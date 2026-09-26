from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from scripts.benchmark_v1.operators import (
    LayerOperatorSpec,
    MultiLayerContextOperator,
    orthonormalize,
    remove_protected_subspace,
)


def safe_scale(values: torch.Tensor, minimum: float = 1e-4) -> torch.Tensor:
    """Return a finite per-coordinate scale suitable for standardization."""
    if values.ndim != 2:
        raise ValueError("values must be a rank-two matrix")
    scale = values.float().std(dim=0, unbiased=False)
    if not torch.isfinite(scale).all():
        raise FloatingPointError("non-finite standardization scale")
    return torch.clamp(scale, min=float(minimum))


def clip_relative_correction(
    correction: torch.Tensor,
    reference: torch.Tensor,
    maximum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip each row to ``maximum * ||reference||`` and return applied scale."""
    if maximum <= 0:
        raise ValueError("maximum relative correction must be positive")
    if correction.shape != reference.shape:
        raise ValueError("correction and reference shapes differ")
    correction_norm = torch.linalg.vector_norm(correction.float(), dim=-1, keepdim=True)
    reference_norm = torch.linalg.vector_norm(reference.float(), dim=-1, keepdim=True)
    budget = float(maximum) * reference_norm
    scale = torch.clamp(budget / torch.clamp(correction_norm, min=1e-12), max=1.0)
    return correction * scale, scale


def signed_governance_factor(
    history_logit: torch.Tensor,
    task_logit: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Signed mismatch coordinate: positive is too obedience-like for the task."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if history_logit.shape != task_logit.shape:
        raise ValueError("history and task logits must have identical shapes")
    return torch.tanh((history_logit - task_logit) / float(temperature))


class AdaptiveGovernanceSiteEditor(nn.Module):
    """Nonlinear, sample-conditioned activation editor for one locked site.

    The Qwen base model is never registered as a child module. Only the two
    state classifiers and the compact correction network are trainable. Bases,
    centers, scales, and the trust radius are immutable buffers/attributes.
    """

    def __init__(
        self,
        *,
        boundary_basis: torch.Tensor,
        context_basis: torch.Tensor,
        output_basis: torch.Tensor,
        boundary_center: torch.Tensor,
        context_center: torch.Tensor,
        boundary_scale: torch.Tensor,
        context_scale: torch.Tensor,
        maximum_relative_correction: float,
        hidden_width: int = 96,
        temperature: float = 1.0,
        head_input_mode: str = "pca_coordinates",
        boundary_head_scale: float = 1.0,
        context_head_scale: float = 1.0,
    ):
        super().__init__()
        if boundary_basis.ndim != 2 or context_basis.ndim != 2 or output_basis.ndim != 2:
            raise ValueError("editor bases must be rank two")
        hidden_size = boundary_basis.shape[0]
        if context_basis.shape[0] != hidden_size or output_basis.shape[0] != hidden_size:
            raise ValueError("all bases must share the hidden dimension")
        boundary_rank = boundary_basis.shape[1]
        context_rank = context_basis.shape[1]
        output_rank = output_basis.shape[1]
        if boundary_center.shape != (hidden_size,) or context_center.shape != (hidden_size,):
            raise ValueError("centers must match the hidden dimension")
        if boundary_scale.shape != (boundary_rank,) or context_scale.shape != (context_rank,):
            raise ValueError("coordinate scales have the wrong shape")
        if torch.any(boundary_scale <= 0) or torch.any(context_scale <= 0):
            raise ValueError("coordinate scales must be positive")
        if maximum_relative_correction <= 0:
            raise ValueError("maximum_relative_correction must be positive")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if head_input_mode not in {"pca_coordinates", "full_state"}:
            raise ValueError("unsupported governance-head input mode")
        if boundary_head_scale <= 0 or context_head_scale <= 0:
            raise ValueError("full-state head scales must be positive")

        self.register_buffer("boundary_basis", orthonormalize(boundary_basis.float()))
        self.register_buffer("context_basis", orthonormalize(context_basis.float()))
        self.register_buffer("output_basis", orthonormalize(output_basis.float()))
        self.register_buffer("boundary_center", boundary_center.float().clone())
        self.register_buffer("context_center", context_center.float().clone())
        self.register_buffer("boundary_scale", boundary_scale.float().clone())
        self.register_buffer("context_scale", context_scale.float().clone())
        self.maximum_relative_correction = float(maximum_relative_correction)
        self.temperature = float(temperature)
        self.head_input_mode = str(head_input_mode)
        # Plain immutable scalars keep the V1 state_dict schema load-compatible.
        self.boundary_head_scale = float(boundary_head_scale)
        self.context_head_scale = float(context_head_scale)
        feature_width = boundary_rank + context_rank + 3
        history_width = hidden_size if head_input_mode == "full_state" else boundary_rank
        task_width = hidden_size if head_input_mode == "full_state" else context_rank
        self.history_head = nn.Linear(history_width, 1)
        self.task_head = nn.Linear(task_width, 1)
        self.conditioner = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )
        self.output_head = nn.Linear(hidden_width, output_rank)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def coordinates(
        self,
        boundary_state: torch.Tensor,
        context_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        boundary = boundary_state.float()
        context = context_state.float()
        if boundary.ndim == 1:
            boundary = boundary.unsqueeze(0)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if boundary.ndim != 2 or context.ndim != 2:
            raise ValueError("boundary/context states must be [batch, hidden] or [hidden]")
        if boundary.shape[0] not in (1, context.shape[0]):
            raise ValueError("boundary batch is incompatible with context batch")
        if boundary.shape[0] == 1 and context.shape[0] > 1:
            boundary = boundary.expand(context.shape[0], -1)
        boundary_code = ((boundary - self.boundary_center) @ self.boundary_basis) / self.boundary_scale
        context_code = ((context - self.context_center) @ self.context_basis) / self.context_scale
        return boundary_code, context_code

    def correction(
        self,
        *,
        value: torch.Tensor,
        boundary_state: torch.Tensor,
        context_state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if value.ndim == 2:
            token_value = value.float()
        elif value.ndim == 3:
            token_value = value[:, -1].float()
        else:
            raise ValueError("value must have shape [batch, hidden] or [batch, tokens, hidden]")
        if context_state.ndim == 3:
            current_context = context_state[:, -1]
        else:
            current_context = context_state
        history_logit, task_logit, boundary_code, context_code = self.governance_logits(
            boundary_state, current_context
        )
        history_probability = torch.sigmoid(history_logit)
        task_probability = torch.sigmoid(task_logit)
        signed_factor = signed_governance_factor(
            history_logit, task_logit, temperature=self.temperature
        )
        features = torch.cat(
            (
                boundary_code,
                context_code,
                history_probability.unsqueeze(-1),
                task_probability.unsqueeze(-1),
                signed_factor.unsqueeze(-1),
            ),
            dim=-1,
        )
        coordinates = self.output_head(self.conditioner(features))
        raw = signed_factor.unsqueeze(-1) * (coordinates @ self.output_basis.transpose(0, 1))
        clipped, trust_scale = clip_relative_correction(
            raw, token_value, self.maximum_relative_correction
        )
        diagnostics = {
            "history_logit": history_logit,
            "task_logit": task_logit,
            "history_probability": history_probability,
            "task_probability": task_probability,
            "signed_governance_factor": signed_factor,
            "raw_correction": raw,
            "clipped_correction": clipped,
            "trust_scale": trust_scale.squeeze(-1),
        }
        return clipped, diagnostics

    def governance_logits(
        self,
        boundary_state: torch.Tensor,
        context_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return governance logits plus the compact conditioner coordinates."""
        boundary_code, context_code = self.coordinates(boundary_state, context_state)
        if self.head_input_mode == "full_state":
            boundary_value = boundary_state.float()
            context_value = context_state.float()
            if boundary_value.ndim == 1:
                boundary_value = boundary_value.unsqueeze(0)
            if context_value.ndim == 1:
                context_value = context_value.unsqueeze(0)
            if boundary_value.shape[0] == 1 and context_value.shape[0] > 1:
                boundary_value = boundary_value.expand(context_value.shape[0], -1)
            history_input = (
                boundary_value - self.boundary_center
            ) / self.boundary_head_scale
            task_input = (context_value - self.context_center) / self.context_head_scale
        else:
            history_input = boundary_code
            task_input = context_code
        history_logit = self.history_head(history_input).squeeze(-1)
        task_logit = self.task_head(task_input).squeeze(-1)
        return history_logit, task_logit, boundary_code, context_code

    def forward(
        self,
        value: torch.Tensor,
        gate: torch.Tensor | float,
        boundary_state: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_state is None or context_state is None:
            raise ValueError("adaptive editor requires boundary and current-token states")
        gate_tensor = torch.as_tensor(gate, device=value.device, dtype=torch.float32)
        # Return the original object to make the zero-gate audit bitwise exact.
        if not bool(torch.any(gate_tensor != 0).detach().cpu()):
            self.last_diagnostics = {}
            return value
        correction, diagnostics = self.correction(
            value=value,
            boundary_state=boundary_state,
            context_state=context_state,
        )
        while gate_tensor.ndim < correction.ndim:
            gate_tensor = gate_tensor.unsqueeze(-1)
        applied = gate_tensor * correction
        edited = value.float().clone()
        if edited.ndim == 2:
            edited = edited + applied
        else:
            edited[:, -1] = edited[:, -1] + applied
        diagnostics["applied_correction"] = applied
        self.last_diagnostics = diagnostics
        return edited.to(value.dtype)


@dataclass(frozen=True)
class GovernanceSite:
    layer: int
    component: str
    rank: int

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.component}"


class AdaptiveMultiSiteGovernanceEditor(nn.Module):
    """Trainable three-site editor with a reversible inference-hook adapter."""

    def __init__(self, sites: dict[GovernanceSite, AdaptiveGovernanceSiteEditor]):
        super().__init__()
        if not sites:
            raise ValueError("at least one governance site is required")
        ordered = sorted(sites, key=lambda site: site.layer)
        if len({site.key for site in ordered}) != len(ordered):
            raise ValueError("duplicate governance site")
        self.site_specs = tuple(ordered)
        self.site_editors = nn.ModuleDict(
            {self._module_name(site): sites[site] for site in self.site_specs}
        )

    @staticmethod
    def _module_name(site: GovernanceSite) -> str:
        return f"layer_{site.layer}__{site.component}"

    def editor(self, site: GovernanceSite) -> AdaptiveGovernanceSiteEditor:
        return self.site_editors[self._module_name(site)]

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def hook_manager(self) -> MultiLayerContextOperator:
        operators = {
            LayerOperatorSpec(site.layer, site.component, site.rank): self.editor(site)
            for site in self.site_specs
        }
        return MultiLayerContextOperator(operators)


def editor_from_checkpoint(checkpoint: dict) -> AdaptiveMultiSiteGovernanceEditor:
    """Reconstruct a governance editor without importing or mutating base weights."""
    if checkpoint.get("schema_version") != 1 or checkpoint.get("kind") != "adaptive_governance_editor":
        raise ValueError("unsupported governance-editor checkpoint")
    sites: dict[GovernanceSite, AdaptiveGovernanceSiteEditor] = {}
    for record in checkpoint.get("sites", []):
        spec = GovernanceSite(
            layer=int(record["layer"]),
            component=str(record["component"]),
            rank=int(record["output_rank"]),
        )
        tensors = record["constructor_tensors"]
        site = AdaptiveGovernanceSiteEditor(
            boundary_basis=tensors["boundary_basis"],
            context_basis=tensors["context_basis"],
            output_basis=tensors["output_basis"],
            boundary_center=tensors["boundary_center"],
            context_center=tensors["context_center"],
            boundary_scale=tensors["boundary_scale"],
            context_scale=tensors["context_scale"],
            maximum_relative_correction=float(record["maximum_relative_correction"]),
            hidden_width=int(record["hidden_width"]),
            temperature=float(record["temperature"]),
            head_input_mode=str(record.get("head_input_mode", "pca_coordinates")),
            boundary_head_scale=float(tensors.get("boundary_head_scale", 1.0)),
            context_head_scale=float(tensors.get("context_head_scale", 1.0)),
        )
        site.load_state_dict(record["state_dict"], strict=True)
        sites[spec] = site
    editor = AdaptiveMultiSiteGovernanceEditor(sites)
    expected = int(checkpoint.get("trainable_parameter_count", -1))
    if editor.trainable_parameter_count != expected:
        raise ValueError("checkpoint trainable-parameter count mismatch")
    return editor


def protected_residual_output_basis(
    teacher_deltas: torch.Tensor,
    protected_states: torch.Tensor,
    output_rank: int,
    protected_rank: int,
) -> torch.Tensor:
    """Construct the locked output basis used by the editor fit."""
    if teacher_deltas.ndim != 2 or protected_states.ndim != 2:
        raise ValueError("teacher and protected matrices must be rank two")
    if teacher_deltas.shape[1] != protected_states.shape[1]:
        raise ValueError("teacher and protected hidden dimensions differ")
    if output_rank <= 0 or protected_rank < 0:
        raise ValueError("invalid requested ranks")
    teacher = teacher_deltas.float()
    teacher = teacher - teacher.mean(dim=0)
    teacher_q = min(output_rank + 8, min(teacher.shape))
    _, _, teacher_v = torch.pca_lowrank(teacher, q=teacher_q, center=False, niter=4)
    harmful = teacher_v[:, :output_rank]
    if protected_rank:
        centered = protected_states.float() - protected_states.float().mean(dim=0)
        protected_q = min(protected_rank + 8, min(centered.shape))
        _, _, protected_v = torch.pca_lowrank(
            centered, q=protected_q, center=False, niter=4
        )
        protected = protected_v[:, :protected_rank]
    else:
        protected = None
    basis = remove_protected_subspace(harmful, protected)
    if basis.shape[1] != output_rank:
        raise RuntimeError(
            f"protected residualization retained rank {basis.shape[1]}, expected {output_rank}"
        )
    return basis
