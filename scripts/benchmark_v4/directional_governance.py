from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from scripts.benchmark_v1.operators import (
    LayerOperatorSpec,
    MultiLayerContextOperator,
    orthonormalize,
)


def calibrated_probability(
    raw_logit: torch.Tensor,
    location: torch.Tensor | float,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """Convert a classifier score to a probability on its own calibrated scale.

    ``location`` and ``scale`` are fitted separately for each classifier.  A
    positive affine rescaling of a raw classifier and its calibration values
    therefore leaves this probability unchanged.  No score from one
    classifier is ever subtracted from a score produced by another classifier.
    """
    value = raw_logit.float()
    if not bool(torch.isfinite(value).all().detach().cpu()):
        raise FloatingPointError("raw classifier logit is non-finite")
    center = torch.as_tensor(location, device=value.device, dtype=torch.float32)
    width = torch.as_tensor(scale, device=value.device, dtype=torch.float32)
    if not bool(torch.isfinite(width).all().detach().cpu()) or not bool(
        torch.all(width > 0).detach().cpu()
    ):
        raise ValueError("calibration scale must be finite and positive")
    if not bool(torch.isfinite(center).all().detach().cpu()):
        raise ValueError("calibration location must be finite")
    return torch.sigmoid((value - center) / width)


def structural_directional_routes(
    history_probability: torch.Tensor,
    task_probability: torch.Tensor,
    *,
    activation_threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return disjoint, abstaining routes for the two mismatch directions.

    Probability one denotes obedience/delegated-choice state for both heads.
    The positive route handles obedience-history -> verification-task; the
    negative route handles verification-history -> delegated-choice.  The
    confidence dead zone makes matched and ambiguous states exactly inactive.
    """
    if history_probability.shape != task_probability.shape:
        raise ValueError("history and task probabilities must have identical shapes")
    if not 0.0 < activation_threshold < 1.0:
        raise ValueError("activation_threshold must lie strictly between zero and one")
    history = history_probability.float()
    task = task_probability.float()
    for name, value in (("history", history), ("task", task)):
        if not bool(torch.isfinite(value).all().detach().cpu()):
            raise FloatingPointError(f"non-finite {name} probability")
        if not bool(torch.all((value >= 0) & (value <= 1)).detach().cpu()):
            raise ValueError(f"{name} probability lies outside [0, 1]")
    positive_evidence = history * (1.0 - task)
    negative_evidence = (1.0 - history) * task
    denominator = 1.0 - float(activation_threshold)
    positive_route = torch.relu(positive_evidence - float(activation_threshold)) / denominator
    negative_route = torch.relu(negative_evidence - float(activation_threshold)) / denominator
    active = (positive_route > 0) | (negative_route > 0)
    return {
        "positive_evidence": positive_evidence,
        "negative_evidence": negative_evidence,
        "positive_route": positive_route,
        "negative_route": negative_route,
        "structural_gate_active": active,
    }


class _DirectionalExpert(nn.Module):
    def __init__(self, feature_width: int, hidden_width: int, output_rank: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_width),
            nn.Linear(feature_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, output_rank),
        )
        final = self.network[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class DirectionalGovernanceSiteEditor(nn.Module):
    """Single-site, two-expert, structurally gated activation editor.

    The base model is never registered here.  The editor has independent
    direction-specific correction maps and output bases.  It abstains exactly
    unless one calibrated mismatch route exceeds the frozen confidence gate.
    """

    def __init__(
        self,
        *,
        boundary_basis: torch.Tensor,
        context_basis: torch.Tensor,
        positive_output_basis: torch.Tensor,
        negative_output_basis: torch.Tensor,
        boundary_center: torch.Tensor,
        context_center: torch.Tensor,
        boundary_scale: torch.Tensor,
        context_scale: torch.Tensor,
        maximum_relative_correction: float,
        activation_threshold: float = 0.5,
        hidden_width: int = 64,
        head_input_mode: str = "full_state",
        boundary_head_scale: float = 1.0,
        context_head_scale: float = 1.0,
        history_logit_location: float = 0.0,
        history_logit_scale: float = 1.0,
        task_logit_location: float = 0.0,
        task_logit_scale: float = 1.0,
    ):
        super().__init__()
        bases = (
            boundary_basis,
            context_basis,
            positive_output_basis,
            negative_output_basis,
        )
        if any(value.ndim != 2 for value in bases):
            raise ValueError("all editor bases must be rank two")
        hidden_size = boundary_basis.shape[0]
        if any(value.shape[0] != hidden_size for value in bases[1:]):
            raise ValueError("all editor bases must share the hidden dimension")
        if boundary_center.shape != (hidden_size,) or context_center.shape != (hidden_size,):
            raise ValueError("centers must match the hidden dimension")
        if boundary_scale.shape != (boundary_basis.shape[1],):
            raise ValueError("boundary scale has the wrong shape")
        if context_scale.shape != (context_basis.shape[1],):
            raise ValueError("context scale has the wrong shape")
        if torch.any(boundary_scale <= 0) or torch.any(context_scale <= 0):
            raise ValueError("coordinate scales must be positive")
        if maximum_relative_correction <= 0:
            raise ValueError("maximum_relative_correction must be positive")
        if not 0.0 < activation_threshold < 1.0:
            raise ValueError("activation_threshold must lie strictly between zero and one")
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if head_input_mode != "full_state":
            raise ValueError("V3 requires full-state governance heads")
        for name, value in (
            ("boundary_head_scale", boundary_head_scale),
            ("context_head_scale", context_head_scale),
            ("history_logit_location", history_logit_location),
            ("history_logit_scale", history_logit_scale),
            ("task_logit_location", task_logit_location),
            ("task_logit_scale", task_logit_scale),
        ):
            finite = bool(torch.isfinite(torch.tensor(float(value))))
            if not finite or (name.endswith("scale") and float(value) <= 0):
                qualifier = "finite and positive" if name.endswith("scale") else "finite"
                raise ValueError(f"{name} must be {qualifier}")

        for name, value in (
            ("boundary_basis", boundary_basis),
            ("context_basis", context_basis),
            ("positive_output_basis", positive_output_basis),
            ("negative_output_basis", negative_output_basis),
            ("boundary_center", boundary_center),
            ("context_center", context_center),
            ("boundary_scale", boundary_scale),
            ("context_scale", context_scale),
        ):
            if not bool(torch.isfinite(value.float()).all().detach().cpu()):
                raise FloatingPointError(f"non-finite editor constructor tensor: {name}")

        orthonormal_bases = {
            "boundary_basis": orthonormalize(boundary_basis.float()),
            "context_basis": orthonormalize(context_basis.float()),
            "positive_output_basis": orthonormalize(positive_output_basis.float()),
            "negative_output_basis": orthonormalize(negative_output_basis.float()),
        }
        for name, original in (
            ("boundary_basis", boundary_basis),
            ("context_basis", context_basis),
            ("positive_output_basis", positive_output_basis),
            ("negative_output_basis", negative_output_basis),
        ):
            if orthonormal_bases[name].shape != original.shape:
                raise ValueError(f"{name} is rank deficient")
            self.register_buffer(name, orthonormal_bases[name])
        self.register_buffer("boundary_center", boundary_center.float().clone())
        self.register_buffer("context_center", context_center.float().clone())
        self.register_buffer("boundary_scale", boundary_scale.float().clone())
        self.register_buffer("context_scale", context_scale.float().clone())
        self.register_buffer(
            "history_logit_location", torch.tensor(float(history_logit_location))
        )
        self.register_buffer("history_logit_scale", torch.tensor(float(history_logit_scale)))
        self.register_buffer("task_logit_location", torch.tensor(float(task_logit_location)))
        self.register_buffer("task_logit_scale", torch.tensor(float(task_logit_scale)))
        self.maximum_relative_correction = float(maximum_relative_correction)
        self.activation_threshold = float(activation_threshold)
        self.hidden_width = int(hidden_width)
        self.head_input_mode = str(head_input_mode)
        self.boundary_head_scale = float(boundary_head_scale)
        self.context_head_scale = float(context_head_scale)

        self.history_head = nn.Linear(hidden_size, 1)
        self.task_head = nn.Linear(hidden_size, 1)
        feature_width = boundary_basis.shape[1] + context_basis.shape[1] + 4
        self.positive_expert = _DirectionalExpert(
            feature_width, hidden_width, positive_output_basis.shape[1]
        )
        self.negative_expert = _DirectionalExpert(
            feature_width, hidden_width, negative_output_basis.shape[1]
        )
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
            raise ValueError("boundary/context states must be rank one or two")
        if boundary.shape[0] not in (1, context.shape[0]):
            raise ValueError("boundary batch is incompatible with context batch")
        if boundary.shape[0] == 1 and context.shape[0] > 1:
            boundary = boundary.expand(context.shape[0], -1)
        boundary_code = ((boundary - self.boundary_center) @ self.boundary_basis) / self.boundary_scale
        context_code = ((context - self.context_center) @ self.context_basis) / self.context_scale
        return boundary_code, context_code

    def governance_logits(
        self,
        boundary_state: torch.Tensor,
        context_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        boundary_code, context_code = self.coordinates(boundary_state, context_state)
        boundary = boundary_state.float()
        context = context_state.float()
        if boundary.ndim == 1:
            boundary = boundary.unsqueeze(0)
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if boundary.shape[0] == 1 and context.shape[0] > 1:
            boundary = boundary.expand(context.shape[0], -1)
        history_input = (boundary - self.boundary_center) / self.boundary_head_scale
        task_input = (context - self.context_center) / self.context_head_scale
        return (
            self.history_head(history_input).squeeze(-1),
            self.task_head(task_input).squeeze(-1),
            boundary_code,
            context_code,
        )

    def correction(
        self,
        *,
        value: torch.Tensor,
        boundary_state: torch.Tensor,
        context_state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        positive_coordinates, negative_coordinates, diagnostics = (
            self.correction_coordinates(
                value=value,
                boundary_state=boundary_state,
                context_state=context_state,
            )
        )
        positive = (
            diagnostics["positive_expert_coordinates"]
            @ self.positive_output_basis.transpose(0, 1)
        )
        negative = (
            diagnostics["negative_expert_coordinates"]
            @ self.negative_output_basis.transpose(0, 1)
        )
        correction = (
            positive_coordinates @ self.positive_output_basis.transpose(0, 1)
            + negative_coordinates @ self.negative_output_basis.transpose(0, 1)
        )
        raw = (
            diagnostics["positive_routed_coordinates"]
            @ self.positive_output_basis.transpose(0, 1)
            + diagnostics["negative_routed_coordinates"]
            @ self.negative_output_basis.transpose(0, 1)
        )
        diagnostics.update(
            {
                "positive_expert_correction": positive,
                "negative_expert_correction": negative,
                "raw_correction": raw,
                "clipped_correction": correction,
            }
        )
        return correction, diagnostics

    def correction_coordinates(
        self,
        *,
        value: torch.Tensor,
        boundary_state: torch.Tensor,
        context_state: torch.Tensor,
        reference_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return exact low-rank correction coordinates before basis expansion.

        Because the two thresholded routes are disjoint and every output basis
        is orthonormal, clipping and relative-norm losses are exact in these
        coordinates.  CPU fitting can therefore avoid repeated 4096-wide basis
        expansion without changing the inference edit.
        """
        if value.ndim == 2:
            token_value = value.float()
        elif value.ndim == 3:
            token_value = value[:, -1].float()
        else:
            raise ValueError("value must have shape [batch, hidden] or [batch, tokens, hidden]")
        current_context = context_state[:, -1] if context_state.ndim == 3 else context_state
        history_logit, task_logit, boundary_code, context_code = self.governance_logits(
            boundary_state, current_context
        )
        history_probability = calibrated_probability(
            history_logit, self.history_logit_location, self.history_logit_scale
        )
        task_probability = calibrated_probability(
            task_logit, self.task_logit_location, self.task_logit_scale
        )
        routes = structural_directional_routes(
            history_probability,
            task_probability,
            activation_threshold=self.activation_threshold,
        )
        if bool(
            torch.any(
                (routes["positive_route"] > 0) & (routes["negative_route"] > 0)
            ).detach().cpu()
        ):
            raise RuntimeError("directional routes are unexpectedly non-disjoint")
        if not bool(routes["structural_gate_active"].any().detach().cpu()):
            positive_zeros = token_value.new_zeros(
                (token_value.shape[0], self.positive_output_basis.shape[1])
            )
            negative_zeros = token_value.new_zeros(
                (token_value.shape[0], self.negative_output_basis.shape[1])
            )
            if self.training:
                # Preserve a zero-gradient graph so a fully abstaining failed
                # router becomes an auditable fit falsifier instead of making
                # loss.backward() crash before checkpoints can be materialized.
                positive_anchor = next(self.positive_expert.parameters()).reshape(-1)[0] * 0.0
                negative_anchor = next(self.negative_expert.parameters()).reshape(-1)[0] * 0.0
                positive_zeros = positive_zeros + positive_anchor
                negative_zeros = negative_zeros + negative_anchor
            diagnostics = {
                "history_logit": history_logit,
                "task_logit": task_logit,
                "history_probability": history_probability,
                "task_probability": task_probability,
                **routes,
                "positive_expert_coordinates": positive_zeros,
                "negative_expert_coordinates": negative_zeros,
                "positive_routed_coordinates": positive_zeros,
                "negative_routed_coordinates": negative_zeros,
                "trust_scale": torch.ones(
                    token_value.shape[0], device=token_value.device, dtype=torch.float32
                ),
            }
            return positive_zeros, negative_zeros, diagnostics
        features = torch.cat(
            (
                boundary_code,
                context_code,
                history_probability.unsqueeze(-1),
                task_probability.unsqueeze(-1),
                routes["positive_route"].unsqueeze(-1),
                routes["negative_route"].unsqueeze(-1),
            ),
            dim=-1,
        )
        positive = self.positive_expert(features)
        negative = self.negative_expert(features)
        positive_routed = routes["positive_route"].unsqueeze(-1) * positive
        negative_routed = routes["negative_route"].unsqueeze(-1) * negative
        raw_norm_squared = (
            torch.sum(positive_routed.float().square(), dim=-1, keepdim=True)
            + torch.sum(negative_routed.float().square(), dim=-1, keepdim=True)
        )
        # This is algebraically identical to clamping the norm at 1e-12,
        # while avoiding sqrt'(0) during the experts' zero-initialized step.
        raw_norm = torch.sqrt(torch.clamp(raw_norm_squared, min=1e-24))
        if reference_norm is None:
            current_reference_norm = torch.linalg.vector_norm(
                token_value.float(), dim=-1, keepdim=True
            )
        else:
            current_reference_norm = reference_norm.to(
                device=token_value.device, dtype=torch.float32
            )
            if current_reference_norm.ndim == 1:
                current_reference_norm = current_reference_norm.unsqueeze(-1)
            if current_reference_norm.shape != (token_value.shape[0], 1):
                raise ValueError("reference_norm must contain one scalar per batch row")
            if not bool(
                torch.isfinite(current_reference_norm).all().detach().cpu()
            ) or not bool(torch.all(current_reference_norm >= 0).detach().cpu()):
                raise ValueError("reference_norm must be finite and non-negative")
        budget = self.maximum_relative_correction * current_reference_norm
        trust_scale = torch.clamp(
            budget / raw_norm, max=1.0
        )
        positive_clipped = positive_routed * trust_scale
        negative_clipped = negative_routed * trust_scale
        active = routes["structural_gate_active"].unsqueeze(-1)
        positive_clipped = torch.where(
            active, positive_clipped, torch.zeros_like(positive_clipped)
        )
        negative_clipped = torch.where(
            active, negative_clipped, torch.zeros_like(negative_clipped)
        )
        diagnostics = {
            "history_logit": history_logit,
            "task_logit": task_logit,
            "history_probability": history_probability,
            "task_probability": task_probability,
            **routes,
            "positive_expert_coordinates": positive,
            "negative_expert_coordinates": negative,
            "positive_routed_coordinates": positive_routed,
            "negative_routed_coordinates": negative_routed,
            "trust_scale": trust_scale.squeeze(-1),
        }
        return positive_clipped, negative_clipped, diagnostics

    def forced_direction_coordinates(
        self,
        *,
        direction: str,
        value: torch.Tensor,
        boundary_state: torch.Tensor,
        context_state: torch.Tensor,
        reference_norm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Stress one expert independently of the learned structural router.

        This path is reserved for protected-control fitting and auditing.  It
        prevents router abstention from hiding a large expert correction on a
        protected sample.  Production inference never calls this method.
        """
        if direction not in {"positive", "negative"}:
            raise ValueError("forced direction must be positive or negative")
        if value.ndim == 2:
            token_value = value.float()
        elif value.ndim == 3:
            token_value = value[:, -1].float()
        else:
            raise ValueError("value must have shape [batch, hidden] or [batch, tokens, hidden]")
        current_context = context_state[:, -1] if context_state.ndim == 3 else context_state
        history_logit, task_logit, boundary_code, context_code = self.governance_logits(
            boundary_state, current_context
        )
        history_probability = calibrated_probability(
            history_logit, self.history_logit_location, self.history_logit_scale
        )
        task_probability = calibrated_probability(
            task_logit, self.task_logit_location, self.task_logit_scale
        )
        positive_route = torch.ones_like(history_probability)
        negative_route = torch.zeros_like(history_probability)
        forced_history_probability = torch.ones_like(history_probability)
        forced_task_probability = torch.zeros_like(task_probability)
        if direction == "negative":
            positive_route, negative_route = negative_route, positive_route
            forced_history_probability, forced_task_probability = (
                forced_task_probability,
                forced_history_probability,
            )
        features = torch.cat(
            (
                boundary_code,
                context_code,
                forced_history_probability.unsqueeze(-1),
                forced_task_probability.unsqueeze(-1),
                positive_route.unsqueeze(-1),
                negative_route.unsqueeze(-1),
            ),
            dim=-1,
        )
        positive = self.positive_expert(features) if direction == "positive" else None
        negative = self.negative_expert(features) if direction == "negative" else None
        active_coordinates = positive if positive is not None else negative
        assert active_coordinates is not None
        raw_norm_squared = torch.sum(
            active_coordinates.float().square(), dim=-1, keepdim=True
        )
        raw_norm = torch.sqrt(torch.clamp(raw_norm_squared, min=1e-24))
        if reference_norm is None:
            current_reference_norm = torch.linalg.vector_norm(
                token_value.float(), dim=-1, keepdim=True
            )
        else:
            current_reference_norm = reference_norm.to(
                device=token_value.device, dtype=torch.float32
            )
            if current_reference_norm.ndim == 1:
                current_reference_norm = current_reference_norm.unsqueeze(-1)
            if current_reference_norm.shape != (token_value.shape[0], 1):
                raise ValueError("reference_norm must contain one scalar per batch row")
            if not bool(
                torch.isfinite(current_reference_norm).all().detach().cpu()
            ) or not bool(torch.all(current_reference_norm >= 0).detach().cpu()):
                raise ValueError("reference_norm must be finite and non-negative")
        budget = self.maximum_relative_correction * current_reference_norm
        trust_scale = torch.clamp(budget / raw_norm, max=1.0)
        active_coordinates = active_coordinates * trust_scale
        positive_coordinates = (
            active_coordinates
            if direction == "positive"
            else token_value.new_zeros(
                (token_value.shape[0], self.positive_output_basis.shape[1])
            )
        )
        negative_coordinates = (
            active_coordinates
            if direction == "negative"
            else token_value.new_zeros(
                (token_value.shape[0], self.negative_output_basis.shape[1])
            )
        )
        diagnostics = {
            "history_logit": history_logit,
            "task_logit": task_logit,
            "history_probability": history_probability,
            "task_probability": task_probability,
            "forced_history_probability": forced_history_probability,
            "forced_task_probability": forced_task_probability,
            "positive_route": positive_route,
            "negative_route": negative_route,
            "trust_scale": trust_scale.squeeze(-1),
        }
        return positive_coordinates, negative_coordinates, diagnostics

    def forward(
        self,
        value: torch.Tensor,
        gate: torch.Tensor | float,
        boundary_state: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_state is None or context_state is None:
            raise ValueError("directional editor requires boundary and current-token states")
        gate_tensor = torch.as_tensor(gate, device=value.device, dtype=torch.float32)
        if not bool(torch.any(gate_tensor != 0).detach().cpu()):
            self.last_diagnostics = {}
            return value
        correction, diagnostics = self.correction(
            value=value,
            boundary_state=boundary_state,
            context_state=context_state,
        )
        if not bool(diagnostics["structural_gate_active"].any().detach().cpu()):
            diagnostics["applied_correction"] = torch.zeros_like(correction)
            self.last_diagnostics = diagnostics
            return value
        if gate_tensor.ndim == 0:
            gate_tensor = gate_tensor.expand(correction.shape[0])
        elif gate_tensor.numel() == 1:
            gate_tensor = gate_tensor.reshape(()).expand(correction.shape[0])
        elif gate_tensor.shape != (correction.shape[0],):
            raise ValueError("external gate must be scalar or have one value per batch row")
        gate_rows = gate_tensor != 0
        gate_tensor = gate_tensor.unsqueeze(-1)
        applied = gate_tensor * correction
        route_active = diagnostics["structural_gate_active"]
        active = route_active & gate_rows
        original = value.float()
        edited = original.clone()
        if edited.ndim == 2:
            candidate = original + applied
            edited = torch.where(active.unsqueeze(-1), candidate, original)
        else:
            candidate = original[:, -1] + applied
            edited[:, -1] = torch.where(active.unsqueeze(-1), candidate, original[:, -1])
        diagnostics["applied_correction"] = torch.where(
            active.unsqueeze(-1), applied, torch.zeros_like(applied)
        )
        self.last_diagnostics = diagnostics
        return edited.to(value.dtype)


@dataclass(frozen=True)
class DirectionalGovernanceSite:
    layer: int
    component: str
    positive_rank: int
    negative_rank: int

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.component}"


class DirectionalSingleSiteGovernanceEditor(nn.Module):
    """A reversible wrapper that deliberately refuses naive multi-site stacking."""

    def __init__(
        self,
        site: DirectionalGovernanceSite,
        editor: DirectionalGovernanceSiteEditor,
    ):
        super().__init__()
        self.site = site
        self.site_editor = editor

    @property
    def trainable_parameter_count(self) -> int:
        return self.site_editor.trainable_parameter_count

    def hook_manager(self) -> MultiLayerContextOperator:
        spec = LayerOperatorSpec(
            self.site.layer,
            self.site.component,
            max(self.site.positive_rank, self.site.negative_rank),
        )
        return MultiLayerContextOperator({spec: self.site_editor})


def editor_from_checkpoint(checkpoint: dict) -> DirectionalSingleSiteGovernanceEditor:
    """Rebuild a V3 editor while refusing base weights and multi-site payloads."""
    if checkpoint.get("schema_version") != 2:
        raise ValueError("unsupported directional-editor checkpoint schema")
    if checkpoint.get("kind") != "directional_structural_governance_editor":
        raise ValueError("unsupported directional-editor checkpoint kind")
    if checkpoint.get("method") != "DSGE-V3":
        raise ValueError("unsupported directional-editor checkpoint method")
    if checkpoint.get("base_model_weights_included") is not False:
        raise ValueError("directional checkpoint must explicitly exclude base weights")
    if checkpoint.get("final_test_open") is not False:
        raise ValueError("directional checkpoint unexpectedly opens final test")
    if checkpoint.get("final_test_open_count") != 0:
        raise ValueError("directional checkpoint final-test count must be zero")
    if checkpoint.get("production_rollout_approved") is not False:
        raise ValueError("directional checkpoint unexpectedly approves production")
    records = checkpoint.get("sites", [])
    if len(records) != 1:
        raise ValueError("V3 checkpoints must contain exactly one selected site")
    record = records[0]
    state_dict = record.get("state_dict", {})
    if any(str(name).startswith(("model.", "base_model.")) for name in state_dict):
        raise ValueError("directional checkpoint contains base-model state")
    tensors = record["constructor_tensors"]
    site = DirectionalGovernanceSite(
        layer=int(record["layer"]),
        component=str(record["component"]),
        positive_rank=int(record["positive_output_rank"]),
        negative_rank=int(record["negative_output_rank"]),
    )
    editor = DirectionalGovernanceSiteEditor(
        boundary_basis=tensors["boundary_basis"],
        context_basis=tensors["context_basis"],
        positive_output_basis=tensors["positive_output_basis"],
        negative_output_basis=tensors["negative_output_basis"],
        boundary_center=tensors["boundary_center"],
        context_center=tensors["context_center"],
        boundary_scale=tensors["boundary_scale"],
        context_scale=tensors["context_scale"],
        maximum_relative_correction=float(record["maximum_relative_correction"]),
        activation_threshold=float(record["activation_threshold"]),
        hidden_width=int(record["hidden_width"]),
        head_input_mode=str(record["head_input_mode"]),
        boundary_head_scale=float(tensors["boundary_head_scale"]),
        context_head_scale=float(tensors["context_head_scale"]),
        history_logit_location=float(tensors["history_logit_location"]),
        history_logit_scale=float(tensors["history_logit_scale"]),
        task_logit_location=float(tensors["task_logit_location"]),
        task_logit_scale=float(tensors["task_logit_scale"]),
    )
    editor.load_state_dict(record["state_dict"], strict=True)
    wrapped = DirectionalSingleSiteGovernanceEditor(site, editor)
    if wrapped.trainable_parameter_count != int(checkpoint.get("trainable_parameter_count", -1)):
        raise ValueError("directional checkpoint parameter-count mismatch")
    return wrapped
