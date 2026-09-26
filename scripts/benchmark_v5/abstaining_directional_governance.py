from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from scripts.benchmark_v1.operators import LayerOperatorSpec, MultiLayerContextOperator
from scripts.benchmark_v4.directional_governance import (
    DirectionalSingleSiteGovernanceEditor,
    calibrated_probability,
    editor_from_checkpoint as v3_editor_from_checkpoint,
    structural_directional_routes,
)
from scripts.benchmark_v5.abstaining_split import APPLICATION_FEATURE_WIDTH


def application_features(
    editor: DirectionalSingleSiteGovernanceEditor,
    boundary_state: torch.Tensor,
    context_state: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return answer-blind V3 state features used by the V4 applicability veto."""
    current_context = context_state[:, -1] if context_state.ndim == 3 else context_state
    history_logit, task_logit, boundary_code, context_code = (
        editor.site_editor.governance_logits(boundary_state, current_context)
    )
    history_probability = calibrated_probability(
        history_logit,
        editor.site_editor.history_logit_location,
        editor.site_editor.history_logit_scale,
    )
    task_probability = calibrated_probability(
        task_logit,
        editor.site_editor.task_logit_location,
        editor.site_editor.task_logit_scale,
    )
    routes = structural_directional_routes(
        history_probability,
        task_probability,
        activation_threshold=editor.site_editor.activation_threshold,
    )
    features = torch.cat(
        (
            boundary_code,
            context_code,
            history_probability.unsqueeze(-1),
            task_probability.unsqueeze(-1),
            routes["positive_evidence"].unsqueeze(-1),
            routes["negative_evidence"].unsqueeze(-1),
            routes["positive_route"].unsqueeze(-1),
            routes["negative_route"].unsqueeze(-1),
        ),
        dim=-1,
    )
    if features.shape[-1] != APPLICATION_FEATURE_WIDTH:
        raise ValueError(f"unexpected V4 applicability feature width: {features.shape[-1]}")
    if not bool(torch.isfinite(features).all().detach().cpu()):
        raise FloatingPointError("non-finite V4 applicability feature")
    diagnostics = {
        "history_probability": history_probability,
        "task_probability": task_probability,
        **routes,
    }
    return features, diagnostics


class ApplicationVetoHead(nn.Module):
    """Small classifier whose hard decision can only veto a V3 route."""

    def __init__(
        self,
        *,
        feature_center: torch.Tensor,
        feature_scale: torch.Tensor,
        hidden_widths: Iterable[int],
        threshold_logit: float = 0.0,
    ):
        super().__init__()
        center = feature_center.float().clone()
        scale = feature_scale.float().clone()
        if center.ndim != 1 or scale.shape != center.shape:
            raise ValueError("application feature normalization must be aligned vectors")
        if center.numel() != APPLICATION_FEATURE_WIDTH:
            raise ValueError(
                f"application veto requires exactly {APPLICATION_FEATURE_WIDTH} input features"
            )
        if not bool(torch.isfinite(center).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("application feature normalization must be finite")
        if not bool(torch.all(scale > 0)):
            raise ValueError("application feature scale must be positive")
        widths = [int(value) for value in hidden_widths]
        if any(value <= 0 for value in widths):
            raise ValueError("application hidden widths must be positive")
        layers: list[nn.Module] = []
        previous = center.numel()
        for width in widths:
            layers.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.register_buffer("feature_center", center)
        self.register_buffer("feature_scale", scale)
        self.register_buffer("threshold_logit", torch.tensor(float(threshold_logit)))
        self.hidden_widths = tuple(widths)

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_center.numel():
            raise ValueError(
                f"application features must have shape [batch, {APPLICATION_FEATURE_WIDTH}]"
            )
        normalized = (features.float() - self.feature_center) / self.feature_scale
        logits = self.network(normalized).squeeze(-1)
        if not bool(torch.isfinite(logits).all().detach().cpu()):
            raise FloatingPointError("non-finite application-veto logit")
        return logits

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(features)
        active = logits > self.threshold_logit
        return logits, active


class AbstainingDirectionalSiteEditor(nn.Module):
    """Freeze a successful V3 expert and place an exact internal veto before it."""

    def __init__(
        self,
        v3_editor: DirectionalSingleSiteGovernanceEditor,
        application_head: ApplicationVetoHead,
    ):
        super().__init__()
        for parameter in v3_editor.parameters():
            parameter.requires_grad_(False)
        self.v3_editor = v3_editor
        self.application_head = application_head
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.application_head.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        value: torch.Tensor,
        gate: torch.Tensor | float,
        boundary_state: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_state is None or context_state is None:
            raise ValueError("V4 editor requires boundary and current-token states")
        gate_tensor = torch.as_tensor(gate, device=value.device, dtype=torch.float32)
        if not bool(torch.any(gate_tensor != 0).detach().cpu()):
            self.last_diagnostics = {}
            return value

        features, feature_diagnostics = application_features(
            self.v3_editor, boundary_state, context_state
        )
        application_logit, application_active = self.application_head(features)
        correction, v3_diagnostics = self.v3_editor.site_editor.correction(
            value=value,
            boundary_state=boundary_state,
            context_state=context_state,
        )
        route_active = v3_diagnostics["structural_gate_active"]
        if gate_tensor.ndim == 0:
            gate_tensor = gate_tensor.expand(correction.shape[0])
        elif gate_tensor.numel() == 1:
            gate_tensor = gate_tensor.reshape(()).expand(correction.shape[0])
        elif gate_tensor.shape != (correction.shape[0],):
            raise ValueError("external gate must be scalar or one value per batch row")
        active = route_active & application_active & (gate_tensor != 0)
        applied = gate_tensor.unsqueeze(-1) * correction
        original = value.float()
        edited = original.clone()
        if edited.ndim == 2:
            edited = torch.where(active.unsqueeze(-1), original + applied, original)
        elif edited.ndim == 3:
            edited[:, -1] = torch.where(
                active.unsqueeze(-1), original[:, -1] + applied, original[:, -1]
            )
        else:
            raise ValueError("V4 edit value must have rank two or three")
        self.last_diagnostics = {
            **v3_diagnostics,
            **feature_diagnostics,
            "application_logit": application_logit,
            "application_probability": torch.sigmoid(application_logit),
            "application_gate_active": application_active,
            "v4_gate_active": active,
            "applied_correction": torch.where(
                active.unsqueeze(-1), applied, torch.zeros_like(applied)
            ),
        }
        return edited.to(value.dtype)


@dataclass(frozen=True)
class AbstainingDirectionalSite:
    layer: int
    component: str
    rank: int

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.component}"


class AbstainingDirectionalGovernanceEditor(nn.Module):
    def __init__(
        self,
        site: AbstainingDirectionalSite,
        site_editor: AbstainingDirectionalSiteEditor,
    ):
        super().__init__()
        self.site = site
        self.site_editor = site_editor

    @property
    def trainable_parameter_count(self) -> int:
        return self.site_editor.trainable_parameter_count

    @property
    def frozen_v3(self) -> DirectionalSingleSiteGovernanceEditor:
        return self.site_editor.v3_editor

    def hook_manager(self) -> MultiLayerContextOperator:
        spec = LayerOperatorSpec(self.site.layer, self.site.component, self.site.rank)
        return MultiLayerContextOperator({spec: self.site_editor})


def editor_from_checkpoint(checkpoint: dict) -> AbstainingDirectionalGovernanceEditor:
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported V4 checkpoint schema")
    if checkpoint.get("kind") != "abstaining_directional_governance_editor":
        raise ValueError("unsupported V4 checkpoint kind")
    if checkpoint.get("method") != "ADSGE-V4":
        raise ValueError("unsupported V4 checkpoint method")
    for field, expected in {
        "base_model_weights_included": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
        "v3_experts_frozen": True,
    }.items():
        if checkpoint.get(field) != expected:
            raise ValueError(f"V4 checkpoint safety mismatch: {field}")
    v3_checkpoint = checkpoint.get("v3_checkpoint")
    if not isinstance(v3_checkpoint, dict):
        raise ValueError("V4 checkpoint is missing its frozen V3 editor")
    v3 = v3_editor_from_checkpoint(v3_checkpoint)
    if v3.site.key != "27:mlp":
        raise ValueError("V4 checkpoint must wrap the selected 27:mlp V3 site")
    record = checkpoint.get("application_head", {})
    state_dict = record.get("state_dict", {})
    if any(str(name).startswith(("model.", "base_model.")) for name in state_dict):
        raise ValueError("V4 application head contains base-model state")
    head = ApplicationVetoHead(
        feature_center=record["feature_center"],
        feature_scale=record["feature_scale"],
        hidden_widths=record["hidden_widths"],
        threshold_logit=float(record["threshold_logit"]),
    )
    head.load_state_dict(state_dict, strict=True)
    site = AbstainingDirectionalSite(layer=27, component="mlp", rank=16)
    wrapped = AbstainingDirectionalGovernanceEditor(
        site, AbstainingDirectionalSiteEditor(v3, head)
    )
    if wrapped.trainable_parameter_count != int(
        checkpoint.get("application_trainable_parameter_count", -1)
    ):
        raise ValueError("V4 application parameter-count mismatch")
    return wrapped
