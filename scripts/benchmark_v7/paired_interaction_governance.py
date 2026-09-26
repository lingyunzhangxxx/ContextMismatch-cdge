from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from scripts.benchmark_v1.operators import LayerOperatorSpec, MultiLayerContextOperator
from scripts.benchmark_v4.directional_governance import (
    DirectionalSingleSiteGovernanceEditor,
    editor_from_checkpoint as v3_editor_from_checkpoint,
)
from scripts.benchmark_v5.abstaining_directional_governance import (
    APPLICATION_FEATURE_WIDTH,
    application_features,
)


BOUNDARY_WIDTH = 16
CONTEXT_WIDTH = 32
PAIR_INTERACTION_FEATURE_WIDTH = (
    APPLICATION_FEATURE_WIDTH
    + 3 * CONTEXT_WIDTH
    + BOUNDARY_WIDTH * CONTEXT_WIDTH
)
HEAD_NAMES = ("all_negative", "matched_state")
FEATURE_SCHEMA = "v3-54-plus-repeat-product-absdiff-sqdiff-and-16x32-outer-v1"


def expand_pair_interaction_features(base: torch.Tensor) -> torch.Tensor:
    """Add fixed boundary/context interactions without learning a projection."""
    if base.ndim != 2 or base.shape[1] != APPLICATION_FEATURE_WIDTH:
        raise ValueError(
            f"base application features must have shape [batch, {APPLICATION_FEATURE_WIDTH}]"
        )
    boundary = base[:, :BOUNDARY_WIDTH].float()
    context = base[:, BOUNDARY_WIDTH : BOUNDARY_WIDTH + CONTEXT_WIDTH].float()
    repeated_boundary = boundary.repeat(1, CONTEXT_WIDTH // BOUNDARY_WIDTH)
    difference = repeated_boundary - context
    outer = torch.einsum("bi,bj->bij", boundary, context).flatten(1)
    expanded = torch.cat(
        (
            base.float(),
            repeated_boundary * context,
            torch.abs(difference),
            difference.square(),
            outer,
        ),
        dim=-1,
    )
    if expanded.shape[1] != PAIR_INTERACTION_FEATURE_WIDTH:
        raise ValueError("PAIR-GE feature expansion width mismatch")
    if not bool(torch.isfinite(expanded).all().detach().cpu()):
        raise FloatingPointError("PAIR-GE feature expansion contains non-finite values")
    return expanded


def paired_interaction_features(
    editor: DirectionalSingleSiteGovernanceEditor,
    boundary_state: torch.Tensor,
    context_state: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base, diagnostics = application_features(editor, boundary_state, context_state)
    return expand_pair_interaction_features(base), diagnostics


class PairedInteractionRouter(nn.Module):
    """One shared nonlinear trunk with two independently thresholded veto heads."""

    def __init__(
        self,
        *,
        feature_center: torch.Tensor,
        feature_scale: torch.Tensor,
        hidden_widths: Iterable[int],
        threshold_logits: torch.Tensor | Iterable[float] = (0.0, 0.0),
    ):
        super().__init__()
        center = feature_center.float().clone()
        scale = feature_scale.float().clone()
        if center.ndim != 1 or scale.shape != center.shape:
            raise ValueError("PAIR-GE normalization must be aligned vectors")
        if center.numel() != PAIR_INTERACTION_FEATURE_WIDTH:
            raise ValueError(
                f"PAIR-GE requires {PAIR_INTERACTION_FEATURE_WIDTH} input features"
            )
        if not bool(torch.isfinite(center).all()) or not bool(torch.isfinite(scale).all()):
            raise ValueError("PAIR-GE normalization must be finite")
        if not bool(torch.all(scale > 0)):
            raise ValueError("PAIR-GE normalization scale must be positive")
        widths = tuple(int(value) for value in hidden_widths)
        if widths != (96, 48):
            raise ValueError("PAIR-GE V5.2 requires the frozen 96-48 shared trunk")
        layers: list[nn.Module] = []
        previous = center.numel()
        for width in widths:
            layers.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        thresholds = torch.as_tensor(threshold_logits, dtype=torch.float32).clone()
        if thresholds.shape != (2,) or not bool(torch.isfinite(thresholds).all()):
            raise ValueError("PAIR-GE requires two finite threshold logits")
        self.trunk = nn.Sequential(*layers)
        self.output_heads = nn.ModuleDict(
            {name: nn.Linear(previous, 1) for name in HEAD_NAMES}
        )
        self.register_buffer("feature_center", center)
        self.register_buffer("feature_scale", scale)
        self.register_buffer("threshold_logits", thresholds)
        self.hidden_widths = widths

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != PAIR_INTERACTION_FEATURE_WIDTH:
            raise ValueError(
                "PAIR-GE features must have shape "
                f"[batch, {PAIR_INTERACTION_FEATURE_WIDTH}]"
            )
        normalized = (features.float() - self.feature_center) / self.feature_scale
        hidden = self.trunk(normalized)
        logits = torch.cat(
            tuple(self.output_heads[name](hidden) for name in HEAD_NAMES), dim=-1
        )
        if logits.shape != (features.shape[0], 2):
            raise ValueError("PAIR-GE output-head shape mismatch")
        if not bool(torch.isfinite(logits).all().detach().cpu()):
            raise FloatingPointError("PAIR-GE logits contain non-finite values")
        return logits

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.logits(features)
        head_active = logits > self.threshold_logits
        return logits, head_active


class PairedInteractionSiteEditor(nn.Module):
    """Apply the frozen V3 correction only after two PAIR-GE heads agree."""

    def __init__(
        self,
        v3_editor: DirectionalSingleSiteGovernanceEditor,
        router: PairedInteractionRouter,
    ):
        super().__init__()
        for parameter in v3_editor.parameters():
            parameter.requires_grad_(False)
        self.v3_editor = v3_editor
        self.router = router
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.router.parameters()
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
            raise ValueError("PAIR-GE requires boundary and current-token states")
        gate_tensor = torch.as_tensor(gate, device=value.device, dtype=torch.float32)
        if not bool(torch.any(gate_tensor != 0).detach().cpu()):
            self.last_diagnostics = {}
            return value

        features, feature_diagnostics = paired_interaction_features(
            self.v3_editor, boundary_state, context_state
        )
        logits, head_active = self.router(features)
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
        consensus_active = head_active.all(dim=-1)
        active = route_active & consensus_active & (gate_tensor != 0)
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
            raise ValueError("PAIR-GE edit value must have rank two or three")
        self.last_diagnostics = {
            **v3_diagnostics,
            **feature_diagnostics,
            "all_negative_application_logit": logits[:, 0],
            "matched_state_application_logit": logits[:, 1],
            "all_negative_application_active": head_active[:, 0],
            "matched_state_application_active": head_active[:, 1],
            "pair_ge_consensus_active": consensus_active,
            "pair_ge_gate_active": active,
            "applied_correction": torch.where(
                active.unsqueeze(-1), applied, torch.zeros_like(applied)
            ),
        }
        return edited.to(value.dtype)


@dataclass(frozen=True)
class PairedInteractionSite:
    layer: int
    component: str
    rank: int

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.component}"


class PairedInteractionGovernanceEditor(nn.Module):
    def __init__(
        self,
        site: PairedInteractionSite,
        site_editor: PairedInteractionSiteEditor,
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


def editor_from_checkpoint(checkpoint: dict) -> PairedInteractionGovernanceEditor:
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported PAIR-GE checkpoint schema")
    if checkpoint.get("kind") != "paired_interaction_governance_editor":
        raise ValueError("unsupported PAIR-GE checkpoint kind")
    if checkpoint.get("method") != "PAIR-GE-V5.2":
        raise ValueError("unsupported PAIR-GE checkpoint method")
    for field, expected in {
        "base_model_weights_included": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
        "v3_expert_frozen": True,
        "feature_schema": FEATURE_SCHEMA,
    }.items():
        if checkpoint.get(field) != expected:
            raise ValueError(f"PAIR-GE checkpoint safety mismatch: {field}")
    v3_checkpoint = checkpoint.get("v3_checkpoint")
    if not isinstance(v3_checkpoint, dict):
        raise ValueError("PAIR-GE checkpoint is missing its frozen V3 editor")
    v3 = v3_editor_from_checkpoint(v3_checkpoint)
    if v3.site.key != "27:mlp":
        raise ValueError("PAIR-GE must wrap the selected 27:mlp V3 site")
    record = checkpoint.get("router", {})
    state_dict = record.get("state_dict", {})
    if any(str(name).startswith(("model.", "base_model.")) for name in state_dict):
        raise ValueError("PAIR-GE router contains base-model state")
    router = PairedInteractionRouter(
        feature_center=record["feature_center"],
        feature_scale=record["feature_scale"],
        hidden_widths=record["hidden_widths"],
        threshold_logits=record["threshold_logits"],
    )
    router.load_state_dict(state_dict, strict=True)
    editor = PairedInteractionGovernanceEditor(
        PairedInteractionSite(layer=27, component="mlp", rank=16),
        PairedInteractionSiteEditor(v3, router),
    )
    if editor.trainable_parameter_count != int(
        checkpoint.get("application_trainable_parameter_count", -1)
    ):
        raise ValueError("PAIR-GE trainable parameter-count mismatch")
    return editor
