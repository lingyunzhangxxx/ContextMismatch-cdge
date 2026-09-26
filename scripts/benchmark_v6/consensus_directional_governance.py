from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from scripts.benchmark_v1.operators import LayerOperatorSpec, MultiLayerContextOperator
from scripts.benchmark_v4.directional_governance import (
    DirectionalSingleSiteGovernanceEditor,
    editor_from_checkpoint as v3_editor_from_checkpoint,
)
from scripts.benchmark_v5.abstaining_directional_governance import (
    ApplicationVetoHead,
    application_features,
)


class ConsensusDirectionalSiteEditor(nn.Module):
    """Apply a frozen V3 edit only when both hard V5 application heads agree."""

    def __init__(
        self,
        v3_editor: DirectionalSingleSiteGovernanceEditor,
        all_negative_head: ApplicationVetoHead,
        matched_state_head: ApplicationVetoHead,
    ):
        super().__init__()
        for parameter in v3_editor.parameters():
            parameter.requires_grad_(False)
        self.v3_editor = v3_editor
        self.all_negative_head = all_negative_head
        self.matched_state_head = matched_state_head
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for head in (self.all_negative_head, self.matched_state_head)
            for parameter in head.parameters()
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
            raise ValueError("V5 editor requires boundary and current-token states")
        gate_tensor = torch.as_tensor(gate, device=value.device, dtype=torch.float32)
        if not bool(torch.any(gate_tensor != 0).detach().cpu()):
            self.last_diagnostics = {}
            return value

        features, feature_diagnostics = application_features(
            self.v3_editor, boundary_state, context_state
        )
        all_negative_logit, all_negative_active = self.all_negative_head(features)
        matched_state_logit, matched_state_active = self.matched_state_head(features)
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
        consensus_active = all_negative_active & matched_state_active
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
            raise ValueError("V5 edit value must have rank two or three")
        self.last_diagnostics = {
            **v3_diagnostics,
            **feature_diagnostics,
            "all_negative_application_logit": all_negative_logit,
            "all_negative_application_active": all_negative_active,
            "matched_state_application_logit": matched_state_logit,
            "matched_state_application_active": matched_state_active,
            "consensus_application_active": consensus_active,
            "v5_gate_active": active,
            "applied_correction": torch.where(
                active.unsqueeze(-1), applied, torch.zeros_like(applied)
            ),
        }
        return edited.to(value.dtype)


@dataclass(frozen=True)
class ConsensusDirectionalSite:
    layer: int
    component: str
    rank: int

    @property
    def key(self) -> str:
        return f"{self.layer}:{self.component}"


class ConsensusDirectionalGovernanceEditor(nn.Module):
    def __init__(
        self,
        site: ConsensusDirectionalSite,
        site_editor: ConsensusDirectionalSiteEditor,
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


def _head_from_record(record: dict) -> ApplicationVetoHead:
    state_dict = record.get("state_dict", {})
    if any(str(name).startswith(("model.", "base_model.")) for name in state_dict):
        raise ValueError("V5 application head contains base-model state")
    head = ApplicationVetoHead(
        feature_center=record["feature_center"],
        feature_scale=record["feature_scale"],
        hidden_widths=[],
        threshold_logit=float(record["threshold_logit"]),
    )
    head.load_state_dict(state_dict, strict=True)
    return head


def editor_from_checkpoint(checkpoint: dict) -> ConsensusDirectionalGovernanceEditor:
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported V5 checkpoint schema")
    if checkpoint.get("kind") != "group_robust_consensus_directional_governance_editor":
        raise ValueError("unsupported V5 checkpoint kind")
    if checkpoint.get("method") != "GRC-DGE-V5":
        raise ValueError("unsupported V5 checkpoint method")
    for field, expected in {
        "base_model_weights_included": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
        "v3_expert_frozen": True,
    }.items():
        if checkpoint.get(field) != expected:
            raise ValueError(f"V5 checkpoint safety mismatch: {field}")
    v3_checkpoint = checkpoint.get("v3_checkpoint")
    if not isinstance(v3_checkpoint, dict):
        raise ValueError("V5 checkpoint is missing its frozen V3 editor")
    v3 = v3_editor_from_checkpoint(v3_checkpoint)
    if v3.site.key != "27:mlp":
        raise ValueError("V5 checkpoint must wrap the selected 27:mlp V3 site")
    records = checkpoint.get("application_heads", {})
    if set(records) != {"all_negative", "matched_state"}:
        raise ValueError("V5 checkpoint must contain exactly two named heads")
    editor = ConsensusDirectionalGovernanceEditor(
        ConsensusDirectionalSite(layer=27, component="mlp", rank=16),
        ConsensusDirectionalSiteEditor(
            v3,
            _head_from_record(records["all_negative"]),
            _head_from_record(records["matched_state"]),
        ),
    )
    if editor.trainable_parameter_count != int(
        checkpoint.get("application_trainable_parameter_count", -1)
    ):
        raise ValueError("V5 application parameter-count mismatch")
    return editor
