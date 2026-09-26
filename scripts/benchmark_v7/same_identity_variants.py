from __future__ import annotations

import torch
from torch import nn

from scripts.benchmark_v4.directional_governance import (
    DirectionalSingleSiteGovernanceEditor,
    calibrated_probability,
    structural_directional_routes,
)


V3_ABLATION_METHODS = {
    "shared_single_expert",
    "positive_only_expert",
    "negative_only_expert",
    "dge_without_structural_routing",
}


def _token_value(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.float()
    if value.ndim == 3:
        return value[:, -1].float()
    raise ValueError("value must have rank two or three")


class DGEAblationSiteEditor(nn.Module):
    """Inference-only, reversible ablations of one frozen DGE site.

    The wrapper owns no base-model parameters and keeps the V3 trust-region
    cap.  Its four modes correspond exactly to the frozen supplement contract.
    """

    def __init__(self, frozen_v3: DirectionalSingleSiteGovernanceEditor, method: str):
        super().__init__()
        if method not in V3_ABLATION_METHODS:
            raise ValueError(f"unsupported DGE ablation: {method}")
        self.method = method
        self.v3 = frozen_v3
        for parameter in self.v3.parameters():
            parameter.requires_grad_(False)
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @property
    def base(self):
        return self.v3.site_editor

    def _states(
        self,
        value: torch.Tensor,
        boundary_state: torch.Tensor,
        context_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        token_value = _token_value(value)
        current_context = context_state[:, -1] if context_state.ndim == 3 else context_state
        history_logit, task_logit, boundary_code, context_code = self.base.governance_logits(
            boundary_state, current_context
        )
        history_probability = calibrated_probability(
            history_logit,
            self.base.history_logit_location,
            self.base.history_logit_scale,
        )
        task_probability = calibrated_probability(
            task_logit,
            self.base.task_logit_location,
            self.base.task_logit_scale,
        )
        if self.method == "dge_without_structural_routing":
            positive_route = (task_probability < 0.5).to(torch.float32)
            negative_route = (task_probability >= 0.5).to(torch.float32)
            routes = {
                "positive_route": positive_route,
                "negative_route": negative_route,
                "structural_gate_active": torch.ones_like(positive_route, dtype=torch.bool),
            }
        else:
            routes = structural_directional_routes(
                history_probability,
                task_probability,
                activation_threshold=self.base.activation_threshold,
            )
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
        diagnostics = {
            "history_logit": history_logit,
            "task_logit": task_logit,
            "history_probability": history_probability,
            "task_probability": task_probability,
            **routes,
        }
        return token_value, features, history_probability, diagnostics, task_probability

    def _cap(self, correction: torch.Tensor, token_value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw_norm = torch.linalg.vector_norm(correction.float(), dim=-1, keepdim=True)
        reference_norm = torch.linalg.vector_norm(token_value.float(), dim=-1, keepdim=True)
        trust_scale = torch.clamp(
            self.base.maximum_relative_correction * reference_norm
            / torch.clamp(raw_norm, min=1e-12),
            max=1.0,
        )
        return correction * trust_scale, trust_scale.squeeze(-1)

    def correction(
        self,
        *,
        value: torch.Tensor,
        boundary_state: torch.Tensor,
        context_state: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        token_value, features, _, diagnostics, _ = self._states(
            value, boundary_state, context_state
        )
        positive_route = diagnostics["positive_route"].unsqueeze(-1)
        negative_route = diagnostics["negative_route"].unsqueeze(-1)
        if self.method == "shared_single_expert":
            shared = self.base.positive_expert(features)
            shared_coordinates = (positive_route + negative_route) * shared
            raw = shared_coordinates @ self.base.positive_output_basis.transpose(0, 1)
            diagnostics.update(
                {
                    "positive_expert_coordinates": shared,
                    "negative_expert_coordinates": shared,
                    "shared_expert_coordinates": shared,
                }
            )
        else:
            positive = self.base.positive_expert(features)
            negative = self.base.negative_expert(features)
            if self.method == "positive_only_expert":
                negative_route = torch.zeros_like(negative_route)
            elif self.method == "negative_only_expert":
                positive_route = torch.zeros_like(positive_route)
            raw = (
                (positive_route * positive)
                @ self.base.positive_output_basis.transpose(0, 1)
                + (negative_route * negative)
                @ self.base.negative_output_basis.transpose(0, 1)
            )
            diagnostics.update(
                {
                    "positive_expert_coordinates": positive,
                    "negative_expert_coordinates": negative,
                }
            )
        clipped, trust_scale = self._cap(raw, token_value)
        diagnostics["trust_scale"] = trust_scale
        diagnostics["raw_correction"] = raw
        diagnostics["clipped_correction"] = clipped
        diagnostics["effective_route_active"] = torch.any(clipped != 0, dim=-1)
        return clipped, diagnostics

    def forward(
        self,
        value: torch.Tensor,
        gate: torch.Tensor | float,
        boundary_state: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_state is None or context_state is None:
            raise ValueError("DGE ablation requires boundary and context states")
        gate_tensor = torch.as_tensor(gate, device=value.device, dtype=torch.float32)
        if not bool(torch.any(gate_tensor != 0).detach().cpu()):
            self.last_diagnostics = {}
            return value
        correction, diagnostics = self.correction(
            value=value,
            boundary_state=boundary_state,
            context_state=context_state,
        )
        if gate_tensor.ndim == 0:
            gate_tensor = gate_tensor.expand(correction.shape[0])
        elif gate_tensor.numel() == 1:
            gate_tensor = gate_tensor.reshape(()).expand(correction.shape[0])
        elif gate_tensor.shape != (correction.shape[0],):
            raise ValueError("external gate has incompatible shape")
        active = diagnostics["effective_route_active"] & (gate_tensor != 0)
        applied = correction * gate_tensor.unsqueeze(-1)
        original = value.float()
        edited = original.clone()
        if edited.ndim == 2:
            edited = torch.where(active.unsqueeze(-1), original + applied, original)
        else:
            edited[:, -1] = torch.where(
                active.unsqueeze(-1), original[:, -1] + applied, original[:, -1]
            )
        diagnostics["applied_correction"] = torch.where(
            active.unsqueeze(-1), applied, torch.zeros_like(applied)
        )
        self.last_diagnostics = diagnostics
        return edited.to(value.dtype)
