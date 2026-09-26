from __future__ import annotations

import math

import torch
from torch import nn


def _unit(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    norm = torch.linalg.vector_norm(value)
    if not bool(torch.isfinite(norm)) or float(norm) <= 1e-12:
        raise ValueError("steering direction is non-finite or null")
    return value / norm


def _clip_relative(value: torch.Tensor, correction: torch.Tensor, cap: float) -> torch.Tensor:
    correction_norm = torch.linalg.vector_norm(correction.float(), dim=-1, keepdim=True)
    value_norm = torch.linalg.vector_norm(value.float(), dim=-1, keepdim=True)
    scale = torch.clamp(cap * value_norm / torch.clamp(correction_norm, min=1e-12), max=1.0)
    return correction.float() * scale


class _ContextualAdapter(nn.Module):
    def __init__(self, maximum_relative_correction: float):
        super().__init__()
        if not 0 < maximum_relative_correction <= 1:
            raise ValueError("maximum_relative_correction must lie in (0, 1]")
        self.maximum_relative_correction = float(maximum_relative_correction)
        self.target_obedience = 0.0
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def set_task(self, target_obedience: float) -> None:
        if target_obedience not in (0.0, 1.0):
            raise ValueError("target_obedience must be binary")
        self.target_obedience = float(target_obedience)

    def _apply_correction(
        self,
        value: torch.Tensor,
        correction: torch.Tensor,
        gate: torch.Tensor | float,
    ):
        correction = _clip_relative(value, correction, self.maximum_relative_correction)
        external = torch.as_tensor(gate, device=value.device, dtype=torch.float32)
        while external.ndim < correction.ndim:
            external = external.unsqueeze(-1)
        edited = value.float() + external * correction
        return edited.to(value.dtype)


class CAAAdapter(_ContextualAdapter):
    def __init__(self, direction: torch.Tensor, strength: float, maximum_relative_correction: float):
        super().__init__(maximum_relative_correction)
        self.register_buffer("direction", _unit(direction))
        self.strength = float(strength)

    def forward(self, value, gate, boundary_state=None, context_state=None):
        del boundary_state, context_state
        sign = 1.0 if self.target_obedience == 1.0 else -1.0
        correction = sign * self.strength * self.direction.to(value.device)
        while correction.ndim < value.ndim:
            correction = correction.unsqueeze(0)
        self.last_diagnostics = {
            "active": torch.tensor(1.0, device=value.device),
            "target_obedience": torch.tensor(self.target_obedience, device=value.device),
        }
        return self._apply_correction(value, correction.expand_as(value), gate)


class CASTAdapter(CAAAdapter):
    def __init__(
        self,
        direction: torch.Tensor,
        strength: float,
        condition_direction: torch.Tensor,
        condition_threshold: float,
        maximum_relative_correction: float,
    ):
        super().__init__(direction, strength, maximum_relative_correction)
        self.register_buffer("condition_direction", _unit(condition_direction))
        self.condition_threshold = float(condition_threshold)

    def forward(self, value, gate, boundary_state=None, context_state=None):
        del context_state
        if boundary_state is None:
            raise ValueError("CAST requires a boundary state")
        boundary = boundary_state.float().to(value.device)
        score = boundary @ self.condition_direction.to(value.device)
        inferred = (score >= self.condition_threshold).float()
        target = torch.tensor(self.target_obedience, device=value.device)
        active = (inferred != target).float()
        sign = 1.0 if self.target_obedience == 1.0 else -1.0
        correction = sign * self.strength * self.direction.to(value.device)
        while correction.ndim < value.ndim:
            correction = correction.unsqueeze(0)
        while active.ndim < correction.ndim:
            active = active.unsqueeze(-1)
        correction = correction.expand_as(value) * active
        self.last_diagnostics = {
            "condition_score": score.reshape(-1)[0],
            "active": active.reshape(-1)[0],
            "target_obedience": target,
        }
        return self._apply_correction(value, correction, gate)


class LoReFTAdapter(_ContextualAdapter):
    """Row-vector form of h + R^T(W h + b - R h)."""

    def __init__(
        self,
        basis: torch.Tensor,
        weights: torch.Tensor,
        bias: torch.Tensor,
        maximum_relative_correction: float,
    ):
        super().__init__(maximum_relative_correction)
        if basis.ndim != 2 or weights.shape != (2, basis.shape[1], basis.shape[0]):
            raise ValueError("invalid LoReFT tensor shapes")
        if bias.shape != (2, basis.shape[1]):
            raise ValueError("invalid LoReFT bias shape")
        self.register_buffer("basis", basis.float())
        self.register_buffer("weights", weights.float())
        self.register_buffer("bias", bias.float())

    def forward(self, value, gate, boundary_state=None, context_state=None):
        del boundary_state, context_state
        index = int(self.target_obedience)
        flat = value.float()
        projected = flat @ self.basis.to(value.device)
        target = flat @ self.weights[index].to(value.device).transpose(0, 1)
        target = target + self.bias[index].to(value.device)
        correction = (target - projected) @ self.basis.to(value.device).transpose(0, 1)
        self.last_diagnostics = {
            "active": torch.tensor(1.0, device=value.device),
            "target_obedience": torch.tensor(self.target_obedience, device=value.device),
        }
        return self._apply_correction(value, correction, gate)


class RePSAdapter(_ContextualAdapter):
    def __init__(self, vectors: torch.Tensor, maximum_relative_correction: float):
        super().__init__(maximum_relative_correction)
        if vectors.ndim != 2 or vectors.shape[0] != 2:
            raise ValueError("RePS requires two task-specific vectors")
        self.register_buffer("vectors", vectors.float())

    def forward(self, value, gate, boundary_state=None, context_state=None):
        del boundary_state, context_state
        correction = self.vectors[int(self.target_obedience)].to(value.device)
        while correction.ndim < value.ndim:
            correction = correction.unsqueeze(0)
        self.last_diagnostics = {
            "active": torch.tensor(1.0, device=value.device),
            "target_obedience": torch.tensor(self.target_obedience, device=value.device),
        }
        return self._apply_correction(value, correction.expand_as(value), gate)


def adapter_from_checkpoint(checkpoint: dict, method: str) -> _ContextualAdapter:
    cap = float(checkpoint["maximum_relative_correction"])
    if method == "caa":
        return CAAAdapter(checkpoint["caa_direction"], checkpoint["caa_strength"], cap)
    if method == "cast":
        return CASTAdapter(
            checkpoint["caa_direction"], checkpoint["caa_strength"],
            checkpoint["cast_condition_direction"], checkpoint["cast_condition_threshold"], cap,
        )
    if method == "loreft":
        return LoReFTAdapter(
            checkpoint["loreft_basis"], checkpoint["loreft_weights"],
            checkpoint["loreft_bias"], cap,
        )
    if method == "reps":
        return RePSAdapter(checkpoint["reps_vectors"], cap)
    raise ValueError(f"unsupported activation baseline: {method}")


def finite_checkpoint(checkpoint: dict) -> bool:
    for value in checkpoint.values():
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value.float()).all()):
            return False
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True
