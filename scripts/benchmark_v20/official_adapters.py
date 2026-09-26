from __future__ import annotations

import math
from pathlib import Path

import torch

from scripts.benchmark_v19.external_baselines import CAAAdapter, _ContextualAdapter
from scripts.benchmark_v20.official_source import (
    load_loreft_class,
    load_reps_class,
    verify_required_sources,
)


class OfficialCASTAdapter(CAAAdapter):
    def __init__(
        self,
        direction: torch.Tensor,
        strength: float,
        condition_direction: torch.Tensor,
        condition_threshold: float,
        condition_polarity: int,
        maximum_relative_correction: float,
    ):
        super().__init__(direction, strength, maximum_relative_correction)
        condition_direction = condition_direction.float()
        condition_direction = condition_direction / torch.clamp(torch.linalg.vector_norm(condition_direction), min=1e-12)
        self.register_buffer("condition_direction", condition_direction)
        self.condition_threshold = float(condition_threshold)
        if condition_polarity not in (-1, 1):
            raise ValueError("CAST condition polarity must be -1 or 1")
        self.condition_polarity = int(condition_polarity)

    def forward(self, value, gate, boundary_state=None, context_state=None):
        del context_state
        if boundary_state is None:
            raise ValueError("CAST requires a boundary state")
        boundary = boundary_state.float().to(value.device)
        direction = self.condition_direction.to(value.device)
        coefficient = (boundary @ direction) / torch.clamp(direction @ direction, min=1e-12)
        projected = coefficient.unsqueeze(-1) * direction
        numerator = torch.sum(boundary * torch.tanh(projected), dim=-1)
        denominator = torch.linalg.vector_norm(boundary, dim=-1) * torch.linalg.vector_norm(torch.tanh(projected), dim=-1)
        score = numerator / torch.clamp(denominator, min=1e-12)
        high = score >= self.condition_threshold
        inferred = high.float() if self.condition_polarity == 1 else (~high).float()
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


class OfficialLoReFTAdapter(_ContextualAdapter):
    def __init__(self, checkpoint: dict, source_root: Path):
        super().__init__(float(checkpoint["maximum_relative_correction"]))
        cls = load_loreft_class(source_root, checkpoint["official_source_files_sha256"]["loreft"])
        self.modules_by_task = torch.nn.ModuleList()
        for state in checkpoint["loreft_state_dicts"]:
            module = cls(
                embed_dim=int(checkpoint["hidden_size"]),
                low_rank_dimension=int(checkpoint["loreft_rank"]),
                dtype=torch.float32,
                dropout=0.0,
                act_fn=None,
            )
            module.load_state_dict(state)
            module.eval()
            self.modules_by_task.append(module)

    def forward(self, value, gate, boundary_state=None, context_state=None):
        del boundary_state, context_state
        module = self.modules_by_task[int(self.target_obedience)]
        proposed = module(value.float())
        correction = proposed.float() - value.float()
        self.last_diagnostics = {
            "active": torch.tensor(1.0, device=value.device),
            "target_obedience": torch.tensor(self.target_obedience, device=value.device),
        }
        return self._apply_correction(value, correction, gate)


class OfficialRePSAdapter(_ContextualAdapter):
    def __init__(self, checkpoint: dict, source_root: Path):
        super().__init__(float(checkpoint["maximum_relative_correction"]))
        cls = load_reps_class(source_root, checkpoint["official_source_files_sha256"]["reps_intervention"])
        states = checkpoint["reps_state_dicts"]
        if not isinstance(states, list) or len(states) != 2:
            raise ValueError("RePS requires two task-specific official rank-1 modules")
        self.interventions = torch.nn.ModuleList()
        for state in states:
            intervention = cls(
                embed_dim=int(checkpoint["hidden_size"]),
                low_rank_dimension=1,
                dropout=0.0,
                intervention_positions_dropout=0.0,
            )
            intervention.load_state_dict(state)
            intervention.eval()
            self.interventions.append(intervention)

    def forward(self, value, gate, boundary_state=None, context_state=None):
        del boundary_state, context_state
        squeezed = value.ndim == 2
        base = value.unsqueeze(0) if squeezed else value
        batch = base.shape[0]
        task = int(self.target_obedience)
        subspaces = {
            "subspaces": [0] * batch,
            "steering_factor": torch.ones(batch, device=base.device, dtype=torch.float32),
        }
        proposed = self.interventions[task](base.float(), subspaces=subspaces).output
        if squeezed:
            proposed = proposed.squeeze(0)
        correction = proposed.float() - value.float()
        self.last_diagnostics = {
            "active": torch.tensor(1.0, device=value.device),
            "target_obedience": torch.tensor(self.target_obedience, device=value.device),
        }
        return self._apply_correction(value, correction, gate)


def adapter_from_official_checkpoint(checkpoint: dict, method: str, source_root: Path | None):
    if source_root is None:
        raise ValueError("official source root is required")
    verify_required_sources(source_root, checkpoint["official_source_files_sha256"])
    cap = float(checkpoint["maximum_relative_correction"])
    if method == "caa":
        return CAAAdapter(checkpoint["caa_direction"], checkpoint["caa_strength"], cap)
    if method == "cast":
        return OfficialCASTAdapter(
            checkpoint["cast_behavior_direction"], checkpoint["cast_strength"],
            checkpoint["cast_condition_direction"], checkpoint["cast_condition_threshold"],
            int(checkpoint["cast_condition_polarity"]), cap,
        )
    if method == "loreft":
        return OfficialLoReFTAdapter(checkpoint, source_root)
    if method == "reps":
        return OfficialRePSAdapter(checkpoint, source_root)
    raise ValueError(f"unsupported official-derived baseline: {method}")


def finite_checkpoint(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value.float()).all())
    if isinstance(value, dict):
        return all(finite_checkpoint(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_checkpoint(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True
