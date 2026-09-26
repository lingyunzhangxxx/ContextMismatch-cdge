from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def orthonormalize(matrix: torch.Tensor, tolerance: float = 1e-6) -> torch.Tensor:
    """Return an oriented orthonormal basis, dropping numerically null columns.

    QR column signs are mathematically arbitrary and differ across LAPACK/NPU
    backends.  That ambiguity is not harmless for a one-sided operator: flipping
    a basis vector swaps the side selected by ``relu(score - threshold)``.
    Canonicalizing the diagonal of R to be positive preserves the orientation
    supplied by the fitted basis and makes inference backend-stable.
    """
    if matrix.ndim != 2:
        raise ValueError("basis matrix must be rank two")
    q, r = torch.linalg.qr(matrix, mode="reduced")
    diagonal = torch.diagonal(r)
    keep = torch.abs(diagonal) > tolerance
    signs = torch.where(diagonal >= 0, torch.ones_like(diagonal), -torch.ones_like(diagonal))
    return (q * signs.unsqueeze(0))[:, keep]


def remove_protected_subspace(harmful: torch.Tensor, protected: torch.Tensor | None) -> torch.Tensor:
    if protected is None or protected.numel() == 0:
        return orthonormalize(harmful)
    protected_q = orthonormalize(protected)
    residual = harmful - protected_q @ (protected_q.transpose(0, 1) @ harmful)
    return orthonormalize(residual)


def bilinear_transport_features(
    history_scores: torch.Tensor,
    threshold: torch.Tensor,
    context_scores: torch.Tensor | None = None,
    one_sided: bool = True,
) -> torch.Tensor:
    """Build `[r, vec(r c^T)]` features used by v4 fitting and inference."""
    if history_scores.shape[-1] != threshold.shape[0]:
        raise ValueError("history score and threshold ranks differ")
    history = (
        torch.relu(history_scores - threshold)
        if one_sided
        else history_scores
    )
    if context_scores is None or context_scores.shape[-1] == 0:
        return history
    if history.shape[:-1] != context_scores.shape[:-1]:
        raise ValueError("history and context feature batch shapes differ")
    interaction = torch.einsum("...r,...c->...rc", history, context_scores)
    return torch.cat((history, interaction.flatten(start_dim=-2)), dim=-1)


def fit_ridge_transport(
    features: torch.Tensor,
    targets: torch.Tensor,
    penalty: float,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Closed-form ridge map from trigger/context features to correction coordinates."""
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must be aligned rank-two matrices")
    if penalty < 0:
        raise ValueError("ridge penalty must be non-negative")
    x = features.float()
    y = targets.float()
    if sample_weight is not None:
        if sample_weight.shape != (features.shape[0],):
            raise ValueError("sample_weight must have one entry per row")
        if torch.any(sample_weight < 0):
            raise ValueError("sample weights must be non-negative")
        root_weight = torch.sqrt(sample_weight.float()).unsqueeze(-1)
        x = x * root_weight
        y = y * root_weight
    gram = x.transpose(0, 1) @ x
    if penalty:
        gram = gram + penalty * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    right = x.transpose(0, 1) @ y
    return torch.linalg.solve(gram, right)


class MismatchGate(nn.Module):
    """Product gate for history regime and current-task governance requirement."""

    def __init__(self, history_weight: torch.Tensor, history_bias: float = 0.0):
        super().__init__()
        if history_weight.ndim != 1:
            raise ValueError("history gate weight must be a vector")
        self.register_buffer("history_weight", history_weight.detach().clone())
        self.register_buffer("history_bias", torch.tensor(float(history_bias)))

    def forward(self, boundary_state: torch.Tensor, verification_required: torch.Tensor) -> torch.Tensor:
        history_probability = torch.sigmoid(boundary_state.float() @ self.history_weight.float() + self.history_bias)
        requirement_probability = verification_required.float().clamp(0.0, 1.0)
        return history_probability * requirement_probability


class SignedGovernanceAlignmentGate(nn.Module):
    """Answer-blind bidirectional gate for governance-state mismatch.

    ``target_obedience`` is supplied by the application-level task contract:
    zero means independent verification is required and one means legitimate
    user deference is required.  A positive result means that the carried state
    is too obedience-like and the fitted correction should be subtracted.  A
    negative result means that it is too verification-like and the same
    correction should be added.  ``applicable`` closes the intervention for
    non-governance tasks such as factual-memory retrieval.
    """

    def __init__(
        self,
        history_weight: torch.Tensor,
        history_bias: float = 0.0,
        deadzone: float = 0.0,
    ):
        super().__init__()
        if history_weight.ndim != 1:
            raise ValueError("history gate weight must be a vector")
        if not 0.0 <= deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        self.register_buffer("history_weight", history_weight.detach().clone().float())
        self.register_buffer("history_bias", torch.tensor(float(history_bias)))
        self.deadzone = float(deadzone)

    def history_probability(self, boundary_state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            boundary_state.float() @ self.history_weight.float() + self.history_bias
        )

    def forward(
        self,
        boundary_state: torch.Tensor,
        target_obedience: torch.Tensor | float,
        applicable: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        probability = self.history_probability(boundary_state)
        target = torch.as_tensor(
            target_obedience,
            device=probability.device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        application = torch.as_tensor(
            applicable,
            device=probability.device,
            dtype=torch.float32,
        ).clamp(0.0, 1.0)
        error = probability - target
        if self.deadzone:
            magnitude = torch.relu(torch.abs(error) - self.deadzone) / (1.0 - self.deadzone)
            error = torch.sign(error) * magnitude
        return application * error


class OneSidedProtectedProjection(nn.Module):
    """Low-rank harmful-side removal in residual/component-output space."""

    def __init__(
        self,
        harmful_basis: torch.Tensor,
        threshold: torch.Tensor,
        strength: torch.Tensor,
        center: torch.Tensor | None = None,
        protected_basis: torch.Tensor | None = None,
        one_sided: bool = True,
        max_relative_correction: float | None = None,
    ):
        super().__init__()
        basis = remove_protected_subspace(harmful_basis.float(), protected_basis.float() if protected_basis is not None else None)
        rank = basis.shape[1]
        if threshold.shape != (rank,) or strength.shape != (rank,):
            raise ValueError(f"threshold/strength must have shape {(rank,)}")
        if torch.any(strength < 0):
            raise ValueError("projection strengths must be non-negative")
        if max_relative_correction is not None and max_relative_correction <= 0:
            raise ValueError("max_relative_correction must be positive")
        self.register_buffer("basis", basis)
        self.register_buffer("threshold", threshold.float())
        self.register_buffer("strength", strength.float())
        self.register_buffer("center", torch.zeros(basis.shape[0]) if center is None else center.float())
        self.one_sided = bool(one_sided)
        self.max_relative_correction = max_relative_correction

    def forward(
        self,
        value: torch.Tensor,
        gate: torch.Tensor | float,
        boundary_state: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del boundary_state, context_state
        original_dtype = value.dtype
        basis = self.basis.to(device=value.device)
        threshold = self.threshold.to(device=value.device)
        strength = self.strength.to(device=value.device)
        center = self.center.to(device=value.device)
        centered = value.float() - center
        scores = centered @ basis
        if self.one_sided:
            removable = torch.relu(scores - threshold)
        else:
            removable = scores
        correction = (removable * strength) @ basis.transpose(0, 1)
        if self.max_relative_correction is not None:
            correction_norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
            value_norm = torch.linalg.vector_norm(value.float(), dim=-1, keepdim=True)
            budget = self.max_relative_correction * value_norm
            correction = correction * torch.clamp(
                budget / torch.clamp(correction_norm, min=1e-12), max=1.0
            )
        gate_tensor = torch.as_tensor(gate, device=value.device, dtype=torch.float32)
        while gate_tensor.ndim < correction.ndim:
            gate_tensor = gate_tensor.unsqueeze(-1)
        return (value.float() - gate_tensor * correction).to(original_dtype)


class FixedDirectionTranslation(nn.Module):
    """Sign-fixed translation retained only as a label-balance diagnostic."""

    def __init__(
        self,
        direction: torch.Tensor,
        strength: float,
        max_relative_correction: float | None = None,
    ):
        super().__init__()
        if direction.ndim != 1:
            raise ValueError("translation direction must be a vector")
        if strength < 0:
            raise ValueError("translation strength must be non-negative")
        if max_relative_correction is not None and max_relative_correction <= 0:
            raise ValueError("max_relative_correction must be positive")
        unit = direction.float() / torch.clamp(
            torch.linalg.vector_norm(direction.float()), min=1e-12
        )
        self.register_buffer("direction", unit)
        self.strength = float(strength)
        self.max_relative_correction = max_relative_correction

    def forward(
        self,
        value: torch.Tensor,
        gate: torch.Tensor | float,
        boundary_state: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del boundary_state, context_state
        correction = self.strength * self.direction.to(device=value.device)
        while correction.ndim < value.ndim:
            correction = correction.unsqueeze(0)
        if self.max_relative_correction is not None:
            correction_norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
            value_norm = torch.linalg.vector_norm(value.float(), dim=-1, keepdim=True)
            budget = self.max_relative_correction * value_norm
            correction = correction * torch.clamp(
                budget / torch.clamp(correction_norm, min=1e-12), max=1.0
            )
        gate_tensor = torch.as_tensor(gate, device=value.device, dtype=torch.float32)
        while gate_tensor.ndim < value.ndim:
            gate_tensor = gate_tensor.unsqueeze(-1)
        return (value.float() - gate_tensor * correction).to(value.dtype)


class ConditionalLowRankTransport(nn.Module):
    """Gated low-rank transport with distinct trigger and correction subspaces.

    The v2 projection is constrained to detect and remove a feature in the same
    basis. This operator detects harmful-side scores in a trigger basis, maps
    them through a small learned matrix, and applies the correction in a
    protected output basis. A relative trust region bounds each token's edit.
    """

    def __init__(
        self,
        trigger_basis: torch.Tensor,
        output_basis: torch.Tensor,
        transport: torch.Tensor,
        threshold: torch.Tensor,
        center: torch.Tensor | None = None,
        protected_basis: torch.Tensor | None = None,
        max_relative_correction: float | None = None,
        one_sided: bool = True,
    ):
        super().__init__()
        trigger_q = orthonormalize(trigger_basis.float())
        output_q = remove_protected_subspace(
            output_basis.float(),
            protected_basis.float() if protected_basis is not None else None,
        )
        if threshold.shape != (trigger_q.shape[1],):
            raise ValueError(f"threshold must have shape {(trigger_q.shape[1],)}")
        if transport.shape != (trigger_q.shape[1], output_q.shape[1]):
            raise ValueError(
                "transport must have shape "
                f"{(trigger_q.shape[1], output_q.shape[1])} after basis residualization"
            )
        if max_relative_correction is not None and max_relative_correction <= 0:
            raise ValueError("max_relative_correction must be positive")
        self.register_buffer("trigger_basis", trigger_q)
        self.register_buffer("output_basis", output_q)
        self.register_buffer("transport", transport.float())
        self.register_buffer("threshold", threshold.float())
        self.register_buffer(
            "center",
            torch.zeros(trigger_q.shape[0]) if center is None else center.float(),
        )
        self.max_relative_correction = max_relative_correction
        self.one_sided = bool(one_sided)

    def forward(
        self,
        value: torch.Tensor,
        gate: torch.Tensor | float,
        boundary_state: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del boundary_state, context_state
        original_dtype = value.dtype
        device = value.device
        trigger = self.trigger_basis.to(device=device)
        output = self.output_basis.to(device=device)
        transport = self.transport.to(device=device)
        threshold = self.threshold.to(device=device)
        center = self.center.to(device=device)
        scores = (value.float() - center) @ trigger
        activated = torch.relu(scores - threshold) if self.one_sided else scores
        correction = (activated @ transport) @ output.transpose(0, 1)
        if self.max_relative_correction is not None:
            correction_norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
            value_norm = torch.linalg.vector_norm(value.float(), dim=-1, keepdim=True)
            budget = self.max_relative_correction * value_norm
            scale = torch.clamp(budget / torch.clamp(correction_norm, min=1e-12), max=1.0)
            correction = correction * scale
        gate_tensor = torch.as_tensor(gate, device=device, dtype=torch.float32)
        while gate_tensor.ndim < correction.ndim:
            gate_tensor = gate_tensor.unsqueeze(-1)
        return (value.float() - gate_tensor * correction).to(original_dtype)


class ContextConditionedLowRankTransport(nn.Module):
    """Boundary-triggered, task-conditioned low-rank correction (v4).

    A history-side trigger is read from a frozen task-boundary state, while a
    bounded context code is read from each current-task token. Their bilinear
    features select a task-dependent correction in a protected output basis.
    This preserves label/task equivariance without giving the operator access
    to the correct answer label.
    """

    def __init__(
        self,
        history_trigger_basis: torch.Tensor,
        output_basis: torch.Tensor,
        transport: torch.Tensor,
        threshold: torch.Tensor,
        context_basis: torch.Tensor | None = None,
        history_center: torch.Tensor | None = None,
        context_center: torch.Tensor | None = None,
        context_scale: torch.Tensor | None = None,
        history_reference: torch.Tensor | None = None,
        protected_basis: torch.Tensor | None = None,
        max_relative_correction: float | None = None,
        one_sided: bool = True,
        edit_last_token_only: bool = True,
    ):
        super().__init__()
        history_q = orthonormalize(history_trigger_basis.float())
        if context_basis is None:
            context_q = torch.empty(
                (history_q.shape[0], 0),
                device=history_q.device,
                dtype=history_q.dtype,
            )
        else:
            context_q = orthonormalize(context_basis.float())
        output_q = remove_protected_subspace(
            output_basis.float(),
            protected_basis.float() if protected_basis is not None else None,
        )
        history_rank = history_q.shape[1]
        context_rank = context_q.shape[1]
        feature_rank = history_rank * (1 + context_rank)
        if threshold.shape != (history_rank,):
            raise ValueError(f"threshold must have shape {(history_rank,)}")
        if transport.shape != (feature_rank, output_q.shape[1]):
            raise ValueError(
                "transport must have shape "
                f"{(feature_rank, output_q.shape[1])} after basis construction"
            )
        if max_relative_correction is not None and max_relative_correction <= 0:
            raise ValueError("max_relative_correction must be positive")
        if context_scale is None:
            scale = torch.ones(context_rank, dtype=torch.float32)
        else:
            if context_scale.shape != (context_rank,):
                raise ValueError(f"context_scale must have shape {(context_rank,)}")
            if torch.any(context_scale <= 0):
                raise ValueError("context_scale entries must be positive")
            scale = context_scale.float()
        self.register_buffer("history_trigger_basis", history_q)
        self.register_buffer("context_basis", context_q)
        self.register_buffer("output_basis", output_q)
        self.register_buffer("transport", transport.float())
        self.register_buffer("threshold", threshold.float())
        self.register_buffer(
            "history_center",
            torch.zeros(history_q.shape[0]) if history_center is None else history_center.float(),
        )
        self.register_buffer(
            "context_center",
            torch.zeros(context_q.shape[0]) if context_center is None else context_center.float(),
        )
        self.register_buffer("context_scale", scale)
        if history_reference is not None and history_reference.shape != (history_rank,):
            raise ValueError(f"history_reference must have shape {(history_rank,)}")
        self.register_buffer(
            "history_reference",
            torch.empty(0, dtype=torch.float32)
            if history_reference is None
            else history_reference.float(),
        )
        self.max_relative_correction = max_relative_correction
        self.one_sided = bool(one_sided)
        self.edit_last_token_only = bool(edit_last_token_only)

    def forward(
        self,
        value: torch.Tensor,
        gate: torch.Tensor | float,
        boundary_state: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_state is None:
            raise ValueError("v4 transport requires a task-boundary state")
        original_dtype = value.dtype
        device = value.device
        value_float = value.float()
        trigger_basis = self.history_trigger_basis.to(device=device)
        context_basis = self.context_basis.to(device=device)
        output_basis = self.output_basis.to(device=device)
        transport = self.transport.to(device=device)
        threshold = self.threshold.to(device=device)
        history_center = self.history_center.to(device=device)
        context_center = self.context_center.to(device=device)
        context_scale = self.context_scale.to(device=device)
        history_reference = self.history_reference.to(device=device)

        boundary = boundary_state.to(device=device, dtype=torch.float32)
        if boundary.ndim == 1:
            boundary = boundary.unsqueeze(0)
        if boundary.ndim != 2 or boundary.shape[-1] != trigger_basis.shape[0]:
            raise ValueError("boundary_state must have shape [batch, hidden] or [hidden]")
        if boundary.shape[0] not in (1, value.shape[0]):
            raise ValueError("boundary_state batch is incompatible with value batch")
        if boundary.shape[0] == 1 and value.shape[0] != 1:
            boundary = boundary.expand(value.shape[0], -1)
        history_scores = (boundary - history_center) @ trigger_basis
        if history_reference.numel():
            history_features = history_reference.unsqueeze(0).expand(boundary.shape[0], -1)
        else:
            history_features = (
                torch.relu(history_scores - threshold) if self.one_sided else history_scores
            )

        if value_float.ndim == 2:
            token_value = value_float.unsqueeze(1)
            squeeze_tokens = True
        elif value_float.ndim == 3:
            token_value = value_float
            squeeze_tokens = False
        else:
            raise ValueError("value must have shape [batch, hidden] or [batch, tokens, hidden]")
        if context_state is None:
            context_value = token_value
        else:
            context_value = context_state.to(device=device, dtype=torch.float32)
            if context_value.ndim == 2:
                context_value = context_value.unsqueeze(1)
            if context_value.shape != token_value.shape:
                raise ValueError("context_state must match the edited value shape")
        history_tokens = history_features.unsqueeze(1).expand(-1, token_value.shape[1], -1)
        if context_basis.shape[1]:
            context_scores = (context_value - context_center) @ context_basis
            context_code = torch.tanh(context_scores / context_scale)
            features = bilinear_transport_features(
                history_tokens,
                torch.zeros_like(self.threshold, device=device),
                context_code,
                one_sided=False,
            )
        else:
            features = history_tokens
        correction = (features @ transport) @ output_basis.transpose(0, 1)
        if self.max_relative_correction is not None:
            correction_norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
            value_norm = torch.linalg.vector_norm(token_value, dim=-1, keepdim=True)
            budget = self.max_relative_correction * value_norm
            scale = torch.clamp(budget / torch.clamp(correction_norm, min=1e-12), max=1.0)
            correction = correction * scale
        if self.edit_last_token_only and token_value.shape[1] > 1:
            token_mask = torch.zeros(
                (*token_value.shape[:-1], 1),
                device=device,
                dtype=torch.float32,
            )
            token_mask[:, -1] = 1.0
            correction = correction * token_mask
        gate_tensor = torch.as_tensor(gate, device=device, dtype=torch.float32)
        while gate_tensor.ndim < correction.ndim:
            gate_tensor = gate_tensor.unsqueeze(-1)
        edited = token_value - gate_tensor * correction
        if squeeze_tokens:
            edited = edited.squeeze(1)
        return edited.to(original_dtype)


@dataclass(frozen=True)
class LayerOperatorSpec:
    layer: int
    component: str
    rank: int


class MultiLayerContextOperator:
    """Registers reversible component hooks for one suffix forward."""

    def __init__(self, operators: dict[LayerOperatorSpec, nn.Module]):
        self.operators = operators
        self._handles = []
        self.last_stats: dict[LayerOperatorSpec, dict[str, float]] = {}

    def to(self, device: torch.device | str):
        for operator in self.operators.values():
            operator.to(device)
        return self

    @staticmethod
    def _replace_output(output, edited):
        return (edited,) + output[1:] if isinstance(output, tuple) else edited

    def install(
        self,
        model,
        gate: torch.Tensor | float | dict[LayerOperatorSpec, torch.Tensor | float],
        module_resolver,
        boundary_state: torch.Tensor | dict[LayerOperatorSpec, torch.Tensor] | None = None,
        collect_diagnostics: bool = False,
    ) -> None:
        if self._handles:
            raise RuntimeError("operator hooks are already installed")
        self.last_stats.clear()
        for spec, operator in self.operators.items():
            module = module_resolver(model, spec.layer, spec.component)
            current_boundary = (
                boundary_state.get(spec)
                if isinstance(boundary_state, dict)
                else boundary_state
            )
            current_gate = gate.get(spec) if isinstance(gate, dict) else gate
            if current_gate is None:
                raise ValueError(f"missing gate for operator site {spec}")
            context_holder: dict[str, torch.Tensor] = {}

            def pre_hook(_module, inputs, kwargs, holder=context_holder):
                value = inputs[0] if inputs else kwargs.get("hidden_states")
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError("operator site did not expose a hidden-state tensor")
                holder["value"] = value

            self._handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))

            def hook(
                _module,
                _inputs,
                output,
                current=operator,
                layer_boundary=current_boundary,
                layer_gate=current_gate,
                layer_spec=spec,
                holder=context_holder,
            ):
                value = output[0] if isinstance(output, tuple) else output
                if "value" not in holder:
                    raise RuntimeError(f"missing component input for operator site {layer_spec}")
                edited = current(
                    value,
                    layer_gate,
                    boundary_state=layer_boundary,
                    context_state=holder.pop("value"),
                )
                if collect_diagnostics:
                    difference = (edited.float() - value.float()).reshape(value.shape[0], -1)
                    original = value.float().reshape(value.shape[0], -1)
                    relative = torch.linalg.vector_norm(difference, dim=-1) / torch.clamp(
                        torch.linalg.vector_norm(original, dim=-1), min=1e-12
                    )
                    self.last_stats[layer_spec] = {
                        "mean_relative_intervention_norm": float(relative.mean().detach().cpu()),
                        "max_relative_intervention_norm": float(relative.max().detach().cpu()),
                    }
                return self._replace_output(output, edited)

            self._handles.append(module.register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.remove()
