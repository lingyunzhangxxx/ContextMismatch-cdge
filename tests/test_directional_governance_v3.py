from __future__ import annotations

import unittest
from types import SimpleNamespace


try:
    import torch

    from scripts.benchmark_v4.directional_governance import (
        DirectionalGovernanceSite,
        DirectionalGovernanceSiteEditor,
        DirectionalSingleSiteGovernanceEditor,
        calibrated_probability,
        editor_from_checkpoint,
        structural_directional_routes,
    )
    from scripts.benchmark_v4.fit_directional_governance import (
        _coordinate_dot,
        _coordinate_reconstruction_relative_squared,
        _coordinate_relative_squared,
        _fit_platt,
        _gradient_output_basis,
    )
    from scripts.benchmark_v4.run_margin_gradient_capture import (
        _suffix_margin_gradient_capture,
    )

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for V3 editor tests")
class DirectionalGovernanceV3Tests(unittest.TestCase):
    def _site(self) -> DirectionalGovernanceSiteEditor:
        hidden = 8
        editor = DirectionalGovernanceSiteEditor(
            boundary_basis=torch.eye(hidden)[:, :2],
            context_basis=torch.eye(hidden)[:, 2:4],
            positive_output_basis=torch.eye(hidden)[:, 4:6],
            negative_output_basis=torch.eye(hidden)[:, 6:8],
            boundary_center=torch.zeros(hidden),
            context_center=torch.zeros(hidden),
            boundary_scale=torch.ones(2),
            context_scale=torch.ones(2),
            maximum_relative_correction=0.1,
            activation_threshold=0.5,
            hidden_width=12,
        )
        with torch.no_grad():
            editor.history_head.weight.zero_()
            editor.task_head.weight.zero_()
            editor.history_head.bias.zero_()
            editor.task_head.bias.zero_()
            positive_final = editor.positive_expert.network[-1]
            negative_final = editor.negative_expert.network[-1]
            positive_final.weight.zero_()
            negative_final.weight.zero_()
            positive_final.bias.fill_(1.0)
            negative_final.bias.fill_(-1.0)
        return editor

    def test_matched_and_ambiguous_states_are_structurally_zero(self):
        history = torch.tensor([0.999, 0.001, 0.5])
        task = torch.tensor([0.999, 0.001, 0.5])
        routes = structural_directional_routes(history, task, activation_threshold=0.5)
        self.assertTrue(torch.equal(routes["positive_route"], torch.zeros(3)))
        self.assertTrue(torch.equal(routes["negative_route"], torch.zeros(3)))
        self.assertFalse(bool(routes["structural_gate_active"].any()))

    def test_mismatch_directions_enter_different_experts(self):
        history = torch.tensor([0.999, 0.001])
        task = torch.tensor([0.001, 0.999])
        routes = structural_directional_routes(history, task, activation_threshold=0.5)
        self.assertGreater(float(routes["positive_route"][0]), 0.99)
        self.assertEqual(float(routes["negative_route"][0]), 0.0)
        self.assertEqual(float(routes["positive_route"][1]), 0.0)
        self.assertGreater(float(routes["negative_route"][1]), 0.99)

    def test_calibrated_route_is_invariant_to_positive_logit_rescaling(self):
        history_logit = torch.tensor([3.0, -2.0])
        task_logit = torch.tensor([-4.0, 5.0])
        history = calibrated_probability(history_logit, 0.25, 1.5)
        task = calibrated_probability(task_logit, -0.5, 2.0)
        original = structural_directional_routes(history, task, activation_threshold=0.5)
        factor = 7.0
        rescaled_history = calibrated_probability(
            factor * history_logit, factor * 0.25, factor * 1.5
        )
        rescaled_task = calibrated_probability(
            factor * task_logit, factor * -0.5, factor * 2.0
        )
        rescaled = structural_directional_routes(
            rescaled_history, rescaled_task, activation_threshold=0.5
        )
        self.assertTrue(torch.allclose(original["positive_route"], rescaled["positive_route"]))
        self.assertTrue(torch.allclose(original["negative_route"], rescaled["negative_route"]))

    def test_nonfinite_raw_logit_fails_closed(self):
        with self.assertRaisesRegex(FloatingPointError, "raw classifier logit"):
            calibrated_probability(torch.tensor([float("nan")]), 0.0, 1.0)

    def test_gradient_output_basis_uses_directional_gradient_space(self):
        gradients = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [1.0, 0.1, 0.0, 0.0],
            ]
        )
        protected = torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
            ]
        )
        basis = _gradient_output_basis(gradients, protected, 1, protected_rank=1)
        self.assertEqual(tuple(basis.shape), (4, 1))
        self.assertGreater(float(torch.abs(basis[0, 0])), 0.99)
        self.assertLess(float(torch.abs(basis[2, 0])), 1e-6)

    def test_separate_platt_calibration_is_finite_and_monotone(self):
        raw = torch.tensor([-4.0, -2.0, 2.0, 4.0])
        target = torch.tensor([0.0, 0.0, 1.0, 1.0])
        location, scale, trace = _fit_platt(
            raw,
            target,
            learning_rate=0.05,
            steps=40,
            minimum_scale=1e-4,
        )
        probability = calibrated_probability(raw, location, scale)
        self.assertGreater(scale, 0.0)
        self.assertTrue(bool(torch.all(probability[1:] > probability[:-1])))
        self.assertTrue(trace)

    def test_external_zero_gate_returns_same_tensor_object(self):
        editor = self._site()
        value = torch.randn(2, 3, 8)
        result = editor(
            value,
            0.0,
            boundary_state=torch.randn(2, 8),
            context_state=torch.randn(2, 3, 8),
        )
        self.assertIs(result, value)
        self.assertEqual(editor.last_diagnostics, {})

    def test_matched_structural_gate_is_bitwise_identity(self):
        editor = self._site()
        with torch.no_grad():
            editor.history_head.bias.fill_(8.0)
            editor.task_head.bias.fill_(8.0)
        value = torch.randn(2, 3, 8)
        result = editor(
            value,
            1.0,
            boundary_state=torch.randn(2, 8),
            context_state=torch.randn(2, 3, 8),
        )
        self.assertTrue(torch.equal(result, value))
        self.assertFalse(bool(editor.last_diagnostics["structural_gate_active"].any()))
        self.assertTrue(
            torch.equal(
                editor.last_diagnostics["positive_expert_correction"],
                torch.zeros(2, 8),
            )
        )

    def test_protected_stress_bypasses_router_and_checks_both_experts(self):
        editor = self._site()
        with torch.no_grad():
            editor.history_head.bias.fill_(8.0)
            editor.task_head.bias.fill_(8.0)
        value = torch.ones(2, 8)
        boundary = torch.zeros(2, 8)
        context = torch.zeros(2, 8)
        routed, diagnostics = editor.correction(
            value=value,
            boundary_state=boundary,
            context_state=context,
        )
        self.assertTrue(torch.equal(routed, torch.zeros_like(routed)))
        self.assertFalse(bool(diagnostics["structural_gate_active"].any()))

        positive, positive_other, _ = editor.forced_direction_coordinates(
            direction="positive",
            value=value,
            boundary_state=boundary,
            context_state=context,
        )
        negative_other, negative, _ = editor.forced_direction_coordinates(
            direction="negative",
            value=value,
            boundary_state=boundary,
            context_state=context,
        )
        self.assertGreater(float(torch.linalg.vector_norm(positive)), 0.0)
        self.assertGreater(float(torch.linalg.vector_norm(negative)), 0.0)
        self.assertTrue(torch.equal(positive_other, torch.zeros_like(positive_other)))
        self.assertTrue(torch.equal(negative_other, torch.zeros_like(negative_other)))
        budget = 0.1 * torch.linalg.vector_norm(value, dim=-1)
        self.assertTrue(
            bool(torch.all(torch.linalg.vector_norm(positive, dim=-1) <= budget + 1e-7))
        )
        self.assertTrue(
            bool(torch.all(torch.linalg.vector_norm(negative, dim=-1) <= budget + 1e-7))
        )

    def test_directional_experts_have_independent_output_spaces(self):
        editor = self._site()
        value = torch.ones(2, 8)
        boundary = torch.zeros(2, 8)
        context = torch.zeros(2, 8)
        with torch.no_grad():
            editor.history_head.bias.copy_(torch.tensor(8.0))
            editor.task_head.bias.copy_(torch.tensor(-8.0))
        positive, positive_diag = editor.correction(
            value=value, boundary_state=boundary, context_state=context
        )
        with torch.no_grad():
            editor.history_head.bias.copy_(torch.tensor(-8.0))
            editor.task_head.bias.copy_(torch.tensor(8.0))
        negative, negative_diag = editor.correction(
            value=value, boundary_state=boundary, context_state=context
        )
        self.assertGreater(float(positive_diag["positive_route"].min()), 0.99)
        self.assertGreater(float(negative_diag["negative_route"].min()), 0.99)
        self.assertTrue(torch.equal(positive[:, 6:], torch.zeros_like(positive[:, 6:])))
        self.assertTrue(torch.equal(negative[:, 4:6], torch.zeros_like(negative[:, 4:6])))
        self.assertFalse(torch.equal(positive, -negative))

    def test_low_rank_coordinates_exactly_reconstruct_full_correction(self):
        editor = self._site()
        value = torch.ones(3, 8)
        boundary = torch.zeros(3, 8)
        context = torch.zeros(3, 8)
        with torch.no_grad():
            editor.history_head.bias.fill_(8.0)
            editor.task_head.bias.fill_(-8.0)
        reference_norm = torch.linalg.vector_norm(value.float(), dim=-1)
        positive, negative, coordinate_diagnostics = editor.correction_coordinates(
            value=value,
            boundary_state=boundary,
            context_state=context,
            reference_norm=reference_norm,
        )
        full, full_diagnostics = editor.correction(
            value=value,
            boundary_state=boundary,
            context_state=context,
        )
        reconstructed = (
            positive @ editor.positive_output_basis.transpose(0, 1)
            + negative @ editor.negative_output_basis.transpose(0, 1)
        )
        self.assertTrue(torch.allclose(full, reconstructed, atol=1e-7, rtol=1e-6))
        self.assertTrue(
            torch.allclose(
                coordinate_diagnostics["trust_scale"],
                full_diagnostics["trust_scale"],
                atol=1e-7,
                rtol=1e-6,
            )
        )

    def test_low_rank_norm_gradient_dot_and_teacher_error_match_full_space(self):
        editor = self._site()
        value = torch.arange(1, 25, dtype=torch.float32).reshape(3, 8) / 10.0
        boundary = torch.zeros(3, 8)
        context = torch.zeros(3, 8)
        with torch.no_grad():
            editor.history_head.bias.fill_(-8.0)
            editor.task_head.bias.fill_(8.0)
        positive, negative, _ = editor.correction_coordinates(
            value=value,
            boundary_state=boundary,
            context_state=context,
        )
        full, _ = editor.correction(
            value=value,
            boundary_state=boundary,
            context_state=context,
        )
        value_norm_squared = torch.sum(value.square(), dim=-1)
        coordinate_relative = _coordinate_relative_squared(
            positive,
            negative,
            value_norm_squared,
        )
        full_relative = torch.sum(full.square(), dim=-1) / torch.clamp(
            value_norm_squared, min=1e-6
        )
        self.assertTrue(
            torch.allclose(coordinate_relative, full_relative, atol=1e-7, rtol=1e-6)
        )

        gradient = torch.randn_like(value)
        coordinate_gain = _coordinate_dot(
            positive,
            negative,
            gradient @ editor.positive_output_basis,
            gradient @ editor.negative_output_basis,
        )
        full_gain = torch.sum(gradient * full, dim=-1)
        self.assertTrue(torch.allclose(coordinate_gain, full_gain, atol=1e-6, rtol=1e-5))

        target = torch.randn_like(value)
        coordinate_teacher = _coordinate_reconstruction_relative_squared(
            positive,
            negative,
            target @ editor.positive_output_basis,
            target @ editor.negative_output_basis,
            torch.sum(target.square(), dim=-1),
        )
        full_teacher = torch.sum((full - target).square(), dim=-1) / torch.clamp(
            torch.sum(target.square(), dim=-1), min=1e-6
        )
        self.assertTrue(
            torch.allclose(coordinate_teacher, full_teacher, atol=1e-6, rtol=1e-5)
        )

    def test_zero_initialized_expert_has_finite_first_backward(self):
        editor = self._site()
        with torch.no_grad():
            for expert in (editor.positive_expert, editor.negative_expert):
                final = expert.network[-1]
                final.weight.zero_()
                final.bias.zero_()
            editor.history_head.bias.fill_(8.0)
            editor.task_head.bias.fill_(-8.0)
        positive, negative, diagnostics = editor.correction_coordinates(
            value=torch.ones(4, 8),
            boundary_state=torch.randn(4, 8),
            context_state=torch.randn(4, 8),
        )
        self.assertTrue(torch.equal(positive, torch.zeros_like(positive)))
        self.assertTrue(torch.equal(negative, torch.zeros_like(negative)))
        self.assertTrue(torch.equal(diagnostics["trust_scale"], torch.ones(4)))
        loss = -(positive.sum() + negative.sum())
        loss.backward()
        gradients = [parameter.grad for parameter in editor.parameters() if parameter.requires_grad]
        self.assertTrue(all(value is not None for value in gradients))
        self.assertTrue(all(bool(torch.isfinite(value).all()) for value in gradients))
        self.assertTrue(any(float(torch.linalg.vector_norm(value)) > 0 for value in gradients))

    def test_checkpoint_refuses_multiple_sites_and_contains_no_base_weights(self):
        editor = self._site()
        site = DirectionalGovernanceSite(31, "mlp", 2, 2)
        wrapper = DirectionalSingleSiteGovernanceEditor(site, editor)
        record = {
            "layer": 31,
            "component": "mlp",
            "positive_output_rank": 2,
            "negative_output_rank": 2,
            "maximum_relative_correction": 0.1,
            "activation_threshold": 0.5,
            "hidden_width": 12,
            "head_input_mode": "full_state",
            "constructor_tensors": {
                "boundary_basis": editor.boundary_basis,
                "context_basis": editor.context_basis,
                "positive_output_basis": editor.positive_output_basis,
                "negative_output_basis": editor.negative_output_basis,
                "boundary_center": editor.boundary_center,
                "context_center": editor.context_center,
                "boundary_scale": editor.boundary_scale,
                "context_scale": editor.context_scale,
                "boundary_head_scale": editor.boundary_head_scale,
                "context_head_scale": editor.context_head_scale,
                "history_logit_location": float(editor.history_logit_location),
                "history_logit_scale": float(editor.history_logit_scale),
                "task_logit_location": float(editor.task_logit_location),
                "task_logit_scale": float(editor.task_logit_scale),
            },
            "state_dict": editor.state_dict(),
        }
        checkpoint = {
            "schema_version": 2,
            "kind": "directional_structural_governance_editor",
            "method": "DSGE-V3",
            "base_model_weights_included": False,
            "trainable_parameter_count": wrapper.trainable_parameter_count,
            "sites": [record],
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        rebuilt = editor_from_checkpoint(checkpoint)
        self.assertEqual(rebuilt.site, site)
        self.assertFalse(checkpoint["base_model_weights_included"])
        self.assertFalse(
            any(name.startswith(("model.", "base_model.")) for name in record["state_dict"])
        )
        checkpoint["sites"] = [record, record]
        with self.assertRaisesRegex(ValueError, "exactly one selected site"):
            editor_from_checkpoint(checkpoint)

    def test_checkpoint_refuses_open_final_test(self):
        editor = self._site()
        record = {
            "layer": 31,
            "component": "mlp",
            "positive_output_rank": 2,
            "negative_output_rank": 2,
            "maximum_relative_correction": 0.1,
            "activation_threshold": 0.5,
            "hidden_width": 12,
            "head_input_mode": "full_state",
            "constructor_tensors": {
                "boundary_basis": editor.boundary_basis,
                "context_basis": editor.context_basis,
                "positive_output_basis": editor.positive_output_basis,
                "negative_output_basis": editor.negative_output_basis,
                "boundary_center": editor.boundary_center,
                "context_center": editor.context_center,
                "boundary_scale": editor.boundary_scale,
                "context_scale": editor.context_scale,
                "boundary_head_scale": editor.boundary_head_scale,
                "context_head_scale": editor.context_head_scale,
                "history_logit_location": float(editor.history_logit_location),
                "history_logit_scale": float(editor.history_logit_scale),
                "task_logit_location": float(editor.task_logit_location),
                "task_logit_scale": float(editor.task_logit_scale),
            },
            "state_dict": editor.state_dict(),
        }
        checkpoint = {
            "schema_version": 2,
            "kind": "directional_structural_governance_editor",
            "method": "DSGE-V3",
            "base_model_weights_included": False,
            "trainable_parameter_count": editor.trainable_parameter_count,
            "sites": [record],
            "final_test_open": True,
            "final_test_open_count": 1,
            "production_rollout_approved": False,
        }
        with self.assertRaisesRegex(ValueError, "opens final test"):
            editor_from_checkpoint(checkpoint)

    def test_true_margin_gradient_capture_backpropagates_through_frozen_model(self):
        class TinyBlock(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.self_attn = torch.nn.Linear(width, width, bias=False)
                self.mlp = torch.nn.Linear(width, width, bias=False)

            def forward(self, value):
                return self.mlp(self.self_attn(value))

        class TinyInner(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.layers = torch.nn.ModuleList([TinyBlock(width), TinyBlock(width)])

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = TinyInner(4)
                self.embedding = torch.nn.Embedding(16, 4)
                self.output = torch.nn.Linear(4, 3, bias=False)

            def forward(self, input_ids, **_kwargs):
                value = self.embedding(input_ids)
                for block in self.model.layers:
                    value = block(value)
                return SimpleNamespace(logits=self.output(value), past_key_values=None)

        torch.manual_seed(3)
        model = TinyModel()
        model.requires_grad_(False)
        gradients, logits = _suffix_margin_gradient_capture(
            model,
            torch.device("cpu"),
            None,
            2,
            [3, 4],
            {"A": 0, "B": 1},
            [(0, "self_attn"), (1, "mlp")],
            "A",
            "B",
        )
        self.assertEqual(set(gradients), {(0, "self_attn"), (1, "mlp")})
        self.assertTrue(all(tuple(value.shape) == (4,) for value in gradients.values()))
        self.assertTrue(
            all(float(torch.linalg.vector_norm(value.float())) > 0 for value in gradients.values())
        )
        self.assertTrue(all(torch.isfinite(torch.tensor(logits))))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
