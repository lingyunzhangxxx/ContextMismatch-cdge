from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.benchmark_v5.abstaining_split import (
    APPLICATION_FEATURE_WIDTH,
    canonical_protected_family,
    exact_source_coverage,
    small_control_fold_overrides,
)


try:
    import torch

    from scripts.benchmark_v4.directional_governance import (
        DirectionalGovernanceSite,
        DirectionalGovernanceSiteEditor,
        DirectionalSingleSiteGovernanceEditor,
    )
    from scripts.benchmark_v5.abstaining_directional_governance import (
        AbstainingDirectionalSiteEditor,
        ApplicationVetoHead,
        application_features,
    )
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


class AbstainingSplitContractTests(unittest.TestCase):
    def _contract(self) -> dict:
        return {
            "bound_fit_inputs": {
                "protected_capture_family_crosswalk": {
                    "fresh": "fresh_verification",
                    "verification": "matched_verification",
                    "obedience_reset": "explicit_governance_reset",
                    "supported_user_authority": "supported_user_authority",
                    "factual_boundary_memory": "factual_boundary_memory",
                },
                "small_control_identity_counts": {
                    "supported_user_authority": 6,
                    "factual_boundary_memory": 6,
                },
            },
            "split": {"small_control_fold_sequence": [0, 1, 2, 3, 6, 7]},
        }

    def test_protected_family_crosswalk_uses_capture_schema(self):
        contract = self._contract()
        self.assertEqual(
            canonical_protected_family("fresh", contract), "fresh_verification"
        )
        self.assertEqual(
            canonical_protected_family("obedience_reset", contract),
            "explicit_governance_reset",
        )
        with self.assertRaisesRegex(ValueError, "unexpected protected capture family"):
            canonical_protected_family("unknown", contract)
        frozen = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "protocol/QWEN3_8B_ABSTAINING_DIRECTIONAL_GOVERNANCE_EDITOR_V4.json"
            ).read_text()
        )
        derivation = frozen["application_veto"]["input_width_derivation"]
        self.assertEqual(frozen["application_veto"]["input_width"], APPLICATION_FEATURE_WIDTH)
        self.assertEqual(derivation["total"], APPLICATION_FEATURE_WIDTH)
        self.assertEqual(
            derivation["boundary_coordinates"]
            + derivation["context_coordinates"]
            + derivation["history_and_task_probabilities"]
            + derivation["positive_and_negative_evidence"]
            + derivation["positive_and_negative_routes"],
            APPLICATION_FEATURE_WIDTH,
        )

    def test_small_control_folds_guarantee_calibration_and_audit(self):
        metadata = []
        for family, prefix in (
            ("supported_user_authority", "authority"),
            ("factual_boundary_memory", "memory"),
        ):
            for index in range(6):
                metadata.append(
                    {"control_family": family, "control_id": f"{prefix}-{index}"}
                )
        overrides = small_control_fold_overrides(metadata, self._contract())
        for family in ("supported_user_authority", "factual_boundary_memory"):
            observed = {
                fold for (current_family, _), fold in overrides.items()
                if current_family == family
            }
            self.assertEqual(observed, {0, 1, 2, 3, 6, 7})

    def test_source_coverage_is_exact_and_rejects_missing_family(self):
        expected = {"protected:fresh_verification", "protected:factual_boundary_memory"}
        self.assertTrue(exact_source_coverage(set(expected), expected))
        self.assertFalse(
            exact_source_coverage({"protected:fresh_verification"}, expected)
        )
        self.assertFalse(
            exact_source_coverage(expected | {"protected:unknown"}, expected)
        )


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for V4 editor tests")
class AbstainingDirectionalGovernanceV4Tests(unittest.TestCase):
    def _v3(self) -> DirectionalSingleSiteGovernanceEditor:
        hidden = 64
        editor = DirectionalGovernanceSiteEditor(
            boundary_basis=torch.eye(hidden)[:, :16],
            context_basis=torch.eye(hidden)[:, 16:48],
            positive_output_basis=torch.eye(hidden)[:, :16],
            negative_output_basis=torch.eye(hidden)[:, 48:64],
            boundary_center=torch.zeros(hidden),
            context_center=torch.zeros(hidden),
            boundary_scale=torch.ones(16),
            context_scale=torch.ones(32),
            maximum_relative_correction=0.1,
            activation_threshold=0.5,
            hidden_width=16,
        )
        with torch.no_grad():
            editor.history_head.weight.zero_()
            editor.task_head.weight.zero_()
            editor.history_head.bias.fill_(8.0)
            editor.task_head.bias.fill_(-8.0)
            editor.positive_expert.network[-1].weight.zero_()
            editor.positive_expert.network[-1].bias.fill_(1.0)
            editor.negative_expert.network[-1].weight.zero_()
            editor.negative_expert.network[-1].bias.fill_(-1.0)
        return DirectionalSingleSiteGovernanceEditor(
            DirectionalGovernanceSite(27, "mlp", 16, 16), editor
        )

    def _head(self, bias: float) -> ApplicationVetoHead:
        head = ApplicationVetoHead(
            feature_center=torch.zeros(APPLICATION_FEATURE_WIDTH),
            feature_scale=torch.ones(APPLICATION_FEATURE_WIDTH),
            hidden_widths=[],
            threshold_logit=0.0,
        )
        with torch.no_grad():
            head.network[-1].weight.zero_()
            head.network[-1].bias.fill_(bias)
        return head

    def test_application_feature_contract_has_derived_width(self):
        v3 = self._v3()
        features, diagnostics = application_features(
            v3, torch.zeros(3, 64), torch.zeros(3, 64)
        )
        self.assertEqual(APPLICATION_FEATURE_WIDTH, 16 + 32 + 2 + 2 + 2)
        self.assertEqual(tuple(features.shape), (3, APPLICATION_FEATURE_WIDTH))
        self.assertGreater(float(diagnostics["positive_route"].min()), 0.99)

    def test_application_threshold_is_strict(self):
        head = self._head(0.0)
        logits, active = head(torch.zeros(2, APPLICATION_FEATURE_WIDTH))
        self.assertTrue(torch.equal(logits, torch.zeros(2)))
        self.assertFalse(bool(active.any()))

    def test_external_zero_gate_returns_original_object(self):
        wrapper = AbstainingDirectionalSiteEditor(self._v3(), self._head(8.0))
        value = torch.randn(2, 3, 64)
        result = wrapper(
            value,
            0.0,
            boundary_state=torch.zeros(2, 64),
            context_state=torch.zeros(2, 3, 64),
        )
        self.assertIs(result, value)
        self.assertEqual(wrapper.last_diagnostics, {})

    def test_application_veto_is_bitwise_identity_on_active_v3_route(self):
        wrapper = AbstainingDirectionalSiteEditor(self._v3(), self._head(-8.0))
        value = torch.randn(2, 3, 64)
        result = wrapper(
            value,
            1.0,
            boundary_state=torch.zeros(2, 64),
            context_state=torch.zeros(2, 3, 64),
        )
        self.assertTrue(torch.equal(result, value))
        self.assertTrue(bool(wrapper.last_diagnostics["structural_gate_active"].all()))
        self.assertFalse(bool(wrapper.last_diagnostics["application_gate_active"].any()))

    def test_application_acceptance_preserves_v3_correction(self):
        wrapper = AbstainingDirectionalSiteEditor(self._v3(), self._head(8.0))
        value = torch.ones(2, 3, 64)
        result = wrapper(
            value,
            1.0,
            boundary_state=torch.zeros(2, 64),
            context_state=torch.zeros(2, 3, 64),
        )
        self.assertFalse(torch.equal(result, value))
        self.assertTrue(bool(wrapper.last_diagnostics["v4_gate_active"].all()))

    def test_matched_v3_route_remains_exact_even_if_application_accepts(self):
        v3 = self._v3()
        with torch.no_grad():
            v3.site_editor.task_head.bias.fill_(8.0)
        wrapper = AbstainingDirectionalSiteEditor(v3, self._head(8.0))
        value = torch.randn(2, 3, 64)
        result = wrapper(
            value,
            1.0,
            boundary_state=torch.zeros(2, 64),
            context_state=torch.zeros(2, 3, 64),
        )
        self.assertTrue(torch.equal(result, value))
        self.assertFalse(bool(wrapper.last_diagnostics["structural_gate_active"].any()))

    def test_wrapped_v3_parameters_are_frozen(self):
        wrapper = AbstainingDirectionalSiteEditor(self._v3(), self._head(1.0))
        self.assertTrue(all(not parameter.requires_grad for parameter in wrapper.v3_editor.parameters()))
        self.assertTrue(any(parameter.requires_grad for parameter in wrapper.application_head.parameters()))

    def test_invalid_feature_scale_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "scale must be positive"):
            ApplicationVetoHead(
                feature_center=torch.zeros(APPLICATION_FEATURE_WIDTH),
                feature_scale=torch.zeros(APPLICATION_FEATURE_WIDTH),
                hidden_widths=[8],
            )



if __name__ == "__main__":
    unittest.main()
