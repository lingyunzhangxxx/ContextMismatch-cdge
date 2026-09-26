from __future__ import annotations

import importlib.util
import unittest

from scripts.benchmark_v8.analyze_v4_protected_controls_diagnostic import _identity, _routing


HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch

    from scripts.benchmark_v4.directional_governance import (
        DirectionalGovernanceSite,
        DirectionalGovernanceSiteEditor,
        DirectionalSingleSiteGovernanceEditor,
    )
    from scripts.benchmark_v4.run_directional_controls import ForcedDirectionalSiteEditor
    from scripts.benchmark_v5.abstaining_directional_governance import (
        AbstainingDirectionalSiteEditor,
        ApplicationVetoHead,
    )
    from scripts.benchmark_v5.abstaining_split import APPLICATION_FEATURE_WIDTH


class V4ProtectedControlsAnalysisTests(unittest.TestCase):
    def test_identity_counts_nonzero_rows(self):
        rows = [
            {"application_gated_selected_logit_error": 0.0},
            {"application_gated_selected_logit_error": 0.25},
            {"application_gated_selected_logit_error": 0.0},
        ]
        report = _identity(rows)
        self.assertEqual(report["changed_rows"], 1)
        self.assertAlmostEqual(report["changed_fraction"], 1 / 3)
        self.assertFalse(report["exact"])

    def test_routing_counts_structural_application_and_composite(self):
        rows = [
            {"application_gated_controller_diagnostics": {
                "structural_gate_active": 1.0,
                "application_gate_active": 1.0,
                "v4_gate_active": 1.0,
            }},
            {"application_gated_controller_diagnostics": {
                "structural_gate_active": 1.0,
                "application_gate_active": 0.0,
                "v4_gate_active": 0.0,
            }},
            {"application_gated_controller_diagnostics": {}},
        ]
        report = _routing(rows)
        self.assertEqual(report["structural_gate_active_rows"], 2)
        self.assertEqual(report["application_gate_active_rows"], 1)
        self.assertEqual(report["v4_gate_active_rows"], 1)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for V4 protected-control tests")
class V4ProtectedControlsTorchTests(unittest.TestCase):
    def _v3(self) -> DirectionalSingleSiteGovernanceEditor:
        hidden = 64
        site_editor = DirectionalGovernanceSiteEditor(
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
            site_editor.history_head.weight.zero_()
            site_editor.task_head.weight.zero_()
            site_editor.history_head.bias.fill_(8.0)
            site_editor.task_head.bias.fill_(-8.0)
            site_editor.positive_expert.network[-1].weight.zero_()
            site_editor.positive_expert.network[-1].bias.fill_(1.0)
        return DirectionalSingleSiteGovernanceEditor(
            DirectionalGovernanceSite(27, "mlp", 16, 16), site_editor
        )

    def test_forced_expert_bypasses_v4_application_veto(self):
        v3 = self._v3()
        head = ApplicationVetoHead(
            feature_center=torch.zeros(APPLICATION_FEATURE_WIDTH),
            feature_scale=torch.ones(APPLICATION_FEATURE_WIDTH),
            hidden_widths=[],
            threshold_logit=0.0,
        )
        with torch.no_grad():
            head.network[-1].weight.zero_()
            head.network[-1].bias.fill_(-8.0)
        v4 = AbstainingDirectionalSiteEditor(v3, head)
        value = torch.ones(2, 3, 64)
        boundary = torch.zeros(2, 64)
        context = torch.zeros(2, 3, 64)
        self.assertTrue(torch.equal(v4(value, 1.0, boundary, context), value))
        forced = ForcedDirectionalSiteEditor(v4.v3_editor.site_editor, "positive")
        self.assertFalse(torch.equal(forced(value, 1.0, boundary, context), value))


if __name__ == "__main__":
    unittest.main()
