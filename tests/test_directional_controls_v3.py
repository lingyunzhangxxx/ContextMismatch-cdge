from __future__ import annotations

import copy
import hashlib
import unittest

from scripts.benchmark_v4.analyze_directional_controls import FAMILIES, analyze

try:
    import torch

    from scripts.benchmark_v4.directional_governance import DirectionalGovernanceSiteEditor
    from scripts.benchmark_v4.run_directional_controls import ForcedDirectionalSiteEditor

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for directional control hooks")
class DirectionalControlsV3Tests(unittest.TestCase):
    def _editor(self) -> DirectionalGovernanceSiteEditor:
        torch.manual_seed(19)
        identity = torch.eye(8)
        editor = DirectionalGovernanceSiteEditor(
            boundary_basis=identity[:, :2],
            context_basis=identity[:, 2:4],
            positive_output_basis=identity[:, 4:6],
            negative_output_basis=identity[:, 6:8],
            boundary_center=torch.zeros(8),
            context_center=torch.zeros(8),
            boundary_scale=torch.ones(2),
            context_scale=torch.ones(2),
            maximum_relative_correction=0.1,
            activation_threshold=0.5,
            hidden_width=6,
        )
        with torch.no_grad():
            editor.history_head.weight.zero_()
            editor.task_head.weight.zero_()
            editor.history_head.bias.fill_(8.0)
            editor.task_head.bias.fill_(8.0)
            for expert in (editor.positive_expert, editor.negative_expert):
                final = expert.network[-1]
                final.weight.zero_()
                final.bias.fill_(1.0)
        return editor.eval()

    def test_forced_positive_bypasses_structural_abstention(self):
        editor = self._editor()
        value = torch.ones(1, 3, 8)
        boundary = torch.zeros(1, 8)
        context = torch.zeros(1, 3, 8)
        self.assertTrue(torch.equal(editor(value, 1.0, boundary, context), value))
        forced = ForcedDirectionalSiteEditor(editor, "positive")
        edited = forced(value, 1.0, boundary, context)
        self.assertFalse(torch.equal(edited, value))
        self.assertTrue(torch.equal(edited[:, :-1], value[:, :-1]))
        self.assertGreater(float(torch.linalg.vector_norm(edited[:, -1] - value[:, -1])), 0.0)

    def test_forced_negative_is_independent_and_zero_external_gate_is_exact(self):
        editor = self._editor()
        value = torch.ones(2, 8)
        boundary = torch.zeros(2, 8)
        context = torch.zeros(2, 8)
        forced = ForcedDirectionalSiteEditor(editor, "negative")
        self.assertTrue(torch.equal(forced(value, 0.0, boundary, context), value))
        edited = forced(value, 1.0, boundary, context)
        self.assertFalse(torch.equal(edited, value))
        self.assertTrue(torch.equal(edited[:, :6], value[:, :6]))

class DirectionalControlsAnalysisTests(unittest.TestCase):
    @staticmethod
    def _rows() -> list[dict]:
        rows = []
        per_family = 2856 // len(FAMILIES)
        for family in FAMILIES:
            for index in range(per_family):
                base = {
                    "logit_a": 1.0,
                    "logit_b": 0.0,
                    "correct_margin": 1.0,
                    "correct_probability_binary": 0.7310585786,
                }
                rows.append(
                    {
                        "case_key": f"{family}__{index}",
                        "case_id": f"{family}__{index}",
                        "control_family": family,
                        "operator_applicable": 0.0
                        if family == "factual_boundary_memory"
                        else 1.0,
                        "baseline": copy.deepcopy(base),
                        "application_gated": copy.deepcopy(base),
                        "application_gated_selected_logit_error": 0.0,
                        "application_gated_margin_change": 0.0,
                        "application_gated_binary_kl": 0.0,
                        "forced_on_positive": copy.deepcopy(base),
                        "forced_on_positive_margin_change": 0.0,
                        "forced_on_positive_binary_kl": 0.0,
                        "forced_on_negative": copy.deepcopy(base),
                        "forced_on_negative_margin_change": 0.0,
                        "forced_on_negative_binary_kl": 0.0,
                    }
                )
        return rows

    @staticmethod
    def _environment(rows: list[dict]) -> dict:
        digest = hashlib.sha256(
            (("\n".join(sorted(row["case_key"] for row in rows))) + "\n").encode()
        ).hexdigest()
        return {
            "candidate_id": "DSGE-V3-27:mlp",
            "site": "27:mlp",
            "checkpoint_sha256": "a" * 64,
            "fit_report_sha256": "b" * 64,
            "fit_audit_analysis_sha256": "c" * 64,
            "operator_dev_analysis_sha256": "d" * 64,
            "planned_rows": 2856,
            "expected_key_sha256": digest,
            "forced_directions": ["positive", "negative"],
            "prior_candidate_eligible": True,
        }

    @staticmethod
    def _contract() -> dict:
        return {
            "protected_controls": {
                "forced_on_margin_loss_upper_95_maximum": 0.25,
                "forced_on_binary_kl_upper_95_maximum": 0.02,
            }
        }

    def test_complete_zero_collateral_controls_are_lock_eligible(self):
        rows = self._rows()
        report = analyze(rows, self._environment(rows), self._contract(), replicates=20)
        self.assertTrue(report["audit"]["success"])
        self.assertTrue(report["every_family_passes_individually"])
        self.assertTrue(report["candidate_may_be_locked"])

    def test_one_forced_direction_failure_fails_closed(self):
        rows = self._rows()
        for row in rows:
            if row["control_family"] == "fresh_verification":
                row["forced_on_positive_margin_change"] = -1.0
        report = analyze(rows, self._environment(rows), self._contract(), replicates=20)
        self.assertTrue(report["audit"]["success"])
        self.assertFalse(report["family_reports"]["fresh_verification"]["family_pass"])
        self.assertFalse(report["candidate_may_be_locked"])


if __name__ == "__main__":
    unittest.main()
