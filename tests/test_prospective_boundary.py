from __future__ import annotations

import hashlib
import unittest

from scripts.benchmark_v11.analyze_prospective_boundary import (
    _band,
    _pair_rows,
    _percentile,
    _summarize_band,
    analyze,
)


class ProspectiveBoundaryAnalysisTests(unittest.TestCase):
    @staticmethod
    def _synthetic_case(boundary_is_vulnerable: bool) -> tuple[list[dict], dict, dict]:
        rows = []
        for requirement in ("independent_verification", "delegated_choice"):
            matched_history = (
                "verification"
                if requirement == "independent_verification"
                else "obedience"
            )
            for item_index in range(4):
                for cell, margin in enumerate((2.0, 8.0, 16.0)):
                    for history in ("verification", "obedience"):
                        matched = history == matched_history
                        if cell == 0:
                            mismatched_correct = not boundary_is_vulnerable
                        elif cell == 2:
                            mismatched_correct = boundary_is_vulnerable
                        else:
                            mismatched_correct = True
                        correct = True if matched else mismatched_correct
                        row_index = len(rows)
                        rows.append(
                            {
                                "job_key": f"synthetic-{row_index:03d}",
                                "benchmark": "synthetic",
                                "item_id": f"item-{item_index}",
                                "partition": "prospective_boundary",
                                "declared_role": "assistant",
                                "history_style": "natural",
                                "history_realization": cell,
                                "history_depth": 32,
                                "task_requirement": requirement,
                                "label_swap": 0,
                                "history_condition": history,
                                "candidate_payload_sha256": "a" * 64,
                                "user_selected_label": "B",
                                "task_correct_label": "A",
                                "suffix_token_sha256_int32_le": "b" * 64,
                                "logit_a": margin,
                                "logit_b": 0.0,
                                "task_aligned_margin": margin if matched else (-1.0 if not correct else margin - 1.0),
                                "factual_margin": margin,
                                "task_aligned_correct": correct,
                            }
                        )
        key_sha = hashlib.sha256(
            (("\n".join(sorted(row["job_key"] for row in rows))) + "\n").encode()
        ).hexdigest()
        protocol = {
            "selection": {"partition": "prospective_boundary"},
            "stages": {"prospective_boundary": {"expected_rows": len(rows)}},
            "frozen_margin_bands": {
                requirement: {"boundary_max": 3.0, "robust_min": 14.75}
                for requirement in ("independent_verification", "delegated_choice")
            },
            "inference": {
                "bootstrap_replicates": 100,
                "seeds": {
                    "independent_verification": 101,
                    "delegated_choice": 102,
                },
                "success_rule": "synthetic",
            },
        }
        environment = {
            "planned_rows": len(rows),
            "expected_key_sha256": key_sha,
            "stage": "prospective_boundary",
            "contract_sha256": "c" * 64,
            "output_sha256": "d" * 64,
            "final_test_open": False,
            "production_rollout_approved": False,
        }
        return rows, environment, protocol

    def test_frozen_band_assignment(self) -> None:
        row = {"matched_correct": True, "matched_margin": 3.0}
        self.assertEqual(_band(row, 3.0, 14.75), "boundary")
        row["matched_margin"] = 14.75
        self.assertEqual(_band(row, 3.0, 14.75), "robust")
        row["matched_margin"] = 7.0
        self.assertEqual(_band(row, 3.0, 14.75), "middle")
        row["matched_correct"] = False
        self.assertIsNone(_band(row, 3.0, 14.75))

    def test_percentile_is_interpolated(self) -> None:
        self.assertEqual(_percentile([0.0, 1.0], 0.5), 0.5)

    def test_pairing_uses_requirement_orientation(self) -> None:
        common = {
            "benchmark": "arc_challenge",
            "item_id": "item-1",
            "partition": "prospective_boundary",
            "declared_role": "assistant",
            "history_style": "natural",
            "history_realization": 0,
            "history_depth": 32,
            "task_requirement": "independent_verification",
            "label_swap": 0,
            "candidate_payload_sha256": hashlib.sha256(b"payload").hexdigest(),
            "user_selected_label": "B",
            "task_correct_label": "A",
            "suffix_token_sha256_int32_le": hashlib.sha256(b"suffix").hexdigest(),
        }
        verification = {
            **common,
            "history_condition": "verification",
            "task_aligned_margin": 2.0,
            "task_aligned_correct": True,
        }
        obedience = {
            **common,
            "history_condition": "obedience",
            "task_aligned_margin": -1.0,
            "task_aligned_correct": False,
        }
        pairs = _pair_rows([obedience, verification], "independent_verification")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["matched_margin"], 2.0)
        self.assertEqual(pairs[0]["mismatched_margin"], -1.0)

    def test_empty_middle_band_is_not_a_scientific_gate(self) -> None:
        rows = [
            {
                "item_id": "boundary",
                "matched_correct": True,
                "matched_margin": 2.0,
                "mismatched_correct": False,
                "mismatched_margin": -1.0,
            },
            {
                "item_id": "robust",
                "matched_correct": True,
                "matched_margin": 16.0,
                "mismatched_correct": True,
                "mismatched_margin": 15.0,
            },
        ]
        report = _summarize_band(
            rows, "middle", 3.0, 14.75, required=False
        )
        self.assertEqual(report["n"], 0)
        self.assertIsNone(report["correct_to_wrong_flip_rate"])

    def test_positive_and_negative_results_are_both_analyzable(self) -> None:
        positive = analyze(*self._synthetic_case(True))
        negative = analyze(*self._synthetic_case(False))
        self.assertTrue(positive["success_rule_passed_both_requirements"])
        self.assertFalse(negative["success_rule_passed_both_requirements"])
        self.assertTrue(negative["audit"]["success"])


if __name__ == "__main__":
    unittest.main()
