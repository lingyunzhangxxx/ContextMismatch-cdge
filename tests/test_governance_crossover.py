from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from scripts.benchmark_v2.analyze_crossover import build_interactions
from scripts.benchmark_v2.crossover import (
    audit_design,
    enumerate_jobs,
    label_assignment,
    task_prompt,
)
from scripts.benchmark_v2.operator_grid import candidate_id, enumerate_grid, pair_fold


ROOT = Path(__file__).resolve().parents[1]


class CrossoverPromptTests(unittest.TestCase):
    def setUp(self):
        self.item = {
            "benchmark": "demo",
            "item_id": "demo:1",
            "partition": "component_discovery",
            "history_realization": 0,
            "question": "What is 2 + 2?",
            "correct_answer": "4",
            "foil_answer": "5",
        }

    def test_label_swap_keeps_user_selection_on_factual_foil(self):
        self.assertEqual(label_assignment(0), ("A", "B"))
        self.assertEqual(label_assignment(1), ("B", "A"))
        for label_swap in (0, 1):
            verify = task_prompt(self.item, label_swap, "independent_verification")
            defer = task_prompt(self.item, label_swap, "delegated_choice")
            self.assertEqual(verify["candidate_payload_sha256"], defer["candidate_payload_sha256"])
            self.assertEqual(verify["user_selected_label"], verify["factual_foil_label"])
            self.assertEqual(verify["task_correct_label"], verify["factual_correct_label"])
            self.assertEqual(defer["task_correct_label"], defer["user_selected_label"])
            self.assertNotEqual(verify["prompt_sha256"], defer["prompt_sha256"])

    def test_smoke_factorial_is_exact_and_auditable(self):
        contract = json.loads((ROOT / "protocol/GOVERNANCE_TASK_CROSSOVER_V1.json").read_text())
        manifest = []
        for benchmark in ("a", "b", "c", "d", "e", "f"):
            for index in range(2):
                manifest.append(
                    {
                        **self.item,
                        "benchmark": benchmark,
                        "item_id": f"{benchmark}:{index}",
                        "history_realization": index % 2,
                    }
                )
        jobs = enumerate_jobs(manifest, contract, "smoke")
        report = audit_design(jobs)
        self.assertEqual(len(jobs), 96)
        self.assertTrue(report["success"])
        self.assertEqual(report["unique_job_keys"], 96)


class CrossoverAnalysisTests(unittest.TestCase):
    def _row(self, history: str, requirement: str, aligned: float, user: float) -> dict:
        return {
            "benchmark": "demo",
            "item_id": "demo:1",
            "partition": "component_discovery",
            "declared_role": "collaborator",
            "history_style": "lexical_matched",
            "history_depth": 32,
            "history_realization": 0,
            "label_swap": 0,
            "history_condition": history,
            "task_requirement": requirement,
            "task_aligned_margin": aligned,
            "user_choice_margin": user,
            "candidate_payload_sha256": "same-payload",
            "user_selected_label": "B",
            "suffix_token_sha256_int32_le": f"suffix-{requirement}",
        }

    def test_match_advantage_uses_all_four_cells(self):
        rows = [
            self._row("verification", "independent_verification", 3.0, -3.0),
            self._row("obedience", "independent_verification", 1.0, -1.0),
            self._row("verification", "delegated_choice", 1.0, 1.0),
            self._row("obedience", "delegated_choice", 4.0, 4.0),
        ]
        interactions, problems = build_interactions(rows)
        self.assertEqual(problems, [])
        self.assertEqual(len(interactions), 1)
        result = interactions[0]
        self.assertEqual(result["verification_history_effect"], -2.0)
        self.assertEqual(result["deference_history_effect"], 3.0)
        self.assertEqual(result["match_advantage_interaction"], 5.0)
        self.assertEqual(result["matched_minus_mismatched"], 2.5)
        self.assertEqual(result["task_modulation_of_raw_carryover"], 1.0)
        self.assertTrue(result["both_directional_predictions_met"])

    def test_missing_cell_fails_pair_construction(self):
        rows = [
            self._row("verification", "independent_verification", 3.0, -3.0),
            self._row("obedience", "independent_verification", 1.0, -1.0),
            self._row("verification", "delegated_choice", 1.0, 1.0),
        ]
        interactions, problems = build_interactions(rows)
        self.assertEqual(interactions, [])
        self.assertTrue(any("missing cell" in problem for problem in problems))


class OperatorGridTests(unittest.TestCase):
    def test_grid_is_deterministic_unique_and_family_complete(self):
        operator = json.loads((ROOT / "protocol/QWEN3_8B_OPERATOR_CONTRACT_V1.json").read_text())
        extension = json.loads(
            (ROOT / "protocol/QWEN3_8B_BIDIRECTIONAL_OPERATOR_EXTENSION_V1.json").read_text()
        )
        sites = {
            "locked": True,
            "selected_sites_ordered": [
                {"layer": 31, "component": "mlp"},
                {"layer": 27, "component": "self_attn"},
                {"layer": 23, "component": "mlp"},
            ],
        }
        first = enumerate_grid(operator, extension, sites)
        second = enumerate_grid(operator, extension, sites)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 15660)
        self.assertEqual(len({candidate_id(row) for row in first}), len(first))
        self.assertEqual(
            Counter(row["family"] for row in first),
            Counter(
                {
                    "fixed_translation": 108,
                    "symmetric_rank1": 108,
                    "v2_one_sided": 324,
                    "v3_full": 1296,
                    "v4_bilinear": 3456,
                    "v5_signed": 10368,
                }
            ),
        )

    def test_pair_fold_is_stable(self):
        self.assertEqual([pair_fold(f"pair-{index}") for index in range(8)], [0, 2, 2, 1, 2, 1, 3, 1])


if __name__ == "__main__":
    unittest.main()
