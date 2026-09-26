import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_v13.lock_qwen35_cdge_v4_1 import build_report_and_lock
from scripts.benchmark_v13.qwen35_eval_contract import (
    CANDIDATE_ID,
    CONTROL_KEY,
    CONTROL_ROWS,
    FINAL_KEY,
    FINAL_ROWS,
    LOCK_MANIFEST_ID,
    LOCK_STAGE,
    METHOD,
    OPERATOR_KEY,
    OPERATOR_ROWS,
    PERFORMANCE_CELLS,
    PERFORMANCE_ROWS,
    require_pareto_lock,
)


def receipt(prefix: str, job_id: int) -> dict:
    return {
        "run_id": f"{prefix}20260731T070000Z",
        "job_id": job_id,
        "archive_sha256": f"{job_id % 10}" * 64,
        "cluster_shared_copy_verified": True,
        "host_data_copy_verified": True,
        "local_copy_verified": True,
        "slurm_terminal_record": (
            f"JobId={job_id} UserId=researcher(10005) "
            "JobState=COMPLETED ExitCode=0:0"
        ),
    }


class Qwen35EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.contract = json.loads(
            Path("protocol/QWEN3_5_9B_CDGE_V4_1_EVALUATION_V1.json").read_text()
        )
        self.behavior = {
            "method": METHOD,
            "stage": "qwen35_cdge_operator_dev",
            "model": "Qwen3.5-9B",
            "candidate_eligible": True,
            "candidate_may_be_locked": False,
            "behavior_gate_checks": {
                "gap_lower_95": True,
                "normalized_gap_lower_95": True,
                "both_directions_positive": True,
                "both_label_swaps_positive": True,
                "all_benchmarks_nonnegative": True,
                "zero_gate_exact": True,
            },
            "audit": {
                "success": True,
                "row_count": OPERATOR_ROWS,
                "unique_job_keys": OPERATOR_ROWS,
                "expected_key_sha256": OPERATOR_KEY,
                "observed_key_sha256": OPERATOR_KEY,
                "zero_gate_max_error": 0.0,
            },
            "matched_minus_mismatched_reduction": {
                "equal_weight_benchmark_mean": 0.4
            },
            "matched_minus_mismatched_reduction_bootstrap": {"ci95": [0.3, 0.5]},
            "normalized_gap_reduction_bootstrap": {"ci95": [0.2, 0.4]},
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        family = {
            "application_gated_identity": {"exact": True},
            "forced_on": {
                direction: {
                    "reference_gate_pass": True,
                    "same_frozen_expert_metric_as_v3": True,
                }
                for direction in ("positive", "negative")
            },
        }
        self.controls = {
            "method": METHOD,
            "stage": "qwen35_cdge_protected_controls",
            "model": "Qwen3.5-9B",
            "audit": {
                "success": True,
                "row_count": CONTROL_ROWS,
                "unique_case_keys": CONTROL_ROWS,
                "expected_key_sha256": CONTROL_KEY,
                "observed_key_sha256": CONTROL_KEY,
            },
            "family_reports": {
                name: json.loads(json.dumps(family))
                for name in self.contract["protected_controls"]["families"]
            },
            "application_gated_identity": {"exact": True},
            "external_zero_gate_identity": {"exact": True},
            "all_six_application_gated_exact_identity": True,
            "all_forced_direction_reference_gates_pass": True,
            "candidate_may_be_locked": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        self.performance = {
            "method": METHOD,
            "stage": "governance_composite_performance",
            "model": "Qwen3.5-9B",
            "audit": {
                "success": True,
                "measurement_rows": PERFORMANCE_ROWS,
                "unique_measurement_keys": PERFORMANCE_ROWS,
                "cells": PERFORMANCE_CELLS,
                "paired_repeats_complete": True,
                "all_method_repeat_groups_complete": True,
                "same_prompt_and_tokens_within_cells": True,
            },
            "method_reports": {name: {} for name in ("baseline", "DSGE-V3", METHOD)},
            "paired_reports": {
                "DSGE-V3": {},
                METHOD: {
                    "latency_overhead_percent": 3.0,
                    "throughput_ratio_to_baseline": 0.97,
                    "peak_npu_memory_delta_bytes": 1024,
                },
            },
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        self.receipts = {
            "behavior_receipt": receipt("qwen3-5-9b-cdge-v4-1-behavior-", 9401),
            "controls_receipt": receipt(
                "qwen3-5-9b-cdge-v4-1-protected-controls-", 9402
            ),
            "performance_receipt": receipt(
                "qwen3-5-9b-cdge-v4-1-performance-", 9403
            ),
        }

    def build(self):
        return build_report_and_lock(
            behavior=self.behavior,
            controls=self.controls,
            performance=self.performance,
            evaluation_contract=self.contract,
            sources={"synthetic": {"sha256": "a" * 64}},
            **self.receipts,
        )

    def test_frozen_identities(self):
        self.assertEqual(self.contract["operator_dev"]["expected_rows"], OPERATOR_ROWS)
        self.assertEqual(self.contract["operator_dev"]["expected_key_sha256"], OPERATOR_KEY)
        self.assertEqual(self.contract["protected_controls"]["expected_rows"], CONTROL_ROWS)
        self.assertEqual(self.contract["protected_controls"]["expected_key_sha256"], CONTROL_KEY)
        self.assertEqual(self.contract["performance"]["expected_measurements"], PERFORMANCE_ROWS)
        self.assertEqual(self.contract["final_test"]["expected_rows"], FINAL_ROWS)
        self.assertEqual(self.contract["final_test"]["expected_key_sha256"], FINAL_KEY)

    def test_complete_terminal_evidence_builds_lock(self):
        report, lock = self.build()
        self.assertTrue(report["admissible"])
        self.assertIsNotNone(lock)
        self.assertEqual(lock["manifest_id"], LOCK_MANIFEST_ID)
        self.assertEqual(lock["stage"], LOCK_STAGE)
        self.assertEqual(lock["candidate_id"], CANDIDATE_ID)
        self.assertFalse(lock["final_test_open"])
        self.assertEqual(lock["final_test_open_count"], 0)

    def test_missing_terminal_provenance_keeps_final_closed(self):
        self.receipts["behavior_receipt"]["slurm_terminal_record"] = (
            "JobId=9401 JobState=COMPLETED ExitCode=1:0"
        )
        report, lock = self.build()
        self.assertFalse(report["admissible"])
        self.assertIsNone(lock)
        self.assertIn("receipts:behavior", report["rejection_reasons"])

    def test_failed_control_gate_keeps_final_closed(self):
        self.controls["all_forced_direction_reference_gates_pass"] = False
        report, lock = self.build()
        self.assertFalse(report["admissible"])
        self.assertIsNone(lock)
        self.assertIn("controls:forced_reference_gates", report["rejection_reasons"])

    def test_lock_source_sha_is_revalidated(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text("{}\n")
            _, lock = self.build()
            lock["sources"] = {
                "source": {"path": str(source), "sha256": __import__(
                    "hashlib"
                ).sha256(source.read_bytes()).hexdigest()}
            }
            lock_path = Path(directory) / "lock.json"
            lock_path.write_text(json.dumps(lock) + "\n")
            require_pareto_lock(lock_path, expected_sources={"source": source})
            source.write_text('{"changed":true}\n')
            with self.assertRaisesRegex(ValueError, "source mismatch"):
                require_pareto_lock(lock_path, expected_sources={"source": source})


if __name__ == "__main__":
    unittest.main()
