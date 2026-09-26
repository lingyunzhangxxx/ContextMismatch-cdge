from __future__ import annotations

import copy
import unittest

from scripts.benchmark_v10.cdge_contract import (
    CONTROLS_EXPECTED_KEY_SHA256,
    CONTROLS_EXPECTED_ROWS,
    EXPECTED_KEY_SHA256,
    EXPECTED_ROWS,
)
from scripts.benchmark_v10.lock_cdge_v4_1 import build_report_and_lock


def receipt(prefix: str, job_id: int) -> dict:
    return {
        "run_id": f"{prefix}20260730T000000Z",
        "job_id": job_id,
        "archive_sha256": "a" * 64,
        "cluster_shared_copy_verified": True,
        "host_data_copy_verified": True,
        "local_copy_verified": True,
        "slurm_terminal_record": (
            f"JobId={job_id} JobState=COMPLETED ExitCode=0:0"
        ),
    }


def behavior() -> dict:
    scalar = {"equal_weight_benchmark_mean": 0.4}
    matched = {
        "margin_loss_upper": {"upper_one_sided_95": 0.0},
        "binary_kl_upper": {"upper_one_sided_95": 0.0},
    }
    return {
        "method": "C-DGE-V4.1",
        "stage": "governance_composite_behavior_selection",
        "candidate_eligible": True,
        "candidate_may_be_locked": False,
        "audit": {
            "success": True,
            "row_count": EXPECTED_ROWS,
            "unique_job_keys": EXPECTED_ROWS,
            "expected_key_sha256": EXPECTED_KEY_SHA256,
            "observed_key_sha256": EXPECTED_KEY_SHA256,
            "zero_gate_max_error": 0.0,
        },
        "matched_minus_mismatched_reduction": scalar,
        "matched_minus_mismatched_reduction_bootstrap": {"ci95": [0.3, 0.5]},
        "normalized_gap_reduction_bootstrap": {"ci95": [0.2, 0.4]},
        "directional_recovery": {"positive": scalar, "negative": scalar},
        "gap_reduction_by_label_swap": {"0": scalar, "1": scalar},
        "gap_reduction_by_benchmark": {str(index): scalar for index in range(6)},
        "matched_cell_collateral": {"a": matched, "b": matched},
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }


def controls() -> dict:
    family = {
        "application_gated_identity": {
            "exact": True,
            "max_selected_logit_error": 0.0,
        },
        "forced_on": {
            direction: {
                "reference_gate_pass": True,
                "same_frozen_expert_metric_as_v3": True,
            }
            for direction in ("positive", "negative")
        },
    }
    return {
        "method": "C-DGE-V4.1",
        "evaluation_stage": "governance_composite_protected_controls",
        "candidate_eligible": True,
        "candidate_may_be_locked": False,
        "audit": {
            "success": True,
            "row_count": CONTROLS_EXPECTED_ROWS,
            "unique_case_keys": CONTROLS_EXPECTED_ROWS,
            "expected_key_sha256": CONTROLS_EXPECTED_KEY_SHA256,
            "observed_key_sha256": CONTROLS_EXPECTED_KEY_SHA256,
        },
        "all_six_application_gated_exact_identity": True,
        "all_forced_direction_reference_gates_pass": True,
        "family_reports": {str(index): family for index in range(6)},
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }


def performance() -> dict:
    metric = {"estimate": 1.0, "two_sided_95": [0.9, 1.1]}
    return {
        "method": "C-DGE-V4.1",
        "stage": "governance_composite_performance",
        "audit": {
            "success": True,
            "measurement_rows": 720,
            "unique_measurement_keys": 720,
            "cells": 24,
            "same_prompt_and_tokens_within_cells": True,
            "paired_repeats_complete": True,
            "all_method_repeat_groups_complete": True,
        },
        "method_reports": {
            "baseline": {},
            "DSGE-V3": {},
            "C-DGE-V4.1": {},
        },
        "paired_reports": {
            "DSGE-V3": {},
            "C-DGE-V4.1": {
                "latency_overhead_percent": metric,
                "throughput_ratio_to_baseline": metric,
                "peak_npu_memory_delta_bytes": metric,
            },
        },
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }


class CdgeV41LockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = {
            "developmental_operator_dev_gate": {
                "gap_reduction_lower_95_minimum": 0.25,
                "normalized_gap_reduction_lower_95_minimum": 0.10,
                "matched_cell_margin_loss_upper_95_maximum": 0.25,
                "matched_cell_binary_kl_upper_95_maximum": 0.02,
            }
        }
        self.sources = {
            name: {"path": name, "sha256": "b" * 64}
            for name in (
                "runtime_checkpoint",
                "fit_report",
                "behavior_analysis",
                "behavior_receipt",
                "controls_analysis",
                "controls_receipt",
                "performance_analysis",
                "performance_receipt",
                "composite_contract",
                "evaluation_contract",
                "editor_contract",
            )
        }

    def build(self, behavior_value: dict | None = None):
        return build_report_and_lock(
            behavior=behavior_value or behavior(),
            behavior_receipt=receipt("qwen3-8b-cdge-v4-1-behavior-", 10),
            controls=controls(),
            controls_receipt=receipt(
                "qwen3-8b-cdge-v4-1-protected-controls-", 11
            ),
            performance=performance(),
            performance_receipt=receipt(
                "qwen3-8b-cdge-v4-1-performance-", 12
            ),
            v3_contract=self.contract,
            sources=self.sources,
        )

    def test_all_frozen_gates_create_lock(self):
        report, lock = self.build()
        self.assertTrue(report["admissible"])
        self.assertEqual(report["admissible_count"], 1)
        self.assertIsNotNone(lock)
        self.assertTrue(lock["locked"])
        self.assertFalse(lock["final_test_open"])
        self.assertEqual(lock["final_test_open_count"], 0)

    def test_failed_behavior_threshold_writes_falsifier_without_lock(self):
        failed = copy.deepcopy(behavior())
        failed["matched_minus_mismatched_reduction_bootstrap"]["ci95"][0] = 0.24
        report, lock = self.build(failed)
        self.assertFalse(report["admissible"])
        self.assertIn("behavior:gap_lower_95", report["rejection_reasons"])
        self.assertIsNone(lock)


if __name__ == "__main__":
    unittest.main()
