import json
import copy
import unittest
from pathlib import Path

from scripts.benchmark_v1.common import load_jsonl
from scripts.benchmark_v10.cdge_performance_design import (
    EXPECTED_CELLS,
    EXPECTED_CELL_KEY_SHA256,
    EXPECTED_MEASUREMENTS,
    METHODS,
    TIMED_REPEATS,
    WARMUP_REPEATS,
    cell_key_sha256,
    measurement_key,
    selected_performance_jobs,
    cell_key,
)
from scripts.benchmark_v10.analyze_cdge_v4_1_performance import analyze


ROOT = Path(__file__).resolve().parents[1]


class CdgePerformanceDesignTest(unittest.TestCase):
    def values(self):
        manifest = load_jsonl(ROOT / "artifacts/qwen3-8b-v1/benchmark_manifest.jsonl")
        crossover = json.loads((ROOT / "protocol/GOVERNANCE_TASK_CROSSOVER_V1.json").read_text())
        jobs = selected_performance_jobs(manifest, crossover)
        rows = []
        for job in jobs:
            cell = cell_key(job)
            for method_index, method in enumerate(METHODS):
                for repeat in range(TIMED_REPEATS):
                    latency = 1.0 + 0.05 * method_index + 0.001 * repeat
                    rows.append({
                        "measurement_key": measurement_key(cell, method, repeat),
                        "cell_key": cell,
                        "method": method,
                        "repeat": repeat,
                        "operator_dev_job_key": job["item"]["item_id"],
                        "prefix_tokens": 20,
                        "suffix_tokens": 10,
                        "prefix_token_sha256_int32_le": "a" * 64,
                        "suffix_token_sha256_int32_le": "b" * 64,
                        "paired_latency_seconds": latency,
                        "examples_per_second": 1.0 / latency,
                        "suffix_tokens_per_second": 10.0 / latency,
                        "resident_npu_memory_bytes": 1000 + method_index,
                        "peak_npu_memory_bytes": 2000 + method_index * 100,
                        "incremental_peak_npu_memory_bytes": 1000 + method_index * 99,
                        "selected_label_logits": {"A": 1.0, "B": -1.0},
                    })
        environment = {
            "stage": "governance_composite_performance",
            "method": "C-DGE-V4.1",
            "methods": list(METHODS),
            "expected_cells": EXPECTED_CELLS,
            "expected_cell_key_sha256": EXPECTED_CELL_KEY_SHA256,
            "expected_measurements": EXPECTED_MEASUREMENTS,
            "batch_size": 1,
            "warmup_repeats": WARMUP_REPEATS,
            "timed_repeats": TIMED_REPEATS,
            "attn_implementation": "eager",
            "peak_memory_reset_after_model_load": True,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        contract = json.loads(
            (ROOT / "protocol/QWEN3_8B_CDGE_V4_1_POST_ELIGIBILITY_EVALUATION_V1.json").read_text()
        )
        return rows, environment, contract

    def test_frozen_24_cell_design(self):
        manifest = load_jsonl(ROOT / "artifacts/qwen3-8b-v1/benchmark_manifest.jsonl")
        contract = json.loads((ROOT / "protocol/GOVERNANCE_TASK_CROSSOVER_V1.json").read_text())
        jobs = selected_performance_jobs(manifest, contract)
        self.assertEqual(len(jobs), EXPECTED_CELLS)
        self.assertEqual(len({job["item"]["benchmark"] for job in jobs}), 6)
        self.assertEqual(len({measurement_key("cell", method, repeat) for method in METHODS for repeat in range(TIMED_REPEATS)}), 30)
        self.assertEqual(EXPECTED_MEASUREMENTS, 720)
        self.assertEqual(WARMUP_REPEATS, 3)
        self.assertEqual(TIMED_REPEATS, 10)
        self.assertEqual(cell_key_sha256(jobs), EXPECTED_CELL_KEY_SHA256)

    def test_contract_matches_implementation(self):
        performance = json.loads(
            (ROOT / "protocol/QWEN3_8B_CDGE_V4_1_POST_ELIGIBILITY_EVALUATION_V1.json").read_text()
        )["performance"]
        self.assertEqual(tuple(performance["methods"]), METHODS)
        self.assertEqual(performance["expected_cells"], EXPECTED_CELLS)
        self.assertEqual(performance["warmup_repeats"], WARMUP_REPEATS)
        self.assertEqual(performance["timed_repeats"], TIMED_REPEATS)
        self.assertEqual(performance["batch_size"], 1)
        self.assertEqual(performance["attn_implementation"], "eager")

    def test_synthetic_performance_audit(self):
        rows, environment, contract = self.values()
        report = analyze(rows, environment, contract, replicates=200)
        self.assertTrue(report["audit"]["success"])
        self.assertGreater(
            report["paired_reports"]["C-DGE-V4.1"]["latency_overhead_percent"]["estimate"],
            0.0,
        )
        broken = copy.deepcopy(rows)
        broken[-1]["suffix_token_sha256_int32_le"] = "c" * 64
        self.assertFalse(analyze(broken, environment, contract, replicates=20)["audit"]["success"])


if __name__ == "__main__":
    unittest.main()
