from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_v1.common import load_jsonl
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key
from scripts.benchmark_v6.materialize_consensus_router_capture_authorization import (
    MINIMUM_CODE_VERSION,
    _design_audit_bindings,
)


ROOT = Path(__file__).resolve().parents[1]


class ConsensusRouterV5AuthorizationCompatibilityTests(unittest.TestCase):
    def test_new_bundle_binds_both_equal_design_audit_fields(self) -> None:
        self.assertEqual(MINIMUM_CODE_VERSION, 35)
        with tempfile.TemporaryDirectory() as temporary:
            audit = Path(temporary) / "component_discovery_design_audit.json"
            audit.write_text('{"audit":{"success":true}}\n', encoding="utf-8")
            bindings = _design_audit_bindings(audit)
        self.assertEqual(
            set(bindings),
            {
                "design_audit_sha256",
                "component_discovery_design_audit_sha256",
            },
        )
        self.assertEqual(
            bindings["design_audit_sha256"],
            bindings["component_discovery_design_audit_sha256"],
        )
        self.assertEqual(len(bindings["design_audit_sha256"]), 64)


class ConsensusRouterV5CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(
            (ROOT / "protocol/QWEN3_8B_GROUP_ROBUST_CONSENSUS_GOVERNANCE_EDITOR_V5.json").read_text()
        )
        cls.crossover = json.loads(
            (ROOT / "protocol/GOVERNANCE_TASK_CROSSOVER_V1.json").read_text()
        )
        cls.manifest = load_jsonl(ROOT / "artifacts/qwen3-8b-v1/benchmark_manifest.jsonl")

    def test_component_discovery_governance_identity_is_frozen(self) -> None:
        jobs = enumerate_jobs(self.manifest, self.crossover, "discovery")
        self.assertEqual(len(jobs), 6144)
        self.assertEqual(
            {job["item"]["partition"] for job in jobs}, {"component_discovery"}
        )
        digest = hashlib.sha256(
            ("\n".join(sorted(job_key(job) for job in jobs)) + "\n").encode("utf-8")
        ).hexdigest()
        self.assertEqual(
            digest,
            self.contract["developmental_audit_capture"][
                "governance_expected_key_sha256"
            ],
        )

    def test_consumed_v4_folds_are_forbidden(self) -> None:
        fit = self.contract["fit_data"]
        self.assertEqual(fit["allowed_hash_folds"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(fit["forbidden_consumed_hash_folds"], [6, 7])
        self.assertTrue(fit["v4_calibration_or_audit_rows_may_not_be_loaded"])
        self.assertTrue(fit["v4_error_identities_may_not_be_loaded"])

    def test_new_small_controls_are_identity_disjoint_and_frozen(self) -> None:
        old = json.loads((ROOT / "protocol/mitigation_controls.json").read_text())
        new = json.loads(
            (ROOT / "protocol/mitigation_controls_v5_audit.json").read_text()
        )
        self.assertTrue(new["frozen_before_any_v5_capture_forward"])
        self.assertTrue(new["output_blind_construction"])
        for family in ("supported_user_authority", "factual_boundary_memory"):
            old_ids = {row["id"] for row in old[family]}
            new_ids = {row["id"] for row in new[family]}
            self.assertEqual(len(new_ids), 6)
            self.assertFalse(old_ids & new_ids)

if __name__ == "__main__":
    unittest.main()
