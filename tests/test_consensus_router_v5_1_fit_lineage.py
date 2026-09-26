import copy
import json
import unittest
from pathlib import Path

from scripts.benchmark_v6.materialize_consensus_router_fit_authorization_v54 import (
    CAPTURE_ARCHIVE_SHA256,
    EXPECTED_FAMILIES,
    GOVERNANCE_AUTH_SHA256,
    PROTECTED_AUTH_SHA256,
    ROUTER_SHA256,
    SELECTION_CONTRACT,
    _validate_contracts,
    _validate_manifest,
    _validate_receipt,
    _validate_selection,
)


ROOT = Path(__file__).resolve().parents[1]


class ConsensusRouterV51FitLineageTests(unittest.TestCase):
    def _contracts(self):
        protocol = ROOT / "protocol"
        router = json.loads(
            (protocol / "QWEN3_8B_GROUP_ROBUST_CONSENSUS_GOVERNANCE_EDITOR_V5.json").read_text()
        )
        inherited = json.loads(
            (protocol / "QWEN3_8B_GROUP_ROBUST_CONSENSUS_GOVERNANCE_FIT_V5.json").read_text()
        )
        recovery = json.loads(
            (protocol / "QWEN3_8B_GROUP_ROBUST_CONSENSUS_GOVERNANCE_FIT_V5_1_RECOVERY.json").read_text()
        )
        return router, inherited, recovery

    def test_recovery_contract_inherits_every_scientific_section(self):
        _validate_contracts(*self._contracts())

    def test_scientific_change_is_rejected(self):
        router, inherited, recovery = self._contracts()
        recovery = copy.deepcopy(recovery)
        recovery["training"]["epochs"] += 1
        with self.assertRaisesRegex(ValueError, "scientific fit contract changed"):
            _validate_contracts(router, inherited, recovery)

    def test_cpu_selector_contract_is_exact(self):
        selection = {
            "node": "a06",
            "state": "IDLE",
            "cpu_free": 192,
            "mem_free_mib": 2_000_000,
            "selection_contract": copy.deepcopy(SELECTION_CONTRACT),
        }
        _validate_selection(selection, "a06")
        selection["selection_contract"]["npus"] = 1
        with self.assertRaisesRegex(ValueError, "selector resource contract mismatch"):
            _validate_selection(selection, "a06")

    def test_dual_manifest_authorizations_are_not_interchangeable(self):
        governance = {
            "partition": "component_discovery",
            "rows": 6144,
            "authorization_sha256": GOVERNANCE_AUTH_SHA256,
            "router_contract_sha256": ROUTER_SHA256,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        protected = {
            **governance,
            "rows": 4008,
            "authorization_sha256": PROTECTED_AUTH_SHA256,
            "family_rows": EXPECTED_FAMILIES,
        }
        _validate_manifest(
            governance, name="governance", rows=6144,
            authorization_sha256=GOVERNANCE_AUTH_SHA256,
        )
        _validate_manifest(
            protected, name="protected", rows=4008,
            authorization_sha256=PROTECTED_AUTH_SHA256,
        )
        with self.assertRaisesRegex(ValueError, "authorization_sha256"):
            _validate_manifest(
                protected, name="protected", rows=4008,
                authorization_sha256=GOVERNANCE_AUTH_SHA256,
            )

    def test_three_copy_terminal_receipt_is_required(self):
        receipt = {
            "schema_version": 1,
            "job_id": 9203,
            "run_id": "qwen3-8b-governance-consensus-router-capture-recovery-20260729T134830Z",
            "archive_sha256": CAPTURE_ARCHIVE_SHA256,
            "cluster_shared_copy_verified": True,
            "host_data_copy_verified": True,
            "local_copy_verified": True,
            "slurm_terminal_record": "JobId=9203 JobState=COMPLETED ExitCode=0:0",
        }
        _validate_receipt(receipt)
        receipt["local_copy_verified"] = False
        with self.assertRaisesRegex(ValueError, "local_copy_verified"):
            _validate_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
