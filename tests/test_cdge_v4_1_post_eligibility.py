import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_v10.cdge_contract import (
    ARCHIVE_SHA256,
    EXPECTED_KEY_SHA256,
    EXPECTED_ROWS,
    FINAL_EXPECTED_KEY_SHA256,
    FINAL_EXPECTED_ROWS,
    FINAL_STAGE,
    METHOD,
    STAGE,
    promote_behavior_outputs,
    promote_final_outputs,
    validate_behavior_prerequisite,
    validate_composite_prerequisite,
)


class CompositePostEligibilityTest(unittest.TestCase):
    def values(self):
        report = {
            "method": METHOD,
            "audit_complete": True,
            "candidate_eligible": True,
            "candidate_may_be_locked": False,
            "operator_dev_accessed": False,
            "protected_behavior_outputs_accessed": False,
            "gate_checks": {"one": True, "two": True},
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        receipt = {
            "run_id": "qwen3-8b-cdge-v4-1-composite-eligibility-20260730T012711Z",
            "job_id": 9267,
            "archive_sha256": ARCHIVE_SHA256,
            "cluster_shared_copy_verified": True,
            "host_data_copy_verified": True,
            "local_copy_verified": True,
            "slurm_terminal_record": "JobId=9267 JobState=COMPLETED ExitCode=0:0",
        }
        return report, receipt

    def test_terminal_composite_prerequisite_passes(self):
        report, receipt = self.values()
        validate_composite_prerequisite(report, receipt)

    def test_failed_composite_gate_is_rejected(self):
        report, receipt = self.values()
        report["gate_checks"]["two"] = False
        with self.assertRaisesRegex(ValueError, "every gate"):
            validate_composite_prerequisite(report, receipt)

    def test_missing_slurm_terminal_evidence_is_rejected(self):
        report, receipt = self.values()
        receipt["slurm_terminal_record"] = "JobId=9267 JobState=RUNNING ExitCode=0:0"
        with self.assertRaisesRegex(ValueError, "JobState=COMPLETED"):
            validate_composite_prerequisite(report, receipt)

    def test_frozen_operator_contract(self):
        contract = json.loads(
            Path("protocol/QWEN3_8B_CDGE_V4_1_POST_ELIGIBILITY_EVALUATION_V1.json").read_text()
        )
        self.assertEqual(contract["method"], METHOD)
        self.assertEqual(contract["operator_dev"]["stage"], STAGE)
        self.assertEqual(contract["operator_dev"]["expected_rows"], EXPECTED_ROWS)
        self.assertEqual(contract["operator_dev"]["expected_key_sha256"], EXPECTED_KEY_SHA256)
        self.assertFalse(contract["final_test_open"])
        self.assertEqual(contract["final_test_open_count"], 0)

    def test_terminal_behavior_prerequisite_passes(self):
        report = {
            "method": METHOD,
            "stage": STAGE,
            "behavior_evaluation_complete": True,
            "original_v4_fit_eligible": False,
            "composite_eligible": True,
            "candidate_eligible": True,
            "candidate_may_be_locked": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
            "audit": {
                "success": True,
                "row_count": EXPECTED_ROWS,
                "unique_job_keys": EXPECTED_ROWS,
                "observed_key_sha256": EXPECTED_KEY_SHA256,
                "zero_gate_max_error": 0.0,
            },
        }
        receipt = {
            "run_id": "qwen3-8b-cdge-v4-1-behavior-20260730T030216Z",
            "job_id": 9275,
            "archive_sha256": "a" * 64,
            "cluster_shared_copy_verified": True,
            "host_data_copy_verified": True,
            "local_copy_verified": True,
            "slurm_terminal_record": "JobId=9275 JobState=COMPLETED ExitCode=0:0",
        }
        validate_behavior_prerequisite(report, receipt)
        receipt["slurm_terminal_record"] = "JobId=9275 JobState=RUNNING ExitCode=0:0"
        with self.assertRaisesRegex(ValueError, "JobState=COMPLETED"):
            validate_behavior_prerequisite(report, receipt)

    def test_runtime_outputs_promote_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bound = []
            for name in ("composite", "evaluation", "report", "authorization", "receipt"):
                path = root / f"{name}.json"
                path.write_text("{}\n")
                bound.append(path)
            output = root / "formal.jsonl"
            environment = root / "formal.environment.json"
            identity = root / "formal.identity.json"
            runtime_output = root / ".formal.jsonl.runtime"
            runtime_environment = root / ".formal.environment.json.runtime"
            runtime_identity = root / ".formal.identity.json.runtime"
            runtime_output.write_text(json.dumps({"method": "ADSGE-V4", "value": 1.0}) + "\n")
            runtime_environment.write_text(json.dumps({"method": "ADSGE-V4"}) + "\n")
            runtime_identity.write_text(json.dumps({"success": True, "max_error": 0.0}) + "\n")
            args = argparse.Namespace(
                output=output,
                environment_output=environment,
                identity_output=identity,
                composite_contract=bound[0],
                evaluation_contract=bound[1],
                composite_report=bound[2],
                composite_authorization=bound[3],
                composite_receipt=bound[4],
            )
            promote_behavior_outputs(
                runtime_output=runtime_output,
                runtime_environment=runtime_environment,
                runtime_identity=runtime_identity,
                output=args.output,
                environment_output=args.environment_output,
                identity_output=args.identity_output,
                composite_contract=args.composite_contract,
                evaluation_contract=args.evaluation_contract,
                composite_report=args.composite_report,
                composite_authorization=args.composite_authorization,
                composite_receipt=args.composite_receipt,
            )
            promoted = json.loads(output.read_text())
            self.assertEqual(promoted["method"], METHOD)
            self.assertEqual(promoted["runtime_checkpoint_method"], "ADSGE-V4")
            self.assertEqual(json.loads(environment.read_text())["stage"], STAGE)
            self.assertEqual(json.loads(identity.read_text())["method"], METHOD)
            self.assertFalse(runtime_output.exists())
            self.assertFalse(runtime_environment.exists())
            self.assertFalse(runtime_identity.exists())

    def test_final_outputs_require_one_time_open_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "formal-final.jsonl"
            environment = root / "formal-final.environment.json"
            identity = root / "formal-final.identity.json"
            runtime_output = root / ".formal-final.jsonl.runtime"
            runtime_environment = root / ".formal-final.environment.json.runtime"
            runtime_identity = root / ".formal-final.identity.json.runtime"
            lock = root / "pareto-lock.json"
            lock.write_text('{"locked":true}\n')
            row = json.dumps(
                {
                    "method": "ADSGE-V4",
                    "final_test_open": True,
                    "final_test_open_count": 1,
                    "production_rollout_approved": False,
                },
                sort_keys=True,
            )
            runtime_output.write_text((row + "\n") * FINAL_EXPECTED_ROWS)
            runtime_environment.write_text(
                json.dumps(
                    {
                        "expected_rows": FINAL_EXPECTED_ROWS,
                        "planned_rows": FINAL_EXPECTED_ROWS,
                        "expected_key_sha256": FINAL_EXPECTED_KEY_SHA256,
                        "final_test_open": True,
                        "final_test_open_count": 1,
                        "production_rollout_approved": False,
                    }
                )
                + "\n"
            )
            runtime_identity.write_text(
                json.dumps(
                    {
                        "success": True,
                        "max_error": 0.0,
                        "final_test_open": True,
                        "final_test_open_count": 1,
                        "production_rollout_approved": False,
                    }
                )
                + "\n"
            )
            promote_final_outputs(
                runtime_output=runtime_output,
                runtime_environment=runtime_environment,
                runtime_identity=runtime_identity,
                output=output,
                environment_output=environment,
                identity_output=identity,
                pareto_lock=lock,
            )
            with output.open() as handle:
                self.assertEqual(sum(1 for _ in handle), FINAL_EXPECTED_ROWS)
            self.assertEqual(json.loads(environment.read_text())["stage"], FINAL_STAGE)
            self.assertEqual(json.loads(identity.read_text())["method"], METHOD)
            self.assertFalse(runtime_output.exists())


if __name__ == "__main__":
    unittest.main()
