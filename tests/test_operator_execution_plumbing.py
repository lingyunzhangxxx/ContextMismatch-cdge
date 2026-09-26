from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class OperatorLedgerMergeTests(unittest.TestCase):
    def _artifact(self, root: Path, name: str, value: dict | bytes) -> dict[str, str]:
        path = root / name
        if isinstance(value, bytes):
            path.write_bytes(value)
        else:
            write_json(path, value)
        return {
            "local_path": str(path),
            "remote_path": f"{REMOTE_ROOT}/operator-artifacts/{name}",
            "sha256": sha256(path),
        }

    def _ledger(self, root: Path, mode: str, candidate_id: str) -> Path:
        tensor = self._artifact(root, f"{candidate_id}.pt", b"immutable tensor fixture")
        manifest = self._artifact(
            root,
            f"{candidate_id}.manifest.json",
            {"candidate_id": candidate_id, "tensor_sha256": tensor["sha256"]},
        )
        analysis = self._artifact(
            root,
            f"{candidate_id}.{mode}.analysis.json",
            {"candidate_id": candidate_id, "audit": {"success": True}},
        )
        identity = self._artifact(
            root,
            f"{candidate_id}.{mode}.identity.json",
            {"candidate_id": candidate_id, "success": True, "max_error": 0.0},
        )
        path = root / f"{mode}.ledger.json"
        write_json(
            path,
            {
                "mode": mode,
                "candidate_count": 1,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "candidate_manifest": manifest,
                        "candidate_tensor": tensor,
                        "analysis": analysis,
                        "identity_report": identity,
                    }
                ],
                "final_test_open": False,
                "production_rollout_approved": False,
            },
        )
        return path

    def test_screening_ledger_merge_preserves_local_remote_sha_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = self._ledger(root, "behavior_screening", "candidate-a")
            output = root / "merged.json"
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.benchmark_v2.merge_operator_evaluation_ledgers",
                    "--mode",
                    "screening",
                    "--behavior-ledger",
                    str(ledger),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            merged = json.loads(output.read_text())
            self.assertEqual(merged["candidate_count"], 1)
            row = merged["candidates"][0]
            self.assertTrue(Path(row["candidate_manifest"]).is_absolute())
            self.assertTrue(row["candidate_manifest_remote"].startswith(f"{REMOTE_ROOT}/"))
            self.assertEqual(sha256(Path(row["candidate_tensor"])), row["candidate_tensor_sha256"])
            self.assertFalse(merged["final_test_open"])
            self.assertFalse(merged["production_rollout_approved"])

    def test_duplicate_candidate_across_ledgers_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = self._ledger(root, "behavior_screening", "candidate-a")
            output = root / "merged.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.benchmark_v2.merge_operator_evaluation_ledgers",
                    "--mode",
                    "screening",
                    "--behavior-ledger",
                    str(ledger),
                    "--behavior-ledger",
                    str(ledger),
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("duplicate candidate", result.stderr)
            self.assertFalse(output.exists())


class OperatorFinalAuthorizationTests(unittest.TestCase):
    def _fixture(self, root: Path, *, final_test_open: bool = False) -> tuple[Path, Path]:
        extension = json.loads(
            (ROOT / "protocol/QWEN3_8B_BIDIRECTIONAL_OPERATOR_EXTENSION_V1.json").read_text()
        )
        extension["status"] = "frozen_before_operator_forward"
        extension["frozen_utc"] = "2026-07-28T02:00:00Z"
        extension_path = root / "extension.json"
        write_json(extension_path, extension)

        tensor = root / "candidate.pt"
        tensor.write_bytes(b"locked candidate tensor fixture")
        manifest = root / "candidate.manifest.json"
        write_json(
            manifest,
            {"candidate_id": "candidate-final", "tensor_sha256": sha256(tensor)},
        )
        operator_contract = ROOT / "protocol/QWEN3_8B_OPERATOR_CONTRACT_V1.json"
        lock = root / "operator-lock.json"
        write_json(
            lock,
            {
                "locked": True,
                "selection_partition": "operator_dev only",
                "chosen_candidate_id": "candidate-final",
                "operator_contract_sha256": sha256(operator_contract),
                "extension_contract_sha256": sha256(extension_path),
                "candidate_manifest": str(manifest),
                "candidate_manifest_remote": f"{REMOTE_ROOT}/operator-artifacts/candidate.manifest.json",
                "candidate_manifest_sha256": sha256(manifest),
                "candidate_tensor": str(tensor),
                "candidate_tensor_remote": f"{REMOTE_ROOT}/operator-artifacts/candidate.pt",
                "candidate_tensor_sha256": sha256(tensor),
                "final_test_open": final_test_open,
                "final_test_open_count": 1 if final_test_open else 0,
                "production_rollout_approved": False,
            },
        )
        return extension_path, lock

    def _command(self, root: Path, extension: Path, lock: Path, output: Path) -> list[str]:
        return [
            sys.executable,
            "-m",
            "scripts.benchmark_v2.materialize_operator_final_authorization",
            "--code-version",
            "18",
            "--bundle-manifest-sha256",
            "a" * 64,
            "--operator-lock",
            str(lock),
            "--operator-lock-remote",
            f"{REMOTE_ROOT}/locks/operator-lock.json",
            "--crossover-contract",
            str(ROOT / "protocol/GOVERNANCE_TASK_CROSSOVER_V1.json"),
            "--operator-contract",
            str(ROOT / "protocol/QWEN3_8B_OPERATOR_CONTRACT_V1.json"),
            "--extension-contract",
            str(extension),
            "--manifest",
            str(ROOT / "artifacts/qwen3-8b-v1/benchmark_manifest.jsonl"),
            "--design-audit",
            str(ROOT / "artifacts/qwen3-8b-v1/governance_crossover_final_test_design_audit_v2.json"),
            "--model-manifest-remote",
            f"{REMOTE_ROOT}/runs/model-verification.json",
            "--model-manifest-sha256",
            "b" * 64,
            "--created-utc",
            "2026-07-28T02:00:00Z",
            "--output",
            str(output),
        ]

    def test_final_authorization_fixture_is_one_time_sha_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension, lock = self._fixture(root)
            output = root / "final-authorization.json"
            subprocess.run(
                self._command(root, extension, lock, output),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            value = json.loads(output.read_text())
            self.assertTrue(value["final_test_open"])
            self.assertEqual(value["final_test_open_count"], 1)
            self.assertEqual(value["expected_rows"], 6144)
            self.assertEqual(value["operator_lock"]["sha256"], sha256(lock))
            self.assertFalse(value["production_rollout_approved"])

    def test_final_authorization_refuses_already_open_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extension, lock = self._fixture(root, final_test_open=True)
            output = root / "final-authorization.json"
            result = subprocess.run(
                self._command(root, extension, lock, output),
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already opened", result.stderr)
            self.assertFalse(output.exists())



if __name__ == "__main__":
    unittest.main()
