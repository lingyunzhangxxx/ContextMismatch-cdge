import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_v16.materialize_native_behavior_authorization import (
    EXPECTED_KEY_SHA256,
    EXPECTED_ROWS,
    _candidate_records,
    _require_receipt,
)
from scripts.benchmark_v16.run_native_behavior_shard import _behavior_checks


class NativeBehaviorTests(unittest.TestCase):
    def _report(self):
        return {
            "directional_recovery": {
                "obedience": {"equal_weight_benchmark_mean": 0.1},
                "verification": {"equal_weight_benchmark_mean": 0.2},
            },
            "gap_reduction_by_label_swap": {
                "0": {"equal_weight_benchmark_mean": 0.1},
                "1": {"equal_weight_benchmark_mean": 0.1},
            },
            "gap_reduction_by_benchmark": {
                name: {"equal_weight_benchmark_mean": 0.01}
                for name in ("arc", "bbh", "gsm", "math", "mmlu", "musr")
            },
            "matched_selected_logit_max_error": 0.0,
            "audit": {"zero_gate_max_error": 0.0},
        }

    def test_fold_seven_identity_is_frozen(self):
        self.assertEqual(EXPECTED_ROWS, 896)
        self.assertEqual(
            EXPECTED_KEY_SHA256,
            "738cda7c0964ab75ef864405412028a95d207a1915dd4b58569428b525b874b1",
        )

    def test_behavior_gate_requires_fit_and_all_strata(self):
        checks = _behavior_checks(self._report(), True)
        self.assertTrue(all(checks.values()))
        self.assertFalse(all(_behavior_checks(self._report(), False).values()))
        broken = copy.deepcopy(self._report())
        broken["gap_reduction_by_benchmark"]["musr"]["equal_weight_benchmark_mean"] = -0.01
        self.assertFalse(
            _behavior_checks(broken, True)[
                "all_six_benchmark_point_estimates_nonnegative"
            ]
        )

    def test_identity_must_be_exact(self):
        report = self._report()
        report["matched_selected_logit_max_error"] = 1e-12
        self.assertFalse(
            _behavior_checks(report, True)["matched_selected_logit_exact_identity"]
        )

    def test_fit_shard_paths_and_receipt_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fit_root = root / "fit"
            fit_root.mkdir()
            profiles = []
            candidates = []
            shard_profiles = []
            for profile_index, rank in enumerate(("compact", "balanced", "wide")):
                profile_id = f"profile-{profile_index}"
                profile_dir = fit_root / profile_id
                cap_dir = profile_dir / "cap_checkpoints"
                cap_dir.mkdir(parents=True)
                contract = profile_dir / "directional_contract.json"
                source_fit = profile_dir / "directional_fit" / "directional_fit_report.json"
                source_fit.parent.mkdir()
                contract.write_text("{}\n")
                source_fit.write_text("{}\n")
                cap_rows = []
                for cap_index, cap in enumerate((0.025, 0.05, 0.075)):
                    candidate_id = f"candidate-{profile_index}-{cap_index}"
                    checkpoint = cap_dir / f"{candidate_id}.pt"
                    checkpoint.write_bytes(candidate_id.encode())
                    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                    cap_rows.append({
                        "candidate_id": candidate_id,
                        "checkpoint": checkpoint.name,
                        "checkpoint_sha256": digest,
                    })
                    candidates.append({
                        "candidate_id": candidate_id,
                        "config": {"layer": 11, "component": "mlp", "rank_profile": rank,
                                   "max_relative_correction": cap},
                    })
                cap_manifest = cap_dir / "cap_checkpoint_manifest.json"
                cap_manifest.write_text(json.dumps({"candidate_count": 3, "checkpoints": cap_rows}))
                profiles.append({
                    "fit_profile_id": profile_id,
                    "fit_profile_sha256": str(profile_index) * 64,
                    "site": {"layer": 11, "component": "mlp"},
                })
                sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
                shard_profiles.append({
                    "fit_profile_id": profile_id,
                    "directional_contract": str(contract.relative_to(fit_root)),
                    "directional_contract_sha256": sha(contract),
                    "directional_fit_report": str(source_fit.relative_to(fit_root)),
                    "directional_fit_report_sha256": sha(source_fit),
                    "cap_checkpoint_manifest": str(cap_manifest.relative_to(fit_root)),
                    "cap_checkpoint_manifest_sha256": sha(cap_manifest),
                    "fit_eligible": profile_index != 0,
                })
            (fit_root / "native_fit_shard_report.json").write_text(json.dumps({
                "stage": "qwen35_cdge_v4_2_native_fit_shard", "method": "C-DGE-V4.2",
                "site": "11:mlp", "fit_complete": True, "fit_profile_count": 3,
                "training_run_count": 3, "materialized_candidate_count": 9,
                "operator_dev_accessed": False, "protected_behavior_outputs_accessed": False,
                "final_test_open": False, "final_test_open_count": 0,
                "production_rollout_approved": False, "profiles": shard_profiles,
            }))
            records = _candidate_records(
                site="11:mlp", fit_run_dir=root,
                grid={"candidates": candidates}, profiles={"profiles": profiles},
            )
            self.assertEqual(len(records), 9)
            self.assertEqual(sum(row["fit_eligible"] for row in records), 6)

            receipt = root / "receipt.json"
            receipt.write_text(json.dumps({
                "run_id": root.name,
                "archive_sha256": "a" * 64,
                "cluster_shared_path": "/workspace/one.tar.gz",
                "durable_mirror_path": "/workspace/one.tar.gz",
                "cluster_shared_copy_verified": True,
                "host_data_copy_verified": True,
                "local_copy_verified": True,
                "slurm_terminal_record": "JobState=COMPLETED ExitCode=0:0",
            }))
            _require_receipt(receipt, root.name)


if __name__ == "__main__":
    unittest.main()
