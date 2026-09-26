import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v14.native_discovery import materialize_candidate_grid
from scripts.benchmark_v15.native_fit_profiles import materialize_fit_profiles
import scripts.benchmark_v15.materialize_native_fit_authorization as fit_auth


class NativeFitAuthorizationTest(unittest.TestCase):
    def _write(self, path: Path, value: object) -> Path:
        atomic_write_text(path, json.dumps(value, sort_keys=True) + "\n")
        return path

    def test_binds_one_site_three_profiles_and_nine_cap_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sites = {
                "stage": "qwen35_cdge_v4_2_native_site_selection",
                "method": "C-DGE-V4.2",
                "selection_partition": "subspace_fit",
                "locked": True,
                "operator_dev_accessed": False,
                "final_test_open": False,
                "final_test_open_count": 0,
                "production_rollout_approved": False,
                "selected_sites_ordered": [
                    {"layer": 3, "component": "self_attn"},
                    {"layer": 11, "component": "mlp"},
                    {"layer": 29, "component": "self_attn"},
                ],
            }
            protocol_value = {
                "method_short_name": "C-DGE-V4.2",
                "status": "frozen_before_qwen3_5_native_discovery_forward",
                "native_discovery": {
                    "candidate_grid": {
                        "rank_profiles": [
                            {"name": "compact", "boundary_rank": 8, "context_rank": 16, "positive_output_rank": 8, "negative_output_rank": 8},
                            {"name": "balanced", "boundary_rank": 16, "context_rank": 32, "positive_output_rank": 16, "negative_output_rank": 16},
                            {"name": "wide", "boundary_rank": 32, "context_rank": 64, "positive_output_rank": 32, "negative_output_rank": 32},
                        ],
                        "protected_rank": 32,
                        "maximum_relative_corrections": [0.025, 0.05, 0.075],
                    },
                    "fit_split": {},
                    "behavior_selection_rule": {},
                },
                "final_test_open": False,
                "final_test_open_count": 0,
                "production_rollout_approved": False,
            }
            site_path = self._write(root / "sites.json", sites)
            grid = materialize_candidate_grid(sites, protocol_value)
            grid_path = self._write(root / "grid.json", grid)
            profiles = materialize_fit_profiles(grid)
            profiles["candidate_grid_file_sha256"] = sha256_file(grid_path)
            profiles_path = self._write(root / "profiles.json", profiles)
            protocol = self._write(root / "protocol.json", protocol_value)
            ordinary = {}
            for name in ("crossover", "template", "archive", "coordinator", "pull"):
                ordinary[name] = self._write(root / f"{name}.json", {"name": name})
            snapshot = self._write(root / "snapshot.json", {
                "node": "a06",
                "selection_contract": {"npus": 0, "cpus": 32, "mem_mib": 196608},
            })
            manifests = {}
            receipts = {}
            site_rows = [
                {"layer": 3, "component": "self_attn"},
                {"layer": 11, "component": "mlp"},
                {"layer": 29, "component": "self_attn"},
            ]
            for index, stage in enumerate(("governance", "gradient", "protected"), 1):
                rows = 4008 if stage == "protected" else 6144
                manifest = {
                    "stage": f"qwen35_cdge_v4_2_{stage}_capture",
                    "partition": "subspace_fit", "rows": rows, "complete": True,
                    "sites": site_rows, "final_test_open": False,
                    "final_test_open_count": 0, "production_rollout_approved": False,
                }
                if stage != "protected":
                    manifest.update({
                        "unique_job_keys": 6144,
                        "expected_key_sha256": fit_auth.EXPECTED_KEY_SHA256,
                        "observed_key_sha256": fit_auth.EXPECTED_KEY_SHA256,
                    })
                else:
                    manifest["family_rows"] = fit_auth.EXPECTED_PROTECTED_FAMILIES
                if stage == "gradient":
                    manifest["base_model_parameter_gradients"] = False
                manifests[stage] = self._write(root / f"{stage}_manifest.json", manifest)
                run_id = f"qwen3-5-9b-cdge-v4-2-{stage}-capture-20260731T00000{index}Z"
                receipts[stage] = self._write(root / f"{stage}_receipt.json", {
                    "run_id": run_id, "job_id": 100 + index,
                    "cluster_shared_copy_verified": True,
                    "host_data_copy_verified": True,
                    "local_copy_verified": True,
                    "archive_sha256": str(index) * 64,
                    "slurm_terminal_record": "JobState=COMPLETED ExitCode=0:0",
                })
            output = root / "authorization.json"
            args = argparse.Namespace(
                code_root=Path("/workspace/context-mismatch-qwen3-5-9b/code-v97"),
                selector_snapshot=snapshot, protocol=protocol,
                crossover_contract=ordinary["crossover"], v3_template=ordinary["template"],
                site_manifest=site_path, candidate_grid=grid_path, fit_profiles=profiles_path,
                governance_manifest=manifests["governance"], gradient_manifest=manifests["gradient"],
                protected_manifest=manifests["protected"],
                governance_receipt=receipts["governance"], gradient_receipt=receipts["gradient"],
                protected_receipt=receipts["protected"], archive_helper=ordinary["archive"],
                archive_coordinator=ordinary["coordinator"], pull_helper=ordinary["pull"],
                output=output, selected_node="a06", site="11:mlp",
                created_utc="2026-07-31T00:00:00Z",
            )
            original_sha = sha256_file
            bundle = str(args.code_root / "bundle.sha256")
            with mock.patch.object(
                fit_auth,
                "sha256_file",
                side_effect=lambda path: "f" * 64 if str(path) == bundle else original_sha(path),
            ):
                value = fit_auth.materialize(args)
            self.assertEqual(value["site"], "11:mlp")
            self.assertEqual(value["fit_profile_count"], 3)
            self.assertEqual(value["training_candidate_count"], 3)
            self.assertEqual(value["materialized_candidate_count"], 9)
            self.assertTrue(value["authorization_id"].endswith("-code-v97"))
            self.assertEqual(len(value["fit_profile_ids"]), 3)
            self.assertFalse(value["final_test_open"])


if __name__ == "__main__":
    unittest.main()
