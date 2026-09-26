import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v14.native_discovery import materialize_candidate_grid
from scripts.benchmark_v15.native_fit_profiles import materialize_fit_profiles


@unittest.skipIf(torch is None, "Torch is required for checkpoint materialization")
class CapCheckpointMaterializationTest(unittest.TestCase):
    def test_clones_weights_and_changes_only_candidate_cap_metadata(self):
        from scripts.benchmark_v15.materialize_cap_checkpoints import (
            materialize_cap_checkpoints,
        )

        protocol = {"native_discovery": {"candidate_grid": {
            "rank_profiles": [
                {"name": "compact", "boundary_rank": 8, "context_rank": 16, "positive_output_rank": 8, "negative_output_rank": 8},
                {"name": "balanced", "boundary_rank": 16, "context_rank": 32, "positive_output_rank": 16, "negative_output_rank": 16},
                {"name": "wide", "boundary_rank": 32, "context_rank": 64, "positive_output_rank": 32, "negative_output_rank": 32},
            ],
            "protected_rank": 32,
            "maximum_relative_corrections": [0.025, 0.05, 0.075],
        }, "fit_split": {}, "behavior_selection_rule": {}}}
        sites = {"locked": True, "operator_dev_accessed": False, "selected_sites_ordered": [
            {"layer": 3, "component": "self_attn"},
            {"layer": 11, "component": "mlp"},
            {"layer": 29, "component": "self_attn"},
        ]}
        grid = materialize_candidate_grid(sites, protocol)
        profiles = materialize_fit_profiles(grid)
        profile = profiles["profiles"][0]
        candidate = next(
            row for row in grid["candidates"]
            if row["candidate_id"] == profile["training_candidate_id"]
        )
        config = candidate["config"]
        checkpoint = {
            "schema_version": 2,
            "kind": "directional_structural_governance_editor",
            "method": "DSGE-V3",
            "base_model_weights_included": False,
            "trainable_parameter_count": 1,
            "sites": [{
                "layer": config["layer"], "component": config["component"],
                "boundary_rank": config["boundary_rank"], "context_rank": config["context_rank"],
                "positive_output_rank": config["positive_output_rank"],
                "negative_output_rank": config["negative_output_rank"],
                "maximum_relative_correction": 0.075,
                "state_dict": {"positive_expert.weight": torch.ones(1)},
            }],
            "final_test_open": False, "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            grid_path = root / "grid.json"
            profiles_path = root / "profiles.json"
            checkpoint_path = root / "training.pt"
            atomic_write_text(grid_path, json.dumps(grid, sort_keys=True) + "\n")
            profiles["candidate_grid_file_sha256"] = sha256_file(grid_path)
            atomic_write_text(profiles_path, json.dumps(profiles, sort_keys=True) + "\n")
            torch.save(checkpoint, checkpoint_path)
            manifest = materialize_cap_checkpoints(
                training_checkpoint=checkpoint_path,
                candidate_grid_path=grid_path,
                fit_profiles_path=profiles_path,
                fit_profile_id=profile["fit_profile_id"],
                output_dir=root / "caps",
            )
            self.assertEqual(manifest["candidate_count"], 3)
            observed = []
            for row in manifest["checkpoints"]:
                value = torch.load(root / "caps" / row["checkpoint"], weights_only=False)
                observed.append(value["sites"][0]["maximum_relative_correction"])
                self.assertTrue(torch.equal(
                    value["sites"][0]["state_dict"]["positive_expert.weight"],
                    checkpoint["sites"][0]["state_dict"]["positive_expert.weight"],
                ))
            self.assertEqual(sorted(observed), [0.025, 0.05, 0.075])


if __name__ == "__main__":
    unittest.main()
