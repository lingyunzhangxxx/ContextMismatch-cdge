import copy
import unittest

from scripts.benchmark_v14.native_discovery import materialize_candidate_grid, value_sha256
from scripts.benchmark_v15.native_fit_profiles import materialize_fit_profiles


class NativeFitProfilesTest(unittest.TestCase):
    def setUp(self):
        self.protocol = {
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
            }
        }
        self.site_manifest = {
            "locked": True,
            "operator_dev_accessed": False,
            "selected_sites_ordered": [
                {"layer": 3, "component": "self_attn"},
                {"layer": 11, "component": "mlp"},
                {"layer": 29, "component": "self_attn"},
            ],
        }

    def test_groups_27_candidates_into_nine_fit_profiles(self):
        grid = materialize_candidate_grid(self.site_manifest, self.protocol)
        profiles = materialize_fit_profiles(grid)
        self.assertEqual(profiles["fit_profile_count"], 9)
        self.assertEqual(profiles["candidate_count"], 27)
        self.assertEqual(
            {tuple(row["max_relative_correction"] for row in profile["cap_candidates"])
             for profile in profiles["profiles"]},
            {(0.025, 0.05, 0.075)},
        )
        self.assertTrue(all(profile["training_cap"] == 0.075 for profile in profiles["profiles"]))

    def test_rejects_missing_cap(self):
        grid = materialize_candidate_grid(self.site_manifest, self.protocol)
        broken = copy.deepcopy(grid)
        broken["candidates"].pop()
        with self.assertRaisesRegex(ValueError, "semantic SHA"):
            materialize_fit_profiles(broken)

    def test_rejects_forged_semantic_sha(self):
        grid = materialize_candidate_grid(self.site_manifest, self.protocol)
        grid["candidate_grid_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "semantic SHA"):
            materialize_fit_profiles(grid)

    def test_is_deterministic(self):
        grid = materialize_candidate_grid(self.site_manifest, self.protocol)
        first = materialize_fit_profiles(grid)
        reversed_grid = copy.deepcopy(grid)
        reversed_grid["candidates"].reverse()
        reversed_grid.pop("candidate_grid_sha256")
        reversed_grid["candidate_grid_sha256"] = value_sha256(reversed_grid)
        second = materialize_fit_profiles(reversed_grid)
        self.assertEqual(first["profiles"], second["profiles"])

    def test_accepts_provenance_when_semantic_sha_is_recomputed(self):
        grid = materialize_candidate_grid(self.site_manifest, self.protocol)
        grid.pop("candidate_grid_sha256")
        grid.update({
            "protocol_sha256": "a" * 64,
            "site_manifest_file_sha256": "b" * 64,
        })
        grid["candidate_grid_sha256"] = value_sha256(grid)
        profiles = materialize_fit_profiles(grid)
        self.assertEqual(profiles["candidate_grid_sha256"], grid["candidate_grid_sha256"])
        self.assertEqual(profiles["candidate_count"], 27)


if __name__ == "__main__":
    unittest.main()
