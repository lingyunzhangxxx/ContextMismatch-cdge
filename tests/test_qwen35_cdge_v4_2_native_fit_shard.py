import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "Torch is required for native fit-shard tests")
class NativeFitShardTest(unittest.TestCase):
    def test_accepts_one_locked_site_from_three_site_native_gradient_capture(self):
        from scripts.benchmark_v4.fit_directional_governance import (
            _validate_gradient_site_scope,
        )

        manifest = {
            "stage": "qwen35_cdge_v4_2_gradient_capture",
            "method": "C-DGE-V4.2",
            "sites": [
                {"layer": 31, "component": "mlp"},
                {"layer": 30, "component": "mlp"},
                {"layer": 28, "component": "mlp"},
            ],
        }
        _validate_gradient_site_scope(manifest, ["31:mlp"], qwen35=True)
        with self.assertRaisesRegex(ValueError, "site set mismatch"):
            _validate_gradient_site_scope(manifest, ["27:mlp"], qwen35=True)
        with self.assertRaisesRegex(ValueError, "site set mismatch"):
            _validate_gradient_site_scope(manifest, ["31:mlp"], qwen35=False)

    def test_derives_locked_v42_fold_and_trust_region_contract(self):
        from scripts.benchmark_v15.run_native_fit_shard import (
            derive_directional_contract,
        )

        repo = Path(__file__).resolve().parents[1]
        template = json.loads((
            repo / "protocol/QWEN3_8B_DIRECTIONAL_STRUCTURAL_GOVERNANCE_EDITOR_V3.json"
        ).read_text())
        protocol = json.loads((
            repo / "protocol/QWEN3_5_9B_CDGE_V4_2_NATIVE_DISCOVERY_V1.json"
        ).read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            for name in ("site", "grid", "profiles", "capture", "protected", "crossover"):
                path = root / f"{name}.json"
                value = {"name": name}
                if name == "grid":
                    value["candidate_grid_sha256"] = "b" * 64
                path.write_text(json.dumps(value) + "\n")
                paths[name] = path
            args = Namespace(
                site="11:mlp",
                site_manifest=paths["site"],
                candidate_grid=paths["grid"],
                fit_profiles=paths["profiles"],
                capture_manifest=paths["capture"],
                protected_manifest=paths["protected"],
                crossover_contract=paths["crossover"],
            )
            profile = {
                "fit_profile_id": "CDGE42FIT-0123456789abcdef",
                "fit_profile_sha256": "a" * 64,
                "rank_profile": "balanced",
            }
            candidate = {
                "candidate_id": "CDGE42-0123456789abcdef",
                "config": {
                    "layer": 11, "component": "mlp",
                    "boundary_rank": 16, "context_rank": 32,
                    "positive_output_rank": 16, "negative_output_rank": 16,
                    "protected_rank": 32, "rank_profile": "balanced",
                    "max_relative_correction": 0.075,
                },
            }
            contract = derive_directional_contract(
                template=template,
                protocol=protocol,
                profile=profile,
                training_candidate=candidate,
                args=args,
                created_utc="2026-07-31T00:00:00Z",
            )
        self.assertEqual(contract["status"], "native_discovery_locked_before_candidate_fit")
        self.assertEqual(contract["fit_split"]["train_folds"], [0, 1, 2, 3, 4])
        self.assertEqual(contract["fit_split"]["calibration_fold"], 5)
        self.assertEqual(contract["fit_split"]["structural_audit_fold"], 6)
        self.assertEqual(contract["fit_split"]["direct_behavior_audit_fold"], 7)
        self.assertEqual(len(contract["candidate_sites"]), 1)
        self.assertEqual(contract["candidate_sites"][0]["max_relative_correction"], 0.075)
        self.assertEqual(
            contract["fit_gates_per_candidate_without_cross_site_averaging"][
                "maximum_observed_relative_correction"
            ],
            0.075001,
        )
        self.assertFalse(contract["final_test_open"])


if __name__ == "__main__":
    unittest.main()
