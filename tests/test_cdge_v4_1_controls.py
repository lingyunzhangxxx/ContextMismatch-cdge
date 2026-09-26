import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.benchmark_v10.cdge_contract import (
    CONTROLS_EXPECTED_KEY_SHA256 as EXPECTED_KEY_SHA256,
    CONTROLS_EXPECTED_ROWS as EXPECTED_ROWS,
    CONTROLS_STAGE as STAGE,
    formalize_control_row,
)
from scripts.benchmark_v10.run_cdge_v4_1_controls_v74 import _promote


FAMILIES = {
    "fresh_verification",
    "matched_verification",
    "matched_delegated_choice",
    "explicit_governance_reset",
    "supported_user_authority",
    "factual_boundary_memory",
}


class CdgeControlsAdapterTest(unittest.TestCase):
    def test_frozen_control_contract(self):
        contract = json.loads(
            Path("protocol/QWEN3_8B_CDGE_V4_1_POST_ELIGIBILITY_EVALUATION_V1.json").read_text()
        )["protected_controls"]
        self.assertEqual(contract["stage"], STAGE)
        self.assertEqual(contract["expected_rows"], EXPECTED_ROWS)
        self.assertEqual(contract["expected_key_sha256"], EXPECTED_KEY_SHA256)
        self.assertEqual(set(contract["control_families"]), FAMILIES)
        self.assertTrue(contract["application_gated_identity_required"])
        self.assertTrue(contract["forced_direction_reference_gates_required"])

    def test_compatibility_promotion_is_explicit(self):
        original = {"method": "ADSGE-V4", "case_key": "one", "margin": 1.25}
        promoted = formalize_control_row(original)
        self.assertEqual(original["method"], "ADSGE-V4")
        self.assertEqual(promoted["method"], "C-DGE-V4.1")
        self.assertEqual(promoted["runtime_checkpoint_method"], "ADSGE-V4")
        self.assertEqual(promoted["candidate_id"], "C-DGE-V4.1-27:mlp")
        self.assertTrue(promoted["composite_eligible"])
        self.assertEqual(promoted["margin"], original["margin"])

    def test_v74_promotes_hidden_runtime_files_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_rows = root / ".controls.jsonl.runtime"
            runtime_environment = root / ".environment.json.runtime"
            output = root / "controls.jsonl"
            environment_output = root / "environment.json"
            compatibility_dir = root / "compatibility"
            compatibility_dir.mkdir()
            runtime_rows.write_text(json.dumps({"method": "ADSGE-V4", "case_key": "one"}) + "\n")
            runtime_environment.write_text("{}\n")
            inputs = []
            for name in ("behavior.json", "evaluation.json", "authorization.json"):
                path = root / name
                path.write_text("{}\n")
                inputs.append(path)
            compatibility_files = []
            for name in ("legacy_behavior.json", "legacy_contract.json", "legacy_auth.json"):
                path = compatibility_dir / name
                path.write_text("{}\n")
                compatibility_files.append(path)
            args = Namespace(
                output=output,
                environment_output=environment_output,
                compatibility_dir=compatibility_dir,
                candidate_behavior_analysis=inputs[0],
                evaluation_contract=inputs[1],
                execution_authorization=inputs[2],
            )
            _promote(
                args,
                tuple(compatibility_files),
                runtime_output=runtime_rows,
                runtime_environment=runtime_environment,
            )
            self.assertTrue(output.is_file())
            self.assertTrue(environment_output.is_file())
            self.assertEqual(json.loads(output.read_text())["method"], "C-DGE-V4.1")
            self.assertEqual(json.loads(environment_output.read_text())["stage"], STAGE)
            self.assertTrue((compatibility_dir / "adapter_manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
