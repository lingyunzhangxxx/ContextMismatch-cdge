import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_v17.merge_native_behavior import _finite, _run_dir
from scripts.benchmark_v17.lock_native_v4_2 import build


class NativeMergeTests(unittest.TestCase):
    def test_finite_is_recursive(self):
        self.assertTrue(_finite({"x": [1.0, 2.0]}))
        self.assertFalse(_finite({"x": [float("nan")]}))

    def test_extracted_bundle_layout_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            run = bundle / "runs" / "run-one"
            (run / "behavior").mkdir(parents=True)
            self.assertEqual(_run_dir(bundle, "run-one"), run)
            with self.assertRaises(ValueError):
                _run_dir(bundle, "run-two")

    def test_lock_requires_terminal_controls(self):
        finalist = {
            "candidate_id": "CDGE42-" + "a" * 16,
            "candidate_config": {"max_relative_correction": 0.025, "rank_profile": "compact"},
            "behavior_eligible": True,
            "normalized_gap_reduction_lower_95": 0.1,
            "gap_reduction_lower_95": 0.2,
        }
        merge = {
            "behavior_complete": True, "candidate_count": 27,
            "control_finalist_selected": True, "control_finalist": finalist,
            "final_test_open": False, "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        controls = {
            "method": "C-DGE-V4.2", "stage": "qwen35_cdge_v4_2_native_protected_controls",
            "candidate_id": finalist["candidate_id"], "audit": {
                "success": True, "row_count": 2856, "unique_case_keys": 2856,
                "observed_key_sha256": "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77",
                "expected_key_sha256": "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77",
            }, "controls_admissible": True, "final_test_open": False,
            "final_test_open_count": 0, "production_rollout_approved": False,
        }
        report, lock = build(merge=merge, controls=controls, receipt={"archive_sha256": "a" * 64}, sources={})
        self.assertTrue(report["admissible"])
        self.assertTrue(lock["locked"])
        controls["controls_admissible"] = False
        report, lock = build(merge=merge, controls=controls, receipt={"archive_sha256": "a" * 64}, sources={})
        self.assertFalse(report["admissible"])
        self.assertIsNone(lock)


if __name__ == "__main__":
    unittest.main()
