import unittest

from scripts.benchmark_v18.lock_native_v4_2 import build
from scripts.benchmark_v18.materialize_native_controls_authorization import frozen_tie


def candidate(candidate_id, cap, normalized=0.25, raw=0.24):
    return {
        "candidate_id": candidate_id, "behavior_eligible": True, "site": "31:mlp",
        "normalized_gap_reduction_lower_95": normalized, "gap_reduction_lower_95": raw,
        "candidate_config": {"max_relative_correction": cap, "rank_profile": "balanced", "layer": 31},
    }


def control(candidate_id, correction):
    return {
        "method": "C-DGE-V4.2", "stage": "qwen35_cdge_v4_2_native_protected_controls",
        "candidate_id": candidate_id, "audit": {"success": True, "row_count": 2856,
        "unique_case_keys": 2856,
        "observed_key_sha256": "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77",
        "expected_key_sha256": "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77"},
        "controls_admissible": True, "protected_forced_on_relative_correction": {
            "aggregation": "maximum of the two per-direction means over protected-control rows",
            "maximum": correction}, "final_test_open": False, "final_test_open_count": 0,
        "production_rollout_approved": False,
    }


class TiedControlsTests(unittest.TestCase):
    def setUp(self):
        self.a = candidate("CDGE42-" + "a" * 16, 0.05)
        self.b = candidate("CDGE42-" + "b" * 16, 0.075)
        self.merge = {"behavior_complete": True, "candidate_count": 27,
                      "candidates": [self.a, self.b], "final_test_open": False,
                      "final_test_open_count": 0, "production_rollout_approved": False}

    def test_frozen_tie(self):
        self.assertEqual({x["candidate_id"] for x in frozen_tie(self.merge)}, {self.a["candidate_id"], self.b["candidate_id"]})

    def test_correction_precedes_cap(self):
        report, lock = build(self.merge, [control(self.a["candidate_id"], 0.02), control(self.b["candidate_id"], 0.01)], [{}, {}], [])
        self.assertTrue(report["admissible"])
        self.assertEqual(lock["candidate_id"], self.b["candidate_id"])

    def test_cap_breaks_correction_tie(self):
        report, lock = build(self.merge, [control(self.a["candidate_id"], 0.01), control(self.b["candidate_id"], 0.01)], [{}, {}], [])
        self.assertTrue(report["admissible"])
        self.assertEqual(lock["candidate_id"], self.a["candidate_id"])

    def test_both_controls_are_required(self):
        report, lock = build(self.merge, [control(self.a["candidate_id"], 0.01)], [{}], [])
        self.assertFalse(report["admissible"])
        self.assertIsNone(lock)


if __name__ == "__main__":
    unittest.main()
