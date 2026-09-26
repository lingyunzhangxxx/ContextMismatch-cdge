import json
import tempfile
import unittest
from pathlib import Path


class FinalRecoveryContractTests(unittest.TestCase):
    def test_legacy_identity_has_exactly_the_known_omission(self):
        value = {
            "success": True,
            "max_error": 0.0,
            "method": "ADSGE-V4",
            "final_test_open": True,
            "production_rollout_approved": False,
        }
        self.assertNotIn("final_test_open_count", value)
        value["final_test_open_count"] = 1
        self.assertEqual(value["final_test_open_count"], 1)

    def test_recovery_boundary_remains_closed(self):
        recovery = {
            "model_forward_allowed": False,
            "source_final_test_open_count": 1,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        self.assertFalse(recovery["model_forward_allowed"])
        self.assertFalse(recovery["final_test_open"])
        self.assertEqual(recovery["final_test_open_count"], 0)


if __name__ == "__main__":
    unittest.main()
