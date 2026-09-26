from __future__ import annotations

import importlib.util
import unittest

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    from scripts.benchmark_v3.run_governance_editor import _fit_gate_status


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for V4 behavior diagnostics")
class V4BehaviorDiagnosticTests(unittest.TestCase):
    def test_v4_failed_gate_schema_is_recognized(self):
        self.assertFalse(_fit_gate_status({"method": "ADSGE-V4", "fit_eligible": False}))

    def test_adaptive_gate_schema_is_preserved(self):
        self.assertTrue(_fit_gate_status({"all_fit_gates_pass": True}))

    def test_unknown_gate_schema_fails_closed(self):
        with self.assertRaises(ValueError):
            _fit_gate_status({"method": "unknown"})


if __name__ == "__main__":
    unittest.main()
