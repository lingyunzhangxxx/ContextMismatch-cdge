from __future__ import annotations

import unittest
from types import SimpleNamespace

try:
    import torch

    from scripts.benchmark_v9.audit_composite_eligibility import (
        _exact_zero_and_coverage,
        _tensor_tree_equal,
        route_metrics,
    )

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for composite eligibility tests")
class CompositeEligibilityTests(unittest.TestCase):
    def _dataset(self):
        return SimpleNamespace(
            target=torch.tensor([1.0, 1.0, 0.0, 0.0]),
            sources=(
                "governance_mismatch",
                "governance_mismatch",
                "governance_matched:matched_verification",
                "protected:fresh_verification",
            ),
            directions=("positive", "negative", "none", "none"),
            label_swaps=(0, 1, 0, -1),
        )

    def test_composite_route_is_boolean_conjunction(self):
        application = torch.tensor([True, True, True, False])
        structural = torch.tensor([True, False, False, True])
        composite = application & structural
        self.assertTrue(torch.equal(composite, torch.tensor([True, False, False, False])))

    def test_route_metrics_keep_source_and_direction_counts(self):
        active = torch.tensor([True, False, False, False])
        report = route_metrics(active, torch.arange(4), self._dataset())
        self.assertEqual(report["rows"], 4)
        self.assertEqual(report["active_rows"], 1)
        self.assertEqual(report["mismatch_true_positive_rate"], 0.5)
        self.assertEqual(report["negative_active_fraction"], 0.0)
        self.assertEqual(report["by_direction"]["positive"]["true_positive_rate"], 1.0)
        self.assertEqual(report["by_direction"]["negative"]["true_positive_rate"], 0.0)

    def test_exact_zero_requires_exact_family_coverage(self):
        report = route_metrics(torch.zeros(4, dtype=torch.bool), torch.arange(4), self._dataset())
        coverage, exact = _exact_zero_and_coverage(
            report,
            prefix="governance_matched:",
            expected_names={"matched_verification"},
        )
        self.assertTrue(coverage)
        self.assertTrue(exact)
        coverage, exact = _exact_zero_and_coverage(
            report,
            prefix="governance_matched:",
            expected_names={"matched_verification", "matched_delegated_choice"},
        )
        self.assertFalse(coverage)
        self.assertFalse(exact)

    def test_tensor_tree_equality_is_exact(self):
        left = {"a": torch.tensor([1.0, 2.0]), "b": [False, 3]}
        right = {"a": torch.tensor([1.0, 2.0]), "b": [False, 3]}
        self.assertTrue(_tensor_tree_equal(left, right))
        right["a"][1] = 2.0001
        self.assertFalse(_tensor_tree_equal(left, right))


if __name__ == "__main__":
    unittest.main()
