import unittest

from scripts.benchmark_v2.analyze_boundary_susceptibility import (
    _band,
    _benchmark_difficulty_test,
    _quantile,
)


class BoundarySusceptibilityTest(unittest.TestCase):
    def test_quantile_uses_linear_interpolation(self):
        self.assertEqual(_quantile([0.0, 1.0, 2.0, 3.0], 0.25), 0.75)
        self.assertEqual(_quantile([0.0, 1.0, 2.0, 3.0], 0.75), 2.25)

    def test_band_excludes_matched_failures(self):
        self.assertIsNone(_band({"matched_correct": False, "matched_margin": -1.0}, 1.0, 3.0))
        self.assertEqual(_band({"matched_correct": True, "matched_margin": 1.0}, 1.0, 3.0), "boundary")
        self.assertEqual(_band({"matched_correct": True, "matched_margin": 2.0}, 1.0, 3.0), "middle")
        self.assertEqual(_band({"matched_correct": True, "matched_margin": 3.0}, 1.0, 3.0), "robust")

    def test_benchmark_test_reports_all_groups(self):
        rows = []
        for benchmark, matched, mismatched in (
            ("a", [True, True], [True, False]),
            ("b", [True, False], [False, False]),
            ("c", [False, False], [False, False]),
        ):
            for base, mismatch in zip(matched, mismatched):
                rows.append(
                    {
                        "benchmark": benchmark,
                        "matched_correct": base,
                        "mismatched_correct": mismatch,
                    }
                )
        report = _benchmark_difficulty_test(rows)
        self.assertEqual(len(report["benchmarks"]), 3)
        self.assertEqual(report["permutations"], 6)


if __name__ == "__main__":
    unittest.main()
