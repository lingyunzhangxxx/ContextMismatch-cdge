#!/usr/bin/env python3
"""Run the same-identity editor suite and reject skips as non-evidence."""

from __future__ import annotations

import argparse
import json
import unittest
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-tests", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.expected_tests <= 0:
        raise ValueError("expected test count must be positive")
    if args.output.exists():
        raise FileExistsError(f"refusing existing same-identity Torch report: {args.output}")

    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.test_same_identity_v3_supplement"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = (
        result.testsRun == args.expected_tests
        and not result.failures
        and not result.errors
        and not result.skipped
        and not result.unexpectedSuccesses
        and result.wasSuccessful()
    )
    report = {
        "schema_version": 1,
        "stage": "same_identity_v3_v5_torch_tests",
        "expected_tests": args.expected_tests,
        "tests_run": result.testsRun,
        "failure_count": len(result.failures),
        "error_count": len(result.errors),
        "skipped_count": len(result.skipped),
        "unexpected_success_count": len(result.unexpectedSuccesses),
        "all_tests_passed": passed,
        "operator_dev_accessed_for_fit": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("same-identity Torch tests did not pass without skips")


if __name__ == "__main__":
    main()
