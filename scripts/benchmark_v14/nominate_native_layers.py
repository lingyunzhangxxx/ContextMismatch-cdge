#!/usr/bin/env python3
"""Audit the full residual scan and freeze the component-scan layer set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v14.native_discovery import audit_scan_rows, load_protocol, _score_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scan-plan", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = load_protocol(args.protocol)
    plan = json.loads(args.scan_plan.read_text())
    environment = json.loads(args.environment.read_text())
    rows = load_jsonl(args.rows)
    audit = audit_scan_rows(rows, {
        "row_count": environment["planned_rows"],
        "key_sha256": environment["expected_key_sha256"],
    })
    if not audit["success"]:
        raise ValueError("residual scan audit failed")
    ranking = [
        _score_rows(rows, layer=layer, component="residual")
        for layer in plan["residual_layers"]
    ]
    ranking.sort(key=lambda row: (-row["ranking_score"], row["layer"]))
    nominated = [row["layer"] for row in ranking[: plan["residual_top_k"]]]
    value = {
        "schema_version": 1,
        "stage": "qwen35_cdge_v4_2_native_residual_nomination",
        "method": "C-DGE-V4.2",
        "locked": True,
        "nominated_layers": nominated,
        "ranking": ranking,
        "audit": audit,
        "protocol_sha256": sha256_file(args.protocol),
        "scan_plan_sha256": sha256_file(args.scan_plan),
        "rows_sha256": sha256_file(args.rows),
        "environment_sha256": sha256_file(args.environment),
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"nominated_layers": nominated, "output_sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
