#!/usr/bin/env python3
"""Build the untouched prospective-boundary manifest from frozen dataset revisions."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from scripts.benchmark_v1.build_manifest import build_benchmark
from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file


def _expanded_spec(spec: dict) -> dict:
    value = copy.deepcopy(spec)
    value["sample_count"] = 160
    if value["id"] == "bbh":
        value["sample_count_per_config"] = 20
    if value["id"] == "musr":
        value["sample_count_by_split"] = {
            "murder_mysteries": 52,
            "object_placements": 54,
            "team_allocation": 54,
        }
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--benchmark-contract", type=Path, required=True)
    parser.add_argument("--old-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.report):
        if path.exists():
            raise FileExistsError(f"refusing existing prospective artifact: {path}")
    protocol = json.loads(args.protocol.read_text())
    contract = json.loads(args.benchmark_contract.read_text())
    if protocol.get("status") != "frozen_before_any_prospective_boundary_model_forward":
        raise ValueError("prospective protocol is not frozen")
    if protocol.get("production_rollout_approved") is not False:
        raise ValueError("production boundary changed")
    seed = int(protocol["selection"]["seed"])
    old = load_jsonl(args.old_manifest)
    old_ids = {(row["benchmark"], row["native_id"]) for row in old}
    rows: list[dict] = []
    by_benchmark: dict[str, dict] = {}
    for spec in contract["benchmarks"]:
        benchmark = spec["id"]
        expanded = build_benchmark(_expanded_spec(spec), seed)
        selected = [
            row for row in expanded
            if (benchmark, row["native_id"]) not in old_ids
        ]
        if len(selected) != 32:
            raise RuntimeError(f"{benchmark}: expected 32 new items, observed {len(selected)}")
        for row in selected:
            row["partition"] = "prospective_boundary"
        rows.extend(selected)
        by_benchmark[benchmark] = {
            "rows": len(selected),
            "old_overlap": sum((benchmark, row["native_id"]) in old_ids for row in selected),
            "dataset_repository": spec["repository"],
            "dataset_revision": spec["revision"],
        }
    rows.sort(key=lambda row: (row["benchmark"], row["item_id"]))
    overlap = sum((row["benchmark"], row["native_id"]) in old_ids for row in rows)
    if len(rows) != 192 or overlap != 0:
        raise RuntimeError("prospective manifest row-count/overlap audit failed")
    atomic_write_text(args.output, "".join(canonical_json(row) + "\n" for row in rows))
    report = {
        "schema_version": 1,
        "protocol_sha256": sha256_file(args.protocol),
        "benchmark_contract_sha256": sha256_file(args.benchmark_contract),
        "old_manifest_sha256": sha256_file(args.old_manifest),
        "manifest_sha256": sha256_file(args.output),
        "rows": len(rows),
        "unique_item_ids": len({row["item_id"] for row in rows}),
        "old_item_overlap": overlap,
        "partition": "prospective_boundary",
        "by_benchmark": by_benchmark,
        "postselected_on_model_outputs": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False
    }
    atomic_write_text(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
