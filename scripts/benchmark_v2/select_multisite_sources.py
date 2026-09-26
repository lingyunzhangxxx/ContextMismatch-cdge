#!/usr/bin/env python3
"""Choose deterministic admissible single-site sources for two/three-site candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v2.select_operator_pareto import _candidate_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-site-ledger", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    ledger = json.loads(args.single_site_ledger.read_text())
    extension = json.loads(args.extension_contract.read_text())
    site_manifest = json.loads(args.operator_site_manifest.read_text())
    locked_sites = {
        f"{int(row['layer'])}:{row['component']}"
        for row in site_manifest["selected_sites_ordered"]
    }
    candidates = [
        _candidate_metrics(entry, args.single_site_ledger.parent, extension)
        for entry in ledger["candidates"]
    ]
    eligible = [
        row
        for row in candidates
        if row["admissible"]
        and len(row["site_order"]) == 1
        and row["site_order"][0] in locked_sites
    ]
    by_family_site = defaultdict(list)
    for row in eligible:
        by_family_site[(row["config"]["family"], row["site_order"][0])].append(row)
    family_sources = {}
    for family in sorted({key[0] for key in by_family_site}):
        sources = []
        for site in sorted(locked_sites):
            rows = by_family_site.get((family, site), [])
            if not rows:
                continue
            sources.append(
                min(
                    rows,
                    key=lambda row: (
                        -row["gap_reduction"],
                        row["worst_protection_fraction"],
                        row["total_rank"],
                        row["mean_relative_intervention_norm"],
                        row["candidate_id"],
                    ),
                )
            )
        if len(sources) >= 2:
            family_sources[family] = sorted(
                sources,
                key=lambda row: (
                    -row["gap_reduction"],
                    row["worst_protection_fraction"],
                    row["candidate_id"],
                ),
            )
    if not family_sources:
        raise SystemExit("no family has admissible sources at two locked sites")
    family = min(
        family_sources,
        key=lambda name: (
            -len(family_sources[name]),
            -sum(row["gap_reduction"] for row in family_sources[name])
            / len(family_sources[name]),
            max(row["worst_protection_fraction"] for row in family_sources[name]),
            name,
        ),
    )
    sources = family_sources[family]
    plans = []
    for site_count in (2, 3):
        if len(sources) >= site_count:
            plans.append(
                {
                    "site_count": site_count,
                    "family": family,
                    "sources": sources[:site_count],
                }
            )
    report = {
        "schema_version": 1,
        "single_site_ledger_sha256": sha256_file(args.single_site_ledger),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "extension_contract_sha256": sha256_file(args.extension_contract),
        "eligible_single_site_candidates": len(eligible),
        "chosen_family": family,
        "family_source_counts": {
            key: len(value) for key, value in sorted(family_sources.items())
        },
        "plans": plans,
        "selection_partition": "operator_dev selection half and protected controls only",
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
