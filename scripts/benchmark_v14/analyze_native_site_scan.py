#!/usr/bin/env python3
"""Audit model-native causal scans and freeze the Qwen3.5 site manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v14.native_discovery import (
    audit_scan_rows,
    load_protocol,
    materialize_candidate_grid,
    select_native_sites,
    value_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scan-plan", type=Path, required=True)
    parser.add_argument("--residual-rows", type=Path, required=True)
    parser.add_argument("--residual-environment", type=Path, required=True)
    parser.add_argument("--component-rows", type=Path, required=True)
    parser.add_argument("--component-environment", type=Path, required=True)
    parser.add_argument("--site-output", type=Path, required=True)
    parser.add_argument("--grid-output", type=Path, required=True)
    args = parser.parse_args()
    if args.site_output.exists() or args.grid_output.exists():
        raise FileExistsError("native site/grid outputs must be fresh")
    protocol = load_protocol(args.protocol)
    plan = json.loads(args.scan_plan.read_text())
    if plan.get("protocol_sha256") != sha256_file(args.protocol):
        raise ValueError("scan plan protocol binding mismatch")
    residual_environment = json.loads(args.residual_environment.read_text())
    component_environment = json.loads(args.component_environment.read_text())
    residual_rows = load_jsonl(args.residual_rows)
    component_rows = load_jsonl(args.component_rows)
    residual_audit = audit_scan_rows(residual_rows, {
        "row_count": residual_environment["planned_rows"],
        "key_sha256": residual_environment["expected_key_sha256"],
    })
    component_audit = audit_scan_rows(component_rows, {
        "row_count": component_environment["planned_rows"],
        "key_sha256": component_environment["expected_key_sha256"],
    })
    if not residual_audit["success"] or not component_audit["success"]:
        raise ValueError("cannot select sites from a failed scan audit")
    site = select_native_sites(
        residual_rows=residual_rows,
        component_rows=component_rows,
        protocol=protocol,
    )
    site.update({
        "protocol_sha256": sha256_file(args.protocol),
        "scan_plan_sha256": sha256_file(args.scan_plan),
        "residual_rows_sha256": sha256_file(args.residual_rows),
        "residual_environment_sha256": sha256_file(args.residual_environment),
        "component_rows_sha256": sha256_file(args.component_rows),
        "component_environment_sha256": sha256_file(args.component_environment),
        "audits": {"residual": residual_audit, "component": component_audit},
    })
    site.pop("selection_sha256", None)
    site["selection_sha256"] = value_sha256(site)
    atomic_write_text(args.site_output, json.dumps(site, indent=2, sort_keys=True) + "\n")
    grid = materialize_candidate_grid(site, protocol)
    grid.pop("candidate_grid_sha256", None)
    grid.update({
        "protocol_sha256": sha256_file(args.protocol),
        "site_manifest_file_sha256": sha256_file(args.site_output),
    })
    grid["candidate_grid_sha256"] = value_sha256(grid)
    atomic_write_text(args.grid_output, json.dumps(grid, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "selected_sites": site["selected_sites_ordered"],
        "candidate_count": grid["candidate_count"],
        "site_manifest_sha256": sha256_file(args.site_output),
        "candidate_grid_sha256": sha256_file(args.grid_output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
