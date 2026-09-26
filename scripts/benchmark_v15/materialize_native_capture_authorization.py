#!/usr/bin/env python3
"""Materialize one SHA-bound Qwen3.5 C-DGE V4.2 capture authorization."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


EXPECTED_ROWS = {"governance": 6144, "gradient": 6144, "protected": 4008}
EXPECTED_KEY_SHA256 = "488835b013ebdf7413d29fa3490bf937700e1b542bac9b0539aa31ba966dcd35"


def _source(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _site_manifest(path: Path) -> tuple[dict, list[str]]:
    value = json.loads(path.read_text())
    required = {
        "stage": "qwen35_cdge_v4_2_native_site_selection",
        "method": "C-DGE-V4.2",
        "selection_partition": "subspace_fit",
        "locked": True,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ValueError(f"site-manifest mismatch: {field}")
    rows = value.get("selected_sites_ordered", [])
    sites = [f"{int(row['layer'])}:{row['component']}" for row in rows]
    if len(sites) != 3 or len(set(sites)) != 3:
        raise ValueError("site manifest must contain exactly three unique sites")
    audits = value.get("audits", {})
    if set(audits) != {"residual", "component"} or not all(
        audit.get("success") is True for audit in audits.values()
    ):
        raise ValueError("site manifest does not contain two successful scan audits")
    return value, sites


def materialize(args: argparse.Namespace) -> dict:
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("method_short_name") != "C-DGE-V4.2":
        raise ValueError("unexpected V4.2 protocol method")
    for field, expected in {
        "status": "frozen_before_qwen3_5_native_discovery_forward",
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if protocol.get(field) != expected:
            raise ValueError(f"protocol mismatch: {field}")
    _, sites = _site_manifest(args.site_manifest)
    created = args.created_utc or dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-5-9b-cdge-v4-2-{args.stage}-capture-code-v90",
        "created_utc": created,
        "stage": f"qwen35_cdge_v4_2_{args.stage}_capture",
        "method": "C-DGE-V4.2",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(
            args.code_root / "bundle.sha256"
        ),
        "execution_node": args.selected_node,
        "slurm_partition": "a01",
        "execution_allowed": True,
        "replication_protocol_sha256": sha256_file(args.protocol),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "model_contract_sha256": sha256_file(args.model_contract),
        "site_manifest": _source(args.site_manifest),
        "site_manifest_sha256": sha256_file(args.site_manifest),
        "sites": sites,
        "expected_rows": EXPECTED_ROWS[args.stage],
        "node_selection_snapshot": _source(args.selector_snapshot),
        "model_contract": _source(args.model_contract),
        "monitoring_contract": {
            "continuous_watch_required": True,
            "poll_seconds": 1,
            "archive_helper": _source(args.archive_helper),
            "archive_coordinator": _source(args.archive_coordinator),
            "pull_helper": _source(args.pull_helper),
        },
        "resource_contract": {
            "partition": "a01", "nodes": 1, "ntasks": 1,
            "cpus_per_task": 8, "mem_mib": 131072,
            "npu_type": "910B3", "npus": 1,
            "time_limit": "12:00:00", "node": args.selected_node,
        },
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if args.stage == "protected":
        if args.controls is None:
            raise ValueError("protected capture requires controls")
        value["controls_sha256"] = sha256_file(args.controls)
    else:
        if args.design_audit is None:
            raise ValueError("governance/gradient capture requires a design audit")
        value["design_audit_sha256"] = sha256_file(args.design_audit)
        value["expected_key_sha256"] = EXPECTED_KEY_SHA256
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=EXPECTED_ROWS, required=True)
    for name in (
        "code_root", "selector_snapshot", "protocol", "crossover_contract",
        "manifest", "model_contract", "site_manifest", "archive_helper",
        "archive_coordinator", "pull_helper", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc")
    parser.add_argument("--design-audit", type=Path)
    parser.add_argument("--controls", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = materialize(args)
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "stage": value["stage"], "sites": value["sites"],
        "authorization_sha256": sha256_file(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
