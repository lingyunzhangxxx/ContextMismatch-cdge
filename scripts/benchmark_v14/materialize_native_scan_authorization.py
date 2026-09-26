#!/usr/bin/env python3
"""Create a SHA-bound code-v89 authorization for one native scan stage."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v14.native_discovery import (
    load_protocol, scan_factorial_axes, scan_key, select_scan_items,
    smoke_scan_axes,
)


def source(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "residual", "component"), required=True
    )
    for name in (
        "code_root", "selector_snapshot", "protocol", "scan_plan", "layer_manifest",
        "manifest", "model_contract", "archive_helper", "archive_coordinator",
        "pull_helper", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc")
    args = parser.parse_args()
    if args.output is None or args.output.exists():
        raise FileExistsError(args.output)
    expected_root = Path("/workspace/context-mismatch-qwen3-5-9b/code-v89")
    if args.code_root != expected_root:
        raise ValueError("native discovery authorization requires exact code-v89")
    if not re.fullmatch(r"a[0-9]{2}", args.selected_node):
        raise ValueError("invalid selected node")
    snapshot = json.loads(args.selector_snapshot.read_text())
    if snapshot.get("node") != args.selected_node:
        raise ValueError("selector snapshot node mismatch")
    selection = snapshot.get("selection_contract", {})
    if any(selection.get(field) != value for field, value in {
        "npus": 1, "cpus": 8, "mem_mib": 131072,
    }.items()):
        raise ValueError("native scan selector resource mismatch")
    protocol = load_protocol(args.protocol)
    plan = json.loads(args.scan_plan.read_text())
    items = select_scan_items(load_jsonl(args.manifest), protocol, mode=args.mode)
    factorial = scan_factorial_axes(protocol, args.mode)
    if args.mode == "smoke":
        layers, components = smoke_scan_axes(protocol, plan)
    elif args.mode == "residual":
        layers, components = list(plan["residual_layers"]), ["residual"]
    else:
        if args.layer_manifest is None:
            raise ValueError("component authorization requires layer manifest")
        layer_value = json.loads(args.layer_manifest.read_text())
        if layer_value.get("locked") is not True:
            raise ValueError("component layer manifest is not locked")
        layers = [int(value) for value in layer_value["nominated_layers"]]
        components = list(protocol["native_discovery"]["components"])
    keys = {
        scan_key(args.mode, item["item_id"], role, style, swap, source_regime, layer, component, coefficient)
        for item in items
        for role in factorial["declared_roles"]
        for style in factorial["history_styles"]
        for swap in factorial["label_swaps"]
        for source_regime in factorial["source_regimes"]
        for layer in layers
        for component in components
        for coefficient in protocol["native_discovery"]["patch_coefficients"]
    }
    key_sha = hashlib.sha256(
        "".join(f"{key}\n" for key in sorted(keys)).encode("utf-8")
    ).hexdigest()
    created = args.created_utc or dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-5-9b-cdge-v4-2-native-{args.mode}-scan-code-v89-{created}",
        "created_utc": created,
        "stage": f"qwen35_cdge_v4_2_native_{args.mode}_scan",
        "method": "C-DGE-V4.2",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "execution_node": args.selected_node,
        "execution_allowed": True,
        "protocol_sha256": sha256_file(args.protocol),
        "scan_plan_sha256": sha256_file(args.scan_plan),
        "layer_manifest_sha256": sha256_file(args.layer_manifest) if args.layer_manifest else None,
        "layer_manifest": source(args.layer_manifest) if args.layer_manifest else None,
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "model_contract_sha256": sha256_file(args.model_contract),
        "layers": layers,
        "components": components,
        "expected_rows": len(keys),
        "expected_key_sha256": key_sha,
        "scientific_efficacy_eligible": args.mode != "smoke",
        "smoke_contract": ({
            "items_per_benchmark": 1,
            "layers": layers,
            "components": components,
            "factorial_axes": factorial,
            "purpose": "engineering-only hook, identity, monitor, archive, and pull validation",
        } if args.mode == "smoke" else None),
        "node_selection_snapshot": source(args.selector_snapshot),
        "monitoring_contract": {
            "continuous_watch_required": True,
            "poll_seconds": 1,
            "archive_helper": source(args.archive_helper),
            "archive_coordinator": source(args.archive_coordinator),
            "pull_helper": source(args.pull_helper),
        },
        "resource_contract": {
            "partition": "a01", "nodes": 1, "ntasks": 1,
            "cpus_per_task": 8, "mem_mib": 131072,
            "npu_type": "910B3", "npus": 1,
            "time_limit": "01:00:00" if args.mode == "smoke" else "12:00:00",
            "node": args.selected_node,
        },
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
