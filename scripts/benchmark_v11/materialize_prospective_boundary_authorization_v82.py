#!/usr/bin/env python3
"""Create one SHA-bound authorization for the prospective boundary run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


SHA_RE = re.compile(r"[0-9a-f]{64}")
WORK_ROOT = "/workspace/context-mismatch-qwen3-8b"


def _sha(value: str, name: str) -> str:
    if not SHA_RE.fullmatch(value):
        raise ValueError(f"invalid SHA256 for {name}")
    return value


def _source(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--selector-snapshot", type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--archive-helper", type=Path, required=True)
    parser.add_argument("--archive-coordinator", type=Path, required=True)
    parser.add_argument("--pull-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if str(args.code_root) != f"{WORK_ROOT}/code-v82":
        raise ValueError("prospective authorization requires immutable code-v82")
    if not re.fullmatch(r"a[0-9]{2}", args.selected_node):
        raise ValueError("invalid selected node")
    protocol = json.loads(args.protocol.read_text())
    report = json.loads(args.manifest_report.read_text())
    design = json.loads(args.design_audit.read_text())
    model = json.loads(args.model_manifest.read_text())
    if protocol.get("status") != "frozen_before_any_prospective_boundary_model_forward":
        raise ValueError("prospective protocol is not frozen")
    for field, expected in {
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if protocol.get(field) != expected:
            raise ValueError(f"protocol safety mismatch: {field}")
    protocol_sha = sha256_file(args.protocol)
    manifest_sha = sha256_file(args.manifest)
    if report.get("manifest_sha256") != manifest_sha or report.get("rows") != 192:
        raise ValueError("prospective manifest report mismatch")
    if report.get("old_item_overlap") != 0 or report.get("postselected_on_model_outputs") is not False:
        raise ValueError("prospective selection invariants failed")
    if design.get("stage") != "prospective_boundary" or design.get("audit", {}).get("success") is not True:
        raise ValueError("prospective design audit failed")
    if design.get("contract_sha256") != protocol_sha or design.get("manifest_sha256") != manifest_sha:
        raise ValueError("prospective design binding mismatch")
    expected_rows = int(protocol["stages"]["prospective_boundary"]["expected_rows"])
    if design["audit"].get("row_count") != expected_rows:
        raise ValueError("prospective row-count mismatch")
    expected_key_sha = _sha(design["audit"].get("expected_key_sha256", ""), "job keys")
    if not (model.get("verified") or model.get("success")):
        raise ValueError("model manifest is unverified")
    if model.get("revision") != protocol["base_model"]["revision"]:
        raise ValueError("model revision mismatch")
    snapshot = json.loads(args.selector_snapshot.read_text())
    if snapshot.get("node") != args.selected_node:
        raise ValueError("selector snapshot node mismatch")
    created = args.created_utc or (
        dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", created):
        raise ValueError("invalid created UTC")
    bundle_sha = sha256_file(args.code_root / "bundle.sha256")
    monitoring = {
        "continuous_watch_required": True,
        "poll_seconds": 1,
        "archive_helper": _source(args.archive_helper),
        "archive_coordinator": _source(args.archive_coordinator),
        "pull_helper": _source(args.pull_helper),
    }
    authorization = {
        "schema_version": 1,
        "authorization_id": "qwen3-8b-prospective-boundary-code-v82",
        "created_utc": created,
        "stage": "prospective_boundary",
        "method": "base-model-context-mismatch",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": bundle_sha,
        "execution_node": args.selected_node,
        "slurm_partition": "a01",
        "execution_allowed": True,
        "protocol_sha256": protocol_sha,
        "contract_sha256": protocol_sha,
        "benchmark_manifest_sha256": manifest_sha,
        "manifest_sha256": manifest_sha,
        "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
        "design_audit_sha256": sha256_file(args.design_audit),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "expected_rows": expected_rows,
        "expected_key_sha256": expected_key_sha,
        "expected_matched_pairs": expected_rows // 2,
        "node_selection_snapshot": _source(args.selector_snapshot),
        "model_manifest": _source(args.model_manifest),
        "monitoring_contract": monitoring,
        "resource_contract": {
            "partition": "a01",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 8,
            "mem_mib": 131072,
            "npu_type": "910B3",
            "npus": 1,
            "time_limit": "06:00:00",
            "node": args.selected_node,
        },
        "postselected_on_model_outputs": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    print(json.dumps(authorization, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
