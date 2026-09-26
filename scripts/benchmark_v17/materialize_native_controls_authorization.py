#!/usr/bin/env python3
"""Materialize the SHA-bound code-v93 authorization for native protected controls."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v17.run_native_controls import EXPECTED_KEY, EXPECTED_ROWS


def _source(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def _receipt(path: Path, run_id: str) -> dict:
    value = json.loads(path.read_text())
    if value.get("run_id") != run_id:
        raise ValueError("behavior receipt run mismatch")
    for field in ("cluster_shared_copy_verified", "host_data_copy_verified", "local_copy_verified"):
        if value.get(field) is not True:
            raise ValueError(f"behavior receipt is not three-copy closed: {field}")
    terminal = str(value.get("slurm_terminal_record", ""))
    if "JobState=COMPLETED" not in terminal or "ExitCode=0:0" not in terminal:
        raise ValueError("behavior receipt lacks terminal provenance")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "code_root", "selector_snapshot", "protocol", "candidate_grid", "behavior_merge",
        "behavior_run_dir", "behavior_receipt", "crossover_contract", "manifest",
        "manifest_report", "controls", "model_manifest", "archive_helper",
        "archive_coordinator", "pull_helper", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if str(args.code_root) != "/workspace/context-mismatch-qwen3-5-9b/code-v93":
        raise ValueError("native controls require immutable code-v93")
    if not re.fullmatch(r"a[0-9]{2}", args.selected_node):
        raise ValueError("invalid selected node")
    selector = json.loads(args.selector_snapshot.read_text())
    selection = selector.get("selection_contract", {})
    if selector.get("node") != args.selected_node:
        raise ValueError("selector node mismatch")
    if (selection.get("npus"), selection.get("cpus"), selection.get("mem_mib")) != (1, 8, 131072):
        raise ValueError("selector resource mismatch")
    protocol = json.loads(args.protocol.read_text())
    grid = json.loads(args.candidate_grid.read_text())
    merge = json.loads(args.behavior_merge.read_text())
    if protocol.get("method_short_name") != "C-DGE-V4.2" or grid.get("candidate_count") != 27:
        raise ValueError("native protocol/grid lineage is incomplete")
    selected = merge.get("control_finalist")
    if merge.get("control_finalist_selected") is not True or not isinstance(selected, dict):
        raise ValueError("behavior merge lacks one control finalist")
    candidate_id = selected["candidate_id"]
    run_id = selected["source_run_id"]
    if args.behavior_run_dir.name != run_id:
        raise ValueError("behavior run path mismatch")
    _receipt(args.behavior_receipt, run_id)
    shard_path = args.behavior_run_dir / "behavior" / "native_behavior_shard_report.json"
    shard = json.loads(shard_path.read_text())
    if shard.get("behavior_complete") is not True or shard.get("site") != selected["site"]:
        raise ValueError("selected behavior shard is incomplete")
    matches = [row for row in shard.get("candidates", []) if row.get("candidate_id") == candidate_id]
    if len(matches) != 1 or matches[0].get("behavior_eligible") is not True:
        raise ValueError("selected candidate is absent or ineligible")
    row = matches[0]
    behavior_analysis = args.behavior_run_dir / "behavior" / row["analysis"]
    if sha256_file(behavior_analysis) != row["analysis_sha256"]:
        raise ValueError("selected behavior analysis SHA mismatch")
    behavior_auth_path = args.behavior_run_dir / "execution_authorization.json"
    behavior_auth = json.loads(behavior_auth_path.read_text())
    records = [item for item in behavior_auth.get("candidates", []) if item.get("candidate_id") == candidate_id]
    if len(records) != 1:
        raise ValueError("selected candidate artifact record is missing")
    record = records[0]
    profile_id = record["fit_profile_id"]
    compat_fit = args.behavior_run_dir / "behavior" / profile_id / "compatibility_fit_report.json"
    artifacts = {
        "protocol": args.protocol,
        "candidate_grid": args.candidate_grid,
        "behavior_merge": args.behavior_merge,
        "behavior_shard_report": shard_path,
        "behavior_authorization": behavior_auth_path,
        "behavior_analysis": behavior_analysis,
        "behavior_receipt": args.behavior_receipt,
        "compatibility_fit_report": compat_fit,
        "crossover_contract": args.crossover_contract,
        "manifest": args.manifest,
        "manifest_report": args.manifest_report,
        "controls": args.controls,
        "model_manifest": args.model_manifest,
        "archive_helper": args.archive_helper,
        "archive_coordinator": args.archive_coordinator,
        "pull_helper": args.pull_helper,
    }
    for name in ("checkpoint", "editor_contract", "source_fit_report", "cap_manifest"):
        source = record[name]
        path = Path(source["path"])
        if sha256_file(path) != source["sha256"]:
            raise ValueError(f"selected candidate artifact SHA mismatch: {name}")
        artifacts[name] = path
    bound = {name: _source(path) for name, path in artifacts.items()}
    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-5-9b-cdge-v4-2-native-controls-{candidate_id}-{args.created_utc}",
        "created_utc": args.created_utc,
        "stage": "qwen35_cdge_v4_2_native_protected_controls",
        "method": "C-DGE-V4.2",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "execution_allowed": True,
        "candidate_id": candidate_id,
        "site": selected["site"],
        "candidate_config": selected["candidate_config"],
        "behavior_finalist_eligible": True,
        "expected_rows": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY,
        "forced_directions": ["positive", "negative"],
        "bound_artifacts": bound,
        "resource_contract": {
            "partition": "a01", "nodes": 1, "ntasks": 1,
            "node": args.selected_node, "cpus_per_task": 8,
            "mem_mib": 131072, "npu_type": "910B3", "npus": 1,
            "time_limit": "12:00:00",
        },
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "candidate_id": candidate_id, "site": selected["site"],
        "authorization_sha256": sha256_file(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
