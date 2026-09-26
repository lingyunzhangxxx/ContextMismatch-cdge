#!/usr/bin/env python3
"""Create one immutable behavior authorization for a nine-candidate native-site shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key


EXPECTED_ROWS = 896
EXPECTED_KEY_SHA256 = "738cda7c0964ab75ef864405412028a95d207a1915dd4b58569428b525b874b1"


def _key_hash(rows: list[dict]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(job_key(row) for row in rows)) + "\n").encode()
    ).hexdigest()


def _fold(item_id: str) -> int:
    return int.from_bytes(hashlib.sha256(item_id.encode()).digest()[:8], "big") % 8


def _artifact(path: Path) -> dict:
    if not path.is_file():
        raise ValueError(f"missing bound artifact: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _require_receipt(path: Path, run_id: str) -> None:
    value = json.loads(path.read_text())
    if value.get("run_id") != run_id:
        raise ValueError("native-fit archive receipt is not three-copy closed")
    for field in (
        "cluster_shared_copy_verified", "host_data_copy_verified", "local_copy_verified"
    ):
        if value.get(field) is not True:
            raise ValueError(f"native-fit receipt is not three-copy closed: {field}")
    terminal = str(value.get("slurm_terminal_record", ""))
    if "JobState=COMPLETED" not in terminal or "ExitCode=0:0" not in terminal:
        raise ValueError("native-fit receipt lacks terminal Slurm provenance")
    for field in ("archive_sha256", "cluster_shared_path", "durable_mirror_path"):
        if not value.get(field):
            raise ValueError(f"native-fit receipt is incomplete: {field}")


def _candidate_records(
    *, site: str, fit_run_dir: Path, grid: dict, profiles: dict,
) -> list[dict]:
    shard_path = fit_run_dir / "fit" / "native_fit_shard_report.json"
    shard = json.loads(shard_path.read_text())
    required = {
        "stage": "qwen35_cdge_v4_2_native_fit_shard",
        "method": "C-DGE-V4.2",
        "site": site,
        "fit_complete": True,
        "fit_profile_count": 3,
        "training_run_count": 3,
        "materialized_candidate_count": 9,
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if shard.get(field) != expected:
            raise ValueError(f"native-fit shard mismatch: {field}")
    candidate_map = {row["candidate_id"]: row for row in grid.get("candidates", [])}
    profile_map = {row["fit_profile_id"]: row for row in profiles.get("profiles", [])}
    records: list[dict] = []
    for profile_record in shard.get("profiles", []):
        profile_id = profile_record["fit_profile_id"]
        profile = profile_map.get(profile_id)
        if profile is None or f"{profile['site']['layer']}:{profile['site']['component']}" != site:
            raise ValueError("fit shard contains an unauthorized profile")
        fit_root = fit_run_dir / "fit"
        contract = fit_root / profile_record["directional_contract"]
        fit_report = fit_root / profile_record["directional_fit_report"]
        cap_manifest_path = fit_root / profile_record["cap_checkpoint_manifest"]
        for path, expected_sha in (
            (contract, profile_record["directional_contract_sha256"]),
            (fit_report, profile_record["directional_fit_report_sha256"]),
            (cap_manifest_path, profile_record["cap_checkpoint_manifest_sha256"]),
        ):
            if sha256_file(path) != expected_sha:
                raise ValueError(f"native-fit profile SHA mismatch: {path}")
        cap_manifest = json.loads(cap_manifest_path.read_text())
        if cap_manifest.get("candidate_count") != 3:
            raise ValueError("cap checkpoint manifest is incomplete")
        for checkpoint_record in cap_manifest.get("checkpoints", []):
            candidate_id = checkpoint_record["candidate_id"]
            candidate = candidate_map.get(candidate_id)
            if candidate is None:
                raise ValueError("cap checkpoint candidate is absent from frozen grid")
            config = candidate["config"]
            if f"{config['layer']}:{config['component']}" != site:
                raise ValueError("cap checkpoint belongs to another site")
            checkpoint = cap_manifest_path.parent / checkpoint_record["checkpoint"]
            if sha256_file(checkpoint) != checkpoint_record["checkpoint_sha256"]:
                raise ValueError("cap checkpoint SHA mismatch")
            records.append({
                "candidate_id": candidate_id,
                "candidate_config": config,
                "fit_profile_id": profile_id,
                "fit_profile_sha256": profile["fit_profile_sha256"],
                "fit_eligible": bool(profile_record["fit_eligible"]),
                "checkpoint": _artifact(checkpoint),
                "editor_contract": _artifact(contract),
                "source_fit_report": _artifact(fit_report),
                "cap_manifest": _artifact(cap_manifest_path),
            })
    records.sort(key=lambda row: row["candidate_id"])
    if len(records) != 9 or len({row["candidate_id"] for row in records}) != 9:
        raise ValueError("behavior shard must bind exactly nine unique candidates")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "code_root", "selector_snapshot", "protocol", "candidate_grid", "fit_profiles",
        "fit_run_dir", "fit_receipt", "crossover_contract", "manifest", "manifest_report",
        "design_audit", "model_manifest", "archive_helper", "archive_coordinator",
        "pull_helper", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not re.fullmatch(r"[0-9]+:(self_attn|mlp)", args.site):
        raise ValueError("invalid native behavior site")
    if str(args.code_root) != "/workspace/context-mismatch-qwen3-5-9b/code-v92":
        raise ValueError("native behavior requires immutable code-v92")
    if not re.fullmatch(r"a[0-9]{2}", args.selected_node):
        raise ValueError("invalid selected node")
    selector = json.loads(args.selector_snapshot.read_text())
    if selector.get("node") != args.selected_node:
        raise ValueError("selector snapshot node mismatch")
    selection = selector.get("selection_contract", {})
    if selection.get("npus") != 1 or selection.get("cpus") != 8:
        raise ValueError("native behavior selector CPU/NPU contract mismatch")
    if selection.get("mem_mib") != 131072:
        raise ValueError("native behavior selector memory contract mismatch")
    protocol = json.loads(args.protocol.read_text())
    grid = json.loads(args.candidate_grid.read_text())
    profiles = json.loads(args.fit_profiles.read_text())
    if protocol.get("method_short_name") != "C-DGE-V4.2":
        raise ValueError("unexpected native protocol")
    if grid.get("candidate_count") != 27 or profiles.get("fit_profile_count") != 9:
        raise ValueError("native grid/profile lineage is incomplete")
    manifest = load_jsonl(args.manifest)
    crossover = json.loads(args.crossover_contract.read_text())
    rows = [
        row for row in enumerate_jobs(manifest, crossover, "replication")
        if _fold(row["item"]["item_id"]) == 7
    ]
    if len(rows) != EXPECTED_ROWS or _key_hash(rows) != EXPECTED_KEY_SHA256:
        raise ValueError("frozen fold-7 identity changed")
    design = json.loads(args.design_audit.read_text())
    if design.get("stage") != "replication" or design.get("audit", {}).get("success") is not True:
        raise ValueError("replication design audit failed")
    if design.get("contract_sha256") != sha256_file(args.crossover_contract):
        raise ValueError("replication design audit contract mismatch")
    run_id = args.fit_run_dir.name
    _require_receipt(args.fit_receipt, run_id)
    records = _candidate_records(site=args.site, fit_run_dir=args.fit_run_dir, grid=grid, profiles=profiles)
    bound = {
        "protocol": _artifact(args.protocol),
        "candidate_grid": _artifact(args.candidate_grid),
        "fit_profiles": _artifact(args.fit_profiles),
        "fit_shard_report": _artifact(args.fit_run_dir / "fit" / "native_fit_shard_report.json"),
        "fit_receipt": _artifact(args.fit_receipt),
        "crossover_contract": _artifact(args.crossover_contract),
        "manifest": _artifact(args.manifest),
        "manifest_report": _artifact(args.manifest_report),
        "design_audit": _artifact(args.design_audit),
        "model_manifest": _artifact(args.model_manifest),
        "archive_helper": _artifact(args.archive_helper),
        "archive_coordinator": _artifact(args.archive_coordinator),
        "pull_helper": _artifact(args.pull_helper),
    }
    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-5-9b-cdge-v4-2-native-behavior-{args.site.replace(':', '-')}-{args.created_utc}",
        "created_utc": args.created_utc,
        "stage": "qwen35_cdge_v4_2_native_behavior_shard",
        "method": "C-DGE-V4.2",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "execution_allowed": True,
        "site": args.site,
        "candidate_count": 9,
        "rows_per_candidate": EXPECTED_ROWS,
        "expected_key_sha256_per_candidate": EXPECTED_KEY_SHA256,
        "total_rows": 9 * EXPECTED_ROWS,
        "candidate_grid_sha256": grid["candidate_grid_sha256"],
        "candidates": records,
        "bound_artifacts": bound,
        "resource_contract": {
            "partition": "a01", "nodes": 1, "ntasks": 1,
            "node": args.selected_node, "cpus_per_task": 8,
            "mem_mib": 131072, "npu_type": "910B3", "npus": 1,
            "time_limit": "12:00:00",
        },
        "fit_gate_failure_does_not_suppress_behavior_characterization": True,
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"site": args.site, "candidate_count": 9, "total_rows": 8064,
                      "authorization_sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
