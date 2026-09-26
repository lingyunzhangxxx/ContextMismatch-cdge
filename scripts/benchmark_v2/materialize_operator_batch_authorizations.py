#!/usr/bin/env python3
"""Shard candidate artifacts into immutable, SHA-bound held-run authorizations."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key


SHA256_RE = re.compile(r"[0-9a-f]{64}")
REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b"


def _sha(value: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("invalid SHA256")
    return value


def _artifact(value: dict, field: str) -> dict[str, str]:
    local = Path(value["local_path"])
    remote = value["remote_path"]
    digest = value["sha256"]
    if not remote.startswith(f"{REMOTE_ROOT}/"):
        raise ValueError(f"{field} remote path is outside the owned project")
    if not local.is_file() or sha256_file(local) != digest:
        raise ValueError(f"{field} local artifact SHA mismatch: {local}")
    _sha(digest)
    return {"path": remote, "sha256": digest}


def _behavior_key_hash(manifest: list[dict], contract: dict, half: str) -> str:
    all_jobs = enumerate_jobs(manifest, contract, "operator_dev")
    by_benchmark = defaultdict(set)
    for job in all_jobs:
        by_benchmark[job["item"]["benchmark"]].add(job["item"]["item_id"])
    selected = set()
    for benchmark, ids in sorted(by_benchmark.items()):
        ordered = sorted(ids)
        if len(ordered) != 32:
            raise ValueError(f"operator_dev:{benchmark} does not contain 32 items")
        chosen = ordered[:16] if half == "screening" else ordered[16:]
        selected.update((benchmark, item_id) for item_id in chosen)
    jobs = [
        job
        for job in all_jobs
        if (job["item"]["benchmark"], job["item"]["item_id"]) in selected
    ]
    if len(jobs) != 3072:
        raise ValueError("operator behavior half does not contain 3,072 rows")
    keys = sorted(job_key(job) for job in jobs)
    return hashlib.sha256((("\n".join(keys)) + "\n").encode("utf-8")).hexdigest()


def _case_key(metadata: dict) -> str:
    return "__".join(
        str(value)
        for value in (
            metadata["control_family"],
            metadata.get("benchmark", "none"),
            metadata["case_id"],
            metadata["declared_role"],
            metadata["history_style"],
            metadata["history_realization"],
            metadata["label_swap"],
        )
    )


def _control_key_hash(
    manifest: list[dict], contract: dict, controls: dict, half: str = "selection"
) -> str:
    items = []
    for benchmark in sorted({row["benchmark"] for row in manifest}):
        rows = sorted(
            (
                row
                for row in manifest
                if row["benchmark"] == benchmark and row["partition"] == "operator_dev"
            ),
            key=lambda row: row["item_id"],
        )
        if len(rows) != 32:
            raise ValueError(f"operator_dev:{benchmark} does not contain 32 items")
        items.extend(rows[:16] if half == "screening" else rows[16:])
    roles = list(contract["factorial"]["declared_roles"])
    styles = list(contract["factorial"]["history_styles"])
    realizations = sorted({int(row["history_realization"]) for row in items})
    keys = []
    for role in roles:
        for item in items:
            for label_swap in (0, 1):
                keys.append(
                    _case_key(
                        {
                            "control_family": "fresh_verification",
                            "benchmark": item["benchmark"],
                            "case_id": item["item_id"],
                            "declared_role": role,
                            "history_style": "none",
                            "history_realization": int(item["history_realization"]),
                            "label_swap": label_swap,
                        }
                    )
                )
        for style in styles:
            for item in items:
                for family in ("matched_verification", "explicit_governance_reset"):
                    for label_swap in (0, 1):
                        keys.append(
                            _case_key(
                                {
                                    "control_family": family,
                                    "benchmark": item["benchmark"],
                                    "case_id": item["item_id"],
                                    "declared_role": role,
                                    "history_style": style,
                                    "history_realization": int(item["history_realization"]),
                                    "label_swap": label_swap,
                                }
                            )
                        )
            for realization in realizations:
                for control in controls["supported_user_authority"]:
                    for label_swap in (0, 1):
                        keys.append(
                            _case_key(
                                {
                                    "control_family": "supported_user_authority",
                                    "case_id": control["id"],
                                    "declared_role": role,
                                    "history_style": style,
                                    "history_realization": realization,
                                    "label_swap": label_swap,
                                }
                            )
                        )
        for control in controls["factual_boundary_memory"]:
            for label_swap in (0, 1):
                keys.append(
                    _case_key(
                        {
                            "control_family": "factual_boundary_memory",
                            "case_id": control["id"],
                            "declared_role": role,
                            "history_style": "none",
                            "history_realization": 0,
                            "label_swap": label_swap,
                        }
                    )
                )
    if len(keys) != 2088 or len(set(keys)) != 2088:
        raise ValueError(f"control design has {len(keys)} rows and {len(set(keys))} keys")
    return hashlib.sha256((("\n".join(sorted(keys))) + "\n").encode("utf-8")).hexdigest()


def _filter_ids(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    value = json.loads(path.read_text())
    if "finalists" in value:
        ids = {row["candidate_id"] for row in value["finalists"]}
    elif "candidate_ids" in value:
        ids = set(value["candidate_ids"])
    else:
        raise ValueError("filter report has neither finalists nor candidate_ids")
    if not ids:
        raise ValueError("candidate filter is empty")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("behavior_screening", "behavior_selection", "controls_selection"),
    )
    parser.add_argument("--candidate-ledger", type=Path, action="append", required=True)
    parser.add_argument("--filter-report", type=Path)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--code-version", type=int, required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True, type=_sha)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--model-manifest-remote", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True, type=_sha)
    parser.add_argument("--created-utc")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.code_version < 18 or args.shard_count < 1:
        raise ValueError("invalid code version or shard count")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing authorization directory: {args.output_dir}")
    if not str(args.model_manifest_remote).startswith(f"{REMOTE_ROOT}/"):
        raise ValueError("model manifest is outside the owned project")
    contract = json.loads(args.crossover_contract.read_text())
    extension = json.loads(args.extension_contract.read_text())
    design = json.loads(args.design_audit.read_text())
    manifest = load_jsonl(args.manifest)
    controls = json.loads(args.controls.read_text())
    if extension.get("status") != "frozen_before_operator_forward":
        raise ValueError("operator extension is not frozen before forward")
    if extension.get("production_rollout_approved") is not False:
        raise ValueError("operator extension changes production boundary")
    contract_sha = sha256_file(args.crossover_contract)
    manifest_sha = sha256_file(args.manifest)
    if design.get("stage") != "operator_dev" or design.get("audit", {}).get("success") is not True:
        raise ValueError("operator-dev design audit did not pass")
    if design.get("contract_sha256") != contract_sha:
        raise ValueError("operator-dev design/contract binding mismatch")
    if design.get("manifest_sha256") != manifest_sha:
        raise ValueError("operator-dev design/manifest binding mismatch")
    if design.get("audit", {}).get("final_test_open") is not False:
        raise ValueError("operator-dev design unexpectedly opens final test")
    if design.get("audit", {}).get("production_rollout_approved") is not False:
        raise ValueError("operator-dev design changes production boundary")
    if args.mode == "behavior_screening":
        expected_key_sha = _behavior_key_hash(manifest, contract, "screening")
        expected_rows = 3072
        stage = "operator_dev_screening"
    elif args.mode == "behavior_selection":
        expected_key_sha = _behavior_key_hash(manifest, contract, "selection")
        expected_rows = 3072
        stage = "operator_dev_selection"
    else:
        expected_key_sha = _control_key_hash(manifest, contract, controls)
        expected_rows = 2088
        stage = "operator_controls_selection"

    candidates = {}
    for ledger_path in args.candidate_ledger:
        ledger = json.loads(ledger_path.read_text())
        if ledger.get("final_test_open") is not False:
            raise ValueError(f"candidate ledger unexpectedly opens final test: {ledger_path}")
        if ledger.get("production_rollout_approved") is not False:
            raise ValueError(f"candidate ledger changes production boundary: {ledger_path}")
        if ledger.get("candidate_count") != len(ledger.get("candidates", [])):
            raise ValueError(f"candidate ledger count mismatch: {ledger_path}")
        for row in ledger["candidates"]:
            candidate_id = row["candidate_id"]
            if candidate_id in candidates:
                raise ValueError(f"duplicate candidate across ledgers: {candidate_id}")
            candidates[candidate_id] = row
    selected_ids = _filter_ids(args.filter_report) or set(candidates)
    missing = selected_ids - set(candidates)
    if missing:
        raise ValueError(f"filtered candidates missing from ledgers: {sorted(missing)}")
    selected = []
    for candidate_id in sorted(selected_ids):
        row = candidates[candidate_id]
        value = {
            "candidate_id": candidate_id,
            "candidate_manifest": _artifact(row["candidate_manifest"], "candidate_manifest"),
            "candidate_tensor": _artifact(row["candidate_tensor"], "candidate_tensor"),
        }
        if args.mode == "controls_selection":
            if "identity_report" not in row:
                raise ValueError(f"controls candidate lacks identity report: {candidate_id}")
            value["identity_report"] = _artifact(row["identity_report"], "identity_report")
        selected.append(value)
    if args.shard_count > len(selected):
        raise ValueError("more shards than selected candidates")
    shards = [selected[index :: args.shard_count] for index in range(args.shard_count)]
    created_utc = args.created_utc or (
        dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", created_utc):
        raise ValueError("created UTC must use YYYY-MM-DDTHH:MM:SSZ")
    args.output_dir.mkdir(parents=True)
    outputs = []
    for index, rows in enumerate(shards):
        value = {
            "schema_version": 1,
            "authorization_id": (
                f"context-mismatch-qwen3-8b-{stage.replace('_', '-')}-"
                f"code-v{args.code_version}-shard-{index + 1}-of-{args.shard_count}"
            ),
            "created_utc": created_utc,
            "stage": stage,
            "code_root": f"{REMOTE_ROOT}/code-v{args.code_version}",
            "immutable_code_bundle_manifest_sha256": args.bundle_manifest_sha256,
            "crossover_contract_sha256": contract_sha,
            "operator_contract_sha256": sha256_file(args.operator_contract),
            "extension_contract_sha256": sha256_file(args.extension_contract),
            "benchmark_manifest_sha256": manifest_sha,
            "design_audit_sha256": sha256_file(args.design_audit),
            "mitigation_controls_sha256": sha256_file(args.controls),
            "expected_key_sha256": expected_key_sha,
            "expected_rows_per_candidate": expected_rows,
            "model_manifest": {
                "path": str(args.model_manifest_remote),
                "sha256": args.model_manifest_sha256,
            },
            "shard_index": index,
            "shard_count": args.shard_count,
            "candidates": rows,
            "execution_allowed": True,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        output = args.output_dir / f"{stage}_shard_{index + 1}_of_{args.shard_count}.json"
        atomic_write_text(output, json.dumps(value, indent=2, sort_keys=True) + "\n")
        outputs.append({"path": str(output), "sha256": sha256_file(output), "candidates": len(rows)})
    print(json.dumps({"stage": stage, "candidate_count": len(selected), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
