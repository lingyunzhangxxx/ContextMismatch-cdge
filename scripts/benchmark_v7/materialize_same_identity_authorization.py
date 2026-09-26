#!/usr/bin/env python3
"""Create one fresh SHA-bound code-v37 same-identity shard authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key
from scripts.benchmark_v2.run_operator_candidate import _half_jobs
from scripts.benchmark_v7.run_same_identity import (
    CANDIDATE_METHODS,
    EXPECTED_KEY_SHA256,
    EXPECTED_ROWS,
    METHOD_SHARDS,
)


REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b"
CODE_VERSION = 37
EXPECTED_CANDIDATES = {
    "fixed_negative_vector": {
        "candidate_id": "d08f09058bbcce1f",
        "family": "fixed_translation",
        "tensor_sha256": "83d7a1000a6fdfc0e183a3812d18e58d6cf1e872fd030daa97627b91dd909e24",
    },
    "symmetric_rank_one": {
        "candidate_id": "237b1257eea89717",
        "family": "symmetric_rank1",
        "tensor_sha256": "87e07e76d5ecfc6198c08bffd818786034e4abb24f7b55286bf57dab35f49722",
    },
}
EXPECTED_V3_CHECKPOINT_SHA256 = "ed5262bbd27470047c3379172e58de23e6bb98a542b9fb5101f828138b11d752"
EXPECTED_V3_FIT_REPORT_SHA256 = "0cda38b3f3ad44460e6f91278277bc1cd66349a5d34f128890d6bc60f60a5f0c"
SUPPLEMENT_SHARDS = ("operators", "experts", "routing")


def _key_hash(jobs: list[dict]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(job_key(job) for job in jobs)) + "\n").encode("utf-8")
    ).hexdigest()


def _record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = {"path": str(path), "sha256": sha256_file(path)}
    if not value["path"].startswith(f"{REMOTE_ROOT}/"):
        raise ValueError("source artifact must remain inside the cluster project root")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--method-shard", choices=SUPPLEMENT_SHARDS, required=True)
    parser.add_argument("--supplement-contract", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path)
    parser.add_argument("--v3-fit-report", type=Path)
    parser.add_argument("--fixed-tensor", type=Path)
    parser.add_argument("--fixed-manifest", type=Path)
    parser.add_argument("--symmetric-tensor", type=Path)
    parser.add_argument("--symmetric-manifest", type=Path)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if args.code_root != Path(f"{REMOTE_ROOT}/code-v{CODE_VERSION}"):
        raise ValueError("same-identity supplement is fixed to immutable code-v37")
    bundle = args.code_root / "bundle.sha256"
    if not bundle.is_file():
        raise FileNotFoundError("code-v36 bundle manifest is missing")
    supplement = json.loads(args.supplement_contract.read_text())
    if supplement.get("status") != "frozen_before_any_same_identity_supplement_forward":
        raise ValueError("same-identity supplement is not frozen")
    scope = supplement["scope"]
    if scope.get("expected_rows_per_method") != EXPECTED_ROWS:
        raise ValueError("supplement row count changed")
    if scope.get("expected_job_key_sha256") != EXPECTED_KEY_SHA256:
        raise ValueError("supplement key SHA changed")
    crossover = json.loads(args.crossover_contract.read_text())
    design = json.loads(args.design_audit.read_text())
    if design.get("stage") != "operator_dev" or design.get("audit", {}).get("success") is not True:
        raise ValueError("operator-dev design audit is invalid")
    if design.get("contract_sha256") != sha256_file(args.crossover_contract):
        raise ValueError("design/crossover binding mismatch")
    if design.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("design/manifest binding mismatch")
    jobs = _half_jobs(
        enumerate_jobs(load_jsonl(args.manifest), crossover, "operator_dev"), "selection"
    )
    if len(jobs) != EXPECTED_ROWS or _key_hash(jobs) != EXPECTED_KEY_SHA256:
        raise ValueError("frozen same-identity job enumeration changed")

    methods = METHOD_SHARDS[args.method_shard]
    sources: dict[str, object] = {"operators": {}}
    if any(method not in CANDIDATE_METHODS for method in methods):
        if args.v3_checkpoint is None or args.v3_fit_report is None:
            raise ValueError("DGE ablations require the frozen V3 checkpoint and fit report")
        checkpoint = _record(args.v3_checkpoint)
        fit_report = _record(args.v3_fit_report)
        if checkpoint["sha256"] != EXPECTED_V3_CHECKPOINT_SHA256:
            raise ValueError("unexpected V3 checkpoint")
        if fit_report["sha256"] != EXPECTED_V3_FIT_REPORT_SHA256:
            raise ValueError("unexpected V3 fit report")
        sources["v3_checkpoint"] = checkpoint
        sources["v3_fit_report"] = fit_report
    if args.method_shard == "operators":
        paths = {
            "fixed_negative_vector": (args.fixed_tensor, args.fixed_manifest),
            "symmetric_rank_one": (args.symmetric_tensor, args.symmetric_manifest),
        }
        for method, (tensor_path, manifest_path) in paths.items():
            if tensor_path is None or manifest_path is None:
                raise ValueError(f"missing source files for {method}")
            tensor = _record(tensor_path)
            manifest_record = _record(manifest_path)
            manifest_value = json.loads(manifest_path.read_text())
            expected = EXPECTED_CANDIDATES[method]
            if manifest_value.get("candidate_id") != expected["candidate_id"]:
                raise ValueError(f"unexpected candidate for {method}")
            if manifest_value.get("config", {}).get("family") != expected["family"]:
                raise ValueError(f"unexpected candidate family for {method}")
            if tensor["sha256"] != expected["tensor_sha256"]:
                raise ValueError(f"unexpected tensor SHA for {method}")
            if manifest_value.get("tensor_sha256") != tensor["sha256"]:
                raise ValueError(f"candidate manifest tensor binding mismatch for {method}")
            sources["operators"][method] = {
                "tensor": tensor,
                "manifest": manifest_record,
            }

    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-8b-dge-same-identity-{args.method_shard}-code-v37",
        "created_utc": args.created_utc,
        "stage": "dge_same_identity_supplement",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(bundle),
        "execution_allowed": True,
        "explicit_user_execution_authorization": True,
        "method_shard": args.method_shard,
        "methods": list(methods),
        "supplement_contract_sha256": sha256_file(args.supplement_contract),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "design_audit_sha256": sha256_file(args.design_audit),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "model_manifest": _record(args.model_manifest),
        "expected_rows_per_method": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "sources": sources,
        "operator_dev_accessed_for_fit": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
