#!/usr/bin/env python3
"""Create immutable SHA-bound DSGE-V3 execution authorizations."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key


REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b"
STAGES = (
    "directional_torch_tests",
    "governance_gradient_capture_smoke",
    "governance_gradient_capture_full",
    "governance_directional_fit",
    "governance_directional_fit_audit",
    "governance_directional_operator_dev",
)


def _key_hash(jobs: list[dict]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(job_key(job) for job in jobs)) + "\n").encode("utf-8")
    ).hexdigest()


def _fold(item_id: str) -> int:
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 8


def _smoke_jobs(jobs: list[dict]) -> list[dict]:
    first_items: dict[str, str] = {}
    for job in sorted(jobs, key=lambda row: (row["item"]["benchmark"], row["item"]["item_id"])):
        first_items.setdefault(job["item"]["benchmark"], job["item"]["item_id"])
    selected = [
        job
        for job in jobs
        if first_items[job["item"]["benchmark"]] == job["item"]["item_id"]
    ]
    if len(selected) != 192:
        raise ValueError("gradient smoke must contain exactly 192 rows")
    return selected


def _operator_selection_jobs(jobs: list[dict]) -> list[dict]:
    by_benchmark: dict[str, list[str]] = {}
    for job in jobs:
        by_benchmark.setdefault(job["item"]["benchmark"], []).append(
            job["item"]["item_id"]
        )
    selected_items = {
        (benchmark, item_id)
        for benchmark, values in by_benchmark.items()
        for item_id in sorted(set(values))[16:]
    }
    selected = [
        job
        for job in jobs
        if (job["item"]["benchmark"], job["item"]["item_id"])
        in selected_items
    ]
    if len(selected) != 3072:
        raise ValueError("operator_dev selection must contain exactly 3,072 rows")
    return selected


def _artifact(path: Path | None, remote: str | None, name: str) -> dict:
    if path is None or not path.is_file():
        raise ValueError(f"{name} local artifact is required")
    if not remote or not remote.startswith(f"{REMOTE_ROOT}/"):
        raise ValueError(f"{name} remote path must be inside the project root")
    return {"path": remote, "sha256": sha256_file(path)}


def _require_safe_contract(path: Path) -> dict:
    value = json.loads(path.read_text())
    required = {
        "status": "design_locked_after_v2_terminal_failure_audit_before_any_v3_forward",
        "method_short_name": "DSGE-V3",
        "code_version_minimum": 29,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"unsafe or unexpected DSGE-V3 contract field: {key}")
    return value


def _validate_design_audit(
    path: Path,
    *,
    expected_stage: str,
    crossover_contract: Path,
    manifest: Path,
) -> None:
    value = json.loads(path.read_text())
    if value.get("stage") != expected_stage or value.get("audit", {}).get("success") is not True:
        raise ValueError("design audit stage or success flag is invalid")
    if value.get("contract_sha256") != sha256_file(crossover_contract):
        raise ValueError("design audit crossover-contract binding mismatch")
    if value.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("design audit benchmark-manifest binding mismatch")


def _validate_model_manifest(path: Path | None, crossover: dict) -> None:
    if path is None or not path.is_file():
        raise ValueError("model manifest is required")
    value = json.loads(path.read_text())
    if not (value.get("verified") or value.get("success")):
        raise ValueError("model manifest is not verified")
    if value.get("revision") != crossover["base_model"]["revision"]:
        raise ValueError("model manifest revision mismatch")


def _require_fields(value: dict, expected: dict, name: str) -> None:
    for field, required in expected.items():
        if value.get(field) != required:
            raise ValueError(f"{name} field mismatch: {field}")


def _validate_fit_inputs(
    *,
    contract: dict,
    editor_contract: Path,
    capture_manifest: Path | None,
    protected_capture_manifest: Path | None,
    gradient_capture_manifest: Path | None,
    full_expected_key_sha256: str,
) -> None:
    paths = {
        "governance capture": capture_manifest,
        "protected capture": protected_capture_manifest,
        "gradient capture": gradient_capture_manifest,
    }
    for name, path in paths.items():
        if path is None or not path.is_file():
            raise ValueError(f"{name} manifest is required")
    assert capture_manifest is not None
    assert protected_capture_manifest is not None
    assert gradient_capture_manifest is not None

    bound = contract.get("bound_inputs", {})
    if bound.get("governance_capture_manifest_sha256") != sha256_file(capture_manifest):
        raise ValueError("governance capture is not bound by the V3 contract")
    if bound.get("protected_capture_manifest_sha256") != sha256_file(
        protected_capture_manifest
    ):
        raise ValueError("protected capture is not bound by the V3 contract")

    capture = json.loads(capture_manifest.read_text())
    _require_fields(
        capture,
        {
            "stage": "governance_capture_full",
            "partition": "subspace_fit",
            "rows": 6144,
            "unique_job_keys": 6144,
            "complete": True,
            "expected_key_sha256": full_expected_key_sha256,
            "observed_key_sha256": full_expected_key_sha256,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "governance capture manifest",
    )
    protected = json.loads(protected_capture_manifest.read_text())
    _require_fields(
        protected,
        {
            "partition": "subspace_fit",
            "rows": 4008,
            "application_gated_and_forced_on_stress_required": True,
            "final_test_open": False,
            "production_rollout_approved": False,
        },
        "protected capture manifest",
    )
    gradient = json.loads(gradient_capture_manifest.read_text())
    _require_fields(
        gradient,
        {
            "stage": "governance_gradient_capture_full",
            "partition": "subspace_fit",
            "rows": 6144,
            "unique_job_keys": 6144,
            "complete": True,
            "expected_key_sha256": full_expected_key_sha256,
            "observed_key_sha256": full_expected_key_sha256,
            "editor_contract_sha256": sha256_file(editor_contract),
            "base_model_parameter_gradients": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "gradient capture manifest",
    )
    fractions = gradient.get("nonzero_gradient_fraction_by_site", {})
    if set(fractions) != {"23:self_attn", "27:mlp", "31:mlp"}:
        raise ValueError("gradient capture site set mismatch")
    if any(float(value) < 0.99 for value in fractions.values()):
        raise ValueError("gradient capture nonzero fraction is below 0.99")


def _validate_fit_report(path: Path, editor_contract: Path) -> dict:
    value = json.loads(path.read_text())
    _require_fields(
        value,
        {
            "stage": "governance_directional_fit",
            "method": "DSGE-V3",
            "fit_complete": True,
            "candidate_count": 3,
            "all_three_candidate_checkpoints_materialized": True,
            "editor_contract_sha256": sha256_file(editor_contract),
            "operator_dev_accessed": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "directional fit report",
    )
    candidates = value.get("candidate_reports", [])
    if len(candidates) != 3:
        raise ValueError("directional fit report candidate set is incomplete")
    contract = json.loads(editor_contract.read_text())
    expected_sites = {
        f"{int(site['layer'])}:{site['component']}"
        for site in contract.get("candidate_sites", [])
    }
    if {row.get("site") for row in candidates} != expected_sites:
        raise ValueError("directional fit report site set differs from the V3 contract")
    checkpoint_shas = {row.get("checkpoint_sha256") for row in candidates}
    if len(checkpoint_shas) != 3 or any(
        not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)
        for sha in checkpoint_shas
    ):
        raise ValueError("directional fit checkpoint set is invalid")
    for row in candidates:
        if row.get("candidate_id") != f"DSGE-V3-{row.get('site')}":
            raise ValueError("directional fit candidate identity is inconsistent")
        if not isinstance(row.get("fit_eligible"), bool):
            raise ValueError("directional fit eligibility is not Boolean")
        if row.get("direct_behavior_characterization_required_even_if_ineligible") is not True:
            raise ValueError("directional fit report suppresses required behavior characterization")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--code-version", type=int, required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path)
    parser.add_argument("--model-manifest-remote")
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--capture-manifest-remote")
    parser.add_argument("--protected-capture-manifest", type=Path)
    parser.add_argument("--protected-capture-manifest-remote")
    parser.add_argument("--gradient-capture-manifest", type=Path)
    parser.add_argument("--gradient-capture-manifest-remote")
    parser.add_argument("--fit-report", type=Path)
    parser.add_argument("--fit-report-remote")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-remote")
    parser.add_argument("--prior-analysis", type=Path)
    parser.add_argument("--prior-analysis-remote")
    parser.add_argument("--expected-tests", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if args.code_version != 29:
        raise ValueError("this immutable workflow materializes code-v29 only")
    if not re.fullmatch(r"[0-9a-f]{64}", args.bundle_manifest_sha256):
        raise ValueError("bundle manifest SHA must be lowercase SHA256 hex")
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", args.created_utc):
        raise ValueError("created UTC must use YYYY-MM-DDTHH:MM:SSZ")
    contract = _require_safe_contract(args.editor_contract)
    crossover = json.loads(args.crossover_contract.read_text())
    manifest = load_jsonl(args.manifest)
    expected_audit_stage = (
        "operator_dev" if args.stage.endswith("operator_dev") else "replication"
    )
    _validate_design_audit(
        args.design_audit,
        expected_stage=expected_audit_stage,
        crossover_contract=args.crossover_contract,
        manifest=args.manifest,
    )
    code_root = f"{REMOTE_ROOT}/code-v{args.code_version}"
    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-8b-{args.stage}-code-v29",
        "created_utc": args.created_utc,
        "stage": args.stage,
        "code_root": code_root,
        "immutable_code_bundle_manifest_sha256": args.bundle_manifest_sha256,
        "execution_allowed": True,
        "editor_contract_filename": args.editor_contract.name,
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "design_audit_sha256": sha256_file(args.design_audit),
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if args.stage == "directional_torch_tests":
        if args.expected_tests <= 0:
            raise ValueError("expected Torch test count must be positive")
        value["expected_tests"] = args.expected_tests
    elif args.stage.startswith("governance_gradient_capture_"):
        jobs = enumerate_jobs(manifest, crossover, "replication")
        if args.stage.endswith("smoke"):
            jobs = _smoke_jobs(jobs)
        _validate_model_manifest(args.model_manifest, crossover)
        model = _artifact(args.model_manifest, args.model_manifest_remote, "model manifest")
        value.update(
            {
                "expected_rows": len(jobs),
                "expected_key_sha256": _key_hash(jobs),
                "model_manifest": model,
                "model_manifest_sha256": model["sha256"],
            }
        )
    elif args.stage == "governance_directional_fit":
        full_jobs = enumerate_jobs(manifest, crossover, "replication")
        _validate_fit_inputs(
            contract=contract,
            editor_contract=args.editor_contract,
            capture_manifest=args.capture_manifest,
            protected_capture_manifest=args.protected_capture_manifest,
            gradient_capture_manifest=args.gradient_capture_manifest,
            full_expected_key_sha256=_key_hash(full_jobs),
        )
        capture = _artifact(
            args.capture_manifest, args.capture_manifest_remote, "capture manifest"
        )
        protected = _artifact(
            args.protected_capture_manifest,
            args.protected_capture_manifest_remote,
            "protected capture manifest",
        )
        gradient = _artifact(
            args.gradient_capture_manifest,
            args.gradient_capture_manifest_remote,
            "gradient capture manifest",
        )
        value.update(
            {
                "capture_manifest": capture,
                "capture_manifest_sha256": capture["sha256"],
                "protected_capture_manifest": protected,
                "protected_capture_manifest_sha256": protected["sha256"],
                "gradient_capture_manifest": gradient,
                "gradient_capture_manifest_sha256": gradient["sha256"],
                "operator_dev_accessed": False,
            }
        )
    else:
        fit_report = _artifact(args.fit_report, args.fit_report_remote, "fit report")
        checkpoint = _artifact(args.checkpoint, args.checkpoint_remote, "checkpoint")
        _validate_model_manifest(args.model_manifest, crossover)
        model = _artifact(args.model_manifest, args.model_manifest_remote, "model manifest")
        assert args.fit_report is not None
        fit_value = _validate_fit_report(args.fit_report, args.editor_contract)
        checkpoint_sha = checkpoint["sha256"]
        candidates = [
            row
            for row in fit_value.get("candidate_reports", [])
            if row.get("checkpoint_sha256") == checkpoint_sha
        ]
        if len(candidates) != 1:
            raise ValueError("checkpoint is not uniquely bound by the fit report")
        value["authorization_id"] = (
            f"qwen3-8b-{args.stage}-code-v29-{checkpoint_sha[:16]}"
        )
        if args.stage.endswith("fit_audit"):
            jobs = [
                job
                for job in enumerate_jobs(manifest, crossover, "replication")
                if _fold(job["item"]["item_id"]) == 7
            ]
        else:
            if candidates[0].get("fit_eligible") is not True:
                raise ValueError("fit-ineligible candidate cannot reach operator_dev")
            prior = _artifact(
                args.prior_analysis, args.prior_analysis_remote, "fit-audit analysis"
            )
            assert args.prior_analysis is not None
            prior_value = json.loads(args.prior_analysis.read_text())
            _require_fields(
                prior_value,
                {
                    "method": "DSGE-V3",
                    "evaluation_stage": "governance_directional_fit_audit",
                    "checkpoint_sha256": checkpoint_sha,
                    "fit_report_sha256": fit_report["sha256"],
                    "editor_contract_sha256": sha256_file(args.editor_contract),
                    "fit_eligible": True,
                    "final_test_open": False,
                    "final_test_open_count": 0,
                    "production_rollout_approved": False,
                },
                "fit-audit analysis",
            )
            if prior_value.get("audit", {}).get("success") is not True or prior_value.get(
                "selection_gate", {}
            ).get("candidate_eligible") is not True:
                raise ValueError("operator_dev requires a passing checkpoint-bound fit audit")
            value["fit_audit_analysis"] = prior
            value["fit_audit_analysis_sha256"] = prior["sha256"]
            jobs = _operator_selection_jobs(
                enumerate_jobs(manifest, crossover, "operator_dev")
            )
        value.update(
            {
                "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                "fit_report": fit_report,
                "fit_report_sha256": fit_report["sha256"],
                "model_manifest": model,
                "model_manifest_sha256": model["sha256"],
                "expected_rows": len(jobs),
                "expected_key_sha256": _key_hash(jobs),
            }
        )
    if contract["final_test_open"] or contract["production_rollout_approved"]:
        raise ValueError("contract safety boundary changed during authorization")
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
