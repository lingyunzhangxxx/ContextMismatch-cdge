"""Frozen identities and strict lineage checks for Qwen3.5 C-DGE evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_v1.common import sha256_file

METHOD = "C-DGE-V4.1"
RUNTIME_METHOD = "ADSGE-V4"
WORK_ROOT = Path("/workspace/context-mismatch-qwen3-5-9b")
MODEL_PATH = Path("/workspace/node-local/context-mismatch-ascend/models/Qwen3.5-9B")
OPERATOR_ROWS = 3072
OPERATOR_KEY = "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e"
CONTROL_ROWS = 2856
CONTROL_KEY = "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77"
FINAL_ROWS = 6144
FINAL_KEY = "42520481f1fb73449f45c18604a72c32736a8eb3ba238b501f78a0575e4cf6ea"
PERFORMANCE_ROWS = 720
PERFORMANCE_CELLS = 24
CANDIDATE_ID = "Qwen3.5-9B-C-DGE-V4.1-27:mlp"
LOCK_STAGE = "qwen35_cdge_pareto_lock"
LOCK_MANIFEST_ID = "context-mismatch-qwen3-5-9b-cdge-v4-1-lock-v1"


def require(value: dict, expected: dict, label: str) -> None:
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ValueError(f"{label} mismatch: {field}")


def require_receipt(path: Path, *, run_prefix: str | None = None) -> dict:
    receipt = json.loads(path.read_text())
    run_id = receipt.get("run_id", "")
    job_id = receipt.get("job_id")
    archive_sha = receipt.get("archive_sha256", "")
    if run_prefix is not None and not run_id.startswith(run_prefix):
        raise ValueError("receipt run identity mismatch")
    if not isinstance(job_id, int) or job_id <= 0:
        raise ValueError("receipt job identity mismatch")
    if not isinstance(archive_sha, str) or len(archive_sha) != 64:
        raise ValueError("receipt archive SHA mismatch")
    require(
        receipt,
        {
            "cluster_shared_copy_verified": True,
            "host_data_copy_verified": True,
            "local_copy_verified": True,
        },
        "three-copy receipt",
    )
    terminal = receipt.get("slurm_terminal_record", "")
    for fragment in (f"JobId={job_id}", "JobState=COMPLETED", "ExitCode=0:0"):
        if fragment not in terminal:
            raise ValueError(f"receipt terminal evidence missing: {fragment}")
    return receipt


def require_fit(
    *,
    fit_report: Path,
    fit_receipt: Path,
    directional_checkpoint: Path,
    applicability_checkpoint: Path,
    composite_report: Path,
    evaluation_contract: Path,
) -> tuple[dict, dict, dict]:
    fit = json.loads(fit_report.read_text())
    composite = json.loads(composite_report.read_text())
    contract = json.loads(evaluation_contract.read_text())
    receipt = require_receipt(fit_receipt, run_prefix="qwen3-5-9b-cdge-fit-")
    require(
        fit,
        {
            "stage": "qwen35_cdge_fit",
            "method": METHOD,
            "fit_complete": True,
            "all_fit_gates_pass": True,
            "directional_fit_eligible": True,
            "operator_dev_accessed": False,
            "protected_behavior_outputs_accessed": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "Qwen3.5 fit report",
    )
    require(
        composite,
        {
            "method": METHOD,
            "audit_complete": True,
            "candidate_eligible": True,
            "candidate_may_be_locked": False,
            "operator_dev_accessed": False,
            "protected_behavior_outputs_accessed": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "Qwen3.5 composite report",
    )
    if not composite.get("gate_checks") or not all(composite["gate_checks"].values()):
        raise ValueError("Qwen3.5 composite gate checks did not all pass")
    if fit["directional_checkpoint_sha256"] != sha256_file(directional_checkpoint):
        raise ValueError("directional checkpoint SHA mismatch")
    if fit["applicability"]["checkpoint_sha256"] != sha256_file(applicability_checkpoint):
        raise ValueError("applicability checkpoint SHA mismatch")
    lineage = contract.get("bound_lineage", {})
    expected = {
        "fit_archive_sha256": receipt["archive_sha256"],
        "fit_receipt_sha256": sha256_file(fit_receipt),
        "fit_report_sha256": sha256_file(fit_report),
        "directional_checkpoint_sha256": sha256_file(directional_checkpoint),
        "applicability_checkpoint_sha256": sha256_file(applicability_checkpoint),
        "composite_report_sha256": sha256_file(composite_report),
    }
    require(lineage, expected, "evaluation lineage")
    require(
        contract,
        {
            "method": METHOD,
            "status": "frozen_before_operator_dev_behavior",
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "evaluation contract",
    )
    return fit, composite, contract


def require_authorization(path: Path, expected: dict) -> dict:
    authorization = json.loads(path.read_text())
    require(
        authorization,
        {
            **expected,
            "method": METHOD,
            "execution_allowed": True,
            "production_rollout_approved": False,
        },
        "Qwen3.5 evaluation authorization",
    )
    code_root = Path(authorization["code_root"])
    if code_root != WORK_ROOT / "code-v88":
        raise ValueError("Qwen3.5 evaluation requires immutable code-v88")
    if authorization.get("immutable_code_bundle_manifest_sha256") != sha256_file(
        code_root / "bundle.sha256"
    ):
        raise ValueError("code-v88 bundle binding mismatch")
    return authorization


def require_pareto_lock(path: Path, *, expected_sources: dict[str, Path] | None = None) -> dict:
    lock = json.loads(path.read_text())
    require(
        lock,
        {
            "manifest_id": LOCK_MANIFEST_ID,
            "stage": LOCK_STAGE,
            "method": METHOD,
            "model": "Qwen3.5-9B",
            "candidate_id": CANDIDATE_ID,
            "locked": True,
            "admissible": True,
            "candidate_may_be_locked": True,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "Qwen3.5 Pareto lock",
    )
    gate_checks = lock.get("gate_checks", {})
    if set(gate_checks) != {"behavior", "controls", "performance", "receipts"}:
        raise ValueError("Qwen3.5 Pareto lock gate families mismatch")
    if not all(
        isinstance(group, dict) and group and all(value is True for value in group.values())
        for group in gate_checks.values()
    ):
        raise ValueError("Qwen3.5 Pareto lock contains a failed gate")
    if expected_sources is not None:
        sources = lock.get("sources", {})
        for name, source_path in expected_sources.items():
            source = sources.get(name, {})
            if source.get("sha256") != sha256_file(source_path):
                raise ValueError(f"Qwen3.5 Pareto lock source mismatch: {name}")
    return lock
