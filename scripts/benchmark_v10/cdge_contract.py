"""Pure C-DGE V4.1 constants and terminal composite-evidence validation."""

import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file

METHOD = "C-DGE-V4.1"
RUNTIME_METHOD = "ADSGE-V4"
STAGE = "governance_composite_behavior_selection"
EXPECTED_ROWS = 3072
EXPECTED_KEY_SHA256 = "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e"
ARCHIVE_SHA256 = "602ad58c15ae47b3b24cf4de18bc1ba1195edf694cce5656e26bb7a4aba4ccdc"
CONTROLS_STAGE = "governance_composite_protected_controls"
CONTROLS_EXPECTED_ROWS = 2856
CONTROLS_EXPECTED_KEY_SHA256 = "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77"
FINAL_STAGE = "governance_composite_final_test"
FINAL_EXPECTED_ROWS = 6144
FINAL_EXPECTED_KEY_SHA256 = "42520481f1fb73449f45c18604a72c32736a8eb3ba238b501f78a0575e4cf6ea"


def _require(value: dict, expected: dict, label: str) -> None:
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ValueError(f"{label} mismatch: {field}")


def validate_composite_prerequisite(report: dict, receipt: dict) -> None:
    _require(
        report,
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
        "composite report",
    )
    checks = report.get("gate_checks", {})
    if not checks or not all(value is True for value in checks.values()):
        raise ValueError("composite report does not pass every gate")
    _require(
        receipt,
        {
            "run_id": "qwen3-8b-cdge-v4-1-composite-eligibility-20260730T012711Z",
            "job_id": 9267,
            "archive_sha256": ARCHIVE_SHA256,
            "cluster_shared_copy_verified": True,
            "host_data_copy_verified": True,
            "local_copy_verified": True,
        },
        "composite receipt",
    )
    terminal = receipt.get("slurm_terminal_record", "")
    for fragment in ("JobId=9267", "JobState=COMPLETED", "ExitCode=0:0"):
        if fragment not in terminal:
            raise ValueError(f"composite receipt lacks terminal evidence: {fragment}")


def validate_behavior_prerequisite(report: dict, receipt: dict) -> None:
    """Require fresh audited operator-dev evidence before protected controls."""
    _require(
        report,
        {
            "method": METHOD,
            "stage": STAGE,
            "behavior_evaluation_complete": True,
            "original_v4_fit_eligible": False,
            "composite_eligible": True,
            "candidate_eligible": True,
            "candidate_may_be_locked": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "behavior report",
    )
    _require(
        report.get("audit", {}),
        {
            "success": True,
            "row_count": EXPECTED_ROWS,
            "unique_job_keys": EXPECTED_ROWS,
            "observed_key_sha256": EXPECTED_KEY_SHA256,
            "zero_gate_max_error": 0.0,
        },
        "behavior audit",
    )
    run_id = receipt.get("run_id", "")
    job_id = receipt.get("job_id")
    archive_sha = receipt.get("archive_sha256", "")
    if not (
        isinstance(run_id, str)
        and run_id.startswith("qwen3-8b-cdge-v4-1-behavior-")
        and isinstance(job_id, int)
        and job_id > 0
        and isinstance(archive_sha, str)
        and len(archive_sha) == 64
        and all(character in "0123456789abcdef" for character in archive_sha)
    ):
        raise ValueError("behavior receipt identity is invalid")
    _require(
        receipt,
        {
            "cluster_shared_copy_verified": True,
            "host_data_copy_verified": True,
            "local_copy_verified": True,
        },
        "behavior receipt",
    )
    terminal = receipt.get("slurm_terminal_record", "")
    for fragment in (f"JobId={job_id}", "JobState=COMPLETED", "ExitCode=0:0"):
        if fragment not in terminal:
            raise ValueError(f"behavior receipt lacks terminal evidence: {fragment}")


def formalize_control_row(row: dict) -> dict:
    """Return an explicitly relabeled fresh control row without changing numbers."""
    value = dict(row)
    value["runtime_checkpoint_method"] = value.get("method")
    value["method"] = METHOD
    value["candidate_id"] = "C-DGE-V4.1-27:mlp"
    value["composite_eligible"] = True
    return value


def promote_behavior_outputs(
    *,
    runtime_output: Path,
    runtime_environment: Path,
    runtime_identity: Path,
    output: Path,
    environment_output: Path,
    identity_output: Path,
    composite_contract: Path,
    evaluation_contract: Path,
    composite_report: Path,
    composite_authorization: Path,
    composite_receipt: Path,
) -> None:
    """Promote hidden ADSGE runtime files into immutable C-DGE formal outputs."""
    if any(path.exists() for path in (output, environment_output, identity_output)):
        raise FileExistsError("refusing existing formal C-DGE behavior output")
    environment = json.loads(runtime_environment.read_text())
    environment.update(
        {
            "stage": STAGE,
            "method": METHOD,
            "runtime_checkpoint_method": RUNTIME_METHOD,
            "original_v4_fit_eligible": False,
            "fit_gates_passed": False,
            "composite_eligible": True,
            "candidate_eligible": True,
            "candidate_may_be_locked": False,
            "composite_contract_sha256": sha256_file(composite_contract),
            "evaluation_contract_sha256": sha256_file(evaluation_contract),
            "composite_report_sha256": sha256_file(composite_report),
            "composite_authorization_sha256": sha256_file(composite_authorization),
            "composite_receipt_sha256": sha256_file(composite_receipt),
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
    )
    identity = json.loads(runtime_identity.read_text())
    identity.update(
        {
            "method": METHOD,
            "runtime_checkpoint_method": RUNTIME_METHOD,
            "composite_eligible": True,
            "candidate_eligible": True,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
    )
    promoted = []
    for row in load_jsonl(runtime_output):
        if row.get("method") != RUNTIME_METHOD:
            raise ValueError("unexpected raw runtime method")
        row["runtime_checkpoint_method"] = RUNTIME_METHOD
        row["method"] = METHOD
        row["composite_eligible"] = True
        promoted.append(canonical_json(row))
    atomic_write_text(environment_output, json.dumps(environment, indent=2, sort_keys=True) + "\n")
    atomic_write_text(identity_output, json.dumps(identity, indent=2, sort_keys=True) + "\n")
    atomic_write_text(output, "\n".join(promoted) + "\n")
    runtime_output.unlink()
    runtime_environment.unlink()
    runtime_identity.unlink()


def promote_final_outputs(
    *,
    runtime_output: Path,
    runtime_environment: Path,
    runtime_identity: Path,
    output: Path,
    environment_output: Path,
    identity_output: Path,
    pareto_lock: Path,
) -> None:
    """Promote hidden ADSGE runtime files into one-time C-DGE final outputs."""
    if any(path.exists() for path in (output, environment_output, identity_output)):
        raise FileExistsError("refusing existing formal C-DGE final output")
    environment = json.loads(runtime_environment.read_text())
    _require(
        environment,
        {
            "expected_rows": FINAL_EXPECTED_ROWS,
            "planned_rows": FINAL_EXPECTED_ROWS,
            "expected_key_sha256": FINAL_EXPECTED_KEY_SHA256,
            "final_test_open": True,
            "final_test_open_count": 1,
            "production_rollout_approved": False,
        },
        "runtime final environment",
    )
    environment.update(
        {
            "stage": FINAL_STAGE,
            "method": METHOD,
            "runtime_checkpoint_method": RUNTIME_METHOD,
            "candidate_id": "C-DGE-V4.1-27:mlp",
            "composite_eligible": True,
            "candidate_eligible": True,
            "candidate_may_be_locked": True,
            "pareto_lock_sha256": sha256_file(pareto_lock),
        }
    )
    identity = json.loads(runtime_identity.read_text())
    _require(
        identity,
        {
            "success": True,
            "max_error": 0.0,
            "final_test_open": True,
            "final_test_open_count": 1,
            "production_rollout_approved": False,
        },
        "runtime final identity",
    )
    identity.update(
        {
            "stage": FINAL_STAGE,
            "method": METHOD,
            "runtime_checkpoint_method": RUNTIME_METHOD,
            "candidate_id": "C-DGE-V4.1-27:mlp",
            "composite_eligible": True,
            "candidate_eligible": True,
            "candidate_may_be_locked": True,
            "pareto_lock_sha256": sha256_file(pareto_lock),
        }
    )
    promoted = []
    for row in load_jsonl(runtime_output):
        if row.get("method") != RUNTIME_METHOD:
            raise ValueError("unexpected raw runtime final method")
        if row.get("final_test_open") is not True:
            raise ValueError("runtime row does not belong to the final test")
        row["runtime_checkpoint_method"] = RUNTIME_METHOD
        row["method"] = METHOD
        row["candidate_id"] = "C-DGE-V4.1-27:mlp"
        row["composite_eligible"] = True
        row["candidate_eligible"] = True
        row["candidate_may_be_locked"] = True
        promoted.append(canonical_json(row))
    if len(promoted) != FINAL_EXPECTED_ROWS:
        raise ValueError("runtime final row count mismatch")
    atomic_write_text(environment_output, json.dumps(environment, indent=2, sort_keys=True) + "\n")
    atomic_write_text(identity_output, json.dumps(identity, indent=2, sort_keys=True) + "\n")
    atomic_write_text(output, "\n".join(promoted) + "\n")
    runtime_output.unlink()
    runtime_environment.unlink()
    runtime_identity.unlink()
