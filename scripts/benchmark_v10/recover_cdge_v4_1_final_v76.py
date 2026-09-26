#!/usr/bin/env python3
"""Recover complete C-DGE final rows after the v73 post-processing schema bug.

This command never runs a model.  It copies the immutable runtime outputs from
the failed producer, adds the final-test count that the legacy identity writer
omitted, and delegates promotion to the frozen C-DGE contract implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from scripts.benchmark_v10.cdge_contract import promote_final_outputs


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(value: dict, expected: dict, label: str) -> None:
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ValueError(f"{label} mismatch: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--recovery-authorization", type=Path, required=True)
    parser.add_argument("--pareto-lock", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    auth = json.loads(args.recovery_authorization.read_text())
    require(
        auth,
        {
            "stage": "governance_composite_final_postprocessing_recovery",
            "method": "C-DGE-V4.1",
            "execution_allowed": True,
            "model_forward_allowed": False,
            "source_job_id": "9324",
            "source_final_test_open": True,
            "source_final_test_open_count": 1,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "recovery authorization",
    )
    if Path(auth["source_run"]) != args.source_run:
        raise ValueError("recovery source path mismatch")
    if sha256_file(args.pareto_lock) != auth["pareto_lock_sha256"]:
        raise ValueError("recovery Pareto-lock mismatch")

    names = {
        "rows": ".cdge_v4_1_final_test.jsonl.runtime",
        "identity": ".cdge_v4_1_final_test.identity.json.runtime",
        "environment": ".cdge_v4_1_final_test.environment.json.runtime",
    }
    sources: dict[str, Path] = {}
    for key, name in names.items():
        source = args.source_run / name
        if not source.is_file() or sha256_file(source) != auth["source_artifacts"][key]["sha256"]:
            raise ValueError(f"recovery source artifact mismatch: {key}")
        sources[key] = source
    source_status = json.loads((args.source_run / "exit_status.json").read_text())
    require(
        source_status,
        {
            "job_id": "9324",
            "exit_code": 1,
            "stage": "governance_composite_final_test",
            "final_test_open": True,
            "final_test_open_count": 1,
            "production_rollout_approved": False,
        },
        "failed producer status",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime_rows = args.output_dir / names["rows"]
    runtime_environment = args.output_dir / names["environment"]
    runtime_identity = args.output_dir / names["identity"]
    for target in (runtime_rows, runtime_environment, runtime_identity):
        if target.exists():
            raise FileExistsError(f"refusing stale recovery target: {target}")
    shutil.copyfile(sources["rows"], runtime_rows)
    shutil.copyfile(sources["environment"], runtime_environment)
    identity = json.loads(sources["identity"].read_text())
    require(
        identity,
        {
            "success": True,
            "max_error": 0.0,
            "method": "ADSGE-V4",
            "final_test_open": True,
            "production_rollout_approved": False,
        },
        "legacy runtime identity",
    )
    if "final_test_open_count" in identity:
        raise ValueError("recovery applies only to the omitted identity count bug")
    identity["final_test_open_count"] = 1
    runtime_identity.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")

    output = args.output_dir / "cdge_v4_1_final_test.jsonl"
    environment = args.output_dir / "cdge_v4_1_final_test.environment.json"
    identity_output = args.output_dir / "cdge_v4_1_final_test.identity.json"
    promote_final_outputs(
        runtime_output=runtime_rows,
        runtime_environment=runtime_environment,
        runtime_identity=runtime_identity,
        output=output,
        environment_output=environment,
        identity_output=identity_output,
        pareto_lock=args.pareto_lock,
    )
    lineage = {
        "schema_version": 1,
        "stage": "governance_composite_final_postprocessing_recovery",
        "source_job_id": "9324",
        "source_run": str(args.source_run),
        "source_artifacts": auth["source_artifacts"],
        "recovery_authorization_sha256": sha256_file(args.recovery_authorization),
        "model_forward_reexecuted": False,
        "final_test_reopened": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    (args.output_dir / "recovery_lineage.json").write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
