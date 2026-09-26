#!/usr/bin/env python3
"""Materialize one immutable SHA-bound code-v30 controls authorization."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v1.controls import load_controls
from scripts.benchmark_v3.governance_control_design import (
    governance_control_groups,
    governance_control_key_hash,
    operator_selection_items,
)


PROJECT_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
CODE_ROOT = PROJECT_ROOT / "code-v30"


def _source(path: Path) -> dict:
    resolved = path.resolve()
    if not str(resolved).startswith(str(PROJECT_ROOT) + "/"):
        raise ValueError(f"source is outside the project root: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--fit-audit-analysis", type=Path, required=True)
    parser.add_argument("--operator-dev-analysis", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.parent.resolve() != (PROJECT_ROOT / "authorizations").resolve():
        raise ValueError("authorization output must be in the project authorization directory")
    if args.output.suffix != ".json":
        raise ValueError("authorization output must be JSON")

    editor = CODE_ROOT / "protocol/QWEN3_8B_DIRECTIONAL_STRUCTURAL_GOVERNANCE_EDITOR_V3.json"
    crossover_path = CODE_ROOT / "protocol/GOVERNANCE_TASK_CROSSOVER_V1.json"
    controls_path = CODE_ROOT / "protocol/mitigation_controls.json"
    manifest_path = CODE_ROOT / "artifacts/benchmark_manifest.jsonl"
    bundle_path = CODE_ROOT / "bundle.sha256"
    for required in (editor, crossover_path, controls_path, manifest_path, bundle_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    checkpoint = _source(args.checkpoint)
    fit_report = _source(args.fit_report)
    fit_audit = _source(args.fit_audit_analysis)
    operator_dev = _source(args.operator_dev_analysis)
    model_manifest = _source(args.model_manifest)
    fit_value = json.loads(args.fit_report.read_text())
    candidates = [
        row
        for row in fit_value.get("candidate_reports", [])
        if row.get("checkpoint_sha256") == checkpoint["sha256"]
    ]
    if len(candidates) != 1 or candidates[0].get("fit_eligible") is not True:
        raise ValueError("checkpoint is not the unique fit-eligible candidate in the fit report")
    fit_audit_value = json.loads(args.fit_audit_analysis.read_text())
    operator_value = json.loads(args.operator_dev_analysis.read_text())
    for value, stage in (
        (fit_audit_value, "governance_directional_fit_audit"),
        (operator_value, "governance_directional_operator_dev"),
    ):
        if value.get("method") != "DSGE-V3" or value.get("evaluation_stage") != stage:
            raise ValueError(f"invalid prior stage: {stage}")
        if value.get("checkpoint_sha256") != checkpoint["sha256"]:
            raise ValueError(f"prior checkpoint mismatch: {stage}")
        if value.get("fit_report_sha256") != fit_report["sha256"]:
            raise ValueError(f"prior fit-report mismatch: {stage}")
        if value.get("audit", {}).get("success") is not True:
            raise ValueError(f"prior audit incomplete: {stage}")
        if value.get("selection_gate", {}).get("candidate_eligible") is not True:
            raise ValueError(f"prior candidate gate failed: {stage}")
        if value.get("final_test_open") is not False or value.get("final_test_open_count") != 0:
            raise ValueError(f"prior final-test boundary changed: {stage}")
        if value.get("production_rollout_approved") is not False:
            raise ValueError(f"prior production boundary changed: {stage}")

    crossover = json.loads(crossover_path.read_text())
    items = operator_selection_items(load_jsonl(manifest_path))
    groups = governance_control_groups(
        items,
        load_controls(controls_path),
        list(crossover["factorial"]["declared_roles"]),
        list(crossover["factorial"]["history_styles"]),
    )
    expected_key_sha = governance_control_key_hash(groups)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-8b-governance_directional_controls-code-v30-{checkpoint['sha256'][:16]}",
        "created_utc": now.isoformat().replace("+00:00", "Z"),
        "stage": "governance_directional_controls",
        "code_root": str(CODE_ROOT),
        "immutable_code_bundle_manifest_sha256": sha256_file(bundle_path),
        "execution_allowed": True,
        "checkpoint": checkpoint,
        "checkpoint_sha256": checkpoint["sha256"],
        "fit_report": fit_report,
        "fit_report_sha256": fit_report["sha256"],
        "fit_audit_analysis": fit_audit,
        "fit_audit_analysis_sha256": fit_audit["sha256"],
        "operator_dev_analysis": operator_dev,
        "operator_dev_analysis_sha256": operator_dev["sha256"],
        "model_manifest": model_manifest,
        "model_manifest_sha256": model_manifest["sha256"],
        "editor_contract_sha256": sha256_file(editor),
        "crossover_contract_sha256": sha256_file(crossover_path),
        "controls_sha256": sha256_file(controls_path),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "expected_rows": 2856,
        "expected_key_sha256": expected_key_sha,
        "forced_directions": ["positive", "negative"],
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "authorization": str(args.output),
                "authorization_sha256": sha256_file(args.output),
                "expected_rows": 2856,
                "expected_key_sha256": expected_key_sha,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
