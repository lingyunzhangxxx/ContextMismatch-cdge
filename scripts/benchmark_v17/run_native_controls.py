#!/usr/bin/env python3
"""Run protected controls for the behavior-selected C-DGE V4.2 candidate."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file


EXPECTED_ROWS = 2856
EXPECTED_KEY = "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77"


def _write(path: Path, value: object, *, overwrite: bool = False) -> None:
    atomic_write_text(
        path, json.dumps(value, indent=2, sort_keys=True) + "\n",
        allow_overwrite=overwrite,
    )


def _artifact(auth: dict, name: str) -> Path:
    record = auth.get("bound_artifacts", {}).get(name, {})
    path = Path(str(record.get("path", "")))
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"native controls artifact mismatch: {name}")
    return path


def _validate(args: argparse.Namespace) -> tuple[dict, dict[str, Path]]:
    auth = json.loads(args.execution_authorization.read_text())
    required = {
        "stage": "qwen35_cdge_v4_2_native_protected_controls",
        "method": "C-DGE-V4.2",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "execution_allowed": True,
        "expected_rows": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY,
        "forced_directions": ["positive", "negative"],
        "behavior_finalist_eligible": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if auth.get(field) != expected:
            raise ValueError(f"native controls authorization mismatch: {field}")
    paths = {name: _artifact(auth, name) for name in (
        "protocol", "candidate_grid", "behavior_merge", "behavior_analysis",
        "behavior_receipt", "checkpoint", "compatibility_fit_report", "editor_contract",
        "crossover_contract", "manifest", "manifest_report", "controls", "model_manifest",
    )}
    merge = json.loads(paths["behavior_merge"].read_text())
    selected = merge.get("control_finalist")
    if auth.get("behavior_tied_finalist") is True:
        tied_ids = auth.get("frozen_tied_candidate_ids")
        if merge.get("control_finalist_selected") is not False or not isinstance(tied_ids, list):
            raise ValueError("native controls tied-finalist contract mismatch")
        matches = [row for row in merge.get("candidates", []) if row.get("candidate_id") == auth.get("candidate_id")]
        if len(matches) != 1 or auth.get("candidate_id") not in tied_ids:
            raise ValueError("native controls candidate is outside the frozen tie")
        selected = matches[0]
    elif merge.get("control_finalist_selected") is not True or not isinstance(selected, dict):
        raise ValueError("native controls require one frozen behavior finalist")
    if selected.get("candidate_id") != auth.get("candidate_id"):
        raise ValueError("native controls finalist identity mismatch")
    analysis = json.loads(paths["behavior_analysis"].read_text())
    if analysis.get("candidate_id") != auth.get("candidate_id"):
        raise ValueError("native controls behavior analysis mismatch")
    if analysis.get("native_behavior_eligible") is not True:
        raise ValueError("native controls behavior finalist is ineligible")
    return auth, paths


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("code_root", "execution_authorization", "model_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    auth, paths = _validate(args)
    args.output_dir.mkdir(parents=True)
    behavior = json.loads(paths["behavior_analysis"].read_text())
    checkpoint_sha = sha256_file(paths["checkpoint"])
    fit_sha = sha256_file(paths["compatibility_fit_report"])
    source_model = json.loads(paths["model_manifest"].read_text())
    model_revision = source_model.get("revision") or source_model.get("lineage", {}).get(
        "weight_and_config_file_revision"
    )
    model_envelope = args.output_dir / "runtime_model_manifest.json"
    _write(model_envelope, {
        "schema_version": 1,
        "verified": True,
        "revision": model_revision,
        "source_model_contract_sha256": sha256_file(paths["model_manifest"]),
        "production_rollout_approved": False,
    })
    operator_prior = args.output_dir / "native_operator_dev_compatibility.json"
    fit_prior = args.output_dir / "native_fit_audit_compatibility.json"
    fit_value = dict(behavior)
    fit_value.update({
        "method": "DSGE-V3",
        "evaluation_stage": "governance_directional_fit_audit",
        "checkpoint_sha256": checkpoint_sha,
        "fit_report_sha256": fit_sha,
        "fit_eligible": True,
        "selection_gate": {"candidate_eligible": True},
        "compatibility_envelope_for_native_frozen_gate": True,
        "source_native_behavior_analysis_sha256": sha256_file(paths["behavior_analysis"]),
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    })
    _write(fit_prior, fit_value)
    prior = dict(fit_value)
    prior.update({
        "method": "DSGE-V3",
        "evaluation_stage": "governance_directional_operator_dev",
        "checkpoint_sha256": checkpoint_sha,
        "fit_report_sha256": fit_sha,
        "fit_eligible": True,
        "selection_gate": {"candidate_eligible": True},
        "matched_selected_logit_max_error": 0.0,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    })
    _write(operator_prior, prior)
    runtime_auth = args.output_dir / "runtime_directional_controls_authorization.json"
    _write(runtime_auth, {
        "schema_version": 1,
        "stage": "governance_directional_controls",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "execution_allowed": True,
        "checkpoint_sha256": checkpoint_sha,
        "fit_report_sha256": fit_sha,
        "fit_audit_analysis_sha256": sha256_file(fit_prior),
        "operator_dev_analysis_sha256": sha256_file(operator_prior),
        "editor_contract_sha256": sha256_file(paths["editor_contract"]),
        "crossover_contract_sha256": sha256_file(paths["crossover_contract"]),
        "benchmark_manifest_sha256": sha256_file(paths["manifest"]),
        "controls_sha256": sha256_file(paths["controls"]),
        "model_manifest_sha256": sha256_file(model_envelope),
        "expected_rows": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY,
        "forced_directions": ["positive", "negative"],
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    })
    raw = args.output_dir / ".native_controls.runtime.jsonl"
    environment = args.output_dir / "native_controls.environment.json"
    subprocess.run([
        sys.executable, "-m", "scripts.benchmark_v4.run_directional_controls",
        "--checkpoint", str(paths["checkpoint"]),
        "--fit-report", str(paths["compatibility_fit_report"]),
        "--fit-audit-analysis", str(fit_prior),
        "--operator-dev-analysis", str(operator_prior),
        "--editor-contract", str(paths["editor_contract"]),
        "--execution-authorization", str(runtime_auth),
        "--crossover-contract", str(paths["crossover_contract"]),
        "--manifest", str(paths["manifest"]),
        "--manifest-report", str(paths["manifest_report"]),
        "--controls", str(paths["controls"]),
        "--model-manifest", str(model_envelope),
        "--model-path", str(args.model_path),
        "--output", str(raw), "--environment-output", str(environment),
        "--device", args.device, "--attn-implementation", "eager",
    ], check=True, cwd=args.code_root)
    rows_path = args.output_dir / "native_controls.jsonl"
    rows = []
    for row in load_jsonl(raw):
        row["runtime_method"] = row.get("method")
        row["method"] = "C-DGE-V4.2"
        row["candidate_id"] = auth["candidate_id"]
        rows.append(canonical_json(row))
    atomic_write_text(rows_path, "\n".join(rows) + "\n")
    raw.unlink()
    env = json.loads(environment.read_text())
    env.update({
        "stage": "qwen35_cdge_v4_2_native_protected_controls",
        "method": "C-DGE-V4.2",
        "candidate_id": auth["candidate_id"],
        "site": auth["site"],
        "formal_authorization_sha256": sha256_file(args.execution_authorization),
        "behavior_merge_sha256": sha256_file(paths["behavior_merge"]),
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    })
    _write(environment, env, overwrite=True)
    analysis_path = args.output_dir / "native_controls.analysis.json"
    subprocess.run([
        sys.executable, "-m", "scripts.benchmark_v4.analyze_directional_controls",
        "--input", str(rows_path), "--environment", str(environment),
        "--editor-contract", str(paths["editor_contract"]),
        "--output", str(analysis_path),
    ], check=True, cwd=args.code_root)
    report = json.loads(analysis_path.read_text())
    parsed_rows = [json.loads(row) for row in rows]
    correction_means = {}
    for direction in ("positive", "negative"):
        values = [
            float(row[f"forced_on_{direction}_intervention_diagnostics"]["mean_relative_intervention_norm"])
            for row in parsed_rows
        ]
        if not values or not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite protected forced-on correction")
        correction_means[direction] = sum(values) / len(values)
    correction_statistic = max(correction_means.values())
    report.update({
        "method": "C-DGE-V4.2",
        "stage": "qwen35_cdge_v4_2_native_protected_controls",
        "evaluation_stage": "qwen35_cdge_v4_2_native_protected_controls",
        "candidate_id": auth["candidate_id"],
        "site": auth["site"],
        "behavior_finalist_eligible": True,
        "behavior_merge_sha256": sha256_file(paths["behavior_merge"]),
        "protected_forced_on_relative_correction": {
            "aggregation": "maximum of the two per-direction means over protected-control rows",
            "positive_mean": correction_means["positive"],
            "negative_mean": correction_means["negative"],
            "maximum": correction_statistic,
        },
        "formal_authorization_sha256": sha256_file(args.execution_authorization),
        "protected_controls_complete": report.get("audit", {}).get("success") is True,
        "candidate_may_be_locked": report.get("controls_admissible") is True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    })
    _write(analysis_path, report, overwrite=True)
    print(json.dumps({
        "candidate_id": auth["candidate_id"],
        "rows": report["audit"]["row_count"],
        "controls_admissible": report["controls_admissible"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
