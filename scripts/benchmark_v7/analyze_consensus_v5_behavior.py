#!/usr/bin/env python3
"""Audit downstream V5 behavior against full DGE on an identical job set."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v7.merge_same_identity import (
    _improvements,
    _paired_full_dge_comparison,
)
from scripts.benchmark_v7.run_same_identity import EXPECTED_KEY_SHA256, EXPECTED_ROWS


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _active(row: dict) -> bool:
    diagnostics = row.get("controller_diagnostics", {})
    if set(diagnostics) != {"27:mlp"}:
        raise ValueError("V5 row must contain exactly the selected 27:mlp diagnostics")
    value = diagnostics["27:mlp"].get("v5_gate_active")
    if value not in (0.0, 1.0):
        raise ValueError("V5 gate diagnostic must be binary")
    return bool(value)


def _router_summary(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        mismatch = not bool(row["matched_context"])
        direction = (
            "obedience_to_verification"
            if row["history_condition"] == "obedience"
            else "verification_to_delegated"
        )
        groups["all"].append(row)
        groups["mismatched" if mismatch else "matched"].append(row)
        groups[f"direction:{direction}"].append(row)
        groups[f"label_swap:{int(row['label_swap'])}"].append(row)
        groups[f"benchmark:{row['benchmark']}"] .append(row)
    result = {}
    for key, values in sorted(groups.items()):
        active = [_active(row) for row in values]
        changed = [float(row["selected_logit_max_error"]) > 0.0 for row in values]
        result[key] = {
            "rows": len(values),
            "active_fraction": _mean([float(value) for value in active]),
            "behavior_changed_fraction": _mean([float(value) for value in changed]),
            "active_but_logit_identity_rows": sum(
                1 for is_active, did_change in zip(active, changed, strict=True)
                if is_active and not did_change
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing V5 behavior report: {args.output}")
    shard_audit_path = args.output_dir / "shard_audit.json"
    systems_path = args.output_dir / "systems_metrics.json"
    shard_audit = json.loads(shard_audit_path.read_text())
    if shard_audit.get("stage") != "governance_v5_same_identity_behavior":
        raise ValueError("unexpected V5 behavior stage")
    if shard_audit.get("method_shard") != "v5" or shard_audit.get("methods") != [
        "full_dge",
        "consensus_v5",
    ]:
        raise ValueError("V5 behavior method pair changed")
    if shard_audit.get("success") is not True:
        raise ValueError("V5 behavior shard audit failed")
    if shard_audit.get("paired_baseline_logits_identical") is not True:
        raise ValueError("V5/full-DGE baselines differ")

    rows = {
        method: load_jsonl(args.output_dir / "rows" / f"{method}.jsonl")
        for method in ("full_dge", "consensus_v5")
    }
    for method, values in rows.items():
        if len(values) != EXPECTED_ROWS or len({row["job_key"] for row in values}) != EXPECTED_ROWS:
            raise ValueError(f"{method}: incomplete V5 behavior rows")
        analysis = json.loads(
            (args.output_dir / "analysis" / f"{method}.json").read_text()
        )
        audit = analysis.get("audit", {})
        if audit.get("success") is not True:
            raise ValueError(f"{method}: behavior analysis audit failed")
        if audit.get("observed_key_sha256") != EXPECTED_KEY_SHA256:
            raise ValueError(f"{method}: behavior key SHA mismatch")

    full_improvements = _improvements(rows["full_dge"])
    v5_improvements = _improvements(rows["consensus_v5"])
    paired = _paired_full_dge_comparison(
        full_improvements,
        v5_improvements,
        args.bootstrap_replicates,
        51001,
    )
    router = _router_summary(rows["consensus_v5"])
    systems = json.loads(systems_path.read_text())
    value = {
        "schema_version": 1,
        "stage": "governance_v5_same_identity_behavior_analysis",
        "rows_per_method": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "shard_audit_sha256": sha256_file(shard_audit_path),
        "systems_metrics_sha256": sha256_file(systems_path),
        "full_dge_analysis_sha256": sha256_file(
            args.output_dir / "analysis/full_dge.json"
        ),
        "consensus_v5_analysis_sha256": sha256_file(
            args.output_dir / "analysis/consensus_v5.json"
        ),
        "router": router,
        "full_dge_minus_consensus_v5": paired,
        "systems": systems,
        "behavioral_interpretation": {
            "positive_full_dge_minus_consensus_v5_favors_full_dge_efficacy": True,
            "matched_active_fraction_is_false_trigger_rate": True,
            "mismatched_active_fraction_is_router_behavioral_coverage": True,
        },
        "success": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
