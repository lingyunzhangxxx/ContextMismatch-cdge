#!/usr/bin/env python3
"""Reproduce the terminal-evidence diagnosis that motivated the V3 router."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


EXPECTED_ROWS = 3072


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matched(row: dict) -> bool:
    history_obedience = row["history_condition"] == "obedience"
    task_obedience = row["task_requirement"] == "delegated_choice"
    return history_obedience == task_obedience


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "minimum": min(values),
        "mean": statistics.fmean(values),
        "maximum": max(values),
    }


def audit(rows_path: Path, *, activation_threshold: float) -> dict:
    if not 0.0 < activation_threshold < 1.0:
        raise ValueError("activation threshold must lie strictly between zero and one")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    keys = [str(row["job_key"]) for row in rows]
    if len(rows) != EXPECTED_ROWS or len(set(keys)) != EXPECTED_ROWS:
        raise ValueError("V2 selection evidence is not exactly 3,072 unique rows")
    sites = sorted(rows[0]["controller_diagnostics"])
    if sites != ["23:self_attn", "27:mlp", "31:mlp"]:
        raise ValueError("unexpected V2 controller sites")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if set(row["controller_diagnostics"]) != set(sites):
            raise ValueError("inconsistent controller diagnostics")
        if not math.isfinite(float(row["delta_task_aligned_margin"])):
            raise FloatingPointError("non-finite behavior margin")
        grouped["matched" if _matched(row) else "mismatch"].append(row)
    if {name: len(values) for name, values in grouped.items()} != {
        "matched": 1536,
        "mismatch": 1536,
    }:
        raise ValueError("matched/mismatch cells are incomplete")

    site_reports = {}
    all_matched_new_active = []
    all_mismatch_new_active = []
    for site in sites:
        states = {}
        for state in ("matched", "mismatch"):
            old_abs = []
            proposed_max = []
            proposed_active = []
            for row in grouped[state]:
                values = row["controller_diagnostics"][site]
                history = float(values["history_probability"])
                task = float(values["task_probability"])
                old = abs(float(values["signed_governance_factor"]))
                if not all(math.isfinite(value) for value in (history, task, old)):
                    raise FloatingPointError("non-finite controller diagnostic")
                positive = history * (1.0 - task)
                negative = (1.0 - history) * task
                route = max(
                    max(0.0, positive - activation_threshold),
                    max(0.0, negative - activation_threshold),
                ) / (1.0 - activation_threshold)
                old_abs.append(old)
                proposed_max.append(route)
                proposed_active.append(route > 0.0)
            states[state] = {
                "old_absolute_factor": _summary(old_abs),
                "old_saturation_fraction_at_0_99": statistics.fmean(
                    value >= 0.99 for value in old_abs
                ),
                "proposed_max_route": _summary(proposed_max),
                "proposed_active_fraction": statistics.fmean(proposed_active),
            }
            if state == "matched":
                all_matched_new_active.extend(proposed_active)
            else:
                all_mismatch_new_active.extend(proposed_active)
        site_reports[site] = states

    return {
        "schema_version": 1,
        "audit": "AMSGE_V2_ROUTER_STRUCTURAL_FAILURE",
        "source_rows": str(rows_path),
        "source_rows_sha256": _sha256(rows_path),
        "rows": len(rows),
        "unique_job_keys": len(set(keys)),
        "sites": site_reports,
        "finding": {
            "raw_logit_subtraction_invalid": True,
            "old_router_matched_saturated_at_every_site": all(
                site_reports[site]["matched"]["old_saturation_fraction_at_0_99"] == 1.0
                for site in sites
            ),
            "old_router_mismatch_saturated_at_every_site": all(
                site_reports[site]["mismatch"]["old_saturation_fraction_at_0_99"] == 1.0
                for site in sites
            ),
            "proposed_activation_threshold": activation_threshold,
            "proposed_router_any_site_matched_active_fraction": statistics.fmean(
                all_matched_new_active
            ),
            "proposed_router_any_site_mismatch_active_fraction": statistics.fmean(
                all_mismatch_new_active
            ),
            "v2_router_falsified": True,
        },
        "safety": {
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--activation-threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(
        json.dumps(
            audit(args.input, activation_threshold=args.activation_threshold),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
