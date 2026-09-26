#!/usr/bin/env python3
"""Recompute the final paper's primary C-DGE values from released model outputs.

This is an offline analysis of real frozen runs; it does not perform inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_v2.analyze_crossover import build_interactions, summarize
from scripts.benchmark_v2.analyze_operator_candidate import _view
from scripts.benchmark_v10.cdge_contract import (
    CONTROLS_EXPECTED_KEY_SHA256, CONTROLS_EXPECTED_ROWS,
    FINAL_EXPECTED_KEY_SHA256, FINAL_EXPECTED_ROWS,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAIR_FIELDS = (
    "benchmark", "item_id", "partition", "declared_role", "history_style",
    "history_depth", "history_realization", "label_swap", "task_requirement",
)


def load_rows(name: str, key: str, expected_rows: int, expected_sha: str) -> list[dict]:
    rows = [json.loads(line) for line in (RESULTS / name).read_text().splitlines() if line.strip()]
    keys = [row[key] for row in rows]
    digest = hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest()
    if len(rows) != expected_rows or len(set(keys)) != expected_rows or digest != expected_sha:
        raise ValueError(f"frozen row identity mismatch: {name}")
    return rows


def reproduce() -> dict:
    rows = load_rows("cdge_v4_1_final_test.jsonl", "job_key", FINAL_EXPECTED_ROWS, FINAL_EXPECTED_KEY_SHA256)
    before, problems_before = build_interactions(_view(rows, "baseline"))
    after, problems_after = build_interactions(_view(rows, "edited"))
    if problems_before or problems_after or len(before) != FINAL_EXPECTED_ROWS // 4:
        raise ValueError("final test does not form complete paired four-cell interactions")
    reductions = []
    for raw, edited in zip(before, after):
        gap = raw["matched_minus_mismatched"]
        reductions.append({"benchmark": raw["benchmark"], "value": (gap - edited["matched_minus_mismatched"]) / max(abs(gap), 1.0)})
    ngr = summarize(reductions, "value")["equal_weight_benchmark_mean"]
    pairs = defaultdict(dict)
    matched_changed = 0
    for row in rows:
        for field in ("baseline", "edited"):
            if not all(math.isfinite(float(v)) for v in row[field].values()):
                raise ValueError("nonfinite final-test output")
        matched = (row["history_condition"] == "obedience") == (row["task_requirement"] == "delegated_choice")
        pairs[tuple(row[field] for field in PAIR_FIELDS)]["matched" if matched else "mismatched"] = row
        matched_changed += bool(matched and row["baseline"] != row["edited"])
    raw_correct = edited_correct = flips = rescued = harmed = 0
    for pair in pairs.values():
        matched, mismatched = pair["matched"], pair["mismatched"]
        mc = matched["baseline"]["task_aligned_margin"] > 0
        rc = mismatched["baseline"]["task_aligned_margin"] > 0
        ec = mismatched["edited"]["task_aligned_margin"] > 0
        raw_correct += rc
        edited_correct += ec
        flips += mc and not rc
        rescued += mc and not rc and ec
        harmed += rc and not ec
    controls = load_rows("cdge_v4_1_protected_controls.jsonl", "case_key", CONTROLS_EXPECTED_ROWS, CONTROLS_EXPECTED_KEY_SHA256)
    families = defaultdict(lambda: {"rows": 0, "changed": 0})
    for row in controls:
        changed = row["baseline"] != row["application_gated"]
        if changed != (row["application_gated_selected_logit_error"] != 0):
            raise ValueError("control identity and selected-logit error disagree")
        families[row["control_family"]]["rows"] += 1
        families[row["control_family"]]["changed"] += changed
    n = len(pairs)
    summary = {
        "method": "C-DGE-V4.1", "full_rows": len(rows), "mismatched_rows": n,
        "baseline_correct": raw_correct, "edited_correct": edited_correct,
        "baseline_accuracy": raw_correct / n, "edited_accuracy": edited_correct / n,
        "accuracy_gain_pp": 100 * (edited_correct - raw_correct) / n,
        "normalized_gap_reduction": ngr, "mismatch_induced_flips": flips,
        "rescued_flips": rescued, "raw_correct_harmed": harmed,
        "matched_changed_rows": matched_changed,
        "protected_rows": len(controls),
        "protected_changed_rows": sum(v["changed"] for v in families.values()),
        "protected_families": dict(sorted(families.items())),
    }
    expected = json.loads((RESULTS / "cdge_final_boundary_rescue.json").read_text())["overall"]
    for ours, theirs in {"baseline_accuracy": "raw_mismatch_accuracy", "edited_accuracy": "edited_mismatch_accuracy", "accuracy_gain_pp": "accuracy_gain_pp", "rescued_flips": "rescued_flips", "mismatch_induced_flips": "mismatch_induced_flips"}.items():
        if not math.isclose(summary[ours], expected[theirs], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"final row recomputation disagrees with boundary report: {ours}")
    final = json.loads((RESULTS / "cdge_v4_1_final_test.analysis.json").read_text())
    if not math.isclose(ngr, final["normalized_gap_reduction"]["equal_weight_benchmark_mean"], rel_tol=1e-12):
        raise ValueError("row recomputation disagrees with final NGR report")
    if matched_changed or summary["protected_changed_rows"] or len(families) != 6:
        raise ValueError("formal matched/control identity failed")
    return summary


def main() -> None:
    result = reproduce()
    target = ROOT / "generated/final_paper_metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
