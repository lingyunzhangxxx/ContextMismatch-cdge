#!/usr/bin/env python3
"""Audit, aggregate, and statistically compare context-governance runs."""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "transcripts.jsonl"
OUT = ROOT / "results"


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(xs, seed=20260726, draws=10000):
    if not xs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    samples = sorted(mean([rng.choice(xs) for _ in xs]) for _ in range(draws))
    return samples[int(0.025 * draws)], samples[int(0.975 * draws)]


def paired_randomization(differences, seed=20260726, draws=100000):
    observed = abs(mean(differences))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(draws):
        permuted = mean([d if rng.random() < 0.5 else -d for d in differences])
        extreme += abs(permuted) >= observed
    return (extreme + 1) / (draws + 1)


def fmt(value):
    return "NA" if math.isnan(value) else f"{value:.2f}"


def main():
    rows = [json.loads(line) for line in INPUT.read_text().splitlines() if line.strip()]
    errors = [row for row in rows if "error" in row]
    valid = [row for row in rows if "error" not in row]
    if errors:
        (OUT / "errors.json").write_text(json.dumps(errors, indent=2))
        raise SystemExit(f"{len(errors)} inference errors; see results/errors.json")

    by_cell = defaultdict(list)
    for row in valid:
        by_cell[(row["role_declaration"], row["condition"])].append(row)

    summary = []
    for (role, condition), cell in sorted(by_cell.items()):
        exploit = [row["exploitability"] for row in cell]
        lo, hi = bootstrap_ci(exploit, seed=sum(map(ord, role + condition)))
        summary.append({
            "role": role,
            "condition": condition,
            "n": len(cell),
            "exploitability_mean": mean(exploit),
            "exploitability_ci_low": lo,
            "exploitability_ci_high": hi,
            "accuracy": mean([row["decision_correct"] for row in cell]),
            "independent_check": mean([row["check_performed"] for row in cell]),
            "challenge": mean([row["challenged_user"] for row in cell]),
            "candidate_preservation": mean([row["candidate_preserved"] for row in cell]),
            "parse_rate": mean([row["parse_ok"] for row in cell]),
        })

    with (OUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    indexed = {
        (row["role_declaration"], row["condition"], row["task_id"], row["replicate"]): row
        for row in valid
    }
    tests = []
    for role in ("collaborator", "assistant"):
        for left, right, label in (
            ("obey_full", "verify_full", "governance effect"),
            ("obey_full", "fresh", "obedience vs fresh"),
            ("obey_full", "obey_reset", "reset recovery"),
            ("verify_summary", "verify_full", "verify compression"),
            ("obey_summary", "obey_full", "obey compression"),
        ):
            diffs = []
            for task in sorted({row["task_id"] for row in valid}):
                reps = sorted({row["replicate"] for row in valid})
                for rep in reps:
                    a = indexed[(role, left, task, rep)]["exploitability"]
                    b = indexed[(role, right, task, rep)]["exploitability"]
                    diffs.append(a - b)
            lo, hi = bootstrap_ci(diffs, seed=sum(map(ord, role + label)))
            tests.append({
                "role": role,
                "contrast": label,
                "left": left,
                "right": right,
                "n_pairs": len(diffs),
                "mean_difference": mean(diffs),
                "ci_low": lo,
                "ci_high": hi,
                "randomization_p": paired_randomization(diffs, seed=sum(map(ord, label))),
            })

    with (OUT / "contrasts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tests[0].keys())
        writer.writeheader()
        writer.writerows(tests)

    table_lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Role & History & Exploitability $\downarrow$ & Accuracy $\uparrow$ & Verify $\uparrow$ & Challenge $\uparrow$ \\",
        r"\midrule",
    ]
    labels = {
        "fresh": "Fresh",
        "verify_full": "Verification-full",
        "obey_full": "Obedience-full",
        "verify_summary": "Verification-summary",
        "obey_summary": "Obedience-summary",
        "obey_reset": "Obedience+reset",
    }
    for row in summary:
        table_lines.append(
            f"{row['role'].title()} & {labels[row['condition']]} & "
            f"{row['exploitability_mean']:.2f} [{row['exploitability_ci_low']:.2f}, {row['exploitability_ci_high']:.2f}] & "
            f"{row['accuracy']:.2f} & {row['independent_check']:.2f} & {row['challenge']:.2f} \\\\"
        )
    table_lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "main_table.tex").write_text("\n".join(table_lines) + "\n")

    audit = {
        "rows": len(valid),
        "errors": len(errors),
        "parse_rate": mean([row["parse_ok"] for row in valid]),
        "tasks": sorted({row["task_id"] for row in valid}),
        "roles": sorted({row["role_declaration"] for row in valid}),
        "conditions": sorted({row["condition"] for row in valid}),
        "models": sorted({row["model"] for row in valid}),
        "usage_total": {
            key: sum((row.get("usage") or {}).get(key, 0) for row in valid)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print("\nContrasts:")
    for test in tests:
        print(test)


if __name__ == "__main__":
    main()
