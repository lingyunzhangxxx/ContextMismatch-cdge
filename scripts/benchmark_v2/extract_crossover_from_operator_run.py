#!/usr/bin/env python3
"""Extract the unedited crossover view from a locked-operator final-test run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--operator-environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.environment_output.exists():
        raise FileExistsError("refusing existing crossover extraction output")
    operator_environment = json.loads(args.operator_environment.read_text())
    if operator_environment.get("stage") != "operator_final_test":
        raise ValueError("crossover extraction is restricted to the one-time final-test run")
    rows = load_jsonl(args.input)
    output_rows = []
    for row in rows:
        baseline = row["baseline"]
        answer = "A" if float(baseline["logit_a"]) >= float(baseline["logit_b"]) else "B"
        output_rows.append(
            {
                "job_key": row["job_key"],
                "stage": "final_test",
                "benchmark": row["benchmark"],
                "item_id": row["item_id"],
                "partition": row["partition"],
                "declared_role": row["declared_role"],
                "history_condition": row["history_condition"],
                "history_style": row["history_style"],
                "history_depth": row["history_depth"],
                "history_realization": row["history_realization"],
                "task_requirement": row["task_requirement"],
                "target_obedience": row["target_obedience"],
                "label_swap": row["label_swap"],
                "task_correct_label": row["task_correct_label"],
                "factual_correct_label": row["factual_correct_label"],
                "user_selected_label": row["user_selected_label"],
                "logit_a": baseline["logit_a"],
                "logit_b": baseline["logit_b"],
                "task_aligned_margin": baseline["task_aligned_margin"],
                "factual_margin": baseline["factual_margin"],
                "user_choice_margin": baseline["user_choice_margin"],
                "answer": answer,
                "task_aligned_correct": answer == row["task_correct_label"],
                "factual_correct": answer == row["factual_correct_label"],
                "followed_user_selection": answer == row["user_selected_label"],
                "candidate_payload_sha256": row["candidate_payload_sha256"],
                "suffix_token_sha256_int32_le": row["suffix_token_sha256_int32_le"],
                "source_operator_final_test_sha256": sha256_file(args.input),
            }
        )
    atomic_write_text(
        args.output,
        "".join(canonical_json(row) + "\n" for row in output_rows),
    )
    environment = {
        "schema_version": 1,
        "stage": "final_test",
        "source_operator_environment_sha256": sha256_file(args.operator_environment),
        "source_operator_output_sha256": sha256_file(args.input),
        "expected_key_sha256": operator_environment["expected_key_sha256"],
        "planned_rows": operator_environment["planned_rows"],
        "final_test_open": True,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.environment_output,
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "rows": len(output_rows),
                "output_sha256": sha256_file(args.output),
                "final_test_open": True,
                "production_rollout_approved": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
