"""Pure-Python frozen row design for six-family AMSGE controls."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from scripts.benchmark_v1.controls import (
    factual_memory_messages,
    factual_memory_prompt,
    supported_authority_prompt,
)
from scripts.benchmark_v1.histories import prefix_messages, system_message
from scripts.benchmark_v2.crossover import task_prompt


def operator_selection_items(manifest: list[dict]) -> list[dict]:
    selected = []
    for benchmark in sorted({row["benchmark"] for row in manifest}):
        rows = sorted(
            (
                row
                for row in manifest
                if row["benchmark"] == benchmark and row["partition"] == "operator_dev"
            ),
            key=lambda row: row["item_id"],
        )
        if len(rows) != 32:
            raise ValueError(f"operator_dev:{benchmark} has {len(rows)} items")
        selected.extend(rows[16:])
    if len(selected) != 96:
        raise ValueError("operator selection-half controls require 96 benchmark items")
    return selected


def governance_control_case_key(case: dict) -> str:
    metadata = case["metadata"]
    return "__".join(
        str(value)
        for value in (
            metadata["control_family"],
            metadata.get("benchmark", "none"),
            metadata["case_id"],
            metadata["declared_role"],
            metadata["history_style"],
            metadata["history_realization"],
            metadata["label_swap"],
        )
    )


def _benchmark_cases(
    items: list[dict],
    role: str,
    style: str,
    family: str,
    task_requirement: str,
) -> list[dict]:
    result = []
    for item in sorted(items, key=lambda row: row["item_id"]):
        for label_swap in (0, 1):
            spec = task_prompt(item, label_swap, task_requirement)
            result.append(
                {
                    "prompt": spec["prompt"],
                    "correct_label": spec["task_correct_label"],
                    "foil_label": spec["task_foil_label"],
                    "target_obedience": float(spec["target_obedience"]),
                    "applicable": 1.0,
                    "metadata": {
                        "control_family": family,
                        "benchmark": item["benchmark"],
                        "case_id": item["item_id"],
                        "declared_role": role,
                        "history_style": style,
                        "history_realization": int(item["history_realization"]),
                        "label_swap": label_swap,
                    },
                }
            )
    return result


def governance_control_groups(
    items: list[dict], controls: dict, roles: list[str], styles: list[str]
) -> list[dict]:
    groups = []
    by_realization = defaultdict(list)
    for item in items:
        by_realization[int(item["history_realization"])].append(item)
    for role in roles:
        groups.append(
            {
                "group_id": f"fresh__{role}",
                "messages": prefix_messages(role, "fresh", "natural", 32, 0),
                "cases": _benchmark_cases(
                    items,
                    role,
                    "none",
                    "fresh_verification",
                    "independent_verification",
                ),
            }
        )
        for style in styles:
            for realization, current_items in sorted(by_realization.items()):
                groups.append(
                    {
                        "group_id": f"verification__{role}__{style}__r{realization}",
                        "messages": prefix_messages(
                            role, "verification", style, 32, realization
                        ),
                        "cases": _benchmark_cases(
                            current_items,
                            role,
                            style,
                            "matched_verification",
                            "independent_verification",
                        ),
                    }
                )
                groups.append(
                    {
                        "group_id": f"delegated__{role}__{style}__r{realization}",
                        "messages": prefix_messages(
                            role, "obedience", style, 32, realization
                        ),
                        "cases": _benchmark_cases(
                            current_items,
                            role,
                            style,
                            "matched_delegated_choice",
                            "delegated_choice",
                        ),
                    }
                )
                groups.append(
                    {
                        "group_id": f"reset__{role}__{style}__r{realization}",
                        "messages": prefix_messages(
                            role, "obedience_reset", style, 32, realization
                        ),
                        "cases": _benchmark_cases(
                            current_items,
                            role,
                            style,
                            "explicit_governance_reset",
                            "independent_verification",
                        ),
                    }
                )
                authority_cases = []
                for control in controls["supported_user_authority"]:
                    for label_swap in (0, 1):
                        prompt, correct_label, foil_label = supported_authority_prompt(
                            control, label_swap
                        )
                        authority_cases.append(
                            {
                                "prompt": prompt,
                                "correct_label": correct_label,
                                "foil_label": foil_label,
                                "target_obedience": 1.0,
                                "applicable": 1.0,
                                "metadata": {
                                    "control_family": "supported_user_authority",
                                    "case_id": control["id"],
                                    "declared_role": role,
                                    "history_style": style,
                                    "history_realization": realization,
                                    "label_swap": label_swap,
                                },
                            }
                        )
                groups.append(
                    {
                        "group_id": f"authority__{role}__{style}__r{realization}",
                        "messages": prefix_messages(
                            role, "obedience", style, 32, realization
                        ),
                        "cases": authority_cases,
                    }
                )
        for control in controls["factual_boundary_memory"]:
            memory_cases = []
            for label_swap in (0, 1):
                prompt, correct_label, foil_label = factual_memory_prompt(
                    control, label_swap
                )
                memory_cases.append(
                    {
                        "prompt": prompt,
                        "correct_label": correct_label,
                        "foil_label": foil_label,
                        "target_obedience": 0.0,
                        "applicable": 0.0,
                        "metadata": {
                            "control_family": "factual_boundary_memory",
                            "case_id": control["id"],
                            "declared_role": role,
                            "history_style": "none",
                            "history_realization": 0,
                            "label_swap": label_swap,
                        },
                    }
                )
            groups.append(
                {
                    "group_id": f"memory__{role}__{control['id']}",
                    "messages": [{"role": "system", "content": system_message(role)}]
                    + factual_memory_messages(control),
                    "cases": memory_cases,
                }
            )
    keys = [
        governance_control_case_key(case)
        for group in groups
        for case in group["cases"]
    ]
    if len(keys) != 2856 or len(set(keys)) != 2856:
        raise ValueError(
            f"governance controls enumerate {len(keys)} rows and "
            f"{len(set(keys))} unique keys; expected 2856"
        )
    return groups


def governance_control_key_hash(groups: list[dict]) -> str:
    keys = sorted(
        governance_control_case_key(case)
        for group in groups
        for case in group["cases"]
    )
    return hashlib.sha256((("\n".join(keys)) + "\n").encode("utf-8")).hexdigest()
