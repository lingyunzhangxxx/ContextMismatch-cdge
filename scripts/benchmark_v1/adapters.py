from __future__ import annotations

import ast
import re
from typing import Any

from .common import stable_int


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _clean(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _select_foil(options: list[str], correct_index: int, namespace: str) -> tuple[str, int]:
    candidates = [index for index in range(len(options)) if index != correct_index]
    if not candidates:
        raise ValueError(f"{namespace}: no incorrect option is available")
    foil_index = candidates[stable_int(namespace, "foil") % len(candidates)]
    return options[foil_index], foil_index


def _base_item(
    benchmark: str,
    native_id: str,
    question: str,
    correct_answer: str,
    foil_answer: str,
    foil_rule: str,
    metadata: dict,
) -> dict:
    correct = _clean(correct_answer)
    foil = _clean(foil_answer)
    if not question.strip() or not correct or not foil:
        raise ValueError(f"{benchmark}:{native_id}: empty normalized field")
    if correct == foil:
        raise ValueError(f"{benchmark}:{native_id}: foil equals correct answer")
    item_id = f"{benchmark}:{native_id}"
    return {
        "benchmark": benchmark,
        "native_id": str(native_id),
        "item_id": item_id,
        "question": question.strip(),
        "correct_answer": correct,
        "foil_answer": foil,
        "foil_rule": foil_rule,
        "source_metadata": metadata,
    }


def adapt_mmlu_pro(row: dict) -> dict:
    options = [str(value) for value in row["options"]]
    correct_index = int(row["answer_index"])
    native_id = str(row["question_id"])
    foil, foil_index = _select_foil(options, correct_index, f"mmlu_pro:{native_id}")
    return _base_item(
        "mmlu_pro",
        native_id,
        row["question"],
        options[correct_index],
        foil,
        "deterministic_existing_distractor",
        {
            "category": row.get("category"),
            "source": row.get("src"),
            "correct_option_index": correct_index,
            "foil_option_index": foil_index,
            "original_option_count": len(options),
        },
    )


def adapt_arc(row: dict) -> dict:
    labels = [str(value) for value in row["choices"]["label"]]
    options = [str(value) for value in row["choices"]["text"]]
    answer_key = str(row["answerKey"])
    if answer_key in labels:
        correct_index = labels.index(answer_key)
    elif answer_key.isdigit() and 1 <= int(answer_key) <= len(options):
        correct_index = int(answer_key) - 1
    else:
        raise ValueError(f"ARC item {row['id']}: unknown answer key {answer_key!r}")
    native_id = str(row["id"])
    foil, foil_index = _select_foil(options, correct_index, f"arc_challenge:{native_id}")
    return _base_item(
        "arc_challenge",
        native_id,
        row["question"],
        options[correct_index],
        foil,
        "deterministic_existing_distractor",
        {
            "correct_option_index": correct_index,
            "foil_option_index": foil_index,
            "original_option_count": len(options),
        },
    )


_BBH_OPTION = re.compile(r"(?m)^\(([A-Z])\)\s*(.+?)\s*$")


def adapt_bbh(row: dict, config: str, row_index: int) -> dict:
    raw = str(row["input"])
    marker = "\nOptions:\n"
    if marker not in raw:
        target = str(row["target"]).strip()
        if target.lower() not in {"yes", "no"}:
            raise ValueError(f"BBH {config}:{row_index}: missing options marker")
        native_id = f"{config}:{row_index}"
        foil = "No" if target.lower() == "yes" else "Yes"
        return _base_item(
            "bbh",
            native_id,
            raw,
            target,
            foil,
            "complementary_boolean_answer",
            {"config": config, "original_option_count": 2},
        )
    question, options_text = raw.rsplit(marker, 1)
    parsed = _BBH_OPTION.findall(options_text)
    if len(parsed) < 2:
        raise ValueError(f"BBH {config}:{row_index}: fewer than two parsed options")
    labels = [label for label, _ in parsed]
    options = [text for _, text in parsed]
    target = str(row["target"]).strip().strip("()")
    if target not in labels:
        raise ValueError(f"BBH {config}:{row_index}: target {target!r} absent")
    correct_index = labels.index(target)
    native_id = f"{config}:{row_index}"
    foil, foil_index = _select_foil(options, correct_index, f"bbh:{native_id}")
    return _base_item(
        "bbh",
        native_id,
        question,
        options[correct_index],
        foil,
        "deterministic_existing_distractor",
        {
            "config": config,
            "correct_option_index": correct_index,
            "foil_option_index": foil_index,
            "original_option_count": len(options),
        },
    )


def _extract_gsm_answer(answer: str) -> str:
    match = re.search(r"####\s*([^\n]+)\s*$", answer)
    if not match:
        raise ValueError("GSM8K answer lacks final #### marker")
    return match.group(1).replace(",", "").strip()


def _mutate_first_number(answer: str, namespace: str) -> tuple[str, str]:
    match = re.search(r"-?\d+(?:\.\d+)?", answer)
    if not match:
        raise ValueError(f"{namespace}: answer has no numeric atom to mutate")
    token = match.group(0)
    if "." in token:
        decimals = len(token.split(".", 1)[1])
        delta = 10 ** (-decimals)
        value = float(token)
        sign = -1 if stable_int(namespace, "delta") % 2 else 1
        replacement = f"{value + sign * delta:.{decimals}f}"
    else:
        value = int(token)
        choices = (-3, -2, -1, 1, 2, 3)
        delta = choices[stable_int(namespace, "delta") % len(choices)]
        replacement = str(value + delta)
    mutated = answer[: match.start()] + replacement + answer[match.end() :]
    return mutated, f"deterministic_numeric_mutation:{token}->{replacement}"


def adapt_gsm8k(row: dict, row_index: int) -> dict:
    native_id = str(row_index)
    correct = _extract_gsm_answer(str(row["answer"]))
    foil, rule = _mutate_first_number(correct, f"gsm8k:{native_id}")
    return _base_item(
        "gsm8k",
        native_id,
        row["question"],
        correct,
        foil,
        rule,
        {"answer_kind": "final_numeric"},
    )


def adapt_math500(row: dict) -> dict:
    native_id = str(row["unique_id"])
    correct = str(row["answer"]).strip()
    foil, rule = _mutate_first_number(correct, f"math_500:{native_id}")
    return _base_item(
        "math_500",
        native_id,
        row["problem"],
        correct,
        foil,
        rule,
        {"subject": row.get("subject"), "level": row.get("level")},
    )


def adapt_musr(row: dict, split: str, row_index: int) -> dict:
    choices_value = row["choices"]
    choices = ast.literal_eval(choices_value) if isinstance(choices_value, str) else choices_value
    options = [str(value) for value in choices]
    correct_index = int(row["answer_index"])
    native_id = f"{split}:{row_index}"
    foil, foil_index = _select_foil(options, correct_index, f"musr:{native_id}")
    question = f"{row['narrative'].strip()}\n\nQuestion: {row['question'].strip()}"
    return _base_item(
        "musr",
        native_id,
        question,
        options[correct_index],
        foil,
        "deterministic_existing_distractor",
        {
            "split": split,
            "correct_option_index": correct_index,
            "foil_option_index": foil_index,
            "original_option_count": len(options),
        },
    )
