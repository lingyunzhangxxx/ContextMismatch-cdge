"""Pure-Python frozen 24-cell C-DGE performance design."""

from __future__ import annotations

import hashlib
from collections import defaultdict

from scripts.benchmark_v2.crossover import enumerate_jobs, job_key


METHODS = ("baseline", "DSGE-V3", "C-DGE-V4.1")
TASK_REQUIREMENTS = ("delegated_choice", "independent_verification")
LABEL_SWAPS = (0, 1)
EXPECTED_CELLS = 24
EXPECTED_CELL_KEY_SHA256 = "007c94197c130c3e154d90160cf0fea3c59643a4a12fbcf76e10815be0c5d16a"
WARMUP_REPEATS = 3
TIMED_REPEATS = 10
EXPECTED_MEASUREMENTS = EXPECTED_CELLS * len(METHODS) * TIMED_REPEATS


def _selection_half(jobs: list[dict]) -> list[dict]:
    selected_ids = set()
    by_benchmark = defaultdict(set)
    for job in jobs:
        by_benchmark[job["item"]["benchmark"]].add(job["item"]["item_id"])
    for benchmark, item_ids in sorted(by_benchmark.items()):
        ordered = sorted(item_ids)
        if len(ordered) != 32:
            raise ValueError(f"operator_dev:{benchmark} has {len(ordered)} items instead of 32")
        selected_ids.update((benchmark, item_id) for item_id in ordered[16:])
    result = [
        job
        for job in jobs
        if (job["item"]["benchmark"], job["item"]["item_id"]) in selected_ids
    ]
    if len(result) != 3072:
        raise ValueError(f"operator-dev selection has {len(result)} rows instead of 3072")
    return result


def cell_key(job: dict) -> str:
    return "__".join(
        (
            job["item"]["benchmark"],
            job["task_requirement"],
            str(job["label_swap"]),
        )
    )


def selected_performance_jobs(manifest: list[dict], contract: dict) -> list[dict]:
    """Select the lexicographically first selection-half job in each frozen cell."""
    selection = _selection_half(enumerate_jobs(manifest, contract, "operator_dev"))
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for job in selection:
        grouped[
            (
                job["item"]["benchmark"],
                job["task_requirement"],
                int(job["label_swap"]),
            )
        ].append(job)
    jobs = [min(values, key=job_key) for _key, values in sorted(grouped.items())]
    keys = [cell_key(job) for job in jobs]
    if len(jobs) != EXPECTED_CELLS or len(set(keys)) != EXPECTED_CELLS:
        raise ValueError("performance design does not contain exactly 24 unique cells")
    if {job["task_requirement"] for job in jobs} != set(TASK_REQUIREMENTS):
        raise ValueError("performance task-requirement coverage mismatch")
    if {int(job["label_swap"]) for job in jobs} != set(LABEL_SWAPS):
        raise ValueError("performance label-swap coverage mismatch")
    if any(job != min((row for row in selection if cell_key(row) == cell_key(job)), key=job_key) for job in jobs):
        raise ValueError("performance selection rule mismatch")
    if cell_key_sha256(jobs) != EXPECTED_CELL_KEY_SHA256:
        raise ValueError("performance cell-key hash mismatch")
    return jobs


def cell_key_sha256(jobs: list[dict]) -> str:
    keys = sorted(cell_key(job) for job in jobs)
    return hashlib.sha256((("\n".join(keys)) + "\n").encode("utf-8")).hexdigest()


def measurement_key(cell: str, method: str, repeat: int) -> str:
    if method not in METHODS or repeat < 0 or repeat >= TIMED_REPEATS:
        raise ValueError("invalid performance measurement identity")
    return f"{cell}__{method}__r{repeat:02d}"
