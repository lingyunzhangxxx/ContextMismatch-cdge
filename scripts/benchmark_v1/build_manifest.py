#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adapters import (
    adapt_arc,
    adapt_bbh,
    adapt_gsm8k,
    adapt_math500,
    adapt_mmlu_pro,
    adapt_musr,
)
from .common import (
    DEFAULT_CONTRACT,
    assign_partitions,
    atomic_write_text,
    canonical_json,
    deterministic_take,
    sha256_file,
)


def _load_dataset(repository: str, config: str | None, split: str, revision: str):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("build_manifest requires the Hugging Face datasets package") from error
    return load_dataset(repository, config, split=split, revision=revision)


def build_benchmark(spec: dict, seed: int) -> list[dict]:
    benchmark = spec["id"]
    repository = spec["repository"]
    revision = spec["revision"]
    if benchmark == "mmlu_pro":
        dataset = _load_dataset(repository, spec["config"], spec["split"], revision)
        eligible = [adapt_mmlu_pro(dict(row)) for row in dataset]
        selected = deterministic_take(eligible, spec["sample_count"], seed, benchmark)
    elif benchmark == "arc_challenge":
        dataset = _load_dataset(repository, spec["config"], spec["split"], revision)
        eligible = [adapt_arc(dict(row)) for row in dataset]
        selected = deterministic_take(eligible, spec["sample_count"], seed, benchmark)
    elif benchmark == "bbh":
        selected = []
        for config in spec["configs"]:
            dataset = _load_dataset(repository, config, spec["split"], revision)
            eligible = [adapt_bbh(dict(row), config, index) for index, row in enumerate(dataset)]
            selected.extend(
                deterministic_take(eligible, spec["sample_count_per_config"], seed, f"{benchmark}:{config}")
            )
    elif benchmark == "gsm8k":
        dataset = _load_dataset(repository, spec["config"], spec["split"], revision)
        eligible = [adapt_gsm8k(dict(row), index) for index, row in enumerate(dataset)]
        selected = deterministic_take(eligible, spec["sample_count"], seed, benchmark)
    elif benchmark == "math_500":
        dataset = _load_dataset(repository, spec["config"], spec["split"], revision)
        eligible = []
        for row in dataset:
            try:
                eligible.append(adapt_math500(dict(row)))
            except ValueError as error:
                if "no numeric atom to mutate" not in str(error):
                    raise
        selected = deterministic_take(eligible, spec["sample_count"], seed, benchmark)
    elif benchmark == "musr":
        selected = []
        for split, quota in spec["sample_count_by_split"].items():
            dataset = _load_dataset(repository, spec["config"], split, revision)
            eligible = [adapt_musr(dict(row), split, index) for index, row in enumerate(dataset)]
            selected.extend(deterministic_take(eligible, quota, seed, f"{benchmark}:{split}"))
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    if len(selected) != spec["sample_count"]:
        raise AssertionError(f"{benchmark}: selected {len(selected)} != {spec['sample_count']}")
    for item in selected:
        item["dataset_repository"] = repository
        item["dataset_revision"] = revision
    return assign_partitions(selected, seed, benchmark)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    seed = int(contract["manifest"]["seed"])
    rows = []
    for spec in contract["benchmarks"]:
        rows.extend(build_benchmark(spec, seed))
    rows.sort(key=lambda row: (row["benchmark"], row["partition"], row["item_id"]))
    text = "".join(canonical_json(row) + "\n" for row in rows)
    atomic_write_text(args.output, text, args.allow_overwrite)
    report = {
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256_file(args.contract),
        "manifest_path": str(args.output),
        "manifest_sha256": sha256_file(args.output),
        "row_count": len(rows),
        "benchmarks": {
            benchmark: sum(row["benchmark"] == benchmark for row in rows)
            for benchmark in sorted({row["benchmark"] for row in rows})
        },
        "partitions": {
            partition: sum(row["partition"] == partition for row in rows)
            for partition in contract["manifest"]["partitions"]
        },
        "postselected_on_model_outputs": False,
    }
    atomic_write_text(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n", args.allow_overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
