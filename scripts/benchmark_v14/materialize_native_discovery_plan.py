#!/usr/bin/env python3
"""Materialize the model-derived layer scan plan before any discovery forward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v14.native_discovery import build_scan_plan, load_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol = load_protocol(args.protocol)
    model = json.loads(args.model_contract.read_text())
    if sha256_file(args.model_contract) != protocol["frozen_lineage"]["model_contract_sha256"]:
        raise ValueError("model contract SHA differs from frozen protocol")
    plan = build_scan_plan(protocol, model)
    plan.update({
        "protocol_sha256": sha256_file(args.protocol),
        "model_contract_sha256": sha256_file(args.model_contract),
    })
    atomic_write_text(args.output, json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(json.dumps(plan, sort_keys=True))


if __name__ == "__main__":
    main()
