#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.crossover import audit_design, enumerate_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    manifest = load_jsonl(args.manifest)
    jobs = enumerate_jobs(manifest, contract, args.stage)
    report = {
        "schema_version": 1,
        "stage": args.stage,
        "contract_sha256": sha256_file(args.contract),
        "manifest_sha256": sha256_file(args.manifest),
        "audit": audit_design(jobs),
    }
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["audit"]["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
