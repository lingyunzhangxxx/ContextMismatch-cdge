#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v2.operator_grid import candidate_id, canonical_json, enumerate_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    operator_contract = json.loads(args.operator_contract.read_text())
    extension = json.loads(args.extension_contract.read_text())
    site_manifest = json.loads(args.operator_site_manifest.read_text())
    configs = enumerate_grid(operator_contract, extension, site_manifest)
    records = [{"candidate_id": candidate_id(config), "config": config} for config in configs]
    text = "".join(canonical_json(record) + "\n" for record in records)
    atomic_write_text(args.output, text)
    report = {
        "schema_version": 1,
        "grid_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "grid_rows": len(records),
        "family_counts": dict(sorted(Counter(row["config"]["family"] for row in records).items())),
        "operator_contract_sha256": sha256_file(args.operator_contract),
        "extension_contract_sha256": sha256_file(args.extension_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "single_site_only": True,
        "selection_performed": False,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
