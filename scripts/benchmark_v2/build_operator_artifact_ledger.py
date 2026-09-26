#!/usr/bin/env python3
"""Build a portable local/cluster ledger from one archived operator run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


def _artifact(local: Path, remote: Path) -> dict[str, str]:
    if not local.is_file():
        raise FileNotFoundError(local)
    return {
        "local_path": str(local.resolve()),
        "remote_path": str(remote),
        "sha256": sha256_file(local),
    }


def _candidate_from_fit(local_run: Path, remote_run: Path) -> list[dict]:
    records = []
    root = local_run / "operator_candidates"
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        manifests = list(directory.glob("operator_candidate_*.manifest.json"))
        tensors = list(directory.glob("operator_candidate_*.pt"))
        if len(manifests) != 1 or len(tensors) != 1:
            raise ValueError(f"candidate directory is ambiguous: {directory}")
        manifest = json.loads(manifests[0].read_text())
        if manifest.get("candidate_id") != directory.name:
            raise ValueError(f"candidate directory/id mismatch: {directory}")
        if manifest.get("tensor_sha256") != sha256_file(tensors[0]):
            raise ValueError(f"candidate tensor/manifest mismatch: {directory}")
        records.append(
            {
                "candidate_id": directory.name,
                "candidate_manifest": _artifact(
                    manifests[0], remote_run / "operator_candidates" / directory.name / manifests[0].name
                ),
                "candidate_tensor": _artifact(
                    tensors[0], remote_run / "operator_candidates" / directory.name / tensors[0].name
                ),
            }
        )
    return records


def _candidates_from_batch(local_run: Path, remote_run: Path, mode: str) -> list[dict]:
    ledger = json.loads((local_run / "batch_ledger.json").read_text())
    if ledger.get("mode") != mode:
        raise ValueError("batch ledger mode mismatch")
    records = []
    for row in ledger["candidates"]:
        candidate_id = row["candidate_id"]
        record = {"candidate_id": candidate_id}
        for field in ("candidate_manifest", "candidate_tensor", "identity_report", "analysis"):
            relative = Path(row[field])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe batch-ledger path: {relative}")
            record[field] = _artifact(local_run / relative, remote_run / relative)
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("fit_candidates", "behavior_screening", "behavior_selection", "controls_selection"),
    )
    parser.add_argument("--local-run-dir", type=Path, required=True)
    parser.add_argument("--remote-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing ledger: {args.output}")
    if not args.local_run_dir.is_dir():
        raise FileNotFoundError(args.local_run_dir)
    if not str(args.remote_run_dir).startswith(
        "/workspace/context-mismatch-qwen3-8b/runs/"
    ):
        raise ValueError("remote run directory is outside the owned run root")
    if args.mode == "fit_candidates":
        records = _candidate_from_fit(args.local_run_dir, args.remote_run_dir)
    else:
        records = _candidates_from_batch(args.local_run_dir, args.remote_run_dir, args.mode)
    if not records or len({row["candidate_id"] for row in records}) != len(records):
        raise ValueError("artifact ledger is empty or has duplicate candidate ids")
    value = {
        "schema_version": 1,
        "mode": args.mode,
        "local_run_dir": str(args.local_run_dir.resolve()),
        "remote_run_dir": str(args.remote_run_dir),
        "candidate_count": len(records),
        "candidates": records,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": args.mode, "candidate_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
