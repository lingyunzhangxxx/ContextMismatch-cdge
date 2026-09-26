#!/usr/bin/env python3
"""Merge archived operator batch ledgers into selector-ready immutable ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b/"


def _load_artifact(value: dict, field: str) -> dict[str, str]:
    required = {"local_path", "remote_path", "sha256"}
    if set(value) != required:
        raise ValueError(f"{field} artifact fields are {sorted(value)}, expected {sorted(required)}")
    local = Path(value["local_path"])
    remote = str(value["remote_path"])
    digest = str(value["sha256"])
    if not local.is_file() or sha256_file(local) != digest:
        raise ValueError(f"{field} local artifact SHA mismatch: {local}")
    if not remote.startswith(REMOTE_ROOT):
        raise ValueError(f"{field} remote path is outside the owned project: {remote}")
    return {
        "local_path": str(local.resolve()),
        "remote_path": remote,
        "sha256": digest,
    }


def _read_ledgers(paths: list[Path], expected_mode: str) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in paths:
        ledger = json.loads(path.read_text())
        if ledger.get("mode") != expected_mode:
            raise ValueError(f"ledger mode mismatch in {path}: {ledger.get('mode')!r}")
        if ledger.get("final_test_open") is not False:
            raise ValueError(f"ledger unexpectedly opens final test: {path}")
        if ledger.get("production_rollout_approved") is not False:
            raise ValueError(f"ledger changes production boundary: {path}")
        for row in ledger.get("candidates", []):
            candidate_id = row.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                raise ValueError(f"invalid candidate id in {path}")
            if candidate_id in records:
                raise ValueError(f"duplicate candidate across {expected_mode} ledgers: {candidate_id}")
            record = {"candidate_id": candidate_id, "source_ledger": str(path.resolve())}
            for field in ("candidate_manifest", "candidate_tensor", "identity_report", "analysis"):
                record[field] = _load_artifact(row[field], f"{candidate_id}:{field}")
            manifest = json.loads(Path(record["candidate_manifest"]["local_path"]).read_text())
            analysis = json.loads(Path(record["analysis"]["local_path"]).read_text())
            identity = json.loads(Path(record["identity_report"]["local_path"]).read_text())
            if manifest.get("candidate_id") != candidate_id:
                raise ValueError(f"candidate manifest identity mismatch: {candidate_id}")
            if manifest.get("tensor_sha256") != record["candidate_tensor"]["sha256"]:
                raise ValueError(f"candidate tensor binding mismatch: {candidate_id}")
            if analysis.get("candidate_id") != candidate_id:
                raise ValueError(f"candidate analysis identity mismatch: {candidate_id}")
            if analysis.get("audit", {}).get("success") is not True:
                raise ValueError(f"candidate analysis audit failed: {candidate_id}")
            if identity.get("candidate_id") != candidate_id:
                raise ValueError(f"candidate identity-report mismatch: {candidate_id}")
            if identity.get("success") is not True or float(identity.get("max_error", 1.0)) != 0.0:
                raise ValueError(f"candidate zero-gate identity failed: {candidate_id}")
            records[candidate_id] = record
    if not records:
        raise ValueError(f"no candidates in {expected_mode} ledgers")
    return records


def _common_artifacts(left: dict, right: dict, candidate_id: str) -> None:
    for field in ("candidate_manifest", "candidate_tensor"):
        if left[field]["sha256"] != right[field]["sha256"]:
            raise ValueError(f"{field} differs across evaluation stages: {candidate_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("screening", "selection_controls"))
    parser.add_argument("--behavior-ledger", type=Path, action="append", required=True)
    parser.add_argument("--controls-ledger", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing merged ledger: {args.output}")

    behavior_mode = "behavior_screening" if args.mode == "screening" else "behavior_selection"
    behavior = _read_ledgers(args.behavior_ledger, behavior_mode)
    controls: dict[str, dict] = {}
    if args.mode == "screening":
        if args.controls_ledger:
            raise ValueError("screening merge does not accept controls ledgers")
    else:
        if not args.controls_ledger:
            raise ValueError("selection_controls merge requires controls ledgers")
        controls = _read_ledgers(args.controls_ledger, "controls_selection")
        if set(behavior) != set(controls):
            raise ValueError(
                "selection/control candidate sets differ: "
                f"selection_only={sorted(set(behavior) - set(controls))}, "
                f"controls_only={sorted(set(controls) - set(behavior))}"
            )

    candidates = []
    for candidate_id, row in sorted(behavior.items()):
        manifest = row["candidate_manifest"]
        tensor = row["candidate_tensor"]
        output = {
            "candidate_id": candidate_id,
            "candidate_manifest": manifest["local_path"],
            "candidate_manifest_remote": manifest["remote_path"],
            "candidate_manifest_sha256": manifest["sha256"],
            "candidate_tensor": tensor["local_path"],
            "candidate_tensor_remote": tensor["remote_path"],
            "candidate_tensor_sha256": tensor["sha256"],
            "identity_report": row["identity_report"]["local_path"],
            "identity_report_sha256": row["identity_report"]["sha256"],
        }
        if args.mode == "screening":
            output["screening_analysis"] = row["analysis"]["local_path"]
            output["screening_analysis_sha256"] = row["analysis"]["sha256"]
        else:
            control = controls[candidate_id]
            _common_artifacts(row, control, candidate_id)
            output["selection_analysis"] = row["analysis"]["local_path"]
            output["selection_analysis_sha256"] = row["analysis"]["sha256"]
            output["controls_analysis"] = control["analysis"]["local_path"]
            output["controls_analysis_sha256"] = control["analysis"]["sha256"]
        candidates.append(output)

    value = {
        "schema_version": 1,
        "mode": args.mode,
        "candidate_count": len(candidates),
        "behavior_ledger_sha256": [sha256_file(path) for path in args.behavior_ledger],
        "controls_ledger_sha256": [sha256_file(path) for path in args.controls_ledger],
        "candidates": candidates,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"mode": args.mode, "candidate_count": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
