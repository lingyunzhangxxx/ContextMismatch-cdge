#!/usr/bin/env python3
"""Validate and summarize the released experiment results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from scripts.reproduce_paper import reproduce


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(results: Path = RESULTS) -> int:
    manifest = results / "MANIFEST.sha256"
    entries = 0
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = results / relative
        if path.resolve().parent != results.resolve() or relative in seen:
            raise ValueError(f"invalid or duplicate manifest target: {relative}")
        seen.add(relative)
        if not path.is_file():
            raise FileNotFoundError(f"manifest target missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        entries += 1
    actual_files = {p.name for p in results.iterdir() if p.is_file() and p != manifest}
    if seen != actual_files:
        raise ValueError(f"manifest coverage mismatch: {sorted(seen ^ actual_files)}")
    return entries


def reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON number: {value}")


def check_finite(value) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite JSON number")
    if isinstance(value, dict):
        for child in value.values():
            check_finite(child)
    elif isinstance(value, list):
        for child in value:
            check_finite(child)


def verify_sources() -> None:
    provenance = json.loads((ROOT / "artifacts/release_provenance.json").read_text())
    for relative, record in provenance["files"].items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT.resolve()) or sha256_file(path) != record["release_sha256"]:
            raise ValueError(f"release provenance mismatch: {relative}")
    upstream = json.loads((ROOT / "third_party/UPSTREAM_SOURCE_MANIFEST_V2.json").read_text())
    for record in upstream["required_files"].values():
        path = ROOT / "third_party/upstream" / record["relative_path"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"fixed baseline source mismatch: {record['relative_path']}")
    for record in upstream["repositories"].values():
        path = ROOT / "third_party/upstream" / record["license_relative_path"]
        if sha256_file(path) != record["license_sha256"]:
            raise ValueError(f"fixed baseline license mismatch: {record['license_relative_path']}")


def validate_results() -> tuple[int, int, int]:
    json_files = 0
    jsonl_files = 0
    jsonl_rows = 0
    for path in sorted(RESULTS.iterdir()):
        if path.suffix == ".json":
            check_finite(json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_nonfinite))
            json_files += 1
        elif path.suffix == ".jsonl":
            seen_job_keys: set[str] = set()
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line, parse_constant=reject_nonfinite)
                    check_finite(row)
                    job_key = row.get("job_key", row.get("case_key", row.get("measurement_key")))
                    if job_key is not None:
                        if job_key in seen_job_keys:
                            raise ValueError(f"duplicate job_key in {path.name}:{line_number}")
                        seen_job_keys.add(job_key)
                    jsonl_rows += 1
            jsonl_files += 1
    return json_files, jsonl_files, jsonl_rows


def main() -> None:
    manifest_entries = verify_manifest()
    verify_sources()
    json_files, jsonl_files, jsonl_rows = validate_results()
    claims = json.loads((RESULTS / "paper_claims.json").read_text(encoding="utf-8"))
    production_approved = claims["interpretation"]["production_rollout_approved"]
    if production_approved is not False:
        raise ValueError("release must not claim production approval")
    summary = reproduce()
    if claims.get("cdge_final") != summary:
        raise ValueError("consolidated C-DGE claims differ from released rows")

    print("ContextMismatch-C-DGE release validation passed")
    print(f"  manifest entries: {manifest_entries}")
    print(f"  parsed JSON files: {json_files}")
    print(f"  parsed JSONL files: {jsonl_files}")
    print(f"  parsed JSONL rows: {jsonl_rows}")
    print(f"  consolidated behavior rows: {claims['behavior']['full_rows']}")
    print(f"  final mismatch accuracy: {100 * summary['baseline_accuracy']:.2f}% -> {100 * summary['edited_accuracy']:.2f}% ({summary['mismatched_rows']} rows)")
    print(f"  final normalized gap reduction: {100 * summary['normalized_gap_reduction']:.2f}%")
    print(f"  rescued mismatch flips: {summary['rescued_flips']} / {summary['mismatch_induced_flips']}")
    print(f"  protected controls changed: {summary['protected_changed_rows']} / {summary['protected_rows']}")
    print("  production rollout approved: false")


if __name__ == "__main__":
    main()
