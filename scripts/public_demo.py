#!/usr/bin/env python3
"""Validate and summarize the released experiment results."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest() -> int:
    manifest = RESULTS / "MANIFEST.sha256"
    entries = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = RESULTS / relative
        if not path.is_file():
            raise FileNotFoundError(f"manifest target missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch: {relative}")
        entries += 1
    return entries


def validate_results() -> tuple[int, int, int]:
    json_files = 0
    jsonl_files = 0
    jsonl_rows = 0
    for path in sorted(RESULTS.iterdir()):
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            json_files += 1
        elif path.suffix == ".jsonl":
            seen_job_keys: set[str] = set()
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    job_key = row.get("job_key")
                    if job_key is not None:
                        if job_key in seen_job_keys:
                            raise ValueError(f"duplicate job_key in {path.name}:{line_number}")
                        seen_job_keys.add(job_key)
                    jsonl_rows += 1
            jsonl_files += 1
    return json_files, jsonl_files, jsonl_rows


def main() -> None:
    manifest_entries = verify_manifest()
    json_files, jsonl_files, jsonl_rows = validate_results()
    claims = json.loads((RESULTS / "paper_claims.json").read_text(encoding="utf-8"))
    production_approved = claims["interpretation"]["production_rollout_approved"]
    if production_approved is not False:
        raise ValueError("release must not claim production approval")

    print("ContextMismatch-C-DGE release validation passed")
    print(f"  manifest entries: {manifest_entries}")
    print(f"  parsed JSON files: {json_files}")
    print(f"  parsed JSONL files: {jsonl_files}")
    print(f"  parsed JSONL rows: {jsonl_rows}")
    print(f"  consolidated behavior rows: {claims['behavior']['full_rows']}")
    print("  production rollout approved: false")


if __name__ == "__main__":
    main()
