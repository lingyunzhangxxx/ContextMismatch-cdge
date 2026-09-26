#!/usr/bin/env python3
"""Verify, restore, and recompute the complete retained experiment records."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import tarfile
import urllib.request
import urllib.parse
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

from scripts.benchmark_v2.analyze_crossover import build_interactions, summarize
from scripts.benchmark_v2.analyze_operator_candidate import _view
from scripts.reproduce_paper import reproduce

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "records"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_path(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe relative record path: {name}")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Record path escapes its destination")
    return path


def read_index() -> dict:
    # The external manifest binds the index as well as every compressed blob.
    manifest = RECORDS / "MANIFEST.sha256"
    seen = set()
    for line in manifest.read_text().splitlines():
        expected, name = line.split("  ", 1)
        if name in seen:
            raise ValueError("Duplicate manifest entry")
        seen.add(name)
        if digest(safe_path(RECORDS, name).read_bytes()) != expected:
            raise ValueError(f"Record manifest mismatch: {name}")
    actual = {p.relative_to(RECORDS).as_posix() for p in RECORDS.rglob("*")
              if p.is_file() and p != manifest}
    if seen != actual:
        raise ValueError("Record manifest does not have exact file coverage")
    return json.loads((RECORDS / "index.json").read_text())


def read_record(entry: dict) -> bytes:
    if "repository_file" in entry:
        raw = safe_path(ROOT, entry["repository_file"]).read_bytes()
        if digest(raw) != entry["released_sha256"]:
            raise ValueError("Repository evidence checksum mismatch")
        return raw
    data = safe_path(RECORDS, entry["blob"]).read_bytes()
    if digest(data) != entry["blob_sha256"]:
        raise ValueError("Compressed blob checksum mismatch")
    raw = gzip.decompress(data)
    if digest(raw) != entry["released_sha256"]:
        raise ValueError("Decompressed record checksum mismatch")
    return raw


def finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return True


def verify(index: dict) -> dict:
    sources = set()
    total_rows = 0
    nonfinite_reports = []
    blobs = set()
    tensors = set()
    for entry in index["files"]:
        name = entry["source_relative"]
        safe_path(ROOT / "generated/evidence", name)
        if name in sources:
            raise ValueError("Duplicate scientific source path")
        sources.add(name)
        if entry["kind"] == "pt":
            tensors.add(entry["released_sha256"])
            if entry["source_sha256"] != entry["released_sha256"]:
                raise ValueError("Released tensor differs from frozen source")
            continue
        if "blob" in entry:
            blobs.add(entry["blob"])
        raw = read_record(entry)
        if entry["kind"] == "jsonl":
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
            if len(rows) != entry["row_count"] or not finite(rows):
                raise ValueError(f"Invalid measurement rows: {name}")
            # Public row records must be byte-identical to retained originals.
            if entry["released_sha256"] != entry["source_sha256"]:
                raise ValueError(f"Raw measurement file was altered: {name}")
            total_rows += len(rows)
        elif not finite(json.loads(raw)):
            nonfinite_reports.append(name)
    observed = {p.relative_to(RECORDS).as_posix() for p in (RECORDS / "blobs").glob("*")}
    if observed != blobs:
        raise ValueError("Index does not have exact blob coverage")
    # Never treat a nonfinite historical report as verified final evidence.
    if sorted(nonfinite_reports) != sorted(index["historical_nonfinite_reports"]):
        raise ValueError("Uncatalogued nonfinite historical report")
    if total_rows != index["counts"]["jsonl_rows"]:
        raise ValueError("Index row total mismatch")
    if len(tensors) != index["counts"]["unique_tensors"]:
        raise ValueError("Index tensor count mismatch")
    if len({entry["source_sha256"] for entry in index["files"]}) != len(sources):
        raise ValueError("Index contains duplicated scientific content")
    return {"files": len(sources), "measurement_rows": total_rows,
            "unique_tensor_files": len(tensors),
            "historical_nonfinite_reports_excluded_from_claims": len(nonfinite_reports)}


def restore(index: dict, output: Path, asset_dir: Path | None) -> dict:
    restored = 0
    for entry in index["files"]:
        if entry["kind"] == "pt":
            continue
        destination = safe_path(output, entry["source_relative"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(read_record(entry))
        restored += 1
    tensor_count = 0
    if asset_dir is not None:
        locations = defaultdict(list)
        for entry in index["files"]:
            if entry["kind"] == "pt":
                locations[(entry["asset"], entry["tensor_member"])].append(entry)
        for asset in index["assets"]:
            path = asset_dir / asset["name"]
            if digest(path.read_bytes()) != asset["sha256"]:
                raise ValueError(f"Tensor archive checksum mismatch: {path.name}")
            seen = set()
            with tarfile.open(path, "r:gz") as archive:
                for member in archive:
                    entries = locations.get((path.name, member.name))
                    if not member.isfile() or not entries or member.name in seen:
                        raise ValueError("Unexpected or unsafe tensor archive member")
                    seen.add(member.name)
                    raw = archive.extractfile(member).read()
                    if digest(raw) != entries[0]["released_sha256"]:
                        raise ValueError("Tensor member checksum mismatch")
                    for entry in entries:
                        destination = safe_path(output, entry["source_relative"])
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(raw)
                        tensor_count += 1
            expected = {name for asset_name, name in locations if asset_name == path.name}
            if seen != expected or len(seen) != asset["members"]:
                raise ValueError("Incomplete tensor archive")
    return {"restored_text_records": restored, "restored_tensors_including_copies": tensor_count,
            "output": str(output)}


def download(index: dict, asset_dir: Path) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    for asset in index["assets"]:
        path = asset_dir / asset["name"]
        if path.exists() and digest(path.read_bytes()) == asset["sha256"]:
            continue
        temporary = path.with_suffix(path.suffix + ".partial")
        print(f"Downloading {asset['name']} ({asset['bytes'] / 2**20:.1f} MiB)", flush=True)
        # A freshly published or renamed GitHub release may have a cached 404
        # for its canonical URL. Bind the request query to the immutable hash.
        url = asset["url"] + ("&" if "?" in asset["url"] else "?") + urllib.parse.urlencode(
            {"verify_sha256": asset["sha256"]})
        request = urllib.request.Request(url, headers={
            "User-Agent": "ContextMismatch-evidence-verifier", "Cache-Control": "no-cache"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        if digest(temporary.read_bytes()) != asset["sha256"]:
            raise ValueError("Downloaded tensor archive checksum mismatch")
        temporary.replace(path)


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Recomputed value differs from frozen report: {label}")


def behavior_metrics(rows: list[dict], report: dict) -> dict:
    before, problems_before = build_interactions(_view(rows, "baseline"))
    after, problems_after = build_interactions(_view(rows, "edited"))
    if problems_before or problems_after or len(before) != len(rows) // 4:
        raise ValueError("Incomplete counterbalanced behavior rows")
    reductions = [{"benchmark": raw["benchmark"], "value":
                   (raw["matched_minus_mismatched"] - edited["matched_minus_mismatched"])
                   / max(abs(raw["matched_minus_mismatched"]), 1.0)}
                  for raw, edited in zip(before, after)]
    ngr = summarize(reductions, "value")["equal_weight_benchmark_mean"]
    close(ngr, report["normalized_gap_reduction"]["equal_weight_benchmark_mean"], "NGR")
    mismatch = [row for row in rows if not row["matched_context"]]
    return {"rows": len(rows), "normalized_gap_reduction": ngr,
            "mismatched_baseline_correct": sum(row["baseline"]["task_aligned_margin"] > 0 for row in mismatch),
            "mismatched_edited_correct": sum(row["edited"]["task_aligned_margin"] > 0 for row in mismatch)}


def recompute(index: dict) -> dict:
    lookup = {entry["source_relative"]: entry for entry in index["files"]}
    by_original = {entry["source_sha256"]: entry for entry in index["files"]}

    def obj(entry):
        return json.loads(read_record(entry))

    def rows(entry):
        return [json.loads(line) for line in read_record(entry).splitlines() if line.strip()]

    result = {"primary_final_test": reproduce(), "official_comparators": {},
              "reported_ablations": {}}
    ablations = json.loads((ROOT / "results/dge_same_identity_comparison.json").read_text())
    for method, report in ablations["method_reports"].items():
        if method == "full_dge":
            raw = by_original[ablations["full_dge_sources"]["rows_sha256"]]
        else:
            raw = next(entry for entry in index["files"]
                       if entry["source_relative"].endswith(f"same_identity/rows/{method}.jsonl"))
        result["reported_ablations"][method] = behavior_metrics(rows(raw), report)
    comparison = json.loads((ROOT / "results/qwen3_8b_official_derived_baseline_comparison.json").read_text())
    for method, summary in comparison["methods"].items():
        analysis = by_original[summary["analysis_sha256"]]
        raw = lookup[str(PurePosixPath(analysis["source_relative"]).with_suffix(""))
                     .removesuffix(".analysis") + ".jsonl"]
        result["official_comparators"][method] = behavior_metrics(rows(raw), obj(analysis))
    native_path = ROOT / "results/qwen35_cdge_v4_2_native_behavior_merge.json"
    native = json.loads(native_path.read_text()) if native_path.exists() else {"candidates": []}
    reported_finalists = {"CDGE42-e22e2b34b3a88394", "CDGE42-f1c7b6e3c13ae32c"}
    for candidate in native["candidates"]:
        if candidate["candidate_id"] not in reported_finalists:
            continue
        analysis = by_original[candidate["analysis_sha256"]]
        raw = by_original[candidate["rows_sha256"]]
        if analysis["source_sha256"] != candidate["analysis_sha256"]:
            raise ValueError("Native candidate report source binding mismatch")
        result.setdefault("native_qwen35_behavior_candidates", {})[candidate["candidate_id"]] = behavior_metrics(rows(raw), obj(analysis))
    for entry in index["files"]:
        if entry["kind"] != "jsonl" or "native-controls-CDGE42-" not in entry["source_relative"]:
            continue
        measurements = rows(entry)
        report = obj(lookup[entry["source_relative"].removesuffix(".jsonl") + ".analysis.json"])
        if entry["source_sha256"] != report["input_sha256"]:
            raise ValueError("Native controls do not match the frozen report input")
        family_results = {}
        for family in sorted({row["control_family"] for row in measurements}):
            selected = [row for row in measurements if row["control_family"] == family]
            error = max(row["application_gated_selected_logit_error"] for row in selected)
            frozen = report["family_reports"][family]
            close(error, frozen["application_gated_identity"]["max_selected_logit_error"], family)
            family_results[family] = {"rows": len(selected), "max_selected_logit_error": error,
                                      "changed_rows": sum(row["baseline"] != row["application_gated"] for row in selected)}
        result.setdefault("native_qwen35_controls", {})[report["candidate_id"]] = family_results
    # The final-test environment must bind actual released editor state.
    final_env = next(entry for entry in index["files"]
                     if "cdge-v4-1-final-recovery" in entry["source_relative"]
                     and entry["source_relative"].endswith(".environment.json"))
    environment = obj(final_env)
    bindings = {key: value for key, value in environment.items()
                if "checkpoint_sha256" in key}
    for key, value in bindings.items():
        if value not in by_original or by_original[value]["kind"] != "pt":
            raise ValueError(f"Final test checkpoint is not released: {key}")
    result["final_checkpoint_bindings"] = bindings
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("verify", "metrics", "restore", "download"), default="verify")
    parser.add_argument("--output", type=Path, default=ROOT / "generated/evidence")
    parser.add_argument("--asset-dir", type=Path)
    args = parser.parse_args()
    index = read_index()
    result = verify(index)
    if args.action == "metrics":
        result = recompute(index)
        target = ROOT / "generated/complete_record_metrics.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    elif args.action == "restore":
        result.update(restore(index, args.output, args.asset_dir))
    elif args.action == "download":
        download(index, args.asset_dir or ROOT / "generated/tensor-assets")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
