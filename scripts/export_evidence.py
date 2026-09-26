#!/usr/bin/env python3
"""Build an auditable public copy of selected final experimental evidence.

Deployment logs and archive-transfer receipts are deliberately excluded. JSON
measurements, fit reports, scientific manifests, and tensors are retained.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

from scripts.privacy_audit import TEXT_PATTERNS

PRIVATE_PATHS = [
    (r"/Users/[A-Za-z0-9._-]+/", "/workspace/"),
    (r"/home/[A-Za-z0-9._-]+/", "/workspace/"),
    (r"/[r]oot/", "/workspace/"),
    (r"/WORK/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/", "/workspace/"),
    (r"/WORK/[A-Za-z0-9._-]+/", "/workspace/"),
    (r"/data[0-9]*/[A-Za-z0-9._-]+/", "/workspace/"),
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize(data: bytes) -> bytes:
    text = data.decode("utf-8")
    for pattern, replacement in PRIVATE_PATHS:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"data_[A-Za-z0-9]+_copy", "independent_archive_copy", text)
    for label, pattern in TEXT_PATTERNS.items():
        if pattern.search(text):
            raise ValueError(f"Unsanitized {label}")
    return text.encode("utf-8")


def scientific_files(source: Path) -> list[Path]:
    excluded = {"evidence-archives", "node-selections", "archive-receipts"}
    return sorted(
        p for directory in ("artifacts", "results")
        for p in (source / directory).rglob("*")
        if p.is_file() and p.suffix in {".json", ".jsonl", ".pt"}
        and not excluded.intersection(p.relative_to(source).parts)
        and not any(s in p.name.lower() for s in ("archive_receipt", "node_selection"))
    )


def check_tensor(path: Path) -> None:
    import torch

    # These are locally produced activation captures and editor states, not
    # foundation model weights. No arbitrary pickle globals are permitted.
    payload = torch.load(path, map_location="cpu", weights_only=True)

    def walk(value):
        if isinstance(value, str):
            for label, pattern in TEXT_PATTERNS.items():
                if pattern.search(value):
                    raise ValueError(f"Tensor metadata contains {label}: {path.name}")
        elif isinstance(value, dict):
            if value.get("base_model_weights_included") is True:
                raise ValueError("Foundation model weights must not be exported")
            for key, item in value.items():
                walk(key)
                walk(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)

    walk(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--extra-source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("records"))
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True,
                        help="Reviewed JSON manifest of final evidence source paths")
    parser.add_argument("--release-base-url", required=True)
    parser.add_argument("--part-mib", type=int, default=600)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    blobs = args.output / "blobs"
    blobs.mkdir(exist_ok=True)
    entries = []
    tensors = {}
    row_total = 0
    nonfinite_reports = []
    selection = json.loads(args.selection.read_text())
    selected = set(selection["files"])
    available = {}
    for source in [args.source, *args.extra_source]:
        for path in scientific_files(source):
            available.setdefault(path.relative_to(source).as_posix(), path)
    if not selected.issubset(available):
        raise ValueError("Selected scientific evidence is missing")
    seen_hashes = set()
    repository_root = Path(__file__).resolve().parents[1]
    existing_results = {sha(p.read_bytes()): p.relative_to(repository_root).as_posix()
                        for p in (repository_root / "results").iterdir()
                        if p.is_file() and p.suffix in {".json", ".jsonl"}}
    for relative, path in sorted(available.items()):
        if relative not in selected:
            continue
        data = path.read_bytes()
        original_sha = sha(data)
        if original_sha in seen_hashes:
            raise ValueError("Selection contains duplicate scientific content")
        seen_hashes.add(original_sha)
        entry = {"source_relative": relative,
                 "source_sha256": original_sha, "source_bytes": len(data),
                 "kind": path.suffix[1:]}
        if path.suffix == ".pt":
            if original_sha not in tensors:
                check_tensor(path)
                tensors[original_sha] = path
            entry.update({"released_sha256": original_sha,
                          "tensor_member": f"tensors/{original_sha}.pt",
                          "transformation": "none"})
        else:
            try:
                public = sanitize(data)
            except ValueError as error:
                raise ValueError(f"{relative}: {error}") from error
            public_sha = sha(public)
            name = f"blobs/{public_sha}.{path.suffix[1:]}.gz"
            blob = args.output / name
            if not blob.exists():
                blob.write_bytes(gzip.compress(public, compresslevel=6, mtime=0))
            entry.update({"released_sha256": public_sha, "blob": name,
                          "blob_sha256": sha(blob.read_bytes()),
                          "released_bytes": len(public),
                          "transformation": "none" if data == public else "private location strings redacted"})
            if public_sha in existing_results:
                entry["repository_file"] = existing_results[public_sha]
                del entry["blob"]
                del entry["blob_sha256"]
            if path.suffix == ".jsonl":
                rows = [json.loads(line) for line in public.splitlines() if line.strip()]
                entry["row_count"] = len(rows)
                row_total += len(rows)
            else:
                payload = json.loads(public)
                from scripts.evidence_records import finite
                if not finite(payload):
                    nonfinite_reports.append(entry["source_relative"])
        entries.append(entry)
    parts = []
    current = []
    size = 0
    groups = []
    for digest, path in sorted(tensors.items()):
        if current and size + path.stat().st_size > args.part_mib * 2**20:
            groups.append(current)
            current = []
            size = 0
        current.append((digest, path))
        size += path.stat().st_size
    if current:
        groups.append(current)
    for number, group in enumerate(groups, 1):
        name = f"scientific-tensors-{number:02d}.tar.gz"
        destination = args.asset_dir / name
        with destination.open("wb") as stream:
            with gzip.GzipFile(fileobj=stream, mode="wb", compresslevel=1, mtime=0, filename="") as compressed:
                with tarfile.open(fileobj=compressed, mode="w|") as tar:
                    for digest, path in group:
                        info = tarfile.TarInfo(f"tensors/{digest}.pt")
                        info.size = path.stat().st_size
                        info.mode = 0o644
                        tar.addfile(info, io.BytesIO(path.read_bytes()))
        parts.append({"name": name, "sha256": sha(destination.read_bytes()),
                      "bytes": destination.stat().st_size, "members": len(group),
                      "url": args.release_base_url.rstrip("/") + "/" + name})
        for digest, _ in group:
            for entry in entries:
                if entry.get("released_sha256") == digest and entry["kind"] == "pt":
                    entry["asset"] = name
        print(f"Built {name}: {destination.stat().st_size / 2**20:.1f} MiB", flush=True)
    index = {"schema_version": 1,
             "description": selection["description"],
             "tensor_loading": "torch.load(path, map_location='cpu', weights_only=True)",
             "counts": {"files": len(entries), "jsonl_rows": row_total,
                        "unique_tensors": len(tensors)},
             "historical_nonfinite_reports": nonfinite_reports,
             "assets": parts, "files": entries}
    (args.output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    referenced = {e["blob"] for e in entries if "blob" in e}
    for blob in blobs.glob("*"):
        if blob.relative_to(args.output).as_posix() not in referenced:
            blob.unlink()
    (args.output / "MANIFEST.sha256").write_text("".join(
        sha(p.read_bytes()) + "  " + p.relative_to(args.output).as_posix() + "\n"
        for p in sorted(args.output.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.sha256"))
    print(json.dumps(index["counts"]))


if __name__ == "__main__":
    main()
