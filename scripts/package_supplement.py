#!/usr/bin/env python3
"""Create a complete anonymous ZIP with fixed timestamps and no Git metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from scripts.evidence_records import read_index, verify
from scripts.privacy_audit import ROOT, files_to_scan, main as privacy_check
from scripts.public_demo import validate_results, verify_manifest, verify_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    args = parser.parse_args()
    verify_manifest()
    verify_sources()
    validate_results()
    privacy_check()
    index = read_index()
    verify(index)
    for asset in index["assets"]:
        path = args.asset_dir / asset["name"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError("Supplement tensor archive checksum mismatch")
        # The anonymous archive contains the assets themselves. Its index must
        # not link to a public account or repository that identifies authors.
        asset.pop("url", None)
    index_data = (json.dumps(index, indent=2) + "\n").encode()
    manifest = (ROOT / "records/MANIFEST.sha256").read_text()
    manifest = "\n".join(
        (hashlib.sha256(index_data).hexdigest() + "  index.json")
        if line.endswith("  index.json") else line
        for line in manifest.splitlines()) + "\n"
    overrides = {"records/index.json": index_data,
                 "records/MANIFEST.sha256": manifest.encode()}
    excluded = {".git", ".github", ".venv", "venv", "generated", "__pycache__", ".pytest_cache"}
    paths = {p.relative_to(ROOT).as_posix(): p for p in files_to_scan()
             if not excluded.intersection(p.relative_to(ROOT).parts)}
    for asset in index["assets"]:
        paths["tensor-assets/" + asset["name"]] = args.asset_dir / asset["name"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", allowZip64=True) as archive:
        for relative, path in sorted(paths.items()):
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (0o100755 if relative == "run.sh" else 0o100644) << 16
            info.compress_type = zipfile.ZIP_STORED if path.suffix == ".gz" else zipfile.ZIP_DEFLATED
            if relative in overrides:
                archive.writestr(info, overrides[relative])
            else:
                info.file_size = path.stat().st_size
                with archive.open(info, "w", force_zip64=True) as target, path.open("rb") as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
    print(f"Created complete anonymous supplement: {args.output} ({args.output.stat().st_size / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
