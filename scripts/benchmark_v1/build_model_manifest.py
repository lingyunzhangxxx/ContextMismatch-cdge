#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .common import atomic_write_text, canonical_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    template = json.loads(args.template.read_text())
    config = json.loads((args.model_dir / "config.json").read_text())
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())
    if config.get("architectures") != [template["architecture"]]:
        raise ValueError(f"architecture mismatch: {config.get('architectures')}")
    for key in ("hidden_size", "num_hidden_layers", "max_position_embeddings"):
        if int(config[key]) != int(template[key]):
            raise ValueError(f"config mismatch for {key}: {config[key]} != {template[key]}")
    if config.get("quantization_config") is not None:
        raise ValueError("quantized checkpoint is forbidden")
    weight_files = sorted(set(index["weight_map"].values()))
    if len(weight_files) != template["weight_shards_expected"]:
        raise ValueError(f"expected {template['weight_shards_expected']} weight shards, found {len(weight_files)}")
    files = []
    for path in sorted(value for value in args.model_dir.rglob("*") if value.is_file()):
        relative = path.relative_to(args.model_dir).as_posix()
        if ".cache/" in relative or relative.startswith(".cache/"):
            continue
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    paths = {entry["path"] for entry in files}
    missing_weights = sorted(set(weight_files) - paths)
    if missing_weights:
        raise FileNotFoundError(f"weight shards absent from manifest: {missing_weights}")
    tree_digest = hashlib.sha256()
    for entry in files:
        tree_digest.update((canonical_json(entry) + "\n").encode())
    result = dict(template)
    result.update(
        {
            "source_directory": str(args.model_dir.resolve()),
            "files": files,
            "file_count": len(files),
            "total_bytes": sum(entry["bytes"] for entry in files),
            "weight_bytes_from_index": int(index.get("metadata", {}).get("total_size", 0)),
            "tree_sha256": tree_digest.hexdigest(),
            "verified": True,
        }
    )
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n", args.allow_overwrite)
    print(json.dumps({key: result[key] for key in ("contract_id", "file_count", "total_bytes", "weight_bytes_from_index", "tree_sha256", "verified")}, indent=2))


if __name__ == "__main__":
    main()
