#!/usr/bin/env python3
"""Materialize a one-use, SHA-bound node-local runtime restore authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


PROJECT_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
CODE_ROOT = PROJECT_ROOT / "code-v45"
RUNTIME_ARCHIVE = PROJECT_ROOT / "runtime-archives/ascend-runtime-v1-code-v45.tar.gz"
EXPECTED_BASE_SHA256 = "af7b5881a1f73c30f710661ab4934559befbf45c5f22d7ae65e44531d8c98555"
EXPECTED_OVERLAY_SHA256 = "9e0ce3ebe2054e56b1bb28ce8f24448a794019aef991ec44752c5d2f0a479bd9"
EXPECTED_PYTHON_ARCHIVE_SHA256 = "71caba13c43177b4b14b9c7fabad807904f70578710357c5d885704f29ac932a"
EXPECTED_SELECTION_CONTRACT = {
    "partition": "a01",
    "npu_type": "910B3",
    "npus": 3,
    "cpus": 24,
    "mem_mib": 393216,
    "allow_nodes": [],
    "exclude_nodes": [],
}
BLOCKED_NODE_STATES = {
    "DOWN", "DRAIN", "DRAINED", "FAIL", "FAILING", "FUTURE", "INVAL",
    "MAINT", "NO_RESPOND", "PLANNED", "POWER_DOWN", "POWERING_DOWN",
    "REBOOT_REQUESTED", "RESERVED", "UNKNOWN",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    if not str(path).startswith(f"{PROJECT_ROOT}/"):
        raise ValueError("bound artifact escaped the project root")
    return {"path": str(path), "sha256": sha256_file(path)}


def node_states(value: str) -> set[str]:
    return {token for token in re.split(r"[+~#$*!%-]+", value.upper()) if token}


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--selector-snapshot", type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if args.code_root != CODE_ROOT:
        raise ValueError("runtime restore requires immutable code-v45")
    if not re.fullmatch(r"a[0-9]{2}", args.selected_node):
        raise ValueError("invalid selected node")
    bundle = args.code_root / "bundle.sha256"
    if not bundle.is_file() or not RUNTIME_ARCHIVE.is_file():
        raise FileNotFoundError("immutable bundle or runtime archive is missing")

    selection = json.loads(args.selector_snapshot.read_text())
    if selection.get("node") != args.selected_node:
        raise ValueError("selector snapshot/node mismatch")
    if selection.get("selection_contract") != EXPECTED_SELECTION_CONTRACT:
        raise ValueError("selector contract mismatch")
    if node_states(str(selection.get("state", ""))) & BLOCKED_NODE_STATES:
        raise ValueError("selector snapshot contains a blocked node state")
    if int(selection.get("npu_free", -1)) < 3 or int(selection.get("cpu_free", -1)) < 24:
        raise ValueError("selector snapshot lacks aggregate compute capacity")
    if int(selection.get("mem_free_mib", -1)) < 393216:
        raise ValueError("selector snapshot lacks aggregate memory capacity")

    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-8b-runtime-restore-code-v45-{args.created_utc}",
        "created_utc": args.created_utc,
        "stage": "node_local_runtime_restore",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(bundle),
        "execution_allowed": True,
        "execution_node": args.selected_node,
        "slurm_partition": "a01",
        "resource_contract": {
            "partition": "a01",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 8,
            "mem_mib": 65536,
            "npus": 0,
            "time_limit": "02:00:00",
            "node": args.selected_node,
        },
        "future_experiment_resource_contract": EXPECTED_SELECTION_CONTRACT,
        "node_selection_snapshot": record(args.selector_snapshot),
        "runtime_archive": record(RUNTIME_ARCHIVE),
        "runtime_content": {
            "base_tree_sha256": EXPECTED_BASE_SHA256,
            "overlay_tree_sha256": EXPECTED_OVERLAY_SHA256,
            "python_archive_sha256": EXPECTED_PYTHON_ARCHIVE_SHA256,
        },
        "target_runtime_root": "/workspace/node-local/context-mismatch-ascend/runtime-v1",
        "scientific_forward_allowed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
