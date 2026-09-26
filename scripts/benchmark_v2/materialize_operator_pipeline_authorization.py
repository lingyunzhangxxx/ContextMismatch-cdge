#!/usr/bin/env python3
"""Create a SHA-bound authorization for one pre-final operator pipeline stage."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


SHA256_RE = re.compile(r"[0-9a-f]{64}")
REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b"


def _sha(value: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("invalid SHA256")
    return value


def _remote_artifact(value: str) -> dict[str, str]:
    path, separator, digest = value.rpartition("=")
    if not separator or not path.startswith(f"{REMOTE_ROOT}/"):
        raise argparse.ArgumentTypeError(f"artifact must be {REMOTE_ROOT}/...=SHA256")
    return {"path": path, "sha256": _sha(digest)}


def _source(value: str) -> tuple[str, dict[str, str]]:
    name, separator, artifact = value.partition("=")
    if not separator or not re.fullmatch(r"[a-z_]+", name):
        raise argparse.ArgumentTypeError("source must be NAME=/remote/path=SHA256")
    return name, _remote_artifact(artifact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        required=True,
        choices=("subspace_capture", "protected_capture", "fit_prefilter"),
    )
    parser.add_argument("--code-version", required=True, type=int)
    parser.add_argument("--bundle-manifest-sha256", required=True, type=_sha)
    parser.add_argument("--mechanism-contract", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--operator-erratum", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--component-manifest", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[], type=_source)
    parser.add_argument("--prerequisite", action="append", default=[], type=_remote_artifact)
    parser.add_argument("--created-utc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.code_version < 18:
        raise ValueError("operator pipeline code version must be at least 18")
    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    extension = json.loads(args.extension_contract.read_text())
    if extension.get("status") != "frozen_before_operator_forward":
        raise ValueError("operator extension is not frozen")
    if extension.get("production_rollout_approved") is not False:
        raise ValueError("production boundary changed")
    source_values = dict(args.source)
    if len(source_values) != len(args.source):
        raise ValueError("duplicate source name")
    required_sources = {
        "subspace_capture": {"behavior_analysis", "model_manifest"},
        "protected_capture": {"behavior_analysis", "model_manifest"},
        "fit_prefilter": {"subspace_capture_manifest", "protected_capture_manifest"},
    }[args.stage]
    if set(source_values) != required_sources:
        raise ValueError(
            f"{args.stage} sources are {sorted(source_values)}, expected {sorted(required_sources)}"
        )
    created_utc = args.created_utc or (
        dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", created_utc):
        raise ValueError("created UTC must use YYYY-MM-DDTHH:MM:SSZ")
    authorization = {
        "schema_version": 1,
        "authorization_id": (
            f"context-mismatch-qwen3-8b-operator-{args.stage.replace('_', '-')}-"
            f"code-v{args.code_version}"
        ),
        "created_utc": created_utc,
        "stage": args.stage,
        "code_root": f"{REMOTE_ROOT}/code-v{args.code_version}",
        "execution_allowed": True,
        "production_rollout_approved": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "immutable_code_bundle_manifest_sha256": args.bundle_manifest_sha256,
        "mechanism_contract_sha256": sha256_file(args.mechanism_contract),
        "operator_contract_sha256": sha256_file(args.operator_contract),
        "operator_erratum_sha256": sha256_file(args.operator_erratum),
        "execution_contract_sha256": sha256_file(args.execution_contract),
        "extension_contract_sha256": sha256_file(args.extension_contract),
        "component_manifest_sha256": sha256_file(args.component_manifest),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "mitigation_controls_sha256": sha256_file(args.controls),
        "sources": source_values,
        "prerequisite_artifacts": args.prerequisite,
    }
    atomic_write_text(args.output, json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    print(json.dumps(authorization, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
