from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "protocol" / "BENCHMARK_CONTRACT_V1.json"
PARTITIONS = (
    "component_discovery",
    "subspace_fit",
    "operator_dev",
    "final_test",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hex(*parts: object) -> str:
    joined = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def stable_int(*parts: object) -> int:
    return int(stable_hex(*parts)[:16], 16)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def deterministic_take(rows: Iterable[dict], count: int, seed: int, namespace: str) -> list[dict]:
    ranked = sorted(
        rows,
        key=lambda row: stable_hex(seed, namespace, row["native_id"]),
    )
    if len(ranked) < count:
        raise ValueError(f"{namespace}: requested {count} items but only {len(ranked)} eligible")
    return ranked[:count]


def assign_partitions(rows: list[dict], seed: int, benchmark: str) -> list[dict]:
    if len(rows) % len(PARTITIONS):
        raise ValueError(f"{benchmark}: row count {len(rows)} is not divisible by four")
    ranked = sorted(rows, key=lambda row: stable_hex(seed, benchmark, "partition", row["native_id"]))
    size = len(ranked) // len(PARTITIONS)
    for partition_index, partition in enumerate(PARTITIONS):
        for row in ranked[partition_index * size : (partition_index + 1) * size]:
            row["partition"] = partition
            row["history_realization"] = stable_int(row["item_id"], "history-realization") % 3
    return ranked


def expected_full_mechanism_rows(
    contract: dict,
    mode: str,
    items_per_benchmark: int,
) -> int:
    """Derive the full mechanism row count from the scientific factorial."""
    if mode not in {"residual", "component"}:
        raise ValueError(f"unsupported mechanism mode: {mode}")
    factorial = contract["factorial"]
    rows = (
        len(contract["data_split"]["benchmarks"])
        * items_per_benchmark
        * len(factorial["declared_roles"])
        * len(factorial["history_styles"])
        * len(factorial["label_swaps"])
        * len(factorial["source_regimes"])
    )
    if mode == "residual":
        return (
            rows
            * len(contract["residual_scan"]["layers"])
            * len(contract["residual_scan"]["coefficients"])
        )
    return (
        rows
        * contract["residual_scan"]["component_nomination_rule"]["select_top_k"]
        * len(contract["component_scan"]["components"])
        * len(contract["component_scan"]["coefficients"])
    )


def effective_mechanism_items_per_benchmark(
    contract: dict,
    erratum: dict,
    contract_sha256: str,
) -> int:
    """Validate the zero-row arithmetic erratum and return its effective sample size."""
    if erratum.get("production_rollout_approved") is not False:
        raise ValueError("mechanism erratum changes the production boundary")
    if erratum.get("mechanism_contract_sha256") != contract_sha256:
        raise ValueError("mechanism erratum contract SHA mismatch")
    if erratum.get("affected_field") != "data_split.items_per_benchmark":
        raise ValueError("mechanism erratum changes an unsupported field")
    if not erratum.get("audit_finding", {}).get("detected_before_affected_forward"):
        raise ValueError("mechanism erratum was not frozen before affected forward execution")
    if erratum.get("audit_finding", {}).get("failed_job_scientific_rows") != 0:
        raise ValueError("mechanism erratum is not a zero-scientific-row correction")
    replacement = erratum.get("replacement", {})
    items_per_benchmark = replacement.get("items_per_benchmark")
    if not isinstance(items_per_benchmark, int) or items_per_benchmark <= 0:
        raise ValueError("invalid effective mechanism item count")
    if items_per_benchmark > contract["data_split"]["items_per_benchmark"]:
        raise ValueError("mechanism erratum expands the original item ceiling")
    expected = {
        "residual": contract["residual_scan"]["expected_full_rows"],
        "component": contract["component_scan"]["expected_full_rows"],
    }
    replacement_expected = {
        "residual": replacement.get("residual_expected_full_rows"),
        "component": replacement.get("component_expected_full_rows"),
    }
    for mode in ("residual", "component"):
        derived = expected_full_mechanism_rows(contract, mode, items_per_benchmark)
        if derived != expected[mode] or derived != replacement_expected[mode]:
            raise ValueError(
                f"mechanism erratum {mode} row mismatch: "
                f"derived={derived} contract={expected[mode]} replacement={replacement_expected[mode]}"
            )
    return items_per_benchmark


def atomic_write_text(path: Path, text: str, allow_overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.incoming")
    if temporary.exists():
        raise FileExistsError(f"preserving interrupted incoming artifact: {temporary}")
    temporary.write_text(text)
    temporary.replace(path)
