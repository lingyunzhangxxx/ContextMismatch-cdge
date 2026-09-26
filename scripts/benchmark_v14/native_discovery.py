"""Deterministic model-native site and hyperparameter discovery for C-DGE.

This module deliberately contains no model-specific layer constants.  A model
contract determines the valid layer range; a frozen discovery protocol
determines the scan and ranking rules; and every emitted candidate is bound to
the site evidence by SHA256 before any direct behavior evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from scripts.benchmark_v1.common import sha256_file


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def value_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def item_fold(item_id: str, folds: int = 8) -> int:
    if folds < 2:
        raise ValueError("fold count must be at least two")
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def site_key(layer: int, component: str) -> str:
    if component not in {"residual", "self_attn", "mlp"}:
        raise ValueError(f"unsupported component: {component}")
    return f"{int(layer)}:{component}"


def scan_key(
    mode: str, item: str, role: str, style: str, swap: int,
    source: str, layer: int, component: str, coefficient: float,
) -> str:
    return "__".join(map(str, (
        "cdge42", mode, item, role, style, swap, source,
        layer, component, coefficient,
    )))


def select_scan_items(
    manifest: list[dict], protocol: dict, *, mode: str = "residual"
) -> list[dict]:
    rule = protocol["native_discovery"]
    allowed = set(int(value) for value in rule["site_scan_folds"])
    count = 1 if mode == "smoke" else int(rule["items_per_benchmark"])
    result = []
    for benchmark in sorted({row["benchmark"] for row in manifest}):
        eligible = sorted(
            (
                row for row in manifest
                if row["benchmark"] == benchmark
                and row["partition"] == rule["partition"]
                and item_fold(row["item_id"]) in allowed
            ),
            key=lambda row: row["item_id"],
        )
        if len(eligible) < count:
            raise ValueError(f"insufficient native discovery items for {benchmark}")
        result.extend(eligible[:count])
    return result


def load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    required = {
        "method_short_name": "C-DGE-V4.2",
        "status": "frozen_before_qwen3_5_native_discovery_forward",
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if protocol.get(field) != expected:
            raise ValueError(f"native discovery protocol mismatch: {field}")
    return protocol


def build_scan_plan(protocol: dict, model_contract: dict) -> dict:
    base = protocol["base_model"]
    layers = int(base["num_hidden_layers"])
    if layers != int(model_contract["architecture"]["num_hidden_layers"]):
        raise ValueError("model contract layer count differs from discovery protocol")
    if layers < 4:
        raise ValueError("model is too shallow for the frozen discovery design")
    eligible = list(range(layers))
    plan = {
        "schema_version": 1,
        "stage": "qwen35_cdge_v4_2_native_site_scan_plan",
        "model_revision": base["revision"],
        "num_hidden_layers": layers,
        "residual_layers": eligible,
        "component_candidates": ["self_attn", "mlp"],
        "residual_top_k": int(protocol["native_discovery"]["residual_top_k"]),
        "final_site_top_k": int(protocol["native_discovery"]["final_site_top_k"]),
        "selection_partition": protocol["native_discovery"]["partition"],
        "selection_folds": list(protocol["native_discovery"]["site_scan_folds"]),
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    plan["plan_sha256"] = value_sha256(plan)
    return plan


def smoke_scan_axes(protocol: dict, plan: dict) -> tuple[list[int], list[str]]:
    """Return the engineering-only endpoints and hook types for a short smoke."""
    last_layer = int(plan["num_hidden_layers"]) - 1
    if last_layer < 1:
        raise ValueError("native smoke requires at least two model layers")
    return [0, last_layer], [
        "residual", *list(protocol["native_discovery"]["components"])
    ]


def scan_factorial_axes(protocol: dict, mode: str) -> dict[str, list]:
    """Return full scientific axes or the deliberately minimal smoke axes."""
    factorial = protocol["native_discovery"]["factorial"]
    axes = {
        "declared_roles": list(factorial["declared_roles"]),
        "history_styles": list(factorial["history_styles"]),
        "label_swaps": list(factorial["label_swaps"]),
        "source_regimes": list(factorial["source_regimes"]),
    }
    if mode == "smoke":
        axes["declared_roles"] = axes["declared_roles"][:1]
        axes["history_styles"] = axes["history_styles"][:1]
        axes["label_swaps"] = axes["label_swaps"][:1]
    return axes


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty cell")
    return sum(values) / len(values)


def _score_rows(rows: list[dict], *, layer: int, component: str) -> dict:
    selected = [
        row for row in rows
        if int(row["layer"]) == int(layer)
        and row["component"] == component
        and float(row["patch_coefficient"]) == 1.0
    ]
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in selected:
        grouped[(row["benchmark"], row["source_regime"], int(row["label_swap"]))].append(row)
    if not grouped:
        raise ValueError(f"site has no coefficient-one evidence: {site_key(layer, component)}")
    cells = []
    for (benchmark, direction, label_swap), values in sorted(grouped.items()):
        effect = _mean(float(row["patch_effect"]) for row in values)
        gap = _mean(float(row["target_gap"]) for row in values)
        ratio = abs(effect / gap) if gap else 0.0
        expected_sign = effect > 0 if direction == "obedience" else effect < 0
        cells.append({
            "benchmark": benchmark,
            "source_regime": direction,
            "label_swap": label_swap,
            "mean_patch_effect": effect,
            "mean_target_gap": gap,
            "absolute_mediation_ratio": ratio,
            "ranking_ratio_clipped_0_1": min(1.0, max(0.0, ratio)),
            "expected_direction": expected_sign,
        })
    return {
        "layer": int(layer),
        "component": component,
        "site_key": site_key(layer, component),
        "ranking_score": _mean(cell["ranking_ratio_clipped_0_1"] for cell in cells),
        "expected_direction_fraction": _mean(float(cell["expected_direction"]) for cell in cells),
        "all_direction_label_cells_expected": all(cell["expected_direction"] for cell in cells),
        "cells": cells,
    }


def audit_scan_rows(rows: list[dict], expected: dict) -> dict:
    keys = [str(row["job_key"]) for row in rows]
    finite_fields = ("patch_effect", "target_gap", "source_cached_margin", "correct_logit_margin")
    nonfinite = sum(
        not all(math.isfinite(float(row[field])) for field in finite_fields)
        for row in rows
    )
    zero = [row for row in rows if float(row["patch_coefficient"]) == 0.0]
    zero_error = max(
        (abs(float(row["correct_logit_margin"]) - float(row["source_cached_margin"])) for row in zero),
        default=math.inf,
    )
    observed_sha = hashlib.sha256(
        "".join(f"{key}\n" for key in sorted(set(keys))).encode("utf-8")
    ).hexdigest()
    audit = {
        "row_count": len(rows),
        "unique_job_keys": len(set(keys)),
        "duplicate_job_keys": len(keys) - len(set(keys)),
        "nonfinite_rows": nonfinite,
        "coefficient_zero_rows": len(zero),
        "coefficient_zero_max_absolute_margin_error": zero_error,
        "observed_key_sha256": observed_sha,
        "expected_row_count": int(expected["row_count"]),
        "expected_key_sha256": expected["key_sha256"],
    }
    audit["success"] = (
        audit["row_count"] == audit["expected_row_count"]
        and audit["unique_job_keys"] == audit["row_count"]
        and nonfinite == 0
        and zero_error == 0.0
        and observed_sha == audit["expected_key_sha256"]
    )
    return audit


def select_native_sites(
    *, residual_rows: list[dict], component_rows: list[dict], protocol: dict
) -> dict:
    rule = protocol["native_discovery"]
    layer_count = int(protocol["base_model"]["num_hidden_layers"])
    residual_ranking = [
        _score_rows(residual_rows, layer=layer, component="residual")
        for layer in range(layer_count)
    ]
    residual_ranking.sort(key=lambda row: (-row["ranking_score"], row["layer"]))
    nominated_layers = [
        row["layer"] for row in residual_ranking[: int(rule["residual_top_k"])]
    ]
    component_ranking = []
    for layer in nominated_layers:
        for component in rule["components"]:
            component_ranking.append(
                _score_rows(component_rows, layer=layer, component=component)
            )
    tie_order = {name: index for index, name in enumerate(rule["component_tie_order"])}
    component_ranking.sort(
        key=lambda row: (
            -row["ranking_score"],
            -row["expected_direction_fraction"],
            row["layer"],
            tie_order[row["component"]],
        )
    )
    selected = component_ranking[: int(rule["final_site_top_k"])]
    result = {
        "schema_version": 1,
        "stage": "qwen35_cdge_v4_2_native_site_selection",
        "method": "C-DGE-V4.2",
        "selection_partition": rule["partition"],
        "selection_folds": list(rule["site_scan_folds"]),
        "nominated_layers": nominated_layers,
        "selected_sites_ordered": [
            {key: row[key] for key in (
                "layer", "component", "site_key", "ranking_score",
                "expected_direction_fraction", "all_direction_label_cells_expected",
            )}
            for row in selected
        ],
        "residual_ranking": residual_ranking,
        "component_ranking": component_ranking,
        "selection_rule": rule["ranking_rule"],
        "locked": True,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    result["selection_sha256"] = value_sha256(result)
    return result


def materialize_candidate_grid(site_manifest: dict, protocol: dict) -> dict:
    if site_manifest.get("locked") is not True:
        raise ValueError("site manifest is not locked")
    if site_manifest.get("operator_dev_accessed") is not False:
        raise ValueError("site discovery accessed operator_dev")
    candidates = []
    grid = protocol["native_discovery"]["candidate_grid"]
    for site in site_manifest["selected_sites_ordered"]:
        for profile in grid["rank_profiles"]:
            for cap in grid["maximum_relative_corrections"]:
                config = {
                    "layer": int(site["layer"]),
                    "component": site["component"],
                    "boundary_rank": int(profile["boundary_rank"]),
                    "context_rank": int(profile["context_rank"]),
                    "positive_output_rank": int(profile["positive_output_rank"]),
                    "negative_output_rank": int(profile["negative_output_rank"]),
                    "protected_rank": int(grid["protected_rank"]),
                    "max_relative_correction": float(cap),
                    "rank_profile": profile["name"],
                }
                candidates.append({
                    "candidate_id": "CDGE42-" + value_sha256(config)[:16],
                    "config": config,
                })
    candidates.sort(key=lambda row: row["candidate_id"])
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise RuntimeError("candidate ID collision")
    manifest = {
        "schema_version": 1,
        "stage": "qwen35_cdge_v4_2_native_candidate_grid",
        "method": "C-DGE-V4.2",
        "site_manifest_sha256": value_sha256(site_manifest),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "fit_split": protocol["native_discovery"]["fit_split"],
        "behavior_selection_rule": protocol["native_discovery"]["behavior_selection_rule"],
        "selection_performed": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    manifest["candidate_grid_sha256"] = value_sha256(manifest)
    return manifest


def validate_checkpoint_against_candidate(
    checkpoint: dict, candidate: dict, *, candidate_grid_sha256: str
) -> None:
    if checkpoint.get("base_model_weights_included") is not False:
        raise ValueError("checkpoint contains or does not exclude base weights")
    if checkpoint.get("candidate_grid_sha256") != candidate_grid_sha256:
        raise ValueError("checkpoint candidate-grid binding mismatch")
    sites = checkpoint.get("sites", [])
    if len(sites) != 1:
        raise ValueError("native candidate checkpoint must contain exactly one site")
    site = sites[0]
    config = candidate["config"]
    fields = (
        "layer", "component", "boundary_rank", "context_rank",
        "positive_output_rank", "negative_output_rank",
    )
    for field in fields:
        if site.get(field) != config[field]:
            raise ValueError(f"dynamic checkpoint/site mismatch: {field}")
    if float(site.get("maximum_relative_correction", math.nan)) != float(
        config["max_relative_correction"]
    ):
        raise ValueError("dynamic checkpoint correction cap mismatch")
    forbidden_prefixes = ("model.", "transformer.", "base_model.", "lm_head.")
    for key in checkpoint.get("state_dict", {}):
        if key.startswith(forbidden_prefixes):
            raise ValueError(f"checkpoint includes a base-weight key: {key}")


def require_sha_bound_path(record: dict, field: str, path: Path) -> None:
    source = record.get(field, {})
    if source.get("path") != str(path) or source.get("sha256") != sha256_file(path):
        raise ValueError(f"SHA-bound artifact mismatch: {field}")
