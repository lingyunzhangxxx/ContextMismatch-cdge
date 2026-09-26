from __future__ import annotations

import hashlib
import itertools
import json


FAMILIES = (
    "fixed_translation",
    "symmetric_rank1",
    "v2_one_sided",
    "v3_full",
    "v4_bilinear",
    "v5_signed",
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def candidate_id(config: dict) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()[:16]


def pair_fold(pair_key: str, folds: int = 4) -> int:
    if folds < 2:
        raise ValueError("fold count must be at least two")
    digest = hashlib.sha256(pair_key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _config(
    *,
    family: str,
    site_key: str,
    history_rank: int,
    context_rank: int,
    correction_rank: int,
    protected_rank: int,
    ridge_penalty: float,
    global_trust_radius: float,
    deadzone: float,
    token_scope: str,
    projection_strength: float,
) -> dict:
    return {
        "family": family,
        "site_keys": [site_key],
        "history_rank": int(history_rank),
        "context_rank": int(context_rank),
        "correction_rank": int(correction_rank),
        "protected_rank": int(protected_rank),
        "ridge_penalty": float(ridge_penalty),
        "global_trust_radius": float(global_trust_radius),
        "deadzone": float(deadzone),
        "token_scope": token_scope,
        "projection_strength": float(projection_strength),
    }


def enumerate_grid(operator_contract: dict, extension: dict, site_manifest: dict) -> list[dict]:
    if extension.get("production_rollout_approved") is not False:
        raise ValueError("production boundary changed")
    if not site_manifest.get("locked"):
        raise ValueError("operator site manifest is not locked")
    sites = [
        f"{int(row['layer'])}:{row['component']}"
        for row in site_manifest["selected_sites_ordered"]
    ]
    if len(sites) != 3 or len(sites) != len(set(sites)):
        raise ValueError("the initial grid requires exactly three unique locked sites")
    fit = operator_contract["subspace_fit"]
    history_ranks = [int(value) for value in fit["history_ranks"]]
    context_ranks = [int(value) for value in fit["task_context_ranks"]]
    correction_ranks = [int(value) for value in fit["correction_ranks"]]
    protected_ranks = [int(value) for value in fit["protected_ranks"]]
    ridge_penalties = [float(value) for value in fit["ridge_penalties"]]
    radii = [float(value) for value in extension["global_trust_budget"]["radii"]]
    deadzones = [float(value) for value in extension["alignment_gate"]["deadzones"]]
    strengths = [float(value) for value in extension["compute_schedule"]["projection_strengths"]]
    minimum_ridge = min(ridge_penalties)
    rows = []
    for site in sites:
        for family in ("fixed_translation", "symmetric_rank1"):
            for protected_rank, strength, radius in itertools.product(
                protected_ranks, strengths, radii
            ):
                rows.append(
                    _config(
                        family=family,
                        site_key=site,
                        history_rank=1,
                        context_rank=0,
                        correction_rank=1,
                        protected_rank=protected_rank,
                        ridge_penalty=minimum_ridge,
                        global_trust_radius=radius,
                        deadzone=0.0,
                        token_scope="last_token",
                        projection_strength=strength,
                    )
                )
        for correction_rank, protected_rank, strength, radius in itertools.product(
            [value for value in correction_ranks if value in {1, 4, 8}],
            protected_ranks,
            strengths,
            radii,
        ):
            rows.append(
                _config(
                    family="v2_one_sided",
                    site_key=site,
                    history_rank=1,
                    context_rank=0,
                    correction_rank=correction_rank,
                    protected_rank=protected_rank,
                    ridge_penalty=minimum_ridge,
                    global_trust_radius=radius,
                    deadzone=0.0,
                    token_scope="last_token",
                    projection_strength=strength,
                )
            )
        for history_rank, correction_rank, protected_rank, ridge, radius in itertools.product(
            [value for value in history_ranks if value in {2, 4, 8}],
            [value for value in correction_ranks if value in {2, 4, 8}],
            protected_ranks,
            ridge_penalties,
            radii,
        ):
            rows.append(
                _config(
                    family="v3_full",
                    site_key=site,
                    history_rank=history_rank,
                    context_rank=0,
                    correction_rank=correction_rank,
                    protected_rank=protected_rank,
                    ridge_penalty=ridge,
                    global_trust_radius=radius,
                    deadzone=0.0,
                    token_scope="last_token",
                    projection_strength=1.0,
                )
            )
        for family in ("v4_bilinear", "v5_signed"):
            family_deadzones = deadzones if family == "v5_signed" else [0.0]
            for values in itertools.product(
                [value for value in history_ranks if value in {2, 4}],
                [value for value in context_ranks if value in {1, 2, 4}],
                [value for value in correction_ranks if value in {4, 8}],
                protected_ranks,
                ridge_penalties,
                radii,
                family_deadzones,
                ["last_token", "all_suffix"],
            ):
                history_rank, context_rank, correction_rank, protected_rank, ridge, radius, deadzone, scope = values
                rows.append(
                    _config(
                        family=family,
                        site_key=site,
                        history_rank=history_rank,
                        context_rank=context_rank,
                        correction_rank=correction_rank,
                        protected_rank=protected_rank,
                        ridge_penalty=ridge,
                        global_trust_radius=radius,
                        deadzone=deadzone,
                        token_scope=scope,
                        projection_strength=1.0,
                    )
                )
    unique = {canonical_json(config): config for config in rows}
    if len(unique) != len(rows):
        raise RuntimeError("candidate grid contains duplicate configurations")
    return [unique[key] for key in sorted(unique)]
