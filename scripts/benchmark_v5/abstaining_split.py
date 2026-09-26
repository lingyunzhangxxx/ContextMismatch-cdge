"""Pure, output-blind split helpers for the ADSGE-V4 applicability fit."""

from __future__ import annotations

import hashlib


APPLICATION_FEATURE_WIDTH = 54


def canonical_protected_family(raw_family: str, contract: dict) -> str:
    crosswalk = contract["bound_fit_inputs"]["protected_capture_family_crosswalk"]
    if raw_family not in crosswalk:
        raise ValueError(f"unexpected protected capture family: {raw_family}")
    return str(crosswalk[raw_family])


def small_control_fold_overrides(
    metadata: list[dict], contract: dict
) -> dict[tuple[str, str], int]:
    inputs = contract["bound_fit_inputs"]
    expected_counts = {
        str(name): int(count)
        for name, count in inputs["small_control_identity_counts"].items()
    }
    sequence = [int(value) for value in contract["split"]["small_control_fold_sequence"]]
    if sequence != [0, 1, 2, 3, 6, 7]:
        raise ValueError("unexpected V4 small-control fold sequence")
    identities: dict[str, set[str]] = {name: set() for name in expected_counts}
    for row in metadata:
        control_id = row.get("control_id")
        if control_id is None:
            continue
        family = canonical_protected_family(str(row["control_family"]), contract)
        if family not in identities:
            raise ValueError(f"unexpected control-id protected family: {family}")
        identities[family].add(str(control_id))
    overrides: dict[tuple[str, str], int] = {}
    for family, expected in expected_counts.items():
        observed = identities[family]
        if len(observed) != expected or expected != len(sequence):
            raise ValueError(
                f"V4 small-control identity mismatch for {family}: "
                f"observed={len(observed)} expected={expected}"
            )
        ordered = sorted(
            observed,
            key=lambda value: hashlib.sha256(value.encode("utf-8")).digest(),
        )
        for control_id, fold in zip(ordered, sequence, strict=True):
            overrides[(family, control_id)] = fold
    return overrides


def exact_source_coverage(observed: set[str], expected: set[str]) -> bool:
    return bool(expected) and observed == expected
