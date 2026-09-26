"""Dependency-free validation helpers for protected-state capture."""

from __future__ import annotations

from pathlib import Path

from .common import sha256_file


def validate_mitigation_controls_binding(
    *,
    execution: dict,
    execution_contract_path: Path,
    controls_path: Path,
    authorization: dict | None,
    router_contract: dict | None,
) -> None:
    """Validate legacy controls or the separately frozen V5 audit controls.

    The operator execution contract predates the V5 developmental audit and is
    intentionally bound to mitigation_controls.json. A V5 recovery must not
    rewrite that immutable contract. Instead, its fresh execution
    authorization and frozen V5 router contract independently bind the new
    controls. Every non-V5 caller retains the original exact-SHA check.
    """

    controls_sha = sha256_file(controls_path)
    if authorization is None:
        if execution["mitigation_controls_sha256"] != controls_sha:
            raise ValueError("execution/mitigation-controls mismatch")
        return
    if router_contract is None:
        raise ValueError("V5 controls validation requires the router contract")
    if authorization.get("execution_contract_sha256") != sha256_file(
        execution_contract_path
    ):
        raise ValueError("V5 authorization/execution-contract mismatch")
    if authorization.get("v5_audit_controls_sha256") != controls_sha:
        raise ValueError("V5 authorization/mitigation-controls mismatch")
    if (
        router_contract.get("bound_static_inputs", {}).get(
            "v5_audit_controls_sha256"
        )
        != controls_sha
    ):
        raise ValueError("V5 router/mitigation-controls mismatch")
