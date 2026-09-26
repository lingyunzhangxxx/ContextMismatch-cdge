# Selective negative-subspace operator v2

## Why the old vector is insufficient

A fixed translation at the final answer token moves the A-minus-B logit in one
global direction. Under label swap, repairing one half necessarily harms the
other. It also cannot distinguish harmful carryover from legitimate user
direction. The mitigation target is therefore not “obedience” in isolation; it
is the conjunction of an obedience-like history state and a current task that
requires independent verification.

## Operator family

Let `h_b` be the residual state at the task boundary, `z_l` the output of a
localized component at suffix layer `l`, `U_l` a rank-`r` harmful governance
subspace, and `P_l` a protected utility subspace. We first residualize the
harmful basis:

`U*_l = orth((I - P_l P_l^T) U_l)`.

The primary one-sided, context-gated edit is

`z'_l = z_l - q(H,x) U*_l diag(lambda_l) relu(U*_l^T(z_l-mu_l)-tau_l)`.

The mismatch gate is

`q(H,x) = p(obedience-state | h_b) * p(independent-verification-required | x)`.

This differs from an unconditional projection in three ways: it edits only the
harmful side of each component, removes directions used by protected controls
before intervention, and opens only when history governance conflicts with the
current task requirement.

## Nested ablations

The development comparison must include:

1. no edit;
2. rejected fixed translation;
3. symmetric rank-one projection;
4. one-sided rank-one projection;
5. protected one-sided projection;
6. protected and gated low-rank projection;
7. jointly regularized multi-layer operator;
8. boundary KV-value operator if exact Qwen3-8B cache patching nominates it.

Each additional mechanism has a corresponding ablation. Rank, threshold,
layer set, and strength are selected jointly on `operator_dev`; no choice may be
revisited after `final_test` is opened.

## Protected controls

The protected basis and the Pareto objective include fresh benchmark behavior,
verification-match behavior, supported correct user direction, tasks whose rule
explicitly assigns the user final authority, and factual memory carried across
the same boundary. The operator is allowed to reduce mismatch loss only inside
predeclared collateral tolerances. It is not successful if it merely makes the
model more resistant to all user instructions.

## Optimization objective

For development rows, minimize

`L = L_mismatch + w_kl KL(p_edit || p_base) + w_f L_fresh + w_v L_verify + w_s L_supported + w_m L_memory + w_g ||lambda||_1`.

The primary selection rule is lexicographic: first reject any candidate whose
upper 95% confidence bound exceeds a frozen collateral tolerance; among the
remaining candidates, maximize mean mismatch recovery; break ties using lower
rank and lower total intervention norm. The full Pareto frontier is retained.

## Parameter-level claim boundary

Hooks test causal necessity and mitigation selectivity. A hook can be
materialized as a reversible low-rank output adapter only after held-out success.
No base weight is overwritten. A successful adapter explains a fixed circuit
that reads a context-dependent state; it does not mean conversation history
changed the model parameters.
