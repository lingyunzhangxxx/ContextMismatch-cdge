# Task-conditioned negative operator v4

## Why v4 is a necessary nested extension

The fixed translation baseline is structurally unable to repair both A/B label
swaps. V2 improves selectivity by making subtraction one-sided, protected, and
mismatch-gated. V3 separates the history trigger subspace from the correction
subspace, but its transport matrix is still the same for every new task. A
single global map can therefore underfit a governance state whose harmful
readout changes with the current question, candidate order, or benchmark.

V4 adds only one capacity axis: a bounded code of the current-task hidden state
conditions the low-rank transport. It never receives the correct answer, the
correct label, benchmark identity, or a post-hoc correctness signal at
inference time.

## Operator

Let `h_b` be a residual captured at the task boundary, `z_t` the current-token
component output, `H` a history-trigger basis, `C` a task-context basis, and
`U* = orth((I - P P^T)U)` a protected correction basis. Define

`r = relu(H^T(h_b - mu_h) - tau)`

`c_t = tanh(diag(s)^-1 C^T(z_t - mu_c))`

`phi_t = [r ; vec(r c_t^T)]`

`delta_t = U* T^T phi_t`

`z'_t = z_t - q(H,x) clip_norm(delta_t, rho ||z_t||_2)`.

The primary implementation edits only the final decision token at causally
nominated components. An all-suffix-token variant is an explicit ablation. V3
is recovered when `C` has rank zero. The bounded `tanh` context code and relative
trust region prevent an outlying task activation from producing an unbounded
edit.

## Fitting and split discipline

1. Residual and component layers are selected using only
   `component_discovery`.
2. `H`, `C`, `U`, protected basis `P`, and ridge transport `T` are fitted only
   on `subspace_fit`. Candidate labels are balanced and never supplied as
   features.
3. Rank, ridge penalty, threshold, trust radius, token scope, layer set, and
   gate calibration are selected only on `operator_dev`.
4. Selection uses leave-one-benchmark-out development estimates and a frozen
   lexicographic Pareto rule: reject collateral violations, maximize mismatch
   recovery, then prefer lower rank and smaller intervention norm.
5. One manifest is hashed and locked before `final_test` is opened exactly once.

## Required comparison

The final development table must include no edit, fixed translation, symmetric
rank-one projection, v2 one-sided protected projection, v3 diagonal transport,
v3 full transport, v3 plus trust region, v4 without interaction features, v4
bilinear transport, and multi-layer v4. V4 is supported only if its held-out
recovery improves on the best admissible v3 while fresh, verification,
supported-authority, factual-memory, and explicit-reset controls remain within
their frozen collateral tolerances.

## Interpretation boundary

A successful v4 result would show that mismatch can be reduced by reading a
history-side governance feature and mapping it through a distinct,
task-conditioned low-rank correction. It would not show that the subspace is
unique, that model parameters changed during conversation, or that a permanent
weight edit is safe. The base BF16 checkpoint remains immutable; all claim runs
use removable forward hooks.
