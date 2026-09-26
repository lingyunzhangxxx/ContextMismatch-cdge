# Conditional low-rank negative operator v3

## Motivation and claim boundary

The v2 one-sided projection detects an obedience-like feature and removes it in
the same basis. That is deliberately conservative, but exact paired patching
suggests the useful intervention may be a transport from a history-regime
feature into a different late decision direction. Version 3 therefore
separates the **trigger subspace** from the **correction subspace**. It remains a
reversible activation hook; no base checkpoint is modified.

## Operator

For component output `z`, centered trigger basis `V`, protected correction
basis `U* = orth((I - P P^T)U)`, threshold `tau`, and a small transport matrix
`A`, define

`r(z) = relu(V^T(z-mu) - tau)`

`delta(z) = U* A^T r(z)`

`z' = z - q(H,x) clip_norm(delta(z), rho ||z||_2)`.

The mismatch gate remains

`q(H,x) = p(obedience-state | h_boundary) * I[independent verification is required]`.

In the controlled benchmark, the second factor is supplied by the task rule;
it is not inferred from the answer label. The relative trust-region radius
`rho` prevents a low-norm token from receiving an arbitrarily large edit.

V2 is recovered by setting `V=U*`, diagonal non-negative `A`, and disabling the
trust-region clip. A signed, non-diagonal `A` is more expressive: it can detect
several governance features and map them into causally nominated decision
directions without erasing those features globally.

## Fitting without test leakage

1. `component_discovery`: nominate layers/components using exact paired
   residual and component patching only.
2. `subspace_fit`: fit `V` from obedience-versus-verification boundary/component
   inputs. Fit `U` from paired verification-minus-obedience causal targets or
   direct-logit-weighted activation differences. Fit protected `P` from fresh,
   matched verification, supported authority, and factual-memory controls.
3. Fit `A` by ridge reduced-rank regression from harmful-side trigger scores to
   paired causal corrections. No coefficient is selected here.
4. `operator_dev`: select rank, thresholds, gate calibration, trust radius,
   layer set, and regularization using the frozen collateral-constrained Pareto
   rule.
5. Lock one operator manifest and open `final_test` once.

## Required nested comparison

The development table retains all earlier baselines and adds capacity one step
at a time:

1. no edit;
2. fixed translation (known sign-balance failure);
3. symmetric rank-one projection;
4. one-sided rank-one projection;
5. protected one-sided projection (v2);
6. protected and mismatch-gated low-rank projection;
7. v3 diagonal transport with distinct trigger/output bases;
8. v3 full low-rank transport;
9. v3 full transport plus trust-region budget;
10. jointly regularized multi-layer v3.

The extra expressivity is accepted only if mismatch recovery improves on v2
while every protected-control upper confidence bound stays within its frozen
tolerance. Larger raw intervention effects are not, by themselves, evidence of
better mitigation.

## Mechanistic interpretation

A successful v3 operator would support a fixed circuit that first detects a
carried governance state and then reads it through a distinct task-decision
subspace. It would not show that conversation changes model parameters, that
the subspace is unique, or that context mismatch is unavoidable. Failure of v3
while exact patching succeeds would instead imply that the causal map is more
nonlinear, token-distributed, or stateful than a low-rank local transport.
