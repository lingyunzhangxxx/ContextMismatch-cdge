# Negative-vector mitigation protocol

## Claim boundary

“Negative-vector pruning” can refer to activation steering, component-output
projection, neuron ablation, or a permanent weight edit. These are not
interchangeable. The study proceeds from reversible to persistent interventions
and promotes an intervention only after it passes the preceding causal and
selectivity checks.

The fully balanced final-token steering experiment has already rejected a fixed
context-independent translation as a useful mitigation: a sign-fixed change to
the `A-B` logit cannot recover both halves of a correct-label swap. Subsequent
work therefore uses exact paired patching for causal localization and
input-conditioned or low-rank hooks for mitigation. It does not enlarge `alpha`
to optimize an isolated smoke example.

## Split discipline

- Fit governance directions on five task domains and test on the sixth.
- Select layers, components, neurons, and steering strengths on development
  history realizations only.
- Evaluate once on held-out natural paraphrases and a held-out history style.
- Pair every edit run with the unedited model at identical labels, evidence
  order, task facts, and cache positions.

## Primary measures

- mismatch recovery: change in correct-vs-foil logit margin under obedience
  history;
- reverse induction: degradation caused by adding the direction to a matched
  verification history;
- collateral cost: change under fresh and verification histories;
- governance selectivity: preservation of correct, evidence-supported user
  instructions and appropriate assistant deference;
- factual-memory selectivity: retention of non-governance facts from the same
  pre-boundary context.

The concrete supported-authority and factual-boundary controls are frozen in
`protocol/mitigation_controls.json`; they are not selected after viewing an edit's
behavior.

## Parameter-level implementations

For a component output projection `W: R^m -> R^d` and a unit governance
direction `g` in residual space, the rank-one edit

`W' = (I - beta g g^T) W`

removes only the output component parallel to `g`. It is evaluated first without
materializing `W'`, using `z' = z - beta <z,g>g` in a forward hook. For an MLP
down projection, neuron `j` is eligible for pruning only when its paired
activation shift and its direct-logit contribution agree in sign across held-out
domains. For DeltaNet, analogous low-rank subtraction is applied to copied cache
states at a task boundary, never to the sole production checkpoint.

An unconditional left projection and a context-selective negative-vector edit
answer different questions. `(I - beta g g^T)W` removes both positive and
negative use of the output dimension everywhere; it can erase legitimate
verification or supported deference. A selective rank-one edit instead has the
form `W' = W - eta g u^T`, where an input-side trigger `u` is fitted from paired
obedience-versus-verification component inputs on development histories. The
first asks whether the output subspace is necessary; the second asks whether a
governance-selective feature can be disconnected from the harmful readout. Both
must first be emulated with reversible hooks, and neither is promoted solely
because it improves mismatch tasks.

## Success criterion

An edit is successful only if its confidence interval shows meaningful mismatch
recovery and excludes a predeclared collateral-damage tolerance on fresh,
matched, supported-deference, and factual-memory controls. The unedited base
checkpoint remains immutable; any promoted edit is stored as a small reversible
adapter plus its training split and provenance.
