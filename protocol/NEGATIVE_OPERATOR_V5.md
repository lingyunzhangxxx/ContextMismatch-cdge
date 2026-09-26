# Bidirectional task-aligned governance reset (v5)

## Why the current operator is insufficient

The existing mismatch gate is one-sided:

`p(obedience state | history) × I[independent verification required]`.

It can suppress obedience carryover on an audit task, but it cannot repair the
opposite error: a verification-shaped history carried into a task where safe,
explicit user deference is correct. A one-sided anti-sycophancy operator can
therefore improve one half of the problem while silently defining legitimate
deference as out of scope.

## Signed alignment gate

Let `p_O(h_b)` be an answer-blind probability of an obedience-shaped carried
state, estimated from the residual state at the explicit task boundary. Let
`t(x)` be the current task's preregistered target governance state: zero for
independent verification and one for legitimate delegated choice. Let `a(x)`
be an applicability bit that is zero for non-governance controls such as exact
factual-memory retrieval. The gate is

`g(h_b, x) = a(x) soft_deadzone[p_O(h_b) - t(x)]`.

Positive `g` means the carried state is too obedience-like; negative `g` means
it is too verification-like. For the fitted obedience-direction correction
`delta`, the removable hook applies

`z' = z - g(h_b, x) clip_norm(delta, rho ||z||_2)`.

Thus the same learned transport is subtracted on verification mismatch and
added on deference mismatch. When history and task match, or when the task is
outside the governance family, the gate approaches zero. No answer label,
correctness bit, benchmark identity, or post-hoc outcome enters the gate.

For the reverse direction, a verification-shaped boundary may have zero
one-sided obedience activation. V5 therefore freezes the mean positive
obedience trigger amplitude learned on `subspace_fit` and uses that reference
only to synthesize the counterfactual correction; the signed gate supplies the
example-specific magnitude and direction. Without this reference-trigger
decoupling, multiplying a negative gate by a zero ReLU trigger would make the
reverse intervention vacuous.

## Targeted operator improvements

V5 combines four safeguards that are individually auditable:

1. **bidirectional signed correction**, instead of globally reducing user
   influence;
2. **task-target gating**, so safe delegated choice is a protected target rather
   than merely an intervention-off control;
3. **uncertainty deadzone**, preventing low-confidence history decoding from
   triggering a full edit;
4. **global multi-layer trust allocation**, with per-site radii whose sum is no
   greater than a frozen global budget.

The v4 task-conditioned bilinear transport remains the correction model. V5
changes the alignment objective and gate, not the base checkpoint. Claim runs
remain reversible BF16 forward hooks.

## Required nested evaluation

On `operator_dev`, retain the existing no-edit, v2, v3, and v4 ladder, then add:

- v4 with the old one-sided gate;
- v4 with the signed gate and zero deadzone;
- signed v4 with each preregistered deadzone;
- signed v4 with one, two, and three layers under a shared trust budget.

An operator is admissible only if it reduces the positive
matched-minus-mismatched gap (baseline minus edited is positive), both mismatch
directions recover, and the upper confidence bound on collateral loss remains
inside the frozen tolerance for fresh verification, explicit reset, supported
authority, factual memory, and task-aligned matched cells. Final-test opening
still occurs exactly once after one tensor manifest is locked.

## Interpretation

Success would demonstrate selective control of a fixed network's carried
activation state. It would not show that conversation changed model weights,
that a unique governance vector exists, or that a permanent parameter edit is
safe. Materializing the hook as an adapter is a separate, later experiment and
is not production-authorized.
