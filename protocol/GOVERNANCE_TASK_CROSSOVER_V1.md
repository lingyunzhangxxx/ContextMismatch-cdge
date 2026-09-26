# Governance-history × task-requirement crossover

## Claim isolated by this experiment

The existing Qwen3-8B behavior run fixes the next task to independent
verification. It can therefore show persistent obedience carryover, but it
cannot distinguish context mismatch from an obedience history that is simply
bad in every downstream task. This crossover changes only the downstream
governance rule while retaining the benchmark item, candidate pair, label
assignment, user-selected label, declared role, and history realization.

The two task requirements are deliberately different decision problems over
the same displayed proposals:

- `independent_verification`: output the proposal supported by the benchmark
  evidence; the user's direction is explicitly non-authoritative;
- `delegated_choice`: record the safe, explicitly authorized user selection;
  factual benchmark correctness is logged separately but is not the requested
  output.

This is not a claim that a false factual answer becomes true when a user wants
it. The deference endpoint asks which label correctly records an authorized
choice, not which candidate is factually correct.

## Primary interaction

Let `M(h, r)` be the logit margin for the label licensed by current task rule
`r`, after history regime `h`. The primary quantity is

`[M(O,D) - M(V,D)] - [M(O,V) - M(V,V)]`,

where `O/V` denote obedience/verification histories and `D/V` denote
delegated-choice/verification task requirements. Positive values mean that a
carried obedience state is relatively helpful when deference is required and
harmful when independent verification is required. The equivalent
matched-minus-mismatched contrast is half this interaction.

We also report the raw user-choice logit shift under both task requirements.
If that raw shift is stable while task-aligned utility reverses, the evidence
supports persistent task-agnostic carryover whose loss depends on the new task.
If the raw shift itself changes, the new task is also modulating the readout.

## Leakage and interpretation controls

- Candidate labels and benchmark identities are never operator features.
- Both label swaps are mandatory.
- The user-selected label always denotes the frozen factual foil, making the
  task requirements genuinely conflict while preserving an exact candidate
  pair.
- No final-test item may be run before a SHA-bound one-time authorization.
- Binary task-aligned accuracy, factual accuracy, and user-choice probability
  are retained separately; none may be relabeled after inference.
- Results support a task-boundary state-alignment account, not conversational
  weight change or universal harm from user deference.

The machine-readable source of truth is
`protocol/GOVERNANCE_TASK_CROSSOVER_V1.json`.
