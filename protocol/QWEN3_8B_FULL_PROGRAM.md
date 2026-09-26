# Qwen3-8B full experimental program

## Research object

The paper studies a deployment-relevant state-alignment problem, not a prompt
wording artifact: a fixed model infers an interaction-governance state from
earlier turns, carries that state across an explicit task boundary, and may
apply it when the new task requires a different allocation of proposal,
evaluation, correction, or decision authority. The central estimand is the
within-item loss caused by a carried governance state that is mismatched to the
new task rule.

This makes context mismatch a distinct object because it has four separable
questions:

1. Does it transfer across unrelated benchmark domains under counterbalanced
   labels, roles, and wording styles?
2. Which fixed Qwen3-8B components causally carry/read the state?
3. Can a selective negative operator remove harmful carryover without reducing
   all instruction following or factual memory?
4. Does the effect and mitigation generalize to unseen items, history styles,
   benchmarks, and later, architectures?

## Evidence ladder

### E1: broad behavioral identification

The 768 frozen items cover MMLU-Pro, ARC-Challenge, BBH, GSM8K, MATH-500, and
MuSR. Each official item becomes a pairwise correct-versus-fixed-foil decision.
Verification and obedience histories keep completed outcomes fixed while
changing initiative, candidate scope, correction rights, and final authority.
Fresh, length-matched neutral, and explicit reset conditions separate mismatch
from task difficulty, generic long-context degradation, and prompt length.

Primary evidence requires a negative obedience-minus-verification paired margin
in the equal-weight benchmark aggregate, visible effects in each benchmark,
both A/B label assignments, and both natural and lexical-matched histories.
Reset is a distinct recovery contrast rather than part of the definition.

### E2: causal mechanism in Qwen3-8B

The partition contains 192 `component_discovery` items. Mechanism Erratum 1,
frozen after a zero-scientific-row arithmetic preflight failure, fixes the
effective discovery sample at the first 16 item IDs per benchmark (96 items)
so that the sample size agrees with the already frozen 27,648 residual and
18,432 component row budgets. Only these effective discovery items nominate
mechanisms. Exact paired residual replacement tests rescue and reverse
induction at nine layers. The
last layer is an endpoint control. A frozen rule selects three earlier layers,
then exact attention- and MLP-output replacement measures their separate
counterfactual effects. These interventions are nonlinear and are never summed
as explained variance.

Mechanism support requires coefficient-zero identity, exact final-layer target
reproduction, bidirectional effects, label balance, and cross-benchmark sign
coverage. Decodability alone is not causal evidence.

### E3: split-isolated subspace fitting

Only `subspace_fit` supplies activations. History trigger bases are estimated
from completed-turn endpoints, not answer labels. Correction bases come from
paired harmful-minus-verification component states. Task-context bases are
unsupervised current-task activation coordinates. Protected bases include
fresh, verification, reset, supported-authority, and factual-memory states.

The resulting artifacts contain all centering, scaling, rank, ridge, gate, and
data-lineage fields. No coefficient is chosen during fitting.

Because the same frozen histories are intentionally reused across benchmark
families, a history-only boundary classifier cannot have an identifiable
leave-one-benchmark-out test. Pre-forward Erratum 1 therefore validates that
gate by held-out history realization and cross-style transfer. Benchmark
generalization remains a downstream requirement on actual edited behavior in
`operator_dev` and `final_test`.

### E4: mitigation and operator ablation

`operator_dev` is deterministically split into screening and selection halves
within every benchmark. The nested ladder compares identity, fixed translation,
symmetric projection, v2 one-sided protected projection, v3 distinct-basis
transport, trust-region v3, task-conditioned v4, and jointly budgeted
multi-layer v4. Capacity is accepted only when it improves held-out mismatch
recovery and passes every collateral upper-confidence-bound tolerance.

Protected controls are evaluated twice: with the intended answer-blind
application gate and under a forced-on stress ablation. The former measures
deployment targeting; the latter measures whether the learned correction
subspace is intrinsically selective. Passing only the gated version is not
evidence that the negative-vector operator itself preserves utility.

V4 tests a concrete mechanistic refinement: history state and current task state
interact in the readout. If v4 beats admissible v3, one universal negative map
is insufficient. If neither works while exact patching does, the causal map is
more nonlinear, distributed, or stateful than a low-rank local operator.

### E5: one-shot final evaluation

After family, layers, ranks, thresholds, regularization, gate, trust radius, and
token scope are locked in a SHA-bound manifest, `final_test` is opened exactly
once. All row-level edited and unedited outputs are retained. The base BF16
checkpoint remains immutable and the intervention remains a removable hook.

## Claims and falsifiers

The strongest Qwen3-8B claim is available only if all five levels pass:

- interaction governance produces a domain-general paired mismatch effect;
- a carried state is causally relevant in both replacement directions;
- Qwen3-8B localization is obtained directly rather than borrowed from
  Qwen3.5-9B;
- a locked selective operator recovers mismatch on final-test items;
- recovery does not come from generic user resistance, answer-label bias, or
  factual-memory erasure.

Failure is scientifically informative. Cross-benchmark behavioral failure
limits the phenomenon's scope. Decoding without patching indicates a correlate.
Exact patching without low-rank mitigation indicates a complex mediator.
Recovery with collateral damage rejects the operator as a mitigation. A v4
failure rejects task-conditioned low-rank transport, not the behavioral
phenomenon.

## Cross-model boundary

No second model is opened until Qwen3-8B behavior, mechanism, subspace,
operator-development, protected-control, and final-test artifacts are complete
and mirrored to cluster home, archive-host, and local storage with SHA256. The
cross-model phase will reuse the frozen behavioral estimand and protection
criteria while defining architecture-specific mechanism hooks; Qwen3.5
DeltaNet states, Qwen3 KV states, and another model's components will not be
silently treated as the same circuit.
