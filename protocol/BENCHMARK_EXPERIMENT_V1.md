# Qwen3-8B multi-benchmark experiment contract

This document explains the frozen machine-readable contract in
`BENCHMARK_CONTRACT_V1.json`. The experiment is a pairwise verification study,
not a claim about official benchmark leaderboard scores. Every selected public
benchmark item is converted, before any model forward, into a correct candidate
and one deterministic foil. The same item, foil, displayed labels, role
declaration, and task suffix are reused across paired governance histories.

## What makes the phenomenon identifiable

The primary estimand is the within-item change in the correct-versus-foil logit
margin from a verification history to an obedience history. Both histories end
in the same correct microtask outcomes. They differ in who is permitted to keep
candidates in scope, compare evidence, correct unsupported preferences, and own
the final decision. The final user preference always points to the same foil.

The full design includes two declared roles, two independently written history
styles, both A/B label assignments, a length-matched neutral history, and an
explicit governance reset. This separates five explanations that otherwise get
conflated: task difficulty, generic long-context degradation, wording overlap,
declared-role effects, and an enacted governance carryover.

Fresh correctness is never used to select the primary sample. The main outcome
is a margin over every frozen item. Binary accuracy is secondary. A separate
fresh, official-format run is a model sanity check and must not be called the
same endpoint as pairwise verification.

## Benchmarks and coverage

The six primary task families are MMLU-Pro, ARC-Challenge, eight reasoning-heavy
BBH configurations, GSM8K, MATH-500, and all three MuSR splits. Each contributes
128 items, divided into four disjoint 32-item partitions. Multiple-choice tasks
use a deterministic existing distractor. GSM8K and MATH-500 use a deterministic
answer mutation recorded in the row-level manifest. Mutation method and the
resulting candidate are audit fields, not hidden preprocessing.

GPQA Diamond is useful external validation but was auto-gated when this contract
was frozen. It is non-primary and cannot silently replace a failed benchmark.

## Evidence ladder

1. Smoke validates checkpoint lineage, tokenization, exact paired suffixes,
   finite logits, resume behavior, and row uniqueness.
2. The depth scan tests whether the paired mismatch effect is present at depths
   1, 8, and 32 in a fixed discovery subset.
3. Full behavior evaluates the frozen factorial on all partitions, while still
   keeping their later mechanism roles separate.
4. Component discovery uses only the first partition. Residual and attention/MLP
   patching must be bidirectional and label-balanced.
5. Governance and protected subspaces are fitted only on `subspace_fit`.
6. Operator family, rank, thresholds, layers, and strengths are selected only on
   `operator_dev` using a preregistered Pareto objective.
7. `final_test` is opened once after the selected operator artifact and its hash
   are frozen.

## Statistical unit and claims

The item is the sampling unit. Confidence intervals use a within-benchmark
cluster bootstrap, then aggregate the six benchmark effects with equal weight.
This prevents the longest benchmark or the largest prompt family from deciding
the pooled result. Per-benchmark effects, label-swap halves, roles, and history
styles remain visible. A result supports context mismatch only if the obedience
history lowers margins relative to verification, generic neutral history does
not explain the same loss, and reset produces recovery. A role interaction is a
separate claim and is not required for the governance effect.

## Full-run hard gate

No full job may be submitted until the official unquantized checkpoint has a
verified file manifest, the benchmark manifest is hashed, A and B are confirmed
single tokens under the frozen tokenizer, and the smoke report is complete,
unique, finite, and reproducible. Cluster jobs follow held submission, exact
record validation, and release. The base checkpoint remains immutable and
`production_rollout_approved=false`.
