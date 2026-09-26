# Mechanistic research framing

## Object of study

Model weights are fixed at inference time. Context mismatch therefore cannot be
caused by the context literally changing parameters. The candidate cause is a
**governance control state**: a context-dependent activation and cache state
computed by fixed parameters from prior allocations of proposal, scope,
evaluation, and decision rights.

“Parameter-level explanation” therefore means identifying the fixed circuit
that writes, transports, and reads this state. It does not mean that conversation
history updates model weights. The same parameters can implement different
policies because their activations and recurrent/KV states differ.

Let history `H` induce layerwise state `g_l(H)`, and let a new task `x` require
governance regime `r(x)`. Context mismatch is the excess decision loss when the
carried state conflicts with the task requirement:

`CM(H, x) = L(pi(x | H), y*) - L(pi(x | H_reset), y*)`.

The hypothesis is stronger than a behavioral carry-over claim:

1. **Decodability:** governance regime is linearly decodable from residual and
   cache states on held-out task domains.
2. **Persistence:** the governance direction survives unrelated evidence and a
   task boundary, with magnitude increasing or saturating with history length.
3. **Causal restoration:** replacing an obedience-history residual with its
   exact matched verification-history residual reduces mismatch loss.
4. **Reverse induction:** replacing a verification-history residual with its
   matched obedience-history residual increases mismatch loss, providing a
   sufficiency-style test without assuming a universal additive direction.
5. **Circuit localization:** a small set of recurrent/attention components and
   downstream MLP features transmit the direction into the decision logit.

These tests form a causal-mediation sequence: history regime is the treatment,
the governance state is the candidate mediator, and correct-vs-foil logit margin
is the outcome. Cross-style decoding establishes transferable information;
exact paired residual restoration and reverse induction test mediation;
component/state interventions test where that mediator is written and read.

A completed balanced intervention rules out a simpler mechanism. A fixed
final-token additive direction cannot repair both label-swapped halves of the
design: its effect on the `A-B` logit is sign-fixed while the correct label is
counterbalanced. Across 1,728 runs, obedience recovery was only `+0.042` and the
verification change was `-0.076`; early and middle layers were approximately
zero. The governance state may still be causal, but it is not merely a single
context-independent answer-bias vector. Larger steering coefficients are not a
valid response to this falsification.

The balanced depth-32 lexical cache experiment also rules out a simple
single-store reset account. The intact verification-minus-obedience margin gap
was `1.528` and was positive in all six task domains (`p=0.03125`). Zeroing only
convolution state left a gap of `1.453`; zeroing recurrent state left `1.177`;
resetting the complete DeltaNet state left `0.741`; and dropping full-attention
KV left `0.559`. None was a selective rescue: the latter interventions reduced
verification margins and task accuracy as well as the mismatch gap.
`recurrent_zero+kv_drop` reduced the gap to `0.003`, but both regimes collapsed
to about `0.63` mean margin. `delta_reset+kv_drop` made the two histories exactly
identical, as an all-history-erasure positive control must, while accuracy fell
to `0.458` in both. Therefore a vanishing gap after destructive cache clearing
does not localize governance to one cache type. The next cache experiment must
replace states with exact matched counterfactual states, and must include
fresh-position and factual-memory controls.

The strict lexical effect is already present after one history exchange. At
depth 1, the obedience-minus-verification margin was `-1.974`; all 72 matched
pairs and all six task-domain means were negative (`p=0.03125`). The effect was
smaller at depth 8 (`-1.559`) and depth 32 (`-1.521`), so the present data reject
a simple monotonic accumulation account. A governance allocation can be written
immediately and then persist or be partially normalized across later turns.

Residual patching uses a stricter numerical control than behavioral collection.
Across the full balanced set, cached-suffix coefficient-0 margins differed from
the older full-sequence forward in 71/144 items (mean absolute difference
`0.066`, maximum `0.25`), even though all cache-path baselines were invariant
across hook layers. This is consistent with different full/chunk DeltaNet kernel
paths and is material relative to early-layer patch effects. The primary patch
therefore captures target residuals online through the same cached-suffix path;
the old 3,456-row full-path run is retained only as a numerical diagnostic.

The 3,456-row cached-target residual experiment passes all numerical controls:
all rows are unique, coefficient-0 is identical across eight patch layers, and
layer-31 coefficient-1 reproduces all 144 matched target margins exactly. Exact
coefficient-1 patches are near zero through layer 15. Bidirectional mediation
emerges at layer 19 (obedience rescue `+0.111`, reverse induction `-0.170`), then
grows at layer 23 (`+0.314`, `-0.408`) and layer 27 (`+0.351`, `-0.476`). At each
of layers 19, 23, and 27, all six domain means have the predicted sign
(`p=0.03125`), and both label-swap halves have the predicted sign. Layer 31 gives
the expected exact endpoint swap and is treated only as a positive control.
Together with middle-layer decodability, this supports separate write/carry and
late task-conditioned readout stages.

The 2,016-row exact paired-cache experiment closes the corresponding storage
test. It is a complete balanced factorial (72 items per regime/intervention),
has no duplicate or non-finite rows, and complete-cache replacement reproduces
the matched target-history intact margin with maximum absolute error `0.0`.
Replacing convolution state alone is null (`-0.014` in obedience sources and
`+0.019` in verification sources). Replacing recurrent state gives
bidirectional mediation (`+0.453` obedience rescue, `-0.349` reverse induction),
while full-attention KV replacement is larger (`+1.184`, `-1.076`). Replacing
recurrent state and KV jointly nearly exchanges the entire behavioral effect
(`+1.547`, `-1.542`, versus an intact regime gap of `1.528`); the slight
overshoot relative to complete-cache replacement shows that cache components
interact and should not be assigned additive percentages.

Four-layer stage replacement localizes the strongest stored counterfactual to
layers 12--15 (stage 3): `+0.918` rescue and `-0.766` reverse induction. Layers
8--11 (stage 2) are smaller but bidirectional (`+0.415`, `-0.358`); layers 16--31
are weak or null when their pre-boundary caches alone are exchanged. Stage-3
rescue is positive in all six task domains (`p=0.03125`); its reverse effect has
the predicted sign in five of six (`p=0.0625`), with the weak data-integrity
domain near zero. Both label-swap and evidence-order strata retain the expected
aggregate direction. This is consistent with governance information being
stored in middle-layer recurrent/KV state and read into task-conditioned
residuals later, rather than stored in the late caches themselves. It remains a
distributed, non-additive mediation result—not evidence that swapping a single
cache component is a safe mitigation.

The CUDA component-output runner has not supplied a balanced localization
result. Its 36-row smoke covers one procurement item and only validates hook
execution and coefficient-0 invariance. The interrupted CUDA full file contains
1,224/2,592 rows with unequal regime/realization coverage (432 obedience rows
and 792 verification rows). Equal counts across patch coefficients do not repair
that truncation bias. No mixer-versus-MLP effect from either CUDA file is used in
the mechanistic claim. A complete Ascend component experiment is reported below
as a backend-specific result; it is not relabeled as completion of the CUDA run.

## Why long-context hybrid models are diagnostic

The mechanistic main model is the official non-quantized Qwen3.5-9B checkpoint.
It alternates three Gated DeltaNet (linear/recurrent attention) layers with one
full-attention layer and supports 262K tokens. The model is small enough to run
BF16 hooks on two A10 GPUs, avoiding the attribution and weight-edit ambiguity
introduced by GPTQ/FP8 kernels. DeltaNet updates a learned recurrent matrix:

`S_t = exp(g_t) S_{t-1} + k_t (v_t - S_{t-1} k_t)^T beta_t`.

This state is explicitly designed to compress and carry history. It has no
privileged knowledge that an application-level task boundary should erase
governance while retaining factual memory. That makes two competing mechanisms
testable:

- **recurrent-state entanglement:** reset DeltaNet recurrent states at the task
  boundary while preserving full-attention KV;
- **KV retrieval:** preserve DeltaNet states but remove/patch pre-boundary KV in
  full-attention layers.

If the first intervention rescues decisions much more than the second, mismatch
is not merely retrieval of a few obedience phrases; it is compressed into the
model's persistent recurrent control state.

## Ascend replication boundary

The causal numbers before this section come from the CUDA BF16 backend. The Ascend
910B3 port is a separate backend replication and is not numerically pooled with
CUDA. The failed Job 8904 remains useful lifecycle evidence: it stopped before
model loading because the probe reset NPU peak-memory statistics before device
initialization. Job 8905 then changed only that initialization order and passed
the bounded hard gate with the official non-quantized BF16 checkpoint. Model
load took `3.327` seconds, the forward took `25.309` seconds, A/B logits were
`23.75/18.5`, and every inspected residual, DeltaNet state, and KV tensor was
finite. The paired failure and success isolate the earlier problem to probe
lifecycle rather than model/backend incompatibility.

Job 8906 passed the preregistered backend-parity contract on 12 fixed samples:
six tasks, two regimes, depth 32, realization 0, and label/order swap 0. Prompt
token counts, token hashes, and A/B label IDs matched exactly. Full-path margin
MAE/max error/Pearson were `0.0521/0.125/0.9993`; cached-path values were
`0.1042/0.25/0.9976`. Decision and correct-margin-sign agreement were `1.0` on
both paths. This is strong evidence that the selected Ascend execution path is
behaviorally comparable to CUDA at the preregistered gate, not permission to
merge backend rows or an estimate of scientific-effect equivalence.

Job 8907 then passed a 40-row mechanism smoke. Coefficient 0 exactly reproduced
the cached baseline, layer-31 residual coefficient 1 exactly reproduced the
paired target endpoint, all logits were finite, and the cached baseline exactly
matched Job 8906. In its single routing item, verification-to-obedience residual
effects at layers 11/19/27/31 were `0.0/-0.25/-0.75/-3.0`, while the reverse
effects were `0.0/0.0/+0.375/+3.0`; the layer-27 mixer moved the margin by
`-0.25/+0.25` in the corresponding directions. The signs are descriptively
consistent with the CUDA smoke, but one task, one realization, and one
label/order cell cannot localize a component or establish domain-general
mediation.

After separate user authorization, residual Job 8909 completed the full frozen
3,456-row factorial. It has 3,456 unique finite rows, zero missing, duplicate,
or unexpected rows, coefficient-0 maximum logit error `0.0`, and exact layer-31
target endpoints in all 144 cases. The preregistered scientific gate passes at
layers 19, 23, and 27 in both causal directions, both label-swap halves, and all
six task domains. At coefficient 1, reverse-induction effects are
`-0.165/-0.408/-0.470`, while rescue effects are `+0.102/+0.307/+0.352`.
Against the six separate CUDA summary cells, effect MAE is `0.00463`, maximum
absolute difference is `0.00868`, and correlation is `0.99990`. This strongly
rules out a CUDA-only hook or kernel artifact while still keeping backend rows
unpooled.

Component Job 8912 then completed the separate 2,592-row factorial with the
same completeness and finite-value guarantees and coefficient-0 maximum error
`0.0`. Its coefficient-1 effects are:

| Counterfactual | Layer | Full residual | Mixer output | MLP output |
|---|---:|---:|---:|---:|
| Verification baseline + obedience patch (reverse induction) | 19 | -0.1649 | -0.0677 | -0.0313 |
| Verification baseline + obedience patch (reverse induction) | 23 | -0.4080 | -0.0503 | -0.0104 |
| Verification baseline + obedience patch (reverse induction) | 27 | -0.4705 | -0.0503 | -0.0174 |
| Obedience baseline + verification patch (rescue) | 19 | +0.1024 | +0.0365 | +0.0139 |
| Obedience baseline + verification patch (rescue) | 23 | +0.3073 | +0.0503 | -0.0069 |
| Obedience baseline + verification patch (rescue) | 27 | +0.3524 | +0.0399 | +0.0122 |

Mixer output has the expected aggregate sign and is larger in magnitude than
MLP output in all six cells; MLP has the expected sign in five of six. This is
exploratory ordering—the contract deliberately did not preregister mixer versus
MLP—and task-domain and label-swap signs are not uniform for every isolated
component. Moreover, the descriptive mixer-plus-MLP sum is 49--60% of the
layer-19 full-residual effect but only 14--15% at layers 23 and 27. Separate
component counterfactuals are nonlinear and cannot be summed as explained
variance. The defensible conclusion is therefore a mixer-dominant, distributed,
and interacting late readout, not a single-module localization claim.

The combined evidence supports a more specific fixed-parameter mechanism:
middle-layer recurrent/KV state carries a regime-conditioned governance
representation across the boundary; late residual computation reads that state
into a task- and label-conditioned decision margin; mixer outputs participate
more consistently than isolated MLP outputs in that readout. Exact paired
patching makes the state causally relevant, and cross-backend replication makes
the effect operationally robust. It still does not show that one parameter
vector is necessary, that the mechanism is universal across architectures, or
that context mismatch is unavoidable whenever a task changes.

For Qwen3-8B, this earlier hybrid-model localization is only a hypothesis
generator. Qwen3-8B is a 36-layer standard Transformer, so its mechanism cannot
be described as DeltaNet recurrent-state carryover. The frozen Qwen3-8B
discovery design instead uses exact paired residual replacement at layers
3/7/11/15/19/23/27/31/35, then a separately locked attention-versus-MLP scan at
the three preregistered nominated layers. It covers all six benchmark families,
both causal directions, roles, history styles, and label swaps on the disjoint
`component_discovery` partition. Layer 35 is an endpoint control, never an
eligible mechanistic component.

Mitigation now contains a further falsifiable branch. V3 assumes that one
history-trigger-to-correction transport is shared across tasks. V4 augments
that map with a bounded, answer-label-blind code of the current task state. In
algebraic terms it regresses paired causal corrections on both the history
trigger and its bilinear interaction with task-state coordinates. If v4
improves held-out mismatch recovery while v3 does not, the result supports a
task-conditioned readout of governance state. If neither approaches exact
paired patching, the relevant map is likely more nonlinear, token-distributed,
or cache-mediated than a local low-rank operator. This comparison is fitted on
`subspace_fit`, selected on `operator_dev`, and evaluated once on `final_test`;
the correct answer and benchmark identity are never operator inputs.

## Qwen3-8B multi-benchmark behavior boundary

The new Qwen3-8B program separates broad behavioral evidence from the earlier
Qwen3.5-9B causal discovery. Its endpoint is benchmark-derived pairwise
candidate verification on MMLU-Pro, ARC-Challenge, BBH, GSM8K, MATH-500, and
MuSR. Correct candidates, foils, label swaps, task suffixes, and four disjoint
mechanism/mitigation partitions are fixed before any Qwen3-8B forward.

The accepted 96-row smoke shows a paired obedience-minus-verification margin
of `-3.2396` with bootstrap 95% CI `[-4.2188,-2.2500]`; all six benchmark means
are negative. A governance reset moves the margin by `+4.9583`, CI
`[3.9583,5.9482]`. This supports transfer of an interaction-induced response
tendency across unrelated tasks, but the sample is intentionally too small for
the paper's final estimate. It also does not yet establish that the Qwen3.5
localized circuit is the Qwen3-8B mechanism.

The completed Qwen3-8B full behavior run now replaces that smoke estimate for
behavioral claims. Across 6,144 within-item pairs, obedience-minus-verification
is `-6.85075`, item-cluster bootstrap 95% CI `[-7.03746,-6.66750]`, with all
six benchmark means negative. Both lexical-matched (`-3.89046`) and natural
(`-9.81104`) histories, both declared roles, both label swaps, and all four
preserved partitions have the same sign. Verification is statistically
indistinguishable from its length-matched neutral history (`-0.08030`, CI
`[-0.16720,0.00728]`), while obedience-minus-neutral is `-6.93105`; explicit
reset recovers `+10.07928`. Thus the broad Qwen3-8B result identifies a
governance-specific, reversible context effect rather than generic length
degradation. It still does not localize a Qwen3-8B mediator or demonstrate that
the proposed low-rank operator can remove it; those claims remain gated on the
disjoint mechanism and mitigation stages.

The Qwen3-8B residual mechanism smoke then validates the exact paired runner on
the new architecture. All 432 rows are unique and finite; coefficient zero is
an exact identity and layer 35 exactly reproduces every paired target margin.
In the single-item-per-benchmark smoke, bidirectional mediation grows sharply
in the later stack: layer 15 gives obedience rescue / verification reverse
induction of `+1.292/-0.875`, layer 19 gives `+3.917/-3.917`, and layer 23 gives
`+4.208/-4.104`, against a mean intact target gap of magnitude `4.208`. Layer
35 is only the endpoint positive control. These values validate intervention
orientation and motivate the frozen late-layer scan; they are not a mechanism
estimate because every benchmark contributes only one item.

The first full submission also served as a design falsifier. Before an affected
forward, its row-count gate showed that the original `32 items per benchmark`
field implies 55,296 residual rows, inconsistent with the frozen 27,648-row
budget. Job 8946 therefore exited with zero scientific rows. Erratum 1 fixes the
effective sample at 16 item IDs per benchmark (96 items total), which preserves
all roles, styles, label swaps, directions, layers, coefficients, and the later
split boundary while matching both frozen residual and component row counts.
The original contract and failed run remain immutable evidence; corrected full
execution uses separately hashed code-v12 and authorization v2.

The execution audit found a separate Ascend constraint: batch=2/4 changes BF16
cached-prefix logits enough to fail the frozen parity threshold even when
suffix lengths are identical. Scalar execution reproduces exactly across three
runs. This is not evidence for context mismatch; it is a backend numerical
falsifier caught by the gate. All subsequent Qwen3-8B claim runs therefore use
batch size 1, and failed batched rows remain excluded.

The exact factorial, row keys, numerical controls, and interpretation rules
remained frozen in `ascend/full_replication_contract.json`; execution permission
was recorded separately. Residual and component jobs remained separate, layer
31 remained only an endpoint positive control, and CUDA/Ascend rows were never
pooled. Two failed component preflights (Jobs 8910 and 8911) produced zero
scientific rows and are archived separately from successful Job 8912.

The Qwen3-8B full component scan now supplies architecture-specific causal
localization rather than borrowing a component hypothesis from Qwen3.5. Across
18,432 unique finite rows, coefficient zero is an exact identity and the locked
row-key set is complete. Applying the preregistered score independently at each
residual-nominated layer selects three different component sites:

- layer 23 `self_attn`, score `0.318679`;
- layer 31 `mlp`, score `0.183478`;
- layer 27 `mlp`, score `0.155176`.

The alternatives at those layers score `0.038410`, `0.020712`, and `0.025304`,
respectively. More importantly than the ordering, each selected site rescues an
obedience-history source with a positive patch effect and induces the reverse
failure from a verification-history source with a negative effect in both
label-swap halves. At layer 23 attention, those four means are
`+1.0527/+1.6029` and `-1.1797/-1.4336`; at layer 31 MLP they are
`+0.5632/+0.9206` and `-0.6484/-0.8841`; at layer 27 MLP they are
`+0.5794/+0.6992` and `-0.6022/-0.6484`. The result therefore supports a
distributed late-stack read/write account with a particularly strong
attention contribution at layer 23 and later MLP contributions. It does not
license summing component effects as explained variance, nor does it show that
a single negative vector will reproduce exact patching.

These sites are now immutable inputs to the split-isolated operator stage. The
operator will estimate history-trigger, task-context, correction, and protected
bases only from `subspace_fit`, then compare fixed translation, projection,
one-sided transport, task-conditioned transport, and signed bidirectional
alignment under a shared trust budget. This is the direct test of whether the
localized causal map is approximately low-rank and selectively resettable.

## Symmetric task-requirement crossover

The governance account makes a stronger prediction than generic anti-sycophancy:
the sign of a carried interaction policy should depend on what the next task
requires. We therefore cross obedience versus verification histories with two
new-task requirements while keeping the candidate payload and user-selected
label paired: independent factual verification and delegated choice. The
preregistered interaction is

`(obedience - verification)_delegated - (obedience - verification)_verification`.

It is positive in two disjoint 6,144-row Qwen3-8B evaluations. Discovery gives
`+2.44393` with item-cluster bootstrap 95% CI `[2.20321,2.68994]`; replication
gives `+2.19132`, CI `[1.96403,2.41960]`. The directional terms also replicate:
obedience history lowers task-aligned margin on independent verification by
`-0.91772` and `-0.80526`, but raises it on delegated choice by `+1.52620` and
`+1.38607`. All six benchmark interactions are positive in each split, as are
both declared roles, both history styles, and both label swaps.

This rules out the narrow interpretation that an obedience-consistent history
is uniformly harmful. The same carried tendency is beneficial when it matches
the next task's allocation of decision authority and harmful when it does not.
It therefore supports a governance-state alignment account rather than merely
a general long-context or anti-agreement effect. The strict both-direction
criterion holds for `26.69%` of discovery pairs and `25.65%` of replication
pairs, however, so the interaction is an aggregate causal target, not a
deterministic per-example mechanism. The next operator stage must improve this
interaction while protecting appropriate delegation, correct factual memory,
supported authority, fresh behavior, and matched verification. Until that
split-isolated mitigation passes, no claim of selective reset is licensed.

### Decision-boundary susceptibility, not monotonic task difficulty

A retrospective cross-split analysis refines where the behavioral loss occurs.
Thresholds were estimated from discovery matched-correct task-aligned margins
and frozen before applying them to replication. On independent verification,
replication correct-to-wrong flip rates are `29.67%` in the discovery-defined
low-margin band (`m <= 3.0`), `2.51%` in the middle band, and `0%` in the
high-margin band (`m >= 14.75`). The item-cluster bootstrap contrast between
the boundary and robust bands is `+29.67` percentage points, 95% CI
`[23.11,36.56]`. On delegated choice the corresponding rates are `3.37%`,
`0%`, and `0%`; the boundary-minus-robust contrast has 95% CI `[1.31,5.81]`.

This pattern follows the margin-crossing condition. If `m_matched` is the
task-aligned margin under the matched governance history and `Delta_CM` is the
mismatch-induced margin change, a matched-correct answer flips only when

`0 < m_matched <= -Delta_CM`.

Consequently, conventional item or benchmark difficulty is not the operative
variable by itself. Very hard items may already be wrong and thus have no
remaining correct answer to flip (a floor effect), while easy high-margin items
remain stable. The vulnerable region consists of currently correct but
low-positive-margin decisions. Consistent with this distinction, the
six-benchmark replication diagnostic gives Spearman `rho=0.20` between matched
error rate and accuracy drop, exact permutation `p=0.7139`; it provides no
evidence for a monotonic “harder benchmark, larger mismatch effect” law.

This analysis is post-hoc and uses discovery-to-replication threshold transfer,
so it is evidence of cross-split boundary susceptibility rather than a
preregistered confirmation. It uses no final-test data. A prospective held-out
test with frozen margin bands remains required before treating the relation as
a confirmed general law.

## Conditional inevitability proposition

Assume a next-token policy infers an unobserved governance regime `G_t`, the
training distribution has positive regime persistence
`P(G_t = G_{t-1}) > P(G_t != G_{t-1})`, actions depend on `G_t`, and a task reset
is not perfectly observable. Then on reset examples, posterior use of
`P(G_t | H)` produces strictly positive expected regret whenever the action
policies for the two regimes differ. This does **not** make failure universal to
all transformers. It identifies sufficient conditions under which some
carry-over error is Bayes-optimal for the training distribution yet wrong for
the deployed interaction.

There is also a sharper identifiability boundary. If one observable transcript
`o` is compatible with a continuation world requiring action `a_c` and a reset
world requiring a different action `a_r`, then any policy conditioned only on
`o` must choose the same action in both worlds. With posterior reset probability
`q` and minimum wrong-action loss `Delta`, its conditional expected regret is at
least `min(q, 1-q) Delta`. This is **task-boundary state aliasing**: the failure
is unavoidable only when the governance reset is not identifiable from the
model's observations. Our experiment is deliberately stricter than this lower
bound—the final prompt contains an explicit new-task marker—so an observed
effect tests imperfect learned resetting, not logical impossibility. Mechanistic
experiments ask which fixed circuits preserve and read the old posterior despite
that marker.

The completed interventions support two premises of this proposition: histories
do induce persistent regime-conditioned states, and swapping those states can
change task decisions in both directions. They do not measure the training
distribution's regime-persistence prior or prove that the explicit reset marker
is observationally insufficient. “Inevitable” therefore remains a conditional
mechanism theorem, not an empirical claim about every mismatch instance.

## Falsifiers

The hypothesis is weakened or rejected if any of the following holds:

- governance is not decodable above matched lexical controls;
- probes fail leave-one-domain-out generalization;
- activation direction is decodable but causal interventions do not move logits;
- only exact governance tokens, not summaries or unrelated-task transfer, carry
  the effect;
- resetting factual and governance states cannot be dissociated;
- effects disappear under label swap or evidence-order controls.

## Mitigation after localization: negative-vector pruning

Mitigation is intentionally downstream of causal localization. A direction that
is merely decodable is not yet a safe edit target. Let

`g_l = E[h_l | obedience] - E[h_l | verification]`

be estimated without the evaluated task domain. We test a ladder of increasingly
persistent interventions:

1. **Fixed-vector diagnostic (completed and rejected as mitigation):** apply
   `h'_l = h_l - alpha g_l` at the decision token under a fully label-balanced
   design. This tests a universal translation hypothesis, not governance-state
   mediation in general.
2. **Exact paired activation patching:** patch matched verification residuals
   into obedience runs, and perform the reverse induction. This localizes causal
   mediation without requiring the state to be one-dimensional or additive.
3. **Output-subspace pruning:** for a localized component with residual output
   `z`, apply `z' = z - beta <z, g_hat> g_hat`. This is implemented first as a
   hook, exactly equivalent to a rank-one left projection of its output matrix,
   before any weights are saved. The completed component experiment locks layer
   23 `self_attn`, layer 31 `mlp`, and layer 27 `mlp` as the three Qwen3-8B
   operator sites; unselected components remain retained as negative controls.
4. **Feature pruning:** rank MLP features by paired activation change and direct
   contribution to the A-vs-B logit. Ablate only features that are both
   governance-selective and causally implicated.
5. **Low-rank parameter edit:** if the hook intervention is selective, encode it
   as a reversible adapter or projected output weight rather than destructively
   rewriting the base checkpoint.
6. **Recurrent-state pruning/reset:** estimate low-rank obedience directions in
   DeltaNet recurrent state and remove those directions specifically at an
   observable task boundary.

The optimization objective is not unconditional refusal or reduced instruction
following. A valid edit must lower mismatch loss on held-out task domains and
held-out governance paraphrases while preserving fresh-task accuracy,
verification-match accuracy, supported user-direction following, and factual
memory carried across the same boundary. We report the full Pareto curve over
intervention strength rather than selecting an `alpha` on the test tasks.

The component result also sets a realistic mitigation expectation. A rank-one
projection of one component output is unlikely to reproduce the much larger
late full-residual intervention across the locked layers 23, 31, and 27. The next experiment
should fit governance subspaces on disjoint histories and domains, test each
mixer hook without changing weights, then test a jointly regularized multi-layer
projection only if single-layer interventions are selective. Candidate
selection, subspace fitting, coefficient choice, and final evaluation must use
separate splits; otherwise the apparent “negative-vector pruning” benefit would
reuse the same 144 items that nominated the components.

## Split-isolated capture status

The input evidence for this mitigation ladder is now complete on Qwen3-8B.
Job 8969 captured 3,072 subspace-fit states in 12 shards, including 768
answer-blind boundary states. Job 8970 separately captured 4,008 protected
states in 50 shards. Their manifest SHA256 values are respectively
`069ab3db5f1c732f14a3e3a264c75ae831136f3077b12fb062a9fbf8b4fde194`
and `f084c379cf59b5eeffdb67f62ce26d2cbb7418919da5e93a297dea1030a7c690`.
Both runs have job-local exit 0, `COMPLETE`, Slurm `COMPLETED ExitCode=0:0`,
and content-verified copies in cluster shared home, archive-host, and local
storage.

These captures strengthen the experimental separation but are not themselves
evidence that a low-rank reset works. History-trigger, task-context, correction,
and protected bases are fitted without using operator-development answers;
the answer-blind gate is validated by held-out history realization and style,
while actual cross-benchmark generalization remains a downstream behavioral
criterion. The protected capture also makes the selectivity claim falsifiable:
each surviving operator must pass both the intended application gate and a
forced-on stress test, so a successful external gate cannot hide an
indiscriminate negative subspace.

The frozen single-site grid contains 15,660 configurations spanning fixed
translation, symmetric rank-one projection, one-sided v2, full v3 transport,
task-conditioned v4, and signed bidirectional v5. CPU-only Job 8972 completed
the four-fold pair-hash reconstruction prefilter under `code-v21`, with all
15,660 scores finite and 42 unique candidates materialized. Its three-copy
archive SHA256 is
`effef85bfe20f8568352f1724559ecac0cdacb9e80c8502678bcf00069f5d6ca`.
The offline score does not establish mitigation: Jobs 8973--8976 now evaluate
the frozen shortlist on the disjoint operator-development screening half in
four SHA-bound shards. Surviving candidates must still pass the selection half,
followed by gated and forced-on protected controls. No final-test item has been
opened.

## Split-isolated screening result

The single-site operator ladder is now behaviorally falsified under its frozen
screening rule. Jobs 8973--8976 evaluated all 42 materialized candidates on the
disjoint operator-development screening half, with 3,072 unique finite rows per
candidate, exact coefficient-zero identity, and successful numerical audits.
The merged ledger and finalist report bind to SHA256 values
`246e218a91a8e9de8436dcae616352f85dda4031b5e4a8143471ffe6303cf0e5`
and `e7b8ca9c1e1923b53a19aad1e545a6d3519a6b5d0f49775bbb747f1979a37ae4`
respectively. The frozen selector returns zero eligible candidates and zero
family finalists.

The failure mode is selective and informative. Aggregate improvement alone is
common enough--26/42 candidates have positive equal-benchmark-weight gap
reduction--but it does not survive the symmetry requirements. Only 2/42 have
strictly positive recovery in both mismatch directions, only 10/42 improve both
label-swap halves, and no candidate passes aggregate, directional, and
label-swap gates together. The strongest aggregate v3 candidate
`cf53417cddd6f4db` improves the verification-task direction while leaving the
delegated-task direction exactly unchanged. Two signed v5 candidates recover
both directions, but their effect reverses sign in one label-swap half. This
pattern points to directional asymmetry and label-dependent readout coupling,
not missing rows, numerical instability, or a failed identity control.

Consequently, the tested low-rank single-site family is rejected as a
selective mitigation for Qwen3-8B. This does not undo the replicated behavioral
context-mismatch interaction, the decision-boundary susceptibility result, or
the causal component localization; it says that the present intervention
parameterization does not remove the localized effect symmetrically enough to
justify downstream controls. The protocol therefore stops before held
selection, protected controls, multisite composition, Pareto locking, and final
test. Frozen thresholds remain unchanged, `final_test_open=false`,
`final_test_open_count=0`, and `production_rollout_approved=false`.

Jobs 8973--8975 use explicitly labeled run-local provenance exceptions because
their live scheduler records expired while `sacct` could not reach
`mgnt:6819`; their scientific audits are complete, but formal Slurm terminal
provenance is not asserted. Job 8976 has a captured live Slurm
`COMPLETED ExitCode=0:0` record. The distinction is frozen in
`artifacts/qwen3-8b-v1/operator_behavior_screening_provenance_v1.json`, SHA256
`e005b0edfe2af6b3b4a7bfa59de3b90bdd834de60fcc54c25c031ad0dcca2a5a`, and does
not alter the scientific screening thresholds.

## Adaptive internal governance editing: complete AMSGE V1 falsifier

AMSGE V1 is a model-internal activation editor rather than a prompt-level
policy. At each locked site it predicts history regime, predicts the current
task's governance requirement, and emits a task-conditioned low-rank residual
correction subject to a relative-norm trust region. The base checkpoint is not
rewritten; the learned editor checkpoint contains only the governance heads and
correction parameters. This architecture directly tests whether the causal
state localized above is sufficiently readable and selectively correctable by
a compact, conditional internal intervention.

The V1 fit does learn some intended structure, but not enough to satisfy the
frozen contract. On all three sites, teacher-MSE improvement is positive in both
mismatch directions and both label-swap halves, normalized teacher MSE and
trust-region gates pass, and relative correction is capped at approximately
`0.10`. However, layer 23 attention and layer 31 MLP fail history balanced
accuracy; all three sites fail task balanced accuracy and the protected
forced-on gate. Layer 27 MLP alone passes the history classifier gate, with
history/task balanced accuracies `0.750/0.738`, but it still fails the other two
selectivity requirements. The aggregate fit decision is therefore correctly
negative rather than an optimization crash.

We nevertheless ran the full post-failure characterization on data that fit did
not use. The 3,072-row untouched `operator_dev` selection is numerically exact
and shows that the correction increases the mismatch interaction. The
equal-benchmark-weight match-advantage reduction is `-0.147461` with bootstrap
95% CI `[-0.197432,-0.099935]`; matched-minus-mismatched reduction is
`-0.073730`, CI `[-0.099202,-0.049721]`; normalized reduction is `-0.052387`,
CI `[-0.071640,-0.032993]`. One causal direction improves by `+0.082031`, but
the reverse direction worsens by `-0.100911`; both label-swap aggregates and
all six benchmark aggregates are negative. Thus the result is not merely a
strict gate rejecting a small useful effect: under the primary held behavior
metrics, V1 amplifies the target mismatch.

The complete 2,856-row protected-control experiment identifies an additional
selectivity failure. Fresh verification, matched verification, matched
delegated choice, explicit governance reset, and factual boundary memory all
pass their frozen application-gated and forced-on reference checks.
Supported-user-authority does not: its margin-loss estimate is `0.210069`, and
the one-sided 95% upper bound is `0.280382` when gated and `0.281250` when
forced on, above the `0.25` ceiling. The exact zero-gate identity control passes
with maximum selected-logit error `0.0`, so the failure cannot be attributed to
an always-on hook or a broken no-application path.

Together these results falsify the V1 parameterization while preserving the
broader internal-governance hypothesis. A small conditional editor can learn
directionally meaningful corrections, yet its history/task heads and shared
correction readout do not separate the two causal directions, label swaps, and
authority-preservation requirement well enough. The observed pattern is
consistent with underidentified governance state and correction entanglement:
training-level directional improvements coexist with held behavioral reversal
and authority collateral. It is evidence against treating fit loss or
directional teacher improvement as a proxy for actual mismatch mitigation.

V1 is therefore a complete negative-result baseline, not a partial run. Its
fit, execution smoke, untouched behavior selection, and gated/forced-on
controls are all retained. It is permanently ineligible for locking:
`candidate_eligible=false`, `controls_admissible=false`, and
`candidate_may_be_locked=false`. No Pareto or final-test claim may be derived
from it; `final_test_open=false`, `final_test_open_count=0`, and
`production_rollout_approved=false` remain binding.

## GRC-DGE V5.1: consensus restores exact abstention but loses fold-robust sensitivity

V5.1 tests whether two independently trained linear applicability heads can
remove the six matched-state false positives left by V4 without changing the
successful V3 experts. The capture recovery is complete and dual-lineage: the
6,144-row governance manifest remains bound to the original Job 9140
authorization, while the newly completed 4,008-row protected manifest is bound
to the Job 9203 recovery authorization. Both captures are finite, identity
complete, terminally audited, and available in three verified copies.

The fit establishes one useful mechanism result and one decisive failure. The
consensus rule has exact zero activation in both governance-matched families
and all five protected families. It also reaches 100% mismatch recall in both
directions and label-swap halves on the developmental component-discovery
audit. Thus the V4 matched-state tail overlap is not irreducible in the
captured representation. Yet this separation is not fold robust: on six-fold
cross-fit data, consensus recall falls to 75.70%, with 83.92% for the negative
direction and 67.48% for the positive direction. Exact cross-fit negative
abstention is retained.

The failure is therefore not a safety/efficacy scalar tradeoff created by a
looser threshold. It is a generalization failure of the two linear heads under
the worst held-negative calibration: training losses approach zero and the
developmental audit separates perfectly, but held-fold mismatch logits do not
remain above the most conservative negative tail. Requiring both heads to fire
amplifies this instability. The next hypothesis should introduce an explicitly
paired, interaction-aware representation and a group-robust ranking objective,
while retaining exact negative calibration and all existing behavioral gates.
Because V5.1's developmental audit has now been observed, it may be used only
as disclosed developmental evidence for a successor; untouched `operator_dev`
and protected-control stages must carry the confirmatory burden. V5.1 is not
eligible for behavior smoke, lock, or final test.

## AMSGE V2: full-state readout improves selectivity but not efficacy

AMSGE V2 tests a narrower diagnosis of the V1 failure: perhaps governance and
task state are present in the full hidden vector but discarded by the compact
PCA coordinates used by the V1 classifiers. V2 therefore trains normalized
full-state history and task heads while leaving the intervention sites,
reversible hook semantics, low-rank correction output, trust region, and base
model unchanged. Its fit result supports the accessibility part of that
diagnosis: aggregate fit gates pass without reading `operator_dev`, and the
editor checkpoint still contains no base weights.

The complete held behavior result rejects the stronger claim that better
state readout is sufficient for mitigation. On 3,072 untouched selection rows,
matched-minus-mismatched reduction is `-0.050456`, item-cluster bootstrap 95%
CI `[-0.074465,-0.026449]`, and normalized reduction is `-0.033615`, CI
`[-0.050991,-0.015907]`. The intervention helps obedience history on a
verification task by `+0.078613` but harms verification history on a delegated
task by `-0.109212`. Both label-swap halves must be positive; instead they are
`-0.003743` and `-0.097168`. Five of six benchmark aggregates are negative.
This is a statistically resolved reversal, not a threshold rejecting a small
uncertain benefit.

At the same time, V2 fixes the prominent V1 protected-control failure. All six
families pass their application-gated and forced-on checks across 2,856 unique
finite rows, including supported user authority. Exact non-application
identity remains `0.0`; the largest forced-on one-sided margin-loss upper bound
is `0.156250`, and the largest forced-on binary-KL upper bound is `0.003369`,
both inside their reference limits. Compared with V1, full-state heads and the
V2 fit therefore improve collateral selectivity and reduce the magnitude of
the aggregate behavioral harm, but they do not recover the intended symmetric
task-governance interaction.

The mechanistic implication is that readout accessibility and correction
validity are separate bottlenecks. A classifier can decode history/task state
well enough to pass its fit gates while the learned correction still follows
an entangled direction that is appropriate for one causal mismatch and wrong
for the reverse direction or one label encoding. Reconstruction-level teacher
improvement is likewise not a behavioral surrogate: it can coexist with a
held decision-margin reversal. The next method must identify and constrain the
sign of the correction separately by causal direction and label-swap orbit,
rather than relying on a shared signed scalar and common output map.

These post-fit behavior and control runs were performed under an explicitly
labeled `PROTOCOL_DEVIATION` because the fit's live Slurm record had expired
and persistent accounting was unavailable. The selection and controls
themselves have captured live `COMPLETED ExitCode=0:0` records and three
verified archive copies; the missing fit scheduler provenance is not
fabricated. Scientifically, V2 is a complete negative-result characterization:
protected selectivity passes, primary efficacy fails, and the editor cannot be
locked. `final_test_open=false`, `final_test_open_count=0`, and
`production_rollout_approved=false` remain binding.

## DSGE V3: expert efficacy succeeds, application routing fails

DSGE V3 tests the mechanism implied by the V1/V2 directional reversal. Instead
of asking one signed scalar and one correction map to serve both crossover
directions, it learns separate positive and negative directional experts and
applies them through a structural history-by-task router. At layer 27 MLP this
separation resolves the held efficacy bottleneck. Across 3,072 untouched
`operator_dev` rows, normalized gap reduction is `29.12%`, bootstrap 95% CI
`[27.67%,30.58%]`; matched-minus-mismatched reduction is `0.44572`, bootstrap
estimate `0.44604`, CI `[0.42367,0.46802]`. Both causal mismatch directions,
both label-swap halves, and all six benchmark aggregates improve. Matched
cells, matched selected logits, and the external coefficient-zero path retain
exact identity.

The 2,856-row protected experiment isolates a different failure. When either
directional expert is forced on independently, every family remains inside its
frozen margin-loss and binary-KL limits. The correction vectors are therefore
admissible under forced stress; V3 does not fail because either expert is an
intrinsically destructive direction. Application-gated behavior is also exact
for matched verification, matched delegated choice, supported user authority,
and factual boundary memory. The failure is concentrated in fresh verification
and explicit governance reset. The internal positive route activates on
384/384 fresh rows and 768/768 reset rows, changing selected logits on 324 and
639 rows respectively, while the negative route remains inactive. Thus exact
external zero gating is insufficient: an internal router can still assign a
nonzero governance-history state to a semantically fresh or explicitly reset
context.

This result sharpens the model of context mismatch. Direction-specific
corrections are both behaviorally effective and individually safe, but the
state estimator lacks a reliable internal notion of provenance invalidation.
It recognizes several protected matched and authority states, yet it does not
distinguish carried obedience history from the absence of carried state or an
explicit command to clear that state. The next mechanism should therefore add
a fresh/reset suppressor or abstention head that vetoes both directional routes
before the experts are applied. That suppressor must be learned and evaluated
without final-test data, and it must preserve exact identity on matched,
fresh/reset, and external-zero controls. Because the experts already pass
forced-direction stress tests, this is a router-repair hypothesis rather than a
new correction-direction hypothesis.

V3 cannot be locked despite its strong held efficacy: protected selectivity is
a necessary condition, and `controls_admissible=false`. No Pareto report or
final authorization follows from these results. `final_test_open=false`,
`final_test_open_count=0`, and `production_rollout_approved=false` remain
binding.

## ADSGE V4: strict abstention nearly separates applicability but fails exact audit identity

ADSGE V4 tests the router-repair hypothesis directly. It freezes the effective
V3 layer-27 MLP experts and their history/task readouts, then places one learned
binary applicability veto before both expert routes. A rejected route returns
the original activation exactly; there is no continuous rescaling. The
threshold is fixed as the smallest float32 value strictly above the maximum
calibration-negative logit, so calibration cannot trade a small false-positive
rate for additional mismatch recall. The final test remains unopened.

The terminal code-v33 fit shows that this representation is extremely close to
the desired separation. On the calibration fold, the selected linear veto has
zero active negatives and 100% mismatch recall in aggregate, in each causal
direction, and in each label-swap half. On the untouched audit fold, mismatch
recall and all directional/swap minima remain 100%, and every protected family
has exact zero activation: fresh verification, matched verification, explicit
governance reset, supported user authority, and factual boundary memory. The
fresh/reset failure that invalidated V3 is therefore removed at the fit-audit
level without modifying the V3 editor or the base model weights.

Exact governance-matched identity does not generalize perfectly, however. The
veto activates on 4/224 matched delegated-choice audit rows and 2/224 matched
verification audit rows. Those six false positives give an aggregate audit
negative active fraction of `0.005792`; because the frozen contract requires
zero activation in each governance-matched family, the sole failed gate is
`audit_each_governance_matched_family_exact_abstention`, and
`fit_eligible=false`. This is not evidence of numerical instability: all
10,152 rows and metrics are finite, the V3 maximum parameter change is exactly
zero, the checkpoint is editor-only, both Torch suites pass with zero skips,
and Job 9101 has a captured Slurm `COMPLETED ExitCode=0:0` record plus three
content-verified archive copies.

Mechanistically, V4 narrows the remaining problem from broad fresh/reset route
confusion to a small tail-overlap between mismatch and semantically matched
governance states. It does not establish behavioral mitigation because the
fit gate correctly stops before behavior smoke and `operator_dev`. Nor may the
audit fold be used to select one of the other calibrated architectures: doing
so would convert the held fit audit into a tuning set. The frozen V4 procedure
is therefore a scientific falsifier despite its near-separation. Any successor
must change the predeclared representation or training objective and obtain a
new untouched audit, rather than relaxing exact identity or tuning to these six
rows. `final_test_open=false`, `final_test_open_count=0`, and
`production_rollout_approved=false` remain binding.

## PAIR-GE V5.2: interaction capacity improves interpolation but destabilizes conservative calibration

PAIR-GE V5.2 tests the specific diagnosis left by V5.1: independent linear
heads may lack the boundary-by-context interactions needed to generalize
matched-versus-mismatched state. It freezes the effective V3 expert, constructs
662 features from the original 54 coordinates plus products, differences,
squared differences, and a boundary-by-context outer product, then fits a
shared nonlinear trunk with two veto heads. Weighted BCE is combined with a
smooth worst-group term, matched-counterfactual ranking, and top-negative-tail
separation. Six-fold cross-fit, exact negative-tail calibration, and every
V5.1 gate remain unchanged.

The terminal fit rejects that diagnosis as sufficient. Within each training
split, optimization is essentially complete: final row, paired-ranking, and
tail-separation losses are near zero. In the all-fold fit, the largest training
negative and smallest training positive logits are `-9.10012` and `11.33878`
for the all-negative head and `-50.24934` and `64.69508` for the matched-state
head. Thus the MLP can interpolate the fitted pairs and sources with a large
raw-logit gap while leaving V3 bitwise unchanged.

That separation is not stable under held-fold normalization. The largest
standardized held-negative logits are `9.30410` and `30.17385`, so the frozen
worst-tail-plus-margin rule raises the final thresholds to `62.47799` and
`84.95255`. The consensus consequently abstains on every one of 5,260
cross-fit negatives but activates on only `14.685%` of mismatches. Direction
and label-swap minima are the same `14.685%`. On the disclosed developmental
audit, all negative families still have exact abstention, while the consensus
accepts all negative-direction mismatches and no positive-direction
mismatches, producing 50% aggregate recall. The architecture has therefore
not discovered a fold-stable paired decision statistic; it has learned
high-margin, split-specific coordinates whose negative scale and location do
not transport under maximum-tail calibration.

This sharpens the V5.1 result. Extra nonlinear capacity and explicit pairwise
ranking do not solve the generalization problem when ranking constrains only
within-pair order while the deployment rule depends on absolute, independently
normalized head logits. In fact, the conservative tail needed for exact
abstention turns the fold instability into a stronger false-negative collapse
than V5.1. Exact negative identity, frozen-editor integrity, and checkpoint
safety all pass; the failure is specifically fold-robust sensitivity.

PAIR-GE V5.2 is therefore ineligible for behavior smoke, `operator_dev`,
protected controls, locking, or final test. Its developmental audit is retained
as non-confirmatory evidence and may not be relabeled as untouched. A successor
must predeclare a score or calibration rule that is invariant to fold-specific
logit location/scale, or learn the applicability decision through a genuinely
paired difference representation, without weakening exact negative abstention
or reusing the observed developmental audit as confirmatory evidence.
`final_test_open=false`, `final_test_open_count=0`, and
`production_rollout_approved=false` remain binding.

## C-DGE V4.1: composite abstention generalizes to the confirmatory split

C-DGE V4.1 turns the V4 diagnostic into a prospectively frozen composite
decision rule rather than relaxing either constituent gate. For a task-boundary
state \(x\), the editor applies only when

\[
g_{\mathrm{C-DGE}}(x)=g_{\mathrm{structural}}(x)\land
g_{\mathrm{applicability}}(x).
\]

This is an intersection-of-errors construction. A false positive from either
auxiliary classifier is harmless whenever the other gate maps the state to the
absorbing zero-edit branch. The protected controls confirm this mechanism:
the structural route accepts fresh/reset states that the applicability veto
rejects, while the applicability head accepts eleven governance-matched states
that the structural route rejects. The conjunction changes none of the 2,856
protected rows, with selected-logit maximum error exactly zero.

The one-time 6,144-row final test shows that this abstention does not obtain
safety by suppressing the useful correction. Both mismatch directions retain
positive recovery (44.31% and 47.09% raw matched-minus-mismatched reduction),
all six benchmarks improve, and equal-benchmark-weight normalized gap reduction
is 30.36% with bootstrap 95% CI 29.26--31.49%. Both matched task-aligned cells
remain exactly unchanged. Thus the supported mechanism is selective internal
state correction: a small layer-27 MLP intervention (mean relative norm 0.0713%)
is routed only where history-induced governance state and current task demand
conflict, while the logical veto restores an exact identity path elsewhere.

The final artifact lineage includes a transparent schema-only recovery. Job
9324 performed the sole model forward but failed after row generation because a
legacy writer omitted authorization identity fields. Job 9333 restored only
those fields in copied rows, changed no model-derived value, records
`model_forward_reexecuted=false`, and provides the terminal `COMPLETED 0:0`
record and three-copy receipt used for the claim. This supports the numerical
result without treating the unavailable terminal record of Job 9324 as if it
existed. Production rollout remains unapproved.

## ADSGE V4 post-fit behavior: the extra veto is operationally transparent on operator_dev

The terminal post-fit diagnostic separates the V4 head-level fit failure from
the behavior of the composed router. On the exact 3,072-row `operator_dev`
selection used for V3, V4 and V3 have bitwise-identical selected margins on
every row. V4 therefore retains V3's `29.12%` normalized gap reduction, both
directional recoveries, both label-swap recoveries, all six benchmark
reductions, and exact matched-cell identity. This is not merely equality of
rounded aggregates: zero of 3,072 edited rows differ and the maximum selected
margin difference is exactly `0.0`.

The reason is the logical composition of the two gates. The learned V4
applicability head accepts all 1,536 mismatch rows, so it never vetoes a V3
correction in the held behavior set. It also accepts 11 of 1,536 matched rows,
but the frozen structural V3 gate rejects every matched row. Consequently the
final conjunction `V3 structural route AND V4 applicability` is active on
1,536/1,536 mismatches and 0/1,536 matched rows. The six fit-audit false
positives, and the eleven analogous behavior-set acceptances, are false
positives of the auxiliary head rather than false positives of the composed
editor.

This result changes the engineering diagnosis but not the preregistered
decision. Requiring the auxiliary applicability head itself to abstain on
governance-matched states is stronger than required for composite behavioral
identity because the upstream structural gate already makes those states an
absorbing zero-edit region. The frozen V4 fit gate nevertheless failed and is
not reinterpreted after observing behavior, so V4 remains ineligible for a
lock or final test. The scientifically relevant remaining question is narrower:
whether the same applicability head vetoes the V3 structural route on fresh
verification and explicit reset states during a full forward, where the V3
gate was incorrectly active. That requires a protected-control behavior
diagnostic; `operator_dev` alone cannot answer it.

## ADSGE V4 protected controls: composite routing removes V3 collateral without changing its experts

The terminal 2,856-row protected-control diagnostic closes that remaining
behavioral question. Its application rule is a logical conjunction,

\[
g_{\mathrm{V4}}(x)=g_{\mathrm{V3\ structural}}(x)\;\land\;
g_{\mathrm{applicability}}(x).
\]

This composition matters because errors from the two gates need not become
editor errors. On fresh verification and explicit governance reset, V3's
structural route is active on 384/384 and 768/768 rows, respectively. The
applicability veto rejects every one of those 1,152 states, so V4 applies no
edit. On matched verification and matched delegated choice, the applicability
head accepts 5/768 and 6/768 rows, but the structural route rejects all eleven;
again the composite editor applies no edit. Supported authority and factual
boundary memory are rejected by both gates. Across all six families, the final
V4 route is active on 0/2,856 rows, selected logits are exactly identical to the
unedited model, and maximum selected-logit error is `0.0`.

The same-identity V3 comparison makes the improvement concrete. V3 changed
324/384 fresh-verification rows and 639/768 explicit-reset rows. V4 deletes all
963 of these collateral edits while leaving the other four already-exact
families unchanged. Its forced positive and negative expert outputs are
numerically identical to V3 in every family and pass all margin and binary-KL
reference gates. Together with the preceding 3,072-row `operator_dev`
diagnostic, V4 therefore preserves V3's `29.12%` normalized gap reduction,
directional recovery, label-swap recovery, and matched identity while removing
all observed protected-control application edits.

The mechanistic lesson is that auxiliary-head false positives are not
equivalent to composite edits. Exact abstention of the applicability head alone
is a sufficient but unnecessarily strong condition when an upstream structural
route already defines an absorbing zero-edit region. The frozen V4 fit audit
tested the stronger head-level property and thus produced an over-conservative
false negative for the composed behavioral system. This diagnosis is supported
by terminal full-forward evidence, but it does not rewrite the preregistered
decision after seeing the outcome. V4 remains a post-fit-failure diagnostic with
`fit_gates_passed=false`, `candidate_eligible=false`, and
`candidate_may_be_locked=false`; no Pareto lock or final-test claim follows.
`final_test_open=false`, `final_test_open_count=0`, and
`production_rollout_approved=false` remain binding.

## Prospective boundary test: local margin, not benchmark difficulty, predicts mismatch failure

The prospectively frozen test confirms the decision-boundary account on a
fresh 6,144-row `operator_dev` half that was not used to choose the margin
thresholds. For each task requirement, histories are paired while the item,
prompt payload, role, style, realization, and label swap are held fixed. The
analysis conditions only on whether the matched-history decision is correct and
then compares a frozen low-margin boundary band with a frozen high-margin
robust band.

The contrast is large for independent verification: 91/205 boundary cases
flip from correct to wrong under mismatch, versus 0/266 robust cases. The
boundary-minus-robust estimate is 44.39 percentage points, with a one-sided 95%
lower bound of 38.54 points. Delegated choice has a smaller but still
prospectively positive contrast: 21/440 versus 0/345, or 4.77 points, with a
one-sided lower bound of 2.54 points. Both preregistered directional tests pass.

This result distinguishes local susceptibility from broad task difficulty.
Across the six benchmark aggregates, difficulty--accuracy-drop Spearman
correlations are weak and nonsignificant under exact permutation tests
(`0.0857`, `p=0.9194` for independent verification; `0.3133`, `p=0.6000` for
delegated choice). A difficult benchmark can contain robust high-margin
decisions, and an easy benchmark can contain fragile boundary decisions.

The supported mechanism is therefore geometric and sample-specific. Context
mismatch shifts a carried governance state; a decision fails when the induced
readout displacement is large relative to the sample's original task-aligned
margin. Low matched margin is an observable susceptibility proxy, whereas
benchmark-level difficulty is at most a coarse ecological correlate. The
prospective experiment confirms this ordering without postselecting bands on
its outputs and without opening the final test.

## Qwen3.5-9B cross-model transfer: direction transfers, effect magnitude does not meet the frozen gate

The terminal Qwen3.5-9B `operator_dev` experiment tests whether the frozen
C-DGE construction transfers beyond the Qwen3-8B mitigation model. It preserves
the mechanism's qualitative signatures: both mismatch directions improve, both
label-swap orbits improve, all six benchmark aggregates are nonnegative, and
both governance-matched cells remain exactly unchanged. The intervention is
small (mean site-relative norm `0.0008585`, maximum `0.0050642`) and its
normalized gap reduction is 19.05% with bootstrap 95% CI 18.47--19.61%.

This is genuine partial transfer, but it is not the preregistered strength of
transfer. The raw matched-minus-mismatched reduction is `0.19157`, with lower
95% confidence bound `0.18571`, below the frozen `0.25` requirement. The miss
is not attributable to a sign reversal, one label assignment, one benchmark,
matched-cell collateral, nonfinite values, or incomplete execution: all of
those gates pass in the terminal 3,072-row audit. It instead indicates that the
same composite routing and directional-editing design has materially weaker
effect size in the Qwen3.5-9B representation.

The supported mechanistic claim is therefore architecture-bounded. C-DGE's
logical intersection preserves an absorbing zero-edit path across both tested
models, and the learned direction is useful in both, but quantitative efficacy
does not automatically transport at the Qwen3-8B level. The correct next
hypothesis is representation-specific refitting or site/rank selection under a
fresh protocol, not relaxation of the existing gate. Because the frozen
behavior gate fails, Qwen3.5 protected controls, performance evaluation, lock,
and final test remain closed; this negative result enters the paper as a
cross-model limitation rather than being hidden or converted into a weaker
success criterion.

## External comparators: generic interventions confound correction with state replacement

The complete formal same-identity external comparison contains two input
controls and four `official-code-derived task adaptations`; the latter are not
byte-identical official reproductions. It separates three qualitatively
different failure modes that a single aggregate NGR number would conceal.
Input-level replacement methods can score a large nominal ratio while erasing
useful matched behavior: session isolation reaches 40.96% NGR but damages both
mismatch directions and both matched cells. Unconditional representation
steering can similarly move every cell in a favorable global direction: RePS
reaches 10.38% NGR but moves the two matched reference cells by `+9.919271` and
`+0.424316`. CAST reaches 5.02% NGR but remains slightly negative in one
mismatch direction and fails exact matched identity. CAA is near zero
(-0.54%, 95% CI [-1.79%, +0.70%]), while LoReFT produces exactly zero effective
change under the frozen correction cap. The earlier code-v87 CAA/CAST/LoReFT/
RePS results are retained only as exploratory `adapter-only` appendix evidence
and are excluded from this formal comparison.

This supports the mechanism assumed by DGE rather than a generic claim that
larger activation movement is better. Context mismatch is task-directional:
the desired correction changes sign with the history--task relation, and the
same state must remain untouched when governance is already matched. A valid
editor therefore needs both directional experts and an explicit application
rule. The external baselines are useful precisely because they show that
prompt clearing, session replacement, contrastive directions, conditional
activation scaling, low-rank representation editing, and representation
steering each capture only part of this structure. All six were evaluated with
bit-identical baseline logits and exact paired identities, so this conclusion
does not depend on different examples or starting predictions. The four
official-code-derived task adaptations bind fixed upstream commits and official
source archives, while adapting their core algorithms to the paired
ContextMismatch objective. This lineage improves comparability without
claiming an unmodified reproduction of each source paper's original task,
data, or hyperparameter regime.

## Qwen3.5 native rediscovery: efficacy adapts, reset provenance remains unresolved

The V4.2 native pipeline tests the stronger transfer hypothesis suggested by
the earlier 19.05% direct-transfer result: rediscover the causal site and refit
the editor inside the target architecture before applying it. This succeeds on
effect magnitude. The target-specific scan selects layer-31 MLP, and two frozen
tied candidates reach 27.06% normalized reduction with exact matched identity,
substantially closing the gap to Qwen3-8B.

Protected controls identify the remaining architecture-specific failure. Both
native candidates pass forced-on correction safety and all families except
explicit governance reset, where the application route induces selected-logit
error up to 0.125. Thus the learned correction vectors are not intrinsically
unsafe and the architecture-adaptive site search is effective; the unresolved
component is provenance invalidation. The Qwen3.5 route does not reliably
recognize that an explicit reset cancels previously carried governance state.

The cross-model lesson is consequently more precise than either ``transfer
works'' or ``transfer fails.'' Target-model rediscovery can recover most of the
lost efficacy, but target-specific routing and protected-control validation
remain necessary. Layer adaptation alone is insufficient. The frozen Pareto
lock correctly rejects both tied candidates, so the result supports the method's
discovery logic while falsifying its current claim to architecture-general safe
deployment.
