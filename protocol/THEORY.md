# Theory: task-boundary state reuse and context mismatch

## The two-level claim

The paper makes a deliberately two-level claim. At the abstract level, it
studies **context mismatch**: a task-scoped control state induced by one
interaction remains active after a task boundary and is interpreted as if it
were still compatible with the next task. At the concrete level, it tests one
controlled instance of this mechanism, in which the state concerns interaction
governance (whether the model should independently verify or honor an
authorized delegation). The theory is therefore neither a universal theory of
context nor merely a description of one benchmark. It is a small, testable
mechanistic account whose governance experiment instantiates its variables.

The useful abstraction is the **boundary-conditioned state-reuse principle**:
carryover becomes harmful when (i) the prior interaction changes a reusable
state, (ii) the current task reads that state differently, and (iii) the
adverse state-induced shift is large relative to the current decision margin.
This principle can describe other task-scoped states, such as role or
authority, but this paper claims and tests it only for governance state.

The abstraction is useful because it separates two questions that are often
conflated:

1. Is the current-task evidence still available?
2. Is the history-induced computational state still appropriate for the
   current task?

Context mismatch is a positive answer to the first question together with a
negative answer to the second, within the tested task family. It is not defined
as generic prompt distraction, forgetting, or an unconditional obedience
penalty.

## State-reuse model

Let H be a completed history, x the current evidence and candidates, and
T in {I,D} the current requirement, where I is independent verification and D is
delegated choice. Let z_H denote the history-induced governance component of the
internal state. At a local decision site:

m(H,x,T) = b_T(x) + r_T(x)^T z_H + epsilon.

Here b_T(x) is the margin supported by the current task, r_T(x) is the
task-conditioned readout of the carried state, and epsilon is the local
approximation error. A valid boundary transition replaces the out-of-scope
component of z_H with the state appropriate for T; mismatch is the effect of
reusing it. This notation does not assert that the model literally stores one
vector or that all computation is linear. It isolates the smallest functional
coordinate needed to connect the behavioral interaction, the margin analysis,
and the internal-state interventions.

This is a functional model, not a claim that the model stores a single
governance vector or that all computation is linear. Cache, residual, and
component exchanges are interventions used to test whether the proposed
functional state is causally load-bearing.

## Proposition 1: crossover factorization

**Boundary-conditioned state-reuse hypothesis.** Context mismatch should
appear as an interaction, not as a one-directional obedience penalty: a
history-induced state must persist, the current task must read it differently,
and its adverse contribution must be large relative to the current margin.
The governance crossover, margin-band prediction, and bidirectional state
interventions are three tests of this same hypothesis.

Let Delta z = z_O - z_V be the difference between obedience- and
verification-history states. Subtracting the two histories within each current
task gives:

Gamma = Delta z^T (r_D(x) - r_I(x)) + epsilon_Gamma.

The symmetric crossover therefore requires two factors:

1. the history must change the carried state (Delta z != 0);
2. the current task must read that state differently (r_D != r_I).

A history effect without this interaction is not sufficient evidence for
context mismatch. The model predicts that the same history can help one task
and hurt another, because compatibility is relational between the carried state
and the current task.

## Proposition 2: boundary crossing

Let m_0(x,T)>0 be the margin under valid governance and let
Delta_{H,T} be the adverse contribution of the reused state. A
matched-correct item can flip only if:

0 < m_0(x,T) <= -Delta_{H,T}.

The relevant susceptibility ratio is:

rho(H,x,T) = [-Delta_{H,T}]_+ / max(m_0(x,T), epsilon).

Thus rho >= 1 predicts boundary crossing. This is why a benchmark-level
difficulty score need not predict mismatch risk: difficulty matters only insofar
as it produces a small positive decision margin.

## Proposition 3: symmetric editing constraint

Suppose the two mismatch transitions require opposite corrections, d and -d,
and matched states must remain unchanged. A history- and task-independent
edit cannot guarantee both corrections and exact matched-state identity when
d != 0. A valid repair must therefore condition on the history--task pair,
represent both directions, and abstain when no conflict is detected.

This proposition motivates the architecture of DGE and C-DGE. It does not say
that every one-direction dataset defeats a global vector.

## Empirical correspondence

The theory makes three claim-bearing predictions:

1. a symmetric history-by-task crossover, replicated on a disjoint split;
2. higher flip rates in frozen low-margin bands than in robust bands;
3. bidirectional state exchange that rescues one mismatch and induces the
   reverse failure.

The current protocol maps these predictions to the behavioral crossover,
prospective boundary test, and exact paired cache/residual/component
interventions. Label swaps, evidence-order reversal, matched histories,
explicit resets, and factual-memory controls separate state reuse from answer
priming, lexical contamination, and irreversible evidence loss.

## From theory to mitigation

The privileged counterfactual repair is the matched-state replacement:

delta h*(H,x,T) = h*(x,T) - h^0(H,x,T).

It is a diagnostic oracle because it requires a second matched execution.
C-DGE is an amortized approximation that predicts only the task-relevant
projection of this correction from the ongoing run. Its selectivity veto is a
repair invariant, not part of the causal claim about the phenomenon. The
method follows from the theory rather than defining it: the theory says that a
repair must be conditional on the boundary state and current task, while the
governance experiment supplies the two directions and the protected-state
criterion used to implement that repair.

## Security boundary

An adversarially constructed prior history could instantiate the same mismatch
mechanism as a cross-task influence risk. This is an implication and a future
threat model, not the primary claim of this paper. The paper must not call
C-DGE a general prompt-injection defense or claim universal security without a
separate attacker-controlled evaluation.

The theory is supported only within the stated governance setting and model
scope. It does not identify a unique latent variable, prove universality across
architectures, or establish a causal effect in natural human conversations.
