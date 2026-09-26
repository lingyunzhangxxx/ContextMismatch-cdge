# Reproduction paths

## Reproduce the final paper numbers

From a fresh clone, run `./run.sh` to verify checksums, row identities, numeric
finiteness, paired task construction, and the final paper's primary values.
This requires only Python 3.9+ and does not contact a model service.
Run `./run.sh records metrics` to verify the complete selected evidence index
and recompute the official comparators and reported ablations
against their frozen reports. Candidate screening
history is retained as a frozen selection report; unused candidate records are
excluded. Coverage is listed in `RESULTS.md`.

Run `./run.sh reproduce` to write `generated/final_paper_metrics.json`. It
recomputes the 6,144-row final test as 3,072 matched/mismatched pairs, checks
candidate and suffix identity across all four history/task cells, averages
normalized gap reduction equally across benchmark families, and recomputes
exact preservation of the 2,856 formal protected controls.

Accuracy uses a strict positive task-aligned margin; zero-margin ties count as
failures. The reported 74.48% to 75.68% accuracy refers to the 3,072 mismatched
rows, not the entire 6,144-row test. Conditional rescue (32/135) differs from the
net increase in correct decisions (37/3,072).

Install the analysis extra and run `python analysis/summarize_and_plot.py` to
regenerate `generated/paper_claims.json` and the historical aggregate figures.
The claims file includes the final C-DGE test, formal controls, prospective
boundary confirmation, and performance summaries. The DGE comparison figures
describe development experiments; they are not a rerun of the final manuscript's
later routing-geometry artwork.

## Verify the implementation

The lightweight Docker image uses Python 3.11 and runs `./run.sh all`, with
tensor-dependent tests explicitly skipped. To exercise all tests on CPU:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-experiment.txt
./run.sh all
```

No model download is needed for the tests. They exercise directional experts,
protected-subspace projection, gradient capture through a frozen model,
identity gates, norm caps, checkpoint loading, native Qwen3.5 site handling,
and the fixed upstream baseline classes. The direct dependency versions are
pinned to the CPU environment verified for this release.

## Run new model experiments

The `pilot` and `length-scan` commands in the README perform new behavioral
inference against an OpenAI-compatible local endpoint. They are distinct from
offline validation of the released results.

The full frozen experiment implementation is in versioned modules:

| Stage | Modules |
|---|---|
| Dataset construction, behavior and causal scans | `scripts.benchmark_v1`, `scripts.benchmark_v2` |
| Paired state and protected activation capture | `scripts.benchmark_v3.run_governance_capture`, `scripts.benchmark_v1.run_protected_capture` |
| Task-margin gradients and directional expert fitting | `scripts.benchmark_v4.run_margin_gradient_capture`, `scripts.benchmark_v4.fit_directional_governance` |
| Applicability veto and composite eligibility | `scripts.benchmark_v5.fit_abstaining_router`, `scripts.benchmark_v9.audit_composite_eligibility` |
| C-DGE behavior, controls, performance and final test | `scripts.benchmark_v10` |
| Prospective frozen-margin confirmation | `scripts.benchmark_v11` |
| Native Qwen3.5 rediscovery and refitting | `scripts.benchmark_v12` through `scripts.benchmark_v18` |
| External comparator adaptations | `scripts.benchmark_v19`, `scripts.benchmark_v20` |

Invoke the modules with `python -m`, for example:

```bash
python -m scripts.benchmark_v3.run_governance_capture --help
python -m scripts.benchmark_v4.fit_directional_governance --help
python -m scripts.benchmark_v10.run_cdge_v4_1_final --help
python -m scripts.benchmark_v13.run_qwen35_cdge_performance --help
```

Accelerator-scale reruns need locally obtained model weights and newly
SHA-bound execution manifests. The original saved activation/gradient tensors
and fitted editor checkpoints can be downloaded and restored with:

```bash
./run.sh records download
./run.sh records restore --asset-dir generated/tensor-assets
```

For an anonymous ZIP containing `tensor-assets/`, use that directory instead.
Restoration verifies archive hashes and every tensor member, rejects unsafe
archive paths, and reconstructs the source-relative scientific directory tree
under `generated/evidence/`. The final editor's checkpoint hash is explicitly
checked against the final-test environment by `records metrics`.

Load these locally produced files with `torch.load(path, map_location='cpu',
weights_only=True)`. They contain captured scientific states and learned
editors, never foundation model weights.

The residual/component mechanism scanner used transient prefix KV caches and
suffix activation captures in memory. Those full runtime caches were never
persisted. The release includes the original per-intervention outputs and all
saved fitting/protected activations and task-margin gradients. Re-running the
mechanism modules recreates the transient caches; the saved fit captures must
not be described as those full mechanism-scan KV snapshots.

The frozen contracts enforce split isolation, model identity, and one-time final
evaluation. Their `/workspace/...` paths are public replacements for historical
cluster locations; they are not ready-made credentials or runnable job receipts.
Configure paths for your runtime and rebuild bindings for a new experiment.
Do not present a new run as the original frozen run.

This release verifies the original outputs and executes all implementation
tests on CPU. It does not claim that the entire Ascend experiment was rerun on
the release machine or that the CUDA model path was tested at paper scale.

## Source integrity

`artifacts/release_provenance.json` binds restored code and terminal result
copies to their original frozen source hashes and their sanitized release
hashes. The release combines the canonical source with later fixes from the
complete historical archive; the capture-authorization compatibility fix is
retained from the canonical source. Numeric experiment values and job keys are
unchanged. `results/MANIFEST.sha256` covers every released result file.
