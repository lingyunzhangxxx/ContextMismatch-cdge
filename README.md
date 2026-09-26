# ContextMismatch-C-DGE

Reproducible code and released results for:

> **Context Mismatch in LLMs: Stale Interaction Policies Across Task Boundaries**

This repository studies whether an interaction policy learned in one task
remains active after the conversation moves to a different task, and includes
the implementation of Composite Directional Governance Editing (C-DGE).

## One-command start

No model download, GPU, or API key is required for the public smoke test:

```bash
./run.sh
```

This validates every released JSON/JSONL result and recomputes the primary
Qwen3-8B final-test values from released model-output rows:

```text
final mismatch accuracy: 74.48% -> 75.68% (3072 rows)
final normalized gap reduction: 30.36%
rescued mismatch flips: 32 / 135
protected controls changed: 0 / 2856
```

It is an offline result-reproduction command. New model inference is described
below. Python 3.9+ is sufficient for this command; tests require Python 3.10+.

For the complete code test in an isolated Python 3.11 environment:

```bash
docker build -t context-mismatch-cdge .
docker run --rm context-mismatch-cdge
```

The container runs the release validator, the unit suite, and the privacy
audit. Tests that require PyTorch are skipped in the lightweight image; use the
optional experiment environment below to run them.

The verified full CPU environment passes **191 tests with no skips**. The
lightweight container explicitly skips 85 tests that need tensor dependencies.

## Repository layout

```text
.
├── analysis/       # Paper-result aggregation and figure generation
├── artifacts/      # Public benchmark manifests and design audits
├── docs/           # Reproduction paths and experiment-module map
├── protocol/       # Frozen experiment contracts and task definitions
├── records/        # Deduplicated final evidence and tensor asset index
├── results/        # Released row-level and summary experiment outputs
├── scripts/        # Behavioral, mechanistic, C-DGE, and baseline code
├── tests/          # Contract and implementation tests
├── third_party/    # Fixed upstream baseline files and original licenses
├── Dockerfile
└── run.sh          # Single entry point
```

Final experimental records are indexed in `records/index.json`: unchanged
row outputs, scientific environment reports, activation/gradient captures,
and fitted editor checkpoints. Repeated copies, unused candidate checkpoints,
smoke runs, and unrelated failed attempts are excluded. Large scientific
tensors are distributed as checksum-bound Release assets; foundation model
weights and deployment credentials are excluded. See [the evidence map](RESULTS.md).

## Local commands

Python 3.10 or newer is required for the full test suite.

```bash
./run.sh demo       # validate and summarize released results
./run.sh review     # offline reviewer check: evidence and publication privacy
./run.sh test       # run the unit suite
./run.sh audit      # scan tracked files for secrets/private paths
./run.sh all        # run all three checks
./run.sh reproduce  # write recomputed final metrics under generated/
./run.sh records metrics   # recompute final, baseline, ablation and native values
./run.sh records download  # download and verify the original scientific tensors
./run.sh records restore --asset-dir generated/tensor-assets
```

If your default Python is older than 3.10, select a newer interpreter explicitly
for tests, for example `PYTHON=python3.11 ./run.sh all`, or use Docker.

The behavioral pilot talks to a local OpenAI-compatible endpoint without an
authentication header. Model-service credentials belong in your private service
configuration, not in this repository.

```bash
python3 scripts/run_experiment.py \
  --base-url http://127.0.0.1:8127/v1 \
  --model YOUR_MODEL_ID \
  --replicates 3 \
  --workers 12
python3 scripts/analyze.py
```

The counterbalanced history-length experiment is:

```bash
python3 scripts/run_length_scan.py \
  --base-url http://127.0.0.1:8127/v1 \
  --depths 1 8 32 64 \
  --conditions verification obedience obedience_reset \
  --realizations 3 \
  --workers 12
python3 scripts/analyze_length_scan.py
```

To regenerate the consolidated claims artifact and paper figures from the
released results:

```bash
python -m pip install '.[analysis]'
python analysis/summarize_and_plot.py
```

Generated files are written under `generated/`; the immutable files in
`results/` are not overwritten.

## Full model environment

The lightweight validation path uses only the Python standard library. Full
mechanistic and C-DGE runs require PyTorch, NumPy, Transformers, sufficient
accelerator memory, and locally obtained Qwen weights. Install the optional
CUDA-oriented dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-experiment.txt
./run.sh all
```

The recorded Ascend environment used Python 3.11.13, PyTorch 2.7.1,
torch-npu 2.7.1, and Transformers 5.13.1. NPU users should install the wheels
matching their CANN/runtime image rather than the CUDA requirements file.

See [the reproduction guide](docs/REPRODUCIBILITY.md) for the complete C-DGE
module map, required capture inputs, and the distinction between offline
reproduction, CPU implementation tests, and accelerator-scale reruns.

## Results and integrity

See [RESULTS.md](RESULTS.md) for the released evidence map and interpretation
limits. `results/MANIFEST.sha256` binds every result file. Run `./run.sh demo`
to verify the manifest and parse every JSON/JSONL row.

All retained formal evidence that declares a production decision has
`production_rollout_approved=false`. The code
is a research artifact, not a deployment-safety claim.

## Citation

Please cite the accompanying paper by title. Author and proceedings metadata
will be added after the anonymous review period.

## License

Project code is released under the [MIT License](LICENSE). Vendored comparator
files retain their [upstream licenses](third_party/README.md). Dataset
and model licenses remain with their original providers; see the frozen
protocol manifests for source revisions.
