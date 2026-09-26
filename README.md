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

This validates every released JSON/JSONL result and prints an auditable summary.
For the complete code test in an isolated Python 3.11 environment:

```bash
docker build -t context-mismatch-cdge .
docker run --rm context-mismatch-cdge
```

The container runs the release validator, 141 unit tests, and the privacy
audit. Tests that require PyTorch are skipped in the lightweight image; use the
optional experiment environment below to run them.

## Repository layout

```text
.
├── analysis/       # Paper-result aggregation and figure generation
├── artifacts/      # Public benchmark manifests and design audits
├── ascend/         # Sanitized terminal verification/archive helpers
├── protocol/       # Frozen experiment contracts and task definitions
├── results/        # Released row-level and summary experiment outputs
├── scripts/        # Behavioral, mechanistic, C-DGE, and baseline code
├── snapshots/      # Frozen code snapshot used by later diagnostics
├── tests/          # Contract and implementation tests
├── Dockerfile
└── run.sh          # Single entry point
```

Large model weights, caches, private credentials, machine-specific receipts,
and raw cluster archives are intentionally excluded. The released results
contain the compact claim-bearing rows and summaries needed to audit the paper.

## Local commands

Python 3.10 or newer is required for the full test suite.

```bash
./run.sh demo       # validate and summarize released results
./run.sh test       # run the unit suite
./run.sh audit      # scan tracked files for secrets/private paths
./run.sh all        # run all three checks
```

The behavioral pilot talks to any OpenAI-compatible endpoint. Credentials are
read only by that endpoint; this repository does not require or store them.

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
```

The recorded Ascend environment used Python 3.11.13, PyTorch 2.7.1,
torch-npu 2.7.1, and Transformers 5.13.1. NPU users should install the wheels
matching their CANN/runtime image rather than the CUDA requirements file.

## Results and integrity

See [RESULTS.md](RESULTS.md) for the released evidence map and interpretation
limits. `results/MANIFEST.sha256` binds every result file. Run `./run.sh demo`
to verify the manifest and parse every JSON/JSONL row.

All retained formal evidence has `production_rollout_approved=false`. The code
is a research artifact, not a deployment-safety claim.

## Citation

Please cite the accompanying paper by title. Author and proceedings metadata
will be added after the anonymous review period.

## License

Code in this repository is released under the [MIT License](LICENSE). Dataset
and model licenses remain with their original providers; see the frozen
protocol manifests for source revisions.
