# Final experimental evidence

`results/` provides the compact paper reports and primary row tables.
`records/index.json` supplies the selected final experiments and the scientific
inputs needed to reproduce them. Every row file is byte-identical to its
retained original. Text metadata has personal filesystem locations redacted;
the index records both the original and public hashes. No numerical values or
row identifiers were changed.

Repeated content is stored once. Files already present in `results/` are
referenced there rather than copied into compressed storage. Smoke runs,
failed experiments and unused editor checkpoints are excluded. Frozen
candidate-grid and selection reports remain as evidence of the selection rule.
Cross-model experiments that did not pass the final protected-control check
are excluded from this successful-run release.

The final index contains 224 distinct scientific source files: 27 row tables
with 149,600 rows, 96 JSON reports/manifests, and 101 original tensor files.
Tensor assets total approximately 489 MiB. Dataset manifests and required
fitting inputs are included in these counts.

| Final experiment | Released evidence | Offline check |
|---|---|---|
| Locked Qwen3-8B C-DGE final test | 6,144 complete output rows, environment, identity audit, final lock and actual runtime editor | Accuracy, NGR, matched identity, boundary rescue and checkpoint binding |
| Formal protected controls | 2,856 rows in six families | Exact gated identity |
| Formal development and performance | 3,072 development rows; 720 performance rows | Original measurements and analysis reports |
| Prospective boundary confirmation | 6,144 rows and frozen-margin manifest | Original measurements and final report |
| Behavioral discovery and replication | Full behavior and two 6,144-row counterbalanced crossover runs | Original outputs, scientific manifests and analysis code |
| Causal residual/component scans | 27,648 residual and 18,432 component intervention rows | Original logits, intervention parameters and analysis code |
| Official CAA/CAST/LoReFT/RePS and two input controls | Six 3,072-row output tables and original fitted baseline state | Per-method NGR and accuracy recomputation |
| Directional core and reported ablations | Seven same-identity methods, 3,072 rows each | Per-method NGR and accuracy recomputation |
| Scientific fitting inputs | Original paired/protected activations and task-margin gradient shards | Archive/member SHA-256 and safe tensor loading |

Run `./run.sh records metrics` to verify the complete index and produce
`generated/complete_record_metrics.json`. It recomputes point estimates from
rows and compares them with the frozen reports. Confidence intervals remain
in the original reports and can be regenerated using the versioned analysis
modules with their recorded bootstrap seeds and replicate counts.

The primary final test reproduces 2,288/3,072 to 2,325/3,072 mismatched correct
decisions (74.48% to 75.68%), 30.36% NGR, 32/135 rescued mismatch-induced flips,
and zero changes in 2,856 protected controls. The net gain of 37 decisions and
the 32 rescued mismatch-induced flips concern different populations.

Scientific tensors are available in the repository's **Complete experimental
records v1** Release. `records/index.json` specifies each asset URL and hash,
plus every original tensor's source path and hash. Use:

```bash
./run.sh records download
./run.sh records restore --asset-dir generated/tensor-assets
```

Restoration reconstructs the scientific directory tree under
`generated/evidence/`, including the original final editor checkpoint and its
frozen directional parent. No foundation model weights or tokenizer cache is
included. Obtain model weights separately under the applicable model license.

The original residual/component scanner did not persist full runtime prefix
KV caches or all suffix activations. It retained per-intervention outputs;
separate fitting/protected activation and gradient tensors were saved and are
included here. Those are different artifacts. Re-running the scanner recreates
its transient caches. This release does not claim to supply snapshots that
were never saved, or to have rerun the full Ascend experiment on the release
machine.

`results/MANIFEST.sha256` and `records/MANIFEST.sha256` enforce exact file
coverage. The reviewer command checks hashes, finite row values, paired task
construction, final editor binding and scientific report agreement. Scientific
state files must be loaded with `torch.load(..., weights_only=True)`.
