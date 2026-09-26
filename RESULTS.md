# Released results

`results/` contains the compact, claim-bearing outputs retained from the frozen
experiment snapshot. It includes:

- counterbalanced context-mismatch discovery and disjoint replication;
- full behavioral, residual, component, and cache analyses;
- DGE/C-DGE behavior, protected-control, boundary, and system summaries;
- exact same-identity comparisons and external-baseline summaries;
- row-level DGE development and protected-control tables;
- the consolidated `paper_claims.json` used to render paper values.

## Final paper evidence

| Released file | Evidence | Size of evaluated population |
|---|---|---|
| `cdge_v4_1_final_test.jsonl` and `.analysis.json` | Locked C-DGE final test | 6,144 rows, including 3,072 mismatched rows |
| `cdge_v4_1_protected_controls.jsonl` and `.analysis.json` | Fresh, matched, reset, authority, and memory controls | 2,856 rows across six families |
| `cdge_v4_1_behavior_operator_dev.analysis.json` | Formal development selection | 3,072 rows |
| `cdge_v4_1_performance.jsonl` and `.analysis.json` | Original Ascend performance measurements | 720 rows, 24 cells |
| `prospective_boundary.jsonl` and `.analysis.json` | Preregistered frozen-margin confirmation | 6,144 rows |
| `cdge_final_boundary_rescue.json` | Descriptive final boundary rescue | 32 of 135 mismatch-induced flips rescued |

`./run.sh reproduce` recomputes final mismatched accuracy (2,288/3,072 to
2,325/3,072), normalized gap reduction (30.36%), matched identity, and formal
protected-control identity directly from the released rows. It checks these
values against the frozen summaries and writes the result under `generated/`.
The net accuracy gain is 37 decisions; 32 of those recover mismatch-induced
flips. These are different populations and must not be conflated.

The older `adsge_v4_*_diagnostic.json` files are retained development diagnostics.
They are distinct from the later formal `cdge_v4_1_*` evaluations. Qwen3.5-9B
architecture-native results include the retained explicit-reset control failure;
the release does not replace that outcome with a successful-transfer claim.

The release validator parses every JSON and JSONL file, verifies the SHA-256
manifest with exact file coverage, rejects nonfinite numeric outputs, checks
row uniqueness where an experiment key is present, and confirms that
the consolidated artifact does not mark the system as approved for production.

The following are deliberately not included:

- model weights or tokenizer caches;
- raw hidden-state tensors and accelerator caches;
- private cluster receipts, SSH details, credentials, and local paths;
- reinstallable environments or third-party repositories.

The released results support audit and paper-number reproduction. Re-running
the accelerator-scale mechanistic experiment requires the model/runtime setup
described in the README and the corresponding upstream model licenses.
