# Released results

`results/` contains the compact, claim-bearing outputs retained from the frozen
experiment snapshot. It includes:

- counterbalanced context-mismatch discovery and disjoint replication;
- full behavioral, residual, component, and cache analyses;
- DGE/C-DGE behavior, protected-control, boundary, and system summaries;
- exact same-identity comparisons and external-baseline summaries;
- row-level DGE development and protected-control tables;
- the consolidated `paper_claims.json` used to render paper values.

The release validator parses every JSON and JSONL file, verifies the SHA-256
manifest, checks row uniqueness where a `job_key` is present, and confirms that
the consolidated artifact does not mark the system as approved for production.

The following are deliberately not included:

- model weights or tokenizer caches;
- raw hidden-state tensors and accelerator caches;
- private cluster receipts, SSH details, credentials, and local paths;
- reinstallable environments or third-party repositories.

The released results support audit and paper-number reproduction. Re-running
the accelerator-scale mechanistic experiment requires the model/runtime setup
described in the README and the corresponding upstream model licenses.
