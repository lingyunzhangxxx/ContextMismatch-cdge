# Release verification — 2026-09-26

The release was checked in an isolated macOS arm64 environment with Python
3.12.14, PyTorch 2.7.1, Transformers 5.13.1, NumPy 2.2.6, Accelerate 1.15.0,
ReportLab 4.4.10, and Datasets 4.8.5.

| Check | Verified result |
|---|---|
| Complete CPU unit suite | 192 passed; zero skipped |
| Lightweight Python 3.11 Docker suite | 107 passed; 85 tensor-dependent tests explicitly skipped |
| Release checksums | All 46 result files covered |
| Result parsing | 38 JSON files; 6 JSONL files; 21,792 JSONL rows |
| Final C-DGE row reproduction | 74.48% to 75.68% mismatched accuracy; 30.36% NGR; 32/135 flips rescued |
| Formal controls | 0/2,856 changed across six families |
| Consolidated claims regeneration | Byte-identical to the released `paper_claims.json` |
| Core capture, fit, final-test and cross-model performance CLI imports | All four `--help` invocations passed |
| Publication privacy scan | No blocked credentials, private paths, or model-weight files found |

Gitleaks 8.24.3 is used for an additional scan of both Git history and current
files. Its default generic-key detector also flags scientific hashes, timed
measurement identifiers, and one literal model-class import. `.gitleaks.toml`
exempts only those specific formats; provider credential detection remains
enabled. JSONL hash fragments at scanner chunk boundaries are handled by the
same field-specific exemptions. Public result checks still verify the complete
files and hashes independently.

These checks reproduce published outputs and test implementation behavior.
They do not measure new Qwen inference or rerun the full Ascend training job.
