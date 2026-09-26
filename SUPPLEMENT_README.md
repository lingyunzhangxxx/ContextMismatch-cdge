# Anonymous code and result supplement

After extracting this ZIP, open a terminal in the extracted directory and run:

```bash
./run.sh review
```

Python 3.9+ is sufficient. This command performs offline checksum, row-identity,
numeric and paired-design checks, recomputes the primary Qwen3-8B C-DGE final
values, and scans the published files for credentials and identifying paths.
No API key, model download, or accelerator is needed.

Expected primary output:

```text
final mismatch accuracy: 74.48% -> 75.68% (3072 rows)
final normalized gap reduction: 30.36%
rescued mismatch flips: 32 / 135
protected controls changed: 0 / 2856
```

Run `./run.sh reproduce` to produce `generated/final_paper_metrics.json`.
For implementation tests in a lightweight Python 3.11 container:

```bash
docker build -t context-mismatch-review .
docker run --rm context-mismatch-review
```

The ZIP includes the original scientific tensor archives in `tensor-assets/`.
To verify every tensor and restore the final experimental directory tree:

```bash
./run.sh records restore --asset-dir tensor-assets
```

`./run.sh records metrics` recomputes the primary final test, six official
comparators and seven reported ablations.
The CPU tensor-test environment is documented in `README.md` and
`docs/REPRODUCIBILITY.md`. Evidence coverage is listed in `RESULTS.md`.
Full runtime KV caches were not persisted by the original mechanism scan;
saved fitting activations and gradients are included, with that distinction
made explicit.

The ZIP excludes Git metadata, cluster launchers, operational snapshots and
execution diaries. Fixed third-party source files retain their original
licenses and attribution.
