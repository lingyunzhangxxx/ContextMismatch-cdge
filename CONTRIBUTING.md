# Contributing

Use Python 3.10 or newer and keep changes deterministic. Before opening a pull
request, run:

```bash
./run.sh all
```

Do not commit model weights, raw private transcripts, credentials, machine-
specific paths, or generated environments. New result files must be documented
in `RESULTS.md` and included in `results/MANIFEST.sha256`.
