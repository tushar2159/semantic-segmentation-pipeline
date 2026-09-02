# Semantic Segmentation Pipeline

Config-driven end-to-end semantic segmentation pipeline with training, evaluation, prediction, tests, CI, and release governance.

**Portfolio intent:** demonstrate clean AI engineering architecture without publishing client data, employer code, private model weights, commercial AOIs, or proprietary production metrics.

## Architecture

```text
config/default.yaml
        │
        ▼
  data preparation
        │
        ▼
   TinyUNet
        │
   ┌────┴────┐
   ▼         ▼
evaluate   predict
   │         │
   └────┬────┘
        ▼
  release manifest
```

## Repository layout

```text
.
├── .github/workflows/ci.yml
├── config/
│   └── default.yaml
├── releases/
│   └── v0.1.0/manifest.json
├── src/segmentation_pipeline/
├── tests/
├── tools/
├── ENGINEERING_PROCESS.md
├── EXPERIMENTS.md
└── pyproject.toml
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## Runbook

```bash
segmentation-pipeline prepare
segmentation-pipeline train
segmentation-pipeline evaluate
segmentation-pipeline predict
segmentation-pipeline self-check
```

## Engineering principles demonstrated

- one repository per model product
- configuration-driven tunables
- pip-installable `src/` package
- fast contract/regression tests
- GitHub Actions CI
- append-only experiment log
- release manifest / model-registry pattern
- reproducible promoted tools
- no large binaries or datasets committed
- no unverified production-accuracy claims

## Public-data policy

This repository uses synthetic/demo inputs by default so it is safe to share publicly.
Real datasets can be integrated through adapters while remaining outside git under `runs/`.

## Status

**v0.1.0 — portfolio engineering baseline.** This is a complete runnable architecture baseline,
not a claim that a commercial model has been trained or deployed from this public repository.

## License

MIT
