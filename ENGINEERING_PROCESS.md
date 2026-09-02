# Semantic Segmentation Pipeline Engineering Process

This repository follows the AI/ML engineering standard supplied for this portfolio.

## 1. Branching model

Single-owner public portfolio mode:

- `main` — release-ready code.
- `experiment/<slug>` — hypothesis-driven model experiments.
- `feature/<slug>` — known product capabilities.
- `chore/<slug>` — tooling/documentation/refactors.
- `hotfix/<slug>` — production-style defects with regression tests.

All meaningful changes should flow through pull requests. A multi-contributor version can add a protected `develop` branch.

## 2. Artifact and promotion

The model artifact is:

`(checkpoint, locked config, optional calibration metadata, code SHA)`

Lifecycle:

`Experiment → Candidate → Production`

Large checkpoints and prepared datasets belong in `runs/` and are intentionally gitignored.

## 3. Versioning

- MAJOR — architecture/output contract change.
- MINOR — retraining with changed data/loss/weights.
- PATCH — calibration or compatible serving change.

## 4. Quality gates

PR gate:

```bash
ruff check src tests
pytest -q
```

Merge/self-check gate:

```bash
segmentation_pipeline self-check
```

Release gate requires locked config, frozen benchmark evaluation, visual QA of failure cases, and a populated release manifest.

## 5. Defects, RCA, and learning

Any escaped defect should add:

1. a short RCA entry,
2. a permanent regression test,
3. a fix linked to that test.

## 6. Working conventions

Package: `segmentation_pipeline`

Main CLI:

```bash
segmentation_pipeline prepare
segmentation_pipeline train
segmentation_pipeline evaluate
segmentation_pipeline predict
segmentation_pipeline self-check
```

All tunables belong under `config/`.

## 7. Explicit deviations

| Standard prescription | This repo | Reason |
|---|---|---|
| mandatory non-author review | not enforced | public single-owner portfolio repository |
| frozen real benchmark | synthetic demo benchmark only | no private or client dataset is published |

**Current production state:** portfolio/demo architecture only; no claim of deployment to a commercial production environment.
