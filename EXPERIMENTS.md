# Semantic Segmentation Pipeline Experiment Log

Append-only, forward-corrections only.

## 1. Synthetic pipeline smoke test

**Hypothesis / question:** Can the complete data → model → inference path execute deterministically on synthetic inputs?

**Pre-registered decision rule:** ADOPT only if the self-check completes without shape, NaN, or contract failures.

**Configs / commits:** `config/default.yaml`; initial public portfolio implementation.

**Result:** Covered by automated contract tests and `segmentation_pipeline self-check`.

**Decision:** ADOPT as the baseline repository architecture.

**What this does NOT establish:** This does not establish accuracy on any real deployment domain, customer data, or private benchmark.
