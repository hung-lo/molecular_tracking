# Phase D1 `cpsam_v2` Scientific Pilot (2026-09-16)

This is a validation-only benchmark. No production rescue path was enabled.

## BASELINE

- starting commit: `6ec8773`
- ending commit: `6ec8773` (the pilot itself made no repository-code changes)
- canonical run provenance commit: `8604f82a1478e97cf8a92de49fd65305cde976c3`
- focused tests: `.venv/bin/python -m pytest -q tests/test_local_segmentation_rescue.py` — **19 passed**
- Cellpose-environment check: direct evaluator completed; `pytest` is not installed in `.venv-cellpose`
- full suite: `.venv/bin/python -m pytest -q` — **347 passed, 2 skipped**

## ENVIRONMENT

- cellpose: `4.2.1.1`
- torch: `2.14.0+cu130`
- CUDA: available (`torch.cuda.is_available() == True`)
- GPU: NVIDIA GeForce RTX 3050
- model: `cpsam_v2`
- device: `cuda`
- execution interpreter: `.venv-cellpose/bin/python`
- explicit preflight: Cellpose imports, CUDA is visible, and `cpsam_v2` model load succeeded
- approved environment additions: `pandas 3.0.5`, `matplotlib 3.11.2` (no Cellpose/Torch/CUDA changes)

## TRI_1 PILOT

- mouse: `Fucci-Tri_1`
- context: `endpoint_one_sided`
- trusted eligible: `100` (108,194 trusted-ineligible rows excluded)
- attempted: `100`
- candidate generation rate: `100/100 = 1.00`
- identity correct rate: `0.00`
- wrong-neighbor rate: `0.00`
- merge rate: `0.96`
- ambiguous rate: `0.04`
- no-candidate rate: `0.00`
- median Dice: `0.8739065500320035`
- median IoU: `0.7760519019032492`
- median centroid error um: `1.0287877785316804`
- median volume ratio: `0.9262530893089786`

The segmentation overlap metrics are not sufficient to establish identity: 96 cases were classified as merged/multiple-cell and 4 as ambiguous, with zero identity-correct cases. The descriptive decision is **scientifically poor**; stop after diagnostics.

## SCALE-UP

- run: **NO** (100-case identity failure requires adjudication first)

## DEAD_1 VALIDATION

- run: **NO**

## REAL PROPOSALS

- run: **NO** (synthetic pilot was not interpretable)
- evaluator no-mask cases: `0 / unavailable`
- attempted: `0`
- single candidate: `0`
- multiple candidates: `0`
- no candidate: `0`
- excluded: `0`

## MANUAL REVIEW

- category-aware panels generated: **YES** (`12` PNG panels; `10` summary plots)
- obvious identity-failure mode: merged/multiple-cell masks (`96%`), with an additional ambiguous group (`4%`)
- notes: review artifacts are in `review_panels/` and `summary_plots/`. Hidden truth is used only in the post-selection evaluation panel. No ranking parameter was tuned against hidden truth.

## REPRODUCIBILITY

- run directory: `/mnt/d/_data/_newAAV_2026/molecular_tracking_derivatives/Fucci-Tri_1/longitudinal/1050/runs/1050_20260511_to_20260828_33s_graph_affine_balanced`
- backend parameters: `cellpose_sam`, `cpsam_v2`, `cuda`, 3-D, `z_axis=0`, `channel_axis=3`, `min_size=100`, crop `(25,96,96)`, seed `20260916`
- observed wall runtime: approximately `44 minutes` (external process monitoring)
- source summary: `pilot_summary.json`

## HARD CONSTRAINTS

- canonical masks changed: **NO**
- canonical tracks changed: **NO**
- primary extraction changed: **NO**
- state/intensity used for identity: **NO**
- production rescue enabled: **NO**
- production acceptance thresholds defined: **NO**

**Stop condition:** evidence and review artifacts are complete. Await manual scientific adjudication; do not proceed automatically to production or Phase E.
