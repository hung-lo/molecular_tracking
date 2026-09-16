# Phase D1 Local Segmentation Rescue Fix Report (2026-09-15)

- baseline: `e803388`
- validity-fix commit: `fe80523`
- acquisition-QC commit (separate): `206b413`

## Truth leakage

- target truth volume used before selection: **NO**
- target truth mask used before selection: **NO**
- ranking is geometry-only and synthetic endpoint cases ignore future observations: **YES**

## Benchmark context

- `endpoint_one_sided` and `internal_gap_two_sided` are explicit context classes.
- existing threshold outputs remain a development/negative-control baseline; they are not a Cellpose-SAM validation.

## Segmenter

- scientific backend: `cellpose_sam` (`cpsam_v2`, `do_3D=True`, `z_axis=0`, `channel_axis=3`)
- `threshold_baseline` is explicit for tests/negative controls only.
- Cellpose/GPU unavailability is reported as `backend_unavailable`; no silent fallback occurs.

## Real proposals

- only explicit evaluator rows classified `no_mask_near_prediction` are accepted;
- missing evaluator artifact is reported unavailable;
- generic internal-gap fallback: **NO**.

## Identity and measurement safeguards

- canonical-label overlap fields and identity categories are recorded separately from Dice/IoU;
- measurement fields are explicitly `raw_mask_mean_*` diagnostics;
- no canonical masks, tracks, track IDs, matcher thresholds, or primary extraction outputs are written.

## Tests

- focused Phase D1 tests: **6 passed**
- acquisition/catalog/master focused tests: **40 passed**
- full repository suite: **315 passed, 2 skipped**

## Status

- production-ready: **NO**
- ready to design production acceptance gates: **NO**
- reason: a Cellpose-SAM pilot and manual review are still required before setting production gates.

