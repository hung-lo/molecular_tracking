# Phase D1 Local Segmentation Rescue Report (2026-09-15)

## STARTING/ENDING COMMIT
- Plan baseline: `804bed014569c87603ab3f6018f5e710ae398005`
- Workspace start: `90b3a74978623fc18888232861183eabb752ebf5`
- Ending commit: `a6dbc5e` (`Record Phase D1 rescue validation results`)
- Canonical run provenance commit: `04d522fe8ad7365b04eb6cf94217ebad8e2c19f6`
- Commits created: `1384d2c` implementation/tests; `a6dbc5e` validation summaries/review artifacts.
- Mode: `real_proposals`

## IMPLEMENTATION
- Module: `postprocessing/local_segmentation_rescue.py`
- Segmenter: `threshold_connected_components_v1_cpu_fallback`
- Spacing (ZYX um): `(5.0, 0.693, 0.693)` (matching/run_log.json)
- Crop: `(25, 96, 96)` voxels, configurable via CLI
- Transform evidence: direct, composed, fallback, or unavailable; prediction and ranking are state-free.
- Truth leakage: target label/mask is loaded only after candidate segmentation for synthetic evaluation.

## BENCHMARK/PROPOSALS
- Eligible/attempted: `298` / `298`
- Candidate generated: `298`; multiple: `298`; no candidate: `0`
- Status counts: `{'multiple_candidates': 298}`

## TESTS
- Focused: `.venv/bin/pytest -q tests/test_local_segmentation_rescue.py` — 4 passed
- Full suite: `.venv/bin/pytest -q` — 308 passed, 2 skipped

## MEASUREMENT BIAS
- Green/Red/ratio are evaluation-only; ECLIPSE fields remain unavailable unless supplied by an existing extraction table.
- Measurement-bias table: `None`

## REVIEW ARTIFACTS
- Panels: `/mnt/d/codex_folder/molecular_tracking/validation/phase_d1_dead1_real_proposals_20260915/review_panels`
- Summary plots: `/mnt/d/codex_folder/molecular_tracking/validation/phase_d1_dead1_real_proposals_20260915/summary_plots`

## HARD CONSTRAINTS
- Canonical masks/tracks/track IDs changed: **NO**
- Matching thresholds or ECLIPSE calculation changed: **NO**
- ECLIPSE/Green/Red/state used for identity: **NO**
- Rescued masks or measurements written to primary extraction: **NO**

## CONCLUSION
- Production acceptance threshold: **not defined in Phase D1**
- Scientific promise: **UNCERTAIN pending benchmark distributions and review**
- Next step: inspect review PNGs and benchmark distributions before any production gate.
