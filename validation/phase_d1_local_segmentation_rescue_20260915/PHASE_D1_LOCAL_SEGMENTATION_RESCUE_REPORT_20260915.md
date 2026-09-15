# Phase D1 Local Segmentation Rescue Report (2026-09-15)

## STARTING/ENDING COMMIT
- Plan baseline: `804bed014569c87603ab3f6018f5e710ae398005`
- Workspace start: `90b3a74978623fc18888232861183eabb752ebf5`
- Ending commit: `0617f85` (`Store Phase D1 benchmark tables`)
- Canonical run provenance commit: `fede6d127e5458a01e9b8428b594c5eb3d2fe296`
- Commits created: `1384d2c` implementation/tests; `a6dbc5e` summaries/review artifacts; `0617f85` benchmark tables.
- Mode: `synthetic_benchmark` plus optional Dead-1 `real_proposals`

## IMPLEMENTATION
- Module: `postprocessing/local_segmentation_rescue.py`
- Segmenter: `threshold_connected_components_v1_cpu_fallback`
- Spacing (ZYX um): `(5.0, 0.693, 0.693)` (matching/run_log.json)
- Crop: `(25, 96, 96)` voxels, configurable via CLI
- Transform evidence: direct, composed, fallback, or unavailable; prediction and ranking are state-free.
- Truth leakage: target label/mask is loaded only after candidate segmentation for synthetic evaluation.

## BENCHMARK/PROPOSALS
- Eligible/attempted: `1000` / `1000`
- Candidate generated: `1000`; multiple: `1000`; no candidate: `0`
- Status counts: `{'multiple_candidates': 1000}`
- Correct-cell rate: `0.103`
- Median/P90 centroid error (um): `15.864818874886101` / `24.57955545151099`
- Median Dice/IoU: `0.006109132464702625` / `0.003064378002010649`
- Median volume ratio: `0.7916967769908947`
- Wrong-neighbor rate: `0.897`

## REAL NO-MASK PROPOSALS
- Dead-1 run: `1050_20260819_to_20260824_4s_first4_graph_affine_balanced`
- Eligible/attempted: `298` / `298`
- Candidate generated: `298`; ambiguous/multiple: `298`; no candidate: `0`
- Proposal-only; no canonical mask or extraction row was written.

## MEASUREMENT BIAS
- Green/Red/ratio are evaluation-only; ECLIPSE fields remain unavailable unless supplied by an existing extraction table.
- Measurement-bias table: `/mnt/d/codex_folder/molecular_tracking/validation/phase_d1_local_segmentation_rescue_20260915/synthetic_hide_rescue_measurement_bias.csv`

## REVIEW ARTIFACTS
- Panels: `/mnt/d/codex_folder/molecular_tracking/validation/phase_d1_local_segmentation_rescue_20260915/review_panels`
- Summary plots: `/mnt/d/codex_folder/molecular_tracking/validation/phase_d1_local_segmentation_rescue_20260915/summary_plots`

## TESTS
- Focused: `.venv/bin/pytest -q tests/test_local_segmentation_rescue.py` — 4 passed
- Full suite: `.venv/bin/pytest -q` — 308 passed, 2 skipped

## HARD CONSTRAINTS
- Canonical masks/tracks/track IDs changed: **NO**
- Matching thresholds or ECLIPSE calculation changed: **NO**
- ECLIPSE/Green/Red/state used for identity: **NO**
- Rescued masks or measurements written to primary extraction: **NO**

## CONCLUSION
- Production acceptance threshold: **not defined in Phase D1**
- Scientific promise: **UNCERTAIN pending benchmark distributions and review**
- Next step: inspect review PNGs and benchmark distributions before any production gate.
