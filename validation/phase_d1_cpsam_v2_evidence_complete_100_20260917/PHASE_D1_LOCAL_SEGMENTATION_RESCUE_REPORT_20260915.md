# Phase D1 Local Segmentation Rescue Report (2026-09-15)

## STARTING/ENDING COMMIT
- Canonical run provenance commit: `8604f82a1478e97cf8a92de49fd65305cde976c3`
- Evaluator repository commit: `6a97a41888ee68e88e99c7ea19d95f6ccf30368b`
- Mode: `synthetic_benchmark`

## IMPLEMENTATION
- Module: `postprocessing/local_segmentation_rescue.py`
- Segmenter backend: `cellpose_sam`
- Spacing (ZYX um): `(5.0, 0.693, 0.693)` (matching/run_log.json)
- Crop: `(25, 96, 96)` voxels, configurable via CLI
- Transform evidence: direct, composed, fallback, or unavailable; prediction and ranking are state-free.
- Truth leakage: target label/mask is loaded only after candidate segmentation for synthetic evaluation.

## BENCHMARK/PROPOSALS
- Eligible/attempted: `100` / `100`
- Candidate generated: `100`; multiple: `100`; no candidate: `0`
- Status counts: `{'multiple_candidates': 100}`
- Truth top-1: `96` / `0.96`
- Not truth top-1: `4` / `0.04`
- Dominant wrong-label: `3` / `0.03`
- No canonical overlap: `1` / `0.01`
- No truth overlap flag: `4` / `0.04`
- Contamination categories (descriptive): `{'single_label_like': 71, 'minor_neighbor_contamination': 20, 'substantial_neighbor_contamination': 8, 'no_canonical_overlap': 1}`
- Median top-1/top-2 candidate fractions: `0.935957800743793` / `0.0012639640585575696`
- Median/P90 centroid error (um): `1.0287877785316804` / `3.1570051295057024`
- Median Dice/IoU: `0.8739065500320035` / `0.7760519019032492`
- Median volume ratio: `0.9262530893089786`

## MEASUREMENT BIAS
- Green/Red/ratio are evaluation-only; ECLIPSE fields remain unavailable unless supplied by an existing extraction table.
- Measurement-bias table: `/mnt/d/codex_folder/molecular_tracking/validation/phase_d1_cpsam_v2_evidence_complete_100_20260917/synthetic_hide_rescue_measurement_bias.csv`

## REVIEW ARTIFACTS
- Panels: `/mnt/d/codex_folder/molecular_tracking/validation/phase_d1_cpsam_v2_evidence_complete_100_20260917/review_panels`
- Summary plots: `/mnt/d/codex_folder/molecular_tracking/validation/phase_d1_cpsam_v2_evidence_complete_100_20260917/summary_plots`

## HARD CONSTRAINTS
- Canonical masks/tracks/track IDs changed: **NO**
- Matching thresholds or ECLIPSE calculation changed: **NO**
- ECLIPSE/Green/Red/state used for identity: **NO**
- Rescued masks or measurements written to primary extraction: **NO**

## CONCLUSION
- Production acceptance threshold: **not defined in Phase D1**
- Scientific promise: **UNCERTAIN pending benchmark distributions and review**
- Next step: inspect review PNGs and benchmark distributions before any production gate.


