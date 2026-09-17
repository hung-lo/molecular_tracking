# Phase D1 `cpsam_v2` identity re-adjudication (2026-09-17)

This is a pure re-analysis of the saved 100-case pilot. The original pilot tables and PNGs were not overwritten, Cellpose was not rerun, and ranking parameters were not changed.

The evaluator now records identity ranking (`dominant_truth_overlap` / `dominant_wrong_label` / `no_canonical_overlap` / `no_candidate`) separately from descriptive contamination and persists top-1/top-2 overlap fields plus the raw canonical-overlap vector for future runs. Those code changes were not used to regenerate this pilot.

## OLD VERSUS NEW DESCRIPTIVE IDENTITY

| measure | original report | re-adjudication |
|---|---:|---:|
| selected candidates | 100/100 | 100/100 |
| `identity_correct_rate` | 0/100 | not used |
| `merged_multiple_cells` | 96/100 | not used as an identity gate |
| `truth_is_top1` | not reported | **96/100 (0.96)** |
| `not_truth_top1` | not reported | **4/100 (0.04)** |
| dominant wrong canonical label (positive top-1 overlap) | not reported | **3/100 (0.03)** |
| no canonical overlap | not reported | **1/100 (0.01)** |
| no truth overlap (independent selected-candidate flag) | not reported | **4/100 (0.04)** |

The old classifier treated any second-label voxel as a merge. The new descriptive identity result is 96 dominant-truth selections, 3 positive-overlap wrong-label selections, and 1 no-canonical-overlap selection. `not_truth_top1` remains the complete four-case failure count; `no_truth_overlap` is an independent flag and is true for all four selected candidates. This does not establish production acceptability.

## OVERLAP DECOMPOSITION

- median top-1 canonical fraction of selected candidate: **0.936** (q05 0.523, q95 0.997)
- median truth fraction of selected candidate: **0.935** (q05 0.377, q95 0.997)
- median `1 - top1_fraction` proxy: **0.064** (q05 0.003, q95 0.477)
- descriptive proxy counts: 13 single-label-like (≤1%), 29 minor (1–5%), 58 substantial (>5%)

The `1 - top1_fraction` value is an upper bound on non-top-1 material: it includes background as well as any neighboring canonical labels. It is not a production contamination threshold.

The original pilot did not persist candidate masks or per-canonical-label overlap vectors. Consequently, exact top-2 label, top-2 voxel, top-2 fraction, top-1-minus-top-2, and ≥1/5/10% label-count fields are explicitly unavailable rather than reconstructed. They are present as `NaN` with a provenance status in `selected_overlap_decomposition.csv`. A future pilot must persist those vectors before exact top-2 adjudication.

## MASK QUALITY AND MEASUREMENT BIAS

- median Dice / IoU: **0.874 / 0.776**
- median centroid error: **1.03 µm**
- median volume ratio: **0.926**
- Green relative difference: median **+0.36%**, q05–q95 **−2.37% to +5.27%**
- Red relative difference: median **+1.03%**, q05–q95 **−6.11% to +10.68%**
- color-ratio relative difference: median **−0.69%**, q05–q95 **−4.32% to +4.68%**

These measurements are diagnostic only and were not used for identity selection.

## FOUR DOMINANT-LABEL FAILURES

Dedicated panels are in `review_panels_four_failures/`. The saved geometric ranking explains the selections:

| track | selected truth overlap? | selected top-1 canonical overlap | truth label | selected ID (distance / score) | best truth-overlap candidate |
|---|---|---|---:|---|---|
| `...20260519:7222` | no | label 6412, fraction 0.994 | 6988 | 85 (17.75 / 17.75) | none |
| `...20260511:2564` | no | label 2564, fraction 0.878 | 1941 | 26 (11.47 / 11.47) | ID 10 (Dice 0.048; distance 22.25) |
| `...20260528:6586` | no | none, fraction 0 | 6691 | 69 (12.94 / 12.94) | ID 51 (Dice 0.413; distance 14.47; score 16.47) |
| `...20260708:5047` | no | label 3943, fraction 0.937 | 4825 | 66 (13.67 / 13.67) | none |

The complete per-case fields, including top-1 presence, labels, selected rank metadata, and the first five ranked candidates, are in `summary_reclassified.json` under `failure_details`.

No ranking change was attempted.

## REVIEW OUTPUTS

- `selected_overlap_decomposition.csv`
- `summary_reclassified.json`
- `summary_plots/` (overlap, identity, contamination, quality, and bias plots)
- `review_panels_four_failures/` (4 dedicated panels)
- `review_panels_truth_top1/` (6 stratified truth-top-1 panels)

Panels show source Red, target Red, predicted location, hidden truth, saved candidate/competitor geometry, and rank metadata. Candidate masks were not persisted in the original pilot, so candidate overlays are explicitly rendered as saved bbox/centroid geometry.

## DECISION

**D. Mixed / requires more manual adjudication.** Ranking appears promising (96% truth top-1), but exact neighbor contamination is unavailable and the non-top-1 proxy is substantial in many cases. Do not run the 500-case scale-up, Dead_1, real proposals, or production rescue until the overlap vectors are persisted and the panels are manually adjudicated.

## HARD CONSTRAINTS

- canonical masks/tracks/track IDs changed: **NO**
- matcher, stitcher, extraction, ECLIPSE, Green/Red normalization changed: **NO**
- Cellpose rerun: **NO**
- ranking parameters changed: **NO**
- state/intensity used for identity: **NO**
- production rescue enabled: **NO**
