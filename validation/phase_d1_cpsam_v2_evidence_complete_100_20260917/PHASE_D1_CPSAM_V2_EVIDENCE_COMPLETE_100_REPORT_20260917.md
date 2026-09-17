# Phase D1 Evidence-Complete 100-Case `cpsam_v2` Validation (2026-09-17)

This is an independent, frozen repeat of the Fucci-Tri_1 endpoint-one-sided pilot. Hidden truth is used only after candidate generation and selection. Canonical outputs were not modified.

## BASELINE
- starting commit: `bbc92d17fe9f80d46de2aede3e8b348e8c28076b`
- ending commit: `6a97a41888ee68e88e99c7ea19d95f6ccf30368b`
- git dirty: `True` (pre-existing unrelated worktree changes are preserved)
- focused tests: **24 passed**
- full suite: **352 passed, 2 skipped**

## ENVIRONMENT
- python: `/mnt/d/codex_folder/molecular_tracking/.venv-cellpose/bin/python` (3.11.16 (main, Sep  1 2026, 14:18:37) [Clang 22.1.3 ])
- cellpose: `4.2.1.1`
- torch: `2.14.0+cu130`
- CUDA: `13.0`, available `True`
- GPU: `NVIDIA GeForce RTX 3050`
- model: `cpsam_v2`
- device: `cuda`

## RUN
- mouse: `Fucci-Tri_1`
- context: `endpoint_one_sided`
- trusted eligible: `100`
- attempted: `100`
- candidate generation rate: `100/100`
- backend failures: `0`

## IDENTITY
- truth_is_top1: `96/100` (0.96)
- not_truth_top1: `4/100` (0.04)
- dominant_wrong_label: `3/100` (0.03)
- no_canonical_overlap: `1/100` (0.01)
- no_candidate: `0/100`
- no_truth_overlap: `4/100` (0.04)

## OVERLAP
- median truth fraction: `0.9352857193846593`
- median top1 fraction: `0.935957800743793`
- median top2 fraction: `0.011889280741803482`
- median top1-minus-top2: `0.9295127092078315`
- median top1/top2 ratio: `77.30000000000001`
- q95 top2 fraction: `0.1603797200060863`

## MASK QUALITY
- median Dice: `0.8739065500320035`
- median IoU: `0.7760519019032492`
- median centroid error: `1.0287877785316804`
- median volume ratio: `0.9262530893089786`

## MEASUREMENT BIAS
- median Green relative difference: `0.0035797089379974003`
- median Red relative difference: `0.0102863694856849`
- median ratio relative difference: `-0.00693510590942105`

## ORIGINAL PILOT COMPARISON
| metric | original pilot | evidence-complete repeat |
|---|---:|---:|
| `truth_is_top1_rate` | 0.96 | 0.96 |
| `not_truth_top1_rate` | 0.04 | 0.04 |
| `dominant_wrong_label_rate` | 0.03 | 0.03 |
| `no_canonical_overlap_rate` | 0.01 | 0.01 |
| `no_candidate_rate` | 0.0 | 0.0 |
| `median_dice` | 0.8739065500320035 | 0.8739065500320035 |
| `median_iou` | 0.7760519019032492 | 0.7760519019032492 |
| `median_centroid_error_um` | 1.0287877785316804 | 1.0287877785316804 |
| `median_volume_ratio` | 0.9262530893089786 | 0.9262530893089786 |
| `median_green_bias` | 0.0035797089379974003 | 0.0035797089379974003 |
| `median_red_bias` | 0.0102863694856849 | 0.0102863694856849 |
| `median_ratio_bias` | -0.00693510590942105 | -0.00693510590942105 |
- case IDs identical in order: **True**

## MANUAL REVIEW
- panels generated: `12` PNGs
- dominant failure mode: descriptive identity failures remain separate from contamination bins
- neighbor contamination interpretation: exact top-2 evidence is persisted for every candidate; review is still required
- notes: panels show source Red, target Red, all candidate outlines, selected outline, truth, top-1/top-2 canonical outlines, and rank metadata

## DECISION
- D. mixed / requires more adjudication
- reason: numerical repeat is evidence-complete, but manual review is required before interpreting neighbor contamination
- recommend frozen 500-case validation: **NO — wait for manual scientific review**

## HARD CONSTRAINTS
- Cellpose parameters changed: **NO**
- ranking changed: **NO**
- canonical masks changed: **NO**
- canonical tracks changed: **NO**
- primary extraction changed: **NO**
- state/intensity used for identity: **NO**
- production rescue enabled: **NO**
- production thresholds defined: **NO**

## RUNTIME
- elapsed seconds: `2596.416`
- runtime metadata: `runtime.json`
- stop condition: evidence-complete 100-case validation finished; no 500-case, Dead_1, real-proposal, production, or Phase E run was started.
