# Phase D1 Local Segmentation Rescue Fix Report (2026-09-15)

- baseline: `fe80523`
- ending commit: `6a97a41888ee68e88e99c7ea19d95f6ccf30368b`
- commits created: post-fe80523 validity hardening

## TRUTH LEAKAGE
- target truth volume used before selection: **NO**
- target truth mask used before selection: **NO**
- synthetic and real ranking semantics aligned: **YES**

## BENCHMARK CONTEXT
- endpoint_one_sided n: `100`
- internal_gap_two_sided n: `0`
- metrics reported separately: **YES**
- trust criteria: consensus track, no cycle conflict/unchecked edge, reliable transform, non-edge target, stable context, bounded distance/ambiguity
- trusted eligible/excluded: `100` / `108194`

## SEGMENTER
- scientific backend: `cellpose_sam`
- model/version: `cpsam_v2` / `4.2.1.1`
- device: `cuda`
- threshold backend retained only as baseline/test: **YES**

## IDENTITY METRICS
- truth top-1: `96` / `0.96`
- not truth top-1: `4` / `0.04`
- dominant wrong-label: `3` / `0.03`
- no canonical overlap: `1` / `0.01`
- no-truth-overlap flag: `4` / `0.04`
- contamination categories (descriptive): `{'single_label_like': 71, 'minor_neighbor_contamination': 20, 'substantial_neighbor_contamination': 8, 'no_canonical_overlap': 1}`
- median top-1/top-2 fractions: `0.935957800743793` / `0.0012639640585575696`

## MASK METRICS
- median Dice: `0.8739065500320035`
- median IoU: `0.7760519019032492`
- median centroid error: `1.0287877785316804`
- median volume ratio: `0.9262530893089786`

## REAL PROPOSALS
- evaluator artifact: `unavailable`
- artifact SHA256: `n/a`
- evaluator no_mask_near_prediction rows: `None`
- rows parsed successfully: `None`
- eligible/attempted: `0` / `0`
- target derived from end_session_index + 1: **NO/UNAVAILABLE**
- generic-gap fallback used: **NO**

## MEASUREMENT BIAS
- production extraction semantics reused: **NO; outputs are explicitly raw_mask_mean diagnostics**

## TESTS
- focused: run by validation command
- full suite: run by validation command

## HARD CONSTRAINTS
- canonical masks changed: **NO**
- canonical tracks changed: **NO**
- track IDs changed: **NO**
- matcher thresholds changed: **NO**
- state/intensity used for identity: **NO**
- primary extraction changed: **NO**
- runner executable mode: **100755**

## PHASE D1 STATUS
- production-ready: **NO**
- ready to design production acceptance gates: **NO**
- reason: scientific Cellpose-SAM benchmark and manual review remain required.
