# Daywise matcher evaluator

The evaluator diagnoses endpoint loss, candidate reappearance, fragmentation,
state-associated endpoint rates, and matcher runtime from an existing daywise
matching run. It is read-only: it never reruns matching, edits tracks, adds
links, interpolates masks, or recomputes ECLIPSE.

## Run

```bash
python matching/evaluate_daywise_tracking.py \
  --match-dir /path/to/matching \
  --output-dir /path/to/matching/evaluation \
  --policy graph --lookahead 3 --search-radius-um 15 \
  --crop-radius-um 45 --z-radius 1 --max-review-panels 100
```

Use `--policy balanced` when graph artifacts are unavailable. The evaluator
does not silently mix policy outputs. Optional `--extraction-dir` and
`--state-csv` enrich review rows; state CSVs must contain `track_uid` and
`session_index`, and existing `eclipse_z`/state values are copied verbatim.

## Inputs and outputs

Required inputs are `session_manifest_resolved.csv`, `roi_features.csv`,
`pairwise_transforms.csv`, and `tracks_<policy>.csv`. Existing candidate,
accepted-match, pair-summary, and runtime tables are joined when present.

The output directory contains endpoint events, future candidates, deterministic
triage classifications, a pseudo-gap benchmark, a spatial/state review
manifest, dropout and runtime summaries, `matcher_evaluation_summary.json`,
and `evaluation_run_log.json`. Review figures are PNG files under
`review_panels/`; no PDF is generated.

Endpoint classifications are intentionally conservative:
`same_track_gap_recovered`, `nearby_new_track_candidate`,
`nearby_singleton_candidate`, `multiple_nearby_candidates`,
`no_mask_near_prediction`, `edge_or_out_of_fov`, `transform_unreliable`, and
`insufficient_input`. Automatic classifications are not manual ground truth.
Manual labels are `segmentation_dropout`, `matcher_fragment`,
`true_disappearance`, `edge_or_fov_loss`, `ambiguous`, and `ignore`.

## Transform convention

Stored transforms map later-session coordinates B into earlier-session
coordinates A:

```text
z_A = z_intercept + z_scale * z_B
[y_A, x_A] = intercept + matrix @ [y_B, x_B]
```

The evaluator inverts direct `t -> t+1/t+2` transforms for forward search and
composes adjacent transforms for evaluator-only `t -> t+3` projections.
Near-singular maps are reported as unreliable.

## Review and future stitching

Inspect the PNG panel, candidate evidence, mask/edge flags, and preceding
match/cycle metadata. Record manual labels and any selected target in the
blank review columns. The synthetic benchmark measures how often a trusted
current identity is recovered after one or two missing sessions. Its output
can inform a future endpoint stitcher, but this evaluator does not assign or
stitch links.

The current implementation uses a per-session centroid KD-tree and caches
arrays during each panel. Runtime summaries describe existing matcher timing;
canonical pair stages are not parallelized here.
