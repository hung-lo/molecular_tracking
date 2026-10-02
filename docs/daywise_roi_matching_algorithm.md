# Daywise ROI Matching Algorithm

This repository's baseline daywise matcher uses an affine-overlap pipeline.
It estimates a coarse shift from binary occupancy, fits a restricted B-to-A transform,
constructs a shared candidate table, and then solves independent `high` and `balanced`
one-to-one assignments.

## Key points

- The matcher is deterministic.
- `high` and `balanced` are separate assignments.
- `graph` is an experimental refinement that builds on the baseline output.
- The matcher consumes daily masks and writes pairwise, track, cycle, and summary CSV files.

## Public output families

- `session_manifest_resolved.csv`
- `roi_features.csv`
- `pairwise_summary.csv`
- `pairwise_transforms.csv`
- `pairwise_matches_high.csv`
- `pairwise_matches_balanced.csv`
- `tracks_high.csv`
- `tracks_balanced.csv`
- `track_edges_high.csv`
- `track_edges_balanced.csv`
- `cycle_consistency_high.csv`
- `cycle_consistency_balanced.csv`
- `cycle_edge_checks_high.csv`
- `cycle_edge_checks_balanced.csv`
- `track_length_summary.csv`

See `README.md` for the CLI commands.

## Optional image registration

The production runners keep the mask-only path as the default and expose the validated
red-anatomy registration as an opt-in mode:

```bash
.venv/bin/python matching/run_daywise_graph_matching.py \
  --manifest path/to/session_manifest.csv \
  --output-dir path/to/output \
  --registration-mode image_affine_local \
  --registration-smoothing-um 15
```

`legacy`, `image_affine`, and `image_affine_local` are supported. The image modes fit
geometry from the red volume, resample masks with nearest-neighbor interpolation, and
reuse the existing candidate and graph thresholds. Affine/local QC can fall back to the
next safe stage; the selected stage and reason are recorded in
`pairwise_registration_qc.csv` and `pairwise_registration_identity_conflicts.csv`.
Selected image transforms are stored under `registration_transforms/`. Registration
mode, smoothing, and red-image hashes participate in the run fingerprint, so a legacy
output cannot be resumed as an image-registration run.

The master runner accepts the same `--registration-mode` and
`--registration-smoothing-um` options. Its default run naming preserves existing legacy
names and adds the selected non-legacy mode to new run names.
