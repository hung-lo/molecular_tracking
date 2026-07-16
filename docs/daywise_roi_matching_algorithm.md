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
