# QC Interpretation

The daywise QC bundle is generated automatically after matching unless `--skip-qc` is used.
It should be treated as a review aid, not as the source of truth for matching.

## What to look for

- Pairwise examples that look geometrically misaligned.
- Tracks with cycle conflicts.
- One-gap bridges and other unusual connections.
- Transform fallbacks or weak affine support.
- Graph changes that differ from the baseline balanced branch.

## Policy panels

- Baseline runs show `high` and `balanced`.
- Graph runs also show `graph` when the files exist.

## Failure behavior

If QC fails, the matching output is still valid unless `--require-qc-success` is requested.
