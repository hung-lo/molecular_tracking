# Image registration pilot and production comparison

`tools/evaluate_image_registration.py` evaluates a full 3D affine and a smooth
local displacement field against existing saved pairwise matches. Production
runners use `image_affine_local` by default with guarded fallback to affine and
then legacy; this evaluator remains a separate **pairwise** comparison tool, not
a replacement for longitudinal graph validation.

## Run

Create a CSV with these columns (extra descriptive columns are allowed):

```csv
matching_dir,session_a,session_b
/absolute/path/to/run/matching,session_A,session_B
```

Choose a new output folder outside the source run and source sessions:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tools/evaluate_image_registration.py \
  --pairs pairs.csv \
  --output analysis/image_registration_experiment \
  --smoothing-um 15 \
  --workers 3
```

Run a smoothing-scale robustness sweep with the same pair list:

```bash
.venv/bin/python tools/evaluate_image_registration.py \
  --pairs pairs.csv --output analysis/image_registration_robustness \
  --smoothing-grid 10 15 20 25 --workers 3
```

This writes one result folder per scale and combines the requested fields in
`robustness_summary.csv`.

The matching directory must contain `session_manifest_resolved.csv`,
`roi_features.csv`, `pairwise_transforms.csv`, and saved graph or balanced pairwise
matches. Native masks must still agree with the saved features. Voxel spacing is
verified from the features' pixel and physical coordinates. Input images/masks
must share a grid. An existing output folder is rejected; interrupted runs leave
their partial outputs available for inspection rather than silently resuming.

## Four comparisons

1. **baseline:** saved graph pairwise matches, or balanced if graph is absent.
2. **recomputed_overlap:** existing affine centroid transform, with mask overlap
   recalculated after nearest-neighbor resampling into the A grid.
3. **image_affine:** full 3D affine fitted to red-channel anatomy, including
   coupling between depth and XY. Mask overlap is recalculated.
4. **smooth_local:** additional 3D TV-L1 displacement estimated on approximately
   isotropic 5–5.5 um voxels and Gaussian-smoothed with **sigma 15 um** by default.
   Sigma is converted independently on each physical axis. This smooths the
   displacement field, not the masks or intensity measurements.

The affine optimizer uses background-subtracted red images, spatially uniform
samples, and deterministic initialization. Local deformation uses only images;
ROI identities do not participate in registration fitting. Both new methods
reuse the existing candidate generator, balanced rules, and greedy one-to-one
assignment. The graph refinement is not rerun on experimental candidates.
Mask overlaps use volumes measured in the resampled grid; the size-ratio rule
continues to use native ROI volumes. Forward ROI centroids are obtained by
inverting the inverse displacement field, with round-trip error checked.

## Outputs and fallback

Each `pair_NNN` directory records input provenance, stage-specific candidate and
match CSVs, transform NPZs, QC metrics, optimization status, and a separate
`guarded_matches.csv`. The root `summary.csv` collects completed pairs.
Each non-baseline stage also writes `*_identity_conflicts.csv`, including old/new
ROI evidence and any available cycle classification. When independently evaluated
A-B, B-C, and A-C pairs are present, `cycle_audit_summary.csv` and
`cycle_audit_details.csv` use that three-pair result; otherwise the saved A-C
comparison is audit-only and is not treated as identity ground truth.

The experimental guard requires:

- no decrease in direct match count;
- image correlation no more than 0.005 below baseline and common sampled support
  no more than 2 percentage points below baseline;
- physical affine singular values within 0.8–1.25;
- no folding, with 1st/99th percentiles of the local inverse Jacobian within
  0.5–2.0;
- 99th-percentile local displacement at most 25 um and maximum centroid
  round-trip error at most 0.25 um;
- optimizer convergence; and for the local field, at least 0.005 additional
  image correlation over image-affine registration.

Strong correspondences have baseline Dice >=0.65, distance <=3 um, native size
ratio >=0.55, and ambiguity <=0.7. Registration/deformation QC is kept separate
from identity auditing. Old strong matches are not ground truth: a changed old
assignment is recorded, not forcibly restored. The guard rejects widespread
identity disruption; it also rejects a sub-99% strong-match retention or a changed
identity contradicted by an available independent cycle, unless the changed
assignment is explicitly supported by that cycle. It selects the most refined
eligible stage, falling back to the original matches if necessary. Production
matching records the selected stage, fallback reason, red-image hashes, and an
explicit image-registration algorithm version in its run log and resume fingerprint.
Missing or unreadable red images and expected registration failures preserve the
legacy mask match for that pair instead of aborting the complete run. Use
`--registration-mode legacy` for the explicit mask-only escape hatch.

These are engineering QC bounds, not validated biological accuracy thresholds.
Correlation is measured on 30,000 fixed evaluation pixels distinct from the
affine optimizer's samples; local-flow fitting nevertheless sees the images, so
this is **not independent held-out anatomy**. Better image correlation, more
matches, and agreement with previous strong matches do not prove correctness.
Review changed identities, newly recovered matches, and three-session consistency
before promoting a method. Broad 10–20 um nearest-neighbor proximity is not proof
of the same cell in a dense field.

All percentages in this evaluator are `200 * direct_matches / (N_A + N_B)`.
They are not shared-track overlaps, which can include paths through other days.

## Checks

```bash
.venv/bin/python -m pytest tests/test_image_registration.py \
  tests/test_affine_overlap_matcher.py tests/test_spatial_graph_matcher.py
```
