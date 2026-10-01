# Opt-in image registration pilot

`tools/evaluate_image_registration.py` evaluates a full 3D affine and a smooth
local displacement field against existing saved pairwise matches. It does not
change the primary pipeline, segmentation, fluorescence measurements, graph
tracks, or production defaults. Results are experimental **pairwise** assignments,
not a replacement longitudinal identity product.

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
When the requested pairs include A-B and B-C, `cycle_audit_summary.csv` and
`cycle_audit_details.csv` compare their compositions against saved high-confidence
A-C matches. That reference still uses production registration and is not ground
truth; evaluate A-C with the same experimental method for an independent
three-pair consistency test. Cycle results are reported, not silently used to
alter the pairwise fallback selection.

The experimental guard requires:

- no decrease in direct match count;
- image correlation no more than 0.005 below baseline and common sampled support
  no more than 2 percentage points below baseline;
- physical affine singular values within 0.8–1.25;
- no folding, with 1st/99th percentiles of the local inverse Jacobian within
  0.5–2.0;
- 99th-percentile local displacement at most 25 um and maximum centroid
  round-trip error at most 0.25 um;
- no reassignment of a strong existing correspondence and at least 99% retention
  of strong existing correspondences;
- optimizer convergence; and for the local field, at least 0.005 additional
  image correlation over image-affine registration.

Strong correspondences have baseline Dice >=0.65, distance <=3 um, native size
ratio >=0.55, and ambiguity <=0.7. Their number is reported: very few anchors
provide weak assurance even if all survive. The guard does not union or forcibly
lock old identities into new assignments. It selects the most refined eligible
stage, falling back to the original matches if necessary.

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
