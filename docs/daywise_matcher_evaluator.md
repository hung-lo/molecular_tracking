# Daywise matcher evaluator

## Purpose and safety boundary

The daywise matcher evaluator is a **read-only diagnostic layer** for an
existing matching run. It is intended to answer whether current track endings
are associated with an existing one-gap recovery, a nearby later track start,
a nearby singleton, no nearby segmented ROI, edge/FOV geometry, or unreliable
registration. It also provides a silver-standard pseudo-gap benchmark,
state-dependent dropout summaries when an external state table is supplied,
and matcher runtime summaries.

The evaluator does **not** rerun canonical matching and does not change affine
thresholds, high/balanced/graph rules, gap acceptance, graph construction,
canonical track IDs, extraction eligibility, or ECLIPSE. It never creates a
canonical `t -> t+3` link and never interpolates a missing mask. All figures are
PNG; the evaluator never writes PDFs.

The working failure-mode distinction is:

- **segmentation dropout**: raw signal is visibly present but no valid mask is
  available (manual label only; never inferred automatically);
- **matcher fragment**: a plausible later mask belongs to a different track
  (manual label after review);
- **true disappearance**: no convincing later cell is visible;
- **edge/FOV loss**: the expected location is at unreliable imaging geometry;
- **ambiguous**: more than one identity remains plausible.

A missing observation is preferable to a wrong biological identity. The
outputs are evidence for review and future endpoint stitching, not revised
canonical tracks.

## CLI

Typical graph-policy run:

```bash
python matching/evaluate_daywise_tracking.py \
  --match-dir /path/to/matching \
  --output-dir /path/to/matching/evaluation \
  --policy graph \
  --lookahead 3 \
  --search-radius-um 15 \
  --crop-radius-um 45 \
  --z-radius 1 \
  --max-review-panels 100
```

Optional enrichment:

```bash
  --extraction-dir /path/to/extraction \
  --state-csv /path/to/dead_referenced_eclipse_long.csv
```

The state CSV must contain `track_uid` and `session_index`. Existing fields
such as `eclipse_z`, `eclipse_core_state`/`eclipse_state_bin`, `green`, `red`,
and `elapsed_days` are copied when present. **ECLIPSE is never recomputed.**

The pseudo-gap trust set is configurable:

```text
--synthetic-min-score 0.35
--synthetic-min-dice 0.10
--synthetic-max-distance-um 5
--synthetic-max-ambiguity 0.85
--[no-]synthetic-require-consensus
--[no-]synthetic-require-no-cycle-conflict
--[no-]synthetic-require-no-transform-fallback
--[no-]synthetic-require-interior
```

`--overwrite` replaces evaluator-generated products only. Existing manual
review fields in `endpoint_classification.csv` are restored by `endpoint_id`
when possible. The evaluator output directory must be separate from the
matching directory.

## Input discovery and policy isolation

Required inputs are:

```text
session_manifest_resolved.csv
roi_features.csv
pairwise_transforms.csv
tracks_<policy>.csv
```

The evaluator discovers and reuses these when present:

```text
track_edges_<policy>.csv
pairwise_candidates.csv
pairwise_matches_high.csv
pairwise_matches_balanced.csv
pairwise_matches_graph.csv
pairwise_summary.csv
pairwise_summary_<policy>.csv
cycle_consistency_<policy>.csv
cycle_edge_checks_<policy>.csv
graph_match_changes.csv
run_log.json
```

A requested policy is never silently replaced by another one. For example,
`--policy graph` requires `tracks_graph.csv`; the existence of balanced files
is not treated as permission to fall back.

Matcher voxel spacing is taken from `run_log.json` when available. Repository
default spacing is used only when exact run provenance is unavailable, and the
source is recorded in `evaluation_run_log.json` and the summary JSON.

Optional extraction enrichment currently recognizes the repository's stable
keys in tables such as:

```text
matched_roi_geometry_qc_long.csv
matched_roi_log_ratio_metrics_all_observed.csv
matched_roi_trajectory_observations_eligible.csv
matched_track_qc_summary.csv
graph_affine_agreement_track_metadata.csv
```

Session-level tables join on `track_uid, session_index`; track-level tables
join on `track_uid`.

## Transform convention

Canonical stored transforms are **B -> A**, where B is the later session:

```text
z_A = z_intercept + z_scale * z_B

[y_A]   [y_intercept]   [y_from_y  y_from_x] [y_B]
[x_A] = [x_intercept] + [x_from_y  x_from_x] [x_B]
```

For evaluator forward projection from an endpoint in A into B, the 2x2 XY
matrix and Z mapping are inverted. Near-singular XY or Z transforms are
rejected.

Projection behavior is:

- `t -> t+1`: prefer the direct stored transform and invert it;
- `t -> t+2`: prefer the direct stored transform; if it is absent, compose the
  two adjacent transforms as an evaluator fallback. When both direct and
  adjacent-composed forms exist, their projection disagreement is retained as
  QC;
- `t -> t+3`: compose the three adjacent transforms and record
  `transform_source = composed_adjacent`.

`t+3` is **search-only** in this evaluator. It never becomes a canonical edge.
Component methods, fallback reasons, and available residual QC are retained in
candidate rows.

## Endpoint definition and right censoring

An endpoint event is an observed ROI at session `t` whose **same current
track** has no ROI at `t+1`. The final acquired session is right-censored and
is not called an endpoint because no `t+1` observation exists.

For every endpoint, the evaluator also searches the current track through the
configured lookahead and records:

```text
same_track_returns
same_track_return_gap
same_track_return_session_index
```

Thus an existing canonical `t -> t+2` recovery is identified before any
fragmentation triage.

`endpoint_events.csv` contains track metadata, endpoint geometry/edge fields,
optional supplied ECLIPSE/intensity fields, local ROI density, and available
preceding accepted-edge/graph/cycle evidence.

## Future candidate search

For every endpoint and each future session through `lookahead` (maximum 3), the
evaluator:

1. projects the endpoint centroid into that session;
2. queries a per-session physical-coordinate KD-tree within
   `search_radius_um`;
3. ranks masks deterministically by projected distance then label;
4. identifies current track ownership and whether the target track starts at
   that session or is a singleton;
5. joins existing baseline/high/balanced/graph evidence when the same
   source-target label pair already exists in canonical matcher tables.

`endpoint_candidates.csv` includes projected/target coordinates, distance,
volume ratio, ownership, accepted-match flags, Dice/IoU/ambiguity, base and
refined scores, graph support/residual fields, second-neighbor margin,
transform provenance, and direct-vs-composed `t+2` QC where available.

No candidate row assigns a new identity.

## Deterministic automatic triage

`endpoint_classification.csv` uses these diagnostic categories:

```text
same_track_gap_recovered
nearby_new_track_candidate
nearby_singleton_candidate
multiple_nearby_candidates
no_mask_near_prediction
edge_or_out_of_fov
transform_unreliable
insufficient_input
```

Only reliable nearby masks whose current track starts at the target session
are considered endpoint-stitch candidates. Nearby masks already owned by an
older track make the local field ambiguous rather than being mislabeled as a
new-track candidate.

The following fields are intentionally blank until human review:

```text
manual_class
manual_target_track_uid
manual_target_session_index
manual_target_label
manual_confidence
reviewer_notes
```

Allowed manual classes are:

```text
segmentation_dropout
matcher_fragment
true_disappearance
edge_or_fov_loss
ambiguous
ignore
```

The evaluator never automatically declares `segmentation_dropout` because that
requires inspection of raw imagery.

## PNG endpoint review workflow

`review_panels/` contains one PNG per selected endpoint plus a PNG contact
sheet. The main panel uses columns:

```text
t-1 | t | t+1 | t+2 | t+3
```

and rows:

```text
Red raw crop
Green raw crop
Red + segmentation/candidate overlays
```

The crop follows the same physical location using the matcher transforms.
Panels show the endpoint/source contour when available, projected centroid
crosshairs, future candidate contours, candidate rank/distance/role/track ID,
session/date, and a footer with state/intensity/volume/edge information,
preceding match evidence, cycle/graph information, candidate summary, and
transform QC. Sessions without a mask or raw file remain safe placeholders;
no interpolated observation is drawn.

TIFF access prefers memory mapping, arrays are cached across review panels,
and contact-sheet thumbnails are downsampled to avoid repeatedly retaining
full rendered panels in memory.

## Synthetic pseudo-gap benchmark

`synthetic_gap_benchmark.csv` is a silver-standard diagnostic built only from
trusted current identities. It evaluates:

- one missing intermediate session: source-to-target gap 2;
- two missing intermediate sessions: source-to-target gap 3.

A case must be continuously observed across the entire source-to-target
interval before observations are conceptually hidden. A track that already has
a real internal gap is therefore not used as pseudo-ground-truth for that
interval. Configured trust filters can require consensus, no cycle conflict,
no transform fallback, interior geometry, score/Dice minima, and
distance/ambiguity maxima.

Crucially, failed searches remain in the benchmark. A known target outside the
search radius is recorded with `true_target_in_search_radius = false` rather
than being dropped from the denominator. The known target's global distance
rank is computed against all masks in the target session when projection is
available. Fixed inputs/configuration yield deterministic output.

This benchmark does not rerun segmentation.

## Manual review manifest

`manual_review_manifest.csv` contains two complementary samples:

1. **general spatial**: approximately one endpoint per occupied cell of an
   8 x 8 XY grid;
2. **state dependent**: Low endpoints are oversampled, with a Middle endpoint
   control selected from the same session when available.

Middle controls are approximately matched using standardized distance over
available Red intensity, volume, XYZ position, local density, and track
length. The manifest records the Low-control pairing, match distance, and
covariates used.

## Dropout and state summaries

`dropout_summary.csv` contains tracking-session summaries and, when state data
are supplied, state-dependent strata. State dropout denominators include
eligible Low/Middle observations that actually have a possible `t+1`; the
final session is excluded as right-censored.

When data permit, state rows report Low/Middle dropout rate, risk ratio, and
risk difference overall and by practical strata including session, Red
quintile, volume quintile, edge status, local-density quintile, and track
source.

These are tracking-loss associations. Manual segmentation-dropout,
matcher-fragment, true-disappearance, and edge/FOV-loss rates are reported only
after corresponding manual labels exist. The evaluator does not claim manual
ground truth before review.

## Runtime profiling

`matching_runtime_summary.csv` normalizes available per-pair timing/count
information from `pairwise_summary.csv` and joins candidate and graph counts.
The evaluator also searches nearby master `run_manifest.json` files for
existing stage-duration summaries.

Current `pairwise_summary_graph.csv` copies the affine pair `elapsed_sec` and
does **not** provide graph-stage pair wall time. The evaluator therefore leaves
`graph_stage_seconds` missing unless a dedicated graph timing field exists; it
does not mislabel affine time as graph time. The summary JSON reports only
bottlenecks directly supported by the available timing data.

Required runtime figures are PNG only:

```text
matching_runtime_by_pair.png
matching_runtime_vs_candidate_count.png
matching_runtime_by_gap.png
```

This patch does not parallelize canonical matching.

## Outputs

The evaluator writes:

```text
endpoint_events.csv
endpoint_candidates.csv
endpoint_classification.csv
synthetic_gap_benchmark.csv
manual_review_manifest.csv
dropout_summary.csv
matching_runtime_summary.csv
matcher_evaluation_summary.json
evaluation_run_log.json
matching_runtime_by_pair.png
matching_runtime_vs_candidate_count.png
matching_runtime_by_gap.png
review_panels/*.png
```

`evaluation_run_log.json` records discovered/missing files, policy, optional
enrichment availability, exact/fallback spacing provenance, package/version
metadata, row counts, omissions, trust configuration, and explicit flags that
canonical matching was not rerun and ECLIPSE was not recomputed.

## Limitations

- Raw-image review can classify segmentation dropout only after a human labels
  the panel; the evaluator does not infer visibility from raw intensity.
- `t+3` projections depend on all required adjacent transforms and accumulate
  registration uncertainty; they remain evaluator-only.
- Per-pair graph wall time cannot be recovered from historical runs that did
  not record it explicitly.
- The pseudo-gap benchmark is silver-standard, not biological ground truth;
  it evaluates recoverability of identities that the current matcher already
  tracks confidently.
- Missing optional state/extraction inputs skip only their dependent analyses.

## How this feeds endpoint stitching

After real-data review, the evaluator outputs can be used to design a separate
conservative endpoint-to-later-track-start stage. Candidate distance,
uniqueness margin, volume/shape evidence, graph consistency, transform QC,
local density, image evidence, and gap length can then be evaluated against the
pseudo-gap benchmark and manual labels. Any future stitcher should use global
one-to-one assignment, require stronger evidence for longer gaps, preserve old
and new IDs/provenance, and keep unvalidated intermediate measurements
missing.

That future stage is intentionally **not implemented here**.
