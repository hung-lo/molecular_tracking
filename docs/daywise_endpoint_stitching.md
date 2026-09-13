# Conservative endpoint stitching

`matching/run_endpoint_stitching.py` is a post-hoc, state-blind stage that proposes links between unresolved canonical track endpoints and later canonical track starts. It never edits the matching directory. Accepted links produce a separate stitched identity view only when explicitly requested and the pseudo-fragment benchmark passes (or the force flag is supplied).

## Run

```bash
python matching/run_endpoint_stitching.py \
  --match-dir /path/to/matching \
  --evaluation-dir /path/to/matcher_evaluation \
  --output-dir /path/to/endpoint_stitching \
  --policy graph \
  --max-gap-sessions 3 \
  --search-radius-um 15 \
  --profile conservative_v1 \
  --benchmark-replicates 20 \
  --random-seed 0 \
  --max-review-panels 100 \
  --overwrite
```

This writes proposals, assignments, benchmark results, summaries, and PNG review panels. Add `--write-stitched-tracks` to request `tracks_graph_stitched.csv` and `track_uid_stitch_map.csv`. A failed benchmark blocks those two files. `--force-write-despite-benchmark-failure` overrides only that write guard and is recorded in the summary. `--proposal-only` always suppresses stitched-track output.

`--state-csv` and `--extraction-dir` are accepted as review-only provenance. ECLIPSE, red/green intensity, and `color_z` are removed before tiering and never enter assignment costs.

## Candidate and automatic gates

The evaluator's 15 µm search is only an envelope. Candidates must connect a true final observation to a different track whose first observation is 1–3 sessions later. Same-track returns, right-censored endpoints, non-start targets, overlap, and out-of-range gaps are not candidates.

Every retained edge records forward distance/ranks/margin, reverse distance/rank/margin, transform QC, local canonical-anchor residuals, up to three source-history observations, up to three later target observations, canonical track quality, and available matcher evidence. Dice, area ratio, volume ratio, and intensity are diagnostic only.

`conservative_v1` requires all of the following for `auto_accept_candidate`:

- source and target each contain at least two observations;
- neither track has a cycle warning, edge warning, or canonical transform-fallback warning;
- the candidate transform and all composed gap-3 components are reliable;
- forward all-mask and track-start rank are 1, reverse endpoint rank is 1, and the forward margin is at least 3 µm;
- projected distance is at most 4 µm for gap 1 and 5 µm for gaps 2–3;
- at least three local anchors, median anchor residual at most 4 µm, and anchor inlier fraction at least 2/3 using a 6 µm inlier radius;
- at least two source-history observations with median residual at most 5 µm;
- at least one observation after the target start with median future residual at most 5 µm.

Structural, unreliable-transform, edge, distance, anchor, history, or future inconsistency produces `reject`. Plausible candidates that fail only context or uniqueness requirements—including singleton fragments, missing anchors/future support, canonical warnings, or non-rank-1 geometry—remain `manual_review` with explicit reason codes.

Only auto-eligible edges enter a connected-component linear assignment. Each source and target is used at most once. A dedicated dummy column gives every source an explicit no-link option. Assignment cost only breaks collisions among already-safe edges: projected distance dominates, followed by anchor residual, history, future, margins, gap, and stable UID/edge ordering.

## Derived identities and invariants

Accepted edges are unioned transitively. The member with the earliest first session (then lexical UID) becomes `stitched_track_uid`; all canonical UIDs remain in provenance tables. A same-session collision raises an error. The stage counts observed `(session_id, ROI label)` nodes before and after merging and refuses a non-conserving result. Missing sessions remain `NA`; no masks or observations are interpolated. Gap-3 links are labeled only as endpoint-stitch edges and never become canonical pairwise matches.

## Outputs

- `stitch_candidates.csv`: all valid endpoint-to-start evidence and tiers.
- `stitch_proposals.csv`: auto and manual-review candidates.
- `stitch_assignments.csv`: all candidates plus global assignment status/cost.
- `track_stitch_edges.csv`: fixed audit schema for stitch decisions.
- `synthetic_stitch_benchmark.csv`: positive pseudo-fragments plus no-successor and target-only negatives for gaps 1–3.
- `synthetic_stitch_threshold_sweep.csv`: precision/FPR guardrail table.
- `manual_stitch_review_manifest.csv`: deterministic review rows; existing manual labels are restored by `stitch_edge_id` on overwrite.
- `review_panels/*.png` and `review_panels/stitch_contact_sheet.png`: state-blind review artifacts. No PDFs are emitted.
- `stitch_summary.json`: counts, gap distributions, conservation checks, benchmark metrics, and immutable/state-blind flags.
- `stitch_run_log.json`: paths, spacing, configuration, canonical input hashes, and provenance.
- `tracks_<policy>_stitched.csv` and `track_uid_stitch_map.csv`: optional guarded derived identity outputs.

Important provenance fields in `track_stitch_edges.csv` include source/target IDs and labels, session/elapsed gaps, forward and reciprocal ranks/margins, transform source/QC, anchor support and residuals, history/future support, optional shape/matcher diagnostics, reason codes, assignment cost/component, and algorithm version. Unavailable optional measurements remain empty/NaN.

## Benchmark guardrail

The benchmark splits trusted continuous canonical tracks into source/target pseudo-fragments, hides the intervening observations, and runs the same candidate, gate, and assignment code used on real endpoints. It includes gaps 1, 2, and 3, repeated deterministic samples, explicit sources with no successor, and target-only controls. It reports precision, recall by gap, negative false-positive rate, collision outcomes, and a Wilson precision interval.

Default write guardrails are precision ≥99.9% and negative false-positive rate ≤0.1%. An empty benchmark fails closed. These thresholds control derived file writing; they never loosen candidate gates.
