# High ROI matching QC performance investigation

**Date:** 2026-11-06
**Repository revision inspected:** `eda390baba4d617d44ff27c6c48f5b82bbdb1606` (`master`)
**Scope:** optimize matching QC lookup work while preserving accepted identities, QC tables, deterministic sampling, and figures. Matcher products and the pre-existing QC directories are read-only.

## Baseline from saved QC runs

The measurements below come from each run's existing `matching/qc/run_log.json`; no old QC run was repeated. CSV row counts are data rows (header excluded). Pairwise candidate and ROI distributions were streamed from the corresponding CSVs. Runtime counters in the old log do not include source-lookup time.

| Metric | Tri_4 | Tri_7 |
|---|---:|---:|
| Sessions | 23 | 22 |
| ROI features | 53,138 | 183,245 |
| ROIs/session, min / median / max | 2,204 / 2,304 / 2,404 | 6,089 / 8,403.5 / 9,619 |
| Candidate rows | 98,450 | 365,368 |
| Candidates per session pair, min / median / max | 1,412 / 2,288 / 2,417 | 6,448 / 9,085 / 10,493 |
| Session pairs with candidates | 43 | 41 |
| Graph accepted rows | 85,651 | 245,667 |
| Existing QC wall time | 447.23 s (7:27) | 5,208.06 s (1:26:48) |
| `raw_load_seconds` | 81.34 s | 74.86 s |
| `pair_render_seconds` | 98.44 s | 93.28 s |
| `full_stack_reads` | 172 | 164 |
| `shape_metadata_reads` / `geometry_fallbacks` | 0 / 0 | 0 / 0 |
| Existing QC PNG / CSV / total files | 358 / 131 / 491 | 342 / 125 / 469 |

Both saved runs used `sample_limit=20`, `review_seed=0`, `include_skip_pairs=true`, `dpi=150`, `max_examples_per_category=20`, `max_total_examples=100`, and `random_seed=0`.

## Initial finding

In `matching/daywise_roi_matcher_qc_plots.py`, `_generate_spatial_pair_qc` determines each source ROI's graph acceptance by running `any(...)` over the entire `accepted_graph` set. This happens once per source ROI, for every candidate session pair. The scan is quadratic in source/pair count against the global accepted-edge collection and is consistent with the large unexplained Tri_7 runtime: saved logs account for about 168 seconds in image loading and pair rendering, but do not expose the remaining time by phase.

The initial safe optimization is to index string-normalized accepted source labels once per `(session_a, session_b)` and use `Series.map(str).isin(...)` for each pair. TIFF loading changes are deferred unless the optimized profile shows they remain material; the saved baseline spends under 1.5 minutes in raw loads.

## Baseline provenance

Tri_7 matching directory: `/mnt/d/_data/_newAAV_2026/molecular_tracking_derivatives/Fucci-Tri_7/longitudinal/1050/runs/1050_20261001_to_20261105_22s_graph_affine_balanced_image_affine_local/matching`
Tri_4 matching directory: `/mnt/d/_data/_newAAV_2026/molecular_tracking_derivatives/Fucci-Tri_4/longitudinal/1050/runs/1050_20260917_to_20261106_23s_graph_affine_balanced_image_affine_local/matching`

## Change

`_generate_spatial_pair_qc` now builds one map from `(session_a, session_b)` to the string-normalized accepted source labels. Each pair annotates its source ROI table with `label_a.map(str).isin(accepted_labels)`, preserving pair identity, label normalization, duplicate target behavior, and the source-ROI denominator. The QC run log now records `accepted_source_index_seconds` and `accepted_source_annotation_seconds`. No matcher, TIFF loading, image selection, or cache behavior changed.

## Before / after standalone QC

Both new runs used the settings recorded by the old run logs and wrote to separate output directories under `/tmp/molecular_tracking_qc_speedup_20261106/`.

| Metric | Tri_4 old | Tri_4 optimized | Tri_7 old | Tri_7 optimized |
|---|---:|---:|---:|---:|
| Wall time | 447.23 s (7:27) | 195.60 s (3:16) | 5,208.06 s (1:26:48) | 204.71 s (3:25) |
| Speedup | — | 2.29× | — | 25.44× |
| Accepted-source index build | not recorded | 0.023 s | not recorded | 0.088 s |
| Accepted-source annotation | not recorded | 0.036 s | not recorded | 0.099 s |
| Raw-load time | 81.34 s | 80.83 s | 74.86 s | 76.69 s |
| Pair-render time | 98.44 s | 91.76 s | 93.28 s | 88.90 s |
| Full-stack reads | 172 | 172 | 164 | 164 |
| QC CSV / PNG paths | 131 / 358 | 131 / 358 | 125 / 342 | 125 / 342 |

The Tri_7 QC finished below the 10–15 minute goal. If the other stages of the previously reported 2:07:56 full master run remained unchanged, replacing its saved QC time with this measurement projects to about 44:33 total; the full master pipeline was not rerun, so that is only an estimate.

### Output equivalence

- For both datasets, every QC CSV and PNG path was present in both the saved output and new output; saved plot names and counts matched.
- Every PNG was pixel-identical after RGBA decoding.
- Every CSV was byte-identical except `manual_review_sample.csv`. That file has the same 100 selected `(review_category, cluster_id, track_uid)` rows and identical values in all shared columns. The new output includes six additional graph-track columns: `n_accepted_graph_edges`, `n_consensus_edges`, `n_graph_only_edges`, `consensus_edge_fraction`, `has_graph_only_edge`, and `track_match_source`.
- The saved QC logs predate their current `tracks_graph.csv` files: by about 22 seconds for Tri_7 and 7 seconds for Tri_4. This accounts for the older review sample lacking those current track columns. The changed lookup does not affect the review sampler.
- SHA-256 hashes for all 14 input `pairwise_matches_{high,balanced,graph}.csv`, `tracks_{high,balanced,graph}.csv`, and `track_edges_graph.csv` files were identical before and after both standalone QC runs.

### Validation

- Focused QC tests: **13 passed**.
- Full suite: `.venv/bin/python -m pytest tests/ -q` — **446 passed, 2 skipped** in 51.11 seconds.
- No TIFF cache was added. Peak RSS was not measured; raw-load time remained about 77–81 seconds, so a cache change was not justified by these profiles.
- The two benchmark runs were on different dates and filesystem cache/system-load conditions were not controlled. The source-index timer and exact output comparisons directly validate the targeted change; reported wall-time ratios should still be treated as observed run timings, not a controlled lab benchmark.

### Reproduction

```bash
.venv/bin/python matching/daywise_roi_matcher_qc_plots.py \
  --match-dir "$TRI7_MATCH_DIR" \
  --output-dir /tmp/molecular_tracking_qc_speedup_20261106/tri7 \
  --sample-limit 20 --review-seed 0 --dpi 150 \
  --max-examples-per-category 20 --max-total-examples 100 --random-seed 0
```

For Tri_4, substitute `$TRI4_MATCH_DIR` and the `tri4` output path. Both match directories are listed above. The originals and their existing `qc/` folders remain untouched.
