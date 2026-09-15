# Overnight QC validation and matcher performance report

- Started: 2026-09-14
- Completed: 2026-09-15
- Starting Git commit: `de3cdefbbcffb00ee71f544d56065202e3269424`
- Ending Git commit before this report: `764a4b4`
- Segmentation-QC real-data validation passed: **YES**

Commits created:

- `4eaa337` — Add deterministic profiled pair multiprocessing
- `c91846a` — Add exact matcher output equivalence checker
- `21ec6d4` — Ignore resolved output path in matcher equivalence
- `d6bb81b` — Cache exact graph anchor neighborhoods
- `764a4b4` — Record graph runner commit after resumed baseline

## Segmentation QC validation

- Run: `Fucci-Dead_1/longitudinal/1050/runs/1050_20260819_to_20260824_4s_first4_graph_affine_balanced`
- Upstream counts remained exact: 6,983 all tracks; 5,054 at least two sessions; 3,251 complete; 3,251 zero-hit complete; 3,216 cycle-QC.
- Segmentation-QC count changed from the erroneous 0 to 3,216.
- Run-level status: `not_configured_bypassed`.
- Filter-step status: `bypassed_not_configured`.
- Per-track status/NA/failure fields all match the required semantics.
- Graph primary output: 3,216. High/balanced named outputs are empty because the exact original run requested only graph policy, not because of QC.
- Validation: **PASS**.

## Baseline profile

- Preferred 20-session Fucci-Tri_1 dataset, 37 session pairs.
- End-to-end 1-worker wall: 468.48 s; second repetition 449.09 s.
- Largest stage: graph support/anchor/pair assignment, 211.05 s, including 208.15 s graph-support scoring.
- Affine pair candidate/transform/assignment: 119.18 s, dominated by overlap/transform at 69.71 s.
- Peak GNU-time RSS: 1,320,852–1,321,324 KiB for serial repetitions.
- QC plotting was skipped consistently in matcher benchmarks; canonical summary/run-log writing remained enabled.

## Pair-level parallel benchmark before KD-tree optimization

| Workers | Wall rep 1 | Wall rep 2 | Mean wall | Mean speedup | Parallel efficiency | GNU-time max RSS range | Equivalence |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 468.48 s | 449.09 s | 458.79 s | 1.00× | 100% | 1,320,852–1,321,324 KiB | PASS |
| 2 | 284.90 s | 286.14 s | 285.52 s | 1.61× | 80.3% | 1,116,448–1,116,696 KiB | PASS |
| 4 | 222.01 s | 213.27 s | 217.64 s | 2.11× | 52.7% | 1,116,848–1,117,524 KiB | PASS |

The first serial run had the coldest filesystem state; its 4% difference from the second run is reported rather than hidden. Two- and four-worker repetitions were highly stable. GNU time's max RSS is not an aggregate simultaneous sum across workers.

## Exact equivalence

- 1 vs 2 workers: **PASS**
- 1 vs 4 workers: **PASS**
- Repeated identical 1/2/4-worker invocations: **PASS**
- Scientific hashes/content identical: **YES**
- Pre/post KD-tree optimization: **PASS**
- Optimized 1 vs 4 workers: **PASS**

All candidate, accepted-match, transform, graph/cycle, edge, track-ID, track-table, and manifest artifacts were identical. Timing fields and output-specific paths were the only exclusions.

## Optimizations

1. Added `--pair-workers N` to affine, graph, and master runners. Default and `N=1` preserve serial behavior; `N>1` computes independent pairs in worker processes, collects results in canonical pair order, and leaves final track graph construction serial.
2. Added detailed non-scientific runtime profiling at matcher, graph, and per-pair stages.
3. Added an automated exact-output equivalence checker and tests.
4. After profiling showed graph support dominant, cached per-pair anchor coordinate arrays and exact KD-tree radius neighborhoods. Exact Euclidean radius rechecks and original neighbor ranking preserve boundary and tie semantics.

The KD-tree/cache optimization reduced serial graph pair processing from 211.05 s to 20.38 s (10.36×), graph stage wall from 240.00 s to 48.60 s, and serial end-to-end wall from 468.48 s to 265.60 s (43.3% shorter). The combined optimized 4-worker run completed in 161.37 s (2:41.37), 2.90× faster than the original serial baseline. Its graph pair work was 5.98 s; affine pair computation and serial construction/serialization are now the larger regions.

No thresholds, scores, acceptance rules, candidate definitions, anchor definitions, ambiguity rules, cycle logic, or assignment logic changed.

## Tests

- Segmentation-QC focused test: `8 passed`.
- Matcher/parallel/equivalence/QC focused selection: `41 passed` before the KD-tree addition.
- KD-tree and graph/equivalence focused selection: `5 passed`.
- Full repository suite: **300 passed, 2 skipped** in 117.55 s.
- Documented prior baseline: 291 passed, 1 skipped. The current count includes newly added tests and an additional environment-appropriate skip; there were no failures.

## Hard constraints

- Canonical matcher behavior changed: **NO**
- Matching thresholds changed: **NO**
- Track IDs changed: **NO**
- Endpoint stitcher changed: **NO**
- ECLIPSE/intensity logic changed: **NO**
- Green/Red normalization changed: **NO**
- Segmentation scientific criteria changed: **NO**
- Phase D segmentation rescue started: **NO**
- Endpoint-targeted search implemented: **NO**
- Canonical matching/extraction outputs overwritten: **NO**

## Recommended next steps

1. Profile affine overlap/transform internals under 4 workers; this is now the largest genuinely parallel compute region.
2. Reduce serial track construction/CSV serialization overhead only with the same exact-output gate.
3. Measure aggregate process-tree memory with an external sampler if production capacity planning requires a true concurrent-memory figure.

