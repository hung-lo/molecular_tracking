# Matcher output equivalence report

- Scientific output equivalence: **PASS**
- Repeated 1-worker runs: **PASS**
- Repeated 2-worker runs: **PASS**
- Repeated 4-worker runs: **PASS**
- 1 worker vs 2 workers: **PASS**
- 1 worker vs 4 workers: **PASS**
- Before vs after KD-tree/cache optimization: **PASS**
- Optimized 1 worker vs optimized 4 workers: **PASS**

All 24 canonical matcher CSV artifacts were checked. Candidate tables, transforms, high/balanced/graph matches, graph changes, cycle outputs, track edges, track tables/IDs, length summaries, ROI features, and the resolved session manifest were byte-identical. Only `elapsed_sec` was excluded from pair summary CSVs. Run-log comparison excluded only timing/timestamps, output-specific paths, process configuration (`pair_workers`), and other explicitly non-scientific runtime metadata.

