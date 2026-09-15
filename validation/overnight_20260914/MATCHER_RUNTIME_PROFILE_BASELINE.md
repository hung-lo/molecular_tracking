# Matcher runtime profile baseline

- Dataset: `Fucci-Tri_1/1050_20260511_to_20260617_20s_first20_graph_affine_balanced`
- Profile commit: `c91846a`
- Sessions/pairs: 20/37 (19 gap-1, 18 gap-2)
- Configuration: graph + affine balanced, `--pair-workers 1`, QC rendering skipped consistently for timing
- End-to-end wall time: 468.48 s (7:48.48)
- User/system CPU: 433.23/15.67 s
- Reported CPU utilization: 95%
- Maximum resident set reported by GNU time: 1,320,852 KiB

## Stage profile

| Stage | Seconds |
|---|---:|
| input discovery and hashing | 11.09 |
| feature loading and serialization | 20.58 |
| affine pair candidate/transform/assignment | 119.18 |
| affine pair output serialization | 14.89 |
| affine track graph construction and serialization | 33.35 |
| graph input and transform loading | 3.56 |
| graph support/anchor/pair assignment | 211.05 |
| graph track construction | 10.25 |
| graph pair/track serialization | 13.39 |
| QC rendering | skipped consistently |

The per-pair affine subtotal was 119.16 s: global shift 6.41 s, overlap/transform 69.71 s, candidate generation 6.54 s, and assignment 1.28 s; remaining time is mask loading and wrapper overhead. The graph per-pair subtotal was 211.05 s, of which graph-support scoring was 208.15 s.

There were 308,742 scored candidate rows: 156,269 for gap-1 pairs and 152,473 for gap-2 pairs. The pair summaries recorded 197,716 high matches and 215,664 balanced matches. Gap-1 pair wall-time sum was 62.99 s and gap-2 was 56.16 s.

Slowest affine pairs:

| Pair | Gap | Scored pair time (s) |
|---|---:|---:|
| 20260511 → 20260512 | 1 | 4.052 |
| 20260604 → 20260605 | 1 | 3.683 |
| 20260512 → 20260513 | 1 | 3.630 |
| 20260519 → 20260521 | 1 | 3.574 |
| 20260601 → 20260602 | 1 | 3.543 |

Slowest graph-support pairs:

| Pair | Gap | Graph pair time (s) |
|---|---:|---:|
| 20260513 → 20260515 | 2 | 8.755 |
| 20260511 → 20260513 | 2 | 8.587 |
| 20260526 → 20260527 | 1 | 7.677 |
| 20260602 → 20260604 | 2 | 7.413 |
| 20260603 → 20260605 | 2 | 7.043 |

GNU time's maximum RSS is a maximum reported for the measured process tree, not an aggregate simultaneous sum across worker processes. It should not be interpreted as total multi-process memory occupancy.

