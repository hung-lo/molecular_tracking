# Output Schema

## Baseline tables

- `pairwise_summary.csv`: one row per matched session pair.
- `pairwise_transforms.csv`: restricted transform parameters for each pair.
- `pairwise_matches_high.csv`: one-to-one high-confidence pair assignments.
- `pairwise_matches_balanced.csv`: one-to-one balanced assignments.
- `tracks_high.csv` and `tracks_balanced.csv`: longitudinal tracks by policy.
- `track_edges_high.csv` and `track_edges_balanced.csv`: accepted edges.
- `cycle_consistency_high.csv` and `cycle_consistency_balanced.csv`: adjacent triplet checks.
- `cycle_edge_checks_high.csv` and `cycle_edge_checks_balanced.csv`: row-level cycle evidence.
- `track_length_summary.csv`: track counts grouped by policy and track length.

## Graph tables

- `pairwise_matches_graph.csv`
- `pairwise_summary_graph.csv`
- `tracks_graph.csv`
- `cycle_consistency_graph.csv`
- `cycle_edge_checks_graph.csv`
- `track_edges_graph.csv`
- `track_length_summary_graph.csv`
- `graph_match_changes.csv`

Empty files keep the same headers as populated files.
