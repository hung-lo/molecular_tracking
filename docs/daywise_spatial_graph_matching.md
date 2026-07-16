# Daywise Spatial Graph Matching

The spatial-graph matcher is a conservative refinement layer that reuses the baseline
pairwise candidates and high-confidence anchors.

## Design

- Strict `high` matches are used as anchors.
- Candidate refinement is limited to baseline-balanced pairs.
- B-session centroids are transformed into A-session physical space.
- The graph score is a local geometric consistency score, not a learned model.

## Outputs

The graph runner writes baseline `high` and `balanced` outputs plus:

- `pairwise_matches_graph.csv`
- `pairwise_summary_graph.csv`
- `tracks_graph.csv`
- `cycle_consistency_graph.csv`
- `cycle_edge_checks_graph.csv`
- `track_edges_graph.csv`
- `track_length_summary_graph.csv`
- `graph_match_changes.csv`

The graph policy is experimental and should not be treated as the default baseline.
