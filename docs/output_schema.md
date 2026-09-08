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

## Daywise analysis tables

- `matched_session_population_roi_metrics.csv`: every native ROI in every required session, with corrected signals and signal-validity flags.
- `matched_session_population_signal_qc_summary.csv`: per-session zero-hit and positive-signal counts.
- `matched_daywise_green_red_linear_fit_summary.csv`: canonical OLS fits from all signal-valid native session ROIs.
- `matched_roi_day_table_all.csv`: every observed longitudinal ROI/session row.
- `matched_roi_day_table_complete.csv`: only tracks observed in every required session.
- `matched_roi_metrics_with_session_normalized_residuals_all_observed.csv`: canonical fit residuals for complete and partial observations.
- `matched_daywise_green_red_linear_fit_summary_complete_track_sensitivity.csv`: legacy complete-track-only fit for sensitivity analysis.
- `matched_roi_trajectory_eligibility.csv`: track-level usable-session counts and eligibility reasons.
- `matched_roi_trajectory_observations_eligible.csv`: usable long-form observations from eligible tracks.
- `matched_roi_trajectory_signed_distance_matrix.csv`: signed-distance trajectories with missing values retained as `NaN`.
- `matched_roi_trajectory_observation_mask.csv`: matching 1/0 observation mask.
- `matched_roi_trajectory_missingness_by_session.csv`: per-session missingness summary.
- `matched_roi_trajectory_signed_distance_matrix_centered_with_nan.csv`: observed values centered by session, with `NaN` retained.
- `matched_roi_pca_complete_case_raw.csv` and `matched_roi_pca_complete_case_centered.csv`: complete-case PCA inputs; the centered file has no imputation or variance scaling.

## Fucci color-state postprocessing

- `postprocess/fucci_color_state/normalization/session_population_eclipse_state.csv`: all native session ROIs scored with modal expected reporter, ECLIPSE deviation, fixed Dead-referenced `eclipse_z`, and state labels.
- Canonical ECLIPSE score columns include `eclipse_ratio`, `log2_eclipse_ratio`, `eclipse_deviation_log2`, `eclipse_z`, `eclipse_state_bin`, `eclipse_core_state`, `eclipse_state_qc_pass`, and `eclipse_state_qc_reason`; current green is reporter and red is reference.
- `postprocess/fucci_color_state/normalization/matched_roi_color_state_all_observed.csv`: matched observations with the same canonical `eclipse_*` fields; legacy `color_*` aliases remain during the transition release.
- `postprocess/fucci_color_state/normalization/eclipse_state_occupancy_by_session.csv`: primary state occupancy from all valid native session ROIs.
- `postprocess/fucci_color_state/normalization/matched_only_eclipse_state_occupancy_by_session.csv`: explicitly matched-only occupancy diagnostic.
- `postprocess/fucci_color_state/normalization/color_state_session_fits.csv`: per-session native-population modal fits and diagnostics.
- `postprocess/fucci_color_state/trajectory/`: color-Z eligibility and NaN-preserving trajectory matrices.
- `postprocess/fucci_color_state/pca/`: complete-case NumPy SVD scores, loadings, and explained variance.
- `postprocess/fucci_color_state/events/`: hysteretic middle-to-low/high entry events and event-aligned summaries.
- `postprocess/fucci_color_state/run_manifest.json`: source hashes, reference hash, settings, and output provenance.
