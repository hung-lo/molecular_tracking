# Daywise Pipeline

The daywise workflow is:

1. build a manifest of daily masks and daily registered images;
2. run `matching/run_daywise_roi_matching.py` or `matching/run_daywise_graph_matching.py`;
3. run `core/run_daywise_matched_roi_pipeline.py` to extract ROI intensities;
4. run `plotting/run_weekly_matched_output_quick_plots.py` for quick inspection.

## Timing fields

- `session_index` is the zero-based manifest order.
- `acquisition_date` is the actual acquisition date.
- `elapsed_days` is calendar days since the first required session.

## Completeness fields

- `matched_roi_day_table_all.csv` includes every ROI-session row for the chosen policy.
- `matched_roi_day_table_complete.csv` includes only structurally and intensity-complete tracks.

The pipeline separates missing segmentation from failed intensity extraction.

## Normalization and trajectories

Session normalization uses signal-valid native ROIs after a conservative, session-local Green artifact rule (modified z-score plus an upper-tail IQR check). Raw measurements remain unchanged; the OLS-with-intercept fit is saved once per session and applied unchanged to complete and partial track observations. Fit inclusion/exclusion counts and the rule are stored in the fit summary.

Canonical biological outputs use the post-cycle-QC `primary_final_tracks.csv` population. `primary_final_strict_tracks.csv` is the review-free subset; broader complete-track tables remain sensitivity/QC outputs. `policy_status.csv` distinguishes policies that were not requested from policies evaluated with zero tracks.

Trajectory eligibility is a downstream identity/coverage criterion and does not influence the session normalization fit. Its defaults require two usable sessions, with no fraction, internal-gap, first-session, or last-session requirement.

Standard PCA complete-case matrices are centered across the actual PCA cell population. Partial-track matrices preserve `NaN` and are not imputed or variance-scaled automatically.

Cross-laser 920/1050 correspondence is a validation layer and is not required for inclusion in the primary 1050 longitudinal trajectory analysis.

Project-mode master runs also write `acquisition_settings_by_session.csv` and `acquisition_settings_qc.json` for the selected sessions. Changes warn by default; `--require-acquisition-settings-consistent` stops before matching when a required numeric setting changes. Resume reuses extraction only when its `run_log.json` proves v0.3.1 normalization provenance and all required v0.3.1 outputs exist.
