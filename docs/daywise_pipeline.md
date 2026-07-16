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
