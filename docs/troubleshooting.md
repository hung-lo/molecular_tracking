# Troubleshooting

## No QC output

- Re-run the matcher without `--skip-qc`.
- Check `run_log.json` for `qc_status` and `qc_error`.

## Empty match files

- Confirm the manifest contains at least two sessions.
- Confirm the required mask files exist and contain positive ROI labels.

## Policy errors

- Requested policies must exist in the selected match directory.
- `graph` requires `tracks_graph.csv` and the other graph outputs.

## Unexpected complete-table rows

- Check whether the ROI is structurally present on every required day.
- Check whether intensity extraction failed or produced invalid values.
