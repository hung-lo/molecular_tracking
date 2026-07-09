# molecular_tracking

Small repository for ROI and molecular-tracking analysis code with a clean split between reusable logic, plotting scripts, matching utilities, tests, and intentionally kept notebooks.

## Repository layout

```text
molecular_tracking/
├── core/
│   ├── analysis_paths.py
│   ├── roi_log_ratio_analysis.py
│   ├── run_920_two_day_cp3nuclei_analysis.py
│   └── run_registered_roi_pipeline.py
├── matching/
│   ├── roi_matcher.py
│   └── roi_matcher_qc_plots.py
├── notebooks/
│   ├── demo_registered_roi_pipeline_1050.ipynb
│   ├── roi_intensity_manual_plotting_20260526.ipynb
│   ├── roi_raw_space_triplet_panels_1050.ipynb
│   └── roi_shared_raw_space_group_panel_1050.ipynb
├── plotting/
│   ├── raw_space_triplet_panels.py
│   ├── run_daywise_green_red_fit_residuals.py
│   ├── run_daywise_green_red_fit_residuals_increasing.py
│   ├── run_daywise_green_red_linear_fit_summary.py
│   ├── run_raw_space_inverse_mask_validation.py
│   └── shared_raw_space_group_panel.py
├── tests/
│   ├── test_roi_log_ratio_analysis.py
│   ├── test_roi_matcher.py
│   ├── test_roi_matcher_qc_plot_contours.py
│   ├── test_roi_matcher_qc_plot_single_plane.py
│   ├── test_roi_matcher_qc_plot_style.py
│   └── test_roi_matcher_qc_plots.py
├── .gitignore
└── README.md
```

## What goes where

- `core/`: reusable analysis code and main pipeline entry points.
- `plotting/`: figure generation, summaries, and raw-space validation panels.
- `matching/`: ROI matching logic plus QC plotting helpers tied to matching.
- `tests/`: tests for reusable analysis and matching behavior.
- `notebooks/`: only intentionally kept demo or reference notebooks.

## Data and outputs

This repo is meant to stay code-first and reproducible.

- Keep `1050_data/` and `920_data/` as external or local-only data folders.
- Do not commit regenerated analysis outputs, figures, CSV exports, caches, or checkpoints.
- Track source code, tests, and curated notebooks; regenerate outputs locally as needed.

## Working style

- Put reusable logic in `core/` or `matching/`, not in notebooks.
- Keep one-off plotting or figure scripts in `plotting/`.
- Add tests in `tests/` when reusable behavior changes.
- Keep notebooks slim and move stabilized logic back into Python modules.

## Running from the repo root

Examples:

```bash
python core/run_registered_roi_pipeline.py
python core/run_920_two_day_cp3nuclei_analysis.py
python plotting/run_daywise_green_red_linear_fit_summary.py
pytest tests/
```

If dataset paths need to change, update `core/analysis_paths.py` or pass dataset-specific paths through script arguments where supported.
