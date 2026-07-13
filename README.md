# molecular_tracking

Small repository for ROI and molecular-tracking analysis code with a clean split between reusable logic, plotting scripts, matching utilities, tests, and intentionally kept notebooks.

## Packages
- antspy
- cellpose


## Repository layout

```text
molecular_tracking/
├── core/
│   ├── analysis_paths.py
│   ├── run_daywise_matched_roi_pipeline.py
│   ├── roi_log_ratio_analysis.py
│   ├── run_920_two_day_cp3nuclei_analysis.py
│   └── run_registered_roi_pipeline.py
├── matching/
│   ├── affine_overlap_matcher.py
│   ├── daywise_roi_matcher_qc_plots.py
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
│   ├── test_affine_overlap_matcher.py
│   ├── test_daywise_matched_roi_pipeline.py
│   ├── test_daywise_roi_matcher_qc_plots.py
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
python core/run_daywise_matched_roi_pipeline.py
python core/run_920_two_day_cp3nuclei_analysis.py
python matching/run_daywise_roi_matching.py
python plotting/run_daywise_green_red_linear_fit_summary.py
pytest tests/
```

## Choosing a workflow

- Use the legacy weekly workflow when you have weekly average masks plus the daywise registered `*_SyN.tif` images that came out of the `weeklyRegister` notebook.
- Use the new daywise workflow when you segment each day separately, for example with Cellpose or SAM, and want matching to run directly on those daily masks.
- The new daywise workflow does not replace the weekly workflow. It runs alongside it and reuses the same downstream dark-correction and green/red metric helpers.

## Weekly workflow example (`crop_512`)

Example data folder:
- Windows path: `D:\_data\_newAAV_2026\weekly_registration_test\crop_512`
- WSL path used in commands below: `/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512`

Use WSL-style paths when running from the bash terminal in this repo. In other words, use `/mnt/d/...` instead of `D:\...`.

Overall order:
1. Run the `weeklyRegister` notebook first.
2. Run `matching/roi_matcher.py` on the weekly registered average masks.
3. Run `matching/roi_matcher_qc_plots.py` to sanity-check the matches.
4. Run `core/run_weekly_matched_roi_pipeline.py` to extract daywise green/red ROI values from the matched ROIs.
5. Run `plotting/run_weekly_matched_output_quick_plots.py` on the step-4 output directory.

What step 1 should leave in the dataset folder:
- Registered daywise images such as `20260511_R_crop_512_SyN.tif` and `20260511_G_crop_512_SyN.tif`.
- Weekly registered masks for the matcher such as `week1_average_cp_masks_SyN.tif` through `week9_average_cp_masks_SyN.tif`.
- Weekly non-SyN Cellpose masks for the weekly matched ROI pipeline such as `week1_average_cp_masks.tif` or `week1_average_cp_mask.tif`.

### Step 2: ROI matcher

```bash
uv run python matching/roi_matcher.py \
  --masks \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week1_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week2_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week3_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week4_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week5_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week6_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week7_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week8_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week9_average_cp_masks_SyN.tif" \
  --days week1 week2 week3 week4 week5 week6 week7 week8 week9 \
  --output-prefix /mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/roi_match_runs/20260710_9wks
```

This writes the matcher CSVs into `roi_match_runs/`, including:
- `/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/roi_match_runs/20260710_9wks.csv`
- `/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/roi_match_runs/20260710_9wks_qc.csv`
- `/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/roi_match_runs/20260710_9wks_run_log.json`

### Step 3: ROI matcher QC plots

```bash
uv run python matching/roi_matcher_qc_plots.py \
  --masks \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week1_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week2_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week3_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week4_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week5_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week6_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week7_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week8_average_cp_masks_SyN.tif" \
    "/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/week9_average_cp_masks_SyN.tif" \
  --days week1 week2 week3 week4 week5 week6 week7 week8 week9 \
  --output-dir /mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/roi_match_runs/20260710_9wks_qc \
  --examples-per-group 4
```

This step is optional but strongly recommended before you trust the matched ROI table.

### Step 4: Weekly matched ROI pipeline

```bash
uv run python core/run_weekly_matched_roi_pipeline.py \
  --dataset /mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512 \
  --match-csv /mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/roi_match_runs/20260710_9wks.csv \
  --start-date 20260511
```

Notes:
- `--dataset` should point to the folder that contains the weekly masks plus the daywise `*_SyN.tif` images.
- `--start-date` can be omitted if the earliest non-SyN raw TIFF in that folder is your true day 0, but keeping it explicit is safer.
- The weekly mask name usually does not need to be passed because the script already tries both `{week_name}_average_cp_masks.tif` and `{week_name}_average_cp_mask.tif`.
- This script uses the matched ROI CSV from step 2, the non-SyN weekly average masks, and the daywise registered `*_SyN.tif` images.
- At the end it prints `output_dir=...`. Copy that path for step 5.

### Step 5: Quick plots from the weekly matched output

Replace `PASTE_OUTPUT_DIR_FROM_STEP_4` with the exact `output_dir=...` path printed by step 4.

```bash
uv run python plotting/run_weekly_matched_output_quick_plots.py \
  --analysis-dir PASTE_OUTPUT_DIR_FROM_STEP_4 \
  --start-date 20260511 \
  --top-n 30
```

If you only want to re-run from an existing matcher CSV later, skip steps 1 to 3 and start directly from step 4.

Reusable re-run template:

```bash
uv run python core/run_weekly_matched_roi_pipeline.py \
  --dataset /mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512 \
  --match-csv /mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_512/roi_match_runs/YOUR_MATCH_FILE.csv \
  --start-date 20260511
```

If dataset paths need to change, update `core/analysis_paths.py` or pass dataset-specific paths through script arguments where supported.

## Daywise workflow example (`cellpose_or_sam_daily_masks`)

Use this path when every day has its own segmentation mask and the mask, red image, and green image all share the same voxel space.

Required manifest columns:

- `session_index`
- `session_id`
- `acquisition_date`
- `mask_path`
- `red_image_path`
- `green_image_path`
- `required`

Example manifest row:

```csv
session_index,session_id,acquisition_date,mask_path,red_image_path,green_image_path,required
0,20260511,2026-05-11,masks/20260511_R_cp_masks.tif,images/20260511_R.tif,images/20260511_G.tif,true
```

### Step 1: run the daywise matcher

```bash
uv run python matching/run_daywise_roi_matching.py \
  --manifest /path/to/daywise_session_manifest.csv \
  --output-dir /path/to/roi_match_runs/20260713_affine_overlap_v1 \
  --xy-um-per-px 0.693359375 \
  --z-um-per-plane 5.0 \
  --max-pair-gap 2
```

Common optional flags:

- `--save-candidates` writes the full candidate table for audit/debugging.
- `--overwrite` replaces an existing output directory.
- `--resume` reuses an existing output directory only when the manifest, inputs, parameters, and algorithm version match exactly.

The matcher writes:

- `session_manifest_resolved.csv`
- `roi_features.csv`
- `pairwise_summary.csv`
- `pairwise_transforms.csv`
- `pairwise_matches_high.csv`
- `pairwise_matches_balanced.csv`
- `cycle_consistency_high.csv`
- `cycle_consistency_balanced.csv`
- `tracks_high.csv`
- `tracks_balanced.csv`
- `track_length_summary.csv`
- `run_log.json`

### Step 2: optional QC plots

```bash
uv run python matching/daywise_roi_matcher_qc_plots.py \
  --match-dir /path/to/roi_match_runs/20260713_affine_overlap_v1 \
  --output-dir /path/to/roi_match_runs/20260713_affine_overlap_v1/qc_plots
```

This creates review-sample CSVs and pair/track QC figures.

### Step 3: run the daywise matched ROI pipeline

```bash
uv run python core/run_daywise_matched_roi_pipeline.py \
  --dataset /path/to/dataset_root \
  --manifest /path/to/daywise_session_manifest.csv \
  --match-dir /path/to/roi_match_runs/20260713_affine_overlap_v1 \
  --policies high balanced \
  --green-dark 319 \
  --red-dark 534
```

This step:

- validates that the manifest matches the matcher output;
- extracts ROI intensities from the original daily images using the matched daily masks;
- applies the existing dark-correction and green/red metric helpers;
- writes the daywise analysis tables and QC summaries.

The daywise analysis output directory contains tables such as:

- `matched_roi_intensity_results_raw.csv`
- `matched_roi_intensity_results_dark_corrected.csv`
- `matched_roi_day_table_complete.csv`
- `matched_track_qc_summary.csv`
- `matched_daywise_green_red_linear_fit_summary.csv`
- `primary_high_complete_matching.csv`
- `sensitivity_balanced_complete.csv`
- `review_flagged_tracks.csv`

### How it fits the current workflow

- The weekly workflow stays exactly as it is today.
- The daywise workflow is a parallel path for datasets where daily segmentation is the right starting point.
- Both workflows share the same downstream intensity-analysis helpers, so the daywise tables should feel familiar if you already use the registered ROI pipeline.
- If you are unsure which path to use, start with the weekly workflow for older datasets and use the daywise workflow when you already trust the daily masks.
