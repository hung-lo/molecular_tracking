# Phase 1 path-impact checklist

- `core/analysis_paths.py`: updated; no source-directory or global-mouse fallback. Explicit
  legacy paths remain supported, and aliases require an explicit legacy root.
- `core/project_config.py`, `core/project_cli.py`: added validated, non-nested roots; shared
  selector; mouse validation; exact-run validation; processed-product resolution; stale-ready
  manifest checks; selected mouse/laser output containment.
- `core/run_daywise_master_pipeline.py`: updated for mutually exclusive legacy/project modes,
  validated automatic manifests, XML spacing and overrides, run-contained extraction work, and
  complete mouse/catalog/session provenance.
- `core/run_daywise_matched_roi_pipeline.py`: updated for shared project selection and an
  explicit output root. Legacy dataset/manifest/match paths remain supported.
- `core/run_registered_roi_pipeline.py`: updated; project mode resolves only a prepared
  `registered` product and writes under selected derivatives runs. Legacy explicit paths remain.
- `core/run_weekly_matched_roi_pipeline.py`: updated; project mode resolves only a prepared
  `weekly_registered` product, requires its published metadata, and validates it before creating
  a run output directory. Legacy explicit paths remain.
- `core/run_920_two_day_cp3nuclei_analysis.py`: updated; explicit legacy mode is valid even
  though the wrapper forces 920, while project mode validates a prepared 920 `registered`
  product and passes the project run root explicitly so outputs land under
  `<mouse>/longitudinal/920/runs`.
- `core/roi_log_ratio_analysis.py`: updated; reusable reference dates are explicit/nullable.
- Residual, linear-summary, raw-space, and quick-plot CLIs under `plotting/`: updated for exact
  analysis/run selection; project inputs are validated against an explicit config/mouse/laser;
  no latest-run selection exists. Legacy exact inputs remain supported.
- `plotting/raw_space_triplet_panels.py` and `plotting/shared_raw_space_group_panel.py`: updated;
  reusable dates are explicit/nullable.
- `notebooks/weeklyRegister_20260531.ipynb`: setup/path cell now reads flat source TIFFs
  from `registered/`, stages the full weekly plus cross-week compatibility product under
  `weekly_registered/.staging/`, publishes the flat validated `weekly_registered/` product only
  once after complete validation, and records stable cropped and uncropped filenames plus explicit
  crop metadata. The published metadata covers every allowlisted file, including cross-week
  registered masks. ANTs calls receive XML-derived ZYX spacing. Crop, `genericLabel`, and
  geometry logic is retained, and the workflow currently keeps weekly averages red-only unless a
  separate green product is implemented explicitly.
- `notebooks/cellposeSAM_batch_segmentation_20260712.ipynb`: setup/input/output cell now uses
  explicit project selection and derivative preprocessing/segmentation paths.
- Four historical analysis notebooks: intentionally retained as legacy/reference notebooks with
  prominent first-cell warnings; no mouse is inferred.
- `tests/test_roi_log_ratio_analysis.py`: old global-root assertions replaced by explicit-path
  compatibility tests; synthetic dates are now passed explicitly.
- `tests/test_roi_matcher.py`: optional real-data test is gated by
  `MOLECULAR_TRACKING_REAL_MATCHER_DIR`; synthetic tests remain the CI requirement.
- `README.md`, `cli_text.txt`, `.gitignore`, and `examples/README.md`: updated; project workflow is
  primary, old commands are labeled legacy, and local config/generated metadata are ignored.
- `examples/daywise_session_manifest.csv`: intentionally retained as a labeled synthetic
  legacy-format example.
- Low-level matching CLIs under `matching/`: intentionally unchanged exact-path tools.

Phase 1 performs logical discovery and derivative catalog/manifest preparation only. Physical
data migration and preprocessing execution remain separate future operations requiring a dry-run
plan and explicit approval. Raw ThorImage folders are never moved or rewritten.
