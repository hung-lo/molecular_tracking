# Legacy derivatives migration audit

This repository now includes a read-only planner for the Phase 2A legacy derivatives review.
It is intentionally conservative: it never moves, copies, renames, deletes, or rewrites source
data. Its only write outputs are the two audit CSVs beneath:

`<derivatives_root>/_catalog/phase2a_audit/`

## What the planner does

- reads the project configuration
- reads the canonical acquisition catalog
- scans only the allowlisted top-level product roots under `legacy_fucci_tri_root`
  - exact roots: `1050_data`, `920_data`, `2wks_1050_data`, `1050_small_test_fireants`
- qc roots: every direct-child root matching `roi_matcher_qc_examples_*` is included when present
- classifies each legacy source file as `session`, `longitudinal`, or `unmapped`
- writes the two audit CSVs atomically beneath `<derivatives_root>/_catalog/phase2a_audit/` using a temporary file plus `os.replace()`:
  - `legacy_derivatives_inventory.csv`
  - `legacy_derivatives_migration_plan.csv`

## Core classification rule

A file may be assigned to a session only when the acquisition date comes from an anchored,
recognized session filename such as:

- `20260511_R.tif`
- `20260511_G.tif`
- `20260511_R_cp_masks_cp_v3_nuclei20.tif`
- `20260511_R_SyN.tif`

The planner does not infer a session from an arbitrary ancestor directory. That is the key
correction for Phase 2A. The CLI also reports the included top-level roots and the ignored entries by name/type so the scan scope is auditable. The first audit recorded 2,920 rows from its effective implementation. The corrected declarative allowlist may produce 2,929 rows if the styled QC tree exists, and that difference is an explicit scope correction rather than a source-data change.

## Longitudinal rule

The following are treated as longitudinal unless the file itself gives stronger explicit
product-specific evidence:

- anything beneath an `analysis/` directory
- complete pipeline-run directories
- ROI tables and cross-day summaries
- population plots and selected-ROI panels
- match/QC output trees
- run metadata and provenance
- run directories with timestamp suffixes such as `YYYYMMDD_HHMMSS`
- `roi_matcher_qc_examples_*` trees

The important part is that a date-like token in a run directory name is not treated as an
acquisition session date.

## Output contract

The inventory includes, at minimum:

- `relative_source_path`
- `source_path`
- `size_bytes`
- `mtime_utc`
- `extension`
- `inferred_mouse_id`
- `inferred_laser_nm`
- `product_class`
- `target_scope`
- `inferred_session_date`
- `date_token_role`
- `catalog_session_match`
- `inference_status`
- `notes`

The migration plan includes, at minimum:

- `source_path`
- `proposed_target`
- `inferred_mouse_id`
- `inferred_laser_nm`
- `product_class`
- `target_scope`
- `inferred_session_date`
- `date_token_role`
- `catalog_session_match`
- `inference_status`
- `action`
- `collision_status`
- `source_size_bytes`
- `source_mtime_utc`
- `reason_evidence`

Every planned action remains `review_required`.

## Example CLI

```bash
python tools/build_legacy_derivatives_plan.py \
  --project-config config/project.local.toml \
  --acquisition-catalog /path/to/acquisitions.generated.csv \
  --include-root 1050_data \
  --include-root 920_data
```

Use `--dry-run` if you want the summary without writing the CSVs. Each `--include-root`
value must be a direct child of `legacy_fucci_tri_root`.

