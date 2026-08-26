# Phase 1 path-impact checklist

Status for the authoritative audit:

- `core/analysis_paths.py`: updated to require explicit paths or an explicit legacy root.
- `core/project_cli.py`: added shared mutually-exclusive selection and raw-safe output contract.
- `core/run_daywise_master_pipeline.py`, matched/registered/weekly/920 wrappers: legacy explicit
  paths remain supported; full project-mode wiring is pending final integration.
- Plotting entry points and reusable reference-date defaults: audited; project exact-run wiring
  and removal of historical defaults remain pending final integration.
- Matching engines: intentionally unchanged because they already consume explicit paths.
- Operational and historical notebooks: retained unchanged pending targeted cell-only edits.
- Tests: new config, selector, XML, and catalog-focused synthetic coverage added; old path
  assertions still require conversion.
- `README.md`, `cli_text.txt`, examples, and `.gitignore`: audited; detailed workflow docs were
  added, with existing primary docs still pending consolidation.

This checklist intentionally does not claim complete Phase 1 integration while those pending
items remain.
