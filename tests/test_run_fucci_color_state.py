import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from run_fucci_color_state import run_color_state
from run_fucci_state_analysis import run_state_analysis


def _master_run(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "master"
    extraction = root / "extraction"
    extraction.mkdir(parents=True)
    session_ids = ["s0", "s1", "s2"]
    native_rows = []
    for session_index, session_id in enumerate(session_ids):
        for roi in range(60):
            red = 20 + roi
            native_rows.append({"session_id": session_id, "session_index": session_index, "day": session_index, "acquisition_date": f"2026-01-{session_index + 1:02d}", "elapsed_days": session_index, "red": red, "green": 5 + 0.25 * red + (roi % 3) * 0.1, "ratio_qc_pass": True})
    pd.DataFrame(native_rows).to_csv(extraction / "matched_session_population_roi_metrics.csv", index=False)
    matched = pd.DataFrame({"match_policy": "graph", "track_uid": "track1", "roi_id": 1, "session_id": session_ids, "session_index": [0, 1, 2], "acquisition_date": ["2026-01-01", "2026-01-02", "2026-01-03"], "elapsed_days": [0, 1, 2], "green": [10.0, 9.0, 8.0], "red": [20.0, 20.0, 20.0], "ratio_qc_pass": True})
    matched.to_csv(extraction / "matched_roi_log_ratio_metrics_all_observed.csv", index=False)
    pd.DataFrame({"match_policy": "graph", "roi_id": [1, 1, 1], "session_index": [0, 1, 2], "geometry_qc_pass": True}).to_csv(extraction / "matched_roi_geometry_qc_long.csv", index=False)
    pd.DataFrame({"match_policy": "graph", "track_uid": ["track1"], "roi_id": [1]}).to_csv(extraction / "matched_track_qc_summary.csv", index=False)
    (extraction / "run_log.json").write_text(json.dumps({"analysis_version": "0.3.1", "normalization": {"population": "all_valid_session_rois"}}))
    (root / "run_manifest.json").write_text(json.dumps({"project": {"mouse_id": "Tri_1", "laser_nm": 1050}, "manifest": {"session_ids": session_ids}}))
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({"schema_version": "fucci_dead_color_reference_v1", "spread": {"reference_robust_sd_log2": 0.2}, "fit": {"min_fit_rois": 50, "bandwidth_scale": 0.6}, "state_thresholds_z": {}}))
    return root, reference


def test_color_state_output_is_additive_and_protected(tmp_path: Path) -> None:
    root, reference = _master_run(tmp_path)
    result = run_color_state(root, reference)
    output = Path(result["output_paths"]["observations"]).parent.parent
    assert (output / "run_manifest.json").is_file()
    scored = pd.read_csv(output / "normalization/matched_roi_color_state_all_observed.csv")
    assert {"predicted_green_modal", "log2_green_over_expected", "color_z"}.issubset(scored.columns)
    assert output != root / "extraction"
    with pytest.raises(FileExistsError):
        run_color_state(root, reference)


def test_state_analysis_writes_color_z_trajectory_and_pca_outputs(tmp_path: Path) -> None:
    root, reference = _master_run(tmp_path)
    result = run_color_state(root, reference)
    output = Path(result["output_paths"]["observations"]).parent.parent
    run_state_analysis(output, event_min_sessions=2)
    assert (output / "trajectory/color_z_trajectory_matrix.csv").is_file()
    assert (output / "pca/color_z_pca_scores.csv").is_file()
    assert (output / "events/color_state_entry_events.csv").is_file()
