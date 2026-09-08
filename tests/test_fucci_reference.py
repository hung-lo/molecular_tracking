import numpy as np
import pandas as pd
import json
from pathlib import Path

from fucci_reference import aggregate_equal_mouse_reference, build_dead_reference


def test_reference_aggregation_weights_mice_equally() -> None:
    sessions = pd.DataFrame({
        "mouse_id": ["dead1", "dead1", "dead1", "dead2"],
        "residual_robust_sd_log2": [0.1, 0.2, 0.3, 0.5],
    })
    mice, reference = aggregate_equal_mouse_reference(sessions)
    assert np.isclose(mice.set_index("mouse_id").loc["dead1", "median_session_robust_sd_log2"], 0.2)
    assert np.isclose(reference, 0.35)


def _reference_run(tmp_path: Path, mouse_id: str, seed: int) -> Path:
    root = tmp_path / mouse_id
    extraction = root / "extraction"
    extraction.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    rows = []
    for session_index, session_id in enumerate(("s0", "s1")):
        for roi in range(60):
            red = 20 + roi
            rows.append({"session_id": session_id, "session_index": session_index, "day": session_index, "acquisition_date": f"2026-01-{session_index + 1:02d}", "elapsed_days": session_index, "red": red, "green": 5 + 0.25 * red + rng.normal(0, 0.5), "ratio_qc_pass": True})
    path = extraction / "matched_session_population_roi_metrics.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    (extraction / "run_log.json").write_text(json.dumps({"analysis_version": "0.3.1", "normalization": {"population": "all_valid_session_rois"}}))
    (root / "run_manifest.json").write_text(json.dumps({"project": {"mouse_id": mouse_id, "laser_nm": 1050}, "manifest": {"session_ids": ["s0", "s1"]}}))
    return root


def test_build_dead_reference_writes_equal_mouse_reference_bundle(tmp_path: Path) -> None:
    output = tmp_path / "reference"
    reference = build_dead_reference([_reference_run(tmp_path, "Dead_1", 1), _reference_run(tmp_path, "Dead_2", 2)], output)
    assert reference["schema_version"] == "fucci_dead_color_reference_v1"
    assert reference["spread"]["reference_robust_sd_log2"] > 0
    assert len(reference["reference_runs"]) == 2
    assert (output / "fucci_dead_reference_sessions.csv").is_file()
    assert (output / "plots/dead_reference_robust_sd_by_session.png").is_file()
