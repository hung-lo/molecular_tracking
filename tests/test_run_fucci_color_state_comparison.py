import json
from pathlib import Path

import pandas as pd

from run_fucci_color_state_comparison import compare_color_state_runs


def _color_state_dir(tmp_path: Path, name: str, low_percentages: list[float]) -> Path:
    source = tmp_path / f"{name}_master"
    extraction = source / "extraction"
    extraction.mkdir(parents=True)
    native = pd.DataFrame({"session_id": ["s0", "s1"], "session_index": [0, 1], "red": [10.0, 10.0], "green": [12.0, 12.0], "ratio_qc_pass": [True, True]})
    native.to_csv(extraction / "matched_session_population_roi_metrics.csv", index=False)
    color_dir = tmp_path / name
    (color_dir / "normalization").mkdir(parents=True)
    scored = pd.DataFrame({"color_z": [-2.5, 0.0], "session_id": ["s0", "s1"]})
    scored.to_csv(color_dir / "normalization/matched_roi_color_state_all_observed.csv", index=False)
    occupancy = pd.DataFrame({"session_id": ["s0", "s1"], "n_valid_observations": [10, 10], "n_strong_low": [1, 2], "n_low_transition": [1, 1], "n_middle": [6, 6], "n_high_transition": [1, 1], "n_strong_high": [1, 0], "pct_strong_low": low_percentages, "pct_low_transition": [10, 10], "pct_middle": [60, 60], "pct_high_transition": [10, 10], "pct_strong_high": [10, 0]})
    occupancy.to_csv(color_dir / "normalization/eclipse_state_occupancy_by_session.csv", index=False)
    pd.DataFrame({"session_id": ["s0", "s1"], "session_index": [0, 1], "modal_slope": [0.2, 0.2], "modal_intercept": [5, 5]}).to_csv(color_dir / "normalization/color_state_session_fits.csv", index=False)
    (color_dir / "run_manifest.json").write_text(json.dumps({"source_master_run_dir": str(source)}))
    return color_dir


def test_comparison_chooses_and_records_representative_session(tmp_path: Path) -> None:
    first = _color_state_dir(tmp_path, "dead1", [5, 20])
    second = _color_state_dir(tmp_path, "dead2", [15, 25])
    output = tmp_path / "comparison"
    result = compare_color_state_runs([first, second], output, labels=["Dead_1", "Dead_2"])
    assert result["representative_sessions"]["Dead_1"]["session_index"] == 0
    assert (output / "comparison_session_state_occupancy.csv").is_file()
    assert (output / "comparison_green_red_sd_zones.png").is_file()
