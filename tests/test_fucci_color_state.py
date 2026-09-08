import numpy as np
import pandas as pd

from fucci_color_state import (
    color_core_state,
    color_state_bin,
    fit_linear_modal_backbone,
    robust_sd_mad,
    score_color_state_table,
)


def _line_data(seed: int = 0, n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    red = np.linspace(10, 100, n)
    green = 5 + 0.25 * red + rng.normal(0, 0.4, n)
    return pd.DataFrame({"red": red, "green": green, "ratio_qc_pass": True})


def test_modal_fit_recovers_clean_backbone_and_is_deterministic() -> None:
    data = _line_data()
    first = fit_linear_modal_backbone(data)
    second = fit_linear_modal_backbone(data)
    assert abs(first.modal_intercept - 5) < 0.5
    assert abs(first.modal_slope - 0.25) < 0.02
    assert np.isclose(first.modal_intercept, second.modal_intercept)
    assert np.isclose(first.modal_slope, second.modal_slope)


def test_modal_fit_resists_lower_tail_contamination() -> None:
    data = _line_data(seed=4)
    contaminated = data.copy()
    contaminated.loc[:31, "green"] *= 0.3
    modal = fit_linear_modal_backbone(contaminated)
    ols_slope = np.polyfit(contaminated["red"], contaminated["green"], 1)[0]
    assert abs(modal.modal_slope - 0.25) < abs(ols_slope - 0.25)
    assert (contaminated.loc[:31, "green"] < modal.predict(contaminated.loc[:31, "red"])).all()


def test_robust_sd_mad_and_invalid_score() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    mad, robust = robust_sd_mad(values, return_mad=True)
    assert mad == 1.0
    assert robust == 1.4826
    assert robust < np.std(values, ddof=1)
    fits = pd.DataFrame({"session_id": ["s0"], "modal_slope": [-1.0], "modal_intercept": [1.0], "modal_bandwidth_green_units": [1.0]})
    scored = score_color_state_table(pd.DataFrame({"session_id": ["s0"], "green": [5.0], "red": [2.0], "ratio_qc_pass": [True]}), fits, 0.2)
    assert pd.isna(scored.loc[0, "color_z"])
    assert scored.loc[0, "color_state_qc_reason"] == "nonpositive_or_nonfinite_predicted_green"


def test_color_state_boundaries() -> None:
    assert color_state_bin(-2.001) == "strong_low"
    assert color_state_bin(-2) == "low_transition"
    assert color_state_bin(-1) == "middle"
    assert color_state_bin(1) == "middle"
    assert color_state_bin(1.001) == "high_transition"
    assert color_state_bin(2) == "high_transition"
    assert color_state_bin(2.001) == "strong_high"
    assert color_core_state(-2) is None
    assert color_core_state(-2.001) == "low"
    assert color_core_state(2.001) == "high"
