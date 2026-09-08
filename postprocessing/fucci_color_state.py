"""Modal green-vs-red fitting and Dead-referenced Fucci color state scores."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


MODAL_METHOD = "linear_gaussian_kernel_modal"
MODAL_BANDWIDTH_SCALE = 0.6
STATE_THRESHOLDS = {
    "strong_low": -2.0,
    "middle_low": -1.0,
    "middle_high": 1.0,
    "strong_high": 2.0,
}


def _boolean_values(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    normalized = values.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def robust_sd_mad(
    values: Any, *, return_mad: bool = False, dropna: bool = True
) -> float | tuple[float, float]:
    """Return ``1.4826 * MAD``; NaNs are dropped by default."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if dropna:
        array = array[np.isfinite(array)]
    elif not np.isfinite(array).all():
        raise ValueError("robust_sd_mad received non-finite values")
    if array.size == 0:
        result = (math.nan, math.nan)
    else:
        median = float(np.median(array))
        mad = float(np.median(np.abs(array - median)))
        result = (mad, 1.4826 * mad)
    return result if return_mad else result[1]


def _fit_arrays(
    data: pd.DataFrame | Any,
    green: Any | None,
    ratio_qc_pass: Any | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if isinstance(data, pd.DataFrame):
        if not {"red", "green"}.issubset(data.columns):
            raise ValueError("Modal fit requires red and green columns")
        red_values = pd.to_numeric(data["red"], errors="coerce").to_numpy(dtype=float)
        green_values = pd.to_numeric(data["green"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(red_values) & (red_values > 0) & np.isfinite(green_values) & (green_values > 0)
        if "ratio_qc_pass" in data:
            valid &= _boolean_values(data["ratio_qc_pass"]).to_numpy()
    else:
        if green is None:
            raise TypeError("green values are required when the first argument is not a DataFrame")
        red_values = np.asarray(data, dtype=float).reshape(-1)
        green_values = np.asarray(green, dtype=float).reshape(-1)
        if red_values.shape != green_values.shape:
            raise ValueError("red and green values must have the same shape")
        valid = np.isfinite(red_values) & (red_values > 0) & np.isfinite(green_values) & (green_values > 0)
        if ratio_qc_pass is not None:
            ratio_values = _boolean_values(pd.Series(np.asarray(ratio_qc_pass).reshape(-1))).to_numpy()
            if ratio_values.shape != valid.shape:
                raise ValueError("ratio_qc_pass must have the same shape as red and green")
            valid &= ratio_values
    return red_values[valid], green_values[valid], int(valid.sum())


@dataclass(frozen=True)
class ModalFitResult:
    n_fit_rois: int
    red_mean: float
    red_std: float
    ols_slope: float
    ols_intercept: float
    modal_slope: float
    modal_intercept: float
    modal_bandwidth_scale: float
    modal_bandwidth_green_units: float
    robust_initial_scale_green_units: float
    robust_scale_fallback: bool
    modal_fit_success: bool
    modal_fit_status: int
    modal_fit_message: str
    n_iterations: int | None
    objective_value: float

    def predict(self, red: Any) -> np.ndarray:
        return self.modal_intercept + self.modal_slope * np.asarray(red, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def fit_linear_modal_backbone(
    data: pd.DataFrame | Any,
    green: Any | None = None,
    *,
    ratio_qc_pass: Any | None = None,
    min_fit_rois: int = 50,
    modal_bandwidth_scale: float = MODAL_BANDWIDTH_SCALE,
) -> ModalFitResult:
    """Fit the deterministic Gaussian-kernel modal line in standardized-x space."""

    if min_fit_rois < 2:
        raise ValueError("min_fit_rois must be at least 2")
    if not np.isfinite(modal_bandwidth_scale) or modal_bandwidth_scale <= 0:
        raise ValueError("modal_bandwidth_scale must be positive and finite")
    red, green_values, n_fit = _fit_arrays(data, green, ratio_qc_pass)
    if n_fit < min_fit_rois:
        raise ValueError(f"Modal fit requires at least {min_fit_rois} valid ROIs; found {n_fit}")

    red_mean = float(np.mean(red))
    red_std = float(np.std(red))
    if not np.isfinite(red_std) or red_std <= 0:
        raise ValueError("Modal fit requires finite, non-constant red values")
    z = (red - red_mean) / red_std
    design = np.column_stack((np.ones(n_fit), z))
    ols_intercept_z, ols_slope_z = np.linalg.lstsq(design, green_values, rcond=None)[0]
    ols_residuals = green_values - (ols_intercept_z + ols_slope_z * z)
    residual_mad, robust_scale = robust_sd_mad(ols_residuals, return_mad=True)
    scale_fallback = False
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        robust_scale = float(np.std(ols_residuals, ddof=1))
        scale_fallback = True
    if not np.isfinite(robust_scale) or robust_scale <= 0:
        raise ValueError("Modal fit residual scale is invalid or zero")
    bandwidth = float(modal_bandwidth_scale * robust_scale)

    def objective(parameters: np.ndarray) -> float:
        residuals = green_values - (parameters[0] + parameters[1] * z)
        return float(-np.mean(np.exp(-0.5 * (residuals / bandwidth) ** 2)))

    result = minimize(
        objective,
        np.array([ols_intercept_z, ols_slope_z], dtype=float),
        method="Nelder-Mead",
    )
    if not result.success:
        raise RuntimeError(
            "Modal optimization failed: "
            f"status={result.status}, message={result.message!s}, "
            f"objective={result.fun!r}"
        )
    modal_intercept_z, modal_slope_z = (float(value) for value in result.x)
    return ModalFitResult(
        n_fit_rois=n_fit,
        red_mean=red_mean,
        red_std=red_std,
        ols_slope=float(ols_slope_z / red_std),
        ols_intercept=float(ols_intercept_z - ols_slope_z * red_mean / red_std),
        modal_slope=float(modal_slope_z / red_std),
        modal_intercept=float(modal_intercept_z - modal_slope_z * red_mean / red_std),
        modal_bandwidth_scale=float(modal_bandwidth_scale),
        modal_bandwidth_green_units=bandwidth,
        robust_initial_scale_green_units=float(robust_scale),
        robust_scale_fallback=scale_fallback,
        modal_fit_success=bool(result.success),
        modal_fit_status=int(result.status),
        modal_fit_message=str(result.message),
        n_iterations=int(result.nit) if hasattr(result, "nit") else None,
        objective_value=float(-result.fun),
    )


def fit_session_modal_fits(
    population: pd.DataFrame,
    *,
    min_fit_rois: int = 50,
    modal_bandwidth_scale: float = MODAL_BANDWIDTH_SCALE,
) -> pd.DataFrame:
    """Fit one modal backbone per session and calculate native-population diagnostics."""

    if "session_id" not in population:
        raise ValueError("Session modal fits require a session_id column")
    rows: list[dict[str, Any]] = []
    for session_id, group in population.groupby(population["session_id"].astype(str), sort=False):
        fit = fit_linear_modal_backbone(
            group,
            min_fit_rois=min_fit_rois,
            modal_bandwidth_scale=modal_bandwidth_scale,
        )
        valid = _fit_arrays(group, None, None)
        red, green, _ = valid
        predicted = fit.predict(red)
        residual_valid = np.isfinite(predicted) & (predicted > 0)
        log_residuals = np.log2(green[residual_valid] / predicted[residual_valid])
        residual_mad, residual_sd = robust_sd_mad(log_residuals, return_mad=True)
        source = group.iloc[0]
        row = {
            "session_id": str(session_id),
            "session_index": int(source["session_index"]),
            "day": int(source["day"]) if "day" in source and pd.notna(source["day"]) else int(source["session_index"]),
            "acquisition_date": source.get("acquisition_date"),
            "elapsed_days": source.get("elapsed_days", np.nan),
            "n_native_rois": int(len(group)),
            "n_fit_rois": int(fit.n_fit_rois),
            "modal_slope": fit.modal_slope,
            "modal_intercept": fit.modal_intercept,
            "ols_slope": fit.ols_slope,
            "ols_intercept": fit.ols_intercept,
            "modal_bandwidth_scale": fit.modal_bandwidth_scale,
            "modal_bandwidth_green_units": fit.modal_bandwidth_green_units,
            "modal_fit_success": fit.modal_fit_success,
            "modal_fit_status": fit.modal_fit_status,
            "modal_fit_message": fit.modal_fit_message,
            "n_iterations": fit.n_iterations,
            "objective_value": fit.objective_value,
            "robust_initial_scale_green_units": fit.robust_initial_scale_green_units,
            "robust_scale_fallback": fit.robust_scale_fallback,
            "residual_median_log2": float(np.median(log_residuals)),
            "residual_mad_log2": float(residual_mad),
            "residual_robust_sd_log2": float(residual_sd),
        }
        if residual_sd > 0 and np.isfinite(residual_sd):
            row.update({
                "fraction_below_minus_1_session_rsd": float(np.mean(log_residuals < -residual_sd)),
                "fraction_below_minus_2_session_rsd": float(np.mean(log_residuals < -2 * residual_sd)),
                "fraction_above_plus_1_session_rsd": float(np.mean(log_residuals > residual_sd)),
                "fraction_above_plus_2_session_rsd": float(np.mean(log_residuals > 2 * residual_sd)),
            })
        else:
            row.update({key: np.nan for key in (
                "fraction_below_minus_1_session_rsd", "fraction_below_minus_2_session_rsd",
                "fraction_above_plus_1_session_rsd", "fraction_above_plus_2_session_rsd",
            )})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("session_index").reset_index(drop=True)


def color_state_bin(color_z: float) -> str | None:
    if not np.isfinite(color_z):
        return None
    if color_z < -2:
        return "strong_low"
    if color_z < -1:
        return "low_transition"
    if color_z <= 1:
        return "middle"
    if color_z <= 2:
        return "high_transition"
    return "strong_high"


def color_core_state(color_z: float) -> str | None:
    if not np.isfinite(color_z):
        return None
    if color_z < -2:
        return "low"
    if -1 <= color_z <= 1:
        return "middle"
    if color_z > 2:
        return "high"
    return None


def score_color_state_table(
    observations: pd.DataFrame,
    session_fits: pd.DataFrame,
    reference_robust_sd_log2: float,
) -> pd.DataFrame:
    """Append modal residual, fixed Dead-Z, and state labels to observations."""

    if not np.isfinite(reference_robust_sd_log2) or reference_robust_sd_log2 <= 0:
        raise ValueError("reference_robust_sd_log2 must be positive and finite")
    required = {"session_id", "green", "red", "ratio_qc_pass"}
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"Color-state observations are missing: {', '.join(sorted(missing))}")
    fit_columns = [
        "session_id", "modal_slope", "modal_intercept", "modal_bandwidth_green_units",
    ]
    result = observations.copy()
    fit_table = session_fits[fit_columns].copy()
    fit_table["session_id"] = fit_table["session_id"].astype(str)
    result["session_id"] = result["session_id"].astype(str)
    result = result.merge(fit_table, on="session_id", how="left", validate="many_to_one")
    green = pd.to_numeric(result["green"], errors="coerce")
    red = pd.to_numeric(result["red"], errors="coerce")
    predicted = result["modal_intercept"] + result["modal_slope"] * red
    result["predicted_green_modal"] = predicted
    result["green_over_expected"] = np.nan
    result["log2_green_over_expected"] = np.nan
    result["dead_reference_robust_sd_log2"] = float(reference_robust_sd_log2)
    result["color_z"] = np.nan
    result["color_state_bin"] = None
    result["color_core_state"] = None
    result["color_state_qc_pass"] = False
    reasons = pd.Series("", index=result.index, dtype=object)
    ratio_valid = _boolean_values(result["ratio_qc_pass"])
    if (~ratio_valid).any():
        reasons.loc[~ratio_valid] = "ratio_qc_fail"
    valid_signal = np.isfinite(green) & (green > 0) & np.isfinite(red) & (red > 0)
    reasons.loc[ratio_valid & ~valid_signal] = "nonpositive_or_nonfinite_signal"
    valid_prediction = np.isfinite(predicted) & (predicted > 0)
    reasons.loc[ratio_valid & valid_signal & ~valid_prediction] = "nonpositive_or_nonfinite_predicted_green"
    valid = ratio_valid & valid_signal & valid_prediction
    result.loc[valid, "green_over_expected"] = green[valid] / predicted[valid]
    result.loc[valid, "log2_green_over_expected"] = np.log2(result.loc[valid, "green_over_expected"])
    result.loc[valid, "color_z"] = result.loc[valid, "log2_green_over_expected"] / reference_robust_sd_log2
    result.loc[valid, "color_state_qc_pass"] = True
    result.loc[valid, "color_state_bin"] = result.loc[valid, "color_z"].map(color_state_bin)
    result.loc[valid, "color_core_state"] = result.loc[valid, "color_z"].map(color_core_state)
    result["color_state_qc_reason"] = reasons.replace("", "valid")
    return result
