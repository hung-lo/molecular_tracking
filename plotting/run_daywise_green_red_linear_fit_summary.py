"""Summarize day-wise linear fits between corrected red and green ROI values.

This script reads the current mean-merge SAM size+shape-filtered ROI metrics
table, fits a separate linear model of corrected green intensity versus
corrected red intensity on each imaging day, and saves the results to a dated
analysis-run directory.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _import_dir in (
    _REPO_ROOT / "core",
    _REPO_ROOT / "plotting",
    _REPO_ROOT / "matching",
):
    _import_dir_str = str(_import_dir)
    if _import_dir_str not in sys.path:
        sys.path.append(_import_dir_str)


from analysis_paths import get_shape_qc_analysis_dir, resolve_dataset_dir
from roi_log_ratio_analysis import summarize_daily_green_red_linear_fits


def format_duration_seconds(duration_seconds: float) -> str:
    """Format elapsed wall-clock time as ``HH:MM:SS``."""

    total_seconds = max(0, int(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_message(run_start_seconds: float, message: str) -> None:
    """Print one elapsed-time progress message."""

    elapsed_seconds = time.perf_counter() - run_start_seconds
    print(f"[{format_duration_seconds(elapsed_seconds)}] {message}", flush=True)



def make_day_date_labels(
    day_values: np.ndarray,
    start_date: str = "20260511",
    acquisition_dates: pd.Series | np.ndarray | list[str] | None = None,
) -> list[str]:
    """Convert integer day offsets or acquisition dates into date labels.

    Parameters
    ----------
    day_values : numpy.ndarray
        One-dimensional array of integer day offsets relative to ``start_date``.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format.
    acquisition_dates : sequence of str, optional
        When provided, each entry is parsed as the actual acquisition date for
        the matching day value and used directly in the output labels.

    Returns
    -------
    list[str]
        Date labels in ``YYYYMMDD`` format, one for each input day value.
    """

    if acquisition_dates is not None:
        acquisition_list = list(acquisition_dates)
        if len(acquisition_list) == len(day_values):
            labels: list[str] = []
            for raw_value in acquisition_list:
                parsed_value = pd.to_datetime(raw_value, errors="coerce")
                if pd.notna(parsed_value):
                    labels.append(parsed_value.strftime("%Y%m%d"))
                else:
                    labels.append(str(raw_value))
            return labels

    reference_date = pd.to_datetime(start_date, format="%Y%m%d")
    labels: list[str] = []
    for day_value in day_values:
        date_value = reference_date + pd.to_timedelta(int(day_value), unit="D")
        labels.append(date_value.strftime("%Y%m%d"))
    return labels


def _filter_green_artifacts(
    day_table: pd.DataFrame,
    green_artifact_threshold: float,
) -> tuple[pd.DataFrame, int]:
    """Drop rows that look like obvious green-channel artifacts."""

    if day_table.empty:
        return day_table.copy(), 0
    cleaned = day_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["red", "green"]).copy()
    if cleaned.empty:
        return cleaned, 0
    artifact_mask = cleaned["green"].gt(float(green_artifact_threshold))
    excluded_count = int(artifact_mask.sum())
    if excluded_count == 0:
        return cleaned.reset_index(drop=True), 0
    return cleaned.loc[~artifact_mask].reset_index(drop=True), excluded_count


def _resolve_day_date_labels(
    roi_metrics: pd.DataFrame,
    day_values: np.ndarray,
    start_date: str = "20260511",
) -> list[str]:
    """Prefer actual acquisition dates when they are available."""

    if "acquisition_date" in roi_metrics.columns and "day" in roi_metrics.columns:
        date_lookup = (
            roi_metrics.loc[:, ["day", "acquisition_date"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["day", "acquisition_date"])
            .assign(
                day=lambda frame: pd.to_numeric(frame["day"], errors="coerce"),
                acquisition_date=lambda frame: pd.to_datetime(
                    frame["acquisition_date"], errors="coerce"
                ),
            )
            .dropna(subset=["day", "acquisition_date"])
            .drop_duplicates(subset=["day"])
            .set_index("day")["acquisition_date"]
            .to_dict()
        )
        acquisition_dates = [date_lookup.get(int(day_value)) for day_value in day_values]
        if all(date is not None and pd.notna(date) for date in acquisition_dates):
            return make_day_date_labels(
                day_values,
                start_date=start_date,
                acquisition_dates=acquisition_dates,
            )
    return make_day_date_labels(day_values, start_date=start_date)


def compute_regression_ci_band(
    x_values: np.ndarray,
    y_values: np.ndarray,
    alpha: float = 0.05,
    n_grid_points: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a fitted line and 95% confidence band for one linear regression.

    Parameters
    ----------
    x_values : numpy.ndarray
        One-dimensional predictor values in corrected red fluorescence units.
    y_values : numpy.ndarray
        One-dimensional response values in corrected green fluorescence units.
    alpha : float, default=0.05
        Two-sided type-I error rate for the confidence band.
    n_grid_points : int, default=200
        Number of x positions used to draw the fitted line and confidence band.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        ``(x_grid, y_hat, y_low, y_high)`` arrays for the fitted line and the
        confidence interval of the mean prediction, all in corrected
        fluorescence units.
    """

    from scipy import stats

    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)
    fit = stats.linregress(x_values, y_values)

    x_grid = np.linspace(float(np.min(x_values)), float(np.max(x_values)), n_grid_points)
    y_hat = fit.intercept + fit.slope * x_grid

    degrees_of_freedom = len(x_values) - 2
    if degrees_of_freedom <= 0:
        return x_grid, y_hat, np.full_like(y_hat, np.nan), np.full_like(y_hat, np.nan)

    residuals = y_values - (fit.intercept + fit.slope * x_values)
    residual_standard_error = np.sqrt(np.sum(residuals**2) / degrees_of_freedom)
    x_mean = float(np.mean(x_values))
    sum_squared_x = float(np.sum((x_values - x_mean) ** 2))
    if sum_squared_x == 0:
        return x_grid, y_hat, np.full_like(y_hat, np.nan), np.full_like(y_hat, np.nan)

    t_critical = float(stats.t.ppf(1.0 - alpha / 2.0, df=degrees_of_freedom))
    mean_prediction_stderr = residual_standard_error * np.sqrt(
        1.0 / len(x_values) + ((x_grid - x_mean) ** 2) / sum_squared_x
    )
    y_low = y_hat - t_critical * mean_prediction_stderr
    y_high = y_hat + t_critical * mean_prediction_stderr
    return x_grid, y_hat, y_low, y_high


def plot_daywise_scatter_summary(
    roi_metrics: pd.DataFrame,
    fit_summary: pd.DataFrame,
    output_path: Path,
    start_date: str = "20260511",
    green_artifact_threshold: float = 1500.0,
) -> None:
    """Plot per-day red-vs-green scatters with fitted lines and CI bands.

    Parameters
    ----------
    roi_metrics : pandas.DataFrame
        ROI/day table with columns ``day``, ``red``, and ``green`` in corrected
        fluorescence units.
    fit_summary : pandas.DataFrame
        Day-wise fit summary table returned by
        :func:`summarize_daily_green_red_linear_fits`.
    output_path : pathlib.Path
        PNG path for the saved scatter summary figure.
    start_date : str, default="20260511"
        Reference date used to convert day offsets into date labels.
    green_artifact_threshold : float, default=1500.0
        Rows with corrected green values above this threshold are treated as
        artifacts and excluded from the fitted lines.
    """

    if roi_metrics.empty:
        raise ValueError("No ROI metrics were available for scatter plotting.")

    filtered_metrics, excluded_total = _filter_green_artifacts(
        roi_metrics,
        green_artifact_threshold=green_artifact_threshold,
    )
    if filtered_metrics.empty:
        raise ValueError("No ROI metrics remained after filtering green artifacts.")
    plot_fit_summary = summarize_daily_green_red_linear_fits(filtered_metrics)
    day_values = plot_fit_summary["day"].to_numpy(dtype=int)
    date_labels = _resolve_day_date_labels(
        roi_metrics,
        day_values,
        start_date=start_date,
    )

    panels: list[tuple[int, pd.DataFrame, int]] = []
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    for day_value in day_values:
        day_table = (
            roi_metrics.loc[
                roi_metrics["day"].eq(day_value),
                [
                    column
                    for column in ["red", "green", "track_match_source"]
                    if column in roi_metrics.columns
                ],
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["red", "green"])
            .reset_index(drop=True)
        )
        filtered_day_table, excluded_count = _filter_green_artifacts(
            day_table,
            green_artifact_threshold=green_artifact_threshold,
        )
        if filtered_day_table.empty and not day_table.empty:
            filtered_day_table = day_table.copy()
            excluded_count = 0
        if not filtered_day_table.empty:
            x_min = min(x_min, float(filtered_day_table["red"].min()))
            x_max = max(x_max, float(filtered_day_table["red"].max()))
            y_min = min(y_min, float(filtered_day_table["green"].min()))
            y_max = max(y_max, float(filtered_day_table["green"].max()))
        panels.append((int(day_value), filtered_day_table, excluded_count))

    figure, axes = plt.subplots(
        1,
        len(day_values),
        figsize=(22, 4.8),
        sharex=True,
        sharey=True,
        facecolor="white",
    )
    if len(day_values) == 1:
        axes = [axes]

    for axis, (day_value, day_table, excluded_count), date_label in zip(
        axes, panels, date_labels, strict=True
    ):
        x_values = day_table["red"].to_numpy(dtype=float)
        y_values = day_table["green"].to_numpy(dtype=float)
        if len(day_table) >= 2:
            x_grid, y_hat, y_low, y_high = compute_regression_ci_band(x_values, y_values)
        else:
            x_grid = np.asarray([], dtype=float)
            y_hat = np.asarray([], dtype=float)
            y_low = np.asarray([], dtype=float)
            y_high = np.asarray([], dtype=float)

        fit_row = plot_fit_summary.loc[plot_fit_summary["day"].eq(day_value)].iloc[0]
        axis.scatter(
            x_values,
            y_values,
            s=9,
            alpha=0.15,
            color="#1f3b4d",
            edgecolors="none",
            rasterized=True,
        )
        if len(x_grid) > 0 and np.all(np.isfinite(y_low)) and np.all(np.isfinite(y_high)):
            axis.fill_between(x_grid, y_low, y_high, color="#8ecae6", alpha=0.35, linewidth=0)
            axis.plot(x_grid, y_hat, color="#d62828", linewidth=2.0)

        axis.set_title(f"Day {day_value}\n{date_label}", fontsize=11)
        axis.set_xlabel("Corrected red intensity", fontsize=10)
        if day_value == int(day_values[0]):
            axis.set_ylabel("Corrected green intensity", fontsize=10)
        axis.tick_params(labelsize=9)
        axis.text(
            0.04,
            0.96,
            (
                f"slope = {fit_row['slope']:.3f}\n"
                f"intercept = {fit_row['intercept']:.1f}\n"
                f"R² = {fit_row['r_squared']:.3f}\n"
                f"n = {int(fit_row['n_rois'])}"
                + (
                    f"\nexcluded green>{green_artifact_threshold:g}: {excluded_count}"
                    if excluded_count
                    else ""
                )
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
        )

    if np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max):
        x_pad = max((x_max - x_min) * 0.05, 1.0)
        y_pad = max((y_max - y_min) * 0.05, 1.0)
        for axis in axes:
            axis.set_xlim(x_min - x_pad, x_max + x_pad)
            axis.set_ylim(y_min - y_pad, y_max + y_pad)

    figure.suptitle(
        "Day-wise corrected red vs green ROI values with separate linear fits",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.02,
        (
            "Each panel shows one imaging day from the current mean-merge SAM "
            "size+shape-filtered ROI set. The red line is the fitted linear "
            "relationship between corrected red and green values, and the blue "
            "band is the 95% confidence interval of the mean fit. "
            f"Green values above {green_artifact_threshold:g} were excluded from the fit; "
            f"total excluded rows = {excluded_total}."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.07, 1.0, 0.92))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_fit_parameter_summary(
    fit_summary: pd.DataFrame,
    output_path: Path,
    roi_metrics: pd.DataFrame | None = None,
    start_date: str = "20260511",
    green_artifact_threshold: float = 1500.0,
) -> None:
    """Plot day-wise slope, intercept, fit quality, and ROI count summaries.

    Parameters
    ----------
    fit_summary : pandas.DataFrame
        Day-wise fit summary table returned by
        :func:`summarize_daily_green_red_linear_fits`.
    output_path : pathlib.Path
        PNG path for the saved summary figure.
    roi_metrics : pandas.DataFrame, optional
        Raw ROI/day table. When provided, the plotted fit summary is recomputed
        after excluding rows whose corrected green values exceed
        ``green_artifact_threshold``.
    start_date : str, default="20260511"
        Reference date used to convert day offsets into date labels.
    green_artifact_threshold : float, default=1500.0
        Rows with corrected green values above this threshold are treated as
        artifacts and excluded from the plotted fit summary.
    """

    plot_fit_summary = fit_summary.copy()
    excluded_total = 0
    if roi_metrics is not None and not roi_metrics.empty:
        filtered_metrics, excluded_total = _filter_green_artifacts(
            roi_metrics,
            green_artifact_threshold=green_artifact_threshold,
        )
        if not filtered_metrics.empty:
            plot_fit_summary = summarize_daily_green_red_linear_fits(filtered_metrics)
            roi_metrics = filtered_metrics

    day_values = plot_fit_summary["day"].to_numpy(dtype=int)
    date_labels = _resolve_day_date_labels(
        roi_metrics if roi_metrics is not None else plot_fit_summary,
        day_values,
        start_date=start_date,
    )
    x_positions = np.arange(len(day_values))

    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor="white")
    slope_axis, intercept_axis, r2_axis, n_axis = axes.ravel()

    slope_axis.errorbar(
        x_positions,
        plot_fit_summary["slope"].to_numpy(dtype=float),
        yerr=np.vstack(
            [
                plot_fit_summary["slope"].to_numpy(dtype=float)
                - plot_fit_summary["slope_ci_low"].to_numpy(dtype=float),
                plot_fit_summary["slope_ci_high"].to_numpy(dtype=float)
                - plot_fit_summary["slope"].to_numpy(dtype=float),
            ]
        ),
        fmt="o-",
        color="#1d3557",
        ecolor="#457b9d",
        capsize=4,
    )
    slope_axis.set_title("Slope by day", fontsize=11)
    slope_axis.set_ylabel("Green / red slope", fontsize=10)

    intercept_axis.errorbar(
        x_positions,
        plot_fit_summary["intercept"].to_numpy(dtype=float),
        yerr=np.vstack(
            [
                plot_fit_summary["intercept"].to_numpy(dtype=float)
                - plot_fit_summary["intercept_ci_low"].to_numpy(dtype=float),
                plot_fit_summary["intercept_ci_high"].to_numpy(dtype=float)
                - plot_fit_summary["intercept"].to_numpy(dtype=float),
            ]
        ),
        fmt="o-",
        color="#6a040f",
        ecolor="#9d0208",
        capsize=4,
    )
    intercept_axis.set_title("Intercept by day", fontsize=11)
    intercept_axis.set_ylabel("Corrected green offset", fontsize=10)

    r2_axis.plot(
        x_positions,
        plot_fit_summary["r_squared"].to_numpy(dtype=float),
        "o-",
        color="#2a9d8f",
    )
    r2_axis.set_title("Fit quality by day", fontsize=11)
    r2_axis.set_ylabel("R²", fontsize=10)
    r2_axis.set_ylim(0.0, 1.02)

    n_axis.bar(
        x_positions,
        plot_fit_summary["n_rois"].to_numpy(dtype=int),
        color="#adb5bd",
        edgecolor="#6c757d",
    )
    n_axis.set_title("ROI count by day", fontsize=11)
    n_axis.set_ylabel("Number of filtered ROIs", fontsize=10)

    for axis in axes.ravel():
        axis.set_xticks(x_positions, date_labels, rotation=30, ha="right")
        axis.tick_params(labelsize=9)

    figure.suptitle("Day-wise linear-fit parameter summary", fontsize=14)
    figure.text(
        0.5,
        0.02,
        (
            "This summary tracks how the fitted corrected green-vs-red "
            "relationship changes across imaging days for the current mean-merge "
            "SAM size+shape-filtered ROI set. "
            f"Green values above {green_artifact_threshold:g} were excluded from the plotted fit; "
            f"total excluded rows = {excluded_total}."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.07, 1.0, 0.93))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_run_summary(
    output_dir: Path,
    input_metrics_path: Path,
    fit_summary_path: Path,
    scatter_plot_path: Path,
    parameter_plot_path: Path,
) -> None:
    """Write a short human-readable summary of the day-wise fit analysis.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory for the current analysis run.
    input_metrics_path : pathlib.Path
        CSV file that provided the filtered ROI/day values.
    fit_summary_path : pathlib.Path
        CSV file containing the day-wise fit coefficients.
    scatter_plot_path : pathlib.Path
        PNG file with the day-wise scatter summary.
    parameter_plot_path : pathlib.Path
        PNG file with the fit-parameter summary.
    """

    summary_lines = [
        "# Day-wise Green-vs-Red Linear Fit Summary",
        "",
        "Goal: summarize the linear relationship between corrected red and green",
        "ROI intensities on each imaging day for the current mean-merge SAM",
        "size+shape-filtered ROI set.",
        "",
        "Inputs:",
        f"- Filtered ROI/day metrics: `{input_metrics_path}`",
        "",
        "Outputs:",
        f"- Day-wise fit table: `{fit_summary_path}`",
        f"- Day-wise scatter summary: `{scatter_plot_path}`",
        f"- Fit-parameter summary: `{parameter_plot_path}`",
        "",
        "Interpretation note:",
        "- These fits are most useful as QC summaries.",
        "- Dividing green values by the daily slope or z-scoring each day would",
        "  change the biological signal in a harder-to-interpret way, so no",
        "  normalization is applied here.",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")


def write_run_log(
    output_dir: Path,
    input_metrics_path: Path,
    fit_summary: pd.DataFrame,
    total_duration_seconds: float,
) -> None:
    """Write a plain-text run log for reproducibility.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory for the current analysis run.
    input_metrics_path : pathlib.Path
        CSV file that provided the filtered ROI/day values.
    fit_summary : pandas.DataFrame
        Day-wise fit summary table returned by
        :func:`summarize_daily_green_red_linear_fits`.
    total_duration_seconds : float
        Total wall-clock duration for the current run in seconds.
    """

    log_lines = [
        f"run_timestamp={datetime.now().isoformat()}",
        f"input_metrics_path={input_metrics_path}",
        f"n_days={len(fit_summary)}",
        f"days={','.join(str(day) for day in fit_summary['day'].tolist())}",
        f"roi_counts={','.join(str(int(count)) for count in fit_summary['n_rois'].tolist())}",
        f"total_duration_seconds={float(total_duration_seconds):.3f}",
        f"total_duration_hms={format_duration_seconds(total_duration_seconds)}",
    ]
    (output_dir / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the day-wise fit summary script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with the dataset alias or path under ``dataset``.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="1050",
        help="Dataset alias (e.g. 1050 or 920) or an explicit dataset directory path.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the day-wise corrected red-vs-green fit summary analysis."""

    run_start_seconds = time.perf_counter()
    args = parse_args()
    log_message(run_start_seconds, f"Starting day-wise green-vs-red linear fit summary | dataset={args.dataset}")
    base_dir = resolve_dataset_dir(args.dataset)
    shape_qc_dir = get_shape_qc_analysis_dir(args.dataset)
    input_metrics_path = (
        shape_qc_dir
        / "roi_log_ratio_metrics_size_and_shape_filtered.csv"
    )
    output_root = shape_qc_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"daywise_green_red_linear_fit_summary_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log_message(run_start_seconds, f"Output directory: {output_dir}")

    log_message(run_start_seconds, f"Loading filtered ROI metrics from {input_metrics_path}")
    roi_metrics = pd.read_csv(input_metrics_path)
    log_message(run_start_seconds, f"Loaded {len(roi_metrics)} ROI/day rows; fitting day-wise green-vs-red models")
    fit_summary = summarize_daily_green_red_linear_fits(roi_metrics)

    fit_summary_path = output_dir / "daywise_green_red_linear_fit_summary.csv"
    fit_summary.to_csv(fit_summary_path, index=False)

    scatter_plot_path = output_dir / "daywise_green_red_linear_fit_scatters.png"
    parameter_plot_path = output_dir / "daywise_green_red_linear_fit_parameters.png"
    log_message(run_start_seconds, "Rendering day-wise scatter and fit-parameter summary plots")
    plot_daywise_scatter_summary(
        roi_metrics=roi_metrics,
        fit_summary=fit_summary,
        output_path=scatter_plot_path,
    )
    plot_fit_parameter_summary(
        fit_summary=fit_summary,
        output_path=parameter_plot_path,
        roi_metrics=roi_metrics,
    )

    shutil.copy2(__file__, output_dir / Path(__file__).name)
    write_run_summary(
        output_dir=output_dir,
        input_metrics_path=input_metrics_path,
        fit_summary_path=fit_summary_path,
        scatter_plot_path=scatter_plot_path,
        parameter_plot_path=parameter_plot_path,
    )
    total_duration_seconds = time.perf_counter() - run_start_seconds
    write_run_log(
        output_dir=output_dir,
        input_metrics_path=input_metrics_path,
        fit_summary=fit_summary,
        total_duration_seconds=total_duration_seconds,
    )

    print(f"[{format_duration_seconds(total_duration_seconds)}] Completed day-wise green-vs-red linear fit summary", flush=True)
    print(f"output_dir={output_dir}")
    print(f"fit_summary_path={fit_summary_path}")
    print(f"scatter_plot_path={scatter_plot_path}")
    print(f"parameter_plot_path={parameter_plot_path}")
    print(f"total_duration={format_duration_seconds(total_duration_seconds)}")


if __name__ == "__main__":
    main()
