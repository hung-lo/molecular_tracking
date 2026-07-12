"""Compute day-wise green-vs-red fit residuals and plot ranked ROI subsets.

This script builds a residual-based companion analysis for the current
mean-merge SAM size+shape-filtered ROI set. The main residual metric is the
signed vertical deviation from each day's fitted corrected green-vs-red line.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil
import argparse
import time

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from analysis_paths import get_shape_qc_analysis_dir, resolve_dataset_dir


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

from roi_log_ratio_analysis import (
    compute_green_red_fit_residuals,
    select_ranked_roi_days,
    summarize_residual_sign_changes,
    summarize_daily_green_red_linear_fits,
)


def make_day_labels(
    day_values: np.ndarray,
    start_date: str = "20260511",
) -> list[str]:
    """Convert day indices into date labels.

    Parameters
    ----------
    day_values : numpy.ndarray
        One-dimensional integer array of day offsets relative to ``start_date``.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format.

    Returns
    -------
    list[str]
        Date labels in ``YYYYMMDD`` format, one per input day value.
    """

    reference_date = pd.to_datetime(start_date, format="%Y%m%d")
    labels: list[str] = []
    for day_value in day_values:
        labels.append(
            (reference_date + pd.to_timedelta(int(day_value), unit="D")).strftime("%Y%m%d")
        )
    return labels


def plot_directional_roi_trajectories_scatter(
    ranked_roi_table: pd.DataFrame,
    fit_summary: pd.DataFrame,
    output_path: Path,
    title_prefix: str,
    direction_label: str,
    start_date: str = "20260511",
) -> None:
    """Plot ranked ROIs in corrected red-green space against the average fit.

    Parameters
    ----------
    ranked_roi_table : pandas.DataFrame
        ROI/day table for a ranked ROI subset. Required columns are
        ``roi_id``, ``selection_rank``, ``day``, ``red``, and ``green``. The
        values must be corrected fluorescence intensities.
    fit_summary : pandas.DataFrame
        Day-wise fit summary with columns ``slope`` and ``intercept``.
    output_path : pathlib.Path
        PNG path for the saved plot.
    title_prefix : str
        Short label such as ``"Top 10"`` or ``"Top 30"`` used in the title.
    direction_label : str
        Human-readable direction label such as ``"decreasing"`` or
        ``"increasing"`` used in the plot title.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format used for day labels.
    """

    average_slope = float(fit_summary["slope"].mean())
    average_intercept = float(fit_summary["intercept"].mean())

    day_values = np.sort(ranked_roi_table["day"].unique())
    day_labels = make_day_labels(day_values, start_date=start_date)
    color_values = plt.cm.viridis(np.linspace(0.0, 1.0, len(day_values)))

    figure, axis = plt.subplots(figsize=(10.0, 8.0), facecolor="white")

    x_min = float(ranked_roi_table["red"].min())
    x_max = float(ranked_roi_table["red"].max())
    x_grid = np.linspace(x_min, x_max, 300)
    axis.plot(
        x_grid,
        average_intercept + average_slope * x_grid,
        color="#d62828",
        linewidth=2.5,
        label=(
            f"Average fit: green = {average_intercept:.1f} + "
            f"{average_slope:.3f} * red"
        ),
        zorder=1,
    )

    for roi_id, roi_table in ranked_roi_table.groupby("roi_id", sort=True):
        roi_table = roi_table.sort_values("day").reset_index(drop=True)
        red_values = roi_table["red"].to_numpy(dtype=float)
        green_values = roi_table["green"].to_numpy(dtype=float)
        rank_value = int(roi_table["selection_rank"].iloc[0])

        axis.plot(
            red_values,
            green_values,
            color="0.75",
            linewidth=1.0,
            alpha=0.8,
            zorder=2,
        )
        for point_index, (red_value, green_value) in enumerate(
            zip(red_values, green_values, strict=True)
        ):
            axis.plot(
                red_value,
                green_value,
                color=color_values[point_index],
                marker="o",
                linestyle="None",
                markersize=6.0,
                markeredgecolor="black",
                markeredgewidth=0.3,
                zorder=3,
            )

        axis.text(
            red_values[-1],
            green_values[-1],
            f" {rank_value}:{roi_id}",
            fontsize=8,
            ha="left",
            va="center",
            color="0.2",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_values[index],
            markeredgecolor="black",
            markeredgewidth=0.3,
            markersize=6.0,
            label=f"Day {int(day_value)} ({day_label})",
        )
        for index, (day_value, day_label) in enumerate(zip(day_values, day_labels, strict=True))
    ]
    legend_handles.append(
        Line2D([0], [0], color="#d62828", linewidth=2.5, label="Average fitted line")
    )

    axis.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=True)
    axis.set_xlabel("Corrected red intensity", fontsize=11)
    axis.set_ylabel("Corrected green intensity", fontsize=11)
    axis.set_title(
        f"{title_prefix} {direction_label} ROIs in corrected red-green space\n"
        "Colored markers show day progression; labels are rank:ROI",
        fontsize=13,
    )
    axis.tick_params(labelsize=10)

    figure.text(
        0.5,
        0.03,
        (
            "Each gray polyline tracks one ROI across days in corrected red-green "
            "space. Colored points show day order from early to late, and the red "
            "line is the average of the day-wise population fits."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.04, 0.10, 1.0, 0.95))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_directional_roi_residuals_vs_day(
    ranked_roi_table: pd.DataFrame,
    output_path: Path,
    title_prefix: str,
    direction_label: str,
    start_date: str = "20260511",
) -> None:
    """Plot signed green-fit residual trajectories for a ranked ROI subset.

    Parameters
    ----------
    ranked_roi_table : pandas.DataFrame
        ROI/day table with columns ``roi_id``, ``selection_rank``, ``day``, and
        ``green_fit_residual``. Residual values are in corrected green
        fluorescence units.
    output_path : pathlib.Path
        PNG path for the saved figure.
    title_prefix : str
        Short label such as ``"Top 10"`` or ``"Top 30"`` used in the title.
    direction_label : str
        Human-readable direction label such as ``"decreasing"`` or
        ``"increasing"`` used in the plot title.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format used for day labels.
    """

    day_values = np.sort(ranked_roi_table["day"].unique())
    day_labels = make_day_labels(day_values, start_date=start_date)
    color_values = plt.cm.viridis(np.linspace(0.0, 1.0, len(day_values)))

    figure, axis = plt.subplots(figsize=(10.0, 7.5), facecolor="white")
    axis.axhline(0.0, color="#d62828", linewidth=1.8, linestyle="--", alpha=0.9, zorder=1)

    for roi_id, roi_table in ranked_roi_table.groupby("roi_id", sort=True):
        roi_table = roi_table.sort_values("day").reset_index(drop=True)
        x_values = roi_table["day"].to_numpy(dtype=int)
        residual_values = roi_table["green_fit_residual"].to_numpy(dtype=float)
        rank_value = int(roi_table["selection_rank"].iloc[0])

        axis.plot(
            x_values,
            residual_values,
            color="0.78",
            linewidth=1.0,
            alpha=0.8,
            zorder=2,
        )
        for point_index, (day_value, residual_value) in enumerate(
            zip(x_values, residual_values, strict=True)
        ):
            axis.plot(
                day_value,
                residual_value,
                color=color_values[point_index],
                marker="o",
                linestyle="None",
                markersize=5.5,
                markeredgecolor="black",
                markeredgewidth=0.25,
                zorder=3,
            )

        axis.text(
            x_values[-1] + 0.03,
            residual_values[-1],
            f"{rank_value}:{roi_id}",
            fontsize=7,
            ha="left",
            va="center",
            color="0.2",
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color_values[index],
            markeredgecolor="black",
            markeredgewidth=0.25,
            markersize=5.5,
            label=f"Day {int(day_value)} ({day_label})",
        )
        for index, (day_value, day_label) in enumerate(zip(day_values, day_labels, strict=True))
    ]
    legend_handles.append(
        Line2D([0], [0], color="#d62828", linewidth=1.8, linestyle="--", label="Residual = 0")
    )
    axis.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True, ncol=1)
    axis.set_xticks(day_values, [f"Day {int(day)}\n{label}" for day, label in zip(day_values, day_labels, strict=True)])
    axis.set_xlabel("Imaging day", fontsize=11)
    axis.set_ylabel("Signed green-fit residual", fontsize=11)
    axis.set_title(
        f"{title_prefix} {direction_label} ROIs: signed residual trajectories\n"
        "Crossing zero means crossing the day-specific fitted green-red relationship",
        fontsize=13,
    )
    axis.tick_params(labelsize=10)

    figure.text(
        0.5,
        0.03,
        (
            "Residual = observed corrected green minus the green value predicted "
            "from that day's fitted red-green line. Positive values are greener "
            "than expected; negative values are less green than expected."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.04, 0.10, 1.0, 0.95))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_summary_markdown(
    output_dir: Path,
    residual_table_path: Path,
    top30_table_path: Path,
    scatter_plot_path: Path,
    residual_plot_path: Path,
    direction_label: str,
) -> None:
    """Write a short human-readable summary of the residual analysis run.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory for the current analysis run.
    residual_table_path : pathlib.Path
        CSV file containing the ROI/day residual table.
    top30_table_path : pathlib.Path
        CSV file containing the selected top-30 decreasing ROI residual table.
    scatter_plot_path : pathlib.Path
        PNG file containing the trajectory scatter plot.
    residual_plot_path : pathlib.Path
        PNG file containing the residual-vs-day plot.
    direction_label : str
        Human-readable direction label such as ``"decreasing"`` or
        ``"increasing"`` used in the summary text.
    """

    summary_lines = [
        "# Day-wise Green-vs-Red Fit Residuals",
        "",
        "Goal: measure how far each ROI sits above or below the day-specific",
        f"population green-vs-red fit, then visualize the top ranked {direction_label}",
        "ROIs in corrected red-green space.",
        "",
        "Key metric:",
        "- `green_fit_residual = observed green - predicted green from that day's fit`",
        "",
        "Outputs:",
        f"- Full ROI/day residual table: `{residual_table_path}`",
        f"- Top 30 {direction_label} residual subset: `{top30_table_path}`",
        f"- Top 30 {direction_label} red-green scatter plot: `{scatter_plot_path}`",
        f"- Top 30 {direction_label} residual-vs-day plot: `{residual_plot_path}`",
        "",
        "Interpretation note:",
        "- A positive residual means the ROI is greener than expected for its red",
        "  value on that day.",
        "- A negative residual means it is less green than expected.",
        "- Crossing zero across days is a useful sign that the ROI moved across",
        "  the population relationship rather than simply sliding along it.",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")


def write_run_log(
    output_dir: Path,
    average_slope: float,
    average_intercept: float,
    top30_roi_ids: list[int],
    direction_label: str,
    total_duration_seconds: float,
) -> None:
    """Write a short run log for reproducibility.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory for the current analysis run.
    average_slope : float
        Mean slope across the day-wise linear fits.
    average_intercept : float
        Mean intercept across the day-wise linear fits.
    top30_roi_ids : list[int]
        ROI IDs included in the plotted top-30 ranked set.
    direction_label : str
        Human-readable direction label such as ``"decreasing"`` or
        ``"increasing"`` used in the log metadata key.
    total_duration_seconds : float
        Total wall-clock duration for the current run in seconds.
    """

    log_lines = [
        f"run_timestamp={datetime.now().isoformat()}",
        f"average_slope={average_slope}",
        f"average_intercept={average_intercept}",
        f"top30_{direction_label}_roi_ids={','.join(str(roi_id) for roi_id in top30_roi_ids)}",
        f"total_duration_seconds={float(total_duration_seconds):.3f}",
        f"total_duration_hms={format_duration_seconds(total_duration_seconds)}",
    ]
    (output_dir / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


def run_directional_residual_analysis(
    dataset: str | Path | None,
    direction_label: str,
    output_dir_prefix: str,
) -> Path:
    """Run the directional residual analysis for one ranked ROI set.

    Parameters
    ----------
    dataset : str, pathlib.Path, or None
        Dataset alias or explicit dataset directory. Supported aliases are
        defined in :mod:`analysis_paths`.
    direction_label : str
        Human-readable direction label such as ``"decreasing"`` or
        ``"increasing"`` used in titles and output filenames.
    output_dir_prefix : str
        Prefix used when creating the dated output directory.

    Returns
    -------
    pathlib.Path
        Path to the created dated output directory.
    """
    run_start_seconds = time.perf_counter()
    log_message(run_start_seconds, f"Starting {direction_label} green-fit residual analysis | dataset={dataset}")
    base_dir = resolve_dataset_dir(dataset)
    shape_qc_dir = get_shape_qc_analysis_dir(dataset)
    metrics_path = (
        shape_qc_dir
        / "roi_log_ratio_metrics_size_and_shape_filtered.csv"
    )
    top30_path = (
        shape_qc_dir
        / f"top30_{direction_label}_rois_size_and_shape_filtered.csv"
    )
    output_root = shape_qc_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{output_dir_prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    log_message(run_start_seconds, f"Output directory: {output_dir}")

    log_message(run_start_seconds, f"Loading ROI metrics from {metrics_path}")
    roi_metrics = pd.read_csv(metrics_path)
    log_message(run_start_seconds, f"Loaded {len(roi_metrics)} ROI/day rows; computing day-wise fits and residuals")
    fit_summary = summarize_daily_green_red_linear_fits(roi_metrics)
    residual_table = compute_green_red_fit_residuals(
        roi_metrics=roi_metrics,
        fit_summary=fit_summary,
    )
    residual_summary = summarize_residual_sign_changes(residual_table)

    full_residual_table = residual_table.merge(residual_summary, on="roi_id", how="left")
    residual_table_path = output_dir / "roi_metrics_with_green_red_fit_residuals.csv"
    full_residual_table.to_csv(residual_table_path, index=False)

    fit_summary_path = output_dir / "daywise_green_red_linear_fit_summary.csv"
    fit_summary.to_csv(fit_summary_path, index=False)

    top30_rois = pd.read_csv(top30_path).sort_values("selection_rank").head(30).copy()
    top30_roi_ids = top30_rois["roi_id"].astype(int).tolist()
    top30_table = select_ranked_roi_days(
        roi_day_table=full_residual_table,
        ranking_table=top30_rois,
        top_n=30,
        ranking_columns=["selection_rank", "min_delta_log2_green_over_red"],
    )
    top30_table_path = output_dir / f"top30_{direction_label}_green_red_fit_residuals.csv"
    top30_table.to_csv(top30_table_path, index=False)
    log_message(run_start_seconds, f"Selected {len(top30_table['roi_id'].unique())} ranked {direction_label} ROIs; rendering plots")

    scatter_plot_path = output_dir / f"top30_{direction_label}_green_red_trajectory_scatter_average_fit.png"
    plot_directional_roi_trajectories_scatter(
        ranked_roi_table=top30_table,
        fit_summary=fit_summary,
        output_path=scatter_plot_path,
        title_prefix="Top 30",
        direction_label=direction_label,
    )
    residual_plot_path = output_dir / f"top30_{direction_label}_green_fit_residual_vs_day.png"
    plot_directional_roi_residuals_vs_day(
        ranked_roi_table=top30_table,
        output_path=residual_plot_path,
        title_prefix="Top 30",
        direction_label=direction_label,
    )

    shutil.copy2(__file__, output_dir / Path(__file__).name)
    write_summary_markdown(
        output_dir=output_dir,
        residual_table_path=residual_table_path,
        top30_table_path=top30_table_path,
        scatter_plot_path=scatter_plot_path,
        residual_plot_path=residual_plot_path,
        direction_label=direction_label,
    )
    total_duration_seconds = time.perf_counter() - run_start_seconds
    write_run_log(
        output_dir=output_dir,
        average_slope=float(fit_summary["slope"].mean()),
        average_intercept=float(fit_summary["intercept"].mean()),
        top30_roi_ids=top30_roi_ids,
        direction_label=direction_label,
        total_duration_seconds=total_duration_seconds,
    )

    print(f"[{format_duration_seconds(total_duration_seconds)}] Completed {direction_label} green-fit residual analysis", flush=True)
    print(f"output_dir={output_dir}")
    print(f"residual_table_path={residual_table_path}")
    print(f"top30_table_path={top30_table_path}")
    print(f"scatter_plot_path={scatter_plot_path}")
    print(f"residual_plot_path={residual_plot_path}")
    print(f"total_duration={format_duration_seconds(total_duration_seconds)}")
    return output_dir


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the residual-analysis script.

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
    """Run the day-wise green-vs-red residual analysis for decreasing ROIs."""

    args = parse_args()
    run_directional_residual_analysis(
        dataset=args.dataset,
        direction_label="decreasing",
        output_dir_prefix="daywise_green_red_fit_residuals",
    )


if __name__ == "__main__":
    main()
