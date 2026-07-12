"""Run the shared registered-space green/red ROI analysis pipeline.

This script provides a dataset-agnostic entrypoint for the Fucci ROI workflow.
It extracts ROI intensities from registered image stacks, applies dark-value
correction, performs size and shape QC, computes green/red metrics and fit
residuals, ranks changing ROIs, renders registered-image ROI panels, and
optionally renders raw-space validation panels when inverse-warped masks are
available.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import argparse
import json
import shutil
import time

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import tifffile

from analysis_paths import get_dataset_analysis_dir, resolve_dataset_dir
from roi_log_ratio_analysis import (
    add_day0_normalized_column,
    apply_channel_dark_correction,
    attach_roi_size_metrics,
    build_inverse_warped_mask_lookup,
    build_raw_image_lookup,
    build_registered_image_lookup,
    compute_green_red_fit_residuals,
    compute_log_ratio_metrics,
    compute_roi_size_table,
    estimate_size_filter_bounds,
    extract_registered_dataset_roi_intensity_table,
    filter_complete_rois,
    flag_shape_qc_rois,
    prepare_stack_for_display,
    project_roi_stack_view,
    select_ranked_roi_days,
    select_top_changing_rois,
    summarize_daily_green_red_linear_fits,
    summarize_residual_sign_changes,
    summarize_roi_metrics,
    wide_table_from_long_table,
)
from run_daywise_green_red_fit_residuals import (
    plot_directional_roi_residuals_vs_day,
    plot_directional_roi_trajectories_scatter,
)
from run_daywise_green_red_linear_fit_summary import (
    compute_regression_ci_band,
    plot_daywise_scatter_summary,
    plot_fit_parameter_summary,
)
from run_raw_space_inverse_mask_validation import build_magenta_green_colormaps


ANALYSIS_VERSION = "0.2.0"


@dataclass(frozen=True)
class RegisteredPipelineConfig:
    """Store parameters for the registered-space ROI analysis workflow.

    Attributes
    ----------
    dataset : str
        Dataset alias or explicit dataset path understood by
        :func:`analysis_paths.resolve_dataset_dir`.
    start_date : str or None
        Reference date in ``YYYYMMDD`` format that defines day 0. When ``None``,
        the earliest raw TIFF date in the dataset directory is used.
    mask_name : str
        Filename of the ROI mask TIFF in the dataset directory.
    green_dark : float
        Fixed green-channel dark offset in arbitrary fluorescence units.
    red_dark : float
        Fixed red-channel dark offset in arbitrary fluorescence units.
    epsilon : float
        Positive offset added before computing ``log2((green + eps)/(red + eps))``.
    xy_um_per_px : float
        XY pixel size in micrometers per pixel for ROI size reporting.
    z_um_per_plane : float
        Z step size in micrometers per plane for ROI size reporting.
    red_cv_max : float
        Maximum allowed red-channel coefficient of variation for candidate ROI
        ranking.
    min_day0_brightness_quantile : float
        Lower day-0 brightness quantile used to suppress dim ROI candidates.
    max_top_rois : int
        Maximum number of ranked increasing or decreasing ROIs to export.
    pad_xy : int
        XY padding in pixels for registered-space per-ROI display crops.
    min_crop_size : int
        Minimum crop width and height in pixels for registered-space per-ROI
        display crops.
    inverse_mask_suffix : str
        Suffix used to locate day-specific inverse-warped ROI masks in raw
        image space.
    inverse_mask_channel : {"red", "green"} or None
        Preferred channel used to disambiguate inverse ROI masks when both red
        and green versions exist for the same day.
    raw_space_half_window_z : int
        Number of z planes to include above and below the centroid slice for
        raw-space validation panels.
    skip_raw_space_validation : bool
        Whether to skip optional raw-space inverse-mask validation panels even
        if inverse masks are available. This defaults to ``True`` because the
        raw-space validation figures are often the slowest stage.
    """

    dataset: str = "1050"
    start_date: str | None = None
    mask_name: str = ""
    green_dark: float = 319.0
    red_dark: float = 534.0
    epsilon: float = 1.0
    xy_um_per_px: float = 0.693
    z_um_per_plane: float = 5.0
    red_cv_max: float = 0.10
    min_day0_brightness_quantile: float = 0.25
    max_top_rois: int = 30
    pad_xy: int = 20
    min_crop_size: int = 48
    inverse_mask_suffix: str = "_ROI_mask_SyN_inversed.tif"
    inverse_mask_channel: str | None = None
    raw_space_half_window_z: int = 5
    skip_raw_space_validation: bool = True


@dataclass(frozen=True)
class RawSpaceRoiDayGeometry:
    """Describe one ROI location inside one day-specific raw-space mask."""

    z_center: int
    y_center: int
    x_center: int
    width_px: int
    height_px: int


def unique_sorted_days_from_channel_lookup(
    channel_lookup: dict[tuple[int, str], object],
) -> list[int]:
    """Return unique sorted day indices from a day/channel lookup table.

    Parameters
    ----------
    channel_lookup : dict[tuple[int, str], object]
        Mapping whose keys are ``(day, channel)`` tuples. The mapped values are
        not inspected.

    Returns
    -------
    list[int]
        Sorted unique day indices.
    """

    return sorted({int(day) for day, _channel in channel_lookup})


def infer_default_inverse_mask_channel(dataset: str | Path | None) -> str:
    """Infer the default inverse-mask channel from a dataset identifier.

    Parameters
    ----------
    dataset : str, pathlib.Path, or None
        Dataset alias or path.

    Returns
    -------
    str
        ``"green"`` for 920-like datasets and ``"red"`` otherwise.
    """

    dataset_text = "" if dataset is None else str(dataset).lower()
    dataset_name = Path(dataset_text).name
    if dataset_text == "920" or dataset_name.startswith("920") or "920_data" in dataset_text:
        return "green"
    return "red"


def compute_raw_space_roi_day_geometry(
    mask_stack: np.ndarray,
    roi_id: int,
) -> RawSpaceRoiDayGeometry:
    """Measure one ROI inside one raw-space day-specific label stack.

    Parameters
    ----------
    mask_stack : numpy.ndarray
        Integer ROI label image with shape ``(z, y, x)`` in raw image space.
    roi_id : int
        Positive ROI label to locate in ``mask_stack``.

    Returns
    -------
    RawSpaceRoiDayGeometry
        Centroid and bounding-box measurements for the requested ROI in pixel
        coordinates.
    """

    roi_mask = mask_stack == roi_id
    if not np.any(roi_mask):
        raise ValueError(f"ROI {roi_id} was not found in the mask stack.")

    z_coords, y_coords, x_coords = np.nonzero(roi_mask)
    return RawSpaceRoiDayGeometry(
        z_center=int(np.clip(np.round(np.mean(z_coords)), 0, mask_stack.shape[0] - 1)),
        y_center=int(np.round(np.mean(y_coords))),
        x_center=int(np.round(np.mean(x_coords))),
        width_px=int(x_coords.max() - x_coords.min() + 1),
        height_px=int(y_coords.max() - y_coords.min() + 1),
    )


def compute_fixed_crop_bounds(
    image_shape_yx: tuple[int, int],
    y_center: int,
    x_center: int,
    crop_height_px: int,
    crop_width_px: int,
) -> tuple[int, int, int, int]:
    """Build a fixed-size XY crop around a center point.

    Parameters
    ----------
    image_shape_yx : tuple[int, int]
        Raw image plane shape as ``(y, x)`` in pixels.
    y_center : int
        Crop center y coordinate in pixels.
    x_center : int
        Crop center x coordinate in pixels.
    crop_height_px : int
        Requested crop height in pixels.
    crop_width_px : int
        Requested crop width in pixels.

    Returns
    -------
    tuple[int, int, int, int]
        ``(y_start, y_stop, x_start, x_stop)`` bounds with inclusive starts and
        exclusive stops in pixel units.
    """

    image_height, image_width = image_shape_yx

    y_start = int(round(y_center - crop_height_px / 2))
    y_stop = y_start + int(crop_height_px)
    x_start = int(round(x_center - crop_width_px / 2))
    x_stop = x_start + int(crop_width_px)

    if y_start < 0:
        y_stop -= y_start
        y_start = 0
    if x_start < 0:
        x_stop -= x_start
        x_start = 0
    if y_stop > image_height:
        shift = y_stop - image_height
        y_start = max(0, y_start - shift)
        y_stop = image_height
    if x_stop > image_width:
        shift = x_stop - image_width
        x_start = max(0, x_start - shift)
        x_stop = image_width

    return y_start, y_stop, x_start, x_stop


def infer_start_date_from_dataset_dir(image_dir: str | Path) -> str:
    """Infer the day-0 date from the earliest raw image filename.

    Parameters
    ----------
    image_dir : str or pathlib.Path
        Dataset directory that contains raw per-day TIFF files named like
        ``YYYYMMDD_R.tif`` or ``YYYYMMDD_G.tif``.

    Returns
    -------
    str
        Earliest detected raw-image date in ``YYYYMMDD`` format.

    Raises
    ------
    FileNotFoundError
        If no raw per-day TIFF files matching the expected naming convention
        are found.
    """

    image_dir = Path(image_dir)
    candidate_dates: list[str] = []
    for path in image_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != ".tif":
            continue

        parts = path.stem.split("_")
        if len(parts) < 2 or len(parts[0]) != 8 or not parts[0].isdigit():
            continue
        if parts[1] not in {"R", "G"}:
            continue

        trailing_lower = [token.lower() for token in parts[2:]]
        if "syn" in trailing_lower:
            continue
        if any("mask" in token for token in trailing_lower) or "roi" in trailing_lower:
            continue
        candidate_dates.append(parts[0])

    if not candidate_dates:
        raise FileNotFoundError(
            f"Could not infer a start date from raw TIFF files in {image_dir}."
        )

    return min(candidate_dates)


def format_duration_seconds(duration_seconds: float) -> str:
    """Format an elapsed wall-clock duration as ``HH:MM:SS``.

    Parameters
    ----------
    duration_seconds : float
        Elapsed wall-clock time in seconds.

    Returns
    -------
    str
        Zero-padded ``HH:MM:SS`` string.
    """

    total_seconds = max(0, int(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def log_stage_start(stage_label: str, pipeline_start_seconds: float) -> float:
    """Print a pipeline stage start message and return the stage start time.

    Parameters
    ----------
    stage_label : str
        Human-readable stage label for terminal progress output.
    pipeline_start_seconds : float
        Pipeline start time from :func:`time.perf_counter` in seconds.

    Returns
    -------
    float
        Stage start time from :func:`time.perf_counter` in seconds.
    """

    elapsed_seconds = time.perf_counter() - pipeline_start_seconds
    print(f"[{format_duration_seconds(elapsed_seconds)}] Starting: {stage_label}")
    return time.perf_counter()


def log_stage_end(
    stage_key: str,
    stage_label: str,
    pipeline_start_seconds: float,
    stage_start_seconds: float,
    stage_durations_seconds: dict[str, float],
    detail: str | None = None,
) -> float:
    """Record and print timing information for one completed pipeline stage.

    Parameters
    ----------
    stage_key : str
        Stable dictionary key used in ``stage_durations_seconds``.
    stage_label : str
        Human-readable stage label for terminal progress output.
    pipeline_start_seconds : float
        Pipeline start time from :func:`time.perf_counter` in seconds.
    stage_start_seconds : float
        Stage start time from :func:`time.perf_counter` in seconds.
    stage_durations_seconds : dict[str, float]
        Mapping that stores stage durations in seconds.
    detail : str or None, default=None
        Optional short message appended to the terminal output.

    Returns
    -------
    float
        Stage duration in seconds.
    """

    stage_duration_seconds = time.perf_counter() - stage_start_seconds
    stage_durations_seconds[stage_key] = stage_duration_seconds
    total_elapsed_seconds = time.perf_counter() - pipeline_start_seconds
    suffix = "" if detail is None else f" | {detail}"
    print(
        f"[{format_duration_seconds(total_elapsed_seconds)} | +{format_duration_seconds(stage_duration_seconds)}] "
        f"Finished: {stage_label}{suffix}"
    )
    return stage_duration_seconds


def make_day_date_labels(day_values: np.ndarray, start_date: str) -> list[str]:
    """Convert integer day offsets into date labels.

    Parameters
    ----------
    day_values : numpy.ndarray
        One-dimensional integer day offsets relative to ``start_date``.
    start_date : str
        Reference date in ``YYYYMMDD`` format.

    Returns
    -------
    list[str]
        Date labels in ``YYYYMMDD`` format.
    """

    reference_date = pd.to_datetime(start_date, format="%Y%m%d")
    return [
        (reference_date + pd.to_timedelta(int(day_value), unit="D")).strftime("%Y%m%d")
        for day_value in day_values
    ]


def plot_filter_counts_bargraph(
    filter_counts: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot ROI retention counts across the QC steps.

    Parameters
    ----------
    filter_counts : pandas.DataFrame
        Table with columns ``step``, ``count``, ``pct_of_start``, and
        ``pct_of_previous`` summarizing the retained ROI count at each filter
        stage.
    output_path : pathlib.Path
        PNG path for the saved bar graph.
    """

    figure, axis = plt.subplots(figsize=(8.4, 5.0), facecolor="white")
    x_positions = np.arange(len(filter_counts))
    counts = filter_counts["count"].to_numpy(dtype=int)
    bars = axis.bar(x_positions, counts, color="#3a86ff", alpha=0.9)
    axis.set_xticks(x_positions, filter_counts["step"].tolist(), rotation=20, ha="right")
    axis.set_ylabel("ROI count", fontsize=11)
    axis.set_title("ROI retention across QC steps", fontsize=13)
    axis.tick_params(labelsize=10)

    for bar, (_, row) in zip(bars, filter_counts.iterrows(), strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            (
                f"{int(row['count'])}\n"
                f"{row['pct_of_start']:.1f}% start\n"
                f"{row['pct_of_previous']:.1f}% prev"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.text(
        0.5,
        0.02,
        (
            "Counts show how many ROIs survive completeness, size, and shape QC "
            "in the registered-space ROI workflow."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.08, 1.0, 0.96))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_population_summary(
    roi_metrics: pd.DataFrame,
    output_path: Path,
    start_date: str,
    include_traces: bool,
) -> None:
    """Plot population summaries for raw and normalized green/red metrics.

    Parameters
    ----------
    roi_metrics : pandas.DataFrame
        ROI/day table with columns ``roi_id``, ``day``, ``green_fraction``,
        ``log2_green_over_red``, and ``delta_log2_green_over_red``.
    output_path : pathlib.Path
        PNG path for the saved figure.
    start_date : str
        Reference date in ``YYYYMMDD`` format used for axis labels.
    include_traces : bool
        Whether to overlay faint single-ROI trajectories behind the population
        summaries.
    """

    day_values = np.sort(roi_metrics["day"].unique())
    date_labels = make_day_date_labels(day_values, start_date=start_date)
    summary = (
        roi_metrics.groupby("day")
        .agg(
            raw_log2_median=("log2_green_over_red", "median"),
            raw_log2_q25=("log2_green_over_red", lambda values: np.quantile(values, 0.25)),
            raw_log2_q75=("log2_green_over_red", lambda values: np.quantile(values, 0.75)),
            delta_log2_median=("delta_log2_green_over_red", "median"),
            delta_log2_q25=("delta_log2_green_over_red", lambda values: np.quantile(values, 0.25)),
            delta_log2_q75=("delta_log2_green_over_red", lambda values: np.quantile(values, 0.75)),
            green_fraction_median=("green_fraction", "median"),
            green_fraction_q25=("green_fraction", lambda values: np.quantile(values, 0.25)),
            green_fraction_q75=("green_fraction", lambda values: np.quantile(values, 0.75)),
        )
        .reset_index()
    )

    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), facecolor="white")
    panel_specs = [
        (
            "log2_green_over_red",
            "raw_log2_median",
            "raw_log2_q25",
            "raw_log2_q75",
            "Raw log2(G/R)",
            "#264653",
        ),
        (
            "delta_log2_green_over_red",
            "delta_log2_median",
            "delta_log2_q25",
            "delta_log2_q75",
            "Delta log2(G/R)",
            "#d62828",
        ),
        (
            "green_fraction",
            "green_fraction_median",
            "green_fraction_q25",
            "green_fraction_q75",
            "Green fraction",
            "#2a9d8f",
        ),
    ]

    for axis, (value_column, median_column, q25_column, q75_column, title, color) in zip(
        axes,
        panel_specs,
        strict=True,
    ):
        if include_traces:
            for _, roi_table in roi_metrics.groupby("roi_id", sort=False):
                roi_table = roi_table.sort_values("day")
                axis.plot(
                    roi_table["day"].to_numpy(dtype=int),
                    roi_table[value_column].to_numpy(dtype=float),
                    color=color,
                    alpha=0.12,
                    linewidth=0.8,
                    zorder=1,
                )

        axis.fill_between(
            summary["day"].to_numpy(dtype=int),
            summary[q25_column].to_numpy(dtype=float),
            summary[q75_column].to_numpy(dtype=float),
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=2,
        )
        axis.plot(
            summary["day"].to_numpy(dtype=int),
            summary[median_column].to_numpy(dtype=float),
            color=color,
            linewidth=2.2,
            marker="o",
            markersize=5.0,
            zorder=3,
        )
        if value_column == "delta_log2_green_over_red":
            axis.axhline(0.0, color="0.4", linestyle="--", linewidth=1.0, zorder=0)
        axis.set_title(title, fontsize=12)
        axis.set_xticks(day_values, [f"Day {int(day)}\n{label}" for day, label in zip(day_values, date_labels, strict=True)])
        axis.tick_params(labelsize=9)

    figure.suptitle(
        "Size+shape-filtered ROI population summaries",
        fontsize=14,
    )
    caption = (
        "Solid lines show the median across ROIs and shaded bands show the "
        "interquartile range. Faint lines show individual ROIs."
        if include_traces
        else "Solid lines show the median across ROIs and shaded bands show the interquartile range."
    )
    figure.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9)
    figure.tight_layout(rect=(0.02, 0.08, 1.0, 0.92))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_single_day_red_green_scatter(
    roi_metrics: pd.DataFrame,
    fit_summary: pd.DataFrame,
    output_path: Path,
    day: int,
    start_date: str,
    log_scale: bool,
) -> None:
    """Plot corrected red versus green ROI values for one day.

    Parameters
    ----------
    roi_metrics : pandas.DataFrame
        ROI/day table with corrected ``red`` and ``green`` columns.
    fit_summary : pandas.DataFrame
        Day-wise fit table with one row per day.
    output_path : pathlib.Path
        PNG path for the saved scatter plot.
    day : int
        Day index to visualize.
    start_date : str
        Reference date in ``YYYYMMDD`` format used for the title.
    log_scale : bool
        Whether to display both axes on a logarithmic scale.
    """

    day_table = roi_metrics.loc[roi_metrics["day"] == int(day), ["red", "green"]].copy()
    if log_scale:
        day_table = day_table[(day_table["red"] > 0) & (day_table["green"] > 0)].copy()
    day_table = day_table.dropna().reset_index(drop=True)

    x_values = day_table["red"].to_numpy(dtype=float)
    y_values = day_table["green"].to_numpy(dtype=float)
    x_grid, y_hat, y_low, y_high = compute_regression_ci_band(x_values=x_values, y_values=y_values)

    fit_row = fit_summary.loc[fit_summary["day"] == int(day)].iloc[0]
    date_label = make_day_date_labels(np.array([day], dtype=int), start_date=start_date)[0]

    figure, axis = plt.subplots(figsize=(6.2, 5.4), facecolor="white")
    axis.scatter(
        x_values,
        y_values,
        s=10,
        alpha=0.14,
        color="#264653",
        edgecolors="none",
        rasterized=True,
    )
    if np.all(np.isfinite(y_low)):
        axis.fill_between(x_grid, y_low, y_high, color="#8ecae6", alpha=0.30, linewidth=0)
    axis.plot(x_grid, y_hat, color="#d62828", linewidth=2.0)

    if log_scale:
        axis.set_xscale("log")
        axis.set_yscale("log")

    axis.set_xlabel("Corrected red intensity", fontsize=11)
    axis.set_ylabel("Corrected green intensity", fontsize=11)
    axis.set_title(
        f"Day {day} corrected red vs green ROI values\n{date_label}",
        fontsize=13,
    )
    axis.text(
        0.04,
        0.96,
        (
            f"slope = {fit_row['slope']:.3f}\n"
            f"intercept = {fit_row['intercept']:.1f}\n"
            f"R² = {fit_row['r_squared']:.3f}\n"
            f"n = {int(fit_row['n_rois'])}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )
    figure.text(
        0.5,
        0.02,
        (
            "Each point is one size+shape-filtered ROI on the requested day. "
            "The red line is the fitted linear relationship and the blue band is "
            "the 95% confidence interval of the mean fit."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.08, 1.0, 0.95))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_ranked_heatmap(
    ranked_roi_days: pd.DataFrame,
    value_column: str,
    output_path: Path,
    start_date: str,
    title: str,
    colorbar_label: str,
    cmap: str | mcolors.Colormap = "viridis",
    center_zero: bool = False,
) -> pd.DataFrame:
    """Render a day-by-ROI heatmap for one ranked ROI subset.

    Parameters
    ----------
    ranked_roi_days : pandas.DataFrame
        ROI/day table containing at least ``selection_rank``, ``roi_id``,
        ``day``, and ``value_column``.
    value_column : str
        Column to pivot into the heatmap matrix.
    output_path : pathlib.Path
        PNG path for the saved heatmap.
    start_date : str
        Reference date in ``YYYYMMDD`` format used for column labels.
    title : str
        Plot title.
    colorbar_label : str
        Label for the colorbar.
    cmap : str or matplotlib.colors.Colormap, default="viridis"
        Colormap for the heatmap.
    center_zero : bool, default=False
        Whether to center the colormap at zero using a diverging normalization.

    Returns
    -------
    pandas.DataFrame
        Wide matrix used for the heatmap, with one row per ranked ROI.
    """

    required_columns = {"selection_rank", "roi_id", "day", value_column}
    missing_columns = required_columns.difference(ranked_roi_days.columns)
    if missing_columns:
        missing_str = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required heatmap columns: {missing_str}")

    heatmap_table = ranked_roi_days.copy()
    heatmap_table["row_label"] = heatmap_table.apply(
        lambda row: f"{int(row['selection_rank']):02d}:{int(row['roi_id'])}",
        axis=1,
    )
    wide_table = (
        heatmap_table.pivot_table(
            index="row_label",
            columns="day",
            values=value_column,
            aggfunc="first",
        )
        .sort_index()
    )
    day_values = wide_table.columns.to_numpy(dtype=int)
    date_labels = make_day_date_labels(day_values, start_date=start_date)

    matrix = wide_table.to_numpy(dtype=float)
    figure_height = max(5.0, 0.25 * len(wide_table))
    figure, axis = plt.subplots(figsize=(4.6, figure_height), facecolor="white")
    if center_zero:
        vmax = float(np.nanmax(np.abs(matrix)))
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    else:
        norm = None

    image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    axis.set_xticks(np.arange(len(day_values)), [f"Day {int(day)}\n{label}" for day, label in zip(day_values, date_labels, strict=True)])
    axis.set_yticks(np.arange(len(wide_table)), wide_table.index.tolist())
    axis.set_title(title, fontsize=12)
    axis.tick_params(labelsize=8)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label, fontsize=10)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return wide_table.reset_index()


def format_roi_panel_title(roi_row: pd.Series, direction_label: str) -> str:
    """Format a metadata-rich title for one registered-image ROI panel.

    Parameters
    ----------
    roi_row : pandas.Series
        One row from a ranked ROI summary table. The series must contain ROI
        identity, rank, size, and shape-QC fields.
    direction_label : str
        Ranking direction label such as ``"decreasing"`` or ``"increasing"``.

    Returns
    -------
    str
        Multi-line title string for the ROI panel.
    """

    if direction_label == "decreasing":
        metric_text = f"min dlog2(G/R) {roi_row['min_delta_log2_green_over_red']:.3f}"
    else:
        metric_text = f"max dlog2(G/R) {roi_row['max_delta_log2_green_over_red']:.3f}"

    return (
        f"ROI {int(roi_row['roi_id'])} | rank {int(roi_row['selection_rank']):02d} | {metric_text}\n"
        f"area {roi_row['proj_area_px']:.0f} px | eq diam {roi_row['eq_diam_um']:.2f} um | "
        f"red CV {roi_row['red_cv']:.3f} | circ {roi_row['circularity']:.2f} | "
        f"sol {roi_row['solidity']:.2f} | ar {roi_row['axis_ratio']:.2f}"
    )


def render_registered_roi_panel(
    roi_row: pd.Series,
    image_cache: dict[tuple[int, str], np.ndarray],
    mask_stack: np.ndarray,
    output_path: Path,
    start_date: str,
    direction_label: str,
    display_mode: str,
    green_dark: float,
    red_dark: float,
    pad_xy: int,
    min_crop_size: int,
) -> None:
    """Render one two-row ROI panel from the registered image stacks.

    Parameters
    ----------
    roi_row : pandas.Series
        Ranked ROI summary row for the ROI to render.
    image_cache : dict[tuple[int, str], numpy.ndarray]
        Mapping from ``(day, channel)`` to registered image stacks with shape
        ``(z, y, x)`` in arbitrary fluorescence units.
    mask_stack : numpy.ndarray
        Integer ROI label stack with shape ``(z, y, x)`` that matches the
        registered image geometry.
    output_path : pathlib.Path
        PNG path for the saved panel.
    start_date : str
        Reference date in ``YYYYMMDD`` format used for day labels.
    direction_label : str
        Ranking direction label such as ``"decreasing"`` or ``"increasing"``.
    display_mode : {"raw", "dark_subtracted"}
        Whether to render the raw registered intensities or a display-only
        dark-subtracted version.
    green_dark : float
        Green-channel dark offset in arbitrary fluorescence units.
    red_dark : float
        Red-channel dark offset in arbitrary fluorescence units.
    pad_xy : int
        XY crop padding in pixels.
    min_crop_size : int
        Minimum crop size in pixels.
    """

    roi_id = int(roi_row["roi_id"])
    days = unique_sorted_days_from_channel_lookup(image_cache)
    date_labels = make_day_date_labels(np.asarray(days, dtype=int), start_date=start_date)
    magenta_cmap, green_cmap = build_magenta_green_colormaps()

    row_defs = [("red", "Red"), ("green", "Green")]
    crop_lookup: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    red_values: list[np.ndarray] = []
    green_values: list[np.ndarray] = []

    for day in days:
        for channel, _label in row_defs:
            image_stack = image_cache[(day, channel)]
            if display_mode == "dark_subtracted":
                dark_value = red_dark if channel == "red" else green_dark
                display_stack = prepare_stack_for_display(
                    image_stack=image_stack,
                    dark_value=dark_value,
                    clip_floor=0.0,
                )
            else:
                display_stack = prepare_stack_for_display(
                    image_stack=image_stack,
                    dark_value=0.0,
                    clip_floor=0.0,
                )
            image_projection, roi_projection, _bounds = project_roi_stack_view(
                image_stack=display_stack,
                mask_stack=mask_stack,
                roi_id=roi_id,
                pad_xy=pad_xy,
                min_crop_size=min_crop_size,
                z_pad=1,
            )
            crop_lookup[(day, channel)] = (image_projection, roi_projection)
            if channel == "red":
                red_values.append(image_projection)
            else:
                green_values.append(image_projection)

    red_vmax = float(np.percentile(np.concatenate([value.ravel() for value in red_values]), 99.5))
    green_vmax = float(np.percentile(np.concatenate([value.ravel() for value in green_values]), 99.5))
    red_vmax = max(red_vmax, 1.0)
    green_vmax = max(green_vmax, 1.0)

    figure_width = max(7.5, len(days) * 2.3 + 1.2)
    fig = plt.figure(figsize=(figure_width, 5.8), facecolor="white")
    grid = GridSpec(
        nrows=2,
        ncols=len(days) + 2,
        figure=fig,
        width_ratios=[1.0] * len(days) + [0.06, 0.06],
        wspace=0.06,
        hspace=0.10,
        left=0.05,
        right=0.96,
        top=0.86,
        bottom=0.10,
    )
    row_to_mappable: dict[int, object] = {}

    for row_index, (channel, row_label) in enumerate(row_defs):
        cmap = magenta_cmap if channel == "red" else green_cmap
        vmax = red_vmax if channel == "red" else green_vmax
        for col_index, (day, date_label) in enumerate(zip(days, date_labels, strict=True)):
            axis = fig.add_subplot(grid[row_index, col_index])
            image_projection, roi_projection = crop_lookup[(day, channel)]
            mappable = axis.imshow(
                image_projection,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )
            axis.contour(roi_projection > 0, levels=[0.5], colors="white", linewidths=0.8)
            if row_index == 0:
                axis.set_title(f"Day {day}\n{date_label}", fontsize=10)
            if col_index == 0:
                axis.set_ylabel(row_label, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            row_to_mappable[row_index] = mappable

    red_cax = fig.add_subplot(grid[0, len(days)])
    green_cax = fig.add_subplot(grid[1, len(days)])
    fig.colorbar(row_to_mappable[0], cax=red_cax, label="Red intensity")
    fig.colorbar(row_to_mappable[1], cax=green_cax, label="Green intensity")
    fig.suptitle(format_roi_panel_title(roi_row, direction_label=direction_label), fontsize=12)
    fig.text(
        0.5,
        0.02,
        (
            "Max-projection registered views centered on each ROI. White contour "
            f"shows the analysis ROI mask. Display mode: {display_mode.replace('_', ' ')}."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_directional_subset(
    direction_label: str,
    roi_summary: pd.DataFrame,
    roi_metrics_with_residuals: pd.DataFrame,
    fit_summary: pd.DataFrame,
    image_cache: dict[tuple[int, str], np.ndarray],
    mask_stack: np.ndarray,
    output_dir: Path,
    config: RegisteredPipelineConfig,
) -> pd.DataFrame:
    """Export tables and plots for one ranked ROI direction.

    Parameters
    ----------
    direction_label : str
        Ranking direction label, either ``"decreasing"`` or ``"increasing"``.
    roi_summary : pandas.DataFrame
        One-row-per-ROI summary table that already includes size and shape
        fields.
    roi_metrics_with_residuals : pandas.DataFrame
        Size+shape-filtered ROI/day table with residual columns added.
    fit_summary : pandas.DataFrame
        Day-wise red-vs-green fit summary table.
    image_cache : dict[tuple[int, str], numpy.ndarray]
        Registered image stacks indexed by ``(day, channel)``.
    mask_stack : numpy.ndarray
        Integer ROI mask stack used for extraction and display.
    output_dir : pathlib.Path
        Root directory for the current analysis run.
    config : RegisteredPipelineConfig
        Analysis parameter bundle.
    """

    top_rois = select_top_changing_rois(
        roi_summary=roi_summary,
        max_rois=config.max_top_rois,
        red_cv_max=config.red_cv_max,
        min_day0_brightness_quantile=config.min_day0_brightness_quantile,
        direction=direction_label,
    )
    top_rois_path = output_dir / f"top{len(top_rois)}_{direction_label}_rois_size_and_shape_filtered.csv"
    top_rois.to_csv(top_rois_path, index=False)

    ranking_columns = [
        "selection_rank",
        "min_delta_log2_green_over_red",
        "max_delta_log2_green_over_red",
        "proj_area_px",
        "eq_diam_um",
        "red_cv",
        "circularity",
        "solidity",
        "axis_ratio",
    ]
    ranked_roi_days = select_ranked_roi_days(
        roi_day_table=roi_metrics_with_residuals,
        ranking_table=top_rois,
        top_n=len(top_rois),
        ranking_columns=ranking_columns,
    )
    ranked_roi_days_path = output_dir / f"top{len(top_rois)}_{direction_label}_roi_day_metrics.csv"
    ranked_roi_days.to_csv(ranked_roi_days_path, index=False)

    delta_heatmap_path = output_dir / f"top{len(top_rois)}_{direction_label}_delta_log2_heatmap.png"
    delta_heatmap_table = plot_ranked_heatmap(
        ranked_roi_days=ranked_roi_days,
        value_column="delta_log2_green_over_red",
        output_path=delta_heatmap_path,
        start_date=config.start_date,
        title=f"Top {len(top_rois)} {direction_label} ROIs: delta log2(G/R)",
        colorbar_label="Delta log2(G/R)",
        cmap="coolwarm",
        center_zero=True,
    )
    delta_heatmap_table.to_csv(
        output_dir / f"top{len(top_rois)}_{direction_label}_delta_log2_heatmap_table.csv",
        index=False,
    )

    green_day_table = add_day0_normalized_column(ranked_roi_days, source_column="green")
    green_raw_heatmap_path = output_dir / f"top{len(top_rois)}_{direction_label}_green_raw_heatmap.png"
    green_raw_heatmap_table = plot_ranked_heatmap(
        ranked_roi_days=green_day_table,
        value_column="green",
        output_path=green_raw_heatmap_path,
        start_date=config.start_date,
        title=f"Top {len(top_rois)} {direction_label} ROIs: corrected green",
        colorbar_label="Corrected green intensity",
        cmap="viridis",
    )
    green_raw_heatmap_table.to_csv(
        output_dir / f"top{len(top_rois)}_{direction_label}_green_raw_heatmap_table.csv",
        index=False,
    )

    green_norm_heatmap_path = (
        output_dir / f"top{len(top_rois)}_{direction_label}_green_day0_normalized_heatmap.png"
    )
    green_norm_heatmap_table = plot_ranked_heatmap(
        ranked_roi_days=green_day_table,
        value_column="green_normalized_to_day0",
        output_path=green_norm_heatmap_path,
        start_date=config.start_date,
        title=f"Top {len(top_rois)} {direction_label} ROIs: green/day0 green",
        colorbar_label="Green normalized to day 0",
        cmap="viridis",
    )
    green_norm_heatmap_table.to_csv(
        output_dir / f"top{len(top_rois)}_{direction_label}_green_day0_normalized_heatmap_table.csv",
        index=False,
    )

    scatter_output = output_dir / f"top{len(top_rois)}_{direction_label}_green_red_trajectory_scatter_average_fit.png"
    plot_directional_roi_trajectories_scatter(
        ranked_roi_table=ranked_roi_days,
        fit_summary=fit_summary,
        output_path=scatter_output,
        title_prefix=f"Top {len(top_rois)}",
        direction_label=direction_label,
        start_date=config.start_date,
    )

    residual_output = output_dir / f"top{len(top_rois)}_{direction_label}_green_fit_residual_vs_day.png"
    plot_directional_roi_residuals_vs_day(
        ranked_roi_table=ranked_roi_days,
        output_path=residual_output,
        title_prefix=f"Top {len(top_rois)}",
        direction_label=direction_label,
        start_date=config.start_date,
    )

    for display_mode in ("raw", "dark_subtracted"):
        panel_dir = output_dir / f"top{len(top_rois)}_{direction_label}_{display_mode}_display_roi_panels"
        panel_dir.mkdir(parents=True, exist_ok=False)
        for _, roi_row in top_rois.iterrows():
            panel_path = panel_dir / (
                f"rank_{int(roi_row['selection_rank']):02d}_roi_{int(roi_row['roi_id'])}.png"
            )
            render_registered_roi_panel(
                roi_row=roi_row,
                image_cache=image_cache,
                mask_stack=mask_stack,
                output_path=panel_path,
                start_date=config.start_date,
                direction_label=direction_label,
                display_mode=display_mode,
                green_dark=config.green_dark,
                red_dark=config.red_dark,
                pad_xy=config.pad_xy,
                min_crop_size=config.min_crop_size,
            )

    return top_rois


def format_z_offset_label(offset: int) -> str:
    """Format one z-offset label for raw-space panel rows.

    Parameters
    ----------
    offset : int
        Signed z offset in planes relative to the ROI centroid slice.

    Returns
    -------
    str
        Human-readable label such as ``"z+5"``, ``"z"``, or ``"z-2"``.
    """

    if offset == 0:
        return "z"
    sign = "+" if offset > 0 else ""
    return f"z{sign}{offset}"


def render_raw_space_roi_panel(
    roi_row: pd.Series,
    raw_lookup: dict[tuple[int, str], Path],
    mask_lookup: dict[int, Path],
    output_path: Path,
    start_date: str,
    direction_label: str,
    half_window_z: int,
    crop_pad_xy: int,
    min_crop_size_px: int,
) -> None:
    """Render one raw-space ROI panel using day-specific inverse-warped masks.

    Parameters
    ----------
    roi_row : pandas.Series
        Ranked ROI summary row for the ROI to render.
    raw_lookup : dict[tuple[int, str], pathlib.Path]
        Mapping from ``(day, channel)`` to raw TIFF paths.
    mask_lookup : dict[int, pathlib.Path]
        Mapping from day index to the ROI mask TIFF in raw image space.
    output_path : pathlib.Path
        PNG path for the saved panel.
    start_date : str
        Reference date in ``YYYYMMDD`` format used for day labels.
    direction_label : str
        Ranking direction label such as ``"decreasing"`` or ``"increasing"``.
    half_window_z : int
        Number of z planes to include above and below the ROI centroid slice.
    crop_pad_xy : int
        Extra XY padding in pixels added around the largest observed ROI crop.
    min_crop_size_px : int
        Minimum crop width and height in pixels.
    """

    roi_id = int(roi_row["roi_id"])
    days = unique_sorted_days_from_channel_lookup(raw_lookup)
    date_labels = make_day_date_labels(np.asarray(days, dtype=int), start_date=start_date)
    magenta_cmap, green_cmap = build_magenta_green_colormaps()

    geometry_by_day: dict[int, RawSpaceRoiDayGeometry] = {}
    raw_cache: dict[tuple[int, str], np.ndarray] = {}
    mask_cache: dict[int, np.ndarray] = {}
    for day in days:
        mask_stack = tifffile.imread(mask_lookup[day])
        mask_cache[day] = mask_stack
        geometry_by_day[day] = compute_raw_space_roi_day_geometry(mask_stack=mask_stack, roi_id=roi_id)
        for channel in ("red", "green"):
            raw_cache[(day, channel)] = tifffile.imread(raw_lookup[(day, channel)]).astype(float)

    crop_width_px = max(
        min_crop_size_px,
        max(geometry.width_px for geometry in geometry_by_day.values()) + 2 * crop_pad_xy,
    )
    crop_height_px = max(
        min_crop_size_px,
        max(geometry.height_px for geometry in geometry_by_day.values()) + 2 * crop_pad_xy,
    )

    z_offsets = list(range(int(half_window_z), -int(half_window_z) - 1, -1))
    row_definitions = [("red", offset) for offset in z_offsets] + [("green", offset) for offset in z_offsets]

    crop_lookup: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]] = {}
    red_values: list[np.ndarray] = []
    green_values: list[np.ndarray] = []
    for day in days:
        geometry = geometry_by_day[day]
        mask_stack = mask_cache[day]
        y_start, y_stop, x_start, x_stop = compute_fixed_crop_bounds(
            image_shape_yx=mask_stack.shape[1:],
            y_center=geometry.y_center,
            x_center=geometry.x_center,
            crop_height_px=crop_height_px,
            crop_width_px=crop_width_px,
        )
        for channel, offset in row_definitions:
            z_index = geometry.z_center + int(offset)
            if 0 <= z_index < mask_stack.shape[0]:
                image_crop = raw_cache[(day, channel)][z_index, y_start:y_stop, x_start:x_stop]
                mask_crop = mask_stack[z_index, y_start:y_stop, x_start:x_stop]
            else:
                image_crop = np.zeros((y_stop - y_start, x_stop - x_start), dtype=float)
                mask_crop = np.zeros((y_stop - y_start, x_stop - x_start), dtype=np.uint16)
            crop_lookup[(day, channel, offset)] = (image_crop, mask_crop)
            if channel == "red":
                red_values.append(image_crop)
            else:
                green_values.append(image_crop)

    red_vmax = float(np.percentile(np.concatenate([value.ravel() for value in red_values]), 99.5)) if red_values else 1.0
    green_vmax = float(np.percentile(np.concatenate([value.ravel() for value in green_values]), 99.5)) if green_values else 1.0
    red_vmax = max(red_vmax, 1.0)
    green_vmax = max(green_vmax, 1.0)

    figure_width = max(11.0, len(days) * 2.1 + 1.2)
    figure_height = max(12.0, 0.85 * len(row_definitions) + 1.5)
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    grid = GridSpec(
        nrows=len(row_definitions),
        ncols=len(days) + 2,
        figure=fig,
        width_ratios=[1.0] * len(days) + [0.06, 0.06],
        wspace=0.06,
        hspace=0.05,
        left=0.05,
        right=0.96,
        top=0.94,
        bottom=0.05,
    )
    row_to_mappable: dict[str, object] = {}

    for row_index, (channel, offset) in enumerate(row_definitions):
        cmap = magenta_cmap if channel == "red" else green_cmap
        vmax = red_vmax if channel == "red" else green_vmax
        for col_index, (day, date_label) in enumerate(zip(days, date_labels, strict=True)):
            axis = fig.add_subplot(grid[row_index, col_index])
            image_crop, mask_crop = crop_lookup[(day, channel, offset)]
            mappable = axis.imshow(
                image_crop,
                cmap=cmap,
                vmin=0.0,
                vmax=vmax,
                interpolation="nearest",
            )
            if np.any(mask_crop == roi_id):
                axis.contour(mask_crop == roi_id, levels=[0.5], colors="white", linewidths=0.8)
            if row_index == 0:
                axis.set_title(f"Day {day}\n{date_label}", fontsize=10)
            if col_index == 0:
                axis.set_ylabel(f"{channel.title()} {format_z_offset_label(offset)}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            row_to_mappable[channel] = mappable

    red_cax = fig.add_subplot(grid[0:len(z_offsets), len(days)])
    green_cax = fig.add_subplot(grid[len(z_offsets):len(row_definitions), len(days)])
    fig.colorbar(row_to_mappable["red"], cax=red_cax, label="Raw red intensity")
    fig.colorbar(row_to_mappable["green"], cax=green_cax, label="Raw green intensity")
    fig.suptitle(format_roi_panel_title(roi_row, direction_label=direction_label), fontsize=12)
    fig.text(
        0.5,
        0.02,
        (
            "Raw-space ROI validation using day-specific inverse-warped masks. "
            f"Each channel shows the centroid slice plus/minus {half_window_z} z planes where available."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def export_raw_space_validation_panels(
    direction_label: str,
    top_rois: pd.DataFrame,
    base_dir: Path,
    output_dir: Path,
    config: RegisteredPipelineConfig,
) -> dict[str, object]:
    """Export raw-space validation panels when inverse masks are available.

    Parameters
    ----------
    direction_label : str
        Ranking direction label, either ``"decreasing"`` or ``"increasing"``.
    top_rois : pandas.DataFrame
        Ranked ROI summary table for the requested direction.
    base_dir : pathlib.Path
        Dataset directory that contains raw images and inverse-warped masks.
    output_dir : pathlib.Path
        Root analysis output directory for the current run.
    config : RegisteredPipelineConfig
        Analysis parameter bundle.

    Returns
    -------
    dict[str, object]
        Small status dictionary describing whether the export ran and how many
        panels were written.
    """

    preferred_channel = config.inverse_mask_channel or infer_default_inverse_mask_channel(config.dataset)
    raw_lookup = build_raw_image_lookup(base_dir, start_date=config.start_date)
    try:
        mask_lookup = build_inverse_warped_mask_lookup(
            image_dir=base_dir,
            day0_mask_name=config.mask_name,
            inverse_mask_suffix=config.inverse_mask_suffix,
            preferred_channel=preferred_channel,
            start_date=config.start_date,
        )
    except (FileNotFoundError, ValueError) as error:
        return {"available": False, "reason": str(error)}

    required_days = unique_sorted_days_from_channel_lookup(raw_lookup)
    missing_days = sorted(set(required_days).difference(mask_lookup))
    if missing_days:
        return {
            "available": False,
            "reason": f"Missing inverse ROI masks for days: {missing_days}",
        }

    panel_dir = output_dir / f"top{len(top_rois)}_{direction_label}_raw_space_inverse_mask_panels"
    panel_dir.mkdir(parents=True, exist_ok=False)
    skipped_rows: list[dict[str, object]] = []
    rendered_count = 0
    for _, roi_row in top_rois.iterrows():
        output_path = panel_dir / f"rank_{int(roi_row['selection_rank']):02d}_roi_{int(roi_row['roi_id'])}.png"
        try:
            render_raw_space_roi_panel(
                roi_row=roi_row,
                raw_lookup=raw_lookup,
                mask_lookup=mask_lookup,
                output_path=output_path,
                start_date=config.start_date,
                direction_label=direction_label,
                half_window_z=config.raw_space_half_window_z,
                crop_pad_xy=config.pad_xy,
                min_crop_size_px=config.min_crop_size,
            )
            rendered_count += 1
        except ValueError as error:
            skipped_rows.append(
                {
                    "roi_id": int(roi_row["roi_id"]),
                    "selection_rank": int(roi_row["selection_rank"]),
                    "reason": str(error),
                }
            )

    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(
            output_dir / f"top{len(top_rois)}_{direction_label}_raw_space_skipped_rois.csv",
            index=False,
        )

    return {
        "available": True,
        "panel_dir": str(panel_dir),
        "rendered_count": rendered_count,
        "preferred_channel": preferred_channel,
        "skipped_count": len(skipped_rows),
    }


def write_summary_markdown(
    output_dir: Path,
    config: RegisteredPipelineConfig,
    filter_counts: pd.DataFrame,
    fit_summary: pd.DataFrame,
    raw_space_status: dict[str, object],
) -> None:
    """Write a short markdown summary for one registered ROI pipeline run.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory for the current analysis run.
    config : RegisteredPipelineConfig
        Analysis parameter bundle.
    filter_counts : pandas.DataFrame
        ROI-retention table across the QC steps.
    fit_summary : pandas.DataFrame
        Day-wise red-vs-green fit summary table.
    raw_space_status : dict[str, object]
        Status dictionary describing whether optional raw-space validation was
        run.
    """

    n_days = len(fit_summary)
    summary_lines = [
        "# Registered ROI Pipeline",
        "",
        "Goal: run the shared registered-space ROI pipeline for one dataset,",
        "including ROI extraction, dark correction, QC filtering, green/red",
        "metrics, fit residuals, ranked ROI exports, and optional raw-space",
        "inverse-mask validation.",
        "",
        "Key analysis choices:",
        f"- Dataset: `{config.dataset}`",
        f"- Start date (day 0): `{config.start_date}`",
        f"- ROI mask: `{config.mask_name}`",
        f"- Dark values: green `{config.green_dark}`, red `{config.red_dark}`",
        f"- Number of imaging days detected after filtering: `{n_days}`",
        "",
        "QC summary:",
    ]
    for _, row in filter_counts.iterrows():
        summary_lines.append(
            f"- {row['step']}: `{int(row['count'])}` ROIs "
            f"({row['pct_of_start']:.1f}% of starting mask ROIs)"
        )
    summary_lines.extend(["", "Green-vs-red fit summary:"])
    for _, row in fit_summary.iterrows():
        summary_lines.append(
            f"- Day {int(row['day'])}: slope `{row['slope']:.3f}`, "
            f"intercept `{row['intercept']:.1f}`, R² `{row['r_squared']:.3f}`"
        )
    summary_lines.extend(["", "Raw-space validation:"])
    if raw_space_status.get('available'):
        summary_lines.append(
            f"- Exported raw-space validation panels using preferred inverse-mask channel `{raw_space_status.get('preferred_channel')}`."
        )
        summary_lines.append(
            f"- Decreasing ROI panels: `{raw_space_status.get('decreasing_panel_dir')}`"
        )
        summary_lines.append(
            f"- Increasing ROI panels: `{raw_space_status.get('increasing_panel_dir')}`"
        )
    else:
        summary_lines.append(
            f"- Skipped raw-space validation: `{raw_space_status.get('reason', 'not available')}`"
        )
    summary_lines.extend(
        [
            "",
            "Outputs include:",
            "- Raw and dark-corrected ROI tables",
            "- Size and shape QC tables",
            "- Population summaries",
            "- Day-wise red-vs-green fit summaries",
            "- Top increasing and decreasing ROI heatmaps",
            "- Top increasing and decreasing ROI residual plots",
            "- Registered-image ROI panels in raw and dark-subtracted display modes",
        ]
    )
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")



def write_run_log(
    output_dir: Path,
    config: RegisteredPipelineConfig,
    filter_counts: pd.DataFrame,
    raw_space_status: dict[str, object],
    stage_durations_seconds: dict[str, float],
    total_duration_seconds: float,
) -> None:
    """Write a JSON run log for reproducibility.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory for the current analysis run.
    config : RegisteredPipelineConfig
        Analysis parameter bundle.
    filter_counts : pandas.DataFrame
        ROI-retention table across the QC steps.
    raw_space_status : dict[str, object]
        Status dictionary describing whether optional raw-space validation was
        run.
    stage_durations_seconds : dict[str, float]
        Mapping from pipeline stage names to wall-clock durations in seconds.
    total_duration_seconds : float
        Total wall-clock duration for the full pipeline run in seconds.
    """

    log_payload = {
        'analysis_version': ANALYSIS_VERSION,
        'run_timestamp': datetime.now().isoformat(),
        'config': asdict(config),
        'filter_counts': filter_counts.to_dict(orient='records'),
        'raw_space_status': raw_space_status,
        'stage_durations_seconds': stage_durations_seconds,
        'total_duration_seconds': float(total_duration_seconds),
        'total_duration_hms': format_duration_seconds(total_duration_seconds),
    }
    (output_dir / 'run_log.json').write_text(
        json.dumps(log_payload, indent=2),
        encoding='utf-8',
    )



def run_registered_roi_pipeline(config: RegisteredPipelineConfig) -> Path:
    """Run the shared registered-space ROI analysis pipeline.

    Parameters
    ----------
    config : RegisteredPipelineConfig
        Analysis parameter bundle.

    Returns
    -------
    pathlib.Path
        Output directory created for this analysis run.
    """

    if not config.mask_name:
        raise ValueError('mask_name must be provided.')

    pipeline_start_seconds = time.perf_counter()
    stage_durations_seconds: dict[str, float] = {}
    base_dir = resolve_dataset_dir(config.dataset)
    effective_start_date = config.start_date or infer_start_date_from_dataset_dir(base_dir)
    config = RegisteredPipelineConfig(**{**asdict(config), "start_date": effective_start_date})
    analysis_root = get_dataset_analysis_dir(config.dataset)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mask_stem = Path(config.mask_name).stem
    output_dir = analysis_root / f'registered_roi_pipeline_{mask_stem}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=False)
    print(
        f"[{format_duration_seconds(0.0)}] Pipeline start | dataset={config.dataset} | "
        f"day0={config.start_date} | mask={config.mask_name}"
    )
    print(f"[{format_duration_seconds(0.0)}] Output directory: {output_dir}")

    stage_start_seconds = log_stage_start(
        stage_label='Extract ROI intensities from registered stacks',
        pipeline_start_seconds=pipeline_start_seconds,
    )
    mask_path = base_dir / config.mask_name
    mask_stack = tifffile.imread(mask_path)
    raw_roi_table = extract_registered_dataset_roi_intensity_table(
        image_dir=base_dir,
        mask_path=mask_path,
        start_date=config.start_date,
    )
    raw_roi_table.to_csv(output_dir / 'roi_intensity_results_raw.csv', index=False)
    log_stage_end(
        stage_key='extract_roi_intensities',
        stage_label='Extract ROI intensities from registered stacks',
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
        detail=f"{raw_roi_table['roi_id'].nunique()} ROIs observed in raw extraction table",
    )

    stage_start_seconds = log_stage_start(
        stage_label='Dark correction and complete-day ROI table',
        pipeline_start_seconds=pipeline_start_seconds,
    )
    corrected_roi_table = apply_channel_dark_correction(
        intensity_table=raw_roi_table,
        green_dark=config.green_dark,
        red_dark=config.red_dark,
        intensity_column='mean_intensity',
        corrected_column='mean_intensity_corrected',
        clip_floor=None,
    )
    corrected_roi_table.to_csv(output_dir / 'roi_intensity_results_dark_corrected.csv', index=False)

    roi_day_table = wide_table_from_long_table(
        corrected_roi_table,
        intensity_column='mean_intensity_corrected',
        start_date=config.start_date,
    )
    complete_roi_day_table = filter_complete_rois(roi_day_table)
    complete_roi_day_table.to_csv(output_dir / 'roi_day_table_complete.csv', index=False)
    log_stage_end(
        stage_key='dark_correction_and_complete_days',
        stage_label='Dark correction and complete-day ROI table',
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
        detail=f"{complete_roi_day_table['roi_id'].nunique()} ROIs present on all detected days",
    )

    stage_start_seconds = log_stage_start(
        stage_label='Size and shape QC metrics',
        pipeline_start_seconds=pipeline_start_seconds,
    )
    roi_metrics = compute_log_ratio_metrics(
        complete_roi_day_table,
        epsilon=config.epsilon,
    )
    roi_size_table = compute_roi_size_table(
        mask_stack=mask_stack,
        xy_um_per_px=config.xy_um_per_px,
        z_um_per_plane=config.z_um_per_plane,
    )
    size_bounds = estimate_size_filter_bounds(roi_size_table, area_column='proj_area_px')
    roi_size_table['size_qc_pass'] = roi_size_table['proj_area_px'].between(
        size_bounds['lower_bound'],
        size_bounds['upper_bound'],
        inclusive='both',
    )
    roi_size_table['size_qc_lower_bound_px'] = float(size_bounds['lower_bound'])
    roi_size_table['size_qc_upper_bound_px'] = float(size_bounds['upper_bound'])
    roi_size_table = flag_shape_qc_rois(roi_size_table)
    roi_size_table.to_csv(output_dir / 'roi_size_table_with_shape_qc.csv', index=False)

    metrics_with_size = roi_metrics.merge(roi_size_table, on='roi_id', how='left', validate='many_to_one')
    metrics_size_filtered = metrics_with_size.loc[metrics_with_size['size_qc_pass']].copy()
    metrics_size_shape_filtered = metrics_with_size.loc[
        metrics_with_size['size_qc_pass'] & metrics_with_size['shape_qc_pass']
    ].copy()
    metrics_size_filtered.to_csv(output_dir / 'roi_log_ratio_metrics_size_filtered.csv', index=False)
    metrics_size_shape_filtered.to_csv(
        output_dir / 'roi_log_ratio_metrics_size_and_shape_filtered.csv',
        index=False,
    )
    log_stage_end(
        stage_key='size_and_shape_qc',
        stage_label='Size and shape QC metrics',
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
        detail=(
            f"{metrics_size_shape_filtered['roi_id'].nunique()} ROIs pass size+shape QC"
        ),
    )

    stage_start_seconds = log_stage_start(
        stage_label='ROI summaries, linear fits, and residuals',
        pipeline_start_seconds=pipeline_start_seconds,
    )
    roi_summary = summarize_roi_metrics(metrics_size_shape_filtered)
    roi_summary = attach_roi_size_metrics(roi_summary, roi_size_table)
    roi_summary.to_csv(output_dir / 'roi_log_ratio_summary_size_and_shape_filtered.csv', index=False)

    fit_summary = summarize_daily_green_red_linear_fits(metrics_size_shape_filtered)
    fit_summary.to_csv(output_dir / 'daywise_green_red_linear_fit_summary.csv', index=False)

    residual_table = compute_green_red_fit_residuals(
        roi_metrics=metrics_size_shape_filtered,
        fit_summary=fit_summary,
    )
    residual_summary = summarize_residual_sign_changes(residual_table)
    roi_metrics_with_residuals = residual_table.merge(
        residual_summary,
        on='roi_id',
        how='left',
        validate='many_to_one',
    )
    roi_metrics_with_residuals.to_csv(
        output_dir / 'roi_metrics_with_green_red_fit_residuals.csv',
        index=False,
    )
    log_stage_end(
        stage_key='roi_summaries_and_residuals',
        stage_label='ROI summaries, linear fits, and residuals',
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
        detail=f"{len(fit_summary)} imaging days summarized",
    )

    stage_start_seconds = log_stage_start(
        stage_label='Population summaries and QC count plots',
        pipeline_start_seconds=pipeline_start_seconds,
    )
    filter_counts = pd.DataFrame(
        [
            {'step': 'mask_rois', 'count': int(roi_size_table['roi_id'].nunique())},
            {'step': 'complete_days', 'count': int(complete_roi_day_table['roi_id'].nunique())},
            {'step': 'size_filtered', 'count': int(metrics_size_filtered['roi_id'].nunique())},
            {'step': 'size_shape_filtered', 'count': int(metrics_size_shape_filtered['roi_id'].nunique())},
        ]
    )
    starting_count = float(filter_counts['count'].iloc[0])
    filter_counts['pct_of_start'] = 100.0 * filter_counts['count'].astype(float) / starting_count
    filter_counts['pct_of_previous'] = 100.0 * filter_counts['count'].astype(float) / filter_counts['count'].shift(1, fill_value=starting_count).astype(float)
    filter_counts.to_csv(output_dir / 'filter_step_counts_with_percentages.csv', index=False)
    plot_filter_counts_bargraph(filter_counts, output_dir / 'filter_step_counts_bargraph.png')

    plot_population_summary(
        roi_metrics=metrics_size_shape_filtered,
        output_path=output_dir / 'population_longitudinal_summary_size_and_shape_filtered.png',
        start_date=config.start_date,
        include_traces=False,
    )
    plot_population_summary(
        roi_metrics=metrics_size_shape_filtered,
        output_path=output_dir / 'population_longitudinal_summary_size_and_shape_filtered_with_roi_traces.png',
        start_date=config.start_date,
        include_traces=True,
    )
    log_stage_end(
        stage_key='population_summaries',
        stage_label='Population summaries and QC count plots',
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
    )

    stage_start_seconds = log_stage_start(
        stage_label='Day-wise green vs red fit plots',
        pipeline_start_seconds=pipeline_start_seconds,
    )
    plot_daywise_scatter_summary(
        roi_metrics=metrics_size_shape_filtered,
        fit_summary=fit_summary,
        output_path=output_dir / 'daywise_green_red_linear_fit_scatters.png',
        start_date=config.start_date,
    )
    plot_fit_parameter_summary(
        fit_summary=fit_summary,
        output_path=output_dir / 'daywise_green_red_linear_fit_parameters.png',
        start_date=config.start_date,
    )
    plot_single_day_red_green_scatter(
        roi_metrics=metrics_size_shape_filtered,
        fit_summary=fit_summary,
        output_path=output_dir / 'day0_red_vs_green_scatter_size_and_shape_filtered.png',
        day=0,
        start_date=config.start_date,
        log_scale=False,
    )
    plot_single_day_red_green_scatter(
        roi_metrics=metrics_size_shape_filtered,
        fit_summary=fit_summary,
        output_path=output_dir / 'day0_red_vs_green_scatter_size_and_shape_filtered_loglog.png',
        day=0,
        start_date=config.start_date,
        log_scale=True,
    )
    log_stage_end(
        stage_key='fit_plots',
        stage_label='Day-wise green vs red fit plots',
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
    )

    stage_start_seconds = log_stage_start(
        stage_label='Load registered image cache',
        pipeline_start_seconds=pipeline_start_seconds,
    )
    registered_lookup = build_registered_image_lookup(
        image_dir=base_dir,
        start_date=config.start_date,
    )
    image_cache = {
        key: tifffile.imread(path).astype(float)
        for key, path in sorted(registered_lookup.items())
    }
    log_stage_end(
        stage_key='load_registered_image_cache',
        stage_label='Load registered image cache',
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
        detail=f"{len(image_cache)} day/channel stacks loaded",
    )

    stage_start_seconds = log_stage_start(
        stage_label='Ranked decreasing ROI exports',
        pipeline_start_seconds=pipeline_start_seconds,
    )
    decreasing_top_rois = export_directional_subset(
        direction_label='decreasing',
        roi_summary=roi_summary,
        roi_metrics_with_residuals=roi_metrics_with_residuals,
        fit_summary=fit_summary,
        image_cache=image_cache,
        mask_stack=mask_stack,
        output_dir=output_dir,
        config=config,
    )
    log_stage_end(
        stage_key='decreasing_roi_exports',
        stage_label='Ranked decreasing ROI exports',
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
        detail=f"{len(decreasing_top_rois)} decreasing ROIs exported",
    )

    stage_start_seconds = log_stage_start(
        stage_label='Ranked increasing ROI exports',
        pipeline_start_seconds=pipeline_start_seconds,
    )
    increasing_top_rois = export_directional_subset(
        direction_label='increasing',
        roi_summary=roi_summary,
        roi_metrics_with_residuals=roi_metrics_with_residuals,
        fit_summary=fit_summary,
        image_cache=image_cache,
        mask_stack=mask_stack,
        output_dir=output_dir,
        config=config,
    )
    log_stage_end(
        stage_key='increasing_roi_exports',
        stage_label='Ranked increasing ROI exports',
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
        detail=f"{len(increasing_top_rois)} increasing ROIs exported",
    )

    raw_space_status: dict[str, object]
    stage_start_seconds = log_stage_start(
        stage_label='Raw-space inverse-mask validation',
        pipeline_start_seconds=pipeline_start_seconds,
    )
    if config.skip_raw_space_validation:
        raw_space_status = {'available': False, 'reason': 'skipped by configuration'}
    else:
        decreasing_status = export_raw_space_validation_panels(
            direction_label='decreasing',
            top_rois=decreasing_top_rois,
            base_dir=base_dir,
            output_dir=output_dir,
            config=config,
        )
        increasing_status = export_raw_space_validation_panels(
            direction_label='increasing',
            top_rois=increasing_top_rois,
            base_dir=base_dir,
            output_dir=output_dir,
            config=config,
        )
        if decreasing_status.get('available') and increasing_status.get('available'):
            raw_space_status = {
                'available': True,
                'preferred_channel': decreasing_status.get('preferred_channel'),
                'decreasing_panel_dir': decreasing_status.get('panel_dir'),
                'increasing_panel_dir': increasing_status.get('panel_dir'),
                'decreasing_rendered_count': decreasing_status.get('rendered_count'),
                'increasing_rendered_count': increasing_status.get('rendered_count'),
                'decreasing_skipped_count': decreasing_status.get('skipped_count'),
                'increasing_skipped_count': increasing_status.get('skipped_count'),
            }
        else:
            raw_space_status = {
                'available': False,
                'reason': decreasing_status.get('reason') or increasing_status.get('reason') or 'inverse masks not available for all days',
            }
    log_stage_end(
        stage_key='raw_space_validation',
        stage_label='Raw-space inverse-mask validation',
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
        detail=raw_space_status.get('reason') if not raw_space_status.get('available') else 'completed',
    )

    stage_start_seconds = log_stage_start(
        stage_label='Finalize outputs and run metadata',
        pipeline_start_seconds=pipeline_start_seconds,
    )
    shutil.copy2(__file__, output_dir / Path(__file__).name)
    write_summary_markdown(
        output_dir=output_dir,
        config=config,
        filter_counts=filter_counts,
        fit_summary=fit_summary,
        raw_space_status=raw_space_status,
    )
    log_stage_end(
        stage_key='finalize_outputs',
        stage_label='Finalize outputs and run metadata',
        pipeline_start_seconds=pipeline_start_seconds,
        stage_start_seconds=stage_start_seconds,
        stage_durations_seconds=stage_durations_seconds,
    )

    total_duration_seconds = time.perf_counter() - pipeline_start_seconds
    stage_durations_seconds['total'] = total_duration_seconds
    write_run_log(
        output_dir=output_dir,
        config=config,
        filter_counts=filter_counts,
        raw_space_status=raw_space_status,
        stage_durations_seconds=stage_durations_seconds,
        total_duration_seconds=total_duration_seconds,
    )
    print(f"[{format_duration_seconds(total_duration_seconds)}] Pipeline completed")
    print(f'output_dir={output_dir}')
    print(f'n_mask_rois={int(filter_counts.loc[0, "count"])}')
    print(f'n_size_shape_filtered_rois={int(filter_counts.loc[3, "count"])}')
    return output_dir



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the registered ROI pipeline.

    Parameters
    ----------
    argv : list[str] or None, default=None
        Optional command-line argument list. When ``None``, arguments are read
        from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='1050', help='Dataset alias (e.g. 1050 or 920) or an explicit dataset directory path.')
    parser.add_argument('--start-date', default=None, help='Optional reference date in YYYYMMDD format that defines day 0. If omitted, the earliest raw TIFF date in the dataset directory is used.')
    parser.add_argument('--mask-name', required=True, help='ROI mask filename inside the dataset directory.')
    parser.add_argument('--green-dark', type=float, default=319.0, help='Green-channel dark offset in arbitrary fluorescence units.')
    parser.add_argument('--red-dark', type=float, default=534.0, help='Red-channel dark offset in arbitrary fluorescence units.')
    parser.add_argument('--xy-um-per-px', type=float, default=0.693, help='XY resolution in micrometers per pixel.')
    parser.add_argument('--z-um-per-plane', type=float, default=5.0, help='Z step size in micrometers per plane.')
    parser.add_argument('--max-top-rois', type=int, default=30, help='Maximum number of ranked increasing or decreasing ROIs to export.')
    parser.add_argument('--inverse-mask-suffix', default='_ROI_mask_SyN_inversed.tif', help='Filename suffix used to find inverse-warped ROI masks in raw image space.')
    parser.add_argument('--inverse-mask-channel', choices=['auto', 'red', 'green'], default='auto', help='Preferred channel used when both red and green inverse ROI masks exist.')
    parser.add_argument('--raw-space-half-window-z', type=int, default=5, help='Number of z planes to include above and below the ROI centroid for raw-space validation.')
    raw_space_group = parser.add_mutually_exclusive_group()
    raw_space_group.add_argument('--enable-raw-space-validation', action='store_true', help='Enable optional raw-space inverse-mask validation. This is off by default because it is often the slowest stage.')
    raw_space_group.add_argument('--skip-raw-space-validation', action='store_true', help='Compatibility flag that keeps raw-space inverse-mask validation disabled.')
    return parser.parse_args(argv)



def main() -> None:
    """Run the shared registered-space ROI pipeline from the command line."""

    args = parse_args()
    inverse_mask_channel = None if args.inverse_mask_channel == 'auto' else args.inverse_mask_channel
    skip_raw_space_validation = True
    if args.enable_raw_space_validation:
        skip_raw_space_validation = False
    if args.skip_raw_space_validation:
        skip_raw_space_validation = True
    config = RegisteredPipelineConfig(
        dataset=args.dataset,
        start_date=args.start_date,
        mask_name=args.mask_name,
        green_dark=args.green_dark,
        red_dark=args.red_dark,
        xy_um_per_px=args.xy_um_per_px,
        z_um_per_plane=args.z_um_per_plane,
        max_top_rois=args.max_top_rois,
        inverse_mask_suffix=args.inverse_mask_suffix,
        inverse_mask_channel=inverse_mask_channel,
        raw_space_half_window_z=args.raw_space_half_window_z,
        skip_raw_space_validation=skip_raw_space_validation,
    )
    run_registered_roi_pipeline(config)


if __name__ == '__main__':
    main()
