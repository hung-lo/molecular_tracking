"""Standalone raw-space ROI panel rendering helpers.

This module is meant to support notebook-based exploration of a small set of
selected ROIs in raw image space. It keeps the plotting logic separate from the
main processing pipeline so the figure layout can be edited later without
touching the analysis scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from run_registered_roi_pipeline import (
    compute_fixed_crop_bounds,
    compute_raw_space_roi_day_geometry,
)


@dataclass(frozen=True)
class RawSpaceTripletPanelGeometry:
    """Summarize one ROI location inside one day-specific raw-space mask.

    Attributes
    ----------
    z_center : int
        Centroid-centered z index in pixels.
    y_center : int
        Mean ROI y coordinate in pixels.
    x_center : int
        Mean ROI x coordinate in pixels.
    width_px : int
        Bounding-box width in pixels.
    height_px : int
        Bounding-box height in pixels.
    """

    z_center: int
    y_center: int
    x_center: int
    width_px: int
    height_px: int


def make_day_date_labels(
    day_values: np.ndarray,
    start_date: str = "20260511",
) -> list[str]:
    """Convert integer day offsets into ``YYYYMMDD`` labels.

    Parameters
    ----------
    day_values : numpy.ndarray
        One-dimensional array of integer day offsets relative to day 0.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format.

    Returns
    -------
    list[str]
        One formatted date string per input day.
    """

    reference_date = pd.to_datetime(start_date, format="%Y%m%d")
    labels: list[str] = []
    for day_value in day_values:
        day_date = reference_date + pd.to_timedelta(int(day_value), unit="D")
        labels.append(day_date.strftime("%Y%m%d"))
    return labels


def format_z_offset_label(offset: int) -> str:
    """Format one z-offset label for figure rows.

    Parameters
    ----------
    offset : int
        Integer z offset relative to the ROI centroid slice.

    Returns
    -------
    str
        Human-readable label such as ``"z+5"``, ``"z"``, or ``"z-2"``.
    """

    if offset == 0:
        return "z"
    sign = "+" if offset > 0 else ""
    return f"z{sign}{offset}"


def measure_roi_geometry(mask_stack: np.ndarray, roi_id: int) -> RawSpaceTripletPanelGeometry:
    """Measure the centroid and bounding box for one ROI mask label.

    Parameters
    ----------
    mask_stack : numpy.ndarray
        Integer label stack with shape ``(z, y, x)`` in raw image space.
    roi_id : int
        Positive ROI label to locate.

    Returns
    -------
    RawSpaceTripletPanelGeometry
        Geometry summary for the requested ROI.
    """

    geometry = compute_raw_space_roi_day_geometry(mask_stack=mask_stack, roi_id=roi_id)
    return RawSpaceTripletPanelGeometry(
        z_center=geometry.z_center,
        y_center=geometry.y_center,
        x_center=geometry.x_center,
        width_px=geometry.width_px,
        height_px=geometry.height_px,
    )


def build_composite_rgb(
    red_plane: np.ndarray,
    green_plane: np.ndarray,
    red_vmax: float,
    green_vmax: float,
) -> np.ndarray:
    """Build a magenta-green RGB composite from one raw red and green plane.

    Parameters
    ----------
    red_plane : numpy.ndarray
        Two-dimensional red-channel raw image plane.
    green_plane : numpy.ndarray
        Two-dimensional green-channel raw image plane.
    red_vmax : float
        Upper display bound for red intensities.
    green_vmax : float
        Upper display bound for green intensities.

    Returns
    -------
    numpy.ndarray
        RGB image with shape ``(y, x, 3)`` in the range ``[0, 1]``.
    """

    red_plane = np.asarray(red_plane, dtype=float)
    green_plane = np.asarray(green_plane, dtype=float)

    red_norm = np.clip(red_plane / max(float(red_vmax), 1.0), 0.0, 1.0)
    green_norm = np.clip(green_plane / max(float(green_vmax), 1.0), 0.0, 1.0)

    rgb = np.zeros((*red_norm.shape, 3), dtype=float)
    rgb[..., 0] = red_norm
    rgb[..., 2] = red_norm
    rgb[..., 1] = green_norm
    return np.clip(rgb, 0.0, 1.0)


def _format_roi_title(roi_id: int, roi_summary: pd.Series | None) -> str:
    """Build a metadata-rich title for one selected ROI."""

    if roi_summary is None:
        return f"ROI {roi_id}"

    parts = [f"ROI {roi_id}"]
    if "proj_area_px" in roi_summary and not pd.isna(roi_summary["proj_area_px"]):
        parts.append(f"area {float(roi_summary['proj_area_px']):.0f} px")
    if "eq_diam_um" in roi_summary and not pd.isna(roi_summary["eq_diam_um"]):
        parts.append(f"eq diam {float(roi_summary['eq_diam_um']):.2f} um")
    if "day0_brightness" in roi_summary and not pd.isna(roi_summary["day0_brightness"]):
        parts.append(f"day0 bright {float(roi_summary['day0_brightness']):.1f}")
    if "red_cv" in roi_summary and not pd.isna(roi_summary["red_cv"]):
        parts.append(f"red CV {float(roi_summary['red_cv']):.3f}")
    if "circularity_x" in roi_summary and not pd.isna(roi_summary["circularity_x"]):
        parts.append(f"circ {float(roi_summary['circularity_x']):.2f}")
    if "solidity_x" in roi_summary and not pd.isna(roi_summary["solidity_x"]):
        parts.append(f"sol {float(roi_summary['solidity_x']):.2f}")
    if "axis_ratio_x" in roi_summary and not pd.isna(roi_summary["axis_ratio_x"]):
        parts.append(f"ar {float(roi_summary['axis_ratio_x']):.2f}")
    return " | ".join(parts)


def render_raw_space_triplet_panel(
    roi_id: int,
    raw_stack_lookup: dict[tuple[int, str], np.ndarray],
    mask_stack_lookup: dict[int, np.ndarray],
    output_path: Path,
    roi_summary: pd.Series | None = None,
    start_date: str = "20260511",
    half_window_z: int = 5,
    crop_pad_xy: int = 20,
    min_crop_size_px: int = 48,
) -> list[dict[str, object]]:
    """Render one selected ROI as a raw-space multi-day z-stack panel.

    Parameters
    ----------
    roi_id : int
        Positive ROI label to render.
    raw_stack_lookup : dict[tuple[int, str], numpy.ndarray]
        Mapping from ``(day, channel)`` to raw image stacks with shape
        ``(z, y, x)``. Channel names must be ``"red"`` and ``"green"``.
    mask_stack_lookup : dict[int, numpy.ndarray]
        Mapping from day index to inverse-warped ROI label stacks with shape
        ``(z, y, x)``. Day 0 should already be in raw space.
    output_path : pathlib.Path
        PNG file to create.
    roi_summary : pandas.Series or None, default=None
        Optional metadata row for the selected ROI. If present, the title uses
        any available size and shape summary columns.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format used for column labels.
    half_window_z : int, default=5
        Number of z planes to display above and below the centroid slice.
    crop_pad_xy : int, default=20
        XY padding added around the largest ROI footprint when building the
        shared crop window.
    min_crop_size_px : int, default=48
        Minimum crop width and height in pixels.

    Returns
    -------
    list[dict[str, object]]
        One metadata dictionary per displayed day and z-plane combination.
    """

    if half_window_z < 0:
        raise ValueError("half_window_z must be non-negative.")
    if "red" not in {channel for _day, channel in raw_stack_lookup}:
        raise ValueError("raw_stack_lookup must include red-channel stacks.")
    if "green" not in {channel for _day, channel in raw_stack_lookup}:
        raise ValueError("raw_stack_lookup must include green-channel stacks.")

    days = sorted(mask_stack_lookup)
    if not days:
        raise ValueError("mask_stack_lookup must contain at least one day.")

    date_labels = make_day_date_labels(np.asarray(days, dtype=int), start_date=start_date)
    geometry_by_day = {
        day: measure_roi_geometry(mask_stack=mask_stack_lookup[day], roi_id=roi_id)
        for day in days
    }

    crop_width_px = max(
        min_crop_size_px,
        max(geometry.width_px for geometry in geometry_by_day.values()) + 2 * crop_pad_xy,
    )
    crop_height_px = max(
        min_crop_size_px,
        max(geometry.height_px for geometry in geometry_by_day.values()) + 2 * crop_pad_xy,
    )
    z_offsets = list(range(int(half_window_z), -int(half_window_z) - 1, -1))

    red_planes: list[np.ndarray] = []
    green_planes: list[np.ndarray] = []
    plane_cache: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]] = {}
    metadata_rows: list[dict[str, object]] = []

    for day in days:
        mask_stack = mask_stack_lookup[day]
        geometry = geometry_by_day[day]
        y_start, y_stop, x_start, x_stop = compute_fixed_crop_bounds(
            image_shape_yx=mask_stack.shape[1:],
            y_center=geometry.y_center,
            x_center=geometry.x_center,
            crop_height_px=crop_height_px,
            crop_width_px=crop_width_px,
        )

        for offset in z_offsets:
            z_index = int(np.clip(geometry.z_center + int(offset), 0, mask_stack.shape[0] - 1))
            red_plane = raw_stack_lookup[(day, "red")][z_index, y_start:y_stop, x_start:x_stop]
            green_plane = raw_stack_lookup[(day, "green")][z_index, y_start:y_stop, x_start:x_stop]
            mask_crop = mask_stack[z_index, y_start:y_stop, x_start:x_stop]

            plane_cache[(day, "red", offset)] = (red_plane, mask_crop)
            plane_cache[(day, "green", offset)] = (green_plane, mask_crop)
            red_planes.append(np.asarray(red_plane, dtype=float))
            green_planes.append(np.asarray(green_plane, dtype=float))

            metadata_rows.append(
                {
                    "roi_id": int(roi_id),
                    "day": int(day),
                    "date": date_labels[days.index(day)],
                    "z_offset": int(offset),
                    "z_index": int(z_index),
                    "y_start": int(y_start),
                    "y_stop": int(y_stop),
                    "x_start": int(x_start),
                    "x_stop": int(x_stop),
                    "crop_height_px": int(crop_height_px),
                    "crop_width_px": int(crop_width_px),
                }
            )

    red_vmax = float(np.percentile(np.concatenate([plane.ravel() for plane in red_planes]), 99.5))
    green_vmax = float(np.percentile(np.concatenate([plane.ravel() for plane in green_planes]), 99.5))
    red_vmax = max(red_vmax, 1.0)
    green_vmax = max(green_vmax, 1.0)

    fig = plt.figure(figsize=(max(11.0, len(days) * 2.2 + 1.2), max(11.5, 0.95 * len(z_offsets) + 1.6)), facecolor="white")
    grid = GridSpec(
        nrows=len(z_offsets),
        ncols=len(days),
        figure=fig,
        wspace=0.04,
        hspace=0.04,
        left=0.05,
        right=0.98,
        top=0.91,
        bottom=0.06,
    )

    for row_index, offset in enumerate(z_offsets):
        for col_index, day in enumerate(days):
            axis = fig.add_subplot(grid[row_index, col_index])
            red_plane, mask_crop = plane_cache[(day, "red", offset)]
            green_plane, _ = plane_cache[(day, "green", offset)]
            rgb = build_composite_rgb(
                red_plane=red_plane,
                green_plane=green_plane,
                red_vmax=red_vmax,
                green_vmax=green_vmax,
            )
            axis.imshow(rgb, interpolation="nearest")
            if np.any(mask_crop == roi_id):
                axis.contour(mask_crop == roi_id, levels=[0.5], colors="white", linewidths=0.9)

            if row_index == 0:
                axis.set_title(f"Day {day}\n{date_labels[col_index]}", fontsize=10)
            if col_index == 0:
                axis.set_ylabel(format_z_offset_label(offset), fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])

    fig.suptitle(_format_roi_title(roi_id, roi_summary), fontsize=12)
    fig.text(
        0.5,
        0.02,
        (
            "Raw-space inverse-mask panel. Each tile is a magenta/green composite "
            "from the raw red and green image planes. White contour shows the ROI "
            "mask at that z plane."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return metadata_rows
