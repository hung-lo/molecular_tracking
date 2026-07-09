"""Standalone shared raw-space ROI panel rendering helpers.

This module is intended for notebook-based exploration of a small group of ROIs
in raw image space. It renders one shared field of view centered on the day-0
average centroid of the selected ROIs, while keeping red and green channels in
separate rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from run_registered_roi_pipeline import (
    compute_fixed_crop_bounds,
    compute_raw_space_roi_day_geometry,
)


@dataclass(frozen=True)
class SharedRawSpaceGroupGeometry:
    """Summarize the shared FoV used for a multi-ROI raw-space panel.

    Attributes
    ----------
    z_center : int
        Shared z centroid in pixels.
    y_center : int
        Shared y centroid in pixels.
    x_center : int
        Shared x centroid in pixels.
    width_px : int
        Shared crop width in pixels.
    height_px : int
        Shared crop height in pixels.
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


def build_magenta_green_colormaps() -> tuple[colors.Colormap, colors.Colormap]:
    """Create the display colormaps requested for the raw-space panels.

    Returns
    -------
    tuple[matplotlib.colors.Colormap, matplotlib.colors.Colormap]
        Two colormaps. The first maps black to magenta for the red-channel
        display. The second maps black to green for the green-channel display.
    """

    magenta_dict = {
        "red": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        "green": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "blue": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
    }
    green_dict = {
        "red": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "green": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        "blue": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    }

    return (
        colors.LinearSegmentedColormap("MagentaBlack", magenta_dict),
        colors.LinearSegmentedColormap("GreenBlack", green_dict),
    )


def measure_roi_geometry(mask_stack: np.ndarray, roi_id: int) -> SharedRawSpaceGroupGeometry:
    """Measure the centroid and bounding box for one ROI mask label.

    Parameters
    ----------
    mask_stack : numpy.ndarray
        Integer label stack with shape ``(z, y, x)`` in raw image space.
    roi_id : int
        Positive ROI label to locate.

    Returns
    -------
    SharedRawSpaceGroupGeometry
        Geometry summary for the requested ROI.
    """

    geometry = compute_raw_space_roi_day_geometry(mask_stack=mask_stack, roi_id=roi_id)
    return SharedRawSpaceGroupGeometry(
        z_center=geometry.z_center,
        y_center=geometry.y_center,
        x_center=geometry.x_center,
        width_px=geometry.width_px,
        height_px=geometry.height_px,
    )


def compute_shared_raw_space_group_geometry(
    mask_stack_lookup: dict[int, np.ndarray],
    roi_ids: list[int] | tuple[int, ...],
    center_day: int = 0,
    crop_pad_xy: int = 20,
    min_crop_size_px: int = 48,
) -> SharedRawSpaceGroupGeometry:
    """Measure one shared FoV that can contain several ROIs across days.

    Parameters
    ----------
    mask_stack_lookup : dict[int, numpy.ndarray]
        Mapping from day index to day-specific ROI label stacks with shape
        ``(z, y, x)`` in raw image space.
    roi_ids : sequence of int
        Positive ROI labels to include in the shared FoV.
    center_day : int, default=0
        Day used to define the shared centroid. The centroid is computed from
        the selected ROIs in this day's mask stack.
    crop_pad_xy : int, default=20
        Extra XY padding in pixels added around the union of the selected ROIs.
    min_crop_size_px : int, default=48
        Minimum crop width and height in pixels.

    Returns
    -------
    SharedRawSpaceGroupGeometry
        Shared centroid and crop size in pixel units.
    """

    if center_day not in mask_stack_lookup:
        raise ValueError(f"center_day {center_day} is not available in mask_stack_lookup.")
    if len(roi_ids) == 0:
        raise ValueError("roi_ids must contain at least one ROI.")

    center_mask_stack = mask_stack_lookup[center_day]
    z_centers: list[int] = []
    y_centers: list[int] = []
    x_centers: list[int] = []
    x_coords_all: list[np.ndarray] = []
    y_coords_all: list[np.ndarray] = []

    for roi_id in roi_ids:
        roi_id = int(roi_id)
        geometry = compute_raw_space_roi_day_geometry(mask_stack=center_mask_stack, roi_id=roi_id)
        z_centers.append(int(geometry.z_center))
        y_centers.append(int(geometry.y_center))
        x_centers.append(int(geometry.x_center))
        roi_mask = center_mask_stack == roi_id
        if not np.any(roi_mask):
            raise ValueError(f"ROI {roi_id} was not found in the center-day mask stack.")

    for day, mask_stack in sorted(mask_stack_lookup.items()):
        for roi_id in roi_ids:
            roi_mask = mask_stack == int(roi_id)
            if not np.any(roi_mask):
                continue
            _z_coords, y_coords, x_coords = np.nonzero(roi_mask)
            x_coords_all.append(x_coords)
            y_coords_all.append(y_coords)

    if not x_coords_all or not y_coords_all:
        raise ValueError("No selected ROIs were found in the available mask stacks.")

    x_center = int(np.round(np.mean(x_centers)))
    y_center = int(np.round(np.mean(y_centers)))
    z_center = int(np.round(np.mean(z_centers)))

    x_min = int(min(np.min(coords) for coords in x_coords_all))
    x_max = int(max(np.max(coords) for coords in x_coords_all))
    y_min = int(min(np.min(coords) for coords in y_coords_all))
    y_max = int(max(np.max(coords) for coords in y_coords_all))

    half_width = max(x_center - x_min, x_max - x_center) + int(crop_pad_xy)
    half_height = max(y_center - y_min, y_max - y_center) + int(crop_pad_xy)
    width_px = max(int(min_crop_size_px), 2 * int(half_width) + 1)
    height_px = max(int(min_crop_size_px), 2 * int(half_height) + 1)

    return SharedRawSpaceGroupGeometry(
        z_center=z_center,
        y_center=y_center,
        x_center=x_center,
        width_px=width_px,
        height_px=height_px,
    )


def render_shared_raw_space_group_panel(
    roi_ids: list[int] | tuple[int, ...],
    raw_stack_lookup: dict[tuple[int, str], np.ndarray],
    mask_stack_lookup: dict[int, np.ndarray],
    output_path: Path,
    start_date: str = "20260511",
    half_window_z: int = 1,
    crop_pad_xy: int = 20,
    min_crop_size_px: int = 48,
    contour_colors: dict[int, str] | None = None,
    center_day: int = 0,
) -> list[dict[str, object]]:
    """Render one shared raw-space FoV for several ROIs.

    Parameters
    ----------
    roi_ids : sequence of int
        Positive ROI labels to overlay in the same FoV.
    raw_stack_lookup : dict[tuple[int, str], numpy.ndarray]
        Mapping from ``(day, channel)`` to raw image stacks with shape
        ``(z, y, x)``. Channel names must be ``"red"`` and ``"green"``.
    mask_stack_lookup : dict[int, numpy.ndarray]
        Mapping from day index to inverse-warped ROI label stacks with shape
        ``(z, y, x)``.
    output_path : pathlib.Path
        PNG file to create.
    start_date : str, default="20260511"
        Reference date in ``YYYYMMDD`` format used for day labels.
    half_window_z : int, default=1
        Number of z planes to display above and below the shared centroid
        slice.
    crop_pad_xy : int, default=20
        Extra XY padding in pixels for the shared crop window.
    min_crop_size_px : int, default=48
        Minimum crop width and height in pixels.
    contour_colors : dict[int, str] or None, default=None
        Optional mapping from ROI id to contour color. When omitted, a small
        qualitative palette is used.
    center_day : int, default=0
        Day used to define the shared centroid and crop size.

    Returns
    -------
    list[dict[str, object]]
        One metadata dictionary per rendered day and z-plane combination.
    """

    if half_window_z < 0:
        raise ValueError("half_window_z must be non-negative.")
    if len(roi_ids) == 0:
        raise ValueError("roi_ids must contain at least one ROI.")
    if "red" not in {channel for _day, channel in raw_stack_lookup}:
        raise ValueError("raw_stack_lookup must include red-channel stacks.")
    if "green" not in {channel for _day, channel in raw_stack_lookup}:
        raise ValueError("raw_stack_lookup must include green-channel stacks.")

    days = sorted(mask_stack_lookup)
    if not days:
        raise ValueError("mask_stack_lookup must contain at least one day.")
    if center_day not in mask_stack_lookup:
        raise ValueError(f"center_day {center_day} is not available in mask_stack_lookup.")

    date_labels = make_day_date_labels(np.asarray(days, dtype=int), start_date=start_date)
    shared_geometry = compute_shared_raw_space_group_geometry(
        mask_stack_lookup=mask_stack_lookup,
        roi_ids=roi_ids,
        center_day=center_day,
        crop_pad_xy=crop_pad_xy,
        min_crop_size_px=min_crop_size_px,
    )
    z_offsets = list(range(int(half_window_z), -int(half_window_z) - 1, -1))
    row_definitions = [("red", offset) for offset in z_offsets] + [("green", offset) for offset in z_offsets]

    if contour_colors is None:
        palette = ["#4cc9f0", "#f72585", "#f9c74f", "#90be6d", "#f94144"]
        contour_colors = {
            int(roi_id): palette[index % len(palette)]
            for index, roi_id in enumerate(roi_ids)
        }

    magenta_cmap, green_cmap = build_magenta_green_colormaps()
    crop_lookup: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]] = {}
    red_values: list[np.ndarray] = []
    green_values: list[np.ndarray] = []
    metadata_rows: list[dict[str, object]] = []

    for day in days:
        mask_stack = mask_stack_lookup[day]
        y_start, y_stop, x_start, x_stop = compute_fixed_crop_bounds(
            image_shape_yx=mask_stack.shape[1:],
            y_center=shared_geometry.y_center,
            x_center=shared_geometry.x_center,
            crop_height_px=shared_geometry.height_px,
            crop_width_px=shared_geometry.width_px,
        )
        for channel, offset in row_definitions:
            z_index = shared_geometry.z_center + int(offset)
            if 0 <= z_index < mask_stack.shape[0]:
                image_crop = raw_stack_lookup[(day, channel)][z_index, y_start:y_stop, x_start:x_stop]
                mask_crop = mask_stack[z_index, y_start:y_stop, x_start:x_stop]
            else:
                image_crop = np.zeros((y_stop - y_start, x_stop - x_start), dtype=float)
                mask_crop = np.zeros((y_stop - y_start, x_stop - x_start), dtype=np.uint16)
            crop_lookup[(day, channel, offset)] = (image_crop, mask_crop)
            if channel == "red":
                red_values.append(np.asarray(image_crop, dtype=float))
            else:
                green_values.append(np.asarray(image_crop, dtype=float))

            metadata_rows.append(
                {
                    "day": int(day),
                    "channel": str(channel),
                    "z_offset": int(offset),
                    "z_index": int(z_index),
                    "y_start": int(y_start),
                    "y_stop": int(y_stop),
                    "x_start": int(x_start),
                    "x_stop": int(x_stop),
                    "shared_z_center": int(shared_geometry.z_center),
                    "shared_y_center": int(shared_geometry.y_center),
                    "shared_x_center": int(shared_geometry.x_center),
                    "shared_crop_width_px": int(shared_geometry.width_px),
                    "shared_crop_height_px": int(shared_geometry.height_px),
                }
            )

    red_vmax = float(np.percentile(np.concatenate([value.ravel() for value in red_values]), 99.5))
    green_vmax = float(np.percentile(np.concatenate([value.ravel() for value in green_values]), 99.5))
    red_vmax = max(red_vmax, 1.0)
    green_vmax = max(green_vmax, 1.0)

    figure_width = max(8.0, len(days) * 0.75 + 1.6)
    figure_height = max(11.5, 0.9 * len(row_definitions) + 1.6)
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    grid = GridSpec(
        nrows=len(row_definitions),
        ncols=len(days) + 2,
        figure=fig,
        width_ratios=[1.0] * len(days) + [0.08, 0.08],
        wspace=0.06,
        hspace=0.06,
        left=0.04,
        right=0.98,
        top=0.86,
        bottom=0.08,
    )
    row_to_mappable: dict[int, object] = {}

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
            for roi_id in roi_ids:
                roi_mask = mask_crop == int(roi_id)
                if np.any(roi_mask):
                    axis.contour(
                        roi_mask,
                        levels=[0.5],
                        colors=contour_colors[int(roi_id)],
                        linewidths=1.0,
                    )
            if row_index == 0:
                axis.set_title(f"Day {day}\n{date_label}", fontsize=10)
            if col_index == 0:
                axis.set_ylabel(f"{channel.title()} {format_z_offset_label(offset)}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            row_to_mappable[row_index] = mappable

    red_cax = fig.add_subplot(grid[0:len(z_offsets), len(days)])
    green_cax = fig.add_subplot(grid[len(z_offsets):len(row_definitions), len(days)])
    fig.colorbar(row_to_mappable[0], cax=red_cax, label="Raw red intensity")
    fig.colorbar(row_to_mappable[len(z_offsets)], cax=green_cax, label="Raw green intensity")

    legend_handles = [
        Line2D([0], [0], color=contour_colors[int(roi_id)], lw=2.0, label=f"ROI {int(roi_id)}")
        for roi_id in roi_ids
    ]
    legend_ax = fig.add_axes([0.04, 0.90, 0.92, 0.04])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=len(roi_ids),
        frameon=False,
        fontsize=9,
        handlelength=1.8,
        columnspacing=1.5,
        handletextpad=0.5,
        borderaxespad=0.0,
    )

    roi_id_text = ", ".join(str(int(roi_id)) for roi_id in roi_ids)
    fig.suptitle(
        f"Shared raw-space FoV centered on day-{center_day} average centroid for ROIs {roi_id_text}",
        fontsize=12,
        y=0.985,
    )
    fig.text(
        0.5,
        0.02,
        (
            "Separate red and green rows in raw space. The same FoV is used for all days, "
            "centered on the day-0 average centroid of the selected ROIs."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return metadata_rows
