"""Render raw-space ROI validation panels with inverse-warped ROI masks.

This script validates candidate changing ROIs against the raw image stacks
rather than the registered stacks. For day 0, the SAM ROI mask already lives
in the raw image space. For later days, the script uses inverse-warped ROI
masks so that the ROI contours can be overlaid directly on the raw moving
images.

The output panels are intended as a registration-artifact sanity check for the
top decreasing green/red-ratio ROIs from the current SAM size+shape QC
analysis branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import argparse
import sys
import time

import matplotlib.colors as colors
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import tifffile

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
from roi_log_ratio_analysis import (
    build_inverse_warped_mask_lookup,
    build_raw_image_lookup,
    select_roi_neighbor_z_slices,
)


@dataclass(frozen=True)
class RoiDayGeometry:
    """Describe one ROI location inside one day-specific mask.

    Attributes
    ----------
    z_center : int
        Integer z index in pixels that represents the centroid-centered slice
        used for the middle display row.
    y_center : int
        Integer y coordinate in pixels for the centroid of the ROI footprint in
        the day-specific raw-space mask.
    x_center : int
        Integer x coordinate in pixels for the centroid of the ROI footprint in
        the day-specific raw-space mask.
    width_px : int
        Width of the ROI bounding box in pixels before display padding.
    height_px : int
        Height of the ROI bounding box in pixels before display padding.
    z_minus1 : int
        Integer z index in pixels for the row above the centroid slice.
    z_plus1 : int
        Integer z index in pixels for the row below the centroid slice.
    """

    z_center: int
    y_center: int
    x_center: int
    width_px: int
    height_px: int
    z_minus1: int
    z_plus1: int


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


def build_magenta_green_colormaps() -> tuple[colors.Colormap, colors.Colormap]:
    """Create the display colormaps requested by the user.

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


def compute_roi_day_geometry(
    mask_stack: np.ndarray,
    roi_id: int,
) -> RoiDayGeometry:
    """Measure one ROI inside one raw-space day-specific label stack.

    Parameters
    ----------
    mask_stack : numpy.ndarray
        Integer ROI label image with shape ``(z, y, x)`` in raw image space.
    roi_id : int
        Positive ROI label to locate in ``mask_stack``.

    Returns
    -------
    RoiDayGeometry
        Centroid and bounding-box measurements for the requested ROI in pixel
        coordinates.
    """

    roi_mask = mask_stack == roi_id
    if not np.any(roi_mask):
        raise ValueError(f"ROI {roi_id} was not found in the mask stack.")

    z_coords, y_coords, x_coords = np.nonzero(roi_mask)
    z_lookup = select_roi_neighbor_z_slices(mask_stack=mask_stack, roi_id=roi_id)

    return RoiDayGeometry(
        z_center=z_lookup["z_center"],
        y_center=int(np.round(np.mean(y_coords))),
        x_center=int(np.round(np.mean(x_coords))),
        width_px=int(x_coords.max() - x_coords.min() + 1),
        height_px=int(y_coords.max() - y_coords.min() + 1),
        z_minus1=z_lookup["z_minus1"],
        z_plus1=z_lookup["z_plus1"],
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


def collect_daywise_roi_geometry(
    mask_lookup: dict[int, Path],
    roi_id: int,
) -> dict[int, RoiDayGeometry]:
    """Load every day-specific mask and measure one ROI.

    Parameters
    ----------
    mask_lookup : dict[int, pathlib.Path]
        Mapping from day index to the ROI mask TIFF that lives in that day's
        raw image space.
    roi_id : int
        Positive ROI label to measure across all days.

    Returns
    -------
    dict[int, RoiDayGeometry]
        Day-indexed geometry measurements for the requested ROI.
    """

    geometry_by_day: dict[int, RoiDayGeometry] = {}
    for day, mask_path in sorted(mask_lookup.items()):
        mask_stack = tifffile.imread(mask_path)
        geometry_by_day[day] = compute_roi_day_geometry(mask_stack=mask_stack, roi_id=roi_id)
    return geometry_by_day


def format_roi_title(roi_row: pd.Series) -> str:
    """Format a concise title string for one ROI panel.

    Parameters
    ----------
    roi_row : pandas.Series
        One row from the top-ROI summary table. The series is expected to
        contain ROI identity, ranking, size, QC, and trajectory summary
        columns.

    Returns
    -------
    str
        Human-readable title summarizing the ROI rank and QC metadata.
    """

    circularity = roi_row.get("circularity_x", roi_row.get("circularity_y", np.nan))
    solidity = roi_row.get("solidity_x", roi_row.get("solidity_y", np.nan))
    axis_ratio = roi_row.get("axis_ratio_x", roi_row.get("axis_ratio_y", np.nan))
    return (
        f"ROI {int(roi_row['roi_id'])} | rank {int(roi_row['selection_rank']):02d} | "
        f"min dlog2(G/R) {roi_row['min_delta_log2_green_over_red']:.3f}\n"
        f"area {roi_row['proj_area_px']:.0f} px | eq diam {roi_row['eq_diam_um']:.2f} um | "
        f"red CV {roi_row['red_cv']:.3f} | circ {circularity:.2f} | "
        f"sol {solidity:.2f} | ar {axis_ratio:.2f}"
    )


def render_raw_space_roi_panel(
    roi_row: pd.Series,
    raw_lookup: dict[tuple[int, str], Path],
    mask_lookup: dict[int, Path],
    output_path: Path,
    crop_pad_xy: int = 20,
    min_crop_size_px: int = 48,
) -> list[dict[str, object]]:
    """Render one six-row raw-space validation panel.

    Parameters
    ----------
    roi_row : pandas.Series
        One row from the top-ROI summary table for the ROI to render.
    raw_lookup : dict[tuple[int, str], pathlib.Path]
        Mapping from ``(day, channel)`` to the raw TIFF file for that day and
        channel. Supported channel names are ``"red"`` and ``"green"``.
    mask_lookup : dict[int, pathlib.Path]
        Mapping from day index to the ROI mask TIFF in that day's raw image
        space.
    output_path : pathlib.Path
        Final PNG path for the rendered panel.
    crop_pad_xy : int, default=20
        Padding added around the maximum ROI width and height when building the
        fixed-size display crop, in pixels.
    min_crop_size_px : int, default=48
        Lower bound on the displayed crop width and height in pixels.

    Returns
    -------
    list[dict[str, object]]
        One metadata dictionary per day describing the crop bounds, z indices,
        and input files used for the panel.
    """

    roi_id = int(roi_row["roi_id"])
    geometry_by_day = collect_daywise_roi_geometry(mask_lookup=mask_lookup, roi_id=roi_id)
    days = sorted(geometry_by_day)

    raw_cache: dict[tuple[int, str], np.ndarray] = {}
    mask_cache: dict[int, np.ndarray] = {}
    for day in days:
        mask_cache[day] = tifffile.imread(mask_lookup[day])
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

    row_definitions = [
        ("red", "z_plus1", "Red z+1"),
        ("red", "z_center", "Red z"),
        ("red", "z_minus1", "Red z-1"),
        ("green", "z_plus1", "Green z+1"),
        ("green", "z_center", "Green z"),
        ("green", "z_minus1", "Green z-1"),
    ]
    magenta_cmap, green_cmap = build_magenta_green_colormaps()

    red_values: list[np.ndarray] = []
    green_values: list[np.ndarray] = []
    crop_lookup: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]] = {}
    metadata_rows: list[dict[str, object]] = []

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
        z_lookup = {
            "z_minus1": geometry.z_minus1,
            "z_center": geometry.z_center,
            "z_plus1": geometry.z_plus1,
        }
        metadata_rows.append(
            {
                "roi_id": roi_id,
                "selection_rank": int(roi_row["selection_rank"]),
                "day": day,
                "raw_red_path": str(raw_lookup[(day, "red")]),
                "raw_green_path": str(raw_lookup[(day, "green")]),
                "mask_path": str(mask_lookup[day]),
                "z_minus1": geometry.z_minus1,
                "z_center": geometry.z_center,
                "z_plus1": geometry.z_plus1,
                "y_center": geometry.y_center,
                "x_center": geometry.x_center,
                "y_start": y_start,
                "y_stop": y_stop,
                "x_start": x_start,
                "x_stop": x_stop,
                "crop_height_px": crop_height_px,
                "crop_width_px": crop_width_px,
            }
        )

        for channel, z_key, _row_label in row_definitions:
            z_index = z_lookup[z_key]
            image_crop = raw_cache[(day, channel)][z_index, y_start:y_stop, x_start:x_stop]
            mask_crop = mask_stack[z_index, y_start:y_stop, x_start:x_stop]
            crop_lookup[(day, channel, z_key)] = (
                image_crop,
                mask_crop,
                (y_start, y_stop, x_start, x_stop),
            )
            if channel == "red":
                red_values.append(image_crop)
            else:
                green_values.append(image_crop)

    red_vmax = float(np.percentile(np.concatenate([value.ravel() for value in red_values]), 99.5))
    green_vmax = float(
        np.percentile(np.concatenate([value.ravel() for value in green_values]), 99.5)
    )
    if red_vmax <= 0:
        red_vmax = 1.0
    if green_vmax <= 0:
        green_vmax = 1.0

    figure_width = max(11.0, len(days) * 2.1 + 1.2)
    figure_height = 11.5
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")
    grid = GridSpec(
        nrows=6,
        ncols=len(days) + 2,
        figure=fig,
        width_ratios=[1.0] * len(days) + [0.06, 0.06],
        wspace=0.06,
        hspace=0.08,
        left=0.05,
        right=0.96,
        top=0.90,
        bottom=0.07,
    )

    row_to_mappable: dict[int, object] = {}
    for row_index, (channel, z_key, row_label) in enumerate(row_definitions):
        cmap = magenta_cmap if channel == "red" else green_cmap
        vmax = red_vmax if channel == "red" else green_vmax

        for col_index, day in enumerate(days):
            axis = fig.add_subplot(grid[row_index, col_index])
            image_crop, mask_crop, _bounds = crop_lookup[(day, channel, z_key)]
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
                day_label = raw_lookup[(day, channel)].name[:8]
                axis.set_title(f"Day {day}\n{day_label}", fontsize=10)
            if col_index == 0:
                axis.set_ylabel(row_label, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
            row_to_mappable[row_index] = mappable

    red_cax = fig.add_subplot(grid[0:3, len(days)])
    green_cax = fig.add_subplot(grid[3:6, len(days)])
    fig.colorbar(row_to_mappable[0], cax=red_cax, label="Raw red intensity")
    fig.colorbar(row_to_mappable[3], cax=green_cax, label="Raw green intensity")

    fig.suptitle(format_roi_title(roi_row), fontsize=12)
    fig.text(
        0.5,
        0.02,
        (
            "Raw image slices in each day's native space. White contour shows the "
            "day-specific ROI mask: day 0 uses the SAM mask, later days use the "
            "inverse-warped ROI mask."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    return metadata_rows


def write_summary_markdown(
    output_dir: Path,
    panel_dir: Path,
    selection_table_path: Path,
    metadata_path: Path,
    n_rois: int,
) -> None:
    """Write a short human-readable summary for the analysis run.

    Parameters
    ----------
    output_dir : pathlib.Path
        Root directory for the current analysis run.
    panel_dir : pathlib.Path
        Directory that contains the per-ROI PNG panels.
    selection_table_path : pathlib.Path
        CSV file that defines the ranked ROIs used for this run.
    metadata_path : pathlib.Path
        CSV file containing crop and z-slice metadata for each ROI/day.
    n_rois : int
        Number of ROI panels written in this run.
    """

    summary_path = output_dir / "SUMMARY.md"
    summary_lines = [
        "# Raw-Space Inverse-Mask Validation",
        "",
        "Goal: inspect the top decreasing green/red-ratio ROIs in raw image space",
        "to check whether the apparent green drop survives outside the registered",
        "image stacks.",
        "",
        "Inputs:",
        f"- Top ROI table: `{selection_table_path}`",
        "- Day 0 mask: `mean_image_merge_cp_masks_SAM.tif`",
        "- Days 1-4 masks: `20260512-20260515_R_ROI_mask_SyN_inversed.tif`",
        "- Raw images: `20260511-20260515_[RG].tif`",
        "",
        "Outputs:",
        f"- ROI panels: `{panel_dir}`",
        f"- Per-day panel metadata: `{metadata_path}`",
        f"- Number of ROI panels: `{n_rois}`",
        "",
        "Notes:",
        "- The ROI contours still originate from the fixed-space SAM segmentation.",
        "- This is therefore a stronger interpolation check, not a fully",
        "  registration-independent segmentation analysis.",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


def write_run_log(
    output_dir: Path,
    selection_table_path: Path,
    metadata_path: Path,
    n_panels: int,
    total_duration_seconds: float,
) -> None:
    """Write a compact text run log for the raw-space validation export."""

    log_lines = [
        f"run_timestamp={datetime.now().isoformat()}",
        f"selection_table_path={selection_table_path}",
        f"metadata_path={metadata_path}",
        f"n_panels={int(n_panels)}",
        f"total_duration_seconds={float(total_duration_seconds):.3f}",
        f"total_duration_hms={format_duration_seconds(total_duration_seconds)}",
    ]
    (output_dir / "run_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the raw-space validation script.

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
    """Run the raw-space validation export for the current top decreasing ROIs.

    This function reads the current SAM size+shape-filtered top-30 decreasing
    ROI table, renders one six-row validation panel per ROI, and saves the
    results into a timestamped analysis-run directory.
    """

    run_start_seconds = time.perf_counter()
    args = parse_args()
    log_message(run_start_seconds, f"Starting raw-space inverse-mask validation | dataset={args.dataset}")
    base_dir = resolve_dataset_dir(args.dataset)
    shape_qc_dir = get_shape_qc_analysis_dir(args.dataset)
    selection_table_path = (
        shape_qc_dir
        / "top30_decreasing_rois_size_and_shape_filtered.csv"
    )
    output_root = shape_qc_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"raw_space_inverse_mask_validation_top30_decreasing_{timestamp}"
    panel_dir = output_dir / "individual_roi_panels"

    output_dir.mkdir(parents=True, exist_ok=False)
    panel_dir.mkdir(parents=True, exist_ok=False)
    log_message(run_start_seconds, f"Output directory: {output_dir}")

    log_message(run_start_seconds, f"Loading ranked ROI selection table from {selection_table_path}")
    roi_table = pd.read_csv(selection_table_path).sort_values("selection_rank").reset_index(drop=True)
    raw_lookup = build_raw_image_lookup(base_dir)
    mask_lookup = build_inverse_warped_mask_lookup(base_dir)
    log_message(run_start_seconds, f"Loaded {len(roi_table)} ranked ROIs; rendering raw-space validation panels")

    all_metadata_rows: list[dict[str, object]] = []
    for roi_index, (_, roi_row) in enumerate(roi_table.iterrows(), start=1):
        log_message(
            run_start_seconds,
            f"Rendering ROI {roi_index}/{len(roi_table)} | roi_id={int(roi_row['roi_id'])} | rank={int(roi_row['selection_rank']):02d}",
        )
        output_path = panel_dir / (
            f"rank_{int(roi_row['selection_rank']):02d}_roi_{int(roi_row['roi_id'])}.png"
        )
        all_metadata_rows.extend(
            render_raw_space_roi_panel(
                roi_row=roi_row,
                raw_lookup=raw_lookup,
                mask_lookup=mask_lookup,
                output_path=output_path,
            )
        )

    metadata_path = output_dir / "raw_space_panel_metadata.csv"
    pd.DataFrame(all_metadata_rows).to_csv(metadata_path, index=False)

    script_copy_path = output_dir / Path(__file__).name
    shutil.copy2(__file__, script_copy_path)
    write_summary_markdown(
        output_dir=output_dir,
        panel_dir=panel_dir,
        selection_table_path=selection_table_path,
        metadata_path=metadata_path,
        n_rois=len(roi_table),
    )

    total_duration_seconds = time.perf_counter() - run_start_seconds
    write_run_log(
        output_dir=output_dir,
        selection_table_path=selection_table_path,
        metadata_path=metadata_path,
        n_panels=len(roi_table),
        total_duration_seconds=total_duration_seconds,
    )
    print(f"[{format_duration_seconds(total_duration_seconds)}] Completed raw-space inverse-mask validation", flush=True)
    print(f"output_dir={output_dir}")
    print(f"panel_dir={panel_dir}")
    print(f"metadata_path={metadata_path}")
    print(f"n_panels={len(roi_table)}")
    print(f"total_duration={format_duration_seconds(total_duration_seconds)}")


if __name__ == "__main__":
    main()
