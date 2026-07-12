"""Generate styled QC plots for multi-week ROI matcher outputs.

This module selects representative matched ROI tracks across confidence levels
and renders compact contact sheets in the common registered-space coordinates
used by :mod:`roi_matcher`.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import tifffile

from roi_matcher import MatchParams, extract_roi_records, match_roi_masks


@dataclass(frozen=True)
class PlotStyle:
    """Store display settings for styled QC contact sheets.

    Parameters
    ----------
    figure_background_color : str
        Matplotlib color string used for the overall figure background.
    panel_background_color : str
        Matplotlib color string used for panel pixels classified as background.
    text_color : str
        Matplotlib color string used for titles and labels.
    neighbor_fill_color : str
        Matplotlib color string used to fill neighboring ROIs.
    neighbor_contour_color : str
        Matplotlib color string used for thin contours around neighboring ROIs.
    matched_fill_color : str
        Matplotlib color string used to fill the matched ROI.
    matched_contour_color : str
        Matplotlib color string used for the matched ROI contour.
    centroid_marker_color : str
        Matplotlib color string used for the matched ROI centroid marker.
    """

    figure_background_color: str = "#ffffff"
    panel_background_color: str = "#000000"
    text_color: str = "#111111"
    neighbor_fill_color: str = "#a9a9a9"
    neighbor_contour_color: str = "#d0d0d0"
    matched_fill_color: str = "#ff2d2d"
    matched_contour_color: str = "#ffffff"
    centroid_marker_color: str = "#000000"


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


def merge_track_and_qc_tables(tracks_table: pd.DataFrame, qc_table: pd.DataFrame) -> pd.DataFrame:
    """Merge matcher track and QC tables on cluster identity.

    Parameters
    ----------
    tracks_table : pandas.DataFrame
        Track table returned by :func:`roi_matcher.match_roi_masks`.
    qc_table : pandas.DataFrame
        QC table aligned to ``tracks_table``.

    Returns
    -------
    pandas.DataFrame
        Combined table with one row per track and both tracking and QC columns.
    """

    qc_columns = [column for column in qc_table.columns if column not in tracks_table.columns or column == "cluster_id"]
    qc_subset = qc_table[qc_columns].copy()
    return tracks_table.merge(qc_subset, on="cluster_id", how="left")


def select_example_tracks(
    tracks_table: pd.DataFrame,
    qc_table: pd.DataFrame,
    examples_per_group: int = 6,
) -> dict[str, pd.DataFrame]:
    """Select representative tracks across confidence and QC categories.

    Parameters
    ----------
    tracks_table : pandas.DataFrame
        Track table containing at least ``cluster_id``, ``n_days_present``,
        ``mean_confidence``, and ``min_confidence`` columns.
    qc_table : pandas.DataFrame
        QC table containing at least ``cluster_id`` and ``needs_review``.
    examples_per_group : int, default=6
        Maximum number of tracks returned per example group.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping with ``"high"``, ``"medium"``, ``"low"``, and ``"review"``
        keys. Each value is a table of selected example tracks.
    """

    merged = merge_track_and_qc_tables(tracks_table, qc_table)
    eligible = merged.loc[merged["n_days_present"].fillna(0).astype(int) >= 2].copy()
    if len(eligible) == 0:
        eligible = merged.copy()

    clean = eligible.loc[~eligible["needs_review"].fillna(False)].copy()
    review = eligible.loc[eligible["needs_review"].fillna(False)].copy()

    high = clean.sort_values(["mean_confidence", "min_confidence"], ascending=[False, False]).head(examples_per_group)

    if len(clean) > 0:
        median_confidence = float(clean["mean_confidence"].median())
        medium = clean.assign(distance_to_median=np.abs(clean["mean_confidence"] - median_confidence))
        medium = medium.sort_values(["distance_to_median", "mean_confidence"], ascending=[True, False]).head(examples_per_group)
        medium = medium.drop(columns=["distance_to_median"])
    else:
        medium = eligible.head(examples_per_group)

    low = eligible.sort_values(["min_confidence", "mean_confidence"], ascending=[True, True]).head(examples_per_group)
    review = review.sort_values(["min_confidence", "mean_confidence"], ascending=[True, True]).head(examples_per_group)

    return {"high": high, "medium": medium, "low": low, "review": review}


def compute_track_crop_bounds(
    track_row: pd.Series,
    day_names: list[str],
    records_by_day_label: dict[tuple[str, int], object],
    image_shape_yx: tuple[int, int],
    pad_xy: int = 8,
    min_crop_size: int = 64,
) -> tuple[int, int, int, int]:
    """Compute a shared XY crop window for one matched track.

    Parameters
    ----------
    track_row : pandas.Series
        One track row containing ``"{day}_roi"`` columns with ROI labels or
        missing values.
    day_names : list[str]
        Ordered day names corresponding to the track columns.
    records_by_day_label : dict[tuple[str, int], object]
        Mapping from ``(day_name, roi_label)`` to ROIRecord objects whose
        bounding boxes are expressed in the common registered-space coordinates.
    image_shape_yx : tuple[int, int]
        Full image shape ``(height, width)`` in pixels.
    pad_xy : int, default=8
        Padding added to the union ROI bounding box in pixels.
    min_crop_size : int, default=64
        Minimum crop width and height in pixels.

    Returns
    -------
    tuple[int, int, int, int]
        Crop bounds ``(y0, y1, x0, x1)`` using half-open indexing in pixels.
    """

    y_mins: list[int] = []
    y_maxs: list[int] = []
    x_mins: list[int] = []
    x_maxs: list[int] = []
    for day_name in day_names:
        label_value = track_row.get(f"{day_name}_roi", pd.NA)
        if pd.isna(label_value):
            continue
        record = records_by_day_label[(day_name, int(label_value))]
        y_bounds = record.bbox_zyx[1]
        x_bounds = record.bbox_zyx[2]
        y_mins.append(int(y_bounds[0]))
        y_maxs.append(int(y_bounds[1]))
        x_mins.append(int(x_bounds[0]))
        x_maxs.append(int(x_bounds[1]))
    if not y_mins:
        raise ValueError("track_row does not contain any present ROI labels")

    height, width = image_shape_yx
    y0 = max(0, min(y_mins) - pad_xy)
    y1 = min(height, max(y_maxs) + pad_xy + 1)
    x0 = max(0, min(x_mins) - pad_xy)
    x1 = min(width, max(x_maxs) + pad_xy + 1)

    crop_height = y1 - y0
    crop_width = x1 - x0
    if crop_height < min_crop_size:
        missing = min_crop_size - crop_height
        extend_before = missing // 2
        extend_after = missing - extend_before
        y0 = max(0, y0 - extend_before)
        y1 = min(height, y1 + extend_after)
        if y1 - y0 < min_crop_size:
            y0 = max(0, y1 - min_crop_size)
            y1 = min(height, y0 + min_crop_size)
    if crop_width < min_crop_size:
        missing = min_crop_size - crop_width
        extend_before = missing // 2
        extend_after = missing - extend_before
        x0 = max(0, x0 - extend_before)
        x1 = min(width, x1 + extend_after)
        if x1 - x0 < min_crop_size:
            x0 = max(0, x1 - min_crop_size)
            x1 = min(width, x0 + min_crop_size)
    return int(y0), int(y1), int(x0), int(x1)


def extract_crop_roi_masks(
    mask_stack: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> dict[int, np.ndarray]:
    """Project all ROI labels present in one crop into 2D binary masks.

    Parameters
    ----------
    mask_stack : numpy.ndarray
        Integer label stack with shape ``(z, y, x)`` in registered space.
    y0 : int
        Inclusive crop start along the y axis in pixels.
    y1 : int
        Exclusive crop end along the y axis in pixels.
    x0 : int
        Inclusive crop start along the x axis in pixels.
    x1 : int
        Exclusive crop end along the x axis in pixels.

    Returns
    -------
    dict[int, numpy.ndarray]
        Mapping from positive ROI label to a boolean max-projection mask with
        shape ``(y1 - y0, x1 - x0)`` in crop coordinates.
    """

    crop_stack = mask_stack[:, y0:y1, x0:x1]
    labels = np.unique(crop_stack)
    labels = labels[labels > 0]
    roi_masks: dict[int, np.ndarray] = {}
    for label in labels:
        projection = (crop_stack == int(label)).max(axis=0)
        if np.any(projection):
            roi_masks[int(label)] = projection.astype(bool)
    return roi_masks


def extract_plane_roi_masks(
    mask_stack: np.ndarray,
    z_index: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> dict[int, np.ndarray]:
    """Extract all ROI masks that intersect one displayed z plane and crop.

    Parameters
    ----------
    mask_stack : numpy.ndarray
        Integer label stack with shape ``(z, y, x)`` in registered space.
    z_index : int
        Z plane index displayed in the panel, expressed in planes.
    y0 : int
        Inclusive crop start along the y axis in pixels.
    y1 : int
        Exclusive crop end along the y axis in pixels.
    x0 : int
        Inclusive crop start along the x axis in pixels.
    x1 : int
        Exclusive crop end along the x axis in pixels.

    Returns
    -------
    dict[int, numpy.ndarray]
        Mapping from positive ROI label to a boolean 2D mask with shape
        ``(y1 - y0, x1 - x0)`` for labels present on the chosen plane.
    """

    z_clipped = int(np.clip(z_index, 0, mask_stack.shape[0] - 1))
    plane = mask_stack[z_clipped, y0:y1, x0:x1]
    labels = np.unique(plane)
    labels = labels[labels > 0]
    roi_masks: dict[int, np.ndarray] = {}
    for label in labels:
        mask_2d = plane == int(label)
        if np.any(mask_2d):
            roi_masks[int(label)] = mask_2d.astype(bool)
    return roi_masks


def split_matched_and_neighbor_masks(
    roi_masks: dict[int, np.ndarray],
    matched_label: int | None,
) -> tuple[np.ndarray | None, dict[int, np.ndarray]]:
    """Separate the selected matched ROI from neighboring ROI crop masks.

    Parameters
    ----------
    roi_masks : dict[int, numpy.ndarray]
        Mapping from ROI label to a 2D boolean crop mask.
    matched_label : int or None
        ROI label to highlight. When ``None`` or absent, no matched ROI mask is
        returned.

    Returns
    -------
    tuple[numpy.ndarray or None, dict[int, numpy.ndarray]]
        ``(matched_mask, neighbor_masks)`` where ``matched_mask`` is a boolean
        2D mask for the selected ROI or ``None`` if unavailable, and
        ``neighbor_masks`` contains the remaining ROI masks.
    """

    neighbors = dict(roi_masks)
    if matched_label is None:
        return None, neighbors
    matched_mask = neighbors.pop(int(matched_label), None)
    return matched_mask, neighbors


def choose_display_plane(
    track_row: pd.Series,
    day_name: str,
    day_names: list[str],
    records_by_day_label: dict[tuple[str, int], object],
    max_z_index: int,
) -> int:
    """Choose the z plane displayed for one day panel of one track.

    Parameters
    ----------
    track_row : pandas.Series
        Track row containing ``"{day}_roi"`` columns with ROI labels or missing
        values.
    day_name : str
        Day corresponding to the panel being drawn.
    day_names : list[str]
        Ordered day names available in the track row.
    records_by_day_label : dict[tuple[str, int], object]
        Mapping from ``(day_name, roi_label)`` to ROIRecord objects that contain
        a ``center_plane`` integer in z-plane units.
    max_z_index : int
        Maximum valid z index for clipping.

    Returns
    -------
    int
        Chosen z plane index in planes. If the current day lacks a matched ROI,
        the median center plane across present matched weeks is used for context.
    """

    label_value = track_row.get(f"{day_name}_roi", pd.NA)
    if pd.notna(label_value):
        return int(np.clip(records_by_day_label[(day_name, int(label_value))].center_plane, 0, max_z_index))

    center_planes = []
    for other_day_name in day_names:
        other_label_value = track_row.get(f"{other_day_name}_roi", pd.NA)
        if pd.isna(other_label_value):
            continue
        center_planes.append(int(records_by_day_label[(other_day_name, int(other_label_value))].center_plane))
    if not center_planes:
        return 0
    median_plane = int(round(float(np.median(center_planes))))
    return int(np.clip(median_plane, 0, max_z_index))


def build_single_plane_display_layers(
    matched_mask: np.ndarray | None,
    neighbor_masks: dict[int, np.ndarray],
) -> np.ndarray:
    """Build a categorical display image for one single-plane crop.

    Parameters
    ----------
    matched_mask : numpy.ndarray or None
        Boolean mask with shape ``(y, x)`` for the matched ROI on the displayed
        z plane. When ``None``, no matched ROI is filled.
    neighbor_masks : dict[int, numpy.ndarray]
        Mapping from neighbor label to boolean masks with common shape
        ``(y, x)`` on the displayed z plane.

    Returns
    -------
    numpy.ndarray
        Unsigned integer display layer image with shape ``(y, x)`` where
        ``0=background``, ``1=neighbor ROI``, and ``2=matched ROI``.
    """

    if matched_mask is not None:
        shape_yx = matched_mask.shape
    elif neighbor_masks:
        shape_yx = next(iter(neighbor_masks.values())).shape
    else:
        raise ValueError("At least one mask is required to build display layers")

    display = np.zeros(shape_yx, dtype=np.uint8)
    for neighbor_mask in neighbor_masks.values():
        display[neighbor_mask.astype(bool)] = 1
    if matched_mask is not None:
        display[matched_mask.astype(bool)] = 2
    return display


def format_panel_title(day_name: str, matched_label: int | None, z_index: int) -> str:
    """Format the two-line title for one week panel.

    Parameters
    ----------
    day_name : str
        Session name displayed on the first line.
    matched_label : int or None
        ROI label shown on the second line, or ``None`` for missing weeks.
    z_index : int
        Displayed z-plane index in planes.

    Returns
    -------
    str
        Two-line panel title.
    """

    if matched_label is None:
        return f"{day_name}\nmissing | z={int(z_index)}"
    return f"{day_name}\nROI {int(matched_label)} | z={int(z_index)}"


def format_cluster_header(track_row: pd.Series) -> str:
    """Format the centered row header for one matched cluster.

    Parameters
    ----------
    track_row : pandas.Series
        Track row containing ``cluster_id``, ``n_days_present``, and
        ``mean_confidence`` fields.

    Returns
    -------
    str
        Row-level cluster summary text.
    """

    return (
        f"Cluster {int(track_row['cluster_id'])} "
        f"(n_days={int(track_row['n_days_present'])}, "
        f"confidence={float(track_row['mean_confidence']):.3f})"
    )


def build_track_plot_metadata(track_row: pd.Series, qc_row: pd.Series) -> dict[str, object]:
    """Summarize one track for QC annotation.

    Parameters
    ----------
    track_row : pandas.Series
        One track row containing cluster and confidence columns.
    qc_row : pandas.Series
        One QC row containing review flags aligned to ``track_row``.

    Returns
    -------
    dict[str, object]
        Dictionary with short plot-friendly labels and flag summaries.
    """

    flag_names = []
    if bool(qc_row.get("low_confidence", False)):
        flag_names.append("low_confidence")
    if bool(qc_row.get("gap_bridge_qc", False)):
        flag_names.append("gap_bridge")
    if bool(qc_row.get("edge_qc", False)):
        flag_names.append("edge")
    if bool(qc_row.get("needs_review", False)) and not flag_names:
        flag_names.append("review")
    return {
        "cluster_id": int(track_row["cluster_id"]),
        "n_days_present": int(track_row["n_days_present"]),
        "confidence_label": f"{float(track_row['mean_confidence']):.2f} / {float(track_row['min_confidence']):.2f}",
        "flag_text": ", ".join(flag_names) if flag_names else "clean",
    }


def render_example_group(
    selected_tracks: pd.DataFrame,
    day_names: list[str],
    mask_stacks: list[np.ndarray],
    records_by_day_label: dict[tuple[str, int], object],
    output_path: Path,
    group_name: str,
    style: PlotStyle,
    pad_xy: int = 10,
    min_crop_size: int = 72,
) -> Path | None:
    """Render one multi-row contact sheet for a confidence/QC group.

    Parameters
    ----------
    selected_tracks : pandas.DataFrame
        Subset of merged track/QC rows to visualize.
    day_names : list[str]
        Ordered day names corresponding to ``mask_stacks``.
    mask_stacks : list[numpy.ndarray]
        Registered label stacks with common shape ``(z, y, x)``.
    records_by_day_label : dict[tuple[str, int], object]
        ROIRecord lookup for crop-bound computation.
    output_path : pathlib.Path
        Output PNG path.
    group_name : str
        Example-group name used for the page title.
    style : PlotStyle
        Rendering settings.
    pad_xy : int, default=10
        Additional crop padding in pixels.
    min_crop_size : int, default=72
        Minimum crop size in pixels.

    Returns
    -------
    pathlib.Path or None
        Saved output path when at least one example is rendered, otherwise
        ``None``.
    """

    if selected_tracks is None or len(selected_tracks) == 0:
        return None

    display_cmap = ListedColormap(
        [
            style.panel_background_color,
            style.neighbor_fill_color,
            style.matched_fill_color,
        ]
    )

    n_rows = len(selected_tracks)
    n_cols = len(day_names)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.1 * n_cols, 3.6 * n_rows),
        squeeze=False,
        facecolor=style.figure_background_color,
    )
    image_shape_yx = mask_stacks[0].shape[-2:]
    max_z_index = mask_stacks[0].shape[0] - 1
    row_headers: list[str] = []

    for row_index, (_, track_row) in enumerate(selected_tracks.iterrows()):
        row_headers.append(format_cluster_header(track_row))
        y0, y1, x0, x1 = compute_track_crop_bounds(
            track_row=track_row,
            day_names=day_names,
            records_by_day_label=records_by_day_label,
            image_shape_yx=image_shape_yx,
            pad_xy=pad_xy,
            min_crop_size=min_crop_size,
        )
        for col_index, day_name in enumerate(day_names):
            ax = axes[row_index, col_index]
            ax.set_facecolor(style.panel_background_color)
            z_index = choose_display_plane(
                track_row=track_row,
                day_name=day_name,
                day_names=day_names,
                records_by_day_label=records_by_day_label,
                max_z_index=max_z_index,
            )
            roi_masks = extract_plane_roi_masks(mask_stacks[col_index], z_index=z_index, y0=y0, y1=y1, x0=x0, x1=x1)
            label_value = track_row.get(f"{day_name}_roi", pd.NA)
            matched_label = None if pd.isna(label_value) else int(label_value)
            matched_mask, neighbor_masks = split_matched_and_neighbor_masks(roi_masks, matched_label=matched_label)
            if matched_mask is None and not neighbor_masks:
                display_layers = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            else:
                display_layers = build_single_plane_display_layers(matched_mask=matched_mask, neighbor_masks=neighbor_masks)
            ax.imshow(display_layers, cmap=display_cmap, vmin=0, vmax=2, interpolation="nearest")

            for neighbor_mask in neighbor_masks.values():
                if np.any(neighbor_mask):
                    ax.contour(
                        neighbor_mask.astype(float),
                        levels=[0.5],
                        colors=[style.neighbor_contour_color],
                        linewidths=0.7,
                    )

            if matched_mask is not None and np.any(matched_mask):
                ax.contour(
                    matched_mask.astype(float),
                    levels=[0.5],
                    colors=[style.matched_contour_color],
                    linewidths=1.2,
                )
                matched_record = records_by_day_label[(day_name, matched_label)]
                centroid_x = float(matched_record.centroid_x - x0)
                centroid_y = float(matched_record.centroid_y - y0)
                ax.plot(
                    centroid_x,
                    centroid_y,
                    marker="s",
                    markersize=4.8,
                    markerfacecolor="none",
                    markeredgecolor=style.centroid_marker_color,
                    markeredgewidth=1.0,
                )

            ax.set_title(
                format_panel_title(day_name=day_name, matched_label=matched_label, z_index=z_index),
                color=style.text_color,
                fontsize=11,
                pad=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#444444")
                spine.set_linewidth(0.9)

    fig.subplots_adjust(left=0.04, right=0.985, bottom=0.035, top=0.94, hspace=0.5, wspace=0.18)
    for row_index, header in enumerate(row_headers):
        left_bbox = axes[row_index, 0].get_position()
        right_bbox = axes[row_index, -1].get_position()
        x_center = 0.5 * (left_bbox.x0 + right_bbox.x1)
        y_text = min(0.97, left_bbox.y1 + 0.038)
        fig.text(
            x_center,
            y_text,
            header,
            ha="center",
            va="bottom",
            color=style.text_color,
            fontsize=12,
        )

    fig.suptitle(
        f"{group_name.title()} Confidence ROI Match Examples",
        color=style.text_color,
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(output_path, dpi=180, facecolor=style.figure_background_color)
    plt.close(fig)
    return output_path


def generate_qc_example_plots(
    mask_paths: list[Path | str],
    day_names: list[str],
    output_dir: Path | str,
    params: MatchParams | None = None,
    examples_per_group: int = 6,
    pad_xy: int = 10,
    min_crop_size: int = 72,
) -> dict[str, object]:
    """Run the matcher and render example QC contact sheets.

    Parameters
    ----------
    mask_paths : list[pathlib.Path or str]
        Paths to registered label-mask TIFF stacks, each with shape
        ``(z, y, x)``.
    day_names : list[str]
        Ordered day names aligned one-to-one with ``mask_paths``.
    output_dir : pathlib.Path or str
        Directory where plots and summary files will be written.
    params : MatchParams or None, default=None
        Matcher parameter set used to generate the tracks. When ``None``, the
        default :class:`roi_matcher.MatchParams` values are used.
    examples_per_group : int, default=6
        Maximum number of plotted example tracks per group.
    pad_xy : int, default=10
        Additional crop padding in pixels.
    min_crop_size : int, default=72
        Minimum crop width and height in pixels.

    Returns
    -------
    dict[str, object]
        Summary dictionary containing output paths plus the track, QC, and
        selection tables used to build the plots.
    """

    run_start_seconds = time.perf_counter()
    if params is None:
        params = MatchParams()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    log_message(
        run_start_seconds,
        f"Starting ROI matcher QC example plots | masks={len(mask_paths)} | output_dir={output_path}",
    )

    log_message(run_start_seconds, "Loading registered mask stacks")
    mask_stacks = [tifffile.imread(Path(mask_path)) for mask_path in mask_paths]

    log_message(run_start_seconds, "Running ROI matcher to build track and QC tables")

    def matcher_log(message: str) -> None:
        log_message(run_start_seconds, message)

    tracks_table, pair_tables, qc_table = match_roi_masks(
        mask_stacks=mask_stacks,
        day_names=day_names,
        params=params,
        log_fn=matcher_log,
    )
    merged_table = merge_track_and_qc_tables(tracks_table, qc_table)

    log_message(run_start_seconds, "Extracting ROI records for visualization")
    records_by_day_label = {}
    for day_name, mask_stack in zip(day_names, mask_stacks):
        for record in extract_roi_records(mask_stack, day_name=day_name, patch_radius=params.patch_radius, edge_margin=params.edge_margin):
            records_by_day_label[(day_name, record.label)] = record

    log_message(run_start_seconds, "Selecting representative example tracks")
    selections = select_example_tracks(tracks_table, qc_table, examples_per_group=examples_per_group)
    style = PlotStyle()
    figure_paths: dict[str, str] = {}
    selected_rows = []
    for group_name, selection_table in selections.items():
        merged_selection = merge_track_and_qc_tables(selection_table, qc_table)
        if len(merged_selection) == 0:
            log_message(run_start_seconds, f"No {group_name} confidence examples selected; skipping figure")
        else:
            log_message(
                run_start_seconds,
                f"Rendering {group_name} confidence examples for {len(merged_selection)} tracks",
            )
        figure_path = render_example_group(
            selected_tracks=merged_selection,
            day_names=day_names,
            mask_stacks=mask_stacks,
            records_by_day_label=records_by_day_label,
            output_path=output_path / f"{group_name}_confidence_examples.png",
            group_name=group_name,
            style=style,
            pad_xy=pad_xy,
            min_crop_size=min_crop_size,
        )
        if figure_path is not None:
            figure_paths[group_name] = str(figure_path)
        if len(merged_selection) > 0:
            selection_export = merged_selection.copy()
            selection_export.insert(0, "example_group", group_name)
            selected_rows.append(selection_export)

    selected_examples_table = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    selected_examples_path = output_path / "selected_examples.csv"
    selected_examples_table.to_csv(selected_examples_path, index=False)

    tracks_path = output_path / "tracks.csv"
    qc_path = output_path / "qc.csv"
    tracks_table.to_csv(tracks_path, index=False)
    qc_table.to_csv(qc_path, index=False)
    pair_counts = {f"{day_a}_vs_{day_b}": len(table) for (day_a, day_b), table in pair_tables.items()}
    total_duration_seconds = time.perf_counter() - run_start_seconds

    summary = {
        "mask_paths": [str(Path(mask_path)) for mask_path in mask_paths],
        "day_names": day_names,
        "n_tracks": int(len(tracks_table)),
        "n_full_tracks": int((tracks_table["n_days_present"] == len(day_names)).sum()),
        "full_track_fraction": float((tracks_table["n_days_present"] == len(day_names)).mean()) if len(tracks_table) else 0.0,
        "needs_review_fraction": float(qc_table["needs_review"].mean()) if len(qc_table) else 0.0,
        "pair_counts": pair_counts,
        "figure_paths": figure_paths,
        "examples_per_group": int(examples_per_group),
        "total_duration_seconds": float(total_duration_seconds),
        "total_duration_hms": format_duration_seconds(total_duration_seconds),
    }

    summary_path = output_path / "summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# ROI Matcher QC Example Summary",
                "",
                f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
                f"- Masks: {', '.join(day_names)}",
                f"- Total tracks: {summary['n_tracks']}",
                f"- Full {len(day_names)}-week tracks: {summary['n_full_tracks']} ({summary['full_track_fraction']:.3f})",
                f"- Fraction flagged for review: {summary['needs_review_fraction']:.3f}",
                f"- Pair counts: {json.dumps(pair_counts, sort_keys=True)}",
                f"- Total duration: {summary['total_duration_hms']}",
                "",
                "These figures are QC examples, not calibrated proof that the confidence score is correct.",
            ]
        ),
        encoding="utf-8",
    )
    log_path = output_path / "run_log.json"
    log_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{format_duration_seconds(total_duration_seconds)}] Completed ROI matcher QC example plots", flush=True)
    print(f"total_duration={format_duration_seconds(total_duration_seconds)}")

    return {
        "output_dir": output_path,
        "tracks_table": tracks_table,
        "qc_table": qc_table,
        "merged_table": merged_table,
        "selected_examples_table": selected_examples_table,
        "summary": summary,
        "tracks_path": tracks_path,
        "qc_path": qc_path,
        "selected_examples_path": selected_examples_path,
        "summary_path": summary_path,
        "log_path": log_path,
        "figure_paths": figure_paths,
    }

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for QC example plot generation.

    Parameters
    ----------
    argv : list[str] or None, default=None
        Optional command-line token list. When ``None``, arguments are read from
        ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed command-line options.
    """

    parser = argparse.ArgumentParser(description="Generate ROI matcher QC example contact sheets.")
    parser.add_argument("--masks", nargs="+", required=True, help="Paths to registered ROI mask TIFF stacks.")
    parser.add_argument("--days", nargs="+", required=True, help="Day names aligned to --masks.")
    parser.add_argument("--output-dir", required=True, help="Directory where QC plots will be written.")
    parser.add_argument("--examples-per-group", type=int, default=6)
    parser.add_argument("--pad-xy", type=int, default=10)
    parser.add_argument("--min-crop-size", type=int, default=72)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the QC plotting CLI and write example figures to disk.

    Parameters
    ----------
    argv : list[str] or None, default=None
        Optional command-line token list. When ``None``, arguments are read from
        ``sys.argv``.

    Returns
    -------
    None
        Plot files and summaries are written to disk.
    """

    args = parse_args(argv)
    result = generate_qc_example_plots(
        mask_paths=[Path(mask_path) for mask_path in args.masks],
        day_names=list(args.days),
        output_dir=Path(args.output_dir),
        examples_per_group=args.examples_per_group,
        pad_xy=args.pad_xy,
        min_crop_size=args.min_crop_size,
    )
    print(json.dumps(result["summary"], indent=2))
    print('QC example plots and summary files written to:', result["output_dir"])

if __name__ == "__main__":
    main()
