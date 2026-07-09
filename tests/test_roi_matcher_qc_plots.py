import numpy as np
import pandas as pd

from roi_matcher import extract_roi_records
from roi_matcher_qc_plots import (
    build_track_plot_metadata,
    compute_track_crop_bounds,
    select_example_tracks,
)


def draw_box(
    mask_stack: np.ndarray,
    label: int,
    z_range: tuple[int, int],
    y_range: tuple[int, int],
    x_range: tuple[int, int],
) -> None:
    """Fill one rectangular ROI into a labeled ``(z, y, x)`` mask stack.

    Parameters
    ----------
    mask_stack : numpy.ndarray
        Integer label image with shape ``(z, y, x)``.
    label : int
        Positive label value written into the requested region.
    z_range : tuple[int, int]
        Inclusive-exclusive ROI bounds along the z axis in planes.
    y_range : tuple[int, int]
        Inclusive-exclusive ROI bounds along the y axis in pixels.
    x_range : tuple[int, int]
        Inclusive-exclusive ROI bounds along the x axis in pixels.

    Returns
    -------
    None
        The input array is modified in place.
    """

    z0, z1 = z_range
    y0, y1 = y_range
    x0, x1 = x_range
    mask_stack[z0:z1, y0:y1, x0:x1] = label


def build_records_by_day_label() -> dict[tuple[str, int], object]:
    """Create a small multi-week ROI record lookup for crop-bound tests.

    Parameters
    ----------
    None
        This helper takes no inputs.

    Returns
    -------
    dict[tuple[str, int], object]
        Mapping from ``(day_name, roi_label)`` to ROIRecord objects.
    """

    week1 = np.zeros((3, 30, 30), dtype=np.uint16)
    week2 = np.zeros_like(week1)
    draw_box(week1, 11, (1, 3), (4, 8), (5, 9))
    draw_box(week2, 21, (1, 3), (6, 10), (8, 12))
    draw_box(week2, 22, (1, 3), (0, 3), (0, 4))

    records = {}
    for day_name, mask in [("week1", week1), ("week2", week2)]:
        for record in extract_roi_records(mask, day_name=day_name, patch_radius=4, edge_margin=4):
            records[(day_name, record.label)] = record
    return records


def test_select_example_tracks_returns_requested_groups() -> None:
    tracks = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3, 4, 5],
            "n_days_present": [4, 4, 3, 2, 4],
            "mean_confidence": [0.95, 0.82, 0.61, 0.42, 0.74],
            "min_confidence": [0.90, 0.80, 0.55, 0.30, 0.71],
            "used_gap_bridge": [False, False, False, True, False],
            "missing_intermediate_days": [0, 0, 0, 1, 0],
        }
    )
    qc = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3, 4, 5],
            "needs_review": [False, False, False, True, False],
            "low_confidence": [False, False, False, True, False],
            "gap_bridge_qc": [False, False, False, True, False],
            "edge_qc": [False, False, False, True, False],
        }
    )

    groups = select_example_tracks(tracks, qc, examples_per_group=2)

    assert set(groups) == {"high", "medium", "low", "review"}
    assert groups["high"]["cluster_id"].tolist() == [1, 2]
    assert len(groups["medium"]) == 2
    assert len(groups["low"]) == 2


def test_select_example_tracks_can_filter_needs_review_cases() -> None:
    tracks = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "n_days_present": [4, 4, 4],
            "mean_confidence": [0.91, 0.55, 0.35],
            "min_confidence": [0.88, 0.40, 0.20],
            "used_gap_bridge": [False, True, False],
            "missing_intermediate_days": [0, 1, 0],
        }
    )
    qc = pd.DataFrame(
        {
            "cluster_id": [1, 2, 3],
            "needs_review": [False, True, True],
            "low_confidence": [False, True, True],
            "gap_bridge_qc": [False, True, False],
            "edge_qc": [False, False, True],
        }
    )

    groups = select_example_tracks(tracks, qc, examples_per_group=2)

    assert groups["review"]["cluster_id"].tolist() == [3, 2]


def test_compute_track_crop_bounds_contains_all_present_rois() -> None:
    records_by_day_label = build_records_by_day_label()
    track_row = pd.Series({"week1_roi": 11, "week2_roi": 21})

    y0, y1, x0, x1 = compute_track_crop_bounds(
        track_row=track_row,
        day_names=["week1", "week2"],
        records_by_day_label=records_by_day_label,
        image_shape_yx=(30, 30),
        pad_xy=2,
        min_crop_size=8,
    )

    assert y0 <= 4
    assert y1 >= 10
    assert x0 <= 5
    assert x1 >= 12


def test_compute_track_crop_bounds_clips_edges() -> None:
    records_by_day_label = build_records_by_day_label()
    track_row = pd.Series({"week1_roi": pd.NA, "week2_roi": 22})

    y0, y1, x0, x1 = compute_track_crop_bounds(
        track_row=track_row,
        day_names=["week1", "week2"],
        records_by_day_label=records_by_day_label,
        image_shape_yx=(30, 30),
        pad_xy=5,
        min_crop_size=10,
    )

    assert y0 == 0
    assert x0 == 0
    assert y1 <= 30
    assert x1 <= 30


def test_build_track_plot_metadata_reports_confidence_and_qc_flags() -> None:
    track_row = pd.Series(
        {
            "cluster_id": 7,
            "n_days_present": 3,
            "mean_confidence": 0.62,
            "min_confidence": 0.41,
            "used_gap_bridge": True,
            "missing_intermediate_days": 1,
        }
    )
    qc_row = pd.Series(
        {
            "needs_review": True,
            "low_confidence": True,
            "gap_bridge_qc": True,
            "edge_qc": False,
        }
    )

    metadata = build_track_plot_metadata(track_row, qc_row)

    assert metadata["cluster_id"] == 7
    assert metadata["confidence_label"] == "0.62 / 0.41"
    assert "low_confidence" in metadata["flag_text"]
    assert "gap_bridge" in metadata["flag_text"]
