import numpy as np
import pandas as pd

from roi_matcher import extract_roi_records
from roi_matcher_qc_plots import choose_display_plane, extract_plane_roi_masks


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


def make_single_plane_test_masks() -> dict[str, np.ndarray]:
    """Create small weekly test masks with distinct ROI center planes.

    Parameters
    ----------
    None
        This helper takes no inputs.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping from day name to a label stack with shape ``(4, 24, 24)`` in
        ``(z, y, x)`` order.
    """

    week1 = np.zeros((4, 24, 24), dtype=np.uint16)
    week2 = np.zeros_like(week1)
    week3 = np.zeros_like(week1)

    draw_box(week1, 11, (0, 2), (5, 9), (5, 9))
    draw_box(week2, 21, (2, 4), (6, 10), (6, 10))
    draw_box(week2, 22, (1, 2), (10, 14), (10, 14))
    draw_box(week3, 31, (1, 3), (12, 16), (12, 16))

    return {"week1": week1, "week2": week2, "week3": week3}


def build_records_by_day_label() -> dict[tuple[str, int], object]:
    """Extract ROI records for the single-plane test masks.

    Parameters
    ----------
    None
        This helper takes no inputs.

    Returns
    -------
    dict[tuple[str, int], object]
        Mapping from ``(day_name, roi_label)`` to ROIRecord objects.
    """

    records = {}
    for day_name, mask in make_single_plane_test_masks().items():
        for record in extract_roi_records(mask, day_name=day_name, patch_radius=4, edge_margin=3):
            records[(day_name, record.label)] = record
    return records


def test_extract_plane_roi_masks_keeps_only_labels_present_on_selected_z() -> None:
    week2 = make_single_plane_test_masks()["week2"]

    roi_masks = extract_plane_roi_masks(week2, z_index=2, y0=0, y1=24, x0=0, x1=24)

    assert set(roi_masks) == {21}
    assert roi_masks[21].shape == (24, 24)


def test_extract_plane_roi_masks_excludes_rois_absent_from_selected_z() -> None:
    week2 = make_single_plane_test_masks()["week2"]

    roi_masks = extract_plane_roi_masks(week2, z_index=1, y0=0, y1=24, x0=0, x1=24)

    assert set(roi_masks) == {22}
    assert 21 not in roi_masks


def test_choose_display_plane_uses_matched_center_plane_when_present() -> None:
    records_by_day_label = build_records_by_day_label()
    track_row = pd.Series({"week1_roi": 11, "week2_roi": 21, "week3_roi": pd.NA})

    z_index = choose_display_plane(
        track_row=track_row,
        day_name="week2",
        day_names=["week1", "week2", "week3"],
        records_by_day_label=records_by_day_label,
        max_z_index=3,
    )

    assert z_index == int(records_by_day_label[("week2", 21)].center_plane)


def test_choose_display_plane_uses_median_present_plane_for_missing_week() -> None:
    records_by_day_label = build_records_by_day_label()
    track_row = pd.Series({"week1_roi": 11, "week2_roi": 21, "week3_roi": pd.NA})

    z_index = choose_display_plane(
        track_row=track_row,
        day_name="week3",
        day_names=["week1", "week2", "week3"],
        records_by_day_label=records_by_day_label,
        max_z_index=3,
    )

    expected = int(
        round(
            np.median(
                [
                    records_by_day_label[("week1", 11)].center_plane,
                    records_by_day_label[("week2", 21)].center_plane,
                ]
            )
        )
    )
    assert z_index == expected
