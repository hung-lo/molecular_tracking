import numpy as np

from roi_matcher_qc_plots import (
    extract_crop_roi_masks,
    split_matched_and_neighbor_masks,
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


def make_crop_test_mask() -> np.ndarray:
    """Build a small labeled stack with multiple nearby ROIs.

    Parameters
    ----------
    None
        This helper takes no inputs.

    Returns
    -------
    numpy.ndarray
        Integer label stack with shape ``(3, 24, 24)`` in ``(z, y, x)`` order.
    """

    mask = np.zeros((3, 24, 24), dtype=np.uint16)
    draw_box(mask, 10, (1, 3), (5, 9), (5, 9))
    draw_box(mask, 20, (0, 2), (8, 12), (10, 14))
    draw_box(mask, 30, (1, 3), (14, 18), (14, 18))
    return mask


def test_extract_crop_roi_masks_keeps_all_labels_in_crop() -> None:
    mask = make_crop_test_mask()

    roi_masks = extract_crop_roi_masks(mask, y0=4, y1=16, x0=4, x1=16)

    assert set(roi_masks) == {10, 20, 30}
    assert roi_masks[10].shape == (12, 12)
    assert roi_masks[20].any()
    assert roi_masks[30].any()


def test_extract_crop_roi_masks_excludes_background_and_empty_labels() -> None:
    mask = make_crop_test_mask()

    roi_masks = extract_crop_roi_masks(mask, y0=0, y1=6, x0=0, x1=6)

    assert set(roi_masks) == {10}
    assert 0 not in roi_masks


def test_split_matched_and_neighbor_masks_preserves_matched_label() -> None:
    mask = make_crop_test_mask()
    roi_masks = extract_crop_roi_masks(mask, y0=4, y1=16, x0=4, x1=16)

    matched_mask, neighbor_masks = split_matched_and_neighbor_masks(roi_masks, matched_label=20)

    assert matched_mask is not None
    assert matched_mask.any()
    assert set(neighbor_masks) == {10, 30}


def test_split_matched_and_neighbor_masks_handles_missing_match() -> None:
    mask = make_crop_test_mask()
    roi_masks = extract_crop_roi_masks(mask, y0=4, y1=16, x0=4, x1=16)

    matched_mask, neighbor_masks = split_matched_and_neighbor_masks(roi_masks, matched_label=None)

    assert matched_mask is None
    assert set(neighbor_masks) == {10, 20, 30}
