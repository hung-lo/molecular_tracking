"""Reusable centered cropping for ZYX image volumes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_crop_shape(crop_shape: int | Sequence[int]) -> tuple[int, int]:
    """Normalize a scalar or ``(height, width)`` crop request."""

    if isinstance(crop_shape, (int, np.integer)):
        crop_shape = (int(crop_shape), int(crop_shape))
    if len(crop_shape) != 2:
        raise ValueError(f"crop_shape must be a (height, width) pair, got {crop_shape}")
    crop_h, crop_w = (int(crop_shape[0]), int(crop_shape[1]))
    if crop_h <= 0 or crop_w <= 0:
        raise ValueError(f"crop_shape must be positive, got {crop_shape}")
    return crop_h, crop_w


def center_crop_zyx(image_array: np.ndarray, crop_shape: int | Sequence[int]) -> np.ndarray:
    """Return a centered rectangular crop from a 3D ``(z, y, x)`` array."""

    image_array = np.asarray(image_array)
    if image_array.ndim != 3:
        raise ValueError(f"Expected a 3D ZYX array, received shape {image_array.shape}")
    crop_h, crop_w = normalize_crop_shape(crop_shape)
    z, height, width = image_array.shape
    if crop_h > height or crop_w > width:
        raise ValueError(
            f"Requested crop {(crop_h, crop_w)} does not fit image shape {(height, width)}"
        )
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    cropped = image_array[:, y0:y0 + crop_h, x0:x0 + crop_w]
    assert cropped.shape == (z, crop_h, crop_w)
    return cropped
