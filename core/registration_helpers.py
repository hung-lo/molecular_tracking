"""Small reusable helpers for 3D registration inputs."""

from __future__ import annotations

from typing import Any

import numpy as np

XY_SPACING_UM = 710.0 / 1024.0
Z_SPACING_UM = 5.0
SPACING_ZYX = (Z_SPACING_UM, XY_SPACING_UM, XY_SPACING_UM)


def ants_from_zyx(
    array: np.ndarray,
    *,
    is_label: bool = False,
    spacing_zyx: tuple[float, float, float] | None = None,
    ants_module: Any | None = None,
):
    """Create a 3D ANTs image from a ``(z, y, x)`` NumPy array."""

    if ants_module is None:
        import ants as ants_module
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D ZYX array, received shape {array.shape}")

    dtype = np.uint32 if is_label else np.float32
    return ants_module.from_numpy(
        array.astype(dtype, copy=False),
        spacing=SPACING_ZYX if spacing_zyx is None else tuple(float(value) for value in spacing_zyx),
        origin=(0.0, 0.0, 0.0),
        direction=np.eye(3),
    )
