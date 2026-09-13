from __future__ import annotations

import numpy as np

from affine_overlap_matcher import RestrictedTransform
from endpoint_evaluator import apply_transform_b_to_a, compose_transforms, invert_restricted_transform


def _transform(**updates: float) -> RestrictedTransform:
    values = {
        "z_intercept": 1.0, "z_scale": 2.0, "y_intercept": 3.0, "y_from_y": 2.0,
        "y_from_x": 1.0, "x_intercept": -1.0, "x_from_y": 1.0, "x_from_x": 3.0,
        "method": "test", "fallback_reason": None, "n_seed": 0, "n_inlier": 0,
        "residual_median_um": None, "residual_p95_um": None,
    }
    values.update(updates)
    return RestrictedTransform(**values)


def test_inverse_transform_round_trip() -> None:
    point = np.array([2.0, 4.0, 5.0])
    mapped = apply_transform_b_to_a(point, _transform())
    recovered = apply_transform_b_to_a(mapped, invert_restricted_transform(_transform()))
    assert np.allclose(recovered, point)


def test_composed_transform_matches_sequential_application() -> None:
    first = _transform(z_intercept=1.0, z_scale=1.0, y_intercept=2.0, y_from_y=1.0, y_from_x=0.0, x_intercept=3.0, x_from_y=0.0, x_from_x=1.0)
    second = _transform(z_intercept=4.0, z_scale=1.0, y_intercept=5.0, y_from_y=1.0, y_from_x=0.0, x_intercept=6.0, x_from_y=0.0, x_from_x=1.0)
    point = np.array([1.0, 2.0, 3.0])
    assert np.allclose(apply_transform_b_to_a(point, compose_transforms(first, second)), apply_transform_b_to_a(apply_transform_b_to_a(point, second), first))


def test_near_singular_transform_is_rejected() -> None:
    singular = _transform(y_from_y=1.0, y_from_x=2.0, x_from_y=2.0, x_from_x=4.0)
    try:
        invert_restricted_transform(singular)
    except ValueError as exc:
        assert "singular" in str(exc)
    else:
        raise AssertionError("near-singular transform was accepted")
