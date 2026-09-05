from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from affine_overlap_matcher import RestrictedTransform, VoxelSpacing, extract_roi_features
from cross_laser_roi_mapper import (
    CrossLaserSource,
    _annotate_pair_table,
    build_fixed_coverage,
    build_moving_coverage,
    classify_common_volume,
    inverse_restricted_transform,
    map_cross_laser_source,
    relabel_primary_high_mask,
    resolve_identity_evidence,
)


def _mask_with_objects(objects: list[tuple[int, tuple[slice, slice, slice]]]) -> np.ndarray:
    mask = np.zeros((12, 36, 36), dtype=np.uint16)
    for label, region in objects:
        mask[region] = label
    return mask


def _sources() -> tuple[CrossLaserSource, CrossLaserSource]:
    return (
        CrossLaserSource("1050_red", 1050, "red"),
        CrossLaserSource("920_green_primary", 920, "green"),
    )


def test_cross_laser_mapper_recovers_z_shift_with_permuted_labels() -> None:
    fixed = _mask_with_objects(
        [
            (11, (slice(3, 5), slice(4, 9), slice(4, 9))),
            (22, (slice(5, 7), slice(16, 21), slice(16, 21))),
        ]
    )
    moving = _mask_with_objects(
        [
            (101, (slice(2, 4), slice(4, 9), slice(4, 9))),
            (202, (slice(4, 6), slice(16, 21), slice(16, 21))),
        ]
    )
    fixed_source, moving_source = _sources()

    result = map_cross_laser_source(
        mouse_id="mouse",
        session_id="session",
        acquisition_date="2026-08-19",
        fixed_mask=fixed,
        moving_mask=moving,
        fixed_source=fixed_source,
        moving_source=moving_source,
        spacing=VoxelSpacing(z_um=2.5, y_um=0.7, x_um=0.7),
    )

    assert result.summary["pair_gap"] is None
    assert result.summary["relationship_type"] == "cross_laser_same_session"
    assert set(zip(result.high_matches["label_1050"], result.high_matches["label_920"])) == {
        (11, 101),
        (22, 202),
    }
    assert result.high_matches["aligned_residual_distance_um"].max() < 1e-8
    assert set(result.fixed_coverage["green_status"]) == {"high"}


def test_cross_laser_coverage_keeps_missing_fixed_and_extra_moving_labels() -> None:
    fixed = _mask_with_objects(
        [
            (11, (slice(3, 5), slice(4, 9), slice(4, 9))),
            (22, (slice(5, 7), slice(16, 21), slice(16, 21))),
            (33, (slice(7, 9), slice(26, 31), slice(26, 31))),
        ]
    )
    moving = _mask_with_objects(
        [
            (101, (slice(3, 5), slice(4, 9), slice(4, 9))),
            (202, (slice(5, 7), slice(16, 21), slice(16, 21))),
            (303, (slice(2, 4), slice(28, 33), slice(3, 8))),
        ]
    )
    fixed_source, moving_source = _sources()
    result = map_cross_laser_source(
        mouse_id="mouse",
        session_id="session",
        acquisition_date="2026-08-19",
        fixed_mask=fixed,
        moving_mask=moving,
        fixed_source=fixed_source,
        moving_source=moving_source,
        spacing=VoxelSpacing(),
    )

    fixed_row = result.fixed_coverage.set_index("label_1050").loc[33]
    moving_row = result.moving_coverage.set_index("label_920").loc[303]
    assert fixed_row["green_status"] == "no_candidate"
    assert moving_row["mapping_status"] == "no_candidate"


def test_shape_mismatch_fails_before_matching() -> None:
    fixed = np.zeros((3, 4, 5), dtype=np.uint16)
    fixed[1, 1, 1] = 1
    moving = np.zeros((3, 4, 6), dtype=np.uint16)
    moving[1, 1, 1] = 2
    fixed_source, moving_source = _sources()

    with pytest.raises(ValueError, match="same ZYX shape"):
        map_cross_laser_source(
            mouse_id="mouse",
            session_id="session",
            acquisition_date="2026-08-19",
            fixed_mask=fixed,
            moving_mask=moving,
            fixed_source=fixed_source,
            moving_source=moving_source,
            spacing=VoxelSpacing(),
        )


def test_inverse_transform_round_trip_and_common_volume_status() -> None:
    transform = RestrictedTransform(
        z_intercept=5.0,
        z_scale=1.0,
        y_intercept=0.0,
        y_from_y=1.0,
        y_from_x=0.0,
        x_intercept=0.0,
        x_from_y=0.0,
        x_from_x=1.0,
        method="translation_only",
        fallback_reason=None,
        n_seed=0,
        n_inlier=0,
        residual_median_um=None,
        residual_p95_um=None,
    )
    points = np.asarray([[1.25, 3.5, 4.75], [2.0, 5.0, 6.0]])
    assert np.allclose(
        inverse_restricted_transform(transform, transform.apply(points)), points
    )

    fixed = _mask_with_objects(
        [
            (1, (slice(0, 2), slice(4, 7), slice(4, 7))),
            (2, (slice(4, 7), slice(12, 15), slice(12, 15))),
        ]
    )
    features = extract_roi_features(fixed, "session", VoxelSpacing())
    statuses = classify_common_volume(features, transform, fixed.shape).set_index("label_1050")
    assert statuses.loc[1, "common_volume_status"] == "outside_common_volume"
    assert statuses.loc[2, "common_volume_status"] == "partially_inside_common_volume"
    assert bool(statuses.loc[2, "cross_laser_edge_clipped"])


def test_common_volume_is_conservative_when_no_transformed_corner_is_inside() -> None:
    angle = np.sqrt(0.5)
    transform = RestrictedTransform(
        z_intercept=0.0, z_scale=1.0,
        y_intercept=4.0, y_from_y=angle, y_from_x=-angle,
        x_intercept=4.0 - 8.0 * angle, x_from_y=angle, x_from_x=angle,
        method="translation_only", fallback_reason=None, n_seed=0, n_inlier=0,
        residual_median_um=None, residual_p95_um=None,
    )
    features = pd.DataFrame({"label": [1], "bbox_z0": [1], "bbox_z1": [3], "bbox_y0": [1], "bbox_y1": [8], "bbox_x0": [1], "bbox_x1": [8]})
    status = classify_common_volume(features, transform, (9, 9, 9)).iloc[0]
    assert status["common_volume_status"] == "partially_inside_common_volume"


def test_identity_resolution_keeps_primary_automatic_and_red_provisional() -> None:
    primary = pd.DataFrame(
        {
            "label_1050": [1, 2],
            "green_status": ["high", "no_candidate"],
            "green_high_label_920": [10, np.nan],
        }
    )
    secondary = pd.DataFrame(
        {
            "label_1050": [1, 2],
            "red_status": ["high", "high"],
            "red_high_label_920": [20, 21],
        }
    )

    resolution = resolve_identity_evidence(primary, secondary_fixed_coverage=secondary)
    rows = resolution.set_index("label_1050")
    assert rows.loc[1, "resolved_status"] == "primary_high"
    assert bool(rows.loc[1, "recommended_for_identity"])
    assert rows.loc[2, "resolved_status"] == "secondary_high_rescue_candidate"
    assert bool(rows.loc[2, "provisional_identity"])
    assert bool(rows.loc[2, "review_required"])
    assert not bool(rows.loc[2, "recommended_for_identity"])


def test_identity_resolution_marks_outside_volume_without_candidate_failure() -> None:
    primary = pd.DataFrame(
        {
            "label_1050": [1],
            "green_status": ["no_candidate"],
            "green_high_label_920": [np.nan],
            "common_volume_status": ["outside_common_volume"],
        }
    )
    row = resolve_identity_evidence(primary).iloc[0]
    assert row["resolved_status"] == "outside_common_volume"
    assert row["resolved_920_source"] == ""
    assert pd.isna(row["resolved_label_920"])
    assert not bool(row["recommended_for_identity"])
    assert not bool(row["review_required"])


def test_identity_resolution_blocks_cross_source_conflict() -> None:
    primary = pd.DataFrame(
        {
            "label_1050": [1, 2],
            "green_status": ["high", "no_candidate"],
            "green_high_label_920": [10, np.nan],
        }
    )
    secondary = pd.DataFrame(
        {
            "label_1050": [1, 2],
            "red_status": ["no_candidate", "high"],
            "red_high_label_920": [np.nan, 20],
        }
    )
    consistency = pd.DataFrame({"label_1050": [10], "label_920": [20]})

    resolution = resolve_identity_evidence(
        primary,
        secondary_fixed_coverage=secondary,
        green_red_high_matches=consistency,
    ).set_index("label_1050")
    assert bool(resolution.loc[2, "cross_source_conflict"])
    assert resolution.loc[2, "resolved_status"] == "cross_source_conflict"
    assert not bool(resolution.loc[2, "recommended_for_identity"])


def test_relabelled_primary_mask_preserves_native_geometry_without_source_mutation() -> None:
    source = _mask_with_objects(
        [
            (101, (slice(3, 5), slice(4, 9), slice(4, 9))),
            (202, (slice(5, 7), slice(16, 21), slice(16, 21))),
        ]
    )
    before = source.copy()
    matches = pd.DataFrame({"label_1050": [11], "label_920": [101]})

    relabelled = relabel_primary_high_mask(source, matches)

    assert relabelled.shape == source.shape
    assert np.array_equal(source, before)
    assert set(np.unique(relabelled)) == {0, 11}


def test_empty_match_tables_keep_coverage_and_consistency_schema() -> None:
    fixed = _mask_with_objects([(1, (slice(3, 5), slice(4, 9), slice(4, 9)))])
    moving = _mask_with_objects([(101, (slice(3, 5), slice(4, 9), slice(4, 9)))])
    fixed_source, moving_source = _sources()
    result = map_cross_laser_source(mouse_id="m", session_id="s", acquisition_date="2026-01-01", fixed_mask=fixed, moving_mask=moving, fixed_source=fixed_source, moving_source=moving_source, spacing=VoxelSpacing())
    result.candidates = result.candidates.iloc[:0].copy()
    result.high_matches = result.high_matches.iloc[:0].copy()
    result.balanced_matches = result.balanced_matches.iloc[:0].copy()
    fixed_coverage = build_fixed_coverage(result)
    moving_coverage = build_moving_coverage(result)
    assert fixed_coverage["green_status"].tolist() == ["no_candidate"]
    assert moving_coverage["mapping_status"].tolist() == ["no_candidate"]
    consistency = map_cross_laser_source(mouse_id="m", session_id="s", acquisition_date="2026-01-01", fixed_mask=moving, moving_mask=moving, fixed_source=CrossLaserSource("920_green", 920, "green"), moving_source=CrossLaserSource("920_red", 920, "red"), spacing=VoxelSpacing())
    empty = _annotate_pair_table(pd.DataFrame(columns=["label_a", "label_b", "score", "dice", "distance_um"]), mouse_id="m", session_id="s", acquisition_date="2026-01-01", fixed_source=consistency.fixed_source, moving_source=consistency.source, fixed_features=consistency.fixed_features, moving_features=consistency.moving_features, transform=consistency.transform, spacing=VoxelSpacing())
    assert {"fixed_label", "moving_label", "label_920_green", "label_920_red"}.issubset(empty.columns)
    assert not any(column.startswith("centroid_1050") or column == "label_1050" for column in empty.columns)


def test_competing_moving_labels_do_not_duplicate_fixed_identity() -> None:
    fixed = _mask_with_objects([(1, (slice(3, 5), slice(4, 10), slice(4, 10)))])
    moving = _mask_with_objects([(101, (slice(3, 5), slice(4, 7), slice(4, 10))), (102, (slice(3, 5), slice(7, 10), slice(4, 10)))])
    fixed_source, moving_source = _sources()
    result = map_cross_laser_source(mouse_id="m", session_id="s", acquisition_date="2026-01-01", fixed_mask=fixed, moving_mask=moving, fixed_source=fixed_source, moving_source=moving_source, spacing=VoxelSpacing())
    assert result.high_matches["label_1050"].nunique() <= 1
