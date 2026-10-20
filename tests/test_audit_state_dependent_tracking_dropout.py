from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from audit_state_dependent_tracking_dropout import (  # noqa: E402
    _ambiguous,
    _assessment_candidates,
    _primary_assessable,
    classify_primary_outcome,
    overlap_candidates,
    propagate_source_roi,
    _load_transform,
)


SPACING = (5.0, 0.693, 0.693)


def _candidates(propagated: np.ndarray, labels: np.ndarray) -> list[dict]:
    counts = np.bincount(labels.ravel())
    centroids = np.full((len(counts), 3), np.nan)
    for label in range(1, len(counts)):
        coords = np.argwhere(labels == label)
        if len(coords):
            centroids[label] = coords.mean(axis=0)
    boxes = [None] * max(0, len(counts) - 1)
    return overlap_candidates(propagated, labels, counts, centroids, SPACING, boxes)


def test_canonical_target_with_same_graph_label_is_tracked_ok():
    target = np.zeros((3, 8, 8), dtype=np.int16)
    target[1, 2:4, 2:4] = 7
    coords = np.argwhere(target == 7)
    candidates = _candidates(coords, target)
    assert classify_primary_outcome(
        assessable=True, canonical_candidates=candidates, unfiltered_candidates=candidates,
        graph_target_labels={7}, fragmentation_or_merge=False, red_signal_remains=True,
    ) == "TRACKED_OK"


def test_canonical_target_with_graph_mismatch_is_identity_failure():
    target = np.zeros((3, 8, 8), dtype=np.int16)
    target[1, 2:4, 2:4] = 7
    candidates = _candidates(np.argwhere(target == 7), target)
    assert classify_primary_outcome(
        assessable=True, canonical_candidates=candidates, unfiltered_candidates=candidates,
        graph_target_labels={9}, fragmentation_or_merge=False, red_signal_remains=False,
    ) == "TARGET_PRESENT_GRAPH_IDENTITY_FAILURE"


def test_only_subthreshold_unfiltered_target_is_min_size_dropout():
    target = np.zeros((3, 8, 8), dtype=np.int16)
    target[1, 2:4, 2:4] = 1
    candidates = _candidates(np.argwhere(target == 1), target)
    assert candidates[0]["target_voxels"] == 4
    assert classify_primary_outcome(
        assessable=True, canonical_candidates=[], unfiltered_candidates=candidates,
        graph_target_labels=set(), fragmentation_or_merge=False, red_signal_remains=False,
    ) == "MIN_SIZE_FILTER_DROPOUT"


def test_plausible_mask_correspondence_does_not_require_strong_local_ncc():
    candidate = {"plausible": True, "strict_plausible": True, "target_voxels": 93}
    assert _primary_assessable(
        geometry_assessable=True, image_assessable=False,
        canonical_candidates=[], unfiltered_candidates=[candidate],
    )
    assert not _primary_assessable(
        geometry_assessable=True, image_assessable=False,
        canonical_candidates=[], unfiltered_candidates=[],
    )
    assert not _primary_assessable(
        geometry_assessable=False, image_assessable=False,
        canonical_candidates=[], unfiltered_candidates=[candidate],
    )
    weak_candidate = {"plausible": True, "strict_plausible": False}
    assert not _primary_assessable(
        geometry_assessable=True, image_assessable=False,
        canonical_candidates=[weak_candidate], unfiltered_candidates=[],
    )
    assert not _assessment_candidates([weak_candidate], image_assessable=False)[0]["plausible"]


def test_no_mask_with_residual_signal_is_segmentation_failure():
    assert classify_primary_outcome(
        assessable=True, canonical_candidates=[], unfiltered_candidates=[],
        graph_target_labels=set(), fragmentation_or_merge=False, red_signal_remains=True,
    ) == "NO_MASK_BUT_RED_SIGNAL_REMAINS"


def test_no_mask_without_residual_signal_is_reporter_disappearance_candidate():
    assert classify_primary_outcome(
        assessable=True, canonical_candidates=[], unfiltered_candidates=[],
        graph_target_labels=set(), fragmentation_or_merge=False, red_signal_remains=False,
    ) == "REPORTER_OBJECT_DISAPPEARANCE"


def test_two_jointly_overlapping_components_are_fragmentation_ambiguity():
    target = np.zeros((2, 8, 8), dtype=np.int16)
    source_coords = np.asarray([[0, 2, x] for x in range(2, 10)])
    # Keep coordinates in-bounds while splitting the source into two labels.
    source_coords = np.asarray([[0, 2, x] for x in range(2, 8)])
    target[0, 2, 2:5] = 1
    target[0, 2, 5:8] = 2
    candidates = _candidates(source_coords, target)
    ambiguous, reason = _ambiguous(candidates, len(source_coords))
    assert ambiguous
    assert reason == "multiple_components_jointly_cover_source"
    assert classify_primary_outcome(
        assessable=True, canonical_candidates=candidates, unfiltered_candidates=candidates,
        graph_target_labels={1}, fragmentation_or_merge=ambiguous, red_signal_remains=True,
    ) == "FRAGMENTATION_OR_MERGE_AMBIGUITY"


def test_propagated_roi_outside_target_is_not_assessable():
    class ShiftOutside:
        @staticmethod
        def apply(points):
            return np.asarray(points) + np.asarray([0, 0, 20])

    coords = np.asarray([[1, 2, 2], [1, 2, 3]], dtype=np.int32)
    propagated, coverage = propagate_source_roi(coords, ShiftOutside(), (3, 8, 8))
    assert len(propagated) == 0
    assert coverage == 0
    assert classify_primary_outcome(
        assessable=False, canonical_candidates=[], unfiltered_candidates=[],
        graph_target_labels=set(), fragmentation_or_merge=False, red_signal_remains=False,
    ) == "NOT_ASSESSABLE"


def test_dropout_audit_propagates_with_selected_local_field(tmp_path: Path):
    path = tmp_path / "selected.npz"
    flow = np.zeros((3, 2, 2, 2), dtype=np.float32)
    flow[1] = 0.25
    np.savez_compressed(
        path, matrix=np.eye(3), offset=np.zeros(3), flow=flow,
        flow_step=np.asarray([1.0, 2.0, 2.0]), has_flow=np.asarray(True),
        method=np.asarray("image_affine_local"), fallback_reason=np.asarray(""),
        n_seed=np.asarray(0), n_inlier=np.asarray(0), residual_median_um=np.asarray(np.nan),
        residual_p95_um=np.asarray(np.nan),
    )
    transforms = pd.DataFrame([{
        "day_a": "s0", "day_b": "s1", "z_intercept": 0, "z_scale": 1,
        "y_intercept": 0, "y_from_y": 1, "y_from_x": 0,
        "x_intercept": 0, "x_from_y": 0, "x_from_x": 1,
        "method": "image_affine_local", "fallback_reason": "",
    }])
    registration_qc = pd.DataFrame([{
        "day_a": "s0", "day_b": "s1", "selected_registration_stage": "image_affine_local",
        "transform_path": str(path), "local_qc_pass": True,
    }])
    _stored, source_to_target, _target_to_source = _load_transform(
        transforms, "s0", "s1", registration_qc
    )
    mapped, coverage = propagate_source_roi(np.asarray([[1, 5, 5]], dtype=np.int32), source_to_target, (8, 8, 8))
    assert np.array_equal(mapped, [[1, 6, 5]])
    assert coverage == 1.0
