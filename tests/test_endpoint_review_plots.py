from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from endpoint_review_plots import plot_endpoint_review_panel, plot_track_presence_timeline


def _inputs(tmp_path: Path):
    red = np.ones((2, 12, 12), dtype=np.uint16)
    green = red * 2
    mask = np.zeros_like(red)
    mask[:, 0:2, 0:2] = 1  # Deliberately near an edge to exercise padded crops.
    red_path = tmp_path / "red.tif"; green_path = tmp_path / "green.tif"; mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(red_path, red); tifffile.imwrite(green_path, green); tifffile.imwrite(mask_path, mask)
    sessions = pd.DataFrame([
        {"session_index": 0, "session_id": "s0", "acquisition_date": "2026-01-01", "mask_path": str(mask_path), "red_image_path": str(red_path), "green_image_path": str(green_path)},
        {"session_index": 1, "session_id": "s1", "acquisition_date": "2026-01-02", "mask_path": "", "red_image_path": "", "green_image_path": ""},
    ])
    features = pd.DataFrame([
        {"session_id": "s0", "label": 1, "centroid_z": 0, "centroid_y": 0.5, "centroid_x": 0.5},
        {"session_id": "s1", "label": 2, "centroid_z": 0, "centroid_y": 0.7, "centroid_x": 0.7},
    ])
    endpoint = pd.Series({
        "endpoint_id": "endpoint_1", "track_uid": "s0:1", "end_session_index": 0,
        "end_session_id": "s0", "end_label": 1, "classification": "nearby_new_track_candidate",
        "eclipse_core_state": "Low", "eclipse_z": -2.5, "green": 20, "red": 50,
        "volume_um3": 100, "touches_xy_edge": True,
    })
    candidates = pd.DataFrame([{
        "endpoint_id": "endpoint_1", "target_session_index": 1, "target_label": 2,
        "target_rank_by_distance": 1, "projected_distance_um": 1.2,
        "target_track_uid": "s1:2", "target_track_starts_here": True,
        "target_is_singleton": False, "session_gap": 1,
        "transform_source": "direct_stored", "transform_reliable": True,
    }])
    transforms = pd.DataFrame([{
        "day_a": "s0", "day_b": "s1", "z_intercept": 0, "z_scale": 1,
        "y_intercept": 0, "y_from_y": 1, "y_from_x": 0,
        "x_intercept": 0, "x_from_y": 0, "x_from_x": 1,
    }])
    return endpoint, sessions, features, candidates, transforms


def test_endpoint_panel_is_png_only_and_handles_missing_raw_and_edge_crop(tmp_path: Path) -> None:
    endpoint, sessions, features, candidates, transforms = _inputs(tmp_path)
    output = tmp_path / "endpoint.png"
    plot_endpoint_review_panel(endpoint, sessions, features, candidates, transforms, output_path=output)
    assert output.is_file() and output.stat().st_size > 0
    assert not list(tmp_path.glob("*.pdf"))


def test_endpoint_panel_rejects_non_png_output(tmp_path: Path) -> None:
    endpoint, sessions, features, candidates, transforms = _inputs(tmp_path)
    with pytest.raises(ValueError, match="PNG"):
        plot_endpoint_review_panel(endpoint, sessions, features, candidates, transforms, output_path=tmp_path / "endpoint.pdf")


def test_track_presence_timeline_handles_missing_session(tmp_path: Path) -> None:
    _endpoint, sessions, _features, _candidates, _transforms = _inputs(tmp_path)
    track = pd.Series({"track_uid": "s0:1", "s0_roi": 1, "s1_roi": pd.NA})
    output = tmp_path / "timeline.png"
    plot_track_presence_timeline(track, sessions, output_path=output)
    assert output.is_file()
