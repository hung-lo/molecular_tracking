from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from endpoint_review_plots import plot_endpoint_review_panel, plot_track_presence_timeline


def _inputs(tmp_path: Path):
    red = np.ones((2, 12, 12), dtype=np.uint16)
    green = red * 2
    mask = np.zeros_like(red)
    mask[:, 5:7, 5:7] = 1
    red_path = tmp_path / "red.tif"; green_path = tmp_path / "green.tif"; mask_path = tmp_path / "mask.tif"
    tifffile.imwrite(red_path, red); tifffile.imwrite(green_path, green); tifffile.imwrite(mask_path, mask)
    sessions = pd.DataFrame([{"session_index": 0, "session_id": "s0", "acquisition_date": "2026-01-01", "mask_path": str(mask_path), "red_image_path": str(red_path), "green_image_path": str(green_path)}])
    features = pd.DataFrame([{"session_id": "s0", "label": 1, "centroid_z": 0, "centroid_y": 5.5, "centroid_x": 5.5}])
    endpoint = pd.Series({"endpoint_id": "endpoint_1", "track_uid": "s0:1", "end_session_index": 0, "end_session_id": "s0", "end_label": 1})
    return endpoint, sessions, features


def test_endpoint_panel_is_png_only_and_handles_empty_candidates(tmp_path: Path) -> None:
    endpoint, sessions, features = _inputs(tmp_path)
    output = tmp_path / "endpoint.png"
    plot_endpoint_review_panel(endpoint, sessions, features, pd.DataFrame(), pd.DataFrame(), output_path=output)
    assert output.is_file()
    assert not list(tmp_path.glob("*.pdf"))


def test_track_presence_timeline_handles_missing_session(tmp_path: Path) -> None:
    endpoint, sessions, _features = _inputs(tmp_path)
    track = pd.Series({"track_uid": "s0:1", "s0_roi": 1})
    output = tmp_path / "timeline.png"
    plot_track_presence_timeline(track, sessions, output_path=output)
    assert output.is_file()
