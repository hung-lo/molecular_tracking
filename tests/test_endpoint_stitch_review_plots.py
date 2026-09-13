from __future__ import annotations

import pandas as pd
import numpy as np
import tifffile

from affine_overlap_matcher import VoxelSpacing
from endpoint_stitch_review_plots import plot_stitch_review_panels


def test_review_outputs_png_only_and_handle_missing_coordinates(tmp_path) -> None:
    proposals = pd.DataFrame([{
        "stitch_edge_id": "stitch_1", "source_track_uid": "a", "target_track_uid": "b",
        "source_session_id": "s0", "target_session_id": "s3", "session_gap": 3,
        "candidate_tier": "manual_review", "projected_distance_um": 2.0,
        "review_reasons": "insufficient_target_future", "rejection_reasons": "",
    }])
    paths = plot_stitch_review_panels(proposals, tmp_path, max_panels=2)
    assert len(paths) == 1 and paths[0].suffix == ".png"
    assert (tmp_path / "stitch_contact_sheet.png").is_file()
    assert not list(tmp_path.glob("*.pdf"))


def test_review_panel_renders_available_edge_crops(tmp_path) -> None:
    image = np.zeros((2, 12, 12), dtype=np.uint16); image[:, :3, :3] = 20
    mask = np.zeros_like(image); mask[:, 0:2, 0:2] = 1
    manifest_rows = []
    for index in range(2):
        red, green, labels = (tmp_path / f"r{index}.tif", tmp_path / f"g{index}.tif", tmp_path / f"m{index}.tif")
        tifffile.imwrite(red, image); tifffile.imwrite(green, image); tifffile.imwrite(labels, mask * (index + 1))
        manifest_rows.append({"session_index": index, "session_id": f"s{index}", "red_image_path": red, "green_image_path": green, "mask_path": labels})
    proposals = pd.DataFrame([{
        "stitch_edge_id": "edge_crop", "source_track_uid": "a", "target_track_uid": "b",
        "source_session_index": 0, "target_session_index": 1, "source_session_id": "s0", "target_session_id": "s1",
        "source_label": 1, "target_label": 2, "session_gap": 1, "candidate_tier": "manual_review",
        "projected_distance_um": 0, "review_reasons": "", "rejection_reasons": "",
    }])
    features = pd.DataFrame([
        {"session_id": "s0", "label": 1, "centroid_z": 0, "centroid_y": 0, "centroid_x": 0},
        {"session_id": "s1", "label": 2, "centroid_z": 0, "centroid_y": 0, "centroid_x": 0},
    ])
    transforms = pd.DataFrame([{
        "day_a": "s0", "day_b": "s1", "z_intercept": 0, "z_scale": 1,
        "y_intercept": 0, "y_from_y": 1, "y_from_x": 0,
        "x_intercept": 0, "x_from_y": 0, "x_from_x": 1,
    }])
    paths = plot_stitch_review_panels(
        proposals, tmp_path / "panels", sessions=pd.DataFrame(manifest_rows),
        features=features, transforms=transforms, spacing=VoxelSpacing(z_um=1, y_um=1, x_um=1),
    )
    assert paths[0].is_file()
