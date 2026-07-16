from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np
import tifffile

from run_daywise_graph_matching import run_daywise_graph_matching


def _build_dataset(tmp_path: Path) -> Path:
    mask = np.zeros((2, 3, 3), dtype=np.uint16)
    mask[0, 0, 0] = 1
    mask[0, 1, 1] = 2
    for day in ["20260511", "20260512"]:
        tifffile.imwrite(tmp_path / f"{day}_mask.tif", mask)
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["session_index", "session_id", "acquisition_date", "mask_path", "red_image_path", "green_image_path", "required"],
        )
        writer.writeheader()
        writer.writerow({"session_index": 0, "session_id": "20260511", "acquisition_date": "2026-05-11", "mask_path": str(tmp_path / "20260511_mask.tif"), "red_image_path": "", "green_image_path": "", "required": "true"})
        writer.writerow({"session_index": 1, "session_id": "20260512", "acquisition_date": "2026-05-12", "mask_path": str(tmp_path / "20260512_mask.tif"), "red_image_path": "", "green_image_path": "", "required": "true"})
    return manifest_path


def test_run_daywise_graph_matching_exports_graph_tables(tmp_path: Path) -> None:
    manifest_path = _build_dataset(tmp_path)
    output_dir = run_daywise_graph_matching(
        manifest_path=manifest_path,
        output_dir=tmp_path / "graph_out",
        overwrite=True,
        skip_qc=True,
    )

    assert (output_dir / "pairwise_matches_graph.csv").exists()
    assert (output_dir / "tracks_graph.csv").exists()
    assert (output_dir / "graph_match_changes.csv").exists()

    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["graph_matcher_algorithm_version"] == "local_spatial_graph_v1"
    assert run_log["graph_row_counts"]["tracks_graph"] > 0
    assert run_log["graph_output_paths"]["tracks_graph"].endswith("tracks_graph.csv")
