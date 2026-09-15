from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd
import tifffile

from run_daywise_graph_matching import run_daywise_graph_matching


def _build_dataset(tmp_path: Path) -> Path:
    mask = np.zeros((2, 3, 3), dtype=np.uint16)
    mask[0, 0, 0] = 1
    mask[0, 1, 1] = 2
    for day in ["20260511", "20260512", "20260513"]:
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
        writer.writerow({"session_index": 2, "session_id": "20260513", "acquisition_date": "2026-05-13", "mask_path": str(tmp_path / "20260513_mask.tif"), "red_image_path": "", "green_image_path": "", "required": "true"})
    return manifest_path


def test_run_daywise_graph_matching_exports_graph_tables(tmp_path: Path) -> None:
    manifest_path = _build_dataset(tmp_path)
    stages: list[tuple[str, float]] = []
    output_dir = run_daywise_graph_matching(
        manifest_path=manifest_path,
        output_dir=tmp_path / "graph_out",
        overwrite=True,
        skip_qc=True,
        stage_callback=lambda key, duration: stages.append((key, duration)),
    )

    assert (output_dir / "pairwise_matches_graph.csv").exists()
    assert (output_dir / "tracks_graph.csv").exists()
    assert (output_dir / "graph_match_changes.csv").exists()

    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["graph_matcher_algorithm_version"] == "local_spatial_graph_v1"
    assert run_log["graph_row_counts"]["tracks_graph"] > 0
    assert run_log["graph_output_paths"]["tracks_graph"].endswith("tracks_graph.csv")
    assert [key for key, _duration in stages] == [
        "daywise_affine_roi_matching",
        "graph_roi_matching",
    ]
    assert all(duration >= 0 for _key, duration in stages)


def test_graph_pair_workers_preserve_exact_scientific_outputs(tmp_path: Path) -> None:
    manifest_path = _build_dataset(tmp_path)
    outputs = [
        run_daywise_graph_matching(
            manifest_path=manifest_path,
            output_dir=tmp_path / f"graph_workers_{workers}",
            pair_workers=workers,
            overwrite=True,
            skip_qc=True,
        )
        for workers in (1, 2)
    ]
    scientific_csvs = [
        "pairwise_matches_graph.csv", "tracks_graph.csv", "cycle_consistency_graph.csv",
        "cycle_edge_checks_graph.csv", "track_edges_graph.csv", "track_length_summary_graph.csv",
        "graph_match_changes.csv",
    ]
    for filename in scientific_csvs:
        pd.testing.assert_frame_equal(pd.read_csv(outputs[0] / filename), pd.read_csv(outputs[1] / filename))
    left_summary = pd.read_csv(outputs[0] / "pairwise_summary_graph.csv").drop(columns="elapsed_sec")
    right_summary = pd.read_csv(outputs[1] / "pairwise_summary_graph.csv").drop(columns="elapsed_sec")
    pd.testing.assert_frame_equal(left_summary, right_summary)

    run_log = json.loads((outputs[1] / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["pair_workers"] == 2
    assert len(run_log["runtime_profile"]["graph_pair_timings_seconds"]) == 3
