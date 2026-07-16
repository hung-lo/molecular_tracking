from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tifffile

from affine_overlap_matcher import AffineOverlapParams, VoxelSpacing
from run_daywise_matched_roi_pipeline import DaywiseMatchedPipelineConfig, run_daywise_matched_roi_pipeline
from run_daywise_graph_matching import run_daywise_graph_matching
from run_daywise_roi_matching import run_daywise_roi_matching


def _write_stack(path: Path, data: np.ndarray) -> None:
    tifffile.imwrite(path, data.astype(np.uint16))


def _build_dataset(tmp_path: Path) -> tuple[Path, Path]:
    mask = np.zeros((2, 3, 3), dtype=np.uint16)
    mask[0, 0, 0] = 1
    mask[0, 1, 1] = 2
    mask[1, 2, 2] = 3
    for day in ["20260511", "20260512"]:
        _write_stack(tmp_path / f"{day}_mask.tif", mask)
        red = np.zeros((2, 3, 3), dtype=np.uint16)
        green = np.zeros((2, 3, 3), dtype=np.uint16)
        red[mask == 1] = 10
        red[mask == 2] = 20
        red[mask == 3] = 30
        green[mask == 1] = 40
        green[mask == 2] = 50
        green[mask == 3] = 60
        _write_stack(tmp_path / f"{day}_R.tif", red)
        _write_stack(tmp_path / f"{day}_G.tif", green)

    manifest = pd.DataFrame(
        [
            {
                "session_index": 0,
                "session_id": "20260511",
                "acquisition_date": "2026-05-11",
                "mask_path": str(tmp_path / "20260511_mask.tif"),
                "red_image_path": str(tmp_path / "20260511_R.tif"),
                "green_image_path": str(tmp_path / "20260511_G.tif"),
                "required": True,
            },
            {
                "session_index": 1,
                "session_id": "20260512",
                "acquisition_date": "2026-05-12",
                "mask_path": str(tmp_path / "20260512_mask.tif"),
                "red_image_path": str(tmp_path / "20260512_R.tif"),
                "green_image_path": str(tmp_path / "20260512_G.tif"),
                "required": True,
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    match_dir = tmp_path / "match_out"
    run_daywise_roi_matching(
        manifest_path=manifest_path,
        output_dir=match_dir,
        spacing=VoxelSpacing(),
        params=AffineOverlapParams(),
        save_candidates=False,
        overwrite=True,
    )
    return manifest_path, match_dir


def test_run_daywise_matched_roi_pipeline_exports_expected_tables(tmp_path: Path) -> None:
    manifest_path, match_dir = _build_dataset(tmp_path)
    output_dir = run_daywise_matched_roi_pipeline(
        DaywiseMatchedPipelineConfig(
            dataset=str(tmp_path),
            manifest=str(manifest_path),
            match_dir=str(match_dir),
            green_dark=0.0,
            red_dark=0.0,
        )
    )

    raw = pd.read_csv(output_dir / "matched_roi_intensity_results_raw.csv")
    complete = pd.read_csv(output_dir / "matched_roi_day_table_complete.csv")
    tracks = pd.read_csv(output_dir / "matched_track_qc_summary.csv")
    primary = pd.read_csv(output_dir / "primary_high_complete_matching.csv")
    balanced = pd.read_csv(output_dir / "sensitivity_balanced_complete.csv")
    primary_full_qc = pd.read_csv(output_dir / "primary_high_complete_full_qc.csv")
    filter_counts = pd.read_csv(output_dir / "filter_step_counts_with_percentages.csv")
    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))

    assert raw["channel"].isin(["red", "green"]).all()
    assert complete.shape[0] == 12
    assert set(complete["match_policy"].astype(str)) == {"high", "balanced"}
    assert set(complete["elapsed_days"].astype(int)) == {0, 1}
    assert float(complete.loc[(complete["match_policy"] == "high") & (complete["roi_id"] == 1) & (complete["day"] == 0), "red"].iloc[0]) == 10.0
    assert float(complete.loc[(complete["match_policy"] == "high") & (complete["roi_id"] == 1) & (complete["day"] == 0), "green"].iloc[0]) == 40.0
    assert tracks.shape[0] == 6
    assert set(primary["match_policy"].astype(str)) == {"high"}
    assert set(balanced["match_policy"].astype(str)) == {"balanced"}
    assert primary_full_qc.empty
    assert set(filter_counts["match_policy"].astype(str)) == {"high", "balanced"}
    assert filter_counts["step_order"].min() == 0
    assert run_log["output_paths"]["matched_track_qc_summary"].endswith("matched_track_qc_summary.csv")


def test_run_daywise_matched_roi_pipeline_accepts_graph_policy(tmp_path: Path) -> None:
    manifest_path, _ = _build_dataset(tmp_path)
    graph_match_dir = run_daywise_graph_matching(
        manifest_path=manifest_path,
        output_dir=tmp_path / "graph_match_out",
        overwrite=True,
        skip_qc=True,
    )
    output_dir = run_daywise_matched_roi_pipeline(
        DaywiseMatchedPipelineConfig(
            dataset=str(tmp_path),
            manifest=str(manifest_path),
            match_dir=str(graph_match_dir),
            policies=("graph",),
            green_dark=0.0,
            red_dark=0.0,
        )
    )

    complete = pd.read_csv(output_dir / "matched_roi_day_table_complete.csv")
    fit_summary = pd.read_csv(output_dir / "matched_daywise_green_red_linear_fit_summary.csv")
    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))

    assert set(complete["match_policy"].astype(str)) == {"graph"}
    assert complete.shape[0] == 4
    assert set(complete["elapsed_days"].astype(int)) == {0, 1}
    assert set(fit_summary["match_policy"].astype(str)) == {"graph"}
    assert run_log["output_paths"]["matched_daywise_green_red_linear_fit_summary"].endswith("matched_daywise_green_red_linear_fit_summary.csv")
