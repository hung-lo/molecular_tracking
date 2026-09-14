from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tifffile

from affine_overlap_matcher import AffineOverlapParams, VoxelSpacing
from run_daywise_matched_roi_pipeline import (
    DaywiseMatchedPipelineConfig,
    SegmentationQCConfig,
    _apply_geometry_qc,
    _track_geometry_summary,
    run_daywise_matched_roi_pipeline,
)
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
    summary = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")

    assert raw["channel"].isin(["red", "green"]).all()
    assert complete.shape[0] == 12
    assert set(complete["match_policy"].astype(str)) == {"high", "balanced"}
    assert set(complete["elapsed_days"].astype(int)) == {0, 1}
    assert float(complete.loc[(complete["match_policy"] == "high") & (complete["roi_id"] == 1) & (complete["day"] == 0), "red"].iloc[0]) == 10.0
    assert float(complete.loc[(complete["match_policy"] == "high") & (complete["roi_id"] == 1) & (complete["day"] == 0), "green"].iloc[0]) == 40.0
    assert tracks.shape[0] == 6
    assert set(primary["match_policy"].astype(str)) == {"high"}
    assert set(balanced["match_policy"].astype(str)) == {"balanced"}
    assert not primary_full_qc.empty
    assert set(tracks["segmentation_qc_status"].astype(str)) == {"not_configured"}
    assert tracks["segmentation_qc_pass_fraction"].isna().all()
    assert tracks["segmentation_qc_pass_all_required_days"].isna().all()
    assert not tracks["segmentation_failure"].astype(bool).any()
    assert set(filter_counts["match_policy"].astype(str)) == {"high", "balanced"}
    assert filter_counts["step_order"].min() == 0
    assert set(filter_counts.loc[filter_counts["step"] == "segmentation_qc", "count"]) == {3}
    assert set(filter_counts.loc[filter_counts["step"] == "segmentation_qc", "step_status"]) == {"bypassed_not_configured"}
    assert run_log["segmentation_qc_status"] == "not_configured_bypassed"
    assert "segmentation_qc_not_configured" in run_log["warnings"]
    assert "Segmentation QC status: `not_configured_bypassed`" in summary
    assert run_log["output_paths"]["matched_track_qc_summary"].endswith("matched_track_qc_summary.csv")


def test_configured_segmentation_qc_still_filters_tracks(tmp_path: Path) -> None:
    manifest_path, match_dir = _build_dataset(tmp_path)
    output_dir = run_daywise_matched_roi_pipeline(
        DaywiseMatchedPipelineConfig(
            dataset=str(tmp_path),
            manifest=str(manifest_path),
            match_dir=str(match_dir),
            policies=("high",),
            green_dark=0.0,
            red_dark=0.0,
            min_volume_um3=3.0,
        )
    )

    tracks = pd.read_csv(output_dir / "matched_track_qc_summary.csv")
    primary_full_qc = pd.read_csv(output_dir / "primary_high_complete_full_qc.csv")
    filter_counts = pd.read_csv(output_dir / "filter_step_counts_with_percentages.csv")
    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))
    summary = (output_dir / "SUMMARY.md").read_text(encoding="utf-8")

    assert set(tracks["segmentation_qc_status"].astype(str)) == {"configured"}
    assert tracks["segmentation_failure"].astype(bool).all()
    assert primary_full_qc.empty
    assert filter_counts.loc[filter_counts["step"] == "segmentation_qc", "count"].tolist() == [0]
    assert filter_counts.loc[filter_counts["step"] == "segmentation_qc", "step_status"].tolist() == ["applied"]
    assert run_log["segmentation_qc_status"] == "configured_no_tracks_passed"
    assert "Segmentation QC status: `configured_no_tracks_passed`" in summary


def test_configured_segmentation_qc_reports_no_upstream_tracks(tmp_path: Path) -> None:
    manifest_path, match_dir = _build_dataset(tmp_path)
    tracks_path = match_dir / "tracks_high.csv"
    tracks = pd.read_csv(tracks_path)
    tracks["has_cycle_conflict"] = True
    tracks.to_csv(tracks_path, index=False)

    output_dir = run_daywise_matched_roi_pipeline(
        DaywiseMatchedPipelineConfig(
            dataset=str(tmp_path),
            manifest=str(manifest_path),
            match_dir=str(match_dir),
            policies=("high",),
            green_dark=0.0,
            red_dark=0.0,
            min_volume_um3=3.0,
        )
    )

    filter_counts = pd.read_csv(output_dir / "filter_step_counts_with_percentages.csv")
    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))

    assert filter_counts.loc[filter_counts["step"] == "cycle_qc", "count"].tolist() == [0]
    assert filter_counts.loc[filter_counts["step"] == "segmentation_qc", "count"].tolist() == [0]
    assert run_log["segmentation_qc_status"] == "configured_no_upstream_eligible_tracks"


def _geometry_fixture(volumes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "roi_id": [1, 1],
            "required": [True, True],
            "session_id": ["s0", "s1"],
            "session_index": [0, 1],
            "volume_um3": volumes,
            "bbox_depth_planes": [2, 2],
            "touches_xy_edge": [False, False],
            "touches_z_edge": [False, False],
        }
    )


def test_configured_segmentation_qc_passes_all_required_sessions() -> None:
    config = SegmentationQCConfig(min_volume_um3=3.0)
    summary = _track_geometry_summary(_apply_geometry_qc(_geometry_fixture([4.0, 4.0]), config), config)

    row = summary.iloc[0]
    assert row["segmentation_qc_status"] == "configured"
    assert row["segmentation_qc_pass_fraction"] == 1.0
    assert bool(row["segmentation_qc_pass_all_required_days"])


def test_all_required_segmentation_qc_rejects_one_failed_session() -> None:
    config = SegmentationQCConfig(min_volume_um3=3.0)
    summary = _track_geometry_summary(_apply_geometry_qc(_geometry_fixture([4.0, 2.0]), config), config)

    row = summary.iloc[0]
    assert row["segmentation_qc_pass_fraction"] == 0.5
    assert not bool(row["segmentation_qc_pass_all_required_days"])


def test_fraction_segmentation_qc_uses_inclusive_boundary() -> None:
    geometry = _geometry_fixture([4.0, 2.0])
    at_boundary = SegmentationQCConfig(mode="fraction", min_volume_um3=3.0, min_segmentation_pass_fraction=0.5)
    above_boundary = SegmentationQCConfig(mode="fraction", min_volume_um3=3.0, min_segmentation_pass_fraction=0.5001)

    boundary_row = _track_geometry_summary(_apply_geometry_qc(geometry, at_boundary), at_boundary).iloc[0]
    above_row = _track_geometry_summary(_apply_geometry_qc(geometry, above_boundary), above_boundary).iloc[0]

    assert bool(boundary_row["segmentation_qc_pass_all_required_days"])
    assert not bool(above_row["segmentation_qc_pass_all_required_days"])


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
    assert complete.shape[0] == 6
    assert set(complete["elapsed_days"].astype(int)) == {0, 1}
    assert set(fit_summary["normalization_population"]) == {"all_valid_session_rois"}
    assert run_log["output_paths"]["matched_daywise_green_red_linear_fit_summary"].endswith("matched_daywise_green_red_linear_fit_summary.csv")


def test_track_attrition_does_not_change_native_session_fit(tmp_path: Path) -> None:
    manifest_path, match_dir = _build_dataset(tmp_path)
    baseline_dir = run_daywise_matched_roi_pipeline(DaywiseMatchedPipelineConfig(
        dataset=str(tmp_path), manifest=str(manifest_path), match_dir=str(match_dir),
        output_root=str(tmp_path / "baseline"), policies=("high",), green_dark=0, red_dark=0,
    ))
    tracks_path = match_dir / "tracks_high.csv"
    tracks = pd.read_csv(tracks_path)
    tracks.loc[0, "20260512_roi"] = np.nan
    tracks.to_csv(tracks_path, index=False)
    attrited_dir = run_daywise_matched_roi_pipeline(DaywiseMatchedPipelineConfig(
        dataset=str(tmp_path), manifest=str(manifest_path), match_dir=str(match_dir),
        output_root=str(tmp_path / "attrited"), policies=("high",), green_dark=0, red_dark=0,
    ))

    pd.testing.assert_frame_equal(
        pd.read_csv(baseline_dir / "matched_daywise_green_red_linear_fit_summary.csv"),
        pd.read_csv(attrited_dir / "matched_daywise_green_red_linear_fit_summary.csv"),
    )
    all_rows = pd.read_csv(attrited_dir / "matched_roi_day_table_all.csv")
    complete_rows = pd.read_csv(attrited_dir / "matched_roi_day_table_complete.csv")
    population = pd.read_csv(attrited_dir / "matched_session_population_roi_metrics.csv")
    assert len(all_rows) == 5
    assert len(complete_rows) == 4
    assert len(population) == 6
