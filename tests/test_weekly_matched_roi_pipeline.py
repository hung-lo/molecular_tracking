from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from roi_log_ratio_analysis import build_registered_image_lookup
from run_registered_roi_pipeline import infer_start_date_from_dataset_dir
from run_weekly_matched_roi_pipeline import (
    WeeklyMatchedPipelineConfig,
    run_weekly_matched_roi_pipeline,
)


def _write_line_stack(path: Path, values: list[int]) -> None:
    stack = np.asarray(values, dtype=np.uint16).reshape(1, 1, len(values))
    tifffile.imwrite(path, stack)


def test_infer_start_date_from_dataset_dir_supports_crop_suffixes(tmp_path: Path) -> None:
    _write_line_stack(tmp_path / "20260512_R_crop_256.tif", [1, 2, 3])
    _write_line_stack(tmp_path / "20260511_G_crop_256.tif", [1, 2, 3])
    _write_line_stack(tmp_path / "20260511_R_crop_256_SyN.tif", [1, 2, 3])
    _write_line_stack(tmp_path / "20260511_R_ROI_mask_SyN_inversed.tif", [1, 2, 3])

    assert infer_start_date_from_dataset_dir(tmp_path) == "20260511"


def test_build_registered_image_lookup_supports_crop_names_and_day0_syn_mode(tmp_path: Path) -> None:
    for file_name in [
        "20260511_R_crop_256.tif",
        "20260511_G_crop_256.tif",
        "20260511_R_crop_256_SyN.tif",
        "20260511_G_crop_256_SyN.tif",
        "20260512_R_crop_256.tif",
        "20260512_G_crop_256.tif",
        "20260512_R_crop_256_SyN.tif",
        "20260512_G_crop_256_SyN.tif",
        "20260512_R_ROI_mask_SyN_inversed.tif",
    ]:
        _write_line_stack(tmp_path / file_name, [1, 2, 3])

    lookup = build_registered_image_lookup(tmp_path, start_date="20260511", day0_mode="syn")

    assert lookup[(0, "red")].name == "20260511_R_crop_256_SyN.tif"
    assert lookup[(0, "green")].name == "20260511_G_crop_256_SyN.tif"
    assert lookup[(1, "red")].name == "20260512_R_crop_256_SyN.tif"
    assert lookup[(1, "green")].name == "20260512_G_crop_256_SyN.tif"


def test_run_weekly_matched_roi_pipeline_end_to_end(tmp_path: Path) -> None:
    week1_mask = np.asarray([[[1, 2, 3]]], dtype=np.uint16)
    week2_mask = np.asarray([[[10, 20, 30]]], dtype=np.uint16)
    tifffile.imwrite(tmp_path / "week1_average_cp_masks.tif", week1_mask)
    tifffile.imwrite(tmp_path / "week2_average_cp_masks.tif", week2_mask)

    dates = ["20260511", "20260512", "20260518", "20260519"]
    red_values = {
        "20260511": [10, 30, 50],
        "20260512": [12, 32, 52],
        "20260518": [14, 34, 54],
        "20260519": [16, 36, 56],
    }
    green_values = {
        "20260511": [20, 40, 60],
        "20260512": [22, 42, 62],
        "20260518": [24, 44, 64],
        "20260519": [26, 0, 66],
    }

    for date_key in dates:
        _write_line_stack(tmp_path / f"{date_key}_R_crop_256.tif", red_values[date_key])
        _write_line_stack(tmp_path / f"{date_key}_G_crop_256.tif", green_values[date_key])
        _write_line_stack(tmp_path / f"{date_key}_R_crop_256_SyN.tif", red_values[date_key])
        _write_line_stack(tmp_path / f"{date_key}_G_crop_256_SyN.tif", green_values[date_key])

    match_table = pd.DataFrame(
        [
            {"cluster_id": 1, "week1_roi": 1, "week2_roi": 10},
            {"cluster_id": 2, "week1_roi": 2, "week2_roi": 20},
            {"cluster_id": 3, "week1_roi": 3, "week2_roi": 30},
        ]
    )
    match_csv = tmp_path / "matched_tracks.csv"
    match_table.to_csv(match_csv, index=False)

    output_dir = run_weekly_matched_roi_pipeline(
        WeeklyMatchedPipelineConfig(
            dataset=str(tmp_path),
            match_csv=str(match_csv),
            green_dark=0.0,
            red_dark=0.0,
        )
    )

    raw_table = pd.read_csv(output_dir / "weekly_matched_roi_intensity_results_raw.csv")
    complete_table = pd.read_csv(output_dir / "weekly_matched_roi_day_table_complete.csv")
    filter_counts = pd.read_csv(output_dir / "filter_step_counts_with_percentages.csv")
    week_assignments = pd.read_csv(output_dir / "week_assignments.csv")
    fit_summary = pd.read_csv(output_dir / "weekly_matched_daywise_green_red_linear_fit_summary.csv")

    assert set(raw_table["cluster_id"].unique()) == {1, 2, 3}
    assert set(complete_table["cluster_id"].unique()) == {1, 3}
    assert complete_table.groupby("cluster_id")["day"].nunique().to_dict() == {1: 4, 3: 4}
    assert filter_counts["count"].tolist() == [3, 3, 2]
    assert week_assignments.groupby("week_name")["day"].apply(list).to_dict() == {
        "week1": [0, 1],
        "week2": [7, 8],
    }
    assert fit_summary["day"].tolist() == [0, 1, 7, 8]

    cluster1_week2_green = raw_table.loc[
        (raw_table["cluster_id"] == 1)
        & (raw_table["day"] == 7)
        & (raw_table["channel"] == "green"),
        "mean_intensity",
    ].iloc[0]
    assert np.isclose(cluster1_week2_green, 24.0)

    cluster2_day8 = raw_table.loc[
        (raw_table["cluster_id"] == 2)
        & (raw_table["day"] == 8),
        ["channel", "mean_intensity"],
    ]
    assert set(cluster2_day8["channel"].tolist()) == {"red"}
