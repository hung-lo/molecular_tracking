import json
import numpy as np
import pandas as pd
from pathlib import Path
import tifffile

from analysis_paths import get_dataset_analysis_dir, get_shape_qc_analysis_dir, resolve_dataset_dir
from run_registered_roi_pipeline import (
    RegisteredPipelineConfig,
    format_duration_seconds,
    infer_default_inverse_mask_channel,
    infer_start_date_from_dataset_dir,
    parse_args,
    unique_sorted_days_from_channel_lookup,
    write_run_log,
)
from raw_space_triplet_panels import (
    build_composite_rgb,
    measure_roi_geometry,
    render_raw_space_triplet_panel,
)
from shared_raw_space_group_panel import (
    compute_shared_raw_space_group_geometry,
    render_shared_raw_space_group_panel,
)

from roi_log_ratio_analysis import (
    attach_roi_size_metrics,
    add_day0_normalized_column,
    apply_channel_dark_correction,
    build_inverse_warped_mask_lookup,
    build_raw_image_lookup,
    classify_roi_log_ratio_trajectories,
    compute_roi_size_table,
    compute_log_ratio_metrics,
    compute_green_red_fit_residuals,
    compute_roi_crop_bounds,
    extract_registered_dataset_roi_intensity_table,
    estimate_size_filter_bounds,
    extract_roi_mean_intensities,
    extract_day_from_image_name,
    flag_shape_qc_rois,
    filter_complete_rois,
    prepare_stack_for_display,
    project_roi_stack_view,
    select_ranked_roi_days,
    select_roi_neighbor_z_slices,
    select_top_changing_rois,
    summarize_residual_sign_changes,
    summarize_daily_green_red_linear_fits,
    summarize_roi_metrics,
    wide_table_from_long_table,
    build_registered_image_lookup,
)


def make_long_table() -> pd.DataFrame:
    rows = []
    for day, date_key in enumerate(["20260511", "20260512", "20260513"]):
        red_values = {1: [100, 102, 98], 2: [120, 121, 119], 3: [140, 141, 142]}
        green_values = {1: [40, 20, 5], 2: [60, 61, 60]}

        for roi_id, roi_red_values in red_values.items():
            rows.append(
                {
                    "roi_id": roi_id,
                    "mean_intensity_corrected": float(roi_red_values[day]),
                    "image": f"{date_key}_R_SyN.tif" if day > 0 else f"{date_key}_R.tif",
                    "channel": "red",
                }
            )

        for roi_id, roi_green_values in green_values.items():
            rows.append(
                {
                    "roi_id": roi_id,
                    "mean_intensity_corrected": float(roi_green_values[day]),
                    "image": f"{date_key}_G_SyN.tif" if day > 0 else f"{date_key}_G.tif",
                    "channel": "green",
                }
            )

    return pd.DataFrame(rows)


def test_resolve_dataset_dir_supports_default_and_aliases() -> None:
    project_root = Path("/mnt/d/_data/_newAAV_2026/Fucci-Tri_avg_images")

    assert resolve_dataset_dir().name == "1050_data"
    assert resolve_dataset_dir("1050") == project_root / "1050_data"
    assert resolve_dataset_dir("920") == project_root / "920_data"


def test_dataset_analysis_helpers_build_expected_subdirectories() -> None:
    project_root = Path("/mnt/d/_data/_newAAV_2026/Fucci-Tri_avg_images")

    assert get_dataset_analysis_dir("1050") == project_root / "1050_data" / "analysis"
    assert get_shape_qc_analysis_dir("1050") == (
        project_root
        / "1050_data"
        / "analysis"
        / "roi_log_ratio_outputs_dark_median_corrected_meanMergeCPSAM_ROIs"
        / "shape_qc_filter"
    )


def test_infer_start_date_from_dataset_dir_uses_earliest_raw_tiff(tmp_path: Path) -> None:
    tifffile.imwrite(tmp_path / "20260527_R.tif", np.zeros((2, 2, 2), dtype=np.uint16))
    tifffile.imwrite(tmp_path / "20260526_G.tif", np.zeros((2, 2, 2), dtype=np.uint16))
    tifffile.imwrite(tmp_path / "20260528_R_SyN.tif", np.zeros((2, 2, 2), dtype=np.uint16))

    assert infer_start_date_from_dataset_dir(tmp_path) == "20260526"


def test_extract_day_from_image_name() -> None:
    assert extract_day_from_image_name("20260511_R.tif") == 0
    assert extract_day_from_image_name("20260515_G_SyN.tif") == 4


def test_build_registered_image_lookup_supports_custom_start_date(tmp_path: Path) -> None:
    for file_name in [
        "20260526_R.tif",
        "20260526_G.tif",
        "20260527_R_SyN.tif",
        "20260527_G_SyN.tif",
        "mean_image_R_SyN.tif",
    ]:
        tifffile.imwrite(tmp_path / file_name, np.zeros((2, 2, 2), dtype=np.uint16))

    lookup = build_registered_image_lookup(tmp_path, start_date="20260526")

    assert lookup[(0, "red")].name == "20260526_R.tif"
    assert lookup[(0, "green")].name == "20260526_G.tif"
    assert lookup[(1, "red")].name == "20260527_R_SyN.tif"
    assert lookup[(1, "green")].name == "20260527_G_SyN.tif"


def test_build_registered_image_lookup_ignores_inverse_mask_tiffs(tmp_path: Path) -> None:
    for file_name in [
        "20260526_R.tif",
        "20260526_G.tif",
        "20260527_R_SyN.tif",
        "20260527_G_SyN.tif",
        "20260527_R_ROI_mask_SyN_inversed.tif",
        "20260527_G_ROI_mask_SyN_inversed.tif",
    ]:
        tifffile.imwrite(tmp_path / file_name, np.zeros((2, 2, 2), dtype=np.uint16))

    lookup = build_registered_image_lookup(tmp_path, start_date="20260526")

    assert lookup[(1, "red")].name == "20260527_R_SyN.tif"
    assert lookup[(1, "green")].name == "20260527_G_SyN.tif"


def test_extract_registered_dataset_roi_intensity_table_handles_two_day_dataset(
    tmp_path: Path,
) -> None:
    mask_stack = np.array(
        [
            [[1, 1], [0, 2]],
            [[1, 1], [0, 2]],
        ],
        dtype=np.uint16,
    )
    tifffile.imwrite(tmp_path / "mask.tif", mask_stack)

    image_data = {
        "20260526_R.tif": np.array([[[10, 14], [0, 30]], [[12, 16], [0, 32]]], dtype=np.uint16),
        "20260526_G.tif": np.array([[[40, 44], [0, 20]], [[42, 46], [0, 24]]], dtype=np.uint16),
        "20260527_R_SyN.tif": np.array([[[20, 24], [0, 34]], [[22, 26], [0, 36]]], dtype=np.uint16),
        "20260527_G_SyN.tif": np.array([[[50, 54], [0, 18]], [[52, 56], [0, 22]]], dtype=np.uint16),
    }
    for file_name, stack in image_data.items():
        tifffile.imwrite(tmp_path / file_name, stack)

    extracted = extract_registered_dataset_roi_intensity_table(
        image_dir=tmp_path,
        mask_path=tmp_path / "mask.tif",
        start_date="20260526",
    )

    extracted = extracted.sort_values(["roi_id", "day", "channel"]).reset_index(drop=True)

    assert len(extracted) == 8
    assert extracted["day"].tolist() == [0, 0, 1, 1, 0, 0, 1, 1]
    assert extracted["channel"].tolist() == [
        "green",
        "red",
        "green",
        "red",
        "green",
        "red",
        "green",
        "red",
    ]

    roi1_day0_red = extracted[
        (extracted["roi_id"] == 1) & (extracted["day"] == 0) & (extracted["channel"] == "red")
    ].iloc[0]
    roi2_day1_green = extracted[
        (extracted["roi_id"] == 2) & (extracted["day"] == 1) & (extracted["channel"] == "green")
    ].iloc[0]

    assert np.isclose(roi1_day0_red["mean_intensity"], 13.0)
    assert np.isclose(roi2_day1_green["mean_intensity"], 20.0)


def test_filter_complete_rois_keeps_only_full_day_rois() -> None:
    long_table = make_long_table()
    wide_table = wide_table_from_long_table(long_table)
    filtered = filter_complete_rois(wide_table)

    assert set(filtered["roi_id"].unique()) == {1, 2}
    assert filtered.groupby("roi_id")["day"].nunique().to_dict() == {1: 3, 2: 3}


def test_filter_complete_rois_allows_extra_days_when_required_subset_is_given() -> None:
    roi_day_table = pd.DataFrame(
        {
            "roi_id": [1, 1, 1, 2, 2, 3, 3],
            "day": [0, 1, 2, 0, 1, 0, 2],
            "red": [10.0, 11.0, 12.0, 20.0, 21.0, 30.0, 31.0],
            "green": [5.0, 5.5, 6.0, 7.0, 7.5, 8.0, 8.5],
        }
    )

    filtered = filter_complete_rois(roi_day_table, required_days=[0, 1])

    assert set(filtered["roi_id"].unique()) == {1, 2}
    assert filtered.groupby("roi_id")["day"].nunique().to_dict() == {1: 3, 2: 2}


def test_build_inverse_warped_mask_lookup_prefers_requested_channel(tmp_path: Path) -> None:
    tifffile.imwrite(tmp_path / "mean_image_merge_cp_masks_SAM.tif", np.zeros((2, 2, 2), dtype=np.uint16))
    tifffile.imwrite(tmp_path / "20260527_R_ROI_mask_SyN_inversed.tif", np.ones((2, 2, 2), dtype=np.uint16))
    tifffile.imwrite(tmp_path / "20260527_G_ROI_mask_SyN_inversed.tif", np.ones((2, 2, 2), dtype=np.uint16))

    lookup = build_inverse_warped_mask_lookup(
        image_dir=tmp_path,
        day0_mask_name="mean_image_merge_cp_masks_SAM.tif",
        inverse_mask_suffix="_ROI_mask_SyN_inversed.tif",
        preferred_channel="green",
        start_date="20260526",
    )

    assert lookup[0].name == "mean_image_merge_cp_masks_SAM.tif"
    assert lookup[1].name == "20260527_G_ROI_mask_SyN_inversed.tif"


def test_unique_sorted_days_from_channel_lookup_removes_duplicates() -> None:
    lookup = {
        (0, "red"): "r0",
        (0, "green"): "g0",
        (1, "red"): "r1",
        (1, "green"): "g1",
        (2, "red"): "r2",
    }

    assert unique_sorted_days_from_channel_lookup(lookup) == [0, 1, 2]


def test_infer_default_inverse_mask_channel_uses_dataset_aliases() -> None:
    assert infer_default_inverse_mask_channel("1050") == "red"
    assert infer_default_inverse_mask_channel("920") == "green"


def test_log_ratio_metrics_match_expected_values() -> None:
    long_table = make_long_table()
    wide_table = filter_complete_rois(wide_table_from_long_table(long_table))
    metrics = compute_log_ratio_metrics(wide_table, epsilon=1.0)

    roi1_day0 = metrics[(metrics["roi_id"] == 1) & (metrics["day"] == 0)].iloc[0]
    roi1_day2 = metrics[(metrics["roi_id"] == 1) & (metrics["day"] == 2)].iloc[0]

    expected_day0 = np.log2((40.0 + 1.0) / (100.0 + 1.0))
    expected_day2 = np.log2((5.0 + 1.0) / (98.0 + 1.0))

    assert np.isclose(roi1_day0["brightness"], 140.0)
    assert np.isclose(roi1_day0["log2_green_over_red"], expected_day0)
    assert np.isclose(roi1_day2["log2_green_over_red"], expected_day2)
    assert np.isclose(roi1_day0["delta_log2_green_over_red"], 0.0)
    assert np.isclose(
        roi1_day2["delta_log2_green_over_red"], expected_day2 - expected_day0
    )


def test_day0_normalized_delta_is_zero_on_day0() -> None:
    long_table = make_long_table()
    wide_table = filter_complete_rois(wide_table_from_long_table(long_table))
    metrics = compute_log_ratio_metrics(wide_table, epsilon=1.0)

    day0_rows = metrics[metrics["day"] == 0]
    assert np.allclose(day0_rows["delta_log2_green_over_red"].to_numpy(), 0.0)


def test_green_values_can_be_normalized_to_day0() -> None:
    long_table = make_long_table()
    wide_table = filter_complete_rois(wide_table_from_long_table(long_table))
    metrics = compute_log_ratio_metrics(wide_table, epsilon=1.0)
    normalized = add_day0_normalized_column(metrics, source_column="green")

    roi1_day0 = normalized[(normalized["roi_id"] == 1) & (normalized["day"] == 0)].iloc[0]
    roi1_day1 = normalized[(normalized["roi_id"] == 1) & (normalized["day"] == 1)].iloc[0]
    roi1_day2 = normalized[(normalized["roi_id"] == 1) & (normalized["day"] == 2)].iloc[0]

    assert np.isclose(roi1_day0["green_day0"], 40.0)
    assert np.isclose(roi1_day0["green_normalized_to_day0"], 1.0)
    assert np.isclose(roi1_day1["green_normalized_to_day0"], 0.5)
    assert np.isclose(roi1_day2["green_normalized_to_day0"], 0.125)


def test_channel_dark_correction_uses_expected_mapping_without_clipping() -> None:
    raw_table = pd.DataFrame(
        {
            "roi_id": [1, 1, 2, 2],
            "image": ["20260511_G.tif", "20260511_R.tif", "20260512_G_SyN.tif", "20260512_R_SyN.tif"],
            "channel": ["green", "red", "green", "red"],
            "mean_intensity": [330.0, 600.0, 300.0, 520.0],
        }
    )

    corrected = apply_channel_dark_correction(
        raw_table,
        green_dark=319.0,
        red_dark=534.0,
        intensity_column="mean_intensity",
        corrected_column="mean_intensity_corrected",
        clip_floor=None,
    )

    assert corrected["mean_intensity_corrected"].tolist() == [11.0, 66.0, -19.0, -14.0]


def test_display_dark_subtraction_clips_negative_pixels_only_for_display() -> None:
    stack = np.array(
        [
            [[10.0, 20.0], [40.0, 50.0]],
            [[30.0, 5.0], [60.0, 80.0]],
        ]
    )

    display_stack = prepare_stack_for_display(stack, dark_value=25.0, clip_floor=0.0)

    expected = np.array(
        [
            [[0.0, 0.0], [15.0, 25.0]],
            [[5.0, 0.0], [35.0, 55.0]],
        ]
    )
    assert np.allclose(display_stack, expected)


def test_roi_mean_extraction_respects_zero_exclusion() -> None:
    mask_stack = np.array(
        [
            [[1, 1, 0], [0, 2, 2]],
            [[1, 1, 0], [0, 2, 2]],
        ],
        dtype=np.uint16,
    )
    image_stack = np.array(
        [
            [[10.0, 14.0, 0.0], [0.0, 20.0, 22.0]],
            [[12.0, 16.0, 0.0], [0.0, 0.0, 24.0]],
        ]
    )

    rows = extract_roi_mean_intensities(image_stack, mask_stack, exclude_zero_pixels=True)

    assert len(rows) == 1
    assert rows[0]["roi_id"] == 1
    assert np.isclose(rows[0]["mean_intensity"], 13.0)


def test_roi_mean_extraction_without_zero_exclusion_keeps_all_rois() -> None:
    mask_stack = np.array(
        [
            [[1, 1, 0], [0, 2, 2]],
            [[1, 1, 0], [0, 2, 2]],
        ],
        dtype=np.uint16,
    )
    image_stack = np.array(
        [
            [[10.0, 14.0, 0.0], [0.0, 20.0, 22.0]],
            [[12.0, 16.0, 0.0], [0.0, 0.0, 24.0]],
        ]
    )

    rows = extract_roi_mean_intensities(image_stack, mask_stack, exclude_zero_pixels=False)
    by_roi = {row["roi_id"]: row["mean_intensity"] for row in rows}

    assert set(by_roi) == {1, 2}
    assert np.isclose(by_roi[1], 13.0)
    assert np.isclose(by_roi[2], 16.5)


def test_candidate_ranking_prefers_large_green_drop_with_stable_red() -> None:
    long_table = make_long_table()
    wide_table = filter_complete_rois(wide_table_from_long_table(long_table))
    metrics = compute_log_ratio_metrics(wide_table, epsilon=1.0)
    summary = summarize_roi_metrics(metrics)
    selected = select_top_changing_rois(
        summary,
        max_rois=2,
        red_cv_max=0.1,
        min_day0_brightness_quantile=0.0,
    )

    assert selected["roi_id"].tolist() == [1, 2]
    assert selected["selection_rank"].tolist() == [1, 2]
    assert selected.iloc[0]["min_delta_log2_green_over_red"] < selected.iloc[1][
        "min_delta_log2_green_over_red"
    ]


def test_candidate_ranking_can_select_increasing_ratio_rois() -> None:
    summary = pd.DataFrame(
        {
            "roi_id": [1, 2, 3],
            "day0_brightness": [100.0, 120.0, 130.0],
            "day0_green": [10.0, 20.0, 20.0],
            "red_cv": [0.04, 0.05, 0.03],
            "min_delta_log2_green_over_red": [-0.4, -0.2, -0.1],
            "max_delta_log2_green_over_red": [0.3, 1.2, 0.7],
            "delta_log2_range": [0.7, 1.4, 0.8],
        }
    )

    selected = select_top_changing_rois(
        summary,
        max_rois=2,
        red_cv_max=0.1,
        min_day0_brightness_quantile=0.0,
        direction="increasing",
    )

    assert selected["roi_id"].tolist() == [2, 3]
    assert selected["selection_rank"].tolist() == [1, 2]
    assert np.all(selected["selection_direction"] == "increasing")


def test_candidate_ranking_can_randomly_sample_a_fixed_subset() -> None:
    summary = pd.DataFrame(
        {
            "roi_id": [1, 2, 3, 4, 5, 6],
            "day0_brightness": [100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            "day0_green": [10.0, 20.0, 20.0, 30.0, 25.0, 35.0],
            "red_cv": [0.04, 0.05, 0.03, 0.02, 0.06, 0.01],
            "min_delta_log2_green_over_red": [-0.6, -0.4, -0.2, -0.1, 0.0, 0.1],
            "max_delta_log2_green_over_red": [0.2, 0.5, 0.7, 0.9, 1.1, 1.4],
            "delta_log2_range": [0.8, 0.9, 0.9, 1.0, 1.1, 1.3],
        }
    )

    first = select_top_changing_rois(
        summary,
        max_rois=3,
        red_cv_max=0.1,
        min_day0_brightness_quantile=0.0,
        random_sample=True,
        random_seed=7,
    )
    second = select_top_changing_rois(
        summary,
        max_rois=3,
        red_cv_max=0.1,
        min_day0_brightness_quantile=0.0,
        random_sample=True,
        random_seed=7,
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3
    assert set(first["roi_id"]).issubset({1, 2, 3, 4, 5, 6})
    assert set(first["selection_metric_column"].astype(str)) == {"random_sample"}
    assert set(first["selection_random_seed"].astype(int)) == {7}


def test_size_thresholds_and_size_merge_support_roi_filtering() -> None:
    roi_size_table = pd.DataFrame(
        {
            "roi_id": [1, 2, 3, 4, 5],
            "proj_area_px": [10, 20, 30, 40, 500],
            "proj_area_um2": [1.0, 2.0, 3.0, 4.0, 50.0],
            "eq_diam_px": [3.0, 5.0, 6.0, 7.0, 25.0],
            "eq_diam_um": [2.0, 3.0, 4.0, 5.0, 17.0],
        }
    )
    summary = pd.DataFrame(
        {
            "roi_id": [1, 2, 3, 4, 5],
            "day0_brightness": [100.0, 110.0, 120.0, 130.0, 140.0],
            "day0_green": [20.0, 20.0, 20.0, 20.0, 20.0],
            "red_cv": [0.05, 0.05, 0.05, 0.05, 0.05],
            "min_delta_log2_green_over_red": [-1.0, -0.9, -0.8, -0.7, -0.6],
            "max_delta_log2_green_over_red": [0.1, 0.2, 0.3, 0.4, 0.5],
            "delta_log2_range": [1.1, 1.1, 1.1, 1.1, 1.1],
        }
    )

    bounds = estimate_size_filter_bounds(
        roi_size_table,
        area_column="proj_area_px",
        lower_quantile=0.20,
        upper_iqr_multiplier=1.5,
    )
    merged = attach_roi_size_metrics(summary, roi_size_table)
    selected = select_top_changing_rois(
        merged,
        max_rois=5,
        red_cv_max=0.1,
        min_day0_brightness_quantile=0.0,
        direction="decreasing",
        min_proj_area_px=bounds["lower_bound"],
        max_proj_area_px=bounds["upper_bound"],
    )

    assert bounds["lower_bound"] == 18.0
    assert bounds["upper_bound"] == 70.0
    assert selected["roi_id"].tolist() == [2, 3, 4]
    assert np.all(selected["proj_area_px"].between(18.0, 70.0))


def test_roi_size_table_reports_projected_shape_metrics() -> None:
    mask_stack = np.zeros((2, 12, 14), dtype=np.uint16)
    mask_stack[:, 2:6, 2:6] = 1
    mask_stack[:, 2:9, 9:10] = 2
    mask_stack[:, 7:8, 6:10] = 2

    roi_size_table = compute_roi_size_table(mask_stack, xy_um_per_px=1.0, z_um_per_plane=5.0)

    expected_columns = {
        "roi_id",
        "proj_area_px",
        "eq_diam_px",
        "proj_perimeter_px",
        "circularity",
        "solidity",
        "eccentricity",
        "axis_ratio",
    }
    assert expected_columns.issubset(set(roi_size_table.columns))

    square = roi_size_table.loc[roi_size_table["roi_id"] == 1].iloc[0]
    irregular = roi_size_table.loc[roi_size_table["roi_id"] == 2].iloc[0]

    assert square["circularity"] > irregular["circularity"]
    assert square["solidity"] > irregular["solidity"]
    assert square["axis_ratio"] < irregular["axis_ratio"]


def test_shape_qc_flags_rounder_rois_and_rejects_irregular_ones() -> None:
    roi_size_table = pd.DataFrame(
        {
            "roi_id": [1, 2, 3],
            "proj_area_px": [120, 140, 110],
            "circularity": [0.90, 0.50, 0.80],
            "solidity": [0.95, 0.70, 0.92],
            "axis_ratio": [1.20, 2.60, 1.80],
        }
    )

    flagged = flag_shape_qc_rois(
        roi_size_table,
        min_circularity=0.65,
        min_solidity=0.80,
        max_axis_ratio=2.0,
    )

    assert flagged["shape_qc_pass"].tolist() == [True, False, True]
    assert np.allclose(flagged["shape_qc_min_circularity"], 0.65)
    assert np.allclose(flagged["shape_qc_min_solidity"], 0.80)
    assert np.allclose(flagged["shape_qc_max_axis_ratio"], 2.0)


def test_build_inverse_warped_mask_lookup_maps_day0_and_moving_days(tmp_path) -> None:
    day0_mask = tmp_path / "mean_image_merge_cp_masks_SAM.tif"
    day1_mask = tmp_path / "20260512_R_ROI_mask_SyN_inversed.tif"
    day2_mask = tmp_path / "20260513_R_ROI_mask_SyN_inversed.tif"
    day0_mask.write_bytes(b"")
    day1_mask.write_bytes(b"")
    day2_mask.write_bytes(b"")

    lookup = build_inverse_warped_mask_lookup(tmp_path)

    assert lookup[0] == day0_mask
    assert lookup[1] == day1_mask
    assert lookup[2] == day2_mask


def test_build_raw_image_lookup_ignores_registered_images(tmp_path) -> None:
    raw_day0_red = tmp_path / "20260511_R.tif"
    raw_day1_red = tmp_path / "20260512_R.tif"
    raw_day1_green = tmp_path / "20260512_G.tif"
    registered_red = tmp_path / "20260512_R_SyN.tif"
    mean_image = tmp_path / "mean_image_R_SyN.tif"
    for path in [raw_day0_red, raw_day1_red, raw_day1_green, registered_red, mean_image]:
        path.write_bytes(b"")

    lookup = build_raw_image_lookup(tmp_path)

    assert lookup[(0, "red")] == raw_day0_red
    assert lookup[(1, "red")] == raw_day1_red
    assert lookup[(1, "green")] == raw_day1_green
    assert (1, "green") in lookup
    assert all("SyN" not in path.name for path in lookup.values())


def test_select_roi_neighbor_z_slices_uses_centroid_and_clips_edges() -> None:
    mask_stack = np.zeros((5, 8, 8), dtype=np.uint16)
    mask_stack[0:2, 2:4, 2:4] = 3
    mask_stack[4:5, 5:7, 5:7] = 4

    near_top = select_roi_neighbor_z_slices(mask_stack, roi_id=3)
    near_bottom = select_roi_neighbor_z_slices(mask_stack, roi_id=4)

    assert near_top == {"z_minus1": 0, "z_center": 0, "z_plus1": 1}
    assert near_bottom == {"z_minus1": 3, "z_center": 4, "z_plus1": 4}


def test_measure_roi_geometry_uses_original_mask_labels() -> None:
    mask_stack = np.zeros((4, 6, 6), dtype=np.uint16)
    mask_stack[:, 2:4, 1:4] = 17

    geometry = measure_roi_geometry(mask_stack=mask_stack, roi_id=17)

    assert geometry.z_center == 2
    assert geometry.y_center == 2
    assert geometry.x_center == 2
    assert geometry.width_px == 3
    assert geometry.height_px == 2


def test_build_composite_rgb_mixes_red_and_green_channels() -> None:
    red_plane = np.array([[0.0, 5.0], [10.0, 0.0]])
    green_plane = np.array([[0.0, 2.5], [0.0, 10.0]])

    rgb = build_composite_rgb(red_plane, green_plane, red_vmax=10.0, green_vmax=10.0)

    assert rgb.shape == (2, 2, 3)
    assert np.isclose(rgb[1, 0, 0], 1.0)
    assert np.isclose(rgb[1, 0, 2], 1.0)
    assert np.isclose(rgb[1, 1, 1], 1.0)


def test_render_raw_space_triplet_panel_writes_expected_number_of_tiles(tmp_path: Path) -> None:
    raw_stack_lookup = {
        (0, "red"): np.full((3, 6, 6), 10.0),
        (0, "green"): np.full((3, 6, 6), 20.0),
        (1, "red"): np.full((3, 6, 6), 30.0),
        (1, "green"): np.full((3, 6, 6), 40.0),
    }
    mask0 = np.zeros((3, 6, 6), dtype=np.uint16)
    mask0[:, 2:4, 2:4] = 5
    mask1 = np.zeros((3, 6, 6), dtype=np.uint16)
    mask1[:, 2:4, 2:4] = 5
    mask_stack_lookup = {0: mask0, 1: mask1}
    output_path = tmp_path / "panel.png"

    metadata_rows = render_raw_space_triplet_panel(
        roi_id=5,
        raw_stack_lookup=raw_stack_lookup,
        mask_stack_lookup=mask_stack_lookup,
        output_path=output_path,
        half_window_z=1,
        crop_pad_xy=0,
        min_crop_size_px=2,
    )

    assert output_path.exists()
    assert len(metadata_rows) == 6
    assert {row["z_offset"] for row in metadata_rows} == {-1, 0, 1}
    assert {row["day"] for row in metadata_rows} == {0, 1}


def test_compute_shared_raw_space_group_geometry_uses_day0_average_centroid() -> None:
    mask0 = np.zeros((3, 10, 10), dtype=np.uint16)
    mask0[:, 1:3, 1:3] = 10
    mask0[:, 5:7, 7:9] = 20
    mask1 = np.zeros((3, 10, 10), dtype=np.uint16)
    mask1[:, 2:4, 2:4] = 10
    mask1[:, 6:8, 7:9] = 20

    geometry = compute_shared_raw_space_group_geometry(
        mask_stack_lookup={0: mask0, 1: mask1},
        roi_ids=[10, 20],
        center_day=0,
        crop_pad_xy=0,
        min_crop_size_px=2,
    )

    assert geometry.z_center == 1
    assert geometry.y_center == 4
    assert geometry.x_center == 5
    assert geometry.width_px >= 7
    assert geometry.height_px >= 5


def test_render_shared_raw_space_group_panel_writes_shared_fov_panel(tmp_path: Path) -> None:
    raw_stack_lookup = {
        (0, "red"): np.full((3, 10, 10), 10.0),
        (0, "green"): np.full((3, 10, 10), 20.0),
        (1, "red"): np.full((3, 10, 10), 30.0),
        (1, "green"): np.full((3, 10, 10), 40.0),
    }
    mask0 = np.zeros((3, 10, 10), dtype=np.uint16)
    mask0[:, 1:3, 1:3] = 10
    mask0[:, 5:7, 7:9] = 20
    mask1 = np.zeros((3, 10, 10), dtype=np.uint16)
    mask1[:, 2:4, 2:4] = 10
    mask1[:, 6:8, 7:9] = 20
    output_path = tmp_path / "shared_panel.png"

    metadata_rows = render_shared_raw_space_group_panel(
        roi_ids=[10, 20],
        raw_stack_lookup=raw_stack_lookup,
        mask_stack_lookup={0: mask0, 1: mask1},
        output_path=output_path,
        half_window_z=1,
        crop_pad_xy=0,
        min_crop_size_px=2,
    )

    assert output_path.exists()
    assert len(metadata_rows) == 12
    assert {row["z_offset"] for row in metadata_rows} == {-1, 0, 1}
    assert {row["day"] for row in metadata_rows} == {0, 1}
    assert {row["channel"] for row in metadata_rows} == {"red", "green"}


def test_daily_green_red_fit_summary_recovers_expected_parameters() -> None:
    roi_metrics = pd.DataFrame(
        {
            "roi_id": [1, 2, 3, 1, 2, 3],
            "day": [0, 0, 0, 1, 1, 1],
            "red": [1.0, 2.0, 3.0, 2.0, 4.0, 6.0],
            "green": [5.0, 7.0, 9.0, 0.0, -1.0, -2.0],
        }
    )

    fit_summary = summarize_daily_green_red_linear_fits(roi_metrics)

    assert fit_summary["day"].tolist() == [0, 1]

    day0 = fit_summary.loc[fit_summary["day"] == 0].iloc[0]
    day1 = fit_summary.loc[fit_summary["day"] == 1].iloc[0]

    assert np.isclose(day0["slope"], 2.0)
    assert np.isclose(day0["intercept"], 3.0)
    assert np.isclose(day0["r_squared"], 1.0)
    assert int(day0["n_rois"]) == 3

    assert np.isclose(day1["slope"], -0.5)
    assert np.isclose(day1["intercept"], 1.0)
    assert np.isclose(day1["r_squared"], 1.0)
    assert int(day1["n_rois"]) == 3


def test_daily_green_red_fit_summary_ignores_invalid_rows() -> None:
    roi_metrics = pd.DataFrame(
        {
            "roi_id": [1, 2, 3, 4],
            "day": [0, 0, 0, 0],
            "red": [1.0, 2.0, np.nan, 4.0],
            "green": [3.0, 5.0, 7.0, np.inf],
        }
    )

    fit_summary = summarize_daily_green_red_linear_fits(roi_metrics)
    day0 = fit_summary.iloc[0]

    assert int(day0["n_rois"]) == 2
    assert np.isclose(day0["slope"], 2.0)
    assert np.isclose(day0["intercept"], 1.0)
    assert np.isnan(day0["slope_ci_low"])
    assert np.isnan(day0["intercept_ci_high"])


def test_green_red_fit_residuals_match_expected_signed_deviations() -> None:
    roi_metrics = pd.DataFrame(
        {
            "roi_id": [1, 1, 2, 2],
            "day": [0, 1, 0, 1],
            "red": [10.0, 10.0, 20.0, 20.0],
            "green": [6.0, 3.0, 8.0, 11.0],
        }
    )
    fit_summary = pd.DataFrame(
        {
            "day": [0, 1],
            "slope": [0.2, 0.4],
            "intercept": [3.0, 1.0],
        }
    )

    residual_table = compute_green_red_fit_residuals(
        roi_metrics=roi_metrics,
        fit_summary=fit_summary,
    )

    roi1_day0 = residual_table[(residual_table["roi_id"] == 1) & (residual_table["day"] == 0)].iloc[0]
    roi1_day1 = residual_table[(residual_table["roi_id"] == 1) & (residual_table["day"] == 1)].iloc[0]
    roi2_day1 = residual_table[(residual_table["roi_id"] == 2) & (residual_table["day"] == 1)].iloc[0]

    assert np.isclose(roi1_day0["predicted_green_from_fit"], 5.0)
    assert np.isclose(roi1_day0["green_fit_residual"], 1.0)
    assert np.isclose(roi1_day1["predicted_green_from_fit"], 5.0)
    assert np.isclose(roi1_day1["green_fit_residual"], -2.0)
    assert np.isclose(roi1_day1["delta_green_fit_residual"], -3.0)
    assert np.isclose(roi2_day1["predicted_green_from_fit"], 9.0)
    assert np.isclose(roi2_day1["green_fit_residual"], 2.0)


def test_green_red_fit_residuals_are_zero_relative_to_day0_on_day0() -> None:
    roi_metrics = pd.DataFrame(
        {
            "roi_id": [1, 1, 2, 2],
            "day": [0, 1, 0, 1],
            "red": [5.0, 5.0, 8.0, 8.0],
            "green": [3.0, 4.0, 2.0, 1.0],
        }
    )
    fit_summary = pd.DataFrame(
        {
            "day": [0, 1],
            "slope": [0.5, 0.5],
            "intercept": [0.0, 0.0],
        }
    )

    residual_table = compute_green_red_fit_residuals(
        roi_metrics=roi_metrics,
        fit_summary=fit_summary,
    )

    day0_rows = residual_table[residual_table["day"] == 0]
    assert np.allclose(day0_rows["delta_green_fit_residual"].to_numpy(), 0.0)
    assert np.allclose(day0_rows["delta_green_fit_signed_distance"].to_numpy(), 0.0)


def test_summarize_residual_sign_changes_counts_crossings() -> None:
    residual_table = pd.DataFrame(
        {
            "roi_id": [1, 1, 1, 2, 2, 2],
            "day": [0, 1, 2, 0, 1, 2],
            "green_fit_residual": [2.0, -1.0, -3.0, -2.0, -1.0, -0.5],
        }
    )

    summary = summarize_residual_sign_changes(residual_table)
    roi1 = summary.loc[summary["roi_id"] == 1].iloc[0]
    roi2 = summary.loc[summary["roi_id"] == 2].iloc[0]

    assert int(roi1["green_fit_residual_sign_change_count"]) == 1
    assert np.isclose(roi1["green_fit_residual_range"], 5.0)
    assert int(roi2["green_fit_residual_sign_change_count"]) == 0


def test_select_ranked_roi_days_returns_requested_top_n_rows() -> None:
    roi_day_table = pd.DataFrame(
        {
            "roi_id": [1, 1, 2, 2, 3, 3],
            "day": [0, 1, 0, 1, 0, 1],
            "red": [10.0, 11.0, 20.0, 21.0, 30.0, 31.0],
        }
    )
    ranking_table = pd.DataFrame(
        {
            "roi_id": [3, 1, 2],
            "selection_rank": [1, 2, 3],
            "min_delta_log2_green_over_red": [-2.0, -1.0, -0.5],
        }
    )

    selected = select_ranked_roi_days(
        roi_day_table=roi_day_table,
        ranking_table=ranking_table,
        top_n=2,
        ranking_columns=["selection_rank", "min_delta_log2_green_over_red"],
    )

    assert selected["roi_id"].unique().tolist() == [3, 1]
    assert selected["selection_rank"].unique().tolist() == [1, 2]
    assert len(selected) == 4


def test_trajectory_classifier_marks_stable_roi() -> None:
    roi_metrics = pd.DataFrame(
        {
            "roi_id": [1] * 5,
            "day": [0, 1, 2, 3, 4],
            "delta_log2_green_over_red": [0.0, 0.08, -0.10, 0.12, -0.06],
        }
    )

    classified = classify_roi_log_ratio_trajectories(roi_metrics)

    assert classified.iloc[0]["trajectory_category"] == "stable"


def test_trajectory_classifier_marks_mostly_down_roi() -> None:
    roi_metrics = pd.DataFrame(
        {
            "roi_id": [2] * 5,
            "day": [0, 1, 2, 3, 4],
            "delta_log2_green_over_red": [0.0, -0.12, -0.42, -0.55, -0.38],
        }
    )

    classified = classify_roi_log_ratio_trajectories(roi_metrics)

    assert classified.iloc[0]["trajectory_category"] == "mostly_down"


def test_trajectory_classifier_marks_mostly_up_roi() -> None:
    roi_metrics = pd.DataFrame(
        {
            "roi_id": [3] * 5,
            "day": [0, 1, 2, 3, 4],
            "delta_log2_green_over_red": [0.0, 0.10, 0.36, 0.48, 0.40],
        }
    )

    classified = classify_roi_log_ratio_trajectories(roi_metrics)

    assert classified.iloc[0]["trajectory_category"] == "mostly_up"


def test_trajectory_classifier_marks_oscillatory_roi() -> None:
    roi_metrics = pd.DataFrame(
        {
            "roi_id": [4] * 5,
            "day": [0, 1, 2, 3, 4],
            "delta_log2_green_over_red": [0.0, 0.34, -0.30, 0.42, -0.28],
        }
    )

    classified = classify_roi_log_ratio_trajectories(roi_metrics)

    assert classified.iloc[0]["trajectory_category"] == "oscillatory"
    assert classified.iloc[0]["significant_sign_change_count"] >= 2


def test_roi_crop_panel_helper_returns_expected_bounds() -> None:
    mask_stack = np.zeros((4, 12, 14), dtype=np.uint16)
    mask_stack[1:3, 4:7, 5:9] = 7

    bounds = compute_roi_crop_bounds(mask_stack, roi_id=7, pad_xy=1, min_crop_size=0)

    assert bounds["z_start"] == 1
    assert bounds["z_stop"] == 3
    assert bounds["y_start"] == 3
    assert bounds["y_stop"] == 8
    assert bounds["x_start"] == 4
    assert bounds["x_stop"] == 10


def test_project_roi_stack_view_returns_crop_and_mask_projection() -> None:
    mask_stack = np.zeros((4, 12, 14), dtype=np.uint16)
    mask_stack[1:3, 4:7, 5:9] = 9

    image_stack = np.zeros_like(mask_stack, dtype=float)
    image_stack[1, 5, 6] = 3.0
    image_stack[2, 6, 7] = 5.0

    image_proj, mask_proj, bounds = project_roi_stack_view(
        image_stack,
        mask_stack,
        roi_id=9,
        pad_xy=1,
        min_crop_size=0,
    )

    assert image_proj.shape == mask_proj.shape
    assert image_proj.max() == 5.0
    assert mask_proj.dtype == bool
    assert mask_proj.sum() > 0
    assert bounds["y_start"] == 3
    assert bounds["x_start"] == 4


def test_format_duration_seconds_uses_hh_mm_ss() -> None:
    assert format_duration_seconds(0.0) == "00:00:00"
    assert format_duration_seconds(65.4) == "00:01:05"
    assert format_duration_seconds(3661.9) == "01:01:01"


def test_parse_args_disables_raw_space_by_default_and_can_enable_it() -> None:
    default_args = parse_args(["--mask-name", "mask.tif"])
    enabled_args = parse_args(["--mask-name", "mask.tif", "--enable-raw-space-validation"])
    skipped_args = parse_args(["--mask-name", "mask.tif", "--skip-raw-space-validation"])

    assert default_args.enable_raw_space_validation is False
    assert default_args.skip_raw_space_validation is False
    assert enabled_args.enable_raw_space_validation is True
    assert enabled_args.skip_raw_space_validation is False
    assert skipped_args.enable_raw_space_validation is False
    assert skipped_args.skip_raw_space_validation is True


def test_write_run_log_includes_stage_durations_seconds(tmp_path: Path) -> None:
    config = RegisteredPipelineConfig(dataset="1050", start_date="20260511", mask_name="mask.tif")
    filter_counts = pd.DataFrame([{"step": "mask_rois", "count": 10, "pct_of_start": 100.0, "pct_of_previous": 100.0}])
    raw_space_status = {"available": False, "reason": "skipped by configuration"}
    stage_durations_seconds = {"extract_roi_intensities": 1.25, "total": 3.5}

    write_run_log(
        output_dir=tmp_path,
        config=config,
        filter_counts=filter_counts,
        raw_space_status=raw_space_status,
        stage_durations_seconds=stage_durations_seconds,
    )

    payload = json.loads((tmp_path / "run_log.json").read_text(encoding="utf-8"))
    assert payload["stage_durations_seconds"] == stage_durations_seconds
