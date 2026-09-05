from __future__ import annotations

import pandas as pd

from cross_laser_roi_qc import (
    fixed_coverage_by_long_axis,
    generate_cross_laser_qc,
    high_confidence_long_axis_statistics,
    select_cross_laser_examples,
)


def test_fixed_coverage_denominator_includes_observable_zero_candidate_rois() -> None:
    coverage = pd.DataFrame(
        {
            "label_1050": [1, 2, 3, 4],
            "centroid_1050_y": [1, 1, 1, 1],
            "centroid_1050_x": [1, 3, 7, 9],
            "common_volume_status": [
                "inside_common_volume",
                "inside_common_volume",
                "inside_common_volume",
                "outside_common_volume",
            ],
            "green_high_label_920": [10, None, 30, None],
        }
    )

    summary = fixed_coverage_by_long_axis(
        coverage, image_shape_yx=(10, 10), bins=2
    ).set_index("bin")

    assert int(summary["n_observable_1050"].sum()) == 3
    assert int(summary["n_high"].sum()) == 2
    assert summary.loc[0, "high_fraction"] == 0.5


def test_high_scatter_and_bin_medians_use_exact_same_high_population() -> None:
    pairs = pd.DataFrame(
        {
            "assignment_policy": ["high", "high", "balanced"],
            "moving_source": ["920_green_primary"] * 3,
            "label_1050": [1, 2, 3],
            "centroid_1050_y": [1, 1, 1],
            "centroid_1050_x": [1, 3, 8],
            "raw_delta_z_planes": [1.0, 3.0, 100.0],
            "aligned_residual_distance_um": [2.0, 4.0, 100.0],
        }
    )

    high, medians = high_confidence_long_axis_statistics(
        pairs, image_shape_yx=(10, 10), bins=2
    )

    assert high["label_1050"].tolist() == [1, 2]
    assert int(medians["n_high"].sum()) == 2
    assert medians.loc[medians["bin"].eq(0), "median_raw_delta_z_planes"].item() == 2.0


def test_example_selection_is_deterministic_and_spatially_distributed() -> None:
    rows = pd.DataFrame(
        {
            "label_1050": [5, 4, 3, 2, 1],
            "centroid_1050_y": [1, 1, 1, 1, 1],
            "centroid_1050_x": [0, 2, 4, 6, 8],
            "resolved_status": ["primary_high"] * 5,
            "green_best_candidate_score": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    first = select_cross_laser_examples(rows, limit=3)
    second = select_cross_laser_examples(rows, limit=3)

    assert first["label_1050"].tolist() == second["label_1050"].tolist()
    assert len(first) == 3


def test_source_comparison_csv_matches_qc_categories(tmp_path) -> None:
    coverage = pd.DataFrame({"label_1050": [1], "centroid_1050_y": [1], "centroid_1050_x": [1], "common_volume_status": ["inside_common_volume"], "green_high_label_920": [10]})
    resolution = pd.DataFrame({"label_1050": [1, 2], "primary_green_status": ["high", "no_candidate"], "secondary_red_status": ["high", "high"], "cross_source_conflict": [False, True], "resolved_status": ["primary_high", "secondary_high_rescue_candidate"]})
    outputs = generate_cross_laser_qc(output_dir=tmp_path, fixed_coverage=coverage, accepted_pairs=pd.DataFrame(), image_shape_yx=(10, 10), identity_resolution=resolution)
    table = pd.read_csv(outputs["source_comparison_table"]).set_index("category")
    assert table.loc["both_high", "count"] == 1
    assert table.loc["red_high_only", "count"] == 1
