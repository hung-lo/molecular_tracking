from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from roi_matcher import (
    MatchParams,
    add_qc_flags,
    build_tracks_from_pairwise_matches,
    estimate_pair_shift,
    export_match_tables,
    extract_roi_records,
    generate_candidate_pairs,
    match_roi_masks,
    parse_args,
    score_candidate_pair,
    solve_pairwise_assignment,
)


def draw_box(
    mask_stack: np.ndarray,
    label: int,
    z_range: tuple[int, int],
    y_range: tuple[int, int],
    x_range: tuple[int, int],
) -> None:
    """Fill one rectangular ROI into a labeled ``(z, y, x)`` mask stack.

    Parameters
    ----------
    mask_stack : numpy.ndarray
        Integer label image with shape ``(z, y, x)``.
    label : int
        Positive label value written into the requested region.
    z_range : tuple[int, int]
        Inclusive-exclusive ROI bounds along the z axis in planes.
    y_range : tuple[int, int]
        Inclusive-exclusive ROI bounds along the y axis in pixels.
    x_range : tuple[int, int]
        Inclusive-exclusive ROI bounds along the x axis in pixels.

    Returns
    -------
    None
        The input array is modified in place.
    """

    z0, z1 = z_range
    y0, y1 = y_range
    x0, x1 = x_range
    mask_stack[z0:z1, y0:y1, x0:x1] = label


def make_three_roi_mask() -> np.ndarray:
    """Create a small labeled test stack with three compact ROIs.

    Parameters
    ----------
    None
        This helper takes no inputs.

    Returns
    -------
    numpy.ndarray
        Unsigned integer mask stack with shape ``(4, 32, 32)`` in ``(z, y, x)``
        order.
    """

    mask = np.zeros((4, 32, 32), dtype=np.uint16)
    draw_box(mask, 1, (1, 3), (4, 8), (5, 9))
    draw_box(mask, 2, (0, 2), (14, 18), (16, 20))
    draw_box(mask, 3, (2, 4), (22, 27), (24, 29))
    return mask


def make_translated_mask(source: np.ndarray, shift_yx: tuple[int, int]) -> np.ndarray:
    """Translate a labeled mask stack by integer XY offsets with zero padding.

    Parameters
    ----------
    source : numpy.ndarray
        Integer label image with shape ``(z, y, x)``.
    shift_yx : tuple[int, int]
        Integer translation ``(dy, dx)`` applied to every ROI in pixels.

    Returns
    -------
    numpy.ndarray
        Shifted label image with the same shape and dtype as ``source``.
    """

    dy, dx = shift_yx
    out = np.zeros_like(source)
    z_dim, y_dim, x_dim = source.shape
    for z_index in range(z_dim):
        ys, xs = np.nonzero(source[z_index] > 0)
        for y_coord, x_coord in zip(ys, xs):
            new_y = y_coord + dy
            new_x = x_coord + dx
            if 0 <= new_y < y_dim and 0 <= new_x < x_dim:
                out[z_index, new_y, new_x] = source[z_index, y_coord, x_coord]
    return out


def make_four_week_masks() -> list[np.ndarray]:
    """Build four synthetic weekly masks with one persistent gap and one singleton.

    Parameters
    ----------
    None
        This helper takes no inputs.

    Returns
    -------
    list[numpy.ndarray]
        Four weekly label stacks, each with shape ``(4, 40, 40)`` in ``(z, y, x)``
        order.
    """

    week1 = np.zeros((4, 40, 40), dtype=np.uint16)
    week2 = np.zeros_like(week1)
    week3 = np.zeros_like(week1)
    week4 = np.zeros_like(week1)

    # Track A persists across all weeks with small motion.
    draw_box(week1, 11, (1, 3), (6, 10), (7, 11))
    draw_box(week2, 21, (1, 3), (7, 11), (8, 12))
    draw_box(week3, 31, (1, 3), (8, 12), (9, 13))
    draw_box(week4, 41, (1, 3), (9, 13), (10, 14))

    # Track B is missing in week 3 but reappears in week 4.
    draw_box(week1, 12, (1, 3), (18, 22), (24, 28))
    draw_box(week2, 22, (1, 3), (19, 23), (25, 29))
    draw_box(week4, 42, (1, 3), (21, 25), (27, 31))

    # Track C exists only once.
    draw_box(week3, 33, (0, 2), (28, 32), (5, 9))

    return [week1, week2, week3, week4]


def test_extract_roi_records_reports_centroid_bbox_area_and_edge_metadata() -> None:
    mask = np.zeros((3, 20, 20), dtype=np.uint16)
    draw_box(mask, 7, (1, 3), (2, 6), (3, 7))

    records = extract_roi_records(mask, day_name="week1", patch_radius=4, edge_margin=4)

    assert len(records) == 1
    record = records[0]
    assert record.label == 7
    assert np.isclose(record.centroid_z, 1.5)
    assert np.isclose(record.centroid_y, 3.5)
    assert np.isclose(record.centroid_x, 4.5)
    assert record.area_total == 32
    assert record.area_center == 16
    assert record.bbox_zyx == ((1, 2), (2, 5), (3, 6))
    assert record.is_edge is True
    assert np.isclose(record.dist_to_edge, 3.5)


def test_pair_shift_estimator_recovers_known_small_xy_shift() -> None:
    week1 = make_three_roi_mask()
    week2 = make_translated_mask(week1, shift_yx=(3, -2))

    shift_y, shift_x = estimate_pair_shift(week1, week2, max_shift=8)

    assert abs(shift_y + 3.0) <= 1.0
    assert abs(shift_x - 2.0) <= 1.0




def test_match_params_and_cli_defaults_reflect_relaxed_threshold_revision() -> None:
    params = MatchParams()
    args = parse_args(["--masks", "week1.tif", "week2.tif", "--output-prefix", "roi_match_test"])

    assert params.max_dist_xy == 15.0
    assert params.max_dist_z == 5.0
    assert params.translation_max_shift == 32
    assert args.max_dist_xy == 15.0
    assert args.max_dist_z == 5.0
    assert args.translation_max_shift == 32


def test_match_roi_masks_emits_progress_logs_when_log_fn_provided() -> None:
    masks = make_four_week_masks()
    params = MatchParams(
        patch_radius=5,
        patch_size=24,
        overlap=8,
        max_dist_xy=5.0,
        max_dist_z=2.5,
        min_score=0.45,
        min_gap_support=0.80,
    )
    messages: list[str] = []

    match_roi_masks(
        mask_stacks=masks,
        day_names=["week1", "week2", "week3", "week4"],
        params=params,
        log_fn=messages.append,
    )

    assert any("Extracting ROI records for week1" in message for message in messages)
    assert any("Pair 1/" in message and "week1 vs week2" in message for message in messages)
    assert any("Finished matching:" in message for message in messages)


def test_candidate_generation_excludes_far_rois_after_shift_correction() -> None:
    week1 = make_three_roi_mask()
    week2 = make_translated_mask(week1, shift_yx=(2, -1))
    draw_box(week2, 99, (1, 3), (26, 30), (2, 6))

    records_a = extract_roi_records(week1, day_name="week1", patch_radius=5, edge_margin=4)
    records_b = extract_roi_records(week2, day_name="week2", patch_radius=5, edge_margin=4)
    params = MatchParams(max_dist_xy=6.0, max_dist_z=2.5, patch_radius=5)

    candidates = generate_candidate_pairs(records_a, records_b, shift_yx=(-2.0, 1.0), params=params)
    candidate_labels = {(row["label_a"], row["label_b"]) for row in candidates.to_dict("records")}

    assert (1, 1) in candidate_labels
    assert (2, 2) in candidate_labels
    assert (3, 3) in candidate_labels
    assert all(label_b != 99 for _, label_b in candidate_labels)


def test_scoring_prefers_true_match_over_nearby_shape_decoy() -> None:
    week1 = np.zeros((3, 30, 30), dtype=np.uint16)
    week2 = np.zeros_like(week1)
    draw_box(week1, 1, (1, 3), (10, 15), (10, 15))
    draw_box(week2, 2, (1, 3), (11, 16), (11, 16))
    draw_box(week2, 3, (1, 3), (12, 16), (11, 14))

    records_a = extract_roi_records(week1, day_name="week1", patch_radius=6, edge_margin=4)
    records_b = extract_roi_records(week2, day_name="week2", patch_radius=6, edge_margin=4)
    params = MatchParams(max_dist_xy=8.0, max_dist_z=2.5, patch_radius=6)

    true_score, true_components = score_candidate_pair(records_a[0], records_b[0], (0.0, 0.0), params)
    decoy_score, decoy_components = score_candidate_pair(records_a[0], records_b[1], (0.0, 0.0), params)

    assert true_score > decoy_score
    assert true_components["iou"] > decoy_components["iou"]


def test_assignment_enforces_one_to_one_matches_for_collision_case() -> None:
    candidate_table = pd.DataFrame(
        [
            {"idx_a": 0, "label_a": 10, "idx_b": 0, "label_b": 20, "score": 0.90},
            {"idx_a": 0, "label_a": 10, "idx_b": 1, "label_b": 21, "score": 0.70},
            {"idx_a": 1, "label_a": 11, "idx_b": 0, "label_b": 20, "score": 0.85},
        ]
    )

    assigned = solve_pairwise_assignment(candidate_table)

    assert len(assigned) == 1
    row = assigned.iloc[0]
    assert int(row["label_a"]) == 10
    assert int(row["label_b"]) == 20
    assert np.isclose(row["score"], 0.90)


def test_track_builder_keeps_adjacent_consistency_without_forcing_bad_bridge() -> None:
    pair_tables = {
        ("week1", "week2"): pd.DataFrame(
            [{"label_a": 1, "label_b": 2, "score": 0.90, "confidence": 0.85}]
        ),
        ("week2", "week3"): pd.DataFrame(
            [{"label_a": 2, "label_b": 3, "score": 0.91, "confidence": 0.84}]
        ),
        ("week3", "week4"): pd.DataFrame(columns=["label_a", "label_b", "score", "confidence"]),
        ("week1", "week3"): pd.DataFrame(
            [{"label_a": 1, "label_b": 30, "score": 0.95, "confidence": 0.92}]
        ),
        ("week2", "week4"): pd.DataFrame(
            [{"label_a": 2, "label_b": 4, "score": 0.94, "confidence": 0.91}]
        ),
    }

    tracks = build_tracks_from_pairwise_matches(
        day_names=["week1", "week2", "week3", "week4"],
        pair_tables=pair_tables,
        min_gap_support=0.88,
    )

    main_track = tracks.loc[tracks["week1_roi"] == 1].iloc[0]
    assert int(main_track["week2_roi"]) == 2
    assert int(main_track["week3_roi"]) == 3
    assert pd.isna(main_track["week4_roi"])


def test_skip_week_recovery_fills_single_gap_only_when_pairwise_support_is_strong() -> None:
    pair_tables = {
        ("week1", "week2"): pd.DataFrame(
            [{"label_a": 1, "label_b": 2, "score": 0.91, "confidence": 0.86}]
        ),
        ("week2", "week3"): pd.DataFrame(columns=["label_a", "label_b", "score", "confidence"]),
        ("week3", "week4"): pd.DataFrame(columns=["label_a", "label_b", "score", "confidence"]),
        ("week2", "week4"): pd.DataFrame(
            [{"label_a": 2, "label_b": 4, "score": 0.93, "confidence": 0.90}]
        ),
    }

    tracks = build_tracks_from_pairwise_matches(
        day_names=["week1", "week2", "week3", "week4"],
        pair_tables=pair_tables,
        min_gap_support=0.88,
    )

    recovered_track = tracks.loc[tracks["week1_roi"] == 1].iloc[0]
    assert int(recovered_track["week2_roi"]) == 2
    assert pd.isna(recovered_track["week3_roi"])
    assert int(recovered_track["week4_roi"]) == 4
    assert bool(recovered_track["used_gap_bridge"]) is True


def test_end_to_end_synthetic_four_week_dataset_recovers_expected_tracks() -> None:
    masks = make_four_week_masks()
    params = MatchParams(
        patch_radius=5,
        patch_size=24,
        overlap=8,
        max_dist_xy=5.0,
        max_dist_z=2.5,
        min_score=0.45,
        min_gap_support=0.80,
    )

    tracks, pair_tables, qc_table = match_roi_masks(
        mask_stacks=masks,
        day_names=["week1", "week2", "week3", "week4"],
        params=params,
    )

    assert set(pair_tables) >= {
        ("week1", "week2"),
        ("week2", "week3"),
        ("week3", "week4"),
    }
    assert {"low_confidence", "used_gap_bridge", "missing_intermediate_days"}.issubset(qc_table.columns)

    track_a = tracks.loc[tracks["week1_roi"] == 11].iloc[0]
    assert int(track_a["week2_roi"]) == 21
    assert int(track_a["week3_roi"]) == 31
    assert int(track_a["week4_roi"]) == 41

    track_b = tracks.loc[tracks["week1_roi"] == 12].iloc[0]
    assert int(track_b["week2_roi"]) == 22
    assert pd.isna(track_b["week3_roi"])
    assert int(track_b["week4_roi"]) == 42
    assert bool(track_b["used_gap_bridge"]) is True

    singleton = tracks.loc[tracks["week3_roi"] == 33].iloc[0]
    assert singleton["n_days_present"] == 1


def test_add_qc_flags_marks_low_confidence_edge_and_gap_cases() -> None:
    tracks = pd.DataFrame(
        [
            {
                "cluster_id": 1,
                "week1_roi": 1,
                "week2_roi": 2,
                "week3_roi": pd.NA,
                "week4_roi": 4,
                "n_days_present": 3,
                "mean_confidence": 0.52,
                "min_confidence": 0.41,
                "used_gap_bridge": True,
                "any_edge_roi": True,
                "missing_intermediate_days": 1,
            }
        ]
    )

    flagged = add_qc_flags(tracks, low_confidence_threshold=0.55)

    row = flagged.iloc[0]
    assert bool(row["low_confidence"]) is True
    assert bool(row["gap_bridge_qc"]) is True
    assert bool(row["edge_qc"]) is True
    assert bool(row["needs_review"]) is True


def test_export_match_tables_writes_track_pair_and_qc_csvs(tmp_path: Path) -> None:
    masks = make_four_week_masks()
    params = MatchParams(
        patch_radius=5,
        patch_size=24,
        overlap=8,
        max_dist_xy=5.0,
        max_dist_z=2.5,
        min_score=0.45,
        min_gap_support=0.80,
    )
    tracks, pair_tables, qc_table = match_roi_masks(
        mask_stacks=masks,
        day_names=["week1", "week2", "week3", "week4"],
        params=params,
    )

    output_paths = export_match_tables(
        output_prefix=tmp_path / "roi_match_test",
        tracks_table=tracks,
        pair_tables=pair_tables,
        qc_table=qc_table,
    )

    assert output_paths["tracks"].exists()
    assert output_paths["qc"].exists()
    assert output_paths["pair_tables"]
    assert output_paths["pair_diagnostics"].exists()
    assert output_paths["pair_summary"].exists()
    assert output_paths["track_length_summary"].exists()
    exported_qc = pd.read_csv(output_paths["qc"])
    exported_pair_diagnostics = pd.read_csv(output_paths["pair_diagnostics"])
    exported_pair_summary = pd.read_csv(output_paths["pair_summary"])
    exported_track_length_summary = pd.read_csv(output_paths["track_length_summary"])
    assert "needs_review" in exported_qc.columns
    assert {
        "week_a",
        "week_b",
        "roi_a",
        "roi_b",
        "distance_xy_px",
        "distance_xy_um",
        "distance_z_planes",
        "distance_z_um",
        "match_score",
        "confidence",
    }.issubset(exported_pair_diagnostics.columns)
    assert {
        "candidate_pairs",
        "accepted_reciprocal_matches",
        "median_accepted_xy_px",
        "max_accepted_xy_px",
        "median_accepted_z_planes",
        "max_accepted_z_planes",
        "score_median",
        "confidence_median",
    }.issubset(exported_pair_summary.columns)
    assert exported_track_length_summary["minimum_weeks_present"].tolist() == [4, 5, 6, 7]


def test_cropped_syn_masks_produce_nonempty_pairwise_matches_and_track_summary() -> None:
    mask_paths = [
        Path("/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_256/week1_average_cp_masks_SyN.tif"),
        Path("/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_256/week2_average_cp_masks_SyN.tif"),
        Path("/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_256/week3_average_cp_masks_SyN.tif"),
        Path("/mnt/d/_data/_newAAV_2026/weekly_registration_test/crop_256/week4_average_cp_masks.tif"),
    ]

    if not all(path.exists() for path in mask_paths):
        return

    mask_stacks = [tifffile.imread(path) for path in mask_paths]
    tracks, pair_tables, qc_table = match_roi_masks(
        mask_stacks=mask_stacks,
        day_names=["week1", "week2", "week3", "week4"],
        params=MatchParams(),
    )

    assert len(tracks) > 0
    assert len(qc_table) == len(tracks)
    assert len(pair_tables[("week1", "week2")]) > 0
    assert len(pair_tables[("week2", "week3")]) > 0
    assert len(pair_tables[("week3", "week4")]) > 0
