from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from longest_supported_window import select_longest_supported_window
from roi_log_ratio_analysis import (
    compute_green_red_fit_residuals,
    compute_log_ratio_metrics,
    decide_ranking_population,
    has_usable_day0_baseline,
    select_top_changing_rois,
    summarize_roi_metrics,
    summarize_daily_green_red_linear_fits_tolerant,
)
from session_manifest import SessionRecord
from session_population_normalization import fit_session_population
from run_daywise_matched_roi_pipeline import DaywiseMatchedPipelineConfig, run_daywise_matched_roi_pipeline
from run_weekly_matched_output_quick_plots import build_quick_plots
from run_ranked_roi_quick_views import build_ranked_roi_views


def _records(n: int) -> list[SessionRecord]:
    return [
        SessionRecord(i, f"session_{i}", date(2026, 1, 1) + timedelta(days=i), Path("mask.tif"))
        for i in range(n)
    ]


def test_tolerant_fits_distinguish_missing_single_constant_and_valid_days():
    metrics = pd.DataFrame({
        "roi_id": [11, 21, 22, 31, 32],
        "day": [1, 2, 2, 3, 3],
        "red": [10.0, 20.0, 20.0, 10.0, 20.0],
        "green": [12.0, 22.0, 25.0, 13.0, 24.0],
        "ratio_qc_pass": [True] * 5,
        "fit_population_include": [True] * 5,
    })
    fits = summarize_daily_green_red_linear_fits_tolerant(metrics, expected_days=[0, 1, 2, 3])
    status = fits.set_index("day")
    assert status.loc[0, "fit_status"] == "not_estimable"
    assert status.loc[0, "fit_reason"] == "no_rois"
    assert status.loc[1, "fit_reason"] == "fewer_than_two_fit_eligible_pairs"
    assert status.loc[2, "fit_reason"] == "constant_red_predictor"
    assert status.loc[3, "fit_status"] == "valid"
    assert np.isnan(status.loc[0, "slope"])
    assert np.isnan(status.loc[2, "intercept"])

    residuals = compute_green_red_fit_residuals(
        metrics, fit_summary=fits, baseline_day=0, allow_missing_fits=True,
    )
    assert residuals.loc[residuals["day"].isin([1, 2]), "green_fit_residual"].isna().all()
    assert np.isfinite(residuals.loc[residuals["day"].eq(3), "green_fit_residual"]).all()

    qc_metrics = pd.DataFrame({
        "day": [4, 5], "red": [10.0, 15.0], "green": [20.0, 30.0],
        "red_signal_qc_pass": [False, True], "green_signal_qc_pass": [False, True],
        "fit_population_include": [False, False],
    })
    qc_fits = summarize_daily_green_red_linear_fits_tolerant(qc_metrics, expected_days=[4, 5]).set_index("day")
    assert qc_fits.loc[4, "fit_reason"] == "no_signal_valid_pairs"
    assert qc_fits.loc[5, "fit_reason"] == "no_fit_eligible_pairs"


def test_native_session_fit_reports_zero_one_constant_and_valid_sessions():
    records = _records(4)
    population = pd.DataFrame({
        "day": [1, 2, 2, 3, 3],
        "session_index": [1, 2, 2, 3, 3],
        "session_id": ["session_1", "session_2", "session_2", "session_3", "session_3"],
        "acquisition_date": ["2026-01-02", "2026-01-03", "2026-01-03", "2026-01-04", "2026-01-04"],
        "elapsed_days": [1, 2, 2, 3, 3],
        "ratio_qc_pass": [True] * 5,
        "red": [10.0, 20.0, 20.0, 10.0, 20.0],
        "green": [12.0, 22.0, 25.0, 13.0, 24.0],
    })
    fits = fit_session_population(population, records=records).set_index("session_index")
    assert fits.loc[0, "fit_reason"] == "no_rois"
    assert np.isnan(fits.loc[0, "slope"])
    assert fits.loc[1, "fit_reason"] == "fewer_than_two_fit_eligible_pairs"
    assert fits.loc[2, "fit_reason"] == "constant_red_predictor"
    assert fits.loc[3, "fit_status"] == "valid"
    assert np.isfinite(fits.loc[3, "slope"])


def _window_data(records, memberships):
    metric_rows = []
    native_rows = []
    for day, track_ids in enumerate(memberships):
        record = records[day]
        for track_id in track_ids:
            red = float(100 + int(track_id) * 3 + day)
            green = float(80 + int(track_id) * 2 + day * 1.4)
            metric_rows.append({
                "roi_id": int(track_id), "track_uid": f"track-{track_id}",
                "session_index": day, "day": day, "session_id": record.session_id,
                "session_roi_label": int(track_id), "red": red, "green": green,
                "ratio_qc_pass": True, "log2_green_over_red": np.log2((green + 1) / (red + 1)),
            })
            native_rows.append({
                "session_id": record.session_id, "session_roi_label": int(track_id),
                "fit_population_include": True,
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(native_rows)


def _ranking_metrics(*, day0_ids=(), post_ids=(), policies=None):
    """Small deterministic original-Day-0 metrics for population decisions."""
    rows = []
    policies = policies or {}
    for roi_id in sorted(set(day0_ids) | set(post_ids)):
        days = ([0] if roi_id in day0_ids else []) + ([1, 2] if roi_id in post_ids else [])
        for day in days:
            red = 100.0
            green = 100.0 if day == 0 else 200.0
            rows.append({
                "roi_id": int(roi_id), "track_uid": policies.get(roi_id, f"track-{roi_id}"),
                "match_policy": "graph", "day": day, "session_index": day,
                "session_id": f"session_{day}", "acquisition_date": f"2026-01-0{day + 1}",
                "red": red, "green": green, "brightness": red + green,
                "log2_green_over_red": np.log2((green + 1.0) / (red + 1.0)),
                "delta_log2_green_over_red": 0.0 if day == 0 else np.log2(201.0 / 101.0),
                "ratio_qc_pass": True, "fit_population_include": True,
            })
    return pd.DataFrame(rows)


def _saved_window(track_ids, *, policy="graph"):
    rows = []
    for roi_id in track_ids:
        for session_index in (4, 5):
            red = 100.0
            green = 100.0 if session_index == 4 else 200.0
            rows.append({
                "roi_id": int(roi_id), "track_uid": f"track-{roi_id}",
                "match_policy": policy, "session_index": session_index,
                "session_id": f"session_{session_index}",
                "acquisition_date": f"2026-01-0{session_index + 1}",
                "red": red, "green": green, "brightness": red + green,
                "log2_green_over_red": np.log2((green + 1.0) / (red + 1.0)),
                "delta_log2_green_over_red_from_window_baseline": 0.0 if session_index == 4 else np.log2(201.0 / 101.0),
                "ratio_qc_pass": True, "fit_population_include": True,
            })
    metrics = pd.DataFrame(rows)
    tracks = pd.DataFrame({
        "roi_id": list(track_ids), "track_uid": [f"track-{roi_id}" for roi_id in track_ids],
        "match_policy": [policy] * len(track_ids),
    })
    selection = {
        "status": "valid", "window_session_indices": [4, 5],
        "window_session_ids": ["session_4", "session_5"],
        "window_acquisition_dates": ["2026-01-05", "2026-01-06"],
        "n_complete_tracks": len(track_ids),
    }
    return selection, metrics, tracks


def test_twelve_session_window_keeps_original_indices_and_tracks():
    records = _records(12)
    tracks = list(range(20))
    metrics, native = _window_data(records, [[]] + [tracks] * 11)
    selection, ranking, window_metrics = select_longest_supported_window(metrics, native, records)

    assert selection["status"] == "valid"
    assert selection["window_session_indices"] == list(range(1, 12))
    assert selection["window_session_ids"] == [f"session_{i}" for i in range(1, 12)]
    assert selection["n_complete_tracks"] == 20
    assert window_metrics["track_uid"].nunique() == 20
    assert len(window_metrics) == 220
    assert set(window_metrics["session_index"]) == set(range(1, 12))
    assert len(ranking) > 0
    assert metrics["session_index"].min() == 1

    complete_metrics, complete_native = _window_data(records, [tracks] * 12)
    full_selection, _, _ = select_longest_supported_window(complete_metrics, complete_native, records)
    assert full_selection["window_session_indices"] == list(range(12))


def test_window_tie_breaks_by_cohort_size_then_earlier_start():
    records = _records(4)
    metrics, native = _window_data(records, [
        [1, 2], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [3, 4, 5],
    ])
    selection, _, _ = select_longest_supported_window(metrics, native, records)
    assert selection["window_session_indices"] == [1, 2, 3]
    assert selection["n_complete_tracks"] == 3

    metrics, native = _window_data(records, [
        [1, 2], [1, 2, 3, 4], [1, 2, 3, 4], [3, 4],
    ])
    selection, _, _ = select_longest_supported_window(metrics, native, records)
    assert selection["window_session_indices"] == [0, 1, 2]
    assert selection["n_complete_tracks"] == 2


def test_no_supported_window_is_explicit_and_keeps_observations():
    records = _records(3)
    metrics, native = _window_data(records, [[1], [1], [1]])
    selection, ranking, window_metrics = select_longest_supported_window(metrics, native, records)
    assert selection["status"] == "no_supported_window"
    assert selection["window_length"] == 0
    assert not ranking.empty
    assert window_metrics.empty
    assert len(metrics) == 3


def test_sparse_day0_ranking_uses_only_well_supported_saved_window():
    window = _saved_window(range(1, 101))
    selection, window_metrics, window_tracks = window

    # No Day-0 signal, and one Day-0 signal without any post-baseline row,
    # both defer to the same saved window without inventing observations.
    zero = decide_ranking_population(
        _ranking_metrics(post_ids=range(1, 101)),
        window_selection=selection, window_metrics=window_metrics, window_tracks=window_tracks,
    )
    assert zero.population == "longest_supported_window"
    assert zero.original_day0_signal_valid_count == 0
    assert zero.original_day0_rankable_track_count == 0
    assert zero.window_rankable_track_count == 100
    assert zero.decision_reason == "original_day0_baseline_unavailable"

    one_signal = decide_ranking_population(
        _ranking_metrics(day0_ids=[1], post_ids=range(2, 101)),
        window_selection=selection, window_metrics=window_metrics, window_tracks=window_tracks,
    )
    assert has_usable_day0_baseline(_ranking_metrics(day0_ids=[1], post_ids=range(2, 101)))
    assert one_signal.population == "longest_supported_window"
    assert one_signal.original_day0_signal_valid_count == 1
    assert one_signal.original_day0_rankable_track_count == 0
    assert one_signal.decision_reason == "original_day0_ranking_population_insufficient"


def test_one_rankable_day0_track_yields_to_100_track_window_and_supported_day0_stays_primary():
    selection, window_metrics, window_tracks = _saved_window(range(1, 101))
    one_day0_track = decide_ranking_population(
        _ranking_metrics(day0_ids=[1], post_ids=[1]),
        window_selection=selection, window_metrics=window_metrics, window_tracks=window_tracks,
    )
    assert one_day0_track.population == "longest_supported_window"
    assert one_day0_track.original_day0_signal_valid_count == 1
    assert one_day0_track.original_day0_rankable_track_count == 1
    assert one_day0_track.window_rankable_track_count == 100
    assert one_day0_track.selected_rankable_track_count == 100
    assert one_day0_track.decision_reason == "original_day0_ranking_population_insufficient"
    assert one_day0_track.low_n_warning is False

    two_day0_tracks = decide_ranking_population(
        _ranking_metrics(day0_ids=[1, 2], post_ids=[1, 2]),
        window_selection=selection, window_metrics=window_metrics, window_tracks=window_tracks,
    )
    assert two_day0_tracks.population == "original_day0"
    assert two_day0_tracks.original_day0_rankable_track_count == 2
    assert two_day0_tracks.window_rankable_track_count == 100
    assert two_day0_tracks.selected_rankable_track_count == 2
    assert two_day0_tracks.low_n_warning is True
    assert two_day0_tracks.low_n_threshold == 10


def test_sparse_day0_without_valid_window_is_nonfatal_and_explicit():
    original = _ranking_metrics(day0_ids=[1])
    missing = decide_ranking_population(original)
    assert missing.population == "original_day0"
    assert missing.window_rankable_track_count is None
    assert missing.window_status_reason == "longest_supported_window_metadata_missing"
    assert missing.decision_reason.startswith("original_day0_ranking_population_insufficient;no_qualifying_window:")

    corrupt = decide_ranking_population(
        original,
        window_selection={"status": "valid", "window_session_indices": [4, 5]},
        window_metrics=None,
        window_tracks=None,
        window_unavailable_reason="selected_window_inputs_invalid:ParserError",
    )
    assert corrupt.population == "original_day0"
    assert corrupt.window_rankable_track_count is None
    assert "selected_window_inputs_invalid" in corrupt.decision_reason


def test_rank_population_counts_are_policy_scoped_and_deduplicate_track_identity():
    graph = _ranking_metrics(day0_ids=[1, 2], post_ids=[1, 2], policies={1: "same-track", 2: "same-track"})
    balanced = _ranking_metrics(day0_ids=[3, 4], post_ids=[3, 4])
    balanced["match_policy"] = "balanced"
    all_metrics = pd.concat([graph, balanced], ignore_index=True)
    selection, window_metrics, window_tracks = _saved_window([1, 2, 3, 4])
    window_metrics.loc[window_metrics["roi_id"].isin([3, 4]), "match_policy"] = "balanced"
    window_tracks.loc[window_tracks["roi_id"].isin([3, 4]), "match_policy"] = "balanced"

    graph_decision = decide_ranking_population(
        all_metrics.loc[all_metrics["match_policy"].eq("graph")],
        window_selection=selection,
        window_metrics=window_metrics.loc[window_metrics["match_policy"].eq("graph")],
        window_tracks=window_tracks.loc[window_tracks["match_policy"].eq("graph")],
    )
    balanced_decision = decide_ranking_population(
        all_metrics.loc[all_metrics["match_policy"].eq("balanced")],
        window_selection=selection,
        window_metrics=window_metrics.loc[window_metrics["match_policy"].eq("balanced")],
        window_tracks=window_tracks.loc[window_tracks["match_policy"].eq("balanced")],
    )
    assert graph_decision.original_day0_signal_valid_count == 1
    assert graph_decision.original_day0_rankable_track_count == 1
    assert graph_decision.window_rankable_track_count == 2
    assert graph_decision.population == "longest_supported_window"
    assert balanced_decision.original_day0_signal_valid_count == 2
    assert balanced_decision.original_day0_rankable_track_count == 2
    assert balanced_decision.window_rankable_track_count == 2
    assert balanced_decision.population == "original_day0"


def test_synthetic_twelve_session_extraction_keeps_native_and_partial_track_data(tmp_path):
    records = []
    feature_rows = []
    n_tracks = 20
    mask = np.arange(1, n_tracks + 1, dtype=np.uint16).reshape(1, 4, 5)
    for day in range(12):
        session_id = f"session_{day:02d}"
        acquisition_date = date(2026, 9, 24) + timedelta(days=day)
        mask_path = tmp_path / f"{session_id}_mask.tif"
        red_path = tmp_path / f"{session_id}_red.tif"
        green_path = tmp_path / f"{session_id}_green.tif"
        tifffile.imwrite(mask_path, mask)
        red = (100 + np.arange(n_tracks) * 3 + day).astype(np.uint16).reshape(mask.shape)
        green = (80 + np.arange(n_tracks) * 2 + day * 2).astype(np.uint16).reshape(mask.shape)
        tifffile.imwrite(red_path, red)
        tifffile.imwrite(green_path, green)
        records.append(SessionRecord(day, session_id, acquisition_date, mask_path, red_path, green_path))
        for label in range(1, n_tracks + 1):
            feature_rows.append({
                "session_id": session_id, "label": label, "volume_um3": 1.0,
                "bbox_depth_planes": 1, "touches_xy_edge": False, "touches_z_edge": False,
                "centroid_z_um": 0.0, "centroid_y_um": float(label), "centroid_x_um": float(label),
            })

    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {
            "session_index": record.session_index, "session_id": record.session_id,
            "acquisition_date": record.acquisition_date.isoformat(),
            "mask_path": str(record.mask_path), "red_image_path": str(record.red_image_path),
            "green_image_path": str(record.green_image_path), "required": True,
        }
        for record in records
    ]).to_csv(manifest, index=False)
    match_dir = tmp_path / "synthetic_match"
    match_dir.mkdir()
    pd.DataFrame([
        {
            "session_index": record.session_index, "session_id": record.session_id,
            "acquisition_date": record.acquisition_date.isoformat(),
            "mask_path": str(record.mask_path), "red_image_path": str(record.red_image_path),
            "green_image_path": str(record.green_image_path), "required": True,
        }
        for record in records
    ]).to_csv(match_dir / "session_manifest_resolved.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(match_dir / "roi_features.csv", index=False)
    track_rows = []
    for label in range(1, n_tracks + 1):
        row = {
            "cluster_id": label, "track_uid": f"synthetic-track-{label:02d}",
            "match_policy": "graph", "has_cycle_conflict": False,
            "used_gap_bridge": False, "review_required": False, "review_reasons": "",
            "n_days_present": 11, "first_session_index": 1, "last_session_index": 11,
        }
        for day in range(12):
            row[f"session_{day:02d}_roi"] = label if day > 0 else np.nan
        track_rows.append(row)
    pd.DataFrame(track_rows).to_csv(match_dir / "tracks_graph.csv", index=False)
    (match_dir / "run_log.json").write_text("{}", encoding="utf-8")

    output = run_daywise_matched_roi_pipeline(DaywiseMatchedPipelineConfig(
        dataset=str(tmp_path), manifest=str(manifest), match_dir=str(match_dir),
        output_root=str(tmp_path / "extraction"), policies=("graph",),
        green_dark=0.0, red_dark=0.0,
    ))
    all_observed = pd.read_csv(output / "matched_roi_log_ratio_metrics_all_observed.csv")
    complete = pd.read_csv(output / "matched_roi_day_table_complete.csv")
    native_fits = pd.read_csv(output / "matched_daywise_green_red_linear_fit_summary.csv")
    coverage = pd.read_csv(output / "session_coverage_fit_qc.csv").set_index("session_index")
    window = json.loads((output / "longest_supported_window.json").read_text(encoding="utf-8"))
    window_tracks = pd.read_csv(output / "matched_roi_tracks_longest_supported_window.csv")
    window_metrics = pd.read_csv(output / "matched_roi_metrics_longest_supported_window.csv")

    assert complete.empty
    assert len(all_observed) == 20 * 11
    assert set(all_observed["day"].astype(int)) == set(range(1, 12))
    assert set(all_observed["track_uid"]) == {f"synthetic-track-{label:02d}" for label in range(1, 21)}
    assert native_fits.loc[native_fits["day"].eq(0), "fit_status"].item() == "valid"
    assert coverage.loc[0, "n_native_rois"] == 20
    assert coverage.loc[0, "n_tracked_observed"] == 0
    assert coverage.loc[0, "n_complete_full_manifest"] == 0
    assert window["window_session_indices"] == list(range(1, 12))
    assert window["n_complete_tracks"] == 20
    assert window_tracks["track_uid"].nunique() == 20
    plots = build_quick_plots(output, start_date="20260924", policy="graph", output_dir=tmp_path / "plots")
    assert (plots / "population_longitudinal_summary.png").is_file()
    assert (plots / "daywise_green_red_linear_fit_scatters.png").is_file()
    plot_status = json.loads((plots / "ranked_roi_status.json").read_text())
    assert plot_status["status"] == "evaluated"
    assert plot_status["ranking_population"] == "longest_supported_window"
    assert plot_status["ranking_metric_column"] == "delta_log2_green_over_red_from_window_baseline"
    assert plot_status["ranking_baseline_session_index"] == 1
    assert plot_status["ranking_baseline_session_id"] == "session_01"
    assert plot_status["window_start_session_index"] == 1
    assert plot_status["window_end_session_index"] == 11
    assert plot_status["window_n_source_sessions"] == 11
    assert plot_status["window_n_candidate_tracks"] == 20
    assert "session_00" in plot_status["ranking_baseline_note"]
    assert plot_status["directions"]["increasing"]["status"] == "evaluated"
    n_increasing = plot_status["directions"]["increasing"]["n_ranked_rois"]
    increasing_ranks = pd.read_csv(plots / f"top{n_increasing}_increasing_clusters.csv")
    assert not increasing_ranks.empty
    assert set(increasing_ranks["ranking_population"]) == {"longest_supported_window"}
    assert set(increasing_ranks["ranking_metric_column"]) == {"delta_log2_green_over_red_from_window_baseline"}
    ranked_days = pd.read_csv(plots / f"top{n_increasing}_increasing_roi_day_metrics.csv")
    assert set(ranked_days["session_index"].astype(int)) == set(range(1, 12))
    assert set(ranked_days["day"].astype(int)) == set(range(1, 12))
    assert 0 not in set(ranked_days["session_index"].astype(int))
    first_ranked_id = int(increasing_ranks.iloc[0]["roi_id"])
    ranked_baseline = ranked_days.loc[
        ranked_days["roi_id"].eq(first_ranked_id) & ranked_days["session_index"].eq(1)
    ].iloc[0]
    window_baseline = window_metrics.loc[
        window_metrics["roi_id"].eq(first_ranked_id) & window_metrics["session_index"].eq(1)
    ].iloc[0]
    assert np.isclose(ranked_baseline["ranking_baseline_red"], window_baseline["red"])
    assert np.isclose(ranked_baseline["ranking_baseline_green"], window_baseline["green"])
    assert np.isclose(ranked_baseline["ranking_baseline_brightness"], window_baseline["brightness"])
    assert (plots / f"top{n_increasing}_increasing_delta_log2_heatmap.png").is_file()
    assert (plots / f"top{n_increasing}_increasing_green_fit_residual_vs_day.png").is_file()
    assert plot_status["residual_join_status"] == "joined_by_roi_and_source_session"
    source_residuals = pd.read_csv(output / "matched_roi_metrics_with_session_normalized_residuals_all_observed.csv")
    expected_residuals = source_residuals.set_index(["roi_id", "session_index"])["green_fit_residual"]
    observed_residuals = ranked_days.set_index(["roi_id", "session_index"])["green_fit_residual"]
    pd.testing.assert_series_equal(
        observed_residuals.sort_index(), expected_residuals.reindex(observed_residuals.index).sort_index(),
        check_names=False,
    )

    raw_views = build_ranked_roi_views(
        output, policy="graph", top_n=1, directions=("increasing",),
        output_dir=tmp_path / "ranked_raw_views",
    )
    render_profile = json.loads((raw_views / "ranked_roi_render_profile.json").read_text())
    raw_index = pd.read_csv(raw_views / "ranked_roi_batch_index.csv")
    assert render_profile["ranking_population"] == "longest_supported_window"
    assert render_profile["ranking_baseline_session_index"] == 1
    assert render_profile["ranking_metric_column"] == "delta_log2_green_over_red_from_window_baseline"
    assert render_profile["n_sessions"] == 11
    assert raw_index.loc[0, "track_uid"] == increasing_ranks.iloc[0]["track_uid"]
    raw_png = Path(raw_index.loc[0, "png_path"])
    assert raw_png.is_file()
    raw_metadata = pd.read_csv(raw_index.loc[0, "metadata_csv_path"])
    assert set(raw_metadata["session_index"].astype(int)) == set(range(1, 12))
    assert 0 not in set(raw_metadata["session_index"].astype(int))
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "matched_roi_log_ratio_metrics_all_observed.csv"), all_observed,
    )

    # A usable Day-0 population remains authoritative even when a valid window
    # exists and the decreasing direction has no qualifying ROI.
    source = all_observed.copy()
    baseline_rows = source.loc[source["day"].eq(1)].copy()
    baseline_rows["day"] = 0
    baseline_rows["session_index"] = 0
    baseline_rows["session_id"] = "session_00"
    baseline_rows["acquisition_date"] = "2026-09-24"
    baseline_rows["elapsed_days"] = 0
    baseline_rows["red"] = baseline_rows["red"].astype(float) - 1.0
    baseline_rows["green"] = baseline_rows["green"].astype(float) - 2.0
    day0_complete = compute_log_ratio_metrics(
        pd.concat([baseline_rows, source], ignore_index=True), baseline_day=0,
    )
    day0_complete.to_csv(output / "matched_roi_metrics_primary_final.csv", index=False)
    expected_summary = summarize_roi_metrics(day0_complete)
    expected_increasing = select_top_changing_rois(
        expected_summary, max_rois=20, direction="increasing", ranking_mode="final",
    )
    original_plots = build_quick_plots(
        output, start_date="20260924", policy="graph", output_dir=tmp_path / "day0_plots",
    )
    original_status = json.loads((original_plots / "ranked_roi_status.json").read_text())
    assert original_status["ranking_population"] == "original_day0"
    assert original_status["ranking_metric_column"] == "delta_log2_green_over_red"
    assert original_status["ranking_original_day0_baseline_available"] is True
    assert original_status["directions"]["increasing"]["status"] == "evaluated"
    assert original_status["directions"]["decreasing"]["status"] == "no_eligible_data"
    n_original_increasing = original_status["directions"]["increasing"]["n_ranked_rois"]
    actual_increasing = pd.read_csv(original_plots / f"top{n_original_increasing}_increasing_clusters.csv")
    assert actual_increasing["roi_id"].tolist() == expected_increasing["roi_id"].tolist()
    assert np.allclose(
        actual_increasing["day_last_delta_log2_green_over_red"],
        expected_increasing["day_last_delta_log2_green_over_red"],
    )
    assert (original_plots / "top0_decreasing_clusters.csv").is_file()

    # One valid Day-0 fluorescence row is not enough to suppress a saved
    # ranking window when that row's final Day-0-relative change is zero.
    sparse_baseline = source.loc[source["roi_id"].eq(1)].sort_values("session_index").tail(1).copy()
    sparse_baseline["day"] = 0
    sparse_baseline["session_index"] = 0
    sparse_baseline["session_id"] = "session_00"
    sparse_baseline["acquisition_date"] = "2026-09-24"
    sparse_baseline["elapsed_days"] = 0
    sparse_day0 = compute_log_ratio_metrics(
        pd.concat([sparse_baseline, source], ignore_index=True), baseline_day=0,
    )
    assert has_usable_day0_baseline(sparse_day0)
    sparse_day0.to_csv(output / "matched_roi_metrics_primary_final.csv", index=False)
    sparse_plots = build_quick_plots(
        output, start_date="20260924", policy="graph", output_dir=tmp_path / "sparse_day0_plots",
    )
    sparse_status = json.loads((sparse_plots / "ranked_roi_status.json").read_text())
    assert sparse_status["ranking_population"] == "longest_supported_window"
    assert sparse_status["ranking_decision_reason"] == "original_day0_ranking_population_insufficient"
    assert sparse_status["ranking_original_day0_signal_valid_count"] == 1
    assert sparse_status["ranking_original_day0_rankable_track_count"] == 0
    assert sparse_status["ranking_window_rankable_track_count"] >= 2
    assert sparse_status["ranking_selected_rankable_track_count"] == sparse_status["ranking_window_rankable_track_count"]
    assert sparse_status["ranking_baseline_session_index"] == 1
    assert sparse_status["ranking_baseline_session_id"] == "session_01"
    assert sparse_status["ranking_baseline_source_session_index"] == 1
    assert sparse_status["ranking_baseline_source_session_id"] == "session_01"
    assert sparse_status["ranking_low_n_warning"] is False
    preserved_primary = pd.read_csv(output / "matched_roi_metrics_primary_final.csv")
    assert preserved_primary.loc[preserved_primary["day"].eq(0), "roi_id"].tolist() == [1]
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "matched_roi_log_ratio_metrics_all_observed.csv"), all_observed,
    )
    sparse_increasing = pd.read_csv(
        sparse_plots / f"top{sparse_status['directions']['increasing']['n_ranked_rois']}_increasing_clusters.csv",
    )
    sparse_views = build_ranked_roi_views(
        output, policy="graph", top_n=1, directions=("increasing",),
        output_dir=tmp_path / "sparse_day0_raw_views",
    )
    sparse_profile = json.loads((sparse_views / "ranked_roi_render_profile.json").read_text())
    sparse_raw_index = pd.read_csv(sparse_views / "ranked_roi_batch_index.csv")
    for field in (
        "ranking_population", "ranking_decision_reason", "ranking_baseline_session_index",
        "ranking_baseline_session_id", "ranking_original_day0_signal_valid_count",
        "ranking_original_day0_rankable_track_count", "ranking_window_rankable_track_count",
        "ranking_selected_rankable_track_count", "ranking_baseline_source_session_index",
        "ranking_baseline_source_session_id",
    ):
        assert sparse_profile[field] == sparse_status[field]
    assert int(sparse_raw_index.loc[0, "roi_id"]) == int(sparse_increasing.iloc[0]["roi_id"])
    sparse_raw_metadata = pd.read_csv(sparse_raw_index.loc[0, "metadata_csv_path"])
    assert set(sparse_raw_metadata["session_index"].astype(int)) == set(range(1, 12))
    assert 0 not in set(sparse_raw_metadata["session_index"].astype(int))
    assert "original_day0_ranking_population_insufficient" in (sparse_views / "README.txt").read_text()

    # No valid selected window yields an explicit status and never ranks the
    # all-observed partial tracks as an ad hoc substitute cohort.
    all_observed.to_csv(output / "matched_roi_metrics_primary_final.csv", index=False)
    (output / "longest_supported_window.json").write_text(json.dumps({
        "status": "no_supported_window", "window_session_indices": [],
        "window_session_ids": [], "window_acquisition_dates": [],
        "window_length": 0, "n_complete_tracks": 0,
    }))
    no_window_plots = build_quick_plots(
        output, start_date="20260924", policy="graph", output_dir=tmp_path / "no_window_plots",
    )
    no_window_status = json.loads((no_window_plots / "ranked_roi_status.json").read_text())
    assert no_window_status["status"] == "no_eligible_data"
    assert no_window_status["reason"] == "no_supported_window"
    assert no_window_status["ranking_population"] == "original_day0"
    assert no_window_status["directions"]["increasing"]["n_ranked_rois"] == 0
    assert not list(no_window_plots.glob("top*_increasing_delta_log2_heatmap.png"))
    no_window_raw = build_ranked_roi_views(
        output, policy="graph", top_n=1, directions=("increasing",),
        output_dir=tmp_path / "no_window_raw_views",
    )
    raw_no_window_status = json.loads((no_window_raw / "ranked_roi_render_profile.json").read_text())
    assert raw_no_window_status["status"] == "no_eligible_data"
    assert raw_no_window_status["reason"] == "no_supported_window"
    assert not list(no_window_raw.glob("**/*.png"))
