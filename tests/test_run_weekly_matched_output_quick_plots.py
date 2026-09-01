from __future__ import annotations

from pathlib import Path

import pandas as pd

from run_weekly_matched_output_quick_plots import _filter_table_by_policy, _resolve_output_dir
from run_ranked_roi_quick_views import _resolve_run_inputs
from roi_log_ratio_analysis import select_top_changing_rois


def test_filter_table_by_policy_filters_suffixed_policy_columns() -> None:
    table = pd.DataFrame(
        {
            "roi_id": [1, 2, 3, 4],
            "day": [0, 1, 0, 1],
            "match_policy_x": ["high", "high", "balanced", "balanced"],
            "match_policy_y": ["high", "high", "balanced", "balanced"],
        }
    )

    filtered = _filter_table_by_policy(table, "high")

    assert filtered["roi_id"].tolist() == [1, 2]
    assert set(filtered["match_policy_x"].astype(str)) == {"high"}
    assert set(filtered["match_policy_y"].astype(str)) == {"high"}


def test_filter_table_by_policy_is_noop_without_policy_column() -> None:
    table = pd.DataFrame({"roi_id": [1, 2], "day": [0, 1]})

    filtered = _filter_table_by_policy(table, "high")

    pd.testing.assert_frame_equal(filtered, table)


def test_resolve_output_dir_separates_policies() -> None:
    analysis_dir = Path('/tmp/analysis')

    default_high = _resolve_output_dir(analysis_dir, None, 'high')
    custom_balanced = _resolve_output_dir(analysis_dir, Path('/tmp/custom_quick_plots'), 'balanced')

    assert default_high == analysis_dir / 'quick_plots' / 'high'
    assert custom_balanced == Path('/tmp/custom_quick_plots').resolve() / 'balanced'


def test_ranked_view_resolves_nested_master_run_and_manifest_fallback(tmp_path: Path) -> None:
    extraction = tmp_path / "extraction"
    matching = tmp_path / "matching"
    extraction.mkdir()
    matching.mkdir()
    for name in ("matched_roi_log_ratio_metrics_complete.csv", "matched_roi_intensity_results_raw.csv"):
        (extraction / name).touch()
    (matching / "session_manifest_resolved.csv").touch()
    metrics, raw, manifest = _resolve_run_inputs(tmp_path)
    assert metrics.parent == extraction
    assert raw.parent == extraction
    assert manifest.parent == matching
    (extraction / "session_manifest_resolved.csv").touch()
    assert _resolve_run_inputs(tmp_path)[2].parent == extraction


def test_final_directional_ranking_is_sign_correct_and_not_random() -> None:
    table = pd.DataFrame({
        "roi_id": [1, 2, 3], "day0_brightness": [10., 10., 10.], "day0_green": [1., 1., 1.],
        "red_cv": [0., 0., 0.], "min_delta_log2_green_over_red": [-3., -1., 1.],
        "max_delta_log2_green_over_red": [2., 1., 3.], "delta_log2_range": [4., 2., 5.],
        "day_last_delta_log2_green_over_red": [2., -1., 3.],
    })
    increasing = select_top_changing_rois(table, max_rois=3, direction="increasing", ranking_mode="final")
    decreasing = select_top_changing_rois(table, max_rois=3, direction="decreasing", ranking_mode="final")
    assert increasing.roi_id.tolist() == [3, 1]
    assert decreasing.roi_id.tolist() == [2]
    assert set(increasing.selection_mode) == {"final"}
    assert set(decreasing.selection_metric_column) == {"day_last_delta_log2_green_over_red"}
