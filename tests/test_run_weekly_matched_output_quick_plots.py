from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from run_weekly_matched_output_quick_plots import _filter_table_by_policy, _resolve_output_dir
import run_daywise_green_red_linear_fit_summary as fit_plots
import run_matched_roi_quick_view as raw_view
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


def test_raw_view_reads_only_requested_z_planes(tmp_path: Path, monkeypatch) -> None:
    mask = np.zeros((3, 5, 5), dtype=np.uint16)
    mask[0, 2, 2] = 1
    red = np.ones_like(mask, dtype=np.uint16)
    green = np.full_like(mask, 2, dtype=np.uint16)
    paths = {
        "mask": tmp_path / "mask.tif",
        "red": tmp_path / "red.tif",
        "green": tmp_path / "green.tif",
    }
    tifffile.imwrite(paths["mask"], mask, photometric="minisblack")
    tifffile.imwrite(paths["red"], red, photometric="minisblack")
    tifffile.imwrite(paths["green"], green, photometric="minisblack")
    tracks = pd.DataFrame({"cluster_id": [1], "roi_id": [1], "track_uid": ["t1"], "s0_roi": [1]})
    sessions = pd.DataFrame({
        "session_index": [0], "session_id": ["s0"], "acquisition_date": ["2026-01-01"],
        "elapsed_days": [0], "mask_path": [str(paths["mask"])],
        "red_image_path": [str(paths["red"])], "green_image_path": [str(paths["green"])],
    })
    real_imread = raw_view.tifffile.imread
    calls: list[int | None] = []

    def read(path, *args, **kwargs):
        calls.append(kwargs.get("key"))
        return real_imread(path, *args, **kwargs)

    monkeypatch.setattr(raw_view.tifffile, "imread", read)
    raw_view.plot_matched_roi_raw_slices(
        cluster_id=1, tracks_table=tracks, session_table=sessions,
        output_path=tmp_path / "one_plane.png", render_z_radius=0,
    )
    assert calls == [None, 0, 0]

    calls.clear()
    raw_view.plot_matched_roi_raw_slices(
        cluster_id=1, tracks_table=tracks, session_table=sessions,
        output_path=tmp_path / "two_planes.png", render_z_radius=1,
    )
    assert calls == [None, 1, 1, 0, 0]


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


def test_fit_plots_use_supplied_summary_without_refitting(tmp_path: Path, monkeypatch) -> None:
    metrics = pd.DataFrame({
        "day": [0, 0, 1, 1], "session_id": ["s0"] * 2 + ["s1"] * 2,
        "red": [1., 2., 1., 2.], "green": [1., 2., 1., 2.],
        "red_signal_qc_pass": [True] * 4, "green_signal_qc_pass": [True] * 4,
        "acquisition_date": ["2026-01-01"] * 2 + ["2026-01-03"] * 2,
    })
    fits = pd.DataFrame({
        "day": [0, 1], "slope": [9., 9.], "intercept": [0., 0.], "r_squared": [0.1, 0.1], "n_rois": [2, 2],
        "slope_ci_low": [8., 8.], "slope_ci_high": [10., 10.], "intercept_ci_low": [-1., -1.], "intercept_ci_high": [1., 1.],
    })
    monkeypatch.setattr(fit_plots, "summarize_daily_green_red_linear_fits", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected refit")))
    fit_plots.plot_daywise_scatter_summary(metrics, fits, tmp_path / "scatter.png", start_date="20260101")
    fit_plots.plot_fit_parameter_summary(fits, tmp_path / "params.png", roi_metrics=metrics, start_date="20260101")
    assert (tmp_path / "scatter.png").is_file() and (tmp_path / "params.png").is_file()
