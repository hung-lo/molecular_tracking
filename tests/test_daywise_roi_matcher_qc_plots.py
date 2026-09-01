from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

import daywise_roi_matcher_qc_plots as qc
from affine_overlap_matcher import AffineOverlapParams, VoxelSpacing
from daywise_roi_matcher_qc_plots import (
    DaywiseQCPlotConfig,
    add_spatial_and_z_qc_columns,
    detect_long_axis,
    generate_daywise_qc_plots,
    select_spatial_examples,
    source_roi_match_fraction,
)
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

    manifest = pd.DataFrame(
        [
            {
                "session_index": 0,
                "session_id": "20260511",
                "acquisition_date": "2026-05-11",
                "mask_path": str(tmp_path / "20260511_mask.tif"),
                "red_image_path": "",
                "green_image_path": "",
                "required": True,
            },
            {
                "session_index": 1,
                "session_id": "20260512",
                "acquisition_date": "2026-05-12",
                "mask_path": str(tmp_path / "20260512_mask.tif"),
                "red_image_path": "",
                "green_image_path": "",
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
        overwrite=True,
    )
    return manifest_path, match_dir


def test_generate_daywise_qc_plots_writes_review_sample(tmp_path: Path) -> None:
    _, match_dir = _build_dataset(tmp_path)
    output_dir = generate_daywise_qc_plots(
        DaywiseQCPlotConfig(
            match_dir=str(match_dir),
            output_dir=str(tmp_path / "qc_plots"),
            sample_limit=3,
            review_seed=11,
        )
    )

    review_sample = pd.read_csv(output_dir / "manual_review_sample.csv")
    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))

    assert not review_sample.empty
    assert (output_dir / "cycle_agreement.png").exists()
    assert run_log["review_sample_rows"] == len(review_sample)
    assert run_log["saved_plots"]


def test_spatial_qc_helpers_preserve_selection_and_zero_source_rois() -> None:
    features = pd.DataFrame({"label": [1, 2, 3], "centroid_z": [1, 2, 3], "centroid_y": [2, 5, 8], "centroid_x": [2, 5, 8]})
    candidates = pd.DataFrame({"session_a": ["a", "a"], "session_b": ["b", "b"], "label_a": [1, 2], "label_b": [1, 2], "score": [.9, .8], "dice": [.9, .8], "distance_um": [1., 2.], "ambiguity": [.1, .2], "accepted_for_track": [True, False], "high_rule": [True, False]})
    enriched = add_spatial_and_z_qc_columns(candidates, features, target_features=features, image_shape_yx=(10, 12))
    selected = select_spatial_examples(enriched, accepted=True, limit=8)
    assert detect_long_axis((10, 12)) == "x"
    assert enriched.loc[selected.index, "raw_delta_z_planes"].iloc[0] == 0
    all_sources = enriched.iloc[:0].copy()
    all_sources["label_a"] = [1, 2, 3]
    all_sources["long_axis_position_normalized"] = [0.1, 0.5, 0.9]
    all_sources["accepted_for_track"] = [True, False, False]
    fraction = source_roi_match_fraction(enriched.iloc[:0], all_source_rois=all_sources)
    assert int(fraction["n_source_rois"].sum()) == 3
    assert int(fraction["n_accepted_source_rois"].sum()) == 1


def test_source_fraction_with_empty_candidates_reports_zero_acceptance() -> None:
    sources = pd.DataFrame({"label_a": [1, 2], "long_axis_position_normalized": [0.1, 0.9], "accepted_for_track": [False, False]})
    result = source_roi_match_fraction(pd.DataFrame(), all_source_rois=sources)
    assert int(result["n_source_rois"].sum()) == 2
    assert int(result["n_accepted_source_rois"].sum()) == 0
    assert float(result["accepted_fraction"].sum()) == 0.0


def test_graph_truth_and_high_confidence_pool_are_distinct() -> None:
    table = pd.DataFrame({
        "session_a": ["a", "a", "a"], "session_b": ["b"] * 3,
        "label_a": [1, 2, 3], "label_b": [11, 12, 13],
        "score": [.9, .8, .7], "dice": [.9, .8, .7],
        "distance_um": [1., 1., 1.], "ambiguity": [.1, .1, .1],
        "accepted_graph": [False, True, True], "accepted_for_track": [False, True, True],
        "high_rule": [True, False, True],
    })
    high_pool = table.loc[table.accepted_graph & table.high_rule]
    assert high_pool["label_a"].tolist() == [3]
    assert not bool(table.loc[0, "accepted_for_track"])


def test_selected_example_ids_survive_sampler_index_reset() -> None:
    table = pd.DataFrame({
        "session_a": ["a"] * 3, "session_b": ["b"] * 3,
        "label_a": [1, 2, 3], "label_b": [11, 12, 13],
        "score": [.5, .9, .7], "dice": [.5, .9, .7],
        "distance_um": [3., 1., 2.], "ambiguity": [.3, .1, .2],
        "accepted_for_track": [True, True, False], "high_rule": [True, True, False],
        "spatial_grid_row": [0, 2, 1], "spatial_grid_col": [0, 2, 1],
    }, index=[10, 20, 30])
    selected = select_spatial_examples(table, accepted=True, limit=1)
    assert selected["label_a"].tolist() == [1]
    assert selected["_candidate_row_id"].tolist() == [10]


def test_selected_example_writeback_flags_only_selected_candidate_key() -> None:
    table = pd.DataFrame({
        "session_a": ["a", "a", "a"], "session_b": ["b", "b", "b"],
        "label_a": [1, 2, 3], "label_b": [11, 12, 13],
        "score": [.9, .8, .7], "dice": [.9, .8, .7],
        "distance_um": [1., 2., 3.], "ambiguity": [.1, .2, .3],
        "accepted_for_track": [True, True, False], "high_rule": [True, True, False],
        "spatial_grid_row": [0, 1, 2], "spatial_grid_col": [0, 1, 2],
    }, index=[10, 20, 30])
    table["_candidate_row_id"] = table.index
    selected = select_spatial_examples(table, accepted=True, limit=1)
    table["selected_high_confidence_example"] = table["_candidate_row_id"].isin(selected["_candidate_row_id"])
    flagged = table.loc[table["selected_high_confidence_example"]]
    assert flagged[["session_a", "session_b", "label_a", "label_b"]].to_dict("records") == [{"session_a": "a", "session_b": "b", "label_a": 1, "label_b": 11}]


def test_axial_high_confidence_population_excludes_graph_accepted_nonhigh_rows() -> None:
    table = pd.DataFrame({
        "accepted_graph": [True, True, False], "high_rule": [True, False, True],
        "raw_delta_z_planes": [2., 99., -4.], "raw_abs_delta_z_planes": [2., 99., 4.],
    })
    accepted_high = table.loc[table["accepted_graph"] & table["high_rule"]]
    median_source = accepted_high.copy()
    assert median_source.index.tolist() == [0]
    assert median_source["raw_delta_z_planes"].median() == 2.0
    assert median_source["raw_delta_z_planes"].tolist() == accepted_high["raw_delta_z_planes"].tolist()


def test_large_z_renderer_uses_two_by_seven_centroid_planes(tmp_path: Path, monkeypatch) -> None:
    mask = np.zeros((9, 12, 12), dtype=np.uint16)
    mask[4, 5, 6] = 1
    mask_path = tmp_path / "mask.tif"
    red_path = tmp_path / "red.tif"
    tifffile.imwrite(mask_path, mask)
    tifffile.imwrite(red_path, np.ones_like(mask))
    manifest = pd.DataFrame({"session_id": ["a", "b"], "mask_path": [str(mask_path)] * 2, "red_image_path": [str(red_path)] * 2})
    table = pd.DataFrame({"session_a": ["a"], "session_b": ["b"], "label_a": [1], "label_b": [1], "accepted_for_track": [True], "raw_abs_delta_z_planes": [4.0], "raw_delta_z_planes": [4.0]})
    original_subplots = plt.subplots
    captured = []

    def capture(*args, **kwargs):
        result = original_subplots(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(qc.plt, "subplots", capture)
    qc._render_large_z_examples(table, manifest, tmp_path, pair_name="a_b", dpi=40)
    fig, axes = captured[0]
    assert axes.shape == (2, 7)
    assert [axes[0, i].get_title().split("dz ")[-1] for i in range(7)] == ["+3", "+2", "+1", "0", "-1", "-2", "-3"]
    plt.close(fig)


def test_axial_scatter_histogram_and_medians_share_high_confidence_population(
    tmp_path: Path, monkeypatch
) -> None:
    match_dir = tmp_path / "match"
    output_dir = tmp_path / "qc"
    (output_dir / "tables").mkdir(parents=True)
    match_dir.mkdir()
    mask_path = tmp_path / "mask.tif"
    red_path = tmp_path / "red.tif"
    tifffile.imwrite(mask_path, np.zeros((1, 10, 10), dtype=np.uint16))
    tifffile.imwrite(red_path, np.ones((1, 10, 10), dtype=np.uint16))

    pd.DataFrame(
        {
            "session_index": [0, 1],
            "session_id": ["a", "b"],
            "acquisition_date": ["2026-05-11", "2026-05-12"],
            "mask_path": [str(mask_path)] * 2,
            "red_image_path": [str(red_path)] * 2,
        }
    ).to_csv(match_dir / "session_manifest_resolved.csv", index=False)
    pd.DataFrame(
        {
            "session_id": ["a", "a", "a", "b", "b", "b"],
            "label": [1, 2, 3, 11, 12, 13],
            "centroid_x": [1, 2, 3, 1, 2, 3],
            "centroid_y": [1, 2, 3, 1, 2, 3],
            "centroid_z": [1, 1, 1, 3, 100, 5],
        }
    ).to_csv(match_dir / "roi_features.csv", index=False)
    (match_dir / "run_log.json").write_text("{}", encoding="utf-8")

    candidates = pd.DataFrame(
        {
            "day_a": ["a", "a", "a"],
            "day_b": ["b", "b", "b"],
            "label_a": [1, 2, 3],
            "label_b": [11, 12, 13],
            "score": [0.9, 0.8, 0.7],
            "dice": [0.9, 0.8, 0.7],
            "distance_um": [1.0, 1.0, 1.0],
            "ambiguity": [0.1, 0.1, 0.1],
            "high_rule": [True, False, True],
            "pair_gap": [1, 1, 1],
        }
    )
    high = candidates.iloc[[0]].copy()
    balanced = candidates.iloc[[0]].copy()
    graph = candidates.iloc[[0, 1]].copy()

    scatter_values: list[tuple[np.ndarray, np.ndarray]] = []
    histogram_values: list[np.ndarray] = []
    original_scatter = qc.plt.Axes.scatter
    original_hist = qc.plt.Axes.hist

    def capture_scatter(axis, x, y, *args, **kwargs):
        scatter_values.append((np.asarray(x), np.asarray(y)))
        return original_scatter(axis, x, y, *args, **kwargs)

    def capture_hist(axis, values, *args, **kwargs):
        histogram_values.append(np.asarray(values))
        return original_hist(axis, values, *args, **kwargs)

    monkeypatch.setattr(qc.plt.Axes, "scatter", capture_scatter)
    monkeypatch.setattr(qc.plt.Axes, "hist", capture_hist)
    monkeypatch.setattr(qc, "_render_match_contact_sheet", lambda *args, **kwargs: None)
    monkeypatch.setattr(qc, "_render_large_z_examples", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        qc,
        "_save_figure",
        lambda path, figure, *, dpi: (plt.close(figure) or path),
    )

    qc._generate_spatial_pair_qc(
        match_dir,
        output_dir,
        candidates,
        [high, balanced, graph],
        dpi=72,
    )

    assert len(scatter_values) == 2
    assert all(np.array_equal(values[1], np.array([2.0])) for values in scatter_values)
    assert len(histogram_values) == 2
    assert all(np.array_equal(values, np.array([2.0])) for values in histogram_values)
    medians = pd.read_csv(output_dir / "tables" / "a_b_axial_shift_bins.csv")
    assert medians["n_matches"].tolist() == [1]
    assert medians["median_delta_z_planes"].tolist() == [2.0]


def test_exported_qc_flags_use_candidate_key_for_accepted_and_rejected(
    tmp_path: Path,
) -> None:
    enriched = pd.DataFrame(
        {
            "session_a": ["a", "a", "a"],
            "session_b": ["b", "b", "b"],
            "label_a": [1, 2, 3],
            "label_b": [11, 12, 13],
            "score": [0.9, 0.8, 0.7],
            "dice": [0.9, 0.8, 0.7],
            "distance_um": [1.0, 2.0, 3.0],
            "ambiguity": [0.1, 0.2, 0.3],
            "accepted_for_track": [True, True, False],
            "high_rule": [True, True, False],
            "spatial_grid_row": [0, 1, 2],
            "spatial_grid_col": [0, 1, 2],
        },
        index=[10, 20, 30],
    )
    enriched["_candidate_row_id"] = enriched.index
    selected_accepted = select_spatial_examples(enriched, accepted=True, limit=1)
    selected_rejected = select_spatial_examples(enriched, accepted=False, limit=1)
    enriched["selected_high_confidence_example"] = enriched["_candidate_row_id"].isin(
        selected_accepted["_candidate_row_id"]
    )
    enriched["selected_rejected_example"] = enriched["_candidate_row_id"].isin(
        selected_rejected["_candidate_row_id"]
    )

    exported_path = tmp_path / "matching_qc_examples.csv"
    enriched.to_csv(exported_path, index=False)
    exported = pd.read_csv(exported_path)
    accepted_flagged = exported.loc[exported["selected_high_confidence_example"]]
    rejected_flagged = exported.loc[exported["selected_rejected_example"]]

    key_columns = ["session_a", "session_b", "label_a", "label_b"]
    assert accepted_flagged[key_columns].to_dict("records") == [
        {"session_a": "a", "session_b": "b", "label_a": 1, "label_b": 11}
    ]
    assert rejected_flagged[key_columns].to_dict("records") == [
        {"session_a": "a", "session_b": "b", "label_a": 3, "label_b": 13}
    ]
