from __future__ import annotations

from pathlib import Path
import csv
import json
import shutil

import numpy as np
import pandas as pd
import tifffile

import run_daywise_graph_matching as graph_runner
import run_daywise_roi_matching as affine_runner
from run_daywise_graph_matching import _affine_git_commit_from_log, _pair_table_groups, run_daywise_graph_matching
from tools.compare_matcher_outputs import compare_matcher_outputs


def _build_dataset(tmp_path: Path) -> Path:
    mask = np.zeros((2, 3, 3), dtype=np.uint16)
    mask[0, 0, 0] = 1
    mask[0, 1, 1] = 2
    for day in ["20260511", "20260512", "20260513"]:
        tifffile.imwrite(tmp_path / f"{day}_mask.tif", mask)
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["session_index", "session_id", "acquisition_date", "mask_path", "red_image_path", "green_image_path", "required"],
        )
        writer.writeheader()
        writer.writerow({"session_index": 0, "session_id": "20260511", "acquisition_date": "2026-05-11", "mask_path": str(tmp_path / "20260511_mask.tif"), "red_image_path": "", "green_image_path": "", "required": "true"})
        writer.writerow({"session_index": 1, "session_id": "20260512", "acquisition_date": "2026-05-12", "mask_path": str(tmp_path / "20260512_mask.tif"), "red_image_path": "", "green_image_path": "", "required": "true"})
        writer.writerow({"session_index": 2, "session_id": "20260513", "acquisition_date": "2026-05-13", "mask_path": str(tmp_path / "20260513_mask.tif"), "red_image_path": "", "green_image_path": "", "required": "true"})
    return manifest_path


def test_run_daywise_graph_matching_exports_graph_tables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(affine_runner, "_git_commit", lambda: "same-stage-commit")
    monkeypatch.setattr(graph_runner, "_git_commit", lambda: "same-stage-commit")
    manifest_path = _build_dataset(tmp_path)
    stages: list[tuple[str, float]] = []
    output_dir = run_daywise_graph_matching(
        manifest_path=manifest_path,
        output_dir=tmp_path / "graph_out",
        overwrite=True,
        skip_qc=True,
        stage_callback=lambda key, duration: stages.append((key, duration)),
    )

    assert (output_dir / "pairwise_matches_graph.csv").exists()
    assert (output_dir / "tracks_graph.csv").exists()
    assert (output_dir / "graph_match_changes.csv").exists()

    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["graph_matcher_algorithm_version"] == "local_spatial_graph_v1"
    assert run_log["graph_matcher_implementation_version"] == "kdtree_prefilter_v1"
    assert run_log["graph_row_counts"]["tracks_graph"] > 0
    assert run_log["graph_output_paths"]["tracks_graph"].endswith("tracks_graph.csv")
    assert run_log["affine_matcher_git_commit"] == "same-stage-commit"
    assert run_log["graph_runner_git_commit"] == "same-stage-commit"
    assert run_log["git_commit"] == run_log["graph_runner_git_commit"]
    assert run_log["git_commit_role"] == "graph_runner"
    assert [key for key, _duration in stages] == [
        "daywise_affine_roi_matching",
        "graph_roi_matching",
    ]
    assert all(duration >= 0 for _key, duration in stages)
    graph_timings = run_log["runtime_profile"]["graph_stage_durations_seconds"]
    assert all(graph_timings[key] >= 0 for key in ("pair_refinement_total", "graph_track_building_total", "graph_cycle_consistency_total", "graph_runner_total"))


def test_pair_table_groups_normalize_keys_and_preserve_row_order() -> None:
    table = pd.DataFrame(
        [
            {"day_a": 20260512, "day_b": 20260513, "row": "first"},
            {"day_a": 20260511, "day_b": 20260512, "row": "second"},
            {"day_a": 20260512, "day_b": 20260513, "row": "third"},
        ]
    )
    groups = _pair_table_groups(table)
    assert list(groups) == [("20260512", "20260513"), ("20260511", "20260512")]
    assert groups[("20260512", "20260513")]["row"].tolist() == ["first", "third"]
    assert groups[("20260511", "20260512")]["row"].tolist() == ["second"]


def test_graph_pair_workers_preserve_exact_scientific_outputs(tmp_path: Path) -> None:
    manifest_path = _build_dataset(tmp_path)
    outputs = [
        run_daywise_graph_matching(
            manifest_path=manifest_path,
            output_dir=tmp_path / f"graph_workers_{workers}",
            pair_workers=workers,
            overwrite=True,
            skip_qc=True,
        )
        for workers in (1, 2)
    ]
    scientific_csvs = [
        "pairwise_matches_graph.csv", "tracks_graph.csv", "cycle_consistency_graph.csv",
        "cycle_edge_checks_graph.csv", "track_edges_graph.csv", "track_length_summary_graph.csv",
        "graph_match_changes.csv",
    ]
    for filename in scientific_csvs:
        pd.testing.assert_frame_equal(pd.read_csv(outputs[0] / filename), pd.read_csv(outputs[1] / filename))
    left_summary = pd.read_csv(outputs[0] / "pairwise_summary_graph.csv").drop(columns="elapsed_sec")
    right_summary = pd.read_csv(outputs[1] / "pairwise_summary_graph.csv").drop(columns="elapsed_sec")
    pd.testing.assert_frame_equal(left_summary, right_summary)

    run_log = json.loads((outputs[1] / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["pair_workers"] == 2
    assert len(run_log["runtime_profile"]["graph_pair_timings_seconds"]) == 3


def test_resumed_graph_run_preserves_affine_commit(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _build_dataset(tmp_path)
    output_dir = tmp_path / "resumed_graph"
    monkeypatch.setattr(affine_runner, "_git_commit", lambda: "old_affine_commit")
    monkeypatch.setattr(graph_runner, "_git_commit", lambda: "old_affine_commit")
    run_daywise_graph_matching(
        manifest_path=manifest_path,
        output_dir=output_dir,
        overwrite=True,
        skip_qc=True,
    )
    scientific_before = {
        path.name: path.read_bytes()
        for path in output_dir.glob("*.csv")
    }
    reference_dir = tmp_path / "pre_resume_reference"
    shutil.copytree(output_dir, reference_dir)

    monkeypatch.setattr(graph_runner, "_git_commit", lambda: "current_graph_commit")
    run_daywise_graph_matching(
        manifest_path=manifest_path,
        output_dir=output_dir,
        resume=True,
        skip_qc=True,
    )

    run_log = json.loads((output_dir / "run_log.json").read_text(encoding="utf-8"))
    assert run_log["affine_matcher_git_commit"] == "old_affine_commit"
    assert run_log["graph_runner_git_commit"] == "current_graph_commit"
    assert run_log["git_commit"] == "current_graph_commit"
    assert scientific_before == {path.name: path.read_bytes() for path in output_dir.glob("*.csv")}
    assert compare_matcher_outputs(reference_dir, output_dir)["scientific_output_equivalence"] == "PASS"


def test_resumed_legacy_affine_log_without_commit_uses_null(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _build_dataset(tmp_path)
    output_dir = tmp_path / "legacy_graph"
    monkeypatch.setattr(affine_runner, "_git_commit", lambda: "initial_commit")
    monkeypatch.setattr(graph_runner, "_git_commit", lambda: "initial_commit")
    run_daywise_graph_matching(
        manifest_path=manifest_path,
        output_dir=output_dir,
        overwrite=True,
        skip_qc=True,
    )
    run_log_path = output_dir / "run_log.json"
    legacy_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    for field in ("git_commit", "git_commit_role", "affine_matcher_git_commit", "graph_runner_git_commit"):
        legacy_log.pop(field, None)
    run_log_path.write_text(json.dumps(legacy_log, indent=2, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr(graph_runner, "_git_commit", lambda: "current_graph_commit")
    run_daywise_graph_matching(
        manifest_path=manifest_path,
        output_dir=output_dir,
        resume=True,
        skip_qc=True,
    )

    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    assert run_log["affine_matcher_git_commit"] is None
    assert run_log["graph_runner_git_commit"] == "current_graph_commit"
    assert run_log["git_commit"] == "current_graph_commit"


def test_resumed_legacy_graph_log_does_not_invent_affine_commit(tmp_path: Path, monkeypatch) -> None:
    manifest_path = _build_dataset(tmp_path)
    output_dir = tmp_path / "legacy_graph_with_old_top_level_commit"
    monkeypatch.setattr(affine_runner, "_git_commit", lambda: "old_affine_commit")
    monkeypatch.setattr(graph_runner, "_git_commit", lambda: "old_graph_commit")
    run_daywise_graph_matching(
        manifest_path=manifest_path,
        output_dir=output_dir,
        overwrite=True,
        skip_qc=True,
    )

    run_log_path = output_dir / "run_log.json"
    legacy_graph_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    legacy_graph_log["git_commit"] = "old_graph_commit"
    for field in ("git_commit_role", "affine_matcher_git_commit", "graph_runner_git_commit"):
        legacy_graph_log.pop(field, None)
    # Keep graph-stage markers from the pre-2d750c1 graph runner.
    run_log_path.write_text(json.dumps(legacy_graph_log, indent=2, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr(graph_runner, "_git_commit", lambda: "current_graph_commit")
    run_daywise_graph_matching(
        manifest_path=manifest_path,
        output_dir=output_dir,
        resume=True,
        skip_qc=True,
    )

    run_log = json.loads(run_log_path.read_text(encoding="utf-8"))
    assert run_log["affine_matcher_git_commit"] is None
    assert run_log["graph_runner_git_commit"] == "current_graph_commit"
    assert run_log["git_commit"] == "current_graph_commit"
    assert run_log["git_commit_role"] == "graph_runner"


def test_affine_provenance_helper_distinguishes_log_shapes() -> None:
    assert _affine_git_commit_from_log({"git_commit": "pure_affine"}) == "pure_affine"
    assert _affine_git_commit_from_log({"git_commit": "old_graph", "graph_runner_version": "v1"}) is None
    assert _affine_git_commit_from_log({"affine_matcher_git_commit": "modern_affine", "git_commit": "graph"}) == "modern_affine"
    assert _affine_git_commit_from_log({"graph_params": {}}) is None
