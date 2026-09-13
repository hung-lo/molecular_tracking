from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from endpoint_evaluator import _merge_extraction_enrichment, evaluate_daywise_tracking


REQUIRED_OUTPUTS = {
    "endpoint_events.csv",
    "endpoint_candidates.csv",
    "endpoint_classification.csv",
    "synthetic_gap_benchmark.csv",
    "manual_review_manifest.csv",
    "matching_runtime_summary.csv",
    "matcher_evaluation_summary.json",
    "evaluation_run_log.json",
}


def _matching_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "matching"
    root.mkdir()
    sessions = pd.DataFrame({
        "session_index": [0, 1, 2, 3], "session_id": ["s0", "s1", "s2", "s3"],
        "acquisition_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
        "mask_path": ["", "", "", ""], "red_image_path": ["", "", "", ""], "green_image_path": ["", "", "", ""], "required": [True] * 4,
    })
    sessions.to_csv(root / "session_manifest_resolved.csv", index=False)
    feature_rows = []
    labels_by_session = {"s0": [1, 3, 2], "s1": [20, 4], "s2": [11, 5, 21], "s3": [12, 6]}
    y_by_label = {1: 10, 2: 20, 3: 40, 20: 11, 4: 41, 11: 12, 5: 42, 21: 22, 12: 13, 6: 43}
    for session_index, session_id in enumerate(sessions.session_id):
        for label in labels_by_session[session_id]:
            y = y_by_label[label]
            feature_rows.append({
                "session_index": session_index, "session_id": session_id, "label": label,
                "centroid_z": 1.0, "centroid_y": y, "centroid_x": 10.0,
                "centroid_z_um": 2.0, "centroid_y_um": y * 1.5, "centroid_x_um": 12.5,
                "volume_um3": 10.0, "touches_z_edge": False, "touches_xy_edge": False,
            })
    pd.DataFrame(feature_rows).to_csv(root / "roi_features.csv", index=False)
    transform_rows = []
    for left, right, gap in (("s0", "s1", 1), ("s0", "s2", 2), ("s1", "s2", 1), ("s1", "s3", 2), ("s2", "s3", 1)):
        transform_rows.append({
            "day_a": left, "day_b": right, "pair_gap": gap,
            "z_intercept": 0, "z_scale": 1, "y_intercept": 0, "y_from_y": 1,
            "y_from_x": 0, "x_intercept": 0, "x_from_y": 0, "x_from_x": 1,
            "method": "restricted_affine", "fallback_reason": "", "n_seed": 10, "n_inlier": 10,
            "residual_median_um": 0.2, "residual_p95_um": 0.5,
        })
    pd.DataFrame(transform_rows).to_csv(root / "pairwise_transforms.csv", index=False)
    tracks = pd.DataFrame([
        {"cluster_id": 1, "track_uid": "s0:1", "s0_roi": 1, "s1_roi": pd.NA, "s2_roi": 11, "s3_roi": 12, "match_policy": "graph", "n_days_present": 3, "min_score": .9, "min_dice": .8},
        {"cluster_id": 2, "track_uid": "s0:2", "s0_roi": 2, "s1_roi": 20, "s2_roi": 21, "s3_roi": pd.NA, "match_policy": "graph", "n_days_present": 3, "min_score": .8, "min_dice": .7},
        {"cluster_id": 3, "track_uid": "s0:3", "s0_roi": 3, "s1_roi": 4, "s2_roi": 5, "s3_roi": 6, "match_policy": "graph", "n_days_present": 4, "min_score": .9, "min_dice": .8},
    ])
    tracks.to_csv(root / "tracks_graph.csv", index=False)
    track_edges = pd.DataFrame([{
        "day_a": "s0", "day_b": "s2", "label_a": 1, "label_b": 11, "pair_gap": 2,
        "accepted_for_track": True, "score": .9, "dice": .8, "distance_um": 2.0,
    }])
    track_edges.to_csv(root / "track_edges_graph.csv", index=False)
    candidates = pd.DataFrame([{
        "day_a": "s0", "day_b": "s2", "label_a": 1, "label_b": 11,
        "candidate_source": "both", "dice": .8, "iou": .7, "ambiguity": .5,
        "distance_um": 2.0, "score": .9,
    }])
    candidates.to_csv(root / "pairwise_candidates.csv", index=False)
    matches = candidates.assign(
        match_policy="graph", graph_status="accepted", refined_score=.95,
        graph_support_count=3, graph_support_fraction=.75,
        graph_residual_median_um=.6, graph_inlier_fraction=.8,
        assignment_source="graph",
    )
    matches.to_csv(root / "pairwise_matches_graph.csv", index=False)
    matches.to_csv(root / "pairwise_matches_high.csv", index=False)
    matches.to_csv(root / "pairwise_matches_balanced.csv", index=False)
    pd.DataFrame([
        {"day_a": "s0", "day_b": "s1", "pair_gap": 1, "elapsed_sec": .1, "n_a": 3, "n_b": 2, "transform_method": "restricted_affine", "transform_fallback_reason": ""},
        {"day_a": "s0", "day_b": "s2", "pair_gap": 2, "elapsed_sec": .2, "n_a": 3, "n_b": 3, "transform_method": "restricted_affine", "transform_fallback_reason": ""},
    ]).to_csv(root / "pairwise_summary.csv", index=False)
    pd.DataFrame([
        {"day_a": "s0", "day_b": "s1", "pair_gap": 1, "elapsed_sec": .1, "n_graph": 2, "n_graph_anchors": 1, "n_graph_changed": 0},
        {"day_a": "s0", "day_b": "s2", "pair_gap": 2, "elapsed_sec": .2, "n_graph": 1, "n_graph_anchors": 1, "n_graph_changed": 0},
    ]).to_csv(root / "pairwise_summary_graph.csv", index=False)
    (root / "run_log.json").write_text(json.dumps({
        "git_commit": "fixture-commit", "algorithm_version": "fixture",
        "spacing": {"z_um": 2.0, "y_um": 1.5, "x_um": 1.25},
    }), encoding="utf-8")
    return root


def _hash_tree(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        result[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def test_evaluator_cli_path_is_deterministic_complete_and_read_only(tmp_path: Path) -> None:
    matching = _matching_fixture(tmp_path)
    before = _hash_tree(matching)
    output = tmp_path / "evaluation"
    summary = evaluate_daywise_tracking(matching, output, overwrite=True, max_review_panels=10)

    assert summary["n_endpoints"] == 2
    assert summary["spacing_um"] == {"z_um": 2.0, "y_um": 1.5, "x_um": 1.25}
    assert summary["spacing_source"] == "run_log.json"
    assert REQUIRED_OUTPUTS.issubset({p.name for p in output.iterdir() if p.is_file()})
    endpoints = pd.read_csv(output / "endpoint_events.csv")
    gap = endpoints.loc[endpoints["track_uid"] == "s0:1"].iloc[0]
    assert bool(gap["same_track_returns"])
    assert int(gap["same_track_return_gap"]) == 2
    assert not (endpoints["end_session_index"] == 3).any()

    candidate_rows = pd.read_csv(output / "endpoint_candidates.csv")
    evidence = candidate_rows.loc[(candidate_rows["source_track_uid"] == "s0:1") & (candidate_rows["target_label"] == 11)].iloc[0]
    assert evidence["graph_status"] == "accepted"
    assert bool(evidence["existing_graph_match"])
    assert float(evidence["refined_score"]) == pytest.approx(.95)

    synthetic = pd.read_csv(output / "synthetic_gap_benchmark.csv")
    assert not synthetic.empty
    assert set(synthetic["session_gap"]).issubset({2, 3})
    runtime = pd.read_csv(output / "matching_runtime_summary.csv")
    assert runtime["graph_stage_seconds"].isna().all()
    assert list((output / "review_panels").glob("*.png"))
    assert (output / "matching_runtime_by_pair.png").is_file()
    assert (output / "matching_runtime_vs_candidate_count.png").is_file()
    assert (output / "matching_runtime_by_gap.png").is_file()
    assert not list(output.rglob("*.pdf"))
    assert _hash_tree(matching) == before

    deterministic_files = [
        "endpoint_events.csv", "endpoint_candidates.csv", "endpoint_classification.csv",
        "synthetic_gap_benchmark.csv", "manual_review_manifest.csv", "matching_runtime_summary.csv",
        "matcher_evaluation_summary.json",
    ]
    first_bytes = {name: (output / name).read_bytes() for name in deterministic_files}
    evaluate_daywise_tracking(matching, output, overwrite=True, max_review_panels=10)
    assert {name: (output / name).read_bytes() for name in deterministic_files} == first_bytes
    assert _hash_tree(matching) == before

    cli_output = tmp_path / "evaluation_cli"
    result = subprocess.run(
        [
            sys.executable, "matching/evaluate_daywise_tracking.py",
            "--match-dir", str(matching), "--output-dir", str(cli_output), "--overwrite",
            "--max-review-panels", "2", "--synthetic-min-dice", "0.05",
        ],
        cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (cli_output / "evaluation_run_log.json").is_file()
    run_log = json.loads((cli_output / "evaluation_run_log.json").read_text())
    assert run_log["canonical_matching_rerun"] is False
    assert run_log["eclipse_recomputed"] is False
    assert run_log["png_only"] is True


def test_graph_policy_does_not_silently_fallback_to_balanced(tmp_path: Path) -> None:
    matching = _matching_fixture(tmp_path)
    (matching / "tracks_graph.csv").unlink()
    pd.DataFrame([{"track_uid": "x"}]).to_csv(matching / "tracks_balanced.csv", index=False)
    with pytest.raises(FileNotFoundError, match="tracks_graph.csv"):
        evaluate_daywise_tracking(matching, tmp_path / "evaluation", policy="graph", overwrite=True)


def test_optional_extraction_tables_merge_without_duplicate_key_failure(tmp_path: Path) -> None:
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    for filename, column, value in (
        ("matched_track_qc_summary.csv", "green", 2.0),
        ("matched_roi_geometry_qc_long.csv", "red", 3.0),
        ("graph_affine_agreement_track_metadata.csv", "eclipse_z", -1.0),
    ):
        pd.DataFrame({"track_uid": ["track-1"], "session_index": [0], column: [value]}).to_csv(extraction / filename, index=False)
    endpoints = pd.DataFrame({"track_uid": ["track-1"], "end_session_index": [0], "green": [pd.NA], "red": [pd.NA], "eclipse_z": [pd.NA], "eclipse_core_state": [""]})
    merged = _merge_extraction_enrichment(endpoints, extraction)
    assert merged.loc[0, "green"] == 2.0
    assert merged.loc[0, "red"] == 3.0
    assert merged.loc[0, "eclipse_z"] == -1.0


def test_optional_extraction_enrichment_accepts_string_qc_state_in_all_missing_column(tmp_path: Path) -> None:
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    pd.DataFrame(
        {
            "track_uid": ["track-1"],
            "session_index": [0],
            "segmentation_qc_status": ["not_configured"],
            "geometry_qc_pass": [pd.NA],
        }
    ).to_csv(extraction / "matched_roi_geometry_qc_long.csv", index=False)

    # Mirrors endpoint_events.csv construction: a fixed optional column with no
    # observed values is inferred as float64 before enrichment.
    endpoints = pd.DataFrame(
        {
            "track_uid": ["track-1"],
            "end_session_index": [0],
            "segmentation_qc_status": [float("nan")],
            "geometry_qc_pass": [float("nan")],
        }
    )

    merged = _merge_extraction_enrichment(endpoints, extraction)
    assert merged.loc[0, "segmentation_qc_status"] == "not_configured"
    assert pd.isna(merged.loc[0, "geometry_qc_pass"])


def test_optional_extraction_enrichment_accepts_boolean_qc_state_in_all_missing_column(tmp_path: Path) -> None:
    extraction = tmp_path / "extraction"
    extraction.mkdir()
    pd.DataFrame(
        {
            "track_uid": ["track-1"],
            "segmentation_failure": [False],
            "edge_heavy": [False],
            "review_required": [False],
            "has_graph_only_edge": [False],
        }
    ).to_csv(extraction / "matched_track_qc_summary.csv", index=False)

    endpoints = pd.DataFrame(
        {
            "track_uid": ["track-1"],
            "end_session_index": [0],
            "segmentation_failure": [float("nan")],
            "edge_heavy": [float("nan")],
            "review_required": [float("nan")],
            "has_graph_only_edge": [float("nan")],
        }
    )

    merged = _merge_extraction_enrichment(endpoints, extraction)
    assert merged.loc[0, "segmentation_failure"] is False or merged.loc[0, "segmentation_failure"] == False
    assert merged.loc[0, "edge_heavy"] is False or merged.loc[0, "edge_heavy"] == False
    assert merged.loc[0, "review_required"] is False or merged.loc[0, "review_required"] == False
    assert merged.loc[0, "has_graph_only_edge"] is False or merged.loc[0, "has_graph_only_edge"] == False
