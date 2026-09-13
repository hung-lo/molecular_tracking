from __future__ import annotations

from pathlib import Path

import pandas as pd
import subprocess
import sys

from endpoint_evaluator import evaluate_daywise_tracking


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
            feature_rows.append({"session_index": session_index, "session_id": session_id, "label": label, "centroid_z": 1.0, "centroid_y": y, "centroid_x": 10.0, "centroid_z_um": 5.0, "centroid_y_um": y, "centroid_x_um": 10.0, "volume_um3": 10.0, "touches_z_edge": False, "touches_xy_edge": False})
    pd.DataFrame(feature_rows).to_csv(root / "roi_features.csv", index=False)
    transform_rows = []
    for left, right, gap in (("s0", "s1", 1), ("s0", "s2", 2), ("s1", "s2", 1), ("s1", "s3", 2), ("s2", "s3", 1)):
        transform_rows.append({"day_a": left, "day_b": right, "pair_gap": gap, "z_intercept": 0, "z_scale": 1, "y_intercept": 0, "y_from_y": 1, "y_from_x": 0, "x_intercept": 0, "x_from_y": 0, "x_from_x": 1, "method": "restricted_affine", "fallback_reason": "", "n_seed": 10, "n_inlier": 10})
    pd.DataFrame(transform_rows).to_csv(root / "pairwise_transforms.csv", index=False)
    tracks = pd.DataFrame([
        {"cluster_id": 1, "track_uid": "s0:1", "s0_roi": 1, "s1_roi": pd.NA, "s2_roi": 11, "s3_roi": 12, "match_policy": "graph", "n_days_present": 3, "min_score": .9, "min_dice": .8},
        {"cluster_id": 2, "track_uid": "s0:2", "s0_roi": 2, "s1_roi": 20, "s2_roi": 21, "s3_roi": pd.NA, "match_policy": "graph", "n_days_present": 3, "min_score": .8, "min_dice": .7},
        {"cluster_id": 3, "track_uid": "s0:3", "s0_roi": 3, "s1_roi": 4, "s2_roi": 5, "s3_roi": 6, "match_policy": "graph", "n_days_present": 4, "min_score": .9, "min_dice": .8},
    ])
    tracks.to_csv(root / "tracks_graph.csv", index=False)
    candidates = pd.DataFrame([{"day_a": "s0", "day_b": "s2", "label_a": 1, "label_b": 11, "candidate_source": "both", "dice": .8, "iou": .7, "ambiguity": .5, "score": .9}])
    candidates.to_csv(root / "pairwise_candidates.csv", index=False)
    matches = candidates.assign(match_policy="graph", graph_status="accepted", refined_score=.9)
    matches.to_csv(root / "pairwise_matches_graph.csv", index=False)
    matches.to_csv(root / "pairwise_matches_high.csv", index=False)
    matches.to_csv(root / "pairwise_matches_balanced.csv", index=False)
    pd.DataFrame([{"day_a": "s0", "day_b": "s1", "pair_gap": 1, "elapsed_sec": .1, "n_a": 3, "n_b": 2, "transform_method": "restricted_affine", "transform_fallback_reason": ""}]).to_csv(root / "pairwise_summary.csv", index=False)
    return root


def test_evaluator_cli_path_is_deterministic_and_read_only(tmp_path: Path) -> None:
    matching = _matching_fixture(tmp_path)
    before = (matching / "tracks_graph.csv").read_bytes()
    output = tmp_path / "evaluation"
    summary = evaluate_daywise_tracking(matching, output, overwrite=True, max_review_panels=10)
    assert summary["n_endpoints"] >= 2
    endpoints = pd.read_csv(output / "endpoint_events.csv")
    assert bool(endpoints.loc[endpoints["track_uid"] == "s0:1", "same_track_returns"].iloc[0])
    assert (output / "endpoint_candidates.csv").is_file()
    assert (output / "endpoint_classification.csv").is_file()
    assert (output / "synthetic_gap_benchmark.csv").is_file()
    assert (output / "manual_review_manifest.csv").is_file()
    assert (output / "matching_runtime_summary.csv").is_file()
    assert (output / "matcher_evaluation_summary.json").is_file()
    assert (output / "evaluation_run_log.json").is_file()
    assert list((output / "review_panels").glob("*.png"))
    assert not list(output.rglob("*.pdf"))
    assert (matching / "tracks_graph.csv").read_bytes() == before

    cli_output = tmp_path / "evaluation_cli"
    result = subprocess.run(
        [sys.executable, "matching/evaluate_daywise_tracking.py", "--match-dir", str(matching), "--output-dir", str(cli_output), "--overwrite", "--max-review-panels", "2"],
        cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (cli_output / "evaluation_run_log.json").is_file()
