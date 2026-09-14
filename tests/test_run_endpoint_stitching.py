from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from endpoint_evaluator import detect_endpoint_events, search_endpoint_candidates, classify_endpoint_events
from run_endpoint_stitching import _manual_manifest, _prepare_output, run_endpoint_stitching
from test_endpoint_stitcher import stitch_fixture


def _hashes(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(root.glob("*")) if path.is_file()}


def test_cli_orchestration_is_read_only_and_blocks_failed_benchmark_write(tmp_path) -> None:
    sessions, features, transforms, tracks = stitch_fixture()
    match_dir, evaluation_dir, output_dir = tmp_path / "match", tmp_path / "evaluation", tmp_path / "stitch"
    match_dir.mkdir(); evaluation_dir.mkdir()
    sessions.to_csv(match_dir / "session_manifest_resolved.csv", index=False)
    features.to_csv(match_dir / "roi_features.csv", index=False)
    transforms.to_csv(match_dir / "pairwise_transforms.csv", index=False)
    tracks.to_csv(match_dir / "tracks_graph.csv", index=False)
    pd.DataFrame(columns=["source_track_uid", "target_track_uid"]).to_csv(match_dir / "track_edges_graph.csv", index=False)
    (match_dir / "run_log.json").write_text(json.dumps({"spacing": {"z_um": 1, "y_um": 1, "x_um": 1}}))
    endpoints = detect_endpoint_events(tracks, features, sessions)
    candidates, status = search_endpoint_candidates(endpoints, tracks, features, sessions, transforms)
    classifications = classify_endpoint_events(endpoints, candidates, status)
    endpoints.to_csv(evaluation_dir / "endpoint_events.csv", index=False)
    candidates.to_csv(evaluation_dir / "endpoint_candidates.csv", index=False)
    classifications.to_csv(evaluation_dir / "endpoint_classification.csv", index=False)
    pd.DataFrame(columns=["case_id"]).to_csv(evaluation_dir / "synthetic_gap_benchmark.csv", index=False)
    (evaluation_dir / "evaluation_run_log.json").write_text(json.dumps({
        "evaluator_version": "daywise_endpoint_evaluator_test",
        "evaluator_repo_git_commit": "evaluator-fixture-commit",
    }))
    before = _hashes(match_dir)
    summary = run_endpoint_stitching(
        match_dir, evaluation_dir, output_dir, overwrite=True, benchmark_replicates=0,
        write_stitched_tracks=True, max_review_panels=1,
    )
    assert not summary["synthetic_benchmark_passed"]
    assert summary["stitched_write_blocked_by_benchmark"]
    assert not (output_dir / "tracks_graph_stitched.csv").exists()
    assert (output_dir / "review_panels" / "stitch_contact_sheet.png").is_file()
    assert _hashes(match_dir) == before
    run_log = json.loads((output_dir / "stitch_run_log.json").read_text())
    assert run_log["canonical_matcher_outputs_unchanged"]
    assert run_log["evaluator_algorithm_version"] == "daywise_endpoint_evaluator_test"
    assert run_log["evaluator_repo_git_commit"] == "evaluator-fixture-commit"

    cli_output = tmp_path / "stitch_cli"
    result = subprocess.run([
        sys.executable, "matching/run_endpoint_stitching.py",
        "--match-dir", str(match_dir), "--evaluation-dir", str(evaluation_dir),
        "--output-dir", str(cli_output), "--benchmark-replicates", "1",
        "--write-stitched-tracks", "--overwrite", "--max-review-panels", "0",
    ], cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert not (cli_output / "tracks_graph_stitched.csv").exists()
    assert not (cli_output / "track_uid_stitch_map.csv").exists()
    cli_summary = json.loads((cli_output / "stitch_summary.json").read_text())
    assert not cli_summary["synthetic_benchmark_passed"]
    assert cli_summary["stitched_write_blocked_by_benchmark"]
    first_manifest = pd.read_csv(output_dir / "manual_stitch_review_manifest.csv")
    second_manifest = pd.read_csv(cli_output / "manual_stitch_review_manifest.csv")
    identity = ["stitch_edge_id", "source_track_uid", "source_session_index", "source_label", "target_track_uid", "target_session_index", "target_label"]
    assert first_manifest[identity].astype(str).to_dict("records") == second_manifest[identity].astype(str).to_dict("records")
    assert second_manifest["review_sample_reason"].eq("").all()


def test_overwrite_cleans_all_policy_specific_derived_tracks(tmp_path) -> None:
    output = tmp_path / "stitch"
    output.mkdir()
    for policy in ("graph", "balanced", "high"):
        (output / f"tracks_{policy}_stitched.csv").write_text("stale")
    _prepare_output(output, overwrite=True, policy="balanced")
    assert not list(output.glob("tracks_*_stitched.csv"))


def test_manual_label_restore_requires_composite_identity(tmp_path) -> None:
    proposals = pd.DataFrame([{
        "stitch_edge_id": "stable", "source_track_uid": "source", "source_session_index": 1,
        "source_label": 10, "target_track_uid": "target", "target_session_index": 2,
        "target_label": 20, "session_gap": 1, "candidate_tier": "manual_review",
        "review_reasons": "", "assignment_status": "candidate", "review_sample_reason": "seeded_remainder",
    }])
    previous = proposals.assign(manual_class="accept").copy()
    previous.loc[0, "target_label"] = 99
    restored = _manual_manifest(proposals, previous)
    assert restored.loc[0, "manual_class"] == ""
    assert restored.loc[0, "manual_label_transfer_warning"] == "identity_mismatch_not_restored"


def test_manual_manifest_is_full_and_independent_of_panel_cap() -> None:
    assignments = pd.DataFrame([
        {
            "stitch_edge_id": f"edge_{index}", "source_track_uid": f"source_{index}", "source_session_index": 1,
            "source_label": 10 + index, "target_track_uid": f"target_{index}", "target_session_index": 2,
            "target_label": 20 + index, "session_gap": 1, "candidate_tier": "manual_review",
            "review_reasons": "", "assignment_status": "candidate",
        }
        for index in range(3)
    ])
    full = _manual_manifest(assignments, pd.DataFrame())
    assert len(full) == 3
    assert full["review_sample_reason"].eq("").all()
