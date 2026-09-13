from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from endpoint_evaluator import detect_endpoint_events, search_endpoint_candidates, classify_endpoint_events
from run_endpoint_stitching import run_endpoint_stitching
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
    assert json.loads((output_dir / "stitch_run_log.json").read_text())["canonical_matcher_outputs_unchanged"]

    cli_output = tmp_path / "stitch_cli"
    result = subprocess.run([
        sys.executable, "matching/run_endpoint_stitching.py",
        "--match-dir", str(match_dir), "--evaluation-dir", str(evaluation_dir),
        "--output-dir", str(cli_output), "--benchmark-replicates", "1",
        "--write-stitched-tracks", "--overwrite", "--max-review-panels", "1",
    ], cwd=Path(__file__).resolve().parent.parent, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert (cli_output / "tracks_graph_stitched.csv").is_file()
    assert (cli_output / "track_uid_stitch_map.csv").is_file()
    cli_summary = json.loads((cli_output / "stitch_summary.json").read_text())
    assert cli_summary["synthetic_benchmark_passed"]
    assert cli_summary["stitched_tracks_written"]
