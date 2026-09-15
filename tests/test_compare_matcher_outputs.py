from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tools.compare_matcher_outputs import CSV_ARTIFACTS, compare_matcher_outputs


def _write_output(directory: Path, *, workers: int, scientific_value: int = 1) -> None:
    directory.mkdir()
    for filename in CSV_ARTIFACTS:
        table = pd.DataFrame({"value": [scientific_value]})
        if filename in {"pairwise_summary.csv", "pairwise_summary_graph.csv"}:
            table["elapsed_sec"] = float(workers)
        table.to_csv(directory / filename, index=False)
    (directory / "run_log.json").write_text(json.dumps({
        "affine_matcher_git_commit": f"affine-{workers}",
        "graph_runner_git_commit": f"graph-{workers}",
        "pair_workers": workers,
        "run_started_utc": f"time-{workers}",
        "runtime_profile": {"total": workers},
        "row_counts": {"tracks": scientific_value},
    }))


def test_compare_matcher_outputs_ignores_only_nonscientific_metadata(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_output(reference, workers=1)
    _write_output(candidate, workers=2)
    report = compare_matcher_outputs(reference, candidate)
    assert report["scientific_output_equivalence"] == "PASS"


def test_compare_matcher_outputs_rejects_scientific_difference(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    _write_output(reference, workers=1)
    _write_output(candidate, workers=2)
    pd.DataFrame({"value": [2]}).to_csv(candidate / "tracks_graph.csv", index=False)
    report = compare_matcher_outputs(reference, candidate)
    assert report["scientific_output_equivalence"] == "FAIL"
