"""Compare matcher directories for exact scientific-output equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


CSV_ARTIFACTS = (
    "session_manifest_resolved.csv",
    "roi_features.csv",
    "pairwise_candidates.csv",
    "pairwise_transforms.csv",
    "pairwise_summary.csv",
    "pairwise_matches_high.csv",
    "pairwise_matches_balanced.csv",
    "pairwise_matches_graph.csv",
    "pairwise_summary_graph.csv",
    "graph_match_changes.csv",
    "cycle_consistency_high.csv",
    "cycle_consistency_balanced.csv",
    "cycle_consistency_graph.csv",
    "cycle_edge_checks_high.csv",
    "cycle_edge_checks_balanced.csv",
    "cycle_edge_checks_graph.csv",
    "track_edges_high.csv",
    "track_edges_balanced.csv",
    "track_edges_graph.csv",
    "tracks_high.csv",
    "tracks_balanced.csv",
    "tracks_graph.csv",
    "track_length_summary.csv",
    "track_length_summary_graph.csv",
)
NONSCIENTIFIC_CSV_COLUMNS = {
    "pairwise_summary.csv": ("elapsed_sec",),
    "pairwise_summary_graph.csv": ("elapsed_sec",),
}
NONSCIENTIFIC_JSON_KEYS = {
    "affine_matcher_git_commit",
    "git_commit",
    "git_commit_role",
    "graph_runner_git_commit",
    "graph_output_paths",
    "output_dir",
    "output_paths",
    "pair_workers",
    "qc_artifacts",
    "qc_output_dir",
    "resolved_manifest_path",
    "run_finished_utc",
    "run_started_utc",
    "runtime_profile",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _normalize_json(item)
            for key, item in value.items()
            if key not in NONSCIENTIFIC_JSON_KEYS
        }
    if isinstance(value, list):
        return [_normalize_json(item) for item in value]
    return value


def compare_matcher_outputs(reference_dir: str | Path, candidate_dir: str | Path) -> dict[str, Any]:
    reference_dir = Path(reference_dir).resolve()
    candidate_dir = Path(candidate_dir).resolve()
    artifact_results: list[dict[str, Any]] = []

    for filename in CSV_ARTIFACTS:
        reference_path = reference_dir / filename
        candidate_path = candidate_dir / filename
        result: dict[str, Any] = {"artifact": filename}
        if not reference_path.exists() or not candidate_path.exists():
            result.update({
                "status": "FAIL",
                "reason": "missing_artifact",
                "reference_exists": reference_path.exists(),
                "candidate_exists": candidate_path.exists(),
            })
            artifact_results.append(result)
            continue

        ignored_columns = list(NONSCIENTIFIC_CSV_COLUMNS.get(filename, ()))
        if ignored_columns:
            reference = pd.read_csv(reference_path, low_memory=False)
            candidate = pd.read_csv(candidate_path, low_memory=False)
            try:
                pd.testing.assert_frame_equal(
                    reference.drop(columns=ignored_columns),
                    candidate.drop(columns=ignored_columns),
                    check_dtype=True,
                    check_exact=True,
                    check_like=False,
                )
            except AssertionError as exc:
                result.update({"status": "FAIL", "reason": str(exc)[:2000]})
            else:
                result.update({"status": "PASS", "ignored_columns": ignored_columns})
        else:
            reference_hash = _sha256(reference_path)
            candidate_hash = _sha256(candidate_path)
            result.update({
                "status": "PASS" if reference_hash == candidate_hash else "FAIL",
                "reference_sha256": reference_hash,
                "candidate_sha256": candidate_hash,
            })
        artifact_results.append(result)

    reference_log = reference_dir / "run_log.json"
    candidate_log = candidate_dir / "run_log.json"
    if reference_log.exists() and candidate_log.exists():
        reference_payload = _normalize_json(json.loads(reference_log.read_text(encoding="utf-8")))
        candidate_payload = _normalize_json(json.loads(candidate_log.read_text(encoding="utf-8")))
        log_equal = reference_payload == candidate_payload
        artifact_results.append({
            "artifact": "run_log.json",
            "status": "PASS" if log_equal else "FAIL",
            "ignored_keys": sorted(NONSCIENTIFIC_JSON_KEYS),
        })
    else:
        artifact_results.append({"artifact": "run_log.json", "status": "FAIL", "reason": "missing_artifact"})

    passed = all(result["status"] == "PASS" for result in artifact_results)
    return {
        "reference_dir": str(reference_dir),
        "candidate_dir": str(candidate_dir),
        "scientific_output_equivalence": "PASS" if passed else "FAIL",
        "artifacts": artifact_results,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Matcher output equivalence report",
        "",
        f"- Reference: `{report['reference_dir']}`",
        f"- Candidate: `{report['candidate_dir']}`",
        f"- Scientific output equivalence: **{report['scientific_output_equivalence']}**",
        "",
        "| Artifact | Status |",
        "|---|---|",
    ]
    lines.extend(f"| `{row['artifact']}` | {row['status']} |" for row in report["artifacts"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference_dir")
    parser.add_argument("candidate_dir")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = compare_matcher_outputs(args.reference_dir, args.candidate_dir)
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        _write_markdown(Path(args.markdown_output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["scientific_output_equivalence"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
