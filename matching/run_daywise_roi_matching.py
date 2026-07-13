"""Command-line runner for daywise affine-overlap ROI matching."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _import_dir in (_REPO_ROOT / "core", _REPO_ROOT / "matching"):
    _import_dir_str = str(_import_dir)
    if _import_dir_str not in sys.path:
        sys.path.append(_import_dir_str)

from session_manifest import SessionRecord, load_session_manifest, validate_manifest_for_matching
from affine_overlap_matcher import (
    MATCHER_ALGORITHM_VERSION,
    AffineOverlapParams,
    PairMatchResult,
    VoxelSpacing,
    extract_roi_features,
    match_pair,
)
from roi_track_graph import (
    build_cycle_consistency_tables,
    build_track_length_summary_table,
    build_tracks_from_pair_tables,
    summarize_track_cycle_metadata,
)


def format_duration_seconds(duration_seconds: float) -> str:
    """Format elapsed wall-clock time as ``HH:MM:SS``."""

    total_seconds = max(0, int(duration_seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _sha256_file(path: Path) -> str:
    """Compute the SHA-256 hash of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_package_version(package_name: str) -> str | None:
    """Return a package version or ``None`` when unavailable."""

    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    """Return the current Git commit hash when available."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _load_mask_stack(mask_path: Path) -> np.ndarray:
    """Load a mask stack, preferring memory mapping when supported."""

    try:
        return np.asarray(tifffile.memmap(mask_path))
    except Exception:
        return tifffile.imread(mask_path)


def _resolved_manifest_dataframe(records: list[SessionRecord]) -> pd.DataFrame:
    """Convert manifest records into a resolved CSV-ready table."""

    rows = []
    for record in records:
        rows.append(
            {
                "session_index": int(record.session_index),
                "session_id": str(record.session_id),
                "acquisition_date": record.acquisition_date.isoformat(),
                "mask_path": str(record.mask_path),
                "red_image_path": str(record.red_image_path) if record.red_image_path is not None else "",
                "green_image_path": str(record.green_image_path) if record.green_image_path is not None else "",
                "required": bool(record.required),
            }
        )
    return pd.DataFrame(rows)


def _package_versions() -> dict[str, str | None]:
    """Collect package versions for the run log."""

    return {
        "numpy": _safe_package_version("numpy"),
        "pandas": _safe_package_version("pandas"),
        "scipy": _safe_package_version("scipy"),
        "scikit-image": _safe_package_version("scikit-image"),
        "tifffile": _safe_package_version("tifffile"),
    }


def _load_existing_run_log(output_dir: Path) -> dict[str, object] | None:
    """Read an existing run log when present."""

    run_log_path = output_dir / "run_log.json"
    if not run_log_path.exists():
        return None
    return json.loads(run_log_path.read_text(encoding="utf-8"))


def _fingerprint_run(
    *,
    manifest_hash: str,
    records: list[SessionRecord],
    params: AffineOverlapParams,
    spacing: VoxelSpacing,
    max_pair_gap: int,
) -> dict[str, object]:
    """Build a compact fingerprint for resume comparisons."""

    return {
        "algorithm_version": MATCHER_ALGORITHM_VERSION,
        "manifest_hash": manifest_hash,
        "session_ids": [record.session_id for record in records],
        "mask_hashes": {record.session_id: _sha256_file(record.mask_path) for record in records},
        "spacing": asdict(spacing),
        "params": asdict(params),
        "max_pair_gap": int(max_pair_gap),
    }


def _normalize_pair_table(
    table: pd.DataFrame,
    *,
    day_a: str,
    day_b: str,
    session_index_a: int,
    session_index_b: int,
    pair_gap: int,
    policy: str,
    transform: object,
) -> pd.DataFrame:
    """Attach run provenance to one accepted pair table."""

    if table is None or table.empty:
        columns = list(table.columns) if table is not None else []
        return pd.DataFrame(columns=columns)
    output = table.copy()
    output.insert(0, "match_policy", policy)
    output.insert(1, "day_a", day_a)
    output.insert(2, "day_b", day_b)
    output.insert(3, "pair_gap", int(pair_gap))
    output.insert(4, "session_index_a", int(session_index_a))
    output.insert(5, "session_index_b", int(session_index_b))
    output["transform_method"] = str(getattr(transform, "method", ""))
    output["transform_fallback_reason"] = getattr(transform, "fallback_reason", None)
    output["transform_residual_median_um"] = getattr(transform, "residual_median_um", None)
    output["transform_residual_p95_um"] = getattr(transform, "residual_p95_um", None)
    return output.reset_index(drop=True)


def _export_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON file with stable formatting."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _existing_output_matches(
    output_dir: Path,
    expected_fingerprint: dict[str, object],
) -> bool:
    """Check whether an existing output directory matches the requested run."""

    existing = _load_existing_run_log(output_dir)
    if existing is None:
        return False
    current = existing.get("run_fingerprint")
    return current == expected_fingerprint


def run_daywise_roi_matching(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    spacing: VoxelSpacing | None = None,
    params: AffineOverlapParams | None = None,
    max_pair_gap: int = 2,
    save_candidates: bool = False,
    overwrite: bool = False,
    resume: bool = False,
) -> Path:
    """Run daywise ROI matching and export all canonical outputs."""

    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if max_pair_gap > 2:
        raise ValueError("--max-pair-gap values above 2 are not supported in affine_overlap_v1.")
    spacing = spacing or VoxelSpacing()
    params = params or AffineOverlapParams()

    records = load_session_manifest(manifest_path)
    validate_manifest_for_matching(records)
    manifest_hash = _sha256_file(manifest_path)
    fingerprint = _fingerprint_run(
        manifest_hash=manifest_hash,
        records=records,
        params=params,
        spacing=spacing,
        max_pair_gap=max_pair_gap,
    )

    if output_dir.exists():
        if resume and _existing_output_matches(output_dir, fingerprint):
            return output_dir
        if resume and not _existing_output_matches(output_dir, fingerprint):
            raise FileExistsError(
                f"Existing output directory does not match the requested resume fingerprint: {output_dir}."
            )
        if not overwrite and not resume:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. Use --overwrite or --resume."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    qc_dir = output_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    run_start_seconds = time.perf_counter()
    print(f"[{format_duration_seconds(0.0)}] Daywise ROI matching start | manifest={manifest_path}", flush=True)
    print(f"[{format_duration_seconds(0.0)}] Output directory: {output_dir}", flush=True)

    resolved_manifest = _resolved_manifest_dataframe(records)
    resolved_manifest.to_csv(output_dir / "session_manifest_resolved.csv", index=False)

    feature_rows: list[pd.DataFrame] = []
    features_by_session: dict[str, pd.DataFrame] = {}
    input_hashes: list[dict[str, object]] = []
    for record in records:
        mask_hash = _sha256_file(record.mask_path)
        mask_stack = _load_mask_stack(record.mask_path)
        feature_table = extract_roi_features(mask_stack, session_id=record.session_id, spacing=spacing)
        feature_table = feature_table.copy()
        feature_table.insert(0, "session_index", int(record.session_index))
        feature_table["session_id"] = record.session_id
        feature_table.insert(2, "mask_path", str(record.mask_path))
        feature_table.insert(3, "mask_sha256", mask_hash)
        feature_table["required"] = bool(record.required)
        feature_rows.append(feature_table.reset_index(drop=True))
        features_by_session[record.session_id] = feature_table.set_index("label", drop=False)
        input_hashes.append(
            {
                "session_id": record.session_id,
                "session_index": int(record.session_index),
                "mask_path": str(record.mask_path),
                "mask_sha256": mask_hash,
            }
        )

    roi_features = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
    roi_features.to_csv(output_dir / "roi_features.csv", index=False)

    pair_results: dict[tuple[str, str], PairMatchResult] = {}
    pair_summaries: list[dict[str, object]] = []
    pair_transforms: list[dict[str, object]] = []
    candidate_rows: list[pd.DataFrame] = []
    session_by_id = {record.session_id: record for record in records}
    ordered_sessions = [record.session_id for record in records]

    for index_a, session_a in enumerate(ordered_sessions):
        for index_b in range(index_a + 1, min(len(ordered_sessions), index_a + max_pair_gap + 1)):
            session_b = ordered_sessions[index_b]
            pair_gap = index_b - index_a
            if pair_gap < 1 or pair_gap > max_pair_gap:
                continue
            pair_start_seconds = time.perf_counter()
            mask_a = _load_mask_stack(session_by_id[session_a].mask_path)
            mask_b = _load_mask_stack(session_by_id[session_b].mask_path)
            result = match_pair(
                session_a=session_a,
                session_b=session_b,
                mask_a=mask_a,
                mask_b=mask_b,
                params=params,
                spacing=spacing,
                features_a=features_by_session[session_a],
                features_b=features_by_session[session_b],
                pair_gap=pair_gap,
            )
            del mask_a
            del mask_b
            pair_results[(session_a, session_b)] = result
            pair_elapsed_seconds = time.perf_counter() - pair_start_seconds
            pair_summaries.append(result.summary | {"elapsed_sec": float(pair_elapsed_seconds)})
            transform = result.transform
            pair_transforms.append(
                {
                    "day_a": session_a,
                    "day_b": session_b,
                    "pair_gap": int(pair_gap),
                    "z_intercept": float(transform.z_intercept),
                    "z_scale": float(transform.z_scale),
                    "y_intercept": float(transform.y_intercept),
                    "y_from_y": float(transform.y_from_y),
                    "y_from_x": float(transform.y_from_x),
                    "x_intercept": float(transform.x_intercept),
                    "x_from_y": float(transform.x_from_y),
                    "x_from_x": float(transform.x_from_x),
                    "method": str(transform.method),
                    "fallback_reason": transform.fallback_reason,
                    "n_seed": int(transform.n_seed),
                    "n_inlier": int(transform.n_inlier),
                    "residual_median_um": transform.residual_median_um,
                    "residual_p95_um": transform.residual_p95_um,
                }
            )
            if save_candidates:
                candidate_df = result.candidates.copy()
                candidate_df.insert(0, "day_a", session_a)
                candidate_df.insert(1, "day_b", session_b)
                candidate_df.insert(2, "pair_gap", int(pair_gap))
                candidate_rows.append(candidate_df)

    pairwise_summary = pd.DataFrame(pair_summaries)
    pairwise_summary = pairwise_summary.sort_values(["day_a", "day_b"]).reset_index(drop=True)
    pairwise_summary.to_csv(output_dir / "pairwise_summary.csv", index=False)

    pairwise_transforms = pd.DataFrame(pair_transforms).sort_values(["day_a", "day_b"]).reset_index(drop=True)
    pairwise_transforms.to_csv(output_dir / "pairwise_transforms.csv", index=False)

    combined_high_rows: list[pd.DataFrame] = []
    combined_balanced_rows: list[pd.DataFrame] = []
    high_pair_tables: dict[tuple[str, str], pd.DataFrame] = {}
    balanced_pair_tables: dict[tuple[str, str], pd.DataFrame] = {}
    for (session_a, session_b), result in pair_results.items():
        index_a = session_by_id[session_a].session_index
        index_b = session_by_id[session_b].session_index
        pair_gap = index_b - index_a
        high_df = _normalize_pair_table(
            result.high_matches,
            day_a=session_a,
            day_b=session_b,
            session_index_a=index_a,
            session_index_b=index_b,
            pair_gap=pair_gap,
            policy="high",
            transform=result.transform,
        )
        balanced_df = _normalize_pair_table(
            result.balanced_matches,
            day_a=session_a,
            day_b=session_b,
            session_index_a=index_a,
            session_index_b=index_b,
            pair_gap=pair_gap,
            policy="balanced",
            transform=result.transform,
        )
        high_pair_tables[(session_a, session_b)] = high_df
        balanced_pair_tables[(session_a, session_b)] = balanced_df
        if not high_df.empty:
            combined_high_rows.append(high_df)
        if not balanced_df.empty:
            combined_balanced_rows.append(balanced_df)

    pairwise_matches_high = pd.concat(combined_high_rows, ignore_index=True) if combined_high_rows else pd.DataFrame()
    pairwise_matches_balanced = pd.concat(combined_balanced_rows, ignore_index=True) if combined_balanced_rows else pd.DataFrame()
    pairwise_matches_high.to_csv(output_dir / "pairwise_matches_high.csv", index=False)
    pairwise_matches_balanced.to_csv(output_dir / "pairwise_matches_balanced.csv", index=False)

    if save_candidates:
        candidate_table = pd.concat(candidate_rows, ignore_index=True) if candidate_rows else pd.DataFrame()
        candidate_table.to_csv(output_dir / "pairwise_candidates.csv", index=False)

    tracks_high, edges_high = build_tracks_from_pair_tables(
        day_names=ordered_sessions,
        features_by_session=features_by_session,
        pair_tables=high_pair_tables,
        match_policy="high",
    )
    tracks_balanced, edges_balanced = build_tracks_from_pair_tables(
        day_names=ordered_sessions,
        features_by_session=features_by_session,
        pair_tables=balanced_pair_tables,
        match_policy="balanced",
    )

    cycle_summary_high, cycle_edge_checks_high = build_cycle_consistency_tables(
        day_names=ordered_sessions,
        pair_tables=high_pair_tables,
        tracks_table=tracks_high,
        match_policy="high",
    )
    cycle_summary_balanced, cycle_edge_checks_balanced = build_cycle_consistency_tables(
        day_names=ordered_sessions,
        pair_tables=balanced_pair_tables,
        tracks_table=tracks_balanced,
        match_policy="balanced",
    )
    tracks_high = summarize_track_cycle_metadata(tracks_high, cycle_edge_checks_high)
    tracks_balanced = summarize_track_cycle_metadata(tracks_balanced, cycle_edge_checks_balanced)

    tracks_high.to_csv(output_dir / "tracks_high.csv", index=False)
    tracks_balanced.to_csv(output_dir / "tracks_balanced.csv", index=False)
    cycle_summary_high.to_csv(output_dir / "cycle_consistency_high.csv", index=False)
    cycle_summary_balanced.to_csv(output_dir / "cycle_consistency_balanced.csv", index=False)
    cycle_edge_checks_high.to_csv(output_dir / "cycle_edge_checks_high.csv", index=False)
    cycle_edge_checks_balanced.to_csv(output_dir / "cycle_edge_checks_balanced.csv", index=False)
    edges_high.to_csv(output_dir / "track_edges_high.csv", index=False)
    edges_balanced.to_csv(output_dir / "track_edges_balanced.csv", index=False)

    track_length_summary = build_track_length_summary_table(
        pd.concat([tracks_high, tracks_balanced], ignore_index=True) if not tracks_high.empty or not tracks_balanced.empty else pd.DataFrame()
    )
    track_length_summary.to_csv(output_dir / "track_length_summary.csv", index=False)

    input_hash_records = []
    for record in records:
        input_hash_records.append(
            {
                "session_index": int(record.session_index),
                "session_id": record.session_id,
                "mask_path": str(record.mask_path),
                "mask_sha256": _sha256_file(record.mask_path),
                "red_image_path": str(record.red_image_path) if record.red_image_path is not None else "",
                "green_image_path": str(record.green_image_path) if record.green_image_path is not None else "",
                "red_sha256": _sha256_file(record.red_image_path) if record.red_image_path is not None else None,
                "green_sha256": _sha256_file(record.green_image_path) if record.green_image_path is not None else None,
            }
        )

    run_log_payload = {
        "algorithm_version": MATCHER_ALGORITHM_VERSION,
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "resolved_manifest_path": str(output_dir / "session_manifest_resolved.csv"),
        "output_dir": str(output_dir),
        "run_fingerprint": fingerprint,
        "spacing": asdict(spacing),
        "params": asdict(params),
        "max_pair_gap": int(max_pair_gap),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "git_commit": _git_commit(),
        "input_hashes": input_hash_records,
        "row_counts": {
            "roi_features": int(len(roi_features)),
            "pairwise_summary": int(len(pairwise_summary)),
            "pairwise_transforms": int(len(pairwise_transforms)),
            "pairwise_matches_high": int(len(pairwise_matches_high)),
            "pairwise_matches_balanced": int(len(pairwise_matches_balanced)),
            "cycle_consistency_high": int(len(cycle_summary_high)),
            "cycle_consistency_balanced": int(len(cycle_summary_balanced)),
            "cycle_edge_checks_high": int(len(cycle_edge_checks_high)),
            "cycle_edge_checks_balanced": int(len(cycle_edge_checks_balanced)),
            "track_edges_high": int(len(edges_high)),
            "track_edges_balanced": int(len(edges_balanced)),
            "tracks_high": int(len(tracks_high)),
            "tracks_balanced": int(len(tracks_balanced)),
            "track_length_summary": int(len(track_length_summary)),
        },
        "output_paths": {
            "session_manifest_resolved": str(output_dir / "session_manifest_resolved.csv"),
            "roi_features": str(output_dir / "roi_features.csv"),
            "pairwise_summary": str(output_dir / "pairwise_summary.csv"),
            "pairwise_transforms": str(output_dir / "pairwise_transforms.csv"),
            "pairwise_matches_high": str(output_dir / "pairwise_matches_high.csv"),
            "pairwise_matches_balanced": str(output_dir / "pairwise_matches_balanced.csv"),
            "cycle_consistency_high": str(output_dir / "cycle_consistency_high.csv"),
            "cycle_consistency_balanced": str(output_dir / "cycle_consistency_balanced.csv"),
            "cycle_edge_checks_high": str(output_dir / "cycle_edge_checks_high.csv"),
            "cycle_edge_checks_balanced": str(output_dir / "cycle_edge_checks_balanced.csv"),
            "track_edges_high": str(output_dir / "track_edges_high.csv"),
            "track_edges_balanced": str(output_dir / "track_edges_balanced.csv"),
            "tracks_high": str(output_dir / "tracks_high.csv"),
            "tracks_balanced": str(output_dir / "tracks_balanced.csv"),
            "track_length_summary": str(output_dir / "track_length_summary.csv"),
            "qc_dir": str(qc_dir),
        },
        "warnings": [],
    }
    _export_json(output_dir / "run_log.json", run_log_payload)
    summary_lines = [
        "# Daywise ROI Matching",
        "",
        f"- Algorithm: `{MATCHER_ALGORITHM_VERSION}`",
        f"- Manifest: `{manifest_path}`",
        f"- Output directory: `{output_dir}`",
        f"- Sessions: `{len(records)}`",
        f"- Pair gap limit: `{max_pair_gap}`",
        "",
        "Output counts:",
        f"- High pairwise matches: `{len(pairwise_matches_high)}`",
        f"- Balanced pairwise matches: `{len(pairwise_matches_balanced)}`",
        f"- High tracks: `{len(tracks_high)}`",
        f"- Balanced tracks: `{len(tracks_balanced)}`",
    ]
    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    total_duration_seconds = time.perf_counter() - run_start_seconds
    print(f"[{format_duration_seconds(total_duration_seconds)}] Daywise ROI matching completed", flush=True)
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Resolved session manifest CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for matcher outputs.")
    parser.add_argument("--xy-um-per-px", type=float, default=710.0 / 1024.0, help="XY pixel spacing in micrometers.")
    parser.add_argument("--z-um-per-plane", type=float, default=5.0, help="Z spacing in micrometers.")
    parser.add_argument("--max-pair-gap", type=int, default=2, help="Maximum allowed session gap for pairwise matching.")
    parser.add_argument("--save-candidates", action="store_true", help="Write the full candidate table.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory.")
    parser.add_argument("--resume", action="store_true", help="Reuse a prior exact-matching output directory when possible.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    """CLI entry point."""

    args = parse_args(argv)
    spacing = VoxelSpacing(z_um=args.z_um_per_plane, y_um=args.xy_um_per_px, x_um=args.xy_um_per_px)
    output_dir = run_daywise_roi_matching(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        spacing=spacing,
        params=AffineOverlapParams(),
        max_pair_gap=args.max_pair_gap,
        save_candidates=bool(args.save_candidates),
        overwrite=bool(args.overwrite),
        resume=bool(args.resume),
    )
    print(f"output_dir={output_dir}", flush=True)
    return output_dir


if __name__ == "__main__":
    main()

