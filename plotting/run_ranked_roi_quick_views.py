"""Render top/bottom final-session directional ROIs in native raw space."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
for _directory in (_ROOT / "core", _ROOT / "plotting"):
    if str(_directory) not in sys.path:
        sys.path.append(str(_directory))

from roi_log_ratio_analysis import select_top_changing_rois, summarize_roi_metrics
from run_matched_roi_quick_view import _tracks_from_raw_table, plot_matched_roi_raw_slices


METRIC_NAMES = (
    "weekly_matched_roi_log_ratio_metrics_complete.csv",
    "matched_roi_log_ratio_metrics_complete.csv",
)
RAW_NAMES = (
    "matched_roi_intensity_results_raw.csv",
    "weekly_matched_roi_intensity_results_raw.csv",
)


def _first_existing(root: Path, names: tuple[str, ...]) -> Path:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of {names} was found in {root}")


def _resolve_run_inputs(run_dir: Path) -> tuple[Path, Path, Path]:
    """Resolve current master-run inputs, with support for legacy flat runs."""
    extraction = run_dir / "extraction"
    matching = run_dir / "matching"
    if extraction.is_dir():
        return (
            _first_existing(extraction, METRIC_NAMES),
            _first_existing(extraction, RAW_NAMES),
            _first_existing(extraction, ("session_manifest_resolved.csv",)),
        )
    return (
        _first_existing(run_dir, METRIC_NAMES),
        _first_existing(run_dir, RAW_NAMES),
        _first_existing(run_dir, ("session_manifest_resolved.csv",)),
    )


def _filter_policy(table: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "all" or "match_policy" not in table.columns:
        return table
    filtered = table.loc[table["match_policy"].astype(str).eq(policy)].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for match policy {policy!r}")
    return filtered


def build_ranked_roi_views(
    run_dir: str | Path,
    *,
    policy: str = "graph",
    top_n: int = 30,
    directions: tuple[str, ...] = ("increasing", "decreasing"),
    z_radius: int = 3,
    output_dir: str | Path | None = None,
) -> Path:
    """Rank current metrics and render each selected track with the existing viewer."""
    root = Path(run_dir).resolve()
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    metrics_path, raw_path, manifest_path = _resolve_run_inputs(root)
    metrics = _filter_policy(pd.read_csv(metrics_path), policy)
    raw = _filter_policy(pd.read_csv(raw_path), policy)
    manifest = pd.read_csv(manifest_path)
    tracks = _tracks_from_raw_table(raw, policy)
    output_root = Path(output_dir).resolve() if output_dir else root / "plots" / policy / "single_roi_raw_validation"
    output_root.mkdir(parents=True, exist_ok=True)
    summary = summarize_roi_metrics(metrics)
    index_rows = []
    for direction in directions:
        if direction not in {"increasing", "decreasing"}:
            raise ValueError(f"Unsupported direction: {direction!r}")
        ranked = select_top_changing_rois(summary, max_rois=top_n, direction=direction, ranking_mode="final")
        direction_dir = output_root / f"top{top_n}_{direction}"
        direction_dir.mkdir(parents=True, exist_ok=True)
        for _, selected in ranked.iterrows():
            roi_id = selected.get("roi_id", selected.get("cluster_id", ""))
            cluster_id = selected.get("cluster_id", roi_id)
            rank = int(selected["selection_rank"])
            stem = f"{'increase' if direction == 'increasing' else 'decrease'}_rank{rank:02d}_roi_{roi_id}"
            png_path = direction_dir / f"{stem}.png"
            plot_matched_roi_raw_slices(
                cluster_id=str(cluster_id), tracks_table=tracks, session_table=manifest,
                output_path=png_path, z_radius=z_radius,
            )
            metadata_path = png_path.with_name(png_path.stem + "_metadata.csv")
            index_rows.append({
                "selection_direction": direction, "selection_rank": rank,
                "roi_id": roi_id, "cluster_id": cluster_id,
                "track_uid": selected.get("track_uid", ""),
                "selection_mode": selected["selection_mode"],
                "selection_metric_column": selected["selection_metric_column"],
                "selection_value": selected[selected["selection_metric_column"]],
                "png_path": str(png_path), "metadata_csv_path": str(metadata_path),
            })
    pd.DataFrame(index_rows).to_csv(output_root / "ranked_roi_batch_index.csv", index=False)
    return output_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--policy", choices=["high", "balanced", "graph"], default="graph")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--directions", default="increasing,decreasing")
    parser.add_argument("--z-radius", type=int, default=3)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    output = build_ranked_roi_views(
        args.run_dir, policy=args.policy, top_n=args.top_n,
        directions=tuple(part.strip() for part in args.directions.split(",") if part.strip()),
        z_radius=args.z_radius, output_dir=args.output_dir,
    )
    print(f"ranked_roi_output_dir={output}")


if __name__ == "__main__":
    main()
