"""Generate QC plots and review samples for daywise affine-overlap matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _import_dir in (_REPO_ROOT / "core", _REPO_ROOT / "matching"):
    _import_dir_str = str(_import_dir)
    if _import_dir_str not in sys.path:
        sys.path.append(_import_dir_str)


@dataclass(frozen=True)
class DaywiseQCPlotConfig:
    match_dir: str
    output_dir: str | None = None
    sample_limit: int = 6
    review_seed: int = 7
    include_skip_pairs: bool = True


def _load_csv(match_dir: Path, name: str) -> pd.DataFrame:
    path = match_dir / name
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _load_json(match_dir: Path, name: str) -> dict[str, object]:
    path = match_dir / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_figure(path: Path, fig: plt.Figure) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _figure_for_table(title: str, ncols: int = 2, nrows: int = 2) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows))
    fig.suptitle(title, fontsize=14)
    return fig, np.asarray(axes).reshape(nrows, ncols)


def _plot_pairwise_distributions(output_dir: Path, pairwise_summary: pd.DataFrame, pairwise_matches: pd.DataFrame) -> list[Path]:
    saved: list[Path] = []
    if pairwise_summary.empty and pairwise_matches.empty:
        return saved

    summary = pairwise_summary.copy()
    if not summary.empty and "pair_gap" in summary.columns:
        summary["pair_kind"] = np.where(summary["pair_gap"].astype(int) == 1, "adjacent", "skip")
    else:
        summary["pair_kind"] = "adjacent"

    summary_policies = summary["match_policy"].astype(str) if (not summary.empty and "match_policy" in summary.columns) else pd.Series(dtype=str, index=summary.index)
    match_policies = pairwise_matches["match_policy"].astype(str) if (not pairwise_matches.empty and "match_policy" in pairwise_matches.columns) else pd.Series(dtype=str, index=pairwise_matches.index)
    for policy in sorted(set(summary_policies.dropna().unique().tolist()) | set(match_policies.dropna().unique().tolist())):
        policy_summary = summary.loc[summary_policies == policy].copy() if not summary.empty else pd.DataFrame()
        policy_matches = pairwise_matches.loc[match_policies == policy].copy() if not pairwise_matches.empty else pd.DataFrame()
        if policy_summary.empty and policy_matches.empty:
            continue
        fig, axes = _figure_for_table(f"Daywise {policy} pair distributions")
        if not policy_summary.empty:
            if "n_high" in policy_summary.columns:
                axes[0, 0].hist(policy_summary["n_high"].fillna(0).astype(float), bins=20, color="#2d6cdf", alpha=0.8)
                axes[0, 0].set_title("Accepted high edges per pair")
            if "n_balanced" in policy_summary.columns:
                axes[0, 1].hist(policy_summary["n_balanced"].fillna(0).astype(float), bins=20, color="#31a354", alpha=0.8)
                axes[0, 1].set_title("Accepted balanced edges per pair")
            if "transform_residual_median_um" in policy_summary.columns:
                axes[1, 0].hist(policy_summary["transform_residual_median_um"].dropna().astype(float), bins=20, color="#8c6bb1", alpha=0.8)
                axes[1, 0].set_title("Transform residual median (um)")
            if "elapsed_sec" in policy_summary.columns:
                axes[1, 1].hist(policy_summary["elapsed_sec"].dropna().astype(float), bins=20, color="#f28e2b", alpha=0.8)
                axes[1, 1].set_title("Pair elapsed seconds")
        else:
            axes[0, 0].axis("off")
            axes[0, 1].axis("off")
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")
        for ax in axes.flat:
            ax.grid(alpha=0.2)
        saved.append(_save_figure(output_dir / f"pairwise_summary_{policy}.png", fig))

        if not policy_matches.empty:
            fig, axes = _figure_for_table(f"Daywise {policy} candidate distributions")
            columns = [
                ("score", "Score", "#2d6cdf"),
                ("dice", "Dice", "#31a354"),
                ("distance_um", "Distance (um)", "#f28e2b"),
                ("ambiguity", "Ambiguity", "#8c6bb1"),
            ]
            for ax, (column, label, color) in zip(axes.flat, columns, strict=False):
                if column in policy_matches.columns:
                    ax.hist(pd.to_numeric(policy_matches[column], errors="coerce").dropna(), bins=30, color=color, alpha=0.85)
                ax.set_title(label)
                ax.grid(alpha=0.2)
            saved.append(_save_figure(output_dir / f"candidate_distributions_{policy}.png", fig))

    return saved


def _plot_track_summaries(output_dir: Path, tracks: pd.DataFrame, policy: str) -> list[Path]:
    saved: list[Path] = []
    if tracks.empty:
        return saved

    panels: list[tuple[str, pd.Series, np.ndarray | int, str]] = [
        (
            "Track length",
            tracks["n_days_present"].dropna().astype(float),
            np.arange(0.5, tracks["n_days_present"].max() + 1.5) if not tracks["n_days_present"].dropna().empty else 20,
            "#2d6cdf",
        ),
    ]
    if "missing_internal_days" in tracks.columns:
        panels.append(
            (
                "Missing internal days",
                tracks["missing_internal_days"].dropna().astype(float),
                20,
                "#31a354",
            )
        )
    if "max_volume_fold_change" in tracks.columns:
        panels.append(
            (
                "Track volume fold-change",
                tracks["max_volume_fold_change"].dropna().astype(float),
                20,
                "#f28e2b",
            )
        )
    if "n_edge_sessions" in tracks.columns:
        panels.append(
            (
                "Edge-touching sessions",
                tracks["n_edge_sessions"].dropna().astype(float),
                20,
                "#8c6bb1",
            )
        )

    n_panels = len(panels)
    ncols = 2 if n_panels > 1 else 1
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows))
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax, (title, values, bins, color) in zip(axes_arr.flat, panels, strict=False):
        if len(values):
            ax.hist(values, bins=bins, color=color, alpha=0.85)
        ax.set_title(title)
        ax.grid(alpha=0.2)
    for ax in axes_arr.flat[len(panels):]:
        ax.axis("off")
    saved.append(_save_figure(output_dir / f"track_summary_{policy}.png", fig))

    if "has_cycle_conflict" in tracks.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = tracks["has_cycle_conflict"].fillna(False).astype(bool).value_counts().sort_index()
        ax.bar(["no", "yes"], [int(counts.get(False, 0)), int(counts.get(True, 0))], color=["#31a354", "#d62728"])
        ax.set_title(f"{policy} cycle conflicts")
        ax.grid(axis="y", alpha=0.2)
        saved.append(_save_figure(output_dir / f"cycle_conflicts_{policy}.png", fig))

    return saved


def _build_review_sample(
    tracks_high: pd.DataFrame,
    tracks_balanced: pd.DataFrame,
    sample_limit: int,
    review_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(review_seed)
    rows: list[pd.DataFrame] = []

    def _take(table: pd.DataFrame, category: str, sort_columns: list[str], ascending: list[bool]) -> None:
        if table.empty:
            return
        subset = table.sort_values(sort_columns, ascending=ascending).head(sample_limit).copy()
        subset.insert(0, "review_category", category)
        rows.append(subset)

    _take(tracks_high.sample(min(sample_limit, len(tracks_high)), random_state=review_seed).copy() if not tracks_high.empty else pd.DataFrame(), "random_high", ["cluster_id"], [True])
    if not tracks_high.empty:
        if "min_score" in tracks_high.columns:
            _take(tracks_high, "lowest_score_high", ["min_score", "cluster_id"], [True, True])
        if "max_distance_um" in tracks_high.columns:
            _take(tracks_high, "largest_distance_high", ["max_distance_um", "cluster_id"], [False, True])
        if "min_dice" in tracks_high.columns:
            _take(tracks_high, "lowest_dice_high", ["min_dice", "cluster_id"], [True, True])
        if "has_cycle_conflict" in tracks_high.columns:
            _take(tracks_high.loc[tracks_high["has_cycle_conflict"].fillna(False).astype(bool)], "cycle_conflict", ["cluster_id"], [True])
        if "used_gap_bridge" in tracks_high.columns:
            _take(tracks_high.loc[tracks_high["used_gap_bridge"].fillna(False).astype(bool)], "gap_bridge", ["cluster_id"], [True])
        if "contains_transform_fallback_edge" in tracks_high.columns:
            _take(tracks_high.loc[tracks_high["contains_transform_fallback_edge"].fillna(False).astype(bool)], "transform_fallback", ["cluster_id"], [True])
        edge_mask = tracks_high.get("n_edge_sessions", pd.Series(dtype=float)).fillna(0).astype(float) > 0
        truncated_mask = tracks_high.get("missing_internal_days", pd.Series(dtype=float)).fillna(0).astype(float) > 0
        _take(tracks_high.loc[edge_mask | truncated_mask], "edge_or_truncated", ["n_days_present", "cluster_id"], [True, True])

    if not tracks_balanced.empty and not tracks_high.empty and "component_signature" in tracks_high.columns and "component_signature" in tracks_balanced.columns:
        high_signatures = set(tracks_high["component_signature"].astype(str))
        balanced_only = tracks_balanced.loc[~tracks_balanced["component_signature"].astype(str).isin(high_signatures)].copy()
        _take(balanced_only, "balanced_only", ["cluster_id"], [True])

    if not rows:
        return pd.DataFrame()
    sample = pd.concat(rows, ignore_index=True)
    sample = sample.drop_duplicates(subset=["review_category", "cluster_id", "track_uid"], keep="first")
    return sample.reset_index(drop=True)


def generate_daywise_qc_plots(config: DaywiseQCPlotConfig) -> Path:
    match_dir = Path(config.match_dir).resolve()
    if not match_dir.exists():
        raise FileNotFoundError(f"match_dir was not found: {match_dir}")
    output_dir = Path(config.output_dir).resolve() if config.output_dir else match_dir / "qc_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_summary = _load_csv(match_dir, "pairwise_summary.csv")
    pairwise_matches_high = _load_csv(match_dir, "pairwise_matches_high.csv")
    pairwise_matches_balanced = _load_csv(match_dir, "pairwise_matches_balanced.csv")
    tracks_high = _load_csv(match_dir, "tracks_high.csv")
    tracks_balanced = _load_csv(match_dir, "tracks_balanced.csv")
    cycle_high = _load_csv(match_dir, "cycle_consistency_high.csv")
    cycle_balanced = _load_csv(match_dir, "cycle_consistency_balanced.csv")
    track_length_summary = _load_csv(match_dir, "track_length_summary.csv")

    saved_paths: list[Path] = []
    saved_paths.extend(_plot_pairwise_distributions(output_dir, pairwise_summary, pd.concat([pairwise_matches_high, pairwise_matches_balanced], ignore_index=True) if not pairwise_matches_high.empty or not pairwise_matches_balanced.empty else pd.DataFrame()))
    saved_paths.extend(_plot_track_summaries(output_dir, tracks_high, "high"))
    saved_paths.extend(_plot_track_summaries(output_dir, tracks_balanced, "balanced"))

    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False
    for policy, table, color in (("high", cycle_high, "#2d6cdf"), ("balanced", cycle_balanced, "#31a354")):
        if table.empty or "agreement" not in table.columns:
            continue
        plotted = True
        ax.plot(np.arange(len(table)), table["agreement"].astype(float), marker="o", linewidth=1.5, label=policy, color=color)
    ax.set_title("Cycle agreement by triplet")
    ax.set_xlabel("Triplet index")
    ax.set_ylabel("Agreement fraction")
    ax.grid(alpha=0.2)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No cycle triplets available", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    saved_paths.append(_save_figure(output_dir / "cycle_agreement.png", fig))

    if not track_length_summary.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for policy, color in (("high", "#2d6cdf"), ("balanced", "#31a354")):
            subset = track_length_summary.loc[track_length_summary["match_policy"].astype(str) == policy]
            if subset.empty:
                continue
            ax.step(
                subset["n_days_present"].astype(int),
                subset["n_tracks"].astype(int),
                where="mid",
                label=policy,
                color=color,
            )
        ax.set_title("Track length summary")
        ax.set_xlabel("Days present")
        ax.set_ylabel("Tracks")
        ax.grid(alpha=0.2)
        ax.legend()
        saved_paths.append(_save_figure(output_dir / "track_length_summary.png", fig))

    review_sample = _build_review_sample(tracks_high, tracks_balanced, config.sample_limit, config.review_seed)
    review_sample.to_csv(output_dir / "manual_review_sample.csv", index=False)
    (output_dir / "run_log.json").write_text(
        json.dumps(
            {
                "match_dir": str(match_dir),
                "output_dir": str(output_dir),
                "sample_limit": int(config.sample_limit),
                "review_seed": int(config.review_seed),
                "include_skip_pairs": bool(config.include_skip_pairs),
                "saved_plots": [str(path) for path in saved_paths],
                "review_sample_rows": int(len(review_sample)),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", required=True, help="Matcher output directory.")
    parser.add_argument("--output-dir", default=None, help="Directory for QC plots and review samples.")
    parser.add_argument("--sample-limit", type=int, default=6)
    parser.add_argument("--review-seed", type=int, default=7)
    parser.add_argument("--no-skip-pairs", action="store_true", help="Reserved for future skip-pair filtering.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    config = DaywiseQCPlotConfig(
        match_dir=args.match_dir,
        output_dir=args.output_dir,
        sample_limit=int(args.sample_limit),
        review_seed=int(args.review_seed),
        include_skip_pairs=not bool(args.no_skip_pairs),
    )
    return generate_daywise_qc_plots(config)


if __name__ == "__main__":
    main()
