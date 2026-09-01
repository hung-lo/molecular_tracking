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
import tifffile

_REPO_ROOT = Path(__file__).resolve().parent.parent
for _import_dir in (_REPO_ROOT / "core", _REPO_ROOT / "matching"):
    _import_dir_str = str(_import_dir)
    if _import_dir_str not in sys.path:
        sys.path.append(_import_dir_str)


@dataclass(frozen=True)
class MatchingQCConfig:
    """Configuration for generating QC artifacts from a completed match run."""

    match_dir: str | Path
    output_dir: str | Path | None = None
    sample_limit: int = 6
    review_seed: int = 7
    include_skip_pairs: bool = True
    image_format: str = "png"
    dpi: int = 150
    max_examples_per_category: int = 20
    max_total_examples: int = 100
    generate_visual_examples: bool = True
    random_seed: int = 0


DaywiseQCPlotConfig = MatchingQCConfig


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


def _save_figure(path: Path, fig: plt.Figure, *, dpi: int) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


QC_TABLE_COLUMNS = [
    "session_a", "session_b", "acquisition_date_a", "acquisition_date_b",
    "label_a", "label_b", "accepted_for_track", "rejection_reason", "score",
    "dice", "distance_um", "ambiguity", "candidate_source", "source_x",
    "source_y", "source_z_centroid", "target_x", "target_y", "target_z_centroid",
    "raw_delta_z_planes", "raw_abs_delta_z_planes", "long_axis",
    "long_axis_position_px", "long_axis_position_normalized",
    "short_axis_position_normalized", "spatial_grid_row", "spatial_grid_col",
    "selected_high_confidence_example", "selected_rejected_example",
    "selected_large_z_example",
]


def detect_long_axis(image_shape_yx: tuple[int, int]) -> str:
    """Return the longer native image axis, with deterministic tie handling."""
    height, width = (int(image_shape_yx[0]), int(image_shape_yx[1]))
    if height <= 0 or width <= 0:
        raise ValueError("image_shape_yx must contain positive dimensions")
    return "x" if width >= height else "y"


def add_spatial_and_z_qc_columns(
    candidates: pd.DataFrame,
    features: pd.DataFrame,
    *,
    image_shape_yx: tuple[int, int],
    z_spacing_um: float | None = None,
    target_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add native-coordinate displacement and 3x3 FoV-bin columns."""
    if candidates.empty:
        return candidates.copy()
    required = {"label_a", "label_b"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {', '.join(sorted(missing))}")
    feature_lookup = features.set_index("label") if "label" in features.columns else features
    target_lookup = target_features.set_index("label") if target_features is not None and "label" in target_features.columns else feature_lookup
    rows = candidates.copy()
    height, width = int(image_shape_yx[0]), int(image_shape_yx[1])
    long_axis = detect_long_axis(image_shape_yx)
    sx = feature_lookup.reindex(pd.to_numeric(rows["label_a"], errors="coerce"))["centroid_x"].to_numpy(float)
    sy = feature_lookup.reindex(pd.to_numeric(rows["label_a"], errors="coerce"))["centroid_y"].to_numpy(float)
    sz = feature_lookup.reindex(pd.to_numeric(rows["label_a"], errors="coerce"))["centroid_z"].to_numpy(float)
    tx = target_lookup.reindex(pd.to_numeric(rows["label_b"], errors="coerce"))["centroid_x"].to_numpy(float)
    ty = target_lookup.reindex(pd.to_numeric(rows["label_b"], errors="coerce"))["centroid_y"].to_numpy(float)
    tz = target_lookup.reindex(pd.to_numeric(rows["label_b"], errors="coerce"))["centroid_z"].to_numpy(float)
    rows["source_x"], rows["source_y"], rows["source_z_centroid"] = sx, sy, sz
    rows["target_x"], rows["target_y"], rows["target_z_centroid"] = tx, ty, tz
    rows["raw_delta_z_planes"] = tz - sz
    rows["raw_abs_delta_z_planes"] = np.abs(rows["raw_delta_z_planes"])
    if z_spacing_um is not None and np.isfinite(float(z_spacing_um)):
        rows["raw_delta_z_um"] = rows["raw_delta_z_planes"] * float(z_spacing_um)
        rows["raw_abs_delta_z_um"] = rows["raw_abs_delta_z_planes"] * float(z_spacing_um)
    if long_axis == "x":
        long_px, short_norm = sx, sy / max(height - 1, 1)
        long_norm = sx / max(width - 1, 1)
    else:
        long_px, short_norm = sy, sx / max(width - 1, 1)
        long_norm = sy / max(height - 1, 1)
    rows["long_axis"] = long_axis
    rows["long_axis_position_px"] = long_px
    rows["long_axis_position_normalized"] = np.clip(long_norm, 0, 1)
    rows["short_axis_position_normalized"] = np.clip(short_norm, 0, 1)
    rows["spatial_grid_row"] = np.minimum((np.clip(sy / max(height, 1), 0, 0.999999) * 3).astype(int), 2)
    rows["spatial_grid_col"] = np.minimum((np.clip(sx / max(width, 1), 0, 0.999999) * 3).astype(int), 2)
    return rows


def _finite_sort(table: pd.DataFrame, columns: list[str], ascending: list[bool]) -> pd.DataFrame:
    work = table.copy()
    for col in columns:
        work[col] = pd.to_numeric(work[col], errors="coerce") if col in work else np.nan
    return work.sort_values(columns, ascending=ascending, na_position="last", kind="mergesort")


def select_spatial_examples(candidates: pd.DataFrame, *, accepted: bool, limit: int = 8) -> pd.DataFrame:
    """Select deterministic, non-duplicate examples across populated 3x3 bins."""
    if limit <= 0 or candidates.empty:
        return candidates.iloc[0:0].copy()
    work = candidates.loc[candidates["accepted_for_track"].fillna(False).astype(bool).eq(bool(accepted))].copy()
    if not accepted:
        work = work.loc[work[["score", "dice", "distance_um"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)]
    sort_cols = ["score", "dice", "ambiguity", "distance_um"] if accepted else ["score", "dice", "distance_um", "ambiguity"]
    asc = [False, False, True, True] if accepted else [False, False, True, True]
    work = _finite_sort(work, sort_cols, asc)
    chosen = []
    seen_labels: set[tuple[str, str]] = set()
    for _, group in work.groupby(["spatial_grid_row", "spatial_grid_col"], sort=True):
        for _, row in group.iterrows():
            key = (str(row.get("session_a", "")), str(row.get("label_a", "")))
            if key not in seen_labels:
                chosen.append(row)
                seen_labels.add(key)
                break
        if len(chosen) >= limit:
            break
    for _, row in work.iterrows():
        if len(chosen) >= limit:
            break
        key = (str(row.get("session_a", "")), str(row.get("label_a", "")))
        if key not in seen_labels:
            chosen.append(row)
            seen_labels.add(key)
    return pd.DataFrame(chosen, columns=work.columns).reset_index(drop=True)


def source_roi_match_fraction(table: pd.DataFrame, *, bins: int = 5) -> pd.DataFrame:
    """Summarize acceptance using unique source ROIs, never candidate rows."""
    if table.empty:
        return pd.DataFrame(columns=["bin", "n_source_rois", "n_accepted_source_rois", "accepted_fraction"])
    work = table.copy()
    work["bin"] = np.minimum((pd.to_numeric(work["long_axis_position_normalized"], errors="coerce").clip(0, 0.999999) * bins).astype(int), bins - 1)
    source = work.groupby(["bin", "label_a"], as_index=False)["accepted_for_track"].max()
    out = source.groupby("bin").agg(n_source_rois=("label_a", "size"), n_accepted_source_rois=("accepted_for_track", "sum")).reset_index()
    out["accepted_fraction"] = out["n_accepted_source_rois"] / out["n_source_rois"]
    return out


def _figure_for_table(title: str, ncols: int = 2, nrows: int = 2) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows))
    fig.suptitle(title, fontsize=14)
    return fig, np.asarray(axes).reshape(nrows, ncols)


def _plot_pairwise_distributions(
    output_dir: Path,
    pairwise_summary_long: pd.DataFrame,
    pairwise_matches: pd.DataFrame,
    *,
    dpi: int,
) -> list[Path]:
    saved: list[Path] = []
    if pairwise_summary_long.empty and pairwise_matches.empty:
        return saved

    summary = pairwise_summary_long.copy()
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
            if "n_matches" in policy_summary.columns:
                axes[0, 0].hist(policy_summary["n_matches"].fillna(0).astype(float), bins=20, color="#2d6cdf", alpha=0.8)
                axes[0, 0].set_title("Accepted matches per pair")
                axes[0, 1].hist(policy_summary["n_matches"].fillna(0).astype(float), bins=20, color="#31a354", alpha=0.8)
                axes[0, 1].set_title("Accepted matches per pair")
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
        saved.append(_save_figure(output_dir / f"pairwise_summary_{policy}.png", fig, dpi=dpi))

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
            saved.append(_save_figure(output_dir / f"candidate_distributions_{policy}.png", fig, dpi=dpi))

    return saved


def _plot_track_summaries(output_dir: Path, tracks: pd.DataFrame, policy: str, *, dpi: int) -> list[Path]:
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
    saved.append(_save_figure(output_dir / f"track_summary_{policy}.png", fig, dpi=dpi))

    if "has_cycle_conflict" in tracks.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = tracks["has_cycle_conflict"].fillna(False).astype(bool).value_counts().sort_index()
        ax.bar(["no", "yes"], [int(counts.get(False, 0)), int(counts.get(True, 0))], color=["#31a354", "#d62728"])
        ax.set_title(f"{policy} cycle conflicts")
        ax.grid(axis="y", alpha=0.2)
        saved.append(_save_figure(output_dir / f"cycle_conflicts_{policy}.png", fig, dpi=dpi))

    return saved


def _build_pairwise_summary_long(
    pairwise_summary: pd.DataFrame,
    pairwise_summary_graph: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not pairwise_summary.empty:
        for row in pairwise_summary.itertuples(index=False):
            base = row._asdict()
            for policy, value_key in (("high", "n_high"), ("balanced", "n_balanced")):
                rows.append(
                    {
                        "match_policy": policy,
                        "day_a": base.get("day_a"),
                        "day_b": base.get("day_b"),
                        "pair_gap": base.get("pair_gap"),
                        "n_matches": base.get(value_key),
                        "elapsed_sec": base.get("elapsed_sec"),
                        "transform_residual_median_um": base.get("transform_residual_median_um"),
                        "transform_residual_p95_um": base.get("transform_residual_p95_um"),
                    }
                )
    if not pairwise_summary_graph.empty:
        for row in pairwise_summary_graph.itertuples(index=False):
            base = row._asdict()
            rows.append(
                {
                    "match_policy": str(base.get("match_policy", "graph")),
                    "day_a": base.get("day_a"),
                    "day_b": base.get("day_b"),
                    "pair_gap": base.get("pair_gap"),
                    "n_matches": base.get("n_graph", base.get("n_matches", 0)),
                    "elapsed_sec": base.get("elapsed_sec"),
                    "transform_residual_median_um": base.get("transform_residual_median_um"),
                    "transform_residual_p95_um": base.get("transform_residual_p95_um"),
                }
            )
    return pd.DataFrame(rows)


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


def _candidate_rejection_reason(row: pd.Series) -> str:
    """Describe why a candidate is absent from the canonical accepted table."""
    if bool(row.get("high_rule", False)) and not bool(row.get("balanced_rule", False)):
        return "balanced_rule_false"
    if not bool(row.get("high_rule", False)):
        return "high_rule_false"
    return "assignment_or_policy_conflict"


def _load_pair_features(match_dir: Path) -> dict[str, pd.DataFrame]:
    features = _load_csv(match_dir, "roi_features.csv")
    if features.empty or "session_id" not in features.columns:
        return {}
    return {str(session): table.copy() for session, table in features.groupby("session_id", sort=False)}


def _accepted_keys(table: pd.DataFrame) -> set[tuple[str, str, str, str]]:
    if table.empty:
        return set()
    return {(str(r.day_a), str(r.day_b), str(r.label_a), str(r.label_b)) for r in table.itertuples()}


def _render_match_contact_sheet(table: pd.DataFrame, manifest: pd.DataFrame, output_path: Path, *, title: str, dpi: int, z_radius: int = 0) -> None:
    """Render native red crops for selected pairwise candidates."""
    if table.empty:
        fig, ax = plt.subplots(figsize=(8, 2.5)); ax.axis("off"); ax.text(.5, .5, "No eligible examples", ha="center", va="center"); _save_figure(output_path, fig, dpi=dpi); return
    sessions = manifest.set_index(manifest["session_id"].astype(str))
    required_sessions = set(table["session_a"].astype(str)) | set(table["session_b"].astype(str))
    if any(session not in sessions.index or not Path(str(sessions.loc[session, "red_image_path"])).is_file() or not Path(str(sessions.loc[session, "mask_path"])).is_file() for session in required_sessions):
        fig, ax = plt.subplots(figsize=(8, 2.5)); ax.axis("off"); ax.text(.5, .5, "Raw image/mask files unavailable; visual example skipped", ha="center", va="center"); _save_figure(output_path, fig, dpi=dpi); return
    images: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for session in sorted(set(table["session_a"].astype(str)) | set(table["session_b"].astype(str))):
        if session not in sessions.index: continue
        row = sessions.loc[session]
        images[session] = (tifffile.imread(row["red_image_path"]), tifffile.imread(row["mask_path"]))
    n = len(table); fig, axes = plt.subplots(n, 2, figsize=(8, max(2.4, 2.8 * n)), squeeze=False)
    for i, item in table.iterrows():
        panels = []
        for side, label_col in (("A", "label_a"), ("B", "label_b")):
            session = str(item["session_a"] if side == "A" else item["session_b"]); label = int(item[label_col])
            image, mask = images[session]; coords = np.where(mask == label); z0 = int(round(float(coords[0].mean())))
            yc, xc = int(round(float(coords[1].mean()))), int(round(float(coords[2].mean())))
            h, w = image.shape[1:]; size = max(48, int(max(np.ptp(coords[1]), np.ptp(coords[2])) + 24)); y0=max(0,yc-size//2); x0=max(0,xc-size//2)
            crop=image[z0, y0:min(h,y0+size), x0:min(w,x0+size)]
            panels.append((crop, mask[z0, y0:min(h,y0+size), x0:min(w,x0+size)] == label, z0, session, label))
        for j, (crop, roi, z0, session, label) in enumerate(panels):
            ax=axes[i,j]; ax.imshow(crop, cmap="magma", vmin=0, vmax=np.percentile(crop, 99.5) if np.any(crop) else 1); ax.contour(roi, levels=[.5], colors="cyan", linewidths=.7); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{session} label {label} z={z0}", fontsize=8)
        accepted = bool(item["accepted_for_track"]); reason = "accepted" if accepted else str(item.get("rejection_reason", "rejected"))
        fig.text(.5, (n-i-.98)/n, f"{item['session_a']} -> {item['session_b']} | {item['label_a']} -> {item['label_b']} | {reason} | score={float(item['score']):.3g} dice={float(item['dice']):.3g} dz={float(item['raw_delta_z_planes']):+.2f}", ha="center", fontsize=8)
    fig.suptitle(title); _save_figure(output_path, fig, dpi=dpi)


def _generate_spatial_pair_qc(match_dir: Path, output_dir: Path, candidates: pd.DataFrame, accepted_tables: list[pd.DataFrame], *, dpi: int) -> list[Path]:
    manifest = _load_csv(match_dir, "session_manifest_resolved.csv")
    features_by_session = _load_pair_features(match_dir)
    if candidates.empty or manifest.empty or not features_by_session:
        return []
    accepted = set().union(*(_accepted_keys(t) for t in accepted_tables))
    rows = candidates.copy()
    rows["session_a"] = rows["day_a"].astype(str); rows["session_b"] = rows["day_b"].astype(str)
    rows["accepted_for_track"] = [key in accepted for key in zip(rows.session_a, rows.session_b, rows.label_a.astype(str), rows.label_b.astype(str))]
    rows["rejection_reason"] = rows.apply(lambda r: "" if r.accepted_for_track else _candidate_rejection_reason(r), axis=1)
    paths=[]; summaries=[]; examples_dir=output_dir / "pair_examples"; axial_dir=output_dir / "axial_shift"; tables_dir=output_dir / "tables"
    for pair, pair_rows in rows.groupby(["session_a", "session_b"], sort=True):
        sa, sb = pair; ma = manifest.loc[manifest.session_id.astype(str).eq(sa)];
        if ma.empty: continue
        mask = tifffile.imread(ma.iloc[0]["mask_path"]); shape=(mask.shape[1], mask.shape[2])
        enriched=add_spatial_and_z_qc_columns(pair_rows, features_by_session[sa], target_features=features_by_session[sb], image_shape_yx=shape)
        high=select_spatial_examples(enriched, accepted=True); rejected=select_spatial_examples(enriched, accepted=False)
        for selected, column in ((high, "selected_high_confidence_example"), (rejected, "selected_rejected_example")):
            enriched[column]=False
            if not selected.empty: enriched.loc[selected.index, column]=True
        large=enriched.sort_values("raw_abs_delta_z_planes", ascending=False).head(4); enriched["selected_large_z_example"]=enriched.index.isin(large.index)
        for selected, name, title in ((high, "high_confidence_accepted", "High-confidence accepted"), (rejected, "informative_rejected", "Informative rejected")):
            path=examples_dir/f"{sa}_{sb}_{name}.png"; _render_match_contact_sheet(selected, manifest, path, title=f"{title}: {sa} -> {sb}", dpi=dpi); paths.append(path)
        enriched.to_csv(tables_dir/f"{sa}_{sb}_matching_qc_examples.csv", index=False)
        accepted_high=enriched.loc[enriched.accepted_for_track & enriched.high_rule.astype(bool)]
        n_source = int(enriched["label_a"].nunique())
        n_accepted = int(enriched.loc[enriched.accepted_for_track, "label_a"].nunique())
        summaries.append({"session_a": sa, "session_b": sb, "n_source_rois": n_source, "n_accepted_source_rois": n_accepted, "accepted_fraction": n_accepted / n_source if n_source else np.nan, "n_high_confidence": int(len(accepted_high)), "median_raw_delta_z_planes": accepted_high["raw_delta_z_planes"].median(), "median_abs_raw_delta_z_planes": accepted_high["raw_abs_delta_z_planes"].median(), "p90_abs_raw_delta_z_planes": accepted_high["raw_abs_delta_z_planes"].quantile(.9), "long_axis": enriched["long_axis"].iloc[0]})
        frac=source_roi_match_fraction(enriched)
        fig, axes=plt.subplots(1,3,figsize=(14,4));
        axes[0].scatter(accepted_high.long_axis_position_normalized, accepted_high.raw_delta_z_planes, s=8); axes[0].set_title("Signed raw dz vs long axis"); axes[1].scatter(accepted_high.long_axis_position_normalized, accepted_high.raw_abs_delta_z_planes, s=8); axes[1].set_title("Absolute raw dz vs long axis"); axes[2].bar(frac.bin.astype(str), frac.accepted_fraction); axes[2].set_title("Accepted source fraction")
        for ax in axes: ax.grid(alpha=.2); ax.set_xlabel("normalized long-axis position")
        path=axial_dir/f"{sa}_{sb}_axial_shift_qc.png"; _save_figure(path,fig,dpi=dpi); paths.append(path)
        fig, axes=plt.subplots(1,2,figsize=(9,3.5)); axes[0].hist(accepted_high.raw_delta_z_planes.dropna(), bins=15); axes[0].set_title("Signed raw dz (planes)"); axes[1].hist(accepted_high.raw_abs_delta_z_planes.dropna(), bins=15); axes[1].set_title("Absolute raw dz (planes)"); path=axial_dir/f"{sa}_{sb}_delta_z_distribution.png"; _save_figure(path,fig,dpi=dpi); paths.append(path)
    pd.DataFrame(summaries).to_csv(tables_dir / "matching_qc_pair_summary.csv", index=False)
    return paths


def generate_matching_qc(config: MatchingQCConfig) -> dict[str, Path]:
    match_dir = Path(config.match_dir).resolve()
    if not match_dir.exists():
        raise FileNotFoundError(f"match_dir was not found: {match_dir}")
    output_dir = Path(config.output_dir).resolve() if config.output_dir else match_dir / "qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    pairwise_summary = _load_csv(match_dir, "pairwise_summary.csv")
    pairwise_summary_graph = _load_csv(match_dir, "pairwise_summary_graph.csv")
    pairwise_matches_high = _load_csv(match_dir, "pairwise_matches_high.csv")
    pairwise_matches_balanced = _load_csv(match_dir, "pairwise_matches_balanced.csv")
    pairwise_matches_graph = _load_csv(match_dir, "pairwise_matches_graph.csv")
    pairwise_candidates = _load_csv(match_dir, "pairwise_candidates.csv")
    tracks_high = _load_csv(match_dir, "tracks_high.csv")
    tracks_balanced = _load_csv(match_dir, "tracks_balanced.csv")
    tracks_graph = _load_csv(match_dir, "tracks_graph.csv")
    cycle_high = _load_csv(match_dir, "cycle_consistency_high.csv")
    cycle_balanced = _load_csv(match_dir, "cycle_consistency_balanced.csv")
    cycle_graph = _load_csv(match_dir, "cycle_consistency_graph.csv")
    track_length_summary = _load_csv(match_dir, "track_length_summary.csv")
    track_length_summary_graph = _load_csv(match_dir, "track_length_summary_graph.csv")

    pairwise_summary_long = _build_pairwise_summary_long(pairwise_summary, pairwise_summary_graph)
    pairwise_matches_all = pd.concat(
        [table for table in [pairwise_matches_high, pairwise_matches_balanced, pairwise_matches_graph] if not table.empty],
        ignore_index=True,
    ) if any(not table.empty for table in [pairwise_matches_high, pairwise_matches_balanced, pairwise_matches_graph]) else pd.DataFrame()

    saved_paths: list[Path] = []
    if config.generate_visual_examples:
        for subdir in (output_dir / "pair_examples", output_dir / "axial_shift", output_dir / "tables"):
            subdir.mkdir(parents=True, exist_ok=True)
        if not pairwise_candidates.empty:
            saved_paths.extend(
                _generate_spatial_pair_qc(
                    match_dir,
                    output_dir,
                    pairwise_candidates,
                    [pairwise_matches_high, pairwise_matches_balanced, pairwise_matches_graph],
                    dpi=int(config.dpi),
                )
            )
        saved_paths.extend(
            _plot_pairwise_distributions(
                output_dir,
                pairwise_summary_long,
                pairwise_matches_all,
                dpi=int(config.dpi),
            )
        )
        saved_paths.extend(_plot_track_summaries(output_dir, tracks_high, "high", dpi=int(config.dpi)))
        saved_paths.extend(_plot_track_summaries(output_dir, tracks_balanced, "balanced", dpi=int(config.dpi)))
        if not tracks_graph.empty:
            saved_paths.extend(_plot_track_summaries(output_dir, tracks_graph, "graph", dpi=int(config.dpi)))

    fig, ax = plt.subplots(figsize=(8, 4))
    plotted = False
    for policy, table, color in (("high", cycle_high, "#2d6cdf"), ("balanced", cycle_balanced, "#31a354"), ("graph", cycle_graph, "#f28e2b")):
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
    saved_paths.append(_save_figure(output_dir / "cycle_agreement.png", fig, dpi=int(config.dpi)))

    if not track_length_summary.empty or not track_length_summary_graph.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for policy, table, color in (("high", track_length_summary, "#2d6cdf"), ("balanced", track_length_summary, "#31a354"), ("graph", track_length_summary_graph, "#f28e2b")):
            if table.empty:
                continue
            if "match_policy" in table.columns:
                subset = table.loc[table["match_policy"].astype(str) == policy]
            else:
                subset = table.copy()
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
        saved_paths.append(_save_figure(output_dir / "track_length_summary.png", fig, dpi=int(config.dpi)))

    review_sample = _build_review_sample(tracks_high, tracks_balanced, config.sample_limit, config.review_seed)
    if not tracks_graph.empty:
        graph_sample = tracks_graph.head(min(len(tracks_graph), int(config.sample_limit))).copy()
        if not graph_sample.empty:
            graph_sample.insert(0, "review_category", "graph_preview")
            review_sample = pd.concat([review_sample, graph_sample], ignore_index=True) if not review_sample.empty else graph_sample
    if len(review_sample) > int(config.max_total_examples):
        review_sample = review_sample.head(int(config.max_total_examples)).copy()
    review_sample.to_csv(output_dir / "manual_review_sample.csv", index=False)
    report_path = output_dir / "qc_report.md"
    report_path.write_text(
        "\n".join(
            [
                "# Daywise Matching QC",
                "",
                f"- Match directory: `{match_dir}`",
                f"- Output directory: `{output_dir}`",
                f"- Review sample rows: `{len(review_sample)}`",
                f"- Visual examples enabled: `{bool(config.generate_visual_examples)}`",
                f"- Saved plots: `{len(saved_paths)}`",
            ]
        ),
        encoding="utf-8",
    )
    run_log_path = output_dir / "run_log.json"
    run_log_path.write_text(
        json.dumps(
            {
                "match_dir": str(match_dir),
                "output_dir": str(output_dir),
                "sample_limit": int(config.sample_limit),
                "review_seed": int(config.review_seed),
                "include_skip_pairs": bool(config.include_skip_pairs),
                "image_format": str(config.image_format),
                "dpi": int(config.dpi),
                "max_examples_per_category": int(config.max_examples_per_category),
                "max_total_examples": int(config.max_total_examples),
                "generate_visual_examples": bool(config.generate_visual_examples),
                "random_seed": int(config.random_seed),
                "saved_plots": [str(path) for path in saved_paths],
                "review_sample_rows": int(len(review_sample)),
                "qc_report": str(report_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact_paths: dict[str, Path] = {
        "output_dir": output_dir,
        "manual_review_sample": output_dir / "manual_review_sample.csv",
        "run_log": run_log_path,
        "qc_report": report_path,
    }
    for saved_path in saved_paths:
        artifact_paths[saved_path.stem] = saved_path
    return artifact_paths


def generate_daywise_qc_plots(config: DaywiseQCPlotConfig) -> Path:
    return generate_matching_qc(config)["output_dir"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-dir", required=True, help="Matcher output directory.")
    parser.add_argument("--output-dir", default=None, help="Directory for QC plots and review samples.")
    parser.add_argument("--sample-limit", type=int, default=6)
    parser.add_argument("--review-seed", type=int, default=7)
    parser.add_argument("--no-skip-pairs", action="store_true", help="Reserved for future skip-pair filtering.")
    parser.add_argument("--image-format", default="png")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--max-examples-per-category", type=int, default=20)
    parser.add_argument("--max-total-examples", type=int, default=100)
    parser.add_argument("--no-visual-examples", action="store_true", help="Skip generating figure artifacts.")
    parser.add_argument("--random-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    config = DaywiseQCPlotConfig(
        match_dir=args.match_dir,
        output_dir=args.output_dir,
        sample_limit=int(args.sample_limit),
        review_seed=int(args.review_seed),
        include_skip_pairs=not bool(args.no_skip_pairs),
        image_format=str(args.image_format),
        dpi=int(args.dpi),
        max_examples_per_category=int(args.max_examples_per_category),
        max_total_examples=int(args.max_total_examples),
        generate_visual_examples=not bool(args.no_visual_examples),
        random_seed=int(args.random_seed),
    )
    return generate_daywise_qc_plots(config)


if __name__ == "__main__":
    main()
