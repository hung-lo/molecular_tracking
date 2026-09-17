"""Pure re-analysis of the saved Phase D1 pilot tables (no Cellpose run)."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import tifffile

from postprocessing.local_segmentation_rescue import _load_run_context, compute_crop_bounds


FAILURE_TRACKS = [
    "WT_Fucci-Tri_corFront_20260519:7222",
    "WT_Fucci-Tri_corFront_20260511:2564",
    "WT_Fucci-Tri_corFront_20260528:6586",
    "WT_Fucci-Tri_corFront_20260708:5047",
]


def _parse(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        return ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return []


def _truthy(value) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _parse_overlap_vector(value) -> list[tuple[int, int]] | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    result = []
    for item in parsed if isinstance(parsed, list) else []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            label, voxels = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            continue
        if label > 0 and voxels > 0:
            result.append((voxels, label))
    return sorted(result, key=lambda item: (-item[0], item[1]))


def _stats(values: pd.Series) -> dict[str, float | int | None]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"n": 0, "median": None, "q05": None, "q95": None, "min": None, "max": None}
    return {
        "n": int(len(values)),
        "median": float(values.median()),
        "q05": float(values.quantile(0.05)),
        "q95": float(values.quantile(0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _candidate_rows(candidates: pd.DataFrame, case: pd.Series) -> pd.DataFrame:
    rows = candidates[
        candidates["track_uid"].eq(case["track_uid"])
        & candidates["target_session"].eq(case["target_session"])
    ].copy()
    return rows.sort_values(["geometric_rank_score", "candidate_id"], kind="stable")


def _decompose(cases: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, case in cases.iterrows():
        candidate_rows = _candidate_rows(candidates, case)
        selected = candidate_rows[candidate_rows["candidate_id"].eq(case["selected_candidate_id"])]
        selected_row = selected.iloc[0] if not selected.empty else pd.Series(dtype=object)
        selected_exists = not selected.empty
        selected_voxels = float(selected_row.get("volume_voxels", np.nan))
        truth_voxels = float(case.get("truth_volume_voxels", np.nan))
        truth_label = case.get("target_truth_label", np.nan)
        truth_fraction_of_truth = float(case.get("selected_overlap_truth_fraction", np.nan))
        persisted_overlaps = _parse_overlap_vector(selected_row.get("canonical_overlap_voxels_json", ""))
        has_persisted_overlaps = persisted_overlaps is not None
        truth_overlap_voxels = int(selected_row.get("truth_overlap_voxels", 0)) if has_persisted_overlaps else (round(truth_fraction_of_truth * truth_voxels) if np.isfinite(truth_fraction_of_truth * truth_voxels) else np.nan)
        top1_voxels, top1_label = (persisted_overlaps[0] if persisted_overlaps else (0, np.nan)) if has_persisted_overlaps else (np.nan, case.get("selected_best_overlapping_canonical_label", np.nan))
        top2_voxels, top2_label = (persisted_overlaps[1] if len(persisted_overlaps) > 1 else (np.nan, np.nan)) if has_persisted_overlaps else (np.nan, np.nan)
        top1_fraction = top1_voxels / selected_voxels if has_persisted_overlaps and selected_voxels else (float(case.get("selected_best_overlapping_canonical_fraction", np.nan)) if not has_persisted_overlaps else 0.0)
        top2_fraction = top2_voxels / selected_voxels if has_persisted_overlaps and selected_voxels and len(persisted_overlaps) > 1 else np.nan
        try:
            truth_is_top1 = bool(selected_exists and int(top1_label) == int(truth_label))
        except (TypeError, ValueError):
            truth_is_top1 = False
        has_truth_overlap = bool(np.isfinite(truth_overlap_voxels) and truth_overlap_voxels > 0)
        top1_overlap_present = bool(np.isfinite(top1_fraction) and top1_fraction > 0)
        raw_top1_label = case.get("selected_best_overlapping_canonical_label", np.nan)
        if not top1_overlap_present:
            top1_label = np.nan
        elif not persisted_overlaps:
            top1_label = raw_top1_label
        if not selected_exists:
            dominant_class = "no_candidate"
        elif truth_is_top1:
            dominant_class = "dominant_truth_overlap"
        elif top1_overlap_present:
            dominant_class = "dominant_wrong_label"
        else:
            dominant_class = "no_canonical_overlap"
        non_top1 = 1.0 - top1_fraction if np.isfinite(top1_fraction) else np.nan
        if np.isfinite(non_top1):
            contamination_class = (
                "single_label_like" if non_top1 <= 0.01 else
                "minor_neighbor_contamination_proxy" if non_top1 <= 0.05 else
                "substantial_neighbor_contamination_proxy"
            )
        else:
            contamination_class = "unavailable"
        rows.append({
            "track_uid": case["track_uid"],
            "source_session": case["source_session"],
            "target_session": case["target_session"],
            "target_truth_label": case.get("target_truth_label", np.nan),
            "selected_candidate_id": case.get("selected_candidate_id", np.nan),
            "selected_candidate_voxels": selected_voxels,
            "truth_overlap_voxels": truth_overlap_voxels,
            "truth_overlap_fraction_of_candidate": truth_overlap_voxels / selected_voxels if selected_voxels else np.nan,
            "truth_overlap_fraction_of_truth": truth_fraction_of_truth,
            "top1_canonical_label": top1_label,
            "top1_overlap_voxels": top1_voxels,
            "top1_overlap_fraction_of_candidate": top1_fraction,
            "top2_canonical_label": top2_label,
            "top2_overlap_voxels": top2_voxels,
            "top2_overlap_fraction_of_candidate": top2_fraction,
            "top1_minus_top2_fraction": top1_fraction - top2_fraction if np.isfinite(top1_fraction) and np.isfinite(top2_fraction) else np.nan,
            "top1_to_top2_ratio": top1_fraction / top2_fraction if np.isfinite(top1_fraction) and np.isfinite(top2_fraction) and top2_fraction > 0 else np.nan,
            "n_canonical_labels_overlap_any": len(persisted_overlaps) if has_persisted_overlaps else np.nan,
            "n_canonical_labels_overlap_any_lower_bound": len(persisted_overlaps) if has_persisted_overlaps else (2 if _truthy(case.get("selected_overlaps_multiple_canonical_rois", False)) else 1),
            "n_canonical_labels_overlap_ge_1pct_candidate": sum(voxels / selected_voxels >= 0.01 for voxels, _ in persisted_overlaps) if has_persisted_overlaps and selected_voxels else (0 if has_persisted_overlaps else np.nan),
            "n_canonical_labels_overlap_ge_5pct_candidate": sum(voxels / selected_voxels >= 0.05 for voxels, _ in persisted_overlaps) if has_persisted_overlaps and selected_voxels else (0 if has_persisted_overlaps else np.nan),
            "n_canonical_labels_overlap_ge_10pct_candidate": sum(voxels / selected_voxels >= 0.10 for voxels, _ in persisted_overlaps) if has_persisted_overlaps and selected_voxels else (0 if has_persisted_overlaps else np.nan),
            "truth_is_top1": truth_is_top1,
            "not_truth_top1": bool(not truth_is_top1),
            "truth_overlap_present": has_truth_overlap,
            "no_truth_overlap": not has_truth_overlap,
            "top1_overlap_present": top1_overlap_present,
            "dominant_identity_class": dominant_class,
            "non_top1_fraction_upper_bound": non_top1,
            "contamination_class_descriptive": contamination_class,
            "dice_3d": case.get("dice_3d", np.nan),
            "iou_3d": case.get("iou_3d", np.nan),
            "centroid_error_um": case.get("centroid_error_um", np.nan),
            "volume_ratio_rescue_to_truth": case.get("volume_ratio_rescue_to_truth", np.nan),
            "raw_mask_mean_green_relative_difference": case.get("raw_mask_mean_green_relative_difference", np.nan),
            "raw_mask_mean_red_relative_difference": case.get("raw_mask_mean_red_relative_difference", np.nan),
            "raw_mask_ratio_relative_difference": case.get("raw_mask_ratio_relative_difference", np.nan),
            "top2_data_status": "persisted" if has_persisted_overlaps else "not_persisted_in_original_pilot_artifact",
        })
    return pd.DataFrame(rows)


def _plot_summary(decomp: pd.DataFrame, output: Path) -> None:
    plot_dir = output / "summary_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    for column, label, color in (
        ("top1_overlap_fraction_of_candidate", "top1 canonical fraction", "#2878b5"),
        ("truth_overlap_fraction_of_candidate", "truth fraction of candidate", "#4c9f70"),
        ("non_top1_fraction_upper_bound", "1 − top1 fraction", "#d95f02"),
    ):
        values = pd.to_numeric(decomp[column], errors="coerce").dropna()
        ax.hist(values, bins=20, alpha=0.45, label=label, color=color)
    ax.set_xlabel("fraction of selected candidate"); ax.set_ylabel("cases"); ax.legend(); ax.set_title("overlap decomposition (saved fields)")
    fig.tight_layout(); fig.savefig(plot_dir / "overlap_decomposition_distributions.png", dpi=160); plt.close(fig)

    truth = decomp["truth_is_top1"].astype(bool)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(decomp.loc[truth, "top1_overlap_fraction_of_candidate"], decomp.loc[truth, "dice_3d"], s=18, label="truth top-1", alpha=0.7)
    ax.scatter(decomp.loc[~truth, "top1_overlap_fraction_of_candidate"], decomp.loc[~truth, "dice_3d"], s=28, label="dominant-label failure", marker="x")
    ax.set_xlabel("top1 overlap fraction of candidate"); ax.set_ylabel("Dice"); ax.legend(); ax.set_title("identity ranking vs mask overlap")
    fig.tight_layout(); fig.savefig(plot_dir / "top1_vs_dice.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, column, label in zip(axes.flat, ["dice_3d", "iou_3d", "centroid_error_um", "volume_ratio_rescue_to_truth"], ["Dice", "IoU", "centroid error (um)", "volume ratio"]):
        ax.scatter(decomp["non_top1_fraction_upper_bound"], decomp[column], s=14, alpha=0.65)
        ax.set_xlabel("1 − top1 fraction"); ax.set_ylabel(label)
    fig.suptitle("non-top1 fraction upper bound vs quality")
    fig.tight_layout(); fig.savefig(plot_dir / "contamination_vs_metrics.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, column, label in zip(axes, ["raw_mask_mean_green_relative_difference", "raw_mask_mean_red_relative_difference", "raw_mask_ratio_relative_difference"], ["Green relative difference", "Red relative difference", "ratio relative difference"]):
        values = pd.to_numeric(decomp[column], errors="coerce").dropna()
        ax.hist(values, bins=20, color="#6a3d9a"); ax.set_title(label); ax.axvline(0, color="black", lw=0.8)
    fig.tight_layout(); fig.savefig(plot_dir / "measurement_bias_distributions.png", dpi=160); plt.close(fig)

    labels = ["dominant_truth_overlap", "dominant_wrong_label", "no_canonical_overlap", "no_candidate"]
    counts = decomp["dominant_identity_class"].value_counts().reindex(labels, fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3)); ax.bar(counts.index, counts.values, color=["#4c9f70", "#d95f02", "#7570b3", "#999999"]); ax.tick_params(axis="x", rotation=20); ax.set_ylabel("cases"); ax.set_title("descriptive identity reclassification"); fig.tight_layout(); fig.savefig(plot_dir / "identity_reclassification.png", dpi=160); plt.close(fig)


def _draw_bbox(ax, candidate: pd.Series, start_zyx: tuple[int, int, int], color: str, linestyle: str, label: str) -> None:
    bbox = _parse(candidate.get("bbox_zyx"))
    if len(bbox) != 6:
        return
    _, y0, x0, _, y1, x1 = [float(v) for v in bbox]
    _, sy, sx = start_zyx
    ax.add_patch(Rectangle((x0 - sx, y0 - sy), x1 - x0, y1 - y0, fill=False, edgecolor=color, linestyle=linestyle, linewidth=1.8, label=label))
    center = _parse(candidate.get("centroid_xyz"))
    if len(center) == 3:
        ax.plot(float(center[0]) - sx, float(center[1]) - sy, marker=".", color=color, ms=6)


def _panel(case: pd.Series, candidates: pd.DataFrame, ctx, path: Path, *, note: str) -> None:
    target_session = str(case["target_session"])
    target_row = ctx.sessions[ctx.sessions["session_id"].eq(target_session)].iloc[0]
    source_session = str(case["source_session"])
    source_row = ctx.sessions[ctx.sessions["session_id"].eq(source_session)].iloc[0]
    source_image = tifffile.imread(str(source_row["red_image_path"]))
    source_mask_stack = tifffile.imread(str(source_row["mask_path"]))
    target_image = tifffile.imread(str(target_row["red_image_path"]))
    target_mask_stack = tifffile.imread(str(target_row["mask_path"]))
    predicted_xyz = _parse(case["predicted_xyz"])
    crop_shape = tuple(int(v) for v in _parse(case["crop_shape_zyx"]))
    source_feature = ctx.features[
        ctx.features["session_id"].eq(source_session)
        & ctx.features["label"].eq(int(case["source_label"]))
    ].iloc[0]
    source_xyz = [float(source_feature["centroid_x"]), float(source_feature["centroid_y"]), float(source_feature["centroid_z"])]
    source_bounds = compute_crop_bounds(source_xyz, source_image.shape, crop_shape)
    bounds = compute_crop_bounds(predicted_xyz, target_image.shape, crop_shape)
    source_raw = np.asarray(source_image[source_bounds.slices])
    source_mask = np.asarray(source_mask_stack[source_bounds.slices])
    raw = np.asarray(target_image[bounds.slices])
    target_mask = np.asarray(target_mask_stack[bounds.slices])
    truth_label = int(case["target_truth_label"])
    candidate_rows = _candidate_rows(candidates, case)
    selected = candidate_rows[candidate_rows["candidate_id"].eq(case["selected_candidate_id"])]
    selected = selected.iloc[0] if not selected.empty else pd.Series(dtype=object)
    competitors = candidate_rows[~candidate_rows["candidate_id"].eq(case["selected_candidate_id"])].head(3)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 1, 1, 1.35]})
    axes[0].imshow(np.max(source_raw, axis=0), cmap="gray")
    axes[0].set_title("source Red crop")
    axes[0].contour(np.max(source_mask == int(case["source_label"]), axis=0), levels=[0.5], colors="gold", linewidths=1.7)
    axes[0].legend(["source canonical ROI"], loc="lower right", fontsize=8)
    axes[1].imshow(np.max(raw, axis=0), cmap="gray")
    axes[1].set_title("target Red crop")
    axes[1].plot(predicted_xyz[0] - bounds.start_zyx[2], predicted_xyz[1] - bounds.start_zyx[1], marker="x", color="yellow", ms=9, mew=2, label="predicted location")
    axes[1].legend(loc="lower right", fontsize=8)
    axes[2].imshow(np.max(raw, axis=0), cmap="gray")
    axes[2].contour(np.max(target_mask == truth_label, axis=0), levels=[0.5], colors="lime", linewidths=1.7)
    if not selected.empty:
        _draw_bbox(axes[2], selected, bounds.start_zyx, "cyan", "-", "selected candidate bbox")
    for _, competitor in competitors.iterrows():
        _draw_bbox(axes[2], competitor, bounds.start_zyx, "magenta", "--", "competing bbox")
    axes[2].plot(predicted_xyz[0] - bounds.start_zyx[2], predicted_xyz[1] - bounds.start_zyx[1], marker="x", color="yellow", ms=9, mew=2, label="predicted location")
    axes[2].set_title("hidden truth + candidate geometry")
    handles, labels = axes[2].get_legend_handles_labels()
    if handles: axes[2].legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), loc="lower right", fontsize=7)
    axes[3].axis("off")
    lines = [
        f"track: {case['track_uid']}", f"source → target: {case['source_session']} → {case['target_session']}",
        f"truth label: {truth_label}", f"selected id: {case['selected_candidate_id']}",
        f"top1 canonical label: {case['selected_best_overlapping_canonical_label']}",
        f"top1 fraction: {float(case['selected_best_overlapping_canonical_fraction']):.3f}",
        f"Dice / centroid error: {float(case['dice_3d']):.3f} / {float(case['centroid_error_um']):.2f} um", "",
        "ranker candidates (ascending score):",
    ]
    for _, row in candidate_rows.head(5).iterrows():
        lines.append(f"id {int(row['candidate_id']):>3}  dist {float(row['distance_from_prediction_um']):5.2f}  score {float(row['geometric_rank_score']):5.2f}  canon {row['best_overlapping_canonical_label']}  truth Dice {float(row['truth_dice']):.3f}")
    lines += ["", "bbox/centroid geometry shown; original pilot did not persist candidate masks", note]
    axes[3].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)
    for ax in axes[:3]: ax.axis("off")
    fig.suptitle(f"{case['track_uid']} — dominant-label adjudication", fontsize=13)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _write_panels(cases: pd.DataFrame, candidates: pd.DataFrame, ctx, output: Path, decomp: pd.DataFrame) -> dict[str, int]:
    failure_dir = output / "review_panels_four_failures"; failure_dir.mkdir(parents=True, exist_ok=True)
    for track in FAILURE_TRACKS:
        case = cases[cases["track_uid"].eq(track)].iloc[0]
        _panel(case, candidates, ctx, failure_dir / f"{track.replace(':', '_')}.png", note="explicit failure case; selected candidate was not truth top-1")
    truth = decomp[decomp["truth_is_top1"]].copy()
    selections: list[tuple[str, pd.Series]] = []
    pools = {
        "high_purity_high_dice": truth[(truth["top1_overlap_fraction_of_candidate"] >= .95) & (truth["dice_3d"] >= .9)].sort_values("dice_3d", ascending=False),
        "high_purity_moderate_dice": truth[(truth["top1_overlap_fraction_of_candidate"] >= .9) & (truth["dice_3d"] < .8)].sort_values("dice_3d", ascending=False),
        "minor_neighbor_contamination": truth[truth["non_top1_fraction_upper_bound"] <= .05].sort_values("non_top1_fraction_upper_bound"),
        "largest_second_label_contamination": truth.sort_values("non_top1_fraction_upper_bound", ascending=False),
        "small_candidate_undersegmentation": truth.sort_values("volume_ratio_rescue_to_truth"),
        "large_candidate_oversegmentation": truth.sort_values("volume_ratio_rescue_to_truth", ascending=False),
        "largest_centroid_error_truth_top1": truth.sort_values("centroid_error_um", ascending=False),
    }
    seen: set[str] = set()
    representative_dir = output / "review_panels_truth_top1"; representative_dir.mkdir(parents=True, exist_ok=True)
    for name, pool in pools.items():
        if pool.empty: continue
        case = cases[cases["track_uid"].eq(pool.iloc[0]["track_uid"]) & cases["target_session"].eq(pool.iloc[0]["target_session"])].iloc[0]
        key = f"{case['track_uid']}|{case['target_session']}"
        if key in seen: continue
        seen.add(key); _panel(case, candidates, ctx, representative_dir / f"{name}.png", note=f"stratum: {name}; top2 voxel fields unavailable in source artifact")
    return {"four_failures": len(list(failure_dir.glob("*.png"))), "truth_top1": len(list(representative_dir.glob("*.png")))}


def _failure_details(cases: pd.DataFrame, candidates: pd.DataFrame, decomp: pd.DataFrame) -> list[dict[str, object]]:
    """Return compact, deterministic ranking details for the four saved failures."""

    details: list[dict[str, object]] = []
    for track_uid in FAILURE_TRACKS:
        case = cases[cases["track_uid"].eq(track_uid)].iloc[0]
        decomposition = decomp[decomp["track_uid"].eq(track_uid)].iloc[0]
        rows = _candidate_rows(candidates, case)
        selected_id = case.get("selected_candidate_id", np.nan)
        selected = rows[rows["candidate_id"].eq(selected_id)]
        selected_row = selected.iloc[0] if not selected.empty else pd.Series(dtype=object)
        truth_dice = pd.to_numeric(rows.get("truth_dice", pd.Series(dtype=float)), errors="coerce")
        truth_rows = rows[truth_dice.gt(0)].copy()
        if not truth_rows.empty:
            truth_rows["_truth_dice"] = pd.to_numeric(truth_rows["truth_dice"], errors="coerce")
            truth_rows = truth_rows.sort_values(["_truth_dice", "geometric_rank_score", "candidate_id"], ascending=[False, True, True], kind="stable")
            best_truth = truth_rows.iloc[0]
        else:
            best_truth = pd.Series(dtype=object)

        def _number(row: pd.Series, column: str):
            value = row.get(column, np.nan)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) else None

        def _integer(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return int(value) if np.isfinite(value) else None

        ranking_rows = rows.head(5)
        ranking = []
        for _, row in ranking_rows.iterrows():
            ranking.append({
                "candidate_id": _integer(row.get("candidate_id")),
                "distance_from_prediction_um": _number(row, "distance_from_prediction_um"),
                "geometric_rank_score": _number(row, "geometric_rank_score"),
                "truth_dice": _number(row, "truth_dice"),
                "best_overlapping_canonical_label": _integer(row.get("best_overlapping_canonical_label")),
            })
        details.append({
            "track_uid": track_uid,
            "truth_overlap_present": bool(decomposition["truth_overlap_present"]),
            "no_truth_overlap": bool(decomposition["no_truth_overlap"]),
            "top1_overlap_present": bool(decomposition["top1_overlap_present"]),
            "top1_canonical_label": _integer(decomposition["top1_canonical_label"]),
            "truth_label": _integer(decomposition["target_truth_label"]),
            "top1_fraction": _number(decomposition, "top1_overlap_fraction_of_candidate"),
            "selected_candidate_id": _integer(selected_id),
            "selected_distance_from_prediction_um": _number(selected_row, "distance_from_prediction_um"),
            "selected_geometric_rank_score": _number(selected_row, "geometric_rank_score"),
            "best_truth_overlap_candidate_id": _integer(best_truth.get("candidate_id", np.nan)),
            "best_truth_overlap_candidate_dice": _number(best_truth, "truth_dice"),
            "best_truth_overlap_candidate_distance_from_prediction_um": _number(best_truth, "distance_from_prediction_um"),
            "ranking_top5": ranking,
        })
    return details


def readjudicate(input_dir: str | Path, output_dir: str | Path, run_dir: str | Path) -> dict[str, object]:
    input_dir, output = Path(input_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(input_dir / "pilot_cases.csv", low_memory=False)
    candidates = pd.read_csv(input_dir / "pilot_candidates.csv", low_memory=False)
    decomp = _decompose(cases, candidates)
    decomp.to_csv(output / "selected_overlap_decomposition.csv", index=False)
    _plot_summary(decomp, output)
    ctx = _load_run_context(run_dir)
    panel_counts = _write_panels(cases, candidates, ctx, output, decomp)
    truth_top1 = decomp["truth_is_top1"].astype(bool)
    bias = {column: _stats(decomp[column]) for column in ("raw_mask_mean_green_relative_difference", "raw_mask_mean_red_relative_difference", "raw_mask_ratio_relative_difference")}
    truth_overlap_present = decomp["truth_overlap_present"].astype(bool)
    not_truth_top1 = decomp["not_truth_top1"].astype(bool)
    identity_counts = decomp["dominant_identity_class"].value_counts()
    new_descriptive = {
        "truth_is_top1_count": int(truth_top1.sum()),
        "truth_is_top1_rate": float(truth_top1.mean()),
        "not_truth_top1_count": int(not_truth_top1.sum()),
        "not_truth_top1_rate": float(not_truth_top1.mean()),
        "dominant_wrong_label_count": int(identity_counts.get("dominant_wrong_label", 0)),
        "dominant_wrong_label_rate": float(identity_counts.get("dominant_wrong_label", 0) / len(decomp)) if len(decomp) else None,
        "no_canonical_overlap_count": int(identity_counts.get("no_canonical_overlap", 0)),
        "no_canonical_overlap_rate": float(identity_counts.get("no_canonical_overlap", 0) / len(decomp)) if len(decomp) else None,
        "no_truth_overlap_count": int((~truth_overlap_present).sum()),
        "no_truth_overlap_rate": float((~truth_overlap_present).mean()),
        "identity_ranking_categories": identity_counts.to_dict(),
        "top1_fraction": _stats(decomp["top1_overlap_fraction_of_candidate"]),
        "truth_fraction_of_candidate": _stats(decomp["truth_overlap_fraction_of_candidate"]),
        "top2_fraction": None,
    }
    summary = {
        "input_dir": str(input_dir.resolve()), "n_cases": int(len(decomp)), "old": {"identity_correct_rate": 0.0, "merged_multiple_cells_rate": 0.96},
        "new_descriptive": new_descriptive,
        "contamination_proxy_descriptive_only": {"definition": "1 - top1_overlap_fraction_of_candidate; includes background and any non-top1 labels", "counts": decomp["contamination_class_descriptive"].value_counts().to_dict(), "distribution": _stats(decomp["non_top1_fraction_upper_bound"])},
        "mask_quality": {column: _stats(decomp[column]) for column in ("dice_3d", "iou_3d", "centroid_error_um", "volume_ratio_rescue_to_truth")},
        "measurement_bias": bias,
        "four_dominant_label_failures": FAILURE_TRACKS,
        "failure_details": _failure_details(cases, candidates, decomp),
        "ranker_failure_notes": "The saved artifact contains no candidate masks or per-canonical-label overlap vectors. Failure panels therefore show saved candidate bbox/centroid geometry and rank scores; exact top2 voxel fractions require a future run that persists those vectors.",
        "panel_counts": panel_counts,
        "cellpose_rerun": False, "ranking_changed": False, "production_enabled": False,
    }
    (output / "summary_reclassified.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(readjudicate(args.input_dir, args.output_dir, args.run_dir), indent=2))


if __name__ == "__main__":
    main()
