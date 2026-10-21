#!/usr/bin/env python3
"""Audit Tri_6 exact pairwise candidates without rerunning production matching."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "matching"))

from affine_overlap_matcher import (  # noqa: E402
    AffineOverlapParams,
    VoxelSpacing,
    _nearest_neighbor_data,
    build_sparse_overlap_table,
    greedy_one_to_one,
)
from image_registration import load_image_transform, restricted_transform, warp_volume  # noqa: E402
from pairwise_shadow_diagnostics import (  # noqa: E402
    candidate_rule_diagnostics,
    diagnose_candidate_table,
    trace_missing_candidate_gates,
)


RUN_DIR = Path(
    "/mnt/d/_data/_newAAV_2026/molecular_tracking_derivatives/Fucci-Tri_6/longitudinal/1050/runs/"
    "1050_20260924_to_20261019_14s_graph_affine_balanced_image_affine_local"
)
MATCH_DIR = RUN_DIR / "matching"
PRIOR_AUDIT_DIR = ROOT / "analysis/tri6_tplus2_edge_failure_audit_2026-10-21"
DEFAULT_OUTPUT_DIR = ROOT / "analysis/tri6_pairwise_shadow_ranking_2026-10-21"
RANK_BINS = ["1", "2", "3", "4-5", ">5"]
COMPETITOR_NARROW_MARGIN = 0.02  # Reporting-only threshold; does not affect matching.

CANONICAL_FILENAMES = [
    "pairwise_matches_balanced.csv",
    "pairwise_matches_high.csv",
    "pairwise_matches_graph.csv",
    "tracks_graph.csv",
    "track_edges_graph.csv",
    "cycle_consistency_graph.csv",
    "cycle_edge_checks_graph.csv",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_source_hashes() -> dict[str, str]:
    paths = (
        ROOT / "matching/pairwise_shadow_diagnostics.py",
        ROOT / "tools/audit_tri6_pairwise_shadow_ranking.py",
        ROOT / "tests/test_pairwise_shadow_diagnostics.py",
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _bool(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _row_count_and_hash(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        row_count = sum(1 for _ in handle) - 1
    return {"path": str(path), "row_count": int(row_count), "sha256": _sha256(path)}


def _baseline_matching_hashes(prior_log: dict[str, Any]) -> dict[str, dict[str, Any]]:
    hashes: dict[str, dict[str, Any]] = {}
    for section in ("canonical_hashes_before", "supporting_matching_input_hashes_before"):
        for name, row in prior_log.get(section, {}).items():
            hashes[name] = row
    return hashes


def _check_native_hashes(expected: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for path_text, baseline in expected.items():
        path = Path(path_text)
        actual = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        if actual["bytes"] != int(baseline["bytes"]) or actual["sha256"] != baseline["sha256"]:
            raise AssertionError(f"Native input changed since the previous audit: {path}")
        result[str(path)] = actual
    return result


def _check_matching_invariants(baseline: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for name in CANONICAL_FILENAMES:
        if name not in baseline:
            raise AssertionError(f"Missing pre-audit hash baseline for {name}")
        path = MATCH_DIR / name
        actual = _row_count_and_hash(path)
        expected = baseline[name]
        if actual["sha256"] != expected["sha256"] or actual["row_count"] != int(expected["row_count"]):
            raise AssertionError(f"Canonical matching output changed since the previous audit: {path}")
        result[name] = actual
    candidate_baseline = baseline.get("pairwise_candidates.csv")
    if candidate_baseline:
        candidate_actual = _row_count_and_hash(MATCH_DIR / "pairwise_candidates.csv")
        if candidate_actual["sha256"] != candidate_baseline["sha256"] or candidate_actual["row_count"] != int(candidate_baseline["row_count"]):
            raise AssertionError("Saved pairwise candidate table changed since the previous audit.")
        result["pairwise_candidates.csv"] = candidate_actual
    return result


def _read_pair_rows(table: pd.DataFrame, pair: tuple[str, str]) -> pd.DataFrame:
    return table.loc[(table["day_a"].astype(str) == pair[0]) & (table["day_b"].astype(str) == pair[1])].copy()


def _match_key_set(table: pd.DataFrame) -> set[tuple[int, int]]:
    if table.empty:
        return set()
    return {(int(row.label_a), int(row.label_b)) for row in table.itertuples(index=False)}


def _validate_assignment_replay(candidates: pd.DataFrame, stored: pd.DataFrame, rule: str, pair: tuple[str, str]) -> None:
    replayed = greedy_one_to_one(candidates, rule)
    if _match_key_set(replayed) != _match_key_set(stored):
        raise AssertionError(f"Shadow assignment differs from stored {rule} output for {pair}.")
    if not replayed.empty:
        replay_score = replayed.set_index(["label_a", "label_b"])["score"].sort_index()
        stored_score = stored.set_index(["label_a", "label_b"])["score"].sort_index()
        if not np.allclose(replay_score.to_numpy(), stored_score.to_numpy(), rtol=0, atol=1e-12):
            raise AssertionError(f"Shadow assignment scores differ from stored {rule} output for {pair}.")


def _rank_bin(value: Any) -> str:
    if value is None or pd.isna(value):
        return "MISSING"
    value = int(value)
    if value <= 3:
        return str(value)
    return "4-5" if value <= 5 else ">5"


def _percent(n: int, denominator: int) -> str:
    return f"{n}/{denominator} ({(100.0 * n / denominator) if denominator else 0.0:.1f}%)"


def _md_table(rows: list[list[Any]], headers: list[str]) -> str:
    def cell(value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("|" + "|".join(cell(value) for value in row) + "|" for row in rows)
    return "\n".join(lines)


def _median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return None if values.empty else float(values.median())


def _mean_bool(series: pd.Series) -> float | None:
    values = series.dropna()
    return None if values.empty else float(values.astype(bool).mean())


def _rank_summary(events: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    present = events.loc[events["candidate_present"].astype(bool)].copy()
    rows = []
    for group_key, group in present.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_info = dict(zip(group_columns, group_key, strict=True))
        for rank_type in ("candidate_rank", "reciprocal_candidate_rank"):
            rank_name = "candidate_rank_bin" if rank_type == "candidate_rank" else "reciprocal_rank_bin"
            bins = group.assign(_rank_bin=group[rank_type].map(_rank_bin))
            for rank_bin in RANK_BINS:
                selected = bins.loc[bins["_rank_bin"] == rank_bin]
                row = {
                    **group_info,
                    "rank_type": rank_type,
                    "rank_bin": rank_bin,
                    "N": int(len(selected)),
                    "median_exact_score": _median(selected["raw_candidate_score"]),
                    "median_winning_score": _median(selected["selected_winner_score"]),
                    "n_with_selected_winner": int(selected["selected_winner_score"].notna().sum()),
                    "median_score_margin_to_winner": _median(selected["score_margin_to_winner"]),
                    "median_score_margin_to_second_best": _median(selected["score_margin_to_second_best"]),
                    "median_distance_difference_to_winner_um": _median(selected["distance_margin_to_winner_um"]),
                    "median_dice_difference_to_winner": _median(selected["dice_margin_to_winner"]),
                    "median_overlap_difference_to_winner_iou": _median(selected["overlap_margin_to_winner"]),
                    "balanced_rule_pass_rate": _mean_bool(selected["balanced_rule"]),
                    "high_rule_pass_rate": _mean_bool(selected["high_rule"]),
                    "one_to_one_selection_rate": _mean_bool(selected["one_to_one_selected"]),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _make_case_events(
    cohort: pd.DataFrame,
    diagnostic_rows: dict[tuple[str, str, int, int], dict[str, Any]],
    high_selected: dict[tuple[str, str], set[tuple[int, int]]],
    features_by_session: dict[str, pd.DataFrame],
    transforms_by_pair: dict[tuple[str, str], dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for item in cohort.to_dict(orient="records"):
        for side, source_session_col, source_label_col, target_session_col, target_label_col, stage_col, source_role, target_role in (
            ("AB", "source_session_id", "source_label_A", "missing_intermediate_session_id", "canonical_eligible_label_B", "source_to_missing_stage", "A", "B"),
            ("BC", "missing_intermediate_session_id", "canonical_eligible_label_B", "return_session_id", "return_label_C", "missing_to_return_stage", "B", "C"),
        ):
            source_session = str(item[source_session_col])
            target_session = str(item[target_session_col])
            source_label = int(item[source_label_col])
            target_label = int(item[target_label_col])
            pair = (source_session, target_session)
            exact = diagnostic_rows.get((source_session, target_session, source_label, target_label))
            audit_presence = _bool(item[f"{side}_candidate_exists"])
            if (exact is not None) != audit_presence:
                raise AssertionError(f"Stored candidate presence differs from the fixed audit cohort for {item['endpoint_id']} {side}.")
            source_feature = features_by_session[source_session].loc[source_label]
            target_feature = features_by_session[target_session].loc[target_label]
            transform_record = transforms_by_pair.get(pair, {})
            row = {
                "endpoint_id": item["endpoint_id"],
                "primary_failure_class": item["primary_failure_class"],
                "edge_side": side,
                "source_session_id": source_session,
                "target_session_id": target_session,
                "source_roi": source_label,
                "target_roi": target_label,
                "source_to_missing_stage": item["source_to_missing_stage"],
                "edge_registration_stage": item[stage_col],
                "transform_method": transform_record.get("method"),
                "candidate_present": bool(exact is not None),
                "source_touches_z_edge": _bool(source_feature.get("touches_z_edge")),
                "source_touches_xy_edge": _bool(source_feature.get("touches_xy_edge")),
                "target_touches_z_edge": _bool(target_feature.get("touches_z_edge")),
                "target_touches_xy_edge": _bool(target_feature.get("touches_xy_edge")),
                "source_required_session": _bool(source_feature.get("required")),
                "target_required_session": _bool(target_feature.get("required")),
            }
            if exact is not None:
                row.update({
                    "candidate_source": exact.get("candidate_source"),
                    "raw_candidate_score": float(exact["score"]),
                    "candidate_rank": int(exact["candidate_rank"]),
                    "reciprocal_candidate_rank": int(exact["reciprocal_candidate_rank"]),
                    "candidate_count_for_source": int(exact["candidate_count_for_source"]),
                    "candidate_count_for_target": int(exact["candidate_count_for_target"]),
                    "selected_winner_target": _json_value(exact.get("selected_winner_target")),
                    "selected_winner_score": _json_value(exact.get("selected_winner_score")),
                    "source_best_target": int(exact["source_best_target"]),
                    "source_best_score": float(exact["source_best_score"]),
                    "source_second_best_score": _json_value(exact.get("source_second_best_score")),
                    "score_margin_to_winner": _json_value(exact.get("score_margin_to_winner")),
                    "score_margin_to_second_best": _json_value(exact.get("score_margin_to_source_second_best")),
                    "score_margin_to_source_best": _json_value(exact.get("score_margin_to_source_best")),
                    "distance_um": float(exact["distance_um"]),
                    "distance_margin_to_winner_um": _json_value(exact.get("distance_margin_to_winner_um")),
                    "dice": float(exact["dice"]),
                    "dice_margin_to_winner": _json_value(exact.get("dice_margin_to_winner")),
                    "iou": float(exact["iou"]),
                    "overlap_margin_to_winner": _json_value(exact.get("overlap_margin_to_winner")),
                    "area_ratio": float(exact["area_ratio"]),
                    "ambiguity": float(exact["ambiguity"]),
                    "high_rule": bool(exact["high_rule"]),
                    "balanced_rule": bool(exact["balanced_rule"]),
                    "one_to_one_selected": bool(exact["one_to_one_selected"]),
                    "one_to_one_rejected": bool(exact["one_to_one_rejected"]),
                    "one_to_one_rejection_reason": exact.get("one_to_one_rejection_reason", ""),
                    "one_to_one_assignment_loss": bool(exact["balanced_rule"] and exact["one_to_one_rejected"]),
                    "same_source_competition": bool(exact.get("same_source_competition", False)),
                    "different_source_same_target_competition": bool(exact.get("different_source_same_target_competition", False)),
                    "selected_competing_source": _json_value(exact.get("selected_competing_source")),
                    "selected_competing_target": _json_value(exact.get("selected_competing_target")),
                    "selected_competing_score": _json_value(exact.get("selected_competing_score")),
                    "selected_competing_distance_um": _json_value(exact.get("selected_competing_distance_um")),
                    "selected_competing_dice": _json_value(exact.get("selected_competing_dice")),
                    "selected_competing_iou": _json_value(exact.get("selected_competing_iou")),
                    "selected_competing_candidate_rank": _json_value(exact.get("selected_competing_candidate_rank")),
                    "selected_competing_reciprocal_rank": _json_value(exact.get("selected_competing_reciprocal_rank")),
                    "selected_competing_edges": exact.get("selected_competing_edges", ""),
                    "high_rule_selected": (source_label, target_label) in high_selected.get(pair, set()),
                    "high_rule_assignment_loss": bool(exact["high_rule"] and (source_label, target_label) not in high_selected.get(pair, set())),
                })
                row.update(candidate_rule_diagnostics(exact))
            else:
                row.update({
                    "candidate_source": None,
                    "raw_candidate_score": np.nan,
                    "candidate_rank": np.nan,
                    "reciprocal_candidate_rank": np.nan,
                    "candidate_count_for_source": np.nan,
                    "candidate_count_for_target": np.nan,
                    "selected_winner_target": np.nan,
                    "selected_winner_score": np.nan,
                    "source_best_target": np.nan,
                    "source_best_score": np.nan,
                    "source_second_best_score": np.nan,
                    "score_margin_to_winner": np.nan,
                    "score_margin_to_second_best": np.nan,
                    "score_margin_to_source_best": np.nan,
                    "distance_um": np.nan,
                    "distance_margin_to_winner_um": np.nan,
                    "dice": np.nan,
                    "dice_margin_to_winner": np.nan,
                    "iou": np.nan,
                    "overlap_margin_to_winner": np.nan,
                    "area_ratio": np.nan,
                    "ambiguity": np.nan,
                    "high_rule": np.nan,
                    "balanced_rule": np.nan,
                    "one_to_one_selected": False,
                    "one_to_one_rejected": False,
                    "one_to_one_rejection_reason": "not_in_candidate_pool",
                    "one_to_one_assignment_loss": False,
                    "same_source_competition": False,
                    "different_source_same_target_competition": False,
                    "selected_competing_source": np.nan,
                    "selected_competing_target": np.nan,
                    "selected_competing_score": np.nan,
                    "selected_competing_distance_um": np.nan,
                    "selected_competing_dice": np.nan,
                    "selected_competing_iou": np.nan,
                    "selected_competing_candidate_rank": np.nan,
                    "selected_competing_reciprocal_rank": np.nan,
                    "selected_competing_edges": "",
                    "high_rule_selected": False,
                    "high_rule_assignment_loss": False,
                    "high_rule_only_failure": False,
                    "rule_failure_reason_string": "not_in_candidate_pool",
                    "production_score_gate_present": False,
                    "production_reciprocal_rank_gate_present": False,
                    "production_margin_gate_present": False,
                })
            row["source_role"] = source_role
            row["target_role"] = target_role
            rows.append(row)
    return pd.DataFrame(rows)


def _replay_missing_pairs(
    missing_events: pd.DataFrame,
    features_by_session: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    summaries_by_pair: dict[tuple[str, str], dict[str, Any]],
    transforms_by_pair: dict[tuple[str, str], dict[str, Any]],
    registration_qc_by_pair: dict[tuple[str, str], dict[str, Any]],
    params: AffineOverlapParams,
    spacing: VoxelSpacing,
) -> dict[tuple[str, str, int, int], dict[str, Any]]:
    mask_paths = manifest.set_index("session_id")["mask_path"].astype(str).to_dict()
    masks: dict[str, np.ndarray] = {}
    replayed: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    pair_groups = missing_events.groupby(["source_session_id", "target_session_id"], sort=True)
    for (session_a, session_b), group in pair_groups:
        pair = (str(session_a), str(session_b))
        transform_row = transforms_by_pair[pair]
        summary = summaries_by_pair[pair]
        qc_row = registration_qc_by_pair[pair]
        selected_stage = str(qc_row["selected_registration_stage"])
        transform_path = qc_row.get("transform_path")
        if selected_stage == "legacy":
            transform = restricted_transform(pd.Series(transform_row))
        elif transform_path is not None and not pd.isna(transform_path) and str(transform_path):
            transform = load_image_transform(str(transform_path))
        else:
            raise AssertionError(f"Selected {selected_stage} transform has no saved transform file for {pair}.")
        feature_a = features_by_session[pair[0]]
        feature_b = features_by_session[pair[1]]
        if pair[0] not in masks:
            masks[pair[0]] = tifffile.memmap(mask_paths[pair[0]], mode="r")
        if pair[1] not in masks:
            masks[pair[1]] = tifffile.memmap(mask_paths[pair[1]], mode="r")
        if selected_stage == "legacy":
            mask_b_for_overlap = masks[pair[1]]
            shift = np.asarray([summary["shift_z"], summary["shift_y"], summary["shift_x"]], dtype=float)
            area_b_for_overlap = feature_b["area_voxels"]
        else:
            mask_b_for_overlap = warp_volume(masks[pair[1]], transform, masks[pair[0]].shape, order=0)
            labels_warped, counts_warped = np.unique(mask_b_for_overlap, return_counts=True)
            area_b_for_overlap = pd.Series(counts_warped, index=labels_warped)
            shift = np.zeros(3, dtype=float)
        overlap = build_sparse_overlap_table(
            masks[pair[0]], mask_b_for_overlap, shift,
            feature_a["area_voxels"], area_b_for_overlap,
        )

        labels_a = feature_a.index.to_numpy(dtype=int)
        labels_b = feature_b.index.to_numpy(dtype=int)
        coords_a = feature_a[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
        coords_b_raw = feature_b[["centroid_z", "centroid_y", "centroid_x"]].to_numpy(dtype=float)
        coords_b = transform.apply(coords_b_raw)
        spacing_array = spacing.as_zyx_array()
        physical_a = coords_a * spacing_array
        physical_b = coords_b * spacing_array
        distances_a, indices_a = _nearest_neighbor_data(cKDTree(physical_b), physical_a, k=2)
        _distances_b, indices_b = _nearest_neighbor_data(cKDTree(physical_a), physical_b, k=2)
        label_to_index_a = {int(label): index for index, label in enumerate(labels_a)}
        label_to_index_b = {int(label): index for index, label in enumerate(labels_b)}
        overlap_lookup = {
            (int(row.label_a), int(row.label_b)): row
            for row in overlap.itertuples(index=False)
        }

        for event in group.to_dict(orient="records"):
            label_a, label_b = int(event["source_roi"]), int(event["target_roi"])
            index_a, index_b = label_to_index_a[label_a], label_to_index_b[label_b]
            exact_distance = float(np.linalg.norm(physical_a[index_a] - physical_b[index_b]))
            nearest_b = int(indices_a[index_a, 0])
            nearest_pair = nearest_b == index_b and int(indices_b[index_b, 0]) == index_a
            source_nearest_distance = float(distances_a[index_a, 0])
            overlap_row = overlap_lookup.get((label_a, label_b))
            intersection = 0 if overlap_row is None else int(overlap_row.intersection_voxels)
            dice = 0.0 if overlap_row is None else float(overlap_row.dice)
            iou = 0.0 if overlap_row is None else float(overlap_row.iou)
            area_a = float(feature_a.at[label_a, "area_voxels"])
            area_b = float(feature_b.at[label_b, "area_voxels"])
            source_overlap = intersection / area_a
            target_overlap = intersection / area_b
            area_ratio = min(area_a, area_b) / max(area_a, area_b)
            gates = trace_missing_candidate_gates(
                source_feature_present=True,
                target_feature_present=True,
                transform_available=True,
                source_nearest_distance_um=source_nearest_distance,
                exact_pair_distance_um=exact_distance,
                mutual_nearest_pair=bool(nearest_pair),
                intersection_voxels=intersection,
                dice=dice,
                centroid_candidate_max_distance_um=params.centroid_candidate_max_distance_um,
                overlap_candidate_min_dice=params.overlap_candidate_min_dice,
            )
            if gates["candidate_would_be_generated"]:
                raise AssertionError(f"Missing candidate replay unexpectedly passes a production candidate route: {event['endpoint_id']} {event['edge_side']}")
            replayed[(pair[0], pair[1], label_a, label_b)] = {
                **gates,
                "intersection_voxels": intersection,
                "distance_um": exact_distance,
                "dice": dice,
                "iou": iou,
                "source_overlap": source_overlap,
                "target_overlap": target_overlap,
                "area_ratio": area_ratio,
                "source_nearest_distance_um": source_nearest_distance,
                "exact_pair_distance_um": exact_distance,
                "mutual_nearest_pair": bool(nearest_pair),
                "transform_method": transform.method,
                "transform_fallback_reason": transform.fallback_reason,
                "source_touches_z_edge": _bool(feature_a.at[label_a, "touches_z_edge"]),
                "source_touches_xy_edge": _bool(feature_a.at[label_a, "touches_xy_edge"]),
                "target_touches_z_edge": _bool(feature_b.at[label_b, "touches_z_edge"]),
                "target_touches_xy_edge": _bool(feature_b.at[label_b, "touches_xy_edge"]),
            }
    return replayed


def _make_competition_audit(events: pd.DataFrame) -> pd.DataFrame:
    selected = events.loc[
        (events["primary_failure_class"] == "COMPETING_CANDIDATE_AMBIGUITY")
        & events["candidate_present"].astype(bool)
        & events["balanced_rule"].fillna(False).astype(bool)
        & events["one_to_one_rejected"].astype(bool)
    ].copy()
    rows = []
    for item in selected.to_dict(orient="records"):
        same_source = bool(item["same_source_competition"])
        other_source_same_target = bool(item["different_source_same_target_competition"])
        comp_score = item.get("selected_competing_score")
        exact_score = item.get("raw_candidate_score")
        score_advantage = np.nan if pd.isna(comp_score) or pd.isna(exact_score) else float(comp_score) - float(exact_score)
        if other_source_same_target:
            classification = "GLOBAL_ONE_TO_ONE_CONFLICT"
        elif same_source and np.isfinite(score_advantage) and 0 <= score_advantage <= COMPETITOR_NARROW_MARGIN:
            classification = "NARROW_COMPETITOR_WIN"
        elif same_source and np.isfinite(score_advantage) and score_advantage > COMPETITOR_NARROW_MARGIN:
            classification = "CLEAR_COMPETITOR_WIN"
        else:
            classification = "OTHER"
        row = {
            "endpoint_id": item["endpoint_id"],
            "edge_side": item["edge_side"],
            "source_session_id": item["source_session_id"],
            "target_session_id": item["target_session_id"],
            "source_roi": item["source_roi"],
            "target_roi": item["target_roi"],
            "registration_stage": item["edge_registration_stage"],
            "transform_method": item["transform_method"],
            "source_touches_z_edge": item["source_touches_z_edge"],
            "source_touches_xy_edge": item["source_touches_xy_edge"],
            "target_touches_z_edge": item["target_touches_z_edge"],
            "target_touches_xy_edge": item["target_touches_xy_edge"],
            "exact_score": exact_score,
            "exact_candidate_rank": item["candidate_rank"],
            "exact_reciprocal_rank": item["reciprocal_candidate_rank"],
            "exact_distance_um": item["distance_um"],
            "exact_dice": item["dice"],
            "exact_iou": item["iou"],
            "balanced_rule": item["balanced_rule"],
            "high_rule": item["high_rule"],
            "selected_competing_source": item["selected_competing_source"],
            "selected_competing_target": item["selected_competing_target"],
            "competitor_score": comp_score,
            "competitor_candidate_rank": item["selected_competing_candidate_rank"],
            "competitor_reciprocal_rank": item["selected_competing_reciprocal_rank"],
            "competitor_distance_um": item["selected_competing_distance_um"],
            "competitor_dice": item["selected_competing_dice"],
            "competitor_iou": item["selected_competing_iou"],
            "selected_competing_edges": item["selected_competing_edges"],
            "same_source_different_target": same_source,
            "different_source_same_target": other_source_same_target,
            "global_one_to_one_assignment_deciding": other_source_same_target,
            "competitor_score_advantage": score_advantage,
            "exact_minus_competitor_score": np.nan if not np.isfinite(score_advantage) else -score_advantage,
            "exact_minus_competitor_distance_um": np.nan if pd.isna(item["selected_competing_distance_um"]) else float(item["distance_um"]) - float(item["selected_competing_distance_um"]),
            "exact_minus_competitor_dice": np.nan if pd.isna(item["selected_competing_dice"]) else float(item["dice"]) - float(item["selected_competing_dice"]),
            "exact_minus_competitor_iou": np.nan if pd.isna(item["selected_competing_iou"]) else float(item["iou"]) - float(item["selected_competing_iou"]),
            "classification": classification,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _make_ab_bc_summary(events: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for side in ("AB", "BC"):
        table = events.loc[events["edge_side"] == side]
        present = table.loc[table["candidate_present"].astype(bool)]
        missing_side = missing.loc[missing["edge_side"] == side]
        first_gate_counts = missing_side["first_failed_gate"].value_counts().to_dict()
        overlap_gate_counts = missing_side["overlap_route_failure"].value_counts().to_dict()
        row = {
            "edge_side": side,
            "cohort_n": int(len(table)),
            "candidate_present_n": int(len(present)),
            "candidate_present_rate": float(len(present) / len(table)),
            "candidate_missing_n": int(len(missing_side)),
            "candidate_missing_rate": float(len(missing_side) / len(table)),
            "rank_1_n": int((present["candidate_rank"] == 1).sum()),
            "rank_2_n": int((present["candidate_rank"] == 2).sum()),
            "rank_3_n": int((present["candidate_rank"] == 3).sum()),
            "rank_4_5_n": int(present["candidate_rank"].between(4, 5).sum()),
            "rank_gt_5_n": int((present["candidate_rank"] > 5).sum()),
            "reciprocal_rank_1_n": int((present["reciprocal_candidate_rank"] == 1).sum()),
            "reciprocal_rank_2_n": int((present["reciprocal_candidate_rank"] == 2).sum()),
            "reciprocal_rank_3_n": int((present["reciprocal_candidate_rank"] == 3).sum()),
            "reciprocal_rank_4_5_n": int(present["reciprocal_candidate_rank"].between(4, 5).sum()),
            "reciprocal_rank_gt_5_n": int((present["reciprocal_candidate_rank"] > 5).sum()),
            "median_exact_score": _median(present["raw_candidate_score"]),
            "median_selected_winner_score": _median(present["selected_winner_score"]),
            "median_score_margin_to_winner": _median(present["score_margin_to_winner"]),
            "median_score_margin_to_second_best": _median(present["score_margin_to_second_best"]),
            "median_distance_um": _median(present["distance_um"]),
            "median_dice": _median(present["dice"]),
            "median_iou_overlap": _median(present["iou"]),
            "balanced_rule_pass_n": int(present["balanced_rule"].fillna(False).astype(bool).sum()),
            "balanced_rule_pass_rate": _mean_bool(present["balanced_rule"]),
            "high_rule_pass_n": int(present["high_rule"].fillna(False).astype(bool).sum()),
            "high_rule_pass_rate": _mean_bool(present["high_rule"]),
            "balanced_positive_assignment_loss_n": int(present["one_to_one_assignment_loss"].astype(bool).sum()),
            "balanced_positive_assignment_loss_rate_of_positive": (
                float(present["one_to_one_assignment_loss"].astype(bool).sum() / present["balanced_rule"].fillna(False).astype(bool).sum())
                if present["balanced_rule"].fillna(False).astype(bool).sum() else 0.0
            ),
            "high_positive_assignment_loss_n": int(present["high_rule_assignment_loss"].astype(bool).sum()),
            "one_to_one_selection_rate": _mean_bool(present["one_to_one_selected"]),
            "missing_median_exact_distance_um": _median(missing_side["exact_pair_distance_um"]),
            "missing_median_dice": _median(missing_side["dice"]),
            "missing_median_source_overlap": _median(missing_side["source_overlap"]),
            "missing_median_target_overlap": _median(missing_side["target_overlap"]),
            "missing_first_centroid_gate_counts": ";".join(f"{key}={value}" for key, value in sorted(first_gate_counts.items())),
            "missing_overlap_gate_counts": ";".join(f"{key}={value}" for key, value in sorted(overlap_gate_counts.items())),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _make_stage_summary(events: pd.DataFrame, missing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in ("image_affine_local", "legacy"):
        table = events.loc[events["source_to_missing_stage"] == stage]
        case_n = int(table["endpoint_id"].nunique())
        missing_cases = table.loc[~table["candidate_present"].astype(bool)].groupby("endpoint_id").size()
        both = table.groupby("endpoint_id")["candidate_present"].agg(lambda x: bool(x.all()))
        primary_comp = table.loc[table["primary_failure_class"] == "COMPETING_CANDIDATE_AMBIGUITY", "endpoint_id"].nunique()
        for side in ("AB", "BC"):
            side_table = table.loc[table["edge_side"] == side]
            present = side_table.loc[side_table["candidate_present"].astype(bool)]
            gate = missing.loc[(missing["source_to_missing_stage"] == stage) & (missing["edge_side"] == side)]
            rows.append({
                "source_to_missing_stage": stage,
                "edge_side": side,
                "cohort_cases": case_n,
                "candidate_missing_n": int((~side_table["candidate_present"].astype(bool)).sum()),
                "candidate_missing_rate": float((~side_table["candidate_present"].astype(bool)).mean()),
                "both_candidate_cases_n": int(both.sum()),
                "cases_with_any_candidate_missing_n": int(len(missing_cases)),
                "competition_primary_cases_n": int(primary_comp),
                "candidate_rank_1_n": int((present["candidate_rank"] == 1).sum()),
                "candidate_rank_2_n": int((present["candidate_rank"] == 2).sum()),
                "candidate_rank_3_n": int((present["candidate_rank"] == 3).sum()),
                "candidate_rank_4_5_n": int(present["candidate_rank"].between(4, 5).sum()),
                "candidate_rank_gt_5_n": int((present["candidate_rank"] > 5).sum()),
                "balanced_rule_pass_n": int(present["balanced_rule"].fillna(False).astype(bool).sum()),
                "balanced_rule_pass_rate": _mean_bool(present["balanced_rule"]),
                "high_rule_pass_n": int(present["high_rule"].fillna(False).astype(bool).sum()),
                "high_rule_pass_rate": _mean_bool(present["high_rule"]),
                "balanced_positive_assignment_loss_n": int(present["one_to_one_assignment_loss"].astype(bool).sum()),
                "median_exact_score": _median(present["raw_candidate_score"]),
                "median_score_margin_to_winner": _median(present["score_margin_to_winner"]),
                "median_distance_um": _median(present["distance_um"]),
                "median_dice": _median(present["dice"]),
                "first_centroid_gate_counts": ";".join(f"{k}={v}" for k, v in gate["first_failed_gate"].value_counts().sort_index().items()),
                "overlap_fallback_gate_counts": ";".join(f"{k}={v}" for k, v in gate["overlap_route_failure"].value_counts().sort_index().items()),
            })
    return pd.DataFrame(rows)


def _report_text(
    cohort: pd.DataFrame,
    events: pd.DataFrame,
    both_rank_summary: pd.DataFrame,
    stage_gate_summary: pd.DataFrame,
    ab_bc_summary: pd.DataFrame,
    missing: pd.DataFrame,
    competition: pd.DataFrame,
    matching_invariants: dict[str, dict[str, Any]],
    native_hashes: dict[str, dict[str, Any]],
) -> str:
    n_cases = int(len(cohort))
    both_ids = set(cohort.loc[cohort["AB_candidate_exists"].astype(bool) & cohort["BC_candidate_exists"].astype(bool), "endpoint_id"])
    low = events.loc[(events["primary_failure_class"] == "PAIRWISE_LOW_CONFIDENCE") & events["candidate_present"].astype(bool)]
    stage_counts = cohort["source_to_missing_stage"].value_counts().to_dict()

    rank_rows = []
    for side in ("AB", "BC"):
        for rank_type, field in (("candidate rank", "candidate_rank"), ("reciprocal rank", "reciprocal_candidate_rank")):
            side_rows = events.loc[(events["edge_side"] == side) & events["endpoint_id"].isin(both_ids)]
            for bin_name in RANK_BINS:
                rank_rows.append([side, rank_type, bin_name, int((side_rows[field].map(_rank_bin) == bin_name).sum())])

    low_gate_names = [
        "high_rule_clause_1_dice_pass", "high_rule_clause_1_distance_pass", "high_rule_clause_1_area_ratio_pass",
        "high_rule_clause_2_distance_pass", "high_rule_clause_2_area_ratio_pass", "high_rule_clause_2_ambiguity_pass",
        "balanced_rule_clause_2_dice_pass", "balanced_rule_clause_2_distance_pass", "balanced_rule_clause_2_area_ratio_pass",
        "balanced_rule_clause_3_distance_pass", "balanced_rule_clause_3_area_ratio_pass", "balanced_rule_clause_3_ambiguity_pass",
    ]
    low_gate_rows = []
    denominator = int(len(low))
    for name in low_gate_names:
        failed = int((~low[name].astype(bool)).sum())
        low_gate_rows.append([name.replace("_pass", ""), failed, f"{(100 * failed / denominator) if denominator else 0:.1f}%"])

    competition_counts = competition["classification"].value_counts().to_dict() if not competition.empty else {}
    competition_n = len(competition)
    competition_case_n = int(competition["endpoint_id"].nunique()) if competition_n else 0
    global_n = int(competition["global_one_to_one_assignment_deciding"].astype(bool).sum()) if competition_n else 0
    global_cases = int(competition.loc[competition["global_one_to_one_assignment_deciding"].astype(bool), "endpoint_id"].nunique()) if competition_n else 0
    source_conflict_n = int(competition["same_source_different_target"].astype(bool).sum()) if competition_n else 0
    both_conflict_n = int((competition["same_source_different_target"].astype(bool) & competition["different_source_same_target"].astype(bool)).sum()) if competition_n else 0
    missing_first = missing["first_failed_gate"].value_counts().to_dict()
    missing_overlap = missing["overlap_route_failure"].value_counts().to_dict()
    missing_combos = missing.groupby(["first_failed_gate", "overlap_route_failure"], dropna=False).size().to_dict()
    ab = ab_bc_summary.set_index("edge_side")
    low_failures = int((~low["balanced_rule"].astype(bool)).sum())

    combined_gate_rows = [
        [first_gate, overlap_gate, int(count), _percent(int(count), len(missing))]
        for (first_gate, overlap_gate), count in sorted(missing_combos.items())
    ]
    staged_gate_rows = []
    for (stage, side), subset in missing.groupby(["source_to_missing_stage", "edge_side"], sort=True):
        stage_denominator = int(len(subset))
        for (first_gate, overlap_gate), count in subset.groupby(["first_failed_gate", "overlap_route_failure"], sort=True).size().items():
            staged_gate_rows.append([
                stage, side, first_gate, overlap_gate, int(count),
                _percent(int(count), stage_denominator),
            ])

    detailed_rank_rows = []
    for item in both_rank_summary.to_dict(orient="records"):
        detailed_rank_rows.append([
            item["edge_side"], item["rank_type"], item["rank_bin"], item["N"],
            "" if pd.isna(item["median_exact_score"]) else f"{item['median_exact_score']:.3f}",
            "" if pd.isna(item["median_winning_score"]) else f"{item['median_winning_score']:.3f}",
            item["n_with_selected_winner"],
            "" if pd.isna(item["median_score_margin_to_winner"]) else f"{item['median_score_margin_to_winner']:.3f}",
            "" if pd.isna(item["median_score_margin_to_second_best"]) else f"{item['median_score_margin_to_second_best']:.3f}",
            "" if pd.isna(item["median_distance_difference_to_winner_um"]) else f"{item['median_distance_difference_to_winner_um']:.3f}",
            "" if pd.isna(item["median_dice_difference_to_winner"]) else f"{item['median_dice_difference_to_winner']:.3f}",
            "" if pd.isna(item["median_overlap_difference_to_winner_iou"]) else f"{item['median_overlap_difference_to_winner_iou']:.3f}",
            "" if pd.isna(item["balanced_rule_pass_rate"]) else f"{100*item['balanced_rule_pass_rate']:.1f}%",
            "" if pd.isna(item["high_rule_pass_rate"]) else f"{100*item['high_rule_pass_rate']:.1f}%",
            "" if pd.isna(item["one_to_one_selection_rate"]) else f"{100*item['one_to_one_selection_rate']:.1f}%",
        ])

    stage_rows = []
    for stage in ("image_affine_local", "legacy"):
        sub = stage_gate_summary.loc[stage_gate_summary["source_to_missing_stage"] == stage]
        stage_cases = int(stage_counts.get(stage, 0))
        for side in ("AB", "BC"):
            stage_row = sub.loc[sub["edge_side"] == side].iloc[0]
            existing_candidates = stage_cases - int(stage_row.candidate_missing_n)
            stage_rows.append([
                stage, side,
                f"{int(stage_row.candidate_missing_n)}/{stage_cases} ({100*float(stage_row.candidate_missing_rate):.1f}%)",
                f"{int(stage_row.balanced_rule_pass_n)}/{existing_candidates} ({100*float(stage_row.balanced_rule_pass_rate or 0):.1f}%)",
                f"{int(stage_row.candidate_rank_1_n)}/{existing_candidates}",
                f"{int(stage_row.competition_primary_cases_n)}/{stage_cases}",
            ])
    abbc_rows = []
    for side in ("AB", "BC"):
        row = ab.loc[side]
        abbc_rows.append([
            side,
            _percent(int(row.candidate_present_n), int(row.cohort_n)),
            _percent(int(row.balanced_rule_pass_n), int(row.candidate_present_n)),
            _percent(int(row.high_rule_pass_n), int(row.candidate_present_n)),
            f"{row.median_distance_um:.2f}" if pd.notna(row.median_distance_um) else "",
            f"{row.median_dice:.3f}" if pd.notna(row.median_dice) else "",
            f"{row.median_iou_overlap:.3f}" if pd.notna(row.median_iou_overlap) else "",
            _percent(int(row.balanced_positive_assignment_loss_n), int(row.balanced_rule_pass_n)),
            int(row.candidate_missing_n),
            f"{row.missing_median_exact_distance_um:.2f}" if pd.notna(row.missing_median_exact_distance_um) else "",
            f"{row.missing_median_dice:.4f}" if pd.notna(row.missing_median_dice) else "",
            row.missing_first_centroid_gate_counts,
            row.missing_overlap_gate_counts,
        ])

    invariant_rows = [[name, info["row_count"], info["sha256"], "unchanged"] for name, info in matching_invariants.items()]
    native_count = len(native_hashes)
    native_bytes = sum(info["bytes"] for info in native_hashes.values())

    recommendation = "3. One-to-one assignment experiment"
    recommendation_reason = (
        f"Selected different-source edges occupied the exact target in {global_cases} of the 41 competition cases "
        f"({global_n}/{competition_n} rule-positive edge losses). Keep the experiment shadow-only; the audit does not establish which competing identity is correct."
    )

    return f"""# Tri_6 pairwise shadow ranking and candidate-generation audit

**Audit date:** 2026-10-21\\
**Repository HEAD:** `b28368d`\\
**Selected run:** `{RUN_DIR}`\\
**Cohort:** the existing 900-row canonical-eligible-mask endpoint cohort from `analysis/tri6_tplus2_edge_failure_audit_2026-10-21/`; the cohort was not redefined.\\
**Mode:** state-blind; saved pairwise candidates and assignments were reused. Matching was not rerun. Missing exact pairs were replayed read-only against their stored transforms, global shifts, ROI features, and native masks. No candidate was inserted into a production table.

## A. Existing-candidate cohort

Both exact candidate rows exist for **{len(both_ids)}/900** cases. Candidate ranks use the production order: score descending, Dice descending, distance ascending, then source and target ROI labels ascending. Reciprocal ranks use that same order within each target's incoming candidates.

{_md_table(rank_rows, ["edge", "rank type", "rank bin", "N of 656"])}

Per-bin medians use the exact candidate minus its selected balanced assignment when that source has one. Blank winner metrics mean there was no selected balanced assignment for that source in that bin.

{_md_table(detailed_rank_rows, ["edge", "rank type", "rank bin", "N", "median exact score", "median winning score", "N with selected winner", "median score margin to winner", "median score margin to second", "median distance difference (um)", "median Dice difference", "median IoU difference", "balanced pass", "high pass", "one-to-one selected"])}

The exact A→B candidate is rank 1 in {int(((events.edge_side == 'AB') & events.endpoint_id.isin(both_ids) & (events.candidate_rank == 1)).sum())}/656 cases; the exact B→C candidate is rank 1 in {int(((events.edge_side == 'BC') & events.endpoint_id.isin(both_ids) & (events.candidate_rank == 1)).sum())}/656. A→B has no selected balanced winner for its exact source in this subset; its exact candidates usually rank first but fail the existing rule. For B→C, rank 2 is common, yet only a small fraction pass the existing rule. The CSV tables retain each source/target candidate count, exact rank, selected assignment, and competitor.

## B. Current production rule failures in the 615 low-confidence cases

The rule-component denominator is **{denominator} exact candidate edges** (AB and BC rows for each of the 615 cases). Components are the actual predicates in `_candidate_rules`: high-rule clauses plus the two additional balanced-rule clauses. A failed-predicate count can overlap other failed predicates.

{_md_table(low_gate_rows, ["production predicate", "failed N", "% of 1,230 exact edges"])}

Balanced rule failed on {low_failures}/{denominator} exact edges ({(100*low_failures/max(denominator,1)):.1f}%). The current production rule has no score gate, reciprocal-rank gate, or score-margin gate; these are reported as absent, not as failure causes. Candidate score and ranks affect the greedy ordering only after a candidate passes a rule.

## C. Rule-positive candidates lost at one-to-one assignment

The prior audit's **{competition_case_n} competition cases** produced **{competition_n} rule-positive exact edge rows** in this audit. “Narrow” is report-only: the selected same-source competitor's score advantage is at most {COMPETITOR_NARROW_MARGIN:.2f} score points. The production matcher uses no such threshold.

{_md_table([[name, int(competition_counts.get(name, 0)), int(competition.loc[competition["classification"] == name, "endpoint_id"].nunique()), _percent(int(competition_counts.get(name, 0)), competition_n)] for name in ("NARROW_COMPETITOR_WIN", "CLEAR_COMPETITOR_WIN", "GLOBAL_ONE_TO_ONE_CONFLICT", "OTHER")], ["classification", "edge rows", "distinct cases", "% of rule-positive losses"])}

Selected different-source edges occupying the exact target were the deciding global assignment conflict in **{global_n}/{competition_n}** edge losses across **{global_cases}/{competition_case_n}** cases. Same-source/different-target selected competitors occurred in {source_conflict_n} edge rows; both conflict types applied to {both_conflict_n} rows. The case table identifies all selected blocking edges, source/target ROI, candidate rank, score, geometry margins, registration stage, and FoV-edge flags.

## D. Missing-candidate generation

There are **{len(missing)} missing exact edge rows** across the 244 cases with one or more absent adjacent candidates. The current candidate generator forms a union of (1) mutual nearest-centroid pairs within `centroid_candidate_max_distance_um={10.0:g}` and (2) overlap rows with Dice at least `overlap_candidate_min_dice={0.10:g}`. Legacy transitions use the saved global shift for overlap; image-affine transitions use the saved transformed mask on the source grid. Missing exact pairs must fail both routes.

Centroid route's first failed gate:

{_md_table([[name, int(missing_first.get(name, 0)), _percent(int(missing_first.get(name, 0)), len(missing))] for name in sorted(missing_first)], ["first failed centroid gate", "edge rows", "% of 250 missing edges"])}

Fallback overlap route's failed gate:

{_md_table([[name, int(missing_overlap.get(name, 0)), _percent(int(missing_overlap.get(name, 0)), len(missing))] for name in sorted(missing_overlap)], ["overlap route failure", "edge rows", "% of 250 missing edges"])}

The combined failure of the two candidate routes was:

{_md_table(combined_gate_rows, ["first centroid-route gate", "overlap-route gate", "edge rows", "% of missing edges"])}

Feature rows and pair transforms were present for every replayed pair. The matcher has no per-ROI ownership/exclusion, morphology/size, FoV, or candidate-truncation gate in this candidate-generation function; those labels were not invented. Edge-touch flags are retained as context only. Shadow replay recorded exact-pair distance, Dice, source/target overlap, raw size ratio, transform method, and FoV flags; replay confirmed that none of the missing pairs would enter either production candidate route.

## E. A→B versus B→C

{_md_table(abbc_rows, ["edge", "candidate present", "balanced pass among candidates", "high pass among candidates", "median distance (um)", "median Dice", "median IoU", "balanced-positive loss rate", "missing exact edges", "median missing distance (um)", "median missing Dice", "first centroid gates", "overlap gates"])}

B→C exact candidates are missing in {int(ab.loc['BC','candidate_missing_n'])}/900 cases, compared with {int(ab.loc['AB','candidate_missing_n'])}/900 A→B cases. The 22 A→B absences are 16 AB-only cases plus the 6 cases missing both edges; the 228 B→C absences are 222 BC-only cases plus those same 6 both-missing cases. All 22 missing A→B pairs had some transformed-mask intersection but Dice below 0.10. Of the 228 missing B→C pairs, {int((missing.loc[missing.edge_side == 'BC', 'overlap_route_failure'] == 'no_mask_intersection').sum())} had no intersection and {int((missing.loc[missing.edge_side == 'BC', 'overlap_route_failure'] == 'overlap_candidate_min_dice').sum())} had sub-threshold Dice. Median missing-pair distances were similar ({ab.loc['AB','missing_median_exact_distance_um']:.2f} um AB; {ab.loc['BC','missing_median_exact_distance_um']:.2f} um BC), while median missing Dice was {ab.loc['AB','missing_median_dice']:.4f} AB versus {ab.loc['BC','missing_median_dice']:.4f} BC. Among candidates that did exist, B→C also had lower median Dice and a lower balanced-rule pass rate. These are descriptive differences, not evidence that B itself caused the failures.

## F. Local-selected versus legacy

{_md_table(stage_rows, ["source-to-missing stage", "edge", "candidate missing", "balanced pass among existing rows", "rank 1 among existing", "primary competition cases"])}

The rank-by-stage CSV reports candidate and reciprocal-rank bins, winning-score margins, and rule/assignment rates for the both-candidate subset within each stage. The missing-candidate gate mix is:

{_md_table(staged_gate_rows, ["source-to-missing stage", "edge", "first centroid-route gate", "overlap-route gate", "N", "% within stage/edge missing rows"])}

The same centroid-radius, mutual-nearest, and overlap-Dice gates appear in both stages. These comparisons are descriptive, not causal.

## G. Next implementation experiment

**{recommendation}** — {recommendation_reason} Do not change production matcher settings based on this audit alone.

## Invariants and provenance

- The seven canonical pairwise/graph outputs and `pairwise_candidates.csv` match the hashes and row counts from the prior edge audit.
- All {native_count} native mask and fluorescence assets ({native_bytes:,} bytes total) match their prior SHA-256 and byte counts. Masks were opened read-only for replay. Fluorescence pixels were not loaded or analyzed.
- No molecular-state or fluorescence values were used for ranking, replay, or classification.
- Saved balanced/high assignments were reproduced exactly from the saved candidate rows using the current production greedy matcher.

{_md_table(invariant_rows, ["canonical file", "rows", "SHA-256", "result"])}

## Output files

- `tri6_exact_candidate_shadow_ranks.csv` — 1,800 AB/BC endpoint-edge rows, with rank, reciprocal rank, margins, rule predicates, winner, assignment-loss, and FoV context.
- `tri6_rule_failure_reasons.csv` — exact current-rule booleans and semicolon-delimited reason strings.
- `tri6_competing_assignment_audit.csv` — rule-positive losses from the 41 competition cases.
- `tri6_missing_candidate_gate_audit.csv` — replayed geometry and the actual candidate-union gates for missing exact pairs.
- `tri6_rank_distribution_summary.csv`, `tri6_rank_by_registration_stage.csv`, and `tri6_ab_vs_bc_summary.csv` — aggregate comparisons.
- `run_log.json` — input/output hashes, assignment equivalence, cohort identity, and read-only integrity checks.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite-generated-output", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        marker = output_dir / "run_log.json"
        if not args.overwrite_generated_output or not marker.exists():
            raise FileExistsError(f"Refusing to overwrite existing audit output directory: {output_dir}")
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if (
            previous.get("audit_type") != "fixed-cohort Tri_6 pairwise candidate rank and generation-gate shadow audit"
            or previous.get("selected_run") != str(RUN_DIR)
        ):
            raise FileExistsError(f"Output directory is not a prior run of this audit: {output_dir}")
        shutil.rmtree(output_dir)

    prior_log = json.loads((PRIOR_AUDIT_DIR / "run_log.json").read_text(encoding="utf-8"))
    if int(prior_log.get("canonical_eligible_mask_cohort", -1)) != 900:
        raise AssertionError("The fixed upstream audit does not identify the expected 900-case cohort.")
    input_hashes = _baseline_matching_hashes(prior_log)
    native_baselines = prior_log["native_asset_hashes_before"]
    matching_before = _check_matching_invariants(input_hashes)
    native_before = _check_native_hashes(native_baselines)

    cohort = pd.read_csv(PRIOR_AUDIT_DIR / "tri6_tplus2_eligible_mask_edge_audit.csv")
    if len(cohort) != 900 or cohort["endpoint_id"].duplicated().any():
        raise AssertionError("The fixed endpoint cohort is not the expected unique 900 rows.")
    if cohort["primary_failure_class"].value_counts().to_dict() != prior_log["primary_failure_class_counts"]:
        raise AssertionError("The fixed cohort failure classes differ from the prior audit.")

    run_log = json.loads((MATCH_DIR / "run_log.json").read_text(encoding="utf-8"))
    params = AffineOverlapParams(**run_log["params"])
    spacing = VoxelSpacing(**run_log["spacing"])
    candidates = pd.read_csv(MATCH_DIR / "pairwise_candidates.csv")
    matches_balanced = pd.read_csv(MATCH_DIR / "pairwise_matches_balanced.csv")
    matches_high = pd.read_csv(MATCH_DIR / "pairwise_matches_high.csv")
    transforms = pd.read_csv(MATCH_DIR / "pairwise_transforms.csv")
    summaries = pd.read_csv(MATCH_DIR / "pairwise_summary.csv")
    registration_qc = pd.read_csv(MATCH_DIR / "pairwise_registration_qc.csv")
    manifest = pd.read_csv(MATCH_DIR / "session_manifest_resolved.csv")
    feature_table = pd.read_csv(MATCH_DIR / "roi_features.csv")
    features_by_session = {
        session: rows.sort_values("label").set_index("label", drop=False)
        for session, rows in feature_table.groupby("session_id", sort=False)
    }
    transforms_by_pair = {
        (str(row.day_a), str(row.day_b)): row._asdict()
        for row in transforms.itertuples(index=False)
    }
    summaries_by_pair = {
        (str(row.day_a), str(row.day_b)): row._asdict()
        for row in summaries.itertuples(index=False)
    }
    registration_qc_by_pair = {
        (str(row.day_a), str(row.day_b)): row._asdict()
        for row in registration_qc.itertuples(index=False)
    }
    candidate_groups = {
        (str(pair[0]), str(pair[1])): group.copy()
        for pair, group in candidates.groupby(["day_a", "day_b"], sort=False)
    }
    balanced_groups = {
        (str(pair[0]), str(pair[1])): group.copy()
        for pair, group in matches_balanced.groupby(["day_a", "day_b"], sort=False)
    }
    high_groups = {
        (str(pair[0]), str(pair[1])): group.copy()
        for pair, group in matches_high.groupby(["day_a", "day_b"], sort=False)
    }

    wanted: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for row in cohort.to_dict(orient="records"):
        wanted.setdefault((str(row["source_session_id"]), str(row["missing_intermediate_session_id"])), set()).add((int(row["source_label_A"]), int(row["canonical_eligible_label_B"])))
        wanted.setdefault((str(row["missing_intermediate_session_id"]), str(row["return_session_id"])), set()).add((int(row["canonical_eligible_label_B"]), int(row["return_label_C"])))

    diagnostic_rows: dict[tuple[str, str, int, int], dict[str, Any]] = {}
    high_selected: dict[tuple[str, str], set[tuple[int, int]]] = {}
    assignment_replay_checks = []
    for pair in sorted(wanted):
        pair_candidates = candidate_groups.get(pair, candidates.iloc[0:0].copy())
        pair_balanced = balanced_groups.get(pair, matches_balanced.iloc[0:0].copy())
        pair_high = high_groups.get(pair, matches_high.iloc[0:0].copy())
        _validate_assignment_replay(pair_candidates, pair_balanced, "balanced_rule", pair)
        _validate_assignment_replay(pair_candidates, pair_high, "high_rule", pair)
        high_selected[pair] = _match_key_set(pair_high)
        assignment_replay_checks.append({
            "day_a": pair[0],
            "day_b": pair[1],
            "candidate_rows": int(len(pair_candidates)),
            "stored_balanced_rows": int(len(pair_balanced)),
            "stored_high_rows": int(len(pair_high)),
            "balanced_assignment_identical": True,
            "high_assignment_identical": True,
        })
        if pair_candidates.empty:
            continue
        diagnosed = diagnose_candidate_table(
            pair_candidates,
            pair_balanced,
            rule_column="balanced_rule",
            include_rule_diagnostics=False,
        )
        relevant_keys = wanted[pair]
        exact_rows = diagnosed.loc[
            diagnosed.apply(lambda row: (int(row["label_a"]), int(row["label_b"])) in relevant_keys, axis=1)
        ]
        for item in exact_rows.to_dict(orient="records"):
            diagnostic_rows[(pair[0], pair[1], int(item["label_a"]), int(item["label_b"]))] = item

    events = _make_case_events(cohort, diagnostic_rows, high_selected, features_by_session, transforms_by_pair)
    if len(events) != 1800 or int(events["candidate_present"].sum()) != 1550:
        raise AssertionError("Exact candidate rows do not reproduce 878 A→B and 672 B→C candidates.")
    missing_events = events.loc[~events["candidate_present"].astype(bool)].copy()
    if len(missing_events) != 250 or int(missing_events["endpoint_id"].nunique()) != 244:
        raise AssertionError("Missing exact-candidate cohort differs from the prior 244 cases/250 edges.")

    replayed = _replay_missing_pairs(
        missing_events, features_by_session, manifest, summaries_by_pair,
        transforms_by_pair, registration_qc_by_pair, params, spacing,
    )
    for event_index, item in missing_events.iterrows():
        key = (item["source_session_id"], item["target_session_id"], int(item["source_roi"]), int(item["target_roi"]))
        if key not in replayed:
            raise AssertionError(f"Missing replay result for {key}")
        for name, value in replayed[key].items():
            events.at[event_index, name] = value
    missing = events.loc[~events["candidate_present"].astype(bool)].copy()
    both_ids = set(cohort.loc[cohort["AB_candidate_exists"].astype(bool) & cohort["BC_candidate_exists"].astype(bool), "endpoint_id"])
    events["both_exact_candidates_exist"] = events["endpoint_id"].isin(both_ids)
    rule_failures = events.copy()
    competition = _make_competition_audit(events)
    both_rank_summary = _rank_summary(events.loc[events["both_exact_candidates_exist"]], ["edge_side"])
    stage_summary = _rank_summary(events.loc[events["both_exact_candidates_exist"]], ["edge_side", "source_to_missing_stage"])
    # Add missing-candidate and gate rates to the stage table without mixing them into rank denominators.
    stage_gate = _make_stage_summary(events, missing)
    stage_gate_for_merge = stage_gate.rename(columns={
        "balanced_rule_pass_rate": "overall_balanced_rule_pass_rate",
        "high_rule_pass_rate": "overall_high_rule_pass_rate",
        "median_exact_score": "overall_median_exact_score",
        "median_score_margin_to_winner": "overall_median_score_margin_to_winner",
    })
    rank_by_stage = stage_summary.merge(stage_gate_for_merge, on=["edge_side", "source_to_missing_stage"], how="left", validate="many_to_one")
    ab_bc_summary = _make_ab_bc_summary(events, missing)

    output_dir.mkdir(parents=True, exist_ok=False)
    output_tables = {
        "tri6_exact_candidate_shadow_ranks.csv": events,
        "tri6_rule_failure_reasons.csv": rule_failures,
        "tri6_competing_assignment_audit.csv": competition,
        "tri6_missing_candidate_gate_audit.csv": missing,
        "tri6_rank_distribution_summary.csv": both_rank_summary,
        "tri6_rank_by_registration_stage.csv": rank_by_stage,
        "tri6_ab_vs_bc_summary.csv": ab_bc_summary,
        "tri6_assignment_replay_checks.csv": pd.DataFrame(assignment_replay_checks),
    }
    for name, table in output_tables.items():
        table.to_csv(output_dir / name, index=False)

    matching_after = _check_matching_invariants(input_hashes)
    native_after = _check_native_hashes(native_baselines)
    if matching_before != matching_after or native_before != native_after:
        raise AssertionError("Read-only audit changed a canonical or native input.")

    report = _report_text(cohort, events, both_rank_summary, stage_gate, ab_bc_summary, missing, competition, matching_after, native_after)
    (output_dir / "TRI6_PAIRWISE_SHADOW_RANKING_REPORT.md").write_text(report, encoding="utf-8")
    log = {
        "audit_date": "2026-10-21",
        "audit_type": "fixed-cohort Tri_6 pairwise candidate rank and generation-gate shadow audit",
        "repo_head": "b28368db8c78cafbd6019b55eb1cfbfe574a6a20",
        "selected_run": str(RUN_DIR),
        "cohort_source": str(PRIOR_AUDIT_DIR / "tri6_tplus2_eligible_mask_edge_audit.csv"),
        "cohort_rows": len(cohort),
        "cohort_endpoint_ids_sha256": hashlib.sha256("\n".join(sorted(cohort["endpoint_id"].astype(str))).encode()).hexdigest(),
        "matching_rerun": False,
        "candidate_rows_inserted": 0,
        "state_blind": True,
        "molecular_state_or_fluorescence_used": False,
        "native_mask_access": "read-only tifffile memmap",
        "fluorescence_pixel_access": False,
        "audit_source_sha256": _audit_source_hashes(),
        "production_assignment_replays": assignment_replay_checks,
        "exact_candidate_edge_rows": len(events),
        "exact_candidate_rows_present": int(events["candidate_present"].sum()),
        "missing_candidate_edge_rows": len(missing),
        "missing_candidate_endpoint_cases": int(missing["endpoint_id"].nunique()),
        "rule_positive_competition_edge_rows": len(competition),
        "native_hashes_before": native_before,
        "native_hashes_after": native_after,
        "matching_invariants_before": matching_before,
        "matching_invariants_after": matching_after,
        "output_files": {
            name: {"row_count": len(table), "sha256": _sha256(output_dir / name)}
            for name, table in output_tables.items()
        },
        "report_sha256": _sha256(output_dir / "TRI6_PAIRWISE_SHADOW_RANKING_REPORT.md"),
    }
    (output_dir / "run_log.json").write_text(json.dumps(log, indent=2, default=_json_value) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir}")
    print(f"Exact edge rows: {len(events)}; present: {int(events['candidate_present'].sum())}; missing: {len(missing)}")
    print(f"Competition loss rows: {len(competition)}")


if __name__ == "__main__":
    main()
