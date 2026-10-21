"""Read-only diagnostics for existing affine-overlap pairwise candidates.

This module ranks rows from an already generated candidate table and attributes
assignment losses against the matcher's existing one-to-one output. It never
adds candidates or changes the candidate table.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

try:  # Supports both package and repository-root imports used by existing tools.
    from .affine_overlap_matcher import _candidate_rules, greedy_one_to_one
except ImportError:  # pragma: no cover - exercised by the repository's flat imports
    from affine_overlap_matcher import _candidate_rules, greedy_one_to_one


_ORDER_COLUMNS = ["score", "dice", "distance_um", "label_a", "label_b"]
_ORDER_ASCENDING = [False, False, True, True, True]


def _ordered(table: pd.DataFrame) -> pd.DataFrame:
    """Sort rows with the exact tie-break order used by the production matcher."""

    return table.sort_values(_ORDER_COLUMNS, ascending=_ORDER_ASCENDING, kind="stable")


def candidate_rule_diagnostics(row: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the current compound rule predicates and exact failed components."""

    dice = float(row["dice"])
    distance = float(row["distance_um"])
    area_ratio = float(row["area_ratio"])
    ambiguity = float(row["ambiguity"])

    flags = {
        "high_rule_clause_1_dice_pass": dice >= 0.40,
        "high_rule_clause_1_distance_pass": distance <= 8.0,
        "high_rule_clause_1_area_ratio_pass": area_ratio >= 0.30,
        "high_rule_clause_2_distance_pass": distance <= 3.5,
        "high_rule_clause_2_area_ratio_pass": area_ratio >= 0.55,
        "high_rule_clause_2_ambiguity_pass": ambiguity <= 0.70,
        "balanced_rule_clause_2_dice_pass": dice >= 0.20,
        "balanced_rule_clause_2_distance_pass": distance <= 7.0,
        "balanced_rule_clause_2_area_ratio_pass": area_ratio >= 0.30,
        "balanced_rule_clause_3_distance_pass": distance <= 5.0,
        "balanced_rule_clause_3_area_ratio_pass": area_ratio >= 0.40,
        "balanced_rule_clause_3_ambiguity_pass": ambiguity <= 0.85,
    }
    high_clause_1 = all(flags[name] for name in (
        "high_rule_clause_1_dice_pass",
        "high_rule_clause_1_distance_pass",
        "high_rule_clause_1_area_ratio_pass",
    ))
    high_clause_2 = all(flags[name] for name in (
        "high_rule_clause_2_distance_pass",
        "high_rule_clause_2_area_ratio_pass",
        "high_rule_clause_2_ambiguity_pass",
    ))
    balanced_clause_2 = all(flags[name] for name in (
        "balanced_rule_clause_2_dice_pass",
        "balanced_rule_clause_2_distance_pass",
        "balanced_rule_clause_2_area_ratio_pass",
    ))
    balanced_clause_3 = all(flags[name] for name in (
        "balanced_rule_clause_3_distance_pass",
        "balanced_rule_clause_3_area_ratio_pass",
        "balanced_rule_clause_3_ambiguity_pass",
    ))
    high_rule, balanced_rule = _candidate_rules(dice, distance, area_ratio, ambiguity)
    if high_rule != (high_clause_1 or high_clause_2):
        raise AssertionError("Shadow high-rule predicates diverged from production.")
    if balanced_rule != (high_rule or balanced_clause_2 or balanced_clause_3):
        raise AssertionError("Shadow balanced-rule predicates diverged from production.")

    expected_high = bool(row["high_rule"])
    expected_balanced = bool(row["balanced_rule"])
    if high_rule != expected_high or balanced_rule != expected_balanced:
        raise AssertionError("Candidate rule fields do not match current production rules.")

    reasons: list[str] = []
    if not high_rule:
        reasons.append("high_rule_failed")
        for name, passed in flags.items():
            if name.startswith("high_rule_") and not passed:
                reasons.append(f"{name}_false")
    if not balanced_rule:
        reasons.append("balanced_rule_failed")
        for name, passed in flags.items():
            if name.startswith("balanced_rule_") and not passed:
                reasons.append(f"{name}_false")
    high_rule_only = not high_rule and balanced_rule
    if high_rule_only:
        reasons.append("high_rule_only_failure")

    return {
        **flags,
        "high_rule_recomputed": bool(high_rule),
        "balanced_rule_recomputed": bool(balanced_rule),
        "high_rule_only_failure": bool(high_rule_only),
        "rule_failure_reason_string": ";".join(dict.fromkeys(reasons)) or "PASS",
        # These are explicitly absent from the current production rules.
        "production_score_gate_present": False,
        "production_reciprocal_rank_gate_present": False,
        "production_margin_gate_present": False,
    }


def diagnose_candidate_table(
    candidate_table: pd.DataFrame,
    selected_matches: pd.DataFrame | None = None,
    *,
    rule_column: str = "balanced_rule",
    include_rule_diagnostics: bool = True,
) -> pd.DataFrame:
    """Return a diagnostic copy of one session-pair candidate table.

    Ranks use the production score order: score descending, Dice descending,
    distance ascending, then source and target labels ascending. Reciprocal rank
    uses that same order within each target's incoming candidate set.

    ``selected_matches`` should be the current production one-to-one output for
    ``rule_column``. When omitted, it is replayed with the production matcher.
    """

    required = {
        "label_a", "label_b", "score", "dice", "iou", "distance_um",
        "ambiguity", "area_ratio", "high_rule", "balanced_rule", rule_column,
    }
    missing = sorted(required.difference(candidate_table.columns))
    if missing:
        raise ValueError(f"candidate_table is missing required columns: {missing}")
    if candidate_table.empty:
        return candidate_table.copy()

    original = candidate_table.copy(deep=True).reset_index(drop=True)
    original["_shadow_row_id"] = np.arange(len(original), dtype=int)
    ordered = _ordered(original)
    ordered["candidate_rank"] = ordered.groupby("label_a", sort=False).cumcount() + 1
    ordered["reciprocal_candidate_rank"] = ordered.groupby("label_b", sort=False).cumcount() + 1
    ordered["candidate_count_for_source"] = ordered.groupby("label_a", sort=False)["label_b"].transform("size")
    ordered["candidate_count_for_target"] = ordered.groupby("label_b", sort=False)["label_a"].transform("size")

    second = (
        ordered.loc[ordered["candidate_rank"] == 2, ["label_a", "score"]]
        .rename(columns={"score": "source_second_best_score"})
    )
    source_best = ordered.drop_duplicates("label_a", keep="first")[[
        "label_a", "label_b", "score", "distance_um", "dice", "iou"
    ]].rename(columns={
        "label_b": "source_best_target",
        "score": "source_best_score",
        "distance_um": "source_best_distance_um",
        "dice": "source_best_dice",
        "iou": "source_best_iou",
    })
    ordered = ordered.merge(second, on="label_a", how="left", validate="many_to_one")
    ordered = ordered.merge(source_best, on="label_a", how="left", validate="many_to_one")
    ordered["score_margin_to_source_second_best"] = ordered["score"] - ordered["source_second_best_score"]
    ordered["score_margin_to_source_best"] = ordered["score"] - ordered["source_best_score"]
    ordered["distance_margin_to_source_best"] = ordered["distance_um"] - ordered["source_best_distance_um"]
    ordered["dice_margin_to_source_best"] = ordered["dice"] - ordered["source_best_dice"]
    ordered["overlap_margin_to_source_best"] = ordered["iou"] - ordered["source_best_iou"]

    if selected_matches is None:
        selected = greedy_one_to_one(candidate_table, rule_column)
    else:
        selected = selected_matches.copy(deep=True)
    if selected.duplicated("label_a").any() or selected.duplicated("label_b").any():
        raise ValueError("selected_matches must be a one-to-one assignment.")
    selected = _ordered(selected).reset_index(drop=True)
    selected_by_source = selected[["label_a", "label_b", "score", "distance_um", "dice", "iou"]].rename(columns={
        "label_b": "selected_winner_target",
        "score": "selected_winner_score",
        "distance_um": "selected_winner_distance_um",
        "dice": "selected_winner_dice",
        "iou": "selected_winner_iou",
    })
    ordered = ordered.merge(selected_by_source, on="label_a", how="left", validate="many_to_one")
    ordered["score_margin_to_winner"] = ordered["score"] - ordered["selected_winner_score"]
    ordered["distance_margin_to_winner_um"] = ordered["distance_um"] - ordered["selected_winner_distance_um"]
    ordered["dice_margin_to_winner"] = ordered["dice"] - ordered["selected_winner_dice"]
    ordered["overlap_margin_to_winner"] = ordered["iou"] - ordered["selected_winner_iou"]

    selected_keys = {(int(row.label_a), int(row.label_b)) for row in selected.itertuples(index=False)}
    ordered["one_to_one_selected"] = [
        (int(row.label_a), int(row.label_b)) in selected_keys for row in ordered.itertuples(index=False)
    ]
    ordered["one_to_one_rejected"] = ordered[rule_column].astype(bool) & ~ordered["one_to_one_selected"]
    ordered["one_to_one_rejection_reason"] = ""
    ordered["selected_competing_source"] = np.nan
    ordered["selected_competing_target"] = np.nan
    ordered["selected_competing_score"] = np.nan
    ordered["selected_competing_distance_um"] = np.nan
    ordered["selected_competing_dice"] = np.nan
    ordered["selected_competing_iou"] = np.nan
    ordered["selected_competing_candidate_rank"] = np.nan
    ordered["selected_competing_reciprocal_rank"] = np.nan
    ordered["same_source_competition"] = False
    ordered["different_source_same_target_competition"] = False
    ordered["selected_competing_edges"] = ""

    by_key = ordered.set_index(["label_a", "label_b"], drop=False)
    for row_index in ordered.index[ordered["one_to_one_rejected"]]:
        row = ordered.loc[row_index]
        conflict = selected.loc[
            (selected["label_a"].astype(int) == int(row["label_a"]))
            | (selected["label_b"].astype(int) == int(row["label_b"]))
        ]
        if conflict.empty:
            ordered.at[row_index, "one_to_one_rejection_reason"] = "unattributed_assignment_loss"
            continue
        same_source_conflict = bool((conflict["label_a"].astype(int) == int(row["label_a"])).any())
        same_target_conflict = bool((conflict["label_b"].astype(int) == int(row["label_b"])).any())
        ordered.at[row_index, "same_source_competition"] = same_source_conflict
        ordered.at[row_index, "different_source_same_target_competition"] = same_target_conflict
        ordered.at[row_index, "selected_competing_edges"] = ";".join(
            f"{int(comp.label_a)}:{int(comp.label_b)}" for comp in conflict.itertuples(index=False)
        )
        competitor = conflict.iloc[0]
        same_source = int(competitor["label_a"]) == int(row["label_a"])
        same_target = int(competitor["label_b"]) == int(row["label_b"])
        if same_source_conflict and same_target_conflict:
            reason = "same_source_and_target_conflicts"
        elif same_source:
            reason = "same_source_different_target"
        elif same_target:
            reason = "different_source_same_target"
        else:  # pragma: no cover - the conflict filter makes this unreachable
            reason = "unattributed_assignment_loss"
        ordered.at[row_index, "one_to_one_rejection_reason"] = reason
        ordered.at[row_index, "selected_competing_source"] = int(competitor["label_a"])
        ordered.at[row_index, "selected_competing_target"] = int(competitor["label_b"])
        ordered.at[row_index, "selected_competing_score"] = float(competitor["score"])
        ordered.at[row_index, "selected_competing_distance_um"] = float(competitor["distance_um"])
        ordered.at[row_index, "selected_competing_dice"] = float(competitor["dice"])
        ordered.at[row_index, "selected_competing_iou"] = float(competitor["iou"])
        competitor_key = (int(competitor["label_a"]), int(competitor["label_b"]))
        ordered.at[row_index, "selected_competing_candidate_rank"] = int(by_key.loc[competitor_key, "candidate_rank"])
        ordered.at[row_index, "selected_competing_reciprocal_rank"] = int(by_key.loc[competitor_key, "reciprocal_candidate_rank"])

    if include_rule_diagnostics:
        rule_rows = pd.DataFrame([candidate_rule_diagnostics(row) for row in ordered.to_dict(orient="records")], index=ordered.index)
        ordered = pd.concat([ordered, rule_rows], axis=1)
    ordered = ordered.drop(columns=["_shadow_row_id"])
    return ordered.reset_index(drop=True)


def trace_missing_candidate_gates(
    *,
    source_feature_present: bool,
    target_feature_present: bool,
    transform_available: bool,
    source_nearest_distance_um: float | None,
    exact_pair_distance_um: float | None,
    mutual_nearest_pair: bool | None,
    intersection_voxels: int | None,
    dice: float | None,
    centroid_candidate_max_distance_um: float = 10.0,
    overlap_candidate_min_dice: float = 0.10,
) -> dict[str, Any]:
    """Trace the actual centroid and overlap routes that form the candidate union."""

    if not source_feature_present:
        return {
            "first_failed_gate": "source_roi_feature_missing",
            "centroid_route_accepted": False,
            "overlap_route_accepted": False,
            "candidate_would_be_generated": False,
            "overlap_route_failure": "not_reached",
        }
    if not target_feature_present:
        return {
            "first_failed_gate": "target_roi_feature_missing",
            "centroid_route_accepted": False,
            "overlap_route_accepted": False,
            "candidate_would_be_generated": False,
            "overlap_route_failure": "not_reached",
        }
    if not transform_available:
        return {
            "first_failed_gate": "pair_transform_unavailable",
            "centroid_route_accepted": False,
            "overlap_route_accepted": False,
            "candidate_would_be_generated": False,
            "overlap_route_failure": "not_reached",
        }
    if source_nearest_distance_um is None or not np.isfinite(float(source_nearest_distance_um)):
        centroid_failure = "centroid_distance_unavailable"
        centroid_accepted = False
    elif float(source_nearest_distance_um) > float(centroid_candidate_max_distance_um):
        centroid_failure = "centroid_candidate_max_distance_um"
        centroid_accepted = False
    elif mutual_nearest_pair is None or not bool(mutual_nearest_pair):
        centroid_failure = "mutual_nearest_centroid_pair"
        centroid_accepted = False
    else:
        centroid_failure = ""
        centroid_accepted = True

    if intersection_voxels is None:
        overlap_failure = "overlap_measure_unavailable"
        overlap_accepted = False
    elif int(intersection_voxels) <= 0:
        overlap_failure = "no_mask_intersection"
        overlap_accepted = False
    elif dice is None or not np.isfinite(float(dice)):
        overlap_failure = "overlap_dice_unavailable"
        overlap_accepted = False
    elif float(dice) < float(overlap_candidate_min_dice):
        overlap_failure = "overlap_candidate_min_dice"
        overlap_accepted = False
    else:
        overlap_failure = ""
        overlap_accepted = True

    would_generate = centroid_accepted or overlap_accepted
    first_failed = centroid_failure if not centroid_accepted else ""
    if not first_failed and not overlap_accepted:
        first_failed = overlap_failure
    return {
        "first_failed_gate": first_failed,
        "source_nearest_distance_um": source_nearest_distance_um,
        "exact_pair_distance_um": exact_pair_distance_um,
        "centroid_route_failure": centroid_failure,
        "overlap_route_failure": overlap_failure,
        "centroid_route_accepted": bool(centroid_accepted),
        "overlap_route_accepted": bool(overlap_accepted),
        "candidate_would_be_generated": bool(would_generate),
    }
