"""Build deterministic longitudinal ROI track graphs from pairwise matches."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import math
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrackGraphResult:
    """Bundle track, edge, and cycle artifacts for one policy."""

    tracks: pd.DataFrame
    edge_decisions: pd.DataFrame
    cycle_summary: pd.DataFrame
    cycle_edge_checks: pd.DataFrame


def _component_signature(nodes: Iterable[tuple[str, int]], session_order: dict[str, int]) -> str:
    """Return a deterministic component signature string."""

    ordered = sorted(nodes, key=lambda node: (session_order[node[0]], int(node[1]), node[0]))
    return "|".join(f"{session_id}:{int(label)}" for session_id, label in ordered)


def _track_uid(nodes: Iterable[tuple[str, int]], session_order: dict[str, int]) -> str:
    """Return a stable track identifier derived from the earliest node."""

    ordered = sorted(nodes, key=lambda node: (session_order[node[0]], int(node[1]), node[0]))
    session_id, label = ordered[0]
    return f"{session_id}:{int(label)}"


def _resolve_node_owner(tracks: list[dict[str, object]], day_names: list[str]) -> dict[tuple[str, int], int]:
    """Map every occupied node to its track index."""

    node_to_track: dict[tuple[str, int], int] = {}
    for track_index, track in enumerate(tracks):
        for day_name in day_names:
            value = track.get(f"{day_name}_roi", pd.NA)
            if pd.notna(value):
                node_to_track[(day_name, int(value))] = track_index
    return node_to_track


def _occupied_day_indices(track: dict[str, object], day_names: list[str]) -> list[int]:
    """Return the occupied day indices for one mutable track."""

    return [
        day_index
        for day_index, day_name in enumerate(day_names)
        if pd.notna(track.get(f"{day_name}_roi", pd.NA))
    ]


def _summarize_track(track: dict[str, object], day_names: list[str]) -> None:
    """Refresh derived track fields in place."""

    occupied = _occupied_day_indices(track, day_names)
    track["n_days_present"] = int(len(occupied))
    if occupied:
        first_index = occupied[0]
        last_index = occupied[-1]
        track["first_session_index"] = int(first_index)
        track["last_session_index"] = int(last_index)
        missing_internal = 0
        for day_index in range(first_index, last_index + 1):
            if pd.isna(track.get(f"{day_names[day_index]}_roi", pd.NA)):
                missing_internal += 1
        track["missing_internal_days"] = int(missing_internal)
    else:
        track["first_session_index"] = -1
        track["last_session_index"] = -1
        track["missing_internal_days"] = 0

    edge_rows = list(track.get("_edges", []))
    confidences = [float(row.get("score", np.nan)) for row in edge_rows if pd.notna(row.get("score", np.nan))]
    dices = [float(row.get("dice", np.nan)) for row in edge_rows if pd.notna(row.get("dice", np.nan))]
    distances = [float(row.get("distance_um", np.nan)) for row in edge_rows if pd.notna(row.get("distance_um", np.nan))]
    ambiguities = [float(row.get("ambiguity", np.nan)) for row in edge_rows if pd.notna(row.get("ambiguity", np.nan))]
    track["n_adjacent_edges"] = int(sum(int(row.get("pair_gap", 1)) == 1 for row in edge_rows))
    track["n_gap_edges"] = int(sum(int(row.get("pair_gap", 1)) == 2 for row in edge_rows))
    track["used_gap_bridge"] = bool(track["n_gap_edges"] > 0)
    track["min_score"] = float(np.min(confidences)) if confidences else np.nan
    track["mean_score"] = float(np.mean(confidences)) if confidences else np.nan
    track["min_dice"] = float(np.min(dices)) if dices else np.nan
    track["max_distance_um"] = float(np.max(distances)) if distances else np.nan
    track["max_ambiguity"] = float(np.max(ambiguities)) if ambiguities else np.nan
    track["contains_balanced_only_edge"] = bool(
        any((not bool(row.get("high_rule", False))) and bool(row.get("balanced_rule", False)) for row in edge_rows)
    )
    track["contains_transform_fallback_edge"] = bool(
        any(pd.notna(row.get("transform_fallback_reason", pd.NA)) and str(row.get("transform_fallback_reason")).strip() != "" for row in edge_rows)
    )
    track["review_required"] = bool(
        track["used_gap_bridge"] or track["contains_transform_fallback_edge"]
    )
    review_reasons: list[str] = []
    if track["used_gap_bridge"]:
        review_reasons.append("gap_bridge")
    if track["contains_transform_fallback_edge"]:
        review_reasons.append("transform_fallback")
    track["review_reasons"] = ",".join(review_reasons)


def _build_initial_tracks(day_names: list[str], features_by_session: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    """Create singleton track records for every ROI label."""

    tracks: list[dict[str, object]] = []
    for session_id in day_names:
        features = features_by_session[session_id]
        for label in features.index.to_numpy(dtype=int):
            row: dict[str, object] = {f"{day_name}_roi": pd.NA for day_name in day_names}
            row[f"{session_id}_roi"] = int(label)
            row["_edges"] = []
            row["match_policy"] = ""
            row["review_required"] = False
            row["review_reasons"] = ""
            _summarize_track(row, day_names)
            tracks.append(row)
    return tracks


def _node_session(label_row: pd.Series | dict[str, object]) -> tuple[str, int]:
    """Return node identity from a pairwise edge row."""

    return str(label_row["day"]), int(label_row["label"])


def _node_component(parent: dict[tuple[str, int], tuple[str, int]], node: tuple[str, int]) -> tuple[str, int]:
    """Find the root of a node in the union-find structure."""

    parent.setdefault(node, node)
    if parent[node] != node:
        parent[node] = _node_component(parent, parent[node])
    return parent[node]


def _merge_components(
    parent: dict[tuple[str, int], tuple[str, int]],
    component_nodes: dict[tuple[str, int], set[tuple[str, int]]],
    component_sessions: dict[tuple[str, int], set[str]],
    component_edges: dict[tuple[str, int], list[dict[str, object]]],
    node_a: tuple[str, int],
    node_b: tuple[str, int],
) -> tuple[str, int]:
    """Merge two valid components and return the new root."""

    root_a = _node_component(parent, node_a)
    root_b = _node_component(parent, node_b)
    if root_a == root_b:
        return root_a
    if len(component_nodes[root_a]) < len(component_nodes[root_b]):
        root_a, root_b = root_b, root_a
    parent[root_b] = root_a
    component_nodes[root_a].update(component_nodes.pop(root_b))
    component_sessions[root_a].update(component_sessions.pop(root_b))
    component_edges[root_a].extend(component_edges.pop(root_b))
    return root_a


def _accept_edge(
    parent: dict[tuple[str, int], tuple[str, int]],
    component_nodes: dict[tuple[str, int], set[tuple[str, int]]],
    component_sessions: dict[tuple[str, int], set[str]],
    component_edges: dict[tuple[str, int], list[dict[str, object]]],
    edge_row: dict[str, object],
    node_a: tuple[str, int],
    node_b: tuple[str, int],
) -> None:
    """Merge two components and attach one edge row to the resulting track."""

    root = _merge_components(parent, component_nodes, component_sessions, component_edges, node_a, node_b)
    component_edges[root].append(edge_row)


def _build_edge_decision(
    *,
    match_policy: str,
    edge_type: str,
    pair_gap: int,
    day_a: str,
    day_b: str,
    label_a: int,
    label_b: int,
    score_row: pd.Series,
    accepted: bool,
    rejection_reason: str | None,
) -> dict[str, object]:
    """Create one edge-decision row."""

    return {
        "match_policy": match_policy,
        "edge_type": edge_type,
        "pair_gap": int(pair_gap),
        "day_a": str(day_a),
        "day_b": str(day_b),
        "label_a": int(label_a),
        "label_b": int(label_b),
        "score": float(score_row.get("score", np.nan)),
        "dice": float(score_row.get("dice", np.nan)),
        "distance_um": float(score_row.get("distance_um", np.nan)),
        "ambiguity": float(score_row.get("ambiguity", np.nan)),
        "high_rule": bool(score_row.get("high_rule", False)),
        "balanced_rule": bool(score_row.get("balanced_rule", False)),
        "candidate_source": str(score_row.get("candidate_source", "")),
        "assignment_policy": str(score_row.get("assignment_policy", match_policy)),
        "transform_method": str(score_row.get("transform_method", "")),
        "transform_fallback_reason": score_row.get("transform_fallback_reason", None),
        "accepted_for_track": bool(accepted),
        "rejection_reason": rejection_reason,
    }


def build_tracks_from_pair_tables(
    day_names: list[str],
    features_by_session: dict[str, pd.DataFrame],
    pair_tables: dict[tuple[str, str], pd.DataFrame],
    match_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build tracks and edge-decision tables for one policy."""

    session_order = {session_id: index for index, session_id in enumerate(day_names)}
    parent: dict[tuple[str, int], tuple[str, int]] = {}
    component_nodes: dict[tuple[str, int], set[tuple[str, int]]] = {}
    component_sessions: dict[tuple[str, int], set[str]] = {}
    component_edges: dict[tuple[str, int], list[dict[str, object]]] = {}

    for session_id in day_names:
        features = features_by_session[session_id]
        for label in features.index.to_numpy(dtype=int):
            node = (session_id, int(label))
            parent[node] = node
            component_nodes[node] = {node}
            component_sessions[node] = {session_id}
            component_edges[node] = []

    edge_rows: list[dict[str, object]] = []
    day_pair_order = [
        (day_names[index], day_names[index + 1], 1)
        for index in range(len(day_names) - 1)
    ]
    gap_pair_order = [
        (day_names[index], day_names[index + 2], 2)
        for index in range(len(day_names) - 2)
    ]

    def process_pair(day_a: str, day_b: str, pair_gap: int, sort_gap_edges: bool = False) -> None:
        pair_table = pair_tables.get((day_a, day_b))
        if pair_table is None or pair_table.empty:
            return
        table = pair_table.copy()
        if sort_gap_edges:
            table = table.sort_values(
                ["score", "dice", "distance_um", "label_a", "label_b"],
                ascending=[False, False, True, True, True],
            ).reset_index(drop=True)
        for _, row in table.iterrows():
            node_a = (day_a, int(row["label_a"]))
            node_b = (day_b, int(row["label_b"]))
            edge_row = _build_edge_decision(
                match_policy=match_policy,
                edge_type="gap" if pair_gap == 2 else "adjacent",
                pair_gap=pair_gap,
                day_a=day_a,
                day_b=day_b,
                label_a=int(row["label_a"]),
                label_b=int(row["label_b"]),
                score_row=row,
                accepted=False,
                rejection_reason=None,
            )
            if node_a not in parent or node_b not in parent:
                edge_row["rejection_reason"] = "missing_node"
                edge_rows.append(edge_row)
                continue
            root_a = _node_component(parent, node_a)
            root_b = _node_component(parent, node_b)
            if root_a == root_b:
                edge_row["accepted_for_track"] = False
                edge_row["rejection_reason"] = "already_connected"
                edge_rows.append(edge_row)
                continue
            sessions_a = component_sessions[root_a]
            sessions_b = component_sessions[root_b]
            if sessions_a & sessions_b:
                edge_row["accepted_for_track"] = False
                edge_row["rejection_reason"] = "same_session_component_conflict"
                edge_rows.append(edge_row)
                continue
            _accept_edge(parent, component_nodes, component_sessions, component_edges, edge_row, node_a, node_b)
            edge_row["accepted_for_track"] = True
            edge_row["rejection_reason"] = None
            edge_rows.append(edge_row)

    for day_a, day_b, pair_gap in day_pair_order:
        process_pair(day_a, day_b, pair_gap, sort_gap_edges=False)
    for day_a, day_b, pair_gap in gap_pair_order:
        process_pair(day_a, day_b, pair_gap, sort_gap_edges=True)

    tracks: list[dict[str, object]] = []
    for root, nodes in component_nodes.items():
        row: dict[str, object] = {f"{day_name}_roi": pd.NA for day_name in day_names}
        for session_id, label in nodes:
            row[f"{session_id}_roi"] = int(label)
        row["match_policy"] = match_policy
        row["track_uid"] = _track_uid(nodes, session_order)
        row["component_signature"] = _component_signature(nodes, session_order)
        row["_edges"] = list(component_edges[root])
        _summarize_track(row, day_names)
        tracks.append(row)

    if tracks:
        tracks_table = pd.DataFrame(tracks)
    else:
        tracks_table = pd.DataFrame(
            columns=[
                *[f"{day_name}_roi" for day_name in day_names],
                "match_policy",
                "track_uid",
                "component_signature",
                "n_days_present",
                "first_session_index",
                "last_session_index",
                "missing_internal_days",
                "n_adjacent_edges",
                "n_gap_edges",
                "used_gap_bridge",
                "min_score",
                "mean_score",
                "min_dice",
                "max_distance_um",
                "max_ambiguity",
                "contains_balanced_only_edge",
                "contains_transform_fallback_edge",
                "review_required",
                "review_reasons",
            ]
        )

    for session_id in day_names:
        tracks_table[f"{session_id}_roi"] = tracks_table[f"{session_id}_roi"].astype("Int64")

    tracks_table["first_session"] = tracks_table.apply(
        lambda row: day_names[int(row["first_session_index"])] if int(row["first_session_index"]) >= 0 else pd.NA,
        axis=1,
    )
    tracks_table["last_session"] = tracks_table.apply(
        lambda row: day_names[int(row["last_session_index"])] if int(row["last_session_index"]) >= 0 else pd.NA,
        axis=1,
    )
    tracks_table["n_days_present"] = tracks_table["n_days_present"].astype(int)
    tracks_table["missing_internal_days"] = tracks_table["missing_internal_days"].astype(int)

    tracks_table["cluster_sort_signature"] = tracks_table["component_signature"].astype(str)
    tracks_table = tracks_table.sort_values(
        ["n_days_present", "missing_internal_days", "first_session_index", "track_uid", "cluster_sort_signature"],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)
    tracks_table.insert(0, "cluster_id", np.arange(1, len(tracks_table) + 1))
    return tracks_table, pd.DataFrame(edge_rows)


def build_cycle_consistency_tables(
    day_names: list[str],
    pair_tables: dict[tuple[str, str], pd.DataFrame],
    tracks_table: pd.DataFrame,
    match_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build aggregate and row-level cycle-consistency tables."""

    node_to_track_uid: dict[tuple[str, int], str] = {}
    for _, row in tracks_table.iterrows():
        track_uid = str(row["track_uid"])
        for day_name in day_names:
            value = row.get(f"{day_name}_roi", pd.NA)
            if pd.notna(value):
                node_to_track_uid[(day_name, int(value))] = track_uid

    summary_rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    for index in range(len(day_names) - 2):
        day_a = day_names[index]
        day_b = day_names[index + 1]
        day_c = day_names[index + 2]
        ab = pair_tables.get((day_a, day_b))
        bc = pair_tables.get((day_b, day_c))
        ac = pair_tables.get((day_a, day_c))
        if ab is None or bc is None or ac is None or ab.empty or bc.empty or ac.empty:
            summary_rows.append(
                {
                    "match_policy": match_policy,
                    "day_a": day_a,
                    "day_b": day_b,
                    "day_c": day_c,
                    "n_composed": 0,
                    "n_comparable": 0,
                    "n_agree": 0,
                    "agreement": np.nan,
                }
            )
            continue

        map_ab = {int(row.label_a): int(row.label_b) for row in ab.itertuples(index=False)}
        map_bc = {int(row.label_a): int(row.label_b) for row in bc.itertuples(index=False)}
        map_ac = {int(row.label_a): int(row.label_b) for row in ac.itertuples(index=False)}
        composed = {label_a: map_bc[label_b] for label_a, label_b in map_ab.items() if label_b in map_bc}
        comparable = {label_a: label_c for label_a, label_c in composed.items() if label_a in map_ac}
        agree = sum(int(map_ac[label_a]) == int(label_c) for label_a, label_c in comparable.items())
        summary_rows.append(
            {
                "match_policy": match_policy,
                "day_a": day_a,
                "day_b": day_b,
                "day_c": day_c,
                "n_composed": int(len(composed)),
                "n_comparable": int(len(comparable)),
                "n_agree": int(agree),
                "agreement": float(agree / len(comparable)) if comparable else np.nan,
            }
        )
        for label_a, label_c_composed in comparable.items():
            detail_rows.append(
                {
                    "match_policy": match_policy,
                    "day_a": day_a,
                    "day_b": day_b,
                    "day_c": day_c,
                    "label_a": int(label_a),
                    "label_b_composed": int(map_ab[label_a]),
                    "label_c_composed": int(label_c_composed),
                    "label_c_direct": int(map_ac[label_a]),
                    "cycle_agrees": bool(map_ac[label_a] == label_c_composed),
                    "track_uid": node_to_track_uid.get((day_a, int(label_a)), pd.NA),
                }
            )

    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def summarize_track_cycle_metadata(
    tracks_table: pd.DataFrame,
    cycle_edge_checks: pd.DataFrame,
) -> pd.DataFrame:
    """Attach track-level cycle summaries to the track table."""

    table = tracks_table.copy()
    if cycle_edge_checks is None or cycle_edge_checks.empty:
        table["n_cycle_comparable"] = 0
        table["n_cycle_agree"] = 0
        table["cycle_agreement_fraction"] = np.nan
        table["has_cycle_conflict"] = False
        table["cycle_unchecked"] = True
        return table

    grouped = cycle_edge_checks.groupby("track_uid", dropna=False)
    comparable = grouped.size().rename("n_cycle_comparable")
    agree = grouped["cycle_agrees"].sum().rename("n_cycle_agree")
    merged = table.merge(comparable, on="track_uid", how="left").merge(agree, on="track_uid", how="left")
    merged["n_cycle_comparable"] = merged["n_cycle_comparable"].fillna(0).astype(int)
    merged["n_cycle_agree"] = merged["n_cycle_agree"].fillna(0).astype(int)
    merged["cycle_agreement_fraction"] = np.where(
        merged["n_cycle_comparable"] > 0,
        merged["n_cycle_agree"] / merged["n_cycle_comparable"],
        np.nan,
    )
    merged["has_cycle_conflict"] = (
        merged["n_cycle_comparable"] > 0
    ) & (merged["n_cycle_agree"] < merged["n_cycle_comparable"])
    merged["cycle_unchecked"] = merged["n_cycle_comparable"] == 0
    merged["review_required"] = merged["review_required"].astype(bool) | merged["has_cycle_conflict"].astype(bool)
    merged["review_reasons"] = merged.apply(
        lambda row: ",".join(
            [reason for reason in [
                str(row["review_reasons"]) if pd.notna(row["review_reasons"]) and str(row["review_reasons"]).strip() else "",
                "cycle_conflict" if bool(row["has_cycle_conflict"]) else "",
            ] if reason]
        ),
        axis=1,
    )
    return merged


def build_track_length_summary_table(tracks_table: pd.DataFrame) -> pd.DataFrame:
    """Summarize the track-length distribution."""

    if tracks_table.empty:
        return pd.DataFrame(columns=["policy", "n_days_present", "missing_internal_days", "n_tracks"])
    summary = (
        tracks_table.groupby(["match_policy", "n_days_present", "missing_internal_days"], dropna=False)
        .size()
        .rename("n_tracks")
        .reset_index()
    )
    return summary.sort_values(["match_policy", "n_days_present", "missing_internal_days"], ascending=[True, False, True]).reset_index(drop=True)

