from __future__ import annotations

from pathlib import Path

import pytest

from match_policy_registry import DEFAULT_ANALYSIS_POLICIES, SUPPORTED_MATCH_POLICIES, resolve_requested_policies


def test_resolve_requested_policies_accepts_supported_subset(tmp_path: Path) -> None:
    for policy in SUPPORTED_MATCH_POLICIES:
        (tmp_path / f"tracks_{policy}.csv").write_text("cluster_id\n1\n", encoding="utf-8")

    resolved = resolve_requested_policies(["high", "balanced", "graph"], tmp_path)

    assert resolved == ("high", "balanced", "graph")
    assert DEFAULT_ANALYSIS_POLICIES == ("high", "balanced")


def test_resolve_requested_policies_rejects_missing_policy_file(tmp_path: Path) -> None:
    (tmp_path / "tracks_high.csv").write_text("cluster_id\n1\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        resolve_requested_policies(["high", "graph"], tmp_path)


def test_resolve_requested_policies_normalizes_duplicates_and_preserves_order(tmp_path: Path) -> None:
    for policy in SUPPORTED_MATCH_POLICIES:
        (tmp_path / f"tracks_{policy}.csv").write_text("cluster_id\n1\n", encoding="utf-8")

    resolved = resolve_requested_policies(["graph", "high", "graph", "balanced"], tmp_path)

    assert resolved == ("graph", "high", "balanced")


def test_resolve_requested_policies_rejects_blank_or_unsupported_names(tmp_path: Path) -> None:
    (tmp_path / "tracks_high.csv").write_text("cluster_id\n1\n", encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_requested_policies(["", "high"], tmp_path)
    with pytest.raises(ValueError):
        resolve_requested_policies(["high", "invalid"], tmp_path)
