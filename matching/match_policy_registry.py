"""Shared match policy constants and validation helpers."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Sequence

SUPPORTED_MATCH_POLICIES: tuple[str, ...] = ("high", "balanced", "graph")
DEFAULT_ANALYSIS_POLICIES: tuple[str, ...] = ("high", "balanced")


def resolve_requested_policies(requested: Sequence[str], match_dir: str | Path) -> tuple[str, ...]:
    """Validate requested policies against the match directory contents."""

    match_dir = Path(match_dir)
    normalized: list[str] = []
    seen: set[str] = set()
    for policy in requested:
        policy = str(policy).strip().lower()
        if not policy:
            raise ValueError("Policy names must be non-empty.")
        if policy not in SUPPORTED_MATCH_POLICIES:
            raise ValueError(f"Unsupported match policy: {policy!r}. Supported policies: {', '.join(SUPPORTED_MATCH_POLICIES)}")
        if policy in seen:
            continue
        if not (match_dir / f"tracks_{policy}.csv").exists():
            raise FileNotFoundError(f"Requested match policy {policy!r} was not found in {match_dir}.")
        seen.add(policy)
        normalized.append(policy)
    if not normalized:
        raise ValueError("At least one match policy must be requested.")
    return tuple(normalized)
