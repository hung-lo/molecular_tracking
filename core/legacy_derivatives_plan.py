"""Read-only planner for the phase2a legacy derivatives audit."""
from __future__ import annotations

import csv
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Iterable, Sequence

from project_config import ProjectConfig, validate_output_path

SESSION_FILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?P<date>\d{8})_(?P<channel>[RG])\.tif$", re.IGNORECASE),
    re.compile(r"^(?P<date>\d{8})_(?P<channel>[RG])_cp_masks_cp_v3_nuclei20\.tif$", re.IGNORECASE),
    re.compile(r"^(?P<date>\d{8})_(?P<channel>[RG])_cp_masks(?:_[^/]+)?\.tif$", re.IGNORECASE),
    re.compile(r"^(?P<date>\d{8})_(?P<channel>[RG])_SyN\.tif$", re.IGNORECASE),
    re.compile(r"^(?P<date>\d{8})_(?P<channel>[RG])_ROI_mask_SyN_inversed\.tif$", re.IGNORECASE),
)
RUN_TIMESTAMP_PATTERN = re.compile(r"(?<!\d)\d{8}_\d{6}(?!\d)")
DATE_TOKEN_PATTERN = re.compile(r"(?<!\d)\d{8}(?!\d)")

ALLOWED_EXACT_ROOTS = (
    "1050_data",
    "920_data",
    "2wks_1050_data",
    "1050_small_test_fireants",
)
ALLOWED_GLOB_ROOTS = ("roi_matcher_qc_examples_*",)
DEFAULT_EXCLUDED_ROOTS = {"roi_matcher_qc_examples_syn_20260615_01_styled"}

INVENTORY_FIELDNAMES = [
    "relative_source_path",
    "source_path",
    "size_bytes",
    "mtime_utc",
    "extension",
    "inferred_mouse_id",
    "inferred_laser_nm",
    "product_class",
    "target_scope",
    "inferred_session_date",
    "date_token_role",
    "catalog_session_match",
    "inference_status",
    "notes",
]
PLAN_FIELDNAMES = [
    "source_path",
    "proposed_target",
    "inferred_mouse_id",
    "inferred_laser_nm",
    "product_class",
    "target_scope",
    "inferred_session_date",
    "date_token_role",
    "catalog_session_match",
    "inference_status",
    "action",
    "collision_status",
    "source_size_bytes",
    "source_mtime_utc",
    "reason_evidence",
]


@dataclass(frozen=True)
class CatalogSession:
    mouse_id: str
    acquisition_date: str
    laser_nm: int
    session_id: str
    acquisition_id: str
    source_path: str


@dataclass(frozen=True)
class TreeContext:
    mouse_id: str | None
    laser_nm: int | None
    evidence: tuple[str, ...]


@dataclass(frozen=True)
class LegacyScanSummary:
    included_roots: tuple[str, ...]
    ignored_top_level_entries: tuple[tuple[str, str], ...]


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Acquisition catalog was not found: {source}")
    with source.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_acquisition_catalog(path: str | Path) -> list[dict[str, str]]:
    rows = _read_csv_rows(path)
    required = {
        "mouse_id",
        "session_id",
        "acquisition_date",
        "acquisition_id",
        "source_path",
        "analysis_included",
        "laser_nm",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Acquisition catalog must contain: {', '.join(sorted(required))}")
    return rows


def build_catalog_sessions(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str, int], CatalogSession]:
    sessions: dict[tuple[str, str, int], CatalogSession] = {}
    for row in rows:
        if not _normalize_bool(row.get("analysis_included", "")):
            continue
        key = (str(row["mouse_id"]), str(row["acquisition_date"]), int(row["laser_nm"]))
        sessions[key] = CatalogSession(
            mouse_id=str(row["mouse_id"]),
            acquisition_date=str(row["acquisition_date"]),
            laser_nm=int(row["laser_nm"]),
            session_id=str(row["session_id"]),
            acquisition_id=str(row["acquisition_id"]),
            source_path=str(row["source_path"]),
        )
    return sessions


def _iso_date(token: str) -> str:
    return f"{token[:4]}-{token[4:6]}-{token[6:]}"


def _is_hidden_name(name: str) -> bool:
    return name.startswith(".") or name == "__pycache__"


def _entry_type(entry: Path) -> str:
    if entry.is_dir():
        return "directory"
    if entry.is_file():
        return "file"
    if entry.is_symlink():
        return "symlink"
    return "other"


def _is_allowed_root_name(name: str) -> bool:
    return (
        name not in DEFAULT_EXCLUDED_ROOTS
        and (name in ALLOWED_EXACT_ROOTS or any(fnmatch(name, pattern) for pattern in ALLOWED_GLOB_ROOTS))
    )


def _validate_include_root_name(name: str, legacy_root: Path) -> None:
    candidate = Path(name)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise ValueError(f"include-root must be a direct child name, not a path: {name}")
    if candidate.name != name:
        raise ValueError(f"include-root must be a direct child name, not a path: {name}")
    if not (legacy_root / name).exists():
        raise FileNotFoundError(f"Included root was not found under legacy_fucci_tri_root: {name}")


def _resolve_include_root_names(legacy_root: Path, include_roots: Sequence[str] | None) -> list[str]:
    children = {entry.name: entry for entry in sorted(legacy_root.iterdir(), key=lambda entry: entry.name)}
    if include_roots is None:
        selected = [name for name, entry in children.items() if entry.is_dir() and _is_allowed_root_name(name)]
        if not selected:
            raise ValueError(
                f"No expected legacy product roots were found under {legacy_root}: "
                f"{', '.join(ALLOWED_EXACT_ROOTS)}"
            )
        return selected

    selected: list[str] = []
    seen: set[str] = set()
    for name in include_roots:
        _validate_include_root_name(name, legacy_root)
        if name in seen:
            raise ValueError(f"Duplicate --include-root value: {name}")
        seen.add(name)
        if not (legacy_root / name).is_dir():
            raise FileNotFoundError(f"Included root is not a directory: {name}")
        selected.append(name)
    return selected


def discover_legacy_source_files(
    legacy_root: str | Path,
    *,
    include_roots: Sequence[str] | None = None,
) -> tuple[list[Path], LegacyScanSummary]:
    root = Path(legacy_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"legacy_fucci_tri_root is unreadable: {root}")

    include_root_names = _resolve_include_root_names(root, include_roots)
    children = {entry.name: entry for entry in sorted(root.iterdir(), key=lambda entry: entry.name)}
    ignored = tuple(
        (name, _entry_type(entry))
        for name, entry in children.items()
        if name not in include_root_names
    )

    files: list[Path] = []
    for root_name in include_root_names:
        top_level = root / root_name
        for path in sorted(top_level.rglob("*"), key=lambda candidate: candidate.relative_to(root).as_posix()):
            if not path.is_file():
                continue
            if any(_is_hidden_name(part) for part in path.relative_to(root).parts):
                continue
            files.append(path)

    return files, LegacyScanSummary(tuple(include_root_names), ignored)


def _top_level_name(source_path: Path, legacy_root: Path) -> str:
    relative = source_path.relative_to(legacy_root)
    return relative.parts[0] if relative.parts else source_path.name


def _session_filename_match(name: str) -> dict[str, str] | None:
    for pattern in SESSION_FILE_PATTERNS:
        match = pattern.fullmatch(name)
        if match:
            return match.groupdict()
    return None


def _root_laser_hint(root_name: str) -> int | None:
    if "1050" in root_name:
        return 1050
    if "920" in root_name:
        return 920
    return None


def _is_longitudinal_root(root_name: str) -> bool:
    return root_name == "1050_small_test_fireants" or root_name.startswith("roi_matcher_qc_examples_")


def _contains_run_timestamp(relative_parts: Iterable[str]) -> bool:
    return any(RUN_TIMESTAMP_PATTERN.search(part) for part in relative_parts)


def _target_scope(root_name: str, relative_parts: tuple[str, ...], session_match: dict[str, str] | None) -> str:
    if _is_longitudinal_root(root_name):
        return "longitudinal"
    if any(part == "analysis" for part in relative_parts):
        return "longitudinal"
    if _contains_run_timestamp(relative_parts):
        return "longitudinal"
    if session_match is not None:
        return "session"
    if _is_allowed_root_name(root_name):
        return "longitudinal"
    return "unmapped"


def _product_class(path: Path, root_name: str, session_match: dict[str, str] | None) -> str:
    name = path.name
    suffix = path.suffix.lower()
    lower = name.lower()

    if session_match is not None:
        channel = session_match["channel"].upper()
        date = session_match["date"]
        if channel == "R" and name == f"{date}_R.tif":
            return "raw_red_stack"
        if channel == "G" and name == f"{date}_G.tif":
            return "raw_green_stack"
        if "_cp_masks" in lower:
            return "session_mask"
        if lower.endswith("_syn.tif"):
            return "session_registered_stack"
        if lower.endswith("_roi_mask_syn_inversed.tif"):
            return "session_registered_mask"
        return "session_stack"

    if root_name.startswith("roi_matcher_qc_examples_"):
        if suffix == ".png":
            return "qc_figure"
        if suffix == ".csv":
            return "qc_table"
        if suffix == ".json":
            return "qc_metadata"
        if suffix == ".md":
            return "qc_summary"
        return "qc_output"

    if root_name == "1050_small_test_fireants":
        if suffix == ".png":
            return "test_figure"
        if suffix == ".csv":
            return "test_table"
        if suffix == ".json":
            return "test_metadata"
        if suffix == ".md":
            return "test_summary"
        return "test_output"

    if suffix == ".png":
        return "analysis_figure"
    if suffix == ".csv":
        return "analysis_table"
    if suffix == ".json":
        return "analysis_metadata"
    if suffix == ".md":
        return "analysis_summary"
    if suffix == ".npy":
        return "analysis_array"
    if suffix == ".tif":
        return "analysis_output"
    return "legacy_output"


def _date_token_role(relative_parts: tuple[str, ...], root_name: str, session_match: dict[str, str] | None) -> str:
    if session_match is not None:
        return "acquisition_date"
    if _contains_run_timestamp(relative_parts):
        return "run_timestamp"
    if root_name.startswith("roi_matcher_qc_examples_") and DATE_TOKEN_PATTERN.search(root_name):
        return "run_timestamp"
    if any(DATE_TOKEN_PATTERN.search(part) for part in relative_parts):
        return "run_timestamp"
    return "none"


def _mouse_candidates_for_laser(catalog_sessions: dict[tuple[str, str, int], CatalogSession], laser_nm: int) -> list[str]:
    return sorted({session.mouse_id for session in catalog_sessions.values() if session.laser_nm == laser_nm})


def _infer_tree_contexts(
    legacy_root: Path,
    files: Iterable[Path],
    catalog_sessions: dict[tuple[str, str, int], CatalogSession],
) -> dict[str, TreeContext]:
    grouped: dict[str, list[Path]] = {}
    for path in files:
        grouped.setdefault(_top_level_name(path, legacy_root), []).append(path)

    contexts: dict[str, TreeContext] = {}
    for root_name, root_files in grouped.items():
        laser_hint = _root_laser_hint(root_name)
        candidate_mice: set[str] = set()
        evidence: list[str] = []

        if laser_hint is not None:
            evidence.append(f"laser_hint={laser_hint}")

        for path in root_files:
            session_match = _session_filename_match(path.name)
            if session_match is None or laser_hint is None:
                continue
            session_date = _iso_date(session_match["date"])
            for session in catalog_sessions.values():
                if session.acquisition_date == session_date and session.laser_nm == laser_hint:
                    candidate_mice.add(session.mouse_id)

        if not candidate_mice and laser_hint is not None:
            mice_for_laser = _mouse_candidates_for_laser(catalog_sessions, laser_hint)
            if len(mice_for_laser) == 1:
                candidate_mice.add(mice_for_laser[0])
                evidence.append(f"mouse_inferred_from_unique_laser={mice_for_laser[0]}")

        mouse_id = next(iter(candidate_mice)) if len(candidate_mice) == 1 else None
        if mouse_id is not None:
            evidence.append(f"mouse_inferred={mouse_id}")

        contexts[root_name] = TreeContext(mouse_id=mouse_id, laser_nm=laser_hint, evidence=tuple(evidence))
    return contexts


def _catalog_session_match(
    catalog_sessions: dict[tuple[str, str, int], CatalogSession],
    mouse_id: str | None,
    session_date: str | None,
    laser_nm: int | None,
) -> bool | str:
    if session_date is None or mouse_id is None or laser_nm is None:
        return "not_applicable"
    return (mouse_id, session_date, laser_nm) in catalog_sessions


def _session_target(
    config: ProjectConfig,
    mouse_id: str,
    laser_nm: int,
    session_date: str,
    relative_source_path: str,
) -> Path:
    return (
        config.paths.derivatives_root
        / mouse_id
        / "sessions"
        / session_date.replace("-", "")
        / str(laser_nm)
        / "legacy_import"
        / relative_source_path
    )


def _longitudinal_target(
    config: ProjectConfig,
    mouse_id: str,
    laser_nm: int,
    relative_source_path: str,
) -> Path:
    return (
        config.paths.derivatives_root
        / mouse_id
        / "longitudinal"
        / str(laser_nm)
        / "legacy_import"
        / relative_source_path
    )


def _collision_status(target: Path | None, *, target_counts: Counter[Path]) -> str:
    if target is None:
        return "not_applicable"
    if target_counts[target] > 1:
        return "duplicate_target"
    if target.exists():
        return "exists"
    return "clear"


def _resolve_inference_status(
    target_scope: str,
    mouse_id: str | None,
    laser_nm: int | None,
    proposed_target: Path | None,
    catalog_session_match: bool | str,
) -> str:
    if target_scope == "session":
        return "resolved" if proposed_target is not None and catalog_session_match is True else "ambiguous"
    if target_scope == "longitudinal":
        if proposed_target is not None and mouse_id is not None and laser_nm is not None:
            return "resolved"
        if mouse_id is not None or laser_nm is not None:
            return "ambiguous"
        return "unmapped"
    return "unmapped"


def _reason_evidence(
    *,
    relative_source_path: str,
    product_class: str,
    target_scope: str,
    session_match: dict[str, str] | None,
    tree_context: TreeContext,
    date_token_role: str,
    catalog_session_match: bool | str,
) -> str:
    pieces = [
        f"path={relative_source_path}",
        f"product_class={product_class}",
        f"target_scope={target_scope}",
    ]
    if session_match is not None:
        pieces.append(f"session_filename_date={_iso_date(session_match['date'])}")
        pieces.append(f"session_filename_channel={session_match['channel']}")
    if tree_context.mouse_id is not None:
        pieces.append(f"mouse={tree_context.mouse_id}")
    if tree_context.laser_nm is not None:
        pieces.append(f"laser={tree_context.laser_nm}")
    if tree_context.evidence:
        pieces.extend(tree_context.evidence)
    pieces.append(f"date_token_role={date_token_role}")
    pieces.append(f"catalog_session_match={catalog_session_match}")
    return "; ".join(pieces)


def _atomic_write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def build_legacy_derivatives_audit(
    config: ProjectConfig,
    *,
    acquisition_catalog_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    include_roots: Sequence[str] | None = None,
    write_outputs: bool = True,
    return_summary: bool = False,
):
    if config.paths.legacy_fucci_tri_root is None:
        raise ValueError("legacy_fucci_tri_root is required in the project configuration.")

    legacy_root = Path(config.paths.legacy_fucci_tri_root).expanduser().resolve()
    if not legacy_root.is_dir():
        raise FileNotFoundError(f"legacy_fucci_tri_root is unreadable: {legacy_root}")

    catalog_path = (
        Path(acquisition_catalog_path).expanduser().resolve()
        if acquisition_catalog_path is not None
        else (config.paths.derivatives_root / "_catalog" / "acquisitions.generated.csv").resolve()
    )
    catalog_rows = load_acquisition_catalog(catalog_path)
    catalog_sessions = build_catalog_sessions(catalog_rows)
    source_files, summary = discover_legacy_source_files(legacy_root, include_roots=include_roots)
    tree_contexts = _infer_tree_contexts(legacy_root, source_files, catalog_sessions)

    inventory_rows: list[dict[str, object]] = []
    plan_rows: list[dict[str, object]] = []
    proposed_targets: list[Path] = []

    for path in source_files:
        relative_source_path = path.relative_to(legacy_root).as_posix()
        relative_parts = path.relative_to(legacy_root).parts
        root_name = relative_parts[0]
        session_match = _session_filename_match(path.name)
        target_scope = _target_scope(root_name, relative_parts, session_match)
        product_class = _product_class(path, root_name, session_match)
        date_token_role = _date_token_role(relative_parts, root_name, session_match)
        tree_context = tree_contexts.get(root_name, TreeContext(None, None, ()))
        mouse_id = tree_context.mouse_id
        laser_nm = tree_context.laser_nm

        session_date = _iso_date(session_match["date"]) if session_match is not None and target_scope == "session" else None
        catalog_session_match = _catalog_session_match(catalog_sessions, mouse_id, session_date, laser_nm)

        if target_scope == "session" and catalog_session_match is not True and session_date is not None and laser_nm is not None and mouse_id is None:
            matching_mice = sorted(
                {
                    session.mouse_id
                    for session in catalog_sessions.values()
                    if session.acquisition_date == session_date and session.laser_nm == laser_nm
                }
            )
            if len(matching_mice) == 1:
                mouse_id = matching_mice[0]
                catalog_session_match = _catalog_session_match(catalog_sessions, mouse_id, session_date, laser_nm)

        if target_scope == "longitudinal" and mouse_id is None and laser_nm is not None:
            matching_mice = _mouse_candidates_for_laser(catalog_sessions, laser_nm)
            if len(matching_mice) == 1:
                mouse_id = matching_mice[0]
                catalog_session_match = "not_applicable"

        proposed_target: Path | None = None
        if target_scope == "session" and session_date is not None and catalog_session_match is True and mouse_id is not None and laser_nm is not None:
            proposed_target = _session_target(config, mouse_id, laser_nm, session_date, relative_source_path)
        elif target_scope == "longitudinal" and mouse_id is not None and laser_nm is not None:
            proposed_target = _longitudinal_target(config, mouse_id, laser_nm, relative_source_path)

        proposed_target = validate_output_path(proposed_target, config) if proposed_target is not None else None
        if proposed_target is not None:
            proposed_targets.append(proposed_target)

        inference_status = _resolve_inference_status(target_scope, mouse_id, laser_nm, proposed_target, catalog_session_match)

        inventory_rows.append(
            {
                "relative_source_path": relative_source_path,
                "source_path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                "extension": path.suffix.lower(),
                "inferred_mouse_id": mouse_id or "",
                "inferred_laser_nm": laser_nm if laser_nm is not None else "",
                "product_class": product_class,
                "target_scope": target_scope,
                "inferred_session_date": session_date or "",
                "date_token_role": date_token_role,
                "catalog_session_match": catalog_session_match,
                "inference_status": inference_status,
                "notes": _reason_evidence(
                    relative_source_path=relative_source_path,
                    product_class=product_class,
                    target_scope=target_scope,
                    session_match=session_match,
                    tree_context=tree_context,
                    date_token_role=date_token_role,
                    catalog_session_match=catalog_session_match,
                ),
            }
        )

        plan_rows.append(
            {
                "source_path": path.as_posix(),
                "proposed_target": proposed_target.as_posix() if proposed_target is not None else "",
                "inferred_mouse_id": mouse_id or "",
                "inferred_laser_nm": laser_nm if laser_nm is not None else "",
                "product_class": product_class,
                "target_scope": target_scope,
                "inferred_session_date": session_date or "",
                "date_token_role": date_token_role,
                "catalog_session_match": catalog_session_match,
                "inference_status": inference_status,
                "action": "review_required",
                "collision_status": "not_applicable",
                "source_size_bytes": path.stat().st_size,
                "source_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
                "reason_evidence": _reason_evidence(
                    relative_source_path=relative_source_path,
                    product_class=product_class,
                    target_scope=target_scope,
                    session_match=session_match,
                    tree_context=tree_context,
                    date_token_role=date_token_role,
                    catalog_session_match=catalog_session_match,
                ),
            }
        )

    target_counts = Counter(Path(row["proposed_target"]) for row in plan_rows if row["proposed_target"])
    for row in plan_rows:
        proposed_target = row["proposed_target"]
        if proposed_target:
            row["collision_status"] = _collision_status(Path(proposed_target), target_counts=target_counts)

    inventory_rows.sort(key=lambda row: row["relative_source_path"])
    plan_rows.sort(key=lambda row: row["source_path"])

    audit_dir = (
        validate_output_path(output_dir, config)
        if output_dir is not None
        else validate_output_path(config.paths.derivatives_root / "_catalog" / "phase2a_audit", config)
    )
    if write_outputs:
        _atomic_write_csv(audit_dir / "legacy_derivatives_inventory.csv", inventory_rows, INVENTORY_FIELDNAMES)
        _atomic_write_csv(audit_dir / "legacy_derivatives_migration_plan.csv", plan_rows, PLAN_FIELDNAMES)

    if return_summary:
        return inventory_rows, plan_rows, audit_dir, summary
    return inventory_rows, plan_rows, audit_dir
