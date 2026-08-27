"""Read-only planner for the phase2a legacy derivatives audit."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from project_config import ProjectConfig, validate_output_path

SESSION_FILE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^(?P<date>\d{8})_(?P<channel>[RG])\.tif$", re.IGNORECASE),
    re.compile(r"^(?P<date>\d{8})_(?P<channel>[RG])_cp_masks_cp_v3_nuclei20\.tif$", re.IGNORECASE),
    re.compile(r"^(?P<date>\d{8})_(?P<channel>[RG])_cp_masks(?:_[^/]+)?\.tif$", re.IGNORECASE),
    re.compile(r"^(?P<date>\d{8})_(?P<channel>[RG])_SyN\.tif$", re.IGNORECASE),
    re.compile(r"^(?P<date>\d{8})_(?P<channel>[RG])_ROI_mask_SyN_inversed\.tif$", re.IGNORECASE),
)
RUN_TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(?P<stamp>\d{8}_\d{6})(?!\d)")
DATE_TOKEN_PATTERN = re.compile(r"(?<!\d)(?P<date>\d{8})(?!\d)")
LASER_PATTERN = re.compile(r"(?<!\d)(?P<laser>920|1050)(?!\d)")

LONGITUDINAL_TREE_PREFIXES = (
    "analysis",
    "roi_matcher_qc_examples",
    "roi_matcher_qc_plots",
    "roi_matcher_qc",
    "small_test",
)

SESSION_LIKE_BASENAMES = {
    "R": "raw_red_stack",
    "G": "raw_green_stack",
    "cp_masks": "session_mask",
    "SyN": "session_registered_stack",
    "ROI_mask_SyN_inversed": "session_registered_mask",
}


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
    """Load the canonical acquisition catalog used for session matching."""

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


def iter_legacy_source_files(legacy_root: str | Path) -> list[Path]:
    root = Path(legacy_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"legacy_fucci_tri_root is unreadable: {root}")

    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def _top_level_name(source_path: Path, legacy_root: Path) -> str:
    relative = source_path.relative_to(legacy_root)
    return relative.parts[0] if relative.parts else source_path.name


def _session_filename_match(name: str) -> dict[str, str] | None:
    for pattern in SESSION_FILE_PATTERNS:
        match = pattern.fullmatch(name)
        if match:
            return match.groupdict()
    return None


def _run_timestamp_role(parts: Iterable[str]) -> str:
    for part in parts:
        if RUN_TIMESTAMP_PATTERN.search(part):
            return "run_timestamp"
        if part.startswith("roi_matcher_qc_examples_") and DATE_TOKEN_PATTERN.search(part):
            return "run_timestamp"
    return "none"


def _has_longitudinal_tree_context(relative_parts: tuple[str, ...]) -> bool:
    top_level = relative_parts[0] if relative_parts else ""
    if top_level.startswith(LONGITUDINAL_TREE_PREFIXES) or "small_test" in top_level:
        return True
    return any(part == "analysis" for part in relative_parts)


def _product_class(path: Path, session_match: dict[str, str] | None) -> str:
    name = path.name
    suffix = path.suffix.lower()
    lower = name.lower()
    if session_match:
        channel = session_match["channel"].upper()
        if channel == "R" and name == f"{session_match['date']}_R.tif":
            return "raw_red_stack"
        if channel == "G" and name == f"{session_match['date']}_G.tif":
            return "raw_green_stack"
        if "_cp_masks" in lower:
            return "session_mask"
        if lower.endswith("_syn.tif"):
            return "session_registered_stack"
        if lower.endswith("_roi_mask_syn_inversed.tif"):
            return "session_registered_mask"
        return "session_stack"
    if "analysis" in path.parts:
        return "analysis_output"
    if lower.startswith("mean_image_") and suffix == ".tif":
        return "mean_image"
    if lower.startswith("roi_intensity_results") and suffix == ".csv":
        return "roi_table"
    if lower == "dark_values.tif":
        return "dark_reference"
    if lower == "run_log.json":
        return "run_metadata"
    if "roi_matcher_qc_examples" in "/".join(path.parts):
        if suffix == ".png":
            return "qc_figure"
        if suffix == ".csv":
            return "qc_table"
        if suffix == ".md":
            return "qc_summary"
        if suffix == ".json":
            return "qc_metadata"
    if suffix == ".png":
        return "figure"
    if suffix == ".csv":
        return "table"
    if suffix == ".json":
        return "json_metadata"
    if suffix == ".md":
        return "markdown"
    if suffix == ".ipynb":
        return "notebook"
    if suffix == ".py":
        return "script"
    if suffix == ".npy":
        return "npy_array"
    return "other"


def _candidate_mouse_ids(
    relative_parts: tuple[str, ...],
    session_match: dict[str, str] | None,
    catalog_sessions: dict[tuple[str, str, int], CatalogSession],
    tree_laser_nm: int | None,
) -> tuple[str, ...]:
    if not session_match or tree_laser_nm is None:
        return ()
    acquisition_date = f"{session_match["date"][:4]}-{session_match["date"][4:6]}-{session_match["date"][6:]}"
    candidates = tuple(
        sorted(
            session.mouse_id
            for key, session in catalog_sessions.items()
            if key[1] == acquisition_date and key[2] == tree_laser_nm
        )
    )
    return candidates


def infer_tree_context(
    legacy_root: str | Path,
    source_files: Iterable[Path],
    catalog_sessions: dict[tuple[str, str, int], CatalogSession],
) -> dict[str, TreeContext]:
    root = Path(legacy_root).expanduser().resolve()
    grouped: dict[str, list[Path]] = {}
    for path in source_files:
        top_level = _top_level_name(path, root)
        grouped.setdefault(top_level, []).append(path)

    contexts: dict[str, TreeContext] = {}
    for top_level, paths in grouped.items():
        laser_hint = None
        laser_match = LASER_PATTERN.search(top_level)
        if laser_match:
            laser_hint = int(laser_match.group("laser"))

        session_mouse_candidates: set[str] = set()
        for path in paths:
            relative_parts = path.relative_to(root).parts
            session_match = _session_filename_match(path.name)
            if not session_match:
                continue
            if laser_hint is None:
                laser_match = LASER_PATTERN.search(path.name)
                if laser_match:
                    laser_hint = int(laser_match.group("laser"))
            if laser_hint is None:
                continue
            session_mouse_candidates.update(
                _candidate_mouse_ids(relative_parts, session_match, catalog_sessions, laser_hint)
            )

        mouse_id = next(iter(session_mouse_candidates)) if len(session_mouse_candidates) == 1 else None
        evidence: list[str] = []
        if laser_hint is not None:
            evidence.append(f"laser_hint={laser_hint}")
        if mouse_id is not None:
            evidence.append(f"mouse_inferred={mouse_id}")
        contexts[top_level] = TreeContext(mouse_id=mouse_id, laser_nm=laser_hint, evidence=tuple(evidence))
    return contexts


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


def _catalog_match(
    catalog_sessions: dict[tuple[str, str, int], CatalogSession],
    mouse_id: str | None,
    session_date: str | None,
    laser_nm: int | None,
) -> bool | str:
    if mouse_id is None or session_date is None or laser_nm is None:
        return "not_applicable"
    return (mouse_id, session_date, laser_nm) in catalog_sessions


def _collision_status(target: Path | None, existing_targets: set[Path], duplicate_targets: set[Path]) -> str:
    if target is None:
        return "not_applicable"
    if target in duplicate_targets:
        return "duplicate_target"
    if target.exists():
        return "exists"
    if target in existing_targets:
        return "duplicate_target"
    return "clear"

def _mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _infer_scope(
    relative_parts: tuple[str, ...],
    session_match: dict[str, str] | None,
) -> str:
    if _has_longitudinal_tree_context(relative_parts):
        return "longitudinal"
    if session_match:
        return "session"
    return "unmapped"


def _inference_status(mouse_id: str | None, laser_nm: int | None, target_scope: str) -> str:
    if target_scope == "session":
        return "resolved" if mouse_id is not None and laser_nm is not None else "ambiguous"
    if target_scope == "longitudinal":
        if mouse_id is not None and laser_nm is not None:
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
    pieces = [f"path={relative_source_path}", f"product_class={product_class}", f"target_scope={target_scope}"]
    if session_match is not None:
        pieces.append(f"session_filename_date={session_match['date']}")
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


def build_legacy_derivatives_audit(
    config: ProjectConfig,
    *,
    acquisition_catalog_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> tuple[list[dict[str, object]], list[dict[str, object]], Path]:
    """Build the inventory and review-only migration plan."""

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
    source_files = iter_legacy_source_files(legacy_root)
    tree_contexts = infer_tree_context(legacy_root, source_files, catalog_sessions)

    inventory_rows: list[dict[str, object]] = []
    plan_rows: list[dict[str, object]] = []
    proposed_targets: list[Path] = []

    for path in source_files:
        relative_source_path = path.relative_to(legacy_root).as_posix()
        relative_parts = path.relative_to(legacy_root).parts
        session_match = _session_filename_match(path.name)
        product_class = _product_class(path, session_match)
        target_scope = _infer_scope(relative_parts, session_match)
        tree_context = tree_contexts[_top_level_name(path, legacy_root)]
        laser_nm = tree_context.laser_nm
        mouse_id = tree_context.mouse_id
        session_date = (
            f"{session_match["date"][:4]}-{session_match["date"][4:6]}-{session_match["date"][6:]}"
            if target_scope == "session" and session_match
            else None
        )
        if target_scope == "session" and session_date is not None and laser_nm is not None and mouse_id is None:
            candidates = _candidate_mouse_ids(relative_parts, session_match, catalog_sessions, laser_nm)
            mouse_id = candidates[0] if len(candidates) == 1 else None

        date_token_role = "acquisition_date" if target_scope == "session" and session_date is not None else _run_timestamp_role(relative_parts)
        if target_scope != "session" and date_token_role == "acquisition_date":
            date_token_role = "none"

        catalog_session_match = _catalog_match(catalog_sessions, mouse_id, session_date, laser_nm)
        inference_status = _inference_status(mouse_id, laser_nm, target_scope)
        if target_scope == "session":
            inference_status = "resolved" if catalog_session_match is True and mouse_id is not None and laser_nm is not None else "ambiguous"

        proposed_target: Path | None = None
        if inference_status == "resolved":
            if target_scope == "session" and session_date is not None and catalog_session_match is True:
                proposed_target = _session_target(config, mouse_id or "", laser_nm or 0, session_date, relative_source_path)
            elif target_scope == "longitudinal":
                proposed_target = _longitudinal_target(config, mouse_id or "", laser_nm or 0, relative_source_path)

        if proposed_target is not None:
            proposed_target = validate_output_path(proposed_target, config)
            proposed_targets.append(proposed_target)

        inventory_rows.append(
            {
                "relative_source_path": relative_source_path,
                "source_path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "mtime_utc": _mtime_utc(path),
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
                "collision_status": "clear" if proposed_target is not None else "not_applicable",
                "source_size_bytes": path.stat().st_size,
                "source_mtime_utc": _mtime_utc(path),
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

    duplicate_targets = {target for target in proposed_targets if proposed_targets.count(target) > 1}
    target_lookup = {Path(row["proposed_target"]) for row in plan_rows if row["proposed_target"]}
    for row in plan_rows:
        proposed_target = row["proposed_target"]
        if not proposed_target:
            continue
        target_path = Path(proposed_target)
        row["collision_status"] = _collision_status(target_path, target_lookup, duplicate_targets)

    inventory_rows.sort(key=lambda row: row["relative_source_path"])
    plan_rows.sort(key=lambda row: row["source_path"])

    audit_dir = (
        validate_output_path(output_dir, config)
        if output_dir is not None
        else validate_output_path(config.paths.derivatives_root / "_catalog" / "phase2a_audit", config)
    )
    if write_outputs:
        audit_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(
            audit_dir / "legacy_derivatives_inventory.csv",
            inventory_rows,
            [
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
            ],
        )
        _write_csv(
            audit_dir / "legacy_derivatives_migration_plan.csv",
            plan_rows,
            [
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
            ],
        )
    return inventory_rows, plan_rows, audit_dir


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

