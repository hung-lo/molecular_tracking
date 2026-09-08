"""Safe IO helpers for explicit Fucci master-run postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


MIN_EXTRACTION_VERSION = (0, 3, 1)
SESSION_POPULATION_NAME = "matched_session_population_roi_metrics.csv"
MATCHED_OBSERVATIONS_NAME = "matched_roi_log_ratio_metrics_all_observed.csv"
GEOMETRY_NAME = "matched_roi_geometry_qc_long.csv"
TRACK_SUMMARY_NAME = "matched_track_qc_summary.csv"


@dataclass(frozen=True)
class FucciMasterRunInputs:
    run_dir: Path
    extraction_dir: Path
    run_manifest_path: Path
    extraction_run_log_path: Path
    session_population_path: Path
    matched_observations_path: Path | None
    geometry_path: Path | None
    track_summary_path: Path | None
    mouse_id: str
    laser_nm: int | None
    session_ids: tuple[str, ...]
    run_manifest: dict[str, Any]
    extraction_run_log: dict[str, Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_tuple(value: Any) -> tuple[int, int, int]:
    parts = str(value).split(".")
    if len(parts) > 3 or any(not part.isdigit() for part in parts):
        raise ValueError(f"Invalid analysis version: {value!r}")
    return tuple(int(part) for part in (parts + ["0", "0", "0"])[:3])


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not parse JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _require_columns(path: Path, columns: set[str]) -> None:
    try:
        actual = set(pd.read_csv(path, nrows=0).columns)
    except (OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read CSV headers from {path}") from exc
    missing = sorted(columns - actual)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")


def _manifest_session_ids(manifest: dict[str, Any]) -> tuple[str, ...]:
    candidates = [
        manifest.get("manifest", {}).get("session_ids") if isinstance(manifest.get("manifest"), dict) else None,
        manifest.get("session_ids"),
        manifest.get("config", {}).get("session_ids") if isinstance(manifest.get("config"), dict) else None,
    ]
    for value in candidates:
        if value:
            return tuple(str(item) for item in value)
    return ()


def _metadata_value(manifest: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if manifest.get(key) is not None:
            return manifest[key]
    for parent_key in ("project", "config", "project_provenance"):
        parent = manifest.get(parent_key)
        if isinstance(parent, dict):
            for key in keys:
                if parent.get(key) is not None:
                    return parent[key]
    return None


def resolve_fucci_master_run(
    run_dir: str | Path, *, require_matched: bool
) -> FucciMasterRunInputs:
    """Validate one explicit master run and resolve paths structurally.

    Absolute paths in the master manifest are intentionally not used to locate
    inputs; copied runs continue to work because all paths are relative to the
    explicitly supplied ``run_dir``.
    """

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Master run directory was not found: {root}")
    manifest_path = root / "run_manifest.json"
    extraction = root / "extraction"
    extraction_log_path = extraction / "run_log.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing master run manifest: {manifest_path}")
    if not extraction.is_dir():
        raise FileNotFoundError(f"Missing extraction directory: {extraction}")
    if not extraction_log_path.is_file():
        raise FileNotFoundError(f"Missing extraction run log: {extraction_log_path}")

    manifest = _read_json(manifest_path)
    extraction_log = _read_json(extraction_log_path)
    try:
        current_version = _version_tuple(extraction_log.get("analysis_version", "0.0.0"))
    except ValueError as exc:
        raise ValueError(f"Invalid extraction analysis_version in {extraction_log_path}") from exc
    population = extraction_log.get("normalization", {}).get("population")
    if current_version < MIN_EXTRACTION_VERSION or population != "all_valid_session_rois":
        raise ValueError(
            f"Extraction at {extraction} is not current: requires analysis_version >= "
            f"{'.'.join(map(str, MIN_EXTRACTION_VERSION))} and normalization.population="
            "all_valid_session_rois. Re-run the master extraction first."
        )

    session_population = extraction / SESSION_POPULATION_NAME
    if not session_population.is_file():
        raise FileNotFoundError(f"Missing native session population: {session_population}")
    _require_columns(
        session_population,
        {"session_id", "session_index", "acquisition_date", "elapsed_days", "red", "green", "ratio_qc_pass"},
    )
    population_table = pd.read_csv(session_population, usecols=["session_id", "session_index", "acquisition_date", "elapsed_days"])
    population_table["session_id"] = population_table["session_id"].astype(str)
    metadata = population_table.drop_duplicates()
    if metadata.groupby("session_id", sort=False).size().gt(1).any():
        raise ValueError(f"Session metadata is inconsistent in {session_population}")
    session_ids = tuple(metadata["session_id"].drop_duplicates().tolist())
    manifest_ids = _manifest_session_ids(manifest)
    if manifest_ids and not set(session_ids).issubset(set(manifest_ids)):
        raise ValueError("Session population contains IDs absent from run_manifest.json")

    paths: dict[str, Path | None] = {}
    required = {
        "matched_observations_path": (MATCHED_OBSERVATIONS_NAME, {"session_id", "session_index", "track_uid", "roi_id", "acquisition_date", "elapsed_days", "green", "red", "ratio_qc_pass"}),
        "geometry_path": (GEOMETRY_NAME, {"session_index", "roi_id", "geometry_qc_pass"}),
        "track_summary_path": (TRACK_SUMMARY_NAME, {"track_uid"}),
    }
    for field, (name, columns) in required.items():
        path = extraction / name
        if require_matched:
            if not path.is_file():
                raise FileNotFoundError(f"Missing required Fucci input: {path}")
            _require_columns(path, columns)
        paths[field] = path if path.is_file() else None

    if require_matched and paths["matched_observations_path"] is not None:
        observed_ids = set(pd.read_csv(paths["matched_observations_path"], usecols=["session_id"])["session_id"].astype(str))
        if not observed_ids.issubset(set(session_ids)):
            raise ValueError("Matched observations contain session IDs absent from the native population")

    mouse_id = _metadata_value(manifest, ("mouse_id", "dataset"))
    if mouse_id is None:
        mouse_id = root.name
    laser_value = _metadata_value(manifest, ("laser_nm",))
    try:
        laser_nm = int(laser_value) if laser_value is not None else None
    except (TypeError, ValueError):
        raise ValueError(f"Invalid laser_nm in {manifest_path}: {laser_value!r}")

    return FucciMasterRunInputs(
        run_dir=root,
        extraction_dir=extraction,
        run_manifest_path=manifest_path,
        extraction_run_log_path=extraction_log_path,
        session_population_path=session_population,
        matched_observations_path=paths["matched_observations_path"],
        geometry_path=paths["geometry_path"],
        track_summary_path=paths["track_summary_path"],
        mouse_id=str(mouse_id),
        laser_nm=laser_nm,
        session_ids=session_ids,
        run_manifest=manifest,
        extraction_run_log=extraction_log,
    )
