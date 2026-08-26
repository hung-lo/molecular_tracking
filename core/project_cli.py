"""Shared explicit-legacy versus project-mode CLI path selection."""
from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
from typing import Any

from project_config import ProjectConfig, load_project_config, validate_output_path

@dataclass(frozen=True)
class DatasetContext:
    mouse_id: str
    laser_nm: int
    raw_root: Path
    derivatives_root: Path
    dataset_dir: Path
    analysis_dir: Path
    xy_um_per_px: float | None = None
    z_um_per_plane: float | None = None
    project_config: ProjectConfig | None = None

def add_project_selector(parser: Any, *, legacy_flag: str = "--dataset", require_laser: bool = True) -> None:
    parser.add_argument(legacy_flag, default=None, help="Explicit legacy dataset/input directory")
    parser.add_argument("--project-config", default=None)
    parser.add_argument("--mouse-id", default=None)
    parser.add_argument("--laser-nm", type=int, default=None, required=False)

def resolve_selection(*, dataset: str | Path | None = None, project_config: str | Path | None = None,
                      mouse_id: str | None = None, laser_nm: int | None = None,
                      output_root: str | Path | None = None) -> DatasetContext:
    project_values = any(value is not None for value in (project_config, mouse_id, laser_nm))
    if dataset is not None and project_values:
        raise ValueError("Choose either explicit legacy --dataset mode or project mode, not both.")
    if dataset is not None:
        source = Path(dataset).expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"Dataset directory was not found: {source}")
        analysis = Path(output_root).expanduser().resolve() if output_root else source / "analysis"
        return DatasetContext("legacy", 0, source, analysis, source, analysis)
    if not project_config or not mouse_id:
        raise ValueError("Project mode requires --project-config and --mouse-id.")
    config = load_project_config(project_config)
    selected_laser = int(laser_nm if laser_nm is not None else config.rig.primary_laser_nm)
    if selected_laser not in {config.rig.primary_laser_nm, config.rig.optional_laser_nm}:
        raise ValueError(f"Unsupported laser_nm={selected_laser}")
    base = config.paths.derivatives_root / str(mouse_id) / "longitudinal" / str(selected_laser)
    analysis = validate_output_path(output_root or base / "runs", config)
    return DatasetContext(str(mouse_id), selected_laser, config.paths.raw_root,
                          config.paths.derivatives_root, base, analysis, project_config=config)

def ready_manifest_path(context: DatasetContext) -> Path:
    if context.project_config is None:
        raise ValueError("Ready manifests are resolved only in project mode")
    path = context.dataset_dir / "manifests" / "daywise_session_manifest.csv"
    if not path.is_file():
        plan = path.with_name("session_manifest_plan.csv")
        suffix = f" A preparation plan exists at {plan}." if plan.exists() else ""
        raise FileNotFoundError(f"Analysis-ready manifest was not found: {path}.{suffix}")
    return path


def catalog_spacing(context: DatasetContext) -> tuple[float, float]:
    """Return uniform catalog XY and Z spacing for one mouse/laser."""
    if context.project_config is None:
        raise ValueError("Catalog spacing is available only in project mode")
    catalog = context.derivatives_root / "_catalog" / "acquisitions.generated.csv"
    if not catalog.is_file():
        raise FileNotFoundError(f"Acquisition catalog was not found: {catalog}")
    with catalog.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["mouse_id"] == context.mouse_id and row["laser_nm"] == str(context.laser_nm) and row["analysis_included"].lower() == "true"]
    if not rows:
        raise ValueError(f"No cataloged {context.laser_nm} acquisitions for {context.mouse_id}")
    xy = {(round(float(row["pixel_size_x_um"]), 9), round(float(row["pixel_size_y_um"]), 9)) for row in rows}
    z = {round(float(row["z_step_um"]), 9) for row in rows}
    if len(xy) != 1 or len(z) != 1:
        raise ValueError("Included project sessions do not have uniform XML-derived spacing")
    x_value, y_value = next(iter(xy))
    if x_value != y_value:
        raise ValueError("This pipeline currently requires equal X/Y pixel spacing")
    return x_value, next(iter(z))
