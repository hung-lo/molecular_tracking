"""Shared explicit-legacy versus project-mode CLI selection and validation."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from project_config import ProjectConfig, load_project_config, validate_output_path

@dataclass(frozen=True)
class DatasetContext:
    mode: str
    mouse_id: str | None
    laser_nm: int | None
    raw_root: Path | None
    derivatives_root: Path | None
    dataset_dir: Path
    analysis_dir: Path
    project_config: ProjectConfig | None = None

def add_project_selector(parser: argparse.ArgumentParser, *, dataset_help: str = "Explicit legacy dataset/input directory") -> None:
    """Add the same mutually exclusive selection arguments to an active CLI."""
    parser.add_argument("--dataset", default=None, help=dataset_help)
    parser.add_argument("--project-config", default=None)
    parser.add_argument("--mouse-id", default=None)
    parser.add_argument("--laser-nm", type=int, default=None)

def _mouse_rows(config: ProjectConfig) -> list[dict[str, str]]:
    if not config.paths.mice_csv.is_file():
        raise FileNotFoundError(f"Mouse metadata was not found: {config.paths.mice_csv}")
    with config.paths.mice_csv.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))

def selected_mouse_metadata(context: DatasetContext) -> dict[str, str]:
    if context.project_config is None or context.mouse_id is None:
        raise ValueError("Mouse metadata is available only in project mode")
    matches=[row for row in _mouse_rows(context.project_config) if row.get("mouse_id")==context.mouse_id]
    if len(matches)!=1: raise ValueError(f"Expected one metadata row for mouse {context.mouse_id}; found {len(matches)}")
    return matches[0]

def resolve_selection(*, dataset: str|Path|None=None, project_config: str|Path|None=None,
                      mouse_id: str|None=None, laser_nm: int|None=None,
                      output_root: str|Path|None=None) -> DatasetContext:
    """Resolve legacy mode from dataset, project mode from config+mouse.

    A laser value alone never changes an explicit legacy dataset into project mode.
    """
    if dataset is not None:
        if project_config is not None or mouse_id is not None:
            raise ValueError("Choose either explicit legacy --dataset mode or project mode, not both.")
        source=Path(dataset).expanduser().resolve()
        if not source.is_dir(): raise FileNotFoundError(f"Dataset directory was not found: {source}")
        analysis=Path(output_root).expanduser().resolve() if output_root else source/"analysis"
        return DatasetContext("legacy",None,int(laser_nm) if laser_nm is not None else None,None,None,source,analysis,None)
    if project_config is None and mouse_id is None:
        raise ValueError("Selection requires explicit --dataset mode or project mode with --project-config and --mouse-id.")
    if not project_config or not mouse_id:
        raise ValueError("Project mode requires both --project-config and --mouse-id.")
    config=load_project_config(project_config)
    known={row.get("mouse_id") for row in _mouse_rows(config)}
    if mouse_id not in known: raise ValueError(f"Unknown mouse_id {mouse_id!r}; expected one of {sorted(known)}")
    selected_laser=int(laser_nm if laser_nm is not None else config.rig.primary_laser_nm)
    allowed={config.rig.primary_laser_nm,config.rig.optional_laser_nm}
    if selected_laser not in allowed: raise ValueError(f"Unsupported laser_nm={selected_laser}; expected {sorted(allowed)}")
    base=config.paths.derivatives_root/mouse_id/"longitudinal"/str(selected_laser)
    analysis=validate_output_path(output_root or base/"runs",config)
    runs_root=base/"runs"
    try: analysis.relative_to(runs_root)
    except ValueError as exc: raise ValueError(f"Project output must remain under {runs_root}: {analysis}") from exc
    return DatasetContext("project",mouse_id,selected_laser,config.paths.raw_root,config.paths.derivatives_root,base,analysis,config)

def file_sha256(path: str|Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()

def catalog_path(context: DatasetContext) -> Path:
    if context.derivatives_root is None: raise ValueError("Catalog paths are project-only")
    return context.derivatives_root/"_catalog"/"acquisitions.generated.csv"

def selected_catalog_rows(context: DatasetContext) -> list[dict[str,str]]:
    path=catalog_path(context)
    if not path.is_file(): raise FileNotFoundError(f"Acquisition catalog was not found: {path}")
    with path.open(encoding="utf-8",newline="") as handle:
        rows=[row for row in csv.DictReader(handle) if row["mouse_id"]==context.mouse_id and row["laser_nm"]==str(context.laser_nm) and row["analysis_included"].lower()=="true"]
    if not rows: raise ValueError(f"No usable {context.laser_nm} acquisitions exist for {context.mouse_id}")
    return rows

def catalog_spacing(context: DatasetContext) -> tuple[float,float,float]:
    rows=selected_catalog_rows(context)
    values={(round(float(r["pixel_size_x_um"]),9),round(float(r["pixel_size_y_um"]),9),round(float(r["z_step_um"]),9)) for r in rows}
    if len(values)!=1: raise ValueError("Included project sessions do not have uniform XML-derived X/Y/Z spacing")
    return next(iter(values))

def ready_manifest_path(context: DatasetContext) -> Path:
    """Return a current ready manifest only after state, catalog, rows, and files validate."""
    if context.project_config is None: raise ValueError("Ready manifests are project-only")
    manifest_dir=context.dataset_dir/"manifests"
    path=manifest_dir/"daywise_session_manifest.csv"; state_path=manifest_dir/"manifest_state.json"
    if not state_path.is_file(): raise FileNotFoundError(f"Manifest state was not found: {state_path}")
    state=json.loads(state_path.read_text(encoding="utf-8"))
    if not state.get("ready"): raise FileNotFoundError(f"Project inputs are not ready; see {manifest_dir/'session_manifest_plan.csv'}")
    current_catalog=catalog_path(context)
    if state.get("catalog_path")!=str(current_catalog.resolve()) or state.get("catalog_sha256")!=file_sha256(current_catalog):
        raise ValueError("Manifest is stale because its source acquisition catalog changed")
    if not path.is_file(): raise FileNotFoundError(f"Analysis-ready manifest was not found: {path}")
    allowed={(r["session_id"],r["acquisition_date"]) for r in selected_catalog_rows(context)}
    seen=set()
    with path.open(encoding="utf-8",newline="") as handle:
        rows=list(csv.DictReader(handle))
    for row in rows:
        key=(row["session_id"],row["acquisition_date"])
        if key not in allowed: raise ValueError(f"Manifest session is not in selected mouse/laser catalog: {key}")
        if key in seen: raise ValueError(f"Duplicate manifest session: {key}")
        seen.add(key)
        for field in ("mask_path","red_image_path","green_image_path"):
            if not Path(row[field]).is_file(): raise FileNotFoundError(f"Stale ready manifest: missing {field} {row[field]}")
    if seen!=allowed: raise ValueError("Manifest sessions do not exactly match the selected catalog sessions")
    return path

def resolve_processed_dataset(context: DatasetContext, *, product_name: str="registered") -> Path:
    """Resolve an explicitly prepared flat compatibility product under derivatives."""
    if context.project_config is None: return context.dataset_dir
    product=validate_output_path(context.dataset_dir/product_name,context.project_config)
    if not product.is_dir():
        raise FileNotFoundError(f"Project preprocessing product is not ready: {product}. Run preprocessing first.")
    return product


def resolve_exact_analysis_dir(context: DatasetContext, value: str | Path | None) -> Path:
    """Resolve an exact existing analysis/run directory without latest-run guessing."""
    if value is None:
        raise ValueError("An exact --analysis-dir or --run-dir is required; latest-run guessing is disabled.")
    path=Path(value).expanduser().resolve()
    if not path.is_dir(): raise FileNotFoundError(f"Analysis directory was not found: {path}")
    if context.mode == "project":
        try: path.relative_to(context.analysis_dir)
        except ValueError as exc: raise ValueError(f"Project analysis directory must be under {context.analysis_dir}: {path}") from exc
    return path


def resolve_analysis_selection(*, analysis_dir: str | Path, dataset: str | Path | None = None, project_config: str | Path | None = None, mouse_id: str | None = None, laser_nm: int | None = None) -> tuple[DatasetContext, Path]:
    """Resolve an exact analysis input as legacy, or validate it against a project mouse/laser."""
    if project_config is not None or mouse_id is not None:
        context=resolve_selection(project_config=project_config,mouse_id=mouse_id,laser_nm=laser_nm)
        if dataset is not None: raise ValueError("Do not combine --dataset with project analysis selection")
        return context,resolve_exact_analysis_dir(context,analysis_dir)
    path=Path(analysis_dir).expanduser().resolve()
    if not path.is_dir(): raise FileNotFoundError(f"Analysis directory was not found: {path}")
    if dataset is not None and Path(dataset).expanduser().resolve()!=path:
        raise ValueError("Legacy --dataset and --analysis-dir must identify the same explicit input when both are supplied")
    context=DatasetContext("legacy",None,laser_nm,None,None,path,path,None)
    return context,path
