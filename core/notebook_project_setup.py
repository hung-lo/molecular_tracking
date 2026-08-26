"""Small project-aware path helper used by operational preprocessing notebooks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from project_cli import catalog_spacing,resolve_selection
from project_config import validate_output_path

@dataclass(frozen=True)
class NotebookProjectPaths:
    preprocessing_dir: Path
    registration_dir: Path
    segmentation_dir: Path
    spacing_zyx: tuple[float,float,float]

def resolve_notebook_project_paths(project_config: str|Path,mouse_id:str,laser_nm:int)->NotebookProjectPaths:
    context=resolve_selection(project_config=project_config,mouse_id=mouse_id,laser_nm=laser_nm)
    assert context.project_config is not None
    registration=validate_output_path(context.dataset_dir/"registration",context.project_config)
    preprocessing=validate_output_path(registration/"preprocessing",context.project_config)
    segmentation=validate_output_path(context.dataset_dir/"segmentation",context.project_config)
    x,y,z=catalog_spacing(context)
    return NotebookProjectPaths(preprocessing,registration,segmentation,(z,y,x))
