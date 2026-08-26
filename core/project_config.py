"""Load and validate multi-mouse project configuration."""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import tomllib
from typing import Any

CONFIG_ENV = "MOLECULAR_TRACKING_CONFIG"

@dataclass(frozen=True)
class ProjectPaths:
    raw_root: Path
    derivatives_root: Path
    mice_csv: Path
    legacy_fucci_tri_root: Path | None = None

@dataclass(frozen=True)
class RigConfig:
    primary_laser_nm: int
    optional_laser_nm: int
    pockels_1_laser_nm: int
    pockels_2_laser_nm: int
    chan_a_signal: str
    chan_b_signal: str

@dataclass(frozen=True)
class CanonicalVolumeConfig:
    imaging_planes: int
    flyback_planes: int
    z_step_um: float
    volumes: int

@dataclass(frozen=True)
class ProjectConfig:
    source_path: Path
    paths: ProjectPaths
    rig: RigConfig
    canonical_volume: CanonicalVolumeConfig

def _required(table: dict[str, Any], key: str, section: str) -> Any:
    if key not in table:
        raise ValueError(f"Missing required [{section}] setting: {key}")
    return table[key]

def _path(value: Any, base: Path) -> Path:
    path=Path(str(value)).expanduser()
    return (base/path).resolve() if not path.is_absolute() else path.resolve()

def is_relative_to(path: Path, parent: Path) -> bool:
    try: path.relative_to(parent); return True
    except ValueError: return False

def validate_output_path(path: str|Path, config: ProjectConfig) -> Path:
    resolved=Path(path).expanduser().resolve()
    if resolved==config.paths.raw_root or is_relative_to(resolved,config.paths.raw_root):
        raise ValueError(f"Generated output must not be inside raw_root: {resolved}")
    if not is_relative_to(resolved,config.paths.derivatives_root):
        raise ValueError(f"Generated project output must be inside derivatives_root: {resolved}")
    return resolved

def load_project_config(path: str|Path|None=None) -> ProjectConfig:
    selected=path or os.environ.get(CONFIG_ENV)
    if not selected: raise ValueError(f"Project configuration is required; pass --project-config or set {CONFIG_ENV}.")
    source=Path(selected).expanduser().resolve()
    if not source.is_file(): raise FileNotFoundError(f"Project configuration was not found: {source}")
    try: data=tomllib.loads(source.read_text(encoding="utf-8"))
    except (tomllib.TOMLDecodeError,OSError) as exc: raise ValueError(f"Unable to parse project configuration {source}: {exc}") from exc
    paths,rig,volume=data.get("paths",{}),data.get("rig",{}),data.get("canonical_volume",{})
    raw=_path(_required(paths,"raw_root","paths"),source.parent); derivatives=_path(_required(paths,"derivatives_root","paths"),source.parent); mice=_path(_required(paths,"mice_csv","paths"),source.parent)
    legacy_value=paths.get("legacy_fucci_tri_root"); legacy=_path(legacy_value,source.parent) if legacy_value else None
    if raw==derivatives or is_relative_to(derivatives,raw) or is_relative_to(raw,derivatives): raise ValueError("raw_root and derivatives_root must be separate, non-nested directories")
    return ProjectConfig(source,ProjectPaths(raw,derivatives,mice,legacy),RigConfig(int(_required(rig,"primary_laser_nm","rig")),int(_required(rig,"optional_laser_nm","rig")),int(_required(rig,"pockels_1_laser_nm","rig")),int(_required(rig,"pockels_2_laser_nm","rig")),str(_required(rig,"chan_a_signal","rig")).lower(),str(_required(rig,"chan_b_signal","rig")).lower()),CanonicalVolumeConfig(int(_required(volume,"imaging_planes","canonical_volume")),int(_required(volume,"flyback_planes","canonical_volume")),float(_required(volume,"z_step_um","canonical_volume")),int(_required(volume,"volumes","canonical_volume"))))
