"""Discover ThorImage acquisitions and build deterministic multi-mouse catalogs."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Any

from project_config import ProjectConfig, validate_output_path
from thorimage_xml import ThorImageMetadata, ThorImageParseError, parse_experiment_xml

NULL_COMPAT = {"TBD", "NA", "N/A"}

@dataclass(frozen=True)
class Mouse:
    values: dict[str, str]
    @property
    def mouse_id(self): return self.values["mouse_id"]
    @property
    def raw_mouse_folder(self): return self.values["raw_mouse_folder"]

def load_mice(path: str | Path) -> list[Mouse]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Mouse metadata was not found: {source}")
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"mouse_id", "experimental_group", "cohort", "raw_mouse_folder"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Mouse metadata must contain: {', '.join(sorted(required))}")
    seen: set[str] = set()
    output = []
    for row in rows:
        normalized = {k: ("" if str(v or "").strip().upper() in NULL_COMPAT else str(v or "").strip()) for k,v in row.items()}
        if normalized["mouse_id"] in seen:
            raise ValueError(f"Duplicate mouse_id: {normalized['mouse_id']}")
        seen.add(normalized["mouse_id"]); output.append(Mouse(normalized))
    return output

def _active(value) -> bool: return value is not None and (value.start > 0 or value.stop > 0)

def _classification(name: str, meta: ThorImageMetadata, config: ProjectConfig) -> tuple[str,bool,int|None,list[str]]:
    low = name.lower(); warnings: list[str] = []
    if "_vol10" in low:
        return "alignment_only", False, None, warnings
    if re.search(r"(?:^|_)vol5(?:_|$)", low) or any(token in low for token in ("dark", "ome", "rawformat", "raw_format", "singlez", "singelz")) or meta.z_steps <= 1:
        return "auxiliary_or_test", False, None, warnings
    expected_frames = (config.canonical_volume.imaging_planes + config.canonical_volume.flyback_planes) * config.canonical_volume.volumes
    canonical = (meta.z_steps == config.canonical_volume.imaging_planes and meta.flyback_frames == config.canonical_volume.flyback_planes and abs(meta.z_step_um-config.canonical_volume.z_step_um)<1e-6 and meta.timepoints == config.canonical_volume.volumes and meta.streaming_frames == expected_frames and meta.experiment_status.lower() in {"complete","completed"})
    if not canonical:
        return "noncanonical", False, None, warnings
    expected = 920 if low.endswith("_laser920") else 1050
    p1 = meta.pockels[0] if len(meta.pockels)>0 else None; p2 = meta.pockels[1] if len(meta.pockels)>1 else None
    p920, p1050 = (_active(p1), _active(p2))
    if p920 == p1050 or (expected == 920 and not p920) or (expected == 1050 and not p1050):
        warnings.append("laser_folder_pockels_mismatch")
        return "ambiguous", False, expected, warnings
    if len(meta.pockels) > 2: warnings.append("extra_pockels_nodes")
    return "canonical", True, expected, warnings

def _session_date(name: str) -> str | None:
    match = re.search(r"(\d{8})$", name)
    if not match: return None
    try: return date.fromisoformat(f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}").isoformat()
    except ValueError: return None

def discover_catalog(config: ProjectConfig) -> tuple[list[dict[str,Any]], dict[str,Any]]:
    rows: list[dict[str,Any]]=[]; errors=[]
    mice = load_mice(config.paths.mice_csv)
    for mouse in mice:
        mouse_root = config.paths.raw_root / mouse.raw_mouse_folder
        if not mouse_root.is_dir():
            errors.append({"code":"missing_mouse_folder","mouse_id":mouse.mouse_id,"path":str(mouse_root)}); continue
        for session in sorted(p for p in mouse_root.iterdir() if p.is_dir() and _session_date(p.name)):
            session_date = _session_date(session.name)
            for acquisition in sorted(p for p in session.iterdir() if p.is_dir() and (p/"Experiment.xml").is_file()):
                xml_path=acquisition/"Experiment.xml"
                try: meta=parse_experiment_xml(xml_path)
                except ThorImageParseError as exc:
                    errors.append({"code":"malformed_xml","path":str(xml_path),"message":str(exc)}); continue
                role,included,laser,warnings=_classification(acquisition.name,meta,config)
                if meta.experiment_date != session_date: warnings.append("session_xml_date_mismatch")
                p=list(meta.pockels)+[None,None]
                raw_image=acquisition/"Image_001_001.raw"
                row={"mouse_id":mouse.mouse_id,"experimental_group":mouse.values["experimental_group"],"cohort":mouse.values["cohort"],"session_id":session.name,"acquisition_date":session_date,"acquisition_id":acquisition.name,"source_path":str(acquisition.resolve()),"role":role,"analysis_included":included,"laser_nm":laser,"is_primary":laser==config.rig.primary_laser_nm and included,"xml_date":meta.experiment_date,"software_version":meta.software_version,"experiment_status":meta.experiment_status,"pixel_x":meta.pixel_x,"pixel_y":meta.pixel_y,"width_um":meta.width_um,"height_um":meta.height_um,"pixel_size_x_um":meta.pixel_width_um,"pixel_size_y_um":meta.pixel_height_um,"z_imaging_planes":meta.z_steps,"flyback_planes":meta.flyback_frames,"z_step_um":meta.z_step_um,"timepoints":meta.timepoints,"streaming_frames":meta.streaming_frames,"pockels_920_start_pct":p[0].start if p[0] else None,"pockels_920_stop_pct":p[0].stop if p[0] else None,"pockels_1050_start_pct":p[1].start if p[1] else None,"pockels_1050_stop_pct":p[1].stop if p[1] else None,"pmt_a_gain":meta.pmt_a_gain,"pmt_b_gain":meta.pmt_b_gain,"average_num":meta.average_num,"raw_image_path":str(raw_image.resolve()) if raw_image.exists() else "","warnings":";".join(sorted(set(warnings)))}
                rows.append(row)
    rows.sort(key=lambda r:(r["mouse_id"],r["acquisition_date"],r["acquisition_id"]))
    for mouse in mice:
        sessions=sorted({r["session_id"] for r in rows if r["mouse_id"]==mouse.mouse_id})
        for sid in sessions:
            primary=[r for r in rows if r["mouse_id"]==mouse.mouse_id and r["session_id"]==sid and r["analysis_included"] and r["laser_nm"]==config.rig.primary_laser_nm]
            if len(primary)!=1: errors.append({"code":"missing_or_duplicate_primary","mouse_id":mouse.mouse_id,"session_id":sid,"count":len(primary)})
            companions=[r for r in rows if r["mouse_id"]==mouse.mouse_id and r["session_id"]==sid and r["analysis_included"] and r["laser_nm"]==config.rig.optional_laser_nm]
            if len(companions)>1: errors.append({"code":"duplicate_optional","mouse_id":mouse.mouse_id,"session_id":sid,"count":len(companions)})
            if len(primary)==1 and len(companions)==1:
                fields=("pixel_x","pixel_y","width_um","height_um","z_imaging_planes","flyback_planes","z_step_um","timepoints")
                if any(primary[0][f]!=companions[0][f] for f in fields): errors.append({"code":"paired_geometry_mismatch","mouse_id":mouse.mouse_id,"session_id":sid})
    report={"errors":sorted(errors,key=lambda e:json.dumps(e,sort_keys=True)),"warnings":[{"path":r["source_path"],"codes":r["warnings"].split(";")} for r in rows if r["warnings"]],"summary":catalog_summary(rows)}
    return rows,report

def catalog_summary(rows):
    result={}
    for mouse in sorted({r["mouse_id"] for r in rows}):
        subset=[r for r in rows if r["mouse_id"]==mouse]
        result[mouse]={"sessions":len({r["session_id"] for r in subset if r["analysis_included"] and r["laser_nm"]==1050}),"canonical_1050":sum(r["analysis_included"] and r["laser_nm"]==1050 for r in subset),"canonical_920":sum(r["analysis_included"] and r["laser_nm"]==920 for r in subset),"alignment_only":sum(r["role"]=="alignment_only" for r in subset),"noncanonical":sum(r["role"]=="noncanonical" for r in subset),"auxiliary_or_test":sum(r["role"]=="auxiliary_or_test" for r in subset)}
    return result

def write_catalog(config: ProjectConfig, rows, report) -> Path:
    catalog_dir=validate_output_path(config.paths.derivatives_root/"_catalog",config); catalog_dir.mkdir(parents=True,exist_ok=True)
    acquisitions=catalog_dir/"acquisitions.generated.csv"
    if rows:
        with acquisitions.open("w",encoding="utf-8",newline="") as h:
            writer=csv.DictWriter(h,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    sessions=[]
    for key in sorted({(r["mouse_id"],r["session_id"],r["acquisition_date"]) for r in rows}):
        m,s,d=key; selected=[r for r in rows if (r["mouse_id"],r["session_id"],r["acquisition_date"])==key]
        sessions.append({"mouse_id":m,"session_id":s,"acquisition_date":d,"has_1050":any(r["analysis_included"] and r["laser_nm"]==1050 for r in selected),"has_920":any(r["analysis_included"] and r["laser_nm"]==920 for r in selected)})
    with (catalog_dir/"sessions.generated.csv").open("w",encoding="utf-8",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=["mouse_id","session_id","acquisition_date","has_1050","has_920"]); writer.writeheader(); writer.writerows(sessions)
    (catalog_dir/"validation_report.json").write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    mice_rows = [mouse.values for mouse in load_mice(config.paths.mice_csv)]
    with (catalog_dir / "mice.validated.csv").open("w", encoding="utf-8", newline="") as h:
        writer = csv.DictWriter(h, fieldnames=list(mice_rows[0]))
        writer.writeheader()
        writer.writerows(mice_rows)
    return catalog_dir

def build_manifest_plan(config: ProjectConfig, rows, mouse_id: str, laser_nm: int=1050) -> tuple[Path,bool]:
    selected=[r for r in rows if r["mouse_id"]==mouse_id and r["analysis_included"] and r["laser_nm"]==laser_nm]
    selected.sort(key=lambda r:(r["acquisition_date"],r["session_id"]))
    if not selected: raise ValueError(f"No canonical {laser_nm} acquisitions for mouse {mouse_id}")
    manifest_dir=validate_output_path(config.paths.derivatives_root/mouse_id/"longitudinal"/str(laser_nm)/"manifests",config); manifest_dir.mkdir(parents=True,exist_ok=True)
    planned=[]; ready=True
    for index,row in enumerate(selected):
        session_base=config.paths.derivatives_root/mouse_id/"sessions"/row["acquisition_date"].replace("-","")/str(laser_nm)
        mask=session_base/"segmentation"/"mask.tif"; green=session_base/"preprocessing"/"green.tif"; red=session_base/"preprocessing"/"red.tif"
        status="ready" if all(p.is_file() for p in (mask,red,green)) else "preprocessing_required"; ready &= status=="ready"
        planned.append({"session_index":index,"session_id":row["session_id"],"acquisition_date":row["acquisition_date"],"mask_path":str(mask),"red_image_path":str(red),"green_image_path":str(green),"required":"true","status":status})
    filename="daywise_session_manifest.csv" if ready else "session_manifest_plan.csv"; path=manifest_dir/filename
    fields=["session_index","session_id","acquisition_date","mask_path","red_image_path","green_image_path","required"] + ([] if ready else ["status"])
    with path.open("w",encoding="utf-8",newline="") as h:
        writer=csv.DictWriter(h,fieldnames=fields); writer.writeheader(); writer.writerows(planned)
    return path,ready
