"""Discover ThorImage acquisitions and build deterministic multi-mouse catalogs."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date,datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from project_config import ProjectConfig,validate_output_path
from thorimage_xml import ThorImageMetadata,ThorImageParseError,parse_experiment_xml

NULL_COMPAT={"TBD","NA","N/A"}; CATALOG_VERSION="thorimage_catalog_v1"
_FLAT_SESSION_RE = re.compile(r"^WT_(.+)_(\d{8})$")
# Historical acquisition name mapped to the canonical project mouse.
_FLAT_MOUSE_ALIASES = {"Fucci-Tri_corFront": "Fucci-Tri_1"}

@dataclass(frozen=True)
class Mouse:
    values: dict[str,str]
    normalized_null_fields: tuple[str,...]=()
    @property
    def mouse_id(self): return self.values["mouse_id"]
    @property
    def raw_mouse_folder(self): return self.values["raw_mouse_folder"]

def load_mice(path: str|Path) -> list[Mouse]:
    source=Path(path)
    if not source.is_file(): raise FileNotFoundError(f"Mouse metadata was not found: {source}")
    with source.open(encoding="utf-8",newline="") as handle: rows=list(csv.DictReader(handle))
    required={"mouse_id","experimental_group","cohort","raw_mouse_folder","reference_session_or_folder"}
    if not rows or not required.issubset(rows[0]): raise ValueError(f"Mouse metadata must contain: {', '.join(sorted(required))}")
    ids=set(); folders=set(); output=[]
    for row in rows:
        nulls=tuple(sorted(k for k,v in row.items() if str(v or "").strip().upper() in NULL_COMPAT))
        normalized={k:("" if k in nulls else str(v or "").strip()) for k,v in row.items()}
        if normalized["mouse_id"] in ids: raise ValueError(f"Duplicate mouse_id: {normalized['mouse_id']}")
        if normalized["raw_mouse_folder"] in folders: raise ValueError(f"Duplicate raw_mouse_folder mapping: {normalized['raw_mouse_folder']}")
        ids.add(normalized["mouse_id"]); folders.add(normalized["raw_mouse_folder"]); output.append(Mouse(normalized,nulls))
    return output

def _active(value)->bool: return value is not None and (value.start>0 or value.stop>0)

def _wavelength(name: str,meta: ThorImageMetadata,config: ProjectConfig)->tuple[int|None,list[str]]:
    expected=config.rig.optional_laser_nm if name.lower().endswith(f"_laser{config.rig.optional_laser_nm}") else config.rig.primary_laser_nm
    mapped={config.rig.pockels_1_laser_nm:meta.pockels[0] if len(meta.pockels)>0 else None,config.rig.pockels_2_laser_nm:meta.pockels[1] if len(meta.pockels)>1 else None}
    active=[laser for laser,value in mapped.items() if _active(value)]; warnings=[]
    if len(active)==1:
        inferred=active[0]
        if inferred!=expected: warnings.append("laser_folder_pockels_mismatch")
        return inferred,warnings
    warnings.append("both_mapped_lasers_active" if len(active)>1 else "both_mapped_lasers_inactive")
    return expected,warnings

def _classification(name:str,meta:ThorImageMetadata,config:ProjectConfig)->tuple[str,bool,int|None,list[str]]:
    low=name.lower(); laser,warnings=_wavelength(name,meta,config)
    if "_vol10" in low: return "alignment_only",False,laser,warnings
    auxiliary=re.search(r"(?:^|_)vol5(?:_|$)",low) or any(t in low for t in ("dark","ome","rawformat","raw_format","singlez","singelz")) or meta.z_steps<=1
    if auxiliary: return "auxiliary_or_test",False,laser,warnings
    volume=config.canonical_volume; expected_frames=(volume.imaging_planes+volume.flyback_planes)*volume.volumes
    canonical=meta.experiment_status.lower() in {"complete","completed"} and meta.z_steps==volume.imaging_planes and meta.flyback_frames==volume.flyback_planes and abs(meta.z_step_um-volume.z_step_um)<1e-6 and meta.timepoints==volume.volumes and meta.streaming_frames==expected_frames
    if not canonical: return "noncanonical",False,laser,warnings
    if warnings: return "ambiguous",False,laser,warnings
    return "canonical",True,laser,warnings

def _session_date(name:str, *, flat: bool = False)->str|None:
    match = _FLAT_SESSION_RE.fullmatch(name) if flat else re.search(r"(\d{8})$", name)
    if not match: return None
    token = match.group(2) if flat else match.group(1)
    try:
        return date.fromisoformat(f"{token[:4]}-{token[4:6]}-{token[6:]}").isoformat()
    except ValueError:
        return None

@dataclass(frozen=True)
class DiscoveredSession:
    mouse: Mouse
    path: Path
    session_date: str
    discovery_layout: str

def _flat_mouse_id(name: str) -> str | None:
    match = _FLAT_SESSION_RE.fullmatch(name)
    if not match: return None
    return _FLAT_MOUSE_ALIASES.get(match.group(1), match.group(1))

def _discover_sessions(raw_root: Path, mice: list[Mouse], errors: list[dict], warnings: list[dict]) -> list[DiscoveredSession]:
    by_id = {mouse.mouse_id: mouse for mouse in mice}
    discovered: list[DiscoveredSession] = []
    for mouse in mice:
        root = raw_root / mouse.raw_mouse_folder
        if not root.is_dir(): continue
        for session in sorted(p for p in root.iterdir() if p.is_dir() and _session_date(p.name)):
            discovered.append(DiscoveredSession(mouse, session, _session_date(session.name), "grouped"))
    if raw_root.is_dir():
        grouped_names = {mouse.raw_mouse_folder for mouse in mice}
        for session in sorted(p for p in raw_root.iterdir() if p.is_dir()):
            match = _FLAT_SESSION_RE.fullmatch(session.name)
            if not match:
                if session.name not in grouped_names:
                    warnings.append({"code": "ignored_raw_root_entry", "path": str(session)})
                continue
            session_date = _session_date(session.name, flat=True)
            mouse_id = _flat_mouse_id(session.name)
            if session_date is None:
                warnings.append({"code": "invalid_flat_session_date", "path": str(session)})
            elif mouse_id not in by_id:
                errors.append({"code": "unknown_flat_mouse", "path": str(session), "mouse_name": match.group(1)})
            else:
                discovered.append(DiscoveredSession(by_id[mouse_id], session, session_date, "flat"))
    return discovered

def discover_catalog(config:ProjectConfig)->tuple[list[dict[str,Any]],dict[str,Any]]:
    rows=[]; errors=[]; warnings=[]; mice=load_mice(config.paths.mice_csv)
    for mouse in mice:
        for field in mouse.normalized_null_fields: warnings.append({"code":"compat_null_normalized","mouse_id":mouse.mouse_id,"field":field})
    discovered = _discover_sessions(config.paths.raw_root, mice, errors, warnings)
    by_source_key: dict[tuple[str, str], list[DiscoveredSession]] = {}
    for found in discovered:
        by_source_key.setdefault((found.mouse.mouse_id, found.session_date), []).append(found)
    conflicting_keys = set()
    for (mouse_id, session_date), sources in sorted(by_source_key.items()):
        paths = sorted({str(source.path.resolve()) for source in sources})
        if len(paths) > 1:
            conflicting_keys.add((mouse_id, session_date))
            errors.append({"code": "duplicate_session_source", "mouse_id": mouse_id, "acquisition_date": session_date, "paths": paths})
    discovered = [found for found in discovered if (found.mouse.mouse_id, found.session_date) not in conflicting_keys]
    grouped_mouse_ids = {s.mouse.mouse_id for s in discovered if s.discovery_layout == "grouped"}
    for mouse in mice:
        if mouse.mouse_id not in grouped_mouse_ids and not any(s.mouse.mouse_id == mouse.mouse_id and s.discovery_layout == "flat" for s in discovered):
            errors.append({"code":"missing_mouse_folder","mouse_id":mouse.mouse_id,"path":str(config.paths.raw_root / mouse.raw_mouse_folder)})
    for found in sorted(discovered, key=lambda s: (s.mouse.mouse_id, s.session_date, s.path.name)):
        mouse, session, session_date = found.mouse, found.path, found.session_date
        for acq in sorted(p for p in session.iterdir() if p.is_dir() and (p/"Experiment.xml").is_file()):
                xml=acq/"Experiment.xml"
                try: meta=parse_experiment_xml(xml)
                except ThorImageParseError as exc: errors.append({"code":"malformed_xml","path":str(xml),"message":str(exc)}); continue
                role,included,laser,codes=_classification(acq.name,meta,config)
                if meta.experiment_date!=session_date: codes.append("session_xml_date_mismatch")
                p=list(meta.pockels)+[None,None]; raw=acq/"Image_001_001.raw"
                row={"mouse_id":mouse.mouse_id,"experimental_group":mouse.values["experimental_group"],"cohort":mouse.values["cohort"],"session_id":session.name,"acquisition_date":session_date,"discovery_layout":found.discovery_layout,"acquisition_id":acq.name,"source_path":str(acq.resolve()),"role":role,"analysis_included":included,"laser_nm":laser,"is_primary":laser==config.rig.primary_laser_nm and included,"xml_date":meta.experiment_date,"software_version":meta.software_version,"experiment_status":meta.experiment_status,"pixel_x":meta.pixel_x,"pixel_y":meta.pixel_y,"width_um":meta.width_um,"height_um":meta.height_um,"pixel_size_x_um":meta.pixel_width_um,"pixel_size_y_um":meta.pixel_height_um,"z_imaging_planes":meta.z_steps,"flyback_planes":meta.flyback_frames,"z_step_um":meta.z_step_um,"timepoints":meta.timepoints,"streaming_frames":meta.streaming_frames,"pockels_920_start_pct":p[0].start if p[0] else None,"pockels_920_stop_pct":p[0].stop if p[0] else None,"pockels_1050_start_pct":p[1].start if p[1] else None,"pockels_1050_stop_pct":p[1].stop if p[1] else None,"pockels_node_count":len(meta.pockels),"pmt_a_gain":meta.pmt_a_gain,"pmt_b_gain":meta.pmt_b_gain,"average_num":meta.average_num,"raw_image_path":str(raw.resolve()) if raw.exists() else "","warnings":";".join(sorted(set(codes)))}
                rows.append(row)
    rows.sort(key=lambda r:(r["mouse_id"],r["acquisition_date"],r["acquisition_id"]))
    for mouse in mice:
        sessions=sorted({r["session_id"] for r in rows if r["mouse_id"]==mouse.mouse_id})
        for sid in sessions:
            primary=[r for r in rows if r["mouse_id"]==mouse.mouse_id and r["session_id"]==sid and r["analysis_included"] and r["laser_nm"]==config.rig.primary_laser_nm]
            optional=[r for r in rows if r["mouse_id"]==mouse.mouse_id and r["session_id"]==sid and r["analysis_included"] and r["laser_nm"]==config.rig.optional_laser_nm]
            if len(primary)!=1: errors.append({"code":"missing_or_duplicate_primary","mouse_id":mouse.mouse_id,"session_id":sid,"count":len(primary)})
            if len(optional)>1: errors.append({"code":"duplicate_optional","mouse_id":mouse.mouse_id,"session_id":sid,"count":len(optional)})
            if len(primary)==len(optional)==1:
                fields=("pixel_x","pixel_y","width_um","height_um","z_imaging_planes","flyback_planes","z_step_um","timepoints")
                if any(primary[0][f]!=optional[0][f] for f in fields): errors.append({"code":"paired_geometry_mismatch","mouse_id":mouse.mouse_id,"session_id":sid})
    warnings.extend({"path":r["source_path"],"codes":r["warnings"].split(";")} for r in rows if r["warnings"])
    report={"catalog_version":CATALOG_VERSION,"errors":sorted(errors,key=lambda e:json.dumps(e,sort_keys=True)),"warnings":sorted(warnings,key=lambda e:json.dumps(e,sort_keys=True)),"summary":catalog_summary(rows,config)}
    return rows,report

def catalog_summary(rows,config):
    result={}; primary=config.rig.primary_laser_nm; optional=config.rig.optional_laser_nm
    for mouse in sorted({r["mouse_id"] for r in rows}):
        subset=[r for r in rows if r["mouse_id"]==mouse]
        result[mouse]={"sessions":len({r["session_id"] for r in subset if r["analysis_included"] and r["laser_nm"]==primary}),f"canonical_{primary}":sum(r["analysis_included"] and r["laser_nm"]==primary for r in subset),f"canonical_{optional}":sum(r["analysis_included"] and r["laser_nm"]==optional for r in subset),"alignment_only":sum(r["role"]=="alignment_only" for r in subset),"noncanonical":sum(r["role"]=="noncanonical" for r in subset),"auxiliary_or_test":sum(r["role"]=="auxiliary_or_test" for r in subset)}
    return result

def _atomic_text(path:Path,text:str)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=path.parent,delete=False) as handle: handle.write(text); temp=Path(handle.name)
    os.replace(temp,path)

def _atomic_csv(path:Path,rows:list[dict],fields:list[str])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",encoding="utf-8",newline="",dir=path.parent,delete=False) as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore"); writer.writeheader(); writer.writerows(rows); temp=Path(handle.name)
    os.replace(temp,path)

def write_catalog(config,rows,report)->Path:
    output=validate_output_path(config.paths.derivatives_root/"_catalog",config); output.mkdir(parents=True,exist_ok=True)
    if rows:_atomic_csv(output/"acquisitions.generated.csv",rows,list(rows[0]))
    sessions=[]
    for key in sorted({(r["mouse_id"],r["session_id"],r["acquisition_date"]) for r in rows}):
        chosen=[r for r in rows if (r["mouse_id"],r["session_id"],r["acquisition_date"])==key]
        sessions.append({"mouse_id":key[0],"session_id":key[1],"acquisition_date":key[2],f"has_{config.rig.primary_laser_nm}":any(r["analysis_included"] and r["laser_nm"]==config.rig.primary_laser_nm for r in chosen),f"has_{config.rig.optional_laser_nm}":any(r["analysis_included"] and r["laser_nm"]==config.rig.optional_laser_nm for r in chosen)})
    _atomic_csv(output/"sessions.generated.csv",sessions,list(sessions[0]) if sessions else ["mouse_id","session_id","acquisition_date"])
    mice=[m.values for m in load_mice(config.paths.mice_csv)]; _atomic_csv(output/"mice.validated.csv",mice,list(mice[0]))
    _atomic_text(output/"validation_report.json",json.dumps(report,indent=2,sort_keys=True))
    return output

def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()

def build_manifest_plan(config,rows,mouse_id:str,laser_nm:int|None=None,*,source_catalog:Path|None=None,validation_report:dict|None=None)->tuple[Path,bool]:
    laser=int(laser_nm if laser_nm is not None else config.rig.primary_laser_nm); mice={m.mouse_id:m for m in load_mice(config.paths.mice_csv)}
    if mouse_id not in mice: raise ValueError(f"Unknown mouse_id {mouse_id!r}")
    selected=[r for r in rows if r["mouse_id"]==mouse_id and bool(r["analysis_included"]) and int(r["laser_nm"])==laser]
    grouped={}
    for row in selected: grouped.setdefault((row["session_id"],row["acquisition_date"],laser),[]).append(row)
    duplicates=[key for key,value in grouped.items() if len(value)!=1]
    if duplicates: raise ValueError(f"Duplicate included acquisitions prevent manifest generation: {duplicates}")
    if not selected: raise ValueError(f"No canonical {laser} acquisitions for mouse {mouse_id}")
    relevant=[e for e in (validation_report or {}).get("errors",[]) if e.get("mouse_id")==mouse_id]
    if relevant: raise ValueError(f"Catalog validation errors prevent manifest generation for {mouse_id}: {relevant}")
    selected=[grouped[key][0] for key in sorted(grouped,key=lambda k:(k[1],k[0]))]
    primary=[r for r in rows if r["mouse_id"]==mouse_id and bool(r["analysis_included"]) and int(r["laser_nm"])==config.rig.primary_laser_nm]
    override=mice[mouse_id].values.get("reference_session_or_folder","")
    candidates=[r for r in primary if (r["session_id"]==override or r["acquisition_id"]==override or r["source_path"]==override or Path(r["source_path"]).name==override)] if override else sorted(primary,key=lambda r:(r["acquisition_date"],r["session_id"]))[:1]
    if len(candidates)!=1: raise ValueError(f"Reference override {override!r} resolved to {len(candidates)} acquisitions")
    reference=candidates[0]; manifest_dir=validate_output_path(config.paths.derivatives_root/mouse_id/"longitudinal"/str(laser)/"manifests",config); manifest_dir.mkdir(parents=True,exist_ok=True)
    planned=[]; ready=True
    for index,row in enumerate(selected):
        base=config.paths.derivatives_root/mouse_id/"sessions"/row["acquisition_date"].replace("-","")/str(laser)
        mask=base/"segmentation"/"mask.tif"; green=base/"preprocessing"/"green.tif"; red=base/"preprocessing"/"red.tif"; red_ready=red.is_file(); green_ready=green.is_file(); mask_ready=mask.is_file(); status="preprocessing_required" if not (red_ready and green_ready) else "segmentation_required" if not mask_ready else "ready"; ready &= status=="ready"
        planned.append({"session_index":index,"session_id":row["session_id"],"acquisition_date":row["acquisition_date"],"mask_path":str(mask),"red_image_path":str(red),"green_image_path":str(green),"required":"true","status":status})
    catalog=source_catalog.resolve() if source_catalog else config.paths.derivatives_root/"_catalog"/"acquisitions.generated.csv"; catalog_hash=_sha(catalog) if catalog.is_file() else None
    metadata={"mouse_id":mouse_id,"laser_nm":laser,"reference_session_id":reference["session_id"],"reference_acquisition_id":reference["acquisition_id"],"reference_acquisition_path":reference["source_path"],"source_catalog":str(catalog),"catalog_sha256":catalog_hash,"catalog_version":(validation_report or {}).get("catalog_version",CATALOG_VERSION),"generated_utc":datetime.now(timezone.utc).isoformat(),"selected_session_ids":[r["session_id"] for r in selected],"selected_acquisition_ids":[r["acquisition_id"] for r in selected]}
    _atomic_text(manifest_dir/"manifest_metadata.json",json.dumps(metadata,indent=2,sort_keys=True)); _atomic_text(manifest_dir/"manifest_state.json",json.dumps({"ready":ready,"catalog_path":str(catalog),"catalog_sha256":catalog_hash,"manifest_metadata":"manifest_metadata.json"},indent=2,sort_keys=True))
    fields=["session_index","session_id","acquisition_date","mask_path","red_image_path","green_image_path","required"]
    if ready: path=manifest_dir/"daywise_session_manifest.csv"; _atomic_csv(path,planned,fields)
    else: path=manifest_dir/"session_manifest_plan.csv"; _atomic_csv(path,planned,fields+["status"])
    return path,ready
