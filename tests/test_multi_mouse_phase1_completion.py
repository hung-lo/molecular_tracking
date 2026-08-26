from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest

from dataset_catalog import _classification, build_manifest_plan
from project_cli import ready_manifest_path, resolve_selection
from project_config import load_project_config
from thorimage_xml import parse_experiment_xml

FIX=Path(__file__).parent/"fixtures"/"thorimage"

def make_project(tmp_path:Path):
    raw=tmp_path/"raw"; derivatives=tmp_path/"derivatives"; raw.mkdir(); derivatives.mkdir()
    mice=tmp_path/"mice.csv"; mice.write_text("mouse_id,experimental_group,cohort,viral_constructs,raw_mouse_folder,reference_session_or_folder\nm,g,c,v,folder,\n",encoding="utf-8")
    config_path=tmp_path/"project.toml"; config_path.write_text(f'''[paths]\nraw_root="{raw}"\nderivatives_root="{derivatives}"\nmice_csv="{mice}"\n[rig]\nprimary_laser_nm=1111\noptional_laser_nm=888\npockels_1_laser_nm=888\npockels_2_laser_nm=1111\nchan_a_signal="green"\nchan_b_signal="red"\n[canonical_volume]\nimaging_planes=41\nflyback_planes=1\nz_step_um=5.0\nvolumes=50\n''',encoding="utf-8")
    return config_path,load_project_config(config_path)

def row(session="session_20260820",acquisition="canonical",laser=1111):
    return {"mouse_id":"m","session_id":session,"acquisition_date":"2026-08-20","acquisition_id":acquisition,"source_path":f"/raw/{session}/{acquisition}","analysis_included":True,"laser_nm":laser}

def test_legacy_selection_ignores_forced_wrapper_laser(tmp_path):
    legacy=tmp_path/"legacy920"; legacy.mkdir()
    context=resolve_selection(dataset=legacy,laser_nm=920)
    assert context.mode=="legacy" and context.dataset_dir==legacy.resolve() and context.laser_nm==920

def test_project_selection_validates_mouse(tmp_path):
    config_path,_=make_project(tmp_path)
    with pytest.raises(ValueError,match="Unknown mouse_id"):
        resolve_selection(project_config=config_path,mouse_id="wrong")

def test_config_driven_wavelength_and_vol10_metadata(tmp_path):
    _,config=make_project(tmp_path)
    meta=parse_experiment_xml(FIX/"rectangular_920.xml")
    role,included,laser,warnings=_classification("filed_vol10_laser888",meta,config)
    assert (role,included,laser)==("alignment_only",False,888)
    assert not warnings

def test_duplicate_acquisition_rejected_before_manifest_write(tmp_path):
    _,config=make_project(tmp_path)
    with pytest.raises(ValueError,match="Duplicate included acquisitions"):
        build_manifest_plan(config,[row(acquisition="a"),row(acquisition="b")],"m",1111)
    assert not (config.paths.derivatives_root/"m").exists()

def test_stale_ready_manifest_rejected_when_required_file_disappears(tmp_path):
    config_path,config=make_project(tmp_path)
    catalog_dir=config.paths.derivatives_root/"_catalog"; catalog_dir.mkdir()
    catalog=catalog_dir/"acquisitions.generated.csv"
    full=row(); full.update({"pixel_size_x_um":"1","pixel_size_y_um":"1","z_step_um":"5","experimental_group":"g","cohort":"c"})
    with catalog.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(full)); writer.writeheader(); writer.writerow(full)
    base=config.paths.derivatives_root/"m"/"sessions"/"20260820"/"1111"
    files=[base/"segmentation"/"mask.tif",base/"preprocessing"/"red.tif",base/"preprocessing"/"green.tif"]
    for path in files: path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b"x")
    manifest,ready=build_manifest_plan(config,[full],"m",1111,source_catalog=catalog)
    assert ready
    context=resolve_selection(project_config=config_path,mouse_id="m")
    assert ready_manifest_path(context)==manifest
    files[0].unlink()
    with pytest.raises(FileNotFoundError,match="Stale ready manifest"):
        ready_manifest_path(context)
