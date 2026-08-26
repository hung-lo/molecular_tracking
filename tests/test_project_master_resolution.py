from __future__ import annotations
import csv
from pathlib import Path

from dataset_catalog import build_manifest_plan
from project_config import load_project_config
import run_daywise_master_pipeline as master

def test_project_master_resolution_records_catalog_provenance(tmp_path,monkeypatch):
    raw=tmp_path/"raw"; derivatives=tmp_path/"derivatives"; raw.mkdir(); derivatives.mkdir()
    mice=tmp_path/"mice.csv"; mice.write_text("mouse_id,experimental_group,cohort,viral_constructs,raw_mouse_folder,reference_session_or_folder\nm,group,cohort,construct,folder,\n",encoding="utf-8")
    config_path=tmp_path/"project.toml"; config_path.write_text(f'''[paths]\nraw_root="{raw}"\nderivatives_root="{derivatives}"\nmice_csv="{mice}"\n[rig]\nprimary_laser_nm=1050\noptional_laser_nm=920\npockels_1_laser_nm=920\npockels_2_laser_nm=1050\nchan_a_signal="green"\nchan_b_signal="red"\n[canonical_volume]\nimaging_planes=41\nflyback_planes=1\nz_step_um=5.0\nvolumes=50\n''',encoding="utf-8")
    config=load_project_config(config_path); catalog_dir=derivatives/"_catalog"; catalog_dir.mkdir()
    row={"mouse_id":"m","experimental_group":"group","cohort":"cohort","session_id":"s_20260820","acquisition_date":"2026-08-20","acquisition_id":"a","source_path":"/raw/a","analysis_included":True,"laser_nm":1050,"pixel_size_x_um":0.7,"pixel_size_y_um":0.7,"z_step_um":5.0}
    catalog=catalog_dir/"acquisitions.generated.csv"
    with catalog.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    base=derivatives/"m"/"sessions"/"20260820"/"1050"
    for path in (base/"segmentation"/"mask.tif",base/"preprocessing"/"red.tif",base/"preprocessing"/"green.tif"):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b"x")
    build_manifest_plan(config,[row],"m",1050,source_catalog=catalog)
    monkeypatch.setattr(master,"run_master_pipeline",lambda cfg:cfg)
    resolved=master.main(["--project-config",str(config_path),"--mouse-id","m","--skip-matching-qc","--skip-quick-plots"])
    assert resolved.mode=="project"
    assert resolved.project_provenance["mouse_id"]=="m"
    assert resolved.project_provenance["experimental_group"]=="group"
    assert resolved.project_provenance["cohort"]=="cohort"
    assert resolved.project_provenance["viral_constructs"]=="construct"
    assert resolved.project_provenance["laser_nm"]==1050
    assert resolved.project_provenance["selected_acquisition_ids"]==["a"]
    assert resolved.project_provenance["effective_spacing_um"]=={"x":0.7,"y":0.7,"z":5.0}
