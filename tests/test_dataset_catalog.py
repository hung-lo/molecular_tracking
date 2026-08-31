from pathlib import Path
import shutil
from dataset_catalog import build_manifest_plan, discover_catalog
from project_config import load_project_config

FIX=Path(__file__).parent/"fixtures"/"thorimage"

def _project(tmp_path):
    raw=tmp_path/"raw"; derivatives=tmp_path/"derivatives"; raw.mkdir()
    mice=tmp_path/"mice.csv"; mice.write_text("mouse_id,experimental_group,cohort,raw_mouse_folder,reference_session_or_folder\nmouse_1,group,cohort,folder,\n")
    cfg=tmp_path/"project.toml"; cfg.write_text(f'''[paths]\nraw_root="{raw}"\nderivatives_root="{derivatives}"\nmice_csv="{mice}"\n[rig]\nprimary_laser_nm=1050\noptional_laser_nm=920\npockels_1_laser_nm=920\npockels_2_laser_nm=1050\nchan_a_signal="green"\nchan_b_signal="red"\n[canonical_volume]\nimaging_planes=41\nflyback_planes=1\nz_step_um=5.0\nvolumes=50\n''')
    return load_project_config(cfg),raw

def _acq(root,session,name,fixture):
    path=root/"folder"/session/name; path.mkdir(parents=True,exist_ok=True); shutil.copy(FIX/fixture,path/"Experiment.xml")

def test_discovery_uses_mouse_mapping_and_optional_920(tmp_path):
    config,raw=_project(tmp_path)
    _acq(raw,"session_20260819","filed_vol50","square_1050.xml")
    rows,report=discover_catalog(config)
    assert len(rows)==1 and rows[0]["mouse_id"]=="mouse_1" and rows[0]["laser_nm"]==1050
    assert not report["errors"]

def test_alignment_and_pairing_and_plan(tmp_path):
    config,raw=_project(tmp_path)
    _acq(raw,"session_20260820","filed_vol50","rectangular_1050.xml")
    _acq(raw,"session_20260820","filed_vol50_laser920","rectangular_920.xml")
    _acq(raw,"session_20260820","filed_vol10_001","rectangular_1050.xml")
    rows,report=discover_catalog(config)
    assert [r["role"] for r in rows].count("alignment_only")==1
    assert sum(r["analysis_included"] for r in rows)==2
    assert not report["errors"]
    plan,ready=build_manifest_plan(config,rows,"mouse_1")
    assert plan.name=="session_manifest_plan.csv" and not ready
    assert config.paths.derivatives_root in plan.parents and config.paths.raw_root not in plan.parents


def test_discovery_supports_flat_incoming_sessions_and_alias(tmp_path):
    config, raw = _project(tmp_path)
    flat = raw / "WT_Fucci-Tri_corFront_20260824"
    acq = flat / "filed_vol50"
    acq.mkdir(parents=True)
    shutil.copy(FIX / "rectangular_1050.xml", acq / "Experiment.xml")
    rows, report = discover_catalog(config)
    # This fixture project has only mouse_1 metadata, so the historical alias
    # is correctly reported as unknown rather than creating a new mouse.
    assert not rows
    assert any(e["code"] == "unknown_flat_mouse" for e in report["errors"])


def test_flat_session_date_is_anchored_and_calendar_valid(tmp_path):
    config, raw = _project(tmp_path)
    # Match the configured mouse and ensure unrelated date-like names are ignored.
    flat = raw / "WT_mouse_1_20260820"
    acq = flat / "filed_vol50"
    acq.mkdir(parents=True)
    shutil.copy(FIX / "rectangular_1050.xml", acq / "Experiment.xml")
    (raw / "unrelated_20260820").mkdir()
    (raw / "WT_mouse_1_20261301").mkdir()
    rows, report = discover_catalog(config)
    assert len(rows) == 1 and rows[0]["discovery_layout"] == "flat"
    assert rows[0]["acquisition_date"] == "2026-08-20"
    assert any(w["code"] == "ignored_raw_root_entry" for w in report["warnings"])
    assert any(w["code"] == "invalid_flat_session_date" for w in report["warnings"])
