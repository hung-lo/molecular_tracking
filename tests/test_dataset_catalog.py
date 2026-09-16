from pathlib import Path
import shutil
import pytest
from acquisition_settings_qc import acquisition_settings_qc_table
from dataset_catalog import _is_vol10_acquisition, build_manifest_plan, discover_catalog
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


def test_discovery_recognizes_field_prefixed_acquisition(tmp_path):
    config, raw = _project(tmp_path)
    _acq(raw, "session_20260819", "field100_res1024_ulFoV_zstack100to300_vol50", "square_1050.xml")
    rows, report = discover_catalog(config)
    assert len(rows) == 1 and rows[0]["analysis_included"] is True
    assert not report["errors"]


def test_discovery_preserves_xml_acquisition_with_nonstandard_name(tmp_path):
    config, raw = _project(tmp_path)
    _acq(raw, "session_20260819", "unexpected_acquisition_name", "square_1050.xml")
    rows, report = discover_catalog(config)
    assert len(rows) == 1 and rows[0]["analysis_included"] is True
    assert rows[0]["role"] == "canonical"
    assert not report["errors"]


def test_missing_experiment_xml_acquisition_is_preserved_and_failed(tmp_path):
    config, raw = _project(tmp_path)
    missing = raw / "folder" / "session_20260821" / "filed_vol50"
    missing.mkdir(parents=True)
    rows, report = discover_catalog(config)
    assert len(rows) == 1
    assert rows[0]["role"] == "missing_xml"
    assert rows[0]["settings_qc_pass"] is False
    assert rows[0]["analysis_eligible"] is False
    assert "missing Experiment.xml" in rows[0]["settings_qc_reason"]
    assert any(error["code"] == "missing_experiment_xml" for error in report["row_ineligible"])


def test_vol10_is_token_aware_and_excluded_from_catalog_qc(tmp_path):
    config, raw = _project(tmp_path)
    _acq(raw, "session_20260819", "filed_vol10", "square_1050.xml")
    _acq(raw, "session_20260819", "filed_vol100", "square_1050.xml")
    rows, report = discover_catalog(config)
    by_id = {row["acquisition_id"]: row for row in rows}
    assert _is_vol10_acquisition("vol10") and _is_vol10_acquisition("field_vol10_extra")
    assert not _is_vol10_acquisition("vol100") and not _is_vol10_acquisition("vol105")
    assert by_id["filed_vol10"]["is_vol10_control"] is True
    assert by_id["filed_vol10"]["settings_qc_status"] == "not_applicable_vol10"
    assert by_id["filed_vol10"]["settings_qc_pass"] is None
    assert by_id["filed_vol100"]["is_vol10_control"] is False
    assert report["summary"]["mouse_1"]["vol10_excluded"] == 1


def test_vol10_missing_xml_does_not_block_valid_manifest(tmp_path):
    config, raw = _project(tmp_path)
    _acq(raw, "session_20260819", "filed_vol50", "square_1050.xml")
    (raw / "folder" / "session_20260820" / "filed_vol50").mkdir(parents=True)
    (raw / "folder" / "session_20260821" / "filed_vol10").mkdir(parents=True)
    rows, report = discover_catalog(config)
    assert {row["role"] for row in rows} == {"canonical", "missing_xml", "alignment_only"}
    assert not report["errors"]
    assert any(item["code"] == "missing_experiment_xml" for item in report["row_ineligible"])
    plan, ready = build_manifest_plan(config, rows, "mouse_1")
    assert not ready
    assert "session_20260819" in plan.read_text()
    assert "session_20260820" not in plan.read_text()
    assert "session_20260821" not in plan.read_text()


def test_malformed_xml_does_not_block_valid_manifest(tmp_path):
    config, raw = _project(tmp_path)
    _acq(raw, "session_20260819", "filed_vol50", "square_1050.xml")
    malformed = raw / "folder" / "session_20260820" / "filed_vol50"
    malformed.mkdir(parents=True)
    (malformed / "Experiment.xml").write_text("<Experiment>", encoding="utf-8")
    rows, report = discover_catalog(config)
    assert any(row["role"] == "malformed_xml" for row in rows)
    assert not report["errors"]
    assert any(item["code"] == "malformed_xml" for item in report["row_ineligible"])
    plan, _ = build_manifest_plan(config, rows, "mouse_1")
    assert "session_20260819" in plan.read_text()


def test_pipeline_excluded_fucci_is_kept_in_catalog_but_cannot_make_manifest(tmp_path):
    config, raw = _project(tmp_path)
    config.paths.mice_csv.write_text(
        "mouse_id,experimental_group,cohort,raw_mouse_folder,reference_session_or_folder,pipeline_enabled,pipeline_exclusion_reason\n"
        "Fucci-Tri_2,group,cohort,folder,,false,poor FoV quality\n"
    )
    _acq(raw, "session_20260819", "filed_vol50", "square_1050.xml")
    _acq(raw, "session_20260819", "filed_vol10", "square_1050.xml")
    rows, report = discover_catalog(config)
    canonical = next(row for row in rows if row["acquisition_id"] == "filed_vol50")
    vol10 = next(row for row in rows if row["acquisition_id"] == "filed_vol10")
    assert canonical["settings_qc_status"] == "not_applicable_pipeline_excluded"
    assert canonical["settings_qc_pass"] is None
    assert canonical["analysis_eligible"] is False
    assert vol10["pipeline_enabled"] is False
    assert vol10["pipeline_exclusion_reason"] == "poor FoV quality"
    assert vol10["settings_qc_status"] == "not_applicable_pipeline_excluded"
    assert vol10["analysis_eligible"] is False
    qc_table, qc_summary = acquisition_settings_qc_table(rows)
    qc_vol10 = qc_table.loc[qc_table["acquisition_id"].eq("filed_vol10")].iloc[0]
    assert bool(qc_vol10["pipeline_enabled"]) is False
    assert qc_vol10["settings_qc_status"] == "not_applicable_pipeline_excluded"
    assert bool(qc_vol10["analysis_eligible"]) is False
    assert qc_summary["n_fail"] == 0
    assert not report["errors"]
    assert report["summary"]["Fucci-Tri_2"]["pipeline_exclusion_reason"] == "poor FoV quality"
    with pytest.raises(ValueError, match="excluded from the longitudinal pipeline"):
        build_manifest_plan(config, rows, "Fucci-Tri_2")

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
    config.paths.mice_csv.write_text("mouse_id,experimental_group,cohort,raw_mouse_folder,reference_session_or_folder\nFucci-Tri_1,group,cohort,Fucci-Tri_1,\n")
    acq = raw / "WT_Fucci-Tri_corFront_20260824" / "filed_vol50"
    acq.mkdir(parents=True)
    shutil.copy(FIX / "rectangular_1050.xml", acq / "Experiment.xml")
    rows, report = discover_catalog(config)
    assert rows[0]["mouse_id"] == "Fucci-Tri_1"
    assert rows[0]["acquisition_date"] == "2026-08-24"
    assert rows[0]["discovery_layout"] == "flat"
    assert not any(e["code"] == "unknown_flat_mouse" for e in report["errors"])


def test_unknown_flat_mouse_is_rejected(tmp_path):
    config, raw = _project(tmp_path)
    (raw / "WT_unknown_20260824").mkdir()
    _, report = discover_catalog(config)
    assert any(e["code"] == "unknown_flat_mouse" for e in report["errors"])


def test_duplicate_grouped_and_flat_session_sources_are_rejected(tmp_path):
    config, raw = _project(tmp_path)
    config.paths.mice_csv.write_text("mouse_id,experimental_group,cohort,raw_mouse_folder,reference_session_or_folder\nFucci-Tri_1,group,cohort,Fucci-Tri_1,\n")
    for acq in (raw / "Fucci-Tri_1" / "session_20260824" / "filed_vol50", raw / "WT_Fucci-Tri_corFront_20260824" / "filed_vol50"):
        acq.mkdir(parents=True)
        shutil.copy(FIX / "rectangular_1050.xml", acq / "Experiment.xml")
    rows, report = discover_catalog(config)
    collisions = [e for e in report["errors"] if e["code"] == "duplicate_session_source"]
    assert len(collisions) == 1
    assert collisions[0]["mouse_id"] == "Fucci-Tri_1"
    assert len(collisions[0]["paths"]) == 2
    assert not rows


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


def _manifest_rows(config, count: int = 1) -> list[dict]:
    return [
        {
            "mouse_id": "mouse_1",
            "session_id": f"session_{index}",
            "acquisition_date": f"2026-08-{19 + index:02d}",
            "acquisition_id": f"acq_{index}",
            "source_path": str(config.paths.raw_root / f"source_{index}"),
            "analysis_included": True,
            "laser_nm": 1050,
        }
        for index in range(count)
    ]


def _write_manifest_inputs(config, rows: list[dict], ready: set[int]) -> None:
    for index, row in enumerate(rows):
        base = (
            config.paths.derivatives_root
            / "mouse_1"
            / "sessions"
            / row["acquisition_date"].replace("-", "")
            / "1050"
        )
        if index in ready:
            (base / "preprocessing").mkdir(parents=True, exist_ok=True)
            (base / "segmentation").mkdir(parents=True, exist_ok=True)
            (base / "preprocessing" / "red.tif").touch()
            (base / "preprocessing" / "green.tif").touch()
            (base / "segmentation" / "mask.tif").touch()


@pytest.mark.parametrize("missing_name", ["red.tif", "green.tif"])
def test_manifest_plan_classifies_missing_preprocessing_as_preprocessing_required(
    tmp_path, missing_name
):
    config, _ = _project(tmp_path)
    rows = _manifest_rows(config)
    _write_manifest_inputs(config, rows, {0})
    (config.paths.derivatives_root / "mouse_1" / "sessions" / "20260819" / "1050" / "preprocessing" / missing_name).unlink()

    plan, ready = build_manifest_plan(config, rows, "mouse_1")
    assert not ready
    assert plan.name == "session_manifest_plan.csv"
    assert plan.read_text().splitlines()[-1].endswith("preprocessing_required")


def test_manifest_plan_classifies_missing_mask_as_segmentation_required(tmp_path):
    config, _ = _project(tmp_path)
    rows = _manifest_rows(config)
    base = config.paths.derivatives_root / "mouse_1" / "sessions" / "20260819" / "1050"
    (base / "preprocessing").mkdir(parents=True)
    (base / "preprocessing" / "red.tif").touch()
    (base / "preprocessing" / "green.tif").touch()

    plan, ready = build_manifest_plan(config, rows, "mouse_1")
    assert not ready
    assert plan.read_text().splitlines()[-1].endswith("segmentation_required")


def test_manifest_plan_all_present_writes_ready_manifest(tmp_path):
    config, _ = _project(tmp_path)
    rows = _manifest_rows(config)
    _write_manifest_inputs(config, rows, {0})

    manifest, ready = build_manifest_plan(config, rows, "mouse_1")
    assert ready
    assert manifest.name == "daywise_session_manifest.csv"
    assert "status" not in manifest.read_text().splitlines()[0]


def test_manifest_plan_mixed_sessions_reports_each_status_and_not_ready(tmp_path):
    config, _ = _project(tmp_path)
    rows = _manifest_rows(config, count=3)
    _write_manifest_inputs(config, rows, {0})
    second_base = config.paths.derivatives_root / "mouse_1" / "sessions" / "20260820" / "1050"
    (second_base / "preprocessing").mkdir(parents=True)
    (second_base / "preprocessing" / "red.tif").touch()
    (second_base / "preprocessing" / "green.tif").touch()

    plan, ready = build_manifest_plan(config, rows, "mouse_1")
    assert not ready
    assert plan.name == "session_manifest_plan.csv"
    statuses = {
        row["session_id"]: row["status"]
        for row in __import__("csv").DictReader(plan.open(newline=""))
    }
    assert statuses == {
        "session_0": "ready",
        "session_1": "segmentation_required",
        "session_2": "preprocessing_required",
    }
