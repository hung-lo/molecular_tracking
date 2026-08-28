from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pytest

from legacy_derivatives_plan import build_legacy_derivatives_audit, _collision_status
from project_config import load_project_config


def _project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    raw_root = tmp_path / "raw"
    derivatives_root = tmp_path / "derivatives"
    legacy_root = tmp_path / "Fucci-Tri_avg_images"
    raw_root.mkdir()
    derivatives_root.mkdir()
    legacy_root.mkdir()
    mice_csv = tmp_path / "mice.csv"
    mice_csv.write_text(
        "mouse_id,experimental_group,cohort,raw_mouse_folder,reference_session_or_folder\n"
        "Fucci-Tri_1,Fucci-Tri,2026_05,Fucci-Tri_1,\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "project.toml"
    config_path.write_text(
        f"""
[paths]
raw_root = "{raw_root}"
derivatives_root = "{derivatives_root}"
mice_csv = "{mice_csv}"
legacy_fucci_tri_root = "{legacy_root}"

[rig]
primary_laser_nm = 1050
optional_laser_nm = 920
pockels_1_laser_nm = 920
pockels_2_laser_nm = 1050
chan_a_signal = "green"
chan_b_signal = "red"

[canonical_volume]
imaging_planes = 41
flyback_planes = 1
z_step_um = 5.0
volumes = 50
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path, raw_root, derivatives_root, legacy_root


def _write_catalog(path: Path) -> None:
    rows = [
        {
            "mouse_id": "Fucci-Tri_1",
            "session_id": "session_20260511",
            "acquisition_date": "2026-05-11",
            "acquisition_id": "20260511_R",
            "source_path": "/raw/Fucci-Tri_1/20260511/20260511_R",
            "analysis_included": "True",
            "laser_nm": "1050",
        },
        {
            "mouse_id": "Fucci-Tri_1",
            "session_id": "session_20260526",
            "acquisition_date": "2026-05-26",
            "acquisition_id": "20260526_R",
            "source_path": "/raw/Fucci-Tri_1/20260526/20260526_R",
            "analysis_included": "True",
            "laser_nm": "920",
        },
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mouse_id",
                "session_id",
                "acquisition_date",
                "acquisition_id",
                "source_path",
                "analysis_included",
                "laser_nm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _touch(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def test_collision_status_helper_handles_unique_duplicate_exists_and_missing(tmp_path: Path) -> None:
    unique_target = tmp_path / "clear" / "example.tif"
    duplicate_target = tmp_path / "dup" / "shared.tif"
    existing_target = tmp_path / "exists" / "existing.tif"
    existing_target.parent.mkdir(parents=True, exist_ok=True)
    existing_target.write_text("present", encoding="utf-8")

    assert _collision_status(None, target_counts=Counter()) == "not_applicable"
    assert _collision_status(unique_target, target_counts=Counter({unique_target: 1})) == "clear"
    assert _collision_status(duplicate_target, target_counts=Counter({duplicate_target: 2})) == "duplicate_target"
    assert _collision_status(existing_target, target_counts=Counter({existing_target: 1})) == "exists"


def test_legacy_derivatives_plan_scans_allowlisted_roots_only_and_classifies_scope(tmp_path: Path) -> None:
    config_path, _raw_root, derivatives_root, legacy_root = _project(tmp_path)
    catalog_path = tmp_path / "acquisitions.generated.csv"
    _write_catalog(catalog_path)

    _touch(legacy_root / "1050_data" / "20260511_R.tif")
    _touch(legacy_root / "920_data" / "20260526_R.tif")
    _touch(legacy_root / "2wks_1050_data" / "analysis" / "run" / "output.csv")
    _touch(legacy_root / "1050_small_test_fireants" / "output.tif")
    _touch(legacy_root / "roi_matcher_qc_examples_20260611_01" / "figure.png")
    _touch(legacy_root / "roi_matcher_qc_examples_nonsyn_20260611_02_contours" / "summary.json")
    _touch(legacy_root / "roi_matcher_qc_examples_syn_20260615_01_styled" / "20260615_R.tif")
    _touch(legacy_root / "roi_matcher_qc_examples_syn_20260615_01_styled" / "style.png")

    _touch(legacy_root / "README.md")
    _touch(legacy_root / "core" / "script.py")
    _touch(legacy_root / "docs" / "note.md")
    _touch(legacy_root / ".venv" / "package.py")
    _touch(legacy_root / "unrelated_data" / "file.tif")

    config = load_project_config(config_path)
    inventory_rows, plan_rows, audit_dir, summary = build_legacy_derivatives_audit(
        config,
        acquisition_catalog_path=catalog_path,
        output_dir=derivatives_root / "_catalog" / "phase2a_audit",
        write_outputs=True,
        return_summary=True,
    )

    assert audit_dir == (derivatives_root / "_catalog" / "phase2a_audit").resolve()
    assert (audit_dir / "legacy_derivatives_inventory.csv").is_file()
    assert (audit_dir / "legacy_derivatives_migration_plan.csv").is_file()
    assert summary.included_roots == (
        "1050_data",
        "1050_small_test_fireants",
        "2wks_1050_data",
        "920_data",
        "roi_matcher_qc_examples_20260611_01",
        "roi_matcher_qc_examples_nonsyn_20260611_02_contours",
        "roi_matcher_qc_examples_syn_20260615_01_styled",
    )
    ignored_names = {name for name, _kind in summary.ignored_top_level_entries}
    assert {"README.md", "core", "docs", ".venv", "unrelated_data"}.issubset(ignored_names)
    assert not ignored_names & set(summary.included_roots)

    assert len(inventory_rows) == 8
    assert len(plan_rows) == 8
    assert [row["relative_source_path"] for row in inventory_rows] == sorted(
        row["relative_source_path"] for row in inventory_rows
    )
    assert [row["source_path"] for row in plan_rows] == sorted(row["source_path"] for row in plan_rows)
    assert all(row["action"] == "review_required" for row in plan_rows)

    lookup = {row["relative_source_path"]: row for row in inventory_rows}
    plan_lookup = {row["source_path"]: row for row in plan_rows}

    session_1050 = lookup["1050_data/20260511_R.tif"]
    assert session_1050["target_scope"] == "session"
    assert session_1050["inferred_session_date"] == "2026-05-11"
    assert session_1050["date_token_role"] == "acquisition_date"
    assert session_1050["catalog_session_match"] is True
    assert session_1050["inference_status"] == "resolved"
    assert plan_lookup[session_1050["source_path"]]["collision_status"] == "clear"

    session_920 = lookup["920_data/20260526_R.tif"]
    assert session_920["target_scope"] == "session"
    assert session_920["inferred_session_date"] == "2026-05-26"
    assert session_920["date_token_role"] == "acquisition_date"
    assert session_920["catalog_session_match"] is True
    assert session_920["inference_status"] == "resolved"
    assert plan_lookup[session_920["source_path"]]["collision_status"] == "clear"

    long_run = lookup["2wks_1050_data/analysis/run/output.csv"]
    assert long_run["target_scope"] == "longitudinal"
    assert long_run["inferred_session_date"] == ""
    assert long_run["catalog_session_match"] == "not_applicable"
    assert long_run["inference_status"] == "resolved"
    assert plan_lookup[long_run["source_path"]]["collision_status"] == "clear"

    test_row = lookup["1050_small_test_fireants/output.tif"]
    assert test_row["target_scope"] == "longitudinal"
    assert test_row["catalog_session_match"] == "not_applicable"
    assert test_row["inference_status"] == "resolved"
    assert plan_lookup[test_row["source_path"]]["collision_status"] == "clear"

    qc_row = lookup["roi_matcher_qc_examples_20260611_01/figure.png"]
    assert qc_row["target_scope"] == "longitudinal"
    assert qc_row["catalog_session_match"] == "not_applicable"
    assert plan_lookup[qc_row["source_path"]]["collision_status"] == "not_applicable"

    styled_sessionish = lookup["roi_matcher_qc_examples_syn_20260615_01_styled/20260615_R.tif"]
    assert styled_sessionish["target_scope"] == "longitudinal"
    assert styled_sessionish["inferred_session_date"] == ""
    assert styled_sessionish["date_token_role"] == "acquisition_date"
    assert styled_sessionish["catalog_session_match"] == "not_applicable"
    assert styled_sessionish["inference_status"] == "unmapped"
    assert plan_lookup[styled_sessionish["source_path"]]["collision_status"] == "not_applicable"

    styled_qc = lookup["roi_matcher_qc_examples_syn_20260615_01_styled/style.png"]
    assert styled_qc["target_scope"] == "longitudinal"
    assert styled_qc["inferred_session_date"] == ""
    assert styled_qc["catalog_session_match"] == "not_applicable"
    assert styled_qc["inference_status"] == "unmapped"
    assert plan_lookup[styled_qc["source_path"]]["collision_status"] == "not_applicable"

    assert all(row["collision_status"] == "clear" for row in plan_rows if row["proposed_target"])
    assert all(row["collision_status"] == "not_applicable" for row in plan_rows if not row["proposed_target"])
    assert not any(row["collision_status"] == "duplicate_target" for row in plan_rows)
    assert not any(row["collision_status"] == "exists" for row in plan_rows)
    assert all(row["target_scope"] != "session" or row["product_class"] != "analysis_output" for row in plan_rows)
    assert not any(
        row["source_path"].startswith((legacy_root / "roi_matcher_qc_examples_").as_posix()) and row["target_scope"] == "session"
        for row in plan_rows
    )


def test_legacy_derivatives_plan_marks_timestamped_analysis_runs_as_longitudinal(tmp_path: Path) -> None:
    config_path, _raw_root, derivatives_root, legacy_root = _project(tmp_path)
    catalog_path = tmp_path / "acquisitions.generated.csv"
    _write_catalog(catalog_path)

    _touch(legacy_root / "1050_data" / "analysis" / "demo_registered_roi_pipeline_1050_20260529_154035" / "output.png")
    _touch(legacy_root / "1050_data" / "analysis" / "raw_space_triplet_roi_panels_1050_20260624_155227" / "metadata.csv")

    config = load_project_config(config_path)
    inventory_rows, plan_rows, _audit_dir = build_legacy_derivatives_audit(
        config,
        acquisition_catalog_path=catalog_path,
        output_dir=derivatives_root / "_catalog" / "phase2a_audit",
        write_outputs=False,
    )

    lookup = {row["relative_source_path"]: row for row in inventory_rows}
    plan_lookup = {row["source_path"]: row for row in plan_rows}

    run_row = lookup["1050_data/analysis/demo_registered_roi_pipeline_1050_20260529_154035/output.png"]
    assert run_row["target_scope"] == "longitudinal"
    assert run_row["inferred_session_date"] == ""
    assert run_row["date_token_role"] == "run_timestamp"
    assert run_row["catalog_session_match"] == "not_applicable"
    assert plan_lookup[run_row["source_path"]]["collision_status"] in {"clear", "not_applicable"}

    coincidence_row = lookup["1050_data/analysis/raw_space_triplet_roi_panels_1050_20260624_155227/metadata.csv"]
    assert coincidence_row["target_scope"] == "longitudinal"
    assert coincidence_row["inferred_session_date"] == ""
    assert coincidence_row["date_token_role"] == "run_timestamp"
    assert coincidence_row["catalog_session_match"] == "not_applicable"


def test_legacy_derivatives_plan_refuses_output_outside_derivatives_root(tmp_path: Path) -> None:
    config_path, _raw_root, derivatives_root, legacy_root = _project(tmp_path)
    catalog_path = tmp_path / "acquisitions.generated.csv"
    _write_catalog(catalog_path)
    _touch(legacy_root / "1050_data" / "20260511_R.tif")

    config = load_project_config(config_path)
    with pytest.raises(ValueError, match="inside derivatives_root"):
        build_legacy_derivatives_audit(
            config,
            acquisition_catalog_path=catalog_path,
            output_dir=tmp_path / "outside",
            write_outputs=False,
        )

    _, _, audit_dir = build_legacy_derivatives_audit(
        config,
        acquisition_catalog_path=catalog_path,
        output_dir=derivatives_root / "_catalog" / "phase2a_audit",
        write_outputs=False,
    )
    assert audit_dir == (derivatives_root / "_catalog" / "phase2a_audit").resolve()
