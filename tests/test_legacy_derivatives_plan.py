from __future__ import annotations

import csv
from pathlib import Path

import pytest

from legacy_derivatives_plan import build_legacy_derivatives_audit
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
            "session_id": "session_20260624",
            "acquisition_date": "2026-06-24",
            "acquisition_id": "20260624_R",
            "source_path": "/raw/Fucci-Tri_1/20260624/20260624_R",
            "analysis_included": "True",
            "laser_nm": "1050",
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


def test_legacy_derivatives_plan_distinguishes_session_and_run_timestamp(tmp_path: Path) -> None:
    config_path, _raw_root, derivatives_root, legacy_root = _project(tmp_path)
    catalog_path = tmp_path / "acquisitions.generated.csv"
    _write_catalog(catalog_path)

    _touch(legacy_root / "1050_data" / "20260511_R.tif")
    _touch(legacy_root / "1050_data" / "20260511_G.tif")
    _touch(legacy_root / "1050_data" / "20260511_R_cp_masks_cp_v3_nuclei20.tif")
    _touch(legacy_root / "1050_data" / "analysis" / "demo_registered_roi_pipeline_1050_20260529_154035" / "output.png")
    _touch(legacy_root / "1050_data" / "analysis" / "raw_space_triplet_roi_panels_1050_20260624_155227" / "metadata.csv")
    _touch(legacy_root / "1050_small_test_fireants" / "20260511_R.tif")

    config = load_project_config(config_path)
    inventory_rows, plan_rows, audit_dir = build_legacy_derivatives_audit(
        config,
        acquisition_catalog_path=catalog_path,
        output_dir=derivatives_root / "_catalog" / "phase2a_audit",
        write_outputs=True,
    )

    assert audit_dir == (derivatives_root / "_catalog" / "phase2a_audit").resolve()
    assert (audit_dir / "legacy_derivatives_inventory.csv").is_file()
    assert (audit_dir / "legacy_derivatives_migration_plan.csv").is_file()
    assert [row["relative_source_path"] for row in inventory_rows] == sorted(
        row["relative_source_path"] for row in inventory_rows
    )

    lookup = {row["relative_source_path"]: row for row in inventory_rows}

    session_row = lookup["1050_data/20260511_R.tif"]
    assert session_row["target_scope"] == "session"
    assert session_row["inferred_session_date"] == "2026-05-11"
    assert session_row["date_token_role"] == "acquisition_date"
    assert session_row["catalog_session_match"] is True
    assert session_row["inference_status"] == "resolved"

    run_row = lookup["1050_data/analysis/demo_registered_roi_pipeline_1050_20260529_154035/output.png"]
    assert run_row["target_scope"] == "longitudinal"
    assert run_row["inferred_session_date"] == ""
    assert run_row["date_token_role"] == "run_timestamp"
    assert run_row["catalog_session_match"] == "not_applicable"
    assert run_row["inference_status"] == "resolved"

    coincidence_row = lookup["1050_data/analysis/raw_space_triplet_roi_panels_1050_20260624_155227/metadata.csv"]
    assert coincidence_row["target_scope"] == "longitudinal"
    assert coincidence_row["inferred_session_date"] == ""
    assert coincidence_row["date_token_role"] == "run_timestamp"
    assert coincidence_row["inference_status"] == "resolved"

    test_tree_row = lookup["1050_small_test_fireants/20260511_R.tif"]
    assert test_tree_row["target_scope"] == "longitudinal"
    assert test_tree_row["inferred_session_date"] == ""
    assert test_tree_row["date_token_role"] == "none"
    assert test_tree_row["inference_status"] == "resolved"
    plan_lookup = {row["source_path"]: row for row in plan_rows}
    assert plan_lookup[lookup["1050_small_test_fireants/20260511_R.tif"]["source_path"]]["proposed_target"].endswith(
        "/Fucci-Tri_1/longitudinal/1050/legacy_import/1050_small_test_fireants/20260511_R.tif"
    )

    assert plan_rows == sorted(plan_rows, key=lambda row: row["source_path"])
    assert all(row["action"] == "review_required" for row in plan_rows)
    assert all(row["target_scope"] != "session" or row["product_class"] != "analysis_output" for row in plan_rows)


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

