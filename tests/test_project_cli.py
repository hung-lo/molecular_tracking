from pathlib import Path
import pytest
from project_cli import resolve_selection

def test_explicit_legacy_path(tmp_path):
    dataset=tmp_path/"legacy"; dataset.mkdir()
    context=resolve_selection(dataset=dataset)
    assert context.dataset_dir==dataset.resolve()
def test_bare_and_mixed_selections_rejected(tmp_path):
    with pytest.raises(ValueError,match="requires"): resolve_selection()
    dataset=tmp_path/"legacy"; dataset.mkdir()
    with pytest.raises(ValueError,match="either"): resolve_selection(dataset=dataset,mouse_id="m")


def test_project_selection_rejects_pipeline_excluded_mouse(tmp_path):
    raw=tmp_path/"raw"; derivatives=tmp_path/"derivatives"; raw.mkdir(); derivatives.mkdir()
    mice=tmp_path/"mice.csv"
    mice.write_text(
        "mouse_id,experimental_group,cohort,raw_mouse_folder,reference_session_or_folder,pipeline_enabled,pipeline_exclusion_reason\n"
        "Fucci-Tri_2,g,c,folder,,false,poor FoV quality\n",
        encoding="utf-8",
    )
    config=tmp_path/"project.toml"
    config.write_text(
        f"[paths]\nraw_root=\"{raw}\"\nderivatives_root=\"{derivatives}\"\nmice_csv=\"{mice}\"\n"
        "[rig]\nprimary_laser_nm=1050\noptional_laser_nm=920\npockels_1_laser_nm=920\npockels_2_laser_nm=1050\nchan_a_signal=\"green\"\nchan_b_signal=\"red\"\n"
        "[canonical_volume]\nimaging_planes=41\nflyback_planes=1\nz_step_um=5\nvolumes=50\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="poor FoV quality"):
        resolve_selection(project_config=config, mouse_id="Fucci-Tri_2")
