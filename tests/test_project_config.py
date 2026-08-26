from pathlib import Path
import pytest
from project_config import load_project_config, validate_output_path

def _config(tmp_path: Path, raw="raw", derivatives="derivatives"):
    (tmp_path/"mice.csv").write_text("mouse_id,experimental_group,cohort,raw_mouse_folder\nm,g,c,m\n")
    path=tmp_path/"project.toml"; path.write_text(f'''[paths]\nraw_root="{raw}"\nderivatives_root="{derivatives}"\nmice_csv="mice.csv"\n[rig]\nprimary_laser_nm=1050\noptional_laser_nm=920\npockels_1_laser_nm=920\npockels_2_laser_nm=1050\nchan_a_signal="green"\nchan_b_signal="red"\n[canonical_volume]\nimaging_planes=41\nflyback_planes=1\nz_step_um=5.0\nvolumes=50\n''')
    return path

def test_load_config_and_protect_raw(tmp_path):
    config=load_project_config(_config(tmp_path))
    assert config.rig.primary_laser_nm==1050
    assert validate_output_path(config.paths.derivatives_root/"m"/"run",config).name=="run"
    with pytest.raises(ValueError,match="raw_root"): validate_output_path(config.paths.raw_root/"bad",config)

def test_reject_nested_roots(tmp_path):
    with pytest.raises(ValueError,match="separate"): load_project_config(_config(tmp_path,"root","root/derivatives"))
