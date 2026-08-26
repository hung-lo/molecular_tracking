from pathlib import Path
from thorimage_xml import parse_experiment_xml

FIXTURES=Path(__file__).parent/"fixtures"/"thorimage"
def test_square_1050_metadata():
    m=parse_experiment_xml(FIXTURES/"square_1050.xml")
    assert (m.pixel_x,m.pixel_y,m.z_steps,m.flyback_frames,m.timepoints,m.streaming_frames)==(1024,1024,41,1,50,2100)
    assert m.pixel_width_um==m.pixel_height_um==710/1024
    assert m.pockels[0].start==0 and m.pockels[1].start==50
def test_rectangular_metadata_and_ramp_values():
    m=parse_experiment_xml(FIXTURES/"rectangular_920.xml")
    assert (m.pixel_x,m.pixel_y,m.width_um,m.height_um)==(1536,768,1065,532.5)
    assert m.pockels[0].start==80 and m.pockels[1].stop==0
    assert len(m.pockels) == 4
