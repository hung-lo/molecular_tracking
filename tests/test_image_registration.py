import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from affine_overlap_matcher import VoxelSpacing, extract_roi_features
from image_registration import (
    ImageTransform, warp_volume, match_transformed_masks, correspondence_changes,
    identity_conflicts, classify_identity_conflicts, identity_guard_reasons,
    guard_reasons, prepare_images, fit_image_affine, fit_smooth_field, image_quality,
)


def test_nonrigid_coordinate_direction_roundtrip_and_label_warp():
    shape = (17, 32, 40)
    zz, yy, xx = np.indices(shape)
    flow = np.zeros((3,*shape), np.float32)
    flow[1] = .5*np.sin(xx/12)
    flow[2] = .2*np.cos(yy/10)
    tr = ImageTransform(np.array([[1,0,0],[.03,1,0],[0,.01,1]]),np.array([1,2,-1]),flow,np.ones(3))
    points = np.array([[5.,12.,13.],[9.,18.,25.]])
    assert np.allclose(tr.inverse(tr.apply(points)),points,atol=1e-4)
    ramp = (zz*100+yy*10+xx).astype(float)
    warped = warp_volume(ramp,tr,shape)
    q=np.array([[8,15,20]])
    expected=ndi.map_coordinates(ramp,tr.inverse(q).T,order=1,prefilter=False)
    assert warped[8,15,20] == pytest.approx(expected[0])
    labels=np.zeros(shape,np.uint16);labels[5:10,10:16,12:18]=23
    warped_labels=warp_volume(labels,tr,shape,order=0)
    assert set(np.unique(warped_labels))=={0,23}


def test_recomputed_overlap_and_original_gates_recover_known_translation():
    a=np.zeros((16,32,40),np.uint16)
    a[4:9,8:14,9:15]=7;a[6:12,20:27,25:32]=31
    b=ndi.shift(a,(1,2,3),order=0)
    spacing=VoxelSpacing(5,.7,.7)
    fa=extract_roi_features(a,session_id='a',spacing=spacing)
    fb=extract_roi_features(b,session_id='b',spacing=spacing)
    c,matches,error=match_transformed_masks(a,b,fa,fb,ImageTransform(np.eye(3),[-1,-2,-3]),spacing)
    assert set(zip(matches.label_a,matches.label_b))=={(7,7),(31,31)}
    assert np.allclose(matches.dice,1)
    assert np.allclose(matches.distance_um,0)
    assert error<1e-5


def test_guard_detects_identity_swap_even_when_match_count_increases():
    base=pd.DataFrame(dict(label_a=[1,2],label_b=[11,12],dice=[.9,.9],distance_um=[1.,1.],area_ratio=[1.,1.],ambiguity=[.1,.1]))
    candidate=pd.DataFrame(dict(label_a=[1,2,3],label_b=[12,11,13]))
    changes=correspondence_changes(base,candidate)
    baseline=dict(heldout_ncc=.8,sample_overlap=.99,n_matches=2)
    row=dict(heldout_ncc=.9,sample_overlap=.99,n_matches=3,affine_singular_min=1.,affine_singular_max=1.,
             jacobian_min=1.,jacobian_p01=1.,jacobian_p99=1.,displacement_p99_um=2.,inverse_error_max_um=0.,**changes)
    assert changes['anchor_conflicts']==2
    assert guard_reasons(row,baseline)==[]
    assert 'strong_identity_disruption_widespread' in identity_guard_reasons(
        row, identity_conflicts(base, candidate))
    row.update(**correspondence_changes(base,base),n_matches=2)
    assert guard_reasons(row,baseline)==[]
    assert identity_guard_reasons(row, identity_conflicts(base,base))==[]
    row['jacobian_min']=-.1
    assert 'local_distortion' in guard_reasons(row,baseline)


def test_cycle_supported_identity_change_is_audit_only():
    base=pd.DataFrame(dict(label_a=[1],label_b=[11],dice=[.9],distance_um=[1.],area_ratio=[1.],ambiguity=[.1]))
    candidate=pd.DataFrame(dict(label_a=[1],label_b=[12]))
    bridge=pd.DataFrame(dict(label_a=[11,12],label_b=[101,102]))
    direct=pd.DataFrame(dict(label_a=[1],label_b=[102]))
    conflicts=classify_identity_conflicts(identity_conflicts(base,candidate),bridge,direct)
    assert conflicts.iloc[0].cycle_category == 'new_supported_by_cycle'
    row=correspondence_changes(base,candidate)
    assert identity_guard_reasons(row, conflicts)==[]


def test_image_affine_uses_structure_and_corrects_known_residual():
    rng=np.random.default_rng(4)
    a=ndi.gaussian_filter(rng.normal(size=(17,40,48)).astype(np.float32),1)
    b=ndi.shift(a,(.5,-1.,1.5),order=1)
    initial=ImageTransform(np.eye(3),np.zeros(3))
    fitted,info=fit_image_affine(a,b,initial,np.ones(3))
    q=np.array([[8.,20.,24.]])
    assert np.linalg.norm(fitted.apply(q)-q-[-.5,1.,-1.5])<.3
    spacing=VoxelSpacing(1,1,1)
    assert image_quality(a,b,fitted,np.ones(3),spacing)['heldout_ncc']>.9


def test_physical_units_and_invalid_inputs():
    with pytest.raises(ValueError,match='contrast'):
        prepare_images(np.ones((10,40,40)),np.ones((10,40,40)),VoxelSpacing())
    with pytest.raises(ValueError,match='orientation'):
        ImageTransform(np.diag([-1,1,1]),np.zeros(3))
    tr=ImageTransform(np.eye(3),[0,4,8])
    volume=np.zeros((10,16,20),np.uint16);volume[4,5,6]=8
    warped=warp_volume(volume,tr,volume.shape,step=(1,2,4),order=0)
    assert warped[4,7,8]==8
    assert np.array_equal(tr.apply([[4,10,24]]),[[4,14,32]])


def test_smooth_registration_improves_known_local_deformation_without_folding():
    rng=np.random.default_rng(24)
    a=ndi.gaussian_filter(rng.normal(size=(17,40,48)).astype(np.float32),1)
    coords=np.indices(a.shape,dtype=np.float32)
    coords[2] += 1.2*np.sin(coords[1]/15)
    b=ndi.map_coordinates(a,coords,order=1,prefilter=False)
    initial=ImageTransform(np.eye(3),np.zeros(3));spacing=VoxelSpacing(5,5,5)
    before=image_quality(a,b,initial,np.ones(3),spacing)
    fitted=fit_smooth_field(a,b,initial,np.ones(3),spacing)
    after=image_quality(a,b,fitted,np.ones(3),spacing)
    assert after['heldout_ncc'] > before['heldout_ncc']+.02
    assert after['jacobian_min']>0
    points=np.array([[8.,20.,24.],[10.,15.,30.]])
    assert np.allclose(fitted.inverse(fitted.apply(points)),points,atol=1e-3)


def test_evaluator_refuses_overwriting_output(tmp_path):
    from tools.evaluate_image_registration import main
    pairs=tmp_path/'pairs.csv'
    pd.DataFrame([dict(matching_dir='/unused',session_a='a',session_b='b')]).to_csv(pairs,index=False)
    output=tmp_path/'existing';output.mkdir()
    sentinel=output/'keep.txt';sentinel.write_text('original')
    with pytest.raises(FileExistsError):
        main(['--pairs',str(pairs),'--output',str(output)])
    assert sentinel.read_text()=='original'
