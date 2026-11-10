import hashlib,json
import numpy as np
import pytest
from cla_engine.functional_y import load_functional_context
from cla_engine.functional_fusion import ARRAY_NAMES,FORMAT,validate_fused_arrays

def fixture_arrays():
    a={k:np.zeros((1,4),dtype=np.float64 if k in ['combined_p','combined_q','rms_effect','max_abs_effect'] else np.uint16 if k.endswith('_count') else np.uint8) for k in ARRAY_NAMES}
    a.update(combined_p=np.array([[np.nan,.8,.01,.0001]]),combined_q=np.array([[np.nan,.9,.08,.005]]),rms_effect=np.array([[np.nan,.02,.16,.4]]),max_abs_effect=np.array([[np.nan,.02,.16,.4]]))
    for k,v in {'observed':[0,1,1,1],'membership':[0,0,1,2],'observed_assay_bits':[0,1,1,1],'cohort_count':[0,1,1,2],'calibration_method':[0,1,2,2],'direction':[0,0,1,1],'readout_count':[0,1,2,3],'distinct_readout_count':[0,1,2,3],'positive_evidence_count':[0,0,1,3]}.items():a[k][0]=v
    return a

def write_context(tmp_path):
    (tmp_path/'Y_DEFINITION.json').write_text(json.dumps({'format':FORMAT,'task_weights':{'Pathway':1.,'State':.5}}))
    p=tmp_path/'BRCA/train_label_fold2';p.mkdir(parents=True)
    (p/'AXES.json').write_text(json.dumps({'lncrna_ids':['L1'],'canonical_ids':['S0','P1','S2','P3'],'target_roles':['State','Pathway','State','Pathway|State'],'primary_tasks':['State','Pathway','State','Pathway']}))
    (p/'SPLIT.json').write_text(json.dumps({'TCGA':{'source_patients':['A'],'label_patients':['B']},'CPTAC':{'source_patients':['C'],'label_patients':['D']}}))
    for f in ['INPUT_REFERENCES.json','DEPENDENCE_REFERENCES.json','MULTIPLICITY.json']:(p/f).write_text('[]')
    (p/'READOUT_INDEX.tsv').write_text('canonical_id\ttarget_id\n')
    np.savez_compressed(p/'Y.npz',**fixture_arrays())
    (p/'COMPLETE.json').write_text(json.dumps({'format':FORMAT,'cancer_id':'BRCA','context':'train_label_fold2','files':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir()}}))
    return p

def test_fused_significance_can_accumulate_within_one_assay():
    a=fixture_arrays();validate_fused_arrays(a,(1,4))
    assert a['observed_assay_bits'][0,3]==1 and a['membership'][0,3]==2
    assert a['individually_strict_assay_bits'][0,3]==0

def test_loader_preserves_unknown_and_has_no_invented_negatives(tmp_path):
    write_context(tmp_path);ctx=load_functional_context(tmp_path,'BRCA','train_label_fold2');m=ctx.supervision_masks()
    assert not m['observed'][0,0] and m['unlabelled'][0,1]
    assert m['weak_positive'][0,2] and m['strong_positive'][0,3]
    assert not m['confirmed_negative'].any()
    with pytest.raises(ValueError,match='UNLABELLED_IS_NOT_NEGATIVE'):ctx.binary_labels()
    np.testing.assert_array_equal(ctx.task_weights(),[.5,1,.5,1])

def test_new_fdr_requirement_applies_to_state_too():
    a=fixture_arrays();a['membership'][0,1]=2;a['direction'][0,1]=1
    with pytest.raises(ValueError,match='SUPPORT_CRITERION'):validate_fused_arrays(a,(1,4))

def test_observed_zero_p_is_not_missing_but_missing_cannot_be_zero_filled():
    a=fixture_arrays();a['combined_p'][0,3]=0;a['combined_q'][0,3]=0;validate_fused_arrays(a,(1,4))
    a['combined_p'][0,0]=0
    with pytest.raises(ValueError,match='MISSINGNESS'):validate_fused_arrays(a,(1,4))

def test_fused_dispatch_checks_the_actual_array_file(tmp_path):
    p=write_context(tmp_path);(p/'Y.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='CHECKSUM'):load_functional_context(tmp_path,'BRCA','train_label_fold2')
