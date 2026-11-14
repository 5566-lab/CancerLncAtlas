import numpy as np
import pytest
from cla_engine.multimodal_relation_y import summarize_effect, assay_bit
from cla_engine.functional_fusion import validate_evidence_strata

def test_repeated_identical_readout_does_not_multiply_weight():
    original=summarize_effect([[.2,-.8]],[[True,True]],[[0],[1]])
    repeated=summarize_effect([[.2,.2,-.8]],[[True,True,True]],[[0,1],[2]])
    assert original[0][0]==pytest.approx(np.sqrt(.34))
    np.testing.assert_allclose(original[0],repeated[0])
    np.testing.assert_allclose(original[1],repeated[1])
    assert repeated[3][0]==2

def test_missing_member_evidence_is_not_zero_or_negative():
    rho=np.array([[np.nan,.6],[np.nan,np.nan],[.2,-.4]])
    available=np.isfinite(rho)
    rms,signed,weight,count=summarize_effect(rho,available,[[0],[1]])
    assert rms[0]==pytest.approx(.6)
    assert np.isnan(rms[1]) and count[1]==0
    assert signed[2]==pytest.approx(-.1)
    assert rms[2]==pytest.approx(np.sqrt(.1))
    assert weight[1].sum()==0

def test_whole_pathway_and_component_outputs_keep_distinct_values():
    rho=np.array([[.8,.2],[np.nan,.4]])
    whole=summarize_effect(rho,np.isfinite(rho)&[True,False],[[0],[1]])[0]
    member=summarize_effect(rho,np.isfinite(rho)&[False,True],[[0],[1]])[0]
    np.testing.assert_allclose(whole,[.8,np.nan],equal_nan=True)
    np.testing.assert_allclose(member,[.2,.4])

def test_invalid_association_or_duplicate_partition_rejected():
    with pytest.raises(ValueError,match='AXIS_OR_VALUE'):
        summarize_effect([[1.1]],[[True]],[[0]])
    with pytest.raises(ValueError,match='DUPLICATE_PARTITION'):
        summarize_effect([[.1,.2]],[[True,True]],[[0,1],[1]])

def test_assay_uses_measurement_not_display_category():
    assert assay_bit('phosphoproteomics')==4
    assert assay_bit('RNA')==1
    assert assay_bit('total_protein')==2
    assert assay_bit('DNA methylation')==8

def test_strata_cannot_silently_replace_missing_member_with_zero():
    families=['equivalent','component','RNA','PROTEIN','PHOSPHO','METHYLATION','CNV','VARIANT','OTHER']
    a={'observed':np.array([[1,0]]),'readout_count':np.array([[1,0]]),'observed_assay_bits':np.array([[4,0]])}
    s={'signed_effect':np.array([[.3,np.nan]])}
    for x in families:
        s[x+'_rms']=np.array([[.3 if x in ['component','PHOSPHO'] else np.nan,np.nan]])
        s[x+'_readouts']=np.array([[int(x in ['component','PHOSPHO']),0]],np.uint16)
    validate_evidence_strata(s,a)
    s['component_rms'][0,1]=0
    with pytest.raises(ValueError,match='MISSINGNESS'):validate_evidence_strata(s,a)
