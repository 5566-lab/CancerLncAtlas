
import pandas as pd
import pytest
from scripts.score_tcpa_cptac_paired_protein_pilot import cohort_pairs,protein_source

def test_coad_read_and_centers_are_not_silently_pooled():
    p=pd.DataFrame(dict(dataset=["COADREAD_VU_protein"]*2+["OV_JHU_protein","OV_PNNL_protein"],
        in_current_rppa_rna_scope=[True]*4,cancer_id=["COAD","READ","OV","OV"],patient_id=["c","r","o","o"]))
    assert cohort_pairs(p,"COADREAD_VU","COAD").patient_id.tolist()==["c"]
    assert cohort_pairs(p,"COADREAD_VU","READ").patient_id.tolist()==["r"]
    assert len(cohort_pairs(p,"OV_JHU","OV"))==1
    with pytest.raises(ValueError,match="INCOMPATIBLE"):cohort_pairs(p,"COADREAD_VU","COADREAD")
    with pytest.raises(ValueError,match="DUPLICATE"):cohort_pairs(pd.concat([p,p]),"OV_PNNL","OV")

def test_center_specific_protein_sources_are_required():
    rows=[dict(cancer="OV",center=c,layer=l,url="source/CDAP_iTRAQ_UnsharedLogRatio_r2.cct") for c,l in [("JHU","protein"),("PNNL","protein"),("PNNL","phospho")]]
    assert protein_source({"files":rows},"OV_JHU")["center"]=="JHU"
    assert protein_source({"files":rows},"OV_PNNL")["layer"]=="protein"
    with pytest.raises(ValueError,match="SOURCE_NOT_UNIQUE"):protein_source({"files":rows+rows},"OV_PNNL")

def test_unresolved_precursor_measurements_cannot_enter_calibration():
    r=dict(cancer="COADREAD",center="VU",layer="protein",url="source/CDAP_UnsharedPrecursorArea_r2.cct")
    with pytest.raises(ValueError,match="UNRESOLVED_COAD"):protein_source({"files":[r]},"COADREAD_VU")
    r.update(cancer="BRCA",center="BI")
    with pytest.raises(ValueError,match="UNREVIEWED"):protein_source({"files":[r]},"BRCA_BI")
