from scripts.audit_tcpa_cptac_specimens import specimen_match

def row(vial="TCGA-AA-0001-01A",**kwargs):
    r=dict(pool="No",taxon="Homo sapiens",sample_type="Primary Tumor",sample_status="Qualified",
           case_status="Qualified",aliquot_status="Qualified",sample_submitter_id=vial,aliquot_submitter_id=vial+"-11-TEST-30")
    r.update(kwargs);return r

def test_matching_patient_with_different_vial_is_not_a_specimen_match():
    assert specimen_match([row("TCGA-AA-0001-01B")],"TCGA-AA-0001-01A")[0]=="SAMPLE_VIAL_MISMATCH"
    assert specimen_match([row(),row("TCGA-AA-0001-01B")],"TCGA-AA-0001-01A")[0]=="MULTIPLE_SAMPLE_VIALS_UNRESOLVED"

def test_pools_and_failed_samples_cannot_prove_identity():
    for invalid in [row(pool="Yes"),row(sample_type="Solid Tissue Normal"),row(aliquot_status="Disqualified")]:
        assert specimen_match([invalid],"TCGA-AA-0001-01A")[0]=="NO_QUALIFIED_PRIMARY_TUMOR_METADATA"

def test_technical_aliquots_of_same_vial_retain_their_identifiers():
    status,vials,aliquots=specimen_match([row(),row(aliquot_submitter_id="TCGA-AA-0001-01A-12-TEST-30")],"TCGA-AA-0001-01A")
    assert status=="SAME_VIAL_IN_PDC_COHORT_METADATA" and len(vials)==1 and len(aliquots)==2
