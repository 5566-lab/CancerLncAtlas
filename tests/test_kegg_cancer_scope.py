import copy
import pytest
from cla_engine.kegg_cancer_scope import classify_sources, validate_evidence

def fixture():
    kinds=['Reference','Variant','Pathogen','Reference']
    rows=[dict(source_target_id='S'+str(i),network_id='N'+str(i),network_type=k,
               cancer_scope_status='outside_selected_network_categories')
          for i,k in enumerate(kinds)]
    records={r['network_id']:{'fields':{}} for r in rows}
    records['N2']['fields']['DISEASE']=['H00054 Nasopharyngeal cancer']
    c={'version':'test','malignant_disease_ids':['H00054'],'families':[
        dict(id='repair',network_ids=['N0'],allowed_type='Reference',name='repair',
             pmids=['123'],rationale='Cancer DNA repair',evidence_level='reviewed_cancer_process_component')]}
    return rows,records,c

def test_specific_cancer_disease_and_reference_evidence_survive_category_filter():
    rows,raw,c=fixture()
    result=classify_sources(rows,raw,c)
    assert [r['cancer_relevant'] for r in result]==['True','False','True','False']
    assert result[2]['cancer_evidence_basis']=='official_malignant_disease'
    assert result[1]['cancer_evidence_basis']=='specific_context_not_supported_for_cancer'

def test_count_and_patient_statistics_do_not_admit_mechanisms():
    rows,raw,c=fixture()
    a=classify_sources(rows,raw,c)
    for row in rows:
        row.update(patient_count=100000,lncrna_rho=.99,shared_proteins=9999)
    b=classify_sources(rows,raw,c)
    assert [(r['network_id'],r['cancer_relevant']) for r in a]==[(r['network_id'],r['cancer_relevant']) for r in b]

def test_host_process_citation_cannot_admit_different_variant_context():
    rows,raw,c=fixture()
    c['families'][0]['network_ids']=['N1']
    with pytest.raises(ValueError,match='CONTEXT_TYPE_MISMATCH'):
        classify_sources(rows,raw,c)

def test_missing_record_remains_archival_without_inventing_scope():
    rows,raw,c=fixture();raw.pop('N3')
    r=classify_sources(rows,raw,c)[3]
    assert r['cancer_relevant']=='False'
    assert r['cancer_evidence_basis']=='archival_definition_without_current_record'

def test_evidence_registry_rejects_unretrieved_or_retracted_citation():
    _,_,c=fixture()
    with pytest.raises(ValueError,match='REFERENCE_MISSING'):validate_evidence(c,{})
    ref={'123':dict(pmid='123',title='Cancer mechanism',abstract='Experimental evidence.',
        source_url='https://pubmed.ncbi.nlm.nih.gov/123/',cached_source_sha256='a'*64)}
    assert validate_evidence(c,ref)=={'123'}
    ref['123']['title']='Retraction: Cancer mechanism'
    with pytest.raises(ValueError,match='RETRACTED'):validate_evidence(c,ref)
