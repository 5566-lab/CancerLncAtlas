import copy
import pytest
from cla_engine.relation_evidence_bindings import compile_bindings


def fixture():
    return dict(
        backbone=[{'canonical_id':'P','source_target_id':'KEGG:REF','network_id':'N1','network_type':'Reference'},
                  {'canonical_id':'V','source_target_id':'KEGG:VAR','network_id':'N2','network_type':'Variant'}],
        definitions=[{'node_id':'old','canonical_id':'old','integrated_canonical_id':'P'},
                     {'node_id':'P','canonical_id':'P','integrated_canonical_id':'P'},
                     {'node_id':'K','canonical_id':'K','integrated_canonical_id':'K'},
                     {'node_id':'B','canonical_id':'B','integrated_canonical_id':'B'}],
        source_bindings=[{'source_target_id':'REACTOME:R1','canonical_id':'P'}],
        policies=[{'canonical_id':'P','analysis_role':'primary_pathway'},
                  {'canonical_id':'V','analysis_role':'primary_pathway'},
                  {'canonical_id':'K','analysis_role':'context_only','biological_type':'kinase_program','mechanism_kind':'kinase'},
                  {'canonical_id':'B','analysis_role':'context_only','biological_type':'pathway','mechanism_kind':''}],
        alignments=[{'source_canonical_id':'K','target_canonical_id':dst,'admitted_context_link':'True',
                     'relation':'regulator_component_in_mechanism','alignment_evidence_basis':'exact_regulator_molecular_identity',
                     'direct_regulator_proteins':'U1'} for dst in ['P','V']] +
                   [{'source_canonical_id':'B','target_canonical_id':'P','admitted_context_link':'True',
                     'relation':'protein_component_association','alignment_evidence_basis':'protein_set_enrichment','direct_regulator_proteins':''}],
        kinase_components=[{'canonical_id':'K','uniprot_accessions':'U1'}],
        readouts=[{'cohort':'CPTAC','layer':'RNA','column_index':0,'canonical_id':'old',
                   'source_target_id':'REACTOME:R1','target_id':'R1','measurement_layer':'RNA_EXPRESSION'},
                  {'cohort':'CPTAC','layer':'PHOSPHO','column_index':0,'canonical_id':'K',
                   'source_target_id':'KINASE:K','target_id':'K','measurement_layer':'PHOSPHO'},
                  {'cohort':'CPTAC','layer':'RNA','column_index':1,'canonical_id':'B',
                   'source_target_id':'BROAD:B','target_id':'B','measurement_layer':'RNA_EXPRESSION'}])


def test_equivalence_and_member_evidence_have_distinct_roles():
    bound, unbound=compile_bindings(**fixture())
    assert len(bound)==3
    equivalent=next(x for x in bound if x['target_id']=='R1')
    assert equivalent['output_canonical_id']=='P'
    assert equivalent['evidence_role']=='equivalent_pathway_readout'
    kinase=[x for x in bound if x['target_id']=='K']
    assert {x['output_canonical_id'] for x in kinase}=={'P','V'}
    assert all(x['evidence_role']=='molecular_component_readout' for x in kinase)
    assert {x['output_canonical_id']:x['condition_required'] for x in kinase}=={'P':False,'V':True}
    assert all(x['output_canonical_id']!='K' for x in bound)
    assert next(x for x in unbound if x['target_id']=='B')['disposition']=='structural_context'


def test_only_protein_overlap_cannot_create_supervision():
    x=fixture();x['alignments']=x['alignments'][-1:]
    bound,_=compile_bindings(**x)
    assert len(bound)==1 and bound[0]['target_id']=='R1'


def test_one_subunit_cannot_identify_an_entire_complex():
    x=fixture();x['kinase_components'].append({'canonical_id':'K','uniprot_accessions':'U2'})
    bound,disp=compile_bindings(**x)
    assert len(bound)==1
    assert next(r for r in disp if r['target_id']=='K')['disposition']=='structural_context'


def test_conflicting_readout_identity_is_not_silently_remapped():
    x=fixture();x['readouts'][0]['canonical_id']='B'
    with pytest.raises(ValueError,match='IDENTITY_CONFLICT'):compile_bindings(**x)


def test_duplicate_array_column_and_ambiguous_alias_rejected():
    x=fixture();x['readouts'].append(copy.deepcopy(x['readouts'][0]))
    with pytest.raises(ValueError,match='DUPLICATE_SOURCE_COLUMN'):compile_bindings(**x)
    x=fixture();x['source_bindings'].append({'source_target_id':'REACTOME:R1','canonical_id':'B'})
    with pytest.raises(ValueError,match='AMBIGUOUS_SOURCE_ALIAS'):compile_bindings(**x)


def test_routes_do_not_depend_on_statistics_or_display_category():
    x=fixture();expected=compile_bindings(**x)
    for row in x['readouts']:row.update(rho=1,p=0,q=0,label=1)
    x['policies'][2]['roles']='State'
    assert compile_bindings(**x)==expected
