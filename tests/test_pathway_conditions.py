import pandas as pd
import pytest
from cla_engine.pathway_conditions import condition_definition,applicability_matrix


def row(nid,name,typ,expanded='1956v1 -> 2475'):
    return {'network_id':nid,'canonical_id':'P'+nid,'network_name':name,'network_type':typ,
            'mechanism_definition':name,'expanded_definition':expanded}


def test_arsenic_condition_is_exposure_despite_source_classification():
    r=condition_definition(row('N01392','Arsenic to electron transfer in complex II','Pathogen','C20296 -> CxII'))
    assert r['source_network_type']=='Pathogen' and r['condition_kind']=='exposure'
    assert not r['member_readouts_establish_condition']


def test_missing_and_negative_conditions_are_distinct_and_gene_mutation_is_insufficient():
    rules=pd.DataFrame([condition_definition(row('N0','Reference','Reference')),
                        condition_definition(row('N1','Mutation-activated EGFR to PI3K signaling pathway','Variant'))])
    obs=pd.DataFrame([{'patient_id':'a','network_id':'N1','status':'absent','evidence_type':'curated_activating_variant','source_id':'validated_assay'}])
    value,known=applicability_matrix(rules,['a','b'],obs)
    assert value.loc['a','PN1']==0 and known.loc['a','PN1']
    assert pd.isna(value.loc['b','PN1']) and not known.loc['b','PN1']
    assert value.PN0.eq(1).all()
    obs['evidence_type']='gene_level_mutation_flag'
    with pytest.raises(ValueError,match='NOT_SPECIFIC_ENOUGH'):
        applicability_matrix(rules,['a','b'],obs)


def test_discordant_specific_assays_cannot_be_collapsed_to_positive():
    rules=pd.DataFrame([condition_definition(row('N1','Mutation-activated EGFR to PI3K signaling pathway','Variant'))])
    obs=pd.DataFrame([{'patient_id':'a','network_id':'N1','status':status,'evidence_type':'curated_activating_variant','source_id':status} for status in ['present','absent']])
    value,known=applicability_matrix(rules,['a'],obs)
    assert pd.isna(value.iloc[0,0]) and not known.iloc[0,0]
