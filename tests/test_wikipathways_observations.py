import numpy as np
import pandas as pd
import pytest
from cla_engine.wikipathways_observations import bind_source_members,score_rna,score_rppa,protein_symbol_matrix


def test_existing_assay_id_is_reused_independently_of_hash_version():
    from cla_engine.wikipathways_observations import existing_measurement_ids
    old=pd.DataFrame([dict(cohort='TCGA',kind='member_measurement',layer='RNA',feature_id='A',observation_id='old-id')])
    new=pd.concat([old.assign(observation_id='new-hash'),old.assign(layer='PROTEIN',observation_id='protein-id')],ignore_index=True)
    assert existing_measurement_ids(new,old)=={'new-hash':'old-id','protein-id':'protein-id'}
    with pytest.raises(ValueError,match='WP_DUPLICATE_ASSAY_IDENTITY'):
        existing_measurement_ids(new,pd.concat([old,old],ignore_index=True))

def test_source_binding_preserves_group_and_site_identity():
    members=pd.DataFrame({'source_target_id':['WP1','WP1','WP2'],'ensembl_gene_id':['A','B','A']})
    f=pd.DataFrame({'feature_id':['AB','A_s1','A_s2'],'gene_ids':['A|B','A','A']})
    links=bind_source_members(members,f)
    assert len(links)==6
    assert links.loc[(links.source_target_id=='WP2')&(links.feature_id=='AB'),'evidence_role'].item()=='ambiguous_group_context'
    assert len(links[links.feature_id.str.contains('s')])==4
    assert not links.independent_loss_enabled.any()

def test_rna_retains_minimum_and_query_does_not_change_reference():
    cols=['g'+str(i) for i in range(30)]
    x=pd.DataFrame([np.arange(30),np.arange(30)[::-1]],columns=cols,index=['ref','query'])
    ann=pd.DataFrame({'gene_id':cols,'reference_symbol':cols,'gene_biotype':'protein_coding'})
    sets={'WPvalid':cols[:10],'WPsmall':cols[:9]}
    y,c=score_rna(x,ann,sets)
    x.loc['query']=100*np.arange(30)
    again,_=score_rna(x,ann,sets)
    assert y.loc['ref','WPvalid']==again.loc['ref','WPvalid']
    assert y.WPsmall.isna().all()
    assert c.set_index('target_id').loc['WPsmall','reason']=='FEWER_THAN_10_MAPPED_CODING_GENES'

def test_rppa_reference_only_and_duplicate_antibodies_one_group():
    x=pd.DataFrame({'a':[1,2,3,20],'a2':[2,4,6,40],'b':[3,2,1,5],'c':[1,3,2,8]},index=['r1','r2','r3','q'])
    f=pd.DataFrame({'feature_id':x.columns,'gene_ids':['A','A','B','C']})
    y,n,c,fit=score_rppa(x,f,{'WP1':{'A','B','C'},'WP2':{'A','B'}},x.index[:3])
    assert n.WP1.eq(3).all() and n.WP2.eq(2).all()
    assert y.WP2.isna().all()
    changed=x.copy();changed.loc['q']*=100
    yy,_,_,ff=score_rppa(changed,f,{'WP1':{'A','B','C'},'WP2':{'A','B'}},x.index[:3])
    pd.testing.assert_frame_equal(y.iloc[:3],yy.iloc[:3]);pd.testing.assert_frame_equal(fit,ff)

def test_cptac_protein_symbols_do_not_include_noncoding_or_impute():
    x=pd.DataFrame({'f1':[0,np.nan],'f2':[2,4],'f3':[9,9]})
    a=pd.DataFrame({'feature_id':x.columns,'layer':'protein','reference_presence':'GENCODE_V36',
        'reference_symbol':['A','A','lnc'],'gene_biotype':['protein_coding','protein_coding','lncRNA']})
    out=protein_symbol_matrix(x,a)
    assert out.columns.tolist()==['A']
    assert out.A.tolist()==[1,4]
