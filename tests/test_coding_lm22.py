import numpy as np
import pandas as pd
from cla_engine.coding_lm22 import coding_reference,relative_nusvr,score_coding_lm22


def test_direct_noncoding_names_cannot_be_rescued_by_alias_or_affect_reference():
    sig=pd.DataFrame({'c1':[1.,2.,3.,900.],'c2':[3.,2.,1.,700.]},index=['A','B','oldC','L'])
    ann=pd.DataFrame({'gene_id':['e1','e2','e3','el'],'reference_symbol':['A','B','C','L'],
                      'gene_biotype':['protein_coding']*3+['lncRNA']})
    hgnc=pd.DataFrame({'status':['Approved'],'ensembl_gene_id':['e3'],'symbol':['C'],
                       'alias_symbol':['oldC|L'],'prev_symbol':['']})
    a,audit=coding_reference(sig,ann,hgnc)
    sig.loc['L']=-1000
    b,_=coding_reference(sig,ann,hgnc)
    pd.testing.assert_frame_equal(a,b)
    assert set(a.index)=={'A','B','C'}
    assert audit.set_index('source_symbol').loc['L','decision']=='DIRECT_NONCODING_SYMBOL'


def test_synthetic_mixture_recovery_and_query_independence():
    rng=np.random.default_rng(21)
    B=rng.lognormal(size=(100,3));truth=np.array([.2,.5,.3]);mix=B@truth
    f,_=relative_nusvr(B,mix)
    np.testing.assert_allclose(f,truth,atol=.015)
    ref=pd.DataFrame(B,index=['g'+str(i) for i in range(100)])
    e=pd.DataFrame(np.log2(np.stack([mix,B@np.array([.8,.1,.1])])+1),index=['a','b'],columns=ref.index)
    a,_=score_coding_lm22(e,ref)
    e.loc['b']+=8
    b,_=score_coding_lm22(e,ref)
    np.testing.assert_array_equal(a.loc['a'],b.loc['a'])
    np.testing.assert_allclose(a.sum(axis=1),1)
    e.loc['b','g0']=np.nan
    c,qc=score_coding_lm22(e,ref)
    assert c.loc['b'].isna().all() and qc.iloc[1].reason=='MISSING_MEMBER_MEASUREMENT'
