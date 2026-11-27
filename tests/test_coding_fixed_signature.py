import numpy as np
import pandas as pd
import pytest
from cla_engine.coding_fixed_signature import score_fixed_coding_set


def fixture():
    x=pd.DataFrame([[1.,2.,3.,4.],[5.,1.,2.,0.]],index=['a','b'],columns=list('ABCD'))
    a=pd.DataFrame({'gene_id':['g1','g2','g3','g4'],'gene_biotype':['protein_coding']*4,'reference_symbol':list('ABCD')})
    return x,a


def test_new_or_reordered_patient_does_not_change_existing_score():
    x,a=fixture();first,_=score_fixed_coding_set(x,a,['A','B'],'sig')
    second,_=score_fixed_coding_set(pd.concat([x.iloc[::-1],x.iloc[[0]].rename(index={'a':'new'})]),a,['A','B'],'sig')
    pd.testing.assert_frame_equal(first,second.loc[first.index])


def test_noncoding_background_rejected_and_missing_retained():
    x,a=fixture();a.loc[0,'gene_biotype']='lncRNA'
    with pytest.raises(ValueError,match='CODING_GENE_IDS'):
        score_fixed_coding_set(x,a,['A'],'sig')
    x,a=fixture();x.loc['a','D']=np.nan
    score,_=score_fixed_coding_set(x,a,['A','B'],'sig')
    assert pd.isna(score.loc['a','sig']) and np.isfinite(score.loc['b','sig'])
    with pytest.raises(ValueError,match='MEMBER_DEFINITION'):
        score_fixed_coding_set(x,a,['A','E'],'sig')
