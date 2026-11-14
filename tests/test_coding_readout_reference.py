import numpy as np
import pandas as pd
from cla_engine.coding_readout_reference import coding_progeny


def test_coding_progeny_reference_does_not_use_query_distribution():
    e=pd.DataFrame({'pc':[1.,3.,4.]},index=['a','b','query'])
    model=pd.DataFrame({'P':[2.,100.]},index=['pc','not_in_coding_axis'])
    a,ref=coding_progeny(e,model,['a','b'],minimum_genes=1)
    e.loc['query','pc']=10000
    b,other=coding_progeny(e,model,['a','b'],minimum_genes=1)
    assert ref==other
    assert a.loc[['a','b']].equals(b.loc[['a','b']])
    assert ref['coding_nonzero_members']=={'P':1}
    assert ref['original_nonzero_members']=={'P':2}
    assert np.isclose(a.loc['query','P'],np.sqrt(2))


def test_missing_weighted_member_propagates_only_to_affected_program():
    e=pd.DataFrame({'g1':[1.,2.,np.nan],'g2':[2.,4.,6.]},index=['a','b','q'])
    model=pd.DataFrame({'A':[1.,0.],'B':[0.,1.]},index=['g1','g2'])
    score,_=coding_progeny(e,model,['a','b'],minimum_genes=1)
    assert np.isnan(score.loc['q','A']) and np.isfinite(score.loc['q','B'])
