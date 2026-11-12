import numpy as np
import pandas as pd
import pytest
from cla_engine.protein_coding_scoring import aggregate_coding_genes


def test_mixed_symbol_excludes_lnc_before_tpm_pooling():
    ann=pd.DataFrame({'gene_id':['pc1','pc2','lnc'],'gene_biotype':['protein_coding','protein_coding','lncRNA'],'reference_symbol':['MIX','MIX','MIX']})
    x=pd.DataFrame({'pc1':np.log2([3.,3.]),'pc2':np.log2([4.,4.]),'lnc':[0.,20.]},index=['a','b'])
    gene,symbol=aggregate_coding_genes(x,ann)
    assert gene.columns.tolist()==['pc1','pc2']
    assert np.allclose(symbol.MIX,np.log2(6.))


def test_missing_coding_measurement_is_not_zero_filled():
    ann=pd.DataFrame({'gene_id':['g1','g2'],'gene_biotype':['protein_coding']*2,'reference_symbol':['S','S']})
    x=pd.DataFrame({'g1':[0.,1.],'g2':[np.nan,0.]})
    _,symbol=aggregate_coding_genes(x,ann)
    assert np.isnan(symbol.iloc[0,0]) and symbol.iloc[1,0]==1.
    with pytest.raises(ValueError,match='MISSING_MEASUREMENT_AXIS'):
        aggregate_coding_genes(x[['g1']],ann)
