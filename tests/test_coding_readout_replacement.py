import numpy as np
import pandas as pd
import pytest
from cla_engine.coding_readout_replacement import replace_coding_readouts


def fixture():
    v=pd.DataFrame({'RNA|A':[99.,99.],'NATIVE|B':[.2,.8],'protein|C':[0.,np.nan]},index=['p','q'])
    md=pd.DataFrame({'target_id':v.columns,'column_index':[9,14,20],
                     'measurement_layer':['RNA_EXPRESSION','RNA_DECONVOLUTION','PROTEIN']})
    cv=pd.DataFrame({'RNA|A@@coding':[0.,np.nan]},index=['q','p'])
    cm=pd.DataFrame({'source_score_id':['RNA|A'],'readout_id':['RNA|A@@coding'],
                     'method':['coding_ssgsea'],'method_variant':['coding_v1']})
    return v,md,cv,cm


def test_exact_replacement_reorders_patients_and_preserves_other_measurements():
    v,md,cv,cm=fixture()
    r,meta,mapping=replace_coding_readouts(v,md,cv,cm)
    assert list(r.columns)==['RNA|A@@coding','NATIVE|B','protein|C']
    assert np.isnan(r.loc['p','RNA|A@@coding']) and r.loc['q','RNA|A@@coding']==0
    pd.testing.assert_frame_equal(r[['NATIVE|B','protein|C']],v[['NATIVE|B','protein|C']])
    assert meta.patients_measured.tolist()==[1,2,1]
    assert meta.coding_expression_basis_verified.tolist()==[True,False,False]
    assert meta.measurement_definition.iloc[1]=='PRESERVED_EXTERNAL_RNA_SOURCE'
    assert mapping.original_column_index.tolist()==[9,14,20]


def test_missing_patient_or_duplicate_identity_cannot_be_silently_reindexed():
    v,md,cv,cm=fixture()
    with pytest.raises(ValueError,match='PATIENT_MISMATCH'):
        replace_coding_readouts(v,md,cv.iloc[:1],cm)
    with pytest.raises(ValueError,match='DUPLICATE_AXIS'):
        replace_coding_readouts(v,md,cv,pd.concat([cm,cm]))
