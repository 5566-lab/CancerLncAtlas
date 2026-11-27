import numpy as np
import pandas as pd
import pytest
from cla_engine.fixed_y_bundle import supervision_mask


def test_unknown_event_and_external_readout_are_not_training_labels():
    observed=pd.DataFrame([[True,True],[True,True],[False,True]],index=['p1','p2','p3'],columns=['real','external'])
    eligible=pd.Series([True,False],index=observed.columns)
    applicability=pd.Series([1,np.nan,0],index=observed.index)
    result=supervision_mask(observed,eligible,applicability)
    assert result.to_numpy().tolist()==[[True,False],[False,False],[False,False]]
    assert observed.iloc[1,0] and pd.isna(applicability.iloc[1])


def test_observed_zero_is_not_missing():
    value=pd.DataFrame([[0.0,np.nan]],columns=['zero','missing'])
    observed=value.notna();eligible=pd.Series(True,index=value.columns);app=pd.Series([1],index=value.index)
    assert supervision_mask(observed,eligible,app).iloc[0].tolist()==[True,False]


def test_misaligned_eligibility_is_rejected():
    with pytest.raises(ValueError,match='AXIS_MISMATCH'):
        supervision_mask(pd.DataFrame([[True]],columns=['a']),pd.Series([True],index=['b']),pd.Series([1]))
