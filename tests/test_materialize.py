import pandas as pd
from cc_hhgt.materialize import classify_members

def config():
    return {'materialization':{
        'observed_core_evidence_min':0.7,
        'model_supported_probability_min':0.8,
        'predicted_candidate_probability_min':0.9,
        'predicted_candidate_max_observed_evidence':0.35,
        'max_uncertainty':0.25,
    }}

def test_predicted_candidate_not_shadowed_by_model_supported():
    df=pd.DataFrame({
        'observed_evidence_score':[0.1,0.5,0.8],
        'calibrated_probability':[0.95,0.85,0.75],
        'uncertainty':[0.1,0.1,0.1],
        'direction':['unknown','positive','negative'],
        'predicted_direction':['positive','positive','negative'],
    })
    out=classify_members(df,config())
    assert out.member_class.tolist()==['predicted_candidate','model_supported','observed_core']
