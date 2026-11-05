import numpy as np
import pandas as pd
import pytest
from cc_hhgt.m0_three_arm import select_targets
from cc_hhgt.m0_covariates import uses_purity,complete_rows

def test_rna_state_requires_explicit_contract():
    m=pd.DataFrame(dict(target_id=['RNA_PATHWAY|A','STATE|B','PROTEIN_PATHWAY|C'],layer=['RNA_PATHWAY','STATE','PROTEIN_PATHWAY'],cohort=['CPTAC']*3))
    assert select_targets(m).target_id.tolist()==['PROTEIN_PATHWAY|C']
    z=select_targets(m,layers=['RNA_PATHWAY','STATE'])
    assert z.target_id.tolist()==['RNA_PATHWAY|A','STATE|B']
    assert z.analysis_role.eq('EXPLORATORY_EXISTING_SCORE_RECONSTRUCTION').all()
    with pytest.raises(ValueError):select_targets(m,layers=['UNKNOWN'])

def test_purity_target_can_keep_observed_patients_without_purity_covariate():
    c=pd.DataFrame(dict(cohort=['CPTAC','CPTAC'],age_years=[50.,60.],sex=['Female','Male'],stage=['I','II'],purity=[np.nan,.8]))
    assert not uses_purity('STATE|absolute_purity_ploidy::purity')
    assert complete_rows(c,uses_purity('STATE|absolute_purity_ploidy::purity')).tolist()==[True,True]
    assert complete_rows(c,uses_purity('STATE|ESTIMATE::TumorPurity')).tolist()==[False,True]
