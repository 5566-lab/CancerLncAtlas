import numpy as np
from scripts.score_tcpa_cptac_paired_phospho import residualize

def test_total_protein_residual_fit_uses_training_only_and_is_orthogonal():
    total=np.arange(12,dtype=float);phospho=2+3*total+np.sin(total)
    train=np.arange(12)<9
    r,p=residualize(phospho,total,train)
    assert abs(r[train].sum())<1e-10
    assert abs(np.sum(r[train]*(total[train]-total[train].mean())))<1e-10
    changed=phospho.copy();changed[~train]=1e6
    _,p2=residualize(changed,total,train);assert p==p2
    coef=np.linalg.lstsq(np.column_stack([np.ones(train.sum()),total[train]]),phospho[train],rcond=None)[0]
    np.testing.assert_allclose([p["intercept"],p["slope"]],coef,atol=1e-12)

def test_missing_total_does_not_become_zero_or_unadjusted_phospho():
    total=np.arange(8,dtype=float);total[-1]=np.nan
    r,p=residualize(np.arange(8,dtype=float)**2,total,np.arange(8)<6)
    assert p["status"]=="FITTED" and np.isnan(r[-1])
    r,p=residualize(np.arange(8,dtype=float),np.ones(8),np.arange(8)<6)
    assert p["status"]=="NO_TOTAL_PROTEIN_VARIANCE" and np.isnan(r).all()
