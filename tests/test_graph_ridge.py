import numpy as np
import pandas as pd
from scipy import sparse
from cc_hhgt.graph_ridge import solve_graph,normalized_kernel,fit_transform,transform,ridge_path,target_policy,combined_kernel

def test_graph_cg_and_psd():
    a=np.array([[0,.5,0],[.5,0,1],[0,1,0.]])
    d=a.sum(1);l=np.eye(3)-a/np.sqrt(d[:,None]*d[None,:]);b=np.arange(12.).reshape(3,4)
    actual,info=solve_graph(sparse.csr_matrix(l),b,10)
    np.testing.assert_allclose(actual,np.linalg.solve(np.eye(3)+10*l,b),atol=1e-8)
    assert info['max_relative_residual']<5e-9

def test_dual_matches_unpenalized_primal_with_latent_bridge():
    rng=np.random.default_rng(5);n=20;x=rng.normal(size=(n,4));x[:,1]=0
    z=np.column_stack([np.ones(n),rng.normal(size=n)]);y=rng.normal(size=(n,2));tr=np.arange(15);te=np.arange(15,20)
    a=np.ones((4,4))-np.eye(4);l=np.diag(a.sum(1))-a;q=np.eye(4)+2*l;k=x@np.linalg.solve(q,x.T);lam=.4
    pred,_=ridge_path(k,z,tr,te,y[tr],[lam])
    design=np.column_stack([z,x]);pen=np.zeros((6,6));pen[2:,2:]=len(tr)*lam*q
    coef=np.linalg.solve(design[tr].T@design[tr]+pen,design[tr].T@y[tr])
    np.testing.assert_allclose(pred[0],design[te]@coef,atol=1e-8)

def test_holdout_cannot_change_fit_or_normalization():
    x=np.arange(50.,dtype=float).reshape(10,5);x[0,0]=np.nan;tr=np.arange(7)
    z,sp=fit_transform(x,tr);changed=x.copy();changed[7:]*=-9000
    zz,sp2=fit_transform(changed,tr)
    np.testing.assert_array_equal(sp['index'],sp2['index']);np.testing.assert_allclose(z[tr],zz[tr])
    np.testing.assert_allclose(transform(x,sp),z)
    k,s=normalized_kernel(z,tr);kk,ss=normalized_kernel(zz,tr)
    np.testing.assert_allclose(k[np.ix_(tr,tr)],kk[np.ix_(tr,tr)])
    assert s['scale']==ss['scale']

def test_target_layer_exclusion_including_genomic_state():
    for layer,blocked in [('PROTEIN_PATHWAY','PROTEIN'),('PHOSPHO_PTMSEA','PHOSPHO')]:
        p=target_policy(dict(layer=layer,target_id=layer+'|x'))
        assert blocked not in p['allowed_blocks']
    for fam,b in [('mutation_burden','MUTATION'),('ploidy','CNV')]:
        assert b not in target_policy(dict(layer='STATE',target_id='STATE|x',family=fam))['allowed_blocks']
    p=target_policy(dict(layer='STATE',target_id='STATE|absolute_purity_ploidy::purity'))
    assert not p['use_purity'] and 'CNV' not in p['allowed_blocks'] and 'MUTATION' not in p['allowed_blocks']

def test_excluded_layer_does_not_enter_kernel():
    ctx={'kernels':{('lncrna',0.):np.eye(3),'CNV':np.eye(3)*2,'PROTEIN':np.eye(3)*3}}
    k=combined_kernel(ctx,'lncrna',0,['CNV'],'multiomics_graph');ctx['kernels']['PROTEIN']*=999
    np.testing.assert_array_equal(k,combined_kernel(ctx,'lncrna',0,['CNV'],'multiomics_graph'))

def test_missing_mutation_is_not_wildtype():
    x=np.array([[0.,0.],[1.,1.],[1.,0.],[np.nan,np.nan]])
    z,s=fit_transform(x,np.arange(4),True)
    assert s['median'][0]==1
    assert s['index'][-1]==2
    assert z[-1,-1]!=z[0,-1]

def test_rna_target_does_not_disallow_lncrna_input():
    p=target_policy(dict(layer='RNA_PATHWAY',target_id='RNA_PATHWAY|x'))
    assert p['role']=='RNA_DERIVED_TARGET' and not p['excluded_blocks']
