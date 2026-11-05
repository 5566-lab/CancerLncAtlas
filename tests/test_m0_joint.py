import numpy as np
import pandas as pd
from pathlib import Path
from cc_hhgt.m0_joint_scores import raw_ssgsea,weighted_programs,frozen_caf,frozen_xcell,fit_range
from cc_hhgt.m0_joint_lasso import fit_x_context,fit_final,predict,cohort_screen

def test_native_gsva_tied_rank_reference():
    p=Path(__file__).parent/'fixtures/m0_joint'
    x=pd.read_csv(p/'SSGSEA_TEST_X.tsv',sep='\t').set_index('gene').T
    y=pd.read_csv(p/'SSGSEA_TEST_Y.tsv',sep='\t').set_index('target').T
    got,_=raw_ssgsea(x,{'A':['G1','G3','G5','G10'],'B':['G2','G4','G7','G15'],'C':['G'+str(i) for i in range(6,13)]})
    np.testing.assert_allclose(got,y,atol=1e-12,rtol=0)
    altered=pd.concat([x,pd.DataFrame([np.arange(20)*100],columns=x.columns,index=['NEW'])])
    again,_=raw_ssgsea(altered,{'A':['G1','G3','G5','G10'],'B':['G2','G4','G7','G15'],'C':['G'+str(i) for i in range(6,13)]})
    np.testing.assert_array_equal(got,again.loc[x.index])

def test_score_missing_background_not_zero_filled():
    x=pd.DataFrame([[0,1,2,3],[np.nan,1,2,3]],columns=list('abcd'))
    score,_=raw_ssgsea(x,{'A':['a','b']})
    assert np.isfinite(score.iloc[0,0]) and np.isnan(score.iloc[1,0])

def test_training_only_tf_and_caf_references():
    rng=np.random.default_rng(30);e=pd.DataFrame(rng.normal(size=(25,9)),columns=['G'+str(i) for i in range(9)])
    net=pd.DataFrame({'gene':list(e.columns)*2,'state_id':['P']*9+['Q']*9,'weight':np.r_[np.ones(9),np.arange(9)-4]})
    a,sa=weighted_programs(e,net,np.arange(16));c,sc=frozen_caf(e,list(e),np.arange(16))
    changed=e.copy();changed.iloc[16:]+=rng.normal(100,50,(9,9))
    b,sb=weighted_programs(changed,net,np.arange(16));d,sd=frozen_caf(changed,list(e),np.arange(16))
    np.testing.assert_array_equal(a.iloc[:16],b.iloc[:16]);np.testing.assert_array_equal(sa['gene_mean'],sb['gene_mean'])
    np.testing.assert_array_equal(c[:16],d[:16]);np.testing.assert_array_equal(sc['weight'],sd['weight'])

def test_xcell_offsets_use_only_training():
    x=pd.DataFrame([[2,4,3,5],[3,5,7,8],[4,8,10,15]],columns=['A%x%1','A%x%2','B%x%1','B%x%2'])
    ref={'cells':['A','B'],'fv':[[0,1,1],[0,1,1]],'K':[[1,0],[0,1]]}
    a,s=frozen_xcell(x,[0,1],ref);xx=x.copy();xx.iloc[2]=-500;b,t=frozen_xcell(xx,[0,1],ref)
    np.testing.assert_allclose(a.iloc[:2],b.iloc[:2]);np.testing.assert_array_equal(s['signature_min'],t['signature_min'])

def test_cohort_only_shift_does_not_pass_screen():
    cohort=np.repeat(['A','B'],30);x=np.column_stack([np.r_[np.zeros(30),np.ones(30)*10],np.tile(np.arange(30),2)])
    y=x[:,0]+x[:,1]*.001
    result=cohort_screen(x,y,cohort)
    assert not result['passed'][0,0] and result['passed'][1,0]
    assert result['rho'][0,0]==0

def test_screen_matches_explicit_partial_rank_calculation():
    from scipy.stats import rankdata
    rng=np.random.default_rng(8);x=rng.normal(size=(50,3));cohort=np.repeat(['A','B'],25);y=x[:,0]+rng.normal(0,.1,50)
    result=cohort_screen(x,y,cohort)
    a=rankdata(x,axis=0);b=rankdata(y)
    for c in ['A','B']:
        mask=cohort==c;a[mask]-=a[mask].mean(axis=0);b[mask]-=b[mask].mean()
    expected=a.T@b/np.sqrt((a*a).sum(axis=0)*(b*b).sum())
    np.testing.assert_allclose(result['rho'][:,0],expected,atol=1e-7)

def test_lasso_one_shared_coefficient_with_cohort_intercepts():
    rng=np.random.default_rng(31);cohort=np.repeat(['A','B'],50);x=rng.normal(size=(100,6));x[50:]+=3
    y=2*x[:,0]-x[:,1]+np.where(cohort=='A',5,-9)
    context=fit_x_context(x,cohort);model=fit_final(context,y,.001);pred=predict(model,x,cohort)
    assert np.mean((pred-y)**2)<.001
    np.testing.assert_allclose(model['intercept'],[5,-9],atol=.04)
    assert len(model['coefficient'])==len(model['feature_index'])
    # A query cohort is never used to fit its own calibration.
    original=predict(model,x[:2],np.array(['UNSEEN','UNSEEN']))
    again=predict(model,np.vstack([x[:2],np.full((1,6),10000)]),np.array(['UNSEEN']*3))
    np.testing.assert_array_equal(original,again[:2])

def test_quality_filter_and_screen_mask_are_training_local():
    x=np.array([[1,2,np.nan,0],[2,2,np.nan,1],[3,2,3,0],[4,2,4,1]],float);cohort=np.array(['A','A','B','B'])
    context=fit_x_context(x,cohort,np.array([True,True,True,False]))
    assert context['feature_index'].tolist()==[0]

def test_ssgsea_reference_range_ignores_heldout_values():
    a=np.arange(20.).reshape(5,4);r=fit_range(a,[0,1,2]);a[3:]=1e9
    assert fit_range(a,[0,1,2])==r


def clinical_fixture(n=120):
    rng=np.random.default_rng(121)
    return pd.DataFrame({'cohort':np.repeat(['A','B'],n//2),'age_years':rng.uniform(30,80,n),'sex':np.tile(['female','male'],n//2),'stage':np.tile(['I','II','III'],n//3),'purity':rng.uniform(.2,.95,n)})

def test_unpenalized_covariates_match_weighted_ols_at_zero_lncrna():
    from cc_hhgt.m0_covariates import fit_design
    c=clinical_fixture();rng=np.random.default_rng(88);x=rng.normal(size=(len(c),5));y=3*c.age_years.to_numpy()+5*c.purity.to_numpy()+rng.normal(size=len(c))
    context=fit_x_context(x,c.cohort,covariates=c);model=fit_final(context,y,1.)
    pred=predict(model,x,c.cohort,c)
    z,_=fit_design(c);expected=z@np.linalg.lstsq(z,y,rcond=None)[0]
    assert len(model['coefficient'])==0
    np.testing.assert_allclose(pred,expected,atol=1e-9)

def test_joint_clinical_fit_matches_full_unpenalized_regression_at_small_penalty():
    c=clinical_fixture();rng=np.random.default_rng(89);x=rng.normal(size=(len(c),5));x[:,0]+=(c.age_years.to_numpy()-50)/10
    y=2*x[:,0]-x[:,1]+.3*c.age_years.to_numpy()+4*c.purity.to_numpy()+np.where(c.cohort.eq('A'),8,-3)
    context=fit_x_context(x,c.cohort,covariates=c);model=fit_final(context,y,1e-6)
    pred=predict(model,x,c.cohort,c);assert np.mean((pred-y)**2)<1e-8
    beta=np.zeros(5);beta[model['feature_index']]=model['coefficient'];np.testing.assert_allclose(beta,[2,-1,0,0,0],atol=2e-5)
    held=c.copy();held.loc[0,'age_years']=np.nan;assert np.isnan(predict(model,x,held.cohort,held)[0])

def test_clinical_screen_matches_projection_and_complete_cases():
    from cc_hhgt.m0_covariates import fit_design,complete_rows
    from scipy.stats import rankdata
    c=clinical_fixture();rng=np.random.default_rng(4);x=rng.normal(size=(len(c),3));y=x[:,0]+.3*c.age_years.to_numpy();c.loc[4,'purity']=np.nan
    got=cohort_screen(x,y,c.cohort,covariates=c,target_ids=['RNA_PATHWAY|P']);ok=complete_rows(c);z,_=fit_design(c.loc[ok]);a=rankdata(x[ok],axis=0);b=rankdata(y[ok]);a-=z@np.linalg.lstsq(z,a,rcond=None)[0];b-=z@np.linalg.lstsq(z,b,rcond=None)[0]
    rho=a.T@b/np.sqrt((a*a).sum(axis=0)*(b*b).sum())
    np.testing.assert_allclose(got['rho'][:,0],rho,atol=1e-7);assert (got['n'][:,0]==119).all()

def test_dna_purity_target_excludes_its_own_covariate():
    from cc_hhgt.m0_covariates import uses_purity
    c=clinical_fixture();x=c.purity.to_numpy()[:,None];y=x[:,0]
    tid='STATE_NATIVE_TCGA|absolute_purity_ploidy::purity'
    assert not uses_purity(tid) and uses_purity('STATE|ESTIMATE::TumorPurity')
    a=cohort_screen(x,y,c.cohort,covariates=c,target_ids=[tid]);assert a['passed'][0,0]
    context=fit_x_context(x,c.cohort,covariates=c,use_purity=False);assert 'purity' not in context['design_spec']['numeric']

def test_covariate_encoding_never_learns_query_categories_or_means():
    from cc_hhgt.m0_covariates import fit_design,query_design
    c=clinical_fixture();z,s=fit_design(c.iloc[:90]);q=c.iloc[90:].copy();before,_=query_design(q,s);q.loc[q.index[-1],'age_years']=10000;after,_=query_design(q,s)
    np.testing.assert_array_equal(before[:-1],after[:-1]);q.loc[q.index[0],'stage']='IV';_,ok=query_design(q,s);assert not ok[0]


def test_unequal_cohort_weighted_unpenalized_fit():
    from cc_hhgt.m0_covariates import fit_design
    c=clinical_fixture().iloc[:90].copy();rng=np.random.default_rng(172);x=rng.normal(size=(len(c),3));y=rng.normal(size=len(c))+.2*c.age_years.to_numpy()
    con=fit_x_context(x,c.cohort,covariates=c);model=fit_final(con,y,1.)
    z,_=fit_design(c);w=con['root'];gamma=np.linalg.lstsq(z*w[:,None],y*w,rcond=None)[0]
    np.testing.assert_allclose(predict(model,x,c.cohort,c),z@gamma,atol=1e-10)
