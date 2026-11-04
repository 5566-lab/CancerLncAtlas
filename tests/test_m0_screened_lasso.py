import numpy as np
from scipy.stats import spearmanr, false_discovery_control
from cc_hhgt.m0_screened_lasso import screen, fit_split, fit_fixed, predict_state, path_predictions

def test_pairwise_screen_matches_scipy_and_bh_includes_untestable():
    r=np.random.default_rng(41);x=r.normal(size=(36,7));y=r.normal(size=(36,3))
    x[:9,1]=np.nan;x[:,5]=2;x[:,6]=np.nan;y[3:10,1]=np.nan
    got=screen(x,y)
    for j in range(3):
        pp=[]
        for k in range(7):
            ok=np.isfinite(x[:,k])&np.isfinite(y[:,j])
            if ok.sum()>=10 and np.ptp(x[ok,k])>0:
                ref=spearmanr(x[ok,k],y[ok,j])
                np.testing.assert_allclose(got["rho"][k,j],ref.statistic,atol=1e-7)
                pp.append(ref.pvalue)
            else:pp.append(1.)
        np.testing.assert_allclose(got["p"][:,j],pp,atol=1e-12)
        np.testing.assert_allclose(got["q"][:,j],false_discovery_control(pp),atol=1e-7)

def test_empty_screen_fits_intercept_without_arbitrary_top_features():
    x=np.arange(60.).reshape(20,3);y=np.linspace(1,4,20)
    p,s=fit_fixed(x,y,np.zeros(3,bool),x[:4],.1)
    np.testing.assert_allclose(p,y.mean());assert len(s["coefficient"])==0
    np.testing.assert_allclose(p,predict_state(s,x[:4]))

def test_replay_and_query_perturbation_cannot_change_coefficients():
    r=np.random.default_rng(5);x=r.normal(size=(90,15));y=2*x[:,2]-x[:,4]+r.normal(size=90)*.1
    x[4,2]=np.nan
    args=(x[:50],y[:50],x[50:70],y[50:70])
    p,a,_=fit_split(*args,x[70:],ratios=np.array([1.,.1,.01]))
    _,b,_=fit_split(*args,x[70:]*100+1000,ratios=np.array([1.,.1,.01]))
    for k in ["feature_index","coefficient","median","mean","scale"]:
        np.testing.assert_allclose(a[k],b[k])
    np.testing.assert_allclose(p,predict_state(a,x[70:]),atol=1e-10)

def test_validation_y_does_not_enter_inner_screen():
    r=np.random.default_rng(2);x=r.normal(size=(90,25));y=x[:,0]+r.normal(size=90)*.3
    _,_,a=fit_split(x[:50],y[:50],x[50:70],y[50:70],x[70:])
    _,_,b=fit_split(x[:50],y[:50],x[50:70],y[50:70][::-1],x[70:])
    assert a["n_screened"]==b["n_screened"]
    assert a["n_variable"]==b["n_variable"]

def test_fixed_fit_equals_corresponding_path_prediction():
    r=np.random.default_rng(8);x=r.normal(size=(80,12));y=2*x[:,1]-.7*x[:,3]+r.normal(size=80)*.3
    mask=np.ones(12,bool);ratios=np.array([1.,.1,.01])
    pred,_=path_predictions(x[:60],y[:60],mask,x[60:],ratios=ratios)
    for j,ratio in enumerate(ratios):
        p,s=fit_fixed(x[:60],y[:60],mask,x[60:],ratio)
        np.testing.assert_allclose(pred[:,j],p,atol=5e-4)
        assert s["dual_gap"]<=1.01e-4

def test_missing_y_rows_never_become_training_labels():
    r=np.random.default_rng(19);x=r.normal(size=(30,5));y=x[:,0].copy();y[:4]=np.nan
    _,a=fit_fixed(x,y,np.ones(5,bool),x[:2],.1)
    _,b=fit_fixed(x[4:],y[4:],np.ones(5,bool),x[:2],.1)
    np.testing.assert_allclose(a["coefficient"],b["coefficient"])


def test_lasso_solver_agrees_with_sklearn_reference():
    from sklearn.linear_model import Lasso
    from cc_hhgt.m0_screened_lasso import prepare
    r=np.random.default_rng(72);x=r.normal(size=(100,20));y=2*x[:,0]-x[:,5]+r.normal(size=100)*.2
    mask=np.ones(20,bool)
    z,yy,st=prepare(x[:80],y[:80],mask)
    yn=(yy-st['outcome_mean'])/st['outcome_scale']
    alpha=np.max(np.abs(z.T@yn))/len(yn)*.05
    ref=Lasso(alpha=alpha,fit_intercept=False,tol=1e-10,max_iter=100000).fit(z,yn)
    _,got=fit_fixed(x[:80],y[:80],mask,x[80:],.05,tol=1e-8)
    np.testing.assert_allclose(got['standardized_coefficient'],ref.coef_*st['outcome_scale'],atol=1e-6)
