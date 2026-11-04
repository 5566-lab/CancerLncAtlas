import numpy as np
from sklearn.linear_model import Ridge
from cc_hhgt.m0_linear import fit_group, preprocess, grouped_oof, predict_checkpoint, metrics

def test_dual_ridge_matches_existing_sklearn_baseline():
    rng=np.random.default_rng(20)
    x=rng.normal(size=(70,12));y=np.c_[3*x[:,0]-2*x[:,3],x[:,4]]+rng.normal(scale=.1,size=(70,2))
    r=fit_group(x[:35],y[:35],x[35:50],y[35:50],x[50:],lambdas=[.001])
    z,q,_=preprocess(x[:50],x[50:])
    expected=Ridge(alpha=z.shape[1]*.001).fit(z,y[:50]).predict(q)
    assert not r["intercept_only"].any()
    np.testing.assert_allclose(r["prediction"],expected,rtol=1e-8,atol=1e-8)

def test_validation_controls_selection_and_test_predictors_do_not_fit():
    rng=np.random.default_rng(12);x=rng.normal(size=(80,8));y=(x[:,0]*4)[:,None]
    x[0,1]=np.nan
    r=fit_group(x[:40],y[:40],x[40:60],y[40:60],x[60:])
    s=fit_group(x[:40],y[:40],x[40:60],y[40:60],x[60:]*1000)
    for key in ["median","mean","scale","selected_lambda","dual"]:
        np.testing.assert_allclose(r[key],s[key])
    assert metrics(y[60:,0],r["prediction"][:,0],r["baseline"][:,0])["r2"]>.99

def test_masked_y_checkpoints_and_outer_test_independence(tmp_path):
    rng=np.random.default_rng(13);x=rng.normal(size=(75,7))
    y=np.c_[2*x[:,0],x[:,2],np.zeros(75),np.full(75,np.nan)]
    y[::4,1]=np.nan;folds=np.arange(75)%5
    args=dict(feature_ids=[f"LNC:{i}" for i in range(7)],patient_ids=[str(i) for i in range(75)],
              target_ids=["signal","masked","constant","absent"])
    p,b,a,c=grouped_oof(x,y,folds,output=tmp_path/"one",**args)
    assert np.isnan(p[::4,1]).all() and np.isnan(p[:,3]).all()
    assert metrics(y[:,0],p[:,0],b[:,0])["r2"]>.95
    assert a.loc[a.target_id.eq("constant"),"intercept_only"].all()
    checkpoint=next((tmp_path/"one").glob("fold_0_mask_*.npz"))
    with np.load(checkpoint) as z:
        indices=[args["patient_ids"].index(i) for i in z["test_patient_ids"]]
        columns=[args["target_ids"].index(i) for i in z["target_ids"]]
    np.testing.assert_allclose(p[np.ix_(indices,columns)],predict_checkpoint(checkpoint,x[indices]),atol=1e-10)
    changed=y.copy();changed[folds==0,0]+=10000
    p2,_,a2,_=grouped_oof(x,changed,folds,output=tmp_path/"two",**args)
    np.testing.assert_allclose(p[folds==0,0],p2[folds==0,0])
    assert not a.status.eq("FITTED")[a.target_id.eq("absent")].any()

def test_training_all_missing_constant_features_removed():
    x=np.array([[1,np.nan,0],[2,np.nan,0],[3,np.nan,0]],float)
    z,q,state=preprocess(x,np.array([[100,999,1]],float))
    assert state["feature_index"].tolist()==[0]
    assert z.shape==(3,1) and q.shape==(1,1)

def test_missing_label_prepared_group_equals_sklearn_on_fitted_fold_transform():
    from cc_hhgt.m0_linear import fit_prepared_group
    rng=np.random.default_rng(33);x=rng.normal(size=(80,13));y=(4*x[:,2]+rng.normal(scale=.2,size=80))[:,None]
    zi,zv,_=preprocess(x[:40],x[40:60]);zf,zt,_=preprocess(x[:60],x[60:])
    tr=np.arange(0,40,2);va=np.arange(0,20,2);fit=np.r_[tr,40+va]
    r=fit_prepared_group(zi[tr],y[tr],zv[va],y[40+va],zf[fit],y[fit],zt,[.001])
    expected=Ridge(alpha=13*.001).fit(zf[fit],y[fit]).predict(zt)
    np.testing.assert_allclose(r["prediction"].reshape(-1),expected.reshape(-1),atol=1e-8)
