"""M0: the existing nested Ridge baseline, batched over equal outcome masks.

Predictors are lncRNA expression only. No PCA, covariates or outcome imputation.
The dual solution equals sklearn Ridge(alpha=p*lambda) on training-scaled X.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr
from .v31_continuous_baselines import _impute_from_train

LAMBDAS = np.logspace(4, -4, 17)

def preprocess(train, other):
    train = np.asarray(train, float)
    other = np.asarray(other, float)
    # All-missing/constant features cannot define a fitted regression coefficient.
    observed = np.isfinite(train).any(axis=0)
    keep0 = np.flatnonzero(observed)
    if not len(keep0):
        raise ValueError("NO_OBSERVED_TRAINING_FEATURES")
    a, b = _impute_from_train(train[:, keep0], other[:, keep0])
    mean = a.mean(axis=0)
    scale = a.std(axis=0)
    keep = scale > 1e-12
    if not keep.any():
        raise ValueError("NO_VARIABLE_TRAINING_FEATURES")
    ids = keep0[keep]
    return ((a[:, keep]-mean[keep])/scale[keep],
            (b[:, keep]-mean[keep])/scale[keep],
            dict(feature_index=ids, median=np.nanmedian(train[:, ids], axis=0),
                 mean=mean[keep], scale=scale[keep]))

def spectral(train_z, query_z, y):
    p = train_z.shape[1]
    kernel = train_z @ train_z.T / p
    eigenvalue, eigenvector = eigh(kernel, check_finite=False)
    eigenvalue = np.maximum(eigenvalue, 0)
    return eigenvalue, eigenvector, query_z @ train_z.T / p @ eigenvector, eigenvector.T @ y

def fit_group(x_train, y_train, x_validation, y_validation, x_test, *, lambdas=LAMBDAS):
    """No test outcomes accepted: neither tuning nor preprocessing can inspect them."""
    x_train, x_validation, x_test = [np.asarray(a, float) for a in (x_train,x_validation,x_test)]
    yt, yv = [np.asarray(a, float) for a in (y_train,y_validation)]
    if yt.ndim != 2 or yv.ndim != 2 or yt.shape[1] != yv.shape[1]:
        raise ValueError("Outcome matrices must have the same target axis")
    if not np.isfinite(yt).all() or not np.isfinite(yv).all():
        raise ValueError("Outcome imputation is forbidden")
    grid = np.unique(np.asarray(lambdas,float))[::-1]
    if len(grid)==0 or not np.isfinite(grid).all() or np.any(grid<=0):
        raise ValueError("Invalid lambda grid")
    z, q, _ = preprocess(x_train, x_validation)
    ym = yt.mean(axis=0)
    d,u,a,b = spectral(z,q,yt-ym)
    # Intercept-only is an explicit candidate; ties retain the simpler model.
    best_mse = np.mean((yv-ym)**2,axis=0)
    chosen = np.full(yt.shape[1], -1, int)
    for j, lam in enumerate(grid):
        pred = a @ (b/(d[:,None]+lam)) + ym
        mse = np.mean((pred-yv)**2,axis=0)
        improve = mse < best_mse - 1e-14*np.maximum(1,best_mse)
        chosen[improve] = j
        best_mse[improve] = mse[improve]
    xf = np.concatenate([x_train,x_validation])
    yf = np.concatenate([yt,yv])
    zf, qt, transform = preprocess(xf,x_test)
    mu = yf.mean(axis=0)
    d,u,a,b = spectral(zf,qt,yf-mu)
    selected = grid[np.maximum(chosen,0)]
    dual = u @ (b/(d[:,None]+selected))
    dual[:,chosen<0] = 0
    pred = qt @ zf.T / zf.shape[1] @ dual + mu
    return dict(prediction=pred, baseline=np.broadcast_to(mu,pred.shape).copy(),
                selected_lambda=np.where(chosen<0,np.nan,selected),
                intercept_only=chosen<0, validation_mse=best_mse,
                train_z=zf, dual=dual, outcome_mean=mu, **transform)

def predict_checkpoint(checkpoint, x):
    from pathlib import Path
    with np.load(checkpoint, allow_pickle=False) as z:
        if "shared_transform" in z:
            with np.load(Path(checkpoint).parent/str(z["shared_transform"]),allow_pickle=False) as base:
                raw=np.asarray(x,float)[:,base["feature_index"]]
                q=(np.where(np.isfinite(raw),raw,base["median"])-base["mean"])/base["scale"]
                train=base["train_z"][z["fit_indices"]]
                center=train.mean(axis=0)
                return (q-center) @ (train-center).T / train.shape[1] @ z["dual"] + z["outcome_mean"]
        raw = np.asarray(x,float)[:,z["feature_index"]]
        transformed = (np.where(np.isfinite(raw),raw,z["median"])-z["mean"])/z["scale"]
        return transformed @ z["train_z"].T / z["train_z"].shape[1] @ z["dual"] + z["outcome_mean"]

def fit_prepared_group(z,yt,q,yv,zf,yf,qt,lambdas):
    # Fold preprocessing uses all training X (including rows lacking this Y).
    # Center again over observed Y rows to preserve an unpenalized intercept.
    zmean=z.mean(axis=0);ym=yt.mean(axis=0)
    d,u,a,b=spectral(z-zmean,q-zmean,yt-ym)
    grid=np.unique(np.asarray(lambdas,float))[::-1]
    best=np.mean((yv-ym)**2,axis=0);chosen=np.full(yt.shape[1],-1,int)
    for j,lam in enumerate(grid):
        mse=np.mean((a@(b/(d[:,None]+lam))+ym-yv)**2,axis=0)
        use=mse<best-1e-14*np.maximum(1,best)
        chosen[use]=j;best[use]=mse[use]
    mean=zf.mean(axis=0);mu=yf.mean(axis=0)
    train=zf-mean;query=qt-mean
    d,u,a,b=spectral(train,query,yf-mu)
    selected=grid[np.maximum(chosen,0)]
    dual=u@(b/(d[:,None]+selected));dual[:,chosen<0]=0
    pred=query@train.T/train.shape[1]@dual+mu
    return dict(prediction=pred,baseline=np.broadcast_to(mu,pred.shape).copy(),
                selected_lambda=np.where(chosen<0,np.nan,selected),
                intercept_only=chosen<0,validation_mse=best,train_z=train,
                dual=dual,outcome_mean=mu)

def metrics(observed, predicted, baseline):
    ok = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(baseline)
    y,p,b = observed[ok],predicted[ok],baseline[ok]
    n = len(y)
    variance = np.sum((y-y.mean())**2) if n else 0
    sse = np.sum((y-p)**2)
    bsse = np.sum((y-b)**2)
    rho = float(spearmanr(y,p).statistic) if n>=3 and np.ptp(y)>0 and np.ptp(p)>0 else np.nan
    return dict(n_oof=n, r2=1-sse/variance if n>=3 and variance>1e-16 else np.nan,
                rmse=float(np.sqrt(sse/n)) if n else np.nan,
                spearman=rho, baseline_rmse=float(np.sqrt(bsse/n)) if n else np.nan,
                skill_vs_train_mean=1-sse/bsse if n>=3 and bsse>1e-16 else np.nan)

def grouped_oof(x, y, folds, *, output, feature_ids, patient_ids, target_ids,
                lambdas=LAMBDAS, min_train=10, min_validation=3, min_test=3):
    """Five outer folds; the next fold validates; refit the remaining four folds.

    Targets with identical observed-patient masks share a decomposition, but each
    target selects its own penalty. Unobserved labels never become training rows.
    """
    from pathlib import Path
    import pandas as pd
    x,y,folds = np.asarray(x,float),np.asarray(y,float),np.asarray(folds,int)
    if x.ndim!=2 or y.ndim!=2 or len(x)!=len(y) or len(folds)!=len(y):
        raise ValueError("Patient axes disagree")
    if len(set(patient_ids))!=len(patient_ids) or len(feature_ids)!=x.shape[1] or len(target_ids)!=y.shape[1]:
        raise ValueError("Invalid entity axes")
    if not set(folds)<=set(range(5)):
        raise ValueError("Five patient folds required")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    oof=np.full_like(y,np.nan);baseline=np.full_like(y,np.nan)
    groups={}
    for j in range(y.shape[1]):
        key=np.packbits(np.isfinite(y[:,j])).tobytes()
        groups.setdefault(key,[]).append(j)
    audit=[];top=[]
    for fold in range(5):
        train_all=np.flatnonzero((folds!=fold)&(folds!=(fold+1)%5))
        val_all=np.flatnonzero(folds==(fold+1)%5)
        fit_all=np.flatnonzero(folds!=fold)
        test_all=np.flatnonzero(folds==fold)
        zi,zv,_=preprocess(x[train_all],x[val_all])
        zfit,ztest,transform=preprocess(x[fit_all],x[test_all])
        inner=np.zeros((len(x),zi.shape[1]));inner[train_all]=zi;inner[val_all]=zv
        outer=np.zeros((len(x),zfit.shape[1]));outer[fit_all]=zfit;outer[test_all]=ztest
        shared=f"fold_{fold}_transform.npz"
        np.savez_compressed(output/shared,train_z=zfit,feature_ids=np.asarray(feature_ids,dtype=str),
                            fit_patient_ids=np.asarray(patient_ids,dtype=str)[fit_all],**transform)
        for gi,cols in enumerate(groups.values()):
            cols=np.asarray(cols)
            finite=np.isfinite(y[:,cols[0]])
            tr=np.flatnonzero(finite & (folds!=fold) & (folds!=(fold+1)%5))
            va=np.flatnonzero(finite & (folds==(fold+1)%5))
            te=np.flatnonzero(finite & (folds==fold))
            status="FITTED"
            if len(tr)<min_train or len(va)<min_validation or len(te)<min_test:
                status="INSUFFICIENT_OBSERVED_PATIENTS"
            result=None
            if status=="FITTED":
                try:
                    fit_rows=np.r_[tr,va]
                    result=fit_prepared_group(inner[tr],y[np.ix_(tr,cols)],
                             inner[va],y[np.ix_(va,cols)],outer[fit_rows],
                             y[np.ix_(fit_rows,cols)],outer[te],lambdas)
                    result["feature_index"]=transform["feature_index"]
                except ValueError as exc:
                    if str(exc).startswith("NO_"):status=str(exc)
                    else:raise
            for k,j in enumerate(cols):
                row=dict(target_id=target_ids[j],outer_fold=fold,status=status,
                         n_train=len(tr),n_validation=len(va),n_test=len(te))
                if result is not None:
                    row.update(selected_lambda=result["selected_lambda"][k],
                               intercept_only=bool(result["intercept_only"][k]),
                               validation_mse=result["validation_mse"][k],
                               n_features=len(result["feature_index"]))
                audit.append(row)
            if result is None:continue
            oof[np.ix_(te,cols)]=result["prediction"]
            baseline[np.ix_(te,cols)]=result["baseline"]
            dest=output/f"fold_{fold}_mask_{gi:04d}.npz"
            state={k:v for k,v in result.items() if k not in ("prediction","baseline","train_z","feature_index")}
            state.update(shared_transform=shared,fit_indices=np.searchsorted(fit_all,np.r_[tr,va]),
                         target_ids=np.asarray(target_ids,dtype=str)[cols],
                         fit_patient_ids=np.asarray(patient_ids,dtype=str)[np.r_[tr,va]],
                         test_patient_ids=np.asarray(patient_ids,dtype=str)[te])
            np.savez_compressed(dest,**state)
            # Save interpretable linear weights without materializing all dense coefficients on disk.
            for start in range(0,len(cols),256):
                take=slice(start,start+256)
                coef=result["train_z"].T @ result["dual"][:,take] / result["train_z"].shape[1]
                n=min(20,len(coef))
                ind=np.argpartition(np.abs(coef),-n,axis=0)[-n:]
                for kk,j in enumerate(cols[take]):
                    for rank,fi in enumerate(sorted(ind[:,kk],key=lambda i:-abs(coef[i,kk])),1):
                        if coef[fi,kk]==0:continue
                        top.append(dict(target_id=target_ids[j],outer_fold=fold,rank=rank,
                                        lncrna_id=feature_ids[result["feature_index"][fi]],
                                        standardized_x_coefficient=coef[fi,kk]))
        print(f"outer_fold={fold} completed; heldout_predictions={int(np.isfinite(oof).sum())}",flush=True)
    return oof,baseline,pd.DataFrame(audit),pd.DataFrame(top)
