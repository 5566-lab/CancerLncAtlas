"""Dependency-calibrated Fisher combination for functional relationship evidence."""
import numpy as np

def empirical_fisher(pvalue, available, correlation, groups=None, simulations=100000, seed=20261110, cache=None):
    """Calibrate Fisher statistic against correlated-normal two-sided null draws.

    Monte Carlo p=(1+exceedances)/(B+1); single readouts are returned exactly.
    Unknown dependence uses Bonferroni rather than assuming independence.
    This is a calibrated combination under the estimated Gaussian copula;
    finite-sample real-data null checks remain required before production use.
    """
    import hashlib,json
    from pathlib import Path
    from scipy.special import log_ndtr
    p=np.asarray(pvalue,float);av=np.asarray(available,bool)&np.isfinite(p)
    if p.ndim!=2 or av.shape!=p.shape:raise ValueError('FUSION_AXIS_MISMATCH')
    if np.any((p[av]<0)|(p[av]>1)):raise ValueError('P_OUT_OF_RANGE')
    k=p.shape[1];R=np.asarray(correlation,float)
    if R.shape!=(k,k) or not np.allclose(R,R.T,equal_nan=True) or not np.allclose(np.diag(R),1):raise ValueError('INVALID_CORRELATION')
    groups=[[i] for i in range(k)] if groups is None else groups
    if sorted(i for g in groups for i in g)!=list(range(k)):raise ValueError('INVALID_GROUPS')
    if simulations<1000:raise ValueError('INSUFFICIENT_NULL_RESOLUTION')
    result=np.full(len(p),np.nan);method=np.zeros(len(p),np.uint8);receipts=[]
    patterns,labels=np.unique(av,axis=0,return_inverse=True)
    for a,pattern in enumerate(patterns):
        rows=np.flatnonzero(labels==a);ix=np.flatnonzero(pattern)
        if not len(ix):continue
        if len(ix)==1:result[rows]=p[rows,ix[0]];method[rows]=1;continue
        rr=R[np.ix_(ix,ix)];weight=np.ones(len(ix));positions={old:new for new,old in enumerate(ix)}
        for g in groups:
            active=[positions[i] for i in g if i in positions]
            if active:weight[active]=1/len(active)
        if not np.isfinite(rr).all():
            result[rows]=np.minimum(1.,len(ix)*np.min(p[np.ix_(rows,ix)],axis=1));method[rows]=3;receipts.append({'indices':ix.tolist(),'method':'BONFERRONI_UNKNOWN_DEPENDENCE','rows':len(rows)});continue
        if np.all(np.abs(rr)>=1-1e-12):
            result[rows]=np.exp(np.log(np.clip(p[np.ix_(rows,ix)],np.finfo(float).tiny,1.))@weight/weight.sum());method[rows]=4;continue
        vals,vecs=np.linalg.eigh(rr);negative=float(max(0.,-vals.min()));vals=np.clip(vals,0,None);fixed=(vecs*vals)@vecs.T;sd=np.sqrt(np.diag(fixed));fixed=fixed/sd[:,None]/sd[None,:];np.fill_diagonal(fixed,1.);vals,vecs=np.linalg.eigh(fixed);factor=vecs*np.sqrt(np.clip(vals,0,None))
        info={'correlation':np.round(fixed,12).tolist(),'weights':weight.tolist(),'simulations':simulations,'seed':seed,'method':'TWO_SIDED_EMPIRICAL_FISHER_GAUSSIAN_COPULA_V1'}
        key=hashlib.sha256(json.dumps(info,sort_keys=True).encode()).hexdigest();path=None if cache is None else Path(cache)/(key+'.npy')
        if path is not None and path.exists():null=np.load(path,allow_pickle=False);assert null.shape==(simulations,) and np.all(np.diff(null)>=0)
        else:
            rng=np.random.default_rng(seed);g=rng.standard_normal((simulations,len(ix)))@factor.T
            null=np.sort((-2*(np.log(2.)+log_ndtr(-np.abs(g))))@weight)
            if path is not None:
                path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
                with tmp.open('wb') as f:np.save(f,null,allow_pickle=False)
                tmp.replace(path);path.with_suffix('.json').write_text(json.dumps(info,indent=2))
        obs=(-2*np.log(np.clip(p[np.ix_(rows,ix)],np.finfo(float).tiny,1.)))@weight
        result[rows]=(1+simulations-np.searchsorted(null,obs,side='left'))/(simulations+1);method[rows]=2
        receipts.append({'indices':ix.tolist(),'method':info['method'],'rows':len(rows),'null_cache_key':key,'minimum_p':1/(simulations+1),'source_negative_eigenvalue':negative,'projection_max_change':float(np.max(abs(fixed-rr)))})
    return {'p':result,'available':av.any(1),'calibration_method':method,'references':receipts}
