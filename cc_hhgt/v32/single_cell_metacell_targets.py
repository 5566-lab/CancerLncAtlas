"""Metacell association strength targets; descriptive null, no FDR claim."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

TARGET_POLICY = 'ABS_WITHIN_DONOR_CELLTYPE_METACELL_SPEARMAN_RHO'
SCORE_SEMANTICS = 'EXPECTED_ABS_WITHIN_DONOR_METACELL_SPEARMAN_RHO'


def targets(x, activity, lnc_index, pathway_index, *, min_metacells=20,
            min_detected=5, min_detection_fraction=.1, null_repeats=64, seed=1729):
    x=np.asarray(x,float); activity=np.asarray(activity,float)
    li=np.asarray(lnc_index,int); pi=np.asarray(pathway_index,int)
    if x.ndim!=2 or activity.ndim!=2 or len(x)!=len(activity) or len(li)!=len(pi):
        raise ValueError('TARGET_SHAPE')
    if not np.isfinite(x).all() or not np.isfinite(activity).all():
        raise ValueError('NONFINITE_METACELL_MEASUREMENTS')
    if (li>=x.shape[1]).any() or (pi>=activity.shape[1]).any():
        raise ValueError('TARGET_INDEX')
    n=len(x); reason=np.full(len(li),'',object)
    reason[(li<0)|(pi<0)]='EXACT_FEATURE_UNAVAILABLE'
    if n<min_metacells:
        reason[reason=='']='INSUFFICIENT_METACELLS'
    frame=pd.DataFrame({'rho':np.nan,'association_target':np.nan,
                        'n_metacells':n,'lnc_detection_rate':np.nan,
                        'null_abs_rho_mean':np.nan,'null_abs_rho_sd':np.nan,
                        'unavailable_reason':reason})
    usable=np.flatnonzero(reason=='')
    if not len(usable):
        frame['available']=False
        return frame
    xr=rankdata(x,axis=0); yr=rankdata(activity,axis=0)
    xr-=xr.mean(axis=0); yr-=yr.mean(axis=0)
    xn=np.linalg.norm(xr,axis=0); yn=np.linalg.norm(yr,axis=0)
    detected=(x>0).sum(axis=0)
    frame.loc[usable,'lnc_detection_rate']=detected[li[usable]]/n
    sufficient=detected[li[usable]]>=max(min_detected,int(np.ceil(n*min_detection_fraction)))
    variable=(xn[li[usable]]>0)&(yn[pi[usable]]>0)
    frame.loc[usable[~sufficient],'unavailable_reason']='INSUFFICIENT_LNC_DETECTION'
    frame.loc[usable[sufficient&~variable],'unavailable_reason']='CONSTANT_LNC_OR_PATHWAY'
    usable=usable[sufficient&variable]
    rng=np.random.default_rng(seed)
    permutations=[rng.permutation(n) for _ in range(null_repeats)]
    for start in range(0,len(usable),128):
        rows=usable[start:start+128]
        a=xr[:,li[rows]]/xn[li[rows]]; b=yr[:,pi[rows]]/yn[pi[rows]]
        rho=np.clip((a*b).sum(axis=0),-1,1)
        frame.loc[rows,'rho']=rho;frame.loc[rows,'association_target']=np.abs(rho)
        if permutations:
            null=np.asarray([np.abs((a*b[p]).sum(axis=0)) for p in permutations])
            frame.loc[rows,'null_abs_rho_mean']=null.mean(axis=0)
            frame.loc[rows,'null_abs_rho_sd']=null.std(axis=0)
    frame['available']=frame.unavailable_reason.eq('')
    frame.loc[frame.available,'unavailable_reason']=None
    frame['target_policy']=TARGET_POLICY
    frame['null_semantics']='DESCRIPTIVE_CONDITIONAL_METACELL_PERMUTATION_NOT_FDR_OR_POPULATION_REPLICATION'
    return frame
