"""Explicit paired units for State metrics: target -> family -> cancer.

Pooled rankings are additional diagnostics, never substituted for the registered
State gate. AUPRC and precision use the same evaluable target set within family.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score
from .common import fail


def paired_task_metrics(y,score,baseline,available,target,*,task,targets,families=None,k=100):
    y=np.asarray(y);s=np.asarray(score);b=np.asarray(baseline);ok=np.asarray(available,dtype=bool);tid=np.asarray(target)
    if any(a.shape!=y.shape for a in (s,b,ok,tid)) or y.ndim!=1:
        fail('METRIC_SHAPE_MISMATCH','Paired metric arrays must share one ordered candidate axis')
    if not np.isfinite(s).all() or not np.isfinite(b).all() or not np.isin(y[ok],[0,1]).all():
        fail('METRIC_VALUES_INVALID','Labels/scores must be valid on the declared mask')
    if np.any(tid<0) or np.any(tid>=len(targets)):fail('METRIC_TARGET_OUT_OF_RANGE','Bad target index')
    if not isinstance(k,int) or k<1:fail('METRIC_K_INVALID','k must be a positive integer')
    n=int(ok.sum());r={'n':n,'n_positive':int(y[ok].sum()),'available':False}
    if np.unique(y[ok]).size<2:
        return {**r,'reason':'SINGLE_CLASS_OR_NO_PAIRS'}
    def unit(mask):
        a=y[mask];x=s[mask];z=b[mask];kk=min(k,len(a))
        return {'auprc':float(average_precision_score(a,x)),
            'baseline_auprc':float(average_precision_score(a,z)),
            'precision_at_k':float(a[np.argsort(-x,kind='stable')[:kk]].mean()),
            'baseline_precision_at_k':float(a[np.argsort(-z,kind='stable')[:kk]].mean()),'k_effective':kk}
    pooled=unit(ok)
    r['pooled_diagnostic']=pooled
    if task=='pathway':
        return {**r,**pooled,'available':True,'unit':'CANCER_PATHWAY_CANDIDATE_POOL',
                'delta_auprc':pooled['auprc']-pooled['baseline_auprc']}
    if task!='state' or families is None:fail('METRIC_TASK_INVALID',str(task))
    target_rows=[];by_family={}
    keys=('auprc','baseline_auprc','precision_at_k','baseline_precision_at_k')
    for j,t in enumerate(targets):
        mask=ok&(tid==j);row={'target_id':t,'family':families[t],'n':int(mask.sum()),'n_positive':int(y[mask].sum()),'available':np.unique(y[mask]).size==2}
        if row['available']:
            row.update(unit(mask));by_family.setdefault(families[t],[]).append(row)
        else:row['reason']='NOT_TWO_CLASS_SAME_TARGET;NOT_USED_IN_EITHER_GATE_METRIC'
        target_rows.append(row)
    groups=[]
    for name,items in sorted(by_family.items()):
        groups.append({'family':name,'n_evaluable_targets':len(items),**{key:float(np.mean([i[key] for i in items])) for key in keys}})
    if not groups:return {**r,'targets':target_rows,'reason':'NO_TWO_CLASS_STATE_TARGET'}
    averaged={key:float(np.mean([g[key] for g in groups])) for key in keys}
    return {**r,**averaged,'available':True,'families':groups,'targets':target_rows,
        'unit':'TARGET_MACRO_WITHIN_FAMILY_THEN_FAMILY_MACRO_WITHIN_CANCER',
        'delta_auprc':averaged['auprc']-averaged['baseline_auprc']}
