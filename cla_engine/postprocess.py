"""Descriptive OOF consensus and per-target subtypes, never a test-tuned deployment score."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score,adjusted_rand_score
from .common import read_json,atomic_json,fail,status,verify_manifest,digest
from .lineage import verify_signed_file,FIELDS

def labels_for_distance(D,k):
    import inspect
    kw={'metric':'precomputed'} if 'metric' in inspect.signature(AgglomerativeClustering).parameters else {'affinity':'precomputed'}
    return AgglomerativeClustering(n_clusters=k,linkage='average',**kw).fit_predict(D)

def subtype_matrix(frame,seed=20260726):
    # Signed observed effects used only for descriptive downstream structure.
    mat=frame.pivot(index='cancer_id',columns='lncrna_id',values='effect').fillna(0);names=mat.index.tolist();X=mat.to_numpy(float)
    if len(names)<6 or X.shape[1]<3:return {c:0 for c in names},{'state':'UNAVAILABLE_LOW_COVERAGE','k':0}
    norm=np.linalg.norm(X,axis=1);S=np.divide(X@X.T,norm[:,None]*norm[None,:],out=np.zeros((len(X),len(X))),where=norm[:,None]*norm[None,:]>0);D=np.clip(1-S,0,2);np.fill_diagonal(D,0)
    if np.max(D)<1e-8:return {c:1 for c in names},{'state':'CONSERVED_K1','k':1}
    options=[]
    for k in range(2,min(4,len(X)//3)+1):
        lab=labels_for_distance(D,k)
        if min(np.bincount(lab))<2:continue
        options.append((silhouette_score(D,lab,metric='precomputed'),k,lab))
    if not options:return {c:1 for c in names},{'state':'CONTINUOUS_HETEROGENEITY','k':1}
    score,k,lab=max(options,key=lambda a:(a[0],-a[1]));rng=np.random.default_rng(seed);stability=[]
    for b in range(25):
        idx=rng.integers(0,X.shape[1],X.shape[1]);xb=X[:,idx];n=np.linalg.norm(xb,axis=1);sim=np.divide(xb@xb.T,n[:,None]*n[None,:],out=np.zeros_like(D),where=n[:,None]*n[None,:]>0);dd=np.clip(1-sim,0,2);np.fill_diagonal(dd,0);stability.append(adjusted_rand_score(lab,labels_for_distance(dd,k)))
    stable=score>=.25 and float(np.median(stability))>=.8
    return ({c:int(l)+1 for c,l in zip(names,lab)} if stable else {c:1 for c in names}),{'state':'FEATURE_BOOTSTRAP_SUBTYPES_EXPLORATORY' if stable else 'CONTINUOUS_HETEROGENEITY','k':int(k) if stable else 1,'silhouette':float(score),'lnc_feature_bootstrap_median_ARI':float(np.median(stability)),'patient_or_external_stability_verified':False}

def make_sets(roots,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);receipts=[read_json(Path(p)/'EVALUATION_RECEIPT.json') for p in roots]
    if {r['fold'] for r in receipts}!=set(range(5)) or len(receipts)!=5:fail('FIVE_DISTINCT_FOLDS_REQUIRED','No repeated-fold pseudoreplication')
    cfg_path=Path(roots[0])/'EVALUATION_CONFIG.json'
    if not cfg_path.is_file():fail('POSTPROCESS_LINEAGE_MISSING','R6.2 signed evaluation configuration required')
    cfg=read_json(cfg_path)
    for root,r in zip(roots,receipts):
        verify_manifest(root);verify_signed_file(Path(root)/'MANIFEST.json',cfg,'OUTER_EVALUATION')
        if r.get('profile')!='formal' or r.get('state')!='OUTER_TEST_EVALUATED_NOT_DEPLOYED' or r.get('training_state')!='TRAINED_FORMAL_CANDIDATE':fail('POSTPROCESS_NOT_FORMAL','No diagnostic or incomplete folds')
        if r.get('config_sha256')!=digest(cfg):fail('POSTPROCESS_CONFIG_DRIFT','Inconsistent evaluations')
    for field in FIELDS:
        if len({r.get('lineage',{}).get(field) for r in receipts})!=1 or receipts[0].get('lineage',{}).get(field) is None:fail('POSTPROCESS_MIXED_LINEAGE',field)
    frames=[]
    for root in roots:
        verify_manifest(root);frames.append(pd.read_csv(Path(root)/'OOF_ASSOCIATIONS.tsv.gz',sep='\t'))
    f=pd.concat(frames,ignore_index=True);key=['cancer_id','task','lncrna_id','target_id']
    if f.duplicated(key+['patient_fold_id']).any():fail('OOF_DUPLICATE_KEY','Duplicate relation/fold')
    f['available_bool']=f.availability.eq(True);f['support']=f.available_bool&f.heldout_q.le(.05)&f.heldout_effect.abs().ge(.2)&f.model_logit.ge(0)
    grouped=[]
    for ident,g in f.groupby(key,observed=True):
        good=g.loc[g.support];available=int(g.available_bool.sum());positive=int((good.heldout_effect>0).sum());negative=int((good.heldout_effect<0).sum());consistency=max(positive,negative)/max(len(good),1)
        grouped.append(dict(zip(key,ident))|{'available_folds':available,'supporting_folds':len(good),'direction_consistency':consistency,'effect':float(g.loc[g.available_bool,'heldout_effect'].median()),'mean_model_logit':float(g.model_logit.mean()),'included':available==5 and len(good)>=4 and consistency>=.8})
    catalog=pd.DataFrame(grouped);catalog.to_csv(out/'RELATION_CONSENSUS.tsv.gz',sep='\t',index=False);kept=catalog.loc[catalog.included].copy()
    # A storage/display cap, not a claimed optimum or FDR proof; rank before capping is retained.
    kept=kept.sort_values(['cancer_id','task','target_id','mean_model_logit'],ascending=[True,True,True,False]);kept['set_rank']=kept.groupby(['cancer_id','task','target_id'],observed=True).cumcount()+1
    kept.loc[kept.set_rank.le(50)].to_csv(out/'CANCER_GENE_SETS_EXPLORATORY.tsv',sep='\t',index=False)
    pan=[]
    for ident,g in kept.groupby(['task','lncrna_id','target_id'],observed=True):
        sign=(g.effect>0).sum();con=max(sign,len(g)-sign)/len(g)
        if len(g)>=5 and con>=.8:pan.append(dict(zip(['task','lncrna_id','target_id'],ident))|{'supporting_cancers':len(g),'direction_consistency':con,'cancers':';'.join(g.cancer_id),'lineage_criterion':'NOT_EVALUATED','high_confidence_label_allowed':False})
    pd.DataFrame(pan,columns=['task','lncrna_id','target_id','supporting_cancers','direction_consistency','cancers','lineage_criterion','high_confidence_label_allowed']).to_csv(out/'CROSS_CANCER_CANDIDATES_NOT_FINAL.tsv',sep='\t',index=False)
    sub=[];audits=[]
    for (task,target),g in kept.groupby(['task','target_id'],observed=True):
        assignment,audit=subtype_matrix(g);audits.append({'task':task,'target':target,**audit})
        for c,k in assignment.items():sub.append({'task':task,'target_id':target,'cancer_id':c,'subtype_id':f'{task}:{target}:S{k}' if k else None,'status':audit['state']})
    pd.DataFrame(sub,columns=['task','target_id','cancer_id','subtype_id','status']).to_csv(out/'TARGET_CONTEXT_SUBTYPES_EXPLORATORY.tsv',sep='\t',index=False)
    atomic_json(out/'SUBTYPE_AUDIT.json',audits);atomic_json(out/'POSTPROCESS_RECEIPT.json',{'state':'DESCRIPTIVE_OOF_SETS_NOT_VALIDATED_RELEASE','subtype_feedback':False,'outer_test_used_for_descriptive_aggregation':True,'independent_gene_set_performance_claimed':False,'independent_patient_bootstrap_complete':False,'pan_cancer_lineages_not_bound':True,'production_deploy':False})
    return status(out,'DESCRIPTIVE_OOF_SETS_NOT_VALIDATED_RELEASE')
