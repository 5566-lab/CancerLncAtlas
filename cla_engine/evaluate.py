"""CPU-only outer-test evaluation and all-cancer admission. No test-dependent alpha selection."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
from .common import digest,read_json,atomic_json,verify_manifest,sha,fail,status,commit_manifest
from .train import load_npz
from .lineage import validate_prepared,verify_sealed,verify_completion,sign_file,verify_signed_file,FIELDS


def evaluate_fold(prepared,training,out):
    prepared=Path(prepared);training=Path(training);out=Path(out)
    manifest,spec,cfg,lineage=validate_prepared(prepared)
    from .test_access import freeze_model,record_outer_open
    receipt,_=verify_completion(prepared,training)
    freeze=freeze_model(prepared,training)
    sealed=prepared.parent/(prepared.name+'__sealed_test');verify_sealed(prepared,sealed)
    if out.exists() and (out/'EVALUATION_RECEIPT.json').exists():
        r=read_json(out/'EVALUATION_RECEIPT.json')
        if r['training_receipt_sha256']!=sha(training/'TRAINING_RECEIPT.json'):fail('TEST_REEVALUATION_WITH_CHANGED_MODEL','Do not select a new model on test')
        verify_manifest(out);verify_signed_file(out/'MANIFEST.json',cfg,'OUTER_EVALUATION')
        return r
    record_outer_open(prepared,training,purpose='PRIMARY_OUTER_EVALUATION',output=out)
    out.mkdir(parents=True,exist_ok=True);metrics=[];rows=[]
    objective=spec.get('relation_objective','binary')
    if objective not in {'binary','observed_fused_rms_effect'}:
        fail('RELATION_OBJECTIVE_UNKNOWN','Outer evaluation objective is unknown',objective=objective)
    for cancer in spec['cancers']:
        for task in ['pathway','state']:
            pred=load_npz(training/'predictions'/f'{cancer}__{task}.npz');test=load_npz(sealed/f'{cancer}__validation__{task}.npz');ok=test['available'].astype(bool)
            source=load_npz(prepared/'views'/f'{cancer}__validation'/f'{task}.npz')
            if not np.array_equal(pred['lnc'],source['lnc']) or not np.array_equal(pred['target'],source['target']):fail('TEST_ROW_IDENTITY_MISMATCH','Prediction candidate keys differ')
            if not np.array_equal(test['lnc'],source['lnc']) or not np.array_equal(test['target'],source['target']):fail('SEALED_CANDIDATE_IDENTITY_MISMATCH','Sealed candidates differ from prepared order')
            if len(pred['score'])!=len(ok):fail('TEST_ROW_MISMATCH','Prediction/test identity count differs')
            targets=spec['nodes']['pathway'] if task=='pathway' else spec['states']
            if objective=='observed_fused_rms_effect':
                from .relation_metrics import paired_relation_strength_metrics
                m=paired_relation_strength_metrics(test['label'],pred['score'],pred['baseline'],ok,pred['target'],
                    task=task,targets=targets,families=spec['state_families'])
                if m['available']:
                    m['delta_mse']=m['baseline_mse']-m['mse']
                f=pd.DataFrame({'patient_fold_id':spec['fold'],'cancer_id':cancer,'task':task,
                    'lncrna_id':[spec['nodes']['lncRNA'][i] for i in pred['lnc']],
                    'target_id':[targets[i] for i in pred['target']],
                    'model_strength':pred['score'],'baseline_strength':pred['baseline'],
                    'heldout_rms_effect':np.where(ok,test['label'],np.nan),'availability':ok})
            else:
                from .metrics import paired_task_metrics
                m=paired_task_metrics(test['label'],pred['score'],pred['baseline'],ok,pred['target'],
                    task=task,targets=targets,families=spec['state_families'],k=int(cfg['gates']['precision_k']))
                if m['available']:
                    m['auroc']=float(roc_auc_score(test['label'][ok],pred['score'][ok]))
                    m['auroc_unit']='POOLED_DIAGNOSTIC_NOT_GATE'
                direction_ok=ok & np.isfinite(test['effect']) & (np.abs(test['effect'])>=.15)
                m['direction_n']=int(direction_ok.sum())
                m['direction_accuracy']=float(np.mean((pred['direction_logit'][direction_ok]>=0)==(test['effect'][direction_ok]>=0))) if direction_ok.any() else None
                m['direction_validated_for_release']=False
                f=pd.DataFrame({'patient_fold_id':spec['fold'],'cancer_id':cancer,'task':task,'lncrna_id':[spec['nodes']['lncRNA'][i] for i in pred['lnc']],'target_id':[targets[i] for i in pred['target']],'model_logit':pred['score'],'baseline_logit':pred['baseline'],'predicted_direction':np.where(pred['direction_logit']>=0,'positive','negative'),'heldout_effect':test['effect'],'heldout_p':test['p'],'heldout_q':test['q'],'heldout_n':test['n'],'availability':ok})
            m.update(fold=spec['fold'],cancer_id=cancer,task=task)
            metrics.append(m)
            rows.append(f)
    pd.concat(rows,ignore_index=True).to_csv(out/'OOF_ASSOCIATIONS.tsv.gz',sep='\t',index=False)
    pd.DataFrame(metrics).to_csv(out/'METRICS.tsv',sep='\t',index=False)
    r={'fold':spec['fold'],'profile':read_json(prepared/'MANIFEST.json')['metadata']['profile'],'synthetic_fixture':bool(cfg.get('synthetic_fixture',False)),'config_sha256':digest(cfg),'state':'OUTER_TEST_EVALUATED_NOT_DEPLOYED','alpha_from_validation':receipt['alpha'],'training_receipt_sha256':sha(training/'TRAINING_RECEIPT.json'),'prepared_sha256':sha(prepared/'MANIFEST.json'),'model_freeze_sha256':freeze['freeze_sha256'],'metrics_contract':cfg['metrics_contract'],'relation_objective':objective,'tests_used_to_select_checkpoint':False,'metrics':metrics,'lineage':lineage,'sealed_manifest_sha256':sha(sealed/'MANIFEST.json'),'training_state':receipt['state'],'model_signature':receipt['signature'],'expected_cancers':spec['cancers'],'candidate_catalog':read_json(prepared/'CANDIDATES.json')}
    atomic_json(out/'EVALUATION_CONFIG.json',cfg)
    atomic_json(out/'EVALUATION_RECEIPT.json',r);commit_manifest(out,{'role':'OUTER_TEST_RESULTS','fold':spec['fold']});sign_file(out/'MANIFEST.json',cfg,'OUTER_EVALUATION');status(out,r['state']);return r

def gate(evaluation_roots,cfg,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);receipts=[read_json(Path(p)/'EVALUATION_RECEIPT.json') for p in evaluation_roots]
    objective=cfg.get('relation_objective','binary')
    if objective not in {'binary','observed_fused_rms_effect'} or {r.get('relation_objective','binary') for r in receipts}!={objective}:
        fail('GATE_RELATION_OBJECTIVE_MISMATCH','All folds must use the configured objective')
    if len({r['fold'] for r in receipts})!=len(receipts):fail('DUPLICATE_FOLD','Repeated fold in gate')
    if {r['fold'] for r in receipts}!=set(range(5)) or any(r['profile']!='formal' or r.get('synthetic_fixture',False) for r in receipts):fail('FORMAL_FIVE_FOLDS_REQUIRED','Diagnostic data cannot pass a formal release gate')
    for root,r in zip(evaluation_roots,receipts):
        verify_manifest(root);verify_signed_file(Path(root)/'MANIFEST.json',cfg,'OUTER_EVALUATION')
        if r.get('state')!='OUTER_TEST_EVALUATED_NOT_DEPLOYED' or r.get('training_state')!='TRAINED_FORMAL_CANDIDATE':fail('GATE_TRAINING_STATE_INVALID','Not a completed formal run')
        if not r.get('model_signature') or not r.get('sealed_manifest_sha256'):fail('GATE_LINEAGE_MISSING','Missing run/sealed binding')
        if r.get('lineage',{}).get('fold')!=r['fold']:fail('GATE_FOLD_MISMATCH','Inner fold differs')
        if r.get('metrics_contract')!=cfg['metrics_contract']:fail('GATE_METRICS_CONTRACT_MISMATCH','Wrong metric aggregation units')
        if not r.get('model_freeze_sha256'):fail('GATE_MODEL_FREEZE_MISSING','Outer metrics must bind a pre-existing frozen model')
        if r.get('config_sha256')!=digest(cfg):fail('GATE_CONFIG_DRIFT','Do not edit thresholds after seeing test')
    for field in FIELDS:
        values=[r.get('lineage',{}).get(field) for r in receipts]
        if any(v is None for v in values) or len(set(values))!=1:fail('GATE_MIXED_LINEAGE','Five folds have different common provenance',field=field)
    if len({r['model_signature'] for r in receipts})!=5:fail('GATE_MODEL_SIGNATURE_REUSED','Fold-specific model signatures must differ')
    if len({digest(r['candidate_catalog']) for r in receipts})!=1:fail('GATE_CANDIDATE_DRIFT','Same fixed candidate universe/order required across folds')
    for r in receipts:
        expected={(c,t) for c in r['expected_cancers'] for t in ('pathway','state')}
        metrics=r['metrics'];keys=[(m['cancer_id'],m['task']) for m in metrics]
        if set(keys)!=expected or len(keys)!=len(expected) or any(m['fold']!=r['fold'] for m in metrics):fail('GATE_METRIC_GRID_INVALID','Duplicate/missing/wrong-fold metrics')
    results={};rng=np.random.default_rng(cfg['seed']);all_metrics=pd.DataFrame([m for r in receipts for m in r['metrics']]);overall=True
    if objective=='observed_fused_rms_effect':
        threshold=cfg['gates'].get('formal_min_delta_mse')
        if threshold is None or not np.isfinite(threshold) or threshold<0:
            fail('CONTINUOUS_FORMAL_THRESHOLD_MISSING','Declare a nonnegative MSE improvement threshold')
        if 'delta_mse' not in all_metrics or not np.isfinite(all_metrics.loc[all_metrics.available.eq(True),'delta_mse']).all():
            fail('CONTINUOUS_OUTER_METRIC_INVALID','Observed folds need finite paired MSE differences')
        for task in ('pathway','state'):
            f=all_metrics.loc[all_metrics.task.eq(task)&all_metrics.available.eq(True)]
            by=f.groupby('cancer_id',observed=True).agg(delta=('delta_mse','mean'),folds=('fold','nunique'))
            by=by.loc[by.folds.eq(5)]
            if len(by)<cfg['gates']['minimum_evaluable_cancers']:
                results[task]={'pass':False,'reason':'INSUFFICIENT_FIVE_FOLD_CANCERS','n_cancers':len(by)}
                overall=False
                continue
            d=by.delta.to_numpy();boots=np.mean(rng.choice(d,size=(2000,len(d)),replace=True),axis=1)
            ci=np.quantile(boots,[.025,.975])
            passed=float(d.mean())>=threshold and ci[0]>0 and float((d>0).mean())>=2/3
            results[task]={'pass':bool(passed),'mean_delta_mse':float(d.mean()),
                'cancer_cluster_bootstrap_ci':ci.tolist(),'positive_fraction':float((d>0).mean()),
                'worst_cancer_delta_mse':float(d.min()),'n_cancers':len(d)}
            overall&=passed
        r={'state':'PREDICTION_GATE_PASS_NOT_DEPLOYED' if overall else 'FORMAL_GATE_FAIL',
           'relation_objective':objective,'tasks':results,
           'production_release_allowed':False,'no_per_cancer_model_switching':True}
        atomic_json(out/'GLOBAL_GATE.json',r);status(out,r['state']);return r
    min_delta=cfg['gates']['formal_min_delta']
    for task in ['pathway','state']:
        f=all_metrics.loc[all_metrics.task.eq(task)&all_metrics.available.eq(True)]
        by=f.groupby('cancer_id',observed=True).agg(delta=('delta_auprc','mean'),folds=('fold','nunique'),precision=('precision_at_k','mean'),base_precision=('baseline_precision_at_k','mean'))
        by=by.loc[by.folds.eq(5)]
        if len(by)<cfg['gates']['minimum_evaluable_cancers']:
            results[task]={'pass':False,'reason':'INSUFFICIENT_FIVE_FOLD_CANCERS','n_cancers':len(by)};overall=False;continue
        d=by.delta.to_numpy();boots=np.mean(rng.choice(d,size=(2000,len(d)),replace=True),axis=1);ci=np.quantile(boots,[.025,.975])
        passed=float(d.mean())>=min_delta and ci[0]>0 and float((d>0).mean())>=2/3 and d.min()>=-.02 and float((by.precision-by.base_precision).mean())>=0
        results[task]={'pass':bool(passed),'mean_delta_auprc':float(d.mean()),'cancer_cluster_bootstrap_ci':ci.tolist(),'positive_fraction':float((d>0).mean()),'worst_cancer_delta':float(d.min()),'mean_precision_delta':float((by.precision-by.base_precision).mean()),'n_cancers':len(d)};overall&=passed
    r={'state':'PREDICTION_GATE_PASS_NOT_DEPLOYED' if overall else 'FORMAL_GATE_FAIL','tasks':results,'gene_set_stability_gate':'NOT_ASSESSED_REQUIRES_DOWNSTREAM_VALIDATION','production_release_allowed':False,'no_per_cancer_model_switching':True}
    atomic_json(out/'GLOBAL_GATE.json',r);status(out,r['state']);return r
