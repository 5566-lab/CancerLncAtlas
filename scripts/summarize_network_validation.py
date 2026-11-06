"""Verify completed phases and summarize strictly held-out repeated-CV results."""
import argparse,json,socket,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cla_engine.common import sha
from scripts.prepare_graph_ridge import dump
from scripts.run_network_validation import check_receipt
from scripts.plot_m0_increment import classify
from cc_hhgt.network_ridge import MODES,ARMS
from cc_hhgt.graph_ridge import target_policy
from cc_hhgt.m0_covariates import complete_rows
CANCERS=('BRCA','LUAD','COAD')
KEYS=['target_id','variant','arm']

def sufficient_statistics(p):
    p=p.copy();p['sy']=p.observed;p['sy2']=p.observed**2;p['sp']=p.predicted;p['sp2']=p.predicted**2;p['syp']=p.observed*p.predicted;p['sse']=(p.observed-p.predicted)**2
    a=p.groupby(KEYS,observed=True,sort=False).agg(n=('observed','size'),sy=('sy','sum'),sy2=('sy2','sum'),sp=('sp','sum'),sp2=('sp2','sum'),syp=('syp','sum'),sse=('sse','sum'))
    return a

def summarize(source,inputs,networks,output):
    assert socket.gethostname()=='149';src=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True);dump(out/'PREFLIGHT.json',dict(target_host='149',stage='NETWORK_VALIDATION_REPORT',source=str(src),code=sha(Path(__file__))))
    statuses=[];metrics=[];hyper=[];coverage=[];audits=[];partitions=[];receipt_codes=[]
    for c in CANCERS:
        case=src/c;meta=classify(pd.read_csv(case/'final/TARGETS.tsv',sep='\t'));meta['family']=meta.family.fillna('UNSPECIFIED')
        native=pd.read_parquet(Path(inputs)/c/'Y.parquet');cov=pd.read_parquet(Path(inputs)/c/'COVARIATES.parquet').reindex(native.index)
        meta['n_observed_y']=meta.target_id.map(native.notna().sum());meta['n_covariate_eligible_y']=[int((native[row['target_id']].notna().to_numpy()&complete_rows(cov,target_policy(row)['use_purity'])).sum()) for row in meta.to_dict('records')]
        phases=[f'repeat{r}_outer{k}' for r in range(2) for k in range(5)]+['final'];ad={};stats={0:[],1:[]}
        for phase in phases:
            folder=case/phase
            if not (folder/'COMPLETE.json').exists():raise ValueError('INCOMPLETE_PHASE: '+str(folder))
            check_receipt(folder);launch=json.loads((folder/'LAUNCH_PREFLIGHT.json').read_text());assert launch['target_host']=='149';receipt_codes.append(launch['code'])
            a=pd.read_parquet(folder/'AUDIT.parquet');assert not a.duplicated(KEYS).any();ad[phase]=a[KEYS+['status']];fit=a[a.status.eq('FITTED')].merge(meta[['target_id','display_layer','family']],on='target_id',validate='many_to_one')
            if not fit.inner_validations.eq(6).all() or not fit.n_inner_train_min.ge(12).all() or not fit.n_inner_validation_min.ge(3).all():raise ValueError('NESTED_CV_THRESHOLD_FAILED')
            for layer,excluded in [('PROTEIN_PATHWAY','PROTEIN'),('PHOSPHO_PTMSEA','PHOSPHO')]:
                if fit.loc[fit.display_layer.eq(layer),'allowed_blocks'].str.split(';').map(lambda a:excluded in a).any():raise ValueError('TARGET_ASSAY_LEAK')
            h=fit.groupby(['variant','arm','display_layer','parameter','lam'],observed=True).size().rename('fits').reset_index();h['phase']=phase;h['cancer_id']=c;hyper.append(h)
            audits.append(dict(cancer_id=c,phase=phase,rows=len(a),fitted=int(a.status.eq('FITTED').sum()),failed=int(a.status.ne('FITTED').sum()),checksums_verified=True,inner_validations=6))
            if phase!='final':
                repeat=int(phase[6]);p=pd.read_parquet(folder/'PREDICTIONS.parquet');assert np.isfinite(p[['observed','predicted']].to_numpy()).all();assert not p.duplicated(KEYS+['patient_id']).any();stats[repeat].append(sufficient_statistics(p));print('SUMMARIZED_PHASE',c,phase,len(p),flush=True);del p
        for repeat in [0,1]:
            aa=pd.concat([ad[f'repeat{repeat}_outer{k}'].assign(phase=str(k)) for k in range(5)]+[ad['final'].assign(phase='final')],ignore_index=True)
            success=aa.assign(fitted=aa.status.eq('FITTED')).groupby(KEYS,observed=True).agg(fitted_phases=('fitted','sum'),attempted_phases=('phase','size'));success['success']=success.fitted_phases.eq(6)&success.attempted_phases.eq(6)
            st=success.reset_index();st['repeat']=repeat;st['cancer_id']=c;statuses.append(st.merge(meta,on='target_id',validate='many_to_one'))
            ss=pd.concat(stats[repeat]).groupby(level=KEYS).sum();ss=ss.join(success[['success']]);ss['sst']=ss.sy2-ss.sy**2/ss.n;ss['r2']=1-ss.sse/ss.sst;ss['rmse']=np.sqrt(ss.sse/ss.n);den=np.sqrt(np.maximum(ss.sst*(ss.sp2-ss.sp**2/ss.n),0));ss['pearson']=(ss.syp-ss.sy*ss.sp/ss.n)/den;ss=ss.reset_index();ss['repeat']=repeat;ss['cancer_id']=c
            metrics.append(ss.merge(meta,on='target_id',validate='many_to_one'))
        fm=pd.read_parquet(Path(networks)/c/'FEATURE_MAP.parquet');prep=joblib.load(case/'final/PREPROCESS_purity1.joblib')
        for block,sp in prep['preprocess'].items():
            ff=fm[fm.block.eq(block)].sort_values('position');idx=sp['index'];idx=idx[idx<sp['original_width']];ff=ff.iloc[idx]
            for arm,f in ([('lncrna',ff[ff.rna_arm.eq('lncrna')]),('mrna',ff[ff.rna_arm.eq('mrna')]),('combined',ff)] if block=='RNA' else [('shared',ff)]):
                for mode in MODES:coverage.append(dict(cancer_id=c,block=block,arm=arm,mode=mode,quality_eligible_features=len(f),connected_features=int(f['module_'+mode].ge(0).sum()),fraction=float(f['module_'+mode].ge(0).mean())))
        part=pd.read_csv(Path(networks)/c/'PARTITION_AUDIT.tsv',sep='\t');part['cancer_id']=c;partitions.append(part)
    if any(code!=receipt_codes[0] for code in receipt_codes):raise ValueError('MIXED_TRAINING_CODE')
    met=pd.concat(metrics,ignore_index=True);status=pd.concat(statuses,ignore_index=True);met.to_parquet(out/'ALL_TARGET_METRICS.parquet',index=False);status.to_parquet(out/'ALL_TRAINING_STATUS.parquet',index=False);pd.concat(hyper).to_csv(out/'HYPERPARAMETERS.tsv',sep='\t',index=False);pd.DataFrame(coverage).to_csv(out/'NETWORK_FEATURE_COVERAGE.tsv',sep='\t',index=False);pd.concat(partitions).to_csv(out/'NETWORK_PARTITIONS.tsv',sep='\t',index=False);pd.DataFrame(audits).to_csv(out/'PHASE_AUDIT.tsv',sep='\t',index=False)
    good=met[met.success&np.isfinite(met.r2)].copy();good['query_support_excluded']=good.n_covariate_eligible_y-good.n;good['evaluated_fraction_of_observed_y']=good.n/good.n_observed_y
    good.groupby(['cancer_id','repeat','variant','arm','display_layer']).agg(targets=('target_id','size'),median_observed_y=('n_observed_y','median'),median_clinical_eligible_y=('n_covariate_eligible_y','median'),median_evaluated_patients=('n','median'),minimum_evaluated_patients=('n','min'),median_evaluated_fraction=('evaluated_fraction_of_observed_y','median'),maximum_query_support_exclusions=('query_support_excluded','max')).reset_index().to_csv(out/'PATIENT_COVERAGE_SUMMARY.tsv',sep='\t',index=False)
    keys=['cancer_id','repeat','target_id','display_layer','family','arm'];wide=good.pivot(index=keys,columns='variant',values='r2').reset_index();wide.to_parquet(out/'PAIRED_MODEL_R2.parquet',index=False)
    summary=good.groupby(['repeat','variant','arm','display_layer']).agg(targets=('r2','size'),median_r2=('r2','median'),mean_r2=('r2','mean'),positive=('r2',lambda a:int(a.gt(0).sum()))).reset_index();summary.to_csv(out/'MODEL_SUMMARY.tsv',sep='\t',index=False)
    comparisons=[];contrasts=[('module_ridge_union','ridge_adaptive_omics'),('module_ridge_union','ridge_equal'),('module_ridge_union','rna_laplacian_union'),('ridge_adaptive_omics','ridge_equal'),('rna_laplacian_union','ridge_equal'),('module_ridge_ppi','ridge_adaptive_omics'),('module_ridge_rpi','ridge_adaptive_omics')]+[('module_ridge_union','module_ridge_union_random_'+str(s)) for s in [101,202,303,404,505]]
    for method,baseline in contrasts:
        f=wide.dropna(subset=[method,baseline]).copy();f['delta_r2']=f[method]-f[baseline];f['method']=method;f['baseline']=baseline;comparisons.append(f[keys+['method','baseline','delta_r2']])
    paired=pd.concat(comparisons,ignore_index=True);paired.to_parquet(out/'PAIRED_MODEL_DELTAS.parquet',index=False)
    cs=paired.groupby(['repeat','arm','display_layer','method','baseline']).agg(targets=('delta_r2','size'),improved=('delta_r2',lambda a:int(a.gt(1e-9).sum())),median_delta_r2=('delta_r2','median'),mean_delta_r2=('delta_r2','mean')).reset_index();cs.to_csv(out/'NETWORK_GAIN_SUMMARY.tsv',sep='\t',index=False)
    aw=good[good.arm.isin(ARMS)].pivot(index=['cancer_id','repeat','variant','target_id','display_layer','family'],columns='arm',values='r2').dropna().reset_index();aw['combined_minus_mrna']=aw.combined-aw.mrna;aw['lncrna_minus_mrna']=aw.lncrna-aw.mrna;aw.to_parquet(out/'THREE_ARM_COMPARISONS.parquet',index=False)
    aas=aw.groupby(['repeat','variant','display_layer']).agg(targets=('target_id','size'),median_lncrna=('lncrna','median'),median_mrna=('mrna','median'),median_combined=('combined','median'),median_combined_minus_mrna=('combined_minus_mrna','median'),combined_improved=('combined_minus_mrna',lambda a:int(a.gt(1e-9).sum()))).reset_index();aas.to_csv(out/'THREE_ARM_SUMMARY.tsv',sep='\t',index=False)
    fail=status.groupby(['cancer_id','repeat','variant','arm','display_layer']).agg(attempted=('target_id','size'),successful=('success','sum')).reset_index();fail['failed']=fail.attempted-fail.successful;fail.to_csv(out/'COVERAGE_SUMMARY.tsv',sep='\t',index=False)
    dump(out/'SUMMARY_COMPLETE.json',dict(target_host='149',status='COMPLETE',phases_verified=len(audits),files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='SUMMARY_COMPLETE.json'}));print('SUMMARY_COMPLETE',len(good),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--inputs',required=True);p.add_argument('--networks',required=True);p.add_argument('--output',required=True);a=p.parse_args();summarize(a.source,a.inputs,a.networks,a.output)
