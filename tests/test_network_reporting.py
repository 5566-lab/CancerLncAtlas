"""Exercise report statistics and exports with explicitly synthetic fixtures."""
import json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from cla_engine.common import sha
from cc_hhgt.graph_ridge import target_policy
from cc_hhgt.network_ridge import MODES,ARMS
from scripts.summarize_network_validation import summarize
from scripts.bootstrap_network_validation import bootstrap
from scripts.plot_network_validation import plot
from scripts.report_network_omics_increment import report as report_increment, paired_increment

def seal(folder):
    (folder/'COMPLETE.json').write_text(json.dumps(dict(files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='COMPLETE.json'})))

def test_reporting_statistics_and_figures(tmp_path):
    source=tmp_path/'SIMULATED_FITS';inputs=tmp_path/'SIMULATED_INPUTS';net=tmp_path/'SIMULATED_NETWORKS';out=tmp_path/'SIMULATED_REPORT'
    methods=['rna_only','omics_only','ridge_equal','ridge_adaptive_omics']
    for mode in MODES:
        suffix=mode if not mode.startswith('random_') else 'union_'+mode;methods+=['rna_laplacian_'+suffix,'module_ridge_'+suffix]
    layers=['PROTEIN_PATHWAY','PHOSPHO_PTMSEA','METHYLATION_PATHWAY','RNA_PATHWAY','STATE','STATE','STATE'];families=['hallmark','ptm','hallmark','hallmark','transcription_factor','EMT','mutation_burden'];tids=[layer+'|synthetic_'+str(i) for i,layer in enumerate(layers)];meta=pd.DataFrame(dict(target_id=tids,layer=layers,family=families,method='synthetic',cohort='SIMULATED'));n=30;pids=['synthetic_patient_'+str(i) for i in range(n)];y=np.sin(np.arange(n)[:,None]/4+np.arange(len(tids))[None,:]);phase_names=[f'repeat{r}_outer{f}' for r in range(2) for f in range(5)]+['final']
    for ci,c in enumerate(['BRCA','LUAD','COAD']):
        (inputs/c).mkdir(parents=True);(net/c).mkdir(parents=True);pd.DataFrame(y,index=pids,columns=tids).to_parquet(inputs/c/'Y.parquet');pd.DataFrame({'cohort':'SIMULATED','age_years':np.arange(n)+35.,'sex':['F','M']*(n//2),'stage':'II','purity':.7},index=pids).to_parquet(inputs/c/'COVARIATES.parquet');fm=[];prep={'preprocess':{}}
        for block in ['RNA','CNV','MUTATION','PROTEIN','PHOSPHO']:
            prep['preprocess'][block]=dict(index=np.arange(8),original_width=8)
            for j in range(8):fm.append(dict(block=block,position=j,rna_arm=('lncrna' if j<4 else 'mrna') if block=='RNA' else None,**{'module_'+mode:j%3 for mode in MODES}))
        pd.DataFrame(fm).to_parquet(net/c/'FEATURE_MAP.parquet');pd.DataFrame({'mode':MODES,'communities':[3]*len(MODES),'connected_nodes':[8]*len(MODES),'modularity':[.5]*len(MODES),'min_nodes':[2]*len(MODES),'max_nodes':[4]*len(MODES)}).to_csv(net/c/'PARTITION_AUDIT.tsv',sep='\t',index=False)
        for phase in phase_names:
            d=source/c/phase;d.mkdir(parents=True);final=phase=='final';repeat=-1 if final else int(phase[6]);fold=-1 if final else int(phase[-1]);indices=[] if final else [i for i in range(n) if (i+repeat)%5==fold];audit=[];pred=[]
            for ti,record in enumerate(meta.to_dict('records')):
                pol=target_policy(record)
                for mi,method in enumerate(methods):
                    for ai,arm in enumerate(['shared'] if method=='omics_only' else ARMS):
                        audit.append(dict(target_id=tids[ti],variant=method,arm=arm,status='FITTED',phase=phase,parameter=.5 if 'module' in method else 0.,lam=.01,inner_validations=6,n_inner_train_min=16,n_inner_validation_min=6,allowed_blocks=';'.join(pol['allowed_blocks'])))
                        for i in indices:
                            error=(.25+.01*mi+.02*ai)*np.cos(i+ci+repeat*.1);pred.append(dict(patient_id=pids[i],target_id=tids[ti],variant=method,arm=arm,phase=phase,repeat=repeat,fold=fold,observed=y[i,ti],predicted=y[i,ti]+error))
            pd.DataFrame(audit).to_parquet(d/'AUDIT.parquet');pd.DataFrame(pred).to_parquet(d/'PREDICTIONS.parquet');meta.to_csv(d/'TARGETS.tsv',sep='\t',index=False);(d/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','code':{'explicitly_synthetic_fixture':'true'}}))
            if final:joblib.dump(prep,d/'PREPROCESS_purity1.joblib')
            seal(d)
    summarize(source,inputs,net,out);report_increment(out,source);bootstrap(source,inputs,out,replicates=50);plot(out,source)
    metrics=pd.read_parquet(out/'ALL_TARGET_METRICS.parquet');row=metrics[(metrics.cancer_id=='BRCA')&(metrics.repeat==0)&(metrics.target_id==tids[0])&(metrics.variant=='rna_only')&(metrics.arm=='lncrna')].iloc[0];expected=1-np.sum((.25*np.cos(np.arange(n)))**2)/np.sum((y[:,0]-y[:,0].mean())**2);np.testing.assert_allclose(row.r2,expected,atol=1e-12);assert row.n==30;assert metrics.success.all();assert len(list(out.glob('*.png')))==11;assert len(list(out.glob('*.pdf')))==11
    assert (out/'OPEN_REPORT.html').exists();b=pd.read_csv(out/'PATIENT_BOOTSTRAP.tsv',sep='\t');assert np.isfinite(b[['family_balanced_mean_delta_r2','ci_low','ci_high']]).all().all();assert (b.ci_low<=b.ci_high).all()

    inc=pd.read_parquet(out/'RNA_INCREMENT_OVER_OMICS.parquet');r=inc[(inc.cancer_id=='BRCA')&(inc.repeat==0)&(inc.target_id==tids[0])&(inc.arm=='lncrna')].iloc[0]
    expected_increment=(((.26*np.cos(np.arange(n)))**2).sum()-((.27*np.cos(np.arange(n)))**2).sum())/np.sum((y[:,0]-y[:,0].mean())**2)
    np.testing.assert_allclose(r.delta_r2,expected_increment,atol=1e-12);assert r.n==r.baseline_n==30;assert len(pd.read_csv(out/'RNA_INCREMENT_PATIENT_PAIRING.tsv',sep='\t'))==30

def test_increment_refuses_unequal_patient_support():
    import pytest
    base=dict(cancer_id='TEST',repeat=0,target_id='x',display_layer='PROTEIN_PATHWAY',family='f',success=True,r2=.1)
    frame=pd.DataFrame([dict(base,variant='omics_only',arm='shared',n=20),dict(base,variant='ridge_equal',arm='lncrna',n=19)])
    with pytest.raises(ValueError,match='DIFFERENT_PATIENT_SUPPORT'):paired_increment(frame)
