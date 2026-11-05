import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cc_hhgt.m0_three_arm import feature_axes,select_targets,restrict_split,matched_predictions,compare_target,ARMS


def test_non_rna_provenance_whitelist_excludes_rna_footprints():
    meta=pd.DataFrame({'cohort':['CPTAC']*5,'layer':['RNA_PATHWAY','STATE','PROTEIN_PATHWAY','PHOSPHO_PTMSEA','METHYLATION_PATHWAY'],
        'target_id':['RNA_PATHWAY|P','STATE|PROGENY','PROTEIN_PATHWAY|P','PHOSPHO_PTMSEA|Q','METHYLATION_PATHWAY|R']})
    out=select_targets(meta)
    assert out.target_id.tolist()==['PROTEIN_PATHWAY|P','PHOSPHO_PTMSEA|Q','METHYLATION_PATHWAY|R']
    assert out.analysis_role.tolist()==['PRIMARY','PRIMARY','SECONDARY']


def test_only_disjoint_annotated_measured_gene_classes_are_compared():
    ann=pd.DataFrame({'gene_id':['ENSG3','ENSG2','ENSG1'],'gene_biotype':['rRNA','protein_coding','lncRNA']})
    feat,axes=feature_axes(ann)
    assert feat.gene_id.tolist()==['ENSG1','ENSG2']
    assert set(axes['lncrna']).isdisjoint(axes['mrna'])
    assert np.array_equal(np.r_[axes['lncrna'],axes['mrna']],axes['combined'])
    with pytest.raises(ValueError,match='Duplicate'):feature_axes(pd.concat([ann,ann.iloc[[0]]]))


def test_existing_patient_splits_are_restricted_without_reassignment():
    split={'outer0':{'train':[0,1,3],'test':[2,4]},'inner0':{'train':[0,1],'validation':[3],'test':[2,4]}}
    got=restrict_split(split,['T0','C0','C1','C2','T1'],['C0','C1','C2'])
    assert got['outer0']=={'train':[0,2],'test':[1]}
    assert got['inner0']=={'train':[0],'validation':[2],'test':[1]}
    with pytest.raises(ValueError,match='leakage'):restrict_split({'x':{'train':[0],'test':[0]}},['C0'],['C0'])


def prediction_fixture():
    f=pd.DataFrame({'patient_id':['P'+str(i) for i in range(30)],'target_id':'PROTEIN_PATHWAY|P',
      'phase':['outer'+str(i%5) for i in range(30)],'cohort':'CPTAC','observed':np.linspace(-2,2,30),'predicted':np.linspace(-2,2,30),
      'train_mean_baseline':0.,'covariate_only_baseline':0.})
    return {arm:f.copy() for arm in ARMS}


def test_comparison_rejects_changed_patients_labels_or_baselines():
    frames=prediction_fixture();frames['mrna']=frames['mrna'].iloc[:-1]
    with pytest.raises(ValueError,match='identical'):matched_predictions(frames)
    for col in ['observed','covariate_only_baseline']:
        frames=prediction_fixture();frames['mrna'].loc[0,col]+=1
        with pytest.raises(ValueError,match='mismatch'):matched_predictions(frames)


def test_paired_metric_direction_and_ties():
    frames=prediction_fixture();p=matched_predictions(frames);r=compare_target(p,bootstrap=30)
    assert r['best_arm']=='TIE' and r['delta_r2_combined_vs_mrna']==0
    frames['mrna']['predicted']=0.;frames['lncrna']['predicted']=.5*frames['lncrna'].observed
    r=compare_target(matched_predictions(frames),bootstrap=30)
    assert r['best_arm']=='combined' and r['delta_r2_combined_vs_mrna']==1
    assert r['folds_combined_lower_mse_than_mrna']==5
    assert r['delta_r2_conditional_bootstrap_low']>0


def test_three_arm_end_to_end_is_paired_and_detects_complementary_signal(tmp_path,monkeypatch):
    # Scientific integration test: a hidden target depends on both RNA classes.
    # The fixture also contains an all-missing assay, which must not be imputed.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_m0_three_arm import run
    import run_m0_three_arm
    def forbidden(*a,**kw):raise AssertionError("Correlation screen must never run")
    monkeypatch.setattr(run_m0_three_arm,"cohort_screen",forbidden)
    from cla_engine.common import sha
    rng=np.random.default_rng(2001);n=120;p=20;idx=pd.Index(['P'+str(i) for i in range(n)])
    a=tmp_path/'aligned'/'BRCA';c=tmp_path/'cov'/'BRCA';s=tmp_path/'splits'/'BRCA'
    for d in [a,c,s]:d.mkdir(parents=True)
    genes=['ENSG'+str(i).zfill(5) for i in range(p)]
    raw=pd.DataFrame(rng.normal(4,.6,(n,p)),index=idx,columns=genes)
    ann=pd.DataFrame({'gene_id':genes,'gene_biotype':['lncRNA']*10+['protein_coding']*10,'reference_symbol':['G'+str(i) for i in range(p)]})
    ann.to_csv(a/'GENE_ANNOTATION.tsv',sep='\t',index=False);raw.to_parquet(a/'GENE_LOGTPM.parquet')
    lnc=raw.iloc[:,:10].copy();lnc.columns='LNC:'+lnc.columns;lnc.to_parquet(a/'X.parquet')
    pd.DataFrame({'patient_id':idx,'cohort':'CPTAC'}).to_csv(a/'PATIENTS.tsv',sep='\t',index=False)
    cov=pd.DataFrame({'cohort':'CPTAC','age_years':rng.uniform(30,80,n),'sex':np.tile(['female','male'],n//2),'stage':np.tile(['I','II','III'],n//3),'purity':rng.uniform(.3,.9,n)},index=idx)
    cov.to_parquet(c/'COVARIATES.parquet')
    targets=['PROTEIN_PATHWAY|P','PHOSPHO_PTMSEA|Q','METHYLATION_PATHWAY|ABSENT']
    y=2*raw.iloc[:,0]+2*raw.iloc[:,10]+.02*cov.age_years+rng.normal(0,.03,n)
    pd.DataFrame({targets[0]:y,targets[1]:-y,targets[2]:np.nan}).to_parquet(a/'CPTAC_NATIVE_Y.parquet')
    pd.DataFrame({'target_id':targets,'layer':[x.split('|')[0] for x in targets],'cohort':'CPTAC','method':'TEST_MEASURED','family':['P','Q','R']}).to_csv(a/'NATIVE_TARGETS.tsv',sep='\t',index=False)
    for d in [a,c]:(d/'COMPLETE.json').write_text(json.dumps({'files':{f.name:sha(f) for f in d.iterdir() if f.is_file()}}))
    folds=np.arange(n)%5;sp={}
    for k in range(5):
        sp['outer'+str(k)]={'train':np.flatnonzero(folds!=k).tolist(),'test':np.flatnonzero(folds==k).tolist()}
        sp['inner'+str(k)]={'train':np.flatnonzero((folds!=k)&(folds!=(k+1)%5)).tolist(),'validation':np.flatnonzero(folds==(k+1)%5).tolist(),'test':np.flatnonzero(folds==k).tolist()}
    sp['final']={'train':list(range(n)),'test':[]};(s/'SPLITS.json').write_text(json.dumps(sp))
    output=tmp_path/'out';summary=run(a.parent,s.parent,c.parent,'BRCA',output,1)
    res=pd.read_csv(output/'BRCA/TARGET_COMPARISON.tsv',sep='\t')
    assert summary['n_paired_targets']==2
    assert not res.loc[res.layer.eq('METHYLATION_PATHWAY'),'complete_fivefold'].any()
    res=res.loc[res.complete_fivefold]
    assert len(res)==2
    assert res.n_patients.eq(n).all()
    assert (res.r2_combined>res.r2_mrna+.1).all() and (res.r2_combined>res.r2_lncrna+.1).all()
    meta=pd.read_csv(output/'BRCA/TARGETS.tsv',sep='\t')
    assert meta.loc[meta.layer.eq('METHYLATION_PATHWAY'),'availability'].item()=='NO_MEASURED_Y'
    # Resume must verify existing checksums, not silently re-fit or replace files.
    assert run(a.parent,s.parent,c.parent,'BRCA',output,1)==summary
