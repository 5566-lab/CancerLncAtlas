"""Join fixed clinical/DNA-purity values to joint M0 patients on host 149."""
import argparse,json,socket,sys,re
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cla_engine.common import sha
from run_m0_linear import write_json
from cc_hhgt.m0_covariates import complete_rows

def stage(value):
    if pd.isna(value):return np.nan
    x=re.sub(r'^STAGE\s*','',str(value).upper().strip())
    m=re.fullmatch(r'(IV|III|II|I|0)[ABC]?',x)
    return m.group(1) if m else np.nan

def prepare(aligned,tcga,cptac,phenotypes,output,cancers):
    if socket.gethostname()!='149':raise RuntimeError('Host 149 required')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    paths=[Path(tcga),Path(cptac),Path(phenotypes)]
    write_json(out/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='FIXED_CLINICAL_AND_DNA_PURITY_JOIN',sources=[dict(path=str(p),sha256=sha(p)) for p in paths]))
    t=pd.read_parquet(tcga);c=pd.read_parquet(cptac).set_index('patient_id');p=pd.read_parquet(phenotypes).set_index('patient_id');summary=[]
    for ca in cancers:
        d=out/ca;d.mkdir(exist_ok=True)
        if (d/'COMPLETE.json').exists():raise ValueError('Immutable completed output '+str(d))
        src=Path(aligned)/ca;patients=pd.read_csv(src/'PATIENTS.tsv',sep='\t').rename(columns={'index':'patient_id'}).set_index('patient_id');z=patients[['cohort','cancer_id']].copy()
        for cc in ['TCGA','CPTAC']:
            ids=patients.index[patients.cohort.eq(cc)];source=t.loc[t.cancer_id.eq(ca)].set_index('patient_id') if cc=='TCGA' else c
            if not source.index.is_unique or not ids.isin(source.index).all():raise ValueError('Clinical patient authority mismatch')
            for col in ['age_years','sex','stage']:z.loc[ids,col]=source.loc[ids,col].to_numpy()
            q=source['absolute_purity'] if cc=='TCGA' else p['WES_purity']
            z.loc[ids,'purity']=pd.to_numeric(q.reindex(ids),errors='raise').to_numpy()
            z.loc[ids,'purity_source']='TCGA_ABSOLUTE' if cc=='TCGA' else 'CPTAC_PUBLISHED_WES'
            z.loc[ids,'purity_original_field']='absolute_purity' if cc=='TCGA' else 'WES_purity'
        z['age_years']=pd.to_numeric(z.age_years,errors='raise');z['purity']=pd.to_numeric(z.purity,errors='raise')
        if ((z.purity.dropna()<0)|(z.purity.dropna()>1)).any():raise ValueError('DNA purity outside [0,1]')
        z['sex']=z.sex.astype('string').str.lower().str.strip().map({'male':'male','female':'female'})
        z['stage']=z.stage.map(stage)
        z['clinical_complete']=complete_rows(z,False);z['clinical_and_purity_complete']=complete_rows(z,True)
        z.index.name='patient_id';z.to_parquet(d/'COVARIATES.parquet',compression='zstd');z.to_csv(d/'COVARIATES.tsv',sep='\t')
        for cc,g in z.groupby('cohort'):
            row=dict(cancer_id=ca,cohort=cc,n=len(g),n_clinical_complete=int(g.clinical_complete.sum()),n_clinical_purity_complete=int(g.clinical_and_purity_complete.sum()),n_purity=int(g.purity.notna().sum()));summary.append(row);print(row,flush=True)
        write_json(d/'COMPLETE.json',dict(status='COMPLETE',target_host='149',input_complete_sha256=sha(src/'COMPLETE.json'),code_sha256=sha(__file__),missing_policy='COMPLETE_CASE_NO_CLINICAL_OR_PURITY_IMPUTATION',purity='FIXED_SOURCE_VALUES_NO_RECOMPUTATION_NO_AVERAGING_NO_FALLBACK_TO_RNA',files={q.name:sha(q) for q in d.iterdir() if q.is_file()}))
    pd.DataFrame(summary).to_csv(out/'COVERAGE.tsv',sep='\t',index=False)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',target_host='149',cancers=cancers,coverage=summary))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['aligned','tcga','cptac','phenotypes','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--cancers',nargs='+',default=['BRCA','LUAD','COAD']);a=p.parse_args();prepare(a.aligned,a.tcga,a.cptac,a.phenotypes,a.output,a.cancers)
