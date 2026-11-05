"""Restore shared GENCODE-v36 measured RNA axes for joint TCGA/CPTAC M0."""
from __future__ import annotations
import argparse,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cla_engine.common import sha
from run_m0_linear import write_json,source_read

def prepare(config,tcga_export,previous_inputs,output,cancers):
    if socket.gethostname()!='149':raise RuntimeError('Run on 149')
    if not 1<=len(cancers)<=3 or len(set(cancers))!=len(cancers):raise ValueError('Three-cancer demo scope')
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new output')
    out.mkdir(parents=True)
    write_json(out/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='JOINT_M0_RAW_RNA_ALIGNMENT',cancers=cancers))
    cfg=json.loads(Path(config).read_text());er=Path(tcga_export);pr=Path(previous_inputs)
    sources=[];ec=source_read(er/'COMPLETE.json',lambda p:json.loads(p.read_text()),sources)
    bp=Path(cfg['cptac_input']['complete']['path']);bc=source_read(bp,lambda p:json.loads(p.read_text()),sources,cfg['cptac_input']['complete']['sha256'])
    ap=Path(bc['config']['aligned_complete']['path']);ac=source_read(ap,lambda p:json.loads(p.read_text()),sources,bc['config']['aligned_complete']['sha256'])
    ann=source_read(ap.parent/'FEATURE_ALIGNMENT.parquet',pd.read_parquet,sources,ac['files']['FEATURE_ALIGNMENT.parquet']['sha256'])
    ann=ann.loc[ann.layer.eq('rna') & ann.reference_presence.eq('GENCODE_V36'),['gene_id','reference_symbol','gene_biotype']].drop_duplicates()
    if ann.gene_id.duplicated().any():raise ValueError('Conflicting reference annotations')
    ann=ann.set_index('gene_id');summary=[]
    for cancer in cancers:
        d=out/cancer;d.mkdir();ss=list(sources)
        rec=ec['cancers'][cancer]
        tc=source_read(er/cancer/'GENE_TPM.tsv.gz',lambda p:pd.read_csv(p,sep='\t').set_index('gene_id'),ss,rec['outputs']['tpm_sha256'])
        cohort={'KIRC':'CCRCC','LUSC':'LSCC','HNSC':'HNSCC','PAAD':'PDAC'}.get(cancer,cancer)
        rel=f'matrices/{cohort}/rna_TumorNormal.parquet'
        cp=source_read(ap.parent/rel,lambda p:pd.read_parquet(p).set_index('feature_id'),ss,ac['files'][rel]['sha256'])
        cp=cp[[c for c in cp if c.endswith('::T')]].copy();cp.columns=cp.columns.str.removesuffix('::T')
        if tc.columns.intersection(cp.columns).size:raise ValueError('Overlapping patient identifiers need adjudication')
        cp=np.expm1(cp*np.log(2.))
        genes=sorted(set(tc.index)&set(cp.index)&set(ann.index))
        if len(genes)<50000:raise ValueError('Truncated full measured RNA axis')
        full=pd.concat([tc.loc[genes],cp.loc[genes]],axis=1)
        if np.isinf(full.to_numpy()).any() or (full.to_numpy()<0).any():raise ValueError('Invalid RNA expression')
        full.index.name='gene_id'
        log=np.log2(full+1)
        log.T.to_parquet(d/'GENE_LOGTPM.parquet',compression='zstd')
        a=ann.loc[genes].copy();a['lnc_predictor']=a.gene_biotype.eq('lncRNA')
        a.reset_index().to_csv(d/'GENE_ANNOTATION.tsv',sep='\t',index=False)
        lnc=a.index[a.lnc_predictor];x=log.loc[lnc].T;x.columns='LNC:'+x.columns
        x.to_parquet(d/'X.parquet',compression='zstd')
        pd.DataFrame({'lncrna_id':x.columns,'gene_symbol':a.loc[lnc,'reference_symbol'].to_numpy()}).to_csv(d/'FEATURES.tsv',sep='\t',index=False)
        # Shared symbol aggregation is the SUM of actual TPM, not mean log-expression.
        # Missing measured values propagate; no missing gene or deleted expression is zero-filled.
        usable=a.reference_symbol.notna() & a.reference_symbol.ne('')
        z=full.loc[a.index[usable]].copy();z.index=a.loc[usable,'reference_symbol']
        summed=z.groupby(level=0,sort=True).sum(min_count=1)
        missing=z.isna().groupby(level=0,sort=True).any();summed=summed.mask(missing)
        np.log2(summed+1).T.to_parquet(d/'SYMBOL_LOGTPM.parquet',compression='zstd')
        pc=sorted(set(a.loc[a.gene_biotype.eq('protein_coding') & usable,'reference_symbol']))
        pd.DataFrame({'symbol':pc}).to_csv(d/'PROTEIN_CODING_SYMBOLS.tsv',sep='\t',index=False)
        patient_parts=[];native=[];metadata=[]
        for dataset,cc in [('TCGA',cancer),('CPTAC',cohort)]:
            src=pr/(dataset+'_'+cc);receipt=json.loads((src/'COMPLETE.json').read_text())
            for name in ['PATIENT_FOLDS.tsv','Y.parquet','TARGETS.tsv']:
                if sha(src/name)!=receipt['files'][name]:raise ValueError('Previous input hash changed: '+str(src/name))
            patients=pd.read_csv(src/'PATIENT_FOLDS.tsv',sep='\t');patients['cohort']=dataset;patients['cancer_id']=cancer
            patient_parts.append(patients)
            y=pd.read_parquet(src/'Y.parquet')
            meta=pd.read_csv(src/'TARGETS.tsv',sep='\t')
            # Native outcomes are staged for explicit disposition, never assumed harmonized.
            y.to_parquet(d/(dataset+'_NATIVE_Y.parquet'),compression='zstd')
            meta['cohort']=dataset;metadata.append(meta)
        patients=pd.concat(patient_parts).set_index('patient_id').reindex(x.index)
        if patients.isna().any().any():raise ValueError('Patient/fold authority incomplete')
        patients.rename_axis('patient_id').reset_index().to_csv(d/'PATIENTS.tsv',sep='\t',index=False)
        pd.concat(metadata).to_csv(d/'NATIVE_TARGETS.tsv',sep='\t',index=False)
        info=dict(cancer_id=cancer,n_patients=len(x),cohorts=patients.cohort.value_counts().to_dict(),n_lncrnas=x.shape[1],n_shared_genes=len(genes),n_symbols=len(summed),n_coding_symbols=len(pc),
          expression_unit='log2(TPM+1)',annotation='GENCODE_v36',structural_feature_intersection=True,expression_filter_applied=False,
          missing_rna_cells=int(log.isna().sum().sum()),historical_filtered_values_restored_from_raw=True,
          symbol_aggregation='SUM_TPM_THEN_LOG2_PLUS1_MISSING_PROPAGATES',overlap_audit='No_exact_patient_ID_overlap;biological_alias_crosswalk_unavailable',
          labels_status='NATIVE_STAGED_NOT_JOINT_TRAINING_LABELS',sources=ss)
        write_json(d/'ALIGNMENT.json',info)
        write_json(d/'COMPLETE.json',dict(status='COMPLETE',target_host='149',files={p.name:sha(p) for p in d.iterdir() if p.is_file()},info={k:v for k,v in info.items() if k!='sources'}))
        summary.append({k:v for k,v in info.items() if k not in ['sources','cohorts']})
        print('ALIGNED',cancer,len(x),len(lnc),len(genes),'missing',info['missing_rna_cells'],flush=True)
    pd.DataFrame(summary).to_csv(out/'SUMMARY.tsv',sep='\t',index=False)
    write_json(out/'COMPLETE.json',dict(status='COMPLETE',target_host='149',cancers=cancers,config_sha256=sha(config),code_sha256=sha(__file__)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','tcga-export','previous-inputs','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--cancer-list',nargs='+',default=['BRCA','LUAD','COAD']);a=p.parse_args()
    prepare(a.config,a.tcga_export,a.previous_inputs,a.output,a.cancer_list)
