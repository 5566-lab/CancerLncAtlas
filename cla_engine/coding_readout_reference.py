"""Freeze RNA readout references using coding-only inputs and fit patients."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from .relation_evidence_bindings import file_sha

METHOD_VARIANT = 'PC_GENE_ID_BACKGROUND_R1'


def coding_progeny(expression, model, fit_patients, minimum_genes=5):
    """Use retained coding members of a fixed model, then freeze its reference."""
    if not expression.index.is_unique or not expression.columns.is_unique or not model.index.is_unique:
        raise ValueError('CODING_PROGENY_DUPLICATE_AXIS')
    fit=list(fit_patients)
    if len(fit)<2 or len(fit)!=len(set(fit)) or not set(fit)<=set(expression.index):
        raise ValueError('CODING_PROGENY_FIT_PATIENTS')
    genes=[g for g in model.index if g in expression]
    weights=model.loc[genes].to_numpy(float)
    if not np.isfinite(weights).all():raise ValueError('CODING_PROGENY_MODEL_VALUES')
    x=expression[genes].to_numpy(float)
    raw=np.asarray(csr_matrix(weights).T@x.T).T
    count=(weights!=0).sum(0);raw[:,count<minimum_genes]=np.nan
    raw=pd.DataFrame(raw,index=expression.index,columns=model.columns)
    mu=raw.loc[fit].mean();sd=raw.loc[fit].std(ddof=1)
    sd=sd.mask(~np.isfinite(sd)|(sd<=1e-12));scaled=(raw-mu)/sd
    return scaled, {'method_variant':METHOD_VARIANT,'method':'FIXED_PROGENY_TOP100_WEIGHTS_CODING_MEMBERS_SOURCE_Z',
        'fit_patients':fit,'genes':genes,'original_nonzero_members':(model!=0).sum().to_dict(),
        'coding_nonzero_members':dict(zip(model.columns,count.tolist())),
        'mean':mu.to_dict(),'sd':sd.to_dict()}


def build_coding_readouts(aligned, cache, apm_root, model_path, fit_patients, *, context):
    """Return fitted readouts without native RNA endpoints of unknown lineage."""
    from cc_hhgt.m0_joint_scores import JointScoreBuilder
    aligned,cache,apm_root,model_path=map(Path,(aligned,cache,apm_root,model_path))
    src=json.loads((aligned/'COMPLETE.json').read_text())
    if src.get('target_host')!='149' or src.get('scoring_background')!='PROTEIN_CODING_GENE_IDS_BEFORE_SYMBOL_AGGREGATION':
        raise ValueError('CODING_READOUT_INPUT_CONTRACT')
    cached=json.loads((cache/'COMPLETE.json').read_text())
    if cached['input_complete_sha256']!=file_sha(aligned/'COMPLETE.json'):
        raise ValueError('CODING_READOUT_CACHE_INPUT_DRIFT')
    for folder,rec in [(aligned,src),(cache,cached)]:
        for name,h in rec['files'].items():
            if file_sha(folder/name)!=h:raise ValueError('CODING_READOUT_INPUT_DRIFT')
    apm=json.loads((apm_root/'APM_RUN_COMPLETE.json').read_text())
    if apm['aligned_receipt_sha256']!=file_sha(aligned/'COMPLETE.json'):
        raise ValueError('CODING_READOUT_APM_INPUT_DRIFT')
    for name,h in apm['files'].items():
        if file_sha(apm_root/name)!=h:raise ValueError('CODING_READOUT_APM_CONTENT_DRIFT')
    e=pd.read_parquet(aligned/'SYMBOL_LOGTPM.parquet')
    ann=pd.read_csv(aligned/'SCORING_GENE_ANNOTATION.tsv',sep='\t')
    if not ann.gene_biotype.eq('protein_coding').all() or set(e)!=set(ann.reference_symbol.dropna()):
        raise ValueError('CODING_READOUT_BACKGROUND_MISMATCH')
    fit=list(fit_patients);positions=e.index.get_indexer(fit)
    if len(fit)!=len(set(fit)) or len(fit)<2 or (positions<0).any():
        raise ValueError('CODING_READOUT_FIT_PATIENTS')
    b=JointScoreBuilder.__new__(JointScoreBuilder)
    b.aligned=aligned;b.cache=cache;b.apm=apm_root/'apm';b.e=e
    b.frames={k:pd.read_parquet(cache/(k+'.parquet')).reindex(e.index) for k in
              ['PATHWAY_RAW','STATE_SSGSEA_RAW','HALLMARK_RAW','STATE_FIXED_RAW','XCELL_RAW']}
    b.defs=json.loads((cache/'DEFINITIONS.json').read_text())
    b.xref=json.loads((cache/'XCELL_REFERENCE.json').read_text())
    b.network=pd.read_csv(cache/'PROGRAM_NETWORK.tsv',sep='\t')
    b.catalog=json.loads((cache/'CONFIG_TARGETS.json').read_text())
    b.native={key:pd.DataFrame(index=e.index) for key in ['TCGA','CPTAC']}
    b.target_metadata=None
    scores,reference=b.fit_apply(positions,context)
    model=pd.read_csv(model_path,sep='\t',index_col=0)
    progeny,pgref=coding_progeny(e,model,fit)
    progeny.columns=['STATE|native_program::PROGENY:'+s for s in progeny]
    if not set(progeny)<=set(scores):raise ValueError('CODING_READOUT_PROGENY_AXIS')
    scores.loc[:,progeny.columns]=progeny
    method=b.target_metadata.copy()
    method.loc[method.target_id.isin(progeny.columns),'method']=pgref['method']
    method['method']=method['method'].str.replace('COMMON_ALLGENE_BACKGROUND','COMMON_PC_GENE_ID_BACKGROUND',regex=False)
    method['method_variant']=METHOD_VARIANT
    method['background']='GENCODE_PROTEIN_CODING_GENE_IDS_THEN_SYMBOL_AGGREGATION'
    reference['method_variant']=METHOD_VARIANT
    reference['PROGENY_CODING']=pgref
    reference['progeny_fixed_model_sha256']=file_sha(model_path)
    reference['coding_input_receipt_sha256']=file_sha(aligned/'COMPLETE.json')
    reference['APM_RUN_COMPLETE_sha256']=file_sha(apm_root/'APM_RUN_COMPLETE.json')
    if np.isinf(scores.to_numpy(float)).any():raise ValueError('CODING_READOUT_INFINITE_SCORE')
    return scores,method,reference
