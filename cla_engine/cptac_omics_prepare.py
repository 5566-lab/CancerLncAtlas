"""Prepare measured protein, promoter-related methylation and site-level PTM score inputs."""
from __future__ import annotations
import argparse, json, socket
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json, sha, fail, read_json
from .state_protein_map import pinned

def gmt(path):
    result={}
    for line in Path(path).read_text().splitlines():
        f=line.split('\t')
        if len(f)>2:result[f[0]]=list(dict.fromkeys(f[2:]))
    return result

def methylation_means(frame, sets, minimum=10, maximum=500):
    """Observed-only mean beta; positive does not mean pathway activation."""
    scores={};counts={}
    for sid,genes in sets.items():
        hit=frame.index.intersection(genes)
        n=frame.loc[hit].notna().sum()
        val=frame.loc[hit].mean()
        val[n.lt(minimum) | (len(genes)>maximum)]=np.nan
        scores[sid]=val;counts[sid]=n
    return pd.DataFrame(scores).T,pd.DataFrame(counts).T

def site_input(frame, features):
    """Merge only repeated measurements of the same sequence and gene; never all gene sites."""
    m=features.set_index('source_feature_id').reindex(frame.index)
    ok=m.ptm_sequence_id.fillna('').ne('')
    m=m.loc[ok].copy();values=frame.loc[ok].copy()
    ambiguous=m.groupby('ptm_sequence_id').gene_id.nunique()
    excluded=set(ambiguous.index[ambiguous.gt(1)])
    keep=~m.ptm_sequence_id.isin(excluded)
    m=m.loc[keep];values=values.loc[keep]
    bridge=m[['gene_id','ptm_sequence_id','protein_isoform','site','flanking_sequence']].reset_index()
    values.index=m.ptm_sequence_id
    # This is a duplicate sequence/site measurement collapse, not gene-level PTM aggregation.
    collapsed=values.groupby(level=0,sort=True).mean()
    reference=collapsed.median(axis=1,skipna=True)
    centered=collapsed.sub(reference,axis=0)
    return centered,reference,bridge,sorted(excluded)

def run(cfg, output):
    if socket.gethostname().split('.')[0]!='149':fail('CPTAC_EXECUTION_HOST','Run on 149')
    complete_path=pinned(cfg['aligned_complete']);aligned=complete_path.parent
    ac=read_json(complete_path)
    if ac.get('status')!='COMPLETE':fail('CPTAC_INCOMPLETE','Aligned inputs are incomplete')
    for rel,rec in ac['files'].items():
        if sha(aligned/rel)!=rec['sha256']:fail('CPTAC_ALIGNED_HASH',rel)
    out=Path(output)
    if out.exists() and any(out.iterdir()):fail('CPTAC_OUTPUT_EXISTS','Use a new versioned directory')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'CPTAC_OMICS_SCORE_INPUT_PREPARATION'})
    sets=gmt(pinned(cfg['pathway_gmt']))
    features=pd.read_parquet(aligned/'FEATURE_ALIGNMENT.parquet')
    meth_definition=read_json(pinned(cfg['methylation_definition']))
    if meth_definition['status']!='VERIFIED_ORIGINAL_METHODS':fail('CPTAC_METHYLATION_DEFINITION','Unverified source definition')
    audit=[]
    for cohort in ac['cohorts']:
        dest=out/cohort;dest.mkdir()
        for layer in ('protein','phosphorylation','methylation_gene'):
            p=aligned/'matrices'/cohort/(layer+'_Tumor.parquet')
            if not p.exists():continue
            frame=pd.read_parquet(p).set_index('feature_id')
            if layer=='phosphorylation':
                f=features.loc[features.layer.eq(layer)]
                result,reference,bridge,ambiguous=site_input(frame,f)
                # Original Broad GCT 1.2 reader accepts NaN as missing.
                with (dest/'PHOSPHO_SITE_CENTERED.gct').open('w') as o:
                    o.write('#1.2\n'+str(len(result))+'\t'+str(result.shape[1])+'\n')
                    temp=result.copy();temp.insert(0,'Description','single_site_sequence');temp.index.name='Name'
                    temp.to_csv(o,sep='\t',na_rep='NaN')
                reference.rename('tumor_cohort_median_log2').to_csv(dest/'PHOSPHO_REFERENCE_MEDIANS.tsv',sep='\t',index_label='ptm_sequence_id')
                bridge.to_csv(dest/'PHOSPHO_SEQUENCE_SITE_BRIDGE.tsv.gz',sep='\t',index=False,compression='gzip')
                audit.append({'source_cohort':cohort,'layer':layer,'source_features':len(frame),'scoring_features':len(result),
                              'ambiguous_multi_gene_sequences_excluded':len(ambiguous),
                              'normalization':'SITE_MEDIAN_CENTERED_WITHIN_TUMOR_COHORT;FROZEN_MEDIANS_EXPORTED',
                              'not_inferred':'NO_CAUSAL_OR_ABSOLUTE_ACTIVATION;NO_PROTEIN_ABUNDANCE_ADJUSTMENT'})
                (dest/'PHOSPHO_AMBIGUOUS_SEQUENCES.json').write_text(json.dumps(ambiguous))
            else:
                f=features.loc[features.layer.eq(layer)].drop_duplicates('feature_id').set_index('feature_id')
                f=f.reindex(frame.index)
                valid=f.reference_presence.eq('GENCODE_V36') & f.reference_symbol.fillna('').ne('')
                if layer=='protein':valid &= f.gene_biotype.eq('protein_coding')
                val=frame.loc[valid].copy();val.index=f.loc[valid,'reference_symbol']
                duplicates=int(val.index.duplicated().sum())
                val=val.groupby(level=0,sort=True).mean();val.index.name='gene_symbol'
                val.to_csv(dest/(layer.upper()+'_SYMBOL.tsv.gz'),sep='\t',na_rep='NA',compression='gzip')
                audit.append({'source_cohort':cohort,'layer':layer,'source_features':len(frame),'scoring_features':len(val),
                              'duplicate_symbol_rows_collapsed_mean':duplicates,'normalization':'PUBLISHED_NATIVE_SCALE'})
                if layer=='methylation_gene':
                    scores,counts=methylation_means(val,sets)
                    scores.index.name='pathway_id';counts.index.name='pathway_id'
                    scores.to_csv(dest/'METHYLATION_PATHWAY_MEAN_BETA.tsv.gz',sep='\t',na_rep='NA',compression='gzip')
                    counts.to_csv(dest/'METHYLATION_PATHWAY_OBSERVED_GENES.tsv.gz',sep='\t',compression='gzip')
            print('OMICS_READY',cohort,layer,flush=True)
    pd.DataFrame(audit).to_csv(out/'OMICS_PREPARATION_AUDIT.tsv',sep='\t',index=False)
    result={'format':'CLA_CPTAC_OMICS_SCORE_INPUT_V1','status':'COMPLETE','target_host':'149','config':cfg,
      'methylation_definition':meth_definition,'no_imputation':True,
      'files':{str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()}}
    atomic_json(out/'COMPLETE.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(read_json(a.config),a.output)
if __name__=='__main__':main()
