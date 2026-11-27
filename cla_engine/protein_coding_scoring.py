"""Build RNA scoring inputs from protein-coding gene IDs before symbol pooling."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .measured_readouts import read_partition
from .relation_evidence_bindings import file_sha


def aggregate_coding_genes(gene_logtpm, annotation):
    if not annotation.gene_id.is_unique or not gene_logtpm.columns.is_unique:
        raise ValueError('PC_SCORING_DUPLICATE_GENE')
    selected=annotation.loc[annotation.gene_biotype.eq('protein_coding')].copy()
    ids=selected.gene_id.tolist()
    if not set(ids).issubset(gene_logtpm.columns):
        raise ValueError('PC_SCORING_MISSING_MEASUREMENT_AXIS')
    gene=gene_logtpm.loc[:,ids].copy()
    values=gene.to_numpy(float)
    if np.isinf(values).any() or (values[np.isfinite(values)]<0).any():
        raise ValueError('PC_SCORING_INVALID_LOGTPM')
    named=selected.loc[selected.reference_symbol.notna() & selected.reference_symbol.ne('')]
    tpm=np.expm1(gene[named.gene_id].T*np.log(2))
    tpm.index=named.reference_symbol.to_numpy()
    grouped=tpm.groupby(level=0,sort=True)
    total=grouped.sum(min_count=1)
    missing=tpm.isna().groupby(level=0,sort=True).any()
    symbols=np.log2(total.mask(missing)+1).T
    gene.index.name=symbols.index.name='patient_id'
    return gene,symbols


def materialize_coding_input(aligned, annotation_path, split_path, output):
    from .local_execution import require_host_149
    require_host_149()
    aligned,annotation_path,split_path,output=map(Path,(aligned,annotation_path,split_path,output))
    if output.exists():
        raise ValueError('PC_SCORING_OUTPUT_EXISTS')
    output.mkdir(parents=True)
    def save(name,value):
        (output/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    save('LAUNCH_PREFLIGHT.json',{'target_host':'149','training':False,'operation':'protein-coding gene-ID-only RNA scoring input'})
    receipt=json.loads((aligned/'COMPLETE.json').read_text())
    if receipt.get('target_host')!='149' or file_sha(aligned/'GENE_LOGTPM.parquet')!=receipt['files']['GENE_LOGTPM.parquet']:
        raise ValueError('PC_SCORING_SOURCE_DRIFT')
    split=json.loads(split_path.read_text())
    if split.get('outer_fold')!=0 or split.get('source_folds')!=[2,3,4] or split.get('label_folds')!=[1]:
        raise ValueError('PC_SCORING_REQUIRES_DEVELOPMENT_PARTITION')
    patients=split['source_patients']+split['label_patients']
    ann=pd.read_csv(annotation_path,sep='\t')
    ids=ann.loc[ann.gene_biotype.eq('protein_coding'),'gene_id'].tolist()
    source=read_partition(aligned/'GENE_LOGTPM.parquet',patients,ids)
    gene,symbols=aggregate_coding_genes(source,ann)
    gene.to_parquet(output/'GENE_LOGTPM.parquet',compression='zstd')
    symbols.to_parquet(output/'SYMBOL_LOGTPM.parquet',compression='zstd')
    pd.DataFrame({'symbol':symbols.columns}).to_csv(output/'PROTEIN_CODING_SYMBOLS.tsv',sep='\t',index=False)
    ann[ann.gene_id.isin(ids)].to_csv(output/'SCORING_GENE_ANNOTATION.tsv',sep='\t',index=False)
    save('PATIENT_PARTITIONS.json',split)
    result={'target_host':'149','scoring_background':'PROTEIN_CODING_GENE_IDS_BEFORE_SYMBOL_AGGREGATION',
            'patients':len(gene),'coding_gene_ids':len(gene.columns),'coding_symbols':len(symbols.columns),
            'source_annotation_sha256':file_sha(annotation_path),'source_split_sha256':file_sha(split_path),
            'source_gene_logtpm_sha256':receipt['files']['GENE_LOGTPM.parquet'],
            'files':{p.name:file_sha(p) for p in output.iterdir() if p.is_file()}}
    save('COMPLETE.json',result)
    return result
