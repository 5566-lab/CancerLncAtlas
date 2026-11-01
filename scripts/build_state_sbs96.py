"""Build audited SBS96 counts from MC3's reference-centered context on host 149."""
import collections,gzip,hashlib,json,os,re,socket,time,sys
from pathlib import Path
assert socket.gethostname().split('.')[0]=='149'
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cla_engine.common import atomic_json,sha
from cla_engine.cancer_scope import resolve_cancer_scope
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--maf',required=True);parser.add_argument('--patient-map',required=True);parser.add_argument('--output-root',required=True)
parser.add_argument('--cancer-type','--cancer_type',choices=['all','list'],default='all')
parser.add_argument('--cancer-list','--cancer_list',nargs='+',default=[])
args=parser.parse_args();out=Path(args.output_root)
if out.exists() and any(out.iterdir()):raise FileExistsError('Use a new output-root')
out.mkdir(parents=True,exist_ok=True)
atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'SBS96_PREPARATION'})
import pandas as pd
import numpy as np
source=Path(args.maf);authority=Path(args.patient_map)
fm=pd.read_csv(authority,sep='\t',usecols=['patient_id','cancer_id']).drop_duplicates()
assert not fm.patient_id.duplicated().any()
scope=resolve_cancer_scope({'cancer_type':args.cancer_type,'cancer_list':args.cancer_list},fm.cancer_id.unique())
fm=fm.loc[fm.cancer_id.isin(scope['selected_cancers'])];patients=set(fm.patient_id)
input_sha=sha(source);authority_sha=sha(authority)
cols=['NCBI_Build','Chromosome','Start_Position','Variant_Type','Reference_Allele','Tumor_Seq_Allele2','Tumor_Sample_Barcode','FILTER','CONTEXT']
counts=collections.Counter();audit=collections.Counter();seen=set();examples=[];complement=str.maketrans('ACGT','TGCA')
for chunk in pd.read_csv(source,sep='\t',comment='#',usecols=cols,dtype=str,chunksize=100000,keep_default_na=False):
    audit['input_rows']+=len(chunk)
    for value,n in chunk.FILTER.value_counts().items():audit['filter:'+value]+=int(n)
    chunk=chunk.loc[chunk.Variant_Type.eq('SNP')&chunk.FILTER.eq('PASS')].copy()
    audit['pass_snv_rows']+=len(chunk)
    # Explicit TCGA participant parsing; malformed barcodes never use substring matching.
    ids=chunk.Tumor_Sample_Barcode.str.extract(r'^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-(\d{2})[A-Z]-')
    chunk['patient_id']=ids[0];chunk['sample_type']=pd.to_numeric(ids[1],errors='coerce')
    audit['invalid_tcga_barcode']+=int(ids[0].isna().sum())
    take=chunk.patient_id.isin(patients)&chunk.sample_type.between(1,9)
    audit['not_authority_or_not_tumor']+=int((~take).sum());chunk=chunk.loc[take]
    for row in chunk.itertuples(index=False):
        build=row.NCBI_Build
        if build not in ('GRCh37','37','hg19'):raise ValueError('Unexpected MC3 reference build: '+build)
        ref=row.Reference_Allele.upper();alt=row.Tumor_Seq_Allele2.upper();context=row.CONTEXT.upper()
        if len(ref)!=1 or len(alt)!=1 or ref not in 'ACGT' or alt not in 'ACGT' or ref==alt:
            audit['invalid_snv_alleles']+=1;continue
        if len(context)<3 or len(context)%2!=1:
            audit['invalid_context_length']+=1;continue
        center=len(context)//2;tri=context[center-1:center+2]
        if tri[1]!=ref:
            audit['context_ref_mismatch']+=1;continue
        if any(x not in 'ACGT' for x in tri):audit['ambiguous_context']+=1;continue
        key=(row.patient_id,row.Chromosome,row.Start_Position,ref,alt)
        if key in seen:audit['duplicate_patient_variant']+=1;continue
        seen.add(key)
        if len(examples)<1000:examples.append({'chromosome':row.Chromosome,'position':int(row.Start_Position),'context':tri})
        if ref in 'AG':tri=tri.translate(complement)[::-1];ref=ref.translate(complement);alt=alt.translate(complement)
        context_name=tri[0]+'['+ref+'>'+alt+']'+tri[2]
        counts[(row.patient_id,context_name)]+=1;audit['accepted_unique_snv']+=1
    print(json.dumps({'rows':audit['input_rows'],'accepted_unique_snv':audit['accepted_unique_snv'],'patients':len(set(p for p,c in counts))}),flush=True)
contexts=[a+'['+change+']'+b for change in ['C>A','C>G','C>T','T>A','T>C','T>G'] for a in 'ACGT' for b in 'ACGT']
assert len(contexts)==96
df=pd.DataFrame([(p,c,n) for (p,c),n in counts.items()],columns=['patient_id','mutation_type','count'])
matrix=df.pivot(index='mutation_type',columns='patient_id',values='count').reindex(contexts).fillna(0).astype('int64')
matrix.to_csv(out/'SBS96_MC3_COUNTS.tsv.gz',sep='\t')
totals=matrix.sum(axis=0).rename('snv_count').reset_index().merge(fm,on='patient_id',validate='one_to_one')
totals.to_csv(out/'SBS96_PATIENT_QC.tsv',sep='\t',index=False)
pd.DataFrame(examples).to_csv(out/'SBS96_REFERENCE_CHECK.tsv',sep='\t',index=False)
assert sha(source)==input_sha and sha(authority)==authority_sha
record={'target_host':'149','source':str(source),'source_sha256':input_sha,'patient_authority_sha256':authority_sha,'cancer_scope':scope,'patient_authority':str(authority),'counts_shape':list(matrix.shape),'audit':dict(audit),
        'sample_identity_rule':'Parse validated TCGA participant component from full MC3 aliquot barcode; exact participant match to frozen authority; sample type 01-09 only',
        'deduplication':'participant, chromosome, position, ref, alt','reference':'GRCh37',
        'context':'MC3 CONTEXT center must equal reference allele; pyrimidine-oriented SBS96',
        'zero_mutation_unsequenced_patients_not_fabricated':True,'minimum_fit_snv_count':50,
        'state':'COUNTS_READY_REFERENCE_CHECK_AND_SIGNATURE_REFIT_PENDING'}
assert audit['context_ref_mismatch']==0,record
(out/'SBS96_COUNTS_RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
