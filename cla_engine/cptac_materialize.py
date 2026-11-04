"""CPTAC publication adapters with explicit patient, tissue and feature identity."""
from __future__ import annotations
import argparse, csv, json, re, socket
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json, read_json, sha, digest, fail
from .state_protein_map import reference_genes, ProteinResolver, pinned

FORMAT='CLA_CPTAC_ALIGNED_INPUT_V1'
from .cptac_scope import CANCER_MAP, select_cohorts
UNITS={'rna':'log2(TPM+1)','protein':'BCM_LOG2_REFERENCE_INTENSITY_NORMALIZED','phosphorylation':'BCM_SITE_LOG2_REFERENCE_INTENSITY_NORMALIZED','methylation_gene':'BCM_GENE_BETA_VALUE_PROMOTER_SCOPE_REQUIRES_SOURCE_VERIFICATION'}

def gene_id(value):
    return re.sub(r'\.\d+(?=_PAR_Y$|$)','',str(value))

def case_ids(path):
    result=Path(path).read_text().splitlines()
    if not result or any(not x or x!=x.strip() for x in result) or len(set(result))!=len(result):
        fail('CPTAC_CASE_AUTHORITY','Expected unique nonempty native case IDs')
    return result

def sample_identity(cohort, native, role, filename, tumor, normal):
    if role=='rna' and cohort not in ('BRCA','COAD','OV'):
        match=re.fullmatch(r'(.+)_(T|A)',native)
        if not match:return None
        case,tissue=match[1],('Tumor' if match[2]=='T' else 'Normal')
    else:
        case=native;tissue='Normal' if filename.endswith('_Normal.txt') else 'Tumor'
    if case not in (tumor if tissue=='Tumor' else normal):return None
    patient='CPTAC:'+cohort+':'+case
    return {'cohort_id':'CPTAC','source_cohort':cohort,'cancer_id':CANCER_MAP[cohort],'native_case_id':case,
            'patient_id':patient,'sample_id':patient+('::T' if tissue=='Tumor' else '::N'),'tissue_type':tissue}

def read_matrix(path):
    with Path(path).open() as f:
        reader=csv.reader(f,delimiter='\t');header=next(reader);first=next(reader)
    # R write.table may omit the row-name heading altogether. Validate this
    # explicitly before pandas can mangle duplicate column names.
    if len(first)==len(header)+1: columns=header
    elif len(first)==len(header): columns=header[1:]
    else: fail('CPTAC_MATRIX_HEADER','Unsupported header/row dimensions',path=str(path))
    if len(set(columns))!=len(columns):fail('CPTAC_DUPLICATE_SAMPLE_COLUMN','Repeated original column IDs',path=str(path))
    f=pd.read_csv(path,sep='\t',index_col=0)
    if list(f.columns)!=columns:
        fail('CPTAC_MATRIX_HEADER','Header/feature dimensions disagree',path=str(path))
    if not f.index.is_unique:fail('CPTAC_DUPLICATE_FEATURE','Repeated feature IDs',path=str(path))
    f.index=f.index.astype(str)
    try:f=f.apply(pd.to_numeric,errors='raise')
    except Exception:fail('CPTAC_NONNUMERIC_ASSAY','Invalid measurement',path=str(path))
    values=f.to_numpy(); rr,cc=np.where(np.isinf(values))
    f.attrs['nonfinite_source']=[{'source_feature_id':str(f.index[i]),'native_sample_id':str(f.columns[j]),'raw_value':str(values[i,j])} for i,j in zip(rr,cc)]
    f=f.replace([np.inf,-np.inf],np.nan)
    return f

def feature_record(role, source_id, resolver):
    parts=str(source_id).split('|')
    gid=gene_id(parts[0])
    x=resolver.resolve('',gid)
    rec={'layer':role,'source_feature_id':source_id,'gene_id':gid,
         'model_gene_id':'GENE:'+gid if x['reference_presence']=='GENCODE_V36' else '',
         **{k:x[k] for k in ['reference_symbol','gene_biotype','uniprot_accessions','mapping_status','reference_presence']}}
    if role=='phosphorylation':
        valid=len(parts)==5 and bool(re.fullmatch(r'[STY][0-9]+',parts[2]))
        rec.update(protein_isoform=parts[1] if len(parts)>1 else '',site=parts[2] if len(parts)>2 else '',
                   flanking_sequence=parts[3] if len(parts)>3 else '',source_terminal_field=parts[4] if len(parts)>4 else '')
        valid=valid and bool(re.fullmatch('[A-Z_]{15}',parts[3])) and parts[3][7]==parts[2][0]
        rec['ptm_sequence_id']=parts[3]+'-p' if valid else ''
        rec['ptm_identity_status']='SINGLE_RESIDUE_WITH_MATCHING_CENTER' if valid else 'NOT_A_VERIFIED_SINGLE_SITE_WINDOW'
        rec['feature_id']=source_id
    else:rec['feature_id']=gid
    return rec

def materialize(cfg, output, cancer_type='all', cancer_list=None):
    if socket.gethostname().split('.')[0]!='149':fail('CPTAC_EXECUTION_HOST','Run on host 149')
    out=Path(output)
    if out.exists() and any(out.iterdir()):fail('CPTAC_OUTPUT_EXISTS','Use a new versioned output')
    out.mkdir(parents=True,exist_ok=True)
    download=read_json(pinned(cfg['download_complete']))
    if download['status']!='COMPLETE':fail('CPTAC_INCOMPLETE_DOWNLOAD','Inputs are incomplete')
    selected=select_cohorts(cancer_type,cancer_list,download['cohorts'])
    if not selected or not selected<=set(CANCER_MAP):fail('CPTAC_CANCER_SCOPE','Unknown or empty cohort list')
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'CPTAC_MATERIALIZATION','cohorts':sorted(selected)})
    genes,symbols=reference_genes(pinned(cfg['gencode']))
    hgnc=pd.read_csv(pinned(cfg['hgnc']),sep='\t',dtype=str,keep_default_na=False)
    graph=pd.read_parquet(pinned(cfg['protein_gene']))
    resolver=ProteinResolver(genes,symbols,hgnc,graph)
    files=[r for r in download['files'] if r['cohort'] in selected]
    for rec in files:
        if rec['status']!='COMPLETE' or sha(rec['path'])!=rec['sha256']:fail('CPTAC_RAW_HASH',rec['name'])
    patient_rows=[];sample_rows=[];excluded=[];measurements=[];feature_rows={};clinical=[];phenotypes=[];nonfinite=[];invalid_rna=[];rna_exclusions=[]
    for cohort in sorted(selected):
        records=[r for r in files if r['cohort']==cohort]
        paths={r['name']:Path(r['path']) for r in records}
        tumor=case_ids(paths[cohort+'_Tumor_CaseList.txt'])
        normal=case_ids(paths[cohort+'_Normal_CaseList.txt']) if cohort+'_Normal_CaseList.txt' in paths else []
        for case in tumor:
            patient_rows.append({'cohort_id':'CPTAC','source_cohort':cohort,'cancer_id':CANCER_MAP[cohort],
                                 'native_case_id':case,'patient_id':'CPTAC:'+cohort+':'+case})
        meta=pd.read_csv(paths[cohort+'_meta.txt'],sep='\t',dtype=str,keep_default_na=False)
        meta=meta.loc[meta.case_id.ne('data_type')].rename(columns={'case_id':'native_case_id'})
        survival=pd.read_csv(paths[cohort+'_survival.txt'],sep='\t',dtype=str,keep_default_na=False)
        survival=survival.rename(columns={survival.columns[0]:'native_case_id'})
        if meta.native_case_id.duplicated().any() or survival.native_case_id.duplicated().any():fail('CPTAC_CLINICAL_DUPLICATE',cohort)
        merged=pd.DataFrame({'native_case_id':tumor}).merge(meta,how='left',validate='one_to_one').merge(survival,how='left',on='native_case_id',suffixes=('','_survival'),validate='one_to_one')
        merged.insert(0,'patient_id','CPTAC:'+cohort+':'+merged.native_case_id);merged.insert(1,'source_cohort',cohort);merged.insert(2,'cancer_id',CANCER_MAP[cohort])
        clinical.append(merged)
        ph=pd.read_csv(paths[cohort+'_phenotype.txt'],sep='\t',index_col=0);ph.index=ph.index.astype(str)
        if not ph.index.is_unique:fail('CPTAC_PHENOTYPE_DUPLICATE',cohort)
        ph=ph.reindex(tumor);ph.index=['CPTAC:'+cohort+':'+x for x in ph.index];ph.index.name='patient_id'
        phenotypes.append(ph.reset_index())
        for rec in records:
            role=rec['role']
            if role not in UNITS:continue
            frame=read_matrix(rec['path'])
            nonfinite.extend({'source_file':rec['name'],**x} for x in frame.attrs.get('nonfinite_source',[]))
            selections=[]
            for native in frame:
                identity=sample_identity(cohort,native,role,rec['name'],set(tumor),set(normal))
                if identity is None:
                    excluded.append({'source_file':rec['name'],'native_sample_id':native,'reason':'NOT_IN_PUBLISHED_TUMOR_OR_NORMAL_CASE_LIST'})
                else:selections.append((native,identity));sample_rows.append({**identity,'layer':role,'native_sample_id':native,'source_file':rec['name']})
            frame=frame[[x[0] for x in selections]]
            frame.columns=[x[1]['sample_id'] for x in selections]
            if frame.columns.duplicated().any():fail('CPTAC_SAMPLE_COLLISION',rec['name'])
            if role=='rna':
                a=frame.to_numpy()
                if np.any(a < -0.02) or np.any(a>50):fail('CPTAC_RNA_SCALE','Outside declared log2(TPM+1) bounds')
                ii,jj=np.where(a<0)
                invalid_rna.extend({'source_file':rec['name'],'source_feature_id':str(frame.index[i]),'sample_id':str(frame.columns[j]),'raw_value':float(a[i,j]),'action':'MASK_INVALID_NEGATIVE_LOG2_TPM_PLUS1_NOT_ZERO_IMPUTATION'} for i,j in zip(ii,jj))
                frame=frame.mask(frame<0)
            if role=='methylation_gene':
                a=frame.to_numpy()
                if np.any(a[np.isfinite(a)]<0) or np.any(a[np.isfinite(a)]>1):fail('CPTAC_METHYLATION_SCALE','Expected beta in [0,1]')
            rows=[feature_record(role,x,resolver) for x in frame.index]
            for row in rows:feature_rows[(role,row['source_feature_id'])]=row
            normalized=[x['feature_id'] for x in rows]
            if len(set(normalized))!=len(normalized):fail('CPTAC_NORMALIZED_FEATURE_COLLISION',rec['name'])
            frame.index=normalized;frame.index.name='feature_id'
            folder=out/'matrices'/cohort;folder.mkdir(parents=True,exist_ok=True)
            tissue='Normal' if rec['name'].endswith('_Normal.txt') else ('TumorNormal' if role=='rna' else 'Tumor')
            dest=folder/(role+'_'+tissue+'.parquet');frame.reset_index().to_parquet(dest,index=False)
            for i,(_,identity) in enumerate(selections):
                measurements.append({**identity,'layer':role,'features':len(frame),'observed_features':int(frame.iloc[:,i].notna().sum()),
                    'source_unit':UNITS[role],'matrix_path':str(dest),'source_file':rec['name']})
            if role=='rna':
                tumor_cols=[x[1]['sample_id'] for x in selections if x[1]['tissue_type']=='Tumor']
                symbol_rows=[(i,genes[gid]['symbol'],genes[gid]['biotype']) for i,gid in enumerate(frame.index) if gid in genes and genes[gid]['symbol']]
                valid=frame.iloc[[x[0] for x in symbol_rows]][tumor_cols].copy()
                valid.index=[x[1] for x in symbol_rows]
                bad_symbols=set(valid.index[valid.isna().any(axis=1)])
                rna_exclusions.extend({'source_cohort':cohort,'gene_symbol':x,'reason':'SOURCE_INVALID_OR_MISSING_VALUE;RNA_SCORER_REQUIRES_FINITE_INPUT'} for x in sorted(bad_symbols))
                valid=valid.loc[~valid.index.isin(bad_symbols)]
                # State scoring uses linear TPM; Ensembl loci sharing a symbol are summed before log transformation.
                tpm=np.expm1(valid*np.log(2)).groupby(level=0,sort=False).sum()
                tpm.index.name='gene_symbol';tpm.to_csv(folder/'RNA_TUMOR_SYMBOL_TPM.tsv.gz',sep='\t',compression={'method':'gzip','mtime':0})
                coding={x[1] for x in symbol_rows if x[2]=='protein_coding'}
                pd.DataFrame({'gene_symbol':sorted(coding)}).to_csv(folder/'PROTEIN_CODING_SYMBOLS.tsv',sep='\t',index=False)
            print('MATERIALIZED',cohort,role,tissue,frame.shape,flush=True)
    patients=pd.DataFrame(patient_rows);samples=pd.DataFrame(sample_rows);obs=pd.DataFrame(measurements)
    if patients.patient_id.duplicated().any():fail('CPTAC_PATIENT_COLLISION','Patient namespace collision')
    patients.to_csv(out/'CPTAC_PATIENTS.tsv',sep='\t',index=False)
    samples.to_csv(out/'CPTAC_SAMPLE_ASSAY_MAP.tsv',sep='\t',index=False)
    obs.to_csv(out/'CPTAC_ASSAY_COVERAGE.tsv',sep='\t',index=False)
    pd.DataFrame(invalid_rna,columns=['source_file','source_feature_id','sample_id','raw_value','action']).to_csv(out/'SOURCE_INVALID_RNA_VALUES.tsv',sep='\t',index=False)
    pd.DataFrame(rna_exclusions,columns=['source_cohort','gene_symbol','reason']).to_csv(out/'RNA_SCORING_EXCLUDED_GENES.tsv',sep='\t',index=False)
    pd.DataFrame(nonfinite,columns=['source_file','source_feature_id','native_sample_id','raw_value']).to_csv(out/'SOURCE_NONFINITE_MEASUREMENTS.tsv',sep='\t',index=False)
    pd.DataFrame(excluded).to_csv(out/'CPTAC_EXCLUDED_SAMPLES.tsv',sep='\t',index=False)
    pd.concat(clinical,ignore_index=True).to_csv(out/'CLINICAL.tsv',sep='\t',index=False)
    pd.concat(phenotypes,ignore_index=True).to_parquet(out/'PUBLISHED_PHENOTYPES.parquet',index=False)
    pd.DataFrame(feature_rows.values()).to_parquet(out/'FEATURE_ALIGNMENT.parquet',index=False)
    old=pd.read_csv(pinned(cfg['tcga_patient_map']),sep='\t')
    overlaps=set(patients.native_case_id)&set(old.patient_id)
    if overlaps:fail('CPTAC_TCGA_PATIENT_OVERLAP','Native IDs overlap TCGA; explicit identity reconciliation required',patients=sorted(overlaps))
    coverage=[]
    for cohort,f in patients.groupby('source_cohort'):
        for layer in UNITS:
            hits=obs.loc[obs.source_cohort.eq(cohort)&obs.tissue_type.eq('Tumor')&obs.layer.eq(layer)&obs.observed_features.gt(0),'patient_id'].nunique()
            coverage.append({'source_cohort':cohort,'cancer_id':CANCER_MAP[cohort],'layer':layer,'patients':len(f),'observed_patients':hits,'coverage_percent':100*hits/len(f)})
    pd.DataFrame(coverage).to_csv(out/'COHORT_COVERAGE.tsv',sep='\t',index=False)
    result={'format':FORMAT,'status':'COMPLETE','target_host':'149','patients':len(patients),'cohorts':sorted(selected),
        'cohort_mapping':{k:CANCER_MAP[k] for k in sorted(selected)},'tcga_native_id_overlap':0,
        'sample_authority':'LinkedOmicsKB published Tumor/Normal CaseList; omitted RNA-only publication extras retained in raw and exclusion table',
        'missingness':'PRESERVED_NO_IMPUTATION;SOURCE_INFINITY_MASKED_WITH_EXPLICIT_AUDIT','phosphorylation':'SITE_LEVEL_NOT_GENE_AVERAGE','config':cfg,
        'files':{str(p.relative_to(out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.rglob('*') if p.is_file()}}
    atomic_json(out/'COMPLETE.json',result)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cancer-type','--cancer_type',choices=['all','list'],default='all');p.add_argument('--cancer-list','--cancer_list',nargs='+')
    a=p.parse_args();r=materialize(read_json(a.config),a.output,a.cancer_type,a.cancer_list);print(r['status'],r['patients'])

if __name__=='__main__':main()
