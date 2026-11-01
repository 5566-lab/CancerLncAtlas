"""Read actual table formats; never infer wildtype/measurement from missing records."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .common import fail,sha
from .identity import frame_ids, is_id_column, token, gene, coverage, unique_values

def table(path, columns=None):
    p=Path(path)
    if not p.exists():fail('MATERIAL_NOT_FOUND','Input missing',path=str(p))
    if p.suffix.lower() in ['.parquet','.pq']:
        try:return frame_ids(pd.read_parquet(p,columns=columns))
        except ImportError:fail('DEPENDENCY_MISSING','pyarrow is required to read Parquet. Use the existing cc_hhgt environment.')
    if p.is_dir():
        files=sorted(p.glob('*.parquet'))
        if not files:fail('SCHEMA_MISMATCH','Expected explicit partition or table, not recursive whole-disk search',path=str(p))
        return pd.concat([table(x,columns) for x in files],ignore_index=True)
    sep=',' if '.csv' in p.name.lower() else '\t'
    header=pd.read_csv(p,sep=sep,nrows=0).columns
    types={name:'string' for name in header if is_id_column(name)}
    return frame_ids(pd.read_csv(p,sep=sep,usecols=columns,dtype=types,low_memory=False))

def partition(root,cancer):
    r=Path(root)
    choices=[r/f'cancer_id={cancer}',r/f'cancer={cancer}',r/f'{cancer}.parquet',r/f'{cancer}.tsv.gz',r/f'{cancer}.tsv',r/cancer]
    found=[x for x in choices if x.exists()]
    if len(found)!=1:fail('PARTITION_AMBIGUOUS' if found else 'MATERIAL_NOT_FOUND','Expected exactly one cancer partition',cancer=cancer,root=str(root),matches=list(map(str,found)))
    f=table(found[0]);
    if 'cancer_id' not in f:f['cancer_id']=cancer
    if set(f.cancer_id.dropna().astype(str))!={str(cancer)}:fail('CANCER_ID_DRIFT','Wrong cancer in partition',path=str(found[0]))
    return f,found[0]

def require(f,cols,role):
    if set(cols)-set(f):fail('SCHEMA_MISMATCH',role,missing=sorted(set(cols)-set(f)),found=list(f))

def strict_bool(v):
    # Strings "False" must never be converted via bool(str).
    s=pd.Series(v);mapped=s.map({True:True,False:False,1:True,0:False,'1':True,'0':False,'true':True,'false':False,'True':True,'False':False})
    invalid=s.notna()&mapped.isna()
    if invalid.any():fail('INVALID_BOOLEAN','Unrecognized availability values',examples=s[invalid].astype(str).unique().tolist()[:10])
    return mapped.fillna(False).to_numpy(bool)

def patient_map(path):
    f=table(path);require(f,['cancer_id','sample_id','patient_id','patient_fold_id'],'patient authority')
    for k in ['cancer_id','sample_id','patient_id']:
        if f[k].isna().any():fail('IDENTITY_MISSING',k)
        f[k]=f[k].astype(str)
    f=f.drop_duplicates()
    for keys,value in [(['cancer_id','sample_id'],'patient_id'),(['cancer_id','patient_id'],'patient_fold_id')]:
        if f.groupby(keys,observed=True)[value].nunique().gt(1).any():fail('PATIENT_FOLD_CONFLICT','Patient/sample maps to multiple identities or folds')
    f['patient_fold_id']=pd.to_numeric(f.patient_fold_id,errors='coerce')
    if not f.patient_fold_id.isin(range(5)).all():fail('FOLD_CONTRACT','Expected folds 0..4')
    if f.groupby('patient_id',observed=True).cancer_id.nunique().gt(1).any():fail('PATIENT_CANCER_CONFLICT','Same patient in different cancers')
    return f

def attach_patient(f,mapping, *, minimum_match=1.0, report=None):
    f=frame_ids(f);m=frame_ids(mapping)
    require(f,['cancer_id'],'patient table')
    key='sample_id' if 'sample_id' in f else 'patient_id'
    require(f,[key],'patient table')
    keys=['cancer_id',key];cols=keys+(['patient_id'] if key=='sample_id' else [])
    authority=unique_values(m,keys,cols[len(keys):],'sample/patient authority')
    original=f['patient_id'].copy() if key=='sample_id' and 'patient_id' in f else None
    if original is not None:f=f.rename(columns={'patient_id':'__claimed_patient'})
    out=f.merge(authority,on=keys,how='left',validate='many_to_one',indicator=True)
    bad=out['_merge'].ne('both')
    n=len(out);fraction=(n-int(bad.sum()))/max(n,1)
    record={'role':'sample_patient_join','rows':n,'unmatched_rows':int(bad.sum()),'matched_fraction':fraction,
            'unmatched_exclusion_policy':'MEASUREMENTS_OUTSIDE_FROZEN_PATIENT_AUTHORITY_EXCLUDED_AND_AUDITED_NOT_SILENT_DROP'}
    # Real data carries measurement rows whose samples/patients are outside the
    # frozen patient authority (e.g. extra TCGA aliquots or patients excluded by
    # the routed candidate universe). The current State input contract counts
    # these as explicit exclusions, not failures. They are excluded and audited
    # here; a completely unmatched table still fails, and in-authority sample
    # identity conflicts still fail below.
    if bad.any() and original is not None:
        owns=out.loc[bad,'__claimed_patient'].fillna('')
        inauth=owns.isin(set(m['patient_id'].astype(str)))&owns.ne('')
        record['unmatched_other_samples_of_authority_patients']=int(inauth.sum())
        record['unmatched_patients_outside_authority']=int((owns.ne('')&~inauth).sum())
    if report is not None:report.append(record)
    if n and bad.all():
        fail('PATIENT_AUTHORITY_COVERAGE_FAIL','No silent inner-drop of unassigned measurements',**record,
             examples=out.loc[bad,keys].drop_duplicates().head(8).to_dict('records'))
    if original is not None:
        conflict=~bad & out.__claimed_patient.notna() & out.__claimed_patient.ne(out.patient_id)
        if conflict.any():fail('SAMPLE_PATIENT_CONFLICT','Supplied patient disagrees with sample authority')
        out=out.drop(columns='__claimed_patient')
    return out.loc[~bad].drop(columns='_merge')

def matrix(f,entity,value,patients,entities,mapping, *, minimum_patient_coverage=1.0,
           minimum_entity_coverage=1.0, required_entities=None, report=None):
    f=attach_patient(f,mapping,report=report);require(f,[entity,value],'matrix')
    entities=[gene(x) if entity=='gene_id' else token(x,entity) for x in entities]
    if entity=='gene_id':f[entity]=f[entity].map(gene)
    else:f[entity]=f[entity].map(lambda x:token(x,entity))
    raw=f[value];f[value]=pd.to_numeric(raw,errors='coerce')
    invalid=raw.notna() & f[value].isna()
    if invalid.any() or np.isinf(f[value].to_numpy(float)).any():fail('MEASUREMENT_NONFINITE','Invalid nonmissing molecular value',value=value)
    reports=[coverage(patients,f.patient_id,role=value+':patients',minimum=minimum_patient_coverage),
             coverage(required_entities if required_entities is not None else entities,f[entity],role=value+':'+entity,minimum=minimum_entity_coverage)]
    if report is not None:report.extend(reports)
    keys=['patient_id',entity]
    u=unique_values(f,keys,[value],role=value+'; conflicting multisample records require sample selection authority')
    return u.pivot(index='patient_id',columns=entity,values=value).reindex(index=list(map(str,patients)),columns=entities).to_numpy(dtype=float,copy=True)

def cnv(root,cancer,patients,lncs, *, required_lncs=None, minimum_patient_coverage=1.0, minimum_entity_coverage=1.0, report=None):
    d=Path(root)/f'cancer={cancer}'
    names=['patient_ids.json','lncrna_ids.json','lncrna_signed_mean.npy','lncrna_callable.npy']
    for n in names:
        if not (d/n).exists():fail('MATERIAL_NOT_FOUND','Signed CNV partition incomplete',path=str(d/n))
    pi=[token(x,'CNV patient') for x in json.loads((d/names[0]).read_text())];li=[token(x,'CNV lncRNA') for x in json.loads((d/names[1]).read_text())]
    patients=list(map(str,patients));lncs=list(map(str,lncs))
    audits=[coverage(patients,pi,role='CNV patients',minimum=minimum_patient_coverage),coverage(required_lncs if required_lncs is not None else lncs,li,role='CNV lncRNAs',minimum=minimum_entity_coverage)]
    if report is not None:report.extend(audits)
    if len(set(pi))!=len(pi) or len(set(li))!=len(li):fail('CNV_ID_CONFLICT','Duplicate CNV IDs')
    values=np.load(d/names[2],mmap_mode='r',allow_pickle=False);calls=np.load(d/names[3],mmap_mode='r',allow_pickle=False)
    if values.shape!=calls.shape or values.shape!=(len(pi),len(li)):fail('CNV_SHAPE_MISMATCH','CNV array/ID dimensions differ')
    if not np.isin(calls,[0,1]).all():fail('INVALID_CALLABILITY','CNV mask not binary')
    vp=pd.DataFrame(np.asarray(values),index=pi,columns=li).reindex(index=patients,columns=lncs).to_numpy(dtype=float,copy=True)
    cp=pd.DataFrame(np.asarray(calls),index=pi,columns=li).reindex(index=patients,columns=lncs).fillna(0).to_numpy(dtype=bool,copy=True)
    vp[~cp]=np.nan
    if np.any(cp&~np.isfinite(vp)):fail('CNV_CALLABLE_NONFINITE','Callable CNV has no numeric value')
    return vp,cp,[d/n for n in names]

def state_wide(source,mapping,target_catalog):
    """Exact joins. Sample or patient keys detected from declared aliases, not barcode truncation."""
    p=Path(source['path'])
    if not p.exists() and source.get('mirror'):p=Path(source['mirror'])
    if not p.exists():fail('MATERIAL_NOT_FOUND','State main and mirror missing',path=source['path'])
    if source.get('sha256') and sha(p)!=source['sha256']:fail('STATE_SOURCE_HASH_MISMATCH','State file differs from supplied authority',path=str(p))

    if source.get('bytes') and p.stat().st_size!=source['bytes']:fail('STATE_SIZE_MISMATCH','State byte size differs')
    f=table(p)
    if source.get('columns') and len(f.columns)!=source['columns']:fail('STATE_HEADER_COUNT_MISMATCH','State column count differs',actual=len(f.columns))
    if source.get('format')=='CLA_PATIENT_STATE_V1':
        from .state_materialize import verify_materialized_source
        verify_materialized_source(source,f,mapping,target_catalog)
    aliases={'sample_id':['sample_id','Sample','sample','sample_barcode','SampleID','sampleID','bcr_sample_barcode'], 'patient_id':['patient_id','Patient','patient','patient_barcode','bcr_patient_barcode'], 'cancer_id':['cancer_id','cancer','Cancer','cancer_type','CancerType','project_id']}
    rename={}
    for canonical,options in aliases.items():
        hits=[x for x in options if x in f]
        if canonical not in f and len(hits)==1:rename[hits[0]]=canonical
        elif canonical not in f and len(hits)>1:fail('AMBIGUOUS_STATE_ID','Multiple possible identity columns',columns=hits)
    f=f.rename(columns=rename)
    if 'sample_id' not in f and 'patient_id' not in f:fail('STATE_ID_COLUMN_UNKNOWN','No exact sample/patient key',columns=list(f)[:35])
    # Patient-first authority: prefer the exact patient-level key whenever the state table
    # supplies patient_id and the frozen authority is patient-first. The historical State
    # matrix carries canonical 15-char TCGA sample barcodes plus an explicit patient_id,
    # while the frozen V3.2 authority uses GDC submitted 28-char sample barcodes; the two
    # sample-level spaces are disjoint, so matching sample barcodes would need truncation
    # (forbidden). Patient barcodes are identical on both sides. This mirrors the frozen
    # patient-level alignment contract in cc_hhgt.v32.state_training (exact patient_id
    # join; multiple aliquots per patient averaged inside the patient).
    key='patient_id' if ('patient_id' in f and 'patient_id' in mapping) else ('sample_id' if 'sample_id' in f else 'patient_id')
    f[key]=f[key].astype(str)
    # Mapping supplies cancer and patient identity; unrelated metadata columns are not trained.
    m=mapping[[key,'cancer_id']+(['patient_id'] if key=='sample_id' else [])].drop_duplicates()
    if m.duplicated(key).any():fail('STATE_JOIN_AMBIGUOUS','Ambiguous authority key')
    if 'cancer_id' in f:
        orig=f[['cancer_id',key]].drop_duplicates().merge(m,on=key,suffixes=('_src','_authority'))
        a=orig.cancer_id_src.astype(str).str.removeprefix('TCGA-')
        if not a.eq(orig.cancer_id_authority.astype(str)).all():fail('STATE_CANCER_MISMATCH','State/source cancer does not match frozen mapping')
    if key=='sample_id' and 'patient_id' in f:
        chk=f[['sample_id','patient_id']].merge(m[['sample_id','patient_id']],on='sample_id',suffixes=('_src','_authority'),validate='many_to_one')
        if not chk.patient_id_src.eq(chk.patient_id_authority).all():fail('STATE_PATIENT_MISMATCH','State patient differs from sample authority')
    # Unrelated source samples are not training patients; their exclusion is explicitly audited.
    matched=set(f[key])&set(m[key])
    if not matched:fail('STATE_NO_PATIENT_MATCH','No exact State/authority key match',
        key_used=key,state_keys=[k for k in ('sample_id','patient_id') if k in f],
        authority_keys=list(mapping.columns))
    f=f.drop(columns=['cancer_id']+(['patient_id'] if key=='sample_id' and 'patient_id' in f else []),errors='ignore')
    f=f.merge(m,on=key,how='inner',validate='many_to_one')
    present=[t for t in target_catalog if t['column'] in f]
    absent=[t['id'] for t in target_catalog if t['column'] not in f and t['role'] in ('primary','auxiliary')]
    if absent:fail('REQUIRED_STATE_COLUMN_MISSING','No silent state removal',missing=absent,columns=list(f))
    if f.empty:fail('STATE_NO_PATIENT_MATCH','State/authority have no exact ID match')
    out=[]
    for t in present:
        b=f[['cancer_id','patient_id',t['column']]].rename(columns={t['column']:'value'})
        if key=='patient_id' and b.duplicated(['cancer_id','patient_id']).any():
            b=b.groupby(['cancer_id','patient_id'],observed=True).value.mean().reset_index()
        b['target_id']=t['id'];out.append(b)
    return pd.concat(out,ignore_index=True),p
