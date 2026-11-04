"""Build one native-scale State row per frozen patient, with source receipts."""
from __future__ import annotations
import argparse,json,re,socket
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json,digest,fail,read_json,sha
from .state_catalog import CATALOG_ID,ROUTED_CATALOG_ID,AUGMENTED_CATALOG_ID,catalog,expanded_targets
from .cancer_scope import resolve_cancer_scope

FORMAT='CLA_PATIENT_STATE_V1'
BARCODE=re.compile(r'^(?P<patient>TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})(?:-(?P<type>[0-9]{2})(?:[A-Z](?:-[0-9]{2}[A-Z]-[A-Z0-9]{4}-[0-9]{2})?)?)?$')
TUMOR_PREFERENCE={1:0,3:1,9:2,2:3,4:4,5:5,6:6,7:7,8:8}

def decode_barcode(value):
    """Explicit TCGA source adapter; generic prepared joins remain exact."""
    match=BARCODE.fullmatch(str(value))
    if match is None:return None
    return match['patient'],int(match['type']) if match['type'] else None

def numeric(frame,columns):
    out=frame.copy()
    for col in columns:
        if col not in out:fail('STATE_COLUMN_MISSING',col)
        values=pd.to_numeric(out[col],errors='coerce')
        if (out[col].notna() & values.isna()).any() or np.isinf(values.to_numpy(float)).any():
            fail('STATE_INVALID_NUMERIC','Non-numeric or infinite native score',column=col)
        out[col]=values
    return out

def select_patient_values(frame,authority,columns,kind):
    """Choose a measured tumor record per target; never average tumor and normal."""
    f=numeric(frame,columns)
    if 'cancer_type' in f and 'cancer_id' not in f:f=f.rename(columns={'cancer_type':'cancer_id'})
    if 'sample_id' not in f:fail('STATE_SAMPLE_ID_REQUIRED',kind)
    auth=authority.set_index('patient_id');rows=[];excluded=[]
    for i,row in f.iterrows():
        sample=str(row['sample_id']);decoded=decode_barcode(sample)
        if decoded is None:
            excluded.append({'source':kind,'sample_id':sample,'reason':'UNRECOGNIZED_SOURCE_BARCODE'});continue
        patient,stype=decoded
        if 'patient_id' in f and pd.notna(row['patient_id']) and str(row['patient_id'])!=patient:
            fail('STATE_SOURCE_PATIENT_CONFLICT',sample)
        if stype is not None and stype not in TUMOR_PREFERENCE:
            excluded.append({'source':kind,'sample_id':sample,'reason':'NON_TUMOR_SAMPLE'});continue
        if stype is None and kind!='historical':fail('STATE_RNA_REQUIRES_TUMOR_SAMPLE',sample)
        if patient not in auth.index:
            excluded.append({'source':kind,'sample_id':sample,'reason':'OUTSIDE_SELECTED_PATIENT_AUTHORITY'});continue
        cancer=auth.at[patient,'cancer_id']
        if 'cancer_id' in f and pd.notna(row['cancer_id']) and str(row['cancer_id']).removeprefix('TCGA-')!=cancer:
            fail('STATE_SOURCE_CANCER_CONFLICT',sample)
        if stype is not None and 'sample_type_code' in f and pd.notna(row['sample_type_code']) and int(row['sample_type_code'])!=stype:
            fail('STATE_SAMPLE_TYPE_CONFLICT',sample)
        if stype is not None and 'is_tumor' in f and pd.notna(row['is_tumor']) and str(row['is_tumor']).lower() not in ('true','1','1.0'):
            fail('STATE_TUMOR_FLAG_CONFLICT',sample)
        expected=str(auth.at[patient,'sample_id']);atype=decode_barcode(expected)
        if atype is None or atype[0]!=patient:fail('STATE_AUTHORITY_BARCODE_INVALID',expected)
        # Historical inputs include sample-type barcodes without vial/aliquot.
        # Match parsed components explicitly, without string-prefix joins.
        priority=0 if sample==expected else (1 if stype is not None and stype==atype[1] else 2+TUMOR_PREFERENCE.get(stype,20))
        rows.append((i,patient,cancer,priority,sample))
    key=pd.DataFrame(rows,columns=['_row','patient_id','cancer_id','_priority','sample_id'])
    values=pd.DataFrame(index=auth.index,columns=columns,dtype=float)
    choices=[];conflicts=[]
    if not key.empty:
        data=key.merge(f[columns],left_on='_row',right_index=True,validate='one_to_one')
        data=data.sort_values(['patient_id','_priority','sample_id','_row'],kind='stable')
        for col in columns:
            observed=data.loc[data[col].notna()]
            # Same source key with incompatible values is not an aliquot choice.
            duplicates=observed.groupby(['patient_id','sample_id'],sort=False)[col].nunique()
            if (duplicates>1).any():fail('STATE_DUPLICATE_SOURCE_VALUE',col)
            pick=observed.drop_duplicates('patient_id',keep='first')
            values.loc[pick.patient_id,col]=pick[col].to_numpy()
            for row in pick.itertuples(index=False,name=None):
                choices.append({'source':kind,'patient_id':row[1],'cancer_id':row[2],'column':col,'sample_id':row[4]})
            distinct=observed.groupby('patient_id',sort=False)[col].nunique()
            for patient in distinct.index[distinct>1]:
                conflicts.append({'source':kind,'patient_id':patient,'column':col,'distinct_values':int(distinct[patient]),'action':'DETERMINISTIC_TUMOR_RECORD_SELECTION_NO_AVERAGING'})
    return values,choices,excluded,conflicts

def verify_sbs(scores,qc,receipt):
    targets=[t['column'] for t in expanded_targets() if t['source_kind']=='sbs']
    if receipt.get('state')!='COMPLETE' or receipt.get('targets')!=targets or receipt.get('reference')!='COSMIC_v3.2_GRCh37':
        fail('STATE_SBS_METHOD_DRIFT','SBS receipt/method differs')
    if receipt.get('minimum_snv_count')!=50 or receipt.get('minimum_reconstruction_cosine')!=.8 or receipt.get('artifact_signatures_included') is not False:
        fail('STATE_SBS_QUALITY_POLICY_DRIFT','SBS thresholds/reference differ')
    if scores.patient_id.duplicated().any() or qc.patient_id.duplicated().any() or set(scores.patient_id)!=set(qc.patient_id):
        fail('STATE_SBS_ID_CONFLICT','SBS score/QC keys disagree')
    data=numeric(scores,targets).set_index('patient_id');q=qc.set_index('patient_id').reindex(data.index)
    if q.state_available.dtype!=bool:fail('STATE_SBS_MASK_INVALID','Expected boolean quality mask')
    good=(q.snv_count>=50)&(q.reconstruction_cosine>=.8)
    if not good.eq(q.state_available).all():fail('STATE_SBS_MASK_INVALID','Quality threshold disagreement')
    if not np.isfinite(data.loc[good,targets].to_numpy()).all() or not data.loc[~good,targets].isna().all().all():
        fail('STATE_SBS_MASK_INVALID','Below-threshold patients must be missing')
    a=data.loc[good,targets].to_numpy()
    if np.any(a<0) or np.any(a>1) or not np.allclose(a.sum(axis=1),1,atol=1e-8):fail('STATE_SBS_FRACTION_INVALID','Expected nonnegative relative contributions summing to one')
    return data[targets]

def _entry(path):
    p=Path(path)
    if not p.is_file() or p.is_symlink():fail('STATE_SOURCE_MISSING_OR_LINK',str(p))
    return {'path':str(p.resolve()),'sha256':sha(p),'bytes':p.stat().st_size}

def materialize(historical,rna_root,sbs_root,patient_map,output,*,cancer_type='all',cancer_list=None,method_provenance=None):
    if socket.gethostname().split('.')[0]!='149':fail('STATE_EXECUTION_HOST','State preparation must run on 149')
    out=Path(output)
    if out.exists() and any(out.iterdir()):fail('STATE_OUTPUT_EXISTS','Use a new output directory')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'STATE_MATERIALIZATION','cancer_type':cancer_type,'cancer_list':cancer_list or []})
    sources={'historical':_entry(historical),'patient_map':_entry(patient_map)}
    if sources['historical']['sha256']!=catalog()['historical_source_sha256']:fail('STATE_HISTORICAL_SOURCE_DRIFT','Use the reviewed native-scale matrix')
    authority=pd.read_csv(patient_map,sep='\t',dtype={'patient_id':str,'sample_id':str,'cancer_id':str})
    if authority.patient_id.duplicated().any() or authority.sample_id.duplicated().any():fail('STATE_AUTHORITY_AMBIGUOUS','Expected one frozen sample per patient')
    scope=resolve_cancer_scope({'cancer_type':cancer_type,'cancer_list':cancer_list or []},authority.cancer_id.unique())
    authority=authority.loc[authority.cancer_id.isin(scope['selected_cancers'])].sort_values(['cancer_id','patient_id']).reset_index(drop=True)
    targets=expanded_targets();historical_columns=[t['column'] for t in targets if t['source_kind']=='historical']
    header=list(pd.read_csv(historical,sep='\t',nrows=0).columns)
    metadata=[c for c in ('sample_id','patient_id','cancer_type','sample_type_code','is_tumor') if c in header]
    hist=pd.read_csv(historical,sep='\t',usecols=metadata+historical_columns)
    values,choices,excluded,conflicts=select_patient_values(hist,authority,historical_columns,'historical')
    h=values[['hrd::HRD','hrd::ai1','hrd::lst1','hrd::hrd-loh']].dropna()
    if not np.allclose(h.iloc[:,0],h.iloc[:,1:].sum(axis=1),atol=1e-10):fail('STATE_HRD_COMPONENT_CONFLICT','Published HRD total differs from three components')
    rna_columns=[t['column'] for t in targets if t['source_kind']=='rna'];rnas=[]
    for cancer in scope['selected_cancers']:
        folder=Path(rna_root);path=folder/(cancer+'.tsv.gz');receipt=folder/(cancer+'.json')
        sources['rna_'+cancer]=_entry(path);sources['rna_receipt_'+cancer]=_entry(receipt);rec=read_json(receipt)
        if rec.get('state')!='COMPLETE' or rec.get('failures') or rec.get('cancer_id')!=cancer or rec.get('score_columns')!=18 or rec.get('package_version')!='1.9.5':
            fail('STATE_RNA_INCOMPLETE_OR_METHOD_DRIFT',cancer)
        expected_methods=catalog()['rna_method_contract']
        actual_methods=rec.get('method_records',{})
        if set(actual_methods)!=set(expected_methods):fail('STATE_RNA_METHOD_SET_DRIFT',cancer)
        for method,definition in expected_methods.items():
            if any(actual_methods[method].get(k)!=v for k,v in definition.items()):
                fail('STATE_RNA_METHOD_DEFINITION_DRIFT',cancer,method=method)
        rna=pd.read_csv(path,sep='\t')
        if len(rna)!=rec['samples'] or set(rna.columns)!=set(['sample_id','cancer_id']+rna_columns) or rna.sample_id.duplicated().any():
            fail('STATE_RNA_SHAPE_DRIFT',cancer)
        if not rna.cancer_id.eq(cancer).all():fail('STATE_RNA_CANCER_DRIFT',cancer)
        rnas.append(rna)
    rv,rc,re,rf=select_patient_values(pd.concat(rnas,ignore_index=True),authority,rna_columns,'rna')
    choices+=rc;excluded+=re;conflicts+=rf
    sbs_root=Path(sbs_root)
    for name in ('SBS_STATE_FRACTIONS.tsv.gz','SBS_FIT_QC.tsv','SBS_STATE_RECEIPT.json','COSMIC_V3_2_GRCH37_SBS_REFERENCE.tsv','SBS96_COUNTS_RECEIPT.json'):
        sources[name]=_entry(sbs_root/name)
    sb=verify_sbs(pd.read_csv(sbs_root/'SBS_STATE_FRACTIONS.tsv.gz',sep='\t'),pd.read_csv(sbs_root/'SBS_FIT_QC.tsv',sep='\t'),read_json(sbs_root/'SBS_STATE_RECEIPT.json'))
    result=authority[['cancer_id','patient_id']].set_index('patient_id').join(values).join(rv).join(sb)
    result=result.reset_index()[['patient_id','cancer_id']+[t['column'] for t in targets]]
    if len(result)!=len(authority) or result.patient_id.duplicated().any():fail('STATE_MATERIALIZATION_ID_DRIFT','Patient cardinality changed')
    matrix=out/'STATE_EXPANDED_PATIENT.tsv.gz'
    result.to_csv(matrix,sep='\t',index=False,compression={'method':'gzip','mtime':0},float_format='%.17g',na_rep='NA')
    coverage=[]
    for cancer,frame in result.groupby('cancer_id',sort=True):
        for t in targets:
            x=frame[t['column']]
            coverage.append({'cancer_id':cancer,'target_id':t['id'],'family':t['family'],'role':t['role'],'measurement':t['measurement'],
                'patients':len(frame),'measured':int(x.notna().sum()),'distinct_values':int(x.nunique()),'fraction':float(x.notna().mean())})
    pd.DataFrame(coverage).to_csv(out/'STATE_COVERAGE.tsv',sep='\t',index=False)
    pd.DataFrame(choices,columns=['source','patient_id','cancer_id','column','sample_id']).to_csv(out/'STATE_VALUE_SOURCES.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(excluded,columns=['source','sample_id','reason']).to_csv(out/'STATE_EXCLUDED_ROWS.tsv',sep='\t',index=False)
    pd.DataFrame(conflicts,columns=['source','patient_id','column','distinct_values','action']).to_csv(out/'STATE_MULTIPLE_TUMOR_VALUES.tsv.gz',sep='\t',index=False,compression={'method':'gzip','mtime':0})
    for name,e in sources.items():
        if _entry(e['path'])!=e:fail('STATE_INPUT_CHANGED_DURING_MATERIALIZATION',name)
    if method_provenance is not None:
        sources['method_provenance']=_entry(method_provenance)
        method_record=read_json(method_provenance)
        if method_record.get('format')!='CLA_STATE_METHOD_PROVENANCE_V1' or method_record.get('target_host')!='149':
            fail('STATE_METHOD_PROVENANCE_INVALID','Unrecognized method provenance')
        for key in ('SBS_STATE_FRACTIONS.tsv.gz','SBS_FIT_QC.tsv','SBS_STATE_RECEIPT.json','COSMIC_V3_2_GRCH37_SBS_REFERENCE.tsv'):
            expected=method_record.get('files',{}).get(sources[key]['path'],{})
            if expected.get('sha256')!=sources[key]['sha256']:fail('STATE_METHOD_PROVENANCE_DRIFT',key)
    record={'format':FORMAT,'state':'COMPLETE','target_host':'149','catalog_id':CATALOG_ID,'target_catalog_sha256':digest(targets),
        'cancer_scope':scope,'patients':len(result),'targets':len(targets),'primary_targets':sum(t['role']=='primary' for t in targets),
        'authority_patient_map_sha256':sources['patient_map']['sha256'],'inputs':sources,
        'outputs':{p.name:_entry(p) for p in sorted(out.iterdir()) if p.is_file()},'matrix':_entry(matrix),
        'columns':list(result.columns),'selection_policy':'PER_TARGET: EXACT_AUTHORITY_ALIQUOT; PARSED_AUTHORITY_SAMPLE_TYPE; PRIMARY_TUMOR_PREFERENCE; EXPLICIT_PATIENT_LEVEL_FALLBACK; LEXICAL_TIEBREAK; NO_AVERAGING',
        'known_normal_samples_excluded':sum(r['reason']=='NON_TUMOR_SAMPLE' for r in excluded),'conflicting_tumor_patient_target_groups':len(conflicts),
        'missing_policy':'NO_IMPUTATION; UNMEASURED_IS_NA_NOT_ZERO','scores_rescaled':False,
        'biological_claim':'Native observational State measurements/estimates; no direct pathway activation or causality claim'}
    manifest=out/'STATE_MATERIALIZATION.json';atomic_json(manifest,record)
    binding={'path':str(matrix.resolve()),'sha256':record['matrix']['sha256'],'bytes':matrix.stat().st_size,'columns':len(result.columns),
        'format':FORMAT,'catalog_id':CATALOG_ID,'materialization_manifest':str(manifest.resolve()),'materialization_manifest_sha256':sha(manifest)}
    atomic_json(out/'STATE_SOURCE_BINDING.json',binding)
    verify_materialized_source(binding,result,authority,targets)
    return binding

def verify_materialized_source(source,frame,mapping,targets):
    if source.get('format')!=FORMAT or source.get('catalog_id') not in (CATALOG_ID,ROUTED_CATALOG_ID,AUGMENTED_CATALOG_ID):fail('STATE_MATERIALIZATION_REQUIRED','Expanded State requires a verified patient-level materialization')
    path=Path(source.get('materialization_manifest',''))
    if not path.is_file() or not source.get('materialization_manifest_sha256') or sha(path)!=source['materialization_manifest_sha256']:
        fail('STATE_MATERIALIZATION_HASH','Missing or changed materialization manifest')
    rec=read_json(path)
    if rec.get('format')!=FORMAT or rec.get('state')!='COMPLETE' or rec.get('target_host')!='149' or rec.get('catalog_id')!=source['catalog_id'] or targets!=expanded_targets(source['catalog_id']) or rec.get('target_catalog_sha256')!=digest(targets):
        fail('STATE_MATERIALIZATION_CATALOG','State source and reviewed target definitions differ')
    if not source.get('sha256') or rec['matrix']['sha256']!=source['sha256'] or rec['matrix']['bytes']!=source.get('bytes'):
        fail('STATE_MATERIALIZATION_MATRIX','Matrix binding differs from receipt')
    columns=['patient_id','cancer_id']+[t['column'] for t in targets]
    if list(frame.columns)!=columns or rec.get('columns')!=columns or len(frame)!=rec.get('patients') or frame.patient_id.isna().any() or frame.patient_id.duplicated().any():
        fail('STATE_PATIENT_MATRIX_SCHEMA','Expanded State needs exactly one row per explicit patient and every catalog column')
    if set(mapping.patient_id)-set(frame.patient_id):fail('STATE_PATIENT_MATRIX_COVERAGE','Selected frozen patients missing from matrix')
    chk=mapping[['patient_id','cancer_id']].drop_duplicates().merge(frame[['patient_id','cancer_id']],on='patient_id',suffixes=('_a','_s'),validate='one_to_one')
    if not chk.cancer_id_a.eq(chk.cancer_id_s).all():fail('STATE_SOURCE_CANCER_CONFLICT','Frozen patient cancer differs')
    numeric(frame,[t['column'] for t in targets])
    return rec

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('historical','rna-root','sbs-root','patient-map','output'):p.add_argument('--'+arg,required=True)
    p.add_argument('--method-provenance',help='Optional sealed computation/source provenance to bind alongside score receipts')
    p.add_argument('--cancer-type','--cancer_type',choices=['all','list'],default='all')
    p.add_argument('--cancer-list','--cancer_list',nargs='+',default=[])
    args=p.parse_args();print(json.dumps(materialize(**vars(args)),indent=2))

if __name__=='__main__':main()
