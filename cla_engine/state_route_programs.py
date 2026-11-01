"""Move existing weighted PROGENY/DoRothEA program values unchanged to State."""
import argparse,json,socket
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json,digest,fail,read_json,sha
from .cancer_scope import resolve_cancer_scope
from .state_catalog import CATALOG_ID,ROUTED_CATALOG_ID,expanded_targets
from .state_materialize import FORMAT,_entry,verify_materialized_source
from .tables import table,patient_map,partition,matrix

def route(cfg,output,*,cancer_type='all',cancer_list=None):
    if socket.gethostname().split('.')[0]!='149':fail('STATE_EXECUTION_HOST','Program routing must run on 149')
    out=Path(output)
    if out.exists() and any(out.iterdir()):fail('STATE_OUTPUT_EXISTS','Use a new versioned directory')
    out.mkdir(parents=True,exist_ok=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'NATIVE_SCALE_PROGRAM_STATE_ROUTING','cancer_type':cancer_type})
    fm=patient_map(cfg['sources']['patient_map']['path'])
    scope=resolve_cancer_scope({'cancer_type':cancer_type,'cancer_list':cancer_list or []},fm.cancer_id.unique())
    fm=fm.loc[fm.cancer_id.isin(scope['selected_cancers'])].copy()
    src=cfg['sources']['state'];base=table(src['path'])
    if sha(src['path'])!=src['sha256']:fail('STATE_SOURCE_HASH_MISMATCH','Original State source changed')
    verify_materialized_source(src,base,fm,expanded_targets(CATALOG_ID))
    targets=expanded_targets(ROUTED_CATALOG_ID);programs=[t for t in targets if t['source_kind']=='routed_program'];program_ids=[t['id'] for t in programs]
    base=base.set_index('patient_id');rows=[];audits=[];inputs={'base_state':_entry(src['path']),'base_state_manifest':_entry(src['materialization_manifest']),'patient_map':_entry(cfg['sources']['patient_map']['path'])}
    for cancer in scope['selected_cancers']:
        mp=fm.loc[fm.cancer_id.eq(cancer)];patients=sorted(mp.patient_id.unique())
        f,p=partition(cfg['sources']['pathway_activity_root']['path'],cancer)
        f=f.loc[f.pathway_id.isin(program_ids)]
        available=sorted(set(f.pathway_id));a=[]
        values=matrix(f,'pathway_id','activity_score',patients,program_ids,mp,required_entities=available,report=a)
        block=base.loc[patients].copy()
        block=pd.concat([block,pd.DataFrame(values,index=patients,columns=[t['column'] for t in programs])],axis=1)
        block.index.name='patient_id';block=block.reset_index()[['patient_id','cancer_id']+[t['column'] for t in targets]]
        rows.append(block)
        for q in (sorted(p.glob('*.parquet')) if p.is_dir() else [p]):inputs[str(q)]=_entry(q)
        audits.append({'cancer_id':cancer,'patients':len(patients),'programs_registered':len(program_ids),'programs_measured':len(available),'measured_values':int(np.isfinite(values).sum()),'joins':a})
        print('STATE_ROUTED',cancer,len(patients),len(available),flush=True)
    result=pd.concat(rows,ignore_index=True);p=out/'STATE_ROUTED_PATIENT.parquet';result.to_parquet(p,index=False)
    restored=pd.read_parquet(p)
    if not np.array_equal(result[[t['column'] for t in targets]].to_numpy(),restored[[t['column'] for t in targets]].to_numpy(),equal_nan=True):fail('STATE_SERIALIZATION_DRIFT','Native floating point scores changed')
    atomic_json(out/'STATE_ROUTING_AUDIT.json',{'target_host':'149','value_rescaling':False,'programs':program_ids,'cancers':audits,
      'implementation':'R/07_bulk_regulator_activity.R legacy weighted gene z-scores followed by program z-score; not official PROGENy/VIPER package calls'})
    coverage=[]
    for cancer,g in result.groupby('cancer_id'):
        for t in targets:coverage.append({'cancer_id':cancer,'target_id':t['id'],'family':t['family'],'role':t['role'],'patients':len(g),'measured':int(g[t['column']].notna().sum())})
    pd.DataFrame(coverage).to_csv(out/'STATE_COVERAGE.tsv',sep='\t',index=False)
    record={'format':FORMAT,'state':'COMPLETE','target_host':'149','catalog_id':ROUTED_CATALOG_ID,'target_catalog_sha256':digest(targets),
      'cancer_scope':scope,'patients':len(result),'targets':len(targets),'primary_targets':sum(t['role']=='primary' for t in targets),
      'authority_patient_map_sha256':inputs['patient_map']['sha256'],'inputs':inputs,'matrix':_entry(p),'columns':list(result),
      'outputs':{q.name:_entry(q) for q in sorted(out.iterdir()) if q.is_file()},'scores_rescaled':False,
      'missing_policy':'NO_IMPUTATION; UNMEASURED_IS_NA_NOT_ZERO','selection_policy':'EXACT_FROZEN_PATIENT_AUTHORITY; CONFLICTING_DUPLICATES_REJECTED',
      'biological_claim':'Existing observational scores routed to State unchanged; excluded from pathway MOGSA/CFA'}
    receipt=out/'STATE_MATERIALIZATION.json';atomic_json(receipt,record)
    binding={**_entry(p),'columns':len(result.columns),'format':FORMAT,'catalog_id':ROUTED_CATALOG_ID,
      'materialization_manifest':str(receipt.resolve()),'materialization_manifest_sha256':sha(receipt)}
    atomic_json(out/'STATE_SOURCE_BINDING.json',binding);verify_materialized_source(binding,result,fm,targets)
    return binding

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cancer-type','--cancer_type',choices=['all','list'],default='all')
    p.add_argument('--cancer-list','--cancer_list',nargs='+',default=[])
    a=p.parse_args();print(json.dumps(route(read_json(a.config),a.output,cancer_type=a.cancer_type,cancer_list=a.cancer_list),indent=2))

if __name__=='__main__':main()
