from pathlib import Path
import hashlib,json,socket,sys,time
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1'); W=R/'kegg_cancer_redefinition_20261111_r1'
C=R/'canonical_targets_r12_wikipathways_patients_20261114_r1'; B=W/'wiki_patient_integration_20261114_r1/Y_bundle'
O=W/'semantic_score_context_20261119'; REPO=R/'source_integration/repo'
O.mkdir(exist_ok=True)
(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'semantic_score_context_materialization','training':False,'y_recalculated':False}))
sys.path.insert(0,str(REPO))
import pandas as pd
import numpy as np
from cla_engine.semantic_context import compile_routes,fit_reference_scaler
from cla_engine.semantic_catalog import semantic_catalog,observation_binding
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rules_path=REPO/'configs/progeny_semantic_rules.json';rules=json.loads(rules_path.read_text())
primary,scope,review,aliases,lookup=semantic_catalog(C,rules)
assert len(primary)==476
primary.to_csv(O/'PRIMARY_PATHWAYS.tsv',sep='\t',index=False)
scope.to_csv(O/'SEMANTIC_SCOPE_RELATIONS.tsv',sep='\t',index=False)
review.to_csv(O/'PROGENY_REVIEWED_ROUTES.tsv',sep='\t',index=False)
summaries=[];inputs={str(rules_path):sha(rules_path)}
for name in ['DEFINITION_TO_IDENTITY.tsv','FUNCTIONAL_SCOPE_DECISIONS.tsv','PRIMARY_PATHWAY_TARGETS.tsv','KEGG_SOURCE_SCOPE_DECISIONS.tsv','SOURCE_ID_BINDINGS.tsv','PTM_READOUT_BINDINGS.tsv']:
 inputs[str(C/name)]=sha(C/name)
for cohort in ['TCGA','CPTAC']:
 for cancer in ['BRCA','LUAD','COAD']:
  case=cohort+'_'+cancer;folder=B/'cohorts'/case;dest=O/'cohorts'/case;dest.mkdir(parents=True,exist_ok=True)
  print('BEGIN',case,flush=True)
  receipt=json.loads((folder/'COMPLETE.json').read_text())
  names=['OBSERVATIONS.tsv.gz','TARGET_OBSERVATIONS.tsv.gz','Y_VALUES.parquet','Y_OBSERVED.parquet','TARGET_APPLICABILITY.parquet']
  before={n:sha(folder/n) for n in names}
  assert all(before[n]==receipt['files'][n] for n in names)
  inputs.update({str(folder/n):v for n,v in before.items()})
  obs=pd.read_csv(folder/'OBSERVATIONS.tsv.gz',sep='\t',keep_default_na=False)
  obs=obs[obs.kind.eq('source_readout')].copy()
  bindings=pd.DataFrame([observation_binding(r,aliases,lookup) for r in obs.to_dict('records')])
  values=pd.read_parquet(folder/'Y_VALUES.parquet',columns=obs.observation_id.tolist())
  measured=pd.read_parquet(folder/'Y_OBSERVED.parquet',columns=obs.observation_id.tolist())
  values=values.where(measured.astype(bool))
  pc=W/'measured_y_definition_r1/protein_coding_readouts'/case/'validation'
  meta=pd.read_csv(pc/'READOUTS.tsv',sep='\t',keep_default_na=False)
  pro=meta[meta.source_score_id.str.contains('PROGENY:',regex=False)]
  assert len(pro)==14
  extra=pd.read_parquet(pc/'VALUES.parquet',columns=pro.readout_id.tolist())
  extra=extra.reindex(values.index)
  inputs[str(pc/'VALUES.parquet')]=sha(pc/'VALUES.parquet');inputs[str(pc/'READOUTS.tsv')]=sha(pc/'READOUTS.tsv')
  new=[]
  for r in pro.to_dict('records'):
   scopeid='PROGENY:'+r['source_score_id'].split('PROGENY:',1)[1]
   present=bindings[bindings.source_scope.eq(scopeid)]
   if len(present)>1: raise ValueError('DUPLICATE_PROGENY_OBSERVATIONS')
   if len(present)==1:
    # One original observation is reused, not a second alias score.
    sid=present.iloc[0].score_id
    assert np.allclose(values[sid],extra[r['readout_id']],equal_nan=True)
    continue
   sid='PROGENY::'+r['source_score_id'].split('PROGENY:',1)[1]
   values[sid]=extra[r['readout_id']]
   new.append(dict(score_id=sid,source_scope=scopeid,input_layers='["RNA"]',
       lineage_ids=json.dumps([sid,r['readout_id'],r['source_score_id']]),
       binding_evidence='exact PROGENY program ID from existing coding-only score metadata',
       source_method=r.get('method','FIXED_PROGENY_TOP100_WEIGHTS_CODING_MEMBERS_SOURCE_Z'),
       source_feature_id=r['readout_id'],source_layer='STATE',input_modality='RNA',
       independent_loss_enabled=False,copy_to_target_y=False))
  bindings=pd.concat([bindings,pd.DataFrame(new)],ignore_index=True).sort_values('score_id').reset_index(drop=True)
  values=values[bindings.score_id.tolist()]
  routes=compile_routes(bindings,scope,primary.canonical_id.tolist())
  partition=json.loads((folder/'PATIENT_PARTITION.json').read_text());inputs[str(folder/'PATIENT_PARTITION.json')]=sha(folder/'PATIENT_PARTITION.json')
  scaler=fit_reference_scaler(values,partition['reference_patients'],partition['query_patients']);scaler.index.name='score_id'
  bindings['mapped_main_pathways']=bindings.score_id.map(routes.groupby('score_id').target_id.nunique()).fillna(0).astype(int)
  bindings['patients_measured']=np.isfinite(values).sum().reindex(bindings.score_id).to_numpy()
  bindings.to_csv(dest/'SCORE_BINDINGS.tsv',sep='\t',index=False)
  routes.to_csv(dest/'SCORE_PATHWAY_CONTEXT.tsv',sep='\t',index=False)
  scaler.to_csv(dest/'REFERENCE_SCALER.tsv',sep='\t')
  values.to_parquet(dest/'SOURCE_VALUES.parquet')
  (dest/'PATIENT_PARTITION.json').write_text(json.dumps(partition,indent=2))
  assert all(sha(folder/n)==before[n] for n in names)
  summary=dict(case=case,patients=len(values),source_scores=len(bindings),
    routed_scores=int(routes.score_id.nunique()),routed_main_pathways=int(routes.target_id.nunique()),
    context_edges=len(routes),progeny_scores=int(bindings.source_scope.str.startswith('PROGENY:').sum()),
    progeny_with_routes=int(((bindings.mapped_main_pathways>0)&bindings.source_scope.str.startswith('PROGENY:')).sum()),
    source_y_unchanged=True,reference_patients=len(partition['reference_patients']),query_patients=len(partition['query_patients']))
  files={n:sha(dest/n) for n in ['SCORE_BINDINGS.tsv','SCORE_PATHWAY_CONTEXT.tsv','REFERENCE_SCALER.tsv','SOURCE_VALUES.parquet','PATIENT_PARTITION.json']}
  (dest/'COMPLETE.json').write_text(json.dumps(dict(format='SEMANTIC_SCORE_CONTEXT_V1',target_host='149',files=files,summary=summary),indent=2))
  summaries.append(summary);print('DONE',json.dumps(summary),flush=True)
pd.DataFrame(summaries).to_csv(O/'COHORT_SUMMARY.tsv',sep='\t',index=False)
(O/'INPUT_RECEIPTS.json').write_text(json.dumps(inputs,indent=2))
(O/'CONTRACT.json').write_text(json.dumps(dict(format='SEMANTIC_SCORE_CONTEXT_V1',main_pathway_count=476,
 fixed_y_root=str(B),new_gene_sets=0,new_scores_computed=0,automatic_auxiliary_losses=0,
 event_conditions_unchanged=True,source_score_role='optional_input_context',
 m0_entry='cla_engine.measured_context_models.measured_m0_design',
 hhgt_entry='cla_engine.measured_context_models.measured_hhgt_graph',
 target_exclusion='same_measurement_layer_and_lineage_before_consumption',
 unknown_dependencies='kept_but_not_input_enabled',training_started=False,
 task_weights={'pathway':1.,'state':.5}),indent=2))
print('COMPLETE',str(O),flush=True)
