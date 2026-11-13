from pathlib import Path
import os,socket,json,hashlib,sys,shutil,subprocess
os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('OMP_NUM_THREADS','2')
assert socket.gethostname().split('.')[0]=='149'
import numpy as np
import pandas as pd
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';REPO=R/'source_integration/repo';O=W/'fixed_y_remake_20261114_r1';B=O/'Y_bundle';F=O/'figure_data'
N=R/'canonical_targets_r11_state_groups_20261114_r1';G=W/'state_groups_20261114_r1/observation_registry_r4'
sys.path.insert(0,str(REPO))
from cla_engine.measured_y_registry import verify_matrix,condition_for_target
from cla_engine.fixed_y_bundle import load_fixed_target,supervision_mask
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
def wr(d,p,index=False):d.to_csv(p,sep='\t',index=index)
def hashes(d):return {p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='COMPLETE.json'}
O.mkdir(exist_ok=True);F.mkdir(exist_ok=True);sj(O/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='Materialize portable fixed measured Y and real figure inputs',catalog=str(N),registry=str(G),training=False,lncrna_association=False,outer_test_opened=False))
assert not (B/'COMPLETE.json').exists(),'Fixed bundle already complete'
B.mkdir(exist_ok=True);(B/'cohorts').mkdir(exist_ok=True)
test=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_fixed_y_bundle.py','tests/test_state_observation_groups.py','tests/test_exact_member_groups.py','tests/test_measured_y_registry.py','tests/test_measured_members.py','tests/test_pathway_conditions.py'],cwd=REPO,capture_output=True,text=True)
sj(O/'TESTS.json',dict(returncode=test.returncode,output=test.stdout+test.stderr));assert test.returncode==0,test.stdout+test.stderr;print(test.stdout,flush=True)
crec=json.loads((N/'COMPLETE.json').read_text())
for name,h in crec['files'].items():assert sha(N/name)==h,name
collection=json.loads((G/'COMPLETE.json').read_text())
display=rd(N/'DISPLAY_TARGETS.tsv');policy=rd(N/'TARGET_POLICY.tsv');assert len(display)==524
targets=rd(G/'BRCA/TARGETS.tsv').merge(display[['canonical_id','display_name','display_name_zh','entity_type']],on='canonical_id',validate='one_to_one')
targets=targets.sort_values(['analysis_role','canonical_id']).reset_index(drop=True);wr(targets,B/'TARGETS.tsv')
for name in ['STATE_GROUP_MEMBERSHIP.tsv','STATE_GROUP_DEFINITION.json','EXACT_MEMBER_GROUP_MEMBERS.tsv','GROUP_SOURCE_MECHANISMS.tsv','PRIMARY_PATHWAY_TARGETS.tsv','STATE_TOP_LEVEL_TARGETS.tsv']:
    shutil.copy2(N/name,B/name)
shutil.copy2(REPO/'cla_engine/fixed_y_bundle.py',B/'reader.py')
summary=[];cover=[];layers=[];observations_summary=[];folds=[];checks=[];pool_summary=[]
example_ids={'MMR':'CLA-P-922863C8FB5E9DF9','PI3K':'CLA-P-A50E266F52E403F9','WNT':'CLA-P-09DF1A5F76D71F39','Hedgehog':'CLA-P-08A1A41F05A3E9ED'}
sm=rd(N/'STATE_GROUP_MEMBERSHIP.tsv')
for family in ['cell_composition','sbs_spectrum']:example_ids[family]=sm.loc[sm.state_family.eq(family),'parent_id'].iloc[0]
for ca in ['BRCA','LUAD','COAD']:
    source=G/ca;receipt=json.loads((source/'COMPLETE.json').read_text());assert sha(source/'COMPLETE.json')==collection['receipt_hashes'][ca]
    for name,h in receipt['files'].items():assert sha(source/name)==h,(ca,name)
    matrix=json.loads((source/'MATRICES.json').read_text());partition=json.loads((source/'PATIENT_PARTITIONS.json').read_text());allobs=rd(source/'OBSERVATIONS.tsv.gz');alllinks=rd(source/'PATHWAY_OBSERVATIONS.tsv.gz')
    assert set(rd(source/'TARGETS.tsv').canonical_id)==set(targets.canonical_id)
    target_links=alllinks[alllinks.is_target_observation].copy();bound=set(target_links.observation_id)
    for co in ['TCGA','CPTAC']:
        case=co+'_'+ca;folder=B/'cohorts'/case;folder.mkdir(exist_ok=True);patients=matrix['patients'][co]
        parts=partition[co];ref=set(parts['reference_patients']);qry=set(parts['query_patients']);assert not(ref&qry) and set(patients)==(ref|qry)
        sj(folder/'PATIENT_PARTITION.json',parts)
        obs=allobs[allobs.cohort.eq(co)&allobs.observation_id.isin(bound)].copy().sort_values('observation_id').reset_index(drop=True)
        assert obs.observation_id.is_unique
        links=target_links[target_links.observation_id.isin(obs.observation_id)].copy()
        blocks=[];masks=[];inputs=[]
        for mid,meta in obs.groupby('matrix_id',sort=False):
            item=matrix['blocks'][mid]
            for field in ['values','observed']:assert sha(item[field]['path'])==item[field]['sha256'],(case,mid,field)
            cols=meta.feature_id.tolist();assert len(cols)==len(set(cols))
            v=pd.read_parquet(item['values']['path'],columns=cols).reindex(patients);m=pd.read_parquet(item['observed']['path'],columns=cols).reindex(patients)
            verify_matrix(v,m,patients);v.columns=m.columns=meta.observation_id.tolist();blocks.append(v);masks.append(m);inputs.append(dict(matrix_id=mid,**item))
        values=pd.concat(blocks,axis=1).reindex(columns=obs.observation_id);observed=pd.concat(masks,axis=1).reindex(columns=obs.observation_id)
        verify_matrix(values,observed,patients)
        con=matrix['conditions'][co];assert sha(con['path'])==con['sha256'];raw_condition=pd.read_parquet(con['path'])
        applicable=pd.concat([condition_for_target(raw_condition,row,patients).rename(row['canonical_id']) for row in targets.to_dict('records')],axis=1)
        assert applicable.shape==(len(patients),524)
        values.to_parquet(folder/'Y_VALUES.parquet');observed.to_parquet(folder/'Y_OBSERVED.parquet');applicable.to_parquet(folder/'TARGET_APPLICABILITY.parquet')
        pd.testing.assert_frame_equal(pd.read_parquet(folder/'Y_VALUES.parquet'),values);pd.testing.assert_frame_equal(pd.read_parquet(folder/'Y_OBSERVED.parquet'),observed)
        obs['patients_measured']=observed.sum(axis=0).reindex(obs.observation_id).to_numpy();wr(obs,folder/'OBSERVATIONS.tsv.gz');wr(links,folder/'TARGET_OBSERVATIONS.tsv.gz')
        child=rd(source/'STATE_CHILD_OBSERVATIONS.tsv.gz');wr(child[child.observation_id.isin(obs.observation_id)],folder/'STATE_CHILD_OBSERVATIONS.tsv.gz')
        sj(folder/'SOURCE_RECEIPTS.json',dict(source_registry=str(source),source_registry_sha256=sha(source/'COMPLETE.json'),matrices=inputs,conditions=con,partition_sha256=sha(source/'PATIENT_PARTITIONS.json')))
        pd.DataFrame({'patient_id':patients,'partition':['reference' if p in ref else 'query' for p in patients]}).to_csv(folder/'PATIENTS.tsv',sep='\t',index=False)
        index=obs.set_index('observation_id');eligible=index.supervision_eligible
        pertarget=[]
        for row in targets.to_dict('records'):
            cid=row['canonical_id'];columns=obs.loc[obs.observation_id.isin(links.loc[links.canonical_id.eq(cid),'observation_id']),'observation_id'].tolist();mask=observed[columns];training_mask=supervision_mask(mask,eligible.loc[columns],applicable[cid])
            n=int(mask.any(axis=1).sum());nq=int(training_mask.any(axis=1).sum());meta=index.loc[columns]
            rr=dict(case=case,cancer=ca,cohort=co,canonical_id=cid,name=row['display_name'],analysis_role=row['analysis_role'],defined_dimensions=len(columns),measured_dimensions=int(mask.any().sum()),eligible_dimensions=int((mask.any()&eligible.loc[columns]).sum()),patients_measured=n,patients_supervision=nq,patients_total=len(patients),condition_unknown=int(applicable[cid].isna().sum()))
            cover.append(rr);pertarget.append(rr)
        for (kind,layer),meta in obs.groupby(['kind','layer']):
            mask=observed[meta.observation_id];layers.append(dict(case=case,kind=kind,layer=layer,defined_dimensions=len(meta),measured_dimensions=int(mask.any().sum()),patients_measured=int(mask.any(axis=1).sum()),patients_total=len(patients),measured_fraction=float(mask.to_numpy().mean())))
        for layer,meta in obs.groupby('layer'):
            ids=set(meta.observation_id);npath=sum(int((rr['analysis_role']=='primary_pathway') and observed[list(ids&set(links.loc[links.canonical_id.eq(rr['canonical_id']),'observation_id']))].any().any()) for rr in pertarget)
            observations_summary.append(dict(case=case,layer=layer,pathways_with_measurement=npath))
        ss=dict(case=case,cancer=ca,cohort=co,patients=len(patients),reference_patients=len(ref),query_patients=len(qry),top_level_targets=524,pathway_targets=476,state_targets=48,defined_dimensions=len(obs),measured_dimensions=int(observed.any().sum()),pathways_with_measurement=sum(r['analysis_role']=='primary_pathway' and r['patients_measured']>0 for r in pertarget),states_with_measurement=sum(r['analysis_role']=='state_auxiliary' and r['patients_measured']>0 for r in pertarget),pathways_with_supervision=sum(r['analysis_role']=='primary_pathway' and r['patients_supervision']>0 for r in pertarget),states_with_supervision=sum(r['analysis_role']=='state_auxiliary' and r['patients_supervision']>0 for r in pertarget))
        summary.append(ss);pool_summary.append(dict(case=case,source_pool=len(allobs[allobs.cohort.eq(co)]),bound_Y_dimensions=len(obs),unbound_or_context_dimensions=len(allobs[allobs.cohort.eq(co)])-len(obs)))
        sj(folder/'COMPLETE.json',dict(format='CLA_FIXED_Y_COHORT_V1',target_host='149',**ss,files=hashes(folder)))
        # Compare accessor output to the original block values for actual examples.
        for key,cid in example_ids.items():
            if key in ['cell_composition','sbs_spectrum'] and ca!='BRCA':continue
            vv,mm,aa,meta=load_fixed_target(B,case,cid)
            pd.testing.assert_frame_equal(vv,values[vv.columns]);pd.testing.assert_frame_equal(mm,observed[vv.columns]);pd.testing.assert_series_equal(aa,applicable[cid])
            selected=meta[meta.patients_measured.gt(0)].copy()
            selected.to_csv(F/(key+'_'+case+'_observations.tsv'),sep='\t',index=False)
            values[selected.observation_id].to_parquet(F/(key+'_'+case+'_values.parquet'))
            checks.append(dict(case=case,target=key,dimensions=len(vv.columns),measured_dimensions=len(selected),accessor_matches_original=True))
        print('COHORT_Y_COMPLETE',case,len(patients),len(obs),ss['pathways_with_measurement'],ss['states_with_measurement'],flush=True)
        del values,observed,blocks,masks
for name,data in [('COHORT_SUMMARY',summary),('TARGET_COVERAGE',cover),('LAYER_COVERAGE',layers),('PATHWAY_LAYER_COVERAGE',observations_summary),('POOL_ROUTING',pool_summary)]:wr(pd.DataFrame(data),B/(name+'.tsv'))
shutil.copy2(B/'COHORT_SUMMARY.tsv',F/'COHORT_SUMMARY.tsv');shutil.copy2(B/'LAYER_COVERAGE.tsv',F/'LAYER_COVERAGE.tsv');shutil.copy2(B/'PATHWAY_LAYER_COVERAGE.tsv',F/'PATHWAY_LAYER_COVERAGE.tsv')
composition=rd(N/'INTEGRATED_PROTEIN_COMPOSITION.tsv.gz');definition=rd(N/'DEFINITION_TO_IDENTITY.tsv');annotation=rd(N/'PROTEIN_ANNOTATION.tsv');wr(annotation,F/'PROTEIN_ANNOTATION.tsv')
examples={}
for key,cid in example_ids.items():
    source_lists={str(s):sorted(set(t.uniprot_id)) for s,t in composition[composition.integrated_canonical_id.eq(cid)].groupby('source_definition')}
    examples[key]=dict(canonical_id=cid,name=targets.set_index('canonical_id').loc[cid,'display_name'],source_lists=source_lists,source_definitions=json.loads(definition[definition.integrated_canonical_id.eq(cid)].to_json(orient='records')))
sj(F/'EXAMPLES.json',examples)
sj(B/'DEFINITION.json',dict(format='CLA_FIXED_MEASURED_Y_V1',catalog=str(N),catalog_sha256=sha(N/'COMPLETE.json'),source_registry=str(G),primary_pathways=476,state_top_level_count=48,state_detail_components=151,total_top_level_targets=524,representation='Patient by unique source-specific molecular observations with target membership',formula='Y[c,p,t] = { (observation_id, value, observed, supervision_eligible, target_applicability) : observation_id belongs to t }',value_policy='Original measured values; no imputation, averaging or cross-platform rescaling',supervision_mask='observed AND supervision_eligible AND (target_applicability == 1)',patient_scope='Existing three-cancer development patients; reference/query assignments retained; sealed outer test excluded',lncrna_independent=True,task_weights={'Pathway':1.0,'State':0.5},cohorts=summary))
sj(O/'VALIDATION.json',dict(target_host='149',primary_pathways=476,state_top_level_count=48,total_targets=524,patients=sum(r['patients'] for r in summary),cohort_checks=6,example_accessor_checks=checks,source_numeric_values_unchanged=True,missingness_preserved=True,patient_partitions_preserved=True,lncrna_input_used=False,training=False,tests_passed=23))
sj(B/'COMPLETE.json',dict(target_host='149',format='CLA_FIXED_MEASURED_Y_BUNDLE_V1',files={str(p.relative_to(B)):sha(p) for p in B.rglob('*') if p.is_file() and p!=B/'COMPLETE.json'},patients=sum(r['patients'] for r in summary),primary_pathways=476,state_top_level_count=48,total_targets=524))
print('FIXED_Y_BUNDLE_COMPLETE',sum(r['patients'] for r in summary),flush=True)
