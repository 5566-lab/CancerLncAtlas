import os
os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import json,socket,sys,subprocess,hashlib,time
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
WORK=ROOT/'kegg_cancer_redefinition_20261111_r1';REPO=ROOT/'source_integration/repo'
BASE=WORK/'measured_y_definition_r1';MEMBERS=BASE/'member_measurements_r1'
OUT=BASE/'observation_registry_r1';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(REPO))
from cla_engine.common import sha
from cla_engine.measured_y_registry import observation_id,verify_matrix,source_categories,normalize_links,load_target
import pandas as pd
import numpy as np

def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
save(OUT/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'register fixed pathway patient observations','lncrna_expression_read':False,'lncrna_association':False,'training':False,'outer_test_read':False,'script_sha256':sha(__file__)})
tests=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_measured_y_registry.py','tests/test_measured_members.py','tests/test_pathway_conditions.py'],cwd=REPO,text=True,capture_output=True)
save(OUT/'TESTS.json',{'target_host':'149','returncode':tests.returncode,'output':tests.stdout+tests.stderr})
print(tests.stdout,flush=True)
if tests.returncode:raise SystemExit(tests.returncode)

cat=ROOT/'canonical_targets_r9_cancer_evidence_20261111_r6'
cr=json.loads((cat/'COMPLETE.json').read_text())
for name in ['SOURCE_TASK_ROLES.tsv','TARGET_POLICY.tsv','PROTEIN_SCOPE_ID_REDIRECTS.tsv','PATHWAY_GENE_COMPOSITION.tsv.gz','MECHANISM_CONTEXT_COMPONENTS.tsv','GROUP_MASTER.tsv','ALL_GROUP_MEMBERSHIP.tsv']:
    assert sha(cat/name)==cr['files'][name],name
roles=pd.read_csv(cat/'SOURCE_TASK_ROLES.tsv',sep='\t',keep_default_na=False)
redirects=pd.read_csv(cat/'PROTEIN_SCOPE_ID_REDIRECTS.tsv',sep='\t',keep_default_na=False).set_index('previous_canonical_id').canonical_id.to_dict()
collection=json.loads((MEMBERS/'COMPLETE.json').read_text())
for name,h in {**collection['block_receipts'],**collection['case_metadata']}.items():assert sha(MEMBERS/name)==h,name
all_counts=[]
for cancer in ['BRCA','LUAD','COAD']:
    start=time.time();out=OUT/cancer
    if (out/'COMPLETE.json').exists():raise RuntimeError('completed output exists: '+str(out))
    out.mkdir(exist_ok=True)
    source=BASE/'measurements_coding_r3'/cancer/'validation'
    rec=json.loads((source/'COMPLETE.json').read_text())
    for name,h in rec['files'].items():assert sha(source/name)==h,name
    targets=pd.read_csv(source/'TARGETS.tsv',sep='\t',keep_default_na=False)
    assert len(targets)==674 and targets.canonical_id.is_unique
    targets.to_csv(out/'TARGETS.tsv',sep='\t',index=False)
    readouts=pd.read_csv(source/'READOUTS.tsv',sep='\t',keep_default_na=False)
    readouts['canonical_display_roles']=readouts.display_category
    # RPPA's published project score table uses canonical IDs as assay score IDs.
    # Resolve this verified score type explicitly, without a generic ID fallback.
    absent=readouts[~readouts.source_target_id.isin(roles.source_target_id)]
    extra=[]
    known_ids=set(pd.read_csv(cat/'TARGET_POLICY.tsv',sep='\t').canonical_id)
    for row in absent.to_dict('records'):
        meta=json.loads(row['source_metadata_json'])
        assert row['cohort']=='TCGA' and row['layer']=='PROTEIN'
        assert redirects.get(row['source_target_id'],row['source_target_id'])==row['canonical_id'] and row['canonical_id'] in known_ids
        assert meta['score_method']=='RPPA_TRAIN_Z_GENE_GROUP_MEAN' and meta['family']=='pathway'
        extra.append({'source_target_id':row['source_target_id'],'display_category':'Pathway'})
    effective_roles=pd.concat([roles,pd.DataFrame(extra).drop_duplicates()],ignore_index=True)
    readouts['display_category']=source_categories(readouts,effective_roles)
    readouts['category_basis']=np.where(readouts.source_target_id.isin(roles.source_target_id),'SOURCE_TASK_ROLES','verified RPPA gene-group pathway score')
    rb=pd.read_csv(source/'READOUT_BINDINGS.tsv',sep='\t',keep_default_na=False)
    matrices={'patients':{},'blocks':{},'conditions':{}}
    observations=[];links=[];observed_counts={};matrix_masks={};source_map={}
    partitions=json.loads((source/'PATIENT_PARTITIONS.json').read_text())
    for cohort in ['TCGA','CPTAC']:
        part=partitions[cohort]
        assert part['outer_fold']==0 and part['reference_folds']==[2,3,4] and part['query_folds']==[1]
        patients=part['reference_patients']+part['query_patients']
        assert len(patients)==len(set(patients))
        matrices['patients'][cohort]=patients
        cp=MEMBERS/(cohort+'_'+cancer)/'CONDITION_APPLICABILITY.parquet'
        matrices['conditions'][cohort]={'path':str(cp),'sha256':sha(cp)}
    def register(matrix_id,folder,cohort):
        values=pd.read_parquet(folder/'VALUES.parquet');mask=pd.read_parquet(folder/'OBSERVED.parquet')
        patients=matrices['patients'][cohort]
        assert values.index.is_unique and set(values.index)==set(patients)
        assert values.index.equals(mask.index)
        verify_matrix(values.reindex(patients),mask.reindex(patients),patients)
        matrices['blocks'][matrix_id]={'cohort':cohort,'values':{'path':str(folder/'VALUES.parquet'),'sha256':sha(folder/'VALUES.parquet')},'observed':{'path':str(folder/'OBSERVED.parquet'),'sha256':sha(folder/'OBSERVED.parquet')}}
        matrix_masks[matrix_id]=mask.reindex(patients)
        observed_counts[matrix_id]=mask.sum(axis=0).to_dict()
        return values.columns.tolist()
    for (cohort,layer),frame in readouts.groupby(['cohort','layer'],sort=False):
        matrix_id=cohort+'::readout::'+layer
        columns=register(matrix_id,source/cohort/layer,cohort)
        assert len(frame)==len(columns)
        for row in frame.to_dict('records'):
            feature=columns[int(row['column_index'])]
            oid=observation_id(cohort,'readout',layer,row['target_id'])
            source_map[(cohort,layer,row['target_id'])]=oid
            dependency_verified=not row['measurement_layer'].startswith('RNA_') or bool(row['coding_expression_basis_verified'])
            observations.append(dict(observation_id=oid,cohort=cohort,kind='source_readout',layer=layer,
                matrix_id=matrix_id,feature_id=feature,definition_id=row['target_id'],source_target_id=row['source_target_id'],
                display_category=row['display_category'],canonical_display_roles=row['canonical_display_roles'],
                measurement_layer=row['measurement_layer'],method=row['applied_method'],units='source-specific score; method and reference retained',
                definition_json=row['source_metadata_json'],method_variant=row['method_variant'],
                supervision_eligible=dependency_verified,eligibility_basis='verified measurement dependency' if dependency_verified else 'external RNA background lineage not verified',
                patients_measured=int(observed_counts[matrix_id][feature]),independent_loss_enabled=False))
    for row in rb.to_dict('records'):
        links.append(dict(canonical_id=row['output_canonical_id'],observation_id=source_map[(row['cohort'],row['layer'],row['target_id'])],
            evidence_role=row['evidence_role'],condition_required=row['condition_required'],scope_relation=row['scope_relation'],
            routing_basis=row['routing_basis'],molecular_components=row['molecular_components']))
    for relative in collection['block_receipts']:
        folder=(MEMBERS/relative).parent
        r=json.loads((folder/'COMPLETE.json').read_text())
        if r['case'].split('_')[1]!=cancer:continue
        for name,h in r['files'].items():assert sha(folder/name)==h,name
        cohort=r['cohort'];layer=r['layer'];matrix_id=cohort+'::member::'+layer
        columns=register(matrix_id,folder,cohort)
        features=pd.read_csv(folder/'FEATURES.tsv.gz',sep='\t',keep_default_na=False)
        assert features.feature_id.tolist()==columns
        for row in features.to_dict('records'):
            feature=row['feature_id'];oid=observation_id(cohort,'member',layer,feature)
            observations.append(dict(observation_id=oid,cohort=cohort,kind='member_measurement',layer=layer,
                matrix_id=matrix_id,feature_id=feature,definition_id=feature,source_target_id='',display_category='Pathway',canonical_display_roles='',
                measurement_layer=layer,method='original measured member value',units=r['units'],definition_json=row['source_annotation_json'],
                method_variant='SOURCE_MEASUREMENT',supervision_eligible=True,eligibility_basis='verified original measured member',
                patients_measured=int(observed_counts[matrix_id][feature]),independent_loss_enabled=False))
        for row in pd.read_csv(folder/'PATHWAY_MEMBER_BINDINGS.tsv.gz',sep='\t',keep_default_na=False).to_dict('records'):
            links.append(dict(canonical_id=row['canonical_id'],observation_id=observation_id(cohort,'member',layer,row['feature_id']),
                evidence_role=row['measurement_role'],condition_required=row['condition_required'],scope_relation='original_KEGG_member',
                routing_basis='exact molecular annotation',molecular_components=row['matched_member_gene_ids']))
    obs=pd.DataFrame(observations)
    assert obs.observation_id.is_unique
    link=normalize_links(pd.DataFrame(links),obs,targets)
    obs['bound_target_count']=obs.observation_id.map(link[link.is_target_observation].groupby('observation_id').canonical_id.nunique()).fillna(0).astype(int)
    obs.to_csv(out/'OBSERVATIONS.tsv.gz',sep='\t',index=False,compression='gzip')
    link.to_csv(out/'PATHWAY_OBSERVATIONS.tsv.gz',sep='\t',index=False,compression='gzip')
    readouts.to_csv(out/'SOURCE_READOUT_METHODS.tsv.gz',sep='\t',index=False,compression='gzip')
    save(out/'MATRICES.json',matrices);save(out/'PATIENT_PARTITIONS.json',partitions)
    save(out/'INPUT_REFERENCES.json',{'source_readout_receipt':{'path':str(source/'COMPLETE.json'),'sha256':sha(source/'COMPLETE.json')},
       'member_collection_receipt':{'path':str(MEMBERS/'COMPLETE.json'),'sha256':sha(MEMBERS/'COMPLETE.json')},
       'catalog_receipt':{'path':str(cat/'COMPLETE.json'),'sha256':sha(cat/'COMPLETE.json')}})
    # Observational coverage is independent of variance or lncRNA estimability.
    coverage=[];indexed=obs.set_index('observation_id')
    for cohort in ['TCGA','CPTAC']:
        local=link[link.observation_id.isin(set(obs.loc[obs.cohort.eq(cohort),'observation_id']))]
        for cid in targets.canonical_id:
            choices=local[local.canonical_id.eq(cid) & local.is_target_observation]
            meta=indexed.loc[choices.observation_id.unique()]
            for layer in sorted(obs.layer.unique()):
                chosen=meta[meta.layer.eq(layer) & meta.supervision_eligible]
                measured=pd.Series(False,index=matrices['patients'][cohort])
                for mid,g in chosen.groupby('matrix_id'):
                    measured |= matrix_masks[mid][g.feature_id.tolist()].any(axis=1)
                coverage.append(dict(cancer=cancer,cohort=cohort,canonical_id=cid,layer=layer,
                    defined_observations=len(chosen),observations_with_measurement=int(chosen.patients_measured.gt(0).sum()),
                    patients_measured=int(measured.sum()),patients_total=len(measured),measured_fraction=float(measured.mean())))
    pd.DataFrame(coverage).to_csv(out/'COVERAGE.tsv',sep='\t',index=False)
    complete={'target_host':'149','format':'CLA_PATHWAY_PATIENT_OBSERVATION_REGISTRY_V1','cancer':cancer,
        'primary_pathways':477,'state_identities':197,'targets':len(targets),'observations':len(obs),
        'target_bindings':len(link),'matrix_blocks':len(matrices['blocks']),
        'external_RNA_reference_observations':int((~obs.supervision_eligible).sum()),
        'patients':{k:len(v) for k,v in matrices['patients'].items()},'model_training':False,'lncrna_association':False,
        'automatic_independent_losses':False,'representation':'PATHWAY_GROUPED_PATIENT_OBSERVATIONS',
        'files':{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json'}}
    save(out/'COMPLETE.json',complete)
    # Exercise the actual accessor for both cohorts and a real conditional target.
    conditional=targets[targets.condition_required].canonical_id.iloc[0]
    validations=[]
    for cohort in ['TCGA','CPTAC']:
        v,m,a,meta=load_target(out,conditional,cohort)
        v2,m2,a2,_=load_target(out,conditional,cohort,qualified_only=True)
        pd.testing.assert_frame_equal(v,v2)
        assert not m2.any().any() and a.isna().all()
        validations.append({'cohort':cohort,'conditional_target':conditional,'dimensions':len(meta),'measured_values':int(m.to_numpy().sum()),'unknown_condition_does_not_change_values':True})
    save(out/'ACCESSOR_VALIDATION.json',{'target_host':'149','checks':validations})
    all_counts.append(complete)
    print('REGISTRY_COMPLETE',cancer,len(obs),len(link),round(time.time()-start,1),flush=True)
save(OUT/'COMPLETE.json',{'target_host':'149','format':'CLA_PATHWAY_PATIENT_OBSERVATION_COLLECTION_V1','cases':all_counts,
    'catalog':str(cat),'no_lncRNA_expression_or_association':True,'training':False,
    'unresolved_external_RNA_lineage':['ESTIMATE::TumorPurity','panimmune_160_signatures::Wolf::Chemokine12_score'],
    'receipt_hashes':{c['cancer']:sha(OUT/c['cancer']/'COMPLETE.json') for c in all_counts}})
print('COLLECTION_COMPLETE',flush=True)
