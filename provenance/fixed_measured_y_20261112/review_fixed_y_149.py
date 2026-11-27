from pathlib import Path
import json, socket, sys, subprocess
assert socket.gethostname().split('.')[0] == '149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
REPO=ROOT/'source_integration/repo'; WORK=ROOT/'kegg_cancer_redefinition_20261111_r1'
BASE=WORK/'measured_y_definition_r1'; REG=BASE/'observation_registry_r2'
CAT=ROOT/'canonical_targets_r9_cancer_evidence_20261111_r6'
OUT=BASE/'final_definition_review_r1'; OUT.mkdir(exist_ok=True)
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
save(OUT/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='verify fixed measured Y definitions and all numeric blocks',lncrna_association=False,training=False,outer_test_opened=False))
sys.path.insert(0,str(REPO));sys.path.append('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps')
import pandas as pd
import numpy as np
from cla_engine.common import sha
from cla_engine.measured_y_registry import verify_matrix, normalize_links, load_target, condition_for_target
def read(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
scope=read(CAT/'KEGG_SOURCE_SCOPE_DECISIONS.tsv'); primary=read(CAT/'PRIMARY_PATHWAY_TARGETS.tsv'); roles=read(CAT/'SOURCE_TASK_ROLES.tsv')
assert len(scope)==658 and len(primary)==477 and primary.canonical_id.is_unique
assert int(scope.cancer_relevant.sum())==477
assert scope.loc[scope.cancer_relevant,'cancer_evidence'].str.len().gt(0).all()
assert scope.loc[scope.cancer_relevant,'cancer_rationale'].str.len().gt(0).all()
program=roles[roles.source_target_id.str.startswith(('PROGENY:','DOROTHEA:','KINASE-'))]
assert len(program)>0 and program.display_category.eq('Pathway').all()
counts={'inventory':len(scope),'primary_pathways':len(primary),'excluded':int((~scope.cancer_relevant).sum()),'evidence_basis':scope[scope.cancer_relevant].cancer_evidence_basis.value_counts().to_dict(), 'network_type':scope[scope.cancer_relevant].network_type.value_counts().to_dict(),'program_source_rows_checked':len(program),'id_redirect_rows':len(read(CAT/'PROTEIN_SCOPE_ID_REDIRECTS.tsv'))}
save(OUT/'CATALOG_REVIEW.json',counts)
tests=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_coding_fixed_signature.py','tests/test_measured_y_registry.py','tests/test_measured_members.py','tests/test_pathway_conditions.py'],cwd=REPO,capture_output=True,text=True)
save(OUT/'TESTS.json',dict(returncode=tests.returncode,output=tests.stdout+tests.stderr,target_host='149'));assert tests.returncode==0,tests.stdout+tests.stderr
print(tests.stdout,flush=True)
collection=json.loads((REG/'COMPLETE.json').read_text()); allcases=[]; coverage=[]
for cancer in ['BRCA','LUAD','COAD']:
    folder=REG/cancer; receipt=json.loads((folder/'COMPLETE.json').read_text())
    assert sha(folder/'COMPLETE.json')==collection['receipt_hashes'][cancer]
    for name,h in receipt['files'].items():assert sha(folder/name)==h,(cancer,name)
    target=read(folder/'TARGETS.tsv');obs=read(folder/'OBSERVATIONS.tsv.gz');links=read(folder/'PATHWAY_OBSERVATIONS.tsv.gz')
    assert target.canonical_id.is_unique and obs.observation_id.is_unique
    assert len(target)==674 and set(primary.canonical_id)<=set(target.canonical_id)
    normalized=normalize_links(links,obs,target)
    assert len(normalized)==len(links) and np.array_equal(normalized.is_target_observation,links.is_target_observation)
    assert not obs.independent_loss_enabled.any() and not links.independent_loss_enabled.any()
    matrices=json.loads((folder/'MATRICES.json').read_text());parts=json.loads((folder/'PATIENT_PARTITIONS.json').read_text())
    blocks=[]
    for mid,meta in matrices['blocks'].items():
        for key in ['values','observed']:assert sha(meta[key]['path'])==meta[key]['sha256']
        v=pd.read_parquet(meta['values']['path']);m=pd.read_parquet(meta['observed']['path'])
        patients=matrices['patients'][meta['cohort']]
        assert set(v.index)==set(patients)
        v=v.reindex(patients);m=m.reindex(patients);verify_matrix(v,m,patients)
        sub=obs[obs.matrix_id.eq(mid)];assert set(sub.feature_id)<=set(v.columns)
        assert (sub.set_index('feature_id').patients_measured==m[sub.feature_id].sum()).all()
        blocks.append(dict(matrix_id=mid,patients=len(v),features=v.shape[1],finite_values=int(m.to_numpy().sum())))
    for cohort,patients in matrices['patients'].items():
        part=parts[cohort];assert part['outer_fold']==0 and part['reference_folds']==[2,3,4] and part['query_folds']==[1]
        assert set(part['reference_patients']).isdisjoint(part['query_patients'])
        assert set(patients)==set(part['reference_patients'])|set(part['query_patients'])
        info=matrices['conditions'][cohort];assert sha(info['path'])==info['sha256']
        cond=pd.read_parquet(info['path']);assert set(cond.columns)==set(primary.canonical_id)
        for row in target.to_dict('records'):condition_for_target(cond,row,patients)
        conditional=target[target.condition_required].canonical_id
        assert cond[conditional].isna().all().all()
        reference=set(primary.canonical_id)-set(conditional)
        assert cond[list(reference)].eq(1).all().all()
        selected=links[links.is_target_observation].merge(obs,on='observation_id',validate='many_to_one')
        selected=selected[selected.cohort.eq(cohort)&selected.supervision_eligible&selected.patients_measured.gt(0)]
        for layer,sub in selected.groupby('layer'):
            coverage.append(dict(cancer=cancer,cohort=cohort,layer=layer,primary_pathways=len(set(sub.canonical_id)&set(primary.canonical_id)),unique_measured_observations=sub.observation_id.nunique()))
        for cid in [next(iter(reference)),conditional.iloc[0],'CLA-S-4F0486ABF1B8D70A']:
            v,m,a,metadata=load_target(folder,cid,cohort)
            assert v.shape==m.shape and list(v.index)==patients
            assert metadata.supervision_eligible.all()
            if cid in set(conditional):
                qv,qm,_,_=load_target(folder,cid,cohort,qualified_only=True)
                pd.testing.assert_frame_equal(v,qv);assert not qm.to_numpy().any()
    allcases.append(dict(cancer=cancer,observations=len(obs),bindings=len(links),blocks=blocks,patients={k:len(v) for k,v in matrices['patients'].items()},external_references=obs.loc[~obs.supervision_eligible,['cohort','source_target_id','eligibility_basis']].to_dict('records')))
    print('VERIFIED_ALL_BLOCKS',cancer,len(blocks),flush=True)
pd.DataFrame(coverage).to_csv(OUT/'OBSERVED_COVERAGE.tsv',sep='\t',index=False)
# Audit downloaded genomics as actual evidence of resolution, without interpreting a gene flag as a specific event.
G=Path('/dsk2/user/dengsc/cla_graph_ridge_20261105_r1'); specs=json.loads((G/'DOWNLOAD_SPEC.json').read_text()); genomic=[]
for spec in specs:
    suffix='.tsv.gz' if spec['cancer']=='COAD' and spec['layer']=='CNV' else '.tsv'
    p=G/'raw'/(spec['cancer']+'_'+spec['layer']+suffix)
    head=pd.read_csv(p,sep='\t',nrows=3)
    genomic.append(dict(cancer=spec['cancer'],layer=spec['layer'],path=str(p),sha256=sha(p),definition=spec['description'],source=spec['url'],columns=len(head.columns),format='gene_by_sample',qualified_variant_consequence=False))
save(OUT/'CONDITION_EVIDENCE_REVIEW.json',dict(target_host='149',raw_genomic_files=genomic,pinned_clinical_review=str(BASE/'PINNED_RAW_CLINICAL_FIELD_REVIEW.json'),decision='Gene-level mutation flags and continuous CNV ratios remain source observations; exact activation, loss, fusion, pathogen and exposure applicability is unknown without matching evidence.',unknown_is_negative=False,member_values_deleted=False))
save(OUT/'EXTERNAL_RNA_POLICY.json',dict(target_host='149',TLS12_coding_definition=str(BASE/'tls12_coding_r1/DEFINITION.json'),published_TLS12='Retained as source reference; coding-background score is a distinct numerical version.',ESTIMATE_TumorPurity='Retained as external reference, excluded from default Y; exact original background not established. Recomputed ESTIMATE scores are not relabelled as calibrated purity fractions.',changes_to_biological_identities=False))
save(OUT/'MEASURED_Y_VALIDATION.json',dict(target_host='149',registry=str(REG),registry_sha256=sha(REG/'COMPLETE.json'),catalog=counts,cases=allcases,all_numeric_blocks_verified=True,model_training=False,lncrna_association=False,outer_test_opened=False))
print('FIXED_Y_REVIEW_COMPLETE',flush=True)
