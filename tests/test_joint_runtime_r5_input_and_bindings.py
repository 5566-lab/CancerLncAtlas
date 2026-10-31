"""Regression tests use source-shaped fixtures, not biological conclusions.
No mocked graph backbone is used to claim real PyG training.
"""
from pathlib import Path
import copy,json,shutil,tarfile
import numpy as np
import pandas as pd
import pytest
import torch
from cla_engine.common import Stop,atomic_json,atomic_npz,sha,digest,read_json,commit_manifest,code_digest
from cla_engine.identity import detection_eligibility
from cla_engine.graph_contracts import membership_join,normalize_ppi,normalize_binding
from cla_engine.tables import table,matrix,cnv
from cla_engine.lineage import validate_prepared,verify_sealed,verify_completion,sign_file,verify_signed_file,FIELDS
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def source_shaped_prepared(tmp_path_factory):
    from cla_engine.fixture import make_fixture
    from cla_engine.prepare import prepare
    root=tmp_path_factory.mktemp('real_namespace_shape');cfg=make_fixture(root/'source')
    mapping={f'G{i}':f'ENSG{i+1:011d}' for i in range(6)}
    for role in ('exact_membership','weighted_membership','protein_gene'):
        p=Path(cfg['sources'][role]['path']);f=pd.read_csv(p,sep='\t');f['gene_id']=f.gene_id.map(mapping)
        if role!='exact_membership':f['gene_id']='GENE:'+f.gene_id
        if role=='weighted_membership':
            f['mapping_status']='mapped'
            f=pd.concat([f,pd.DataFrame([{'pathway_id':'P0','gene_id':None,'weight':np.nan,'mapping_status':'unmapped'}])],ignore_index=True)
        f.to_csv(p,sep='\t',index=False);cfg['sources'][role]['sha256']=sha(p)
    for p in Path(cfg['sources']['gene_expression_root']['path']).glob('*.tsv'):
        f=pd.read_csv(p,sep='\t');f['gene_id']='GENE:'+f.gene_id.map(mapping);f.to_csv(p,sep='\t',index=False)
    # Duplicated detection source rows must not increase the cancer count.
    p=Path(cfg['sources']['detection']['path']);f=pd.read_csv(p,sep='\t');pd.concat([f,f.iloc[:5]],ignore_index=True).to_csv(p,sep='\t',index=False);cfg['sources']['detection']['sha256']=sha(p)
    prepared=prepare(cfg,root/'prepared',0,'diagnostic');validate_prepared(prepared)
    return root,prepared,cfg

def test_real_namespace_path_is_connected(source_shaped_prepared):
    root,p,cfg=source_shaped_prepared;spec=read_json(p/'SPEC.json')
    assert all(g.startswith('ENSG') for g in spec['nodes']['gene'])
    with np.load(p/'views/C1__validation/graph.npz') as g:
        assert g['edge::protein|encoded_by_positive|gene'].shape[1]>0
        assert np.any(g['x::gene'][:,2]>0)
        assert np.isfinite(g['x::gene']).all()
    assert read_json(p/'MEMBERSHIP_ALIGNMENT.json')['unmapped_source_rows_excluded']==1
    assert read_json(p/'GRAPH_SIZE.json')['diagnostic_is_formal_memory_estimate'] is False

def test_real_shaped_prepared_sealed_binding_passes(source_shaped_prepared):
    _,p,_=source_shaped_prepared
    r=verify_sealed(p,p.parent/(p.name+'__sealed_test'))
    assert len(r['files'])==6
    assert all(v['outer_patient_hash'] for v in r['files'].values())

@pytest.mark.parametrize('change',['fold','prepared_manifest_sha256','binding_sha256'])
def test_sealed_wrong_binding_fails(source_shaped_prepared,tmp_path,change):
    _,p,_=source_shaped_prepared;sealed=tmp_path/'sealed';shutil.copytree(p.parent/(p.name+'__sealed_test'),sealed)
    meta=read_json(sealed/'MANIFEST.json');meta['metadata'][change]=4 if change=='fold' else 'f'*64;atomic_json(sealed/'MANIFEST.json',meta)
    with pytest.raises(Stop,match=change):verify_sealed(p,sealed)

def test_sealed_same_length_replacement_is_rejected(source_shaped_prepared,tmp_path):
    _,p,_=source_shaped_prepared;sealed=tmp_path/'sealed';shutil.copytree(p.parent/(p.name+'__sealed_test'),sealed)
    fp=next(sealed.glob('*.npz'))
    with np.load(fp) as z:a={k:z[k] for k in z.files}
    a['label']=1-a['label'];atomic_npz(fp,**a)
    meta=read_json(sealed/'MANIFEST.json')['metadata'];commit_manifest(sealed,meta)
    with pytest.raises(Stop) as e:verify_sealed(p,sealed)
    assert e.value.code=='SEALED_LABEL_HASH_MISMATCH'

def test_reordered_candidate_view_rejected_even_rehashed(source_shaped_prepared,tmp_path):
    _,p,_=source_shaped_prepared;q=tmp_path/'copy';shutil.copytree(p,q)
    fp=q/'views/C1__train2/pathway.npz'
    with np.load(fp) as z:a={k:z[k].copy() for k in z.files}
    for k in a:a[k]=a[k][::-1]
    atomic_npz(fp,**a);commit_manifest(q,read_json(q/'MANIFEST.json')['metadata'])
    with pytest.raises(Stop) as e:validate_prepared(q)
    assert e.value.code=='CANDIDATE_VIEW_MISMATCH'

def test_patient_view_hash_changed_rejected(source_shaped_prepared,tmp_path):
    _,p,_=source_shaped_prepared;q=tmp_path/'copy';shutil.copytree(p,q);f=q/'views/C1__validation/PROVENANCE.json'
    a=read_json(f);a['feature_patient_hash']='a'*64;atomic_json(f,a);commit_manifest(q,read_json(q/'MANIFEST.json')['metadata'])
    with pytest.raises(Stop) as e:validate_prepared(q)
    assert e.value.code=='PATIENT_COHORT_HASH_MISMATCH'

def test_detection_counts_distinct_cancers():
    f=pd.DataFrame({'cancer_id':['BRCA']*5,'lncrna_id':['L']*5,'detection_rate_logcpm_gt0':[.8]*5})
    _,eligible,rep=detection_eligibility(f,5)
    assert not eligible

def test_detection_conflicting_duplicate_rejected():
    f=pd.DataFrame({'cancer_id':['BRCA']*2,'lncrna_id':['L']*2,'detection_rate_logcpm_gt0':[.8,.2]})
    with pytest.raises(Stop):detection_eligibility(f,5)

def test_tsv_leading_zero_id_preserved(tmp_path):
    p=tmp_path/'f.tsv';p.write_text('patient_id\tgene_id\tvalue\n000123\tGENE:ENSG00000000001\t1\n')
    f=table(p);assert f.patient_id.iloc[0]=='000123';assert f.gene_id.iloc[0]=='ENSG00000000001'

def test_sample_patient_mismatch_rejected():
    auth=pd.DataFrame({'cancer_id':['BRCA'],'sample_id':['S'],'patient_id':['P']})
    f=pd.DataFrame({'cancer_id':['BRCA'],'sample_id':['S'],'patient_id':['WRONG'],'gene_id':['G'],'x':[1.]})
    with pytest.raises(Stop) as e:matrix(f,'gene_id','x',['P'],['G'],auth)
    assert e.value.code=='SAMPLE_PATIENT_CONFLICT'

def test_unmatched_entity_not_silent_nan():
    auth=pd.DataFrame({'cancer_id':['BRCA'],'patient_id':['P']})
    f=pd.DataFrame({'cancer_id':['BRCA'],'patient_id':['P'],'gene_id':['WRONG'],'x':[1.]})
    with pytest.raises(Stop) as e:matrix(f,'gene_id','x',['P'],['G'],auth)
    assert e.value.code=='ID_COVERAGE_FAIL'

def test_multisample_conflict_not_averaged():
    auth=pd.DataFrame({'cancer_id':['BRCA']*2,'sample_id':['S1','S2'],'patient_id':['P']*2})
    f=auth.assign(gene_id='G',x=[1.,2.])
    with pytest.raises(Stop) as e:matrix(f,'gene_id','x',['P'],['G'],auth)
    assert e.value.code=='DUPLICATE_VALUE_CONFLICT'

@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_cnv_cow_arrays_are_writable_copies(tmp_path,dtype):
    d=tmp_path/'cancer=BRCA';d.mkdir();atomic_json(d/'patient_ids.json',['0001','0002']);atomic_json(d/'lncrna_ids.json',['L'])
    np.save(d/'lncrna_signed_mean.npy',np.array([[.1],[.2]],dtype));np.save(d/'lncrna_callable.npy',np.array([[1],[0]],np.uint8))
    with pd.option_context('mode.copy_on_write',True):a,c,_=cnv(tmp_path,'BRCA',['0001','0002'],['L'])
    assert np.isfinite(a[0,0]) and np.isnan(a[1,0]) and a.flags.writeable
    assert np.load(d/'lncrna_signed_mean.npy')[1,0]==pytest.approx(.2)
    with pytest.raises(Stop):cnv(tmp_path,'BRCA',['MISSING'],['L'])

def test_signed_membership_unmapped_is_not_positive():
    e=pd.DataFrame({'pathway_id':['P']*2,'gene_id':['ENSG00000000001','ENSG00000000002']})
    w=pd.DataFrame({'pathway_id':['P']*2,'gene_id':['GENE:ENSG00000000001',None],'weight':[-1.,np.nan],'mapping_status':['mapped','unmapped']})
    m,r=membership_join(e,w);assert m.weight.iloc[0]==-1 and np.isnan(m.weight.iloc[1]);assert r['unmapped_source_rows_excluded']==1

@pytest.mark.parametrize('weight',[np.inf,0.])
def test_mapped_invalid_weight_rejected(weight):
    e=pd.DataFrame({'pathway_id':['P'],'gene_id':['G']})
    w=e.assign(weight=weight,mapping_status='mapped')
    with pytest.raises(Stop):membership_join(e,w)

def test_mapped_nan_is_explicit_unsigned():
    # R62 real-data semantic: a mapped row with no weight carries no signed
    # effect; it joins as explicit unsigned and the graph emits
    # membership_unspecified (frozen v2_9/g012 source convention).
    e=pd.DataFrame({'pathway_id':['P'],'gene_id':['G']})
    w=e.assign(weight=np.nan,mapping_status='mapped')
    m,r=membership_join(e,w)
    assert m.weight.isna().all() and r['mapped_nan_reclassified_unsigned']==1

def test_signed_unsigned_duplicate_conflict():
    e=pd.DataFrame({'pathway_id':['P'],'gene_id':['G']});w=pd.DataFrame({'pathway_id':['P','P'],'gene_id':['G','G'],'weight':[1.,np.nan],'mapping_status':['mapped','explicit_unsigned']})
    with pytest.raises(Stop):membership_join(e,w)

@pytest.mark.parametrize('weight',[0.,-.4,np.inf,np.nan])
def test_ppi_confidence_rejected(weight):
    with pytest.raises(Stop):normalize_ppi(pd.DataFrame({'protein_id_a':['P1'],'protein_id_b':['P2'],'weight':[weight]}))

def test_ppi_reciprocal_dedup_and_selfloop():
    f=pd.DataFrame({'protein_id_a':['P1','P2'],'protein_id_b':['P2','P1'],'weight':[.4,.4],'physical_score':[400,400],'edge_id':['E1','E2'],'relation_type':['physical_interaction']*2,'source_database':['STRING_physical']*2})
    assert len(normalize_ppi(f))==1
    with pytest.raises(Stop):normalize_ppi(f.assign(protein_id_b=f.protein_id_a))

def test_binding_unknown_not_global():
    f=pd.DataFrame({'lncrna_id':['L'],'protein_id':['P'],'cancer_id':[None],'weight':[.5],'edge_id':['E'],'relation_type':['physical_binding'],'source_database':['measured_binding'],'mapping_status':['MAPPED']})
    with pytest.raises(Stop):normalize_binding(f)
    assert normalize_binding(f.assign(is_context_specific=False)).cancer_id.iloc[0]=='GLOBAL'
    with pytest.raises(Stop):normalize_binding(f.assign(is_context_specific=True))

def _mutation(tmp_path,rows):
    d=tmp_path/'genomic';d.mkdir(exist_ok=True)
    p=d/'sample_lncrna_mutation_mc3_assay_callable.tsv';pd.DataFrame(rows).to_csv(p,sep='\t',index=False)
    from cla_engine.prepare import load_mutation
    return load_mutation('BRCA',['L'],['P1','P2'],{'imports_root':str(tmp_path),'synthetic_fixture':True,'minimum_pairs':2})

def _mr(patient='P1',entity='L',event=True,count=1,**extra):
    return {'cancer_id':'BRCA','patient_id':patient,'lncrna_id':entity,'lncrna_any_mutation':event,'lncrna_mutation_available':True,'exonic_variant_count':count,**extra}

def test_sparse_mc3_does_not_make_positive_only_rate(tmp_path):
    a,ok,r=_mutation(tmp_path,[_mr(entity='__MC3_WES_ASSAY_CALLABLE__',event=False,count=0),_mr(patient='P2',entity='__MC3_WES_ASSAY_CALLABLE__',event=False,count=0),_mr()])
    assert ok[0]==0 and np.all(a==0);assert r['sentinel_rows']==2 and r['observed_positive_records']==1

def test_explicit_locus_denominator_rate(tmp_path):
    a,ok,r=_mutation(tmp_path,[_mr(locus_callable=True),_mr(patient='P2',event=False,count=0,locus_callable=True)])
    assert ok[0]==1;assert a[0,0]==.5

@pytest.mark.parametrize('bad',[-1.,np.inf,-np.inf])
def test_mutation_invalid_count_rejected(tmp_path,bad):
    with pytest.raises(Stop):_mutation(tmp_path,[_mr(count=bad)])

def test_no_wt_for_explicit_noncallable(tmp_path):
    a,ok,r=_mutation(tmp_path,[_mr(locus_callable=True),_mr(patient='P2',event=False,count=0,locus_callable=False)])
    assert ok[0]==0

def test_cnv_data_rank_is_explicit_not_matrix_rank():
    from cla_engine.statistics import pairs
    rng=np.random.default_rng(12);n=30;C=np.exp(np.linspace(0,5,n))[:,None];X=(np.log(C)+rng.normal(size=(n,1)));Y=X+rng.normal(size=(n,1));Z=np.ones((n,1))
    a=pairs(X,Y,C,np.ones_like(C,bool),Z,np.array([0]),np.array([0]),np.arange(n),12,cnv_transform='raw_signed')
    b=pairs(X,Y,C,np.ones_like(C,bool),Z,np.array([0]),np.array([0]),np.arange(n),12,cnv_transform='rank')
    assert a['rank'][0]==b['rank'][0]==2;assert not np.allclose(a['r'],b['r'])

# A synthetic checkpoint validates the identity protocol only, never the PyG math.
def completion_fixture(prepared,run):
    from cla_engine.train import load_npz,aggregate_metrics,model_signature,save_checkpoint
    _,spec,cfg,lin=validate_prepared(prepared);run=Path(run);run.mkdir(parents=True)
    pred={t:[] for t in ('pathway','state')};files={}
    for c in spec['cancers']:
        for t in pred:
            a=load_npz(prepared/'views'/f'{c}__validation'/f'{t}.npz');d=np.zeros(len(a['lnc']),np.float32)
            path=run/'predictions'/f'{c}__{t}.npz';atomic_npz(path,lnc=a['lnc'],target=a['target'],score=a['base'],baseline=a['base'],direction_logit=d,discovery_available=a['discovery_available'])
            files[f'predictions/{path.name}']=sha(path);pred[t].append((c,a,a['base'],d))
    sig=model_signature(prepared,cfg,spec);metrics=aggregate_metrics(pred,spec)
    r={'state':'NO_GO_WITHIN_DIAGNOSTIC_BUDGET','signature':sig,'config_sha256':digest(cfg),'source_profile':'diagnostic','synthetic_fixture':True,'best_step':3,'alpha':0.,'validation':metrics,'gradient_proof':{'pathway_max_shared_grad':1.,'state_max_shared_grad':1.},'task_weights':{'pathway':1.,'state':.5},'prediction_files':files,'prepared_manifest_sha256':sha(prepared/'MANIFEST.json'),'lineage':lin,'attempt_id':'fixture-only-not-a-pyg-training','formal_test_evaluated':False,'not_deployed':True}
    rec=save_checkpoint(run,{'signature':sig,'format':'CLA_FULL_PYG_R62_FULL_TRAINING_STATE_V1','next_step':3,'best':{'step':3,'alpha':0.},'completion':copy.deepcopy(r),'test_fixture_only':True})
    r['last_checkpoint']=rec;atomic_json(run/'TRAINING_RECEIPT.json',r);atomic_json(run/'RUN_CONTRACT.json',{'prepared_path':str(prepared.resolve())})
    return r

def test_bound_completion_and_cpu_evaluation(source_shaped_prepared,tmp_path):
    _,p,_=source_shaped_prepared;r=completion_fixture(p,tmp_path/'run');checked,_=verify_completion(p,tmp_path/'run');assert checked==r
    from cla_engine.evaluate import evaluate_fold
    ev=evaluate_fold(p,tmp_path/'run',tmp_path/'evaluation')
    assert ev['state']=='OUTER_TEST_EVALUATED_NOT_DEPLOYED';assert ev['synthetic_fixture']
    assert (tmp_path/'evaluation/MANIFEST.json.attestation.json').is_file()
    assert all('direction_n' in x for x in ev['metrics'])

@pytest.mark.parametrize('field,value',[('state','ERROR'),('state','GO_TO_FORMAL_EVALUATION'),('alpha',1.),('best_step',7)])
def test_modified_receipt_rejected(source_shaped_prepared,tmp_path,field,value):
    _,p,_=source_shaped_prepared;r=completion_fixture(p,tmp_path/'run');r[field]=value;atomic_json(tmp_path/'run/TRAINING_RECEIPT.json',r)
    with pytest.raises(Stop):verify_completion(p,tmp_path/'run')

def test_stale_predictions_rejected(source_shaped_prepared,tmp_path):
    _,p,_=source_shaped_prepared;completion_fixture(p,tmp_path/'run');atomic_npz(tmp_path/'run/predictions/OLD__state.npz',a=np.ones(10))
    with pytest.raises(Stop) as e:verify_completion(p,tmp_path/'run')
    assert e.value.code=='STALE_PREDICTION_FILE'

def test_controller_signatures_reject_manual_edit(tmp_path):
    cfg=enroll_test_controller({'work_root':str(tmp_path/'work')},tmp_path);p=tmp_path/'JOBS.json';atomic_json(p,{'ok':True});sign_file(p,cfg,'FORMAL_JOB_PLAN');verify_signed_file(p,cfg,'FORMAL_JOB_PLAN')
    atomic_json(p,{'ok':False})
    with pytest.raises(Stop):verify_signed_file(p,cfg,'FORMAL_JOB_PLAN')

def test_formal_queue_requires_signed_admission_before_remote(tmp_path,monkeypatch):
    from cla_engine.queue import run_queue,job_grid
    cfg=read_json(ROOT/'config/149.json');cfg['work_root']=str(tmp_path/'work');enroll_test_controller(cfg,tmp_path);p=tmp_path/'JOBS.json'
    atomic_json(p,{'format':'CLA_FULL_PYG_R62_15_JOB_PLAN_V1','config_sha256':digest(cfg),'jobs':job_grid(cfg,tmp_path/'fake')})
    monkeypatch.setattr('cla_engine.queue.supervise',lambda *a,**k:pytest.fail('must never reach remote'))
    with pytest.raises(Stop) as e:run_queue(ROOT,p,cfg,tmp_path/'run','cla-gpu',1)
    assert e.value.code=='CONTROLLER_ATTESTATION_MISSING'

@pytest.mark.parametrize('field',['code_sha256','source_lineage_sha256','task_contract_sha256','target_catalog_sha256','model_contract_sha256','candidate_catalog_sha256'])
def test_fivefold_mixed_lineage_rejected(tmp_path,field):
    from cla_engine.evaluate import gate
    cfg=read_json(ROOT/'config/149.json');cfg['work_root']=str(tmp_path/'work');enroll_test_controller(cfg,tmp_path);dirs=[]
    for fold in range(5):
        d=tmp_path/f'f{fold}';d.mkdir();lin={k:('G2' if k=='graph_variant' else 'a'*64) for k in FIELDS};lin['fold']=fold
        if fold==3:lin[field]='b'*64
        r={'fold':fold,'profile':'formal','synthetic_fixture':False,'state':'OUTER_TEST_EVALUATED_NOT_DEPLOYED','model_freeze_sha256':'f'*64,'metrics_contract':cfg['metrics_contract'],'model_freeze_sha256':'f'*64,'metrics_contract':cfg['metrics_contract'],
           'training_state':'TRAINED_FORMAL_CANDIDATE','config_sha256':digest(cfg),'lineage':lin,'model_signature':digest(fold),'sealed_manifest_sha256':'s'*64,'candidate_catalog':{},'metrics':[],'expected_cancers':[]}
        atomic_json(d/'EVALUATION_RECEIPT.json',r);commit_manifest(d,{'fixture':True});sign_file(d/'MANIFEST.json',cfg,'OUTER_EVALUATION');dirs.append(d)
    with pytest.raises(Stop) as e:gate(dirs,cfg,tmp_path/'gate')
    assert e.value.code=='GATE_MIXED_LINEAGE'

def test_resume_tar_excludes_status_and_predictions(tmp_path):
    from cla_engine.runtime_guard import make_resume_tar
    r=tmp_path/'run';(r/'checkpoints').mkdir(parents=True);(r/'predictions').mkdir();atomic_json(r/'STATUS.json',{'state':'ERROR'});atomic_json(r/'LATEST.json',{});(r/'checkpoints/x.pt').write_bytes(b'fixture');(r/'predictions/stale.npz').write_bytes(b'fixture')
    atomic_json(r/'LATEST.json',{'file':'checkpoints/x.pt','sha256':sha(r/'checkpoints/x.pt'),'next_step':2});p=tmp_path/'resume.tar';make_resume_tar(r,p)
    with tarfile.open(p) as f:names=f.getnames()
    assert set(names)=={'LATEST.json','checkpoints/x.pt'}

def test_attempt_isolation_and_pid_progress(tmp_path):
    from cla_engine.runtime_guard import begin_attempt,check_remote_status
    atomic_json(tmp_path/'STATUS.json',{'state':'ERROR','attempt_id':'old'});(tmp_path/'predictions').mkdir();(tmp_path/'predictions/x').write_text('old')
    attempt=begin_attempt(tmp_path);assert attempt!='old';assert not (tmp_path/'predictions').exists()
    assert check_remote_status({'state':'ERROR','attempt_id':'old'},attempt,True,20,0,startup_timeout=5)=='STALE_ATTEMPT_STATUS'
    assert check_remote_status({'state':'TRAINING','attempt_id':attempt,'updated_unix':99},attempt,False,100,0)=='WORKER_EXITED_WITHOUT_TERMINAL'
    assert check_remote_status({'state':'TRAINING','attempt_id':attempt,'updated_unix':10},attempt,True,100,0,progress_timeout=30)=='WORKER_PROGRESS_STALLED'

def test_source_freeze_detects_partition_replacement(tmp_path):
    from cla_engine.source_lock import record_sources
    p=tmp_path/'raw.npy';p.write_bytes(b'first');cfg={'work_root':str(tmp_path/'work')}
    record_sources(cfg,{'cnv':{'path':str(p),'sha256':sha(p)}});p.write_bytes(b'other')
    with pytest.raises(Stop) as e:record_sources(cfg,{'cnv':{'path':str(p),'sha256':sha(p)}})
    assert e.value.code=='FROZEN_SOURCE_CHANGED'

def test_global_local_binding_not_double_counted_in_context():
    from cla_engine.graph import graph_npz
    ids={'lncRNA':['L'],'protein':['P']};features={k:np.ones((1,6),np.float32) for k in ids}
    edges={'lncRNA|reported_binding_positive|protein':{'edge_index':np.array([[0,0],[0,0]]),'weight':np.array([.4,.8],np.float32),'context':np.array(['GLOBAL','BRCA'])}}
    dyn={s:(np.empty((2,0),np.int64),np.empty(0,np.float32)) for s in ['positive','negative']}
    out=graph_npz(ids,edges,'BRCA',dyn,features)
    assert out['edge::lncRNA|reported_binding_positive|protein'].shape[1]==1
    assert out['weight::lncRNA|reported_binding_positive|protein'][0]==pytest.approx(.8)

def test_graph_cap_restricts_member_gene_and_family_scope(tmp_path):
    from cla_engine.fixture import make_fixture
    from cla_engine.prepare import prepare
    cfg=make_fixture(tmp_path/'source');cfg['profiles']['diagnostic']['max_pathways']=2
    p=prepare(cfg,tmp_path/'prepared',0,'diagnostic');s=read_json(p/'SPEC.json')
    m=pd.read_csv(cfg['sources']['exact_membership']['path'],sep='\t');h=pd.read_csv(cfg['sources']['hierarchy']['path'],sep='\t')
    assert len(s['nodes']['pathway'])==2
    pg=pd.read_csv(cfg['sources']['protein_gene']['path'],sep='\t')
    assert set(s['nodes']['gene'])==set(m.loc[m.pathway_id.isin(s['nodes']['pathway']),'gene_id'])|set(pg.gene_id)
    assert set(s['nodes']['pathway_family'])==set(h.loc[h.pathway_id.isin(s['nodes']['pathway']),'pathway_family_id'])
    assert read_json(p/'GRAPH_SIZE.json')['diagnostic_is_formal_memory_estimate'] is False

def test_declared_roles_in_yaml_match_active_config():
    import yaml
    cfg=read_json(ROOT/'config/149.json');y=yaml.safe_load((ROOT/'config/model_v3_2_full_multitask.yaml').read_text())
    assert set(y['modules']['state']['targets'])=={t['id'] for t in cfg['targets'] if t['role']=='primary'}
    assert set(y['modules']['state']['auxiliary_targets'])=={t['id'] for t in cfg['targets'] if t['role']=='auxiliary'}
    assert y['shared_core']['checkpoint_lineage']=='CC_HHGT_PYG_FULL_R62_JOINT'

def test_code_capsule_digest_matches_controller(tmp_path):
    from cla_engine.transport import make_code_tar
    import hashlib
    from cla_engine.common import code_digest
    p=tmp_path/'code.tar';make_code_tar(ROOT,p)
    with tarfile.open(p) as t:
        entries={n.name:hashlib.sha256(t.extractfile(n).read()).hexdigest() for n in t.getmembers() if n.isfile() and (n.name=='cla.py' or (Path(n.name).parts[0] in {'cla_engine','cc_hhgt','scripts','config'} and Path(n.name).suffix.lower() in {'.py','.r','.sh','.yaml','.yml','.json'}))}
    assert digest(entries)==code_digest()

def test_hmac_not_shipped_with_code_capsule(tmp_path):
    from cla_engine.transport import make_code_tar
    p=tmp_path/'code.tar';make_code_tar(ROOT,p)
    with tarfile.open(p) as t:assert not any('attestation.key' in n.name for n in t.getmembers())

def test_transport_deadline_limits_long_copy_timeout():
    from cla_engine.transport import SSH
    import time
    r=SSH('cla-gpu');r.deadline=time.time()+10
    assert 0<r.limit(7200)<=10
    r.deadline=time.time()-1
    with pytest.raises(Stop):r.limit(7200)

def test_valid_fivefold_protocol_is_not_always_rejected(tmp_path):
    """Protocol-only signed synthetic records, not a claim that five models trained."""
    from cla_engine.evaluate import gate
    cfg=read_json(ROOT/'config/149.json');cfg['work_root']=str(tmp_path/'work');enroll_test_controller(cfg,tmp_path);dirs=[]
    cancers=['C'+str(i) for i in range(cfg['gates']['minimum_evaluable_cancers'])]
    for fold in range(5):
        d=tmp_path/f'f{fold}';d.mkdir()
        lin={k:('G2' if k=='graph_variant' else 'a'*64) for k in FIELDS};lin['fold']=fold
        metrics=[{'fold':fold,'cancer_id':c,'task':t,'available':True,
                  'delta_auprc':.05,'precision_at_k':.8,'baseline_precision_at_k':.7}
                 for c in cancers for t in ('pathway','state')]
        r={'fold':fold,'profile':'formal','synthetic_fixture':False,'state':'OUTER_TEST_EVALUATED_NOT_DEPLOYED',
           'model_freeze_sha256':'f'*64,'metrics_contract':cfg['metrics_contract'],
           'training_state':'TRAINED_FORMAL_CANDIDATE','config_sha256':digest(cfg),'lineage':lin,
           'model_signature':digest(fold),'sealed_manifest_sha256':digest(['sealed',fold]),
           'candidate_catalog':{'fixed':'protocol-only'},'metrics':metrics,'expected_cancers':cancers}
        atomic_json(d/'EVALUATION_RECEIPT.json',r);commit_manifest(d,{'protocol_fixture':True})
        sign_file(d/'MANIFEST.json',cfg,'OUTER_EVALUATION');dirs.append(d)
    result=gate(dirs,cfg,tmp_path/'gate')
    assert result['state']=='PREDICTION_GATE_PASS_NOT_DEPLOYED'
    assert result['production_release_allowed'] is False

@pytest.mark.parametrize('field,value,code',[('format','OLD','COMPLETION_CHECKPOINT_FORMAT'),('next_step',0,'COMPLETION_STEP_INVALID')])
def test_checkpoint_inner_generation_and_progress_checked(source_shaped_prepared,tmp_path,field,value,code):
    import torch
    _,p,_=source_shaped_prepared;r=completion_fixture(p,tmp_path/'run')
    cp=tmp_path/'run'/r['last_checkpoint']['file'];data=torch.load(cp,weights_only=False);data[field]=value
    torch.save(data,cp);r['last_checkpoint']['sha256']=sha(cp);atomic_json(tmp_path/'run/TRAINING_RECEIPT.json',r)
    with pytest.raises(Stop) as e:verify_completion(p,tmp_path/'run')
    assert e.value.code==code


def enroll_test_controller(cfg,root):
    import sys
    from cla_engine.controller import enroll_controller
    cfg.setdefault('imports_root',str(root/'imports'))
    cfg['controller_contract']={'role':'149','python_executable':sys.executable}
    enroll_controller(cfg)
    return cfg
