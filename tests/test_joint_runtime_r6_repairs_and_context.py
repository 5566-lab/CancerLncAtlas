"""R6 regressions: real CPU functions, sparse source shapes, and adversarial workflows.
Synthetic fixtures are explicitly NOT clinical, GPU, SSH or causal validation.
"""
from pathlib import Path
import copy,json,subprocess,sys,time
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from cla_engine.common import Stop,read_json,atomic_json,sha,commit_manifest
from cla_engine.genomics import normalize_mutation,project_gene_context,FEATURES,SENTINEL
from cla_engine.cnv_sensitivity import common_mask_audit,pathway_cnv
from cla_engine.readiness import validate_task_arrays
from cla_engine.transport import SSH,cleanup_command_allowed,recover_after_deadline
from cla_engine.singlecell_interpretation import expression_table,activity_table,pair_context,donor_correlations,interpret,eligible_manifest
from cla_engine.knowledge_paths import knowledge_paths

ROOT=Path(__file__).resolve().parents[1]

def mutation_rows():
    return pd.DataFrame([
        {'cancer_id':'BRCA','patient_id':'001','gene_id':SENTINEL,'is_mutated':False,'mutation_callable':True,'mutation_count':0,'absence_is_wildtype':False},
        {'cancer_id':'BRCA','patient_id':'002','gene_id':SENTINEL,'is_mutated':False,'mutation_callable':True,'mutation_count':0,'absence_is_wildtype':False},
        {'cancer_id':'BRCA','patient_id':'001','gene_id':'GENE:ENSG01','is_mutated':True,'mutation_callable':True,'mutation_count':2,'absence_is_wildtype':False},
    ])

def test_sparse_positive_is_observation_not_rate():
    c=normalize_mutation(mutation_rows(),['001','002'],['ENSG01','ENSG02'],'gene','BRCA')
    v,r=c.summaries([0,1],minimum=1)
    assert c.events[0,0]==1 and np.isnan(c.events[1,0])
    assert c.callable.sum()==0
    assert v[0,0]==pytest.approx(np.log(2)) and v[0,1]==pytest.approx(np.log(3))
    assert v[0,3]==0 and v[0,4]==0 and v[0,-1]==1
    assert np.all(v[1]==0) and r['all_rates_unavailable']

@pytest.mark.parametrize('burden',[-1,np.inf,-np.inf,.4])
def test_gene_burden_rejected(burden):
    f=mutation_rows();f['mutation_count']=f.mutation_count.astype(float);f.loc[2,'mutation_count']=burden
    with pytest.raises(Stop):normalize_mutation(f,['001','002'],['ENSG01'],'gene','BRCA')

def test_gene_callability_conflict_not_or():
    f=mutation_rows();f['locus_callable']=False;f.loc[2,'absence_is_wildtype']=True
    with pytest.raises(Stop,match='uncallable'):normalize_mutation(f,['001','002'],['ENSG01'],'gene','BRCA')

def test_explicit_locus_rate_and_annotations():
    f=pd.DataFrame([{'cancer_id':'BRCA','patient_id':str(i),'gene_id':'ENSG01','is_mutated':i==0,'mutation_callable':True,'mutation_count':int(i==0),'absence_is_wildtype':True,'consequence':'missense_variant' if i==0 else ''} for i in range(4)])
    c=normalize_mutation(f,[str(i) for i in range(4)],['ENSG01'],'gene','BRCA');v,_=c.summaries(range(4),minimum=2)
    assert v[0,3]==.25 and v[0,4]==1 and v[0,5]==0 and v[0,6]>0
    changed=f.copy();changed.loc[changed.patient_id.eq('3'),'mutation_count']=3;changed.loc[changed.patient_id.eq('3'),'is_mutated']=True
    d=normalize_mutation(changed,[str(i) for i in range(4)],['ENSG01'],'gene','BRCA')
    assert np.array_equal(c.summaries([0,1,2],2)[0],d.summaries([0,1,2],2)[0])

@pytest.mark.parametrize('ix',[[-1],[4],[0,0]])
def test_genotype_bad_cohort_index(ix):
    c=normalize_mutation(mutation_rows(),['001','002'],['ENSG01'],'gene','BRCA')
    with pytest.raises(Stop):c.summaries(ix)

def test_project_partner_is_actual_coding_gene_not_all_pathway():
    values=np.zeros((2,len(FEATURES)),np.float32);values[0,0]=1;values[0,-1]=1;values[1,-1]=1
    pg=pd.DataFrame({'protein_id':['P1','P2'],'gene_id':['GENE:ENSG01','GENE:ENSG02']})
    mem=pd.DataFrame({'pathway_id':['W','W'],'gene_id':['ENSG01','ENSG02']})
    bind=pd.DataFrame({'lncrna_id':['L','L'],'protein_id':['P1','P1'],'cancer_id':['GLOBAL','BRCA'],'weight':[.7,.7],'edge_id':['e1','e2'],'source_database':['measured','measured'],'relation_type':['physical_binding','physical_binding'],'mapping_status':['mapped','mapped']})
    g=project_gene_context(values,['ENSG01','ENSG02'],pg,['P1','P2'],mem,['W'],bind,['L'],'BRCA','G2')
    assert g['partner'][0,0]==1 and g['pathway'][0,0]==.5
    g0=project_gene_context(values,['ENSG01','ENSG02'],pg,['P1','P2'],mem,['W'],bind,['L'],'BRCA','G0')
    assert not g0['partner'].any()

def test_cnv_a2_common_mask_and_no_primary_mutation():
    rng=np.random.default_rng(19);n=80;p=rng.normal(size=(n,1));x=p+rng.normal(0,.3,(n,1));y=p+rng.normal(0,.3,(n,1));c=rng.normal(size=(n,1));z=np.ones((n,1));m=np.ones((n,1),bool)
    saved=x.copy();pm=m.copy();pm[:5]=False
    r=common_mask_audit(x,y,c,m,p,pm,z,[0],[0],range(n),12)
    assert all(v['n'][0]==75 for v in r.values())
    assert abs(r['ranked_A2_common']['r'][0])<abs(r['ranked_A1_common']['r'][0])
    assert np.array_equal(x,saved)
    assert r['ranked_A2_common']['rank'][0]==3

def test_cnv_constant_rank_not_overcounted():
    n=20;v=np.arange(n,dtype=float)[:,None];zero=np.zeros_like(v);m=np.ones_like(v,bool)
    r=common_mask_audit(v,v,zero,m,zero,m,np.ones((n,1)),[0],[0],range(n),12)
    assert r['ranked_A2_common']['rank'][0]==1 and r['ranked_A2_common']['r'][0]==pytest.approx(1)

def test_evidence_contexts_remain_distinct_and_review_conflict_unknown():
    from cc_hhgt.v32.evidence_training import build_exact_event_bags
    from cc_hhgt.v32.evidence_semantics_v2 import independent_events
    a={'cancer_id':'BRCA','lncrna_id':'L','pathway_id':'W','source_record_id':'R','pmid':'123456','source_database':'X','experiment_type':'qPCR','direction':'positive','manual_review_status':'approved','cell_line':'MCF7','tissue':'tumour'}
    b={**a,'cell_line':'T47D','tissue':'organoid'};members=pd.DataFrame({'gene_id':['G'],'pathway_id':['W']})
    x=build_exact_event_bags(pd.DataFrame([a,b]),pd.DataFrame(),members)
    assert len(x.events)==2 and len(independent_events(x.events))==2
    for data in [[a,{**a,'manual_review_status':'rejected'}],[{**a,'manual_review_status':'rejected'},a]]:
        e=build_exact_event_bags(pd.DataFrame(data),pd.DataFrame(),members).events
        assert len(e)==1 and e.confidence_target.isna().all() and e.review_conflict.all()
        assert independent_events(e).confidence_target.isna().all()

@pytest.mark.parametrize('text',['no upregulation','not inhibited','no significant increase','supported','unsupported','not downregulated','upstream'])
def test_direction_not_guessed_from_substrings(text):
    from cc_hhgt.v32.evidence_training import _direction_target
    assert _direction_target(text)==-1

@pytest.mark.parametrize('text,answer',[('positive',2),('downregulated',0),('no effect',1),('unknown',-1)])
def test_explicit_direction_preserved(text,answer):
    from cc_hhgt.v32.evidence_training import _direction_target
    assert _direction_target(text)==answer

def test_legacy_pan_cancer_cannot_override_raw_disease():
    from cc_hhgt.v32.evidence_training import build_exact_event_bags
    row={'cancer_id':'PAN_CANCER','disease_raw':"Alzheimer's disease",'lncrna_id':'L','partner_id':'G','source_record_id':'R'}
    a=build_exact_event_bags(pd.DataFrame(),pd.DataFrame([row]),pd.DataFrame({'gene_id':['G'],'pathway_id':['W']}))
    assert a.events.empty and len(a.rejected)==1

def test_sql_streaming_has_same_semantic_signature_and_conflict_guard():
    # Explicit static assertion; real DuckDB execution is tested only if available.
    from cc_hhgt.v32 import evidence_streaming_training as m
    import inspect
    text=inspect.getsource(m._materialize_exact_tables)
    assert 'review_checked_events' in text and '_qmin <> _qmax' in text
    assert all(x in text for x in ['tissue','cell_line','experiment_id','intervention','mediator_id'])
    assert m.STREAMING_FORMAT.endswith('CONTEXT_R6')

@pytest.mark.parametrize('field,value',[('is_computational',True),('species','Mus musculus'),('mapping_status','ambiguous'),('relation_type','activates')])
def test_static_relation_rejects_semantic_counterexamples(field,value):
    from cla_engine.graph_contracts import normalize_binding
    f=pd.DataFrame([{'lncrna_id':'L','protein_id':'P','cancer_id':'GLOBAL','weight':.6,'edge_id':'E','mapping_status':'MAPPED','relation_type':'physical_binding','source_database':'verified_fixture'}])
    f[field]=value
    with pytest.raises(Stop):normalize_binding(f)

def test_static_source_columns_not_optional():
    from cla_engine.graph_contracts import normalize_ppi
    with pytest.raises(Stop):normalize_ppi(pd.DataFrame({'protein_id_a':['P1'],'protein_id_b':['P2'],'weight':[.5]}))

def sc_fixture():
    return pd.DataFrame([{'dataset_id':'D','cancer_id':'BRCA','lncrna_id':'L','cell_type_major':ct,'cell_state':'s','mean_log_expression':e,'detection_rate':.4,'n_donors':12,'n_cells':100,'n_detected':40} for ct,e in [('Tcell',1.),('Bcell',9.)]])

def query_fixture():return pd.DataFrame({'cancer_id':['BRCA'],'lncrna_id':['L'],'target_id':['W'],'task':['pathway'],'model_probability':[.91]})

def test_singlecell_context_not_broadcast_or_fake_correlated():
    e,_=expression_table(sc_fixture(),{'D'})
    a=activity_table(pd.DataFrame([{'dataset_id':'D','cancer_id':'BRCA','pathway_id':'W','cell_type':'Tcell','cell_state':'s','activity':2.,'activity_kind':'ucell'}]),{'D'})
    r=pair_context(query_fixture(),e,a).set_index('cell_type')
    assert r.loc['Tcell','activity_score']==2 and np.isnan(r.loc['Bcell','activity_score'])
    assert r.correlation.isna().all() and not r.independent_replication_verified.any()
    assert r.model_probability.eq(.91).all()

@pytest.mark.parametrize('rate',[-.1,1.1,np.inf])
def test_sc_invalid_detection_rejected(rate):
    f=sc_fixture();f.loc[0,'detection_rate']=rate
    with pytest.raises(Stop):expression_table(f,{'D'})

def test_sc_ineligible_before_validation_and_vocabulary():
    f=sc_fixture();bad=f.iloc[:1].assign(dataset_id='BAD',cell_type_major='never_shared',detection_rate=99.)
    out,n=expression_table(pd.concat([f,bad]),{'D'})
    assert n==1 and 'never_shared' not in set(out.cell_type)

def test_sc_donor_not_cell_level_and_conflicts():
    rng=np.random.default_rng(1);f=pd.DataFrame([{'dataset_id':'D','cancer_id':'BRCA','donor_id':f'P{i}','lncrna_id':'L','target_id':'W','task':'pathway','cell_type':'Tcell','cell_state':'s','expression':float(i),'activity':float(i)+rng.normal(0,.2),'activity_kind':'ucell'} for i in range(10)])
    a=donor_correlations(f,query_fixture(),8);assert a.n_donors.iloc[0]==10 and a.rho.iloc[0]>.9
    with pytest.raises(Stop):donor_correlations(pd.concat([f,f.iloc[:1].assign(expression=99.)]),query_fixture(),8)

def test_clinical_empty_label_validation_rejected_before_optimizer():
    from cc_hhgt.v32.clinical_training import train_private_clinical_head,ClinicalTrainingError
    arrays={'OS':{'bin':np.zeros(10,dtype=int),'event':np.ones(10,np.float32),'available':np.r_[np.ones(8),np.zeros(2)]}}
    with pytest.raises(ClinicalTrainingError,match='VALIDATION_LABELS'):
        train_private_clinical_head(np.ones((10,3),np.float32),np.ones((10,2),np.float32),arrays,train_indices=np.arange(8),validation_indices=np.arange(8,10),n_bins=3,epochs=2,seed=17)

def test_clinical_nll_no_label_is_not_zero():
    import torch
    from cc_hhgt.v32.clinical_training import discrete_time_nll,ClinicalTrainingError
    with pytest.raises(ClinicalTrainingError):discrete_time_nll(torch.zeros((2,3)),torch.zeros(2,dtype=torch.long),torch.ones(2),torch.zeros(2))

@pytest.mark.parametrize('cmd',['python train.py','kill -TERM 1; python t.py','cat /x && python t.py','scp a b','echo renew > /lease'])
def test_cleanup_cannot_start_or_modify_training(cmd):assert not cleanup_command_allowed(cmd)

@pytest.mark.parametrize('cmd',['cat /job/run/STATUS.json','sha256sum /job/r.tar','ps -p 123 -o stat= -o args=','kill -TERM 123','tar -cf /job/result.tar -C /job/run -- .'])
def test_cleanup_accepts_only_bounded_recovery_commands(cmd):assert cleanup_command_allowed(cmd)

def test_cleanup_deadline_separate_from_optimizer_deadline():
    r=SSH('test-alias');r.deadline=time.time()-2;r.cleanup_deadline=time.time()+25
    with pytest.raises(Stop):r.limit(20)
    assert 0<r.limit(30,cleanup=True)<=25

def test_actual_recovery_routine_persists_checkpoint_after_deadline(tmp_path):
    payload=b'synthetic-checkpoint-for-transport-not-torch';src=tmp_path/'source.pt';src.write_bytes(payload)
    pointer={'file':'checkpoints/g0.pt','sha256':sha(src),'next_step':4,'created_unix':1}
    class FakeRemote:
        def run(self,cmd,check=True,cleanup=False,**kw):
            assert cleanup and cleanup_command_allowed(cmd)
            return SimpleNamespace(returncode=1,stdout='',stderr='')
        def json(self,path,cleanup=False):
            assert cleanup
            return pointer if path.endswith('LATEST.json') else None
        def pull(self,remote,local,cleanup=False):
            assert cleanup;Path(local).write_bytes(payload)
    r=recover_after_deadline(FakeRemote(),'/job',tmp_path/'out',tmp_path/'out/run',tmp_path/'p','a','123',time.time()-1,grace=10)
    assert r['state']=='WORKER_STOPPED_RECOVERY_PARTIAL'
    assert (tmp_path/'out/run/checkpoints/g0.pt').read_bytes()==payload
    assert r['optimizer_budget_extended'] is False

@pytest.fixture(scope='module')
def prepared_r6(tmp_path_factory):
    from cla_engine.fixture import make_fixture
    from cla_engine.prepare import prepare
    p=tmp_path_factory.mktemp('r6whole');cfg=make_fixture(p/'src');prepare(cfg,p/'prepared',0,'diagnostic');return p,cfg

def test_true_cpu_prepare_includes_genomic_nodes_and_no_sc(prepared_r6):
    p,cfg=prepared_r6;s=read_json(p/'prepared/SPEC.json')
    assert s['singlecell_role']=='interpretation_only' and s['genomics_enabled']
    v=p/'prepared/views/C1__validation'
    with np.load(v/'graph.npz') as a:
        assert a['x::gene'].shape[1]>6 and a['x::protein'].shape[1]>6 and np.any(a['x::gene'][:,6:])
        assert not a['sc_available'].any() and not a['singlecell'].any()
    inputs=read_json(p/'prepared/INPUT_HASHES.json')
    assert not any('singlecell' in key.lower() for key in inputs)
    assert (p/'prepared/preflight_report.html').is_file()

def test_no_valid_candidate_stops_on_cpu(tmp_path):
    from cla_engine.fixture import make_fixture
    from cla_engine.prepare import prepare
    cfg=make_fixture(tmp_path/'src');cfg['minimum_pairs']=1000
    with pytest.raises(Stop) as exc:prepare(cfg,tmp_path/'p',0,'diagnostic')
    assert exc.value.code in {'TASK_NOT_TRAINABLE_CPU','COVARIATE_COHORT_NOT_ADMISSIBLE','COEXPRESSION_NO_ESTIMABLE_PAIRS'}
    assert not (tmp_path/'p/MANIFEST.json').exists()

def test_view_index_mismatch_rejected_even_with_recommitted_manifest(prepared_r6,tmp_path):
    import shutil
    from cla_engine.lineage import validate_prepared
    p,cfg=prepared_r6;d=tmp_path/'copy';shutil.copytree(p/'prepared',d)
    v=read_json(d/'VIEWS.json');v[0]['cancer_index']=2;atomic_json(d/'VIEWS.json',v)
    m=read_json(d/'MANIFEST.json');commit_manifest(d,m['metadata'])
    with pytest.raises(Stop):validate_prepared(d)

def test_cnv_audit_end_to_end(prepared_r6,tmp_path):
    from cla_engine.cnv_sensitivity import run_audit
    p,cfg=prepared_r6;r=run_audit(p/'prepared',tmp_path/'sensitivity')
    assert r['primary_training_target_changed'] is False and r['scope']=='source'
    f=pd.read_csv(tmp_path/'sensitivity/C1.tsv.gz',sep='\t');assert len(f.method.unique())==3

def test_knowledge_posthoc_and_singlecell_real_functions(prepared_r6,tmp_path):
    p,cfg=prepared_r6;s=read_json(p/'prepared/SPEC.json')
    q=pd.DataFrame({'cancer_id':['C1'],'task':['pathway'],'lncrna_id':['L0'],'target_id':['P0'],'model_probability':[.8]})
    qp=tmp_path/'q.tsv';q.to_csv(qp,sep='\t',index=False)
    r=knowledge_paths(qp,cfg,tmp_path/'paths',prepared=p/'prepared')
    assert not r['primary_score_changed'] and r['paths']>=1
    paths=pd.read_csv(tmp_path/'paths/KNOWLEDGE_PATHS.tsv.gz',sep='\t')
    assert paths.path_direction.eq('UNKNOWN_NOT_INFERRED_FROM_MEMBERSHIP_OR_BINDING').all()
    assert paths.model_probability.eq(.8).all()
    result=interpret(qp,cfg,tmp_path/'cells')
    assert result['donor_correlation_rows']==0
    f=pd.read_csv(tmp_path/'cells/LNCRNA_TARGET_CELL_CONTEXT.tsv.gz',sep='\t')
    assert f.correlation.isna().all() and set(f.cell_type)=={'Tcell','malignant'}


def test_original_backbone_and_update_guard_unchanged():
    import hashlib
    baseline=ROOT/'provenance/R5_BASELINE_FILES.json'
    if not baseline.exists():pytest.skip('Baseline hash artifact generated during final packaging')
    records=read_json(baseline)
    for file in ['code/cc_hhgt/gnn.py','code/cc_hhgt/v32/optimizer_guard_v2.py']:
        assert sha(ROOT/file)==records['files'][file]['sha256']


@pytest.mark.parametrize("file", ["../x.pt", "/tmp/x.pt", "checkpoints/../../x.pt", "checkpoints/a;rm.pt", "checkpoints/a\\b.pt"])
def test_checkpoint_pointer_cannot_escape(file):
    from cla_engine.runtime_guard import validate_checkpoint_pointer
    with pytest.raises(Stop):validate_checkpoint_pointer({'file':file,'sha256':'a'*64})

def test_resume_refuses_changed_persisted_checkpoint(tmp_path):
    from cla_engine.runtime_guard import make_resume_tar
    root=tmp_path/'run';(root/'checkpoints').mkdir(parents=True)
    cp=root/'checkpoints/x.pt';cp.write_bytes(b'original')
    atomic_json(root/'LATEST.json',{'file':'checkpoints/x.pt','sha256':sha(cp),'next_step':1})
    cp.write_bytes(b'changed')
    with pytest.raises(Stop,match='changed'):make_resume_tar(root,tmp_path/'resume.tar')

def test_physical_facts_keep_tissue_and_experiment_context():
    from cc_hhgt.v32.evidence_training import build_exact_event_bags
    a={'cancer_id':'BRCA','lncrna_id':'L','partner_id':'G','partner_type':'gene',
       'source_record_id':'R','source_database':'X','pmid':'123456','experiment_type':'CLIP',
       'is_experimental':True,'is_computational':False,'is_physical':True,
       'relation_type':'binding','tissue':'tumour','cell_line':'MCF7','species':'human','experiment_id':'E1'}
    b={**a,'cell_line':'T47D','experiment_id':'E2'}
    x=build_exact_event_bags(pd.DataFrame(),pd.DataFrame([a,b]),pd.DataFrame({'pathway_id':['W'],'gene_id':['G']}))
    assert len(x.physical_facts)==2
    assert set(x.physical_facts.cell_line)=={'MCF7','T47D'}
    assert set(x.physical_facts.experiment_id)=={'E1','E2'}

def test_clinical_checkpoint_path_is_not_relative_to_code_tree(tmp_path):
    import ast
    source=(ROOT/'scripts/run_v32_clinical_training.py').read_text();tree=ast.parse(source)
    # Execute the EXACT expression emitted for metadata, with outputs external to code.
    expressions=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Dict):
            expressions.extend(value for key,value in zip(node.keys,node.values) if isinstance(key,ast.Constant) and key.value=='checkpoint_path')
    assert expressions
    cp=tmp_path/'results/fold0/best_model_state.pt';cp.parent.mkdir(parents=True);cp.write_bytes(b'synthetic')
    for expression in expressions:
        value=eval(compile(ast.Expression(expression),'<clinical metadata expression>','eval'),{'checkpoint_path':cp,'root':tmp_path/'code','str':str})
        assert Path(value).resolve()==cp.resolve()


def test_state_activity_keeps_state_role(tmp_path):
    frame=pd.DataFrame({'dataset_id':['D'],'cancer_id':['C1'],'cell_type':['Tcell'],'state_id':['RNAss'],'activity_score':[.5]})
    result=activity_table(frame,{'D'});assert result.task.tolist()==['state']
    from cla_engine.interpretation import queries
    q=tmp_path/'q.tsv';pd.DataFrame({'cancer_id':['C1'],'lncrna_id':['L'],'state_id':['RNAss']}).to_csv(q,sep='\t',index=False)
    data,_=queries(q);assert data.task.tolist()==['state']

def test_bad_activity_coverage_is_not_silent_unknown():
    frame=pd.DataFrame({'dataset_id':['D'],'cancer_id':['C1'],'cell_type':['Tcell'],'pathway_id':['W'],'activity_score':[.5],'coverage_rate':['bad']})
    with pytest.raises(Stop):activity_table(frame,{'D'})

def test_all_mismatched_mutation_patients_stop():
    with pytest.raises(Stop,match='matches'):normalize_mutation(mutation_rows(),['NEW01'],['ENSG01'],'gene','BRCA')
