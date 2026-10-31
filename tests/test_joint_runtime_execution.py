import importlib.util
import copy,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cla_engine.common import Stop,atomic_json,atomic_npz,sha,read_json,verify_manifest,commit_manifest,lock,config
from cla_engine.statistics import pairs,design,bh
from cla_engine.tables import strict_bool,state_wide,patient_map,cnv
from cla_engine.fixture import make_fixture
from cla_engine.prepare import prepare,deterministic_pairs,load_mutation
from cla_engine.train import train,model_signature,load_checkpoint,load_npz,tensor_batch
from cla_engine.model import JointHGT,to_graph,loss_one
from cla_engine.evaluate import evaluate_fold,gate
from cla_engine.evidence import review,direction,audit
from cla_engine.transport import SSH,make_tar

@pytest.fixture(scope='session')
def experiment(tmp_path_factory):
    root=tmp_path_factory.mktemp('experiment');cfg=make_fixture(root/'sources');prep=root/'prepared';prepare(cfg,prep,0,'diagnostic')
    return root,cfg,prep

@pytest.mark.parametrize('text,expected',[('False',False),('True',True),('0',False),('1',True),(False,False),(True,True)])
def test_bool_exact(text,expected):assert strict_bool([text])[0]==expected

def test_invalid_bool():
    with pytest.raises(Stop,match='Unrecognized'):strict_bool(['maybe'])

def test_constant_cnv_valid():
    n=20;x=np.arange(n,dtype=float)[:,None];y=(x[:,0]+np.sin(x[:,0]))[:,None];z=np.ones((n,1));zero=np.zeros_like(x)
    out=pairs(x,y,zero,np.ones_like(x,bool),z,[0],[0],range(n),12)
    assert out['available'][0] and out['rank'][0]==1 and out['r'][0]>.9

def test_unmeasured_not_wildtype():
    n=20;x=np.arange(n,dtype=float)[:,None];out=pairs(x,x,np.zeros_like(x),np.zeros_like(x,bool),np.ones((n,1)),[0],[0],range(n),12)
    assert not out['available'][0] and out['n'][0]==0

def test_target_aware_covariates():
    # R6.2 contract: independent-source purity/composition remain adjustments.
    ids=[f'p{i}' for i in range(20)];f=pd.DataFrame({'patient_id':ids,'age_years':range(20),'purity':np.linspace(.1,.9,20),'purity_source':'ABSOLUTE','leukocyte_fraction':np.linspace(.9,.1,20),'stage':['Stage IIA','Stage IVB']*10,'sex':['F','M']*10})
    for col in ('cell_fraction_Macrophages.M0','cell_fraction_Macrophages.M1','cell_fraction_Macrophages.M2'):f[col]=np.linspace(.1,.3,20)
    Z,r=design(f,ids,ids[:15],family='immune');assert 'purity' in r['used_columns'] and 'leukocyte_fraction' in r['used_columns'];assert 'age_years' in r['used_columns'];assert any('stage=' in s for s in r['used_columns'])

def test_encoding_fit_train_only():
    ids=[f'p{i}' for i in range(20)];f=pd.DataFrame({'patient_id':ids,'age_years':range(20),'sex':['F','M']*10,'stage':['Stage I','Stage II']*10,'purity_source':'ABSOLUTE'})
    for col in ('purity','leukocyte_fraction','cell_fraction_Macrophages.M0','cell_fraction_Macrophages.M1','cell_fraction_Macrophages.M2'):f[col]=np.linspace(.1,.3,20)
    a,r=design(f,ids,ids[:10],family='immune');f.loc[10:,'age_years']=100000;b,r=design(f,ids,ids[:10],family='immune');assert np.array_equal(a[:10],b[:10])


def test_bh_nan():assert np.isnan(bh([.01,np.nan,.1])[1])

def test_candidate_no_duplicates():
    a,b=deterministic_pairs(list('abcde'),list('xyz'),15,1,'A');assert len(set(zip(a,b)))==15

def test_manifest_corruption(tmp_path):
    (tmp_path/'a').write_text('ok');commit_manifest(tmp_path,{})
    (tmp_path/'a').write_text('bad')
    with pytest.raises(Stop):verify_manifest(tmp_path)

def test_wrong_task_weights(tmp_path):
    c=read_json(ROOT/'config/149.json');c['model']['task_weights']={'pathway':.5,'state':.5};atomic_json(tmp_path/'config.json',c)
    with pytest.raises(Stop):config(tmp_path/'config.json')

def test_subtype_feedback_rejected(tmp_path):
    c=read_json(ROOT/'config/149.json');c['subtype_feedback']=True;atomic_json(tmp_path/'c.json',c)
    with pytest.raises(Stop):config(tmp_path/'c.json')

def test_raw_preparation(experiment):
    root,cfg,p=experiment;m=verify_manifest(p);assert m['metadata']['sealed_test_labels_included'] is False;assert not any('sealed_test' in n for n in m['files']);assert read_json(p/'TASK_CONTRACT.json')['task_weights']=={'pathway':1.,'state':.5}

def test_feature_label_patient_disjoint(experiment):
    root,cfg,p=experiment
    for f in (p/'views').glob('*/PROVENANCE.json'):
        x=read_json(f);assert not set(x['feature_folds'])&set(x['label_folds']);assert 0 not in x['feature_folds']+x['label_folds']

def test_fdr_feature_and_lasso(experiment):
    _,_,p=experiment;rec=read_json(p/'BASELINES.json');assert all(x['fdr_feature_present'] for x in rec.values());assert all(x['same_numeric_multimodal_inputs_as_deep'] for x in rec.values())

@pytest.mark.skipif(importlib.util.find_spec('torch_geometric') is None,reason='Actual PyG backend unavailable; no replacement model used')
def test_zero_residual_exact(experiment):
    _,cfg,p=experiment;spec=read_json(p/'SPEC.json');model=JointHGT(spec,cfg['model']).eval();view=read_json(p/'VIEWS.json')[0];g=to_graph(load_npz(p/'views'/view['id']/'graph.npz'),'cpu');enc=model.encode(g,view['cancer_index'])
    for t in ['pathway','state']:
        a=load_npz(p/'views'/view['id']/f'{t}.npz');b=tensor_batch(a,slice(None),'cpu');out=model.forward_task(t,enc,b);assert torch.equal(out['logit'],b['base'])

@pytest.mark.skipif(importlib.util.find_spec('torch_geometric') is None,reason='Actual PyG backend unavailable; no replacement model used')
def test_original_pyg_graph_nodes_and_edges(experiment):
    _,cfg,p=experiment;spec=read_json(p/'SPEC.json');model=JointHGT(spec,cfg['model']).eval();v=read_json(p/'VIEWS.json')[0];g=to_graph(load_npz(p/'views'/v['id']/'graph.npz'),'cpu')
    x,*_=model.encode(g,0);y,*_=model.encode(g,0,frozenset(['reported_PPI']))
    from torch_geometric.nn import HGTConv
    assert all(isinstance(layer,HGTConv) for layer in model.layers)
    assert 'cancer' in g['data'].node_types and not hasattr(model,'node')
    assert not torch.equal(x['protein'],y['protein']);assert all(torch.isfinite(a).all() for a in x.values())

@pytest.mark.skipif(importlib.util.find_spec('torch_geometric') is None,reason='Actual PyG backend unavailable; no replacement model used')
def test_resume_matches_uninterrupted(experiment):
    root,cfg,p=experiment;full=root/'full';res=root/'res'
    a=train(p,full,device='cpu',unsafe_local_debug=True);b=train(p,res,device='cpu',unsafe_local_debug=True,interrupt_after=2);assert b['state']=='PAUSED_TEST_INTERRUPT';b=train(p,res,device='cpu',unsafe_local_debug=True)
    sig=model_signature(p,cfg,read_json(p/'SPEC.json'));ca,_=load_checkpoint(full,sig);cb,_=load_checkpoint(res,sig)
    assert ca['next_step']==cb['next_step'];assert all(torch.equal(ca['model'][k],cb['model'][k]) for k in ca['model']);assert a['gradient_proof']['state_max_shared_grad']>0;assert a['gradient_proof']['pathway_max_shared_grad']>0
    assert all(k in cb for k in ['optimizer','scheduler','scaler','rng','best','next_step','gradient_proof'])
    r=evaluate_fold(p,res,root/'evaluation');assert r['tests_used_to_select_checkpoint'] is False
    # A genuinely trained shared encoder also receives membership-only gradients.
    model=JointHGT(read_json(p/'SPEC.json'),cfg['model']);model.load_state_dict(cb['model']);view=read_json(p/'VIEWS.json')[0];g=to_graph(load_npz(p/'views'/view['id']/'graph.npz'),'cpu');enc=model.encode(g,view['cancer_index']);arr=load_npz(p/'views'/view['id']/'state.npz');b=tensor_batch(arr,slice(None),'cpu');o=model.forward_task('state',enc,b);loss=loss_one(o,b,direction_weight=0.);gr=torch.autograd.grad(loss,list(model.layers[0].parameters()),allow_unused=True);assert sum(float(x.abs().sum()) for x in gr if x is not None)>0

def test_budget_required(experiment):
    root,_,p=experiment
    with pytest.raises(Stop):train(p,root/'no_budget',device='cpu')

def test_expired_budget(experiment):
    root,_,p=experiment
    with pytest.raises(Stop):train(p,root/'expired',device='cpu',deadline=time.time()-1)

def test_double_lock(tmp_path):
    if os.name=='nt':pytest.skip('POSIX child lock test')
    with lock(tmp_path):
        with pytest.raises(Stop):
            with lock(tmp_path):pass

@pytest.mark.parametrize('word',['invalidated','not validated','not approved','rejected'])
def test_evidence_negative_review(word):assert review(word)=='record_not_approved'

def test_unknown_direction():assert direction('not specified')==0

def test_source_evidence_dedup_no_broadcast(tmp_path):
    rows=[{'lncrna_id':'L1','partner_id':'G1','partner_type':'gene','source_database':'DB1','pmid':'12345678','cancer_id':'BRCA','direction':'positive','is_experimental':True,'is_predicted':False}, {'lncrna_id':'L1','partner_id':'G1','partner_type':'gene','source_database':'DB2','pmid':'12345678','cancer_id':'BRCA','direction':'positive','is_experimental':True,'is_predicted':False}]
    p=tmp_path/'raw.tsv';pd.DataFrame(rows).to_csv(p,sep='\t',index=False);r=audit(p,tmp_path/'out');assert r['counts']['deduplicated_events']==1;assert r['counts']['explicit_pathway_events']==0;assert r['review_quality_is_biological_label'] is False

def test_archive_no_self_recursion(tmp_path):
    with pytest.raises(Stop):make_tar(tmp_path,tmp_path/'out.tar')

@pytest.mark.parametrize('alias',['-oProxyCommand=x','gpu;rm -rf /','gpu with space'])
def test_ssh_alias_reject(alias):
    with pytest.raises(Stop):SSH(alias)

def test_gate_rejects_diagnostic_folds(tmp_path):
    dirs=[]
    for f in range(5):
        p=tmp_path/str(f);atomic_json(p/'EVALUATION_RECEIPT.json',{'fold':f,'profile':'diagnostic','metrics':[]});dirs.append(p)
    with pytest.raises(Stop):gate(dirs,read_json(ROOT/'config/149.json'),tmp_path/'gate')

def test_cli_help():
    p=subprocess.run([sys.executable,str(ROOT/'cla.py'),'--help'],capture_output=True,text=True);assert p.returncode==0 and 'supervise' in p.stdout and 'prepare-formal' in p.stdout

def test_corrupt_checkpoint_pointer_falls_back(experiment,tmp_path):
    from cla_engine.train import save_checkpoint
    rec=save_checkpoint(tmp_path,{'signature':'fixed','next_step':3,'tensor':torch.ones(2)})
    (tmp_path/'LATEST.json').write_text('{invalid')
    (tmp_path/'checkpoints'/'unfinished.pt.partial').write_bytes(b'incomplete')
    state,used=load_checkpoint(tmp_path,'fixed');assert state['next_step']==3;assert used['sha256']==rec['sha256']

def test_changed_checkpoint_contract_rejected(tmp_path):
    from cla_engine.train import save_checkpoint
    save_checkpoint(tmp_path,{'signature':'A','next_step':0})
    with pytest.raises(Stop):load_checkpoint(tmp_path,'B')

@pytest.mark.skipif(importlib.util.find_spec('torch_geometric') is None,reason='Actual PyG backend unavailable; no replacement model used')
def test_actual_process_sigkill_resume(experiment,tmp_path):
    if os.name=='nt':pytest.skip('Linux process preemption regression')
    import shutil,signal
    root,cfg,original=experiment;p=tmp_path/'prepared';shutil.copytree(original,p)
    c=read_json(p/'CONFIG.json');c['training'].update(epochs=3,min_epochs=3,steps_per_epoch=12,patience=10);atomic_json(p/'CONFIG.json',c)
    from cla_engine.lineage import prepare_lineage
    atomic_json(p/'LINEAGE.json',prepare_lineage(p))
    meta=read_json(p/'MANIFEST.json')['metadata'];meta['config_sha256']=__import__('cla_engine.common',fromlist=['digest']).digest(c);commit_manifest(p,meta)
    out=tmp_path/'killed';out.mkdir();atomic_json(out/'SUPERVISOR_LEASE.json',{'expires_unix':time.time()+600})
    log=(tmp_path/'child.log').open('w')
    proc=subprocess.Popen([sys.executable,str(ROOT/'cla.py'),'worker','--prepared',str(p),'--output',str(out),'--device','cpu','--deadline',str(time.time()+600)],stdout=log,stderr=log)
    deadline=time.time()+30;observed=False;lease_bound=False
    try:
        while time.time()<deadline and proc.poll() is None:
            active=out/'ACTIVE_ATTEMPT.json'
            if not lease_bound and active.exists():
                attempt=read_json(active)['attempt_id']
                atomic_json(out/'SUPERVISOR_LEASE.json',{'expires_unix':time.time()+600,'attempt_id':attempt})
                lease_bound=True
            latest=out/'LATEST.json'
            if latest.exists() and read_json(latest)['next_step']>=2:observed=True;proc.kill();break
            time.sleep(.01)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        log.close()
    assert observed,'Worker never produced an intermediate committed checkpoint'
    b=train(p,out,device='cpu',deadline=time.time()+600,unsafe_local_debug=True)
    a=train(p,tmp_path/'continuous',device='cpu',unsafe_local_debug=True)
    sig=model_signature(p,c,read_json(p/'SPEC.json'));x,_=load_checkpoint(out,sig);y,_=load_checkpoint(tmp_path/'continuous',sig)
    assert x['next_step']==y['next_step'];assert all(torch.equal(x['model'][k],y['model'][k]) for k in x['model'])

@pytest.mark.skipif(importlib.util.find_spec('torch_geometric') is None,reason='Actual PyG backend unavailable; no replacement model used')
def test_expired_supervisor_lease_stops(experiment,tmp_path):
    _,_,p=experiment;atomic_json(tmp_path/'SUPERVISOR_LEASE.json',{'expires_unix':time.time()-5})
    r=train(p,tmp_path,device='cpu',deadline=time.time()+500,lease_required=True)
    assert r['state']=='PAUSED_BUDGET_OR_SIGNAL' and r['next_step']==0

def test_code_archive_excludes_local_patient_data(tmp_path):
    from cla_engine.transport import make_code_tar
    import tarfile
    pkg=tmp_path/'pkg';(pkg/'cla_engine').mkdir(parents=True);(pkg/'config').mkdir();(pkg/'local_export').mkdir();(pkg/'cla.py').write_text('pass');(pkg/'cla_engine'/'a.py').write_text('pass');(pkg/'local_export'/'sealed.tsv').write_text('patient_secret')
    arc=tmp_path/'code.tar';make_code_tar(pkg,arc)
    with tarfile.open(arc) as t:assert t.getnames()==['cla.py','cla_engine/a.py']

def test_manifest_extra_file_rejected(tmp_path):
    (tmp_path/'a').write_text('a');commit_manifest(tmp_path,{})
    (tmp_path/'secret.tsv').write_text('not in training contract')
    with pytest.raises(Stop):verify_manifest(tmp_path)


def test_formal_admission_rejects_unbound_go_receipt(tmp_path):
    from cla_engine.queue import create_plan
    from cla_engine.common import atomic_json, Stop
    import pytest
    run=tmp_path/'run';run.mkdir()
    atomic_json(run/'TRAINING_RECEIPT.json',{'state':'GO_TO_FORMAL_EVALUATION','alpha':1.0,'config_sha256':'wrong','source_profile':'diagnostic'})
    with pytest.raises(Stop) as e:create_plan({'seed':1},tmp_path/'formal',run)
    assert e.value.code=='DIAGNOSTIC_BINDING_MISMATCH'
