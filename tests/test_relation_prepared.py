import copy,json,time
from pathlib import Path
import numpy as np
import pytest
from cla_engine import relation_prepared as rp
from cla_engine.common import atomic_json,atomic_npz,sha,digest,commit_manifest
from cla_engine.lineage import validate_prepared
from cla_engine.relation_components import CONTEXTS
from test_joint_node_omics import _model_fixture


def test_capsule_worker_preserves_targets_and_rejects_resealed_label_change(tmp_path,monkeypatch):
    spec,model,graph=_model_fixture(True)
    model.update(backend='original_pyg_cc_hhgt',learning_rate=.001,weight_decay=1e-5,direction_weight=.25)
    spec['nodes']['cancer']=['BRCA']
    b=tmp_path/'baseline';b.mkdir();atomic_json(b/'COMPLETE.json',{})
    policy=tmp_path/'POLICY.tsv';policy.write_text('bound policy fixture\n')
    s=tmp_path/'static';s.mkdir();atomic_json(s/'COMPLETE.json',{})
    atomic_json(s/'NODES.json',spec['nodes'])
    atomic_json(s/'PATHWAY_AXIS.json',{'canonical_ids':['P1','P0']})
    numeric_names=['TCGA/RNA/f'+str(i) for i in range(8)]
    base={'receipt':{'cancer':'BRCA'},'preflight':{'config':{'seed':4,'baseline':{}},
          'policy':{'path':str(policy),'sha256':sha(policy)},'components':{}},
          'axes':{'targets':{'pathway':['P0','P1'],'state':['S0','S1']},
          'state_families':{'S0':'A','S1':'B'},'feature_names':numeric_names},'arrays':{},'components':{}}
    paths={}
    for context in CONTEXTS:
        c=tmp_path/context;c.mkdir();atomic_json(c/'COMPLETE.json',{})
        base['preflight']['components'][context]={'path':str(c),'sha256':sha(c/'COMPLETE.json')}
        base['components'][context]={'splits':{'TCGA':{'source_patients':['p2'],'label_patients':['p3']}}}
        g=tmp_path/('graph_'+context);g.mkdir();atomic_json(g/'COMPLETE.json',{})
        atomic_npz(g/'graph.npz',**graph);paths[context]=str(g)
        base['arrays'][context]={}
        for task in ('pathway','state'):
            base['arrays'][context][task]={'lnc':np.array([0,1,0,1]),'target':np.array([0,0,1,1]),
                'numeric':np.zeros((4,8),np.float32),'base':np.array([.1,.2,.1,.2],np.float32),
                'label':np.array([.3,.4,.6,.7],np.float32),'available':np.ones(4,bool),'weight':np.ones(4,np.float32),
                'direction_label':np.zeros(4,np.float32),'direction_available':np.zeros(4,bool)}
    names={k:['field'+str(i) for i in range(6)] for k in spec['nodes']}
    monkeypatch.setattr(rp,'load_relation_baseline',lambda path:base)
    monkeypatch.setattr(rp,'load_relation_graph_view',lambda *args:{'ids':spec['nodes'],'relations':[list(x) for x in spec['relations']],
        'feature_names':names,'audit':{}})
    out=tmp_path/'prepared'
    rp.prepare_relation_capsule({'baseline':str(b),'static':str(s),'graph_views':paths,'model':model,
        'training':{'epochs':2,'min_epochs':2,'steps_per_epoch':3,'batch_size':4,
            'patience':3,'cpu_threads':1,'amp':False,'alpha_grid':[0.,.5,1.],
            'checkpoint_steps':3,'checkpoint_seconds':3600,'diagnostic_full_epochs':True},
        'gates':{'diagnostic_min_delta_mse':0.}},out)
    _,actual,_,_=validate_prepared(out)
    assert actual['nodes']['pathway']==['P1','P0']
    assert actual['pathway_source_ids']==['P0','P1']
    f=out/'views/BRCA__validation/pathway.npz'
    with np.load(f) as source:arr=dict(source)
    np.testing.assert_array_equal(arr['target'],[1,1,0,0])
    np.testing.assert_allclose(arr['label'],[.3,.4,.6,.7])
    from cla_engine.train import train
    from cla_engine.lineage import verify_completion
    run=tmp_path/'run'
    result=train(out,run,device='cpu',deadline=time.time()+120)
    receipt,_=verify_completion(out,run)
    assert receipt['training_progress']['optimizer_steps_total']==6
    assert receipt['task_weights']=={'pathway':1.,'state':.5}
    assert min(receipt['gradient_proof'].values())>0
    for task in ('pathway','state'):
        with np.load(run/'predictions'/('BRCA__'+task+'.npz')) as prediction:
            assert set(prediction.files)=={'lnc','target','score','baseline','direction_logit'}
    repeated=train(out,run,device='cpu',deadline=time.time()+120)
    assert repeated['state']==result['state']
    assert repeated['message']=='Verified completed run; no optimizer steps repeated'
    arr['label'][0]=.9;atomic_npz(f,**arr)
    meta=json.loads((out/'MANIFEST.json').read_text())['metadata'];commit_manifest(out,meta)
    with pytest.raises(ValueError,match='MATCHED_BASELINE_DRIFT'):validate_prepared(out)


def test_modality_groups_keep_both_cohorts_and_separate_phospho():
    names={'lncRNA':[c+'/'+n for c in ('TCGA','CPTAC') for n in ('expression_mean','expression_sd','expression_measured_fraction','local_cnv_mean','local_cnv_sd','local_cnv_callable_fraction')],
           'tcga_protein_assay':['mean','sd','measured'],'cptac_phospho_assay':['mean','sd','measured']}
    groups=rp.modality_groups(names,['TCGA/PROTEIN/rho','CPTAC/PHOSPHO/rho','CPTAC/local_cnv_mean'])
    assert groups['context_feature_groups']['cnv']['lncRNA']==[3,4,5,9,10,11]
    assert groups['numeric_feature_groups']['protein']==[0]
    assert groups['numeric_feature_groups']['phospho']==[1]
