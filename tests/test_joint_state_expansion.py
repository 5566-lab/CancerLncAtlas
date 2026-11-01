import copy
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cla_engine.common import Stop,read_json,sha
from cla_engine.state_catalog import CATALOG_ID,expanded_targets
from cla_engine.state_materialize import select_patient_values,verify_sbs
from cla_engine.execution_contract import validate_config_contract
from cc_hhgt.v32.adjustment_policy import default_adjustment_contract,effective_policy

ROOT=Path(__file__).resolve().parents[1]

def test_complete_reviewed_catalog_and_measurement_separation():
    cfg=read_json(ROOT/'config/149.json');validate_config_contract(cfg)
    catalog={t['id']:t for t in cfg['targets']}
    assert len(catalog)==483 and sum(t['role']=='primary' for t in catalog.values())==482
    assert {t['id'] for t in catalog.values() if t['role']=='auxiliary'}=={'ESTIMATEScore'}
    for target in ('EXTEND','RNAss','DNAss','SenMayo','OXPHOS','IPS','TMB','HRD_LOH','HRD_TAI','HRD_LST','MSI','Aneuploidy','ABSOLUTE_ploidy','TumorPurity','SBS::SBS2','SBS::SBS13'):
        assert catalog[target]['role']=='primary'
    assert catalog['signifinder::CIN_Carter_70']['measurement']=='RNA_EXPRESSION'
    assert catalog['Aneuploidy']['measurement']=='DNA_COPY_NUMBER'
    assert catalog['signifinder::DNArep_Kang']['family']!=catalog['HRD_LOH']['family']
    assert sum(t['source_kind']=='sbs' for t in catalog.values())==60
    assert sum(t['family']=='cell_composition' for t in catalog.values())==96
    assert not any(t.endswith(('::P.value','::RMSE','::Correlation')) for t in catalog)
    bad=copy.deepcopy(cfg);bad['targets'][-1]['column']='RNA_FAKE_SBS'
    with pytest.raises(Stop) as e:validate_config_contract(bad)
    assert e.value.code=='TARGET_ROLE_CATALOG_DRIFT'

def test_composition_self_adjustment_and_source_specific_purity():
    contract=default_adjustment_contract([t['id'] for t in expanded_targets()])
    for m in ('M0','M1','M2'):
        _,p=effective_policy('cibersort_relative::Macrophages.'+m,contract=contract,purity_families=['ABSOLUTE'])
        assert 'cell_fraction_Macrophages.'+m in p['excluded']
        assert 'cell_fraction_Macrophages.'+m not in p['required']
        assert 'leukocyte_fraction' in p['required']
    _,p=effective_policy('ABSOLUTE_purity',contract=contract,purity_families=['ABSOLUTE'])
    assert 'purity' in p['excluded']
    _,p=effective_policy('TumorPurity',contract=contract,purity_families=['ABSOLUTE'])
    assert 'purity' in p['required']

def authority():
    return pd.DataFrame([{'patient_id':'TCGA-AA-0001','sample_id':'TCGA-AA-0001-01A-11R-AAAA-01','cancer_id':'ACC'},
        {'patient_id':'TCGA-AA-0002','sample_id':'TCGA-AA-0002-01A-11R-AAAA-01','cancer_id':'ACC'}])

def test_patient_values_preserve_tumor_records_and_missingness():
    f=pd.DataFrame([{'sample_id':'TCGA-AA-0001-11','x':100.,'y':20.},
        {'sample_id':'TCGA-AA-0001-01','x':4.,'y':np.nan},
        {'sample_id':'TCGA-AA-0001-06','x':6.,'y':8.},
        {'sample_id':'TCGA-AA-0002','x':9.,'y':np.nan}])
    scores,choices,excluded,conflicts=select_patient_values(f,authority(),['x','y'],'historical')
    assert scores.loc['TCGA-AA-0001','x']==4 and scores.loc['TCGA-AA-0001','y']==8
    assert scores.loc['TCGA-AA-0002','x']==9 and np.isnan(scores.loc['TCGA-AA-0002','y'])
    assert excluded==[{'source':'historical','sample_id':'TCGA-AA-0001-11','reason':'NON_TUMOR_SAMPLE'}]
    assert len(conflicts)==1 and not any(c['sample_id'].endswith('-11') for c in choices)

@pytest.mark.parametrize('change,code',[
    ({'patient_id':'TCGA-AA-9999'},'STATE_SOURCE_PATIENT_CONFLICT'),
    ({'cancer_id':'BRCA'},'STATE_SOURCE_CANCER_CONFLICT'),
    ({'x':float('inf')},'STATE_INVALID_NUMERIC'),
    ({'x':'unmeasured'},'STATE_INVALID_NUMERIC'),
])
def test_incompatible_source_identity_and_values_are_rejected(change,code):
    f=pd.DataFrame([{'sample_id':'TCGA-AA-0001-01','x':1.,**change}])
    with pytest.raises(Stop) as e:select_patient_values(f,authority(),['x'],'historical')
    assert e.value.code==code

def test_sbs_unavailable_never_becomes_zero():
    cols=[t['column'] for t in expanded_targets() if t['source_kind']=='sbs']
    scores=pd.DataFrame([[1/60]*60,[np.nan]*60],columns=cols);scores.insert(0,'patient_id',['p1','p2'])
    qc=pd.DataFrame({'patient_id':['p1','p2'],'snv_count':[100,49],'reconstruction_cosine':[.95,np.nan],'state_available':[True,False]})
    receipt={'state':'COMPLETE','targets':cols,'reference':'COSMIC_v3.2_GRCh37','minimum_snv_count':50,'minimum_reconstruction_cosine':.8,'artifact_signatures_included':False}
    result=verify_sbs(scores,qc,receipt);assert result.loc['p2'].isna().all()
    scores.loc[1,cols]=0.
    with pytest.raises(Stop) as e:verify_sbs(scores,qc,receipt)
    assert e.value.code=='STATE_SBS_MASK_INVALID'

def test_all_expanded_targets_enter_prepared_and_shared_gradient(tmp_path):
    import torch
    from cla_engine.fixture import make_fixture
    from cla_engine.prepare import prepare
    from cla_engine.lineage import validate_prepared
    from cla_engine.train import load_npz,tensor_batch
    from cla_engine.model import JointHGT,to_graph,loss_one
    torch.set_num_threads(1)
    cfg=make_fixture(tmp_path/'input',expanded_states=True)
    cfg['cancer_type']='list';cfg['cancer_list']=['C1']
    # An unmeasured genomic target must stay in the catalog with unavailable rows.
    path=Path(cfg['sources']['state']['path']);f=pd.read_csv(path,sep='\t');f['SBS::SBS3']=np.nan;f.to_csv(path,sep='\t',index=False)
    cfg['sources']['state'].update(sha256=sha(path),bytes=path.stat().st_size)
    out=tmp_path/'prepared';prepare(cfg,out)
    _,spec,stored,_=validate_prepared(out)
    assert spec['state_catalog']==CATALOG_ID and len(spec['states'])==197
    assert spec['cancers']==['C1'] and stored['model']['task_weights']=={'pathway':1.,'state':.5}
    view=out/'views/C1__train2';a=load_npz(view/'state.npz')
    absent=spec['states'].index('SBS::SBS3')
    assert np.any(a['target']==absent) and not a['available'][a['target']==absent].any()
    model=JointHGT(spec,cfg['model']);model.train()
    assert model.joint.state_embedding.num_embeddings==197
    graph=to_graph(load_npz(view/'graph.npz'),'cpu');encoded=model.encode(graph,0)
    # Exercise newly added RNA, genomic, composition and formerly auxiliary State IDs.
    new_ids=['SenMayo','TMB','SBS::SBS2','cibersort_relative::Macrophages.M0','ABSOLUTE_ploidy']
    positions=[]
    for target in new_ids:
        hit=np.flatnonzero((a['target']==spec['states'].index(target)) & a['available'])
        assert len(hit),target
        positions.append(hit[0])
    batch=tensor_batch(a,np.asarray(positions),'cpu')
    loss=loss_one(model.forward_task('state',encoded,batch),batch,cfg['model']['direction_weight'])
    assert torch.isfinite(loss)
    before=model.joint.state_embedding.weight.detach().clone()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    optimizer.zero_grad();(.5*loss).backward()
    assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.encoder.parameters() if p.requires_grad)
    optimizer.step()
    for target in new_ids:
        i=spec['states'].index(target);assert not torch.equal(before[i],model.joint.state_embedding.weight[i])
