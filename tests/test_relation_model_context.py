import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('torch_geometric')
from cla_engine.model import JointHGT,to_graph
from cla_engine.common import Stop
from test_joint_node_omics import _model_fixture


def _fixture():
    spec,cfg,arrays=_model_fixture(True)
    spec.update(numeric_features=72,conservation_context_indices=list(range(72)),
        context_feature_groups={'cnv':{'lncRNA':[3,4,5,9,10,11]}},
        node_input_groups={'cnv':{'gene':[2,3]},'methylation':{'gene':[4,5]}},
        numeric_feature_groups={'cnv':[33,34,35,69,70,71]})
    spec['feature_dims']['lncRNA']=12
    arrays['x::lncRNA']=np.ones((2,12),np.float32)
    return spec,cfg,arrays


def test_cnv_ablation_removes_both_cohorts_and_gene_channel_without_mutating_input():
    spec,cfg,arrays=_fixture()
    model=JointHGT(spec,cfg).eval()
    graph=to_graph(arrays,'cpu')
    captured=[]
    hook=model.context_projection['lncRNA'].register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
    encoded=model.encode(graph,0,frozenset({'cnv'}))
    hook.remove()
    assert not captured[0][:,[3,4,5,9,10,11]].any()
    assert captured[0][:,[0,1,2,6,7,8]].eq(1).all()
    assert not encoded[2]['gene'].x[:,2:4].any()
    assert torch.equal(encoded[2]['gene'].x[:,4:6],graph['data']['gene'].x[:,4:6])
    assert graph['data']['gene'].x[:,2:4].any()
    assert graph['features']['lncRNA'].eq(1).all()


def test_all_declared_context_columns_reach_both_heads():
    spec,cfg,arrays=_fixture()
    model=JointHGT(spec,cfg).eval()
    encoded=model.encode(to_graph(arrays,'cpu'),0)
    numeric=torch.arange(144,dtype=torch.float32).reshape(2,72)
    seen=[]
    original=model.joint.forward_task
    def intercept(task,data,batch,**kwargs):
        seen.append(batch['conservation_context'].clone())
        return original(task,data,batch,**kwargs)
    model.joint.forward_task=intercept
    batch={'lnc':torch.tensor([0,1]),'target':torch.tensor([0,1]),'numeric':numeric,'base':torch.zeros(2)}
    for task in ('pathway','state'):
        out=model.forward_task(task,encoded,batch)
        assert torch.isfinite(out['logit']).all()
    assert len(seen)==2 and all(torch.equal(x,numeric) for x in seen)
    spec['conservation_context_indices']=[0,72]
    with pytest.raises(Stop) as err:JointHGT(spec,cfg)
    assert err.value.code=='CONSERVATION_CONTEXT_DIM'
