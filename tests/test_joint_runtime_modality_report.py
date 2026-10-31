import numpy as np
import pytest

from cla_engine.common import Stop
from cla_engine.modality_report import feature_blocks, metric_rows, variation


def spec():
    return dict(singlecell_association_dim=28,singlecell_association_slice=[5,33],
                feature_dims={'lncRNA':18},sc_dim=1,genomics_enabled=True,numeric_features=72,
                nodes={'pathway':['P']},states=['A','B','C'],state_families={'A':'F1','B':'F1','C':'F2'})


def test_numeric_blocks_partition_and_cnv_not_mutation():
    blocks=feature_blocks(spec())
    assert blocks['singlecell_association']==list(range(5,33))
    assert blocks['lncrna_cnv']==[36,37,38]
    assert blocks['partner_mutation']==list(range(39,51))
    assert blocks['lncrna_mutation']==[53,54,55,56]
    assert blocks['pathway_cnv']==[69,70,71]
    assert sorted(i for cols in blocks.values() for i in cols)==list(range(72))


def test_unknown_schema_fails_closed():
    s=spec(); s['numeric_features']=73
    with pytest.raises(Stop,match='layout'): feature_blocks(s)


def test_constant_values_are_not_called_measured_or_beneficial():
    values=np.full((4,2),-0.5)
    assert variation(values,values,[0,1])['status']=='CONSTANT_OR_UNAVAILABLE'
    assert variation(values,values,[])['status']=='NOT_IN_NUMERIC_INPUT'


def test_state_metric_preserves_family_macro_and_delta_sign():
    data=dict(label=np.tile([0,1],3),available=np.ones(6,bool),
              cancer=np.repeat('X',6),target_name=np.repeat(['A','B','C'],2))
    full=np.array([0,1,0,1,0,1]); removed=np.array([0,1,0,1,1,0])
    rows=metric_rows('state',data,spec(),full,removed,'cnv','TEST',k=1)
    overall=next(row for row in rows if row['level']=='overall')
    assert overall['full_auprc']==1
    assert overall['removed_auprc']==0.75
    assert overall['delta_auprc']==0.25
    assert overall['delta_precision_at_k']==0.5
    assert len([r for r in rows if r['level']=='family'])==2


def test_single_class_target_is_unavailable_not_zero_gain():
    data=dict(label=np.array([1,1]),available=np.ones(2,bool),cancer=np.repeat('X',2),target_name=np.repeat('P',2))
    rows=metric_rows('pathway',data,spec(),np.ones(2),np.zeros(2),'cnv','TEST')
    assert all(not row['available'] for row in rows)
    assert all('delta_auprc' not in row for row in rows)


def test_inventory_mode_never_fits_or_loads_checkpoint(monkeypatch,tmp_path):
    from cla_engine import modality_report as module
    from cla_engine.common import atomic_json, read_json
    p=tmp_path/'prepared'; p.mkdir(); atomic_json(p/'MANIFEST.json',{})
    monkeypatch.setattr(module.socket,'gethostname',lambda:'149')
    monkeypatch.setattr(module,'validate_prepared',lambda p: ({},spec(),{},{}))
    monkeypatch.setattr(module,'read_json',lambda p: [])
    monkeypatch.setattr(module,'_assemble',lambda *a: {k:{'numeric':np.zeros((2,72))} for k in ('train','validation')})
    def forbidden(*args,**kwargs): raise AssertionError('Training/inference was not requested')
    monkeypatch.setattr(module,'_fit_ablation',forbidden)
    monkeypatch.setattr(module,'_checkpoint_model',forbidden)
    out=tmp_path/'report'; module.report(p,out)
    contract=read_json(out/'REPORT_CONTRACT.json')
    assert contract['lasso_refits']=='NOT_RUN_WAITING_CONFIRMATION'
    assert not contract['neural_training_started']
    assert (out/'MODALITY_CONTRIBUTIONS.tsv').read_text().count('\n')==1


@pytest.mark.parametrize('port,node,selected',[
    ('lncrna_cnv','lncRNA',slice(3,6)),('pathway_cnv','pathway',slice(18,21)),
    ('partner_mutation','lncRNA',slice(6,18)),('pathway_mutation','pathway',slice(6,18))])
def test_frozen_mask_is_local_and_does_not_mutate_graph(port,node,selected):
    import torch
    from torch import nn
    from torch_geometric.data import HeteroData
    from cla_engine.model import JointHGT
    class Encoder(nn.Module):
        def encode(self,data): return data
    model=JointHGT.__new__(JointHGT); nn.Module.__init__(model)
    model.node_input_dim=2  # Explicit architecture for this constructor-bypassing test double.
    model.spec={'nodes':{'lncRNA':['L'],'pathway':['P']},'genomics_enabled':True,'singlecell_role':'interpretation_only'}
    model.modal_gate=nn.Parameter(torch.ones(1,2));model.global_modal_gate=nn.Parameter(torch.ones(2))
    model.context_projection=nn.ModuleDict({'lncRNA':nn.Identity(),'pathway':nn.Identity()})
    model.mutation_projection=nn.Linear(3,18,bias=False)
    model.joint=nn.Module();model.joint.encoder=Encoder()
    data=HeteroData();data['lncRNA'].x=torch.zeros(1,2);data['pathway'].x=torch.zeros(1,2)
    graph={'data':data,'features':{'lncRNA':torch.ones(1,18),'pathway':torch.ones(1,21)},
           'mutation':torch.zeros(1,3),'mutation_available':torch.zeros(1)}
    normal=model.encode(graph,0)[2]; masked=model.encode(graph,0,frozenset([port]))[2]
    for kind in graph['features']:
        expected=torch.ones_like(graph['features'][kind])
        if kind==node: expected[:,selected]=0
        torch.testing.assert_close(masked[kind].r4_context_embedding,expected)
        torch.testing.assert_close(normal[kind].r4_context_embedding,torch.ones_like(expected))
        torch.testing.assert_close(graph['features'][kind],torch.ones_like(expected))
