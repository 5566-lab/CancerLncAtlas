import copy
import numpy as np
import pytest
from cla_engine.mechanism_context import add_context_graph,catalog_context

def fixture():
    ids={'gene':['ENSG1'],'protein':['P1'],'pathway':['KEGG:1'],'lncRNA':['L1']}
    maps={t:{v:i for i,v in enumerate(values)} for t,values in ids.items()}
    links=[dict(mechanism_id='TF1',pathway_source_id='KEGG:1',weight=.5),
           dict(mechanism_id='KINASE1',pathway_source_id='KEGG:1',weight=.4)]
    components=[dict(mechanism_id='TF1',uniprot_id='P1',gene_id=''),
                dict(mechanism_id='KINASE1',uniprot_id='OTHER_NAMESPACE',gene_id='ENSG1')]
    return ids,maps,links,components

def test_mechanism_context_reaches_primary_pathway_without_new_label_axis():
    ids,maps,links,components=fixture()
    primary=list(ids['pathway']);relations=[];edges={}
    audit=add_context_graph(ids,maps,relations,edges,links,components)
    assert ids['pathway']==primary
    assert set(ids['mechanism'])=={'TF1','KINASE1'}
    assert audit['mapped_mechanism_nodes']==2 and not audit['independent_loss']
    assert edges['mechanism|mechanism_context|pathway']['edge_index'].shape==(2,2)
    assert 'protein|mechanism_component|mechanism' in edges
    assert 'gene|mechanism_component|mechanism' in edges
    for src,rel,dst in relations:
        edge=edges['|'.join((src,rel,dst))]['edge_index']
        assert edge[0].max()<len(ids[src]) and edge[1].max()<len(ids[dst])

def test_no_annotation_match_is_not_silently_treated_as_present():
    ids,maps,links,components=fixture()
    for r in components:r.update(uniprot_id='unknown',gene_id='')
    with pytest.raises(ValueError,match='COVERAGE_ZERO'):
        add_context_graph(ids,maps,[],{},links,components)

def test_context_materialization_excludes_inactive_and_state_targets():
    policy=[dict(canonical_id='TF',analysis_role='context_only',active_in_analysis=True),
            dict(canonical_id='S',analysis_role='state_auxiliary',active_in_analysis=True),
            dict(canonical_id='D',analysis_role='excluded',active_in_analysis=False)]
    align=[dict(source_canonical_id=x,target_canonical_id='P',admitted_context_link='True',jaccard=.5) for x in ['TF','S','D']]
    links,components=catalog_context(policy,align,{'TF':{'U'}},{'U':{'G1','G2'}},
        {'P':dict(source_target_id='KEGG:P',network_id='N1')})
    assert {r['mechanism_id'] for r in links}=={'TF'}
    assert components==[dict(mechanism_id='TF',uniprot_id='U',gene_id='')]

@pytest.mark.parametrize("omics",[False,True])
def test_context_nodes_reach_existing_joint_encoder_without_extra_task_head(omics):
    torch=pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from test_joint_node_omics import _model_fixture
    from cla_engine.model import JointHGT,to_graph
    spec,cfg,arrays=_model_fixture(omics)
    original_pathways=list(spec["nodes"]["pathway"])
    maps={t:{v:i for i,v in enumerate(vs)} for t,vs in spec["nodes"].items()}
    added={}
    add_context_graph(spec["nodes"],maps,spec["relations"],added,
        [dict(mechanism_id="TF_CONTEXT",pathway_source_id="P0",weight=.8)],
        [dict(mechanism_id="TF_CONTEXT",uniprot_id="R0",gene_id="G0")])
    spec["feature_dims"]["mechanism"]=6
    arrays["x::mechanism"]=np.ones((1,6),np.float32)
    if omics:arrays["node_omics::mechanism"]=np.zeros((1,4),np.float32)
    for key,value in added.items():
        arrays["edge::"+key]=value["edge_index"]
        arrays["weight::"+key]=value["weight"]
    torch.manual_seed(19)
    model=JointHGT(spec,cfg).eval()
    graph=to_graph(arrays,"cpu")
    encoded=model.encode(graph,0)
    batch={"lnc":torch.tensor([0,1]),"target":torch.tensor([0,1]),
           "numeric":torch.zeros((2,8)),"base":torch.zeros(2)}
    outputs=[model.forward_task(task,encoded,batch)["logit"] for task in ["pathway","state"]]
    assert all(torch.isfinite(x).all() for x in outputs)
    # The established residual heads are initialized at zero. One optimizer
    # step learns their output weights; the next exposes encoder gradients.
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    loss=sum(torch.nn.functional.binary_cross_entropy_with_logits(x,torch.ones_like(x)) for x in outputs)
    loss.backward();optimizer.step();optimizer.zero_grad()
    encoded=model.encode(graph,0)
    loss=sum(torch.nn.functional.binary_cross_entropy_with_logits(
        model.forward_task(task,encoded,batch)["logit"],torch.ones(2)) for task in ["pathway","state"])
    loss.backward()
    assert model.context_projection["mechanism"].weight.grad.abs().sum()>0
    assert spec["nodes"]["pathway"]==original_pathways
    assert cfg["task_weights"]=={"pathway":1.,"state":.5}
