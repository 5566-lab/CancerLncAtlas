import json
import numpy as np
import pandas as pd
import pytest
from cla_engine.semantic_context import (ContextTask,compile_routes,select_routes,
    fit_reference_scaler,linear_context_design,add_score_graph)
from cla_engine.measured_context_models import measured_m0_design,measured_m0_forward


def fixture():
    bindings=pd.DataFrame([dict(score_id='wnt',source_scope='WNT',input_layers='["RNA"]',
        lineage_ids='["wnt"]',binding_evidence='reviewed ID')])
    scope=pd.DataFrame([dict(source_scope='WNT',target_id='P0',relation='broader_scope_context',
        evidence='KEGG:N00056',evidence_basis='reviewed_perturbation_scope',jaccard=0.01)])
    routes=compile_routes(bindings,scope,['P0','P1'])
    values=pd.DataFrame({'wnt':[0.,2.,4.]},index=['a','b','c'])
    scaler=fit_reference_scaler(values,['a','b'],['c'])
    task=ContextTask('P0','protein_y',frozenset({'PROTEIN'}),frozenset({'RNA'}),frozenset())
    return dict(bindings=bindings,routes=routes,values=values,scaler=scaler),task,scope


def test_semantics_independent_of_membership_and_no_copied_y():
    c,t,scope=fixture();other=scope.copy();other.jaccard=1.
    pd.testing.assert_frame_equal(c['routes'],compile_routes(c['bindings'],other,['P0','P1']))
    assert not c['routes'].copy_to_target_y.any()
    assert not c['routes'].independent_loss_enabled.any()
    assert set(c['routes'].target_id)=={'P0'}
    other.evidence_basis='protein_set_enrichment'
    with pytest.raises(ValueError,match='UNSUPPORTED_SEMANTIC'):compile_routes(c['bindings'],other,['P0'])


def test_duplicate_score_rejected_and_multiple_routes_store_one_column():
    c,t,scope=fixture()
    with pytest.raises(ValueError,match='DUPLICATE'):compile_routes(pd.concat([c['bindings']]*2),scope,['P0'])
    extra=scope.copy();extra.target_id='P1'
    routes=compile_routes(c['bindings'],pd.concat([scope,extra]),['P0','P1'])
    assert len(routes)==2 and c['values'].shape[1]==1


def test_target_layer_and_lineage_exclusion_and_l_only_baseline():
    c,t,_=fixture()
    assert len(select_routes(c['bindings'],c['routes'],t))==1
    for task in [ContextTask('P0','rna_y',frozenset({'RNA'}),frozenset({'RNA'}),frozenset()),
                 ContextTask('P0','protein_y',frozenset({'PROTEIN'}),frozenset(),frozenset()),
                 ContextTask('P0','protein_y',frozenset({'PROTEIN'}),frozenset({'RNA'}),frozenset({'wnt'}))]:
        assert select_routes(c['bindings'],c['routes'],task).empty
    c['bindings']['input_layers']='["UNRESOLVED"]'
    assert select_routes(c['bindings'],c['routes'],t).empty


def test_heldout_never_changes_scaler_and_missing_is_distinct_from_zero():
    c,t,_=fixture();changed=c['values'].copy();changed.loc['c','wnt']=999
    pd.testing.assert_frame_equal(c['scaler'],fit_reference_scaler(changed,['a','b'],['c']))
    c['values'].loc['c','wnt']=np.nan
    design,_=linear_context_design(**c,task=t)
    assert design.loc['a','observed::wnt']==1
    assert design.loc['c','observed::wnt']==0 and design.loc['c','score::wnt']==0
    with pytest.raises(ValueError,match='PARTITION'):fit_reference_scaler(changed,['a','c'],['c'])


def test_m0_uses_numeric_context_and_clinical_covariates_are_unpenalized():
    c,t,_=fixture();rna=pd.DataFrame({'L1':[1.,2.,3.]},index=c['values'].index)
    cov=pd.DataFrame({'purity':[.7,.6,.8]},index=rna.index)
    design,penalty,_=measured_m0_design(rna,cov,c,t)
    assert penalty['rna::L1']==1 and penalty['covariate::purity']==0
    beta=pd.Series(0.,index=design.columns);beta['score::wnt']=2.
    y=measured_m0_forward(design,beta)
    assert y.loc['c']-y.loc['b']==4.


def test_real_original_hhgt_receives_score_values_and_loses_effect_without_edge():
    torch=pytest.importorskip('torch');pytest.importorskip('torch_geometric')
    from cla_engine.model import JointHGT,to_graph
    from test_joint_node_omics import _model_fixture
    c,t,_=fixture();spec,cfg,arrays=_model_fixture(False)
    torch.manual_seed(37)
    extended,a=add_score_graph(spec,arrays,**c,task=t,patient_id='c')
    model=JointHGT(extended,cfg).eval()
    graph=to_graph(a,'cpu');graph['features']['source_score'].requires_grad_()
    h=model.encode(graph,0)[0]['pathway'];h[0].sum().backward()
    assert graph['features']['source_score'].grad[0,0].abs()>0
    changed=dict(a);changed['x::source_score']=a['x::source_score'].copy();changed['x::source_score'][0,0]+=3.
    with torch.no_grad():
        h2=model.encode(to_graph(changed,'cpu'),0)[0]['pathway']
        assert not torch.allclose(h[0],h2[0])
        h3=model.encode(to_graph(a,'cpu'),0,frozenset({'score_context'}))[0]['pathway']
        h4=model.encode(to_graph(changed,'cpu'),0,frozenset({'score_context'}))[0]['pathway']
        torch.testing.assert_close(h3,h4)
    assert extended['score_context_contract']['independent_loss'] is False
    assert cfg['task_weights']=={'pathway':1.,'state':.5}
    assert 'source_score' not in spec['nodes']


def test_missing_or_forbidden_scores_have_no_graph_edges():
    pytest.importorskip('torch');pytest.importorskip('torch_geometric')
    from test_joint_node_omics import _model_fixture
    c,t,_=fixture();spec,cfg,a=_model_fixture(False)
    c['values'].loc['c','wnt']=np.nan
    _,g=add_score_graph(spec,a,**c,task=t,patient_id='c')
    assert not g['x::source_score'].any()
    assert all(v.shape[1]==0 for k,v in g.items() if k.startswith('edge::source_score'))


def test_unresolved_target_dependencies_are_rejected():
    with pytest.raises(ValueError,match='UNRESOLVED_TARGET'):
        ContextTask('P0','y',frozenset({'UNRESOLVED'}),frozenset({'RNA'}),frozenset())


def test_scaler_roundtrip_with_all_missing_and_constant_columns(tmp_path):
    from cla_engine.semantic_context import transformed_context
    values=pd.DataFrame({'empty':[np.nan]*3,'constant':[2.]*3,'varying':[0.,2.,4.]},index=['a','b','c'])
    scaler=fit_reference_scaler(values,['a','b'],['c']);scaler.index.name='score_id'
    path=tmp_path/'scaler.tsv';scaler.to_csv(path,sep='\t')
    loaded=pd.read_csv(path,sep='\t').set_index('score_id')
    z,mask=transformed_context(values,loaded)
    assert np.isfinite(z.to_numpy()).all() and not mask['empty'].any() and not mask['constant'].any()
    assert z.loc['c','varying']==3.


def test_old_overlap_scope_edges_are_removed_but_physical_memberships_survive():
    from cla_engine.semantic_context import semantic_graph_base
    rels=[('mechanism','mechanism_context','pathway'),('protein','mechanism_component','mechanism')]
    spec={'relations':rels};arrays={}
    for r in rels:
        arrays['edge::'+'|'.join(r)]=np.array([[0],[0]])
        arrays['weight::'+'|'.join(r)]=np.ones(1)
    s,a=semantic_graph_base(spec,arrays)
    assert s['relations']==[rels[1]] and len(s['excluded_overlap_scope_relations'])==1
    assert 'edge::'+'|'.join(rels[0]) not in a and 'edge::'+'|'.join(rels[1]) in a


def test_score_as_y_is_predicted_once_and_loss_reaches_original_pathway_encoder():
    torch=pytest.importorskip('torch');pytest.importorskip('torch_geometric')
    from test_joint_node_omics import _model_fixture
    from cla_engine.model import JointHGT,to_graph
    from cla_engine.measured_context_models import score_target_graph,selected_score_loss
    from cla_engine.score_observation_head import ScoreObservationHead
    c,t,_=fixture();spec,cfg,arrays=_model_fixture(False)
    more=c['routes'].copy();more.target_id='P1'
    routes=pd.concat([c['routes'],more])
    s,a=score_target_graph(spec,arrays,c['bindings'],routes,['wnt'])
    assert s['nodes']['score_target']==['wnt'] and not a['x::score_target'].any()
    torch.manual_seed(73)
    model=JointHGT(s,cfg).eval();head=ScoreObservationHead(cfg['hidden'],['wnt'])
    graph=to_graph(a,'cpu');graph['features']['pathway'].requires_grad_()
    pred=head(model.encode(graph,0),s)
    loss=selected_score_loss(pred,torch.tensor([2.]),torch.tensor([True]),['wnt'])
    torch.testing.assert_close(loss,(pred[0]-2.).square())
    loss.backward()
    assert graph['features']['pathway'].grad.abs().sum()>0
    assert selected_score_loss(pred,pred,torch.tensor([True]),[]) is None
    with pytest.raises(ValueError,match='DUPLICATE_SUPERVISION'):
        selected_score_loss(pred,pred,torch.tensor([True]),['wnt','wnt'])


def test_same_scope_scores_have_separate_calibration_heads():
    torch=pytest.importorskip('torch')
    from cla_engine.score_observation_head import ScoreObservationHead
    head=ScoreObservationHead(4,['protein','rna'])
    with torch.no_grad():head.weight.zero_();head.bias.copy_(torch.tensor([1.,5.]))
    actual=head({'score_target':torch.ones(2,4)},{'nodes':{'score_target':['protein','rna']}})
    torch.testing.assert_close(actual,torch.tensor([1.,5.]))
