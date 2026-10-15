"""Actual V4 loaders and frozen forwards on tiny graphs, plus analytic estimands."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.current_core_export import sha256, CHECKPOINT_FORMAT
from cc_hhgt.v32.current_core_prediction import (candidate_table, prediction_batches, stream_predictions,
    materialize_current_predictions, load_current_predictions, CHUNK_ESTIMAND)


def record(path): return {'path':str(path),'sha256':sha256(path)}


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No optimizer or fitting during prediction tests')
    for name in ('Adam','AdamW','SGD'):monkeypatch.setattr(torch.optim,name,forbidden)
    import cc_hhgt.v32.training as training
    monkeypatch.setattr(training,'_batch_loss',forbidden)


class AnalyticModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.scale=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float64));self.calls=[]
    @property
    def encoder(self):return self
    def encode(self,graph):self.calls.append(graph);return graph
    def forward(self,graph,candidate,base,context,available,*,admitted,encoded):
        raw=self.scale*encoded
        gate=torch.full_like(base,.8)
        residual=torch.where(available,gate*torch.tanh(raw),torch.zeros_like(base))
        return {'final_logit':base+residual,'direction_logit':torch.full_like(base,encoded*2),
            'graph_residual':residual,'graph_gate':gate}


def tiny_features():
    batch={'candidate_batch':{'l':torch.tensor([1,0]),'p':torch.tensor([0,0]),'c':torch.tensor([0,0])},
        'base_logit':torch.tensor([2.,-.4],dtype=torch.float64),'conservation_context':torch.ones((2,4),dtype=torch.float64),
        'graph_available':torch.tensor([True,False]),'proxy_label':object(),'direction_label':object()}
    bundle=SimpleNamespace(node_maps={'lncRNA':{'L0':0,'L1':1},'pathway':{'P1':0},'cancer':{'BRCA':0}})
    payload={'bundle':bundle,'train_batches':[batch],'validation_batches':[copy.deepcopy(batch)]}
    candidates=pd.DataFrame({'cancer_id':['BRCA','BRCA'],'lncrna_id':['L0','L1'],'pathway_id':'P1'})
    return payload,candidates


def test_chunk_logits_precede_sigmoid_and_candidate_order_is_restored_without_label_reads():
    payload,candidates=tiny_features();batches=prediction_batches(payload,candidates);model=AnalyticModel()
    frame,audit=stream_predictions(model,payload['bundle'],batches,(-2,1),{-2:.25,1:.75},lambda b,c,device:c,candidate_rows=2)
    sigmoid=lambda x:1/(1+np.exp(-x))
    expected=2+.8*(.25*np.tanh(-2)+.75*np.tanh(1))
    assert frame.association_membership_probability.tolist()==pytest.approx([sigmoid(-.4),sigmoid(expected)])
    wrong=.25*sigmoid(2+.8*np.tanh(-2))+.75*sigmoid(2+.8*np.tanh(1))
    assert abs(frame.association_membership_probability.iloc[1]-wrong)>.001
    assert frame.association_direction_probability.eq(sigmoid(.5)).all()
    assert frame.graph_residual.iloc[0]==0 and frame.l1_probability.iloc[0]==frame.association_membership_probability.iloc[0]
    assert model.calls==[-2,1] and audit['encoder_forward_calls']==2 and not audit['optimizer_started']
    assert model.scale.grad is None and not model.scale.requires_grad
    assert audit['chunk_estimand']==CHUNK_ESTIMAND


@pytest.mark.parametrize('fault',['feature','indices','duplicate','missing','extra','sealed','nan','availability'])
def test_full_candidate_feature_integrity_before_any_forward(fault):
    payload,candidates=tiny_features()
    if fault=='feature':payload['validation_batches'][0]['base_logit'][0]=99
    if fault=='indices':payload['validation_batches'][0]['candidate_batch']['l'][0]=0
    if fault=='duplicate':
        for side in ('train_batches','validation_batches'):payload[side][0]['candidate_batch']['l'][0]=0
    if fault=='missing':candidates=pd.concat([candidates,candidates.iloc[[0]].assign(lncrna_id='L2')],ignore_index=True)
    if fault=='extra':
        for side in ('train_batches','validation_batches'):payload[side][0]['candidate_batch']['replication_effect']=torch.ones(2)
    if fault=='sealed':payload['test_batches']=[]
    if fault=='nan':
        for side in ('train_batches','validation_batches'):payload[side][0]['conservation_context'][0,0]=float('nan')
    if fault=='availability':
        for side in ('train_batches','validation_batches'):payload[side][0]['graph_available']=torch.ones(2)
    with pytest.raises(ValueError):prediction_batches(payload,candidates)


def current_fixture(root):
    from test_v32_current_binding_expansion import write_preparation_inputs
    from scripts.prepare_v32_current_binding_comparison import load_current_fold
    from cc_hhgt.v32.training import _prepared_input_authority_hashes, _runtime_chunk_contract
    from cc_hhgt.gnn import build_model
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual
    root.mkdir()
    source=json.loads(write_preparation_inputs(root).read_text())
    candidates=pd.DataFrame({'cancer_id':['BRCA'],'lncrna_id':['LNC:Z'],'pathway_id':['P1'],'pathway_family_id':['F1']})
    path=root/'candidates.parquet';candidates.to_parquet(path,index=False)
    scope=root/'scope.parquet';candidates[['cancer_id','lncrna_id']].assign(shared_or_local_scope='shared').to_parquet(scope,index=False)
    manifest={'format':'CURRENT_CORE_PREDICTION_INPUT_V1','core_run_id':'CURRENT-V4-SYNTHETIC','graph_variant':'G2',
        'candidate_authority':record(path),'lncrna_scope':record(scope),'fold_inputs':[]}
    for item in source['fold_inputs']:
        fold=item['fold'];payload,_=load_current_fold(item,root)
        torch.manual_seed(fold+17)
        encoder=build_model('cc_hhgt',payload['bundle'],2,payload['legacy_model_config'])
        model=build_v32_cc_hhgt_residual(encoder,hidden_channels=8,context_features=4,dropout=0.)
        with torch.no_grad():model.residual_output.bias.fill_(.5+fold*.1)
        hashes={'config_sha256':source['config']['sha256'],'input_manifest_sha256':item['input_manifest']['sha256'],'code_sha256':'f'*64}
        checkpoint={'checkpoint_format':CHECKPOINT_FORMAT,'model_state':model.state_dict(),'patient_fold':fold,
            'task_id':f'CURRENT-V4-SYNTHETIC-PF{fold}','graph_variant':'G2','artifact_hashes':hashes,
            'input_authority_hashes':_prepared_input_authority_hashes(payload),'prepared_artifact_sha256':item['prepared']['sha256'],'cycle':0}
        path=root/f'checkpoint{fold}.pt';torch.save(checkpoint,path)
        chunks,weights=_runtime_chunk_contract(payload['bundle'])
        receipt={k:v for k,v in checkpoint.items() if k!='model_state'}
        receipt.update(status='SUCCESS',run_id=manifest['core_run_id'],best_model_state_sha256=sha256(path),
            optimizer_steps=1,parameter_delta_positive=True,validation_full_runtime_coverage=True,
            runtime_chunk_weights={str(c):weights[c] for c in chunks},runtime_num_chunks=len(chunks),
            synthetic_serialization_fixture=True,no_optimizer_actually_run=True)
        success=root/f'training{fold}.json';success.write_text(json.dumps(receipt))
        manifest['fold_inputs'].append({**item,'config':source['config'],'checkpoint':record(path),'training_receipt':record(success)})
    path=root/'CURRENT_INPUT.json';path.write_text(json.dumps(manifest));return path


def test_actual_v4_loaders_all_chunks_existing_cli_and_selected_geneset_consumer(tmp_path):
    from scripts.infer_v32_release_fold import main
    from cc_hhgt.v32.selected_genesets import materialize_selected_analysis,SelectedAnalysisQuery
    from cc_hhgt.v32.external_validation_current import producer_scores
    manifest=current_fixture(tmp_path/'inputs');output=tmp_path/'predictions'
    assert main(['--current-manifest',str(manifest),'--expected-core-run-id','CURRENT-V4-SYNTHETIC','--output',str(output)])==0
    path=output/'CURRENT_PREDICTION_BINDING.json';scores,binding=load_current_predictions(path,expected_sha=sha256(path))
    assert len(scores)==1 and scores.association_membership_probability.gt(.5).all()
    assert 'ridge_probability' not in scores and not binding['ridge_available']
    assert not binding['test_payloads_loaded'] and not binding['metrics_computed']
    assert all(r['encoder_forward_calls']==2 and r['core_unchanged_after_prediction'] for r in binding['ensemble_source_records'])
    assert len({r['core_parameter_sha256'] for r in binding['ensemble_source_records']})==5
    expected=np.mean([pd.read_parquet(r['prediction_path']).association_membership_probability.iloc[0] for r in binding['ensemble_source_records']])
    assert scores.association_membership_probability.iloc[0]==expected
    spec={'model_id':'current','run_id':'CURRENT-V4-SYNTHETIC','producer_kind':'current_core_release',
        'producer_binding':record(path),'score_column':'association_membership_probability'}
    read,source=producer_scores(spec,tmp_path)
    np.testing.assert_array_equal(read.score,scores.association_membership_probability)
    selected={'format':'CURRENT_SELECTED_ANALYSIS_INPUT_V1','selected_model':spec,'reference_exact_model':spec,
        'selection':{'kind':'EXPLICIT_SOURCE_REGISTRATION','rationale':'Synthetic current checkpoint route; no winner claim'},
        'geneset':{'min_members':1,'max_members':1,'score_min':.1},
        'subtype':{'score_column':'selected_score','min_members':1,'min_cancers':2,'bootstrap_replicates':2}}
    selected_path=tmp_path/'SELECTED.json';selected_path.write_text(json.dumps(selected))
    result=materialize_selected_analysis(selected_path,tmp_path/'selected')
    query=SelectedAnalysisQuery(result['binding_path'],expected_sha=result['binding_sha256'])
    members=query.query()['rows']
    json.dumps(query.query())
    assert len(members)==1 and members[0]['selected_score']==scores.association_membership_probability.iloc[0]
    assert members[0]['score_source_run_id']=='CURRENT-V4-SYNTHETIC'
    assert Path(query.download('gmt')['path']).read_text().rstrip().endswith('LNC:Z')
    assert all(name not in members[0] for name in ('proxy_label','held_out_proxy_label','direction_label'))
    from scripts.audit_v32_current_candidates import main as audit_main
    assert audit_main(['--binding',str(path),'--expected-sha256',sha256(path),
        '--output',str(tmp_path/'REAUDIT.json'),'--replay-models'])==0
    assert json.loads((tmp_path/'REAUDIT.json').read_text())['frozen_checkpoints_replayed']
    # Rehashing an altered public ensemble cannot bypass actual fold recomputation.
    prediction=Path(binding['prediction']['path']);scores.loc[0,'association_membership_probability']=.99;scores.to_parquet(prediction,index=False)
    binding['prediction']['sha256']=sha256(prediction);path.write_text(json.dumps(binding))
    (output/'SUCCESS.json').write_text(json.dumps({'status':'PASS','binding_sha256':sha256(path)}))
    with pytest.raises(ValueError,match='fold ensemble'):load_current_predictions(path,expected_sha=sha256(path))
    # Even a consistently rewritten fold + ensemble remains inconsistent with
    # the actual selected checkpoint. The explicit deeper replay detects it.
    from cc_hhgt.v32.current_core_prediction import ensemble_predictions,audit_current_predictions
    item=binding['ensemble_source_records'][0];fold_path=Path(item['prediction_path'])
    changed=pd.read_parquet(fold_path);changed['association_membership_probability']=.99;changed.to_parquet(fold_path,index=False)
    item['prediction_sha256']=sha256(fold_path)
    candidate=candidate_table(json.loads(manifest.read_text()),manifest.parent)
    ensemble=ensemble_predictions([r['prediction_path'] for r in binding['ensemble_source_records']],candidate,binding['training_run_id'])
    ensemble.to_parquet(prediction,index=False);binding['prediction']['sha256']=sha256(prediction);path.write_text(json.dumps(binding))
    (output/'SUCCESS.json').write_text(json.dumps({'status':'PASS','binding_sha256':sha256(path)}))
    with pytest.raises(ValueError,match='checkpoint replay'):audit_current_predictions(path,expected_sha=sha256(path),replay_models=True)


@pytest.mark.parametrize('fault',['run','weights','checkpoint'])
def test_current_receipt_or_checkpoint_mismatch_is_not_inferred(tmp_path,fault):
    path=current_fixture(tmp_path/'inputs');manifest=json.loads(path.read_text());item=manifest['fold_inputs'][0]
    receipt_path=Path(item['training_receipt']['path']);receipt=json.loads(receipt_path.read_text())
    if fault=='run':receipt['run_id']='another-core'
    if fault=='weights':receipt['runtime_chunk_weights']={'0':.5,'1':.5}
    if fault=='checkpoint':receipt['best_model_state_sha256']='f'*64
    receipt_path.write_text(json.dumps(receipt));item['training_receipt']=record(receipt_path);path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):materialize_current_predictions(path,tmp_path/'out',expected_run_id='CURRENT-V4-SYNTHETIC')
