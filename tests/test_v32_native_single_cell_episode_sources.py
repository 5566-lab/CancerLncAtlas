"""Actual episode launch/finalization with explicit frozen geometry/LASSO substitutes."""
from pathlib import Path
from types import SimpleNamespace
import sys
import time

import numpy as np
import pytest
import torch

from cc_hhgt.v32 import native_single_cell_episode_sources as sources
from cc_hhgt.v32 import native_single_cell_donors_workflow as donors
from cc_hhgt.v32 import single_cell_nested_episodes as nested_geometry
from cc_hhgt.v32.single_cell_batched_lasso import BatchedLassoResult
from cc_hhgt.v32.formal_expression_sources import read, write, record
from scripts import materialize_single_cell_nested_hhgt_episodes_149 as nested
from scripts import materialize_single_cell_hhgt_training_episodes_149 as assembler
from scripts import train_single_cell_hhgt_149 as trainer
from test_v32_native_single_cell_graph_workflow import completed as graph_completed
from test_v32_canonical_metacell_consumer import no_fit, frozen_groups
from test_v32_native_single_cell_mapping_workflow import forbidden


def frozen_lasso(coding, lncrna, observations, *, response_global_indices, lncrna_ids, gene_ids,
                 source_donors, excluded_donors, **kwargs):
    assert not set(source_donors) & set(excluded_donors)
    assert set(source_donors) <= set(observations.donor_id)
    count = len(response_global_indices)
    return BatchedLassoResult(response_count=count, gene_count=len(gene_ids),
        response_ids=tuple(lncrna_ids[i] for i in response_global_indices), gene_ids=tuple(gene_ids),
        response_index=np.arange(count), gene_index=np.zeros(count,dtype=int),
        coefficient=np.full(count,.25,dtype=np.float32), stability=np.ones(count,dtype=np.float32),
        available=np.ones(count,dtype=bool), unavailable_reason=np.full(count,'',dtype=object),
        selected_alpha=np.full(count,.1), inner_cv_mse=np.full(count,.5), inner_null_mse=np.ones(count),
        bootstrap_successes=np.full(count,8), audit={'synthetic_frozen_lasso':True,'held_out_donors_used':0})


@pytest.fixture(scope='module')
def completed(graph_completed):
    root, _, spec, graph_pin, graph_out = graph_completed
    donor = Path(spec['donor_preparation']['path']); value = read(read(donor)['preflight']['path'])
    nested_settings, _ = donors.build_settings(value, donor.parent, time.time()+600)
    def projection(matrix, labels, allowed, **kwargs):
        return SimpleNamespace(training_donors=tuple(allowed), transform=lambda x:x[:,:2].toarray())
    def grouping(*args, **kwargs):
        assignments, audit = frozen_groups(*args, **kwargs)
        return assignments, {**audit,'training_donors':list(kwargs['allowed_training_donors'])}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(nested_geometry,'LocalSpace',SimpleNamespace(fit=projection))
        patch.setattr(nested_geometry,'local_disjoint_groups',grouping)
        receipts = nested.run_cancer(nested_settings)
    assert any(r['state']=='NESTED_EPISODE_ASSETS_READY' and r['role']=='validation' for r in receipts)
    out = root/'episodes'
    argv=['--run-id','native-synthetic','--source-result-root',str(donor.parent/'metacells'),
          '--graph-asset-root',str(graph_out),'--nested-asset-root',str(donor.parent/'nested'),
          '--output-root',str(out),'--cancer-id','BRCA','--deadline',str(time.time()+600),
          '--native-graph-binding',graph_pin['path']]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler,'fit_context_lasso',forbidden)
        patch.setattr(sys,'argv',['materializer',*argv,'--prepare-only'])
        assert assembler.main()==0
    assert not (out/'episodes').exists() and not (out/'static').exists()
    settings=assembler.settings_from_args(assembler.parser().parse_args(argv))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(assembler,'fit_context_lasso',frozen_lasso)
        result=assembler.run_cancer(settings)
    assert result['role_counts']['fit'] and result['role_counts']['validation'] and result['role_counts']['query']
    settings.cancer_id=None; settings.finalize=True
    manifest=assembler.finalize(settings)
    assert manifest['outer_folds']==5 and manifest['cancers']==['BRCA']
    return root,settings,donor,value,manifest


def test_actual_native_episodes_reach_training_input_collection_without_model(completed,monkeypatch):
    root, settings, _, _, manifest=completed
    monkeypatch.setattr(trainer,'_initialize_model',forbidden)
    config=trainer.TrainingSettings(episode_root=settings.output_root,graph_assets_root=settings.graph_asset_root,
                                    output_root=root/'training-preflight',prepare_only=True)
    descriptors,lineage=trainer.load_episode_manifest(config)
    assert sum(map(len,descriptors.values()))==manifest['episode_count']
    for fold,items in descriptors.items():
        assert {d.role for d in items}=={'fit','validation','query'}
        for descriptor in items:
            payload=torch.load(descriptor.path,weights_only=True)
            assert payload['lasso_audit']['synthetic_frozen_lasso']
            assert not set(payload['source_donors']) & set(payload['label_donors'])
    from cc_hhgt.v32.single_cell_numeric_replay import collect_input_preflight
    collected=collect_input_preflight(config,descriptors,lineage)
    assert len(collected['models'])==10 and not collected['model_constructed'] and not collected['training_started']


def test_native_cancer_cache_checks_sources_without_fitting(completed,monkeypatch):
    _,settings,_,_,_=completed
    monkeypatch.setattr(assembler,'fit_context_lasso',forbidden)
    args=SimpleNamespace(**vars(settings));args.cancer_id='BRCA';args.finalize=False
    result=assembler.run_cancer(args)
    assert result['state']=='CANCER_EPISODES_COMPLETE'


@pytest.mark.parametrize('fault',['missing_outer','partition','target_source','node_order'])
def test_completed_nested_sources_are_checked_before_cached_episodes(completed,fault):
    _,settings,donor,value,_=completed
    root=donor.parent/'nested/cancers/BRCA'
    outer=read(root/'SUCCESS.json')['outer_fold_receipts'][0]
    episode=Path(read(outer['path'])['episode_receipts'][0]['path']); folder=episode.parent
    if fault=='missing_outer':
        target=Path(outer['path']); original=target.read_bytes()
        try:
            target.unlink()
            with pytest.raises((FileNotFoundError,RuntimeError,ValueError)): sources.preflight_sources(settings)
        finally: target.write_bytes(original)
        return
    target={'partition':folder/'DONOR_PARTITION.json','target_source':folder/'SIGNED_TARGET_ROWS.parquet',
            'node_order':root/'MATRIX_IDENTITY.parquet'}[fault]
    files=[target,episode,Path(outer['path']),root/'SUCCESS.json',root/'MATRIX_IDENTITY.json']
    old={p:p.read_bytes() for p in files}
    try:
        if fault=='partition':
            data=read(target);data['source_donors']=data['outer_query_donors'];write(target,data)
        else:
            import pandas as pd
            frame=pd.read_parquet(target)
            if fault=='target_source': frame.loc[0,'query_donor_id']=read(episode)['source_donors'][0]
            else: frame.loc[0,'canonical_id']='WRONG'
            frame.to_parquet(target,index=False)
        if fault=='node_order':
            data=read(root/'MATRIX_IDENTITY.json');data['matrix_identity']=nested.file_record(target);write(root/'MATRIX_IDENTITY.json',data)
            data=read(root/'SUCCESS.json');data['matrix_identity']=nested.file_record(root/'MATRIX_IDENTITY.json');write(root/'SUCCESS.json',data)
        else:
            data=read(episode);data['files'][target.name]=nested.file_record(target);write(episode,data)
            data=read(outer['path']);data['episode_receipts']=[nested.file_record(Path(p['path'])) for p in data['episode_receipts']];write(outer['path'],data)
            data=read(root/'SUCCESS.json');data['outer_fold_receipts']=[nested.file_record(Path(p['path'])) for p in data['outer_fold_receipts']];write(root/'SUCCESS.json',data)
        with pytest.raises((ValueError,RuntimeError,AssertionError)): sources.preflight_sources(settings)
    finally:
        for p,data in old.items():p.write_bytes(data)


def test_native_cached_episode_checks_actual_bytes(completed):
    _,settings,_,_,manifest=completed
    path=Path(manifest['folds'][0]['episodes'][0]['path']); receipt=read(path.with_suffix('.json'))
    old=path.read_bytes()
    try:
        path.write_bytes(old[:-1]+bytes([old[-1]^1]))
        with pytest.raises(RuntimeError): assembler._verify_native_cache(settings,receipt)
    finally: path.write_bytes(old)


@pytest.mark.parametrize('field',['nested_asset_root','source_result_root','cancer_id','deadline'])
def test_native_preflight_rejects_different_sources_and_invalid_deadline(completed,field):
    _,settings,_,_,_=completed
    args=SimpleNamespace(**vars(settings))
    setattr(args,field,{'nested_asset_root':settings.output_root,'source_result_root':settings.output_root,
                        'cancer_id':'LUAD','deadline':float('inf')}[field])
    with pytest.raises(ValueError,match='roots|cancer|deadline'):sources.preflight_sources(args)


def test_changed_lasso_policy_cannot_reuse_native_preflight(completed):
    _,settings,_,_,_=completed
    args=SimpleNamespace(**vars(settings));args.lasso_stability_threshold=.75
    with pytest.raises(RuntimeError,match='PREFLIGHT_DRIFT'):assembler._initialize_preflight(args)
