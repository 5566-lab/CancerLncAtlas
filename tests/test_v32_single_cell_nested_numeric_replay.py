"""Measured-value reconstruction with explicit frozen synthetic grouping, no fit."""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cc_hhgt.v32 import native_single_cell_donors_workflow as donors
from cc_hhgt.v32 import native_single_cell_episode_sources as sources
from cc_hhgt.v32 import single_cell_nested_numeric_replay as replay
from cc_hhgt.v32.formal_expression_sources import read, write, record
from scripts import materialize_single_cell_nested_hhgt_episodes_149 as native
from test_v32_native_single_cell_episode_sources import nested_inputs
from test_v32_native_single_cell_graph_workflow import completed as graph_completed
from test_v32_canonical_metacell_consumer import no_fit
from test_v32_native_single_cell_mapping_workflow import forbidden


def settings(inputs):
    _, donor, value, _, _ = inputs
    return donors.build_settings(value, donor.parent, time.time()+600)[0]


def paths(inputs):
    _, donor, _, _, _ = inputs
    cancer = donor.parent/'nested/cancers/BRCA/SUCCESS.json'
    outer = Path(read(cancer)['outer_fold_receipts'][0]['path'])
    episode = Path(read(outer)['episode_receipts'][0]['path'])
    assert read(episode)['state'] == 'NESTED_EPISODE_ASSETS_READY'
    return cancer, outer, episode


def snapshot(root):
    return {str(p):(record(p)['sha256'],p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


def test_original_complete_cache_reconstructs_and_preserves_inputs(nested_inputs,monkeypatch):
    _,donor,value,_,_=nested_inputs
    before=snapshot(donor.parent); replay._CACHE.clear()
    monkeypatch.setattr(native,'fit_nested_metacells',forbidden)
    original=replay.verify_plan; observed=[]
    def checked(*args):
        observed.append(args[1].as_dict());return original(*args)
    monkeypatch.setattr(replay,'verify_plan',checked)
    result=native.run_cancer(settings(nested_inputs))
    assert len(result)==len(observed)==value['nested_plans']
    assert {p['role'] for p in observed}=={'fit','validation'}
    monkeypatch.setattr(replay,'verify_plan',forbidden)
    assert native.run_cancer(settings(nested_inputs))==result
    assert snapshot(donor.parent)==before


@pytest.mark.parametrize('fault',['coding','lncrna','ucell','target_rho','source_rho',
                                 'candidate','node_features','coverage','summary'])
def test_rehashed_numeric_corruption_fails_original_cached_producer(nested_inputs,monkeypatch,fault):
    cancer,outer,episode=paths(nested_inputs); folder=episode.parent
    names={'coding':'METACELL_CODING.npz','lncrna':'METACELL_LNCRNA.npz',
        'ucell':'METACELL_PATHWAY_ACTIVITY.npy','target_rho':'SIGNED_TARGET_ROWS.parquet',
        'source_rho':'SIGNED_TARGET_ROWS.parquet','candidate':'CANDIDATE_UNIVERSE.parquet',
        'node_features':'NODE_FEATURES.npz','coverage':'CONTEXT_COVERAGE.json','summary':'SUCCESS.json'}
    target=folder/names[fault]; old={p:p.read_bytes() for p in (target,episode,outer,cancer)}
    monkeypatch.setattr(native,'fit_nested_metacells',forbidden)
    try:
        if fault in ('coding','lncrna'):
            matrix=sparse.load_npz(target); assert matrix.nnz
            matrix.data[0]+=1;sparse.save_npz(target,matrix)
        elif fault=='ucell':
            array=np.load(target,allow_pickle=False);array.flat[0]+=.25;np.save(target,array)
        elif fault in ('target_rho','source_rho','candidate'):
            frame=pd.read_parquet(target)
            if fault=='candidate':frame.loc[0,'is_bulk_candidate']=not bool(frame.loc[0,'is_bulk_candidate'])
            else:frame.loc[0,{'target_rho':'rho','source_rho':'source_meta_rho'}[fault]]+=.125
            frame.to_parquet(target,index=False)
        elif fault=='node_features':
            with np.load(target,allow_pickle=False) as arrays: data={key:arrays[key] for key in arrays.files}
            key=next(key for key,array in data.items() if array.size);data[key].flat[0]+=.125
            np.savez_compressed(target,**data)
        elif fault=='coverage':
            value=read(target);value['contexts'][0]['target_rows']+=1;write(target,value)
        else:
            value=read(target);value['assigned_cells']+=1;write(target,value)
        value=read(episode)
        if fault!='summary':value['files'][target.name]=native.file_record(target)
        write(episode,value)
        value=read(outer);value['episode_receipts']=[native.file_record(Path(p['path'])) for p in value['episode_receipts']];write(outer,value)
        value=read(cancer);value['outer_fold_receipts']=[native.file_record(Path(p['path'])) for p in value['outer_fold_receipts']];write(cancer,value)
        with pytest.raises(ValueError,match='Nested reconstruction'):
            native.run_cancer(settings(nested_inputs))
    finally:
        for path,data in old.items():path.write_bytes(data)


def test_partial_cache_checks_numeric_values_before_any_fitting(nested_inputs,monkeypatch):
    cancer,outer,episode=paths(nested_inputs);target=episode.parent/'METACELL_CODING.npz'
    old={p:p.read_bytes() for p in (target,episode,outer,cancer)}
    monkeypatch.setattr(native,'fit_nested_metacells',forbidden)
    try:
        cancer.unlink()
        matrix=sparse.load_npz(target);matrix.data[0]+=1;sparse.save_npz(target,matrix)
        value=read(episode);value['files'][target.name]=native.file_record(target);write(episode,value)
        with pytest.raises(ValueError,match='Nested reconstruction differs: METACELL_CODING'):
            native.run_cancer(settings(nested_inputs))
    finally:
        for path,data in old.items():path.write_bytes(data)


def test_cache_rechecks_output_bytes_and_raw_inputs(nested_inputs,monkeypatch):
    _,donor,value,_,_=nested_inputs
    native.run_cancer(settings(nested_inputs))
    _,_,episode=paths(nested_inputs)
    monkeypatch.setattr(native,'fit_nested_metacells',forbidden)
    for target in (episode.parent/'METACELL_CODING.npz',Path(value['mapped_success']['path']).parent/'CODING_MEASURED.npz'):
        old=target.read_bytes()
        try:
            target.write_bytes(old[:-1]+bytes([old[-1]^1]))
            with pytest.raises((ValueError,RuntimeError)):
                sources.nested_inventory(value,donor)
        finally:target.write_bytes(old)
