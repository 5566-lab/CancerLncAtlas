"""Actual original outer statistics, synthetic measured cells, frozen grouping."""
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from cc_hhgt.v32 import single_cell_outer_numeric_replay as replay
from cc_hhgt.v32.formal_expression_sources import read, write, record
from scripts import materialize_single_cell_metacell_149 as native
from test_v32_canonical_metacell_consumer import completed, no_fit
from test_v32_native_single_cell_mapping_workflow import forbidden
from test_single_cell_metacell_screen import synthetic


def paths(completed):
    settings = completed[3]
    cancer = settings.output_root/'cancers/BRCA/SUCCESS.json'
    return settings, cancer, cancer.parent/'outer_fold=0/SUCCESS.json'


def snapshot(root):
    return {str(p):(record(p)['sha256'],p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


def test_original_context_targets_match_independent_signed_and_leave_one_out_reference():
    x,y,metadata = synthetic()
    metadata['state_id'] = np.tile(['A','B'],len(metadata)//2)
    ids = {'lncrna_ids':['L1','L2'],'pathway_ids':['P1','P2']}
    bulk = pd.DataFrame({'lncrna_id':['L1'],'pathway_id':['P1']})
    settings = SimpleNamespace(deadline=time.time()+60,discovery_top_k=1,sketch_oversample=2,
        sketch_dimensions=32,min_target_metacells=8,min_screen_abs_rho=0.,seed=1729,max_train_donors_per_pair=6)
    source = {'D1','D2','D3'}
    def calculate(values):
        return native._context_targets(values,y,metadata,ids,bulk,source,{'D4'},
            {'dataset_id':'SYNTHETIC','measurement_scale':'log_normalized'},'BRCA',0,settings)
    train,query,universe,_ = calculate(x)
    reference = {d:spearmanr(x[metadata.donor_id.eq(d),0],y[metadata.donor_id.eq(d),0]).statistic
                 for d in ['D1','D2','D3','D4']}
    for table in (train,query):
        rows = table.loc[table.lncrna_id.eq('L1') & table.pathway_id.eq('P1')]
        assert len(rows) == (3 if table is train else 1)
        for row in rows.itertuples():
            donors = source-{row.query_donor_id}
            rhos = [reference[d] for d in donors]
            expected = np.tanh(np.mean(np.arctanh(np.clip(rhos,-.999999,.999999))))
            assert row.rho == pytest.approx(reference[row.query_donor_id],abs=1e-6)
            assert row.association_target == pytest.approx(abs(row.rho))
            assert row.source_meta_rho == pytest.approx(expected,abs=1e-6)
            assert row.source_donor_count == len(donors)
            positive = sum(r>0 for r in rhos)
            assert row.source_direction_consistency == max(positive,len(donors)-positive)/len(donors)
    changed = x.copy(); mask = metadata.donor_id.eq('D4').to_numpy()
    changed[mask] = changed[mask].max(axis=0)-changed[mask]
    other_train,other_query,other_universe,_ = calculate(changed)
    pd.testing.assert_frame_equal(train,other_train,check_exact=True)
    pd.testing.assert_frame_equal(universe,other_universe,check_exact=True)
    keys = ['lncrna_id','pathway_id','query_donor_id']
    columns = [*keys,'source_meta_rho','source_donor_count','source_direction_consistency']
    pd.testing.assert_frame_equal(query[columns],other_query[columns],check_exact=True)
    selected = other_query.loc[other_query.lncrna_id.eq('L1') & other_query.pathway_id.eq('P1')]
    assert selected.rho.iloc[0] == pytest.approx(-reference['D4'],abs=1e-6)


def test_complete_original_cache_reconstructs_all_folds_without_writes(completed, monkeypatch):
    root, _, _, settings = completed
    before = snapshot(root); replay._CACHE.clear(); observed = []
    original = replay.verify_fold
    def checked(*args):
        observed.append(args[2]); return original(*args)
    monkeypatch.setattr(replay,'verify_fold',checked)
    monkeypatch.setattr(native,'local_disjoint_groups',forbidden)
    native.run_cancer(settings)
    assert observed == list(range(5))
    monkeypatch.setattr(replay,'verify_fold',forbidden)
    native.run_cancer(settings)
    assert snapshot(root) == before


@pytest.mark.parametrize('fault',['query_rho','train_rho','source_rho','source_count','source_direction',
                                 'candidate','coverage','summary','inventory','group_count','row_order'])
def test_rehashed_outer_statistics_are_rejected_by_original_cache(completed, monkeypatch, fault):
    settings, cancer, success = paths(completed)
    names = {'query_rho':'QUERY_ROWS.parquet','train_rho':'TRAIN_ROWS.parquet',
        'source_rho':'QUERY_ROWS.parquet','source_count':'QUERY_ROWS.parquet',
        'source_direction':'TRAIN_ROWS.parquet','candidate':'UNIVERSE.parquet',
        'coverage':'CONTEXT_COVERAGE.json','summary':'SUCCESS.json','inventory':'SUCCESS.json',
        'group_count':'METACELL_METADATA.parquet','row_order':'QUERY_ROWS.parquet'}
    target = success.parent/names[fault]
    old = {p:p.read_bytes() for p in (target,success,cancer)}
    monkeypatch.setattr(native,'local_disjoint_groups',forbidden)
    try:
        if fault in ('summary','inventory'):
            data = read(success)
            if fault=='summary': data['unassigned_cells'] += 1
            else: data['files'].pop('UNIVERSE.parquet')
            write(success,data)
        elif fault=='coverage':
            data = read(target); data[0]['bulk_pairs'] += 1; write(target,data)
        else:
            frame = pd.read_parquet(target)
            if fault=='candidate': frame.loc[0,'is_bulk_candidate'] = not bool(frame.loc[0,'is_bulk_candidate'])
            elif fault=='row_order': frame = frame.iloc[::-1].reset_index(drop=True)
            else:
                column = {'query_rho':'rho','train_rho':'rho','source_rho':'source_meta_rho',
                    'source_count':'source_donor_count','source_direction':'source_direction_consistency',
                    'group_count':'n_cells'}[fault]
                frame.loc[0,column] += 1 if fault in ('source_count','group_count') else .125
            frame.to_parquet(target,index=False)
        data = read(success)
        if target != success: data['files'][target.name] = native.file_record(target)
        write(success,data); data = read(cancer)
        data['fold_receipt_sha256']['0'] = record(success)['sha256']; write(cancer,data)
        with pytest.raises(ValueError,match='Outer reconstruction|FROZEN_GROUP'):
            native.run_cancer(settings)
    finally:
        for path,data in old.items(): path.write_bytes(data)


def test_partial_cache_checks_query_values_before_any_fitting(completed,monkeypatch):
    settings,cancer,success = paths(completed); target = success.parent/'QUERY_ROWS.parquet'
    old = {p:p.read_bytes() for p in (target,success,cancer)}
    monkeypatch.setattr(native,'local_disjoint_groups',forbidden)
    try:
        cancer.unlink()
        frame = pd.read_parquet(target); frame.loc[0,'rho'] += .125; frame.to_parquet(target,index=False)
        data = read(success); data['files'][target.name] = native.file_record(target); write(success,data)
        with pytest.raises(ValueError,match='Outer reconstruction differs: QUERY_ROWS'):
            native.run_cancer(settings)
    finally:
        for path,data in old.items(): path.write_bytes(data)


@pytest.mark.parametrize('name',['QUERY_ROWS.parquet','CODING_MEASURED.npz','LNC_MEASURED.npz','ALL_GENE_LIBRARY.npy'])
def test_cached_proof_still_checks_actual_source_and_output_bytes(completed,monkeypatch,name):
    settings,_,success = paths(completed)
    native.run_cancer(settings)
    target = success.parent/name if name.endswith('.parquet') else settings.scratch_root/'BRCA'/name
    old = target.read_bytes()
    monkeypatch.setattr(replay,'verify_fold',forbidden)
    try:
        target.write_bytes(old[:-1]+bytes([old[-1]^1]))
        with pytest.raises((ValueError,RuntimeError)):
            native.run_cancer(settings)
    finally:
        target.write_bytes(old)
