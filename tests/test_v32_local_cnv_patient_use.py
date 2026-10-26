"""Actual native statistics capture exact patient scopes without optimization."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32 import local_cnv_patient_use as scopes
from cc_hhgt.v32.current_model_use import Inventory, inventory_tables
from cc_hhgt.v32.formal_expression_sources import read,write,record
from cc_hhgt.v32.local_cnv_confounding import fold_local_pair_models
from cc_hhgt.v32.local_cnv_source_replay import replay,inspect
from test_v32_native_cnv_workflow import completed
from test_v32_local_cnv_cohort import no_fit
from test_v32_local_cnv_source_replay import binding


def snapshot(root):
    return {str(p):(record(p)['sha256'],p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


def mapped_inventory(completed,root):
    args=completed['correction'][2]; patients=pd.read_csv(args.patient_map,sep='\t',dtype=str)
    digest=record(args.patient_map)['sha256']
    crosswalk=root/'identities.parquet'
    pd.DataFrame({'source_file_sha256':digest,'identity_column':'patient_id',
        'raw_id':patients.patient_id,'canonical_patient_id':patients.patient_id}).to_parquet(crosswalk,index=False)
    return Inventory([],root,{**record(crosswalk),'namespace':'SYNTHETIC_NATIVE_PATIENTS'})


def rows_by_pair(event):
    assert event['pairing']=='ALIGNED_PAIRS'
    return list(zip(json.loads(event['lncrna_ids_json']),json.loads(event['target_ids_json'])))


@pytest.mark.parametrize('kind',['pair','graph','validation'])
def test_actual_replay_records_private_statistical_cohorts_and_counts(completed,tmp_path,kind):
    inventory=mapped_inventory(completed,tmp_path); item=binding(completed,kind=kind)
    before=snapshot(completed['correction'][0])
    with inventory.context('G012_COMPARISON:G2'):
        inspect(inventory,item,tmp_path,'core','native.local_cnv',0)
    tables=inventory_tables(inventory,{'core':{}}); events=tables[scopes.TABLE]
    assert len(events) and events['format'].eq(scopes.FORMAT).all()
    assert events.dependency_context.eq('G012_COMPARISON:G2').all()
    assert events.output_sha256.eq(item['output']['sha256']).all()
    patient_map=pd.read_csv(completed['correction'][2].patient_map,sep='\t')
    observed=pd.read_parquet(item['output']['path'])
    for event in events.to_dict('records'):
        patient_rows=patient_map.iloc[json.loads(event['patient_map_row_indices_json'])]
        assert patient_rows.cancer_id.eq('BRCA').all()
        allowed={2,3,4} if event['split_role']=='train' else {1}
        assert set(patient_rows.patient_fold_id)<=allowed
        assert patient_rows.patient_id.nunique()==event['native_patient_count']
        assert event['candidate_pair_count']>=0
        if kind!='graph' and event['statistic']!='MATCHED_SHARED_COVARIATE_DESIGN':
            column={'A0_FULL':'n_A0_full' if event['split_role']=='train' else 'n_A0_validation',
                    'A0_MATCHED_A1_LOCAL':'n_A1_validation' if kind=='validation' else 'n_A1_local',
                    'A1_ON_A2_A2_LOCAL_PATHWAY':'n_A2_local_pathway'}[event['statistic']]
            for lnc,pathway in rows_by_pair(event):
                selected=observed.loc[observed.lncrna_id.eq(lnc)&observed.pathway_id.eq(pathway)]
                assert selected[column].tolist()==[event['native_patient_count']]
    if kind=='graph':
        tested=events.loc[events.statistic.eq('E0_E1_TESTED_CARTESIAN')]
        assert len(tested) and tested.pairing.eq('CARTESIAN').all()
        for event in tested.to_dict('records'):
            assert event['candidate_pair_count']==len(json.loads(event['lncrna_ids_json']))*len(json.loads(event['target_ids_json']))
    mappings=tables['patient_mapping'].loc[tables['patient_mapping'].source_role.str.contains('.cohort.',regex=False)]
    assert len(mappings) and mappings.unmapped_identity_count.eq(0).all()
    assert tables['patient_usage_private'].patient_token.str.startswith('PATIENT:').all()
    assert not events.astype(str).apply(lambda c:c.str.contains('PRIVATE_',regex=False)).any().any()
    assert snapshot(completed['correction'][0])==before


def test_masks_are_exact_pair_specific_and_do_not_change_statistics(tmp_path):
    count=20; patients=[f'PRIVATE_{i}' for i in range(count)]
    path=tmp_path/'patients.tsv'
    pd.DataFrame({'patient_id':patients,'cancer_id':'BRCA','patient_fold_id':2}).to_csv(path,sep='\t',index=False)
    trace=scopes.Trace(path,record(path),'BRCA',0)
    pairs=pd.DataFrame({'lncrna_id':['L1','L2'],'pathway_id':['P1','P2']})
    rng=np.random.default_rng(22)
    x=rng.normal(size=(count,2)); y=rng.normal(size=(count,2)); z=rng.normal(size=(count,2)); p=z.copy()
    design=np.ones((count,1)); local=np.ones((count,2),bool); pathway=local.copy()
    local[8:,1]=False; pathway[6:,0]=False
    x[2,0]=np.nan; z[3,0]=np.nan; design[4,0]=np.nan
    z[0,0]=0.0  # Measured neutral CNV must remain a patient, not missingness.
    args=(x,y,z,p,design)
    original=fold_local_pair_models(*args,local_callable=local,pathway_callable=pathway)
    result=fold_local_pair_models(*args,local_callable=local,pathway_callable=pathway,
        patient_observer=trace.matched(pairs,patients,'train'))
    for key in original: np.testing.assert_array_equal(result[key],original[key])
    expected_local=local & np.isfinite(x)&np.isfinite(y)&np.isfinite(z)&np.isfinite(design).all(axis=1)[:,None]
    expected_pathway=expected_local & pathway & np.isfinite(p)
    seen=set()
    for event in trace.events:
        expected=expected_local if event['statistic']=='A0_MATCHED_A1_LOCAL' else expected_pathway
        for lnc,target in rows_by_pair(event):
            index=int(lnc[-1])-1; seen.add((event['statistic'],lnc))
            assert json.loads(event['patient_map_row_indices_json'])==np.flatnonzero(expected[:,index]).tolist()
            assert event['calculation_entered']==(int(expected[:,index].sum())>=12)
    assert len(seen)==4


def test_cache_reuses_captured_scopes_in_each_dependency_context(completed,tmp_path,monkeypatch):
    item=binding(completed); inventory=mapped_inventory(completed,tmp_path)
    # A prior numeric-only proof cannot manufacture a patient trace.
    replay(item,tmp_path,cache=inventory.visited)
    inspect(inventory,item,tmp_path,'core','local_cnv',0)
    count=len(inventory.statistical_patient_scopes); assert count>0
    import scripts.run_v32_local_cnv_confounding_audit_r2 as native
    def forbidden(*a,**k): raise AssertionError('Valid captured scopes must reuse checked proof')
    monkeypatch.setattr(native,'association_fold',forbidden)
    with inventory.context('G012_COMPARISON:G1'):
        inspect(inventory,item,tmp_path,'core','local_cnv',0)
    assert len(inventory.statistical_patient_scopes)==2*count
    assert {r['dependency_context'] for r in inventory.statistical_patient_scopes}=={'SELECTED_SCORE','G012_COMPARISON:G1'}
    path=Path(completed['correction'][2].patient_map); old=path.read_bytes()
    try:
        path.write_bytes(old+b'changed')
        with pytest.raises((ValueError,RuntimeError)): inspect(inventory,item,tmp_path,'core','local_cnv',0)
    finally:path.write_bytes(old)


@pytest.mark.parametrize('fault',['test_fold','unknown_patient','duplicate_patient'])
def test_wrong_patient_scope_is_rejected_before_recording(tmp_path,fault):
    path=tmp_path/'patients.tsv'
    pd.DataFrame({'patient_id':['A','B'],'cancer_id':'BRCA','patient_fold_id':[2,0]}).to_csv(path,sep='\t',index=False)
    trace=scopes.Trace(path,record(path),'BRCA',0)
    chosen={'test_fold':['B'],'unknown_patient':['C'],'duplicate_patient':['A','A']}[fault]
    with pytest.raises(ValueError,match='patient scope'):
        trace.emit('A0_FULL','train',chosen,['L'],['P'],'pathway','ALIGNED_PAIRS',True,[True])
    assert not trace.events


def test_native_patient_aliases_are_mapped_using_actual_source_rows(tmp_path):
    path=tmp_path/'patients.tsv'
    pd.DataFrame({'patient_id':['tcga.aa.0001.01a','TCGA-AA-0001-02A','TCGA-BB-0002'],
        'cancer_id':'BRCA','patient_fold_id':[2,2,3]}).to_csv(path,sep='\t',index=False)
    pin=record(path);trace=scopes.Trace(path,pin,'BRCA',0)
    trace.emit('A0_FULL','train',['TCGA-AA-0001'],['L'],['P'],'pathway','ALIGNED_PAIRS',True,[True])
    assert trace.events[0]['native_patient_count']==1
    assert json.loads(trace.events[0]['patient_map_row_indices_json'])==[0,1]
    mapping=tmp_path/'map.parquet'
    pd.DataFrame({'source_file_sha256':pin['sha256'],'identity_column':'patient_id',
        'raw_id':['tcga.aa.0001.01a','TCGA-AA-0001-02A'],'canonical_patient_id':'PERSON_1'}).to_parquet(mapping,index=False)
    inventory=Inventory([],tmp_path,{**record(mapping),'namespace':'SYNTHETIC'})
    scopes.attach(inventory,trace.payload(),{'sources':{'sha256':'a'*64},'output':{'sha256':'b'*64}},
                  tmp_path,'core','local_cnv',0)
    tables=inventory.patient_use.tables()
    assert len(tables['patient_usage_private'])==1
    assert tables['patient_mapping'].mapped_identity_count.tolist()==[2]
    # Without an explicit crosswalk, canonicalization does not imply identity.
    other=Inventory([],tmp_path)
    scopes.attach(other,trace.payload(),{'sources':{'sha256':'a'*64},'output':{'sha256':'b'*64}},
                  tmp_path,'core','local_cnv',0)
    assert other.patient_use.tables()['patient_mapping'].unmapped_identity_count.tolist()==[2]
