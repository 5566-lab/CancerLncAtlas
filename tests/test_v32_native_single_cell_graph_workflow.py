from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from cc_hhgt.v32 import native_single_cell_graph_workflow as workflow
from cc_hhgt.v32 import native_single_cell_mapping_workflow as mapping
from cc_hhgt.v32 import native_single_cell_static_workflow as static
from cc_hhgt.v32 import native_single_cell_donors_workflow as donors
from cc_hhgt.v32 import native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read,write,record
from cc_hhgt.v32.single_cell_graph_assets import validate_frozen_metacell_metadata
from scripts import materialize_single_cell_metacell_149 as base
from scripts import materialize_single_cell_hhgt_assets_149 as native
from scripts import materialize_single_cell_hhgt_training_episodes_149 as assembler
from test_v32_canonical_metacell_consumer import expanded_source,frozen_fit,frozen_groups,no_fit
from test_v32_native_single_cell_mapping_workflow import forbidden


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    root=tmp_path_factory.mktemp('sc_graph'); manifest,_,_,_=expanded_source(root/'raw')
    mapped=mapping.run(manifest,root/'mapping'); ids=read(Path(read(mapped)['mapped_success']['path']).parent/'IDS.json')
    family=root/'family.parquet'
    pd.DataFrame({'pathway_id':ids['pathway_ids'],'pathway_family_id':'F1','is_representative':True}).to_parquet(family,index=False)
    external=root/'external.parquet'
    pd.DataFrame([dict(lncrna_id=ids['lncrna_ids'][0],partner_id=ids['protein_ids'][0],partner_type='gene',
        relation_type='binding',direction='direct',experiment_family='physical_binding',is_experimental=True,
        is_predicted=False,pmid='1',source_database='SYNTHETIC',mapping_status='lncrna_and_partner_mapped')]).to_parquet(external,index=False)
    write(root/'STATIC_INPUT.json',{'format':static.INPUT,'mapping':record(mapped),'pathway_family':record(family),'external_evidence':record(external)})
    stat=static.run(root/'STATIC_INPUT.json',root/'static')
    crosswalk=root/'donors.parquet'
    pd.DataFrame({'dataset_id':'SYNTHETIC','query_donor_id':[f'D{i:02}' for i in range(10)],
        'canonical_donor_id':[f'P{i%9:02}' for i in range(10)]}).to_parquet(crosswalk,index=False)
    candidates=root/'candidates.parquet'
    pd.DataFrame({'cancer_id':['BRCA'],'lncrna_id':[ids['lncrna_ids'][0]],'pathway_id':[ids['pathway_ids'][0]]}).to_parquet(candidates,index=False)
    write(root/'DONOR_INPUT.json',{'format':donors.INPUT,'mapping':record(mapped),'donor_crosswalk':record(crosswalk),
        'graph_preflight':read(stat)['preflight'],'candidates':record(candidates)})
    donor=donors.run(root/'DONOR_INPUT.json',root/'donor'); value=read(read(donor)['preflight']['path'])
    settings,_=donors.build_metacell_settings(value,donor.parent,time.time()+600)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base,'LocalSpace',SimpleNamespace(fit=frozen_fit)); patch.setattr(base,'local_disjoint_groups',frozen_groups)
        base.run_cancer(settings)
    spec={'format':workflow.INPUT,'donor_preparation':record(donor),'static_workflow':record(stat),
          'metacell_success':record(settings.output_root/'cancers/BRCA/SUCCESS.json')}
    manifest=root/'INPUT.json'; write(manifest,spec)
    write(root/'CHAIN_INPUT.json',{'format':chain.INPUT,'chain_id':'sc-graph','stages':[
        {'id':'graph','operation':'single_cell_graph.run','inputs':spec}]})
    chain.initialize(root/'CHAIN_INPUT.json',root/'chain'); chain.advance(root/'chain')
    state=read(root/'chain/CHAIN.json'); assert state['status']=='DECLARED_STAGES_COMPLETE',state
    pin=state['results']['graph']; out=Path(pin['path']).parent
    return root,manifest,spec,pin,out


def test_all_fold_graphs_reaggregate_canonical_measurements(completed,monkeypatch):
    root,_,spec,pin,out=completed; workflow.validate(pin,root)
    graph=read(out/'GRAPH_ASSET_MANIFEST.json'); assert graph['fold_count']==5
    for fold in range(5):
        folder=out/f'cancers/BRCA/outer_fold={fold}'; receipt=read(folder/'SUCCESS.json')
        assert not receipt['geometry_recomputed'] and not receipt['lasso_coefficients_computed']
        assert receipt['metacells']==400 and receipt['query_observations_forbidden_from_lasso']>0
        rows=pd.read_parquet(folder/'LASSO_OBSERVATIONS.parquet')
        assert rows.donor_id.str.startswith('P').all()
        assert not rows.loc[rows.donor_id.isin(receipt['query_donors']),'lasso_input_allowed'].any()
        assert sparse.load_npz(folder/'METACELL_CODING.npz').shape==(400,5)
        assert np.load(folder/'METACELL_PATHWAY_ACTIVITY.npy').shape==(400,2135)
    monkeypatch.setattr(chain,'dispatch',forbidden); chain.advance(root/'chain')
    settings=workflow.settings(read(out/'LAUNCH_PREFLIGHT.json'),out,0,time.time()+300)
    native.materialize_fold(settings)


def test_original_query_assembler_reaches_lasso_boundary_without_fit(completed,tmp_path,monkeypatch):
    _,_,_,_,out=completed
    settings=SimpleNamespace(graph_asset_root=out,output_root=tmp_path,deadline=time.time()+300)
    loaded=assembler._load_cancer_static(settings,'BRCA'); seen=[]
    def boundary(settings,**values):
        assert not set(values['source_donors']) & set(values['outer_query_donors'])
        assert values['rows'].query_donor_id.str.startswith('P').all()
        assert values['features_full']['gene'].shape[0]==5
        seen.append(values['outer_fold_id'])
        return {'synthetic_lasso_boundary':True}
    monkeypatch.setattr(assembler,'_cached_episode',lambda *a,**k:None)
    monkeypatch.setattr(assembler,'_write_episode',boundary)
    result=assembler._query_episodes(settings,'BRCA',loaded,tmp_path/'unused.pt',read(out/'cancers/BRCA/SUCCESS.json'))
    assert len(result)==5 and sorted(seen)==list(range(5))


@pytest.mark.parametrize('fault',['coding','activity','query_allowed'])
def test_repinned_fold_corruption_is_rejected_on_cached_consumer(completed,fault):
    _,_,_,_,out=completed; folder=out/'cancers/BRCA/outer_fold=0'
    filename={'coding':'METACELL_CODING.npz','activity':'METACELL_PATHWAY_ACTIVITY.npy','query_allowed':'LASSO_OBSERVATIONS.parquet'}[fault]
    target=folder/filename; success=folder/'SUCCESS.json'; cancer=out/'cancers/BRCA/SUCCESS.json'; manifest=out/'GRAPH_ASSET_MANIFEST.json'
    old={p:p.read_bytes() for p in (target,success,cancer,manifest)}
    try:
        if fault=='coding':
            frame=sparse.load_npz(target); frame.data[0]+=1; sparse.save_npz(target,frame)
        elif fault=='activity':
            frame=np.load(target); frame[0,0]+=.2; np.save(target,frame)
        else:
            frame=pd.read_parquet(target); frame.loc[frame.observation_role.eq('query'),'lasso_input_allowed']=True; frame.to_parquet(target,index=False)
        receipt=read(success); receipt['files'][filename]=native.file_record(target); write(success,receipt)
        c,m=workflow.envelopes(read(out/'LAUNCH_PREFLIGHT.json'),out); write(cancer,c); write(manifest,m)
        with pytest.raises(ValueError,match='reconstruction differs'):
            native.materialize_fold(workflow.settings(read(out/'LAUNCH_PREFLIGHT.json'),out,0,time.time()+300))
    finally:
        for p,data in old.items():p.write_bytes(data)


def test_missing_canonical_cache_never_falls_back_to_fitting(completed,tmp_path,monkeypatch):
    _,_,spec,_,_=completed; donor=Path(spec['donor_preparation']['path']); value=read(read(donor)['preflight']['path'])
    settings,_=donors.build_metacell_settings(value,donor.parent,time.time()+300)
    monkeypatch.setattr(base,'LocalSpace',SimpleNamespace(fit=forbidden))
    with pytest.raises(RuntimeError,match='fitting is forbidden'):
        base._write_fold('BRCA',0,tmp_path,settings.scratch_root/'BRCA',settings.candidates,settings,cached_only=True)


@pytest.mark.parametrize('fault',['donor','count','mixed'])
def test_frozen_group_metadata_must_match_raw_cells(fault):
    cells=pd.DataFrame({'donor_id':['A','A','B','B'],'cell_type':'T','state_id':'S'})
    assignment=np.array([0,0,1,1]); frozen=pd.DataFrame({'metacell_id':[0,1],'donor_id':['A','B'],'cell_type':'T','state_id':'S','n_cells':[2,2]})
    validate_frozen_metacell_metadata(cells,assignment,frozen)
    if fault=='donor': frozen.loc[0,'donor_id']='B'
    elif fault=='count': frozen.loc[0,'n_cells']=3
    else: assignment=np.array([0,1,0,1])
    with pytest.raises(ValueError,match='FROZEN_GROUP'): validate_frozen_metacell_metadata(cells,assignment,frozen)
