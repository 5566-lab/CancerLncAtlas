"""Native H5/graph/nested/episode sources reach actual frozen model-use inspection."""
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32 import native_single_cell_model_sources as sources
from cc_hhgt.v32.current_model_use import Inventory, implementations
from cc_hhgt.v32.formal_expression_sources import read,write,record
from scripts import materialize_single_cell_hhgt_training_episodes_149 as assembler
from scripts import train_single_cell_hhgt_149 as trainer
from test_v32_native_single_cell_episode_sources import completed as episode_completed,nested_inputs
from test_v32_native_single_cell_graph_workflow import completed as graph_completed
from test_v32_canonical_metacell_consumer import no_fit
from test_v32_single_cell_model_use import checkpoint_fixture,inspect as inspect_model
from test_v32_native_single_cell_mapping_workflow import forbidden


@pytest.fixture(scope='module')
def completed(episode_completed):
    root,settings,donor,value,_=episode_completed
    # Original upstream production is actual; model states/predictions and
    # telemetry below are explicitly frozen serialization fixtures, not fits.
    binding=checkpoint_fixture(root/'model-inspection',settings.output_root,settings.graph_asset_root,
                               candidates=Path(value['candidates']['path']))
    return root,settings,donor,value,binding


def paths(completed):
    _,settings,_,_,_=completed
    return settings.output_root/'SC_HHGT_EPISODE_MANIFEST.json',settings.graph_asset_root/'GRAPH_ASSET_MANIFEST.json'


def snapshot(root):
    return {str(p):(record(p)['sha256'],p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


def test_actual_signed_model_inspector_records_native_reconstructions(completed,monkeypatch):
    root,settings,donor,value,binding=completed
    before=snapshot(root)
    for name in ('_initialize_model','_evaluate_episode','fit_fold_model','run_training'):
        monkeypatch.setattr(trainer,name,forbidden)
    monkeypatch.setattr(assembler,'fit_context_lasso',forbidden)
    inventory=inspect_model(binding,root)
    replayed=pd.DataFrame(inventory.transform_replays)
    assert set(replayed['format'])=={sources.mapping.FORMAT,sources.static.FORMAT,sources.graph.FORMAT,sources.FORMAT,sources.ASSEMBLY}
    assert replayed.model_training_started.eq(False).all()
    assert set(replayed.dependency_context)=={'FUSION_EXPERT:single_cell_positive'}
    assert replayed.loc[replayed['format'].eq(sources.FORMAT),'folds_recomputed'].tolist()==[5]
    assert replayed.outputs_compared.gt(0).all()
    dependencies={r['path'] for r in inventory.dependencies}
    mapped=Path(value['mapped_success']['path'])
    mapping_value=read(read(mapped)['native_mapping_preparation']['path'])
    assert {p['path'] for p in mapping_value['source_files']}<=dependencies
    reconstructed={r['sha256'] for r in inventory.dependencies if '.native.mapping.' in r['role']}
    assert reconstructed
    assert not any(r['sha256'] in reconstructed and r['reason']=='SC_RAW_ARRAY_TRANSFORM_ANCESTRY_NOT_RECONSTRUCTED' for r in inventory.gaps)
    reasons={r['reason'] for r in inventory.gaps}
    assert {'SC_FITTED_GEOMETRY_NOT_INDEPENDENTLY_VERIFIED','SC_FITTED_LASSO_COEFFICIENTS_NOT_RECONSTRUCTED',
            'SC_RAW_ACQUISITION_QC_AND_REFERENCE_ANCESTRY_UNVERIFIED',
            'SC_RAW_STUDY_AND_CROSS_MODULE_PATIENT_ANCESTRY_NOT_COMPLETE'}<=reasons
    public=pd.DataFrame(inventory.usage).astype(str)
    assert not public.apply(lambda c:c.str.contains(r'"donor_id"|D00|P00|CANONICAL_P',regex=True)).any().any()
    assert snapshot(root)==before
    assert set(sources.CODE)<=set(implementations())


@pytest.mark.parametrize('fault',['native_marker','manifest_counts','other_graph','mapped_bytes','nested_bytes'])
def test_native_source_inspection_rejects_broken_or_downgraded_bindings(completed,tmp_path,monkeypatch,fault):
    _,settings,_,value,_=completed
    episode,graph=paths(completed); launch=settings.output_root/'LAUNCH_PREFLIGHT.json'
    old={p:p.read_bytes() for p in (episode,launch)}
    monkeypatch.setattr(assembler,'fit_context_lasso',forbidden)
    try:
        if fault=='native_marker':
            preflight=read(launch);preflight.pop('native_episode_sources');write(launch,preflight)
            data=read(episode);data['launch_preflight']=assembler.file_record(launch);write(episode,data)
        elif fault=='manifest_counts':
            data=read(episode);data['row_count']+=1;write(episode,data)
        elif fault=='other_graph':
            other=tmp_path/'GRAPH_ASSET_MANIFEST.json';other.write_bytes(graph.read_bytes());graph=other
        else:
            if fault=='mapped_bytes':target=Path(value['mapped_success']['path']).parent/'CODING_MEASURED.npz'
            else:
                root=Path(value['mapped_success']['path'])
                donor=completed[2]
                cancer=read(donor.parent/'nested/cancers/BRCA/SUCCESS.json')
                outer=read(cancer['outer_fold_receipts'][0]['path'])
                target=Path(outer['episode_receipts'][0]['path']).parent/'METACELL_CODING.npz'
            old[target]=target.read_bytes();target.write_bytes(old[target][:-1]+bytes([old[target][-1]^1]))
        with pytest.raises((ValueError,RuntimeError)):
            sources.inspect(Inventory([],tmp_path),episode,graph,'model','single_cell')
    finally:
        for path,data in old.items():path.write_bytes(data)


def test_legacy_manifest_has_no_invented_native_proof(tmp_path):
    episode=tmp_path/'episode.json';graph=tmp_path/'graph.json';write(episode,{});write(graph,{})
    inventory=Inventory([],tmp_path)
    assert sources.inspect(inventory,episode,graph,'model','sc')==set()
    assert not inventory.transform_replays
