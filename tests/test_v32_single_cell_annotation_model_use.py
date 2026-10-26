"""Actual frozen SC model-input inspection follows original GTF production."""
import pytest

from cc_hhgt.v32 import native_single_cell_annotation_workflow as annotation
from cc_hhgt.v32.formal_expression_sources import read, record
from test_v32_native_single_cell_annotation_workflow import attach_reference
from test_v32_native_single_cell_model_sources import completed as native_model_completed
from test_v32_native_single_cell_episode_sources import completed as episode_completed, nested_inputs
from test_v32_native_single_cell_graph_workflow import completed as graph_completed
from test_v32_canonical_metacell_consumer import no_fit
from test_v32_single_cell_model_use import inspect as inspect_model
from test_v32_native_single_cell_mapping_workflow import forbidden


@pytest.fixture(scope='module',autouse=True)
def generated_reference():
    import test_v32_native_single_cell_graph_workflow as fixture
    original=fixture.expanded_source; references=[]
    def source(root):
        result=original(root);references.append(attach_reference(result[0],root/'gencode'))
        return result
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fixture,'expanded_source',source)
        yield references


def test_actual_signed_model_input_path_reaches_original_gtf(native_model_completed,generated_reference,monkeypatch):
    from scripts import train_single_cell_hhgt_149 as trainer
    from scripts import materialize_single_cell_hhgt_training_episodes_149 as assembler
    root,settings,donor,value,binding=native_model_completed
    for name in ('_initialize_model','_evaluate_episode','fit_fold_model','run_training'):
        monkeypatch.setattr(trainer,name,forbidden)
    monkeypatch.setattr(assembler,'fit_context_lasso',forbidden)
    assert len(generated_reference)==1
    reference=generated_reference[0]; sc=read(reference);launch=read(sc['preflight']['path'])
    before={str(p):record(p)['sha256'] for p in root.rglob('*') if p.is_file()}
    inventory=inspect_model(binding,root)
    assert launch['gtf']['path'] in {p['path'] for p in inventory.dependencies}
    proofs=[r for r in inventory.transform_replays if r['format']==annotation.FORMAT]
    assert len(proofs)==1 and proofs[0]['outputs_compared']==2 and not proofs[0]['model_training_started']
    assert proofs[0]['dependency_context']=='FUSION_EXPERT:single_cell_positive'
    value,pins=annotation.source_files(sc['annotation']['path'])
    reference_digests={p['sha256'] for p in pins}
    assert not any(r['source_file_sha256'] in reference_digests for r in inventory.usage)
    assert not any(r['sha256'] in reference_digests and r['reason']=='RAW_UPSTREAM_TRANSFORMATIONS_NOT_RECONSTRUCTED' for r in inventory.gaps)
    reasons={r['reason'] for r in inventory.gaps}
    assert {'GENCODE_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED',
            'SC_FITTED_GEOMETRY_NOT_INDEPENDENTLY_VERIFIED','SC_FITTED_LASSO_COEFFICIENTS_NOT_RECONSTRUCTED'}<=reasons
    assert before=={str(p):record(p)['sha256'] for p in root.rglob('*') if p.is_file()}
