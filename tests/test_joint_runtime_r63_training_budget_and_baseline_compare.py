import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from cla_engine.baseline_compare import _ablation_design,hash_collision_report,hashed_design
from cla_engine.exact_baseline import fit_vocabulary
from cla_engine.common import Stop
from cla_engine.diagnostic_matrix import ARMS,_gradient_geometry,_schedule_digest
from cla_engine.singlecell_interpretation import association_candidate_features
from cla_engine.train import should_stop_early


def test_diagnostic_full_epoch_contract_disables_stale_early_stop():
    cfg={'training':{'diagnostic_full_epochs':True,'min_epochs':3,'patience':4}}
    assert not should_stop_early('diagnostic',cfg,72,9,4)
    assert should_stop_early('formal',cfg,72,9,4)


def test_hash4096_ablation_uses_same_rows_and_reports_collisions():
    rows=[('C',f'L{index}','T') for index in range(5000)]
    numeric=np.zeros((len(rows),3),dtype=np.float32)
    design=hashed_design(rows,numeric,4096)
    report=hash_collision_report(rows,4096)
    assert design.shape==(len(rows),4099)
    assert report['unique_train_identity_terms']>report['occupied_hash_buckets']
    assert report['colliding_identity_terms']>0


def test_exact_feature_block_ablation_keeps_declared_rows_and_columns():
    rows=[('C','L1','T'),('C','L2','T')];numeric=np.arange(8,dtype=np.float32).reshape(2,4)
    contract={'max_identity_features':100,'max_vocabulary_gib':.01}
    vectorizer,_=fit_vocabulary(rows,contract)
    design=_ablation_design(rows,numeric,vectorizer,[1,3],True)
    assert design.shape==(2,len(vectorizer.vocabulary_)+2)
    assert np.allclose(design[:,-2:].toarray(),numeric[:,[1,3]])
    assert design.indices.dtype==np.int32
    assert design.indptr.dtype==np.int32
    identity_only=_ablation_design(rows,numeric,vectorizer,[],True)
    assert identity_only.indices.dtype==np.int32
    assert identity_only.indptr.dtype==np.int32


def _sc_manifest():
    return pd.DataFrame([{'dataset_id':'D','cancer_id':'HNSC','formal_eligible':True,
        'model_derived':False,'outcome_derived':False,'source_tier':'approved',
        'quality_status':'PASS','feature_universe_status':'PASS','donor_metadata_available':True}])


def test_singlecell_association_is_exact_celltype_pair_context_without_family_broadcast():
    frame=pd.DataFrame([
        {'dataset_id':'D','cancer_id':'HNSC','cell_type_major':'B_cell','analysis_context':'overall',
         'lncrna_id':'L1','pathway_id':'P1','rho':-.4,'fdr':.01,'source_tier':'approved'},
        {'dataset_id':'D','cancer_id':'HNSC','cell_type_major':'T_cell','analysis_context':'overall',
         'lncrna_id':'L1','pathway_id':'P1','rho':.2,'fdr':.1,'source_tier':'approved'},
    ])
    rows=[('HNSC','L1','P1'),('HNSC','L1','P2')]
    features,report,_=association_candidate_features(frame,_sc_manifest(),rows,['B_cell','T_cell'],'HNSC')
    np.testing.assert_allclose(features[0],[-.4,.4,2.,1.,.2,.2,1.,1.],rtol=1e-6)
    np.testing.assert_array_equal(features[1],np.zeros(8,dtype=np.float32))
    assert report['candidate_rows_with_association']==1
    assert report['exact_pathway_join'] is True
    assert report['family_broadcast'] is False


def test_singlecell_association_never_uses_raw_p_as_fdr():
    frame=pd.DataFrame([{'dataset_id':'D','cancer_id':'HNSC','cell_type_major':'B_cell',
        'analysis_context':'overall','lncrna_id':'L1','pathway_id':'P1','rho':.4,
        'p_value':.01,'source_tier':'approved'}])
    with pytest.raises(Stop,match='FDR') as exc:
        association_candidate_features(frame,_sc_manifest(),[('HNSC','L1','P1')],['B_cell'],'HNSC')
    assert exc.value.code=='SC_ASSOCIATION_FDR_MISSING'


def test_g012_joint_contract_and_schedule_digest_are_fixed():
    assert ARMS=={
        'G0':{'variant':'G0','joint':True,'label':'G0_PATHWAY_PLUS_0.5_STATE'},
        'G1':{'variant':'G1','joint':True,'label':'G1_PATHWAY_PLUS_0.5_STATE'},
        'G2':{'variant':'G2','joint':True,'label':'G2_PATHWAY_PLUS_0.5_STATE'},
    }
    rows=[{'step':0,'view':'v','pathway':np.array([3,1]),'state':np.array([2])}]
    assert _schedule_digest(rows)==_schedule_digest(rows)


@pytest.mark.parametrize('drift', [None, 'numeric', 'coexpression', 'unfixed_background'])
def test_graph_comparison_enforces_identical_numeric_background(monkeypatch, tmp_path, drift):
    from cla_engine import diagnostic_matrix as matrix
    roots = {variant: tmp_path/variant for variant in ARMS}
    for root in roots.values():
        root.mkdir()
        (root/'MANIFEST.json').write_text('{}')
    def validated(root):
        spec = dict(nodes={}, states=[], cancers=['C'], numeric_features=2,
                    feature_dims={}, architecture='fixed', fold=0, graph_variant=root.name)
        cfg = dict(seed=1, model={}, training={},
                   genomics={'enabled':True, 'context_graph_variant':'G2'})
        if drift=='unfixed_background' and root.name=='G0':
            cfg['genomics'].pop('context_graph_variant')
        return {'metadata':{'profile':'diagnostic'}}, spec, cfg, {}
    def arrays(path):
        variant = path.parts[-4]
        if path.name!='graph.npz':
            return {'numeric':np.array([1., 2. if not (drift=='numeric' and variant=='G0') else 3.])}
        result={'x::lncRNA':np.ones((1,2)),
                'edge::lncRNA|coexpression_positive|gene':np.array([[0],[0]])}
        if variant in ('G1','G2'):
            result['edge::lncRNA|reported_binding|protein']=np.array([[0],[0]])
        if variant=='G2':
            result['edge::protein|reported_PPI|protein']=np.array([[0],[0]])
        if drift=='coexpression' and variant=='G0':
            result['edge::lncRNA|coexpression_positive|gene']=np.array([[0],[1]])
        return result
    monkeypatch.setattr(matrix,'validate_prepared',validated)
    monkeypatch.setattr(matrix,'read_json',lambda path:[{'id':'C__train2'}])
    monkeypatch.setattr(matrix,'load_npz',arrays)
    if drift:
        with pytest.raises(Stop):
            matrix._compare_prepared(roots['G0'],roots['G1'],roots['G2'])
    else:
        audit,_,_=matrix._compare_prepared(roots['G0'],roots['G1'],roots['G2'])
        assert audit['task_arrays_equal_including_candidates_labels_masks_weights_numeric_and_exact_lasso']
        assert audit['common_non_variant_graph_arrays_equal']


def test_execution_contract_allows_explicit_g2_numeric_background_for_g0(monkeypatch,tmp_path):
    from cla_engine import execution_contract as contract
    view={'id':'C__validation'};folder=tmp_path/'views'/view['id'];folder.mkdir(parents=True)
    (folder/'graph.npz').touch();(folder/'coexpression_support.npz').touch()
    payloads={
        'COEXPRESSION_AUDIT.json':{'contract':'PAIRWISE_COMMON_MASK_COEXPRESSION_R62_V1',
            'source_patient_hash':'h','minimum_pairs':3,'topk_per_lncrna':8,
            'estimable_pairs':1,'selected_edges':{'positive':1,'negative':1},'source_patients':5},
        'PROVENANCE.json':{'feature_patient_hash':'h','cancer':'C'},
        'GENOMIC_CONTEXT.json':{'partner_context_uses_binding':True,
            'binding_context':{'normalizer':'normalize_binding','cancer_id':'C'}},
    }
    monkeypatch.setattr('cla_engine.common.read_json',lambda path:payloads[path.name])
    class Loaded(dict):
        @property
        def files(self):return list(self)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    graph={'edge::lncRNA|coexpression_positive|gene':np.array([[0],[0]]),
           'weight::lncRNA|coexpression_positive|gene':np.array([.5]),
           'edge::lncRNA|coexpression_negative|gene':np.array([[0],[0]]),
           'weight::lncRNA|coexpression_negative|gene':np.array([.5])}
    support={'positive_n':np.array([5]),'positive_rank':np.array([1]),
             'negative_n':np.array([5]),'negative_rank':np.array([1])}
    monkeypatch.setattr(np,'load',lambda path,allow_pickle=False:Loaded(graph if path.name=='graph.npz' else support))
    contract.verify_graph_view_audit(tmp_path,{'graph_variant':'G0'},view,
        {'minimum_pairs':3,'graph':{'coexpression_topk':8},
         'genomics':{'enabled':True,'context_graph_variant':'G2'}})


class _ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.weight=nn.Parameter(torch.ones(1))


class _ToyPathway(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=_ToyEncoder()


class _ToyJoint(nn.Module):
    def __init__(self):
        super().__init__();self.pathway_model=_ToyPathway()


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__();self.joint=_ToyJoint();self.modal_gate=nn.Parameter(torch.ones(1));self.global_modal_gate=nn.Parameter(torch.ones(1))


def test_gradient_geometry_uses_weighted_same_step_shared_gradients():
    model=_ToyModel();weight=model.joint.pathway_model.encoder.weight
    result=_gradient_geometry(model,weight.sum(),-weight.sum(),7)
    assert result['pathway_weighted_gradient_norm']==1.
    assert result['state_0.5_weighted_gradient_norm']==.5
    assert result['weighted_gradient_cosine']==-1.
    assert result['gradient_conflict'] is True
