import numpy as np
import pytest
from cla_engine.functional_fusion import FusedFunctionalContext
from cla_engine.functional_target_view import project_context
from cla_engine.target_policy import policy_from_rows

def policies():
    rows=[]
    for cid,role,group,weight,display in [('p','primary_pathway','pathway',1.,'Pathway'),('tf','context_only','none',0.,'Pathway'),('s','state_auxiliary','state',.5,'State')]:
        rows.append(dict(canonical_id=cid,roles=display,biological_type='test',analysis_role=role,loss_group=group,loss_weight=weight,active_in_analysis=True,cancer_scope_status='test',backbone_anchor_ids=''))
    return policy_from_rows(rows)
def context():
    return FusedFunctionalContext('BRCA','train',('lnc',),('tf','s','p'),('State','State','Pathway'),('State','State','Pathway'),{}, {'observed':np.array([[1,0,1]]),'membership':np.array([[2,0,0]]),'combined_q':np.array([[.01,np.nan,.7]])})
def test_view_preserves_unlabelled_and_missing():
    result=project_context(context(),policies(),('p','s'))
    assert result.arrays['membership'].tolist()==[[0,0]]
    assert result.arrays['observed'].tolist()==[[1,0]]
    assert result.arrays['combined_q'][0,0]==.7
    assert np.isnan(result.arrays['combined_q'][0,1])
    assert result.task_weights().tolist()==[1.,.5]
    with pytest.raises(ValueError,match='UNLABELLED_IS_NOT_NEGATIVE'):
        result.binary_labels()
def test_context_only_regulator_cannot_be_projected_as_supervision():
    with pytest.raises(ValueError,match='UNSUPERVISED'):
        project_context(context(),policies(),('tf',))
def test_scope_projection_is_not_affected_by_removed_label_values():
    source=context();a=project_context(source,policies(),('p',))
    source.arrays['membership'][0,0]=0;source.arrays['combined_q'][0,0]=.99
    b=project_context(source,policies(),('p',))
    assert np.array_equal(a.arrays['membership'],b.arrays['membership'])
    assert np.array_equal(a.arrays['combined_q'],b.arrays['combined_q'])


def test_renamed_kegg_column_does_not_select_existing_reactome_column():
    source = FusedFunctionalContext(
        "BRCA", "train", ("lnc",), ("old_kegg", "p"),
        ("Pathway", "Pathway"), ("Pathway", "Pathway"), {},
        {"combined_q": np.array([[.04, .9]]),
         "combined_p": np.array([[.01, .8]]),
         "observed": np.array([[1, 1]]),
         "membership": np.array([[2, 0]])})
    projected = project_context(source, policies(), ("p",), ("old_kegg",))
    assert projected.canonical_ids == ("p",)
    assert projected.arrays["combined_q"].tolist() == [[.04]]
    assert projected.arrays["combined_p"].tolist() == [[.01]]
    assert projected.arrays["membership"].tolist() == [[2]]


def test_projection_rejects_reusing_one_source_column_for_two_targets():
    with pytest.raises(ValueError, match="NOT_ONE_TO_ONE"):
        project_context(context(), policies(), ("p", "s"), ("p", "p"))


def test_exact_source_readout_identity_precedes_current_merged_identity():
    from cla_engine.functional_target_view import compile_label_identity_map
    backbone = [{"source_target_id": "KEGG:MMR", "canonical_id": "merged", "network_id": "N01432"}]
    readouts = {
        "BRCA/train": [
            {"source_target_id": "KEGG:MMR", "canonical_id": "old_kegg"},
            {"source_target_id": "REACTOME:MMR", "canonical_id": "merged"}],
        "LUAD/train": []}
    definitions = [{"authority_id": "KEGG.NETWORK:N01432", "node_id": "old_kegg",
                    "integrated_canonical_id": "merged"}]
    result = compile_label_identity_map(backbone, readouts, definitions)
    assert result[0]["source_label_canonical_id"] == "old_kegg"
    assert result[0]["output_canonical_id"] == "merged"
    assert result[0]["evidence_contexts"] == "BRCA/train"


def test_conflicting_cross_context_exact_source_ids_fail():
    from cla_engine.functional_target_view import compile_label_identity_map
    backbone = [{"source_target_id": "KEGG:MMR", "canonical_id": "merged", "network_id": "N01432"}]
    readouts = {key: [{"source_target_id": "KEGG:MMR", "canonical_id": cid}]
                for key, cid in [("BRCA/train", "old_kegg"), ("LUAD/train", "other")]}
    with pytest.raises(ValueError, match="IDENTITY_UNRESOLVED"):
        compile_label_identity_map(backbone, readouts, [])


def test_network_definition_fallback_preserves_original_node():
    from cla_engine.functional_target_view import compile_label_identity_map
    backbone = [{"source_target_id": "KEGG:MMR", "canonical_id": "merged", "network_id": "N01432"}]
    definitions = [{"authority_id": "KEGG.NETWORK:N01432", "node_id": "old_kegg",
                    "integrated_canonical_id": "merged"}]
    result = compile_label_identity_map(backbone, {"BRCA/train": []}, definitions)
    assert result[0]["source_label_canonical_id"] == "old_kegg"
    assert result[0]["mapping_basis"] == "original_network_node_from_definition_to_identity"


def test_identity_map_cannot_silently_fuse_multiple_backbone_definitions():
    from cla_engine.functional_target_view import compile_label_identity_map
    backbone = [{"source_target_id": sid, "canonical_id": cid, "network_id": nid}
                for sid, cid, nid in [("KEGG:A", "a", "N00001"), ("KEGG:B", "b", "N00002")]]
    readouts = {"BRCA/train": [{"source_target_id": sid, "canonical_id": "same_old"}
                               for sid in ("KEGG:A", "KEGG:B")]}
    with pytest.raises(ValueError, match="NOT_ONE_TO_ONE"):
        compile_label_identity_map(backbone, readouts, [])


def test_old_view_without_source_label_mapping_is_not_silently_loaded(tmp_path):
    import json
    from cla_engine.functional_target_view import load_target_view
    (tmp_path / "COMPLETE.json").write_text(json.dumps({"files": {}}))
    with pytest.raises(ValueError, match="EXPLICIT_LABEL_MAPPING_REQUIRED"):
        load_target_view(tmp_path, "BRCA", "train")


def test_unqualified_conditional_readout_is_missing_not_negative_and_source_is_immutable():
    source=context()
    result=project_context(source,policies(),('p','s'),unavailable_target_ids=['p'])
    assert result.arrays['observed'].tolist()==[[0,0]]
    assert result.arrays['membership'].tolist()==[[0,0]]
    assert np.isnan(result.arrays['combined_q'][0,0])
    assert source.arrays['observed'][0,2]==1
    assert source.arrays['combined_q'][0,2]==.7

def test_unknown_conditional_target_is_rejected():
    with pytest.raises(ValueError,match='APPLICABILITY_AXIS_MISMATCH'):
        project_context(context(),policies(),('p',),unavailable_target_ids=['unknown'])

def test_original_msigdb_definition_is_valid_fallback_without_current_network_authority():
    from cla_engine.functional_target_view import compile_label_identity_map
    sid='MSIGDB:KEGG_MEDICUS:exact_variant'
    result=compile_label_identity_map(
        [{'source_target_id':sid,'canonical_id':'new','network_id':'N1'}],{},
        [{'authority_id':sid,'node_id':'old','integrated_canonical_id':'new'}])
    assert result[0]['source_label_canonical_id']=='old'
    assert result[0]['mapping_basis']=='original_exact_source_node_from_definition_to_identity'
