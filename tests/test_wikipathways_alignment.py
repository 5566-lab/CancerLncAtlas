import pytest
from cla_engine.wikipathways_alignment import build_wikipathways_alignment


def example(exact=False):
    definitions = [{'database_family':'WIKIPATHWAYS','node_id':'REF:WikiPathways:WP1',
                    'canonical_id':'','authority_id':'WP1','display_name':'Example source',
                    'integrated_canonical_id':'anchor' if exact else 'CLA-REF-1'}]
    composition = [{'node_id':'universe','source_definition':'background','uniprot_id':f'P{i}'} for i in range(100)]
    composition += [{'node_id':'REF:WikiPathways:WP1','source_definition':'WP1','uniprot_id':f'P{i}'} for i in range(3)]
    composition += [{'node_id':'anchor','source_definition':'KEGG:1','uniprot_id':f'P{i}'} for i in range(3)]
    # A separate source definition under the same identity cannot expand the KEGG anchor.
    composition += [{'node_id':'anchor','source_definition':'OTHER','uniprot_id':'P99'}]
    backbone = [{'canonical_id':'anchor','source_target_id':'KEGG:1','display_name':'Exact mechanism','network_id':'N1'}]
    return definitions, composition, backbone


def test_external_ref_nodes_enter_alignment_without_becoming_equivalent():
    candidates, routes, disposition, summary = build_wikipathways_alignment(*example())
    assert summary['source_count'] == 1
    assert disposition[0]['disposition'] == 'composition_context_supported'
    assert routes[0]['relation'] == 'protein_component_association'
    assert not routes[0]['whole_source_readout_is_exact_target_observation']
    assert not routes[0]['identity_merge']
    assert routes[0]['target_protein_count'] == 3
    assert candidates[0]['jaccard'] == 1.0


def test_only_previously_reviewed_equivalence_routes_whole_source_score():
    _, routes, disposition, _ = build_wikipathways_alignment(*example(exact=True))
    assert len(routes) == 1
    assert routes[0]['relation'] == 'same_functional_identity'
    assert routes[0]['whole_source_readout_is_exact_target_observation']
    assert not routes[0]['score_averaging_allowed']
    assert disposition[0]['exact_target_id'] == 'anchor'


def test_previous_zero_overlap_tests_remain_in_global_bh_family():
    candidates, routes, disposition, summary = build_wikipathways_alignment(
        *example(), previous_summary={'protein_universe_size':100,'tested_pairs':10000})
    assert candidates[0]['overlap_q_wp_family'] < .05
    assert candidates[0]['overlap_q'] > .05
    assert routes == []
    assert summary['total_multiplicity_tests'] == 10001
    assert disposition[0]['disposition'] == 'nonzero_overlap_below_threshold'


def test_frozen_universe_drift_is_rejected():
    with pytest.raises(ValueError, match='FROZEN_PROTEIN_UNIVERSE_CHANGED'):
        build_wikipathways_alignment(*example(), previous_summary={'protein_universe_size':99})


def test_zero_overlap_is_an_explicit_source_disposition():
    defs, composition, backbone = example()
    composition = [r for r in composition if r['node_id'] != 'REF:WikiPathways:WP1']
    composition += [{'node_id':'REF:WikiPathways:WP1','source_definition':'WP1','uniprot_id':'P98'}]
    candidates, routes, disposition, summary = build_wikipathways_alignment(defs,composition,backbone)
    assert candidates == [] and routes == []
    assert disposition[0]['disposition'] == 'zero_backbone_overlap'
    assert summary['zero_overlap_pairs'] == 1


def test_joint_family_refreshes_previous_decisions_too():
    from cla_engine.wikipathways_alignment import merge_alignment_families
    previous = [{'source_canonical_id':'old','target_canonical_id':'anchor',
                 'source_database':'REACTOME','shared_proteins':3,'overlap_p':.01,
                 'overlap_q':.01,'admitted_context_link':'True','direct_regulator_proteins':''}]
    merged, changes, summary = merge_alignment_families(
        previous, {'tested_pairs':1}, [], 1000)
    assert merged[0]['overlap_q'] == 1.0
    assert merged[0]['admitted_context_link'] == 'False'
    assert changes[0]['change'] == 'withdrawn'
    assert summary['previous_links_withdrawn'] == 1
    assert summary['joint_tested_pairs'] == 1001
