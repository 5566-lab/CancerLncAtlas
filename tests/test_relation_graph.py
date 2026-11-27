import numpy as np
import pandas as pd
import pytest

from cla_engine.relation_graph import build_static_relation_graph


def fixture():
    lnc = 'LNC:ENSG00000000001'
    sid = 'MSIGDB:KEGG_MEDICUS:EXACT'
    return dict(
        backbone=[{'source_target_id': sid, 'canonical_id': 'KEGG_A',
                   'backbone_id': 'KEGG.NETWORK:N00001'}],
        gene_sets={sid: ['ENSG00000000002']},
        decisions=[{'source_target_id': sid, 'all_map_ids': 'nt00001|nt00002'}],
        binding=pd.DataFrame([
            {'lncrna_id': lnc, 'protein_id': 'UNIPROT:P00001', 'weight': .8,
             'evidence_class': 'experimental_physical', 'cancer_id': None, 'is_context_specific': False},
            {'lncrna_id': lnc, 'protein_id': 'UNIPROT:P00002', 'weight': .9,
             'evidence_class': 'experimental_physical', 'cancer_id': 'LUAD', 'is_context_specific': True}]),
        protein_gene=pd.DataFrame([
            {'protein_id': 'UNIPROT:P00001', 'gene_id': 'GENE:ENSG00000000002', 'weight': 1.},
            {'protein_id': 'UNIPROT:P00002', 'gene_id': 'GENE:ENSG00000000003', 'weight': 1.}]),
        ppi=pd.DataFrame([{'protein_id_a': 'UNIPROT:P00001', 'protein_id_b': 'UNIPROT:P00002', 'weight': .7}]),
        lncrna_ids=[lnc], cancer='BRCA',
        context_links=[{'mechanism_id': 'CTX', 'pathway_source_id': sid, 'weight': .5,
                        'relation': 'shared_protein_components'}],
        context_components=[{'mechanism_id': 'CTX', 'uniprot_id': 'P00002', 'gene_id': 'ENSG00000000003'}])


def test_exact_membership_survives_context_and_network_union():
    cfg = fixture()
    ids, maps, relations, edges, audit = build_static_relation_graph(**cfg)
    membership = edges['gene|membership_unspecified|pathway']['edge_index']
    assert membership.shape == (2, 1)
    assert ids['gene'][membership[0, 0]] == 'ENSG00000000002'
    assert 'ENSG00000000003' in ids['gene']
    assert audit['context']['mapped_mechanism_nodes'] == 1
    assert audit['context']['independent_loss'] is False
    assert len(ids['pathway']) == 1
    assert len(ids['pathway_family']) == 2
    assert len(edges['lncRNA|reported_binding|protein']['weight']) == 1
    assert np.allclose(edges['lncRNA|reported_binding|protein']['weight'], [.8])
    assert len(edges['protein|reported_PPI|protein']['weight']) == 2
    assert len(relations) == len(set(relations))


def test_ambiguous_encoding_retains_protein_but_does_not_force_gene():
    cfg = fixture()
    cfg['protein_gene'] = pd.concat([cfg['protein_gene'], pd.DataFrame([
        {'protein_id': 'UNIPROT:P00001', 'gene_id': 'GENE:ENSG00000000004', 'weight': .5}])], ignore_index=True)
    ids, maps, _, edges, audit = build_static_relation_graph(**cfg)
    assert 'P00001' in ids['protein']
    encoded_proteins = edges['protein|encoded_by|gene']['edge_index'][0]
    assert maps['protein']['P00001'] not in encoded_proteins
    assert audit['ambiguous_proteins'] == ['P00001']


@pytest.mark.parametrize('column,value,error', [
    ('evidence_class', 'computational', 'NOT_EXPERIMENTAL_PHYSICAL'),
    ('is_context_specific', True, 'CONTEXT_CONFLICT'),
    ('weight', np.nan, 'CONFIDENCE_INVALID'),
    ('protein_id', 'GENE:ENSG00000000002', 'PROTEIN_NAMESPACE')])
def test_ineligible_or_mistyped_binding_fails(column, value, error):
    cfg = fixture()
    cfg['binding'].loc[0, column] = value
    with pytest.raises(ValueError, match=error):
        build_static_relation_graph(**cfg)


def test_unknown_context_is_not_silently_global():
    cfg = fixture()
    cfg['binding']['is_context_specific'] = cfg['binding']['is_context_specific'].astype(object)
    cfg['binding'].loc[0, 'is_context_specific'] = None
    with pytest.raises(ValueError, match='CONTEXT_FLAG_INVALID'):
        build_static_relation_graph(**cfg)


def test_duplicate_source_edges_keep_max_confidence():
    cfg = fixture()
    cfg['ppi'] = pd.concat([cfg['ppi'], cfg['ppi'].assign(weight=.6)], ignore_index=True)
    _, _, _, edges, _ = build_static_relation_graph(**cfg)
    assert np.allclose(edges['protein|reported_PPI|protein']['weight'], [.7, .7])
