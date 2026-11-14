"""Exact KEGG molecular graph for source/label-separated relation prediction.

Original KEGG members define the supervised mechanisms. Cross-library programs
join as context nodes. Experimental RPI and STRING association evidence retain
separate relation types and their original confidence weights.
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .common import sha
from .mechanism_context import add_context_graph
from .prepared_target_policy import load_bound_policy


def _rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def _identity(value, kind):
    if not isinstance(value, str) or not value:
        raise ValueError('RELATION_GRAPH_MISSING_IDENTITY')
    prefix = {'gene': 'GENE:', 'protein': 'UNIPROT:', 'lncRNA': 'LNC:'}[kind]
    value = value[len(prefix):] if value.startswith(prefix) else value
    if kind in ('gene', 'lncRNA'):
        if not re.fullmatch(r'ENSG[0-9]{11}', value):
            raise ValueError('RELATION_GRAPH_GENE_NAMESPACE: ' + value)
    elif not re.fullmatch(r'[A-Z0-9]+(?:-[0-9]+)?', value):
        raise ValueError('RELATION_GRAPH_PROTEIN_NAMESPACE: ' + value)
    return 'LNC:' + value if kind == 'lncRNA' else value


def _weighted(frame, columns):
    if not set(columns + ['weight']) <= set(frame):
        raise ValueError('RELATION_GRAPH_SOURCE_SCHEMA')
    weight = pd.to_numeric(frame['weight'], errors='coerce').to_numpy(float)
    if not np.isfinite(weight).all() or np.any(weight <= 0) or np.any(weight > 1):
        raise ValueError('RELATION_GRAPH_CONFIDENCE_INVALID')
    result = frame.copy()
    result['weight'] = weight
    return result


def build_static_relation_graph(backbone, gene_sets, decisions, binding,
                                protein_gene, ppi, lncrna_ids, cancer,
                                context_links, context_components):
    """Construct topology without patient values, candidates or labels."""
    lncrna_ids = list(lncrna_ids)
    if not lncrna_ids or len(set(lncrna_ids)) != len(lncrna_ids):
        raise ValueError('RELATION_GRAPH_LNCRNA_AXIS_INVALID')
    if any(_identity(v, 'lncRNA') != v for v in lncrna_ids):
        raise ValueError('RELATION_GRAPH_LNCRNA_AXIS_NAMESPACE')
    primary = {row['source_target_id']: row for row in backbone}
    if (not primary or len(primary) != len(backbone)
            or len({r['canonical_id'] for r in backbone}) != len(backbone)
            or set(primary) - set(gene_sets)):
        raise ValueError('RELATION_GRAPH_PRIMARY_AXIS_INVALID')
    members = {(sid, _identity(g, 'gene')) for sid in primary for g in gene_sets[sid]}
    if {sid for sid, _ in members} != set(primary):
        raise ValueError('RELATION_GRAPH_PRIMARY_MEMBERS_MISSING')

    rpi = _weighted(binding, ['lncrna_id', 'protein_id'])
    if not {'evidence_class', 'cancer_id', 'is_context_specific'} <= set(rpi):
        raise ValueError('RELATION_GRAPH_RPI_EVIDENCE_REQUIRED')
    if not rpi.evidence_class.eq('experimental_physical').all():
        raise ValueError('RELATION_GRAPH_RPI_NOT_EXPERIMENTAL_PHYSICAL')
    if not rpi.is_context_specific.isin([True, False]).all():
        raise ValueError('RELATION_GRAPH_RPI_CONTEXT_FLAG_INVALID')
    context = rpi.cancer_id.fillna('').astype(str)
    global_context = context.isin(['', 'GLOBAL'])
    if (global_context & rpi.is_context_specific).any():
        raise ValueError('RELATION_GRAPH_RPI_CONTEXT_CONFLICT')
    rpi['cancer_id'] = context.where(~global_context, 'GLOBAL')
    rpi['lncrna_id'] = rpi.lncrna_id.map(lambda v: _identity(v, 'lncRNA'))
    rpi['protein_id'] = rpi.protein_id.map(lambda v: _identity(v, 'protein'))
    total_rpi = len(rpi)
    rpi = rpi.loc[rpi.lncrna_id.isin(lncrna_ids) & rpi.cancer_id.isin(['GLOBAL', cancer])]
    ppi = _weighted(ppi, ['protein_id_a', 'protein_id_b'])
    for column in ('protein_id_a', 'protein_id_b'):
        ppi[column] = ppi[column].map(lambda v: _identity(v, 'protein'))
    pg = _weighted(protein_gene, ['protein_id', 'gene_id'])
    pg['protein_id'] = pg.protein_id.map(lambda v: _identity(v, 'protein'))
    pg['gene_id'] = pg.gene_id.map(lambda v: _identity(v, 'gene'))
    multiplicity = pg.groupby('protein_id').gene_id.nunique()
    ambiguous = set(multiplicity.index[multiplicity > 1])
    unique_pg = pg.loc[~pg.protein_id.isin(ambiguous)]
    context_components = [dict(row) for row in context_components]
    for row in context_components:
        row['uniprot_id'] = _identity(row['uniprot_id'], 'protein')
        if row.get('gene_id'):
            row['gene_id'] = _identity(row['gene_id'], 'gene')
    genes = sorted({g for _, g in members} | set(pg.gene_id)
                   | {r['gene_id'] for r in context_components if r.get('gene_id')})
    proteins = sorted(set(pg.protein_id) | set(rpi.protein_id)
                      | set(ppi.protein_id_a) | set(ppi.protein_id_b)
                      | {r['uniprot_id'] for r in context_components})
    scope = {row['source_target_id']: row for row in decisions}
    if set(primary) - set(scope):
        raise ValueError('RELATION_GRAPH_NETWORK_CLASSIFICATION_MISSING')
    family_pairs, singleton_count = [], 0
    for sid, anchor in primary.items():
        classes = [v for v in scope[sid].get('all_map_ids', '').split('|') if v]
        if classes and any(not re.fullmatch(r'nt[0-9]{5}', v) for v in classes):
            raise ValueError('RELATION_GRAPH_NETWORK_CLASSIFICATION_INVALID')
        if not classes:
            singleton_count += 1
        families = ['KEGG.NETWORK_MAP:' + v for v in classes] or [anchor['backbone_id']]
        family_pairs.extend((sid, f) for f in families)
    ids = {'lncRNA': lncrna_ids, 'gene': genes, 'protein': proteins,
           'pathway': sorted(primary), 'pathway_family': sorted({f for _, f in family_pairs}),
           'cancer': [cancer]}
    maps = {kind: {v: i for i, v in enumerate(axis)} for kind, axis in ids.items()}
    relations, edges = [], {}

    def add(src, relation, dst, records, reverse=True):
        collapsed = {}
        for a, b, weight, ctx in records:
            if a not in maps[src] or b not in maps[dst]:
                raise ValueError('RELATION_GRAPH_UNMAPPED_EDGE_ENDPOINT')
            key = (maps[src][a], maps[dst][b], ctx)
            collapsed[key] = max(float(weight), collapsed.get(key, 0.))
        order = sorted(collapsed)
        index = np.asarray([[a, b] for a, b, _ in order], np.int64).reshape(-1, 2).T
        values = np.asarray([collapsed[k] for k in order], np.float32)
        if not np.isfinite(values).all() or np.any(values <= 0) or np.any(values > 1):
            raise ValueError('RELATION_GRAPH_EDGE_WEIGHT_INVALID')
        entry = {'edge_index': index, 'weight': values,
                 'context': np.asarray([c for _, _, c in order], str), 'sign': 1}
        rel = (src, relation, dst)
        edges['|'.join(rel)] = entry
        relations.append(rel)
        if reverse:
            rev = (dst, 'rev_' + relation, src)
            edges['|'.join(rev)] = {**entry, 'edge_index': index[::-1].copy()}
            relations.append(rev)

    add('gene', 'membership_unspecified', 'pathway', ((g, sid, 1., '') for sid, g in sorted(members)))
    add('pathway', 'member_of_family', 'pathway_family', ((s, f, 1., '') for s, f in family_pairs))
    add('protein', 'encoded_by', 'gene', ((r.protein_id, r.gene_id, r.weight, '') for r in unique_pg.itertuples()))
    add('lncRNA', 'reported_binding', 'protein', ((r.lncrna_id, r.protein_id, r.weight, r.cancer_id) for r in rpi.itertuples()))
    add('protein', 'reported_PPI', 'protein',
        ((a, b, r.weight, '') for r in ppi.itertuples()
         for a, b in ((r.protein_id_a, r.protein_id_b), (r.protein_id_b, r.protein_id_a)) if a != b), reverse=False)
    context_audit = add_context_graph(ids, maps, relations, edges, context_links, context_components)
    audit = {'node_counts': {k: len(v) for k, v in ids.items()},
             'edge_counts': {k: len(v['weight']) for k, v in edges.items()},
             'primary_membership_edges': len(members), 'original_KEGG_members_only': True,
             'cross_library_context_expands_primary_membership': False,
             'ambiguous_protein_gene_encodings_omitted': len(ambiguous),
             'ambiguous_proteins': sorted(ambiguous),
             'rpi_source_rows': total_rpi, 'rpi_selected_rows': len(rpi),
             'ppi_semantics': 'STRING association confidence; not asserted physical or causal',
             'unclassified_network_singleton_families': singleton_count,
             'context': context_audit, 'patient_values_read': False, 'label_values_read': False}
    return ids, maps, relations, edges, audit


def load_static_relation_graph(catalog, catalog_receipt_sha256, authority_binding,
                               lncrna_ids, cancer):
    """Resolve and hash-check current catalog and experimental network sources."""
    catalog = Path(catalog)
    if sha(catalog / 'COMPLETE.json') != catalog_receipt_sha256:
        raise ValueError('RELATION_GRAPH_CATALOG_RECEIPT_DRIFT')
    receipt = json.loads((catalog / 'COMPLETE.json').read_text())
    for name in ('PREPARE_POLICY_BINDING.json', 'KEGG_SOURCE_SCOPE_DECISIONS.tsv'):
        if sha(catalog / name) != receipt['files'][name]:
            raise ValueError('RELATION_GRAPH_CATALOG_FILE_DRIFT: ' + name)
    cfg = json.loads((catalog / 'PREPARE_POLICY_BINDING.json').read_text())
    bound = load_bound_policy(cfg)
    authority = Path(authority_binding['path'])
    if sha(authority) != authority_binding['sha256']:
        raise ValueError('RELATION_GRAPH_AUTHORITY_DRIFT')
    source = json.loads(authority.read_text())['artifacts']
    names = ('lnc_protein_binding_experimental', 'protein_gene_encoding', 'ppi')
    frames, files = {}, {}
    for name in names:
        item = source[name]
        if any(item.get(k) is not False for k in ('historical_model_output', 'outcome_derived', 'toy_or_synthetic')):
            raise ValueError('RELATION_GRAPH_INELIGIBLE_SOURCE: ' + name)
        path = Path(item['path'])
        if sha(path) != item['sha256']:
            raise ValueError('RELATION_GRAPH_SOURCE_DRIFT: ' + name)
        frames[name] = pd.read_parquet(path)
        files[name] = {'path': str(path), 'sha256': item['sha256']}
    original = json.loads((bound['backbone_path'] / 'GENE_SETS.json').read_text())
    result = build_static_relation_graph(bound['backbone']['rows'], original,
        _rows(catalog / 'KEGG_SOURCE_SCOPE_DECISIONS.tsv'),
        frames['lnc_protein_binding_experimental'], frames['protein_gene_encoding'],
        frames['ppi'], lncrna_ids, cancer,
        bound['context_graph']['MECHANISM_CONTEXT_LINKS.tsv'],
        bound['context_graph']['MECHANISM_CONTEXT_COMPONENTS.tsv'])
    result[-1]['source_bindings'] = files
    result[-1]['catalog_receipt_sha256'] = catalog_receipt_sha256
    result[-1]['authority_binding'] = authority_binding
    return result
