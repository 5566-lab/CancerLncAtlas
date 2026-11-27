"""Cancer scope from verified KEGG network identity and typed map membership.

Gene overlap never grants admission. This module does not read patient data,
lncRNA associations or model predictions, and does not redefine label values.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import csv
import hashlib
import json
import re
import socket

CANCER_CATEGORIES = frozenset({'Cancer', 'Cancer signaling'})
FORMAT = 'CLA_KEGG_NETWORK_SCOPE_V1'


def read_tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write_tsv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(value):
    return tuple(x.strip() for x in str(value or '').split('|') if x.strip())


def truth(value):
    if value is True or str(value).lower() in {'true', '1'}:
        return True
    if value is False or str(value).lower() in {'false', '0'}:
        return False
    raise ValueError('NETWORK_SCOPE_INVALID_BOOLEAN: ' + str(value))


def decide_network_scope(mappings, records, memberships):
    networks = {}
    for row in records:
        n = row['network_id']
        if not re.fullmatch(r'N\d{5}', n) or n in networks:
            raise ValueError('NETWORK_SCOPE_DUPLICATE_OR_INVALID_NETWORK: ' + n)
        networks[n] = row
    maps = defaultdict(dict)
    for row in memberships:
        n, mid = row['network_id'], row['map_id']
        if n not in networks:
            raise ValueError('NETWORK_SCOPE_MEMBERSHIP_WITHOUT_RECORD: ' + n)
        if not re.fullmatch(r'nt\d{5}', mid):
            raise ValueError('NETWORK_SCOPE_INVALID_MAP: ' + mid)
        if truth(row['is_cancer_map']) != (row['category'] in CANCER_CATEGORIES):
            raise ValueError('NETWORK_SCOPE_CATEGORY_FLAG_CONFLICT: ' + mid)
        if mid in maps[n] and maps[n][mid] != row:
            raise ValueError('NETWORK_SCOPE_MAP_CONFLICT: ' + n + ':' + mid)
        maps[n][mid] = row
    result, seen = [], {}
    for source in mappings:
        sid = source['source_target_id']
        status_in = source.get('mapping_status', '')
        if not sid or (sid in seen and ('mapped' in (seen[sid], status_in))):
            raise ValueError('NETWORK_SCOPE_DUPLICATE_SOURCE: ' + sid)
        seen[sid] = status_in
        n = source.get('network_id', '')
        record = networks.get(n, {})
        all_maps = maps[n]
        cancer_maps = sorted(mid for mid, row in all_maps.items() if truth(row['is_cancer_map']))
        mapped = source.get('mapping_status') == 'mapped'
        if not mapped:
            status = 'unresolved_source_identity'
        elif not record:
            status = 'unresolved_network_record'
        elif not record.get('name') or not record.get('definition'):
            status = 'incomplete_network_definition'
        elif cancer_maps:
            status = 'direct_cancer_network'
        elif not all_maps:
            status = 'unresolved_network_classification'
        else:
            status = 'outside_selected_network_categories'
        admitted = status == 'direct_cancer_network'
        result.append({
            **source,
            'network_name': record.get('name', ''),
            'network_type': record.get('network_type', ''),
            'mechanism_definition': record.get('definition', ''),
            'expanded_definition': record.get('expanded_definition', ''),
            'pathway_ids': record.get('pathway_ids', ''),
            'all_map_ids': '|'.join(sorted(all_maps)),
            'cancer_map_ids': '|'.join(cancer_maps),
            'cancer_map_names': '|'.join(all_maps[mid]['map_name'] for mid in cancer_maps),
            'cancer_categories': '|'.join(sorted({all_maps[mid]['category'] for mid in cancer_maps})),
            'cancer_scope_status': status,
            'admitted_by_network_scope': admitted,
            'alignment_key': 'KEGG.NETWORK:' + n if mapped else '',
            'network_source_url': record.get('source_url', ''),
            'network_source_sha256': record.get('source_sha256', ''),
        })
    return result


def _kegg_pathway(value):
    match = re.fullmatch(r'(?:path:)?(?:hsa|map)(\d{5})', value or '')
    return 'hsa' + match.group(1) if match else ''


def cross_database_links(scope, definitions, scope_reviews, external_relations, similarity_pairs=()):
    """Explicit identity review may transfer scope; parent links never merge it.

    A NETWORK element linked to hsa04310 is not thereby equivalent to every
    Reactome or WikiPathways definition mapped to the whole WNT pathway.
    """
    defs = {r['node_id']: r for r in definitions}
    nodes_by_network = defaultdict(list)
    for row in definitions:
        auth = row.get('authority_id', '')
        if auth.startswith('KEGG.NETWORK:'):
            nodes_by_network[auth.split(':', 1)[1]].append(row['node_id'])
    reviews_by_node = defaultdict(list)
    for row in scope_reviews:
        if row.get('relation') == 'same_functional_identity':
            if not row.get('evidence') or not row.get('biological_reason'):
                raise ValueError('NETWORK_SCOPE_REVIEW_EVIDENCE_MISSING')
            reviews_by_node[row['node_a']].append((row['node_b'], row))
            reviews_by_node[row['node_b']].append((row['node_a'], row))
    pairs_by_node = defaultdict(list)
    for row in similarity_pairs:
        pairs_by_node[row['node_a']].append((row['node_b'], row))
        pairs_by_node[row['node_b']].append((row['node_a'], row))
    result = []
    for src in scope:
        if not truth(src['admitted_by_network_scope']):
            continue
        base = {'source_target_id': src['source_target_id'], 'network_id': src['network_id'],
                'source_canonical_id': src['canonical_id'], 'scope_transfer': False,
                'identity_merge': False, 'shared_parent_pathway_id': '',
                'shared_proteins': '', 'jaccard': '', 'overlap_coefficient': ''}
        source_nodes = set(nodes_by_network[src['network_id']])
        # A recovered accession may refer to an old definition that lacks N.
        # Both historical anchors must agree; neither can override a contrary N.
        original = src.get('original_canonical_id', '')
        systematic = set(tokens(src.get('systematic_id', '')))
        original_hits, systematic_hits = set(), set()
        for row in definitions:
            if row.get('source_database') != 'KEGG_MEDICUS':
                continue
            if original and original in {row.get('node_id'), row.get('canonical_id')}:
                original_hits.add(row['node_id'])
            if systematic and row.get('systematic_source_id') in systematic:
                systematic_hits.add(row['node_id'])
        if original_hits and systematic_hits and original_hits != systematic_hits:
            raise ValueError('NETWORK_SCOPE_SOURCE_ANCHOR_CONFLICT: ' + src['source_target_id'])
        for node in original_hits | systematic_hits:
            authority = defs[node].get('authority_id', '')
            if authority.startswith('KEGG.NETWORK:') and authority != 'KEGG.NETWORK:' + src['network_id']:
                raise ValueError('NETWORK_SCOPE_RECOVERED_IDENTITY_CONFLICT: ' + node)
            source_nodes.add(node)
        for node in sorted(source_nodes):
            for target, review in reviews_by_node[node]:
                target_def = defs.get(target, {})
                if not target_def:
                    raise ValueError('NETWORK_SCOPE_REVIEW_TARGET_MISSING: ' + target)
                if target_def.get('source_database') == 'KEGG_MEDICUS':
                    continue
                # Existing reviewed identity is reused; no new union/merge is made.
                result.append({**base, 'target_node_id': target,
                    'target_canonical_id': target_def.get('integrated_canonical_id', ''),
                    'target_resource': target_def.get('source_database', ''),
                    'target_external_id': target_def.get('authority_id', ''),
                    'target_name': target_def.get('display_name', ''),
                    'alignment_relation': 'reviewed_equivalent_mechanism',
                    'scope_transfer': True, 'mapping_relation': review['relation'],
                    'evidence': review['evidence'], 'biological_reason': review['biological_reason']})
            for target, pair in pairs_by_node[node]:
                target_def = defs.get(target, {})
                if not target_def or target_def.get('source_database') == 'KEGG_MEDICUS':
                    continue
                result.append({**base, 'target_node_id': target,
                    'target_canonical_id': target_def.get('integrated_canonical_id', ''),
                    'target_resource': target_def.get('source_database', ''),
                    'target_external_id': target_def.get('authority_id', ''),
                    'target_name': target_def.get('display_name', ''),
                    'alignment_relation': 'protein_similarity_candidate',
                    'mapping_relation': 'protein_set_overlap',
                    'evidence': 'existing_frozen_protein_similarity',
                    'biological_reason': '', **{k: pair.get(k, '') for k in
                        ('shared_proteins', 'jaccard', 'overlap_coefficient')}})
        parents = set(tokens(src['pathway_ids']))
        for rel in external_relations:
            sp = _kegg_pathway(rel.get('source_external_id', '')) if 'kegg' in rel.get('source_resource', '').lower() else ''
            tp = _kegg_pathway(rel.get('target_external_id', '')) if 'kegg' in rel.get('target_resource', '').lower() else ''
            if sp in parents:
                prefix, parent = 'target', sp
            elif tp in parents:
                prefix, parent = 'source', tp
            else:
                continue
            if 'kegg' in rel.get(prefix + '_resource', '').lower():
                continue
            if rel.get('relation') not in {'equivalentTo', 'isPartOf'}:
                continue
            result.append({**base,
                'target_node_id': '',
                'target_canonical_id': rel.get(prefix + '_canonical_id', ''),
                'target_resource': rel.get(prefix + '_resource', ''),
                'target_external_id': rel.get(prefix + '_external_id', ''),
                'target_name': rel.get(prefix + '_name', ''),
                'alignment_relation': 'shared_parent_pathway_candidate',
                'shared_parent_pathway_id': parent,
                'mapping_relation': rel['relation'],
                'mapping_source_resource': rel.get('source_resource', ''),
                'mapping_source_external_id': rel.get('source_external_id', ''),
                'mapping_target_resource': rel.get('target_resource', ''),
                'mapping_target_external_id': rel.get('target_external_id', ''),
                'evidence': rel.get('mapping_source', ''),
                'biological_reason': 'The curated relation refers to the parent pathway, not the exact NETWORK element.'})
    # Duplicate evidence paths are not counted as independent support.
    unique = {json.dumps(row, sort_keys=True, default=str): row for row in result}
    return list(unique.values())


def verify_snapshot(root, manifest_path):
    root = Path(root).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    for item in manifest.get('sources', []) + manifest.get('outputs', []):
        path = (root / item['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha(path) != item['sha256']:
            raise ValueError('NETWORK_SCOPE_SNAPSHOT_CHECKSUM_MISMATCH: ' + str(path))
    if not manifest.get('sources') or not manifest.get('outputs'):
        raise ValueError('NETWORK_SCOPE_EMPTY_SNAPSHOT_MANIFEST')
    return manifest


def materialize_scope(mapping_path, snapshot_root, catalog_root, out,
                      *, snapshot_manifest=None, protein_pairs=None):
    if socket.gethostname() != '149':
        raise RuntimeError('CancerLncAtlas preparation must run on host 149')
    out, snap, cat = Path(out), Path(snapshot_root), Path(catalog_root)
    if out.exists() and any(out.iterdir()):
        raise ValueError('NETWORK_SCOPE_OUTPUT_EXISTS_USE_NEW_VERSION')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'LAUNCH_PREFLIGHT.json').write_text(json.dumps({
        'target_host': '149', 'operation': 'network_scope_materialization',
        'patient_data_used': False, 'training_started': False}, indent=2))
    manifest_path = Path(snapshot_manifest) if snapshot_manifest else snap/'network_snapshot/NETWORK_SNAPSHOT_MANIFEST.json'
    verified_manifest = verify_snapshot(snap, manifest_path)
    frozen_outputs = {item['path'] for item in verified_manifest['outputs']}
    required_outputs = {'network_snapshot/NETWORK_RECORDS.tsv',
                        'network_snapshot/NETWORK_MAP_MEMBERSHIP.tsv'}
    if not required_outputs.issubset(frozen_outputs):
        raise ValueError('NETWORK_SCOPE_REQUIRED_OUTPUT_NOT_FROZEN')
    paths = {
        'mapping': Path(mapping_path),
        'records': snap/'network_snapshot/NETWORK_RECORDS.tsv',
        'memberships': snap/'network_snapshot/NETWORK_MAP_MEMBERSHIP.tsv',
        'definitions': cat/'DEFINITION_TO_IDENTITY.tsv',
        'scope_reviews': cat/'FUNCTIONAL_SCOPE_DECISIONS.tsv',
        'external_relations': cat/'CURATED_EXTERNAL_RELATIONS.tsv',
    }
    rows = {key: read_tsv(path) for key, path in paths.items()}
    decisions = decide_network_scope(rows['mapping'], rows['records'], rows['memberships'])
    missing_path = snap/'network_snapshot/NETWORK_UNRESOLVED.tsv'
    missing_by_network = {}
    if missing_path.exists():
        if 'network_snapshot/NETWORK_UNRESOLVED.tsv' not in frozen_outputs:
            raise ValueError('NETWORK_SCOPE_REQUIRED_OUTPUT_NOT_FROZEN')
        paths['unresolved_records'] = missing_path
        missing_by_network = {r['network_id']: r for r in read_tsv(missing_path)}
    for row in decisions:
        missing = missing_by_network.get(row['network_id'], {})
        if missing and row['cancer_scope_status'] != 'unresolved_network_record':
            raise ValueError('NETWORK_SCOPE_MISSING_RECORD_CONFLICT: ' + row['network_id'])
        row['current_record_issue'] = missing.get('reason', '')
        row['current_record_issue_evidence_url'] = missing.get('source_url', '')
        row['current_record_issue_evidence_sha256'] = missing.get('source_sha256', '')
    pairs = []
    if protein_pairs:
        paths['protein_pairs'] = Path(protein_pairs)
        if str(protein_pairs).endswith('.parquet'):
            import pandas as pd
            pairs = pd.read_parquet(protein_pairs).fillna('').to_dict('records')
        else:
            pairs = read_tsv(protein_pairs)
    links = cross_database_links(decisions, rows['definitions'], rows['scope_reviews'],
                                 rows['external_relations'], pairs)
    decision_fields = list(decisions[0]) if decisions else ['source_target_id']
    write_tsv(out/'SOURCE_SCOPE_DECISIONS.tsv', decisions, decision_fields)
    write_tsv(out/'ADMITTED_SOURCE_DEFINITIONS.tsv',
              [r for r in decisions if r['admitted_by_network_scope']], decision_fields)
    link_fields = ('source_target_id','network_id','source_canonical_id','target_node_id',
                   'target_canonical_id','target_resource','target_external_id','target_name',
                   'alignment_relation','scope_transfer','identity_merge',
                   'shared_parent_pathway_id','mapping_relation','mapping_source_resource',
                   'mapping_source_external_id','mapping_target_resource',
                   'mapping_target_external_id','evidence','biological_reason',
                   'shared_proteins','jaccard','overlap_coefficient')
    write_tsv(out/'CROSS_DATABASE_ALIGNMENT.tsv', links, link_fields)
    counts = Counter(r['cancer_scope_status'] for r in decisions)
    admitted = [r for r in decisions if r['admitted_by_network_scope']]
    summary = {'format': FORMAT, 'target_host': '149',
               'source_definitions': len({r['source_target_id'] for r in decisions}), 'mapping_candidate_rows': len(decisions), 'scope_status_counts': dict(counts),
               'admitted_source_definitions': len(admitted),
               'admitted_unique_networks': len({r['network_id'] for r in admitted}),
               'admitted_existing_canonical_ids': len({r['canonical_id'] for r in admitted}),
               'cross_database_alignment_counts': dict(Counter(r['alignment_relation'] for r in links)),
               'new_identity_merges': 0, 'labels_recomputed': False,
               'formal_catalog_pointer_changed': False,
               'scope_basis': 'official Cancer/Cancer signaling map membership plus exact N identity',
               'cancer_categories': sorted(CANCER_CATEGORIES),
               'inputs': {str(path): sha(path) for path in paths.values()},
               'snapshot_manifest_sha256': sha(manifest_path)}
    outputs = ('SOURCE_SCOPE_DECISIONS.tsv','ADMITTED_SOURCE_DEFINITIONS.tsv','CROSS_DATABASE_ALIGNMENT.tsv')
    summary['outputs'] = {name: sha(out/name) for name in outputs}
    (out/'COMPLETE.json').write_text(json.dumps(summary, indent=2))
    return summary
