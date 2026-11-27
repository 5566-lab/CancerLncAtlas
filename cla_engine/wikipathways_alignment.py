"""Route frozen WikiPathways definitions to the exact KEGG backbone.

External REF nodes remain source definitions. Protein-set enrichment produces
composition context, never an equivalence decision or an activity label.
"""
from collections import defaultdict
import numpy as np
from .backbone_alignment import build_alignment, bh


def build_wikipathways_alignment(definitions, composition, backbone, *,
                                  previous_alignment=(), previous_summary=None):
    wp = [dict(r) for r in definitions if r.get('database_family') == 'WIKIPATHWAYS']
    ids = [r['node_id'] for r in wp]
    if len(ids) != len(set(ids)):
        raise ValueError('DUPLICATE_WIKIPATHWAYS_REF_NODE')
    anchors = {r['canonical_id']: dict(r) for r in backbone}
    if len(anchors) != len(backbone):
        raise ValueError('DUPLICATE_BACKBONE_ID')
    by_node, by_source = defaultdict(set), defaultdict(set)
    universe = set()
    for r in composition:
        u = r.get('uniprot_id', '')
        if not u:
            continue
        universe.add(u)
        by_node[r['node_id']].add(u)
        by_source[r['source_definition']].add(u)
    previous_summary = previous_summary or {}
    expected_size = previous_summary.get('protein_universe_size')
    if expected_size is not None and int(expected_size) != len(universe):
        raise ValueError('FROZEN_PROTEIN_UNIVERSE_CHANGED')
    protein_sets = {'__frozen_catalog_universe__': universe}
    for cid, anchor in anchors.items():
        source = anchor['source_target_id']
        members = by_source.get(source, set())
        if not members:
            raise ValueError('ORIGINAL_KEGG_MEMBERS_MISSING:' + source)
        protein_sets[cid] = set(members)
    master = []
    exact = {}
    for row in wp:
        node = row['node_id']
        if not by_node.get(node):
            raise ValueError('WIKIPATHWAYS_MEMBERS_MISSING:' + node)
        protein_sets[node] = set(by_node[node])
        master.append({'canonical_id': node, 'display_name': row['display_name'],
                       'source_database': 'WIKIPATHWAYS'})
        integrated = row.get('integrated_canonical_id', '')
        if integrated in anchors:
            exact[node] = integrated
    candidates, _, _, added_summary = build_alignment(master, list(anchors.values()), protein_sets)
    previous = list(previous_alignment)
    prior_count = int(previous_summary.get('tested_pairs', 0))
    if prior_count < len(previous):
        raise ValueError('PREVIOUS_TESTED_PAIR_COUNT_TOO_SMALL')
    new_count = len(wp) * len(anchors)
    # Keep zero-overlap tests (p=1) and the original comparisons in the family.
    if prior_count:
        old_p = [float(r['overlap_p']) for r in previous]
        all_p = np.ones(prior_count + new_count, dtype=float)
        all_p[:len(old_p)] = old_p
        all_p[prior_count:prior_count + len(candidates)] = [float(r['overlap_p']) for r in candidates]
        adjusted = bh(all_p)[prior_count:prior_count + len(candidates)]
    else:
        adjusted = [r['overlap_q'] for r in candidates]
    lookup = {}
    for row, q in zip(candidates, adjusted):
        row['overlap_q_wp_family'] = float(row['overlap_q'])
        row['overlap_q'] = float(q)
        row['admitted_context_link'] = str(row['shared_proteins'] >= 3 and q <= .05)
        row['multiplicity_family'] = 'existing_frozen_comparisons_plus_all_WP_x_current_backbone' if prior_count else 'all_WP_x_current_backbone'
        row['identity_merge'] = 'False'
        row['mechanism_context_inherited'] = 'False'
        lookup[(row['source_canonical_id'], row['target_canonical_id'])] = row
    wp_by_node = {r['node_id']: r for r in wp}
    routes = []
    def add_route(node, target, relation, stats):
        a, b = protein_sets[node], protein_sets[target]
        source = wp_by_node[node]
        routes.append({'source_node_id': node, 'source_external_id': source['authority_id'], 'source_target_id': source['authority_id'],
                       'source_name': source['display_name'], 'source_database': 'WIKIPATHWAYS',
                       'target_canonical_id': target, 'canonical_id': target, 'route_type': 'equivalent_definition' if relation == 'same_functional_identity' else 'composition_context', 'target_name': anchors[target].get('display_name', ''),
                       'network_id': anchors[target].get('network_id', ''), 'relation': relation,
                       'shared_proteins': len(a & b), 'source_protein_count': len(a),
                       'target_protein_count': len(b), 'jaccard': len(a & b) / len(a | b),
                       'overlap_p': stats.get('overlap_p', 1), 'overlap_q': stats.get('overlap_q', 1),
                       'shared_uniprot_ids': ';'.join(sorted(a & b)),
                       'source_uniprot_ids': ';'.join(sorted(a)),
                       'whole_source_readout_is_exact_target_observation': relation == 'same_functional_identity',
                       'shared_member_states_can_link_to_target': True,
                       'identity_merge': relation == 'same_functional_identity',
                       'mechanism_context_inherited': False,
                       'routing_basis': 'existing_reviewed_functional_scope' if relation == 'same_functional_identity' else 'protein_set_enrichment_q_le_0.05_shared_ge_3',
                       'score_averaging_allowed': False})
    for node, target in exact.items():
        add_route(node, target, 'same_functional_identity', lookup.get((node, target), {}))
    for row in candidates:
        node, target = row['source_canonical_id'], row['target_canonical_id']
        if row['admitted_context_link'] == 'True' and exact.get(node) != target:
            add_route(node, target, 'protein_component_association', row)
    by_source = defaultdict(list)
    for row in routes:
        by_source[row['source_node_id']].append(row)
    dispositions = []
    backbone_union = set().union(*(protein_sets[c] for c in anchors)) if anchors else set()
    for source in wp:
        node = source['node_id']; members = protein_sets[node]; related = by_source[node]
        overlap = members & backbone_union
        state = ('exact_equivalent' if node in exact else 'composition_context_supported' if related
                 else 'nonzero_overlap_below_threshold' if overlap else 'zero_backbone_overlap')
        dispositions.append({'source_node_id': node, 'source_external_id': source['authority_id'], 'source_target_id': source['authority_id'],
                             'source_name': source['display_name'], 'source_protein_count': len(members),
                             'shared_with_backbone_union': len(overlap), 'disposition': state,
                             'exact_target_id': exact.get(node, ''), 'context_target_count': sum(r['relation'] != 'same_functional_identity' for r in related),
                             'all_linked_target_count': len(related),
                             'shared_member_count_in_admitted_routes': len(set().union(*(set(r['shared_uniprot_ids'].split(';')) for r in related))) if related else 0})
    summary = {'source_count': len(wp), 'backbone_count': len(anchors),
               'protein_universe_size': len(universe), 'original_tested_pairs': prior_count,
               'new_tested_pairs': new_count, 'total_multiplicity_tests': prior_count + new_count,
               'nonzero_overlap_pairs': len(candidates), 'zero_overlap_pairs': new_count - len(candidates),
               'exact_identity_sources': len(exact), 'context_route_count': sum(r['relation'] != 'same_functional_identity' for r in routes),
               'selection': 'frozen protein universe; original KEGG members; BH q<=0.05 and >=3 shared proteins',
               'context_does_not_merge_identity': True, 'patient_expression_used': False,
               'lncrna_information_used': False}
    for status in ['exact_equivalent', 'composition_context_supported', 'nonzero_overlap_below_threshold', 'zero_backbone_overlap']:
        summary[status] = sum(r['disposition'] == status for r in dispositions)
    return candidates, routes, dispositions, summary



def merge_alignment_families(previous_alignment, previous_summary,
                             new_candidates, new_tested_pairs):
    """Refresh both old and new q-values in one frozen comparison family.

    Missing rows are known zero-overlap tests with p=1, not omitted unknowns.
    Direct regulator membership remains an identity-based context rule.
    """
    previous = [dict(r) for r in previous_alignment]
    added = [dict(r) for r in new_candidates]
    old_count = int(previous_summary['tested_pairs'])
    if old_count < len(previous) or int(new_tested_pairs) < len(added):
        raise ValueError('TESTED_PAIR_COUNT_SMALLER_THAN_NONZERO_ROWS')
    p = np.ones(old_count + int(new_tested_pairs), dtype=float)
    p[:len(previous)] = [float(r['overlap_p']) for r in previous]
    p[old_count:old_count + len(added)] = [float(r['overlap_p']) for r in added]
    q = bh(p)
    changes = []
    old_flags, new_flags = [], []
    for i, row in enumerate(previous):
        before = str(row.get('admitted_context_link', '')).lower() == 'true'
        old_q = float(row['overlap_q'])
        direct = bool(row.get('direct_regulator_proteins', ''))
        after = bool(direct or (int(row['shared_proteins']) >= 3 and q[i] <= .05))
        row['overlap_q'] = float(q[i])
        row['admitted_context_link'] = str(bool(after))
        old_flags.append(before); new_flags.append(after)
        if before != after:
            changes.append({'source_canonical_id': row['source_canonical_id'],
                            'target_canonical_id': row['target_canonical_id'],
                            'source_database': row.get('source_database', ''),
                            'shared_proteins': row['shared_proteins'],
                            'overlap_q_previous': old_q, 'overlap_q_joint': float(q[i]),
                            'previous_admitted': before, 'joint_admitted': bool(after),
                            'change': 'added' if after else 'withdrawn'})
    for i, row in enumerate(added):
        row['overlap_q'] = float(q[old_count + i])
        direct = bool(row.get('direct_regulator_proteins', ''))
        row['admitted_context_link'] = str(bool(direct or (int(row['shared_proteins']) >= 3 and row['overlap_q'] <= .05)))
    summary = {'previous_tested_pairs': old_count, 'new_tested_pairs': int(new_tested_pairs),
               'joint_tested_pairs': len(p), 'previous_context_links': sum(old_flags),
               'previous_links_retained': sum(a and b for a,b in zip(old_flags,new_flags)),
               'previous_links_added': sum((not a) and b for a,b in zip(old_flags,new_flags)),
               'previous_links_withdrawn': sum(a and (not b) for a,b in zip(old_flags,new_flags)),
               'previous_context_links_after_joint_bh': sum(new_flags),
               'new_context_links_after_joint_bh': sum(str(r['admitted_context_link']).lower() == 'true' for r in added)}
    return previous + added, changes, summary
