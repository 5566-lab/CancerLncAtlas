"""Deterministic grouping of nonempty, identical primary-pathway protein sets."""
from collections import defaultdict
import hashlib


def exact_member_groups(backbone, source_proteins):
    """Use original source members, without patient values or association inputs.

    A group is an observation unit. Original event definitions remain provenance.
    Empty annotations never establish identity. Reference definitions are preferred
    as representatives, then the existing canonical ID fixes ties reproducibly.
    """
    buckets = defaultdict(list)
    seen = set()
    for row in backbone:
        cid = row['canonical_id']
        if cid in seen:
            raise ValueError('DUPLICATE_PRIMARY_ID')
        seen.add(cid)
        proteins = frozenset(p for p in source_proteins.get(row['source_target_id'], ()) if p)
        key = ('proteins', tuple(sorted(proteins))) if proteins else ('unmapped', cid)
        buckets[key].append(dict(row))
    result = []
    for key, members in buckets.items():
        members.sort(key=lambda r: (r.get('network_type') != 'Reference', r['canonical_id']))
        representative = members[0]['canonical_id']
        result.append(dict(canonical_id=representative,
                           members=members,
                           protein_members=list(key[1]) if key[0] == 'proteins' else [],
                           member_set_sha256=hashlib.sha256('\n'.join(key[1]).encode()).hexdigest() if key[0] == 'proteins' else '',
                           merged=len(members) > 1))
    return sorted(result, key=lambda g: g['canonical_id'])


def merge_redirects(groups):
    return {r['canonical_id']: g['canonical_id'] for g in groups for r in g['members']}


def grouped_applicability(source, groups):
    """Scope shared member observations separately from their source events.

    Merged groups are defined by membership, so no specific upstream event is a
    prerequisite. The source event matrix must be retained by the caller; its
    unknown values are never interpreted as event-positive or event-negative.
    """
    result = source.copy()
    for group in groups:
        ids = [r['canonical_id'] for r in group['members']]
        if any(cid not in result for cid in ids):
            raise ValueError('MISSING_SOURCE_CONDITION_AXIS')
        if group['merged']:
            result[group['canonical_id']] = 1.0
            result = result.drop(columns=[cid for cid in ids if cid != group['canonical_id']])
    return result
