"""Bind measured readouts to exact pathway identities and molecular components.

Routing is structural: this module never reads expression or association values.
An equivalent pathway measurement and a member regulator measurement have
different evidence roles, even when both support the same relationship target.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

FORMAT = 'CLA_MULTIMODAL_RELATION_BINDINGS_V1'
FIELDS = ('cohort', 'layer', 'column_index', 'source_target_id', 'target_id',
          'source_canonical_id', 'resolved_source_canonical_id',
          'output_canonical_id', 'network_id', 'measurement_layer',
          'evidence_role', 'scope_relation', 'molecular_components',
          'condition_required', 'routing_basis')


def _truth(value):
    return value is True or value == 'True'


def _tokens(value):
    return {x.strip() for x in str(value or '').replace(';', '|').split('|') if x.strip()}


def compile_bindings(*, backbone, definitions, source_bindings, policies,
                     alignments, kinase_components, readouts):
    """Return explicit supervision bindings and unbound source dispositions.

    Readouts contain cohort/layer/column_index and their original TARGETS.tsv
    fields. Their row position is the association-array column position.
    Conditional mechanisms retain an applicability requirement on every route.
    Display categories do not assign a loss or create a new target.
    """
    primary = {r['canonical_id']: r for r in backbone}
    if len(primary) != len(backbone):
        raise ValueError('READOUT_BINDING_DUPLICATE_BACKBONE_ID')
    policy = {r['canonical_id']: r for r in policies}
    if set(primary) != {r['canonical_id'] for r in policies if r['analysis_role'] == 'primary_pathway'}:
        raise ValueError('READOUT_BINDING_PRIMARY_POLICY_MISMATCH')
    supervised = set(primary) | {r['canonical_id'] for r in policies if r['analysis_role'] == 'state_auxiliary'}
    nodes = defaultdict(set)
    for r in definitions:
        if r.get('node_id'):
            nodes[r['node_id']].add(r['integrated_canonical_id'])
        if r.get('canonical_id'):
            nodes[r['canonical_id']].add(r['integrated_canonical_id'])
    if any(len(v) != 1 for v in nodes.values()):
        raise ValueError('READOUT_BINDING_AMBIGUOUS_DEFINITION')
    redirect = {k: next(iter(v)) for k, v in nodes.items()}
    aliases = defaultdict(set)
    for r in source_bindings:
        if r.get('source_target_id'):
            aliases[r['source_target_id']].add(redirect.get(r['canonical_id'], r['canonical_id']))
    exact_backbone = {r['source_target_id']: r['canonical_id'] for r in backbone}
    if len(exact_backbone) != len(backbone):
        raise ValueError('READOUT_BINDING_DUPLICATE_KEGG_SOURCE')
    components = defaultdict(set)
    for r in kinase_components:
        components[redirect.get(r['canonical_id'], r['canonical_id'])].update(_tokens(r['uniprot_accessions']))
    molecular = defaultdict(list)
    context = defaultdict(set)
    for r in alignments:
        if not _truth(r['admitted_context_link']):
            continue
        src, dst = r['source_canonical_id'], r['target_canonical_id']
        if dst not in primary:
            raise ValueError('READOUT_BINDING_ALIGNMENT_OUTSIDE_BACKBONE')
        context[src].add(dst)
        if r['relation'] != 'regulator_component_in_mechanism':
            continue
        if r['alignment_evidence_basis'] != 'exact_regulator_molecular_identity':
            raise ValueError('READOUT_BINDING_UNVERIFIED_REGULATOR')
        shared = _tokens(r.get('direct_regulator_proteins'))
        if not shared:
            raise ValueError('READOUT_BINDING_EMPTY_REGULATOR_COMPONENT')
        # A group/complex cannot be identified by just one of its components.
        known = components.get(src)
        if known and not known.issubset(shared):
            continue
        if policy[src]['biological_type'] not in {'tf_program', 'kinase_program', 'transcription_factor_program', 'kinase', 'tf'}:
            # Catalogs may retain an explicit mechanism_kind alongside an older
            # biological_type. The molecular identity, not the display role,
            # is the admission requirement.
            if policy[src].get('mechanism_kind') not in {'TF', 'kinase'}:
                raise ValueError('READOUT_BINDING_NOT_A_REGULATOR')
        molecular[src].append((dst, shared))
    output, dispositions, seen = [], [], set()
    for r in readouts:
        slot = (r['cohort'], r['layer'], int(r['column_index']))
        if slot in seen:
            raise ValueError('READOUT_BINDING_DUPLICATE_SOURCE_COLUMN')
        seen.add(slot)
        original = r['canonical_id']
        resolved = redirect.get(original, original)
        sid = r.get('source_target_id', '')
        candidates = aliases.get(sid, set())
        if len(candidates) > 1:
            raise ValueError('READOUT_BINDING_AMBIGUOUS_SOURCE_ALIAS: ' + sid)
        if candidates and resolved not in candidates:
            raise ValueError('READOUT_BINDING_SOURCE_IDENTITY_CONFLICT: ' + sid)
        if sid in exact_backbone:
            exact = exact_backbone[sid]
            if resolved != exact:
                raise ValueError('READOUT_BINDING_KEGG_IDENTITY_CONFLICT: ' + sid)
        routes = []
        if resolved in supervised:
            role = 'equivalent_pathway_readout' if resolved in primary else 'same_state_readout'
            routes.append((resolved, role, 'same_identity', set(), 'source_definition_identity'))
        if resolved not in primary:
            for dst, member in molecular.get(resolved, []):
                routes.append((dst, 'molecular_component_readout', 'regulator_component_in_mechanism',
                               member, 'exact_regulator_molecular_identity'))
        unique = {}
        for dst, role, relation, member, basis in routes:
            key = (dst, role)
            if key in unique:
                unique[key][3].update(member)
            else:
                unique[key] = [role, relation, basis, set(member)]
        for (dst, _), (role, relation, basis, member) in sorted(unique.items()):
            core = primary.get(dst)
            output.append({
                'cohort': r['cohort'], 'layer': r['layer'], 'column_index': int(r['column_index']),
                'source_target_id': sid, 'target_id': r['target_id'], 'source_canonical_id': original,
                'resolved_source_canonical_id': resolved, 'output_canonical_id': dst,
                'network_id': core['network_id'] if core else '',
                'measurement_layer': r['measurement_layer'], 'evidence_role': role,
                'scope_relation': relation, 'molecular_components': '|'.join(sorted(member)),
                'condition_required': bool(core and core['network_type'] != 'Reference'),
                'routing_basis': basis,
            })
        dispositions.append({'cohort': r['cohort'], 'layer': r['layer'],
            'column_index': int(r['column_index']), 'target_id': r['target_id'],
            'resolved_source_canonical_id': resolved, 'bound_target_count': len(unique),
            'context_target_count': len(context.get(resolved, set())),
            'disposition': 'bound_evidence' if unique else 'structural_context' if context.get(resolved) else 'outside_current_supervision'})
    return output, dispositions


def read_table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_table(path, rows, fields=FIELDS):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter='\t')
        w.writeheader(); w.writerows(rows)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def materialize_bindings(catalog, backbone_dir, stream_specs, output):
    from .local_execution import require_host_149
    require_host_149()
    catalog, backbone_dir, output = Path(catalog), Path(backbone_dir), Path(output)
    if output.exists():
        raise ValueError('READOUT_BINDING_OUTPUT_EXISTS')
    output.mkdir(parents=True)
    (output/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','training':False,'operation':'bind measured readouts to exact pathway and member identities'},indent=2))
    names = ['DEFINITION_TO_IDENTITY.tsv','SOURCE_ID_BINDINGS.tsv','TARGET_POLICY.tsv',
             'BACKBONE_ALIGNMENT.tsv','KINASE_MOLECULAR_COMPONENTS.tsv']
    refs = {name: {'path': str(catalog/name), 'sha256': file_sha(catalog/name)} for name in names}
    refs['backbone'] = {'path':str(backbone_dir/'KEGG_CANCER_BACKBONE.tsv'),'sha256':file_sha(backbone_dir/'KEGG_CANCER_BACKBONE.tsv')}
    reads, streams = [], []
    for spec in stream_specs:
        directory = Path(spec['directory'])
        receipt = json.loads((directory/'COMPLETE.json').read_text())
        if receipt['target_host'] != '149' or file_sha(directory/'TARGETS.tsv') != receipt['files']['TARGETS.tsv']:
            raise ValueError('READOUT_BINDING_SOURCE_METADATA_DRIFT')
        rows=read_table(directory/'TARGETS.tsv')
        for j,r in enumerate(rows):
            reads.append({**r,'cohort':spec['cohort'],'layer':spec['layer'],'column_index':j})
        streams.append({**spec,'source_receipt_sha256':file_sha(directory/'COMPLETE.json'),
                        'targets_sha256':receipt['files']['TARGETS.tsv'],'columns':len(rows)})
    bindings, disposition = compile_bindings(backbone=read_table(backbone_dir/'KEGG_CANCER_BACKBONE.tsv'),
        definitions=read_table(catalog/'DEFINITION_TO_IDENTITY.tsv'),source_bindings=read_table(catalog/'SOURCE_ID_BINDINGS.tsv'),
        policies=read_table(catalog/'TARGET_POLICY.tsv'),alignments=read_table(catalog/'BACKBONE_ALIGNMENT.tsv'),
        kinase_components=read_table(catalog/'KINASE_MOLECULAR_COMPONENTS.tsv'),readouts=reads)
    write_table(output/'READOUT_BINDINGS.tsv',bindings)
    write_table(output/'SOURCE_DISPOSITIONS.tsv',disposition,tuple(disposition[0]))
    (output/'INPUT_REFERENCES.json').write_text(json.dumps({'structural':refs,'streams':streams},indent=2))
    result={'format':FORMAT,'target_host':'149','readout_columns':len(reads),'binding_rows':len(bindings),
            'identity_uses_patient_data':False,'display_category_adds_loss':False,
            'files':{name:file_sha(output/name) for name in ['READOUT_BINDINGS.tsv','SOURCE_DISPOSITIONS.tsv','INPUT_REFERENCES.json']}}
    (output/'COMPLETE.json').write_text(json.dumps(result,indent=2))
    return result
