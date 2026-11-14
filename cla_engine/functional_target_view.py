"""Immutable target views with explicit source-label to output-identity mapping."""
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
import csv
import hashlib
import json
import shutil
import socket
import numpy as np
from .target_policy import resolve_target_policy, policy_for_supervised_axis, active_target_ids
from .functional_fusion import FusedFunctionalContext
from .kegg381_backbone import load_backbone

FORMAT = 'CLA_FUNCTIONAL_TARGET_VIEW_V1'
MAPPING_SCHEMA = 'CLA_SOURCE_LABEL_IDENTITY_MAP_V1'
MAPPING_FIELDS = ('output_canonical_id', 'source_label_canonical_id',
                  'source_target_id', 'network_id', 'mapping_basis',
                  'evidence_contexts', 'additional_source_evidence_fused')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def _write_tsv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, delimiter='\t', fieldnames=MAPPING_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def compile_label_identity_map(backbone_rows, context_readouts, definition_rows):
    """Resolve original KEGG columns by exact source identity, never label values.

    Cross-context evidence may fill an unmeasured readout in one context, but any
    conflicting source identities fail. Definition N nodes are a fallback only.
    """
    by_source = defaultdict(set)
    source_contexts = defaultdict(set)
    for context, rows in context_readouts.items():
        for row in rows:
            sid, cid = row.get('source_target_id', ''), row.get('canonical_id', '')
            if sid and cid:
                by_source[sid].add(cid)
                source_contexts[sid].add(context)
    definition_nodes = defaultdict(set)
    source_definition_nodes = defaultdict(set)
    for row in definition_rows:
        authority = row.get('authority_id', '')
        node = row.get('node_id') or row.get('canonical_id')
        if authority.startswith('MSIGDB:KEGG_MEDICUS:') and node:
            source_definition_nodes[authority].add(node)
        if authority.startswith('KEGG.NETWORK:') and node:
            definition_nodes[authority].add(node)
    result = []
    seen_outputs, seen_sources = set(), set()
    for row in backbone_rows:
        sid = row['source_target_id']
        output = row['canonical_id']
        candidates = by_source.get(sid, set())
        if candidates:
            basis = 'exact_backbone_source_target_id_in_readout_index'
            contexts = '|'.join(sorted(source_contexts[sid]))
        elif source_definition_nodes.get(sid):
            candidates = source_definition_nodes[sid]
            basis = 'original_exact_source_node_from_definition_to_identity'
            contexts = ''
        else:
            candidates = definition_nodes.get('KEGG.NETWORK:' + row['network_id'], set())
            basis = 'original_network_node_from_definition_to_identity'
            contexts = ''
        if len(candidates) != 1:
            raise ValueError('TARGET_VIEW_BACKBONE_LABEL_IDENTITY_UNRESOLVED: ' + sid)
        source = next(iter(candidates))
        if output in seen_outputs or source in seen_sources:
            raise ValueError('TARGET_VIEW_LABEL_MAPPING_NOT_ONE_TO_ONE')
        seen_outputs.add(output)
        seen_sources.add(source)
        result.append({
            'output_canonical_id': output, 'source_label_canonical_id': source,
            'source_target_id': sid, 'network_id': row['network_id'],
            'mapping_basis': basis, 'evidence_contexts': contexts,
            'additional_source_evidence_fused': 'False',
        })
    return result


def project_context(context, policies, target_ids, source_label_ids=None, *, unavailable_target_ids=()):
    """Project exact frozen columns and optionally rename their output identity."""
    target_ids = tuple(target_ids)
    source_ids = tuple(target_ids if source_label_ids is None else source_label_ids)
    selected = policy_for_supervised_axis(target_ids, policies)
    if len(source_ids) != len(target_ids) or len(set(source_ids)) != len(source_ids):
        raise ValueError('TARGET_VIEW_LABEL_MAPPING_NOT_ONE_TO_ONE')
    if len(set(context.canonical_ids)) != len(context.canonical_ids):
        raise ValueError('TARGET_VIEW_DUPLICATE_SOURCE_AXIS')
    index = {cid: i for i, cid in enumerate(context.canonical_ids)}
    missing = set(source_ids) - set(index)
    if missing:
        raise ValueError('TARGET_VIEW_MISSING_LABEL_AXIS: ' + ','.join(sorted(missing)))
    take = np.asarray([index[cid] for cid in source_ids], dtype=np.int64)
    arrays = {key: value[:, take].copy() for key, value in context.arrays.items()}
    unavailable = set(unavailable_target_ids)
    if unavailable - set(target_ids):
        raise ValueError('TARGET_VIEW_APPLICABILITY_AXIS_MISMATCH')
    mask = np.asarray([cid in unavailable for cid in target_ids], bool)
    for values in arrays.values():
        values[:, mask] = np.nan if np.issubdtype(values.dtype, np.floating) else 0

    if not isinstance(context, FusedFunctionalContext):
        return replace(context, canonical_ids=target_ids, arrays=arrays, target_policies=selected)
    return replace(context, canonical_ids=target_ids, arrays=arrays, target_policies=selected,
                   target_roles=tuple(p.roles for p in selected),
                   primary_tasks=tuple(p.primary_task for p in selected))


def build_target_view(source, catalog, output, cancers=('BRCA', 'LUAD', 'COAD')):
    if socket.gethostname() != '149':
        raise RuntimeError('CancerLncAtlas preparation must run on 149')
    source, catalog, out = Path(source).resolve(), Path(catalog).resolve(), Path(output).resolve()
    if out.exists():
        raise ValueError('TARGET_VIEW_OUTPUT_EXISTS')
    original = json.loads((source / 'Y_DEFINITION.json').read_text())
    if original['format'] != 'CLA_FUNCTIONAL_RELATION_FUSED_V1':
        raise ValueError('TARGET_VIEW_EXPECTS_FUSED_SOURCE')
    catdef = json.loads((catalog / 'CATALOG_DEFINITION.json').read_text())
    policies = resolve_target_policy(catalog, catdef)
    wanted = active_target_ids(policies)
    primary = set(active_target_ids(policies, analysis_role='primary_pathway'))
    if len(primary) != catdef['primary_pathway_count'] or not primary:
        raise ValueError('TARGET_VIEW_DECLARED_PRIMARY_COUNT_MISMATCH')
    backbone_path = Path(catdef['backbone_path']).resolve()
    if sha(backbone_path / 'MANIFEST.json') != catdef['backbone_manifest_sha256']:
        raise ValueError('TARGET_VIEW_BACKBONE_MANIFEST_MISMATCH')
    backbone = load_backbone(backbone_path, verify=True)
    if {row['canonical_id'] for row in backbone['rows']} != primary:
        raise ValueError('TARGET_VIEW_BACKBONE_PRIMARY_AXIS_MISMATCH')
    contexts, axes, readouts = {}, {}, {}
    for cancer in cancers:
        dirs = sorted((source / cancer).glob('*/COMPLETE.json'))
        if not dirs:
            raise ValueError('TARGET_VIEW_CANCER_CONTEXTS_MISSING: ' + cancer)
        for receipt_path in dirs:
            base = receipt_path.parent
            receipt = json.loads(receipt_path.read_text())
            key = cancer + '/' + base.name
            axis_path, readout_path = base / 'AXES.json', base / 'READOUT_INDEX.tsv'
            if sha(axis_path) != receipt['files']['AXES.json']:
                raise ValueError('TARGET_VIEW_SOURCE_AXIS_CHECKSUM')
            if sha(readout_path) != receipt['files']['READOUT_INDEX.tsv']:
                raise ValueError('TARGET_VIEW_SOURCE_READOUT_INDEX_CHECKSUM')
            axis = json.loads(axis_path.read_text())['canonical_ids']
            if len(set(axis)) != len(axis):
                raise ValueError('TARGET_VIEW_DUPLICATE_SOURCE_AXIS')
            axes[key] = set(axis)
            readouts[key] = _read_tsv(readout_path)
            contexts[key] = {
                'source_complete_sha256': sha(receipt_path),
                'source_axes_sha256': sha(axis_path),
                'source_readout_index_sha256': sha(readout_path),
            }
    definitions_path = catalog / 'DEFINITION_TO_IDENTITY.tsv'
    mapping = compile_label_identity_map(backbone['rows'], readouts, _read_tsv(definitions_path))
    primary_map = {row['output_canonical_id']: row for row in mapping}
    conditional = {row['canonical_id'] for row in backbone['rows']
                   if backbone['manifest'].get('format') == 'CLA_KEGG_CANCER_BACKBONE_V2'
                   and row['network_type'] != 'Reference'}

    missing_aux = set()
    for key, binding in contexts.items():
        missing_primary = {cid for cid in primary
                           if primary_map[cid]['source_label_canonical_id'] not in axes[key]}
        if missing_primary:
            raise ValueError('TARGET_VIEW_SOURCE_PRIMARY_LABEL_AXIS_MISSING: ' + key)
        selected, selected_sources = [], []
        for cid in wanted:
            source_id = primary_map[cid]['source_label_canonical_id'] if cid in primary else cid
            if source_id not in axes[key]:
                missing_aux.add(cid)
                continue
            selected.append(cid)
            selected_sources.append(source_id)
        if len(set(selected_sources)) != len(selected_sources):
            raise ValueError('TARGET_VIEW_LABEL_MAPPING_NOT_ONE_TO_ONE: ' + key)
        binding['selected_canonical_ids'] = selected
        binding['source_label_canonical_ids'] = selected_sources
        binding['unavailable_conditional_target_ids'] = sorted(conditional & set(selected))
    selected_aux = set().union(*(set(b['selected_canonical_ids']) for b in contexts.values())) - primary
    for cid in sorted(selected_aux):
        mapping.append({
            'output_canonical_id': cid, 'source_label_canonical_id': cid,
            'source_target_id': '', 'network_id': '',
            'mapping_basis': 'unchanged_auxiliary_label_identity',
            'evidence_contexts': '', 'additional_source_evidence_fused': 'False',
        })
    renamed = [r for r in mapping if r['output_canonical_id'] != r['source_label_canonical_id']]
    out.mkdir(parents=True)
    shutil.copy2(catalog / 'TARGET_POLICY.tsv', out / 'TARGET_POLICY.tsv')
    _write_tsv(out / 'LABEL_IDENTITY_MAP.tsv', mapping)
    applicability = {'format': 'CLA_LABEL_APPLICABILITY_V1',
        'conditional_target_ids': sorted(conditional),
        'rule': 'Generic host-component readouts do not identify the named alteration, pathogen or exposure.',
        'conditional_label_qualification': 'No context-qualified label receipt in the frozen source.',
        'missing_representation': 'NaN continuous values; zero observation and evidence codes; never a negative.'}
    (out / 'LABEL_APPLICABILITY.json').write_text(json.dumps(applicability, indent=2) + '\n')

    (out / 'SOURCE_CONTEXTS.json').write_text(json.dumps(contexts, indent=2) + '\n')
    validation = {
        'target_host': '149', 'label_mapping_schema': MAPPING_SCHEMA,
        'primary_pathway_count': len(primary),
        'renamed_primary_label_count': len(renamed),
        'renamed_primary_labels': renamed,
        'source_label_identity_is_one_to_one': True,
        'original_label_columns_retained': True,
        'label_values_changed': bool(conditional),
        'conditional_targets_masked': len(conditional),
        'unconditional_label_values_changed': False,
        'additional_source_evidence_fused': False,
        'scope_note': 'Original KEGG label columns are retained. Output identity redirects do not fuse additional Reactome evidence.',
        'mapping_input_sha256': {
            'catalog_definition': sha(catalog / 'CATALOG_DEFINITION.json'),
            'backbone_manifest': sha(backbone_path / 'MANIFEST.json'),
            'definition_to_identity': sha(definitions_path),
        },
    }
    (out / 'SCOPE_VALIDATION.json').write_text(json.dumps(validation, indent=2) + '\n')
    definition = {
        **original, 'format': FORMAT, 'source_format': original['format'],
        'source_y': str(source), 'source_y_definition_sha256': sha(source / 'Y_DEFINITION.json'),
        'target_policy_required': True, 'target_policy_schema': catdef['target_policy_schema'],
        'target_policy_file': 'TARGET_POLICY.tsv', 'target_policy_sha256': sha(out / 'TARGET_POLICY.tsv'),
        'source_contexts_sha256': sha(out / 'SOURCE_CONTEXTS.json'), 'cancers': list(cancers),
        'label_mapping_schema': MAPPING_SCHEMA, 'label_identity_map_file': 'LABEL_IDENTITY_MAP.tsv',
        'label_identity_map_sha256': sha(out / 'LABEL_IDENTITY_MAP.tsv'),
        'scope_validation_sha256': sha(out / 'SCOPE_VALIDATION.json'),
        'label_applicability_sha256': sha(out / 'LABEL_APPLICABILITY.json'),
        'primary_pathway_count': len(primary), 'missing_auxiliary_definition_ids': sorted(missing_aux),
        'multiplicity': original['multiplicity'],
        'multiplicity_reference': 'Original frozen full target family; column selection and output renaming do not recalculate p/q.',
        'label_values_changed': bool(conditional),
        'conditional_targets_masked': len(conditional),
        'unconditional_label_values_changed': False, 'display_category_determines_loss': False,
        'additional_source_evidence_fused': False,
    }
    (out / 'Y_DEFINITION.json').write_text(json.dumps(definition, indent=2) + '\n')
    receipt = {
        'target_host': '149', 'format': FORMAT, 'source_y': str(source),
        'primary_pathways': len(primary), 'contexts': len(contexts),
        'supervised_axis_sizes': sorted(set(len(x['selected_canonical_ids']) for x in contexts.values())),
        'unobserved_auxiliary_definition_count': len(missing_aux),
        'renamed_primary_label_count': len(renamed),
        'patient_labels_recomputed': False, 'sealed_test_labels_opened': False,
        'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()},
    }
    (out / 'COMPLETE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def load_target_view(root, cancer, context, *, target_policy=None):
    root = Path(root).resolve()
    receipt = json.loads((root / 'COMPLETE.json').read_text())
    required = ('Y_DEFINITION.json', 'SOURCE_CONTEXTS.json', 'TARGET_POLICY.tsv',
                'LABEL_IDENTITY_MAP.tsv', 'SCOPE_VALIDATION.json')
    if any(name not in receipt.get('files', {}) for name in required):
        raise ValueError('TARGET_VIEW_EXPLICIT_LABEL_MAPPING_REQUIRED')
    for name in required:
        if sha(root / name) != receipt['files'][name]:
            raise ValueError('TARGET_VIEW_CHECKSUM_MISMATCH: ' + name)
    definition = json.loads((root / 'Y_DEFINITION.json').read_text())
    if definition.get('label_mapping_schema') != MAPPING_SCHEMA:
        raise ValueError('TARGET_VIEW_LABEL_MAPPING_SCHEMA_MISMATCH')
    if sha(root / 'LABEL_IDENTITY_MAP.tsv') != definition['label_identity_map_sha256']:
        raise ValueError('TARGET_VIEW_LABEL_MAPPING_CHECKSUM')
    applicability = None
    if definition.get('label_applicability_sha256'):
        if sha(root / 'LABEL_APPLICABILITY.json') != definition['label_applicability_sha256']:
            raise ValueError('TARGET_VIEW_APPLICABILITY_CHECKSUM')
        applicability = json.loads((root / 'LABEL_APPLICABILITY.json').read_text())
    policies = resolve_target_policy(root, definition, target_policy)
    source = Path(definition['source_y']).resolve()
    if source == root or sha(source / 'Y_DEFINITION.json') != definition['source_y_definition_sha256']:
        raise ValueError('TARGET_VIEW_SOURCE_DEFINITION_MISMATCH')
    contexts = json.loads((root / 'SOURCE_CONTEXTS.json').read_text())
    binding = contexts.get(cancer + '/' + context)
    if binding is None:
        raise ValueError('TARGET_VIEW_CONTEXT_NOT_SELECTED')
    source_complete = (source / cancer / context / 'COMPLETE.json').resolve()
    if not source_complete.is_relative_to(source) or sha(source_complete) != binding['source_complete_sha256']:
        raise ValueError('TARGET_VIEW_SOURCE_CONTEXT_MISMATCH')
    source_ids = binding.get('source_label_canonical_ids')
    if source_ids is None:
        raise ValueError('TARGET_VIEW_EXPLICIT_LABEL_MAPPING_REQUIRED')
    mapping = {}
    for row in _read_tsv(root / 'LABEL_IDENTITY_MAP.tsv'):
        cid = row['output_canonical_id']
        if cid in mapping:
            raise ValueError('TARGET_VIEW_LABEL_MAPPING_NOT_ONE_TO_ONE')
        mapping[cid] = row['source_label_canonical_id']
    if [mapping.get(cid) for cid in binding['selected_canonical_ids']] != source_ids:
        raise ValueError('TARGET_VIEW_CONTEXT_LABEL_MAPPING_MISMATCH')
    from .functional_y import load_functional_context
    original = load_functional_context(source, cancer, context)
    conditional = binding.get('unavailable_conditional_target_ids', [])
    if applicability is not None:
        expected = set(applicability['conditional_target_ids']) & set(binding['selected_canonical_ids'])
        if set(conditional) != expected:
            raise ValueError('TARGET_VIEW_APPLICABILITY_BINDING')
    elif conditional:
        raise ValueError('TARGET_VIEW_APPLICABILITY_REQUIRED')
    return project_context(original, policies, binding['selected_canonical_ids'], source_ids,
                           unavailable_target_ids=conditional)
