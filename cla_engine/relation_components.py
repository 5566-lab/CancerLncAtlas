"""Validate reusable source feature components before relation preparation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .common import atomic_json, atomic_npz, digest, sha
from .relation_features import COHORTS, LAYERS, ENTITY_FIELDS
from .relation_source import SOURCE_NUMERIC_FIELDS, source_numeric_block

FORMAT = 'CLA_RELATION_SOURCE_FEATURE_COMPONENTS_V1'
CONTEXTS = ('train_label_fold2', 'train_label_fold3', 'train_label_fold4', 'validation')


def materialize_relation_components(bindings, cancer, context_name, out, *, budget=50000,
                                    seed=20261111, progress=None):
    """Export source-derived inputs and separately attach the frozen labels."""
    import dataclasses
    import time
    from .functional_y import load_functional_context
    from .local_execution import require_host_149
    from .prepare import deterministic_pairs
    from .relation_features import build_source_features, verified_file
    from .relation_strength import exact_relation_strength

    require_host_149()
    if context_name not in CONTEXTS or budget < 1:
        raise ValueError('RELATION_COMPONENT_REQUEST_INVALID')
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError('RELATION_COMPONENT_OUTPUT_NOT_EMPTY')
    definition = verified_file(bindings['target_view_definition'])
    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'LAUNCH_PREFLIGHT.json', {'target_host': '149', 'cancer': cancer,
        'context': context_name, 'candidate_budget': budget, 'seed': seed,
        'bindings': bindings, 'training': False, 'sealed_test_opened': False})
    context = load_functional_context(definition.parent, cancer, context_name)
    li, ti = deterministic_pairs(context.lncrna_ids, context.canonical_ids, budget, seed, cancer)
    atomic_json(out / 'AXES.json', {'lncrna_ids': context.lncrna_ids,
        'canonical_ids': context.canonical_ids, 'loss_groups': context.loss_groups,
        'candidate_selection': 'DETERMINISTIC_LABEL_INDEPENDENT', 'seed': seed})
    result = build_source_features(context, bindings, li, ti, progress=progress)
    labels = exact_relation_strength(context, [context.lncrna_ids[i] for i in li],
        [context.canonical_ids[i] for i in ti], source_patients_by_cohort=result['source_patients'],
        expected_context=context_name)
    atomic_npz(out / 'CANDIDATES.npz', lnc=li, target=ti, numeric=result['numeric'],
               label=labels['label'], available=labels['available'])
    atomic_json(out / 'FEATURE_NAMES.json', result['feature_names'])
    atomic_json(out / 'AUDIT.json', result['audits'])
    atomic_json(out / 'READOUT_AXES.json', result['readout_axes'])
    atomic_json(out / 'SPLIT.json', context.cohort_splits)
    for key, association in result['associations'].items():
        arrays = {f.name: getattr(association, f.name) for f in dataclasses.fields(association)
                  if isinstance(getattr(association, f.name), np.ndarray)}
        atomic_npz(out / (key.replace('/', '_') + '_SOURCE_ASSOCIATIONS.npz'), **arrays)
    atomic_npz(out / 'LNC_NODE_FEATURES.npz', **result['lncrna_node_features'])
    groups = np.asarray(context.loss_groups)[ti]
    receipt = {'target_host': '149', 'format': FORMAT, 'cancer': cancer, 'context': context_name,
        'candidates': len(li), 'numeric_features': result['numeric'].shape[1],
        'lncrna_axis': len(context.lncrna_ids), 'target_axis': len(context.canonical_ids),
        'targets_in_candidates': len(np.unique(ti)), 'lncrnas_in_candidates': len(np.unique(li)),
        'observed_labels': int(labels['available'].sum()),
        'observed_labels_by_task': {g: int((labels['available'] & (groups == g)).sum())
                                    for g in ('pathway', 'state')},
        'candidate_hash': digest(list(zip(li.tolist(), ti.tolist()))),
        'feature_computation_reads_label_values': False, 'training_started': False,
        'full_prepared_capsule': False, 'seconds': time.monotonic() - start,
        'code_files': {name: sha(Path(__file__).parent / name) for name in
                       ('relation_components.py', 'relation_features.py', 'relation_source.py', 'relation_strength.py')},
        'files': {p.name: sha(p) for p in out.iterdir() if p.is_file()}}
    atomic_json(out / 'COMPLETE.json', receipt)
    load_relation_components(out)
    return receipt


def _json(root, name):
    return json.loads((root / name).read_text())


def load_relation_components(path):
    """Check frozen splits, candidate selection and every numeric source block."""
    from .prepare import deterministic_pairs

    root = Path(path)
    receipt = _json(root, 'COMPLETE.json')
    if (receipt.get('format') != FORMAT or receipt.get('target_host') != '149'
            or receipt.get('context') not in CONTEXTS
            or receipt.get('full_prepared_capsule') is not False):
        raise ValueError('RELATION_COMPONENT_FORMAT')
    required = {'LAUNCH_PREFLIGHT.json', 'AXES.json', 'CANDIDATES.npz',
                'FEATURE_NAMES.json', 'AUDIT.json', 'READOUT_AXES.json',
                'SPLIT.json', 'LNC_NODE_FEATURES.npz'}
    required.update(f'{cohort}_{layer}_SOURCE_ASSOCIATIONS.npz'
                    for cohort in COHORTS for layer in LAYERS)
    if set(receipt.get('files', {})) != required:
        raise ValueError('RELATION_COMPONENT_FILE_SET')
    for name, expected in receipt['files'].items():
        if Path(name).name != name or sha(root / name) != expected:
            raise ValueError('RELATION_COMPONENT_FILE_DRIFT: ' + name)
    preflight = _json(root, 'LAUNCH_PREFLIGHT.json')
    if any(preflight.get(k) != receipt[k] for k in ('target_host', 'cancer', 'context')):
        raise ValueError('RELATION_COMPONENT_IDENTITY')
    if preflight.get('sealed_test_opened') is not False:
        raise ValueError('RELATION_COMPONENT_SEALED_TEST')
    axes, splits = _json(root, 'AXES.json'), _json(root, 'SPLIT.json')
    lncs, targets, groups = (axes[k] for k in ('lncrna_ids', 'canonical_ids', 'loss_groups'))
    if (len(lncs) != len(set(lncs)) or not lncs or not targets
            or len(targets) != len(set(targets)) or len(groups) != len(targets)
            or set(groups) != {'pathway', 'state'}):
        raise ValueError('RELATION_COMPONENT_AXES')
    if set(splits) != set(COHORTS):
        raise ValueError('RELATION_COMPONENT_COHORTS')
    label_fold = 1 if receipt['context'] == 'validation' else int(receipt['context'][-1])
    source_folds = sorted({2, 3, 4} - {label_fold})
    for cohort, split in splits.items():
        source, label = split['source_patients'], split['label_patients']
        if (not source or not label or len(source) != len(set(source))
                or len(label) != len(set(label)) or set(source) & set(label)
                or split['outer_fold'] != 0 or split['source_folds'] != source_folds
                or split['label_folds'] != [label_fold]):
            raise ValueError('RELATION_COMPONENT_PATIENT_SPLIT: ' + cohort)
    with np.load(root / 'CANDIDATES.npz', allow_pickle=False) as f:
        arrays = {key: f[key] for key in f.files}
    if set(arrays) != {'lnc', 'target', 'numeric', 'label', 'available'}:
        raise ValueError('RELATION_COMPONENT_CANDIDATE_FIELDS')
    li, ti = deterministic_pairs(lncs, targets, preflight['candidate_budget'], preflight['seed'], receipt['cancer'])
    if (axes.get('candidate_selection') != 'DETERMINISTIC_LABEL_INDEPENDENT'
            or axes.get('seed') != preflight['seed']
            or not np.array_equal(arrays['lnc'], li) or not np.array_equal(arrays['target'], ti)
            or receipt['candidate_hash'] != digest(list(zip(li.tolist(), ti.tolist())))):
        raise ValueError('RELATION_COMPONENT_CANDIDATE_SELECTION')
    names = [name for cohort in COHORTS for name in (
        [f'{cohort}/{layer}/{field}' for layer in LAYERS for field in SOURCE_NUMERIC_FIELDS]
        + [f'{cohort}/{field}' for field in ENTITY_FIELDS])]
    if (_json(root, 'FEATURE_NAMES.json') != names or arrays['numeric'].shape != (len(li), len(names))
            or not np.isfinite(arrays['numeric']).all()):
        raise ValueError('RELATION_COMPONENT_NUMERIC_SCHEMA')
    y, mask = arrays['label'], arrays['available']
    if (mask.dtype != bool or mask.shape != (len(li),) or y.shape != mask.shape
            or not np.isfinite(y).all() or np.any(y < 0) or np.any(y > 1 + 1e-6)
            or np.any(y[~mask] != 0)):
        raise ValueError('RELATION_COMPONENT_LABEL_MASK')
    readout_axes, audits = _json(root, 'READOUT_AXES.json'), _json(root, 'AUDIT.json')
    if set(readout_axes) != {c + '/' + layer for c in COHORTS for layer in LAYERS}:
        raise ValueError('RELATION_COMPONENT_READOUT_BLOCKS')
    if [(a['cohort'], a['layer']) for a in audits] != [(c, layer) for c in COHORTS for layer in LAYERS]:
        raise ValueError('RELATION_COMPONENT_AUDIT_BLOCKS')
    with np.load(root / 'LNC_NODE_FEATURES.npz', allow_pickle=False) as f:
        node = {key: f[key] for key in f.files}
    if set(node) != set(COHORTS):
        raise ValueError('RELATION_COMPONENT_LNCRNA_NODE_COHORTS')
    rebuilt, associations = [], {}
    for cohort in COHORTS:
        patients = tuple(splits[cohort]['source_patients'])
        for layer in LAYERS:
            key = cohort + '/' + layer
            axis = readout_axes[key]
            count = len(axis['readout_ids'])
            if (len(set(axis['readout_ids'])) != count or len(axis['canonical_ids']) != count
                    or len(axis['metadata']) != count
                    or [r['target_id'] for r in axis['metadata']] != axis['readout_ids']):
                raise ValueError('RELATION_COMPONENT_READOUT_AXIS')
            with np.load(root / f'{cohort}_{layer}_SOURCE_ASSOCIATIONS.npz', allow_pickle=False) as f:
                a = {field: f[field] for field in f.files}
            fields = {'lncrna_index', 'readout_index', 'rho', 'p', 'n', 'rank', 'available'}
            n = len(a['lncrna_index'])
            if set(a) != fields or any(v.shape != (n,) for v in a.values()):
                raise ValueError('RELATION_COMPONENT_ASSOCIATION_SCHEMA')
            if (a['available'].dtype != bool or np.any(a['lncrna_index'] < 0)
                    or np.any(a['lncrna_index'] >= len(lncs)) or np.any(a['readout_index'] < 0)
                    or np.any(a['readout_index'] >= count)
                    or np.any(~np.isfinite(a['rho'][a['available']]))
                    or np.any(np.abs(a['rho'][a['available']]) > 1 + 1e-6)):
                raise ValueError('RELATION_COMPONENT_ASSOCIATION_VALUES')
            associations[key] = a
            association = SimpleNamespace(**a, cohort=cohort, layer=layer, patients=patients)
            readouts = SimpleNamespace(**axis, cohort=cohort, layer=layer, patients=patients)
            rebuilt.append(source_numeric_block(association, readouts, li, ti, targets))
        summary = node[cohort]
        if (summary.shape != (len(lncs), len(ENTITY_FIELDS)) or not np.isfinite(summary).all()
                or np.any(summary[:, [1, 4]] < 0) or np.any(summary[:, [2, 5]] < 0)
                or np.any(summary[:, [2, 5]] > 1)):
            raise ValueError('RELATION_COMPONENT_LNCRNA_NODE_VALUES')
        rebuilt.append(summary[li])
    if not np.allclose(arrays['numeric'], np.column_stack(rebuilt), rtol=1e-6, atol=1e-7):
        raise ValueError('RELATION_COMPONENT_NUMERIC_SOURCE_MISMATCH')
    observed_by_task = {g: int((mask & (np.asarray(groups)[ti] == g)).sum()) for g in ('pathway', 'state')}
    expected_counts = {'candidates': len(li), 'numeric_features': len(names), 'lncrna_axis': len(lncs),
                       'target_axis': len(targets), 'observed_labels': int(mask.sum()),
                       'observed_labels_by_task': observed_by_task,
                       'targets_in_candidates': len(np.unique(ti)), 'lncrnas_in_candidates': len(np.unique(li))}
    if any(receipt.get(k) != v for k, v in expected_counts.items()):
        raise ValueError('RELATION_COMPONENT_COUNTS')
    return {'receipt': receipt, 'axes': axes, 'splits': splits, 'arrays': arrays,
            'feature_names': names, 'associations': associations, 'readout_axes': readout_axes,
            'lncrna_node_features': node, 'preflight': preflight}
