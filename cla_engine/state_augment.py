"""Append verified existing State measurements without recomputing prior scores."""
from __future__ import annotations
import argparse, socket
from pathlib import Path
import numpy as np
import pandas as pd
from .common import atomic_json, read_json, sha, digest, fail
from .state_catalog import ROUTED_CATALOG_ID, AUGMENTED_CATALOG_ID, expanded_targets, catalog
from .state_materialize import FORMAT, _entry, numeric, select_patient_values, verify_materialized_source
from .cancer_scope import resolve_cancer_scope


def append_columns(base, selected, targets):
    """Preserve every prior value/NA and align additions by explicit patient ID."""
    if base.patient_id.duplicated().any() or not selected.index.is_unique:
        fail('STATE_AUGMENT_PATIENT_DUPLICATE', 'Expected unique patient keys')
    added = [t['column'] for t in targets if t['column'] not in base]
    if set(added) != set(selected):
        fail('STATE_AUGMENT_COLUMN_DRIFT', 'Only the reviewed additional scores are accepted')
    result = base.set_index('patient_id').join(selected[added]).reset_index()
    result = result[['patient_id', 'cancer_id'] + [t['column'] for t in targets]]
    if not np.array_equal(result[base.columns[2:]].to_numpy(), base.iloc[:, 2:].to_numpy(), equal_nan=True):
        fail('STATE_AUGMENT_BASE_CHANGED', 'Existing scores and missing masks must remain identical')
    return result


def augment(cfg, historical, output, *, cancer_type='all', cancer_list=None):
    if socket.gethostname().split('.')[0] != '149':
        fail('STATE_EXECUTION_HOST', 'Run State preparation on 149')
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        fail('STATE_OUTPUT_EXISTS', 'Use a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json', {'target_host': '149', 'action': 'APPEND_VERIFIED_STATE_SCORES'})
    src = cfg['sources']['state']
    if src.get('catalog_id') != ROUTED_CATALOG_ID or sha(src['path']) != src.get('sha256'):
        fail('STATE_AUGMENT_BASE', 'Expected unchanged verified 483-target State source')
    base = pd.read_parquet(src['path'])
    authority = pd.read_csv(cfg['sources']['patient_map']['path'], sep='\t')
    verify_materialized_source(src, base, authority, expanded_targets(ROUTED_CATALOG_ID))
    inputs = {'base_state': _entry(src['path']), 'base_manifest': _entry(src['materialization_manifest']),
              'patient_map': _entry(cfg['sources']['patient_map']['path']), 'historical': _entry(historical)}
    if inputs['historical']['sha256'] != catalog(AUGMENTED_CATALOG_ID)['historical_source_sha256']:
        fail('STATE_HISTORICAL_SOURCE_DRIFT', 'Historical source must match the reviewed snapshot')
    scope = resolve_cancer_scope({'cancer_type': cancer_type, 'cancer_list': cancer_list or []}, authority.cancer_id.unique())
    authority = authority.loc[authority.cancer_id.isin(scope['selected_cancers'])].copy()
    base = base.loc[base.patient_id.isin(authority.patient_id)].copy().reset_index(drop=True)
    targets = expanded_targets(AUGMENTED_CATALOG_ID)
    added = [t for t in targets if t['column'] not in base]
    cols = [t['column'] for t in added]
    header = pd.read_csv(historical, sep='\t', nrows=0).columns
    metadata = [c for c in ('sample_id', 'patient_id', 'cancer_type', 'sample_type_code', 'is_tumor') if c in header]
    hist = pd.read_csv(historical, sep='\t', usecols=metadata+cols)
    values, choices, excluded, conflicts = select_patient_values(hist, authority, cols, 'historical')
    result = append_columns(base, values, targets)
    path = out/'STATE_AUGMENTED_PATIENT.parquet'
    result.to_parquet(path, index=False)
    restored = pd.read_parquet(path)
    if not np.array_equal(result.iloc[:, 2:].to_numpy(), restored.iloc[:, 2:].to_numpy(), equal_nan=True):
        fail('STATE_SERIALIZATION_DRIFT', 'State values changed during serialization')
    for filename, rows in [('STATE_VALUE_SOURCES.tsv', choices), ('STATE_EXCLUDED_ROWS.tsv', excluded),
                           ('STATE_MULTIPLE_TUMOR_VALUES.tsv', conflicts)]:
        pd.DataFrame(rows).to_csv(out/filename, sep='\t', index=False)
    coverage = []
    for cancer, f in result.groupby('cancer_id'):
        for t in targets:
            x = f[t['column']]
            coverage.append({'cancer_id': cancer, 'target_id': t['id'], 'family': t['family'],
                'role': t['role'], 'patients': len(f), 'measured': int(x.notna().sum()),
                'distinct_values': int(x.nunique()), 'fraction': float(x.notna().mean())})
    pd.DataFrame(coverage).to_csv(out/'STATE_COVERAGE.tsv', sep='\t', index=False)
    for key, rec in inputs.items():
        if _entry(rec['path']) != rec:
            fail('STATE_INPUT_CHANGED_DURING_MATERIALIZATION', key)
    record = {'format': FORMAT, 'state': 'COMPLETE', 'target_host': '149', 'catalog_id': AUGMENTED_CATALOG_ID,
        'target_catalog_sha256': digest(targets), 'authority_patient_map_sha256': inputs['patient_map']['sha256'],
        'cancer_scope': scope, 'patients': len(result), 'targets': len(targets),
        'primary_targets': sum(t['role']=='primary' for t in targets), 'inputs': inputs,
        'matrix': _entry(path), 'columns': list(result),
        'outputs': {p.name: _entry(p) for p in out.iterdir() if p.is_file()},
        'prior_targets_unchanged': len(expanded_targets(ROUTED_CATALOG_ID)), 'added_targets': [t['id'] for t in added],
        'scores_rescaled': False, 'scores_recomputed': False, 'missing_policy': 'NO_IMPUTATION; UNMEASURED_IS_NA_NOT_ZERO',
        'selection_policy': 'EXISTING_REVIEWED_PER_TARGET_TUMOR_SELECTION; NO_AVERAGING',
        'biological_claim': 'Original observational expression scores; no absolute functional or causal claim'}
    receipt = out/'STATE_MATERIALIZATION.json'
    atomic_json(receipt, record)
    binding = {**_entry(path), 'format': FORMAT, 'catalog_id': AUGMENTED_CATALOG_ID, 'columns': len(result.columns),
               'materialization_manifest': str(receipt.resolve()), 'materialization_manifest_sha256': sha(receipt)}
    atomic_json(out/'STATE_SOURCE_BINDING.json', binding)
    verify_materialized_source(binding, result, authority, targets)
    return binding


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'historical', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--cancer-type', '--cancer_type', choices=['all', 'list'], default='all')
    p.add_argument('--cancer-list', '--cancer_list', nargs='+', default=[])
    a = p.parse_args()
    print(augment(read_json(a.config), a.historical, a.output, cancer_type=a.cancer_type, cancer_list=a.cancer_list))


if __name__ == '__main__':
    main()
