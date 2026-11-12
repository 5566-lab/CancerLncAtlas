"""Export patient measurements and structural routes without lncRNA statistics.

This is the measured foundation for defining Y. It does not choose a scalar
fusion method, fit a predictor, or infer a lncRNA--pathway relationship.
"""
from __future__ import annotations
import collections
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .relation_evidence_bindings import FORMAT as BINDING_FORMAT, file_sha, read_table, write_table

FORMAT = 'CLA_PATIENT_MEASURED_READOUTS_V1'
CONTEXTS = {'train_label_fold2', 'train_label_fold3', 'train_label_fold4', 'validation'}


def _json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _rows(path, rows, fields):
    write_table(path, rows, fields)


def measurement_mask(frame):
    """A measured zero or a constant readout is still observed."""
    if not frame.index.is_unique or not frame.columns.is_unique:
        raise ValueError('MEASUREMENT_DUPLICATE_AXIS')
    values = frame.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('MEASUREMENT_INFINITE_VALUE')
    return pd.DataFrame(np.isfinite(values), index=frame.index, columns=frame.columns)


def read_partition(path, patient_ids, columns):
    """Select permitted patient rows at the parquet reader; never load AXES.json."""
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError('MEASUREMENT_DUPLICATE_PATIENT')
    metadata = json.loads(pq.ParquetFile(path).metadata.metadata[b'pandas'])
    index = metadata['index_columns']
    if len(index) != 1 or not isinstance(index[0], str):
        raise ValueError('MEASUREMENT_REQUIRES_EXPLICIT_PATIENT_INDEX')
    frame = pd.read_parquet(path, columns=columns, filters=[(index[0], 'in', patient_ids)])
    if set(frame.index) != set(patient_ids) or not frame.index.is_unique:
        raise ValueError('MEASUREMENT_PATIENT_AXIS_MISMATCH')
    frame = frame.reindex(patient_ids)
    if list(frame.columns) != list(columns):
        raise ValueError('MEASUREMENT_COLUMN_AXIS_MISMATCH')
    return frame


def summarize_target_coverage(bindings, masks, target_rows, cohort, context):
    grouped = collections.defaultdict(list)
    for row in bindings:
        if row['cohort'] == cohort:
            grouped[row['output_canonical_id']].append(row)
    patient_index = next(iter(masks.values())).index
    if any(not frame.index.equals(patient_index) for frame in masks.values()):
        raise ValueError('MEASUREMENT_LAYER_PATIENT_MISMATCH')
    result = []
    for target in target_rows:
        cid = target['canonical_id']
        routes = grouped[cid]
        row = {'cohort':cohort,'context':context,'canonical_id':cid,
               'analysis_role':target['analysis_role'],'patients_in_scope':len(patient_index),
               'structurally_bound_readouts':len(routes)}
        for key, roles in [('identity', {'equivalent_pathway_readout','same_state_readout'}),
                           ('member', {'molecular_component_readout'}),
                           ('any', {'equivalent_pathway_readout','same_state_readout','molecular_component_readout'})]:
            selected = [r for r in routes if r['evidence_role'] in roles]
            vectors = [masks[r['layer']][r['target_id']].to_numpy(bool) for r in selected]
            row[key+'_readouts_with_measurement'] = sum(bool(v.any()) for v in vectors)
            row['patients_with_'+key+'_measurement'] = int(np.logical_or.reduce(vectors).sum()) if vectors else 0
        row['condition_required'] = target['condition_required']
        result.append(row)
    return result


def materialize_measured_readouts(binding_dir, output, *, cancer, context, progress=None):
    from .local_execution import require_host_149
    require_host_149()
    if cancer not in {'BRCA','LUAD','COAD'} or context not in CONTEXTS:
        raise ValueError('MEASUREMENT_OUTSIDE_OPEN_DEMO_CONTEXTS')
    binding_dir, output = Path(binding_dir), Path(output)
    if output.exists():
        raise ValueError('MEASUREMENT_OUTPUT_EXISTS')
    output.mkdir(parents=True)
    _json(output/'LAUNCH_PREFLIGHT.json', {'target_host':'149','training':False,
          'operation':'materialize patient measurements independently of lncRNA',
          'cancer':cancer,'context':context})
    receipt = json.loads((binding_dir/'COMPLETE.json').read_text())
    if receipt.get('format') != BINDING_FORMAT or receipt.get('target_host') != '149':
        raise ValueError('MEASUREMENT_BINDING_FORMAT')
    for name, sha in receipt['files'].items():
        if file_sha(binding_dir/name) != sha:
            raise ValueError('MEASUREMENT_BINDING_DRIFT')
    refs = json.loads((binding_dir/'INPUT_REFERENCES.json').read_text())
    for item in refs['structural'].values():
        if file_sha(item['path']) != item['sha256']:
            raise ValueError('MEASUREMENT_STRUCTURAL_DRIFT')
    policies = read_table(refs['structural']['TARGET_POLICY.tsv']['path'])
    primary = {r['canonical_id']:r for r in read_table(refs['structural']['backbone']['path'])}
    active = {r['canonical_id'] for r in policies if r['active_in_analysis'] == 'True'}
    targets = []
    for r in policies:
        if r['analysis_role'] in {'primary_pathway','state_auxiliary'}:
            targets.append({'canonical_id':r['canonical_id'],'analysis_role':r['analysis_role'],
                            'display_category':r['roles'],'network_id':primary.get(r['canonical_id'],{}).get('network_id',''),
                            'condition_required':primary.get(r['canonical_id'],{}).get('network_type','Reference') != 'Reference'})
    if len({r['canonical_id'] for r in targets}) != len(targets):
        raise ValueError('MEASUREMENT_DUPLICATE_TARGET')
    bindings = read_table(binding_dir/'READOUT_BINDINGS.tsv')
    dispositions = {(r['cohort'],r['layer'],int(r['column_index'])):r
                    for r in read_table(binding_dir/'SOURCE_DISPOSITIONS.tsv')}
    mask_by_cohort = collections.defaultdict(dict)
    splits, source_records, measurement_rows, target_coverage = {}, [], [], []
    for spec in refs['streams']:
        cohort, layer = spec['cohort'], spec['layer']
        path = Path(spec['directory'])
        if path.parent.name != context or path.parent.parent.name != cancer:
            raise ValueError('MEASUREMENT_SOURCE_CONTEXT_MISMATCH')
        if layer in mask_by_cohort[cohort]:
            raise ValueError('MEASUREMENT_DUPLICATE_STREAM')
        if file_sha(path/'COMPLETE.json') != spec['source_receipt_sha256']:
            raise ValueError('MEASUREMENT_SOURCE_RECEIPT_DRIFT')
        source_receipt = json.loads((path/'COMPLETE.json').read_text())
        for name in ['TARGETS.tsv','PATIENT_S.parquet']:
            if file_sha(path/name) != source_receipt['files'][name]:
                raise ValueError('MEASUREMENT_SOURCE_CONTENT_DRIFT')
        # Source receipt may also describe association files. They are neither
        # opened nor used to define the measurement axis or any observed mask.
        split = json.loads((path.parent/'SPLIT.json').read_text())
        if split.get('outer_fold') != 0 or 0 in split.get('source_folds',[]) + split.get('label_folds',[]):
            raise ValueError('MEASUREMENT_SEALED_FOLD')
        source, label = split['source_patients'], split['label_patients']
        if set(source) & set(label):
            raise ValueError('MEASUREMENT_PATIENT_PARTITION_OVERLAP')
        patients = source + label
        if not patients:
            raise ValueError('MEASUREMENT_EMPTY_PATIENT_SCOPE')
        if cohort in splits and splits[cohort] != split:
            raise ValueError('MEASUREMENT_SPLIT_DRIFT')
        splits[cohort] = split
        metadata = read_table(path/'TARGETS.tsv')
        selected = [(j,r) for j,r in enumerate(metadata)
                    if dispositions[(cohort,layer,j)]['resolved_source_canonical_id'] in active]
        columns = [r['target_id'] for _,r in selected]
        if len(columns) != len(set(columns)):
            raise ValueError('MEASUREMENT_DUPLICATE_READOUT')
        for b in bindings:
            if b['cohort'] == cohort and b['layer'] == layer:
                if metadata[int(b['column_index'])]['target_id'] != b['target_id'] or b['target_id'] not in columns:
                    raise ValueError('MEASUREMENT_BINDING_AXIS_MISMATCH')
        values = read_partition(path/'PATIENT_S.parquet', patients, columns)
        observed = measurement_mask(values)
        mask_by_cohort[cohort][layer] = observed
        folder = output/cohort/layer
        folder.mkdir(parents=True)
        values.to_parquet(folder/'VALUES.parquet', compression='zstd')
        observed.to_parquet(folder/'OBSERVED.parquet', compression='zstd')
        for j, r in selected:
            measured = observed[r['target_id']]
            measurement_rows.append({'cohort':cohort,'layer':layer,'column_index':j,
                'target_id':r['target_id'],'source_target_id':r['source_target_id'],
                'canonical_id':dispositions[(cohort,layer,j)]['resolved_source_canonical_id'],
                'measurement_layer':r['measurement_layer'],
                'patients_in_scope':len(patients),'patients_measured':int(measured.sum()),
                'measured_fraction':float(measured.mean()),
                'nonconstant':values.loc[measured,r['target_id']].nunique()>1,
                'source_metadata_json':json.dumps(r,ensure_ascii=False,sort_keys=True)})
        source_records.append({'cohort':cohort,'layer':layer,'directory':str(path),
            'receipt_sha256':spec['source_receipt_sha256'],
            'scores_sha256':source_receipt['files']['PATIENT_S.parquet'],
            'metadata_sha256':source_receipt['files']['TARGETS.tsv'],
            'split_sha256':file_sha(path.parent/'SPLIT.json'),
            'retained_readouts':len(columns)})
        if progress:
            progress(f'MEASUREMENTS {cancer} {context} {cohort} {layer}: {len(patients)} patients, {len(columns)} readouts')
    for cohort, masks in mask_by_cohort.items():
        target_coverage.extend(summarize_target_coverage(bindings,masks,targets,cohort,context))
    _rows(output/'TARGETS.tsv',targets,tuple(targets[0]))
    _rows(output/'READOUTS.tsv',measurement_rows,tuple(measurement_rows[0]))
    _rows(output/'READOUT_BINDINGS.tsv',bindings,tuple(bindings[0]))
    _rows(output/'TARGET_MEASUREMENT_COVERAGE.tsv',target_coverage,tuple(target_coverage[0]))
    _json(output/'PATIENT_PARTITIONS.json',splits)
    _json(output/'INPUT_REFERENCES.json',{'binding_directory':str(binding_dir),
          'binding_receipt_sha256':file_sha(binding_dir/'COMPLETE.json'),
          'structural':refs['structural'],'streams':source_records})
    result = {'format':FORMAT,'target_host':'149','cancer':cancer,'context':context,
              'biological_targets':len(targets),'primary_pathways':len(primary),
              'state_identities':sum(r['analysis_role']=='state_auxiliary' for r in targets),
              'readout_columns':len(measurement_rows),
              'patients':{c:len(next(iter(m.values()))) for c,m in mask_by_cohort.items()},
              'files':{str(p.relative_to(output)):file_sha(p) for p in sorted(output.rglob('*')) if p.is_file()}}
    _json(output/'COMPLETE.json',result)
    return result
