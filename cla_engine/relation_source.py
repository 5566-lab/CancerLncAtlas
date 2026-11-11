"""Frozen TCGA/CPTAC source-patient expression for relation preparation."""
from __future__ import annotations

import json
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import sha


@dataclass(frozen=True)
class SourceMatrix:
    cohort: str
    patients: tuple[str, ...]
    entities: tuple[str, ...]
    values: np.ndarray
    measured: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class SourceReadouts:
    cohort: str
    layer: str
    patients: tuple[str, ...]
    readout_ids: tuple[str, ...]
    canonical_ids: tuple[str, ...]
    values: np.ndarray
    measured: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class SourceCovariates:
    cohort: str
    patients: tuple[str, ...]
    frame: pd.DataFrame
    source_sha256: str


@dataclass(frozen=True)
class SourceLocalCnv:
    cohort: str
    patients: tuple[str, ...]
    lncrna_ids: tuple[str, ...]
    values: np.ndarray
    callable: np.ndarray
    source_sha256: str


@dataclass(frozen=True)
class SourceDesign:
    cohort: str
    patients: tuple[str, ...]
    values: np.ndarray
    target_id: str
    report: dict


@dataclass(frozen=True)
class SourceAssociations:
    cohort: str
    layer: str
    patients: tuple[str, ...]
    lncrna_index: np.ndarray
    readout_index: np.ndarray
    rho: np.ndarray
    p: np.ndarray
    n: np.ndarray
    rank: np.ndarray
    available: np.ndarray


def _tsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def load_source_matrix(context, aligned_root, cohort, filename, entity_ids):
    """Align a measured matrix to the relation axis without imputing absent cells."""
    if context.context == 'sealed_test':
        raise ValueError('RELATION_SOURCE_SEALED_TEST_FORBIDDEN')
    if cohort not in {'TCGA', 'CPTAC'} or set(context.cohort_splits) != {'TCGA', 'CPTAC'}:
        raise ValueError('RELATION_SOURCE_COHORT_INVALID')
    if filename not in {'LNC_LOGTPM_FULL.parquet', 'GENE_LOGTPM.parquet'}:
        raise ValueError('RELATION_SOURCE_MATRIX_ROLE_INVALID')
    entities = tuple(entity_ids)
    if not entities or len(entities) != len(set(entities)):
        raise ValueError('RELATION_SOURCE_ENTITY_AXIS_INVALID')
    split = context.cohort_splits[cohort]
    patients = tuple(split['source_patients'])
    labels = tuple(split['label_patients'])
    if not patients or len(patients) != len(set(patients)) or set(patients) & set(labels):
        raise ValueError('RELATION_SOURCE_PATIENT_ROLE_INVALID')
    base = Path(aligned_root) / f'{cohort}_{context.cancer_id}'
    receipt = json.loads((base / 'COMPLETE.json').read_text())
    expected = receipt.get('files', {}).get(filename)
    path = base / filename
    if (receipt.get('target_host') != '149' or receipt.get('dataset') != cohort
            or receipt.get('cohort') != context.cancer_id or not expected
            or not path.is_file() or sha(path) != expected):
        raise ValueError('RELATION_SOURCE_MATRIX_RECEIPT_MISMATCH')
    parquet = pq.ParquetFile(path)
    columns = set(parquet.schema_arrow.names)
    present = [entity for entity in entities if entity in columns]
    if not present:
        raise ValueError('RELATION_SOURCE_NO_MEASURED_ENTITIES')
    frame = pd.read_parquet(path, columns=present)
    if not frame.index.is_unique or frame.index.hasnans:
        raise ValueError('RELATION_SOURCE_PATIENT_INDEX_INVALID')
    missing = set(patients) - set(frame.index.astype(str))
    if missing:
        raise ValueError('RELATION_SOURCE_PATIENT_MISSING: ' + ','.join(sorted(missing)[:3]))
    values = frame.loc[list(patients)].reindex(columns=entities).to_numpy(dtype=np.float32)
    measured = np.isfinite(values)
    if measured.shape != (len(patients), len(entities)):
        raise ValueError('RELATION_SOURCE_MATRIX_SHAPE_INVALID')
    return SourceMatrix(cohort, patients, entities, values, measured, expected)


def load_relation_lncrna_sources(context, aligned_root):
    """Use one exact lncRNA axis for both cohorts; missing CPTAC columns stay NaN."""
    result = {cohort: load_source_matrix(context, aligned_root, cohort,
                                         'LNC_LOGTPM_FULL.parquet', context.lncrna_ids)
              for cohort in ('TCGA', 'CPTAC')}
    if result['TCGA'].entities != result['CPTAC'].entities:
        raise ValueError('RELATION_SOURCE_LNCRNA_AXIS_MISMATCH')
    return result


def load_source_readouts(context, target_view_root, cohort, layer, output_targets):
    """Read only frozen source patients and exact mapped target readout columns."""
    if context.context == 'sealed_test':
        raise ValueError('RELATION_SOURCE_SEALED_TEST_FORBIDDEN')
    if cohort not in {'TCGA', 'CPTAC'} or layer not in {
            'RNA', 'STATE', 'PROTEIN', 'PHOSPHO', 'METHYLATION'}:
        raise ValueError('RELATION_SOURCE_READOUT_ROLE_INVALID')
    view_root = Path(target_view_root)
    definition = json.loads((view_root / 'Y_DEFINITION.json').read_text())
    if definition.get('format') != 'CLA_FUNCTIONAL_TARGET_VIEW_V1':
        raise ValueError('RELATION_SOURCE_TARGET_VIEW_REQUIRED')
    fused_root = Path(definition['source_y']).resolve()
    if sha(fused_root / 'Y_DEFINITION.json') != definition['source_y_definition_sha256']:
        raise ValueError('RELATION_SOURCE_FUSED_DEFINITION_DRIFT')
    base = fused_root / context.cancer_id / context.context
    fused_receipt = json.loads((base / 'COMPLETE.json').read_text())
    for name in ('READOUT_INDEX.tsv', 'INPUT_REFERENCES.json'):
        if sha(base / name) != fused_receipt['files'][name]:
            raise ValueError('RELATION_SOURCE_FUSED_CONTEXT_DRIFT: ' + name)
    mapping_rows = _tsv(view_root / 'LABEL_IDENTITY_MAP.tsv')
    if sha(view_root / 'LABEL_IDENTITY_MAP.tsv') != definition['label_identity_map_sha256']:
        raise ValueError('RELATION_SOURCE_IDENTITY_MAP_DRIFT')
    output_targets = tuple(output_targets)
    if (not output_targets or len(output_targets) != len(set(output_targets))
            or set(output_targets) - set(context.canonical_ids)):
        raise ValueError('RELATION_SOURCE_TARGET_AXIS_INVALID')
    output_to_source = {r['output_canonical_id']: r['source_label_canonical_id']
                        for r in mapping_rows}
    if len(output_to_source) != len(mapping_rows) or set(output_targets) - set(output_to_source):
        raise ValueError('RELATION_SOURCE_IDENTITY_MAP_INVALID')
    source_to_output = {output_to_source[target]: target for target in output_targets}
    if len(source_to_output) != len(output_targets):
        raise ValueError('RELATION_SOURCE_IDENTITY_NOT_ONE_TO_ONE')
    rows = [r for r in _tsv(base / 'READOUT_INDEX.tsv')
            if r['dataset'] == cohort and r['layer'] == layer
            and r['canonical_id'] in source_to_output]
    if len({r['target_id'] for r in rows}) != len(rows):
        raise ValueError('RELATION_SOURCE_READOUT_AXIS_INVALID')
    patients = _source_patients(context, cohort)
    if not rows:
        empty = np.empty((len(patients), 0), dtype=np.float32)
        return SourceReadouts(cohort, layer, patients, (), (), empty,
                              np.empty_like(empty, dtype=bool),
                              fused_receipt['files']['READOUT_INDEX.tsv'])
    directories = {r['directory'] for r in rows}
    references = [r for r in json.loads((base / 'INPUT_REFERENCES.json').read_text())
                  if r['dataset'] == cohort and r['layer'] == layer]
    if len(directories) != 1 or len(references) != 1 or directories != {references[0]['directory']}:
        raise ValueError('RELATION_SOURCE_READOUT_REFERENCE_MISMATCH')
    source = Path(references[0]['directory']).resolve()
    if sha(source / 'COMPLETE.json') != references[0]['receipt_sha256']:
        raise ValueError('RELATION_SOURCE_READOUT_RECEIPT_DRIFT')
    receipt = json.loads((source / 'COMPLETE.json').read_text())
    if (receipt.get('target_host') != '149' or receipt.get('dataset') != cohort
            or receipt.get('cancer_id') != context.cancer_id
            or receipt.get('context') != context.context or receipt.get('layer') != layer):
        raise ValueError('RELATION_SOURCE_READOUT_RECEIPT_IDENTITY')
    for name in ('PATIENT_S.parquet', 'TARGETS.tsv'):
        if sha(source / name) != receipt.get('files', {}).get(name):
            raise ValueError('RELATION_SOURCE_READOUT_FILE_DRIFT: ' + name)
    target_axis = [r['target_id'] for r in _tsv(source / 'TARGETS.tsv')]
    for row in rows:
        index = int(row['column_index'])
        if index < 0 or index >= len(target_axis) or target_axis[index] != row['target_id']:
            raise ValueError('RELATION_SOURCE_READOUT_COLUMN_MISMATCH')
    readout_ids = tuple(r['target_id'] for r in rows)
    frame = pd.read_parquet(source / 'PATIENT_S.parquet', columns=list(readout_ids))
    if (not frame.index.is_unique or set(patients) - set(frame.index.astype(str))):
        raise ValueError('RELATION_SOURCE_READOUT_PATIENT_MISMATCH')
    values = frame.loc[list(patients), list(readout_ids)].to_numpy(dtype=np.float32)
    return SourceReadouts(cohort, layer, patients, readout_ids,
                          tuple(source_to_output[r['canonical_id']] for r in rows),
                          values, np.isfinite(values), receipt['files']['PATIENT_S.parquet'])


def _source_patients(context, cohort):
    if context.context == 'sealed_test':
        raise ValueError('RELATION_SOURCE_SEALED_TEST_FORBIDDEN')
    if cohort not in {'TCGA', 'CPTAC'} or set(context.cohort_splits) != {'TCGA', 'CPTAC'}:
        raise ValueError('RELATION_SOURCE_COHORT_INVALID')
    split = context.cohort_splits[cohort]
    patients = tuple(split['source_patients'])
    if (not patients or len(patients) != len(set(patients))
            or set(patients) & set(split['label_patients'])):
        raise ValueError('RELATION_SOURCE_PATIENT_ROLE_INVALID')
    return patients


def load_source_covariates(context, covariate_root, cohort, expected_sha256):
    """Bind patient covariates to a caller-pinned source file, never label patients."""
    patients = _source_patients(context, cohort)
    path = Path(covariate_root) / f'{cohort}_{context.cancer_id}' / 'COVARIATES.parquet'
    if not expected_sha256 or not path.is_file() or sha(path) != expected_sha256:
        raise ValueError('RELATION_SOURCE_COVARIATE_HASH_MISMATCH')
    frame = pd.read_parquet(path)
    if ('patient_id' not in frame or frame.patient_id.isna().any()
            or not frame.patient_id.is_unique):
        raise ValueError('RELATION_SOURCE_COVARIATE_PATIENT_INDEX_INVALID')
    indexed = frame.set_index('patient_id', drop=False)
    if set(patients) - set(indexed.index):
        raise ValueError('RELATION_SOURCE_COVARIATE_PATIENT_MISSING')
    return SourceCovariates(cohort, patients, indexed.loc[list(patients)].reset_index(drop=True),
                            expected_sha256)


def load_source_local_cnv(context, cnv_root_by_cohort, cohort, lncrna_ids):
    """Preserve local CNV callability while aligning the frozen source axis."""
    patients = _source_patients(context, cohort)
    lncrna_ids = tuple(lncrna_ids)
    if not lncrna_ids or len(lncrna_ids) != len(set(lncrna_ids)):
        raise ValueError('RELATION_SOURCE_CNV_LNCRNA_AXIS_INVALID')
    if set(lncrna_ids) - set(context.lncrna_ids):
        raise ValueError('RELATION_SOURCE_CNV_OUTSIDE_FROZEN_AXIS')
    if set(cnv_root_by_cohort) != {'TCGA', 'CPTAC'}:
        raise ValueError('RELATION_SOURCE_CNV_ROOTS_INVALID')
    base = Path(cnv_root_by_cohort[cohort]) / context.cancer_id
    receipt_path = base / 'COMPLETE.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('cancer_id') != context.cancer_id or receipt.get('target_host') != '149':
        raise ValueError('RELATION_SOURCE_CNV_RECEIPT_IDENTITY')
    names = ('patient_ids.json', 'lncrna_ids.json', 'lncrna_signed_mean.npy',
             'lncrna_callable.npy')
    for name in names:
        item = receipt.get('files', {}).get(name)
        expected = item.get('sha256') if isinstance(item, dict) else item
        if not expected or sha(base / name) != expected:
            raise ValueError('RELATION_SOURCE_CNV_FILE_DRIFT: ' + name)
    patient_axis = json.loads((base / 'patient_ids.json').read_text())
    lnc_axis = json.loads((base / 'lncrna_ids.json').read_text())
    if (len(patient_axis) != len(set(patient_axis)) or len(lnc_axis) != len(set(lnc_axis))
            or set(patients) - set(patient_axis) or set(lncrna_ids) - set(lnc_axis)):
        raise ValueError('RELATION_SOURCE_CNV_AXIS_MISMATCH')
    row = [patient_axis.index(patient) for patient in patients]
    col = [lnc_axis.index(lnc) for lnc in lncrna_ids]
    values_file = np.load(base / 'lncrna_signed_mean.npy', mmap_mode='r', allow_pickle=False)
    callable_file = np.load(base / 'lncrna_callable.npy', mmap_mode='r', allow_pickle=False)
    shape = (len(patient_axis), len(lnc_axis))
    if values_file.shape != shape or callable_file.shape != shape:
        raise ValueError('RELATION_SOURCE_CNV_SHAPE_INVALID')
    values = np.asarray(values_file[np.ix_(row, col)], dtype=np.float32)
    callable_mask = np.asarray(callable_file[np.ix_(row, col)], dtype=bool)
    if not np.array_equal(np.isfinite(values), callable_mask):
        raise ValueError('RELATION_SOURCE_CNV_CALLABILITY_MISMATCH')
    return SourceLocalCnv(cohort, patients, lncrna_ids, values, callable_mask,
                          sha(receipt_path))


def fit_source_design(covariates, cfg, *, family, target_id):
    """Fit the target-aware nuisance design on source patients only."""
    from .fused_adjustment.design import design

    patients = list(covariates.patients)
    z, report = design(covariates.frame, patients, patients, family=family,
                       target_id=target_id, cfg=cfg,
                       admission_cohorts={'source': patients})
    if z.shape[0] != len(patients):
        raise ValueError('RELATION_SOURCE_DESIGN_PATIENT_AXIS_MISMATCH')
    return SourceDesign(covariates.cohort, covariates.patients, z, target_id, report)


def source_readout_associations(expression, readouts, local_cnv, source_design,
                                lncrna_index, readout_index, *, minimum_pairs=12):
    """Recompute fold-local rank associations; partial-pair FDR is not exposed."""
    from .statistics import pairs

    items = (expression, readouts, local_cnv, source_design)
    if (len({item.cohort for item in items}) != 1
            or len({item.patients for item in items}) != 1
            or expression.entities != local_cnv.lncrna_ids):
        raise ValueError('RELATION_SOURCE_ASSOCIATION_AXIS_MISMATCH')
    if (not np.array_equal(expression.measured, np.isfinite(expression.values))
            or not np.array_equal(readouts.measured, np.isfinite(readouts.values))
            or not np.array_equal(local_cnv.callable, np.isfinite(local_cnv.values))):
        raise ValueError('RELATION_SOURCE_ASSOCIATION_MEASUREMENT_MASK_MISMATCH')
    li = np.asarray(lncrna_index)
    ti = np.asarray(readout_index)
    if (li.ndim != 1 or ti.ndim != 1 or li.shape != ti.shape
            or not np.issubdtype(li.dtype, np.integer)
            or not np.issubdtype(ti.dtype, np.integer)
            or np.any(li < 0) or np.any(li >= len(expression.entities))
            or np.any(ti < 0) or np.any(ti >= len(readouts.readout_ids))):
        raise ValueError('RELATION_SOURCE_ASSOCIATION_PAIR_AXIS_INVALID')
    if minimum_pairs < 3:
        raise ValueError('RELATION_SOURCE_ASSOCIATION_MINIMUM_INVALID')
    if not len(li):
        empty = np.empty(0, dtype=np.float32)
        return SourceAssociations(expression.cohort, readouts.layer, expression.patients,
                                  li, ti, empty, empty.copy(), np.empty(0, np.int32),
                                  np.empty(0, np.int16), np.empty(0, bool))
    result = pairs(expression.values, readouts.values, local_cnv.values,
                   local_cnv.callable, source_design.values, li, ti,
                   np.arange(len(expression.patients)), minimum_pairs,
                   cnv_transform='raw_signed')
    return SourceAssociations(expression.cohort, readouts.layer, expression.patients,
                              li, ti, result['r'].astype(np.float32),
                              result['p'].astype(np.float32), result['n'],
                              result['rank'], result['available'])
