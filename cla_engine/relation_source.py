"""Frozen TCGA/CPTAC source-patient expression for relation preparation."""
from __future__ import annotations

import json
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
