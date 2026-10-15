"""Apply source-reviewed sample and donor mappings without inferring malignancy."""
from __future__ import annotations

import pandas as pd


def verified_sample_mask(metadata):
    """Fail closed when an explicit source-review gate is present."""
    if 'source_review_include' not in metadata:
        return pd.Series(True, index=metadata.index)
    for column in ('source_review_include', 'patient_id_verified'):
        if column not in metadata or not pd.api.types.is_bool_dtype(metadata[column]):
            raise ValueError(f'{column} must be an explicit boolean')
        if metadata[column].isna().any():
            raise ValueError(f'{column} cannot be null')
    selected = metadata.source_review_include
    if not metadata.loc[selected, 'patient_id_verified'].all():
        raise ValueError('Selected sample has an unverified patient')
    for column in ('patient_id', 'source_dataset_id', 'source_review_evidence'):
        values = metadata.loc[selected, column].fillna('').astype(str).str.strip()
        if values.eq('').any():
            raise ValueError(f'Selected sample lacks {column}')
    return selected


def apply_sample_policy(legacy, published, policy, cancer):
    """Join exact cell and sample identities, keeping excluded rows H5-aligned."""
    if legacy.cell_id.duplicated().any() or published.cellname.duplicated().any():
        raise ValueError('Duplicate cell identity')
    if policy.published_sample.duplicated().any():
        raise ValueError('Ambiguous sample policy')
    verified_sample_mask(policy)
    cells = published.set_index('cellname').reindex(legacy.cell_id).reset_index(drop=True)
    if cells.SampleID.isna().any():
        raise ValueError('Cell absent from deposited author metadata')
    own = legacy.reset_index(drop=True)
    if not own.patient_id_raw.eq(cells.SampleID).all():
        raise ValueError('Author sample identity mismatch')
    if not own.cell_type_raw.eq(cells.celltype).all():
        raise ValueError('Author cell annotation mismatch')
    if not pd.to_numeric(own.n_count_rna).eq(pd.to_numeric(cells.nCount_RNA)).all():
        raise ValueError('Author count signature mismatch')
    mapped = policy.set_index('published_sample').reindex(cells.SampleID).reset_index(drop=True)
    if mapped.source_review_include.isna().any():
        raise ValueError('Sample missing explicit reviewed disposition')
    if set(mapped.cancer_id) != {cancer}:
        raise ValueError('Cancer identity mismatch')
    mapped['cell_id'] = 'LEGACY_SC_' + cancer + ':' + own.cell_id
    mapped['sample_id'] = cells.SampleID
    mapped['cell_type_major'] = cells.celltype
    mapped['source_sample_class'] = cells.group
    mapped['sample_class'] = cells.group.map({'Tumor': 'Tumor', 'Adjacent': 'Normal', 'Normal': 'Normal'})
    if mapped.sample_class.isna().any():
        raise ValueError('Unknown author sample class')
    mapped['dataset_id'] = 'SC_GSE210347_VERIFIED_' + cancer
    mapped['patient_id'] = mapped.patient_id.fillna('')
    mapped['source_review_include'] = mapped.source_review_include.astype(bool)
    mapped['patient_id_verified'] = mapped.patient_id_verified.astype(bool)
    verified_sample_mask(mapped)
    return mapped
