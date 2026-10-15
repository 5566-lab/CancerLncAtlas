"""Bind verified preprocessing evidence without turning reference labels into CNV calls."""
from __future__ import annotations

import pandas as pd


def preprocessing_mask(metadata, *, required=False):
    if 'preprocessing_pass_qc' not in metadata:
        if required:
            raise ValueError('Required preprocessing evidence is absent')
        return pd.Series(True, index=metadata.index)
    for column in ('preprocessing_pass_qc', 'batch_correction_applied', 'annotation_completed'):
        if column not in metadata or not pd.api.types.is_bool_dtype(metadata[column]):
            raise ValueError(f'{column} must be boolean')
        if metadata[column].isna().any():
            raise ValueError(f'{column} cannot be null')
    selected = metadata.preprocessing_pass_qc
    if not metadata.loc[selected, ['batch_correction_applied', 'annotation_completed']].all().all():
        raise ValueError('QC-passing cell lacks batch correction or annotation evidence')
    for column in ('preprocessing_evidence_sha256', 'preprocessing_method', 'reference_annotation_source'):
        if column not in metadata or metadata.loc[selected, column].fillna('').astype(str).str.strip().eq('').any():
            raise ValueError(f'QC-passing cell lacks {column}')
    return selected


def bind_preprocessing(metadata, annotations, *, evidence_sha256, method, annotation_source):
    if metadata.cell_id.duplicated().any() or annotations.cell_id.duplicated().any():
        raise ValueError('Ambiguous cell identity in preprocessing join')
    if not isinstance(evidence_sha256, str) or len(evidence_sha256) != 64:
        raise ValueError('Invalid preprocessing evidence hash')
    if 'pass_qc' not in annotations or not pd.api.types.is_bool_dtype(annotations.pass_qc):
        raise ValueError('Preprocessing QC must be explicitly boolean')
    if annotations.pass_qc.isna().any():
        raise ValueError('Preprocessing QC is null')
    eligible = metadata.analysis_eligible_tumor_context.astype(bool)
    if not metadata.loc[eligible, 'cell_id'].isin(annotations.cell_id).all():
        raise ValueError('Current eligible cells are absent from preprocessing evidence')
    a = annotations.set_index('cell_id').reindex(metadata.cell_id).reset_index(drop=True)
    out = metadata.reset_index(drop=True).copy()
    qc = a.pass_qc.eq(True)
    out['preprocessing_pass_qc'] = qc
    out['batch_correction_applied'] = qc
    out['annotation_completed'] = qc
    out['preprocessing_evidence_sha256'] = evidence_sha256
    out['preprocessing_method'] = method
    out['reference_annotation_source'] = annotation_source
    for column in ('singler_label', 'singler_pruned_label', 'seurat_cluster', 'singler_delta_next'):
        if column in a:
            out['preprocessing_' + column] = a[column].to_numpy()
    label = a['singler_pruned_label'].fillna('').astype(str).str.strip()
    out['reference_annotation_status'] = 'QC_EXCLUDED'
    out.loc[qc & label.ne(''), 'reference_annotation_status'] = 'REFERENCE_LABEL_AVAILABLE'
    out.loc[qc & label.eq(''), 'reference_annotation_status'] = 'REFERENCE_UNCERTAIN'
    # Author/marker malignancy and canonical lineage remain separate evidence axes.
    out['cell_type_subtype'] = label.where(qc & label.ne(''), 'Not_assessed')
    for column in ('analysis_eligible_tumor_context', 'analysis_eligible_malignant_confirmed',
                   'analysis_eligible_malignant_candidate'):
        out[column] = out[column].astype(bool) & qc
    out.loc[eligible.to_numpy() & ~qc, 'analysis_exclusion_reason'] = 'PREPROCESSING_QC_EXCLUDED'
    preprocessing_mask(out, required=True)
    return out
