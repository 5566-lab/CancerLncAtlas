"""Source features are measured in frozen source patients, with explicit gaps."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cla_engine.common import sha
from cla_engine.relation_source import load_relation_lncrna_sources, load_source_matrix


def fixture(tmp_path):
    context = SimpleNamespace(context='train_label_fold2', cancer_id='BRCA',
        lncrna_ids=('L1', 'L2'), cohort_splits={
            'TCGA': {'source_patients': ['T1', 'T2'], 'label_patients': ['T3']},
            'CPTAC': {'source_patients': ['C1', 'C2'], 'label_patients': ['C3']}})
    for cohort in ('TCGA', 'CPTAC'):
        base = tmp_path / f'{cohort}_BRCA'
        base.mkdir()
        names = context.cohort_splits[cohort]
        columns = {'L1': [1., 2., 3.]}
        if cohort == 'TCGA':
            columns['L2'] = [4., 5., 6.]
        frame = pd.DataFrame(columns, index=names['source_patients'] + names['label_patients'])
        path = base / 'LNC_LOGTPM_FULL.parquet'
        frame.to_parquet(path)
        (base / 'COMPLETE.json').write_text(__import__('json').dumps({
            'target_host': '149', 'dataset': cohort, 'cohort': 'BRCA',
            'files': {path.name: sha(path)}}))
    return context


def test_source_only_alignment_preserves_missing_cptac_lnc(tmp_path):
    context = fixture(tmp_path)
    sources = load_relation_lncrna_sources(context, tmp_path)
    assert sources['TCGA'].values.shape == (2, 2)
    assert sources['TCGA'].measured.all()
    assert np.isnan(sources['CPTAC'].values[:, 1]).all()
    assert not sources['CPTAC'].measured[:, 1].any()
    assert sources['CPTAC'].patients == ('C1', 'C2')
    assert sources['CPTAC'].values[0, 0] == 1.


def test_rejects_patient_role_and_source_checksum_drift(tmp_path):
    context = fixture(tmp_path)
    context.cohort_splits['TCGA']['label_patients'] = ['T1']
    with pytest.raises(ValueError, match='PATIENT_ROLE_INVALID'):
        load_relation_lncrna_sources(context, tmp_path)
    context.cohort_splits['TCGA']['label_patients'] = ['T3']
    pd.DataFrame({'L1': [10.]}, index=['T1']).to_parquet(tmp_path / 'TCGA_BRCA/LNC_LOGTPM_FULL.parquet')
    with pytest.raises(ValueError, match='RECEIPT_MISMATCH'):
        load_relation_lncrna_sources(context, tmp_path)


def test_rejects_sealed_test_and_unknown_matrix(tmp_path):
    context = fixture(tmp_path)
    context.context = 'sealed_test'
    with pytest.raises(ValueError, match='SEALED_TEST_FORBIDDEN'):
        load_relation_lncrna_sources(context, tmp_path)
    context.context = 'train_label_fold2'
    with pytest.raises(ValueError, match='MATRIX_ROLE_INVALID'):
        load_source_matrix(context, tmp_path, 'TCGA', 'Y.npz', ('L1',))
