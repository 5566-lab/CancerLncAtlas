"""Source features are measured in frozen source patients, with explicit gaps."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cla_engine.common import sha
from cla_engine.relation_source import (load_relation_lncrna_sources, load_source_matrix,
                                        load_source_readouts, load_source_covariates,
                                        load_source_local_cnv, source_readout_associations,
                                        source_numeric_block, SourceMatrix, SourceReadouts,
                                        SourceLocalCnv, SourceDesign, SourceAssociations)


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


def readout_fixture(tmp_path):
    context = SimpleNamespace(context='train_label_fold2', cancer_id='BRCA',
        canonical_ids=('P_NEW',), cohort_splits={
            'TCGA': {'source_patients': ['T1', 'T2'], 'label_patients': ['T3']},
            'CPTAC': {'source_patients': ['C1'], 'label_patients': ['C2']}})
    fused = tmp_path / 'fused'
    base = fused / 'BRCA/train_label_fold2'
    base.mkdir(parents=True)
    (fused / 'Y_DEFINITION.json').write_text('{}')
    source = tmp_path / 'scores'
    source.mkdir()
    score = source / 'PATIENT_S.parquet'
    pd.DataFrame({'R1': [1., 2., 999.]}, index=['T1', 'T2', 'T3']).to_parquet(score)
    targets = source / 'TARGETS.tsv'
    targets.write_text('target_id\nR1\n')
    receipt = {'target_host': '149', 'dataset': 'TCGA', 'cancer_id': 'BRCA',
               'context': context.context, 'layer': 'PROTEIN',
               'files': {name: sha(source / name) for name in ('PATIENT_S.parquet', 'TARGETS.tsv')}}
    (source / 'COMPLETE.json').write_text(json.dumps(receipt))
    (base / 'INPUT_REFERENCES.json').write_text(json.dumps([{
        'dataset': 'TCGA', 'layer': 'PROTEIN', 'directory': str(source),
        'receipt_sha256': sha(source / 'COMPLETE.json')}]))
    readout = base / 'READOUT_INDEX.tsv'
    readout.write_text('canonical_id\tdataset\tlayer\ttarget_id\tdirectory\tcolumn_index\n'
                       f'P_OLD\tTCGA\tPROTEIN\tR1\t{source}\t0\n')
    (base / 'COMPLETE.json').write_text(json.dumps({'files': {
        name: sha(base / name) for name in ('READOUT_INDEX.tsv', 'INPUT_REFERENCES.json')}}))
    view = tmp_path / 'view'
    view.mkdir()
    mapping = view / 'LABEL_IDENTITY_MAP.tsv'
    mapping.write_text('output_canonical_id\tsource_label_canonical_id\nP_NEW\tP_OLD\n')
    (view / 'Y_DEFINITION.json').write_text(json.dumps({
        'format': 'CLA_FUNCTIONAL_TARGET_VIEW_V1', 'source_y': str(fused),
        'source_y_definition_sha256': sha(fused / 'Y_DEFINITION.json'),
        'label_identity_map_sha256': sha(mapping)}))
    return context, view, source


def test_readout_mapping_uses_only_source_patients(tmp_path):
    context, view, source = readout_fixture(tmp_path)
    block = load_source_readouts(context, view, 'TCGA', 'PROTEIN', ('P_NEW',))
    assert block.canonical_ids == ('P_NEW',)
    assert block.readout_ids == ('R1',)
    assert block.patients == ('T1', 'T2')
    assert block.values[:, 0].tolist() == [1., 2.]
    assert block.measured.all()
    absent = load_source_readouts(context, view, 'TCGA', 'PHOSPHO', ('P_NEW',))
    assert absent.patients == ('T1', 'T2')
    assert absent.values.shape == (2, 0)
    assert absent.readout_ids == ()
    source.joinpath('TARGETS.tsv').write_text('target_id\nR2\n')
    with pytest.raises(ValueError, match='FILE_DRIFT'):
        load_source_readouts(context, view, 'TCGA', 'PROTEIN', ('P_NEW',))


def test_covariates_and_local_cnv_use_only_frozen_source_patients(tmp_path):
    context = fixture(tmp_path)
    cov_root = tmp_path / 'covariates'
    roots = {}
    for cohort, prefix in (('TCGA', 'T'), ('CPTAC', 'C')):
        cov = cov_root / f'{cohort}_BRCA'
        cov.mkdir(parents=True)
        path = cov / 'COVARIATES.parquet'
        pd.DataFrame({'patient_id': [f'{prefix}1', f'{prefix}2', f'{prefix}3'],
                      'purity': [.7, .8, 999.]}).to_parquet(path)
        block = load_source_covariates(context, cov_root, cohort, sha(path))
        assert block.patients == (f'{prefix}1', f'{prefix}2')
        assert block.frame.purity.tolist() == [.7, .8]
        if cohort == 'TCGA':
            tcga_cov_sha = block.source_sha256
        root = tmp_path / f'{cohort}_cnv'
        base = root / 'BRCA'
        base.mkdir(parents=True)
        roots[cohort] = root
        (base / 'patient_ids.json').write_text(json.dumps([f'{prefix}3', f'{prefix}2', f'{prefix}1']))
        (base / 'lncrna_ids.json').write_text(json.dumps(['L2', 'L1']))
        np.save(base / 'lncrna_signed_mean.npy', np.array([[999., 999.], [np.nan, 2.], [3., 1.]]))
        np.save(base / 'lncrna_callable.npy', np.array([[True, True], [False, True], [True, True]]))
        files = {}
        for name in ('patient_ids.json', 'lncrna_ids.json', 'lncrna_signed_mean.npy',
                     'lncrna_callable.npy'):
            files[name] = {'sha256': sha(base / name)} if cohort == 'TCGA' else sha(base / name)
        (base / 'COMPLETE.json').write_text(json.dumps({
            'target_host': '149', 'cancer_id': 'BRCA', 'files': files}))
    cnv = load_source_local_cnv(context, roots, 'TCGA', context.lncrna_ids)
    assert cnv.patients == ('T1', 'T2')
    assert cnv.values[:, 0].tolist() == [1., 2.]
    assert cnv.values[0, 1] == 3.
    assert np.isnan(cnv.values[1, 1]) and not cnv.callable[1, 1]
    assert load_source_local_cnv(context, roots, 'CPTAC', ('L1',)).values[:, 0].tolist() == [1., 2.]
    pd.DataFrame({'patient_id': ['T1'], 'purity': [.1]}).to_parquet(
        cov_root / 'TCGA_BRCA/COVARIATES.parquet')
    with pytest.raises(ValueError, match='COVARIATE_HASH_MISMATCH'):
        load_source_covariates(context, cov_root, 'TCGA', tcga_cov_sha)
    (roots['TCGA'] / 'BRCA/lncrna_callable.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='CNV_FILE_DRIFT'):
        load_source_local_cnv(context, roots, 'TCGA', context.lncrna_ids)


def test_source_readout_association_keeps_missing_measurements_unavailable():
    patients = tuple(f'T{i}' for i in range(20))
    trend = np.arange(20, dtype=np.float32)
    expression = SourceMatrix('TCGA', patients, ('L1', 'L2'),
        np.column_stack([trend, np.full(20, np.nan, np.float32)]),
        np.column_stack([np.ones(20, bool), np.zeros(20, bool)]), 'expression-sha')
    readouts = SourceReadouts('TCGA', 'PROTEIN', patients, ('R1',), ('P1',),
        trend[:, None], np.ones((20, 1), bool), 'readout-sha')
    cnv = SourceLocalCnv('TCGA', patients, ('L1', 'L2'), np.zeros((20, 2)),
                         np.ones((20, 2), bool), 'cnv-sha')
    design = SourceDesign('TCGA', patients, np.ones((20, 1)), '__pathway__', {})
    result = source_readout_associations(expression, readouts, cnv, design,
                                         np.array([0, 1]), np.array([0, 0]))
    assert result.available.tolist() == [True, False]
    assert result.rho[0] > .99
    assert result.n.tolist() == [20, 0]
    with pytest.raises(ValueError, match='ASSOCIATION_AXIS_MISMATCH'):
        source_readout_associations(expression, readouts,
            SourceLocalCnv('TCGA', patients[::-1], cnv.lncrna_ids, cnv.values,
                           cnv.callable, cnv.source_sha256), design,
            np.array([0]), np.array([0]))
    with pytest.raises(ValueError, match='MEASUREMENT_MASK_MISMATCH'):
        source_readout_associations(expression,
            SourceReadouts('TCGA', 'PROTEIN', patients, ('R1',), ('P1',),
                           readouts.values, np.zeros((20, 1), bool), 'readout-sha'),
            cnv, design, np.array([0]), np.array([0]))


def test_numeric_block_keeps_readout_multiplicity_and_missing_targets_distinct():
    patients = ('T1', 'T2')
    readouts = SourceReadouts('TCGA', 'PROTEIN', patients, ('R1', 'R2'),
                              ('P1', 'P1'), np.ones((2, 2)),
                              np.ones((2, 2), bool), 'sha')
    associations = SourceAssociations('TCGA', 'PROTEIN', patients,
        np.array([0, 0, 1]), np.array([0, 1, 0]),
        np.array([.4, -.2, np.nan]), np.array([.01, .02, np.nan]),
        np.array([20, 18, 0]), np.array([3, 3, 0]),
        np.array([True, True, False]))
    block = source_numeric_block(associations, readouts,
                                 np.array([0, 1, 0]), np.array([0, 0, 1]),
                                 ('P1', 'P2'))
    assert block.shape == (3, 6)
    assert block[0, 0] == pytest.approx(.1)
    assert block[0, 1] == pytest.approx(np.sqrt(.1))
    assert block[0, 2] == pytest.approx(.4)
    assert block[0, 4] == pytest.approx(np.log1p(2))
    assert block[0, 5] == 1.
    assert not block[1:].any()
