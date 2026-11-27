"""Source readouts retain exact identity and target-aware exclusions."""
import numpy as np
import pandas as pd
import pytest

from cla_engine.fused_adjustment.adjustment_policy import default_adjustment_contract
from cla_engine.relation_features import (compute_readout_block, expand_candidate_readouts,
                                         entity_summary, read_exclusions)
from cla_engine.relation_source import (SourceMatrix, SourceReadouts, SourceLocalCnv,
                                       SourceCovariates)
from cla_engine.common import sha


def source_fixture():
    rng = np.random.default_rng(42)
    patients = tuple(f'T{i}' for i in range(40))
    values = rng.normal(size=(40, 2)).astype(np.float32)
    expression = SourceMatrix('TCGA', patients, ('L1', 'L2'), values,
                               np.ones_like(values, bool), 'expression')
    cnv = SourceLocalCnv('TCGA', patients, expression.entities, np.zeros_like(values),
                         np.ones_like(values, bool), 'cnv')
    frame = pd.DataFrame({'patient_id': patients, 'age_years': rng.uniform(30, 70, 40),
        'sex': ['F', 'M'] * 20, 'stage': ['I', 'II'] * 20,
        'purity': rng.uniform(.4, .9, 40), 'purity_source': 'ABSOLUTE',
        **{'cell_fraction_Macrophages.' + m: rng.uniform(.05, .2, 40) for m in ('M0', 'M1', 'M2')}})
    cov = SourceCovariates('TCGA', patients, frame, 'covariates')
    return expression, cnv, cov


def test_state_adjustment_and_direct_member_exclusions():
    expression, cnv, cov = source_fixture()
    metadata = ({'target_id': 'R1', 'source_target_id': 'TumorPurity', 'family': 'purity'},
                {'target_id': 'R2', 'source_target_id': 'RNAss', 'family': 'stemness_telomere'})
    readouts = SourceReadouts('TCGA', 'STATE', expression.patients, ('R1', 'R2'),
        ('P1', 'P2'), np.column_stack([cov.frame.purity, expression.values[:, 0]]),
        np.ones((40, 2), bool), 'readouts', metadata)
    cfg = {'minimum_pairs': 12, 'adjustment_contract': default_adjustment_contract()}
    li, ti = np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])
    numeric, associations, audit = compute_readout_block(expression, readouts, cnv, cov,
        cfg, li, ti, ('P1', 'P2'), ({('L1', 'RNAss')}, set()))
    assert len(audit['adjustment_groups']) == 2
    assert audit['direct_rna_pairs_excluded'] == 1
    assert numeric[1, -1] == 0
    assert numeric[0, -1] == 1  # Purity self-covariate must not erase this target.
    assert len(associations.rho) == 4
    assert np.isfinite(numeric).all()
    assert np.isfinite(associations.rho).sum() == 3


def test_empty_layer_and_mixed_coding_rna_exclusion():
    expression, cnv, cov = source_fixture()
    cfg = {'minimum_pairs': 12, 'adjustment_contract': default_adjustment_contract()}
    empty = SourceReadouts('TCGA', 'PHOSPHO', expression.patients, (), (),
                           np.empty((40, 0)), np.empty((40, 0), bool), 'index')
    numeric, stats, audit = compute_readout_block(expression, empty, cnv, cov, cfg,
        np.array([0, 1]), np.array([0, 0]), ('P1',), (set(), set()))
    assert numeric.shape == (2, 6) and not numeric.any()
    assert len(stats.rho) == 0 and audit['readouts'] == 0
    readout = SourceReadouts('TCGA', 'RNA', expression.patients, ('R1',), ('P1',),
        expression.values[:, :1], np.ones((40, 1), bool), 'scores',
        ({'target_id': 'R1', 'source_target_id': 'RNA_PATHWAY', 'family': 'pathway'},))
    numeric, stats, audit = compute_readout_block(expression, readout, cnv, cov, cfg,
        np.array([0, 1]), np.array([0, 0]), ('P1',), (set(), {'L1'}))
    assert numeric[:, -1].tolist() == [0., 1.]
    assert stats.available.tolist() == [False, True]


def test_expansion_uses_exact_readout_identity_and_rejects_duplicate_candidates():
    readouts = SourceReadouts('TCGA', 'PROTEIN', ('T1',), ('R1', 'R2', 'R3'),
                              ('P2', 'P1', 'P1'), np.ones((1, 3)), np.ones((1, 3), bool), 'sha')
    li, ri = expand_candidate_readouts(np.array([0, 1]), np.array([0, 1]), ('P1', 'P2'), readouts)
    assert list(zip(li, ri)) == [(1, 0), (0, 1), (0, 2)]
    with pytest.raises(ValueError, match='DUPLICATE_CANDIDATE'):
        expand_candidate_readouts(np.array([0, 0]), np.array([0, 0]), ('P1', 'P2'), readouts)


def test_measurement_summary_and_pinned_exclusion_files(tmp_path):
    values = np.array([[1., np.nan], [3., np.nan]])
    result = entity_summary(values, np.isfinite(values))
    np.testing.assert_allclose(result, [[2, 1, 1], [0, 0, 0]])
    direct, mixed = tmp_path / 'direct.tsv', tmp_path / 'mixed.tsv'
    direct.write_text('state_id\tlncrna_id\nRNAss\tLNC:E1\n')
    mixed.write_text('gene_id\nE2\n')
    d, m = {'path': str(direct), 'sha256': sha(direct)}, {'path': str(mixed), 'sha256': sha(mixed)}
    assert read_exclusions(d, m) == ({('LNC:E1', 'RNAss')}, {'LNC:E2'})
    mixed.write_text('gene_id\nE3\n')
    with pytest.raises(ValueError, match='HASH_MISMATCH'):
        read_exclusions(d, m)
