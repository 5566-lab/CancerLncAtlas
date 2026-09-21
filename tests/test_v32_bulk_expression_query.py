from __future__ import annotations

import json
from pathlib import Path

import pytest

from cc_hhgt.v32.bulk_expression_query import (
    BulkExpressionQueryAssetError,
    BulkExpressionQueryInputError,
    BulkExpressionReleaseQuery,
)
from cc_hhgt.v32.bulk_expression_release import artifact_sha256


REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / 'artifacts' / 'v32_bulk_expression_release_20260826_r2_nullable_counts'
BINDING = RELEASE / 'BULK_EXPRESSION_BINDING.json'


@pytest.fixture(scope='module')
def query() -> BulkExpressionReleaseQuery:
    success = json.loads((RELEASE / 'SUCCESS.json').read_text(encoding='utf-8'))
    return BulkExpressionReleaseQuery(
        BINDING,
        expected_binding_sha256=success['binding_sha256'],
    )


def test_formal_release_has_full_16889_by_33_coverage(query) -> None:
    result = query.query_lncrna(lncrna_id='ENSG00000099869')
    assert result['returned_rows'] == 33
    assert result['fresh_computation_from_source_measurements'] is True
    assert result['provenance']['historical_derived_outputs_used'] is False
    assert {row['cancer_id'] for row in result['rows']} == {
        'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM',
        'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
        'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD',
        'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM',
    }


def test_cancer_coverage_is_current_exact_candidate_scope(query) -> None:
    result = query.query_cancer_coverage(cancer_id='acc')
    assert result['returned_rows'] == 1
    row = result['rows'][0]
    assert row['cancer_id'] == 'ACC'
    assert row['annotation_lncrna_count'] == 16_889
    assert row['model_eligible_lncrna_count'] == 2_029
    assert row['tumor_sample_count'] == 79


def test_unavailable_expression_summary_is_null_with_reason(query) -> None:
    result = query.query_lncrna(
        lncrna_id='LNC:ENSG00000083622',
        cancer_id='ACC',
    )
    assert result['returned_rows'] == 1
    row = result['rows'][0]
    assert row['availability'] is False
    assert row['n_samples'] is None
    assert row['median_logcpm'] is None
    assert row['failure_reason'] == (
        'BELOW_WITHIN_CANCER_LOGCPM_GT0_DETECTION_RATE_0_10'
    )
    assert row['detection_rate_logcpm_gt0'] == 0.0


def test_binding_hash_and_freshness_semantics_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(BulkExpressionQueryAssetError, match='SHA mismatch'):
        BulkExpressionReleaseQuery(
            BINDING,
            expected_binding_sha256='0' * 64,
        )
    binding = json.loads(BINDING.read_text(encoding='utf-8'))
    binding['historical_predictions_used'] = True
    copied_binding = tmp_path / BINDING.name
    copied_binding.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    copied_success = json.loads((RELEASE / 'SUCCESS.json').read_text(encoding='utf-8'))
    copied_success['binding_sha256'] = artifact_sha256(copied_binding)
    (tmp_path / 'SUCCESS.json').write_text(
        json.dumps(copied_success, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    with pytest.raises(BulkExpressionQueryAssetError, match='historical_predictions_used'):
        BulkExpressionReleaseQuery(
            copied_binding,
            expected_binding_sha256=artifact_sha256(copied_binding),
        )


def test_query_input_bounds_fail_closed(query) -> None:
    with pytest.raises(BulkExpressionQueryInputError, match='lncrna_id'):
        query.query_lncrna(lncrna_id='')
    with pytest.raises(BulkExpressionQueryInputError, match='limit'):
        query.query_lncrna(lncrna_id='ENSG00000099869', limit=34)
    with pytest.raises(BulkExpressionQueryInputError, match='offset'):
        query.query_cancer_coverage(offset=33)

