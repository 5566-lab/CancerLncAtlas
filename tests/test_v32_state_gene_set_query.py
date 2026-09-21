from __future__ import annotations

import json
from pathlib import Path

import pytest

from cc_hhgt.v32.state_gene_set_query import (
    StateGeneSetQueryAssetError,
    StateGeneSetQueryInputError,
    StateGeneSetReleaseQuery,
)
from cc_hhgt.v32.state_gene_set_release import STATE_IDS, artifact_sha256


REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / 'artifacts' / 'v32_state_gene_sets_20260826_r2_code_bound'
BINDING = RELEASE / 'STATE_GENE_SET_BINDING.json'


@pytest.fixture(scope='module')
def query() -> StateGeneSetReleaseQuery:
    success = json.loads((RELEASE / 'SUCCESS.json').read_text(encoding='utf-8'))
    return StateGeneSetReleaseQuery(
        BINDING,
        expected_binding_sha256=success['binding_sha256'],
    )


def test_formal_state_gene_sets_cover_all_seven_fresh_states(query) -> None:
    result = query.query_gene_sets(limit=500)
    assert result['total_rows'] == 452
    assert result['returned_rows'] == 452
    assert {row['state_id'] for row in result['rows']} == set(STATE_IDS)
    assert result['provenance']['source_training_run_id'] == 'v32-state-fresh-20260825'
    assert result['provenance']['historical_checkpoints_used'] is False
    assert result['provenance']['historical_predictions_used'] is False
    assert result['provenance']['historical_rankings_used'] is False
    assert result['provenance']['historical_gene_sets_used'] is False
    assert result['provenance']['changes_primary_ranking'] is False


def test_rnass_pan_cancer_set_has_ranked_current_members(query) -> None:
    result = query.query_gene_sets(
        state_id='stemness_rna::RNAss',
        scope='pan_cancer',
        direction='positive',
    )
    assert result['total_rows'] == 1
    row = result['rows'][0]
    assert row['gene_set_id'] == 'STATE::PAN_CANCER::stemness_rna::RNAss::positive'
    assert row['member_count'] == 200
    members = query.query_members(gene_set_id=row['gene_set_id'], limit=200)
    assert members['total_rows'] == 200
    assert [item['member_rank'] for item in members['rows']] == list(range(1, 201))
    assert all(item['member_id'].startswith('LNC:') for item in members['rows'])
    assert all(not item['member_gene_id'].startswith('LNC:') for item in members['rows'])
    assert all(item['cancer_count'] >= 5 for item in members['rows'])
    assert all(item['direction_consistency'] >= 0.60 for item in members['rows'])


def test_cancer_specific_filter_and_gmt_report_are_present(query) -> None:
    result = query.query_gene_sets(
        state_id='stemness_dna::DNAss',
        cancer_id='acc',
        scope='cancer_specific',
    )
    assert result['total_rows'] >= 1
    assert all(row['cancer_id'] == 'ACC' for row in result['rows'])
    assert all(row['scope'] == 'CANCER_SPECIFIC' for row in result['rows'])
    report = (RELEASE / 'STATE_GENE_SET_REPORT.md').read_text(encoding='utf-8')
    gmt = (RELEASE / 'state_gene_sets.gmt').read_text(encoding='utf-8')
    for state_id in STATE_IDS:
        assert state_id in report
        assert state_id in gmt


def test_binding_freshness_and_hash_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(StateGeneSetQueryAssetError, match='SHA mismatch'):
        StateGeneSetReleaseQuery(BINDING, expected_binding_sha256='0' * 64)
    binding = json.loads(BINDING.read_text(encoding='utf-8'))
    binding['historical_gene_sets_used'] = True
    copied_binding = tmp_path / BINDING.name
    copied_binding.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    success = json.loads((RELEASE / 'SUCCESS.json').read_text(encoding='utf-8'))
    success['binding_sha256'] = artifact_sha256(copied_binding)
    (tmp_path / 'SUCCESS.json').write_text(
        json.dumps(success, indent=2, sort_keys=True) + '\n', encoding='utf-8'
    )
    with pytest.raises(StateGeneSetQueryAssetError, match='historical_gene_sets_used'):
        StateGeneSetReleaseQuery(
            copied_binding,
            expected_binding_sha256=artifact_sha256(copied_binding),
        )


def test_state_gene_set_query_inputs_fail_closed(query) -> None:
    with pytest.raises(StateGeneSetQueryInputError, match='Unknown state_id'):
        query.query_gene_sets(state_id='not-a-state')
    with pytest.raises(StateGeneSetQueryInputError, match='scope'):
        query.query_gene_sets(scope='legacy')
    with pytest.raises(StateGeneSetQueryInputError, match='limit'):
        query.query_gene_sets(limit=501)
    with pytest.raises(StateGeneSetQueryInputError, match='gene_set_id'):
        query.query_members(gene_set_id='')
