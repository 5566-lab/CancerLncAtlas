from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from cc_hhgt.v32.release_registry import artifact_sha256
from cc_hhgt.v32.single_cell_expression_query import (
    SingleCellExpressionAssetError,
    SingleCellExpressionAuditQuery,
    SingleCellExpressionInputError,
    build_single_cell_expression_binding,
)


REPO = Path(__file__).resolve().parents[1]
AUDIT_ROOT = REPO / 'artifacts' / 'single_cell_lncrna_expression_audit_20260826'
GATE_ROOT = (
    REPO
    / 'artifacts'
    / 'single_cell_h5_33c_partitions_20260826_r6_semantic_correction'
)


@pytest.fixture
def bound_query(tmp_path: Path):
    sources = tmp_path / 'sources'
    sources.mkdir()
    summary = sources / 'AUDIT_SUMMARY.json'
    table = sources / 'single_cell_lncrna_expression_33c.tsv'
    gate = sources / 'FORMAL_33C_INPUT_GATE.json'
    shutil.copy2(AUDIT_ROOT / summary.name, summary)
    shutil.copy2(AUDIT_ROOT / table.name, table)
    shutil.copy2(GATE_ROOT / gate.name, gate)
    built = build_single_cell_expression_binding(
        audit_summary_path=summary,
        expression_table_path=table,
        formal_gate_path=gate,
        output_dir=tmp_path / 'binding',
    )
    query = SingleCellExpressionAuditQuery(
        built['binding_path'],
        expected_binding_sha256=built['binding_sha256'],
    )
    return query, built, summary, table, gate


def test_materializes_hash_bound_33_cancer_audit_not_training(bound_query) -> None:
    query, built, _, _, _ = bound_query
    assert built['binding']['cancer_count'] == 33
    assert built['binding']['cell_count'] == 1_919_578
    assert built['binding']['single_cell_module_complete'] is False
    assert built['binding']['training_started'] is False
    assert built['binding']['pathway_predictions_included'] is False
    result = query.query_summary()
    assert result['returned_rows'] == 33
    assert result['single_cell_module_complete'] is False
    assert result['individual_lncrna_expression_rows_included'] is False
    rows = {row['cancer_id']: row for row in result['rows']}
    assert rows['ACC']['canonical_lnc_detected_ge1_cell'] == 10_731
    assert rows['UCEC']['canonical_lnc_detected_ge1_cell'] == 14_844
    assert rows['CESC']['formal_input_status'] == 'QUALITY_LIMITED_FEATURE_UNIVERSE'
    assert rows['BLCA']['formal_input_status'] == 'BLOCKED_MISSING_METADATA'
    assert rows['KICH']['formal_input_status'] == 'BUILT_NOT_FORMAL_OTHER_GATE'
    assert rows['HNSC']['formal_input_status'] == 'FORMAL_ELIGIBLE'


def test_filters_and_canonical_detection_definitions(bound_query) -> None:
    query, _, _, _, _ = bound_query
    result = query.query_summary(cancer_id='hnsc')
    assert result['returned_rows'] == 1
    assert result['rows'][0]['cancer_id'] == 'HNSC'
    assert result['rows'][0]['canonical_lnc_detected_ge1_cell'] == 979
    assert 'at least 1 cell' in result['detection_definitions']['detected_ge1_cell']
    formal = query.query_summary(formal_input_eligible=True)
    assert formal['returned_rows'] == 17
    assert all(row['formal_input_eligible'] for row in formal['rows'])


def test_query_requires_exact_binding_hash(bound_query) -> None:
    _, built, _, _, _ = bound_query
    with pytest.raises(SingleCellExpressionAssetError, match='expected binding'):
        SingleCellExpressionAuditQuery(
            built['binding_path'],
            expected_binding_sha256=None,
        )
    with pytest.raises(SingleCellExpressionAssetError, match='SHA mismatch'):
        SingleCellExpressionAuditQuery(
            built['binding_path'],
            expected_binding_sha256='0' * 64,
        )


def test_source_artifact_drift_fails_closed(bound_query) -> None:
    _, built, _, table, _ = bound_query
    table.write_text(table.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(SingleCellExpressionAssetError, match='SHA drift'):
        SingleCellExpressionAuditQuery(
            built['binding_path'],
            expected_binding_sha256=built['binding_sha256'],
        )


def test_stale_success_and_binding_semantics_fail_closed(bound_query) -> None:
    _, built, _, _, _ = bound_query
    binding_path = Path(built['binding_path'])
    success_path = Path(built['success_path'])
    success = json.loads(success_path.read_text(encoding='utf-8'))
    success['binding_sha256'] = 'f' * 64
    success_path.write_text(json.dumps(success), encoding='utf-8')
    with pytest.raises(SingleCellExpressionAssetError, match='success marker'):
        SingleCellExpressionAuditQuery(
            binding_path,
            expected_binding_sha256=artifact_sha256(binding_path),
        )


def test_builder_refuses_overwrite_and_query_bounds(bound_query) -> None:
    query, built, summary, table, gate = bound_query
    with pytest.raises(SingleCellExpressionAssetError, match='overwrite'):
        build_single_cell_expression_binding(
            audit_summary_path=summary,
            expression_table_path=table,
            formal_gate_path=gate,
            output_dir=Path(built['binding_path']).parent,
        )
    with pytest.raises(SingleCellExpressionInputError, match='cancer_id'):
        query.query_summary(cancer_id='NOT_A_CANCER')
    with pytest.raises(SingleCellExpressionInputError, match='limit'):
        query.query_summary(limit=34)
    with pytest.raises(SingleCellExpressionInputError, match='offset'):
        query.query_summary(offset=33)

