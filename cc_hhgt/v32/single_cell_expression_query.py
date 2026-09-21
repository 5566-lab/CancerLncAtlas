'''Immutable binding and read-only query for the 33-cancer lncRNA expression audit.'''
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ANALYSIS_VERSION = 'CancerLncAtlas_V3.2_FULL_MULTITASK'
BINDING_FORMAT = 'CC_HHGT_V3_2_SINGLE_CELL_EXPRESSION_AUDIT_BINDING_V1'
AUDIT_TABLE_SHA256 = '61f48ba6de3436877c90fe8ea0663a6c0477fdae24ac7c0f8e2f228b115bec0b'
AUDIT_SUMMARY_SHA256 = 'd3007fa69091b1db42e489b3171c578c59c178e66f66e0ef5150b1678f2df56e'
FORMAL_GATE_SHA256 = 'cbf88cfabaf9b7719cd3514e1896c34c695f5861c40143ed747089d4ec18ec00'
GENCODE_V50_SHA256 = '88ef3d23a2d6cf3422e188a4d5506ba5386b2c05283a99c916794b5cf9f2b6bf'
MAX_QUERY_LIMIT = 33
MAX_QUERY_OFFSET = 32
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_EXPECTED_CANCERS = {
    'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM',
    'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
    'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD',
    'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM',
}
_REQUIRED_COLUMNS = {
    'cancer_id',
    'source_tier',
    'measurement_scale',
    'cell_count',
    'sample_count',
    'donor_count',
    'feature_count',
    'canonical_lnc_feature_unique',
    'canonical_lnc_detected_ge1_cell',
    'canonical_lnc_detected_ge10_cells',
    'one_percent_cell_threshold',
    'canonical_lnc_detected_ge1pct_cells',
    'matrix_data_dtype',
    'stored_nnz',
    'h5_size_bytes',
    'h5_mtime_utc',
    'input_h5_path',
    'quality_flags',
}


class SingleCellExpressionQueryError(RuntimeError):
    '''Base single-cell expression audit query error.'''


class SingleCellExpressionAssetError(SingleCellExpressionQueryError):
    '''Raised when a binding or source artifact is stale or invalid.'''


class SingleCellExpressionInputError(SingleCellExpressionQueryError):
    '''Raised when query filters are invalid.'''


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise SingleCellExpressionAssetError(f'{label} is missing or invalid JSON') from exc
    if not isinstance(value, dict):
        raise SingleCellExpressionAssetError(f'{label} must be a JSON object')
    return value


def _safe_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise SingleCellExpressionAssetError(f'{label} is missing or unsafe: {resolved}')
    return resolved


def _read_table(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep='\t')
    except Exception as exc:
        raise SingleCellExpressionAssetError('Expression audit table is unreadable') from exc
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise SingleCellExpressionAssetError(f'Expression audit table lacks columns: {missing}')
    if len(frame) != 33 or frame['cancer_id'].nunique(dropna=False) != 33:
        raise SingleCellExpressionAssetError('Expression audit table must have 33 unique cancers')
    observed = set(frame['cancer_id'].astype(str))
    if observed != _EXPECTED_CANCERS:
        raise SingleCellExpressionAssetError(
            f'Expression audit cancer universe mismatch: {sorted(_EXPECTED_CANCERS ^ observed)}'
        )
    numeric = [
        'cell_count',
        'feature_count',
        'canonical_lnc_feature_unique',
        'canonical_lnc_detected_ge1_cell',
        'canonical_lnc_detected_ge10_cells',
        'one_percent_cell_threshold',
        'canonical_lnc_detected_ge1pct_cells',
        'stored_nnz',
        'h5_size_bytes',
    ]
    for column in numeric:
        values = pd.to_numeric(frame[column], errors='coerce')
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise SingleCellExpressionAssetError(f'Expression audit has invalid {column}')
        frame[column] = values.astype('int64')
    if (
        (frame['canonical_lnc_detected_ge1_cell'] > frame['canonical_lnc_feature_unique']).any()
        or (
            frame['canonical_lnc_detected_ge10_cells']
            > frame['canonical_lnc_detected_ge1_cell']
        ).any()
        or (
            frame['canonical_lnc_detected_ge1pct_cells']
            > frame['canonical_lnc_detected_ge1_cell']
        ).any()
        or (
            frame['one_percent_cell_threshold']
            != np.ceil(frame['cell_count'] * 0.01).astype('int64')
        ).any()
    ):
        raise SingleCellExpressionAssetError('Expression audit count invariants failed')
    return frame


def _validate_sources(
    summary_path: Path,
    table_path: Path,
    gate_path: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    digests = {
        'audit_summary': artifact_sha256(summary_path),
        'expression_table': artifact_sha256(table_path),
        'formal_input_gate': artifact_sha256(gate_path),
    }
    expected = {
        'audit_summary': AUDIT_SUMMARY_SHA256,
        'expression_table': AUDIT_TABLE_SHA256,
        'formal_input_gate': FORMAL_GATE_SHA256,
    }
    if digests != expected:
        raise SingleCellExpressionAssetError(
            f'Single-cell expression source SHA mismatch: {digests}'
        )
    summary = _load_json(summary_path, 'Expression audit summary')
    required_summary = {
        'audit_type': 'READ_ONLY_SINGLE_CELL_LNCRNA_EXPRESSION_AUDIT_NOT_TRAINING',
        'status': 'SUCCESS_33_CANCERS_CORE_COUNTS',
        'cancer_count': 33,
        'unique_cancer_count': 33,
        'forbidden_inputs_used': False,
    }
    for key, value in required_summary.items():
        if summary.get(key) != value:
            raise SingleCellExpressionAssetError(
                f'Expression audit summary has invalid {key}: {summary.get(key)!r}'
            )
    annotation = summary.get('annotation')
    if (
        not isinstance(annotation, dict)
        or annotation.get('release') != 'GENCODE v50'
        or annotation.get('sha256') != GENCODE_V50_SHA256
        or annotation.get('strict_primary_lnc_unique_stable_gene_ids') != 34866
    ):
        raise SingleCellExpressionAssetError('Expression audit annotation is not formal GENCODE v50')
    table_decl = summary.get('artifacts', {}).get('per_cancer_tsv', {})
    if table_decl.get('sha256') != AUDIT_TABLE_SHA256:
        raise SingleCellExpressionAssetError('Expression audit summary does not bind the table')
    gate = _load_json(gate_path, 'Formal 33-cancer input gate')
    required_gate = {
        'format': 'CC_HHGT_V3_2_SINGLE_CELL_FORMAL_33C_INPUT_GATE_V2',
        'analysis_version': ANALYSIS_VERSION,
        'control_semantics_correction_only': True,
        'blocked_fresh_direct_id_counts_typed_unavailable': True,
        'fresh_direct_id_counts_authoritative_for_built_partitions': True,
        'expression_audit_is_not_training_completion': True,
        'formal_33c_training_ready': False,
        'single_cell_module_complete': False,
        'training_started': False,
        'historical_sc_trajectory_allowed': False,
        'release_ready': False,
        'production_deployed': False,
    }
    for key, value in required_gate.items():
        if gate.get(key) != value:
            raise SingleCellExpressionAssetError(
                f'Formal single-cell gate has invalid {key}: {gate.get(key)!r}'
            )
    formal = set(gate.get('formal_eligible_cancers', []))
    blocked = set(gate.get('blocked_cancers', []))
    limited = set(gate.get('quality_limited_cancers', []))
    if (
        len(formal) != 17
        or len(blocked) != 10
        or len(limited) != 3
        or not (formal | blocked | limited).issubset(_EXPECTED_CANCERS)
        or formal & blocked
    ):
        raise SingleCellExpressionAssetError('Formal single-cell gate cancer sets are invalid')
    return summary, _read_table(table_path), gate


def build_single_cell_expression_binding(
    *,
    audit_summary_path: str | Path,
    expression_table_path: str | Path,
    formal_gate_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    '''Validate immutable source facts and create a non-release audit binding.'''
    summary_path = _safe_file(audit_summary_path, 'Expression audit summary')
    table_path = _safe_file(expression_table_path, 'Expression audit table')
    gate_path = _safe_file(formal_gate_path, 'Formal 33-cancer input gate')
    summary, frame, gate = _validate_sources(summary_path, table_path, gate_path)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    binding_path = destination / 'SINGLE_CELL_EXPRESSION_BINDING.json'
    success_path = destination / 'SUCCESS.json'
    if binding_path.exists() or success_path.exists():
        raise SingleCellExpressionAssetError(
            f'Refusing to overwrite existing single-cell binding: {destination}'
        )
    binding = {
        'format': BINDING_FORMAT,
        'analysis_version': ANALYSIS_VERSION,
        'status': 'SUCCESS_33_CANCER_EXPRESSION_AUDIT_HASH_BOUND',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'result_role': 'EXPRESSION_AUDIT_FACTS_NOT_TRAINED_SINGLE_CELL_PREDICTIONS',
        'cancer_count': 33,
        'cell_count': int(frame['cell_count'].sum()),
        'single_cell_module_complete': False,
        'training_started': False,
        'pathway_predictions_included': False,
        'individual_lncrna_expression_rows_included': False,
        'historical_predictions_used': False,
        'historical_rankings_used': False,
        'historical_checkpoints_used': False,
        'historical_sc_trajectory_used': False,
        'release_ready': False,
        'production_deployed': False,
        'annotation': {
            'release': summary['annotation']['release'],
            'sha256': summary['annotation']['sha256'],
            'strict_primary_lnc_unique_stable_gene_ids': 34866,
        },
        'input_gate': {
            'run_id': gate['run_id'],
            'formal_eligible_count': len(gate['formal_eligible_cancers']),
            'blocked_count': len(gate['blocked_cancers']),
            'quality_limited_count': len(gate['quality_limited_cancers']),
        },
        'artifacts': {
            'audit_summary': {
                'path': str(summary_path),
                'sha256': AUDIT_SUMMARY_SHA256,
            },
            'expression_table': {
                'path': str(table_path),
                'sha256': AUDIT_TABLE_SHA256,
                'rows': 33,
            },
            'formal_input_gate': {
                'path': str(gate_path),
                'sha256': FORMAL_GATE_SHA256,
            },
        },
    }
    binding_path.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    digest = artifact_sha256(binding_path)
    success = {
        'status': binding['status'],
        'binding': binding_path.name,
        'binding_sha256': digest,
        'release_ready': False,
        'production_deployed': False,
    }
    success_path.write_text(
        json.dumps(success, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return {
        'binding_path': str(binding_path),
        'binding_sha256': digest,
        'success_path': str(success_path),
        'binding': binding,
    }


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


class SingleCellExpressionAuditQuery:
    '''Hash-pinned view of per-cancer expression-detection counts.'''

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
    ) -> None:
        source = _safe_file(binding_path, 'Single-cell expression binding')
        expected = str(expected_binding_sha256 or '').lower()
        if not _SHA256.fullmatch(expected):
            raise SingleCellExpressionAssetError(
                'Single-cell expression query requires an expected binding SHA256'
            )
        observed = artifact_sha256(source)
        if observed != expected:
            raise SingleCellExpressionAssetError(
                f'Single-cell expression binding SHA mismatch: {observed} != {expected}'
            )
        binding = _load_json(source, 'Single-cell expression binding')
        required = {
            'format': BINDING_FORMAT,
            'analysis_version': ANALYSIS_VERSION,
            'status': 'SUCCESS_33_CANCER_EXPRESSION_AUDIT_HASH_BOUND',
            'result_role': 'EXPRESSION_AUDIT_FACTS_NOT_TRAINED_SINGLE_CELL_PREDICTIONS',
            'cancer_count': 33,
            'single_cell_module_complete': False,
            'training_started': False,
            'pathway_predictions_included': False,
            'individual_lncrna_expression_rows_included': False,
            'historical_predictions_used': False,
            'historical_rankings_used': False,
            'historical_checkpoints_used': False,
            'historical_sc_trajectory_used': False,
            'release_ready': False,
            'production_deployed': False,
        }
        for key, value in required.items():
            if binding.get(key) != value:
                raise SingleCellExpressionAssetError(
                    f'Single-cell expression binding has invalid {key}: {binding.get(key)!r}'
                )
        success = _load_json(source.parent / 'SUCCESS.json', 'Single-cell binding success marker')
        if (
            success.get('status') != binding['status']
            or success.get('binding') != source.name
            or success.get('binding_sha256') != observed
            or success.get('release_ready') is not False
            or success.get('production_deployed') is not False
        ):
            raise SingleCellExpressionAssetError('Single-cell binding success marker is stale')
        artifacts = binding.get('artifacts')
        if not isinstance(artifacts, dict):
            raise SingleCellExpressionAssetError('Single-cell binding lacks artifacts')
        paths: dict[str, Path] = {}
        for role, expected_sha in {
            'audit_summary': AUDIT_SUMMARY_SHA256,
            'expression_table': AUDIT_TABLE_SHA256,
            'formal_input_gate': FORMAL_GATE_SHA256,
        }.items():
            declaration = artifacts.get(role)
            if (
                not isinstance(declaration, dict)
                or declaration.get('sha256') != expected_sha
            ):
                raise SingleCellExpressionAssetError(
                    f'Single-cell binding has invalid artifact declaration: {role}'
                )
            path = _safe_file(declaration.get('path', ''), role)
            if artifact_sha256(path) != expected_sha:
                raise SingleCellExpressionAssetError(f'Single-cell artifact SHA drift: {role}')
            paths[role] = path
        summary, frame, gate = _validate_sources(
            paths['audit_summary'],
            paths['expression_table'],
            paths['formal_input_gate'],
        )
        formal = set(gate['formal_eligible_cancers'])
        blocked = set(gate['blocked_cancers'])
        limited = set(gate['quality_limited_cancers'])
        frame = frame.copy()
        frame['formal_input_status'] = frame['cancer_id'].map(
            lambda cancer: (
                'FORMAL_ELIGIBLE'
                if cancer in formal
                else 'BLOCKED_MISSING_METADATA'
                if cancer in blocked
                else 'QUALITY_LIMITED_FEATURE_UNIVERSE'
                if cancer in limited
                else 'BUILT_NOT_FORMAL_OTHER_GATE'
            )
        )
        frame['formal_input_eligible'] = frame['cancer_id'].isin(formal)
        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.summary = summary
        self.gate = gate
        self.frame = frame

    def query_summary(
        self,
        *,
        cancer_id: Any | None = None,
        formal_input_eligible: bool | None = None,
        limit: int = 33,
        offset: int = 0,
    ) -> dict[str, Any]:
        if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
            raise SingleCellExpressionInputError(f'limit must be 1..{MAX_QUERY_LIMIT}')
        if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
            raise SingleCellExpressionInputError(f'offset must be 0..{MAX_QUERY_OFFSET}')
        cancer = None
        result = self.frame
        if cancer_id is not None:
            cancer = str(cancer_id).strip().upper()
            if cancer not in _EXPECTED_CANCERS:
                raise SingleCellExpressionInputError('cancer_id is not in the 33-cancer universe')
            result = result[result['cancer_id'] == cancer]
        if formal_input_eligible is not None:
            if not isinstance(formal_input_eligible, bool):
                raise SingleCellExpressionInputError('formal_input_eligible must be boolean')
            result = result[result['formal_input_eligible'] == formal_input_eligible]
        result = result.sort_values('cancer_id').iloc[int(offset): int(offset) + int(limit)]
        rows = [
            {str(key): _json_value(value) for key, value in row.items()}
            for row in result.to_dict('records')
        ]
        return {
            'module': 'single_cell',
            'query_kind': 'lncrna_expression_detection_summary',
            'result_role': self.binding['result_role'],
            'detection_definitions': {
                'detected_ge1_cell': 'canonical lncRNA has non-zero expression in at least 1 cell',
                'detected_ge10_cells': 'canonical lncRNA has non-zero expression in at least 10 cells',
                'detected_ge1pct_cells': (
                    'canonical lncRNA has non-zero expression in at least ceil(1% of cells)'
                ),
            },
            'single_cell_module_complete': False,
            'training_started': False,
            'pathway_predictions_included': False,
            'individual_lncrna_expression_rows_included': False,
            'filters': {
                'cancer_id': cancer,
                'formal_input_eligible': formal_input_eligible,
            },
            'limit': int(limit),
            'offset': int(offset),
            'returned_rows': len(rows),
            'rows': rows,
            'provenance': {
                'analysis_version': ANALYSIS_VERSION,
                'binding_path': str(self.binding_path),
                'binding_sha256': self.binding_sha256,
                'annotation_release': 'GENCODE v50',
                'annotation_sha256': GENCODE_V50_SHA256,
                'historical_results_used': False,
            },
        }


__all__ = [
    'AUDIT_SUMMARY_SHA256',
    'AUDIT_TABLE_SHA256',
    'BINDING_FORMAT',
    'FORMAL_GATE_SHA256',
    'MAX_QUERY_LIMIT',
    'MAX_QUERY_OFFSET',
    'SingleCellExpressionAssetError',
    'SingleCellExpressionAuditQuery',
    'SingleCellExpressionInputError',
    'SingleCellExpressionQueryError',
    'build_single_cell_expression_binding',
]
