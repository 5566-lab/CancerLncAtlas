'''Hash-pinned read-only queries for fresh V3.2 bulk expression facts.'''
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

if __package__:
    from .bulk_expression_release import (
        ANALYSIS_VERSION,
        BINDING_FORMAT,
        DETECTION_GRID_SHA256,
        FORMAL_CANDIDATE_SHA256,
        FORMAL_CANCERS,
        FORMAL_ELIGIBLE_PAIRS,
        FORMAL_GRID_ROWS,
        FORMAL_LNCRNAS,
        THRESHOLD_SUCCESS_SHA256,
        artifact_sha256,
    )
else:
    from bulk_expression_release import (  # type: ignore
        ANALYSIS_VERSION,
        BINDING_FORMAT,
        DETECTION_GRID_SHA256,
        FORMAL_CANDIDATE_SHA256,
        FORMAL_CANCERS,
        FORMAL_ELIGIBLE_PAIRS,
        FORMAL_GRID_ROWS,
        FORMAL_LNCRNAS,
        THRESHOLD_SUCCESS_SHA256,
        artifact_sha256,
    )


MAX_QUERY_LIMIT = 33
MAX_QUERY_OFFSET = 32
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


class BulkExpressionQueryError(RuntimeError):
    '''Base bulk expression query error.'''


class BulkExpressionQueryAssetError(BulkExpressionQueryError):
    '''Raised when a binding or output artifact is stale.'''


class BulkExpressionQueryInputError(BulkExpressionQueryError):
    '''Raised when query filters are invalid.'''


def _safe_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise BulkExpressionQueryAssetError(f'{label} is missing or unsafe: {resolved}')
    return resolved


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise BulkExpressionQueryAssetError(f'{label} is missing or invalid JSON') from exc
    if not isinstance(value, dict):
        raise BulkExpressionQueryAssetError(f'{label} must be a JSON object')
    return value


def _sql_path(path: Path) -> str:
    quote = chr(39)
    return quote + path.resolve().as_posix().replace(quote, quote * 2) + quote


def _canonical_lnc(value: Any) -> str:
    text = str(value or '').strip().upper()
    text = re.sub(r'^(?:LNC|LNCRNA):', '', text)
    text = re.sub(r'\.\d+$', '', text)
    if not text or len(text) > 128 or any(ord(character) < 32 for character in text):
        raise BulkExpressionQueryInputError('lncrna_id is required and must be valid')
    return 'LNC:' + text


def _bounds(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= MAX_QUERY_LIMIT:
        raise BulkExpressionQueryInputError(f'limit must be 1..{MAX_QUERY_LIMIT}')
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise BulkExpressionQueryInputError(f'offset must be 0..{MAX_QUERY_OFFSET}')
    return int(limit), int(offset)


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


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _json_value(value) for key, value in row.items()}
        for row in frame.to_dict('records')
    ]


class BulkExpressionReleaseQuery:
    '''Validated immutable view of the current V3.2 bulk expression release.'''

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
        require_formal_authority: bool = True,
    ) -> None:
        source = _safe_file(binding_path, 'Bulk expression binding')
        expected = str(expected_binding_sha256 or '').lower()
        if not _SHA256.fullmatch(expected):
            raise BulkExpressionQueryAssetError(
                'Bulk expression query requires an expected binding SHA256'
            )
        observed = artifact_sha256(source)
        if observed != expected:
            raise BulkExpressionQueryAssetError(
                f'Bulk expression binding SHA mismatch: {observed} != {expected}'
            )
        binding = _load_json(source, 'Bulk expression binding')
        required = {
            'format': BINDING_FORMAT,
            'analysis_version': ANALYSIS_VERSION,
            'status': 'SUCCESS_FRESH_V32_BULK_EXPRESSION_HASH_BOUND',
            'result_role': 'BULK_EXPRESSION_LANDSCAPE_AND_MODEL_COVERAGE',
            'training_not_applicable_statistical_summary': True,
            'fresh_computation_from_source_measurements': True,
            'historical_derived_outputs_used': False,
            'historical_predictions_used': False,
            'historical_rankings_used': False,
            'historical_checkpoints_used': False,
            'all_output_rows_generated_current_run': True,
            'family_to_exact_broadcast': False,
            'release_ready': False,
            'production_deployed': False,
        }
        for key, value in required.items():
            if binding.get(key) != value:
                raise BulkExpressionQueryAssetError(
                    f'Bulk expression binding has invalid {key}: {binding.get(key)!r}'
                )
        if require_formal_authority and binding.get('formal_authority') is not True:
            raise BulkExpressionQueryAssetError('Bulk expression binding is not formal authority')
        run_id = str(binding.get('computation_run_id', ''))
        if not run_id.startswith('V32-BULK-EXPRESSION-'):
            raise BulkExpressionQueryAssetError('Bulk expression binding lacks current run_id')
        counts = binding.get('counts')
        if not isinstance(counts, dict):
            raise BulkExpressionQueryAssetError('Bulk expression binding lacks counts')
        if require_formal_authority and counts != {
            'summary_rows': FORMAL_GRID_ROWS,
            'cancer_rows': FORMAL_CANCERS,
            'eligible_pairs': FORMAL_ELIGIBLE_PAIRS,
            'annotation_lncrnas': FORMAL_LNCRNAS,
            'cancers': FORMAL_CANCERS,
        }:
            raise BulkExpressionQueryAssetError('Bulk expression formal counts are invalid')
        authorities = binding.get('authorities')
        if not isinstance(authorities, dict):
            raise BulkExpressionQueryAssetError('Bulk expression binding lacks authorities')
        if require_formal_authority:
            expected_authorities = {
                'detection_grid': DETECTION_GRID_SHA256,
                'threshold_success': THRESHOLD_SUCCESS_SHA256,
                'exact_candidates': FORMAL_CANDIDATE_SHA256,
            }
            for role, digest in expected_authorities.items():
                declaration = authorities.get(role)
                if not isinstance(declaration, dict) or declaration.get('sha256') != digest:
                    raise BulkExpressionQueryAssetError(
                        f'Bulk expression {role} is not formal authority'
                    )
        success = _load_json(source.parent / 'SUCCESS.json', 'Bulk expression SUCCESS')
        if (
            success.get('status') != binding['status']
            or success.get('binding') != source.name
            or success.get('binding_sha256') != observed
            or success.get('computation_run_id') != run_id
            or success.get('release_ready') is not False
            or success.get('production_deployed') is not False
        ):
            raise BulkExpressionQueryAssetError('Bulk expression SUCCESS marker is stale')
        artifacts = binding.get('artifacts')
        if not isinstance(artifacts, dict):
            raise BulkExpressionQueryAssetError('Bulk expression binding lacks artifacts')
        paths: dict[str, Path] = {}
        for role in ('expression_summary', 'cancer_coverage', 'source_inputs', 'module_lineage'):
            declaration = artifacts.get(role)
            if (
                not isinstance(declaration, dict)
                or not _SHA256.fullmatch(str(declaration.get('sha256', '')))
            ):
                raise BulkExpressionQueryAssetError(
                    f'Bulk expression artifact declaration is invalid: {role}'
                )
            path = _safe_file(declaration.get('path', ''), role)
            if artifact_sha256(path) != declaration['sha256']:
                raise BulkExpressionQueryAssetError(
                    f'Bulk expression artifact SHA drift: {role}'
                )
            paths[role] = path
        lineage = _load_json(paths['module_lineage'], 'Bulk expression lineage')
        if (
            lineage.get('status') != 'SUCCESS_FRESH_V32_STATISTICAL_SUMMARY'
            or lineage.get('computation_run_id') != run_id
            or lineage.get('historical_derived_outputs_used') is not False
            or lineage.get('all_output_rows_generated_current_run') is not True
        ):
            raise BulkExpressionQueryAssetError('Bulk expression lineage is invalid')
        self.binding = binding
        self.binding_path = source
        self.binding_sha256 = observed
        self.computation_run_id = run_id
        self.paths = paths
        self.require_formal_authority = require_formal_authority
        self._validate_tables()

    def _connect(self):
        return duckdb.connect(':memory:')

    def _validate_tables(self) -> None:
        summary = f'read_parquet({_sql_path(self.paths["expression_summary"])})'
        coverage = f'read_parquet({_sql_path(self.paths["cancer_coverage"])})'
        required_summary = {
            'cancer_id',
            'lncrna_id',
            'detection_rate_logcpm_gt0',
            'detection_rate_cpm_ge1',
            'detection_rate_tpm_gt0',
            'within_cancer_model_eligible',
            'expression_summary_available',
            'n_measurements',
            'n_samples',
            'n_patients',
            'mean_logcpm',
            'median_logcpm',
            'q25_logcpm',
            'q75_logcpm',
            'variance_logcpm',
            'observed_detection_rate_logcpm_gt0',
            'availability',
            'failure_reason',
            'model_coverage_status',
            'analysis_version',
            'computation_run_id',
        }
        required_coverage = {
            'cancer_id',
            'annotation_lncrna_count',
            'model_eligible_lncrna_count',
            'tumor_sample_count',
            'model_eligible_fraction',
            'analysis_version',
            'computation_run_id',
        }
        con = self._connect()
        try:
            summary_columns = {
                row[0] for row in con.execute(f'DESCRIBE SELECT * FROM {summary}').fetchall()
            }
            coverage_columns = {
                row[0] for row in con.execute(f'DESCRIBE SELECT * FROM {coverage}').fetchall()
            }
            if missing := sorted(required_summary - summary_columns):
                raise BulkExpressionQueryAssetError(
                    f'Bulk expression summary lacks columns: {missing}'
                )
            if missing := sorted(required_coverage - coverage_columns):
                raise BulkExpressionQueryAssetError(
                    f'Bulk expression coverage lacks columns: {missing}'
                )
            audit = con.execute(
                f'''
                SELECT count(*),
                       count(DISTINCT (cancer_id, lncrna_id)),
                       count(DISTINCT cancer_id),
                       count(DISTINCT lncrna_id),
                       count_if(within_cancer_model_eligible),
                       count_if(analysis_version <> ? OR computation_run_id <> ?),
                       count_if(detection_rate_logcpm_gt0 NOT BETWEEN 0 AND 1
                                OR detection_rate_cpm_ge1 NOT BETWEEN 0 AND 1
                                OR detection_rate_tpm_gt0 NOT BETWEEN 0 AND 1),
                       count_if(availability IS DISTINCT FROM expression_summary_available
                                OR availability IS DISTINCT FROM within_cancer_model_eligible),
                       count_if(availability AND
                                (n_measurements IS NULL OR n_samples IS NULL OR n_patients IS NULL
                                 OR mean_logcpm IS NULL OR median_logcpm IS NULL
                                 OR q25_logcpm IS NULL OR q75_logcpm IS NULL
                                 OR variance_logcpm IS NULL
                                 OR observed_detection_rate_logcpm_gt0 IS NULL
                                 OR failure_reason IS NOT NULL)),
                       count_if(NOT availability AND
                                (n_measurements IS NOT NULL OR n_samples IS NOT NULL
                                 OR n_patients IS NOT NULL OR mean_logcpm IS NOT NULL
                                 OR median_logcpm IS NOT NULL OR q25_logcpm IS NOT NULL
                                 OR q75_logcpm IS NOT NULL OR variance_logcpm IS NOT NULL
                                 OR observed_detection_rate_logcpm_gt0 IS NOT NULL
                                 OR failure_reason IS NULL)),
                       max(abs(detection_rate_logcpm_gt0
                               - observed_detection_rate_logcpm_gt0))
                           FILTER (WHERE availability)
                FROM {summary}
                ''',
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            cancer_sizes = con.execute(
                f'''
                SELECT min(n), max(n)
                FROM (
                    SELECT cancer_id, count(DISTINCT lncrna_id) AS n
                    FROM {summary}
                    GROUP BY cancer_id
                )
                '''
            ).fetchone()
            coverage_audit = con.execute(
                f'''
                SELECT count(*), count(DISTINCT cancer_id),
                       sum(model_eligible_lncrna_count),
                       count_if(annotation_lncrna_count <= 0
                                OR model_eligible_lncrna_count < 0
                                OR model_eligible_lncrna_count > annotation_lncrna_count
                                OR tumor_sample_count <= 0
                                OR model_eligible_fraction NOT BETWEEN 0 AND 1
                                OR analysis_version <> ?
                                OR computation_run_id <> ?)
                FROM {coverage}
                ''',
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
        finally:
            con.close()
        expected_summary = int(self.binding['counts']['summary_rows'])
        expected_cancers = int(self.binding['counts']['cancers'])
        expected_lncrnas = int(self.binding['counts']['annotation_lncrnas'])
        expected_eligible = int(self.binding['counts']['eligible_pairs'])
        if (
            int(audit[0]) != expected_summary
            or int(audit[1]) != expected_summary
            or int(audit[2]) != expected_cancers
            or int(audit[3]) != expected_lncrnas
            or int(audit[4]) != expected_eligible
            or any(int(value or 0) for value in audit[5:10])
            or float(audit[10] or 0) > 1e-12
            or int(cancer_sizes[0]) != expected_lncrnas
            or int(cancer_sizes[1]) != expected_lncrnas
            or int(coverage_audit[0]) != expected_cancers
            or int(coverage_audit[1]) != expected_cancers
            or int(coverage_audit[2]) != expected_eligible
            or int(coverage_audit[3] or 0)
        ):
            raise BulkExpressionQueryAssetError(
                'Bulk expression output semantic validation failed'
            )

    def _result(
        self,
        kind: str,
        frame: pd.DataFrame,
        filters: dict[str, Any],
        limit: int,
        offset: int,
    ) -> dict[str, Any]:
        return {
            'module': 'expression_landscape',
            'query_kind': kind,
            'result_role': self.binding['result_role'],
            'training_not_applicable_statistical_summary': True,
            'fresh_computation_from_source_measurements': True,
            'filters': filters,
            'limit': limit,
            'offset': offset,
            'returned_rows': len(frame),
            'rows': _records(frame),
            'provenance': {
                'analysis_version': ANALYSIS_VERSION,
                'computation_run_id': self.computation_run_id,
                'binding_path': str(self.binding_path),
                'binding_sha256': self.binding_sha256,
                'historical_derived_outputs_used': False,
                'release_ready': False,
                'production_deployed': False,
            },
        }

    def query_lncrna(
        self,
        *,
        lncrna_id: Any,
        cancer_id: Any | None = None,
        availability: bool | None = None,
        limit: int = 33,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        lnc = _canonical_lnc(lncrna_id)
        cancer = None
        clauses = ['lncrna_id = ?']
        parameters: list[Any] = [lnc]
        if cancer_id is not None:
            cancer = str(cancer_id).strip().upper()
            if not cancer or len(cancer) > 32:
                raise BulkExpressionQueryInputError('cancer_id is invalid')
            clauses.append('cancer_id = ?')
            parameters.append(cancer)
        if availability is not None:
            if not isinstance(availability, bool):
                raise BulkExpressionQueryInputError('availability must be boolean')
            clauses.append('availability = ?')
            parameters.append(availability)
        relation = f'read_parquet({_sql_path(self.paths["expression_summary"])})'
        parameters.extend([limit, offset])
        con = self._connect()
        try:
            frame = con.execute(
                f'''
                SELECT * FROM {relation}
                WHERE {' AND '.join(clauses)}
                ORDER BY cancer_id
                LIMIT ? OFFSET ?
                ''',
                parameters,
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            'lncrna_expression_and_model_coverage',
            frame,
            {'lncrna_id': lnc, 'cancer_id': cancer, 'availability': availability},
            limit,
            offset,
        )

    def query_cancer_coverage(
        self,
        *,
        cancer_id: Any | None = None,
        limit: int = 33,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset)
        cancer = None
        clauses = []
        parameters: list[Any] = []
        if cancer_id is not None:
            cancer = str(cancer_id).strip().upper()
            if not cancer or len(cancer) > 32:
                raise BulkExpressionQueryInputError('cancer_id is invalid')
            clauses.append('cancer_id = ?')
            parameters.append(cancer)
        relation = f'read_parquet({_sql_path(self.paths["cancer_coverage"])})'
        where = 'WHERE ' + ' AND '.join(clauses) if clauses else ''
        parameters.extend([limit, offset])
        con = self._connect()
        try:
            frame = con.execute(
                f'''
                SELECT * FROM {relation}
                {where}
                ORDER BY cancer_id
                LIMIT ? OFFSET ?
                ''',
                parameters,
            ).fetchdf()
        finally:
            con.close()
        return self._result(
            'cancer_model_coverage',
            frame,
            {'cancer_id': cancer},
            limit,
            offset,
        )


__all__ = [
    'BulkExpressionQueryAssetError',
    'BulkExpressionQueryError',
    'BulkExpressionQueryInputError',
    'BulkExpressionReleaseQuery',
    'MAX_QUERY_LIMIT',
    'MAX_QUERY_OFFSET',
]
