'''Fresh V3.2 bulk lncRNA expression landscape from audited GDC inputs.'''
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ANALYSIS_VERSION = 'CancerLncAtlas_V3.2_FULL_MULTITASK'
BINDING_FORMAT = 'CC_HHGT_V3_2_BULK_EXPRESSION_RELEASE_BINDING_V1'
DETECTION_GRID_SHA256 = '69241435b1326b99c177ac70f9497b7e25310b1eb898555f7e418c9ecce7be52'
THRESHOLD_SUCCESS_SHA256 = '281119263e55bfe3ef9e11643ba965a65cedc062f84737fa463e8084b5db9dc5'
FORMAL_CANDIDATE_SHA256 = 'cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f'
FORMAL_GRID_ROWS = 557_337
FORMAL_CANDIDATE_ROWS = 3_300_000
FORMAL_ELIGIBLE_PAIRS = 76_734
FORMAL_LNCRNAS = 16_889
FORMAL_CANCERS = 33
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_EXPECTED_CANCERS = {
    'ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM',
    'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
    'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD',
    'TGCT', 'THCA', 'THYM', 'UCEC', 'UCS', 'UVM',
}


class BulkExpressionReleaseError(RuntimeError):
    '''Raised when formal bulk-expression materialization fails closed.'''


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise BulkExpressionReleaseError(f'{label} is missing or unsafe: {resolved}')
    return resolved


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise BulkExpressionReleaseError(f'{label} is missing or invalid JSON') from exc
    if not isinstance(value, dict):
        raise BulkExpressionReleaseError(f'{label} must be a JSON object')
    return value


def _sql_path(path: Path) -> str:
    quote = chr(39)
    return quote + path.resolve().as_posix().replace(quote, quote * 2) + quote


def _parquet_list(paths: list[Path]) -> str:
    return '[' + ','.join(_sql_path(path) for path in paths) + ']'


def _validate_detection_grid(
    path: Path,
    success_path: Path,
    *,
    strict_formal_authority: bool,
) -> pd.DataFrame:
    if strict_formal_authority:
        if artifact_sha256(path) != DETECTION_GRID_SHA256:
            raise BulkExpressionReleaseError('Formal detection-grid SHA256 mismatch')
        if artifact_sha256(success_path) != THRESHOLD_SUCCESS_SHA256:
            raise BulkExpressionReleaseError('Formal threshold SUCCESS SHA256 mismatch')
    success = _load_json(success_path, 'Threshold-grid SUCCESS')
    required = {
        'status': 'PASS',
        'cancers': FORMAL_CANCERS,
        'stable_lncrna_union': FORMAL_LNCRNAS,
        'stable_lncrna_per_cancer_min': FORMAL_LNCRNAS,
        'stable_lncrna_per_cancer_max': FORMAL_LNCRNAS,
        'primary_metric': 'exact_model_logcpm_gt0',
    }
    for key, value in required.items():
        if strict_formal_authority and success.get(key) != value:
            raise BulkExpressionReleaseError(
                f'Threshold-grid SUCCESS has invalid {key}: {success.get(key)!r}'
            )
    try:
        grid = pd.read_csv(path, sep='\t')
    except Exception as exc:
        raise BulkExpressionReleaseError('Detection grid is unreadable') from exc
    required_columns = {
        'cancer_id',
        'lncrna_id',
        'detection_rate_logcpm_gt0',
        'detection_rate_cpm_ge1',
        'detection_rate_tpm_gt0',
    }
    if missing := sorted(required_columns - set(grid.columns)):
        raise BulkExpressionReleaseError(f'Detection grid lacks columns: {missing}')
    if (
        grid.duplicated(['cancer_id', 'lncrna_id']).any()
        or grid['cancer_id'].nunique() != (FORMAL_CANCERS if strict_formal_authority else grid['cancer_id'].nunique())
    ):
        raise BulkExpressionReleaseError('Detection grid has duplicate keys or invalid cancers')
    if strict_formal_authority:
        if (
            len(grid) != FORMAL_GRID_ROWS
            or set(grid['cancer_id'].astype(str)) != _EXPECTED_CANCERS
            or not grid.groupby('cancer_id')['lncrna_id'].nunique().eq(FORMAL_LNCRNAS).all()
        ):
            raise BulkExpressionReleaseError('Detection grid does not cover 16,889 x 33')
    for column in sorted(required_columns - {'cancer_id', 'lncrna_id'}):
        values = pd.to_numeric(grid[column], errors='coerce')
        if values.isna().any() or not np.isfinite(values).all() or not values.between(0, 1).all():
            raise BulkExpressionReleaseError(f'Detection grid has invalid {column}')
        grid[column] = values.astype(float)
    return grid


def _validate_expression_partitions(
    expression_root: Path,
    *,
    strict_formal_authority: bool,
) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(expression_root.glob('cancer_id=*/part-0.parquet'))
    expected_count = FORMAL_CANCERS if strict_formal_authority else len(paths)
    if len(paths) != expected_count or not paths:
        raise BulkExpressionReleaseError(
            f'Expected {expected_count} expression partitions, observed {len(paths)}'
        )
    declarations: list[dict[str, Any]] = []
    cancers: set[str] = set()
    for path in paths:
        cancer = path.parent.name.split('=', 1)[-1]
        qc_path = _safe_file(path.parent / 'QC.json', f'{cancer} expression QC')
        qc = _load_json(qc_path, f'{cancer} expression QC')
        if qc.get('cancer_id') != cancer:
            raise BulkExpressionReleaseError(f'{cancer} expression QC cancer mismatch')
        for key in ('rows', 'lncrna_n', 'sample_n'):
            if not isinstance(qc.get(key), int) or int(qc[key]) <= 0:
                raise BulkExpressionReleaseError(f'{cancer} expression QC has invalid {key}')
        cancers.add(cancer)
        declarations.append(
            {
                'cancer_id': cancer,
                'path': str(path.resolve()),
                'sha256': artifact_sha256(path),
                'qc_path': str(qc_path),
                'qc_sha256': artifact_sha256(qc_path),
                'rows': int(qc['rows']),
                'lncrna_n': int(qc['lncrna_n']),
                'sample_n': int(qc['sample_n']),
            }
        )
    if strict_formal_authority and cancers != _EXPECTED_CANCERS:
        raise BulkExpressionReleaseError('Expression partition cancer universe mismatch')
    relation = f'read_parquet({_parquet_list(paths)}, hive_partitioning=false)'
    con = duckdb.connect(':memory:')
    try:
        columns = {
            row[0]
            for row in con.execute(f'DESCRIBE SELECT * FROM {relation}').fetchall()
        }
        required = {'cancer_id', 'sample_id', 'patient_id', 'lncrna_id', 'logcpm'}
        if missing := sorted(required - columns):
            raise BulkExpressionReleaseError(f'Expression partitions lack columns: {missing}')
        audit = con.execute(
            f'''
            SELECT count(*) AS rows,
                   count(DISTINCT (cancer_id, sample_id, lncrna_id)) AS unique_keys,
                   count_if(cancer_id IS NULL OR sample_id IS NULL OR patient_id IS NULL
                            OR lncrna_id IS NULL OR logcpm IS NULL
                            OR NOT isfinite(logcpm)) AS invalid_rows,
                   count(DISTINCT cancer_id) AS cancers
            FROM {relation}
            '''
        ).fetchone()
        observed = con.execute(
            f'''
            SELECT cancer_id, count(*) AS rows,
                   count(DISTINCT lncrna_id) AS lncrna_n,
                   count(DISTINCT sample_id) AS sample_n
            FROM {relation}
            GROUP BY cancer_id
            ORDER BY cancer_id
            '''
        ).fetchdf()
    finally:
        con.close()
    if int(audit[0]) != int(audit[1]) or int(audit[2]) or int(audit[3]) != len(paths):
        raise BulkExpressionReleaseError(f'Expression partition semantic audit failed: {audit}')
    declared = pd.DataFrame(declarations)[['cancer_id', 'rows', 'lncrna_n', 'sample_n']]
    compared = declared.merge(observed, on='cancer_id', suffixes=('_declared', '_observed'))
    for key in ('rows', 'lncrna_n', 'sample_n'):
        if not compared[f'{key}_declared'].eq(compared[f'{key}_observed']).all():
            raise BulkExpressionReleaseError(f'Expression QC mismatch for {key}')
    return paths, declarations


def _validate_candidates(
    path: Path,
    *,
    strict_formal_authority: bool,
) -> pd.DataFrame:
    if strict_formal_authority and artifact_sha256(path) != FORMAL_CANDIDATE_SHA256:
        raise BulkExpressionReleaseError('Formal exact candidate SHA256 mismatch')
    con = duckdb.connect(':memory:')
    try:
        relation = f'read_parquet({_sql_path(path)})'
        columns = {
            row[0]
            for row in con.execute(f'DESCRIBE SELECT * FROM {relation}').fetchall()
        }
        if not {'cancer_id', 'lncrna_id', 'pathway_id'}.issubset(columns):
            raise BulkExpressionReleaseError('Formal candidates lack exact-pathway keys')
        row_count = int(con.execute(f'SELECT count(*) FROM {relation}').fetchone()[0])
        pairs = con.execute(
            f'''
            SELECT DISTINCT cancer_id::VARCHAR AS cancer_id,
                            lncrna_id::VARCHAR AS lncrna_id
            FROM {relation}
            ORDER BY cancer_id, lncrna_id
            '''
        ).fetchdf()
    finally:
        con.close()
    if strict_formal_authority and (
        row_count != FORMAL_CANDIDATE_ROWS
        or len(pairs) != FORMAL_ELIGIBLE_PAIRS
        or set(pairs['cancer_id']) != _EXPECTED_CANCERS
    ):
        raise BulkExpressionReleaseError('Formal exact candidate universe has invalid size')
    return pairs


def materialize_bulk_expression_release(
    *,
    detection_grid_path: str | Path,
    threshold_success_path: str | Path,
    expression_root: str | Path,
    exact_candidate_path: str | Path,
    output_root: str | Path,
    runner_path: str | Path,
    strict_formal_authority: bool = True,
) -> dict[str, Any]:
    '''Create fresh summary/coverage tables without using historical outputs.'''
    grid_path = _safe_file(detection_grid_path, 'Detection grid')
    threshold_path = _safe_file(threshold_success_path, 'Threshold SUCCESS')
    candidate_path = _safe_file(exact_candidate_path, 'Exact candidates')
    code_path = _safe_file(Path(__file__), 'Bulk expression materializer code')
    runner = _safe_file(runner_path, 'Bulk expression materializer runner')
    expression = Path(expression_root).resolve()
    if not expression.is_dir() or expression.is_symlink():
        raise BulkExpressionReleaseError(f'Expression root is missing or unsafe: {expression}')
    destination = Path(output_root).resolve()
    if destination.exists():
        raise BulkExpressionReleaseError(
            f'Refusing to overwrite existing bulk expression release: {destination}'
        )

    grid = _validate_detection_grid(
        grid_path,
        threshold_path,
        strict_formal_authority=strict_formal_authority,
    )
    expression_paths, expression_declarations = _validate_expression_partitions(
        expression,
        strict_formal_authority=strict_formal_authority,
    )
    candidates = _validate_candidates(
        candidate_path,
        strict_formal_authority=strict_formal_authority,
    )
    threshold_pairs = grid.loc[
        grid['detection_rate_logcpm_gt0'].ge(0.10),
        ['cancer_id', 'lncrna_id'],
    ].drop_duplicates()
    pair_audit = threshold_pairs.merge(
        candidates,
        on=['cancer_id', 'lncrna_id'],
        how='outer',
        indicator=True,
    )
    if not pair_audit['_merge'].eq('both').all():
        raise BulkExpressionReleaseError(
            'Exact candidate lncRNA pairs do not equal the formal >=0.10 detection scope'
        )

    relation = (
        f'read_parquet({_parquet_list(expression_paths)}, hive_partitioning=false)'
    )
    con = duckdb.connect(':memory:')
    try:
        summary = con.execute(
            f'''
            SELECT cancer_id::VARCHAR AS cancer_id,
                   lncrna_id::VARCHAR AS lncrna_id,
                   count(*)::BIGINT AS n_measurements,
                   count(DISTINCT sample_id)::BIGINT AS n_samples,
                   count(DISTINCT patient_id)::BIGINT AS n_patients,
                   avg(logcpm)::DOUBLE AS mean_logcpm,
                   median(logcpm)::DOUBLE AS median_logcpm,
                   quantile_cont(logcpm, 0.25)::DOUBLE AS q25_logcpm,
                   quantile_cont(logcpm, 0.75)::DOUBLE AS q75_logcpm,
                   var_samp(logcpm)::DOUBLE AS variance_logcpm,
                   avg(CASE WHEN logcpm > 0 THEN 1.0 ELSE 0.0 END)::DOUBLE
                       AS observed_detection_rate_logcpm_gt0
            FROM {relation}
            GROUP BY cancer_id, lncrna_id
            ORDER BY cancer_id, lncrna_id
            '''
        ).fetchdf()
    finally:
        con.close()
    if summary.duplicated(['cancer_id', 'lncrna_id']).any():
        raise BulkExpressionReleaseError('Fresh expression summary has duplicate keys')
    expression_pairs = summary[['cancer_id', 'lncrna_id']]
    expression_audit = expression_pairs.merge(
        candidates,
        on=['cancer_id', 'lncrna_id'],
        how='outer',
        indicator=True,
    )
    if not expression_audit['_merge'].eq('both').all():
        raise BulkExpressionReleaseError(
            'Formal expression partitions do not equal exact candidate lncRNA scope'
        )

    output = grid.merge(
        summary,
        on=['cancer_id', 'lncrna_id'],
        how='left',
        validate='one_to_one',
    )
    output['within_cancer_model_eligible'] = output[
        'detection_rate_logcpm_gt0'
    ].ge(0.10)
    output['expression_summary_available'] = output['n_samples'].notna()
    if not output['within_cancer_model_eligible'].eq(
        output['expression_summary_available']
    ).all():
        raise BulkExpressionReleaseError(
            'Fresh expression summaries do not match the model-eligible scope'
        )
    eligible = output['within_cancer_model_eligible']
    maximum_rate_delta = float(
        (
            output.loc[eligible, 'detection_rate_logcpm_gt0']
            - output.loc[eligible, 'observed_detection_rate_logcpm_gt0']
        )
        .abs()
        .max()
    )
    if not math.isfinite(maximum_rate_delta) or maximum_rate_delta > 1e-12:
        raise BulkExpressionReleaseError(
            f'Expression summary detection rate drift: {maximum_rate_delta}'
        )
    output['availability'] = output['expression_summary_available']
    for column in ('n_measurements', 'n_samples', 'n_patients'):
        output[column] = pd.to_numeric(output[column], errors='coerce').astype('Int64')
    output['failure_reason'] = np.where(
        output['availability'],
        None,
        'BELOW_WITHIN_CANCER_LOGCPM_GT0_DETECTION_RATE_0_10',
    )
    output['model_coverage_status'] = np.where(
        output['within_cancer_model_eligible'],
        'ELIGIBLE_CURRENT_V32_EXACT_CANDIDATE',
        'NOT_ELIGIBLE_CURRENT_V32_EXACT_CANDIDATE',
    )
    run_id = (
        'V32-BULK-EXPRESSION-'
        + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    )
    output['analysis_version'] = ANALYSIS_VERSION
    output['computation_run_id'] = run_id
    output = output.sort_values(
        ['cancer_id', 'lncrna_id'],
        kind='stable',
    ).reset_index(drop=True)

    cancer_coverage = (
        output.groupby('cancer_id', as_index=False, observed=True)
        .agg(
            annotation_lncrna_count=('lncrna_id', 'nunique'),
            model_eligible_lncrna_count=('within_cancer_model_eligible', 'sum'),
            tumor_sample_count=('n_samples', 'max'),
            median_detection_rate_logcpm_gt0=('detection_rate_logcpm_gt0', 'median'),
            mean_detection_rate_logcpm_gt0=('detection_rate_logcpm_gt0', 'mean'),
            max_detection_rate_logcpm_gt0=('detection_rate_logcpm_gt0', 'max'),
        )
    )
    cancer_coverage['model_eligible_lncrna_count'] = cancer_coverage[
        'model_eligible_lncrna_count'
    ].astype('int64')
    cancer_coverage['annotation_lncrna_count'] = cancer_coverage[
        'annotation_lncrna_count'
    ].astype('int64')
    cancer_coverage['tumor_sample_count'] = cancer_coverage[
        'tumor_sample_count'
    ].astype('int64')
    cancer_coverage['model_eligible_fraction'] = (
        cancer_coverage['model_eligible_lncrna_count']
        / cancer_coverage['annotation_lncrna_count']
    )
    cancer_coverage['analysis_version'] = ANALYSIS_VERSION
    cancer_coverage['computation_run_id'] = run_id
    cancer_coverage = cancer_coverage.sort_values('cancer_id').reset_index(drop=True)

    if strict_formal_authority and (
        len(output) != FORMAL_GRID_ROWS
        or int(output['within_cancer_model_eligible'].sum()) != FORMAL_ELIGIBLE_PAIRS
        or len(cancer_coverage) != FORMAL_CANCERS
        or not cancer_coverage['annotation_lncrna_count'].eq(FORMAL_LNCRNAS).all()
    ):
        raise BulkExpressionReleaseError('Fresh bulk expression output has invalid coverage')
    metric_columns = [
        'n_measurements',
        'n_samples',
        'n_patients',
        'mean_logcpm',
        'median_logcpm',
        'q25_logcpm',
        'q75_logcpm',
        'variance_logcpm',
        'observed_detection_rate_logcpm_gt0',
    ]
    if (
        output.loc[eligible, metric_columns].isna().any().any()
        or output.loc[~eligible, metric_columns].notna().any().any()
        or output.loc[eligible, 'failure_reason'].notna().any()
        or output.loc[~eligible, 'failure_reason'].isna().any()
    ):
        raise BulkExpressionReleaseError('Fresh bulk expression availability semantics failed')

    destination.mkdir(parents=True, exist_ok=False)
    summary_path = destination / 'bulk_lncrna_expression_summary.parquet'
    coverage_path = destination / 'bulk_expression_cancer_coverage.parquet'
    source_path = destination / 'SOURCE_INPUTS.json'
    lineage_path = destination / 'MODULE_LINEAGE.json'
    binding_path = destination / 'BULK_EXPRESSION_BINDING.json'
    success_path = destination / 'SUCCESS.json'
    output.to_parquet(summary_path, index=False, compression='zstd')
    cancer_coverage.to_parquet(coverage_path, index=False, compression='zstd')

    source_inputs = {
        'analysis_version': ANALYSIS_VERSION,
        'input_role': 'PROVENANCE_AUDITED_STANDARDIZED_SOURCE_NOT_DERIVED_OUTPUT',
        'historical_predictions_used': False,
        'historical_rankings_used': False,
        'historical_checkpoints_used': False,
        'detection_grid': {
            'path': str(grid_path),
            'sha256': artifact_sha256(grid_path),
        },
        'threshold_success': {
            'path': str(threshold_path),
            'sha256': artifact_sha256(threshold_path),
        },
        'exact_candidates': {
            'path': str(candidate_path),
            'sha256': artifact_sha256(candidate_path),
            'rows': FORMAL_CANDIDATE_ROWS if strict_formal_authority else None,
        },
        'expression_partitions': expression_declarations,
    }
    source_path.write_text(
        json.dumps(source_inputs, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    lineage = {
        'analysis_version': ANALYSIS_VERSION,
        'computation_run_id': run_id,
        'status': 'SUCCESS_FRESH_V32_STATISTICAL_SUMMARY',
        'result_role': 'BULK_EXPRESSION_LANDSCAPE_AND_MODEL_COVERAGE',
        'training_not_applicable_statistical_summary': True,
        'fresh_computation_from_source_measurements': True,
        'historical_derived_outputs_used': False,
        'historical_predictions_used': False,
        'historical_rankings_used': False,
        'historical_checkpoints_used': False,
        'all_output_rows_generated_current_run': True,
        'exact_candidate_sha256': artifact_sha256(candidate_path),
        'release_ready': False,
        'production_deployed': False,
    }
    lineage_path.write_text(
        json.dumps(lineage, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    binding = {
        'format': BINDING_FORMAT,
        'analysis_version': ANALYSIS_VERSION,
        'status': 'SUCCESS_FRESH_V32_BULK_EXPRESSION_HASH_BOUND',
        'computation_run_id': run_id,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'result_role': lineage['result_role'],
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
        'formal_authority': strict_formal_authority,
        'counts': {
            'summary_rows': len(output),
            'cancer_rows': len(cancer_coverage),
            'eligible_pairs': int(output['within_cancer_model_eligible'].sum()),
            'annotation_lncrnas': int(output['lncrna_id'].nunique()),
            'cancers': int(output['cancer_id'].nunique()),
        },
        'semantics': {
            'detection': 'edgeR TMM log2 CPM prior.count=0.25; detected iff logcpm > 0',
            'within_cancer_model_eligibility': 'detection_rate_logcpm_gt0 >= 0.10',
            'unavailable_encoding': 'NULL_WITH_TYPED_REASON',
            'maximum_recomputed_detection_rate_delta': maximum_rate_delta,
        },
        'authorities': {
            'materializer_code': {
                'path': str(code_path),
                'sha256': artifact_sha256(code_path),
            },
            'materializer_runner': {
                'path': str(runner),
                'sha256': artifact_sha256(runner),
            },
            'detection_grid': {
                'path': str(grid_path),
                'sha256': artifact_sha256(grid_path),
            },
            'threshold_success': {
                'path': str(threshold_path),
                'sha256': artifact_sha256(threshold_path),
            },
            'exact_candidates': {
                'path': str(candidate_path),
                'sha256': artifact_sha256(candidate_path),
            },
        },
        'artifacts': {
            'expression_summary': {
                'path': str(summary_path),
                'sha256': artifact_sha256(summary_path),
                'rows': len(output),
            },
            'cancer_coverage': {
                'path': str(coverage_path),
                'sha256': artifact_sha256(coverage_path),
                'rows': len(cancer_coverage),
            },
            'source_inputs': {
                'path': str(source_path),
                'sha256': artifact_sha256(source_path),
            },
            'module_lineage': {
                'path': str(lineage_path),
                'sha256': artifact_sha256(lineage_path),
            },
        },
    }
    binding_path.write_text(
        json.dumps(binding, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    binding_sha = artifact_sha256(binding_path)
    success = {
        'status': binding['status'],
        'binding': binding_path.name,
        'binding_sha256': binding_sha,
        'computation_run_id': run_id,
        'release_ready': False,
        'production_deployed': False,
    }
    success_path.write_text(
        json.dumps(success, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return {
        'binding_path': str(binding_path),
        'binding_sha256': binding_sha,
        'success_path': str(success_path),
        'summary_path': str(summary_path),
        'coverage_path': str(coverage_path),
        'summary_rows': len(output),
        'eligible_pairs': int(output['within_cancer_model_eligible'].sum()),
        'cancer_rows': len(cancer_coverage),
        'computation_run_id': run_id,
    }


__all__ = [
    'ANALYSIS_VERSION',
    'BINDING_FORMAT',
    'BulkExpressionReleaseError',
    'DETECTION_GRID_SHA256',
    'FORMAL_CANDIDATE_SHA256',
    'FORMAL_CANDIDATE_ROWS',
    'FORMAL_CANCERS',
    'FORMAL_ELIGIBLE_PAIRS',
    'FORMAL_GRID_ROWS',
    'FORMAL_LNCRNAS',
    'THRESHOLD_SUCCESS_SHA256',
    'artifact_sha256',
    'materialize_bulk_expression_release',
]
