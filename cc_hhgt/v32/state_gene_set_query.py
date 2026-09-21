"""Hash-pinned read-only queries for fresh V3.2 State Gene Sets."""
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
    from .state_gene_set_release import (
        ANALYSIS_VERSION,
        BINDING_FORMAT,
        FORMAL_CANCER_SPECIFIC_SETS,
        FORMAL_CHECKPOINT_MANIFEST_SHA256,
        FORMAL_GENE_SETS,
        FORMAL_LINEAGE_SHA256,
        FORMAL_MEMBER_ROWS,
        FORMAL_PAN_CANCER_SETS,
        FORMAL_PREDICTION_SHA256,
        STATE_IDS,
        artifact_sha256,
    )
else:
    from state_gene_set_release import (  # type: ignore
        ANALYSIS_VERSION,
        BINDING_FORMAT,
        FORMAL_CANCER_SPECIFIC_SETS,
        FORMAL_CHECKPOINT_MANIFEST_SHA256,
        FORMAL_GENE_SETS,
        FORMAL_LINEAGE_SHA256,
        FORMAL_MEMBER_ROWS,
        FORMAL_PAN_CANCER_SETS,
        FORMAL_PREDICTION_SHA256,
        STATE_IDS,
        artifact_sha256,
    )


MAX_GENE_SET_LIMIT = 500
MAX_MEMBER_LIMIT = 500
MAX_QUERY_OFFSET = 100_000
_SHA256 = re.compile(r'^[0-9a-f]{64}$')
_CANCER = re.compile(r'^[A-Z0-9_-]{2,16}$')


class StateGeneSetQueryError(RuntimeError):
    """Base State gene-set query error."""


class StateGeneSetQueryAssetError(StateGeneSetQueryError):
    """Raised when a binding or output artifact is stale."""


class StateGeneSetQueryInputError(StateGeneSetQueryError):
    """Raised when query filters are invalid."""


def _safe_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise StateGeneSetQueryAssetError(f'{label} is missing or unsafe: {resolved}')
    return resolved


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateGeneSetQueryAssetError(f'{label} is missing or invalid JSON') from exc
    if not isinstance(value, dict):
        raise StateGeneSetQueryAssetError(f'{label} must be a JSON object')
    return value


def _sql_path(path: Path) -> str:
    quote = chr(39)
    return quote + path.resolve().as_posix().replace(quote, quote * 2) + quote


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


def _bounds(limit: int, offset: int, *, maximum: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not 1 <= int(limit) <= maximum:
        raise StateGeneSetQueryInputError(f'limit must be 1..{maximum}')
    if isinstance(offset, bool) or not 0 <= int(offset) <= MAX_QUERY_OFFSET:
        raise StateGeneSetQueryInputError(f'offset must be 0..{MAX_QUERY_OFFSET}')
    return int(limit), int(offset)


def _state(value: str | None) -> str | None:
    if value is None:
        return None
    lookup = {item.casefold(): item for item in STATE_IDS}
    text = str(value).strip().casefold()
    if text not in lookup:
        raise StateGeneSetQueryInputError(f'Unknown state_id: {value}')
    return lookup[text]


def _choice(value: str | None, *, name: str, choices: set[str]) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text not in choices:
        raise StateGeneSetQueryInputError(f'{name} must be one of {sorted(choices)}')
    return text


class StateGeneSetReleaseQuery:
    """Validated immutable view of current V3.2 State gene-set artifacts."""

    def __init__(
        self,
        binding_path: str | Path,
        *,
        expected_binding_sha256: str | None,
        require_formal_authority: bool = True,
    ) -> None:
        source = _safe_file(binding_path, 'State gene-set binding')
        expected = str(expected_binding_sha256 or '').lower()
        if not _SHA256.fullmatch(expected):
            raise StateGeneSetQueryAssetError(
                'State gene-set query requires an expected binding SHA256'
            )
        observed = artifact_sha256(source)
        if observed != expected:
            raise StateGeneSetQueryAssetError(
                f'State gene-set binding SHA mismatch: {observed} != {expected}'
            )
        binding = _load_json(source, 'State gene-set binding')
        required = {
            'format': BINDING_FORMAT,
            'analysis_version': ANALYSIS_VERSION,
            'status': 'SUCCESS_FRESH_V32_STATE_GENE_SETS_HASH_BOUND',
            'result_role': 'SECONDARY_STATE_GENE_SET_AND_REPORT',
            'fresh_computation_from_v32_state_predictions': True,
            'historical_checkpoints_used': False,
            'historical_predictions_used': False,
            'historical_rankings_used': False,
            'historical_gene_sets_used': False,
            'all_output_rows_generated_current_run': True,
            'changes_primary_ranking': False,
            'release_ready': False,
            'production_deployed': False,
        }
        for key, expected_value in required.items():
            if binding.get(key) != expected_value:
                raise StateGeneSetQueryAssetError(
                    f'State gene-set binding has invalid {key}: {binding.get(key)!r}'
                )
        if binding.get('states') != list(STATE_IDS):
            raise StateGeneSetQueryAssetError('State gene-set binding lacks all seven states')
        if require_formal_authority and binding.get('formal_authority') is not True:
            raise StateGeneSetQueryAssetError('State gene-set binding is not formal authority')
        run_id = str(binding.get('computation_run_id', ''))
        if not re.fullmatch(r'V32-STATE-GENE-SETS-[0-9A-F]{16}', run_id):
            raise StateGeneSetQueryAssetError('State gene-set computation_run_id is invalid')
        counts = binding.get('counts')
        formal_counts = {
            'gene_sets': FORMAL_GENE_SETS,
            'member_rows': FORMAL_MEMBER_ROWS,
            'states': 7,
            'cancer_specific_sets': FORMAL_CANCER_SPECIFIC_SETS,
            'pan_cancer_sets': FORMAL_PAN_CANCER_SETS,
        }
        if not isinstance(counts, dict) or (
            require_formal_authority and counts != formal_counts
        ):
            raise StateGeneSetQueryAssetError('State gene-set formal counts are invalid')
        formal_policy = {
            'cancer_max_members': 100,
            'pan_cancer_max_members': 200,
            'minimum_set_members': 3,
            'pan_cancer_min_cancers': 5,
            'pan_cancer_min_direction_consistency': 0.60,
            'primary_order': 'membership_probability_desc_then_effect_strength_desc',
        }
        if require_formal_authority and binding.get('selection_policy') != formal_policy:
            raise StateGeneSetQueryAssetError('State gene-set formal selection policy drifted')
        source_artifacts = binding.get('source_artifacts')
        if not isinstance(source_artifacts, dict):
            raise StateGeneSetQueryAssetError('State gene-set source artifacts are missing')
        expected_sources = {
            'state_predictions': FORMAL_PREDICTION_SHA256,
            'state_module_lineage': FORMAL_LINEAGE_SHA256,
            'state_checkpoint_manifest': FORMAL_CHECKPOINT_MANIFEST_SHA256,
        }
        if require_formal_authority:
            for role, digest in expected_sources.items():
                declaration = source_artifacts.get(role)
                if not isinstance(declaration, dict) or declaration.get('sha256') != digest:
                    raise StateGeneSetQueryAssetError(
                        f'State gene-set source is not formal authority: {role}'
                    )
        for role, declaration in source_artifacts.items():
            if (
                not isinstance(declaration, dict)
                or not _SHA256.fullmatch(str(declaration.get('sha256', '')))
            ):
                raise StateGeneSetQueryAssetError(
                    f'State gene-set source declaration is invalid: {role}'
                )
            source_path = _safe_file(declaration.get('path', ''), f'source {role}')
            if artifact_sha256(source_path) != declaration['sha256']:
                raise StateGeneSetQueryAssetError(f'State gene-set source SHA drift: {role}')
        success = _load_json(source.parent / 'SUCCESS.json', 'State gene-set SUCCESS')
        if (
            success.get('status') != binding['status']
            or success.get('binding') != source.name
            or success.get('binding_sha256') != observed
            or success.get('computation_run_id') != run_id
            or success.get('all_seven_states_present') is not True
            or success.get('historical_state_outputs_used') is not False
            or success.get('release_ready') is not False
            or success.get('production_deployed') is not False
        ):
            raise StateGeneSetQueryAssetError('State gene-set SUCCESS marker is stale')
        artifacts = binding.get('artifacts')
        roles = {
            'catalog', 'members', 'gmt', 'report', 'module_lineage', 'report_manifest'
        }
        if not isinstance(artifacts, dict) or not roles.issubset(artifacts):
            raise StateGeneSetQueryAssetError('State gene-set artifacts are incomplete')
        paths: dict[str, Path] = {}
        for role in sorted(roles):
            declaration = artifacts.get(role)
            if (
                not isinstance(declaration, dict)
                or not _SHA256.fullmatch(str(declaration.get('sha256', '')))
            ):
                raise StateGeneSetQueryAssetError(
                    f'State gene-set artifact declaration is invalid: {role}'
                )
            path = _safe_file(declaration.get('path', ''), role)
            if artifact_sha256(path) != declaration['sha256']:
                raise StateGeneSetQueryAssetError(f'State gene-set artifact SHA drift: {role}')
            paths[role] = path
        lineage = _load_json(paths['module_lineage'], 'State gene-set lineage')
        report_manifest = _load_json(paths['report_manifest'], 'State report manifest')
        if (
            lineage.get('status') != 'SUCCESS_FRESH_V32_STATE_GENE_SETS'
            or lineage.get('computation_run_id') != run_id
            or lineage.get('historical_gene_sets_used') is not False
            or lineage.get('all_output_rows_generated_current_run') is not True
            or lineage.get('changes_primary_ranking') is not False
            or report_manifest.get('status') != 'SUCCESS_FRESH_V32_STATE_REPORT'
            or report_manifest.get('states') != list(STATE_IDS)
            or report_manifest.get('historical_state_outputs_used') is not False
        ):
            raise StateGeneSetQueryAssetError('State lineage or report manifest is invalid')
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
        catalog = f'read_parquet({_sql_path(self.paths["catalog"])})'
        members = f'read_parquet({_sql_path(self.paths["members"])})'
        required_catalog = {
            'gene_set_id', 'gene_set_name', 'gene_set_type', 'scope', 'cancer_id',
            'state_id', 'direction', 'member_count', 'selection_rule',
            'analysis_version', 'source_training_run_id', 'computation_run_id',
            'changes_primary_ranking',
        }
        required_members = {
            'gene_set_id', 'member_rank', 'member_id', 'member_gene_id',
            'membership_score', 'state_effect', 'cancer_count',
            'direction_consistency', 'source_cancer_id', 'analysis_version',
            'source_training_run_id', 'computation_run_id', 'changes_primary_ranking',
        }
        con = self._connect()
        try:
            catalog_columns = {
                row[0] for row in con.execute(f'DESCRIBE SELECT * FROM {catalog}').fetchall()
            }
            member_columns = {
                row[0] for row in con.execute(f'DESCRIBE SELECT * FROM {members}').fetchall()
            }
            if missing := sorted(required_catalog - catalog_columns):
                raise StateGeneSetQueryAssetError(f'State catalog lacks columns: {missing}')
            if missing := sorted(required_members - member_columns):
                raise StateGeneSetQueryAssetError(f'State members lack columns: {missing}')
            catalog_audit = con.execute(
                f'''SELECT count(*), count(DISTINCT gene_set_id),
                           count(DISTINCT state_id),
                           count_if(scope = 'CANCER_SPECIFIC'),
                           count_if(scope = 'PAN_CANCER'),
                           count_if(scope NOT IN ('CANCER_SPECIFIC', 'PAN_CANCER')
                                    OR direction NOT IN ('positive', 'negative')
                                    OR member_count < 3
                                    OR analysis_version <> ?
                                    OR computation_run_id <> ?
                                    OR changes_primary_ranking)
                    FROM {catalog}''',
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            member_audit = con.execute(
                f'''SELECT count(*), count(DISTINCT (gene_set_id, member_id)),
                           count_if(member_rank < 1 OR membership_score NOT BETWEEN 0 AND 1
                                    OR NOT isfinite(membership_score)
                                    OR NOT isfinite(state_effect)
                                    OR direction_consistency NOT BETWEEN 0.5 AND 1
                                    OR cancer_count < 1
                                    OR analysis_version <> ?
                                    OR computation_run_id <> ?
                                    OR changes_primary_ranking)
                    FROM {members}''',
                [ANALYSIS_VERSION, self.computation_run_id],
            ).fetchone()
            count_mismatches = int(
                con.execute(
                    f'''SELECT count(*) FROM (
                            SELECT c.gene_set_id, c.member_count, count(m.member_id) AS observed,
                                   min(m.member_rank) AS minimum_rank,
                                   max(m.member_rank) AS maximum_rank,
                                   count(DISTINCT m.member_rank) AS distinct_ranks
                            FROM {catalog} c
                            LEFT JOIN {members} m USING (gene_set_id)
                            GROUP BY c.gene_set_id, c.member_count
                            HAVING observed <> member_count OR minimum_rank <> 1
                               OR maximum_rank <> member_count OR distinct_ranks <> member_count
                        )'''
                ).fetchone()[0]
            )
            state_values = {
                row[0] for row in con.execute(f'SELECT DISTINCT state_id FROM {catalog}').fetchall()
            }
        finally:
            con.close()
        expected_counts = self.binding['counts']
        if (
            int(catalog_audit[0]) != int(catalog_audit[1])
            or int(catalog_audit[0]) != int(expected_counts['gene_sets'])
            or int(catalog_audit[2]) != 7
            or int(catalog_audit[3]) != int(expected_counts['cancer_specific_sets'])
            or int(catalog_audit[4]) != int(expected_counts['pan_cancer_sets'])
            or int(catalog_audit[5]) != 0
            or int(member_audit[0]) != int(member_audit[1])
            or int(member_audit[0]) != int(expected_counts['member_rows'])
            or int(member_audit[2]) != 0
            or count_mismatches != 0
            or state_values != set(STATE_IDS)
        ):
            raise StateGeneSetQueryAssetError('State gene-set table semantic audit failed')

    def _provenance(self) -> dict[str, Any]:
        return {
            'binding_sha256': self.binding_sha256,
            'computation_run_id': self.computation_run_id,
            'source_training_run_id': self.binding['source_training_run_id'],
            'fresh_computation_from_v32_state_predictions': True,
            'historical_checkpoints_used': False,
            'historical_predictions_used': False,
            'historical_rankings_used': False,
            'historical_gene_sets_used': False,
            'changes_primary_ranking': False,
            'release_ready': False,
            'production_deployed': False,
        }

    def query_gene_sets(
        self,
        *,
        state_id: str | None = None,
        cancer_id: str | None = None,
        scope: str | None = None,
        direction: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset, maximum=MAX_GENE_SET_LIMIT)
        state_id = _state(state_id)
        scope = _choice(
            scope, name='scope', choices={'CANCER_SPECIFIC', 'PAN_CANCER'}
        )
        direction = _choice(
            direction, name='direction', choices={'POSITIVE', 'NEGATIVE'}
        )
        cancer = None
        if cancer_id is not None:
            cancer = str(cancer_id).strip().upper()
            if not _CANCER.fullmatch(cancer):
                raise StateGeneSetQueryInputError('cancer_id is invalid')
        relation = f'read_parquet({_sql_path(self.paths["catalog"])})'
        predicates: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ('state_id', state_id),
            ('cancer_id', cancer),
            ('scope', scope),
            ('direction', direction.lower() if direction else None),
        ):
            if value is not None:
                predicates.append(f'{column} = ?')
                parameters.append(value)
        where = ' WHERE ' + ' AND '.join(predicates) if predicates else ''
        con = self._connect()
        try:
            total = int(
                con.execute(f'SELECT count(*) FROM {relation}{where}', parameters).fetchone()[0]
            )
            rows = con.execute(
                f'''SELECT * FROM {relation}{where}
                    ORDER BY state_id, scope, cancer_id, direction, gene_set_id
                    LIMIT ? OFFSET ?''',
                [*parameters, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return {
            'analysis_version': ANALYSIS_VERSION,
            'query_type': 'state_gene_sets',
            'total_rows': total,
            'returned_rows': int(len(rows)),
            'limit': limit,
            'offset': offset,
            'rows': _records(rows),
            'provenance': self._provenance(),
        }

    def query_members(
        self,
        *,
        gene_set_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> dict[str, Any]:
        limit, offset = _bounds(limit, offset, maximum=MAX_MEMBER_LIMIT)
        identifier = str(gene_set_id or '').strip()
        if not identifier or len(identifier) > 256 or any(ord(char) < 32 for char in identifier):
            raise StateGeneSetQueryInputError('gene_set_id is required and must be valid')
        relation = f'read_parquet({_sql_path(self.paths["members"])})'
        con = self._connect()
        try:
            total = int(
                con.execute(
                    f'SELECT count(*) FROM {relation} WHERE gene_set_id = ?', [identifier]
                ).fetchone()[0]
            )
            rows = con.execute(
                f'''SELECT * FROM {relation} WHERE gene_set_id = ?
                    ORDER BY member_rank LIMIT ? OFFSET ?''',
                [identifier, limit, offset],
            ).fetchdf()
        finally:
            con.close()
        return {
            'analysis_version': ANALYSIS_VERSION,
            'query_type': 'state_gene_set_members',
            'gene_set_id': identifier,
            'total_rows': total,
            'returned_rows': int(len(rows)),
            'limit': limit,
            'offset': offset,
            'rows': _records(rows),
            'provenance': self._provenance(),
        }


__all__ = [
    'MAX_GENE_SET_LIMIT',
    'MAX_MEMBER_LIMIT',
    'MAX_QUERY_OFFSET',
    'StateGeneSetQueryAssetError',
    'StateGeneSetQueryInputError',
    'StateGeneSetReleaseQuery',
]
