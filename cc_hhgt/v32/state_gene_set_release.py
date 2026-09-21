"""Build hash-bound State gene sets from the freshly trained V3.2 State release."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


ANALYSIS_VERSION = 'CancerLncAtlas_V3.2_FULL_MULTITASK'
BINDING_FORMAT = 'CC_HHGT_V3_2_STATE_GENE_SET_BINDING_V1'
FORMAL_PREDICTION_SHA256 = '6f1a12e1d927e06972d7e4ffaf67eed4bd7ecc978d875341bdd19d6cb88e8029'
FORMAL_LINEAGE_SHA256 = '7a5ffdd26f49141694f62ea3b79e518331adeedf8ee6ee5fe077028345fb5b3d'
FORMAL_CHECKPOINT_MANIFEST_SHA256 = 'f67740ba3c5c38019ced7b6822dbe806e3da8c550a807acf5f52f4bc8adb559b'
FORMAL_ROWS = 1_972_971
FORMAL_AVAILABLE_ROWS = 509_253
FORMAL_CANCERS = 33
FORMAL_LNCRNAS = 8_541
FORMAL_GENE_SETS = 452
FORMAL_MEMBER_ROWS = 46_600
FORMAL_CANCER_SPECIFIC_SETS = 438
FORMAL_PAN_CANCER_SETS = 14
STATE_IDS = (
    'EXTEND::published_score',
    'stemness_dna::DMPss',
    'stemness_dna::DNAss',
    'stemness_dna::ENHss',
    'stemness_dna::EREG-METHss',
    'stemness_rna::EREG.EXPss',
    'stemness_rna::RNAss',
)
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


class StateGeneSetReleaseError(RuntimeError):
    """Raised when State gene sets cannot be proven fresh and valid."""


def artifact_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _safe_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise StateGeneSetReleaseError(f'{label} is missing or unsafe: {resolved}')
    return resolved


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateGeneSetReleaseError(f'{label} is missing or invalid JSON') from exc
    if not isinstance(value, dict):
        raise StateGeneSetReleaseError(f'{label} must be a JSON object')
    return value


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f'.{path.name}.tmp')
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f'.{path.stem}.tmp{path.suffix}')
    frame.to_parquet(temporary, index=False, compression='zstd')
    os.replace(temporary, path)


def _validate_sources(
    prediction_path: Path,
    lineage_path: Path,
    checkpoint_path: Path,
    *,
    strict_formal_authority: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if strict_formal_authority:
        required_hashes = {
            prediction_path: FORMAL_PREDICTION_SHA256,
            lineage_path: FORMAL_LINEAGE_SHA256,
            checkpoint_path: FORMAL_CHECKPOINT_MANIFEST_SHA256,
        }
        for path, expected in required_hashes.items():
            observed = artifact_sha256(path)
            if observed != expected:
                raise StateGeneSetReleaseError(
                    f'Formal State authority SHA mismatch for {path.name}: {observed}'
                )
    lineage = _load_json(lineage_path, 'State module lineage')
    checkpoint = _load_json(checkpoint_path, 'State checkpoint manifest')
    required_lineage = {
        'module_id': 'state',
        'analysis_version': ANALYSIS_VERSION,
        'training_status': 'SUCCESS',
        'initialization_policy': 'FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH',
        'folds': 5,
        'old_checkpoint_loaded': False,
        'old_predictions_used_as_features': False,
        'old_rankings_used_as_outputs': False,
        'private_head_trained_from_scratch': True,
        'core_parameters_frozen': True,
        'full_cancer_lncrna_state_universe': True,
    }
    for key, expected in required_lineage.items():
        if lineage.get(key) != expected:
            raise StateGeneSetReleaseError(
                f'State lineage has invalid {key}: {lineage.get(key)!r}'
            )
    if (
        lineage.get('prediction_sha256') != artifact_sha256(prediction_path)
        or lineage.get('checkpoint_manifest_sha256') != artifact_sha256(checkpoint_path)
        or checkpoint.get('analysis_version') != ANALYSIS_VERSION
        or checkpoint.get('module_id') != 'state'
        or checkpoint.get('all_private_heads_fresh') is not True
    ):
        raise StateGeneSetReleaseError('State lineage/checkpoint authority is inconsistent')
    records = checkpoint.get('records')
    if not isinstance(records, dict) or set(records) != {str(index) for index in range(5)}:
        raise StateGeneSetReleaseError('State checkpoint manifest does not contain five folds')
    if any(
        not isinstance(record, dict)
        or record.get('source_checkpoint_sha256') is not None
        or not _SHA256.fullmatch(str(record.get('sha256', '')))
        for record in records.values()
    ):
        raise StateGeneSetReleaseError('A State private head is not fresh-from-scratch')
    try:
        frame = pd.read_parquet(prediction_path)
    except Exception as exc:
        raise StateGeneSetReleaseError('State prediction table is unreadable') from exc
    required_columns = {
        'cancer_id', 'lncrna_id', 'state_id', 'state_membership_probability',
        'state_effect', 'association_direction', 'availability', 'analysis_version',
        'training_run_id', 'changes_primary_ranking',
    }
    if missing := sorted(required_columns - set(frame.columns)):
        raise StateGeneSetReleaseError(f'State prediction table lacks columns: {missing}')
    keys = ['cancer_id', 'lncrna_id', 'state_id']
    if frame.empty or frame.duplicated(keys).any():
        raise StateGeneSetReleaseError('State prediction table is empty or has duplicate keys')
    if set(frame['state_id'].astype(str)) != set(STATE_IDS):
        raise StateGeneSetReleaseError('State prediction table does not contain all seven states')
    if not frame['analysis_version'].eq(ANALYSIS_VERSION).all():
        raise StateGeneSetReleaseError('State prediction analysis_version drifted')
    if frame['training_run_id'].nunique() != 1 or frame['changes_primary_ranking'].any():
        raise StateGeneSetReleaseError('State training run or ranking semantics are invalid')
    frame['availability'] = frame['availability'].astype(bool)
    available = frame['availability']
    probabilities = pd.to_numeric(frame['state_membership_probability'], errors='coerce')
    effects = pd.to_numeric(frame['state_effect'], errors='coerce')
    directions = frame['association_direction'].astype('string')
    if (
        probabilities[available].isna().any()
        or effects[available].isna().any()
        or not np.isfinite(probabilities[available]).all()
        or not np.isfinite(effects[available]).all()
        or not probabilities[available].between(0, 1).all()
        or not directions[available].isin(['positive', 'negative']).all()
        or probabilities[~available].notna().any()
        or effects[~available].notna().any()
        or directions[~available].notna().any()
    ):
        raise StateGeneSetReleaseError('State availability/null/numeric semantics are invalid')
    sign_mismatch = (
        directions[available].eq('positive') & effects[available].lt(0)
    ) | (directions[available].eq('negative') & effects[available].gt(0))
    if sign_mismatch.any():
        raise StateGeneSetReleaseError('State direction and effect signs disagree')
    if strict_formal_authority and (
        len(frame) != FORMAL_ROWS
        or int(available.sum()) != FORMAL_AVAILABLE_ROWS
        or frame['cancer_id'].nunique() != FORMAL_CANCERS
        or frame['lncrna_id'].nunique() != FORMAL_LNCRNAS
        or not frame.groupby('state_id', observed=True).size().eq(
            FORMAL_CANCERS * FORMAL_LNCRNAS
        ).all()
    ):
        raise StateGeneSetReleaseError('Formal State universe has invalid dimensions')
    frame['state_membership_probability'] = probabilities
    frame['state_effect'] = effects
    frame['association_direction'] = directions
    return frame, lineage


def _member_id(lncrna_id: str) -> str:
    return re.sub(r'^LNC:', '', str(lncrna_id), flags=re.IGNORECASE)


def _cancer_specific_sets(
    available: pd.DataFrame,
    *,
    max_members: int,
    min_members: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    catalogs: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    grouped = available.groupby(
        ['cancer_id', 'state_id', 'association_direction'],
        observed=True,
        sort=True,
    )
    for (cancer_id, state_id, direction), group in grouped:
        selected = (
            group.assign(abs_state_effect=group['state_effect'].abs())
            .sort_values(
                ['state_membership_probability', 'abs_state_effect', 'lncrna_id'],
                ascending=[False, False, True],
                kind='stable',
            )
            .drop_duplicates('lncrna_id')
            .head(max_members)
        )
        if len(selected) < min_members:
            continue
        gene_set_id = f'STATE::CANCER::{cancer_id}::{state_id}::{direction}'
        catalogs.append(
            {
                'gene_set_id': gene_set_id,
                'gene_set_name': gene_set_id,
                'gene_set_type': 'state_cancer_specific',
                'scope': 'CANCER_SPECIFIC',
                'cancer_id': str(cancer_id),
                'state_id': str(state_id),
                'direction': str(direction),
                'member_count': int(len(selected)),
                'selection_rule': (
                    f'top_{max_members}_fresh_v32_membership_probability_then_abs_effect'
                ),
            }
        )
        for rank, row in enumerate(selected.itertuples(index=False), start=1):
            members.append(
                {
                    'gene_set_id': gene_set_id,
                    'member_rank': rank,
                    'member_id': str(row.lncrna_id),
                    'member_gene_id': _member_id(row.lncrna_id),
                    'membership_score': float(row.state_membership_probability),
                    'state_effect': float(row.state_effect),
                    'cancer_count': 1,
                    'direction_consistency': 1.0,
                    'source_cancer_id': str(cancer_id),
                }
            )
    return catalogs, members


def _pan_cancer_sets(
    available: pd.DataFrame,
    *,
    max_members: int,
    min_members: int,
    min_cancers: int,
    minimum_direction_consistency: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    working = available.assign(
        positive=available['association_direction'].eq('positive').astype(int),
        abs_state_effect=available['state_effect'].abs(),
    )
    aggregate = (
        working.groupby(['state_id', 'lncrna_id'], observed=True, sort=True)
        .agg(
            cancer_count=('cancer_id', 'nunique'),
            positive_cancers=('positive', 'sum'),
            membership_score=('state_membership_probability', 'mean'),
            state_effect=('state_effect', 'mean'),
            mean_abs_state_effect=('abs_state_effect', 'mean'),
        )
        .reset_index()
    )
    aggregate['negative_cancers'] = (
        aggregate['cancer_count'] - aggregate['positive_cancers']
    )
    aggregate['direction'] = np.where(
        aggregate['positive_cancers'].ge(aggregate['negative_cancers']),
        'positive',
        'negative',
    )
    aggregate['direction_consistency'] = (
        aggregate[['positive_cancers', 'negative_cancers']].max(axis=1)
        / aggregate['cancer_count']
    )
    aggregate = aggregate.loc[
        aggregate['cancer_count'].ge(min_cancers)
        & aggregate['direction_consistency'].ge(minimum_direction_consistency)
    ]
    catalogs: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []
    for (state_id, direction), group in aggregate.groupby(
        ['state_id', 'direction'], observed=True, sort=True
    ):
        selected = group.sort_values(
            [
                'membership_score', 'direction_consistency', 'cancer_count',
                'mean_abs_state_effect', 'lncrna_id',
            ],
            ascending=[False, False, False, False, True],
            kind='stable',
        ).head(max_members)
        if len(selected) < min_members:
            continue
        gene_set_id = f'STATE::PAN_CANCER::{state_id}::{direction}'
        catalogs.append(
            {
                'gene_set_id': gene_set_id,
                'gene_set_name': gene_set_id,
                'gene_set_type': 'state_pan_cancer_consistent',
                'scope': 'PAN_CANCER',
                'cancer_id': 'PAN_CANCER',
                'state_id': str(state_id),
                'direction': str(direction),
                'member_count': int(len(selected)),
                'selection_rule': (
                    f'top_{max_members}_mean_fresh_v32_probability_min_{min_cancers}_cancers_'
                    f'direction_consistency_ge_{minimum_direction_consistency:.2f}'
                ),
            }
        )
        for rank, row in enumerate(selected.itertuples(index=False), start=1):
            members.append(
                {
                    'gene_set_id': gene_set_id,
                    'member_rank': rank,
                    'member_id': str(row.lncrna_id),
                    'member_gene_id': _member_id(row.lncrna_id),
                    'membership_score': float(row.membership_score),
                    'state_effect': float(row.state_effect),
                    'cancer_count': int(row.cancer_count),
                    'direction_consistency': float(row.direction_consistency),
                    'source_cancer_id': None,
                }
            )
    return catalogs, members


def _write_gmt(catalog: pd.DataFrame, members: pd.DataFrame, path: Path) -> None:
    member_groups = {
        gene_set_id: group.sort_values('member_rank')['member_gene_id'].astype(str).tolist()
        for gene_set_id, group in members.groupby('gene_set_id', observed=True, sort=False)
    }
    with path.open('w', encoding='utf-8', newline='\n') as handle:
        for row in catalog.sort_values('gene_set_id').itertuples(index=False):
            description = (
                f'Fresh V3.2 {row.scope} lncRNA set for {row.state_id}; '
                f'direction={row.direction}; no historical predictions/checkpoints'
            )
            handle.write(
                '\t'.join(
                    [row.gene_set_name, description, *member_groups[row.gene_set_id]]
                )
                + '\n'
            )


def materialize_state_gene_set_release(
    *,
    state_prediction_path: str | Path,
    state_lineage_path: str | Path,
    state_checkpoint_manifest_path: str | Path,
    output_root: str | Path,
    runner_path: str | Path | None = None,
    strict_formal_authority: bool = True,
    cancer_max_members: int = 100,
    pan_cancer_max_members: int = 200,
    minimum_set_members: int = 3,
    pan_cancer_min_cancers: int = 5,
    pan_cancer_min_direction_consistency: float = 0.60,
) -> dict[str, Any]:
    """Materialize State Gene Sets without changing the primary pathway ranking."""
    if cancer_max_members < minimum_set_members or pan_cancer_max_members < minimum_set_members:
        raise StateGeneSetReleaseError('Gene-set member limits are smaller than the minimum')
    if pan_cancer_min_cancers < 2 or not 0.5 <= pan_cancer_min_direction_consistency <= 1:
        raise StateGeneSetReleaseError('Pan-cancer selection thresholds are invalid')
    prediction_path = _safe_file(state_prediction_path, 'State predictions')
    lineage_path = _safe_file(state_lineage_path, 'State lineage')
    checkpoint_path = _safe_file(
        state_checkpoint_manifest_path, 'State checkpoint manifest'
    )
    runner = _safe_file(runner_path, 'State gene-set runner') if runner_path else None
    destination = Path(output_root).resolve()
    if destination.exists():
        raise StateGeneSetReleaseError(
            f'Refusing to overwrite existing State gene-set release: {destination}'
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.tmp-{uuid.uuid4().hex}')
    temporary.mkdir()
    try:
        frame, source_lineage = _validate_sources(
            prediction_path,
            lineage_path,
            checkpoint_path,
            strict_formal_authority=strict_formal_authority,
        )
        available = frame.loc[frame['availability']].copy()
        cancer_catalog, cancer_members = _cancer_specific_sets(
            available,
            max_members=cancer_max_members,
            min_members=minimum_set_members,
        )
        pan_catalog, pan_members = _pan_cancer_sets(
            available,
            max_members=pan_cancer_max_members,
            min_members=minimum_set_members,
            min_cancers=pan_cancer_min_cancers,
            minimum_direction_consistency=pan_cancer_min_direction_consistency,
        )
        catalog = pd.DataFrame(cancer_catalog + pan_catalog)
        members = pd.DataFrame(cancer_members + pan_members)
        if catalog.empty or members.empty:
            raise StateGeneSetReleaseError('State selection produced no gene sets')
        if (
            catalog.duplicated('gene_set_id').any()
            or members.duplicated(['gene_set_id', 'member_id']).any()
            or set(catalog['state_id']) != set(STATE_IDS)
            or set(members['gene_set_id']) != set(catalog['gene_set_id'])
        ):
            raise StateGeneSetReleaseError('State gene-set catalog/member invariants failed')
        observed_counts = members.groupby('gene_set_id', observed=True).size()
        declared_counts = catalog.set_index('gene_set_id')['member_count']
        if not observed_counts.sort_index().eq(declared_counts.sort_index()).all():
            raise StateGeneSetReleaseError('State gene-set member counts disagree')

        selection_policy = {
            'cancer_max_members': cancer_max_members,
            'pan_cancer_max_members': pan_cancer_max_members,
            'minimum_set_members': minimum_set_members,
            'pan_cancer_min_cancers': pan_cancer_min_cancers,
            'pan_cancer_min_direction_consistency': pan_cancer_min_direction_consistency,
            'primary_order': 'membership_probability_desc_then_effect_strength_desc',
        }
        computation_run_id = (
            'V32-STATE-GENE-SETS-'
            + _canonical_hash(
                {
                    'prediction_sha256': artifact_sha256(prediction_path),
                    'selection_policy': selection_policy,
                }
            )[:16].upper()
        )
        for table in (catalog, members):
            table['analysis_version'] = ANALYSIS_VERSION
            table['source_training_run_id'] = str(source_lineage['training_run_id'])
            table['computation_run_id'] = computation_run_id
            table['changes_primary_ranking'] = False
        catalog = catalog.sort_values('gene_set_id', kind='stable').reset_index(drop=True)
        members = members.sort_values(
            ['gene_set_id', 'member_rank'], kind='stable'
        ).reset_index(drop=True)

        names = {
            'catalog': 'state_gene_set_catalog.parquet',
            'members': 'state_gene_set_members.parquet',
            'gmt': 'state_gene_sets.gmt',
            'report': 'STATE_GENE_SET_REPORT.md',
            'module_lineage': 'MODULE_LINEAGE.json',
            'report_manifest': 'STATE_REPORT_MANIFEST.json',
        }
        catalog_path = temporary / names['catalog']
        member_path = temporary / names['members']
        gmt_path = temporary / names['gmt']
        report_path = temporary / names['report']
        _atomic_parquet(catalog, catalog_path)
        _atomic_parquet(members, member_path)
        _write_gmt(catalog, members, gmt_path)

        state_counts = (
            catalog.groupby('state_id', observed=True)
            .agg(gene_sets=('gene_set_id', 'size'), members=('member_count', 'sum'))
            .reset_index()
        )
        report_lines = [
            '# CancerLncAtlas V3.2 State Gene Set Report',
            '',
            f'- Computation run: `{computation_run_id}`',
            f"- Source State training run: `{source_lineage['training_run_id']}`",
            f'- Source State prediction SHA256: `{artifact_sha256(prediction_path)}`',
            f'- Gene sets: {len(catalog):,}',
            f'- Member rows: {len(members):,}',
            '- Historical checkpoints, predictions, rankings, and gene sets used: **No**',
            '- Changes primary exact-pathway ranking: **No**',
            '- Release ready / production deployed: **No / No**',
            '',
            '## State coverage',
            '',
            '| State ID | Gene sets | Member rows |',
            '|---|---:|---:|',
        ]
        report_lines.extend(
            f'| `{row.state_id}` | {int(row.gene_sets):,} | {int(row.members):,} |'
            for row in state_counts.itertuples(index=False)
        )
        report_lines.extend(
            [
                '',
                'Cancer-specific sets retain the highest fresh V3.2 State membership '
                'probabilities within each cancer, state, and direction. Pan-cancer sets '
                f'require at least {pan_cancer_min_cancers} cancers and '
                f'{pan_cancer_min_direction_consistency:.0%} direction consistency.',
                '',
            ]
        )
        report_path.write_text('\n'.join(report_lines), encoding='utf-8')

        output_declarations: dict[str, dict[str, Any]] = {}
        for role, path, rows in (
            ('catalog', catalog_path, len(catalog)),
            ('members', member_path, len(members)),
            ('gmt', gmt_path, None),
            ('report', report_path, None),
        ):
            declaration: dict[str, Any] = {
                'path': str(destination / path.name),
                'sha256': artifact_sha256(path),
            }
            if rows is not None:
                declaration['rows'] = int(rows)
            output_declarations[role] = declaration
        source_artifacts = {
            'materialization_module': {
                'path': str(Path(__file__).resolve()),
                'sha256': artifact_sha256(Path(__file__).resolve()),
            },
            'state_predictions': {
                'path': str(prediction_path), 'sha256': artifact_sha256(prediction_path),
            },
            'state_module_lineage': {
                'path': str(lineage_path), 'sha256': artifact_sha256(lineage_path),
            },
            'state_checkpoint_manifest': {
                'path': str(checkpoint_path), 'sha256': artifact_sha256(checkpoint_path),
            },
        }
        if runner is not None:
            source_artifacts['materialization_runner'] = {
                'path': str(runner), 'sha256': artifact_sha256(runner),
            }
        counts = {
            'gene_sets': int(len(catalog)),
            'member_rows': int(len(members)),
            'states': int(catalog['state_id'].nunique()),
            'cancer_specific_sets': int(catalog['scope'].eq('CANCER_SPECIFIC').sum()),
            'pan_cancer_sets': int(catalog['scope'].eq('PAN_CANCER').sum()),
        }
        module_lineage = {
            'module_id': 'state_gene_set',
            'analysis_version': ANALYSIS_VERSION,
            'status': 'SUCCESS_FRESH_V32_STATE_GENE_SETS',
            'computation_run_id': computation_run_id,
            'source_training_run_id': source_lineage['training_run_id'],
            'source_state_module_training_status': source_lineage['training_status'],
            'fresh_computation_from_v32_state_predictions': True,
            'historical_checkpoints_used': False,
            'historical_predictions_used': False,
            'historical_rankings_used': False,
            'historical_gene_sets_used': False,
            'all_output_rows_generated_current_run': True,
            'changes_primary_ranking': False,
            'selection_policy': selection_policy,
            'source_artifacts': source_artifacts,
            'output_artifacts': output_declarations,
            'counts': counts,
            'release_ready': False,
            'production_deployed': False,
        }
        lineage_output_path = temporary / names['module_lineage']
        _atomic_json(module_lineage, lineage_output_path)
        output_declarations['module_lineage'] = {
            'path': str(destination / lineage_output_path.name),
            'sha256': artifact_sha256(lineage_output_path),
        }
        report_manifest = {
            'format': 'CC_HHGT_V3_2_STATE_REPORT_MANIFEST_V1',
            'analysis_version': ANALYSIS_VERSION,
            'status': 'SUCCESS_FRESH_V32_STATE_REPORT',
            'computation_run_id': computation_run_id,
            'states': list(STATE_IDS),
            'artifacts': output_declarations,
            'historical_state_outputs_used': False,
            'release_ready': False,
            'production_deployed': False,
        }
        report_manifest_path = temporary / names['report_manifest']
        _atomic_json(report_manifest, report_manifest_path)
        output_declarations['report_manifest'] = {
            'path': str(destination / report_manifest_path.name),
            'sha256': artifact_sha256(report_manifest_path),
        }
        binding = {
            'format': BINDING_FORMAT,
            'analysis_version': ANALYSIS_VERSION,
            'status': 'SUCCESS_FRESH_V32_STATE_GENE_SETS_HASH_BOUND',
            'result_role': 'SECONDARY_STATE_GENE_SET_AND_REPORT',
            'computation_run_id': computation_run_id,
            'source_training_run_id': source_lineage['training_run_id'],
            'formal_authority': bool(strict_formal_authority),
            'fresh_computation_from_v32_state_predictions': True,
            'historical_checkpoints_used': False,
            'historical_predictions_used': False,
            'historical_rankings_used': False,
            'historical_gene_sets_used': False,
            'all_output_rows_generated_current_run': True,
            'changes_primary_ranking': False,
            'states': list(STATE_IDS),
            'selection_policy': selection_policy,
            'counts': counts,
            'source_artifacts': source_artifacts,
            'artifacts': output_declarations,
            'release_ready': False,
            'production_deployed': False,
        }
        binding_path = temporary / 'STATE_GENE_SET_BINDING.json'
        _atomic_json(binding, binding_path)
        binding_sha256 = artifact_sha256(binding_path)
        success = {
            'status': binding['status'],
            'analysis_version': ANALYSIS_VERSION,
            'computation_run_id': computation_run_id,
            'binding': binding_path.name,
            'binding_sha256': binding_sha256,
            'counts': counts,
            'all_seven_states_present': set(catalog['state_id']) == set(STATE_IDS),
            'historical_state_outputs_used': False,
            'release_ready': False,
            'production_deployed': False,
        }
        _atomic_json(success, temporary / 'SUCCESS.json')
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        'status': 'SUCCESS_FRESH_V32_STATE_GENE_SETS_HASH_BOUND',
        'output_root': str(destination),
        'binding_path': str(destination / 'STATE_GENE_SET_BINDING.json'),
        'binding_sha256': binding_sha256,
        'gene_sets': int(len(catalog)),
        'member_rows': int(len(members)),
        'release_ready': False,
        'production_deployed': False,
    }


__all__ = [
    'ANALYSIS_VERSION',
    'BINDING_FORMAT',
    'STATE_IDS',
    'StateGeneSetReleaseError',
    'artifact_sha256',
    'materialize_state_gene_set_release',
]
