"""Source-patient features for exact cancer-lncRNA-target prediction.

Candidate selection and all feature computation are independent of label values.
Each cohort/assay retains its measured readouts and source-fitted adjustment.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import sha
from .fused_adjustment.adjustment_policy import resolve_policy
from .relation_source import (SOURCE_NUMERIC_FIELDS, SourceAssociations,
    fit_source_design, load_relation_lncrna_sources, load_source_covariates,
    load_source_local_cnv, load_source_readouts, source_numeric_block,
    source_readout_associations)

COHORTS = ('TCGA', 'CPTAC')
LAYERS = ('RNA', 'STATE', 'PROTEIN', 'PHOSPHO', 'METHYLATION')
ENTITY_FIELDS = ('expression_mean', 'expression_sd', 'expression_measured_fraction',
                 'local_cnv_mean', 'local_cnv_sd', 'local_cnv_callable_fraction')


def verified_file(binding):
    path = Path(binding['path'])
    if not binding.get('sha256') or not path.is_file() or sha(path) != binding['sha256']:
        raise ValueError('RELATION_FEATURE_INPUT_HASH_MISMATCH: ' + str(path))
    return path


def read_exclusions(direct_binding, mixed_binding):
    """Use the same RNA self-membership exclusions as the frozen label generator."""
    def rows(binding):
        with verified_file(binding).open(encoding='utf-8-sig', newline='') as handle:
            return list(csv.DictReader(handle, delimiter='\t'))
    direct = rows(direct_binding)
    mixed = rows(mixed_binding)
    if any(not row.get('state_id') or not row.get('lncrna_id') for row in direct):
        raise ValueError('RELATION_FEATURE_DIRECT_MEMBER_SCHEMA')
    if any(not row.get('gene_id') for row in mixed):
        raise ValueError('RELATION_FEATURE_MIXED_MEMBER_SCHEMA')
    return ({(row['lncrna_id'], row['state_id']) for row in direct},
            {'LNC:' + row['gene_id'] for row in mixed})


def expand_candidate_readouts(candidate_lnc, candidate_target, target_axis, readouts):
    """Expand exact target candidates to their source readouts without aliases."""
    li, ti = np.asarray(candidate_lnc), np.asarray(candidate_target)
    if (li.ndim != 1 or ti.shape != li.shape or not np.issubdtype(li.dtype, np.integer)
            or not np.issubdtype(ti.dtype, np.integer) or np.any(li < 0)
            or np.any(ti < 0) or np.any(ti >= len(target_axis))
            or len(target_axis) != len(set(target_axis))):
        raise ValueError('RELATION_FEATURE_CANDIDATE_AXIS_INVALID')
    if len(np.unique(li.astype(np.int64) * len(target_axis) + ti)) != len(li):
        raise ValueError('RELATION_FEATURE_DUPLICATE_CANDIDATE')
    target_lookup = {target: i for i, target in enumerate(target_axis)}
    if set(readouts.canonical_ids) - set(target_lookup):
        raise ValueError('RELATION_FEATURE_READOUT_TARGET_UNKNOWN')
    by_target = {target: li[ti == target_lookup[target]] for target in set(readouts.canonical_ids)}
    pairs = [(by_target[target], np.full(len(by_target[target]), column, np.int64))
             for column, target in enumerate(readouts.canonical_ids)]
    if not pairs:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])


def compute_readout_block(expression, readouts, local_cnv, covariates, cfg,
                          candidate_lnc, candidate_target, target_axis, exclusions):
    """Apply exact State adjustment and RNA self-exclusions before aggregation."""
    if len(readouts.metadata) != len(readouts.readout_ids):
        raise ValueError('RELATION_FEATURE_READOUT_METADATA_REQUIRED')
    li, ri = expand_candidate_readouts(candidate_lnc, candidate_target, target_axis, readouts)
    if np.any(li >= len(expression.entities)):
        raise ValueError('RELATION_FEATURE_LNCRNA_INDEX_INVALID')
    count = len(li)
    rho, p = np.full(count, np.nan, np.float32), np.full(count, np.nan, np.float32)
    n, rank, available = np.zeros(count, np.int32), np.zeros(count, np.int16), np.zeros(count, bool)
    groups, representatives = defaultdict(list), {}
    for column, row in enumerate(readouts.metadata):
        if row.get('target_id') != readouts.readout_ids[column]:
            raise ValueError('RELATION_FEATURE_READOUT_METADATA_MISMATCH')
        target = row.get('source_target_id') if readouts.layer == 'STATE' else '__pathway__'
        if not target:
            raise ValueError('RELATION_FEATURE_ADJUSTMENT_TARGET_MISSING')
        _, policy = resolve_policy(target, contract=cfg['adjustment_contract'])
        key = json.dumps(policy, sort_keys=True)
        groups[key].append(column)
        representatives[key] = (target, row.get('family', 'pathway'))
    reports = []
    for key, columns in groups.items():
        rows = np.flatnonzero(np.isin(ri, columns))
        if not len(rows):
            continue
        target, family = representatives[key]
        design = fit_source_design(covariates, cfg, family=family, target_id=target)
        result = source_readout_associations(expression, readouts, local_cnv, design,
                  li[rows], ri[rows], minimum_pairs=int(cfg['minimum_pairs']))
        rho[rows], p[rows] = result.rho, result.p
        n[rows], rank[rows], available[rows] = result.n, result.rank, result.available
        reports.append({'readout_ids': [readouts.readout_ids[i] for i in columns],
                        'representative_target': target, 'report': design.report})
    direct, mixed = exclusions
    excluded = np.zeros(count, bool)
    if readouts.layer == 'RNA':
        excluded = np.asarray([expression.entities[i] in mixed for i in li], bool)
    elif readouts.layer == 'STATE':
        excluded = np.asarray([(expression.entities[i], readouts.metadata[j]['source_target_id'])
                               in direct for i, j in zip(li, ri)], bool)
    available[excluded] = False
    rho[excluded], p[excluded] = np.nan, np.nan
    associations = SourceAssociations(expression.cohort, readouts.layer, expression.patients,
                                      li, ri, rho, p, n, rank, available)
    numeric = source_numeric_block(associations, readouts, candidate_lnc, candidate_target, target_axis)
    return numeric, associations, {'cohort': expression.cohort, 'layer': readouts.layer,
        'readouts': len(readouts.readout_ids), 'expanded_pairs': count,
        'estimable_pairs': int(available.sum()), 'direct_rna_pairs_excluded': int(excluded.sum()),
        'adjustment_groups': reports, 'source_score_sha256': readouts.source_sha256}


def entity_summary(values, measured):
    if values.shape != measured.shape or not np.array_equal(np.isfinite(values), measured):
        raise ValueError('RELATION_FEATURE_ENTITY_MEASUREMENT_MISMATCH')
    count = measured.sum(0)
    mean = np.divide(np.where(measured, values, 0).sum(0), count,
                     out=np.zeros(values.shape[1], float), where=count > 0)
    squared = np.where(measured, values - mean, 0) ** 2
    variance = np.divide(squared.sum(0), count, out=np.zeros_like(mean), where=count > 0)
    return np.column_stack([mean, np.sqrt(variance), count / max(len(values), 1)]).astype(np.float32)


def build_source_features(context, bindings, candidate_lnc, candidate_target, *, progress=None):
    """Build shared M0/HHGT numeric inputs; no relation-label array is accessed."""
    from .local_execution import require_host_149
    require_host_149()
    cfg = json.loads(verified_file(bindings['adjustment_config']).read_text())
    if cfg.get('target_host') != '149' or context.cancer_id not in cfg.get('cancer_list', []):
        raise ValueError('RELATION_FEATURE_CONFIG_SCOPE_MISMATCH')
    view_definition = verified_file(bindings['target_view_definition'])
    if progress:
        progress({'stage': 'load_source_expression', 'cancer': context.cancer_id})
    target_axis = tuple(context.canonical_ids)
    expressions = load_relation_lncrna_sources(context, bindings['aligned_root'])
    matrices, names, audits, edges, readout_axes, node_features = [], [], [], {}, {}, {}
    for cohort in COHORTS:
        covariates = load_source_covariates(context, bindings['covariate_root'], cohort,
                                            bindings['covariate_sha256'][cohort])
        cnv = load_source_local_cnv(context, bindings['cnv_roots'], cohort, context.lncrna_ids)
        exclusions = read_exclusions(bindings['direct_rna_members'][cohort], bindings['mixed_rna_members'])
        for layer in LAYERS:
            readouts = load_source_readouts(context, view_definition.parent, cohort, layer, target_axis)
            numeric, association, audit = compute_readout_block(expressions[cohort], readouts,
                cnv, covariates, cfg, candidate_lnc, candidate_target, target_axis, exclusions)
            matrices.append(numeric)
            names.extend(f'{cohort}/{layer}/{field}' for field in SOURCE_NUMERIC_FIELDS)
            audits.append(audit)
            if progress:
                progress({k: audit[k] for k in ('cohort', 'layer', 'readouts',
                                                'expanded_pairs', 'estimable_pairs',
                                                'direct_rna_pairs_excluded')})
            key = cohort + '/' + layer
            edges[key] = association
            readout_axes[key] = {'readout_ids': list(readouts.readout_ids),
                                'canonical_ids': list(readouts.canonical_ids),
                                'metadata': list(readouts.metadata)}
        summary = np.column_stack([entity_summary(expressions[cohort].values,
                                                   expressions[cohort].measured),
                                   entity_summary(cnv.values, cnv.callable)])
        node_features[cohort] = summary
        matrices.append(summary[np.asarray(candidate_lnc)])
        names.extend(f'{cohort}/{field}' for field in ENTITY_FIELDS)
    return {'numeric': np.column_stack(matrices).astype(np.float32),
            'feature_names': names, 'audits': audits, 'associations': edges,
            'readout_axes': readout_axes, 'lncrna_node_features': node_features,
            'source_patients': {cohort: list(block.patients) for cohort, block in expressions.items()},
            'input_bindings': bindings}
