"""Typed patient observations for fixed pathways; no lncRNA/association input."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def observation_id(cohort, kind, layer, feature_id):
    identity = json.dumps([cohort, kind, layer, feature_id], ensure_ascii=False)
    return 'CLA-O-' + hashlib.sha256(identity.encode()).hexdigest()[:24].upper()


def verify_matrix(values, observed, patients):
    if not values.index.is_unique or not values.columns.is_unique:
        raise ValueError('Y_DUPLICATE_MEASUREMENT_AXIS')
    if list(values.index) != list(patients):
        raise ValueError('Y_PATIENT_ORDER')
    if not values.index.equals(observed.index) or not values.columns.equals(observed.columns):
        raise ValueError('Y_MASK_AXIS')
    if not all(t == bool for t in observed.dtypes):
        raise ValueError('Y_MASK_NOT_BOOLEAN')
    if np.isinf(values.to_numpy(float)).any():
        raise ValueError('Y_INFINITE_MEASUREMENT')
    if not np.array_equal(np.isfinite(values.to_numpy(float)), observed.to_numpy()):
        raise ValueError('Y_MASK_NOT_MEASURED')


def source_categories(readouts, source_roles):
    """A shared biological identity must not overwrite a source's category."""
    if not source_roles.source_target_id.is_unique:
        raise ValueError('Y_DUPLICATE_SOURCE_ROLE')
    categories = source_roles.set_index('source_target_id').display_category
    result = readouts.source_target_id.map(categories)
    if result.isna().any():
        missing = readouts.loc[result.isna(), 'source_target_id'].unique().tolist()
        raise ValueError('Y_UNKNOWN_SOURCE_ROLE: ' + repr(missing))
    return result


def normalize_links(links, observations, targets):
    required = ['canonical_id', 'observation_id', 'evidence_role', 'condition_required']
    if not set(required) <= set(links):
        raise ValueError('Y_LINK_SCHEMA')
    if not set(links.canonical_id) <= set(targets.canonical_id):
        raise ValueError('Y_LINK_UNKNOWN_TARGET')
    if not set(links.observation_id) <= set(observations.observation_id):
        raise ValueError('Y_LINK_UNKNOWN_OBSERVATION')
    roles = {'same_state_readout', 'equivalent_pathway_readout',
             'molecular_component_readout', 'member_measurement', 'ambiguous_group_context'}
    if not set(links.evidence_role) <= roles:
        raise ValueError('Y_UNKNOWN_EVIDENCE_ROLE')
    out = links.drop_duplicates(['canonical_id', 'observation_id', 'evidence_role']).copy()
    # Structural context is not silently promoted to a target measurement.
    out['is_target_observation'] = out.evidence_role.ne('ambiguous_group_context')
    out['independent_loss_enabled'] = False
    return out


def condition_for_target(applicable, target, patients):
    """Keep unknown conditions unknown; a missing primary condition axis is an error."""
    canonical_id = target['canonical_id']
    if set(applicable.index) != set(patients) or not applicable.index.is_unique:
        raise ValueError('Y_CONDITION_PATIENT_ORDER')
    applicable = applicable.reindex(patients)
    if canonical_id in applicable:
        result = applicable[canonical_id]
        if not result.dropna().isin([0, 1]).all():
            raise ValueError('Y_INVALID_CONDITION_VALUE')
        return result
    if target['analysis_role'] != 'state_auxiliary' or target['condition_required']:
        raise ValueError('Y_MISSING_PRIMARY_CONDITION_AXIS')
    return pd.Series(1.0, index=patients, name=canonical_id)


def load_target(registry, canonical_id, cohort, *, qualified_only=False,
                include_external_reference=False):
    """Return original observed dimensions, masks and separate applicability.

    Values from different platforms are never concatenated on a shared column.
    This accessor does not fit, normalize, average or calculate associations.
    """
    from .relation_evidence_bindings import file_sha
    registry = Path(registry)
    receipt = json.loads((registry / 'COMPLETE.json').read_text())
    for name in ['TARGETS.tsv', 'OBSERVATIONS.tsv.gz', 'PATHWAY_OBSERVATIONS.tsv.gz', 'MATRICES.json']:
        if file_sha(registry / name) != receipt['files'][name]:
            raise ValueError('Y_REGISTRY_DRIFT')
    targets = pd.read_csv(registry / 'TARGETS.tsv', sep='\t', keep_default_na=False)
    if canonical_id not in set(targets.canonical_id):
        raise KeyError(canonical_id)
    obs = pd.read_csv(registry / 'OBSERVATIONS.tsv.gz', sep='\t', keep_default_na=False)
    links = pd.read_csv(registry / 'PATHWAY_OBSERVATIONS.tsv.gz', sep='\t', keep_default_na=False)
    selected = links.loc[links.canonical_id.eq(canonical_id) & links.is_target_observation]
    selected = obs[obs.cohort.eq(cohort) & obs.observation_id.isin(selected.observation_id)]
    if not include_external_reference:
        selected = selected[selected.supervision_eligible]
    matrices = json.loads((registry / 'MATRICES.json').read_text())
    if cohort not in matrices['patients']:
        raise KeyError(cohort)
    patients = matrices['patients'][cohort]
    blocks = []
    masks = []
    for matrix_id, metadata in selected.groupby('matrix_id', sort=False):
        matrix = matrices['blocks'][matrix_id]
        for field in ['values', 'observed']:
            if file_sha(matrix[field]['path']) != matrix[field]['sha256']:
                raise ValueError('Y_MATRIX_DRIFT')
        columns = metadata.feature_id.tolist()
        values = pd.read_parquet(matrix['values']['path'], columns=columns)
        observed = pd.read_parquet(matrix['observed']['path'], columns=columns)
        if set(values.index) != set(patients) or not values.index.equals(observed.index):
            raise ValueError('Y_MATRIX_PATIENTS')
        values = values.reindex(patients)
        observed = observed.reindex(patients)
        verify_matrix(values, observed, patients)
        values.columns = observed.columns = metadata.observation_id.tolist()
        blocks.append(values)
        masks.append(observed)
    values = pd.concat(blocks, axis=1) if blocks else pd.DataFrame(index=patients, dtype=float)
    observed = pd.concat(masks, axis=1) if masks else pd.DataFrame(index=patients, dtype=bool)
    condition = matrices['conditions'][cohort]
    if file_sha(condition['path']) != condition['sha256']:
        raise ValueError('Y_CONDITION_DRIFT')
    applicable = pd.read_parquet(condition['path'])
    target = targets.set_index('canonical_id').loc[canonical_id].to_dict()
    target['canonical_id'] = canonical_id
    applicable = condition_for_target(applicable, target, patients)
    if qualified_only:
        observed = observed.mul(applicable.eq(1), axis=0).astype(bool)
    return values, observed, applicable, selected
