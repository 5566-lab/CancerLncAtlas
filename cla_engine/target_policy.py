"""Explicit functional target admission and loss policy, independent of display roles.

A Pathway display category may contain an auxiliary TF or kinase. Classification
must never choose its loss weight. Older frozen contexts without a policy retain
legacy routing; a policy-bearing catalog must bind every supervised identity.
"""
from __future__ import annotations
from dataclasses import dataclass
import csv
import hashlib
import math
from pathlib import Path

POLICY_SCHEMA = 'CLA_TARGET_POLICY_V1'
REQUIRED_FIELDS = ('canonical_id', 'roles', 'biological_type', 'analysis_role',
                   'loss_group', 'loss_weight', 'active_in_analysis',
                   'cancer_scope_status', 'backbone_anchor_ids')
GROUP_WEIGHTS = {'pathway': 1.0, 'state': 0.5, 'none': 0.0}
SUPERVISED_ROLES = {'primary_pathway', 'regulator_auxiliary', 'state_auxiliary'}
INACTIVE_STATUSES = {'candidate_overlap_only', 'excluded_zero_overlap',
                     'unmapped_definition', 'excluded_scope'}
ROLE_GROUPS = {'primary_pathway': 'pathway', 'regulator_auxiliary': 'state',
               'state_auxiliary': 'state', 'context_only': 'none', 'excluded': 'none'}


def _bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'true', '1'}:
        return True
    if normalized in {'false', '0'}:
        return False
    raise ValueError('TARGET_POLICY_INVALID_BOOLEAN: ' + str(value))


def explicit_task_weights(loss_groups, loss_weights):
    """Validate explicit training groups without consulting a display category."""
    groups, weights = tuple(loss_groups), tuple(loss_weights)
    if len(groups) != len(weights):
        raise ValueError('TARGET_POLICY_WEIGHT_AXIS_MISMATCH')
    result = []
    for group, weight in zip(groups, weights):
        try:
            value = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError('TARGET_POLICY_INVALID_WEIGHT: ' + str(weight)) from exc
        if group not in GROUP_WEIGHTS or not math.isfinite(value) or value != GROUP_WEIGHTS[group]:
            raise ValueError('TARGET_POLICY_GROUP_WEIGHT_MISMATCH: ' + str(group))
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class TargetPolicy:
    canonical_id: str
    roles: str
    biological_type: str
    analysis_role: str
    loss_group: str
    loss_weight: float
    active_in_analysis: bool
    cancer_scope_status: str
    backbone_anchor_ids: str

    @property
    def supervised(self):
        return self.active_in_analysis and self.analysis_role in SUPERVISED_ROLES

    @property
    def primary_task(self):
        return {'pathway': 'Pathway', 'state': 'State', 'none': 'None'}[self.loss_group]


def policy_from_rows(rows):
    """Validate a complete policy; candidate overlap alone never admits a target."""
    policies = {}
    for raw in rows:
        missing = set(REQUIRED_FIELDS) - set(raw)
        if missing:
            raise ValueError('TARGET_POLICY_FIELDS_MISSING: ' + ','.join(sorted(missing)))
        nulls = [field for field in REQUIRED_FIELDS if raw[field] is None]
        if nulls:
            raise ValueError('TARGET_POLICY_NULL_FIELDS: ' + ','.join(nulls))
        data = {field: str(raw[field]).strip() for field in REQUIRED_FIELDS}
        target = data['canonical_id']
        if not target or target in policies:
            raise ValueError('TARGET_POLICY_ID_INVALID_OR_DUPLICATED: ' + target)
        if not data['biological_type'] or not data['cancer_scope_status']:
            raise ValueError('TARGET_POLICY_DEFINITION_MISSING: ' + target)
        roles = set(data['roles'].split('|'))
        if not roles or roles - {'Pathway', 'State'}:
            raise ValueError('TARGET_POLICY_DISPLAY_ROLE_INVALID: ' + target)
        analysis_role = data['analysis_role']
        group = data['loss_group']
        if analysis_role not in ROLE_GROUPS or group != ROLE_GROUPS[analysis_role]:
            raise ValueError('TARGET_POLICY_ANALYSIS_ROLE_GROUP_MISMATCH: ' + target)
        weight = explicit_task_weights([group], [raw['loss_weight']])[0]
        active = _bool(raw['active_in_analysis'])
        if active and (analysis_role == 'excluded' or data['cancer_scope_status'] in INACTIVE_STATUSES
                       or data['cancer_scope_status'].startswith('excluded_')):
            raise ValueError('TARGET_POLICY_INACTIVE_SCOPE_ACTIVATED: ' + target)
        if not active and (weight != 0 or analysis_role not in {'context_only', 'excluded'}):
            raise ValueError('TARGET_POLICY_INACTIVE_TARGET_HAS_LOSS: ' + target)
        policies[target] = TargetPolicy(target, data['roles'], data['biological_type'],
            analysis_role, group, weight, active, data['cancer_scope_status'],
            data['backbone_anchor_ids'])
    if not policies:
        raise ValueError('TARGET_POLICY_EMPTY')
    return policies


def load_target_policy(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError('TARGET_POLICY_FIELDS_MISSING: ' + ','.join(sorted(missing)))
        return policy_from_rows(reader)


def resolve_target_policy(catalog, definition=None, explicit=None):
    """Resolve a policy binding and verify its checksum when the catalog pins one."""
    root = Path(catalog).resolve()
    definition = definition or {}
    schema = definition.get('target_policy_schema')
    if schema is not None and schema != POLICY_SCHEMA:
        raise ValueError('TARGET_POLICY_SCHEMA_UNKNOWN: ' + str(schema))
    declared = definition.get('target_policy_file')
    candidate = explicit if explicit is not None else declared
    if candidate is None and (root / 'TARGET_POLICY.tsv').exists():
        candidate = 'TARGET_POLICY.tsv'
    if candidate is None:
        if definition.get('target_policy_required') or schema is not None or declared:
            raise ValueError('TARGET_POLICY_REQUIRED')
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise ValueError('TARGET_POLICY_FILE_MISSING: ' + str(path))
    expected = definition.get('target_policy_sha256')
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('TARGET_POLICY_CHECKSUM_MISMATCH')
    return load_target_policy(path)


def policy_for_supervised_axis(canonical_ids, policies):
    """Reject excluded, unresolved, context-only, or unbound label columns."""
    result = []
    seen = set()
    for canonical_id in canonical_ids:
        if canonical_id in seen:
            raise ValueError('TARGET_POLICY_DUPLICATE_AXIS: ' + canonical_id)
        seen.add(canonical_id)
        if canonical_id not in policies:
            raise ValueError('TARGET_POLICY_AXIS_ID_MISSING: ' + canonical_id)
        policy = policies[canonical_id]
        if not policy.supervised:
            raise ValueError('TARGET_POLICY_UNSUPERVISED_TARGET_ON_LABEL_AXIS: ' + canonical_id)
        result.append(policy)
    return tuple(result)


def active_target_ids(policies, *, analysis_role=None, loss_group=None):
    """Select supervised identities explicitly; display roles are never consulted."""
    return tuple(key for key, policy in policies.items() if policy.supervised
                 and (analysis_role is None or policy.analysis_role == analysis_role)
                 and (loss_group is None or policy.loss_group == loss_group))
