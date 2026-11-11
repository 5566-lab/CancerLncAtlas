import csv
import hashlib
import json
import numpy as np
import pytest
from cla_engine.target_policy import (REQUIRED_FIELDS, POLICY_SCHEMA,
    active_target_ids, load_target_policy, policy_for_supervised_axis,
    policy_from_rows, resolve_target_policy)
from cla_engine.functional_y import functional_task_weights, load_functional_context
from test_functional_y import make_context
from test_functional_fusion import write_context


def policy_row(target='P1', **updates):
    row = dict(canonical_id=target, roles='Pathway', biological_type='pathway',
               analysis_role='primary_pathway', loss_group='pathway', loss_weight=1.,
               active_in_analysis=True, cancer_scope_status='backbone',
               backbone_anchor_ids='hsa05200')
    row.update(updates)
    return row


def write_policy(root, rows, *, required=False):
    path = root / 'TARGET_POLICY.tsv'
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_FIELDS, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    if required:
        definition_path = root / 'Y_DEFINITION.json'
        definition = json.loads(definition_path.read_text()) if definition_path.exists() else {}
        definition.update(target_policy_schema=POLICY_SCHEMA, target_policy_required=True,
            target_policy_file=path.name,
            target_policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        definition_path.write_text(json.dumps(definition))
    return path


def rehash_axes(base):
    receipt_path = base / 'COMPLETE.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['files']['AXES.json'] = hashlib.sha256((base / 'AXES.json').read_bytes()).hexdigest()
    receipt_path.write_text(json.dumps(receipt))


def test_display_category_does_not_route_explicit_auxiliary_loss():
    weights = functional_task_weights(['Pathway', 'Pathway', 'State'],
        loss_groups=['pathway', 'state', 'state'], loss_weights=[1, .5, .5])
    np.testing.assert_array_equal(weights, [1, .5, .5])
    policies = policy_from_rows([policy_row(), policy_row('TF1', biological_type='TF',
        analysis_role='regulator_auxiliary', loss_group='state', loss_weight=.5)])
    assert active_target_ids(policies, analysis_role='primary_pathway') == ('P1',)
    assert policies['TF1'].roles == 'Pathway'
    assert policies['TF1'].primary_task == 'State'


@pytest.mark.parametrize('updates,match', [
    ({'loss_weight': .5}, 'GROUP_WEIGHT'),
    ({'loss_weight': float('nan')}, 'GROUP_WEIGHT'),
    ({'loss_group': 'state', 'loss_weight': .5}, 'ANALYSIS_ROLE_GROUP'),
    ({'cancer_scope_status': 'candidate_overlap_only'}, 'INACTIVE_SCOPE'),
    ({'cancer_scope_status': 'excluded_zero_overlap'}, 'INACTIVE_SCOPE'),
    ({'active_in_analysis': False}, 'INACTIVE_TARGET_HAS_LOSS'),
    ({'active_in_analysis': 'maybe'}, 'INVALID_BOOLEAN'),
])
def test_invalid_or_implicitly_activated_policies_rejected(updates, match):
    with pytest.raises(ValueError, match=match):
        policy_from_rows([policy_row(**updates)])


def test_inactive_candidates_never_become_label_columns(tmp_path):
    path = write_policy(tmp_path, [policy_row(), policy_row('candidate',
        analysis_role='context_only', loss_group='none', loss_weight=0,
        active_in_analysis=False, cancer_scope_status='candidate_overlap_only')])
    policies = load_target_policy(path)
    assert active_target_ids(policies) == ('P1',)
    with pytest.raises(ValueError, match='UNSUPERVISED_TARGET_ON_LABEL_AXIS'):
        policy_for_supervised_axis(['P1', 'candidate'], policies)
    with pytest.raises(ValueError, match='AXIS_ID_MISSING'):
        policy_for_supervised_axis(['unknown'], policies)


def test_required_policy_cannot_fall_back_to_role_inference(tmp_path):
    with pytest.raises(ValueError, match='TARGET_POLICY_REQUIRED'):
        resolve_target_policy(tmp_path, {'target_policy_required': True})
    write_policy(tmp_path, [policy_row()], required=True)
    definition = json.loads((tmp_path / 'Y_DEFINITION.json').read_text())
    with (tmp_path / 'TARGET_POLICY.tsv').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='CHECKSUM_MISMATCH'):
        resolve_target_policy(tmp_path, definition)


def test_legacy_context_loads_with_explicit_policy_and_rejects_excluded_axis(tmp_path):
    make_context(tmp_path)
    rows = [policy_row('T' + str(i)) for i in range(5)]
    write_policy(tmp_path, rows, required=True)
    context = load_functional_context(tmp_path, 'BRCA', 'sealed_test')
    np.testing.assert_array_equal(context.task_weights(), np.ones(5))
    rows[-1].update(analysis_role='excluded', loss_group='none', loss_weight=0,
                   active_in_analysis=False, cancer_scope_status='excluded_zero_overlap')
    write_policy(tmp_path, rows, required=True)
    with pytest.raises(ValueError, match='UNSUPERVISED_TARGET_ON_LABEL_AXIS'):
        load_functional_context(tmp_path, 'BRCA', 'sealed_test')


def test_fused_context_separates_tf_display_from_loss_routing(tmp_path):
    base = write_context(tmp_path)
    axes_path = base / 'AXES.json'
    axes = json.loads(axes_path.read_text())
    axes['target_roles'][0] = 'Pathway'  # TF moved from State display.
    axes_path.write_text(json.dumps(axes))
    rehash_axes(base)
    rows = [policy_row('S0', biological_type='TF', analysis_role='regulator_auxiliary',
                       loss_group='state', loss_weight=.5),
            policy_row('P1'), policy_row('S2', roles='State', biological_type='phenotype',
                       analysis_role='state_auxiliary', loss_group='state', loss_weight=.5),
            policy_row('P3', roles='Pathway|State')]
    write_policy(tmp_path, rows, required=True)
    context = load_functional_context(tmp_path, 'BRCA', 'train_label_fold2')
    np.testing.assert_array_equal(context.task_weights(), [.5, 1, .5, 1])
    assert context.target_roles[0] == 'Pathway'
    assert context.loss_groups[0] == 'state'
    axes['primary_tasks'][0] = 'Pathway'  # A display-driven rematerializer is rejected.
    axes_path.write_text(json.dumps(axes))
    rehash_axes(base)
    with pytest.raises(ValueError, match='LOSS_GROUP_AXIS_MISMATCH'):
        load_functional_context(tmp_path, 'BRCA', 'train_label_fold2')


def test_explicit_weight_axis_requires_both_fields():
    with pytest.raises(ValueError, match='GROUP_AND_WEIGHT_REQUIRED'):
        functional_task_weights(['Pathway'], loss_groups=['state'])
    with pytest.raises(ValueError, match='WEIGHT_AXIS_MISMATCH'):
        functional_task_weights(['Pathway'], loss_groups=['pathway','state'], loss_weights=[1,.5])


def test_policy_missing_values_are_not_stringified_to_valid_fields():
    with pytest.raises(ValueError, match='NULL_FIELDS'):
        policy_from_rows([policy_row(biological_type=None)])
    with pytest.raises(ValueError, match='Unknown functional target role'):
        functional_task_weights(['unknown'], loss_groups=['state'], loss_weights=[.5])
