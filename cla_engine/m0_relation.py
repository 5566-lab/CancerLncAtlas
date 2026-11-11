"""Matched, policy-scoped linear comparator for prepared CCHHGT relation tasks.

A row is cancer x lncRNA x exact pathway, not a patient's pathway score.
Only the existing prepared numeric block is an input. Graph arrays, graph logits,
and held-out correlation/p/q columns are never features. Fused-only catalogues
have no approved binary negatives and cannot be passed to this runner.
"""
from __future__ import annotations

import csv
import inspect
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .common import atomic_json, atomic_npz, digest, read_json, sha
from .exact_baseline import (decision_batched, fit_vocabulary, resource_plan,
                             training_design, unseen_audit)
from .execution_contract import baseline_contract
from .lineage import validate_prepared
from .local_execution import require_host_149
from .target_policy import active_target_ids, load_target_policy

FORMAT = 'CLA_M0_MATCHED_PREPARED_RELATION_V1'
DEFAULT_CANCERS = ('BRCA', 'LUAD', 'COAD')
_FIELDS = ('numeric', 'label', 'available', 'weight', 'direction_label',
           'direction_available', 'lnc', 'target')


def _identity_map(path):
    if path is None:
        return {}
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    mapping = {}
    for row in rows:
        source, target = row.get('source_target_id', '').strip(), row.get('canonical_id', '').strip()
        if not source or not target or source in mapping:
            raise ValueError('M0_IDENTITY_MAP_INVALID_OR_DUPLICATED')
        mapping[source] = target
    return mapping


def _target_axis(source_targets, policies, identity_map):
    canonical = tuple(identity_map.get(target, target) for target in source_targets)
    unknown = sorted(set(canonical) - set(policies))
    if unknown:
        raise ValueError('M0_TARGET_POLICY_ID_MISSING: ' + ','.join(unknown[:10]))
    # A rename does not fuse readouts. Many-to-one needs a new prepared target,
    # otherwise identical canonical rows would acquire conflicting labels.
    if len(set(canonical)) != len(canonical):
        raise ValueError('M0_MANY_TO_ONE_LABEL_REQUIRES_REMATERIALIZATION')
    allowed = set(active_target_ids(policies, analysis_role='primary_pathway'))
    admitted = tuple(target for target in canonical if target in allowed)
    if not admitted:
        raise ValueError('M0_NO_PRIMARY_PATHWAY_IN_PREPARED_AXIS')
    return canonical, admitted


def _select_rows(blocks, admitted):
    """Filter before scaler, vocabulary, label fitting or validation selection."""
    result = {key: [] for key in _FIELDS}
    identities = []
    allowed = set(admitted)
    for block in blocks:
        rows = block['identities']
        keep = np.asarray([row[2] in allowed for row in rows], bool)
        for key in _FIELDS:
            result[key].append(np.asarray(block[key])[keep])
        identities.extend(tuple(row) for row, take in zip(rows, keep) if take)
    if not blocks:
        raise ValueError('M0_REQUIRED_TRAIN_OR_VALIDATION_VIEW_MISSING')
    result = {key: np.concatenate(parts, axis=0) for key, parts in result.items()}
    result['identities'] = identities
    if not identities:
        raise ValueError('M0_EMPTY_POLICY_FILTERED_CANDIDATES')
    return result


def _metric(y, score, available):
    mask = np.asarray(available, bool)
    y = np.asarray(y)[mask]
    score = np.asarray(score)[mask]
    result = {'n': int(len(y)), 'positive': int(y.sum()), 'estimable': False}
    if np.unique(y).size != 2:
        return {**result, 'reason': 'NO_TWO_CLASS_EVALUATION'}
    probability = 1.0 / (1.0 + np.exp(-np.clip(score, -700, 700)))
    return {**result, 'estimable': True,
            'average_precision': float(average_precision_score(y, score)),
            'roc_auc': float(roc_auc_score(y, score)),
            'log_loss': float(log_loss(y, probability, labels=[0, 1]))}


def _classifier(C, penalty, cfg):
    args = {'C': float(C), 'solver': 'liblinear', 'max_iter': cfg['baseline']['max_iter'],
            'random_state': cfg['seed']}
    if inspect.signature(LogisticRegression).parameters['penalty'].default == 'deprecated':
        args['l1_ratio'] = 1.0 if penalty == 'l1' else 0.0
    else:
        args['penalty'] = penalty
    return LogisticRegression(**args)


def fit_scoped_cancer(train_blocks, validation_blocks, admitted, cfg, *, penalty='l1'):
    """Fit two linear heads with the already-declared CCHHGT labels/masks.

    Membership selects C by validation AP, as in the original linear base.
    Direction selects C by validation BCE, matching the direction loss.
    No test-label argument exists. This function does not alter prepared data.
    """
    if penalty not in {'l1', 'l2'}:
        raise ValueError('M0_PENALTY_MUST_BE_L1_OR_L2')
    train = _select_rows(train_blocks, admitted)
    valid = _select_rows(validation_blocks, admitted)
    cancers = {row[0] for row in train['identities'] + valid['identities']}
    if len(cancers) != 1:
        raise ValueError('M0_MUST_FIT_ONE_CANCER_AT_A_TIME')
    numeric_dim = train['numeric'].shape[1]
    if valid['numeric'].shape[1] != numeric_dim:
        raise ValueError('M0_NUMERIC_FEATURE_WIDTH_MISMATCH')
    bc = baseline_contract(cfg)
    heads = {}
    metrics = []
    for head in ('membership', 'direction'):
        label_key = 'label' if head == 'membership' else 'direction_label'
        train_ok = train['available'].astype(bool)
        valid_ok = valid['available'].astype(bool)
        if head == 'direction':
            train_ok &= train['direction_available'].astype(bool)
            valid_ok &= valid['direction_available'].astype(bool)
        y = train[label_key][train_ok]
        vy = valid[label_key][valid_ok]
        if not np.isin(y, [0, 1]).all() or not np.isin(vy, [0, 1]).all():
            raise ValueError('M0_PREPARED_BINARY_LABEL_REQUIRED')
        if np.unique(y).size != 2 or np.unique(vy).size != 2:
            heads[head] = {'status': 'NOT_ESTIMABLE', 'reason': 'NO_TWO_CLASS_TRAIN_OR_VALIDATION',
                           'train_rows': int(train_ok.sum()), 'validation_rows': int(valid_ok.sum())}
            continue
        budget = resource_plan(int(train_ok.sum()), numeric_dim, len(valid_ok), bc)
        train_rows = [row for row, keep in zip(train['identities'], train_ok) if keep]
        train_numeric = train['numeric'][train_ok]
        scaler = StandardScaler().fit(train_numeric)
        vectorizer, vocabulary = fit_vocabulary(train_rows, bc)
        design = training_design(train_rows, train_numeric, scaler, vectorizer, bc['batch_rows'])
        # Availability and weights are exactly the prepared endpoint's arrays.
        weights = train['weight'][train_ok].astype(float)
        weights /= weights.mean()
        best = None
        trials = []
        for C in cfg['baseline']['C']:
            model = _classifier(C, penalty, cfg)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error', ConvergenceWarning)
                    model.fit(design, y, sample_weight=weights)
            except ConvergenceWarning:
                trials.append({'C': float(C), 'status': 'NONCONVERGED_EXCLUDED'})
                continue
            logits = decision_batched(model, valid['identities'], valid['numeric'],
                                      scaler, vectorizer, bc['batch_rows'])
            pooled = _metric(valid[label_key], logits, valid_ok)
            selection = pooled['average_precision'] if head == 'membership' else -pooled['log_loss']
            trials.append({'C': float(C), 'selection_score': selection})
            if best is None or selection > best[0]:
                best = (selection, float(C), model, logits, pooled)
        if best is None:
            raise ValueError('M0_NO_CONVERGED_LINEAR_MODEL: ' + head)
        selection, C, model, logits, pooled = best
        heads[head] = {'status': 'FITTED', 'selected_C': C, 'trials': trials,
                      'selection': 'VALIDATION_POOLED_AP' if head == 'membership' else 'VALIDATION_NEGATIVE_BCE',
                      'pooled_validation': pooled, 'numeric_mean': scaler.mean_, 'numeric_scale': scaler.scale_,
                      'coef': model.coef_, 'intercept': model.intercept_, 'logits': logits,
                      'vocabulary': vectorizer.feature_names_, 'vocabulary_report': vocabulary,
                      'unseen_validation': unseen_audit(valid['identities'], vectorizer),
                      'resources': budget, 'train_rows': len(y)}
        target_names = np.asarray([row[2] for row in valid['identities']])
        for target in admitted:
            item = _metric(valid[label_key], logits, valid_ok & (target_names == target))
            metrics.append({'canonical_id': target, 'head': head, **item})
    return {'cancer_id': next(iter(cancers)), 'heads': heads, 'exact_target_metrics': metrics,
            'validation': valid, 'admitted_targets': tuple(admitted),
            'train_candidate_rows': len(train['identities']),
            'fit_input_contract': 'PREPARED_NUMERIC_PLUS_TRAIN_ONLY_EXACT_IDENTITIES_NO_GRAPH'}


def _load_blocks(prepared, spec, canonical, cancer):
    blocks = {'train': [], 'validation': []}
    inputs = {}
    for view in read_json(prepared / 'VIEWS.json'):
        if spec['cancers'][view['cancer_index']] != cancer:
            continue
        if view['split'] not in blocks:
            raise ValueError('M0_UNEXPECTED_SPLIT_IN_TRAINING_CAPSULE')
        folder = (prepared / 'views' / view['id']).resolve()
        if not folder.is_relative_to(prepared):
            raise ValueError('M0_VIEW_OUTSIDE_PREPARED')
        path = folder / 'pathway.npz'
        # Whitelist: no automatic concatenation of discovery_q, label p/q/effect,
        # base logits, graph outputs or arbitrary extra NPZ columns.
        with np.load(path, allow_pickle=False) as arrays:
            block = {key: np.array(arrays[key], copy=True) for key in _FIELDS}
        block['identities'] = [(cancer, spec['nodes']['lncRNA'][int(lnc)], canonical[int(target)])
                               for lnc, target in zip(block['lnc'], block['target'])]
        blocks[view['split']].append(block)
        inputs[str(path.relative_to(prepared))] = sha(path)
    return blocks, inputs


def run_m0_relation(prepared, policy, out, *, cancers=DEFAULT_CANCERS, identity_map=None, penalty='l1'):
    """Validate authoritative preparation; fit only requested cancers on host 149."""
    host = require_host_149()
    prepared, out = Path(prepared).resolve(), Path(out).resolve()
    if (prepared / 'Y_DEFINITION.json').exists() and not (prepared / 'SPEC.json').exists():
        raise ValueError('M0_FUSED_Y_REQUIRES_APPROVED_OBJECTIVE_AND_PREPARED_ADAPTER; '
                         'UNLABELLED_IS_NOT_NEGATIVE')
    if out == prepared or out.is_relative_to(prepared):
        raise ValueError('M0_OUTPUT_MUST_NOT_MODIFY_PREPARED_CAPSULE')
    if out.exists() and any(out.iterdir()):
        raise ValueError('M0_OUTPUT_ALREADY_EXISTS_USE_NEW_VERSION')
    cancers = tuple(cancers)
    if not cancers or len(set(cancers)) != len(cancers):
        raise ValueError('M0_CANCER_SCOPE_INVALID')
    policies = load_target_policy(policy)
    manifest, spec, cfg, lineage = validate_prepared(prepared)
    if set(cancers) - set(spec['cancers']):
        raise ValueError('M0_REQUESTED_CANCER_MISSING_FROM_PREPARED')
    canonical, admitted = _target_axis(spec['nodes']['pathway'], policies, _identity_map(identity_map))
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'LAUNCH_PREFLIGHT.json', {'target_host': '149', 'actual_host': host,
                'cancers': cancers, 'outer_fold': spec['fold'], 'paid_compute': False})
    receipt = {'format': FORMAT, 'target_host': host, 'cancers': cancers,
               'penalty': penalty, 'patient_score_regression': False, 'outer_test_read': False,
               'prepared_manifest_sha256': sha(prepared / 'MANIFEST.json'),
               'policy_sha256': sha(policy), 'identity_map_sha256': sha(identity_map) if identity_map else None,
               'outer_fold': spec['fold'], 'validation_fold': spec['validation_fold'],
               'train_folds': spec['train_folds'], 'admitted_targets': admitted,
               'excluded_prepared_targets': sorted(set(canonical) - set(admitted)),
               'primary_policy_targets_not_prepared': sorted(set(active_target_ids(policies, analysis_role='primary_pathway')) - set(canonical)),
               'input_features': 'AUTHORITATIVE_PREPARED_NUMERIC_PLUS_EXACT_IDENTITIES',
               'original_task_contract': read_json(prepared / 'TASK_CONTRACT.json'),
               'clinical_purity_adjustment': 'PRESERVED_IN_PREPARED_PATIENT_STAGE;NOT_FABRICATED_RELATION_COVARIATES',
               'graph_comparison_requires_same_scoped_candidates_features_labels_and_masks': True,
               'numeric_block_matched': True, 'graph_node_input_block_matched': False,
               'pure_graph_topology_ablation': False,
               'comparison_interpretation': 'NUMERIC_IDENTITY_LINEAR_VS_GRAPH_MODEL_WITH_NODE_CONTEXT',
               'fused_catalog_training_claimed': False, 'results': {}, 'inputs': {}}
    rows = []
    for cancer in cancers:
        blocks, inputs = _load_blocks(prepared, spec, canonical, cancer)
        fit = fit_scoped_cancer(blocks['train'], blocks['validation'], admitted, cfg, penalty=penalty)
        receipt['inputs'].update(inputs)
        folder = out / cancer
        folder.mkdir()
        summaries = {}
        for head, result in fit['heads'].items():
            if result['status'] != 'FITTED':
                summaries[head] = result
                continue
            arrays = {key: result[key] for key in ('coef', 'intercept', 'numeric_mean', 'numeric_scale')}
            atomic_npz(folder / (head + '_model.npz'), **arrays)
            atomic_json(folder / (head + '_vocabulary.json'), result['vocabulary'])
            valid = fit['validation']
            atomic_npz(folder / (head + '_validation_predictions.npz'),
                       canonical_id=np.asarray([r[2] for r in valid['identities']]),
                       lncrna_id=np.asarray([r[1] for r in valid['identities']]),
                       source_target_index=valid['target'], lnc_index=valid['lnc'],
                       logit=result['logits'], label=valid['label' if head == 'membership' else 'direction_label'],
                       available=valid['available'] if head == 'membership' else valid['available'] & valid['direction_available'])
            summaries[head] = {k: v for k, v in result.items() if k not in
                               {'coef', 'intercept', 'numeric_mean', 'numeric_scale', 'logits', 'vocabulary'}}
        receipt['results'][cancer] = summaries
        rows.extend({'cancer_id': cancer, **row} for row in fit['exact_target_metrics'])
    keys = ('cancer_id', 'canonical_id', 'head', 'n', 'positive', 'estimable',
            'average_precision', 'roc_auc', 'log_loss', 'reason')
    with (out / 'EXACT_PATHWAY_METRICS.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    receipt['fit_contract_sha256'] = digest({key: receipt[key] for key in
                                           ('prepared_manifest_sha256', 'policy_sha256', 'identity_map_sha256', 'cancers', 'penalty')})
    atomic_json(out / 'M0_RELATION_RECEIPT.json', receipt)
    return receipt
