"""Matched, policy-scoped linear comparator for prepared CCHHGT relation tasks.

A row is cancer x lncRNA x exact pathway, not a patient's pathway score.
Only the existing prepared numeric block is an input. Graph arrays, graph logits,
and held-out correlation/p/q columns are never features. Numeric-only is the
default; exact relation identity terms are an explicit additional ablation.
The existing source-patient discovery statistics remain in numeric. Fused-only catalogues
have no approved binary negatives and cannot be passed to this runner.
"""
from __future__ import annotations

import csv
import inspect
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from .common import atomic_json, atomic_npz, digest, read_json, sha
from .exact_baseline import (decision_batched, design_block, fit_vocabulary, resource_plan,
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



def _prepared_scope_inputs(prepared, spec, policy, identity_map):
    """Use the policy and source-ID map sealed by the actual prepared pipeline."""
    receipt_path = prepared / 'RELATION_TARGET_POLICY.json'
    nested_policy = prepared / 'relation_target_policy' / 'TARGET_POLICY.tsv'
    if receipt_path.exists() or nested_policy.exists():
        if not receipt_path.is_file() or sha(receipt_path) != spec.get('relation_target_policy_sha256'):
            raise ValueError('M0_PREPARED_POLICY_RECEIPT_MISMATCH')
        receipt = read_json(receipt_path)
        if receipt.get('format') != 'CLA_PREPARED_RELATION_TARGET_POLICY_V1':
            raise ValueError('M0_PREPARED_POLICY_RECEIPT_FORMAT')
        sealed_map = prepared / 'relation_target_policy' / 'SOURCE_IDENTITY_MAP.tsv'
        for path in (nested_policy, sealed_map):
            item = receipt.get('files', {}).get(str(path.relative_to(prepared)))
            if not item or not path.is_file() or sha(path) != item.get('sha256'):
                raise ValueError('M0_PREPARED_POLICY_FILE_MISMATCH: ' + str(path))
        if sha(nested_policy) != sha(policy):
            raise ValueError('M0_PREPARED_TARGET_POLICY_MISMATCH')
        if identity_map is not None and sha(identity_map) != sha(sealed_map):
            raise ValueError('M0_PREPARED_IDENTITY_MAP_MISMATCH')
        return nested_policy, sealed_map
    legacy = prepared / 'TARGET_POLICY.tsv'
    if legacy.exists() and sha(legacy) != sha(policy):
        raise ValueError('M0_PREPARED_TARGET_POLICY_MISMATCH')
    return legacy if legacy.exists() else None, identity_map


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


def _strength_metrics(y, prediction, available, identities, admitted, families=None):
    mask = np.asarray(available, dtype=bool)
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if not np.isfinite(prediction).all() or not np.isfinite(y[mask]).all():
        raise ValueError('M0_STRENGTH_NONFINITE')
    if np.any(y[mask] < 0) or np.any(y[mask] > 1 + 1e-6):
        raise ValueError('M0_STRENGTH_LABEL_RANGE')
    targets = np.asarray([row[2] for row in identities])
    rows = []
    for target in admitted:
        take = mask & (targets == target)
        n = int(take.sum())
        row = {'canonical_id': target, 'head': 'strength', 'n': n, 'estimable': n >= 2}
        if n >= 2:
            row['mse'] = float(np.mean((y[take] - prediction[take]) ** 2))
            row['mae'] = float(np.mean(np.abs(y[take] - prediction[take])))
        rows.append(row)
    good = [row for row in rows if row['estimable']]
    if families is not None:
        if set(admitted)-set(families):
            raise ValueError('M0_STATE_FAMILY_MISSING')
        means=[np.mean([row['mse'] for row in good if families[row['canonical_id']]==family])
               for family in sorted({families[row['canonical_id']] for row in good})]
        return rows, float(np.mean(means)) if means else None
    return rows, float(np.mean([row['mse'] for row in good])) if good else None


def fit_strength_cancer(train_blocks, validation_blocks, admitted, cfg, *, identity_mode='numeric_only', families=None):
    """Fit observed fused effects with an L1 linear head and exact-target selection."""
    if identity_mode not in {'numeric_only', 'exact'}:
        raise ValueError('M0_IDENTITY_MODE_INVALID')
    train = _select_rows(train_blocks, admitted)
    valid = _select_rows(validation_blocks, admitted)
    cancers = {row[0] for row in train['identities'] + valid['identities']}
    if len(cancers) != 1:
        raise ValueError('M0_MUST_FIT_ONE_CANCER_AT_A_TIME')
    if valid['numeric'].shape[1] != train['numeric'].shape[1]:
        raise ValueError('M0_NUMERIC_FEATURE_WIDTH_MISMATCH')
    train_ok = train['available'].astype(bool)
    valid_ok = valid['available'].astype(bool)
    if not train_ok.any() or not valid_ok.any():
        raise ValueError('M0_NO_OBSERVED_STRENGTH_TRAIN_OR_VALIDATION')
    for data, mask in ((train, train_ok), (valid, valid_ok)):
        y = data['label'][mask]
        if not np.isfinite(y).all() or np.any(y < 0) or np.any(y > 1 + 1e-6):
            raise ValueError('M0_STRENGTH_LABEL_RANGE')
    bc = {**baseline_contract(cfg), 'identity_mode': identity_mode}
    resource_plan(int(train_ok.sum()), train['numeric'].shape[1], len(valid_ok), bc)
    train_rows = [row for row, take in zip(train['identities'], train_ok) if take]
    scaler = StandardScaler().fit(train['numeric'][train_ok])
    vectorizer, vocabulary = fit_vocabulary(train_rows, bc)
    design = training_design(train_rows, train['numeric'][train_ok], scaler,
                             vectorizer, bc['batch_rows'])
    weights = train['weight'][train_ok].astype(float)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('M0_AVAILABLE_WEIGHTS_MUST_BE_POSITIVE_FINITE')
    weights /= weights.mean()
    y_train = train['label'][train_ok]
    centered = y_train - np.average(y_train, weights=weights)
    alpha_max = float(np.max(np.abs(design.T @ (weights * centered))) / weights.sum())
    alpha_max = max(alpha_max, 1e-8)
    best = None
    trials = []
    for C in cfg['baseline']['C']:
        if not np.isfinite(C) or C <= 0:
            raise ValueError('M0_STRENGTH_C_MUST_BE_POSITIVE')
        alpha = alpha_max / float(C)
        model = Lasso(alpha=alpha, max_iter=cfg['baseline']['max_iter'],
                      random_state=cfg['seed'])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(design, y_train, sample_weight=weights)
        except ConvergenceWarning:
            trials.append({'C': float(C), 'alpha': alpha, 'status': 'NONCONVERGED_EXCLUDED'})
            continue
        prediction = np.empty(len(valid['identities']), dtype=float)
        for start in range(0, len(prediction), bc['batch_rows']):
            stop = min(start + bc['batch_rows'], len(prediction))
            prediction[start:stop] = model.predict(design_block(
                valid['identities'][start:stop], valid['numeric'][start:stop],
                scaler, vectorizer))
        rows, macro_mse = _strength_metrics(valid['label'], prediction, valid_ok,
                                            valid['identities'], admitted, families=families)
        trials.append({'C': float(C), 'alpha': alpha,
                       'validation_exact_target_macro_mse': macro_mse})
        if macro_mse is not None and (best is None or macro_mse < best[0]):
            best = (macro_mse, float(C), alpha, model, prediction, rows)
    if best is None:
        raise ValueError('M0_NO_ESTIMABLE_CONVERGED_STRENGTH_MODEL')
    macro_mse, C, alpha, model, prediction, rows = best
    head = {'status': 'FITTED', 'selected_C': C, 'selected_alpha': alpha,
            'selection': ('VALIDATION_EXACT_TARGET_THEN_FAMILY_MACRO_MSE' if families is not None
                          else 'VALIDATION_EXACT_TARGET_MACRO_MSE'),
            'validation_macro_mse': macro_mse, 'trials': trials,
            'coef': model.coef_[None, :], 'intercept': np.asarray([model.intercept_]),
            'numeric_mean': scaler.mean_, 'numeric_scale': scaler.scale_,
            'prediction': prediction, 'vocabulary': vectorizer.feature_names_,
            'vocabulary_report': vocabulary,
            'unseen_validation': unseen_audit(valid['identities'], vectorizer),
            'train_rows': int(train_ok.sum())}
    return {'cancer_id': next(iter(cancers)), 'heads': {'strength': head},
            'exact_target_metrics': rows, 'validation': valid,
            'admitted_targets': tuple(admitted),
            'train_candidate_rows': len(train['identities']),
            'identity_mode': identity_mode,
            'fit_input_contract': 'PREPARED_NUMERIC_NO_GRAPH' if identity_mode == 'numeric_only'
                                  else 'PREPARED_NUMERIC_PLUS_TRAIN_ONLY_EXACT_IDENTITIES_NO_GRAPH'}


def fit_scoped_cancer(train_blocks, validation_blocks, admitted, cfg, *, penalty='l1', identity_mode='numeric_only'):
    """Fit two linear heads with the already-declared CCHHGT labels/masks.

    Membership selects C by validation AP, as in the original linear base.
    Direction selects C by validation BCE, matching the direction loss.
    No test-label argument exists. This function does not alter prepared data.
    """
    if identity_mode not in {'numeric_only', 'exact'}:
        raise ValueError('M0_IDENTITY_MODE_INVALID')
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
    bc = {**baseline_contract(cfg), 'identity_mode': identity_mode}
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
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('M0_AVAILABLE_WEIGHTS_MUST_BE_POSITIVE_FINITE')
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
                      'identity_mode': identity_mode,
                      'vocabulary': vectorizer.feature_names_,
                      'vocabulary_report': vocabulary,
                      'unseen_validation': unseen_audit(valid['identities'], vectorizer),
                      'resources': budget, 'train_rows': len(y)}
        target_names = np.asarray([row[2] for row in valid['identities']])
        for target in admitted:
            item = _metric(valid[label_key], logits, valid_ok & (target_names == target))
            metrics.append({'canonical_id': target, 'head': head, **item})
    return {'cancer_id': next(iter(cancers)), 'heads': heads, 'exact_target_metrics': metrics,
            'validation': valid, 'admitted_targets': tuple(admitted),
            'train_candidate_rows': len(train['identities']),
            'identity_mode': identity_mode,
            'fit_input_contract': 'PREPARED_NUMERIC_NO_GRAPH' if identity_mode == 'numeric_only' else 'PREPARED_NUMERIC_PLUS_TRAIN_ONLY_EXACT_IDENTITIES_NO_GRAPH'}


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


def run_m0_relation(prepared, policy, out, *, cancers=DEFAULT_CANCERS, identity_map=None, penalty='l1', identity_mode='numeric_only'):
    """Validate authoritative preparation; fit only requested cancers on host 149."""
    host = require_host_149()
    prepared, out = Path(prepared).resolve(), Path(out).resolve()
    # This legacy relation wrapper cannot implement the measured-Y M0.
    if (prepared / 'SPEC.json').exists():
        declared = read_json(prepared / 'SPEC.json')
        if declared.get('relation_objective') == 'observed_fused_rms_effect':
            raise ValueError('ASSOCIATION_STATISTICS_ARE_NOT_MEASURED_Y; '
                             'FIX_MEASURED_Y_BEFORE_M0')
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
    objective = spec.get('relation_objective', 'binary')
    if objective == 'observed_fused_rms_effect':
        raise ValueError('ASSOCIATION_STATISTICS_ARE_NOT_MEASURED_Y; '
                         'FIX_MEASURED_Y_BEFORE_M0')
    if objective != 'binary':
        raise ValueError('M0_RELATION_OBJECTIVE_UNKNOWN')
    if set(cancers) - set(spec['cancers']):
        raise ValueError('M0_REQUESTED_CANCER_MISSING_FROM_PREPARED')
    bound_policy, identity_map = _prepared_scope_inputs(prepared, spec, policy, identity_map)
    canonical, admitted = _target_axis(spec['nodes']['pathway'], policies, _identity_map(identity_map))
    if 'pathway_canonical_ids' in spec and tuple(spec['pathway_canonical_ids']) != canonical:
        raise ValueError('M0_PREPARED_CANONICAL_AXIS_MISMATCH')
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'LAUNCH_PREFLIGHT.json', {'target_host': '149', 'actual_host': host,
                'cancers': cancers, 'outer_fold': spec['fold'], 'paid_compute': False})
    receipt = {'format': FORMAT, 'target_host': host, 'cancers': cancers,
               'relation_objective': objective,
               'penalty': penalty, 'identity_mode': identity_mode,
               'target_policy_bound_in_prepared': bound_policy is not None, 'patient_score_regression': False, 'outer_test_read': False,
               'prepared_manifest_sha256': sha(prepared / 'MANIFEST.json'),
               'policy_sha256': sha(policy), 'identity_map_sha256': sha(identity_map) if identity_map else None,
               'outer_fold': spec['fold'], 'validation_fold': spec['validation_fold'],
               'train_folds': spec['train_folds'], 'admitted_targets': admitted,
               'excluded_prepared_targets': sorted(set(canonical) - set(admitted)),
               'primary_policy_targets_not_prepared': sorted(set(active_target_ids(policies, analysis_role='primary_pathway')) - set(canonical)),
               'input_features': 'AUTHORITATIVE_PREPARED_NUMERIC' if identity_mode == 'numeric_only' else 'AUTHORITATIVE_PREPARED_NUMERIC_PLUS_EXACT_IDENTITIES',
               'source_relation_statistics_in_numeric': True,
               'source_relation_statistics': (['mean_rho','rms_rho','max_abs_rho','mean_log1p_n','log1p_readout_count','available']
                    if objective=='observed_fused_rms_effect' else ['rho','abs_rho','neglog10_q','log1p_n','available']),
               'label_patient_relation_statistics_as_features': False,
               'original_task_contract': read_json(prepared / 'TASK_CONTRACT.json'),
               'clinical_purity_adjustment': 'PRESERVED_IN_PREPARED_PATIENT_STAGE;NOT_FABRICATED_RELATION_COVARIATES',
               'graph_comparison_requires_same_scoped_candidates_features_labels_and_masks': True,
               'numeric_block_matched': True, 'graph_node_input_block_matched': False,
               'pure_graph_topology_ablation': False,
               'comparison_interpretation': ('NUMERIC_LINEAR_VS_GRAPH_MODEL_WITH_NODE_CONTEXT' if identity_mode == 'numeric_only' else 'NUMERIC_IDENTITY_LINEAR_VS_GRAPH_MODEL_WITH_NODE_CONTEXT'),
               'fused_catalog_training_claimed': objective=='observed_fused_rms_effect', 'results': {}, 'inputs': {}}
    rows = []
    for cancer in cancers:
        blocks, inputs = _load_blocks(prepared, spec, canonical, cancer)
        fit = (fit_strength_cancer(blocks['train'], blocks['validation'], admitted, cfg,
                                   identity_mode=identity_mode)
               if objective == 'observed_fused_rms_effect' else
               fit_scoped_cancer(blocks['train'], blocks['validation'], admitted, cfg,
                                 penalty=penalty, identity_mode=identity_mode))
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
            prediction = {'score': result['prediction']} if head == 'strength' else {'logit': result['logits']}
            atomic_npz(folder / (head + '_validation_predictions.npz'),
                       canonical_id=np.asarray([r[2] for r in valid['identities']]),
                       lncrna_id=np.asarray([r[1] for r in valid['identities']]),
                       source_target_index=valid['target'], lnc_index=valid['lnc'],
                       label=valid['label' if head in {'membership', 'strength'} else 'direction_label'],
                       available=valid['available'] if head in {'membership', 'strength'} else valid['available'] & valid['direction_available'],
                       **prediction)
            summaries[head] = {k: v for k, v in result.items() if k not in
                               {'coef', 'intercept', 'numeric_mean', 'numeric_scale', 'logits', 'prediction', 'vocabulary'}}
        receipt['results'][cancer] = summaries
        rows.extend({'cancer_id': cancer, **row} for row in fit['exact_target_metrics'])
    keys = ('cancer_id', 'canonical_id', 'head', 'n', 'positive', 'estimable',
            'average_precision', 'roc_auc', 'log_loss', 'mse', 'mae', 'reason')
    with (out / 'EXACT_PATHWAY_METRICS.tsv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, delimiter='\t')
        writer.writeheader()
        writer.writerows(rows)
    receipt['fit_contract_sha256'] = digest({key: receipt[key] for key in
                                           ('prepared_manifest_sha256', 'policy_sha256', 'identity_map_sha256', 'cancers', 'penalty', 'identity_mode', 'relation_objective')})
    atomic_json(out / 'M0_RELATION_RECEIPT.json', receipt)
    return receipt
