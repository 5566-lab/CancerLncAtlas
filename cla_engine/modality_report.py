"""Validation-only input inventory, matched LASSO refits and frozen-branch sensitivity.

No refit is performed unless explicitly requested. No optimizer/GPU launcher here.
Positive deltas mean full minus removed, not a causal effect or an outer-test gain.
"""
from pathlib import Path
import socket

import numpy as np
import pandas as pd

from .baseline_compare import _assemble, _fit_ablation, _score, load_npz
from .common import atomic_json, commit_manifest, fail, read_json, sha, status, verify_manifest
from .exact_baseline import fit_vocabulary
from .execution_contract import baseline_contract
from .lineage import validate_prepared
from .metrics import paired_task_metrics


def feature_blocks(spec):
    """Follow prepare's serialized numeric layout; reject unknown layouts."""
    sc=int(spec.get('singlecell_association_dim', 0))
    if sc and list(spec.get('singlecell_association_slice', [])) != [5, 5+sc]:
        fail('MODALITY_SCHEMA_DRIFT', 'Unexpected single-cell association slice')
    lnc=5+sc; width=int(spec['feature_dims']['lncRNA']); node=lnc+width
    mutation=node+int(spec['sc_dim'])+1; target=mutation+4
    genomic=bool(spec.get('genomics_enabled'))
    expected=target+(15 if genomic else 0)
    if width != (18 if genomic else 6) or expected != spec['numeric_features']:
        fail('MODALITY_SCHEMA_DRIFT', 'Numeric layout must be explicitly mapped before reporting')
    blocks={
        'discovery_association': list(range(5)),
        'singlecell_association': list(range(5, 5+sc)),
        'lncrna_expression': list(range(lnc, lnc+3)),
        'lncrna_cnv': list(range(lnc+3, lnc+6)),
        'partner_mutation': list(range(lnc+6, node)),
        'singlecell_node_context': list(range(node, mutation)),
        'lncrna_mutation': list(range(mutation, target)),
        'pathway_mutation': list(range(target, target+12)) if genomic else [],
        'pathway_cnv': list(range(target+12, target+15)) if genomic else [],
    }
    if sorted(i for columns in blocks.values() for i in columns) != list(range(expected)):
        fail('MODALITY_SCHEMA_DRIFT', 'Feature groups must partition numeric columns')
    return blocks


def variation(train, validation, columns):
    if not columns:
        return {'varying_columns': 0, 'status': 'NOT_IN_NUMERIC_INPUT'}
    values=np.concatenate([train[:, columns], validation[:, columns]])
    n=int(np.count_nonzero(np.ptp(values, axis=0)>1e-8))
    return {'varying_columns': n, 'status': 'VARIABLE_INPUT' if n else 'CONSTANT_OR_UNAVAILABLE',
            'note': 'Variation is not measurement availability; zeros can be observed biological values.'}


def metric_rows(task, data, spec, full, removed, modality, evidence, k=100):
    """Paired AP/P@K; preserve registered State target->family->cancer averaging."""
    targets=spec['nodes']['pathway'] if task=='pathway' else spec['states']
    lookup={name:i for i,name in enumerate(targets)}
    tid=np.asarray([lookup[name] for name in data['target_name']])
    rows=[]; cancer_metrics=[]
    def add(level, cancer, name, metric):
        row=dict(task=task, modality=modality, evidence=evidence, level=level,
                 cancer_id=cancer, unit_id=name, available=bool(metric.get('available', True)))
        for key in ('n', 'n_positive', 'k_effective', 'reason'):
            if key in metric: row[key]=metric[key]
        if row['available']:
            for key in ('auprc', 'precision_at_k'):
                row['full_'+key]=metric[key]; row['removed_'+key]=metric['baseline_'+key]
                row['delta_'+key]=metric[key]-metric['baseline_'+key]
        rows.append(row)
    for cancer in sorted(set(data['cancer'])):
        mask=data['cancer']==cancer
        result=paired_task_metrics(data['label'][mask], np.asarray(full)[mask], np.asarray(removed)[mask],
            data['available'][mask], tid[mask], task=task, targets=targets,
            families=spec.get('state_families'), k=k)
        add('cancer', cancer, cancer, result)
        if result['available']: cancer_metrics.append(result)
        if task=='state':
            for item in result.get('targets', []): add('target', cancer, item['target_id'], item)
            for item in result.get('families', []): add('family', cancer, item['family'], item)
        else:
            for name in sorted(set(data['target_name'][mask])):
                take=mask & (data['target_name']==name)
                unit=paired_task_metrics(data['label'][take], np.asarray(full)[take], np.asarray(removed)[take],
                    data['available'][take], tid[take], task='pathway', targets=targets, k=k)
                add('target', cancer, name, unit)
    overall={'available': bool(cancer_metrics), 'reason': 'NO_EVALUABLE_CANCER'}
    if cancer_metrics:
        overall={key:float(np.mean([m[key] for m in cancer_metrics]))
                 for key in ('auprc','baseline_auprc','precision_at_k','baseline_precision_at_k')}
    add('overall', 'ALL', 'CANCER_MACRO', overall)
    return rows


def _checkpoint_model(prepared, training, spec, cfg):
    from .model import JointHGT
    from .train import load_checkpoint, model_signature
    from .lineage import verify_completion
    training=Path(training)
    if (training/'DIAGNOSTIC_ARM_RECEIPT.json').exists():
        verify_manifest(training, name='RESULT_MANIFEST.json')
        receipt=read_json(training/'DIAGNOSTIC_ARM_RECEIPT.json')
        contract=read_json(training/'RUN_CONTRACT.json')
        if (receipt.get('state')!='DIAGNOSTIC_ARM_COMPLETE_NOT_FORMAL'
            or receipt.get('prepared_manifest_sha256')!=sha(prepared/'MANIFEST.json')
            or receipt.get('signature')!=contract.get('signature')
            or receipt.get('optimizer_steps_total')!=216):
            fail('MODALITY_RUN_MISMATCH', 'Require a completed, bound diagnostic run')
        checkpoint,record=load_checkpoint(training, receipt['signature'])
        if (record!=receipt['last_checkpoint'] or checkpoint.get('next_step')!=216
            or checkpoint.get('format')!='CLA_R62_DIAGNOSTIC_ARM_STATE_V1'):
            fail('MODALITY_RUN_MISMATCH', 'Require the committed fixed-epoch checkpoint')
        weights=checkpoint['model']; alpha=1.; selection='FIXED_EPOCH24_ALPHA1'
    else:
        receipt,_=verify_completion(prepared, training)
        signature=model_signature(prepared,cfg,spec)
        if receipt['signature']!=signature: fail('MODALITY_RUN_MISMATCH', 'Signature mismatch')
        checkpoint,_=load_checkpoint(training,signature)
        weights=checkpoint['best']['model']; alpha=receipt['alpha']; selection='REGISTERED_SELECTED_CHECKPOINT'
    model=JointHGT(spec,cfg['model']).to('cpu'); model.load_state_dict(weights); model.eval()
    return model, alpha, {'selection': selection, 'alpha': alpha, 'run': str(training), 'signature':receipt['signature']}


def report(prepared, out, refit_lasso=False, training=None):
    if socket.gethostname()!='149': fail('WRONG_HOST','Modality reports execute on 149 only')
    prepared=Path(prepared); out=Path(out)
    if out.exists() and any(out.iterdir()): fail('OUTPUT_EXISTS','Never overwrite a contribution report')
    _,spec,cfg,lineage=validate_prepared(prepared)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out/'LAUNCH_PREFLIGHT.json', {'target_host':'149','refit_lasso':refit_lasso,'neural_training':False})
    views=read_json(prepared/'VIEWS.json'); blocks=feature_blocks(spec)
    inventory=[]; rows=[]; fits={}; assembled={}
    for task in ('pathway','state'):
        data=_assemble(prepared, views, spec, task); assembled[task]=data
        train=data['train']; validation=data['validation']
        for name,columns in blocks.items():
            item={'task':task,'modality':name,'columns':columns,
                  **variation(train['numeric'], validation['numeric'], columns)}
            inventory.append(item)
        if not refit_lasso: continue
        take=np.flatnonzero(train['available'])
        vectorizer,_=fit_vocabulary([train['rows'][i] for i in take],baseline_contract(cfg))
        columns=list(range(spec['numeric_features']))
        full_fit,full=_fit_ablation(task,data,spec,cfg,vectorizer,columns,True,'full',True)
        fits[task]={'full':full_fit,'stored_baseline_macro_ap':_score(task,validation,spec,validation['base'])}
        # Use a matched refit as reference, not historical logits with different solver state.
        for name,removed_columns in blocks.items():
            info=next(i for i in inventory if i['task']==task and i['modality']==name)
            if info['status']!='VARIABLE_INPUT': continue
            kept=[i for i in columns if i not in removed_columns]
            fitted,ablated=_fit_ablation(task,data,spec,cfg,vectorizer,kept,True,name,True)
            fits[task][name]=fitted
            rows.extend(metric_rows(task,validation,spec,full,ablated,name,'MATCHED_LASSO_REFIT'))
    run_record=None
    if training is not None:
        from .train import evaluate
        model,alpha,run_record=_checkpoint_model(prepared,training,spec,cfg)
        args=(model,prepared,views,spec,'cpu',cfg['training']['batch_size'],alpha)
        _,full=evaluate(*args)
        for port in ('singlecell_association','singlecell','lncrna_cnv','pathway_cnv',
                     'lncrna_mutation','partner_mutation','pathway_mutation','mutation',
                     'reported_binding','reported_PPI','coexpression'):
            _,removed=evaluate(*args,disabled=frozenset([port]))
            for task in ('pathway','state'):
                for left,right in zip(full[task],removed[task],strict=True):
                    if left[0]!=right[0] or any(not np.array_equal(left[1][key],right[1][key])
                                               for key in ('lnc','target','label','available')):
                        fail('MODALITY_ROW_DRIFT','Frozen inference changed candidate ordering')
                full_score=np.concatenate([item[2] for item in full[task]])
                removed_score=np.concatenate([item[2] for item in removed[task]])
                rows.extend(metric_rows(task,assembled[task]['validation'],spec,full_score,removed_score,
                                        port,'FROZEN_DEEP_BRANCH_BASELINE_RETAINED'))
    contract={'format':'CLA_R62_MODALITY_REPORT_V1','target_host':'149','prepared_manifest_sha256':sha(prepared/'MANIFEST.json'),
        'lineage':lineage,'outer_test_read':False,'neural_training_started':False,
        'lasso_refits':'COMPLETE' if refit_lasso else 'NOT_RUN_WAITING_CONFIRMATION',
        'frozen_deep_branch':run_record or 'NOT_RUN_NO_CHECKPOINT',
        'full_multimodal_retraining':'NOT_RUN; use matched preparations and identical schedules',
        'graph_retraining':'G1-G0 binding; G2-G1 PPI; use diagnostic-report after all joint runs complete',
        'delta_definition':'FULL_MINUS_REMOVED; positive is descriptive improvement on validation',
        'limitations':['Validation is used to tune C; this is not outer-test evidence or statistical significance.',
          'Frozen-branch results retain the multimodal LASSO; they are not total modality removal.',
          'Zeroing input can be out of distribution; sensitivity does not prove retraining benefit.',
          'Constant/unavailable numeric groups are not refitted or assigned a benefit.',
          'Effects overlap and must not be added; gate magnitudes are not predictive gains.',
          'Partner CNV, ATAC, drug/clinical/evidence annotations are not separate primary numeric inputs here.']}
    atomic_json(out/'REPORT_CONTRACT.json',contract)
    atomic_json(out/'FEATURE_INVENTORY.json',inventory)
    atomic_json(out/'LASSO_REFITS.json',fits)
    fields=['task','modality','evidence','level','cancer_id','unit_id','available','n','n_positive','k_effective','reason',
            'full_auprc','removed_auprc','delta_auprc','full_precision_at_k','removed_precision_at_k','delta_precision_at_k']
    pd.DataFrame(rows,columns=fields).to_csv(out/'MODALITY_CONTRIBUTIONS.tsv',sep='\t',index=False)
    commit_manifest(out,{'role':'VALIDATION_MODALITY_REPORT','outer_test_read':False})
    return status(out,'MODALITY_REPORT_COMPLETE_NOT_TRAINED')
