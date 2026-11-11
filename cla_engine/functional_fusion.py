"""Load fused cancer-lncRNA-functional-target evidence without inventing negatives."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from .target_policy import resolve_target_policy, policy_for_supervised_axis

FORMAT='CLA_FUNCTIONAL_RELATION_FUSED_V1'
ARRAY_NAMES=('combined_p','combined_q','rms_effect','max_abs_effect','observed','membership',
 'observed_assay_bits','individually_strict_assay_bits','individually_supported_assay_bits',
 'cohort_count','individually_supported_cohort_count','calibration_method','direction',
 'readout_count','distinct_readout_count','positive_evidence_count','negative_evidence_count')

@dataclass(frozen=True)
class FusedFunctionalContext:
    cancer_id: str
    context: str
    lncrna_ids: tuple[str,...]
    canonical_ids: tuple[str,...]
    target_roles: tuple[str,...]
    primary_tasks: tuple[str,...]
    cohort_splits: dict
    arrays: dict[str,np.ndarray]
    target_policies: tuple | None = None

    @property
    def loss_groups(self):
        if self.target_policies is not None:
            return tuple(p.loss_group for p in self.target_policies)
        return tuple(task.lower() for task in self.primary_tasks)

    def counts(self):
        a=self.arrays;m=a['membership'];obs=a['observed'].astype(bool)
        return {'cancer_id':self.cancer_id,'context':self.context,'canonical_targets':len(self.canonical_ids),
          'observed_relations':int(obs.sum()),'observed_targets':int(obs.any(0).sum()),
          'strong_relations':int((m==2).sum()),'weak_relations':int((m==1).sum()),
          'supported_targets':int((m>0).any(0).sum()),'unlabelled_observed':int((obs&(m==0)).sum()),
          'joint_cohort_relations':int((a['cohort_count']==2).sum()),
          'mixed_direction_supported_relations':int((a['direction']==3).sum())}

    def supervision_masks(self):
        observed=self.arrays['observed'].astype(bool);membership=self.arrays['membership']
        return {'observed':observed,'positive':membership>0,'strong_positive':membership==2,
          'weak_positive':membership==1,'unlabelled':observed&(membership==0),
          'confirmed_negative':np.zeros(observed.shape,bool)}

    def binary_labels(self):
        raise ValueError('UNLABELLED_IS_NOT_NEGATIVE: select an explicit positive-unlabelled or continuous-evidence objective')

    def task_weights(self):
        if self.target_policies is not None:
            return np.asarray([p.loss_weight for p in self.target_policies], np.float32)
        if set(self.primary_tasks) - {'Pathway', 'State'}:
            raise ValueError('FUSED_Y_TASK_GROUP_INVALID')
        return np.asarray([1.0 if task=='Pathway' else .5 for task in self.primary_tasks],np.float32)


def validate_fused_arrays(arrays,shape):
    if set(arrays)!=set(ARRAY_NAMES):raise ValueError('FUSED_Y_SCHEMA_MISMATCH')
    for name,value in arrays.items():
        if value.shape!=shape:raise ValueError('FUSED_Y_AXIS_MISMATCH: '+name)
    a=arrays;observed=a['observed'].astype(bool);p=a['combined_p'];q=a['combined_q'];m=a['membership']
    if not np.isin(a['observed'],[0,1]).all() or not np.isin(m,[0,1,2]).all():raise ValueError('FUSED_Y_MEMBERSHIP_CODE_INVALID')
    if not np.array_equal(np.isfinite(p),observed) or not np.array_equal(np.isfinite(q),observed):raise ValueError('FUSED_Y_MISSINGNESS_MISMATCH')
    if np.any((p[observed]<0)|(p[observed]>1)|(q[observed]<0)|(q[observed]>1)):raise ValueError('FUSED_Y_PQ_OUT_OF_RANGE')
    if np.any(q[observed]+1e-12<p[observed]):raise ValueError('FUSED_Y_Q_BELOW_P')
    for name in ['rms_effect','max_abs_effect']:
        value=a[name]
        if not np.array_equal(np.isfinite(value),observed) or np.any((value[observed]<0)|(value[observed]>1+1e-6)):raise ValueError('FUSED_Y_EFFECT_INVALID: '+name)
    strong=observed&(q<=.05)&(a['max_abs_effect']>=.20)
    weak=observed&(q<=.10)&(a['max_abs_effect']>=.15)&~strong
    if not np.array_equal(m,np.where(strong,2,np.where(weak,1,0))):raise ValueError('FUSED_Y_SUPPORT_CRITERION_MISMATCH')
    if np.any(a['individually_supported_assay_bits']&~a['observed_assay_bits']) or np.any(a['individually_strict_assay_bits']&~a['individually_supported_assay_bits']):raise ValueError('FUSED_Y_ASSAY_SUPPORT_WITHOUT_OBSERVATION')
    if not np.array_equal(a['observed_assay_bits']>0,observed):raise ValueError('FUSED_Y_ASSAY_OBSERVATION_MISMATCH')
    if np.any(a['individually_supported_cohort_count']>a['cohort_count']) or np.any(a['cohort_count']>2):raise ValueError('FUSED_Y_COHORT_COUNT_INVALID')
    if not np.array_equal(a['cohort_count']>0,observed) or np.any(a['distinct_readout_count']>a['readout_count']):raise ValueError('FUSED_Y_READOUT_COUNT_INVALID')
    if not np.array_equal(a['direction']==0,m==0) or np.any(a['direction']>3):raise ValueError('FUSED_Y_DIRECTION_INVALID')
    if not np.array_equal(a['calibration_method']>0,observed) or np.any(a['calibration_method']>4):raise ValueError('FUSED_Y_CALIBRATION_INVALID')


def load_fused_functional_context(catalog,cancer_id,context,*,target_policy=None):
    root=Path(catalog).resolve();base=(root/cancer_id/context).resolve()
    if not base.is_relative_to(root) or base==root:raise ValueError('FUSED_Y_CONTEXT_OUTSIDE_CATALOG')
    receipt=json.loads((base/'COMPLETE.json').read_text());definition=json.loads((root/'Y_DEFINITION.json').read_text())
    if receipt.get('format')!=FORMAT or definition.get('format')!=FORMAT:raise ValueError('FUSED_Y_FORMAT_MISMATCH')
    if receipt.get('cancer_id')!=cancer_id or receipt.get('context')!=context:raise ValueError('FUSED_Y_CONTEXT_MISMATCH')
    if definition.get('task_weights')!={'Pathway':1.0,'State':.5}:raise ValueError('FUSED_Y_TASK_WEIGHT_DRIFT')
    for name in ['AXES.json','SPLIT.json','Y.npz','INPUT_REFERENCES.json','DEPENDENCE_REFERENCES.json','READOUT_INDEX.tsv','MULTIPLICITY.json']:
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=receipt['files'][name]:raise ValueError('FUSED_Y_CHECKSUM_MISMATCH: '+name)
    ax=json.loads((base/'AXES.json').read_text());splits=json.loads((base/'SPLIT.json').read_text())
    for field in ['lncrna_ids','canonical_ids']:
        if len(set(ax[field]))!=len(ax[field]) or not all(isinstance(x,str) and x for x in ax[field]):raise ValueError('FUSED_Y_IDENTITIES_INVALID: '+field)
    if len(ax['target_roles'])!=len(ax['canonical_ids']) or len(ax['primary_tasks'])!=len(ax['canonical_ids']):raise ValueError('FUSED_Y_ROLE_AXIS_MISMATCH')
    policies = resolve_target_policy(root, definition, target_policy)
    target_policies = None if policies is None else policy_for_supervised_axis(ax['canonical_ids'], policies)
    if target_policies is None:
        expected=['Pathway' if 'Pathway' in role.split('|') else 'State' for role in ax['target_roles']]
        if expected!=ax['primary_tasks']:raise ValueError('FUSED_Y_SHARED_IDENTITY_DOUBLE_COUNTED')
    else:
        if [p.roles for p in target_policies] != ax['target_roles']:
            raise ValueError('TARGET_POLICY_DISPLAY_ROLE_AXIS_MISMATCH')
        if [p.primary_task for p in target_policies] != ax['primary_tasks']:
            raise ValueError('TARGET_POLICY_LOSS_GROUP_AXIS_MISMATCH')
        if 'loss_weights' in ax and [p.loss_weight for p in target_policies] != ax['loss_weights']:
            raise ValueError('TARGET_POLICY_LOSS_WEIGHT_AXIS_MISMATCH')
        if 'loss_groups' in ax and [p.loss_group for p in target_policies] != ax['loss_groups']:
            raise ValueError('TARGET_POLICY_LOSS_GROUP_AXIS_MISMATCH')
    if set(splits)!={'TCGA','CPTAC'}:raise ValueError('FUSED_Y_COHORT_SPLITS_MISSING')
    for ds,split in splits.items():
        if set(split['source_patients'])&set(split['label_patients']):raise ValueError('FUSED_Y_PATIENT_ROLE_OVERLAP: '+ds)
        if ds=='CPTAC' and cancer_id=='BRCA' and 'CPTAC:BRCA:604' in split['source_patients']+split['label_patients']:raise ValueError('FUSED_Y_UNRESOLVED_RETROSPECTIVE_CASE')
    with np.load(base/'Y.npz',allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    validate_fused_arrays(arrays,(len(ax['lncrna_ids']),len(ax['canonical_ids'])))
    result=FusedFunctionalContext(cancer_id,context,tuple(ax['lncrna_ids']),tuple(ax['canonical_ids']),tuple(ax['target_roles']),tuple(ax['primary_tasks']),splits,arrays,target_policies)
    for key,value in result.counts().items():
        if key in receipt and receipt[key]!=value:raise ValueError('FUSED_Y_COUNT_MISMATCH: '+key)
    return result
