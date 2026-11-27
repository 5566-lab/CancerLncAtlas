"""Portable measured Y access: separate observation, eligibility and conditions."""
from pathlib import Path
import hashlib
import json
import pandas as pd


def supervision_mask(observed, eligible, applicable):
    if not observed.columns.equals(eligible.index) or not observed.index.equals(applicable.index):
        raise ValueError('Y_MASK_AXIS_MISMATCH')
    return observed.mul(eligible.astype(bool), axis=1).mul(applicable.eq(1), axis=0).astype(bool)


def load_fixed_target(root, case, canonical_id, *, for_supervision=False):
    root=Path(root); folder=root/'cohorts'/case
    receipt=json.loads((folder/'COMPLETE.json').read_text())
    for name in ['OBSERVATIONS.tsv.gz','TARGET_OBSERVATIONS.tsv.gz','Y_VALUES.parquet','Y_OBSERVED.parquet','TARGET_APPLICABILITY.parquet']:
        actual=hashlib.sha256((folder/name).read_bytes()).hexdigest()
        if actual!=receipt['files'][name]:raise ValueError('FIXED_Y_FILE_DRIFT')
    targets=pd.read_csv(root/'TARGETS.tsv',sep='\t',keep_default_na=False)
    if canonical_id not in set(targets.canonical_id):raise KeyError(canonical_id)
    obs=pd.read_csv(folder/'OBSERVATIONS.tsv.gz',sep='\t',keep_default_na=False)
    links=pd.read_csv(folder/'TARGET_OBSERVATIONS.tsv.gz',sep='\t',keep_default_na=False)
    ids=set(links.loc[links.canonical_id.eq(canonical_id),'observation_id'])
    metadata=obs[obs.observation_id.isin(ids)].copy();columns=metadata.observation_id.tolist()
    values=pd.read_parquet(folder/'Y_VALUES.parquet',columns=columns)
    observed=pd.read_parquet(folder/'Y_OBSERVED.parquet',columns=columns)
    applicable=pd.read_parquet(folder/'TARGET_APPLICABILITY.parquet',columns=[canonical_id])[canonical_id]
    mask=supervision_mask(observed,metadata.set_index('observation_id').supervision_eligible,applicable) if for_supervision else observed
    return values,mask,applicable,metadata


def load_target_with_context(root, context_root, case, canonical_id, observation_id, *, allowed_input_layers):
    """Bind one original supervised observation to optional semantic INPUT context.

    Target values, masks and applicability are returned unchanged. Callers must
    explicitly declare available input layers; a pure lncRNA baseline passes [].
    """
    from .semantic_context import ContextTask, read_context_bundle
    values, mask, applicable, metadata = load_fixed_target(root,case,canonical_id,for_supervision=True)
    meta = metadata.set_index('observation_id')
    if observation_id not in meta.index: raise ValueError('OBSERVATION_NOT_IN_TARGET')
    row = meta.loc[observation_id]
    layer = row['layer']
    if layer not in {'RNA','PROTEIN','PHOSPHO','METHYLATION','CNV','MUTATION'}:
        raise ValueError('UNRESOLVED_TARGET_DEPENDENCIES')
    lineage = frozenset(str(row[k]) for k in ['definition_id','feature_id','source_target_id'] if row[k])
    task = ContextTask(canonical_id,observation_id,frozenset({layer}),frozenset(allowed_input_layers),lineage)
    context = read_context_bundle(context_root,case)
    if not context['values'].index.equals(values.index): raise ValueError('CONTEXT_PATIENT_AXIS_MISMATCH')
    return {'y':values[observation_id], 'mask':mask[observation_id], 'applicable':applicable,
            'task':task, 'context':context, 'target_metadata':row.to_dict()}
