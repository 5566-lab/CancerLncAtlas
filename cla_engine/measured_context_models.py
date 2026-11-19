"""Explicit adapters for fixed measured Y; no relation-label training launch.

M0 and HHGT call the same target/layer/lineage selection. This module never
constructs Y from a score-context link and never registers an extra loss.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .semantic_context import linear_context_design, add_score_graph


def measured_m0_design(rna, covariates, context, task):
    if (not rna.index.equals(covariates.index)
            or not rna.index.equals(context['values'].index)):
        raise ValueError('M0_PATIENT_AXIS_MISMATCH')
    score, routes = linear_context_design(**context,task=task)
    x = pd.concat([rna.add_prefix('rna::'),covariates.add_prefix('covariate::'),score],axis=1)
    if not x.columns.is_unique or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError('M0_DESIGN_NONFINITE_OR_DUPLICATE')
    penalty = pd.Series(1.,index=x.columns)
    penalty.loc[covariates.add_prefix('covariate::').columns] = 0.
    return x, penalty, routes


def measured_m0_forward(design, coefficients, intercept=0.):
    if not design.columns.equals(coefficients.index):
        raise ValueError('M0_COEFFICIENT_AXIS_MISMATCH')
    return design.dot(coefficients) + float(intercept)


def measured_hhgt_graph(spec, arrays, context, task, patient_id):
    return add_score_graph(spec,arrays,**context,task=task,patient_id=patient_id)


def score_target_graph(spec, arrays, bindings, routes, selected_score_ids):
    """Scores used as Y: zero-valued scope nodes, pathway->score prediction edges.

    Only an explicit task selection enables an output. No patient Y value is
    accepted by this constructor. An output is stored once, even with many
    related mechanisms. These edges do not assert that the score equals a
    mechanism's activity or that a mechanism-specific event occurred.
    """
    import copy
    from .semantic_context import RELATIONS,semantic_graph_base
    ids=list(selected_score_ids)
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=set(bindings.score_id):
        raise ValueError('EXPLICIT_UNIQUE_SCORE_TARGET_SELECTION_REQUIRED')
    if 'score_target' in spec['nodes'] or 'source_score' in spec['nodes']:
        raise ValueError('SEPARATE_INPUT_AND_TARGET_SCORE_VIEWS_REQUIRED')
    new,out=semantic_graph_base(spec,arrays)
    new['nodes']['score_target']=ids;new['feature_dims']['score_target']=2
    out['x::score_target']=np.zeros((len(ids),2),np.float32)
    if any(k.startswith('node_omics::') for k in out):
        out['node_omics::score_target']=np.zeros((len(ids),4),np.float32)
    p={key:j for j,key in enumerate(new['nodes']['pathway'])};s={key:j for j,key in enumerate(ids)}
    chosen=routes[routes.score_id.isin(ids)&routes.target_id.isin(p)]
    for relation in RELATIONS:
        edge=('pathway','predicts_observation_'+relation,'score_target')
        new['relations'].append(edge);key='|'.join(edge)
        pairs=sorted({(p[r.target_id],s[r.score_id]) for r in chosen[chosen.relation.eq(relation)].itertuples()})
        out['edge::'+key]=np.asarray(pairs,dtype=np.int64).reshape(-1,2).T.copy()
        out['weight::'+key]=np.ones(len(pairs),np.float32)
    if not set(ids)<=set(chosen.score_id):raise ValueError('SCORE_TARGET_WITHOUT_SUPPORTED_SCOPE')
    new['score_target_contract']={'selected_score_ids':ids,'patient_y_as_input':False,
        'loss_registration':'explicit_caller_only','one_output_per_original_score':True}
    return new,out


def measured_m0_score_forward(design, coefficients, intercept):
    """One linear output per explicitly selected original score, not per route."""
    if (not design.columns.equals(coefficients.index) or not coefficients.columns.equals(intercept.index)
            or not coefficients.columns.is_unique):raise ValueError('M0_SCORE_OUTPUT_AXIS_MISMATCH')
    return design.dot(coefficients).add(intercept,axis=1)


def selected_score_loss(predictions, labels, observed, selected_score_ids):
    """Explicit, once-per-observation MSE; no implicit per-pathway replication.

    Caller selects existing Pathway outputs and combines this term within the
    Pathway loss budget (1.0); State retains 0.5. This is not an extra loss added
    on top by the dictionary. None/empty selection adds no supervision.
    """
    import torch
    if not selected_score_ids:return None
    if len(selected_score_ids)!=len(set(selected_score_ids)):
        raise ValueError('DUPLICATE_SUPERVISION_OBSERVATION')
    if predictions.shape!=labels.shape or labels.shape!=observed.shape or labels.shape[-1]!=len(selected_score_ids):
        raise ValueError('SCORE_SUPERVISION_AXIS_MISMATCH')
    mask=observed.bool()&torch.isfinite(labels)
    if not mask.any():return None
    # No mask multiplication of NaNs, and no degree multiplier from scope edges.
    return (predictions[mask]-labels[mask]).square().mean()
