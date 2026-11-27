"""Evidence-defined score routing. Composition similarity never creates a route.

Patient scores remain observations at their original scope. These links provide
optional INPUT context only; they do not create labels or auxiliary losses.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
import pandas as pd

RELATIONS = ('same_scope', 'broader_scope_context', 'narrower_scope_context',
             'shared_process_context', 'regulator_context')
ROUTE_COLUMNS = ['score_id', 'target_id', 'relation', 'evidence', 'evidence_basis',
                 'source_scope', 'independent_loss_enabled', 'copy_to_target_y']


def compile_routes(bindings, scope_relations, target_ids):
    """Only explicit identity or reviewed, directed scope relations; no fuzzy match.

    No transitive closure over 'shared process', nor over similarity clusters.
    One stored score may have multiple routes, without duplicating its value.
    """
    if bindings.score_id.duplicated().any():
        raise ValueError('DUPLICATE_SCORE_ID')
    targets = set(target_ids)
    routes = []
    allowed = {'official_identity', 'official_hierarchy', 'curated_functional_scope',
               'reviewed_perturbation_scope', 'reviewed_regulator_mechanism'}
    for r in scope_relations.to_dict('records'):
        if r['relation'] not in RELATIONS or r['evidence_basis'] not in allowed or not r['evidence']:
            raise ValueError('UNSUPPORTED_SEMANTIC_EVIDENCE')
    by_scope = {}
    for r in scope_relations.to_dict('records'):
        if r['target_id'] in targets:
            by_scope.setdefault(r['source_scope'], []).append(r)
    for b in bindings.to_dict('records'):
        scope = b['source_scope']
        candidates = list(by_scope.get(scope, []))
        if scope in targets:
            candidates.append(dict(source_scope=scope, target_id=scope, relation='same_scope',
                                   evidence=b['binding_evidence'], evidence_basis='official_identity'))
        for r in candidates:
            routes.append(dict(score_id=b['score_id'], target_id=r['target_id'], relation=r['relation'],
                evidence=r['evidence'], evidence_basis=r['evidence_basis'], source_scope=scope,
                independent_loss_enabled=False, copy_to_target_y=False))
    out = pd.DataFrame(routes, columns=ROUTE_COLUMNS)
    return out.drop_duplicates(['score_id','target_id','relation']).sort_values(
        ['score_id','target_id','relation']).reset_index(drop=True)


@dataclass(frozen=True)
class ContextTask:
    target_id: str
    target_observation_id: str
    target_input_layers: frozenset[str]
    allowed_input_layers: frozenset[str]
    forbidden_lineage_ids: frozenset[str]

    def __post_init__(self):
        if not self.target_id or not self.target_observation_id or not self.target_input_layers:
            raise ValueError('TARGET_DEPENDENCIES_REQUIRED')
        if 'UNRESOLVED' in self.target_input_layers:
            raise ValueError('UNRESOLVED_TARGET_DEPENDENCIES')


def select_routes(bindings, routes, task):
    """Same exclusion for M0 and HHGT, BEFORE linear or graph aggregation."""
    if bindings.score_id.duplicated().any():
        raise ValueError('DUPLICATE_SCORE_ID')
    available = set()
    for b in bindings.to_dict('records'):
        layers = frozenset(json.loads(b['input_layers']))
        lineage = frozenset(json.loads(b['lineage_ids']))
        if (not layers or 'UNRESOLVED' in layers or not layers <= task.allowed_input_layers
                or layers & task.target_input_layers
                or lineage & (task.forbidden_lineage_ids | {task.target_observation_id})
                or b['score_id'] == task.target_observation_id):
            continue
        available.add(b['score_id'])
    return routes[routes.target_id.eq(task.target_id) & routes.score_id.isin(available)].copy()


def fit_reference_scaler(values, reference_patients, query_patients):
    """Freeze only reference-patient moments. Missing is never an observed zero."""
    ref = list(reference_patients); query = list(query_patients)
    if (not ref or len(set(ref)) != len(ref) or set(ref) & set(query)
            or not set(ref + query) <= set(values.index) or not values.index.is_unique
            or not values.columns.is_unique):
        raise ValueError('PATIENT_PARTITION_OR_AXIS_INVALID')
    x = values.loc[ref].replace([np.inf,-np.inf], np.nan)
    count = x.count(); sd = x.std(ddof=0)
    return pd.DataFrame({'mean':x.mean(), 'scale':sd.where(sd.gt(1e-12),1.),
                         'reference_count':count, 'usable':count.ge(2) & sd.gt(1e-12)})


def transformed_context(values, scaler):
    if not values.columns.equals(scaler.index):
        raise ValueError('CONTEXT_SCALER_AXIS_MISMATCH')
    # Use checked positional axes; Arrow-backed Index/Series arithmetic can
    # realign nullable metadata unexpectedly after a TSV/parquet round trip.
    raw = values.to_numpy(dtype=float,na_value=np.nan)
    mean = pd.to_numeric(scaler['mean'],errors='raise').to_numpy(dtype=float,na_value=np.nan)
    scale = pd.to_numeric(scaler['scale'],errors='raise').to_numpy(dtype=float,na_value=np.nan)
    usable = scaler['usable'].to_numpy()
    if not np.isin(usable,[True,False]).all():raise ValueError('SCALER_USABILITY_NOT_BOOLEAN')
    observed = np.isfinite(raw) & usable.astype(bool)[None,:]
    z = np.where(observed,(raw-mean[None,:])/scale[None,:],0.).astype('float32')
    return (pd.DataFrame(z,index=values.index,columns=values.columns),
            pd.DataFrame(observed,index=values.index,columns=values.columns))


def semantic_graph_base(spec, arrays):
    """Drop only old overlap-created scope edges, retaining molecular memberships."""
    updated=copy.deepcopy(spec);out=dict(arrays)
    excluded={'mechanism_context','rev_mechanism_context'}
    removed=[r for r in updated['relations'] if r[1] in excluded]
    updated['relations']=[r for r in updated['relations'] if r[1] not in excluded]
    for r in removed:
        key='|'.join(r)
        out.pop('edge::'+key,None);out.pop('weight::'+key,None)
    updated['excluded_overlap_scope_relations']=[list(r) for r in removed]
    return updated,out


def linear_context_design(values, scaler, bindings, routes, task):
    selected = select_routes(bindings,routes,task)
    columns = sorted(set(selected.score_id))
    z, mask = transformed_context(values[columns],scaler.loc[columns])
    out = pd.DataFrame(index=values.index)
    for key in columns:
        out['score::'+key] = z[key]
        out['observed::'+key] = mask[key].astype('float32')
    return out, selected


def add_score_graph(spec, arrays, values, scaler, bindings, routes, task, patient_id):
    """Patient-specific, task-masked score nodes consumed by the original HGT.

    All score nodes have stable IDs. Ineligible and absent scores have zero
    features AND no outgoing context edge. There are no reverse score edges.
    """
    if 'source_score' in spec['nodes']:
        raise ValueError('SCORE_CONTEXT_ALREADY_ATTACHED')
    if task.target_id not in spec['nodes']['pathway']:
        raise ValueError('TARGET_NOT_IN_GRAPH')
    ids = sorted(bindings.score_id.tolist())
    if not set(ids) <= set(values.columns):
        raise ValueError('MISSING_SCORE_AXIS')
    z, mask = transformed_context(values.loc[[patient_id],ids],scaler.loc[ids])
    selected = select_routes(bindings,routes,task)
    permitted = set(selected.score_id)
    x = np.zeros((len(ids),2),np.float32)
    for j,sid in enumerate(ids):
        if sid in permitted and bool(mask.at[patient_id,sid]):
            x[j] = [z.at[patient_id,sid],1.]
    updated, out = semantic_graph_base(spec,arrays)
    updated['nodes']['source_score'] = ids
    updated['feature_dims']['source_score'] = 2
    out['x::source_score'] = x
    if any(k.startswith('node_omics::') for k in out):
        out['node_omics::source_score'] = np.zeros((len(ids),4),np.float32)
    si = {k:i for i,k in enumerate(ids)}; ti = {k:i for i,k in enumerate(spec['nodes']['pathway'])}
    for relation in RELATIONS:
        edge = ('source_score', 'score_context_'+relation, 'pathway')
        if list(edge) not in updated['relations'] and edge not in updated['relations']:
            updated['relations'].append(edge)
        rows = selected[selected.relation.eq(relation)]
        pairs = [(si[r.score_id],ti[r.target_id]) for r in rows.itertuples()
                 if bool(mask.at[patient_id,r.score_id])]
        key = '|'.join(edge)
        out['edge::'+key] = np.asarray(pairs,dtype=np.int64).reshape(-1,2).T.copy()
        out['weight::'+key] = np.ones(len(pairs),np.float32)
    updated['score_context_contract'] = {'role':'input_context', 'independent_loss':False,
        'target_y_copied':False, 'score_nodes':len(ids), 'patient_specific':True,
        'exclusion':'target_modality_and_lineage', 'edge_weight':'unit_not_jaccard'}
    return updated, out


def read_context_bundle(root, case):
    root = Path(root); folder = root/'cohorts'/case
    receipt = json.loads((folder/'COMPLETE.json').read_text())
    for name,digest in receipt['files'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
            raise ValueError('SCORE_CONTEXT_FILE_DRIFT')
    read = lambda name: pd.read_csv(folder/name,sep='\t',keep_default_na=False)
    return {'values':pd.read_parquet(folder/'SOURCE_VALUES.parquet'),
            'scaler':pd.read_csv(folder/'REFERENCE_SCALER.tsv',sep='\t').set_index('score_id'),
            'bindings':read('SCORE_BINDINGS.tsv'), 'routes':read('SCORE_PATHWAY_CONTEXT.tsv')}
