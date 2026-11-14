"""Molecular graph views computed only from frozen source patients."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from .common import sha, digest
from .graph import coexpression, graph_npz
from .relation_assays import append_assay_nodes, load_source_assays
from .relation_components import load_relation_components
from .relation_features import COHORTS, entity_summary, verified_file, read_exclusions
from .relation_source import (load_relation_lncrna_sources, load_source_matrix,
    load_source_covariates, load_source_local_cnv, fit_source_design)


def _static(path):
    root = Path(path)
    receipt = json.loads((root / 'COMPLETE.json').read_text())
    if receipt.get('format') != 'CLA_RELATION_STATIC_GRAPH_V1' or receipt.get('target_host') != '149':
        raise ValueError('RELATION_VIEW_STATIC_GRAPH_FORMAT')
    for name, expected in receipt['files'].items():
        if Path(name).name != name or sha(root / name) != expected:
            raise ValueError('RELATION_VIEW_STATIC_GRAPH_DRIFT')
    ids = json.loads((root / 'NODES.json').read_text())
    with np.load(root / 'STATIC_EDGES.npz', allow_pickle=False) as f:
        arrays = dict(f)
    edges = {key[6:]: {'edge_index': value, 'weight': arrays['weight::' + key[6:]],
                      'context': arrays['context::' + key[6:]], 'sign': 1}
             for key, value in arrays.items() if key.startswith('edge::')}
    return ids, edges, receipt


def gene_node_omics_for_new_axis(binding, ids, cancer, source_patients):
    """Retain new graph genes as unavailable when the verified product lacks them."""
    from .node_omics import VerifiedGeneNodeOmics, graph_node_omics
    complete = verified_file(binding)
    receipt = json.loads(complete.read_text())
    gene_file = receipt['genes']
    path = Path(gene_file['path'])
    if not path.is_absolute():
        path = complete.parent / path
    if sha(path) != gene_file['sha256']:
        raise ValueError('RELATION_VIEW_OMICS_GENE_AXIS_DRIFT')
    available_axis = set(pd.read_parquet(path).gene_id.astype(str))
    measured_axis = [g for g in ids['gene'] if g in available_axis]
    if not measured_axis:
        raise ValueError('RELATION_VIEW_OMICS_GENE_INTERSECTION_EMPTY')
    config = {'enabled': True, 'channels': ['cnv_signed', 'methylation_promoter_beta'], 'complete': binding}
    reader = VerifiedGeneNodeOmics(config, measured_axis, [cancer])
    subset, audit = reader.build_view(cancer, sorted(source_patients))
    block = np.zeros((len(ids['gene']), 4), np.float32)
    index = {g: i for i, g in enumerate(ids['gene'])}
    block[[index[g] for g in measured_axis]] = subset
    audit['cohort'] = 'TCGA'
    audit['new_graph_genes_absent_from_source_product'] = sorted(set(ids['gene']) - available_axis)
    audit['source_product_gene_axis_changed'] = False
    audit['input_hashes'] = reader.input_hashes()
    return graph_node_omics(ids, block), audit


def build_relation_graph_view(component_root, static_root, molecular_bindings, *, topk=8, progress=None):
    """Build cohort-specific edges without consuming the component label arrays."""
    from .local_execution import require_host_149
    require_host_149()
    component = load_relation_components(component_root)
    metadata = component['receipt']
    ids, static, static_receipt = _static(static_root)
    if metadata['cancer'] != static_receipt['cancer'] or ids['lncRNA'] != component['axes']['lncrna_ids']:
        raise ValueError('RELATION_VIEW_STATIC_AXIS_MISMATCH')
    cancer = metadata['cancer']
    context = SimpleNamespace(cancer_id=cancer, context=metadata['context'],
        lncrna_ids=tuple(ids['lncRNA']), cohort_splits=component['splits'])
    binding = component['preflight']['bindings']
    cfg = json.loads(verified_file(binding['adjustment_config']).read_text())
    annotation = pd.read_csv(verified_file(molecular_bindings['rna_annotation']), sep='\t')
    coding = set(annotation.loc[annotation.gene_biotype.eq('protein_coding'), 'gene_id'])
    expressions = load_relation_lncrna_sources(context, binding['aligned_root'])
    features = {kind: np.zeros((len(axis), 6), np.float32) for kind, axis in ids.items()}
    features['lncRNA'] = np.column_stack([component['lncrna_node_features'][c] for c in COHORTS])
    dynamic, coexpression_audit, support, gene_stats = {}, {}, {}, []
    for cohort in COHORTS:
        if progress:
            progress({'stage': 'source_gene_graph', 'cohort': cohort, 'cancer': cancer, 'context': context.context})
        genes = load_source_matrix(context, binding['aligned_root'], cohort, 'GENE_LOGTPM.parquet', ids['gene'])
        values = genes.values.copy()
        values[:, [i for i, g in enumerate(ids['gene']) if g not in coding]] = np.nan
        gene_stats.append(entity_summary(values, np.isfinite(values)))
        cov = load_source_covariates(context, binding['covariate_root'], cohort, binding['covariate_sha256'][cohort])
        cnv = load_source_local_cnv(context, binding['cnv_roots'], cohort, ids['lncRNA'])
        design = fit_source_design(cov, cfg, family='pathway', target_id='__pathway__')
        _, mixed = read_exclusions(binding['direct_rna_members'][cohort], binding['mixed_rna_members'])
        x = expressions[cohort].values.copy()
        excluded = [i for i, g in enumerate(ids['lncRNA']) if g in mixed]
        x[:, excluded] = np.nan
        edges, audit = coexpression(x, values, cnv.values, cnv.callable, design.values,
            np.arange(len(genes.patients)), topk=topk, min_n=int(cfg['minimum_pairs']),
            cnv_transform='raw_signed', return_audit=True, cache_gene_residuals=True,
            progress=(lambda item: progress({'stage':'coexpression_progress','cohort':cohort,**item})) if progress else None)
        support[cohort] = audit.pop('edge_support')
        audit.update(source_patient_hash=digest(list(genes.patients)), cohort=cohort,
            excluded_mixed_coding_lncrnas=len(excluded), gene_expression_sha256=genes.source_sha256,
            gene_expression_restricted_to_protein_coding=True, adjustment=design.report)
        coexpression_audit[cohort] = audit
        dynamic[cohort] = edges
        if progress:
            progress({'stage': 'source_gene_graph_complete', 'cohort': cohort,
                      'selected_edges': audit['selected_edges'], 'estimable_pairs': audit['estimable_pairs']})
    features['gene'] = np.column_stack(gene_stats)
    # The historical helper's measured_in edges inspect column 2. Replace them
    # below with explicit separate-cohort measured edges.
    arrays = graph_npz(ids, static, cancer, {}, features)
    for key in list(arrays):
        if '|measured_in|' in key or '|rev_measured_in|' in key:
            del arrays[key]
    for cohort_index, cohort in enumerate(COHORTS):
        for sign, (index, weights) in dynamic[cohort].items():
            key = f'lncRNA|{cohort.lower()}_coexpression_{sign}|gene'
            rev = f'gene|rev_{cohort.lower()}_coexpression_{sign}|lncRNA'
            arrays['edge::' + key], arrays['weight::' + key] = index, weights
            arrays['edge::' + rev], arrays['weight::' + rev] = index[::-1].copy(), weights.copy()
        for kind, offset in (('lncRNA', 6 * cohort_index + 2), ('gene', 3 * cohort_index + 2)):
            measured = np.flatnonzero(features[kind][:, offset] > 0)
            index = np.vstack([measured, np.zeros(len(measured), np.int64)])
            key = f'{kind}|{cohort.lower()}_measured_in|cancer'
            rev = f'cancer|rev_{cohort.lower()}_measured_in|{kind}'
            arrays['edge::' + key], arrays['weight::' + key] = index, np.ones(len(measured), np.float32)
            arrays['edge::' + rev], arrays['weight::' + rev] = index[::-1].copy(), np.ones(len(measured), np.float32)
    omics, omics_audit = gene_node_omics_for_new_axis(molecular_bindings['gene_node_omics'], ids,
                                                    cancer, context.cohort_splits['TCGA']['source_patients'])
    arrays.update({'node_omics::' + kind: values for kind, values in omics.items()})
    assays = load_source_assays(context, molecular_bindings)
    assay_audit = append_assay_nodes(ids, arrays, assays)
    feature_names = {kind: ['unused_' + str(i) for i in range(v.shape[1])]
                     for kind, v in features.items()}
    feature_names['lncRNA'] = [c + '/' + f for c in COHORTS for f in
        ('expression_mean', 'expression_sd', 'expression_measured_fraction',
         'local_cnv_mean', 'local_cnv_sd', 'local_cnv_callable_fraction')]
    feature_names['gene'] = [c + '/' + f for c in COHORTS for f in ('expression_mean', 'expression_sd', 'expression_measured_fraction')]
    for kind in assays:
        feature_names[kind] = ['signal_mean', 'signal_sd', 'signal_measured_fraction']
    return {'ids': ids, 'graph': arrays,
        'relations': [key[6:].split('|') for key in arrays if key.startswith('edge::')],
        'feature_names': feature_names, 'coexpression_support': support,
        'audit': {'cancer': cancer, 'context': context.context, 'target_host': '149',
            'component_receipt_sha256': sha(Path(component_root) / 'COMPLETE.json'),
            'static_receipt_sha256': sha(Path(static_root) / 'COMPLETE.json'),
            'coexpression': coexpression_audit, 'gene_node_omics': omics_audit,
            'assays': assay_audit, 'molecular_bindings': molecular_bindings,
            'feature_patient_roles': {c: digest(s['source_patients']) for c,s in context.cohort_splits.items()},
            'label_values_used_in_graph': False}}


def materialize_relation_graph_view(component_root,static_root,molecular_bindings,out,*,progress=None):
    from .common import atomic_json,atomic_npz
    from .local_execution import require_host_149
    from .relation_graph_view import load_relation_graph_view
    require_host_149()
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('RELATION_GRAPH_OUTPUT_NOT_EMPTY')
    out.mkdir(parents=True,exist_ok=True)
    component=json.loads((Path(component_root)/'COMPLETE.json').read_text())
    atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','cancer':component['cancer'],
        'context':component['context'],'molecular_bindings':molecular_bindings,
        'code':{name:sha(Path(__file__).parent/name) for name in ('relation_view.py','relation_assays.py','graph.py')}})
    result=build_relation_graph_view(component_root,static_root,molecular_bindings,progress=progress)
    for name,value in [('NODES.json',result['ids']),('RELATIONS.json',result['relations']),
                       ('FEATURE_NAMES.json',result['feature_names']),('AUDIT.json',result['audit'])]:
        atomic_json(out/name,value)
    atomic_npz(out/'graph.npz',**result['graph'])
    support={c+'::'+s+'::'+k:v for c,edges in result['coexpression_support'].items()
             for s,entry in edges.items() for k,v in entry.items()}
    atomic_npz(out/'COEXPRESSION_SUPPORT.npz',**support)
    atomic_json(out/'COMPLETE.json',{'format':'CLA_RELATION_GRAPH_VIEW_COMPONENT_V1','target_host':'149',
        'cancer':component['cancer'],'context':component['context'],'prepared_capsule':False,'training_started':False,
        'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()}})
    load_relation_graph_view(out,component_root,static_root)
    return str(out)
