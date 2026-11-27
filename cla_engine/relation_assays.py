"""Source-only assay nodes; antibody groups and MS sites retain their identities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from .common import sha
from .relation_features import entity_summary, verified_file
from .relation_source import _source_patients


@dataclass(frozen=True)
class SourceAssay:
    cohort: str
    layer: str
    patients: tuple[str, ...]
    feature_ids: tuple[str, ...]
    gene_groups: tuple[tuple[str, ...], ...]
    values: np.ndarray
    source_files: dict


def verified_product(binding):
    path = verified_file(binding)
    product = json.loads(path.read_text())
    if product.get('target_host') != '149':
        raise ValueError('RELATION_ASSAY_PRODUCT_HOST')
    return path.parent, product


def product_file(root, product, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('RELATION_ASSAY_PRODUCT_PATH')
    item = product['files'].get(name)
    expected = item.get('sha256') if isinstance(item, dict) else item
    if not expected or sha(path) != expected:
        raise ValueError('RELATION_ASSAY_SOURCE_DRIFT: ' + name)
    return path


def align_cptac_assay(frame, patients):
    """Use declared tumor columns; absent assays remain missing patients."""
    if 'feature_id' not in frame or frame.feature_id.isna().any() or frame.feature_id.duplicated().any():
        raise ValueError('RELATION_ASSAY_FEATURE_AXIS')
    columns = [c for c in frame.columns if c != 'feature_id']
    if any(not c.endswith('::T') for c in columns):
        raise ValueError('RELATION_ASSAY_NONTUMOR_COLUMN')
    native = [c[:-3] for c in columns]
    if len(native) != len(set(native)):
        raise ValueError('RELATION_ASSAY_DUPLICATE_PATIENT')
    matrix = frame.set_index('feature_id').T
    matrix.index = native
    return matrix.reindex(list(patients)).to_numpy(np.float32)


def load_source_assays(context, bindings):
    """Load raw abundance by exact source-patient identity, never pooled platforms."""
    result = {}
    root, product = verified_product(bindings['tcpa_rppa500'])
    catalog_path = product_file(root, product, 'FEATURE_CATALOG.tsv')
    catalog = pd.read_csv(catalog_path, sep='\t').fillna('')
    if catalog.feature_id.duplicated().any():
        raise ValueError('RELATION_ASSAY_RPPA_DUPLICATE_FEATURE')
    byid = catalog.set_index('feature_id')
    patients = _source_patients(context, 'TCGA')
    for layer, name in (('protein', 'PROTEIN.parquet'), ('phospho', 'PHOSPHORYLATION.parquet')):
        path = product_file(root, product, name)
        frame = pd.read_parquet(path).set_index('patient_id')
        if not frame.index.is_unique or set(frame.columns) - set(byid.index):
            raise ValueError('RELATION_ASSAY_RPPA_AXIS')
        groups = tuple(tuple(sorted(set(str(byid.loc[f, 'gene_ids']).split('|')) - {''})) for f in frame.columns)
        result['tcga_' + layer + '_assay'] = SourceAssay('TCGA', layer, patients,
            tuple(frame.columns), groups, frame.reindex(list(patients)).to_numpy(np.float32),
            {'matrix': {'path': str(path), 'sha256': sha(path)},
             'annotation': {'path': str(catalog_path), 'sha256': sha(catalog_path)}})
    root, product = verified_product(bindings['cptac_aligned'])
    annotation_path = product_file(root, product, 'FEATURE_ALIGNMENT.parquet')
    annotation = pd.read_parquet(annotation_path, columns=['layer', 'feature_id', 'gene_id'])
    patients = _source_patients(context, 'CPTAC')
    for layer, native in (('protein', 'protein'), ('phospho', 'phosphorylation')):
        name = f'matrices/{context.cancer_id}/{native}_Tumor.parquet'
        path = product_file(root, product, name)
        frame = pd.read_parquet(path)
        features = tuple(frame.feature_id.astype(str))
        ann = annotation.loc[annotation.layer.eq(native)].dropna(subset=['feature_id', 'gene_id'])
        grouped = ann.groupby('feature_id').gene_id.agg(lambda values: tuple(sorted(set(values.astype(str)))))
        groups = tuple(grouped.get(f, ()) for f in features)
        result['cptac_' + layer + '_assay'] = SourceAssay('CPTAC', layer, patients,
            features, groups, align_cptac_assay(frame, patients),
            {'matrix': {'path': str(path), 'sha256': sha(path)},
             'annotation': {'path': str(annotation_path), 'sha256': sha(annotation_path)}})
    for key, block in result.items():
        if np.isinf(block.values).any() or block.values.shape != (len(block.patients), len(block.feature_ids)):
            raise ValueError('RELATION_ASSAY_NONFINITE_OR_SHAPE: ' + key)
    return result


def append_assay_nodes(ids, graph, assays):
    """One node per measurement, even when an antibody recognizes multiple genes."""
    genes = {g: i for i, g in enumerate(ids['gene'])}
    audit = {}
    for kind, assay in assays.items():
        if kind in ids:
            raise ValueError('RELATION_ASSAY_NODE_COLLISION')
        if len(assay.feature_ids) != len(set(assay.feature_ids)) or len(assay.feature_ids) != len(assay.gene_groups):
            raise ValueError('RELATION_ASSAY_METADATA_AXIS')
        ids[kind] = list(assay.feature_ids)
        finite = np.isfinite(assay.values)
        graph['x::' + kind] = entity_summary(assay.values, finite)
        edges = [(j, genes[g]) for j, group in enumerate(assay.gene_groups)
                 for g in sorted(set(group)) if g in genes]
        index = np.asarray(edges, np.int64).reshape(-1, 2).T
        relation = kind + '|assay_group_component|gene'
        reverse = 'gene|rev_assay_group_component|' + kind
        graph['edge::' + relation] = index
        graph['weight::' + relation] = np.ones(len(edges), np.float32)
        graph['edge::' + reverse] = index[::-1].copy()
        graph['weight::' + reverse] = np.ones(len(edges), np.float32)
        if any(k.startswith('node_omics::') for k in graph):
            graph['node_omics::' + kind] = np.zeros((len(assay.feature_ids), 4), np.float32)
        audit[kind] = {'cohort': assay.cohort, 'layer': assay.layer,
            'source_patients': len(assay.patients), 'assay_nodes': len(assay.feature_ids),
            'observed_assay_nodes': int(finite.any(0).sum()), 'observed_cells': int(finite.sum()),
            'group_component_edges': len(edges), 'linked_assay_nodes': len({e[0] for e in edges}),
            'multi_gene_assays': sum(len(g) > 1 for g in assay.gene_groups),
            'measurement_duplicated_per_gene': False, 'phospho_interpretation': 'raw signal; distinct from total abundance and signed activity',
            'source_files': assay.source_files}
    return audit
