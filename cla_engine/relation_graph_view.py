"""Verify a source-patient molecular graph component before prepared assembly."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .common import sha,digest
from .relation_components import load_relation_components

FORMAT='CLA_RELATION_GRAPH_VIEW_COMPONENT_V1'


def validate_graph_arrays(ids,arrays,feature_names):
    """Check actual endpoints, measured-value masks and declared feature axes."""
    if set(feature_names)!=set(ids):raise ValueError('RELATION_GRAPH_FEATURE_NODE_SET')
    for kind,axis in ids.items():
        if not axis or len(axis)!=len(set(axis)):raise ValueError('RELATION_GRAPH_NODE_AXIS')
        value=arrays.get('x::'+kind)
        if value is None or value.shape!=(len(axis),len(feature_names[kind])) or not np.isfinite(value).all():
            raise ValueError('RELATION_GRAPH_FEATURE_MATRIX')
        omics=arrays.get('node_omics::'+kind)
        if omics is None or omics.shape!=(len(axis),4) or not np.isfinite(omics).all():
            raise ValueError('RELATION_GRAPH_OMICS_MATRIX')
        if (not np.isin(omics[:,[1,3]],[0,1]).all() or np.any(omics[:,0][omics[:,1]==0]!=0)
                or np.any(omics[:,2][omics[:,3]==0]!=0) or np.any(omics[:,2]<0) or np.any(omics[:,2]>1)
                or (kind!='gene' and np.any(omics!=0))):
            raise ValueError('RELATION_GRAPH_OMICS_MASK')
    relations=[]
    for key,index in arrays.items():
        if not key.startswith('edge::'):continue
        parts=key[6:].split('|')
        if len(parts)!=3 or parts[0] not in ids or parts[2] not in ids:
            raise ValueError('RELATION_GRAPH_EDGE_TYPE')
        weight=arrays.get('weight::'+key[6:])
        if (index.ndim!=2 or index.shape[0]!=2 or not np.issubdtype(index.dtype,np.integer)
                or weight is None or weight.shape!=(index.shape[1],) or not np.isfinite(weight).all()
                or np.any(weight<=0) or np.any(index<0) or np.any(index[0]>=len(ids[parts[0]]))
                or np.any(index[1]>=len(ids[parts[2]]))):
            raise ValueError('RELATION_GRAPH_EDGE_ENDPOINT_OR_WEIGHT')
        relations.append(parts)
    expected={prefix+kind for kind in ids for prefix in ('x::','node_omics::')}
    expected.update(prefix+'|'.join(rel) for rel in relations for prefix in ('edge::','weight::'))
    if set(arrays)!=expected:raise ValueError('RELATION_GRAPH_UNDECLARED_ARRAY')
    return relations


def load_relation_graph_view(path,component_path,static_path):
    root=Path(path);receipt=json.loads((root/'COMPLETE.json').read_text())
    required={'LAUNCH_PREFLIGHT.json','NODES.json','RELATIONS.json','FEATURE_NAMES.json',
              'AUDIT.json','graph.npz','COEXPRESSION_SUPPORT.npz'}
    if (receipt.get('format')!=FORMAT or receipt.get('target_host')!='149'
            or set(receipt.get('files',{}))!=required or receipt.get('prepared_capsule') is not False):
        raise ValueError('RELATION_GRAPH_COMPONENT_FORMAT')
    for name,expected in receipt['files'].items():
        if sha(root/name)!=expected:raise ValueError('RELATION_GRAPH_COMPONENT_DRIFT')
    component=load_relation_components(component_path)
    def read(name):return json.loads((root/name).read_text())
    ids,names,audit=read('NODES.json'),read('FEATURE_NAMES.json'),read('AUDIT.json')
    for key in ('cancer','context','target_host'):
        if receipt[key]!=component['receipt'][key] or audit[key]!=receipt[key]:
            raise ValueError('RELATION_GRAPH_COMPONENT_IDENTITY')
    if audit.get('label_values_used_in_graph') is not False:raise ValueError('RELATION_GRAPH_LABEL_USE')
    if (audit['component_receipt_sha256']!=sha(Path(component_path)/'COMPLETE.json')
            or audit['static_receipt_sha256']!=sha(Path(static_path)/'COMPLETE.json')):
        raise ValueError('RELATION_GRAPH_SOURCE_BINDING')
    static_ids=json.loads((Path(static_path)/'NODES.json').read_text())
    if any(ids.get(k)!=v for k,v in static_ids.items()):raise ValueError('RELATION_GRAPH_STATIC_NODE_DRIFT')
    assay_types={c+'_'+layer+'_assay' for c in ('tcga','cptac') for layer in ('protein','phospho')}
    if set(ids)!=set(static_ids)|assay_types:raise ValueError('RELATION_GRAPH_ASSAY_TYPE_SET')
    with np.load(root/'graph.npz',allow_pickle=False) as f:arrays=dict(f)
    relations=validate_graph_arrays(ids,arrays,names)
    if relations!=read('RELATIONS.json'):raise ValueError('RELATION_GRAPH_RELATION_AXIS')
    for cohort,split in component['splits'].items():
        expected=digest(split['source_patients'])
        if audit['feature_patient_roles'][cohort]!=expected or audit['coexpression'][cohort]['source_patient_hash']!=expected:
            raise ValueError('RELATION_GRAPH_SOURCE_PATIENTS')
        if audit['coexpression'][cohort]['source_patients']!=len(split['source_patients']):
            raise ValueError('RELATION_GRAPH_SOURCE_COUNT')
        for layer in ('protein','phospho'):
            kind=cohort.lower()+'_'+layer+'_assay';record=audit['assays'][kind]
            if (record['cohort']!=cohort or record['source_patients']!=len(split['source_patients'])
                    or record['assay_nodes']!=len(ids[kind]) or record['measurement_duplicated_per_gene'] is not False):
                raise ValueError('RELATION_GRAPH_ASSAY_IDENTITY')
    expected=np.column_stack([component['lncrna_node_features'][c] for c in ('TCGA','CPTAC')])
    if not np.array_equal(arrays['x::lncRNA'],expected):raise ValueError('RELATION_GRAPH_LNCRNA_FEATURE_DRIFT')
    from .node_omics import _array_digest
    omics_audit=audit['gene_node_omics']
    omitted=set(omics_audit['new_graph_genes_absent_from_source_product'])
    keep=[i for i,g in enumerate(ids['gene']) if g not in omitted]
    block=arrays['node_omics::gene']
    if (omics_audit['graph_gene_axis_sha256']!=digest([ids['gene'][i] for i in keep])
            or omics_audit['block_sha256']!=_array_digest(block[keep])
            or np.any(block[[i for i,g in enumerate(ids['gene']) if g in omitted]]!=0)
            or omics_audit['source_patient_sha256']!=digest(sorted(component['splits']['TCGA']['source_patients']))):
        raise ValueError('RELATION_GRAPH_GENE_OMICS_SOURCE')
    return {'receipt':receipt,'audit':audit,'ids':ids,'graph':arrays,'feature_names':names,'relations':relations}
