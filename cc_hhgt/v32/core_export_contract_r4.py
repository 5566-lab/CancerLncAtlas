"""R6.2 export generation; filename retained solely for original module import compatibility."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path

EXPORT_FORMAT='CC_HHGT_V3_2_FULL_PYG_R62_FROZEN_CORE_V1'
CHECKPOINT_FORMAT='CLA_FULL_PYG_R62_FULL_TRAINING_STATE_V1'
ARCHITECTURE='V32_FULL_PYG_SHARED_HHGT_PATHWAY_STATE_1_05_R62'
BASE_EXPORT_NODES={'cancer','lncRNA','pathway'}

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()

def _validate_declared_exports(root,records,label,*,require_base=False):
    if not isinstance(records,dict) or (require_base and not BASE_EXPORT_NODES.issubset(records)):
        raise RuntimeError(f'R62_CORE_EXPORT_SET_MISMATCH:{label}')
    for node,record in records.items():
        if not isinstance(node,str) or not node or not isinstance(record,dict):
            raise RuntimeError(f'R62_CORE_EXPORT_DECLARATION_INVALID:{label}')
        ep=(root/record.get('path','')).resolve()
        if (not ep.is_relative_to(root) or not ep.is_file()
                or re.fullmatch('[0-9a-f]{64}',str(record.get('sha256',''))) is None
                or file_hash(ep)!=record['sha256']):
            raise RuntimeError(f'R62_CORE_EXPORT_MISMATCH:{label}:{node}')
        if 'rows' in record and (not isinstance(record['rows'],int) or isinstance(record['rows'],bool) or record['rows']<1):
            raise RuntimeError(f'R62_CORE_EXPORT_ROWS_INVALID:{label}:{node}')
        if 'features' in record and (not isinstance(record['features'],int) or isinstance(record['features'],bool) or record['features']<1):
            raise RuntimeError(f'R62_CORE_EXPORT_FEATURES_INVALID:{label}:{node}')

def validate_export(path,*,fold=None,expected_manifest_sha256=None,expected_graph_receipt_sha256=None):
    """Validate all bound exports plus the graph-source receipt before use."""
    path=Path(path).resolve();root=path.parent
    payload=json.loads(path.read_text(encoding='utf-8'))
    required={'export_format':EXPORT_FORMAT,'core_architecture':ARCHITECTURE,
        'training_generation':'V3.2','all_embeddings_from_newly_trained_v32_core':True,
        'historical_checkpoint_loaded':False,'historical_prediction_loaded':False,
        'test_labels_used_in_export':False,'task_weights':{'pathway':1.,'state':.5}}
    for k,v in required.items():
        if payload.get(k)!=v:raise RuntimeError(f'R62_CORE_CONTRACT: {k} differs')
    if expected_manifest_sha256 and file_hash(path)!=expected_manifest_sha256:raise RuntimeError('R62_CORE_MANIFEST_SHA_MISMATCH')
    folds=payload.get('folds',{})
    if set(folds)!=set(map(str,range(5))):raise RuntimeError('R62_CORE_EXACTLY_FIVE_FOLDS_REQUIRED')
    graph=payload['graph_authority_receipt'];gp=(root/graph['path']).resolve()
    if not gp.is_relative_to(root) or file_hash(gp)!=graph['sha256']:raise RuntimeError('R62_GRAPH_RECEIPT_SHA_MISMATCH')
    if expected_graph_receipt_sha256 and graph['sha256']!=expected_graph_receipt_sha256:raise RuntimeError('R62_GRAPH_RECEIPT_BINDING_MISMATCH')
    gr=json.loads(gp.read_text())
    if gr.get('format')!='R62_CONTEXT_GRAPH_AUTHORITY_V1' or gr.get('variant')!='G2':raise RuntimeError('R62_GRAPH_AUTHORITY_NOT_G2')
    if payload.get('formal_graph_variant')!='G2':raise RuntimeError('R62_AUXILIARY_CORE_MUST_USE_PREREGISTERED_G2')
    indices=range(5) if fold is None else [int(fold)]
    for i in indices:
        row=folds[str(i)]
        if row.get('patient_fold')!=i or row.get('checkpoint_format')!=CHECKPOINT_FORMAT:raise RuntimeError('R62_CORE_FOLD_OR_CHECKPOINT_MISMATCH')
        if row.get('old_checkpoint_loaded') is not False or row.get('trained_from_scratch') is not True:raise RuntimeError('R62_CORE_NOT_FRESH')
        for name in ['core_parameter_sha256','checkpoint_sha256','prepared_sha256','training_receipt_sha256']:
            if re.fullmatch('[0-9a-f]{64}',str(row.get(name,''))) is None:raise RuntimeError(f'R62_CORE_SHA_MISSING:{name}')
        if gr['prepared_manifest_sha256'].get(str(i))!=row['prepared_sha256']:raise RuntimeError('R62_CORE_PREPARED_GRAPH_MISMATCH')
        _validate_declared_exports(root,row.get('exports',{}),str(i),require_base=True)
        contexts=row.get('context_exports',{})
        if not isinstance(contexts,dict):raise RuntimeError(f'R62_CORE_CONTEXT_EXPORT_SET_INVALID:{i}')
        for context,records in contexts.items():
            _validate_declared_exports(root,records,f'{i}:context:{context}',require_base=True)
    composite=hashlib.sha256(json.dumps([folds[str(i)]['core_parameter_sha256'] for i in range(5)],separators=(',',':')).encode()).hexdigest()
    return payload,file_hash(path),composite
