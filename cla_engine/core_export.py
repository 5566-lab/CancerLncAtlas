"""Export new PyG joint cores into an explicitly versioned auxiliary interface.

Context-specific exports are retained. Existing private heads receive a declared
mean of source-patient context encodings, NOT a prediction or a test-derived vector.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import read_json,sha,atomic_json,digest,fail,status,verify_manifest,commit_manifest
from .model import JointHGT,to_graph
from .lineage import verify_completion,verify_signed_file,FIELDS
from .train import load_npz,load_checkpoint,model_signature

CORE_EXPORT_NODE_TYPES=('lncRNA','pathway','cancer','gene')

def _context_node_arrays(encoded,spec):
    result={}
    for node in CORE_EXPORT_NODE_TYPES:
        if node not in encoded or node not in spec['nodes']:
            fail('CORE_NODE_MISSING','Joint encoder did not return a declared export node',node=node)
        values=encoded[node].float().cpu().numpy()
        if values.ndim!=2 or values.shape[0]!=len(spec['nodes'][node]) or not np.isfinite(values).all():
            fail('CORE_NONFINITE','Core embeddings have invalid shape or values',node=node)
        result[node]=values
    return result

def _fold_node_arrays(totals,counts,cancer_values):
    result={}
    for node in CORE_EXPORT_NODE_TYPES:
        if node=='cancer':result[node]=cancer_values
        else:
            if counts.get(node,0)<=0:fail('CORE_CONTEXT_MISSING','No source-patient context encoding was observed',node=node)
            result[node]=totals[node]/counts[node]
    return result

def export_core(plan_path,queue_root,out,*,device='cpu'):
    from cc_hhgt.v32.integrated_model import module_state_sha256
    from cc_hhgt.v32.core_export_contract_r4 import EXPORT_FORMAT,CHECKPOINT_FORMAT,ARCHITECTURE,validate_export
    plan_path=Path(plan_path);queue_root=Path(queue_root);out=Path(out)
    plan=read_json(plan_path)
    if plan.get('format')!='CLA_FULL_PYG_R62_15_JOB_PLAN_V1':fail('WRONG_GENERATION','R6.2 bound full job plan required')
    jobs=sorted([j for j in plan['jobs'] if j['variant']=='G2'],key=lambda j:j['fold'])
    if [j['fold'] for j in jobs]!=list(range(5)):fail('CORE_FIVE_FOLDS_REQUIRED','Expected G2 folds 0..4')
    base_cfg=read_json(Path(jobs[0]['prepared'])/'CONFIG.json')
    verify_signed_file(plan_path,base_cfg,'FORMAL_JOB_PLAN')
    common_lineage=None
    if (out/'CORE_EMBEDDING_MANIFEST.json').exists():
        v,_,_=validate_export(out/'CORE_EMBEDDING_MANIFEST.json')
        if v.get('parent_plan_sha256')!=sha(plan_path):fail('CORE_EXPORT_REUSE_DRIFT','Existing core export belongs to another plan')
        return status(out,'CORE_EXPORT_REUSED_VERIFIED_NOT_DEPLOYED')
    out.mkdir(parents=True,exist_ok=True)
    folds={};graph_hashes={};input_hashes={};candidate_rows=[];reference_nodes=None
    for j in jobs:
        f=j['fold'];prepared=Path(j['prepared']);verify_manifest(prepared,expected=j['sha256'])
        cfg=read_json(prepared/'CONFIG.json');spec=read_json(prepared/'SPEC.json');views=read_json(prepared/'VIEWS.json')
        if spec['fold']!=f or spec.get('graph_variant')!='G2' or cfg.get('synthetic_fixture',False):fail('CORE_SOURCE_NOT_FORMAL','No synthetic or other-variant parent')
        if read_json(prepared/'MANIFEST.json')['metadata']['profile']!='formal':fail('CORE_SOURCE_NOT_FORMAL','Formal patient folds required')
        if reference_nodes is None:reference_nodes=spec['nodes']
        elif spec['nodes']!=reference_nodes:fail('CORE_NODE_UNIVERSE_DRIFT','All five folds must use the same node universe/order')
        training=queue_root/'G2'/f'fold_{f}'/'run';rp=training/'TRAINING_RECEIPT.json';receipt,lineage=verify_completion(prepared,training)
        if receipt['state']!='TRAINED_FORMAL_CANDIDATE':fail('CORE_TRAINING_NOT_COMPLETE','Formal completed checkpoint required')
        current_common={k:lineage[k] for k in FIELDS}
        if common_lineage is None:common_lineage=current_common
        elif common_lineage!=current_common:fail('CORE_MIXED_LINEAGE','Five exported cores must share source/task/model provenance')
        signature=model_signature(prepared,cfg,spec)
        if receipt.get('signature')!=signature or receipt.get('formal_test_evaluated') is not False:fail('CORE_TRAINING_BINDING_MISMATCH','Parent training signature invalid')
        saved,record=load_checkpoint(training,signature)
        if saved is None or saved.get('format')!=CHECKPOINT_FORMAT:fail('CORE_CHECKPOINT_MISSING','R4 complete state required')
        if saved['best']['step']==0 or saved['best']['alpha']<=0:fail('CORE_NO_LEARNED_INCREMENT','Refusing to export epoch-zero/fallback as a learned core',fold=f)
        model=JointHGT(spec,cfg['model']).to(device);model.load_state_dict(saved['best']['model'],strict=True);model.eval()
        totals={k:np.zeros((len(spec['nodes'][k]),cfg['model']['hidden']),dtype=np.float64) for k in CORE_EXPORT_NODE_TYPES}
        counts={k:0 for k in totals};cancer_values=np.zeros_like(totals['cancer']);ctx_exports={}
        with torch.no_grad():
            for view in views:
                if view['split']!='validation':continue
                # The graph is made from outer-training patients; no validation/test measurements.
                graph=to_graph(load_npz(prepared/'views'/view['id']/'graph.npz'),device)
                h,ci,_,_=model.encode(graph,view['cancer_index']);cancer=spec['cancers'][ci]
                folder=out/f'patient_fold={f}'/'contexts'/cancer;folder.mkdir(parents=True,exist_ok=True)
                local={}
                for node,values in _context_node_arrays(h,spec).items():
                    totals[node]+=values;counts[node]+=1
                    frame=pd.DataFrame(values,columns=[f'core_feature_{i:04d}' for i in range(values.shape[1])]);frame.insert(0,'node_id',spec['nodes'][node])
                    fp=folder/f'{node}.parquet';frame.to_parquet(fp,index=False)
                    local[node]={'path':str(fp.relative_to(out)),'sha256':sha(fp),'rows':len(frame),'features':values.shape[1]}
                cancer_values[ci]=h['cancer'][ci].float().cpu().numpy();ctx_exports[cancer]=local
                if f==0:
                    a=load_npz(prepared/'views'/view['id']/'pathway.npz')
                    candidate_rows.extend((cancer,spec['nodes']['lncRNA'][int(l)],spec['nodes']['pathway'][int(t)]) for l,t in zip(a['lnc'],a['target']))
                del h,graph
        exports={};folder=out/f'patient_fold={f}';folder.mkdir(parents=True,exist_ok=True)
        for node,values in _fold_node_arrays(totals,counts,cancer_values).items():
            frame=pd.DataFrame(values.astype(np.float32),columns=[f'core_feature_{i:04d}' for i in range(values.shape[1])]);frame.insert(0,'node_id',spec['nodes'][node])
            fp=folder/f'{node}.parquet';frame.to_parquet(fp,index=False)
            exports[node]={'path':str(fp.relative_to(out)),'sha256':sha(fp),'rows':len(frame),'features':values.shape[1]}
        graph_hashes[str(f)]=sha(prepared/'MANIFEST.json');input_hashes[str(f)]=read_json(prepared/'INPUT_HASHES.json')
        folds[str(f)]={'patient_fold':f,'checkpoint_format':CHECKPOINT_FORMAT,'checkpoint_sha256':record['sha256'],
            'checkpoint_path':str((training/record['file']).resolve()),'core_parameter_sha256':module_state_sha256(model.encoder),
            'prepared_sha256':j['sha256'],'training_receipt_sha256':sha(rp),'trained_from_scratch':True,'old_checkpoint_loaded':False,
            'best_step':saved['best']['step'],'formal_graph_variant':'G2','exports':exports,'context_exports':ctx_exports}
        del model,saved,totals
    gp=out/'GRAPH_AUTHORITY_RECEIPT.json'
    atomic_json(gp,{'format':'R62_CONTEXT_GRAPH_AUTHORITY_V1','variant':'G2','prepared_manifest_sha256':graph_hashes,
                    'source_inputs':input_hashes,'parent_plan_sha256':sha(plan_path),'outer_test_used_to_generate_graph':False})
    cp=out/'CANDIDATE_UNIVERSE.parquet';pd.DataFrame(candidate_rows,columns=['cancer_id','lncrna_id','pathway_id']).drop_duplicates().to_parquet(cp,index=False)
    manifest={'export_format':EXPORT_FORMAT,'core_architecture':ARCHITECTURE,'analysis_version':'CancerLncAtlas_V3.2_FULL_PYG_R62',
        'training_generation':'V3.2','all_embeddings_from_newly_trained_v32_core':True,'historical_checkpoint_loaded':False,
        'historical_prediction_loaded':False,'test_labels_used_in_export':False,'task_weights':{'pathway':1.,'state':.5},
        'formal_graph_variant':'G2','parent_plan_sha256':sha(plan_path),'folds':folds,
        'graph_authority_receipt':{'path':gp.name,'sha256':sha(gp)},'candidates':{'path':cp.name,'sha256':sha(cp),'rows':len(set(candidate_rows)),'cancers':len(set(r[0] for r in candidate_rows))},
        'private_head_embedding_role':'MEAN_OF_SOURCE_PATIENT_CANCER_CONTEXTS; cancer row retains own context',
        'context_specific_exports_preserved':True,'not_a_legacy_V1_checkpoint':True,'production_deployed':False}
    atomic_json(out/'CORE_EMBEDDING_MANIFEST.json',manifest);validate_export(out/'CORE_EMBEDDING_MANIFEST.json')
    return status(out,'CORE_EXPORTED_NOT_DEPLOYED',manifest_sha256=sha(out/'CORE_EMBEDDING_MANIFEST.json'),folds=5)
