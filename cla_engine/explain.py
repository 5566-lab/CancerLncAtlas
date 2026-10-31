"""Validation-only contribution tables. Graph-branch ablation != total modality removal."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from .common import read_json,verify_manifest,fail,atomic_json,status
from .train import model_signature,load_checkpoint,evaluate
from .model import JointHGT
from .lineage import verify_completion

def contributions(prepared,training,out,device='cpu'):
    p=Path(prepared);training=Path(training);out=Path(out);out.mkdir(parents=True,exist_ok=True);verify_manifest(p)
    cfg=read_json(p/'CONFIG.json');spec=read_json(p/'SPEC.json');receipt,_=verify_completion(p,training);sig=model_signature(p,cfg,spec)
    if receipt['signature']!=sig:fail('RESULT_BINDING_MISMATCH','Explain only the trained, bound candidate')
    checkpoint,_=load_checkpoint(training,sig);model=JointHGT(spec,cfg['model']).to(device);model.load_state_dict(checkpoint['best']['model']);views=read_json(p/'VIEWS.json');alpha=receipt['alpha'];_,full=evaluate(model,p,views,spec,device,cfg['training']['batch_size'],alpha)
    rows=[]
    for port in ['singlecell','mutation','reported_PPI','reported_binding','coexpression']:
        _,abl=evaluate(model,p,views,spec,device,cfg['training']['batch_size'],alpha,disabled=frozenset([port]))
        for task in ['pathway','state']:
            ids=spec['nodes']['pathway'] if task=='pathway' else spec['states']
            for (c,a,s,_),(c2,a2,s2,_) in zip(full[task],abl[task]):
                if c!=c2 or not np.array_equal(a['target'],a2['target']):fail('EXPLANATION_ROW_DRIFT','Candidate order changed')
                for ti in np.unique(a['target']):
                    ok=a['available'].astype(bool)&(a['target']==ti);y=a['label'][ok];r={'task':task,'cancer_id':c,'target_id':ids[int(ti)],'removed_graph_port':port,'n':int(ok.sum()),'available':np.unique(y).size==2,'baseline_and_numeric_readout_kept':True}
                    if r['available']:
                        r.update(full_auprc=float(average_precision_score(y,s[ok])),ablated_auprc=float(average_precision_score(y,s2[ok])));r['delta_auprc']=r['full_auprc']-r['ablated_auprc']
                    rows.append(r)
    pd.DataFrame(rows).to_csv(out/'GRAPH_PORT_TARGET_CONTRIBUTIONS.tsv',sep='\t',index=False)
    gates=model.modal_gate.detach().cpu().clamp(0,1).numpy();pd.DataFrame({'cancer_id':spec['cancers'],'singlecell_graph_gate':gates[:,0],'mutation_graph_gate':gates[:,1]}).to_csv(out/'LEARNED_GRAPH_INPUT_GATES.tsv',sep='\t',index=False)
    atomic_json(out/'EXPLANATION_CONTRACT.json',{'evaluation_data':'VALIDATION_ONLY','test_read':False,'graph_port_removal_only':True,'full_modality_removal_requires_new_preparation_and_matched_baseline':True,'causal_contribution_claimed':False,'alpha':alpha,'zero_alpha_all_primary_graph_contributions_zero_by_construction':alpha==0})
    return status(out,'VALIDATION_CONTRIBUTIONS_EXPORTED')
