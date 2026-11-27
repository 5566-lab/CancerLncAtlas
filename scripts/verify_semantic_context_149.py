from pathlib import Path
import socket,os,json,sys,hashlib
assert socket.gethostname().split('.')[0]=='149'
os.environ['OMP_NUM_THREADS']='2'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';repo=R/'source_integration/repo'
O=W/'semantic_score_context_20261119';B=W/'wiki_patient_integration_20261114_r1/Y_bundle'
(O/'VERIFY_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'semantic_route_and_forward_self_check','optimizer_steps':0,'training':False}))
sys.path[:0]=[str(repo),str(repo/'tests')]
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(2)
from cla_engine.fixed_y_bundle import load_fixed_target,load_target_with_context
from cla_engine.semantic_context import read_context_bundle,select_routes,ContextTask
from cla_engine.measured_context_models import (measured_m0_design,measured_m0_forward,measured_hhgt_graph,
    score_target_graph,selected_score_loss,measured_m0_score_forward)
from cla_engine.score_observation_head import ScoreObservationHead
from cla_engine.model import JointHGT,to_graph
from test_joint_node_omics import _model_fixture
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
review=pd.read_csv(O/'PROGENY_REVIEWED_ROUTES.tsv',sep='\t',keep_default_na=False)
wnt=review.loc[review.network_id.eq('N00056'),'target_id'].iloc[0]
apc=review.loc[review.network_id.eq('N00057'),'target_id'].iloc[0]
assert not ((review.score=='TGFb') & review.network_name.str.contains('TGFA')).any()
assert not ((review.score=='Estrogen') & review.network_id.isin(['N00317','N01362','N01363','N01365'])).any()
proof=[];scope_counts=[];score_counts=[]
for case in ['TCGA_BRCA','TCGA_LUAD','TCGA_COAD','CPTAC_BRCA','CPTAC_LUAD','CPTAC_COAD']:
 print('VERIFY',case,flush=True)
 folder=B/'cohorts'/case;before={n:sha(folder/n) for n in ['Y_VALUES.parquet','Y_OBSERVED.parquet','TARGET_APPLICABILITY.parquet']}
 context=read_context_bundle(O,case)
 values,mask,app,metadata=load_fixed_target(B,case,wnt,for_supervision=True)
 candidates=metadata[metadata.layer.eq('PROTEIN')&metadata.patients_measured.gt(0)].observation_id.tolist()
 oid=next(k for k in candidates if mask[k].any())
 bound=load_target_with_context(B,O,case,wnt,oid,allowed_input_layers=['RNA'])
 assert bound['y'].equals(values[oid]) and bound['mask'].equals(mask[oid])
 task=bound['task'];context=bound['context']
 sid=context['bindings'].loc[context['bindings'].source_scope.eq('PROGENY:WNT'),'score_id'].iloc[0]
 p=next(i for i in bound['y'].index if bound['mask'].loc[i] and np.isfinite(context['values'].at[i,sid]))
 rna=pd.DataFrame(index=context['values'].index);cov=pd.DataFrame(index=rna.index)
 design,penalty,chosen=measured_m0_design(rna,cov,context,task)
 beta=pd.Series(0.,index=design.columns);beta['score::'+sid]=1.
 original=measured_m0_forward(design,beta)
 changed=dict(context);changed['values']=context['values'].copy()
 changed['values'].at[p,sid]+=float(context['scaler'].at[sid,'scale'])
 design2,_,_=measured_m0_design(rna,cov,changed,task)
 changed_prediction=measured_m0_forward(design2,beta)
 linear_delta=float(changed_prediction.loc[p]-original.loc[p]);assert np.isclose(linear_delta,1.,atol=1e-5)
 spec,cfg,arrays=_model_fixture(False);spec['nodes']['pathway']=[wnt,apc]
 torch.manual_seed(37)
 extended,a=measured_hhgt_graph(spec,arrays,context,task,p)
 extended2,a2=measured_hhgt_graph(spec,arrays,changed,task,p)
 assert extended==extended2
 model=JointHGT(extended,cfg).eval()
 with torch.no_grad():
  h=model.encode(to_graph(a,'cpu'),0)[0]['pathway'];h2=model.encode(to_graph(a2,'cpu'),0)[0]['pathway']
  graph_delta=float((h2[0]-h[0]).abs().max());assert graph_delta>1e-7
  ab=model.encode(to_graph(a,'cpu'),0,frozenset({'score_context'}))[0]['pathway']
  ab2=model.encode(to_graph(a2,'cpu'),0,frozenset({'score_context'}))[0]['pathway']
  removed_delta=float((ab2-ab).abs().max());assert removed_delta==0.
 # A score serving as Y cannot simultaneously be inserted as an RNA input.
 forbidden=ContextTask(wnt,'rna_y',frozenset({'RNA'}),frozenset({'RNA'}),frozenset())
 assert select_routes(context['bindings'],context['routes'],forbidden).empty
 # Predict the SAME original score once, with two scope edges, no value as input.
 s,ga=score_target_graph(spec,arrays,context['bindings'],context['routes'],[sid])
 assert s['nodes']['score_target']==[sid] and not ga['x::score_target'].any()
 torch.manual_seed(73);model_y=JointHGT(s,cfg).eval();head=ScoreObservationHead(cfg['hidden'],[sid])
 graph=to_graph(ga,'cpu');graph['features']['pathway'].requires_grad_()
 pred=head(model_y.encode(graph,0),s)
 label=torch.tensor([float(context['values'].at[p,sid])],dtype=torch.float32)
 loss=selected_score_loss(pred,label,torch.tensor([True]),[sid]);loss.backward()
 gradients=graph['features']['pathway'].grad.abs().sum(dim=1).tolist()
 assert all(g>0 for g in gradients)
 assert selected_score_loss(pred,label,torch.tensor([True]),[]) is None
 # Linear output retains one score column independent of two scope links.
 raw_x=pd.DataFrame({'constant':np.ones(len(rna))},index=rna.index)
 output=measured_m0_score_forward(raw_x,pd.DataFrame({sid:[.2]},index=['constant']),pd.Series({sid:0.}))
 assert output.shape==(len(rna),1)
 assert all(sha(folder/n)==v for n,v in before.items())
 proof.append(dict(case=case,patient_id=p,target_id=wnt,target_observation_id=oid,score_id=sid,
    original_score=float(context['values'].at[p,sid]),original_y=float(bound['y'].loc[p]),
    m0_one_sd_input_delta=linear_delta,hhgt_max_embedding_delta=graph_delta,
    hhgt_delta_without_score_edges=removed_delta,score_target_output_count=int(pred.numel()),
    score_target_pathway_gradients=gradients,original_y_unchanged=True,optimizer_steps=0))
 for rel,g in context['routes'].groupby('relation'):
  scope_counts.append(dict(case=case,relation=rel,edges=len(g),scores=g.score_id.nunique(),pathways=g.target_id.nunique()))
 for r in context['bindings'][context['bindings'].source_scope.str.startswith('PROGENY:')].to_dict('records'):
  score_counts.append(dict(case=case,score=r['source_scope'],patients_measured=r['patients_measured'],mapped_main_pathways=r['mapped_main_pathways']))
pd.DataFrame(scope_counts).to_csv(O/'RELATION_TYPE_COUNTS.tsv',sep='\t',index=False)
pd.DataFrame(score_counts).to_csv(O/'PROGENY_COVERAGE.tsv',sep='\t',index=False)
(O/'FORWARD_VERIFICATION.json').write_text(json.dumps(proof,indent=2))
record=dict(status='PASS',target_host='149',training_started=False,optimizer_steps=0,cohorts_checked=6,
 original_y_unchanged=True,new_gene_sets=0,new_score_calculations=0,
 original_hhgt_encoder_used=True,score_target_gradient_verified=True,
 note='Real scores and masks; fixed/random-weight forward and backward wiring checks, no model fitting or performance evaluation.')
(O/'VERIFICATION.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record),flush=True)
print(pd.DataFrame(scope_counts).to_string(index=False),flush=True)
print(pd.DataFrame(score_counts).query('case=="TCGA_BRCA"').to_string(index=False),flush=True)
