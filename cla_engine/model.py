"""Current adapter (originally added in R4) around the ORIGINAL full-r1 PyG CC-HHGT and bounded residual.

No native HGT, no silent MLP fallback, no node-ID embedding replacement.
The original HGTConv + weighted aggregation remains in cc_hhgt/gnn.py.
R4 additions are explicit context projections, cancer-local graphs and a State
head which shares the *live* encoder. Original no-context behavior is retained.
"""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .common import fail

ARCHITECTURE = 'CC_HHGT_PYG_FULL_R62_JOINT'
NODE_OMICS_PREFIX = 'node_omics::'

def _pyg():
    try:
        from torch_geometric.data import HeteroData
        from torch_geometric.nn import HGTConv
        return HeteroData,HGTConv
    except ImportError as exc:
        fail('PYG_REQUIRED_NO_FALLBACK', 'R6.2 requires real torch_geometric; it never substitutes the R3 native model',error=str(exc))

class JointHGT(nn.Module):
    """Runtime facade; all message passing is provided by the original encoder."""
    def __init__(self,spec,model_config):
        super().__init__()
        HeteroData,_=_pyg()
        if spec.get('architecture') != ARCHITECTURE:
            fail('MODEL_GENERATION_MISMATCH','Do not load previous-generation prepared data into R6.2')
        if 'cancer' not in spec['nodes'] or 'pathway_family' not in spec['nodes']:
            fail('ORIGINAL_NODE_TYPE_MISSING','cancer and pathway_family must be graph nodes')
        if model_config['task_weights'] != {'pathway':1.,'state':.5}:
            fail('TASK_WEIGHT_DRIFT','Pathway=1.0, State=0.5')
        self.spec=spec;self.cfg=model_config;h=int(model_config['hidden'])
        if int(spec.get('singlecell_association_dim',0)):
            fail('SC_PRIMARY_INPUT_RETIRED','Reprepare without SC concatenation; single-cell experts enter secondary fusion only')
        if h%int(model_config['heads']):fail('MODEL_CONFIG','hidden must be divisible by heads')
        node_input_dim=int(spec.get('node_input_dim',2))
        enabled=bool(spec.get('gene_node_omics_enabled',False))
        if node_input_dim not in (2,6) or enabled!=(node_input_dim==6):
            fail('NODE_INPUT_CONTRACT_INVALID','R6.2 accepts the historical two columns or the declared six-column gene-omics contract')
        if enabled and spec.get('node_input_feature_order') != [
                'log1p_degree','constant','cnv_signed_value','cnv_signed_available',
                'methylation_promoter_beta_value','methylation_promoter_beta_available']:
            fail('NODE_INPUT_CONTRACT_INVALID','Six-column node feature order differs from the prepared contract')
        self.node_input_dim=node_input_dim
        data=HeteroData()
        for kind,ids in spec['nodes'].items():data[kind].x=torch.zeros((len(ids),node_input_dim))
        for relation in spec['relations']:
            data[tuple(relation)].edge_index=torch.empty((2,0),dtype=torch.long)
            data[tuple(relation)].edge_weight=torch.empty(0)
        from cc_hhgt.gnn import build_model
        from cc_hhgt.v32.joint_model import JointPathwayStateModel
        legacy={'training':{'hidden_channels':h,'num_layers':int(model_config['layers']),
            'num_heads':int(model_config['heads']),'dropout':float(model_config['dropout']),
            'pair_hidden_channels':h},'task_contract':{'pathway':{'target_level':'exact_pathway'}}}
        encoder=build_model('cc_hhgt',SimpleNamespace(hetero_data=data,node_input_dim=node_input_dim),int(spec['numeric_features']),legacy)
        # Original legacy decoder is not used by the bounded-residual wrapper.
        # It remains in the model for provenance but is explicitly frozen.
        for parameter in encoder.decoder.parameters():parameter.requires_grad_(False)
        self.joint=JointPathwayStateModel(encoder,state_ids=tuple(spec['states']),hidden_channels=h,
            context_features=4,dropout=float(model_config['dropout']))
        self.context_projection=nn.ModuleDict({t:nn.Linear(int(spec['feature_dims'][t]),h,bias=False) for t in spec['nodes']})
        self.sc_projection=nn.Linear(int(spec['sc_dim']),h,bias=False)
        self.mutation_projection=nn.Linear(3,h,bias=False)
        if spec.get('singlecell_role')=='interpretation_only':
            for parameter in self.sc_projection.parameters():parameter.requires_grad_(False)
        # Exact-zero gates are possible; availability masks stay distinct from zero.
        self.modal_gate=nn.Parameter(torch.full((len(spec['cancers']),2),.5))
        self.global_modal_gate=nn.Parameter(torch.full((2,),.5))
    @property
    def encoder(self):return self.joint.encoder
    @property
    def layers(self):return self.encoder.layers
    def regularizer(self):
        selected=slice(1,2) if self.spec.get('singlecell_role')=='interpretation_only' else slice(None)
        return float(self.cfg.get('gate_penalty',1e-5))*(self.modal_gate[:,selected].clamp(0,1).mean()+self.global_modal_gate[selected].clamp(0,1).mean())
    def encode(self,graph,cancer,disabled=frozenset()):
        data=graph['data'].clone()
        if any(data[t].x.shape[0]!=len(ids) for t,ids in self.spec['nodes'].items()):
            fail('NODE_IDENTITY_MISMATCH','Input node ordering/count differs from model spec')
        if any(data[t].x.ndim!=2 or data[t].x.shape[1]!=self.node_input_dim for t in self.spec['nodes']):
            fail('NODE_INPUT_WIDTH_MISMATCH','Graph node inputs differ from the architecture-bound checkpoint width')
        gates=self.modal_gate[int(cancer)].clamp(0,1)*self.global_modal_gate.clamp(0,1)
        for t in data.node_types:
            value=graph['features'][t].clone()
            if t=='lncRNA' and ('cnv' in disabled or 'lncrna_cnv' in disabled):
                value[:,3:6]=0
            if self.spec.get('genomics_enabled'):
                if t=='pathway' and ('cnv' in disabled or 'pathway_cnv' in disabled):
                    value[:,-3:]=0
                if t=='lncRNA' and 'partner_mutation' in disabled:
                    value[:,6:]=0
                if t=='pathway' and 'pathway_mutation' in disabled:
                    value[:,6:-3]=0
            if self.spec.get('genomics_enabled') and value.shape[1]>6:
                stop=value.shape[1]-3 if t=='pathway' else value.shape[1]
                # Non-inplace gate: a slice assignment on the value view breaks
                # the autograd graph of the earlier feature-producing Mul under
                # modern torch (AsStridedBackward0 version bump on real
                # genomics-enabled graphs). Same values, new tensor.
                gate_value=0. if 'mutation' in disabled else gates[1]
                value=torch.cat([value[:,:6],value[:,6:stop]*gate_value,value[:,stop:]],dim=1)
            data[t].r4_context_embedding=self.context_projection[t](value)
        lctx=data['lncRNA'].r4_context_embedding
        if self.spec.get('singlecell_role')!='interpretation_only' and 'singlecell' not in disabled:
            lctx=lctx+gates[0]*self.sc_projection(graph['singlecell'])*graph['sc_available'][:,None]
        if 'mutation' not in disabled and 'lncrna_mutation' not in disabled:
            lctx=lctx+gates[1]*self.mutation_projection(graph['mutation'])*graph['mutation_available'][:,None]
        data['lncRNA'].r4_context_embedding=lctx
        for rel in list(data.edge_types):
            if rel[1] in disabled or any(key in rel[1] for key in disabled):
                data[rel].edge_index=data[rel].edge_index[:,:0];data[rel].edge_weight=data[rel].edge_weight[:0]
        return self.encoder.encode(data),int(cancer),data,frozenset(disabled)
    def forward_task(self,task,encoded,batch,alpha=1.):
        h,ci,data,disabled=encoded
        cancer=torch.full_like(batch['lnc'],ci)
        candidate={'l':batch['lnc'],'c':cancer,('p' if task=='pathway' else 's'):batch['target']}
        # Original gate takes train-origin conservation context. No eval labels.
        context=batch['numeric'][:,:4]
        if context.shape[1]!=4:fail('CONSERVATION_CONTEXT_DIM','Primary context must retain the original four statistics')
        value={'candidate_batch':candidate,'base_logit':batch['base'],'conservation_context':context,
               'graph_available':torch.ones_like(batch['base'],dtype=torch.bool)}
        out=self.joint.forward_task(task,data,value,encoded=h)
        delta=out['graph_residual']*float(self.cfg.get('residual_bound',1.))
        return {'logit':batch['base']+float(alpha)*delta,'delta':delta,'direction':out['direction_logit']}

def loss_one(out,batch,direction_weight=.25,objective='binary'):
    mask=batch['available'].bool()
    if not mask.any():return None
    weight=batch['weight'][mask];weight=weight/weight.sum().clamp_min(1e-12)
    if objective=='binary':
        loss=(F.binary_cross_entropy_with_logits(out['logit'][mask],batch['label'][mask],reduction='none')*weight).sum()
        dm=mask&batch['direction_available'].bool()
        if dm.any():loss=loss+float(direction_weight)*F.binary_cross_entropy_with_logits(out['direction'][dm],batch['direction_label'][dm])
    elif objective=='observed_fused_rms_effect':
        label=batch['label'][mask]
        if not torch.isfinite(label).all() or (label<0).any() or (label>1+1e-6).any():
            fail('RELATION_STRENGTH_LABEL_INVALID','Observed RMS effect must be finite and in [0,1]')
        loss=(F.mse_loss(out['logit'][mask],label,reduction='none')*weight).sum()
    else:
        fail('RELATION_OBJECTIVE_UNKNOWN','Training objective must be explicit',objective=objective)
    return loss+1e-4*out['delta'][mask].square().mean()

def to_graph(arrays,device):
    HeteroData,_=_pyg();data=HeteroData()
    features={k[3:]:torch.as_tensor(v,dtype=torch.float32,device=device) for k,v in arrays.items() if k.startswith('x::')}
    raw_node_omics={k[len(NODE_OMICS_PREFIX):]:np.asarray(v) for k,v in arrays.items() if k.startswith(NODE_OMICS_PREFIX)}
    if raw_node_omics and set(raw_node_omics)!=set(features):
        fail('NODE_OMICS_NODE_TYPE_SET','Every graph node type requires an explicit node-omics block')
    node_omics={}
    for t,raw in raw_node_omics.items():
        value=np.asarray(raw,dtype=np.float32)
        if value.shape!=(len(features[t]),4) or not np.isfinite(value).all():
            fail('NODE_OMICS_SHAPE_OR_VALUE','Node-omics block must be finite node-by-four',node_type=t)
        if not np.isin(value[:,[1,3]],[0,1]).all():
            fail('NODE_OMICS_MASK_INVALID','Availability columns must be binary',node_type=t)
        if np.any(value[:,0][value[:,1]==0]!=0) or np.any(value[:,2][value[:,3]==0]!=0):
            fail('NODE_OMICS_MISSING_VALUE_NONZERO','Unavailable values must use zero plus an explicit zero mask',node_type=t)
        node_omics[t]=torch.as_tensor(value,dtype=torch.float32,device=device)
    degree={t:torch.zeros(len(x),dtype=torch.float32,device=device) for t,x in features.items()}
    for k,index in arrays.items():
        if not k.startswith('edge::'):continue
        rel=tuple(k[6:].split('|'));edge=torch.as_tensor(index,dtype=torch.long,device=device)
        w=torch.as_tensor(arrays['weight::'+k[6:]],dtype=torch.float32,device=device)
        if edge.ndim!=2 or edge.shape[0]!=2 or edge.shape[1]!=len(w):fail('EDGE_SHAPE','Invalid graph relation',relation=rel)
        # Magnitudes in raw array; sign is explicitly encoded in relation name.
        if 'negative' in rel[1]:w=-w.abs()
        data[rel].edge_index=edge;data[rel].edge_weight=w
        if edge.numel():
            degree[rel[0]].index_add_(0,edge[0],torch.ones_like(w));degree[rel[2]].index_add_(0,edge[1],torch.ones_like(w))
    for t,value in features.items():
        # No ID embedding.  Disabled inputs preserve the historical two
        # columns exactly; enabled inputs append value/mask pairs to all types.
        base=torch.stack([torch.log1p(degree[t]),torch.ones_like(degree[t])],1)
        data[t].x=torch.cat([base,node_omics[t]],1) if node_omics else base
    n=len(features['lncRNA'])
    return {'data':data,'features':features,
        'singlecell':torch.as_tensor(arrays.get('singlecell',np.zeros((n,1))),dtype=torch.float32,device=device),
        'sc_available':torch.as_tensor(arrays.get('sc_available',np.zeros(n)),dtype=torch.float32,device=device),
        'mutation':torch.as_tensor(arrays.get('mutation',np.zeros((n,3))),dtype=torch.float32,device=device),
        'mutation_available':torch.as_tensor(arrays.get('mutation_available',np.zeros(n)),dtype=torch.float32,device=device)}
