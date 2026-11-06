"""Predict with frozen module/base ridge models and expose assay coefficients.

No outcomes or fitting enter prediction. Reference columns, final preprocessing,
network membership, and the saved dual are the complete prediction contract.
Legacy RNA Laplacians continue to use their separate kernel implementation.
"""
from __future__ import annotations
import numpy as np
from cc_hhgt.graph_ridge import transform
from cc_hhgt.m0_covariates import query_design

def coefficients(model,preprocess,reference_blocks,feature_map,rna_features):
    if model['kind'] not in ('module','base'):raise ValueError('USE_LEGACY_LAPLACIAN_KERNEL_FOR_THIS_MODEL')
    arm=model['arm'];variant=model['variant'];blocks=[] if variant=='omics_only' else ['RNA']
    if variant!='rna_only':blocks+=list(model['allowed_blocks'])
    entries=[];masses={}
    for block in blocks:
        spec=preprocess['preprocess'][block];x=transform(reference_blocks[block],spec);ix=spec['index'];selector=np.arange(len(ix));scale_key=block
        if block=='RNA':
            types=rna_features.feature_type.to_numpy()[ix];selector=np.arange(len(ix)) if arm=='combined' else np.flatnonzero(types==arm);scale_key=arm
        positions=ix[selector];scale=preprocess['scales'][scale_key];fm=feature_map[feature_map.block.eq(block)].sort_values('position');labels=[]
        for position in positions:
            key=block if block!='RNA' else str(rna_features.iloc[position].feature_type)
            if model['kind']=='module' and model['mode']=='omics':label=arm if block=='RNA' else block
            elif model['kind']=='module':
                module=int(fm.iloc[position]['module_'+model['mode']]) if position<spec['original_width'] else -1;label='module:'+str(module) if module>=0 else 'background:'+key
            else:label='base'
            labels.append(label);masses[label]=masses.get(label,0)+1/(scale*len(blocks))
        entries.append((block,selector,x[:,selector],positions,np.asarray(labels),scale))
    if model['kind']=='module':
        alpha=dict(zip(model['group_labels'],model['group_weights']))
        if set(alpha)!=set(masses):raise ValueError('FROZEN_GROUP_MEMBERSHIP_DRIFT')
        if abs(sum(masses.values())-1)>1e-8:raise ValueError('FROZEN_SCALING_DRIFT')
    result={}
    for block,selector,x,positions,labels,scale in entries:
        weights=np.array([alpha[label]/masses[label] for label in labels]) if model['kind']=='module' else np.ones(len(labels));beta=(x[np.asarray(model['train'],int)].T@np.asarray(model['dual']).ravel())*weights/(scale*len(blocks))
        fm=feature_map[feature_map.block.eq(block)].sort_values('position');names=[str(fm.iloc[i].feature_id) if i<preprocess['preprocess'][block]['original_width'] else block+':ASSAY_AVAILABLE' for i in positions]
        result[block]=dict(selector=selector,feature_positions=positions,feature_ids=names,standardized_coefficients=beta)
    return result

def predict(model,preprocess,reference_blocks,new_blocks,new_covariates,feature_map,rna_features):
    coefs=coefficients(model,preprocess,reference_blocks,feature_map,rna_features);z,supported=query_design(new_covariates,preprocess['design_spec']);pred=z@np.asarray(model['nuisance']).ravel()
    for block,info in coefs.items():
        x=transform(new_blocks[block],preprocess['preprocess'][block]);pred+=x[:,info['selector']]@info['standardized_coefficients']
    pred[~supported]=np.nan
    return pred,supported
