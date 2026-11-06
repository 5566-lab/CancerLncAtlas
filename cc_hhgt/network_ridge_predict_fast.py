"""Vectorized frozen base/module export, with reusable compiled prediction.

This preserves network_ridge_predict's mathematics and feature ordering while
avoiding one pandas row construction per feature. Training code is unchanged.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
from cc_hhgt.graph_ridge import transform
from cc_hhgt.m0_covariates import query_design

def coefficients(model, preprocess, reference_blocks, feature_map, rna_features):
    if model["kind"] not in ("module", "base"):
        raise ValueError("USE_LEGACY_LAPLACIAN_KERNEL_FOR_THIS_MODEL")
    arm=model["arm"];variant=model["variant"]
    blocks=[] if variant=="omics_only" else ["RNA"]
    if variant!="rna_only":blocks+=list(model["allowed_blocks"])
    entries=[];rna_types=rna_features.feature_type.to_numpy()
    for block in blocks:
        spec=preprocess["preprocess"][block]
        x=transform(reference_blocks[block],spec)
        ix=np.asarray(spec["index"])
        selector=np.arange(len(ix));scale_key=block
        if block=="RNA":
            types=rna_types[ix]
            selector=np.arange(len(ix)) if arm=="combined" else np.flatnonzero(types==arm)
            scale_key=arm
        positions=ix[selector];scale=preprocess["scales"][scale_key]
        fm=feature_map[feature_map.block.eq(block)].sort_values("position")
        labels=np.full(len(positions),"base",dtype=object)
        if model["kind"]=="module":
            if model["mode"]=="omics":
                labels[:]=arm if block=="RNA" else block
            else:
                keys=rna_types[positions].astype(str) if block=="RNA" else np.full(len(positions),block)
                labels=np.array(["background:"+key for key in keys],dtype=object)
                valid=positions<spec["original_width"]
                modules=np.full(len(positions),-1,dtype=np.int64)
                modules[valid]=fm["module_"+model["mode"]].to_numpy(dtype=np.int64)[positions[valid]]
                connected=modules>=0
                labels[connected]=np.array(["module:"+str(m) for m in modules[connected]],dtype=object)
        names=np.full(len(positions),block+":ASSAY_AVAILABLE",dtype=object)
        valid=positions<spec["original_width"]
        names[valid]=fm.feature_id.to_numpy()[positions[valid]].astype(str)
        entries.append(dict(block=block,selector=selector,x=x[:,selector],positions=positions,
                            labels=labels,scale=scale,names=names.tolist()))
    alpha=None;mass_lookup={}
    if model["kind"]=="module":
        unique=np.unique(np.concatenate([e["labels"] for e in entries]))
        mass=np.zeros(len(unique),dtype=float)
        for entry in entries:
            # Unbuffered updates preserve the original per-feature summation order.
            loc=np.searchsorted(unique,entry["labels"])
            np.add.at(mass,loc,1/(entry["scale"]*len(blocks)))
        mass_lookup=dict(zip(unique,mass))
        alpha=dict(zip(model["group_labels"],model["group_weights"]))
        if set(alpha)!=set(mass_lookup):raise ValueError("FROZEN_GROUP_MEMBERSHIP_DRIFT")
        if abs(sum(mass_lookup.values())-1)>1e-8:raise ValueError("FROZEN_SCALING_DRIFT")
    result={}
    train=np.asarray(model["train"],int);dual=np.asarray(model["dual"]).ravel()
    for e in entries:
        if alpha is None:weights=np.ones(len(e["labels"]))
        else:weights=np.array([alpha[label]/mass_lookup[label] for label in e["labels"]])
        beta=(e["x"][train].T@dual)*weights/(e["scale"]*len(blocks))
        result[e["block"]]=dict(selector=e["selector"],feature_positions=e["positions"],
            feature_ids=e["names"],standardized_coefficients=beta)
    return result

def compile_model(model, preprocess, reference_blocks, feature_map, rna_features):
    """Export reusable coefficients and frozen preprocessing; retain no patient rows."""
    coefs=coefficients(model,preprocess,reference_blocks,feature_map,rna_features)
    return dict(version=1,variant=model["variant"],arm=model["arm"],
        coefficients=coefs,design_spec=deepcopy(preprocess["design_spec"]),
        preprocess={b:deepcopy(preprocess["preprocess"][b]) for b in coefs},
        nuisance=np.asarray(model["nuisance"]).ravel().copy())

def predict_compiled(compiled, new_blocks, new_covariates):
    if compiled["version"]!=1:raise ValueError("UNSUPPORTED_COMPILED_MODEL_VERSION")
    z,supported=query_design(new_covariates,compiled["design_spec"])
    pred=z@compiled["nuisance"]
    for block,info in compiled["coefficients"].items():
        x=transform(new_blocks[block],compiled["preprocess"][block])
        pred+=x[:,info["selector"]]@info["standardized_coefficients"]
    pred[~supported]=np.nan
    return pred,supported

def predict(model, preprocess, reference_blocks, new_blocks, new_covariates, feature_map, rna_features):
    return predict_compiled(compile_model(model,preprocess,reference_blocks,feature_map,rna_features),
                            new_blocks,new_covariates)
