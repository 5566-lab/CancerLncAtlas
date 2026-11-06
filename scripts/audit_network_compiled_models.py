"""Audit saved base/module models through the vectorized compiled export API."""
from __future__ import annotations
import argparse,json,socket,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cla_engine.common import sha
from scripts import audit_network_final_models as base
from cc_hhgt import network_ridge_predict_fast as fast

def audit_compiled(inputs,networks,result,output,cancer,resolution4=False,validation_receipt=None):
    if socket.gethostname()!="149":raise RuntimeError("HOST_149_REQUIRED")
    binding=dict(wrapper_sha256=sha(Path(__file__)),
                 base_auditor_sha256=sha(ROOT/"scripts/audit_network_final_models.py"),
                 fast_export_sha256=sha(ROOT/"cc_hhgt/network_ridge_predict_fast.py"),
                 resolution4=bool(resolution4))
    if validation_receipt:
        receipt=json.loads(Path(validation_receipt).read_text())
        if receipt["status"]!="PASS" or receipt["fast_export_sha256"]!=binding["fast_export_sha256"]:
            raise ValueError("FAST_EXPORT_VALIDATION_MISMATCH")
        binding["validation_receipt_sha256"]=sha(validation_receipt)
    cache={};counts=dict(compiled_models=0,prediction_calls=0,coefficient_calls=0)
    def compiled(model,preprocess,reference_blocks,feature_map,rna_features):
        key=(id(model),id(preprocess))
        if cache.get("key")!=key:
            cache.clear()
            cache.update(key=key,model=fast.compile_model(model,preprocess,reference_blocks,feature_map,rna_features))
            counts["compiled_models"]+=1
        return cache["model"]
    def predict(model,preprocess,reference_blocks,new_blocks,new_covariates,feature_map,rna_features):
        counts["prediction_calls"]+=1
        return fast.predict_compiled(compiled(model,preprocess,reference_blocks,feature_map,rna_features),new_blocks,new_covariates)
    def coefficients(model,preprocess,reference_blocks,feature_map,rna_features):
        counts["coefficient_calls"]+=1
        return compiled(model,preprocess,reference_blocks,feature_map,rna_features)["coefficients"]
    def dump(path,value):
        if path.name in ("PREFLIGHT.json","COMPLETE.json"):
            value={**value,"export_implementation":"vectorized_compiled","export_binding":binding}
        if path.name=="COMPLETE.json":
            if value["models_checked"]!=counts["compiled_models"]:raise ValueError("COMPILED_MODEL_COUNT_MISMATCH")
            value["compiled_export_calls"]=dict(counts)
        temp=path.with_suffix(path.suffix+".tmp")
        temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n")
        temp.replace(path)
    previous=(base.predict,base.coefficients,base.dump,base.VARIANTS)
    try:
        base.predict=predict;base.coefficients=coefficients;base.dump=dump
        if resolution4:
            base.VARIANTS=("module_resolution4_union",)+tuple("module_resolution4_union_random_"+str(s) for s in (101,202,303,404,505))
        base.audit_cancer(inputs,networks,result,output,cancer)
    finally:
        base.predict,base.coefficients,base.dump,base.VARIANTS=previous

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for name in ("inputs","networks","result","output","cancer"):p.add_argument("--"+name,required=True)
    p.add_argument("--resolution4",action="store_true");p.add_argument("--validation-receipt")
    audit_compiled(**vars(p.parse_args()))
