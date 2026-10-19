"""Replay the native five-fold TMM producer from counts, without model fitting.

Only the current pinned producer is executed, in a temporary directory on 149.
Published inputs/outputs are never rewritten. A completed normalization release
is required; a conversion receipt by itself is not numerical replay evidence.
"""
from pathlib import Path
import subprocess
import tempfile

from .cnv_input_sources import require_host
from .current_core_export import verified_path,sha256
from .formal_expression_sources import read,write,record

ROOT=Path(__file__).resolve().parents[2]
PRODUCER=ROOT/"scripts/materialize_fold_local_expression.R"
HELPER=ROOT/"scripts/frozen_tmm.R"
COMPARATOR=ROOT/"scripts/compare_fold_local_normalization.R"
FORMAT="NATIVE_TMM_NUMERIC_REPLAY_V1"
METADATA=("format","target_host","training_started","cancer_id","fold","normalization",
          "reference_sample","mean_train_effective_library_size","raw_file_exposure",
          "raw_source_ancestry_complete","versions")


def declared_release(item,root):
    path=verified_path(item,root);value=read(path)
    if (value.get("status")!="PASS" or value.get("target_host")!="149"
            or value.get("training_started") is not False or set(value.get("folds",{}))!=set(map(str,range(5)))):
        raise ValueError("Normalization replay requires a completed five-fold release")
    return path,value


def release_sources(item,root):
    from .fold_local_expression import native
    path,value=declared_release(item,root)
    folds={};inputs=None
    for fold,source in value["folds"].items():
        npath,norm,_=native(source,path.parent)
        if norm["fold"]!=int(fold) or norm["cancer_id"]!=value.get("cancer_id"):
            raise ValueError("Normalization replay release fold/cancer differs")
        actual={name:record(verified_path(r,npath.parent)) for name,r in norm["inputs"].items()}
        if inputs is not None and actual!=inputs:
            raise ValueError("Normalization replay folds do not share count/patient inputs")
        inputs=actual;folds[int(fold)]=(npath,norm)
    return path,value,folds,inputs


def replay(item,root,normalization,*,cache=None):
    require_host()
    path,value=declared_release(item,root)
    selected=verified_path(normalization,root)
    if sum(r["sha256"]==normalization["sha256"] for r in value["folds"].values())!=1:
        raise ValueError("Consumed normalization is absent from its completed release")
    key=(FORMAT,item["sha256"])
    if cache is not None and key in cache:
        from .fold_local_expression import native
        native(normalization,root)  # Recheck consumed bytes before using cached numerical proof.
        return path,value
    path,value,folds,inputs=release_sources(item,root)
    expected_code={sha256(PRODUCER),sha256(HELPER)}
    for _,norm in folds.values():
        if {r["sha256"] for r in norm["implementations"]}!=expected_code:
            raise ValueError("Normalization replay needs the pinned native implementation")
    bound=[record(path),record(selected),*inputs.values()]
    for npath,norm in folds.values():
        bound.extend([record(npath),*[record(verified_path(r,npath.parent)) for r in norm["outputs"].values()],
                      *[record(verified_path(r,npath.parent)) for r in norm["implementations"]]])
        if norm.get("counts_assembly"):bound.append(record(verified_path(norm["counts_assembly"],npath.parent)))
    codes=[record(PRODUCER),record(HELPER),record(COMPARATOR)]
    with tempfile.TemporaryDirectory(prefix="lncatlas-tmm-replay-") as temporary:
        directory=Path(temporary)
        write(directory/"LAUNCH_PREFLIGHT.json",{"target_host":"149","operation":FORMAT,
            "training_started":False,"normalization_fitting_replayed":True,"source_release":record(path)})
        output=directory/"recomputed"
        result=subprocess.run(["/usr/bin/Rscript",str(PRODUCER),inputs["counts"]["path"],
            inputs["patient_map"]["path"],value["cancer_id"],str(output)],
            cwd=ROOT,capture_output=True,text=True,timeout=900)
        if result.returncode:raise ValueError("Native TMM producer replay failed: "+result.stderr[-2000:])
        _,_,computed,_=release_sources(record(output/"SUCCESS.json"),directory)
        pairs=[];metadata=[]
        for fold,(actual_path,actual) in sorted(folds.items()):
            expected_path,expected=computed[fold]
            metadata.append({"actual":{k:actual[k] for k in METADATA},"expected":{k:expected[k] for k in METADATA}})
            for name,source in actual["outputs"].items():
                pairs.append({"role":str(fold)+"."+name,"actual":str(verified_path(source,actual_path.parent)),
                    "expected":str(verified_path(expected["outputs"][name],expected_path.parent)),
                    "kind":"rds" if name in {"lncrna","gene","state"} else "table"})
        manifest=directory/"COMPARE.json";write(manifest,{"pairs":pairs,"metadata":metadata})
        result=subprocess.run(["/usr/bin/Rscript",str(COMPARATOR),str(manifest)],
            cwd=ROOT,capture_output=True,text=True,timeout=900)
        if result.returncode:raise ValueError("Native TMM numerical replay differs: "+result.stderr[-2000:])
    for source in [*bound,*codes]:verified_path(source,ROOT)
    if cache is not None:cache.add(key)
    return path,value


def inspect(inventory,release,normalization,root,model,role,fold):
    if release is None:
        inventory.gap(model,role,"NORMALIZATION_REPLAY_RELEASE_NOT_BOUND",fold,normalization["sha256"])
        return False
    path,value=replay(release,root,normalization,cache=inventory.visited)
    inventory.bind(release,root,model,role+".normalization_release",fold)
    inventory.transform_replays.append({"model_id":model,"dependency_context":inventory.context_id,
        "patient_fold_id":fold,"role":role,"format":FORMAT,"source_release_sha256":release["sha256"],
        "normalization_sha256":normalization["sha256"],"folds_recomputed":5,"outputs_compared":25,
        "model_training_started":False})
    return True
