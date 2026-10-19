"""Recompute native five-fold ssGSEA scores from their pinned expression inputs.

This proves numerical consistency with the recorded producer, not independent
biological validation. Upstream normalization and resource ancestry are inspected
separately. Published files are never rewritten and execution is limited to 149.
"""
from pathlib import Path
import subprocess
import tempfile

from .cnv_input_sources import require_host
from .current_core_export import verified_path,sha256
from .formal_expression_sources import read,write,record

ROOT=Path(__file__).resolve().parents[2]
PRODUCER=ROOT/"scripts/materialize_fold_local_ssgsea.R"
HELPER=ROOT/"scripts/frozen_ssgsea.R"
COMPARATOR=ROOT/"scripts/compare_fold_local_normalization.R"
FORMAT="NATIVE_SSGSEA_NUMERIC_REPLAY_V1"
METADATA=("format","target_host","training_started","cancer_id","fold","method",
          "training_score_range","training_detection_required_n","versions")


def declared_release(item,root):
    path=verified_path(item,root);value=read(path)
    if (value.get("status")!="PASS" or value.get("target_host")!="149"
            or value.get("training_started") is not False or set(value.get("folds",{}))!=set(map(str,range(5)))):
        raise ValueError("ssGSEA replay requires a completed five-fold release")
    return path,value


def release_sources(item,root):
    from .fold_local_activity import native
    path,value=declared_release(item,root)
    folds={};inputs=None;gmts=None
    for fold,source in value["folds"].items():
        apath,activity,npath,norm,_=native(source,path.parent)
        if activity["fold"]!=int(fold) or activity["cancer_id"]!=value.get("cancer_id"):
            raise ValueError("ssGSEA replay release fold/cancer differs")
        actual={name:record(verified_path(r,apath.parent)) for name,r in activity["inputs"].items()}
        actual_gmts={name:record(verified_path(r,apath.parent)) for name,r in activity["gmt_sources"].items()}
        if (inputs is not None and actual!=inputs) or (gmts is not None and actual_gmts!=gmts):
            raise ValueError("ssGSEA replay folds do not share expression/annotation/GMT inputs")
        inputs=actual;gmts=actual_gmts;folds[int(fold)]=(apath,activity,npath,norm)
    return path,value,folds,inputs,gmts


def replay(item,root,activity,*,cache=None):
    require_host()
    path,value=declared_release(item,root)
    selected=verified_path(activity,root)
    if sum(r["sha256"]==activity["sha256"] for r in value["folds"].values())!=1:
        raise ValueError("Consumed ssGSEA is absent from its completed release")
    key=(FORMAT,item["sha256"])
    if cache is not None and key in cache:
        from .fold_local_activity import native
        native(activity,root)  # Rehash consumed data and code even on a cache hit.
        return path,value
    path,value,folds,inputs,gmts=release_sources(item,root)
    expected_code={sha256(PRODUCER),sha256(HELPER)}
    for _,info,_,_ in folds.values():
        if {r["sha256"] for r in info["implementations"]}!=expected_code:
            raise ValueError("ssGSEA replay needs the pinned native implementation")
    bound=[record(path),record(selected),*inputs.values(),*gmts.values()]
    for apath,info,npath,norm in folds.values():
        bound.extend([record(apath),record(npath),
            *[record(verified_path(r,apath.parent)) for r in info["outputs"].values()],
            *[record(verified_path(r,apath.parent)) for r in info["implementations"]],
            *[record(verified_path(norm["outputs"][name],npath.parent)) for name in ("gene","normalization_samples")]])
    codes=[record(PRODUCER),record(HELPER),record(COMPARATOR)]
    with tempfile.TemporaryDirectory(prefix="lncatlas-ssgsea-replay-") as temporary:
        directory=Path(temporary)
        write(directory/"LAUNCH_PREFLIGHT.json",{"target_host":"149","operation":FORMAT,
            "training_started":False,"ssgsea_training_scale_replayed":True,"source_release":record(path)})
        output=directory/"recomputed"
        result=subprocess.run(["/usr/bin/Rscript",str(PRODUCER),
            str(Path(inputs["normalization_success"]["path"]).parent),inputs["annotation"]["path"],
            inputs["gmt_manifest"]["path"],str(output)],cwd=ROOT,capture_output=True,text=True,timeout=900)
        if result.returncode:raise ValueError("Native ssGSEA producer replay failed: "+result.stderr[-2000:])
        _,_,computed,recomputed_inputs,recomputed_gmts=release_sources(record(output/"SUCCESS.json"),directory)
        if recomputed_inputs!=inputs or recomputed_gmts!=gmts:
            raise ValueError("ssGSEA replay consumed different input files")
        pairs=[];metadata=[]
        for fold,(actual_path,actual,_,_) in sorted(folds.items()):
            expected_path,expected,_,_=computed[fold]
            metadata.append({"actual":{k:actual[k] for k in METADATA},"expected":{k:expected[k] for k in METADATA}})
            for name,source in actual["outputs"].items():
                pairs.append({"role":str(fold)+"."+name,"actual":str(verified_path(source,actual_path.parent)),
                    "expected":str(verified_path(expected["outputs"][name],expected_path.parent)),
                    "kind":"rds" if name in {"raw","activity","state"} else "table"})
        manifest=directory/"COMPARE.json";write(manifest,{"format":FORMAT,"pairs":pairs,"metadata":metadata})
        result=subprocess.run(["/usr/bin/Rscript",str(COMPARATOR),str(manifest)],
            cwd=ROOT,capture_output=True,text=True,timeout=900)
        if result.returncode:raise ValueError("Native ssGSEA numerical replay differs: "+result.stderr[-2000:])
    for source in [*bound,*codes]:verified_path(source,ROOT)
    if cache is not None:cache.add(key)
    return path,value


def inspect(inventory,release,activity,normalization,root,model,role,fold):
    if release is None:
        inventory.gap(model,role,"SSGSEA_REPLAY_RELEASE_NOT_BOUND",fold,activity["sha256"])
        inventory.gap(model,role,"SSGSEA_RAW_SCORE_TRANSFORMATION_NOT_REPLAYED",fold,activity["sha256"])
        return False
    replay(release,root,activity,cache=inventory.visited)
    inventory.bind(release,root,model,role+".activity_release",fold)
    inventory.transform_replays.append({"model_id":model,"dependency_context":inventory.context_id,
        "patient_fold_id":fold,"role":role,"format":FORMAT,"source_release_sha256":release["sha256"],
        "normalization_sha256":normalization["sha256"],"activity_sha256":activity["sha256"],
        "folds_recomputed":5,"outputs_compared":25,"model_training_started":False})
    return True
