"""Native raw-matrix distal ATAC aggregation with checked downstream replay."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile

import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read,record
from .current_analysis_workflow import write
from .input_lineage import artifact_sha256

ROOT=Path(__file__).resolve().parents[2]
INPUT="NATIVE_DISTAL_ATAC_WORKFLOW_INPUT_V1"
FORMAT="NATIVE_DISTAL_ATAC_WORKFLOW_RESULTS_V1"
ROLES=("matrix","mapping","peak_set","topology_links","patient_folds","candidate_universe")
CODE=("cc_hhgt/v32/native_distal_atac_workflow.py","scripts/run_native_distal_atac_workflow.py",
      "scripts/materialize_v32_distal_atac_lncrna.R","cc_hhgt/v32/distal_topology.py")
RSCRIPT=Path("/usr/bin/Rscript")
_REPLAY_CACHE=set()


def environment():
    command='cat(jsonlite::toJSON(list(r=as.character(getRversion()), packages=setNames(lapply(c("data.table","Matrix","jsonlite"),function(p) as.character(packageVersion(p))),c("data.table","Matrix","jsonlite"))),auto_unbox=TRUE))'
    result=subprocess.run([str(RSCRIPT),"-e",command],capture_output=True,text=True,check=True,timeout=60)
    return {"executable":record(RSCRIPT),"versions":json.loads(result.stdout)}


def scoped_inputs(files,cancers):
    from .methylation_recovery import read_patient_folds
    folds=read_patient_folds(Path(files["patient_folds"]["path"]))
    folds=folds.loc[folds.cancer_id.isin(cancers)].sort_values(["cancer_id","patient_id"]).reset_index(drop=True)
    if set(folds.cancer_id)!=set(cancers) or any(set(g.patient_fold_id)!=set(range(5)) for _,g in folds.groupby("cancer_id")):
        raise ValueError("Distal ATAC requires all five folds in every declared cancer")
    if folds.patient_id.duplicated().any():raise ValueError("Distal ATAC patient occurs in multiple declared cancers")
    return folds


def plan(manifest,output,*,new=True):
    require_host();manifest=Path(manifest).resolve();spec=read(manifest);output=Path(output).resolve()
    if set(spec)!={"format",*ROLES} or spec["format"]!=INPUT:raise ValueError("Invalid native distal ATAC manifest")
    if new and output.exists():raise FileExistsError("Use a new distal ATAC output")
    files={k:record(verified_path(spec[k],manifest.parent)) for k in ROLES}
    from .atac_training import FORMAL_CANCERS,RAW_COVERED_CANCERS
    candidates=pd.read_parquet(files["candidate_universe"]["path"]);keys=["cancer_id","lncrna_id","pathway_id"]
    if (candidates.empty or not set(keys)<=set(candidates) or candidates[keys].isna().any().any()
            or candidates.duplicated(keys).any() or not set(candidates.cancer_id)<=set(FORMAL_CANCERS)):
        raise ValueError("Invalid distal ATAC candidate authority")
    cancers=sorted(candidates.cancer_id.unique());scoped_inputs(files,cancers)
    genes=sorted(candidates.lncrna_id.astype(str).str.removeprefix("LNC:").str.split(".").str[0].unique())
    if not all(genes):raise ValueError("Empty distal ATAC gene identifier")
    links=Path(files["topology_links"]["path"]);audit=Path(str(links)+".audit.json")
    from .distal_topology import topology_lineage,require_shared_predictive_topology
    lineage=require_shared_predictive_topology(topology_lineage(links))
    cohort={"format":"DISTAL_ATAC_COHORT_V1","cancers":cancers,
        "covered_cancers":sorted(set(cancers)&set(RAW_COVERED_CANCERS)),
        "raw_gap_cancers":sorted(set(cancers)-set(RAW_COVERED_CANCERS))}
    return {"input":record(manifest),"target_host":"149","training_started":False,
        "files":files,"topology_audit":record(audit),"topology_lineage":lineage,"cohort":cohort,"genes":genes,
        "environment":environment(),"implementations":[record(ROOT/n) for n in CODE],
        "download_started":False,"source_ancestry_complete":False}


def invoke(value,root):
    root.mkdir(parents=True,exist_ok=True);inputs=root/"inputs";inputs.mkdir()
    pd.DataFrame({"gene_id":["LNC:"+g for g in value["genes"]]}).to_csv(inputs/"genes.tsv",sep="\t",index=False)
    scoped_inputs(value["files"],value["cohort"]["cancers"]).to_csv(inputs/"folds.tsv",sep="\t",index=False)
    write(inputs/"cohort.json",value["cohort"])
    command=[str(RSCRIPT),str(ROOT/"scripts/materialize_v32_distal_atac_lncrna.R")]
    command += [value["files"][k]["path"] for k in ("matrix","mapping","peak_set","topology_links")]
    command += [str(inputs/"genes.tsv"),str(inputs/"folds.tsv"),str(root/"features"),
        "CANCERLNCATLAS_V32_DISTAL_ATAC_LNCRNA_V1",str(inputs/"cohort.json")]
    with (root/"r.log").open("w",encoding="utf-8") as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)


def compare_tsv(left,right):
    actual=pd.read_csv(left,sep="\t");expected=pd.read_csv(right,sep="\t")
    try:pd.testing.assert_frame_equal(actual,expected,check_dtype=True,rtol=0,atol=1e-12)
    except AssertionError as error:raise ValueError("Distal ATAC output differs from raw-matrix replay") from error


def validate_features(item,root):
    require_host();path=verified_path(item,root);audit=read(path)
    if (audit.get("format")!="CANCERLNCATLAS_V32_DISTAL_ATAC_LNCRNA_MATERIALIZATION_V1"
            or audit.get("status")!="PASS" or audit.get("target_host")!="149" or audit.get("training_started") is not False):
        raise ValueError("Incomplete distal ATAC materialization")
    preflight=verified_path(audit["native_distal_atac_preparation"],path.parent);value=read(preflight)
    if path!=preflight.parent/"features/DISTAL_ATAC_AUDIT.json":raise ValueError("Distal ATAC result location differs")
    if value!=plan(verified_path(value["input"],path.parent),preflight.parent,new=False):
        raise ValueError("Distal ATAC inputs or implementation changed")
    names=[c+".distal_accessibility.tsv.gz" for c in value["cohort"]["covered_cancers"]]
    declared=audit.get("output_md5",{}) or {}
    if not isinstance(declared,dict) or {Path(k).name for k in declared}!=set(names):raise ValueError("Distal ATAC output inventory differs")
    for name,digest in declared.items():
        target=Path(name)
        if target!=path.parent/target.name or hashlib.md5(target.read_bytes()).hexdigest()!=digest:
            raise ValueError("Distal ATAC output checksum differs")
    if {p.name for p in path.parent.glob("*.distal_accessibility.tsv.gz")}!=set(names):raise ValueError("Unexpected distal ATAC cancer matrix")
    outputs=[record(path.parent/n) for n in [*names,"DISTAL_LNCRNA_PEAK_TOPOLOGY.tsv.gz"]]
    inputs=[record(preflight.parent/"inputs"/n) for n in ("genes.tsv","folds.tsv","cohort.json")]
    signature=hashlib.sha256(json.dumps({"preflight":value,"audit":record(path),"outputs":outputs,"inputs":inputs},sort_keys=True).encode()).hexdigest()
    if signature not in _REPLAY_CACHE:
        with tempfile.TemporaryDirectory(prefix="native_distal_atac_replay_") as tmp:
            replay=Path(tmp)/"replay";invoke(value,replay);expected=read(replay/"features/DISTAL_ATAC_AUDIT.json")
            excluded={"output_md5","native_distal_atac_preparation"}
            if {k:v for k,v in audit.items() if k not in excluded}!={k:v for k,v in expected.items() if k not in excluded}:
                raise ValueError("Distal ATAC metadata differs from raw-matrix replay")
            for pin in outputs:compare_tsv(Path(pin["path"]),replay/"features"/Path(pin["path"]).name)
            for name in ("genes.tsv","folds.tsv"):compare_tsv(preflight.parent/"inputs"/name,replay/"inputs"/name)
            if read(preflight.parent/"inputs/cohort.json")!=read(replay/"inputs/cohort.json"):
                raise ValueError("Distal ATAC cohort differs from candidate declaration")
        if value!=plan(verified_path(value["input"],path.parent),preflight.parent,new=False):raise ValueError("Distal ATAC sources changed during replay")
        for pin in [record(preflight),item,*outputs,*inputs]:verified_path(pin,path.parent)
        _REPLAY_CACHE.add(signature)
    return [record(path),record(preflight),value["input"],*value["files"].values(),value["topology_audit"],
        *value["implementations"],value["environment"]["executable"],*outputs,*inputs]


def run(manifest,output):
    value=plan(manifest,output);output=Path(output).resolve();output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",value)
    try:
        invoke(value,output);path=output/"features/DISTAL_ATAC_AUDIT.json";audit=read(path)
        audit["native_distal_atac_preparation"]=record(output/"LAUNCH_PREFLIGHT.json");write(path,audit)
        validate_features(record(path),output)
        binding=output/"NATIVE_DISTAL_ATAC_BINDING.json"
        write(binding,{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
            "preflight":record(output/"LAUNCH_PREFLIGHT.json"),"audit":record(path),
            "feature_root":{"path":str(path.parent),"sha256":artifact_sha256(path.parent)},
            "raw_numerical_replay_verified":True,"source_ancestry_complete":False,"production_deployed":False})
        return binding
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise


def validate(item,root):
    require_host();path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or binding.get("raw_numerical_replay_verified") is not True
            or any(binding.get(k) is not False for k in ("training_started","source_ancestry_complete","production_deployed"))):
        raise ValueError("Incomplete native distal ATAC workflow")
    audit=verified_path(binding["audit"],path.parent);preflight=verified_path(binding["preflight"],path.parent)
    if audit!=path.parent/"features/DISTAL_ATAC_AUDIT.json" or preflight!=path.parent/"LAUNCH_PREFLIGHT.json":
        raise ValueError("Distal ATAC workflow location differs")
    if read(audit)["native_distal_atac_preparation"]!=record(preflight):raise ValueError("Distal ATAC preparation identity differs")
    validate_features(record(audit),path.parent)
    if binding["feature_root"]!={"path":str(audit.parent),"sha256":artifact_sha256(audit.parent)}:
        raise ValueError("Distal ATAC handoff directory differs")
    return path,binding
