"""Native HM450 materialization/recovery and checked distal feature handoff.

Recovery reconstructs batches, not unavailable raw beta files. Neither path
deletes input data, fits a model, or upgrades a cohort-selected topology.
"""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import tempfile

import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read,record
from .current_analysis_workflow import write
from .input_lineage import artifact_sha256
from .context_table_replay import compare_tables

ROOT=Path(__file__).resolve().parents[2]
INPUT="NATIVE_METHYLATION_WORKFLOW_INPUT_V1"
FORMAT="NATIVE_METHYLATION_WORKFLOW_RESULTS_V1"
CODE=("cc_hhgt/v32/native_methylation_workflow.py","scripts/run_native_methylation_workflow.py",
      "scripts/materialize_v32_methylation_lncrna.py","cc_hhgt/v32/methylation_recovery.py",
      "cc_hhgt/v32/distal_topology.py","cc_hhgt/v32/context_table_replay.py")


def plan(manifest,output,*,new=True):
    require_host();manifest=Path(manifest).resolve();spec=read(manifest);output=Path(output).resolve()
    required={"format","mode","selected_files","probe_map","patient_folds","download_root"}
    if (not required<=set(spec) or set(spec)-required-{"batches","workers","batch_size"}
            or spec["format"]!=INPUT or spec["mode"] not in {"raw","recovery"}
            or ("batches" in spec)!=(spec["mode"]=="recovery")):
        raise ValueError("Invalid native methylation workflow")
    if new and output.exists():raise FileExistsError("Use a new methylation workflow directory")
    workers=spec.get("workers",8);batch_size=spec.get("batch_size",64)
    if any(type(v) is not int or v<1 for v in (workers,batch_size)):
        raise ValueError("Methylation workers and batch size must be positive integers")
    files={k:record(verified_path(spec[k],manifest.parent)) for k in ("selected_files","probe_map","patient_folds")}
    source_files={v["path"]:v for v in files.values()}
    probe_audit=Path(files["probe_map"]["path"]).parent/"AUDIT.json"
    if probe_audit.is_file():source_files[str(probe_audit)]=record(probe_audit)
    from .distal_topology import methylation_probe_lineage
    from .methylation_recovery import read_patient_folds
    lineage=methylation_probe_lineage(Path(files["probe_map"]["path"]))
    if probe_audit.is_file() and "native_probe_preparation" in read(probe_audit):
        from .native_probe_workflow import validate_map
        for pin in validate_map(Path(files["probe_map"]["path"])):source_files[pin["path"]]=pin
    read_patient_folds(Path(files["patient_folds"]["path"]))
    raw=(manifest.parent/spec["download_root"]).resolve();batches=None
    if spec["mode"]=="raw":
        from scripts.materialize_v32_methylation_lncrna import _verify_file
        selected=pd.read_csv(files["selected_files"]["path"],sep="\t",dtype=str)
        keys={"file_id","file_name","md5sum","file_size","cancer_id","patient_id","sample_id"}
        if (selected.empty or not keys<=set(selected) or selected[list(keys)].isna().any().any()
                or selected.file_id.duplicated().any()):raise ValueError("Invalid selected methylation files")
        for row in selected.to_dict("records"):
            path=(raw/row["file_id"]/row["file_name"]).resolve()
            if raw not in path.parents:raise ValueError("Methylation raw file is outside its declared cache")
            if _verify_file({**row,"path":str(path)})["status"]!="PASS":
                raise ValueError("Methylation raw file checksum/size mismatch: "+row["file_id"])
            source_files[str(path)]=record(path)
    else:
        batches=(manifest.parent/spec["batches"]["path"]).resolve()
        if not batches.is_dir() or artifact_sha256(batches)!=spec["batches"]["sha256"]:
            raise ValueError("Methylation batch directory checksum differs")
        selected=list(sorted(batches.glob("BATCH_*.parquet")))
        if not selected:raise ValueError("No completed methylation batches")
        for path in selected:source_files[str(path)]=record(path)
        for name in ("BATCH_INPUT_AUTHORITY.json","SAMPLE_PROCESSING_RECORDS.parquet"):
            path=batches.parent/name
            if path.is_file():source_files[str(path)]=record(path)
    for source in [raw,*([batches] if batches else [])]:
        if source==output or source in output.parents:raise ValueError("Methylation output must be outside source directories")
    return {"input":record(manifest),"mode":spec["mode"],"target_host":"149","training_started":False,
        "files":files,"download_root":str(raw),"batches":None if batches is None else {
            "path":str(batches),"sha256":artifact_sha256(batches)},
        "workers":workers,"batch_size":batch_size,"probe_topology":lineage,
        "source_files":sorted(source_files.values(),key=lambda r:r["path"]),
        "implementations":[record(ROOT/name) for name in CODE],"raw_deletion_executed":False}


def invoke(value,output):
    from scripts.materialize_v32_methylation_lncrna import main
    args=[part for key,pin in value["files"].items() for part in ("--"+key.replace("_","-"),pin["path"])]
    args += ["--download-root",value["download_root"],"--output",str(output),"--workers",str(value["workers"]),
             "--batch-size",str(value["batch_size"])]
    if value["mode"]=="recovery":args += ["--resume-batches",value["batches"]["path"]]
    if main(args)!=0:raise RuntimeError("Native methylation command failed")


def audit_outputs(value,output):
    audit_path=output/"AUDIT.json";audit=read(audit_path)
    if (audit.get("format")!="CC_HHGT_V3_2_PATIENT_LNCRNA_METHYLATION_FEATURES_V1"
            or audit.get("status")!="PASS_PATIENT_LNCRNA_METHYLATION_FEATURES"
            or audit.get("target_host")!="149" or audit.get("training_started") is not False
            or audit["inputs"]!=value["files"]):raise ValueError("Incomplete native methylation result")
    with tempfile.TemporaryDirectory(prefix="native_methylation_replay_") as temporary:
        replay=Path(temporary)/"features"
        with (Path(temporary)/"replay.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):
            invoke(value,replay)
        expected=read(replay/"AUDIT.json")
        if {k:v for k,v in audit.items() if k!="outputs"}!={k:v for k,v in expected.items() if k!="outputs"}:
            raise ValueError("Methylation metadata differs from native replay")
        if set(audit["outputs"])!=set(expected["outputs"]) or audit["outputs"]["sample_batches"]!=expected["outputs"]["sample_batches"]:
            raise ValueError("Methylation output inventory differs")
        for key,keys in (("patient_features",["cancer_id","patient_id","lncrna_id"]),("sample_records",["file_id"])):
            target=verified_path(audit["outputs"][key],output)
            reference=Path(expected["outputs"][key]["path"])
            if target!=output/reference.name:raise ValueError("Methylation output is outside its producer")
            compare_tables(target,reference,keys,label="Methylation")
        if value["mode"]=="raw":
            actual=sorted((output/"sample_feature_batches").glob("BATCH_*.parquet"))
            references=sorted((replay/"sample_feature_batches").glob("BATCH_*.parquet"))
            if [p.name for p in actual]!=[p.name for p in references]:raise ValueError("Methylation output batch inventory differs")
            for left,right in zip(actual,references):compare_tables(left,right,["file_id","lncrna_id"],label="Methylation sample batch")
            if read(output/"BATCH_INPUT_AUTHORITY.json")!=read(replay/"BATCH_INPUT_AUTHORITY.json"):
                raise ValueError("Methylation output aggregation authority differs")
    return audit


def run(manifest,output):
    value=plan(manifest,output);output=Path(output).resolve();output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",value)
    try:
        with (output/"stage.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):
            invoke(value,output/"features");audit=audit_outputs(value,output/"features")
        if value!=plan(manifest,output,new=False):raise ValueError("Methylation sources changed during execution")
        batches=output/"features/sample_feature_batches" if value["mode"]=="raw" else Path(value["batches"]["path"])
        path=output/"NATIVE_METHYLATION_BINDING.json"
        write(path,{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
            "mode":value["mode"],"preflight":record(output/"LAUNCH_PREFLIGHT.json"),"audit":record(output/"features/AUDIT.json"),
            "outputs":audit["outputs"],"batch_root":{"path":str(batches),"sha256":artifact_sha256(batches)},
            "native_numerical_replay_verified":True,"raw_checksum_reverified":value["mode"]=="raw",
            "raw_deletion_executed":False,"production_deployed":False,"source_ancestry_complete":False})
        return path
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise


def validate(item,root):
    require_host();path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or binding.get("native_numerical_replay_verified") is not True
            or any(binding.get(k) is not False for k in ("training_started","raw_deletion_executed","production_deployed","source_ancestry_complete"))):
        raise ValueError("Incomplete native methylation workflow")
    value=read(verified_path(binding["preflight"],path.parent))
    if value!=plan(verified_path(value["input"],path.parent),path.parent,new=False):raise ValueError("Methylation workflow sources or implementation differ")
    native=verified_path(binding["audit"],path.parent)
    if native!=path.parent/"features/AUDIT.json":raise ValueError("Methylation result location differs")
    audit=audit_outputs(value,native.parent)
    batches=native.parent/"sample_feature_batches" if value["mode"]=="raw" else Path(value["batches"]["path"])
    if (binding["mode"]!=value["mode"] or binding["raw_checksum_reverified"] is not (value["mode"]=="raw")
            or binding["outputs"]!=audit["outputs"] or binding["batch_root"]!={"path":str(batches),"sha256":artifact_sha256(batches)}):
        raise ValueError("Methylation workflow result handoff differs")
    return path,binding


def validate_distal_input(prepared,root):
    """Verify that assembled patient values actually match the bound producer."""
    if "methylation_workflow" not in prepared:return []
    path,binding=validate(prepared["methylation_workflow"],root)
    features=verified_path(binding["outputs"]["patient_features"],path.parent)
    if verified_path(prepared["sources"]["methylation"],root)!=features:
        raise ValueError("Distal input uses a different methylation workflow")
    keys=["cancer_id","patient_id","lncrna_id"]
    names=[f"{region}_methylation_beta{suffix}" for region in ("promoter","distal") for suffix in ("","__available")]
    methylation=pd.read_parquet(features)
    for pin in prepared["outputs"]:
        actual=pd.read_parquet(verified_path(pin,root))
        expected=actual[keys].merge(methylation[keys+names+["patient_fold_id"]],on=keys,how="left",validate="one_to_one")
        for name in names:
            if name.endswith("__available"):expected[name]=expected[name].fillna(False).astype(bool)
        matched=expected.patient_fold_id.notna()
        if not actual.loc[matched,"patient_fold_id"].reset_index(drop=True).eq(expected.loc[matched,"patient_fold_id"].reset_index(drop=True)).all():
            raise ValueError("Distal methylation patient folds differ")
        try:pd.testing.assert_frame_equal(actual[keys+names].reset_index(drop=True),expected[keys+names],check_dtype=False,atol=1e-12,rtol=0)
        except AssertionError as error:raise ValueError("Distal methylation values differ from native workflow") from error
    plan_path=verified_path(binding["preflight"],path.parent);value=read(plan_path)
    return [record(path),record(plan_path),record(features),binding["audit"],value["input"],*value["source_files"],*value["implementations"]]
