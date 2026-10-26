"""Native State no-fit preparation and completed-result publication on 149.

The generated training command is never dispatched by this module. Preparation
uses the actual State CLI's train/validation collectors. Collection checks the
completed native serializer, then executes the existing measurement/forward
replay, selected GeneSet context and publication binding.
"""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from pathlib import Path
import re
import sys

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, write, record
from .native_auxiliary_fusion import Sources
from . import state_training as native

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_STATE_WORKFLOW_INPUT_V1"
PREPARED = "NATIVE_STATE_WORKFLOW_PREPARED_V1"
FORMAT = "NATIVE_STATE_WORKFLOW_RESULTS_V1"
TRAINER = "scripts.run_v32_state_training"
ROLES = {"state_measurements": "tumor_state_long_measurement", "patient_folds": "v32_patient_fold_manifest",
         "core_manifest": "v32_core_embedding_manifest", "lncrna_expression": "lncrna_expression"}
CODE = ("cc_hhgt/v32/native_state_workflow.py", "scripts/run_native_state_workflow.py",
    "cc_hhgt/v32/state_training.py", "scripts/run_v32_state_training.py",
    "cc_hhgt/v32/current_state_context.py", "cc_hhgt/v32/patient_fold_authority.py",
    "cc_hhgt/v32/patient_folds.py", "cc_hhgt/v32/integrated_model.py",
    "cc_hhgt/v32/current_core_export.py", "cc_hhgt/v32/selected_genesets.py")


def arguments(values): return [part for key,value in values.items() for part in ("--"+key.replace("_","-"),str(value))]


def plan(manifest, output, *, new=True):
    require_host(); manifest=Path(manifest).resolve(); output=Path(output).resolve(); spec=read(manifest)
    required={"format","state_run_id","core_run_id","selected_analysis","patient_authority_receipt",*ROLES}
    if (set(spec)-required-{"config","state_sets"} or not required <= set(spec) or spec["format"]!=INPUT
            or not re.fullmatch(r"v32-[a-z0-9][a-z0-9._-]*",str(spec["state_run_id"]))
            or not isinstance(spec["core_run_id"],str) or not spec["core_run_id"].strip()):
        raise ValueError("Invalid native State workflow input")
    if new and output.exists(): raise FileExistsError("Use a new State preparation directory")
    sources=Sources(); files={}
    from .current_state_context import bind_state_input, state_core_sources, StateSetConfig
    for key,role in ROLES.items():
        path=bind_state_input(sources,{**spec[key],"role":role},manifest.parent)
        files[key]={"path":str(path),"sha256":native.path_sha256(path)}
    for key in ("selected_analysis","patient_authority_receipt"):
        files[key]=record(sources.bind(spec[key],manifest.parent))
    _,_,_,_,parents=state_core_sources({**spec,**files},manifest.parent,sources)
    authority=native.state_patient_authority(files["patient_folds"]["path"],files["patient_authority_receipt"]["path"],"current")
    defaults=asdict(native.StateTrainingConfig()); override=spec.get("config",{})
    if not isinstance(override,dict) or set(override)-set(defaults): raise ValueError("Unknown State training configuration")
    cfg=native.StateTrainingConfig(**{**defaults,**override}); cfg.validate(); config=asdict(cfg)
    policy=StateSetConfig(**spec.get("state_sets",{})); policy.validate()
    base={"repo_root":ROOT,"state_measurements":files["state_measurements"]["path"],
        "patient_fold_manifest":files["patient_folds"]["path"],"patient_fold_authority_receipt":files["patient_authority_receipt"]["path"],
        "patient_authority_mode":"current","expected_core_run_id":spec["core_run_id"],
        "core_embedding_manifest":files["core_manifest"]["path"],"lncrna_expression":files["lncrna_expression"]["path"],
        "run_id":spec["state_run_id"],**{k:v for k,v in config.items() if k!="n_folds"}}
    command=[sys.executable,"-m",TRAINER,*arguments({**base,"output_root":output/"training"})]
    prepare=[sys.executable,"-m",TRAINER,*arguments({**base,"output_root":output/"head_preflight"}),"--prepare-only"]
    native.build_arg_parser().parse_args(command[3:]); native.build_arg_parser().parse_args(prepare[3:])
    for name in CODE: sources.bind(record(ROOT/name),ROOT)
    sources.bind(record(manifest),manifest.parent)
    return {"target_host":"149","training_started":False,"input":record(manifest),"files":files,
        "state_run_id":spec["state_run_id"],"core_run_id":spec["core_run_id"],"config":config,"state_sets":asdict(policy),
        "patient_authority":authority,"core_lineage":{str(k):v for k,v in parents.items()},
        "sources":sorted(sources.records.values(),key=lambda r:r["path"]),"prepare_command":prepare,
        "training_command":{"argv":command,"execution_mode":"CODE_ONLY","automatically_executed":False,
                            "requires_separate_explicit_execution":True}}


def expected_preflight(value):
    files=value["files"]; cfg=native.StateTrainingConfig(**value["config"])
    state=native.read_table(files["state_measurements"]["path"],"tumor_state_long measurement")
    patients=native.read_table(files["patient_folds"]["path"],"V3.2 patient fold manifest")
    expression=native.read_table(files["lncrna_expression"]["path"],"lncRNA expression")
    cores,parents,_=native.load_v32_core_embeddings(files["core_manifest"]["path"],repo_root=ROOT)
    if {str(k):v for k,v in parents.items()}!=value["core_lineage"]: raise ValueError("State preparation core parents differ")
    design=native.preflight_state_inputs(state,patients,cores,expression,cfg,parents)
    return {"format":"NATIVE_STATE_HEAD_PREFLIGHT_V1","status":"PASS_READY","target_host":"149",
        "training_started":False,"model_forward_started":False,"test_statistics_computed":False,
        "run_id":value["state_run_id"],"core_run_id":value["core_run_id"],"config":value["config"],
        "patient_authority":value["patient_authority"],"input_artifacts":[{"role":role,**files[key]} for key,role in ROLES.items()],**design}


def execute_preflight(command, log):
    if command[2]!=TRAINER or command.count("--prepare-only")!=1:
        raise ValueError("State workflow cannot dispatch fitting")
    with Path(log).open("w",encoding="utf-8") as handle,redirect_stdout(handle),redirect_stderr(handle):
        if native.main(command[3:])!=0: raise RuntimeError("Native State input preparation failed")


def prepare(manifest, output):
    value=plan(manifest,output); output=Path(output).resolve(); output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",value)
    try:
        execute_preflight(value["prepare_command"],output/"prepare.log")
        ready=output/"head_preflight/HEAD_PREFLIGHT.json"
        if read(ready)!=expected_preflight(value): raise ValueError("State original input collector differs from replay")
        if value!=plan(manifest,output,new=False): raise ValueError("State preparation inputs changed")
        write(output/"TRAINING_COMMAND.json",value["training_command"])
        path=output/"PREPARED.json"
        write(path,{"format":PREPARED,"status":"PASS_READY","target_host":"149","training_started":False,
            "plan":record(output/"LAUNCH_PREFLIGHT.json"),"head_preflight":record(ready),
            "training_command":record(output/"TRAINING_COMMAND.json")})
        return path
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)}); raise


def validate_preparation(item, root):
    require_host(); path=verified_path(item,root); binding=read(path)
    if (binding.get("format")!=PREPARED or binding.get("status")!="PASS_READY"
            or binding.get("target_host")!="149" or binding.get("training_started") is not False):
        raise ValueError("Invalid State preparation binding")
    value=read(verified_path(binding["plan"],path.parent))
    if value!=plan(verified_path(value["input"],path.parent),path.parent,new=False):
        raise ValueError("State preparation plan differs from actual sources")
    ready=read(verified_path(binding["head_preflight"],path.parent))
    if ready!=expected_preflight(value): raise ValueError("State input preflight numerical replay differs")
    if read(verified_path(binding["training_command"],path.parent))!=value["training_command"]:
        raise ValueError("State prepared training command differs")
    return path,binding,value,ready


def validate_completed(preparation, manifest, completed_path, completed, sources, root):
    path,binding,value,ready=validate_preparation(preparation,root)
    if Path(completed_path).resolve()!=path.parent/"training/STATE_TRAINING_LINEAGE.json":
        raise ValueError("Completed State result is outside the prepared output")
    for key in ("state_run_id","core_run_id","state_sets"):
        if manifest.get(key)!=value[key]: raise ValueError("Completed State publication settings differ: "+key)
    for key in ("core_manifest","selected_analysis"):
        actual=verified_path(manifest[key],root)
        if record(actual)!=value["files"][key]: raise ValueError("Completed State selected/core source differs")
    if (completed.get("training_format")!=native.STATE_TRAINING_FORMAT or completed.get("training_status")!="SUCCESS"
            or completed.get("training_run_id")!=value["state_run_id"] or completed.get("config")!=value["config"]
            or completed.get("patient_authority")!=value["patient_authority"]):
        raise ValueError("Completed State run, configuration or patient authority differs from preparation")
    declared=completed.get("input_artifacts",[])
    observed={r["role"]:{"role":r["role"],"path":str((Path(completed_path).parent/r["path"]).resolve()),"sha256":r["sha256"]} for r in declared}
    if len(observed)!=len(declared) or observed!={r["role"]:r for r in ready["input_artifacts"]}:
        raise ValueError("Completed State inputs differ from preparation")
    for row in ready["folds"]:
        actual=completed.get("fold_lineage",{}).get(str(row["patient_fold"]),{})
        if any(actual.get(key)!=v for key,v in row.items()):
            raise ValueError("Completed State collected arrays, splits or parent differ from preparation")
    for key in ("observed_state_ids","missing_state_ids"):
        if completed.get(key)!=ready[key]: raise ValueError("Completed State program coverage differs")
    sources.bind(record(path),root)
    for key in ("plan","head_preflight","training_command"): sources.bind(binding[key],path.parent)
    for source in value["sources"]: sources.bind(source,path.parent)
    return value


def collect(preparation, completed, output):
    require_host(); output=Path(output).resolve()
    if output.exists(): raise FileExistsError("Use a new State collection directory")
    path,_,value,_=validate_preparation(preparation,ROOT)
    completed_path=verified_path(completed,ROOT)
    from .current_state_context import INPUT_FORMAT, materialize_current_state_context
    manifest={"format":INPUT_FORMAT,"core_run_id":value["core_run_id"],"state_run_id":value["state_run_id"],
        "core_manifest":value["files"]["core_manifest"],"selected_analysis":value["files"]["selected_analysis"],
        "state_sets":value["state_sets"],"state_training_lineage":record(completed_path),"workflow_preparation":record(path)}
    validate_completed(preparation,manifest,completed_path,read(completed_path),Sources(),ROOT)
    output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,
        "preparation":record(path),"completed":record(completed_path)})
    try:
        source=output/"STATE_CONTEXT_INPUT.json"; write(source,manifest)
        result=materialize_current_state_context(source,output/"context")
        binding=output/"NATIVE_STATE_BINDING.json"
        write(binding,{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
            "preparation":record(path),"completed":record(completed_path),"context_input":record(source),
            "state_context":record(Path(result["binding_path"])),"source_ancestry_complete":False,
            "independent_predictive_performance_verified":False,"production_deployed":False})
        return binding
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)}); raise


def validate(item, root):
    path=verified_path(item,root); binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or any(binding.get(key) is not False for key in ("training_started","source_ancestry_complete",
                "independent_predictive_performance_verified","production_deployed"))):
        raise ValueError("Invalid native State result binding")
    context=verified_path(binding["state_context"],path.parent); data=read(context)
    source=verified_path(binding["context_input"],path.parent); manifest=read(source)
    if (data.get("input_manifest")!=record(source) or manifest.get("workflow_preparation")!=binding["preparation"]
            or manifest.get("state_training_lineage")!=binding["completed"]):
        raise ValueError("Native State result binding dependencies differ")
    from .current_state_context import audit_current_state_context
    audit_current_state_context(context,expected_sha=binding["state_context"]["sha256"])
    return path,binding
