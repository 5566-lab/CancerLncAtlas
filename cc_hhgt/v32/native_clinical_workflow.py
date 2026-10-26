"""Prepare native Clinical inputs and collect completed results on host 149.

Risk fitting and entity statistics have separate declared commands. This
workflow dispatches only their original CLI input-collection modes; completed
results must subsequently match that preparation before current publication.
"""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path, sha256
from .formal_expression_sources import read, write, record
from .input_lineage import artifact_sha256
from .native_auxiliary_fusion import Sources

ROOT=Path(__file__).resolve().parents[2]
INPUT="NATIVE_CLINICAL_WORKFLOW_INPUT_V1"
PREPARED="NATIVE_CLINICAL_WORKFLOW_PREPARED_V1"
FORMAT="NATIVE_CLINICAL_WORKFLOW_RESULTS_V1"
RISK="scripts.run_v32_clinical_training"
ENTITY="scripts.run_v32_clinical_entity_training"
FILES={"workbook","patient_folds","patient_authority_receipt","expression_root","activity_root",
       "state_measurements","state_lineage","core_manifest","selected_analysis","candidates"}
RISK_PARAMETERS=("seed","epochs","patience","n_bins","min_patients","min_events")
ENTITY_PARAMETERS=("block_size","min_train_patients","min_train_events","min_validation_patients",
    "min_validation_events","min_test_patients","min_test_events","min_train_measurements","min_folds_available")
CODE=("cc_hhgt/v32/native_clinical_workflow.py","scripts/run_native_clinical_workflow.py",
    "cc_hhgt/v32/clinical_training.py","cc_hhgt/v32/clinical_entity_training.py",
    "scripts/run_v32_clinical_training.py","scripts/run_v32_clinical_entity_training.py",
    "cc_hhgt/v32/clinical_numeric_replay.py","cc_hhgt/v32/current_clinical_sources.py",
    "cc_hhgt/v32/current_clinical_context.py","cc_hhgt/v32/patient_fold_authority.py",
    "cc_hhgt/v32/integrated_model.py","cc_hhgt/v32/contracts.py",
    "cc_hhgt/v32/current_core_export.py","cc_hhgt/v32/selected_genesets.py")


def arguments(values): return [part for key,value in values.items() for part in ("--"+key.replace("_","-"),str(value))]


def bind_input(item,root,sources):
    path=(Path(root)/item["path"]).resolve()
    if artifact_sha256(path)!=item["sha256"]: raise ValueError("Clinical workflow input SHA256 mismatch")
    for file in (sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]):
        sources.bind(record(file),root)
    if artifact_sha256(path)!=item["sha256"]: raise ValueError("Clinical input changed during binding")
    return {"path":str(path),"sha256":item["sha256"]}


def configuration(parser,names,overrides):
    defaults={a.dest:a.default for a in parser._actions if a.dest in names}
    if not isinstance(overrides,dict) or set(overrides)-set(names): raise ValueError("Unknown Clinical configuration")
    config={**defaults,**overrides}
    if any(type(v) is not int or (k!="seed" and v<1) for k,v in config.items()):
        raise ValueError("Clinical counts and configuration must be positive integers")
    if config.get("min_folds_available",5)>5: raise ValueError("Clinical available folds cannot exceed five")
    return config


def plan(manifest,output,*,new=True):
    from scripts import run_v32_clinical_training as risk, run_v32_clinical_entity_training as entity
    from .current_clinical_sources import selected_inputs, INPUT_FORMAT
    from .patient_fold_authority import validate_patient_fold_authority
    require_host();manifest=Path(manifest).resolve();output=Path(output).resolve();spec=read(manifest)
    required={"format","patient_risk_run_id","entity_run_id","core_run_id",*FILES}
    if (set(spec)-required-{"risk_config","entity_config","fdr_threshold"} or not required<=set(spec)
            or spec["format"]!=INPUT or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",str(spec[k]))
                for k in ("patient_risk_run_id","entity_run_id","core_run_id"))):
        raise ValueError("Invalid native Clinical workflow input")
    if new and output.exists(): raise FileExistsError("Use a new Clinical preparation directory")
    threshold=spec.get("fdr_threshold",.05)
    if type(threshold) not in (int,float) or not np.isfinite(threshold) or not 0<threshold<=1:
        raise ValueError("Clinical FDR threshold must be in (0,1]")
    context=selected_inputs({**spec,"format":INPUT_FORMAT},manifest.parent);sources=context["sources"]
    files={k:bind_input(spec[k],manifest.parent,sources) for k in sorted(FILES)}
    if files["candidates"]!=record(context["candidate_path"]): raise ValueError("Clinical workflow candidates differ from selected analysis")
    for item in files.values():
        source=Path(item["path"])
        if source.is_dir() and (output==source or source in output.parents):
            raise ValueError("Clinical output cannot be inside an input dataset")
    authority=validate_patient_fold_authority(files["patient_folds"]["path"],files["patient_authority_receipt"]["path"])
    anchor=read(files["state_lineage"]["path"])
    if not str(anchor.get("analysis_version","")).startswith("CancerLncAtlas_V3.2") or anchor.get("old_checkpoint_loaded") is not False:
        raise ValueError("Clinical State generation anchor differs")
    risk_cfg=configuration(risk._parser(),RISK_PARAMETERS,spec.get("risk_config",{}))
    entity_cfg=configuration(entity._parser(),ENTITY_PARAMETERS,spec.get("entity_config",{}))
    common={"patient_fold_authority_receipt":files["patient_authority_receipt"]["path"],"patient_authority_mode":"current"}
    risk_args={**common,"repo_root":ROOT,"workbook":files["workbook"]["path"],"fold_manifest":files["patient_folds"]["path"],
        "expression_root":files["expression_root"]["path"],"activity_root":files["activity_root"]["path"],
        "core_root":Path(files["core_manifest"]["path"]).parent,"training_run_id":spec["patient_risk_run_id"],
        "expected_core_run_id":spec["core_run_id"],**risk_cfg}
    if Path(files["core_manifest"]["path"]).name!="CORE_EMBEDDING_MANIFEST.json":
        raise ValueError("Native Clinical risk CLI requires CORE_EMBEDDING_MANIFEST.json")
    entity_args={**common,"tcga_cdr":files["workbook"]["path"],"patient_folds":files["patient_folds"]["path"],
        "candidates":files["candidates"]["path"],"lncrna_expression_root":files["expression_root"]["path"],
        "pathway_activity_root":files["activity_root"]["path"],"state_measurements":files["state_measurements"]["path"],
        "v32_state_lineage":files["state_lineage"]["path"],"training_run_id":spec["entity_run_id"],**entity_cfg}
    commands={};preparation={}
    for kind,module,base,parser in (("risk",RISK,risk_args,risk._parser()),("entity",ENTITY,entity_args,entity._parser())):
        command=[sys.executable,"-m",module,*arguments({**base,"output_root":output/("risk_training" if kind=="risk" else "entity_calculation")})]
        preflight=[sys.executable,"-m",module,*arguments({**base,"output_root":output/(kind+"_preflight")}),"--prepare-only"]
        parser.parse_args(command[3:]);parser.parse_args(preflight[3:])
        commands[kind]={"argv":command,"execution_mode":"CODE_ONLY","automatically_executed":False,
            "requires_separate_explicit_execution":True,"operation":"NEURAL_HEAD_FITTING" if kind=="risk" else "ENTITY_STATISTICAL_CALCULATION"}
        preparation[kind]=preflight
    for name in CODE:sources.bind(record(ROOT/name),ROOT)
    sources.bind(record(manifest),manifest.parent)
    return {"target_host":"149","training_started":False,"input":record(manifest),"files":files,
        **{k:spec[k] for k in ("patient_risk_run_id","entity_run_id","core_run_id")},"fdr_threshold":float(threshold),
        "risk_config":risk_cfg,"entity_config":entity_cfg,"patient_authority":authority,
        "sources":sorted(sources.records.values(),key=lambda r:r["path"]),"prepare_commands":preparation,"execution_commands":commands}


def expected_risk_preflight(value,output):
    from scripts import run_v32_clinical_training as risk
    from .clinical_training import standardize_tcga_cdr_workbook
    from .current_state_context import _assert_same_table
    files=value["files"];paths={k:Path(v["path"]) for k,v in files.items()};config=value["risk_config"]
    endpoints,covariates=standardize_tcga_cdr_workbook(paths["workbook"])
    folds,endpoints,covariates=risk.align_clinical_cohort(pd.read_csv(paths["patient_folds"],sep="\t"),endpoints,covariates)
    core=read(paths["core_manifest"]);parts=[]
    for fold in range(5):
        collected=risk.collect_fold_inputs(fold,folds,covariates,endpoints,paths["expression_root"],paths["activity_root"],
            paths["core_manifest"].parent,core,**{k:config[k] for k in ("n_bins","min_patients","min_events")})
        feature=Path(output)/f"risk_preflight/patient_fold={fold}/patient_features.fold_local.parquet"
        try:_assert_same_table(pd.read_parquet(feature),collected["features"])
        except AssertionError as error:raise ValueError("Clinical risk preflight feature replay differs") from error
        parts.append({"patient_fold":fold,**collected["description"],"feature_artifact":record(feature)})
    contract={"format":"NATIVE_CLINICAL_RISK_INPUTS_V1","patient_authority_mode":"current",
        "files":{k:files[k] for k in ("workbook","patient_folds","patient_authority_receipt","expression_root","activity_root","core_manifest")}}
    return {"format":"NATIVE_CLINICAL_RISK_PREFLIGHT_V1","status":"PASS_READY","target_host":"149",
        "training_started":False,"model_forward_started":False,"training_run_id":value["patient_risk_run_id"],"core_run_id":value["core_run_id"],
        "source_contract":contract,"training_config":{"parameters":config,"runner_sha256":sha256(Path(risk.__file__)),
            "implementation_sha256":sha256(ROOT/"cc_hhgt/v32/clinical_training.py")},"patient_fold_authority":value["patient_authority"],"folds":parts}


def expected_entity_preflight(value):
    from scripts import run_v32_clinical_entity_training as entity
    args=entity._parser().parse_args(value["prepare_commands"]["entity"][3:]);data=entity.load_inputs(args)
    parts=[entity.collect_partition(data,cancer,kind)["description"] for cancer in data["cancers"] for kind in data["subject_types"]]
    return entity.preflight_record(data,value["entity_run_id"],parts)


def execute_preflight(command,log):
    from scripts import run_v32_clinical_training as risk,run_v32_clinical_entity_training as entity
    if (len(command)<4 or command[1]!="-m" or command[2] not in {RISK,ENTITY} or command.count("--prepare-only")!=1):
        raise ValueError("Clinical workflow cannot dispatch fitting or entity analysis")
    main=risk.main if command[2]==RISK else entity.main
    with Path(log).open("w",encoding="utf-8") as handle,redirect_stdout(handle),redirect_stderr(handle):
        if main(command[3:])!=0:raise RuntimeError("Clinical original input collection failed")


def prepare(manifest,output):
    value=plan(manifest,output);output=Path(output).resolve();output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",value)
    try:
        for kind,command in value["prepare_commands"].items():execute_preflight(command,output/(kind+"_prepare.log"))
        risk=output/"risk_preflight/HEAD_PREFLIGHT.json";entity=output/"entity_preflight/INPUT_PREFLIGHT.json"
        if read(risk)!=expected_risk_preflight(value,output) or read(entity)!=expected_entity_preflight(value):
            raise ValueError("Clinical original input collectors differ from reconstruction")
        if value!=plan(manifest,output,new=False):raise ValueError("Clinical sources changed during preparation")
        write(output/"EXECUTION_COMMANDS.json",value["execution_commands"])
        path=output/"PREPARED.json"
        write(path,{"format":PREPARED,"status":"PASS_READY","target_host":"149","training_started":False,
            "plan":record(output/"LAUNCH_PREFLIGHT.json"),"risk_preflight":record(risk),"entity_preflight":record(entity),
            "execution_commands":record(output/"EXECUTION_COMMANDS.json")})
        return path
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise


def validate_preparation(item,root):
    require_host();path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=PREPARED or binding.get("status")!="PASS_READY"
            or binding.get("target_host")!="149" or binding.get("training_started") is not False):
        raise ValueError("Invalid Clinical preparation binding")
    value=read(verified_path(binding["plan"],path.parent))
    if value!=plan(verified_path(value["input"],path.parent),path.parent,new=False):
        raise ValueError("Clinical preparation differs from actual sources")
    risk=read(verified_path(binding["risk_preflight"],path.parent));entity=read(verified_path(binding["entity_preflight"],path.parent))
    if risk!=expected_risk_preflight(value,path.parent) or entity!=expected_entity_preflight(value):
        raise ValueError("Clinical preparation numerical replay differs")
    if read(verified_path(binding["execution_commands"],path.parent))!=value["execution_commands"]:
        raise ValueError("Clinical execution commands differ")
    return path,binding,value,risk,entity


def validate_completed(context):
    """Called by the actual current Clinical reader, including later re-audits."""
    manifest=context["manifest"];sources=context["sources"];root=context["root"]
    path,binding,value,risk_ready,entity_ready=validate_preparation(manifest["workflow_preparation"],root)
    for key in ("patient_risk_run_id","entity_run_id","core_run_id","fdr_threshold"):
        if manifest.get(key)!=value[key]:raise ValueError("Completed Clinical publication settings differ: "+key)
    for key in ("core_manifest","selected_analysis"):
        if record(verified_path(manifest[key],root))!=value["files"][key]:raise ValueError("Completed Clinical selected/core source differs")
    rp,risk=sources.document(manifest["patient_risk_lineage"],root)
    ep,entity=sources.document(manifest["entity_lineage"],root)
    cp,calc=sources.document(manifest["entity_calculation_manifest"],root)
    if (rp!=path.parent/"risk_training/MODULE_LINEAGE.json" or ep!=path.parent/"entity_calculation/ENTITY_CLINICAL_LINEAGE.json"
            or cp!=ep.parent/"CALCULATION_MANIFEST.json" or verified_path(manifest["patient_risk_predictions"],root)!=rp.parent/"clinical_patient_risk.parquet"):
        raise ValueError("Completed Clinical results are outside the prepared output")
    if (risk.get("training_status")!="SUCCESS" or risk.get("training_run_id")!=value["patient_risk_run_id"]
            or risk.get("core_run_id")!=value["core_run_id"] or risk.get("training_config")!=risk_ready["training_config"]
            or risk.get("source_contract")!=risk_ready["source_contract"]):
        raise ValueError("Completed Clinical risk run, inputs or configuration differ")
    tasks=risk.get("task_lineage",[])
    if len(tasks)!=5 or {r["patient_fold"] for r in tasks}!=set(range(5)):
        raise ValueError("Completed Clinical risk folds differ")
    for row in tasks:
        expected=risk_ready["folds"][row["patient_fold"]]
        if row.get("input_description")!={k:v for k,v in expected.items() if k not in {"patient_fold","feature_artifact"}}:
            raise ValueError("Completed Clinical risk input arrays differ")
    if (entity.get("training_status")!="SUCCESS" or entity.get("training_run_id")!=value["entity_run_id"]
            or entity.get("calculation_config")!=entity_ready["calculation_config"]
            or calc.get("calculation_config")!=entity_ready["calculation_config"]
            or entity.get("source_contract")!=entity_ready["source_contract"] or calc.get("source_contract")!=entity_ready["source_contract"]
            or entity.get("patient_fold_authority")!=entity_ready["patient_fold_authority"]):
        raise ValueError("Completed Clinical entity run, inputs, authority or configuration differ")
    parts=calc.get("partitions",[])
    if ([p.get("input_description") for p in parts]!=entity_ready["partitions"]
            or any(p.get("status")!="SUCCESS" for p in parts)):
        raise ValueError("Completed Clinical entity input matrices differ")
    sources.bind(record(path),root)
    for key in ("plan","risk_preflight","entity_preflight","execution_commands"):sources.bind(binding[key],path.parent)
    for row in risk_ready["folds"]:sources.bind(row["feature_artifact"],path.parent)
    for source in value["sources"]:sources.bind(source,path.parent)
    return value


def collect(preparation,risk,entity,output):
    from .current_clinical_sources import selected_inputs, INPUT_FORMAT
    from .current_clinical_context import materialize_current_clinical_context
    require_host();output=Path(output).resolve()
    if output.exists():raise FileExistsError("Use a new Clinical collection directory")
    path,_,value,_,_=validate_preparation(preparation,ROOT)
    rp=verified_path(risk,ROOT);ep=verified_path(entity,ROOT)
    manifest={"format":INPUT_FORMAT,**{k:value[k] for k in ("patient_risk_run_id","entity_run_id","core_run_id","fdr_threshold")},
        "core_manifest":value["files"]["core_manifest"],"selected_analysis":value["files"]["selected_analysis"],
        "patient_risk_lineage":record(rp),"patient_risk_predictions":record(rp.parent/"clinical_patient_risk.parquet"),
        "entity_lineage":record(ep),"entity_calculation_manifest":record(ep.parent/"CALCULATION_MANIFEST.json"),"workflow_preparation":record(path)}
    validate_completed(selected_inputs(manifest,ROOT))
    output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,
        "preparation":record(path),"risk_completed":record(rp),"entity_completed":record(ep)})
    try:
        source=output/"CLINICAL_CONTEXT_INPUT.json";write(source,manifest)
        result=materialize_current_clinical_context(source,output/"context");binding=output/"NATIVE_CLINICAL_BINDING.json"
        write(binding,{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
            "preparation":record(path),"risk_completed":record(rp),"entity_completed":record(ep),"context_input":record(source),
            "clinical_context":record(Path(result["binding_path"])),"source_ancestry_complete":False,
            "independent_predictive_performance_verified":False,"production_deployed":False})
        return binding
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise


def validate(item,root):
    path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or any(binding.get(k) is not False for k in ("training_started","source_ancestry_complete","independent_predictive_performance_verified","production_deployed"))):
        raise ValueError("Invalid native Clinical result binding")
    source=verified_path(binding["context_input"],path.parent);manifest=read(source)
    context=verified_path(binding["clinical_context"],path.parent);data=read(context)
    if (data.get("input_manifest")!=record(source) or manifest.get("workflow_preparation")!=binding["preparation"]
            or manifest.get("patient_risk_lineage")!=binding["risk_completed"] or manifest.get("entity_lineage")!=binding["entity_completed"]):
        raise ValueError("Clinical result binding dependencies differ")
    from .current_clinical_context import audit_current_clinical_context
    audit_current_clinical_context(context,expected_sha=binding["clinical_context"]["sha256"])
    return path,binding
