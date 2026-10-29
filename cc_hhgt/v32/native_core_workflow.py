"""Typed authority -> formal G012 preparation -> binding comparison on 149.

Formal preparation includes L1 fitting and requires a separate explicit CLI
switch. This workflow never dispatches neural training: all thirty generated
training commands retain the existing CODE_ONLY guard.
"""
from contextlib import redirect_stdout,redirect_stderr
from copy import deepcopy
from pathlib import Path
import re
import sys

import yaml

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read,write,record
from .native_graph_input import NATIVE_MODE

ROOT=Path(__file__).resolve().parents[2]
INPUT="NATIVE_CORE_WORKFLOW_INPUT_V1"
FORMAT="NATIVE_CORE_WORKFLOW_RELEASE_V1"
TYPED="scripts.prepare_v32_typed_graph_authority"
FORMAL="scripts.prepare_v32_formal"
COMPARISON="scripts.prepare_v32_current_binding_comparison"
CODE=["cc_hhgt/v32/native_core_workflow.py","cc_hhgt/v32/prepared_construction.py","cc_hhgt/v32/baselines.py","scripts/run_native_core_workflow.py",
      "cc_hhgt/v32/native_graph_input.py","cc_hhgt/v32/formal_graph_authority.py",
      "cc_hhgt/v32/omics_node_inputs.py","cc_hhgt/v32/omics_node_encoding.py",
      "cc_hhgt/v32/gene_patient_omics.py",
      "cc_hhgt/v32/local_cnv_core_preparation.py","cc_hhgt/v32/training_guard.py",
      "cc_hhgt/v32/training.py","cc_hhgt/v32/current_binding_graph.py",
      "cc_hhgt/v32/current_binding_overlay.py","cc_hhgt/v32/orchestration.py",
      "cc_hhgt/v32/patient_fold_authority.py","cc_hhgt/v32/formal_graph.py",
      "cc_hhgt/v32/fold_local_scope.py","cc_hhgt/v32/fold_local_graph.py",
      "cc_hhgt/v32/native_cnv_workflow.py",
      *[m.replace(".","/")+".py" for m in (TYPED,FORMAL,COMPARISON)]]


def preflight(manifest,output):
    require_host();manifest=Path(manifest).resolve(strict=True);output=Path(output).resolve();spec=read(manifest)
    required={"format","release","typed_graph_manifest","candidates","fold_local_scope","local_cnv_success",
              "validation_success","authorization_hashes","comparison_config","expansion_sources"}
    if (not required<=set(spec) or set(spec)-required-{"native_cnv_workflow"} or spec.get("format")!=INPUT
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}",spec.get("release",""))):
        raise ValueError("Invalid native core workflow input")
    if output.exists():raise FileExistsError("Use a new native core workflow output")
    files={k:record(verified_path(spec[k],manifest.parent)) for k in required-{"format","release","expansion_sources"}}
    typed=read(files["typed_graph_manifest"]["path"]);typed_root=Path(files["typed_graph_manifest"]["path"]).parent
    from scripts.prepare_v32_typed_graph_authority import INPUT as TYPED_INPUT
    if typed.get("format")!=TYPED_INPUT:raise ValueError("Missing typed graph manifest")
    typed_files={k:record(verified_path(typed[k],typed_root)) for k in ("source_receipt","patient_map","patient_receipt","binding_declaration")}
    from .patient_fold_authority import validate_frozen_v32_patient_fold_binding
    patient=validate_frozen_v32_patient_fold_binding(typed_files["patient_map"]["path"],typed_files["patient_receipt"]["path"])
    from .formal_graph_authority import load_formal_graph_input_authority,STATIC_ARTIFACT_IDS,RELATION_SCHEMA_GENERIC_GLOBAL
    graph=read(typed_files["source_receipt"]["path"])
    load_formal_graph_input_authority(receipt_path=typed_files["source_receipt"]["path"],receipt_sha256=typed_files["source_receipt"]["sha256"],
        patient_authority_audit=patient,static_inputs={k:(graph["artifacts"][k]["path"],graph["artifacts"][k]["sha256"]) for k in STATIC_ARTIFACT_IDS},
        fold_expression_pattern=typed["fold_expression_pattern"],fold_coexpression_pattern=typed["fold_coexpression_pattern"],
        relation_schema_generation=graph.get("relation_schema_generation",RELATION_SCHEMA_GENERIC_GLOBAL))
    from .local_cnv_core_preparation import validate_local_cnv_audit,validate_validation_a1_authority
    local=validate_local_cnv_audit(Path(files["local_cnv_success"]["path"]).parent,expected_success_sha256=files["local_cnv_success"]["sha256"])
    valid=validate_validation_a1_authority(Path(files["validation_success"]["path"]).parent,expected_success_sha256=files["validation_success"]["sha256"])
    from .fold_local_scope import release
    _,scope=release(files["fold_local_scope"],manifest.parent,candidates=files["candidates"]["path"],patient_map=typed_files["patient_map"]["path"])
    cohort=local.get("cohort")
    if (not cohort or valid.get("cohort")!=cohort or valid.get("local_cnv_audit_success_sha256")!=local["success_sha256"]
            or cohort["candidates"]!=files["candidates"] or cohort["patient_map"]!=typed_files["patient_map"]
            or cohort["candidate_definition"]!=scope["candidates"]
            or graph.get("local_cnv_adjustment",{}).get("success_sha256")!=local["success_sha256"]):
        raise ValueError("Core workflow graph, scope, train/validation and patient authorities differ")
    native=None
    if "native_cnv_workflow" in spec:
        from .native_cnv_workflow import validate
        path,cnv=validate(spec["native_cnv_workflow"],manifest.parent)
        if (not cnv["corrected_inputs_available"] or cnv["outputs"]["corrected_graph"]!=typed_files["source_receipt"]
                or cnv["outputs"]["association_success"]!=files["local_cnv_success"]
                or cnv["outputs"]["validation_success"]!=files["validation_success"]):
            raise ValueError("Core workflow differs from its completed CNV workflow")
        native=record(path)
    config=yaml.safe_load(Path(files["comparison_config"]["path"]).read_text())
    if config.get("crossfit",{}).get("seed",20260726)!=20260726:
        raise ValueError("Core preparation and comparison require the same frozen seed 20260726")
    contract=config.get("task_contract",{})
    if contract.get("rbp_evidence_mode")!=NATIVE_MODE or contract.get("graph_variant")!="G2":
        raise ValueError("Core workflow requires an unexpanded native G2 comparison config")
    if contract.get("binding_expansion_arm") or any(config.get("training_io",{}).get(k) for k in ("c_graph_overlay_pattern","binding_expansion_overlay_pattern")):
        raise ValueError("Core workflow cannot reuse a historical/expanded graph config")
    from scripts.prepare_v32_current_binding_comparison import load_sources
    # Resolve source paths before moving the eventual comparison manifest.
    expansion=[];expansion_pins=[]
    for entry in spec["expansion_sources"]:
        entry=deepcopy(entry)
        for key in ("edges","lineage","audit","binding"):
            if key in entry:entry[key]=record(verified_path(entry[key],manifest.parent));expansion_pins.append(entry[key])
        expansion.append(entry)
    load_sources(expansion,manifest.parent)
    statics={k:deepcopy(graph["artifacts"][k]) for k in STATIC_ARTIFACT_IDS}
    statics["lnc_protein_binding"]=typed_files["binding_declaration"]
    args=["--hashes",files["authorization_hashes"]["path"],"--patient-folds",typed_files["patient_map"]["path"],
        "--patient-fold-authority-receipt",typed_files["patient_receipt"]["path"],"--graph-authority-receipt",str(output/"typed/GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json"),
        "--graph-authority-receipt-sha256",{"sha256_of":"typed_receipt"},"--graph-relation-schema-generation","TYPED_ASSAY_CLASS_V1",
        "--graph-fold-expression-pattern",typed["fold_expression_pattern"],"--graph-fold-coexpression-pattern",typed["fold_coexpression_pattern"],
        "--formal-candidates",files["candidates"]["path"],"--formal-candidates-sha256",files["candidates"]["sha256"],
        "--fold-local-scope",files["fold_local_scope"]["path"],"--fold-local-scope-sha256",files["fold_local_scope"]["sha256"],
        "--association-mode","local-cnv-adjusted","--local-cnv-audit-root",local["root"],"--local-cnv-audit-success-sha256",local["success_sha256"],
        "--local-cnv-validation-root",valid["root"],"--local-cnv-validation-success-sha256",valid["success_sha256"],
        "--budget-per-cancer",str(cohort["requested_budget_per_cancer"]),"--seed","20260726","--start-fold","0","--fold-count","5","--output",str(output/"prepared")]
    for key,item in statics.items():args.extend(["--graph-"+key.replace("_","-"),item["path"],"--graph-"+key.replace("_","-")+"-sha256",item["sha256"]])
    commands=[{"stage":"typed_graph","argv":[sys.executable,"-m",TYPED,"--manifest",files["typed_graph_manifest"]["path"],"--output",str(output/"typed")]},
        {"stage":"formal_preparation","requires_l1_fitting":True,"argv":[sys.executable,"-m",FORMAL,*args]},
        {"stage":"binding_comparison","argv":[sys.executable,"-m",COMPARISON,"--manifest",str(output/"resolved_comparison.json"),"--output",str(output/"comparison"),"--release",spec["release"]]}]
    return {"target_host":"149","neural_training_started":False,"preparation_includes_l1_baseline_fitting":True,
        "input":record(manifest),"files":files,"typed_files":typed_files,"cohort":cohort,"native_cnv_workflow":native,
        "release":spec["release"],"expansion_sources":expansion,"commands":commands,
        "sources":[record(manifest),*files.values(),*typed_files.values(),*expansion_pins,*([native] if native else [])],
        "implementations":[record(ROOT/name) for name in CODE]}


def resolve(command,output):
    values=[]
    for value in command["argv"]:
        if isinstance(value,dict):
            if value!={"sha256_of":"typed_receipt"}:raise ValueError("Unknown core command dependency")
            value=record(output/"typed/GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json")["sha256"]
        values.append(value)
    return values


def training_commands(output,config_record,release):
    from .orchestration import build_task_manifest,write_task_manifest
    config=yaml.safe_load(Path(config_record["path"]).read_text());commands=[]
    for variant in ("G0","G1","G2"):
        folder=output/"training"/variant;folder.mkdir(parents=True)
        cfg=deepcopy(config);cfg["task_contract"]["graph_variant"]=variant
        if variant!="G2":cfg["task_contract"].pop("rbp_evidence_mode",None)
        cfg["execution_control"]={"execution_mode":"CODE_ONLY","training_authorized":False,"paid_enabled":False,"max_paid_hours":0.,"max_cost_cny":0.}
        cfg.setdefault("runtime_profile",{}).update(device="cpu",target_host="149",mixed_precision=False)
        cfg.setdefault("training_io",{}).update(prepared_fold_pattern=str(output/"prepared"/variant/"PATIENT_FOLD_{fold}.pt"),output_root=str(folder/"outputs"))
        config_path=folder/"config.yaml";config_path.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
        run_id=f"v32-g012-{variant.lower()}-native-{release}"
        tasks=build_task_manifest(run_id=run_id,seed=20260726,owner="server149",hardware_class="SERVER149_CPU")
        write_task_manifest(folder/"TASK_MANIFEST.tsv",tasks)
        for task in tasks:
            commands.append({"graph_variant":variant,"fold":task["patient_fold"],"status":"BLOCKED_CODE_ONLY","argv":[sys.executable,
                "scripts/v32_pipeline.py","run-shard","--repo-root",str(ROOT),"--config",str(config_path),
                "--input-manifest",str(output/"prepared"/variant/"INPUT_MANIFEST.json"),"--task-manifest",str(folder/"TASK_MANIFEST.tsv"),
                "--approval",str(folder/"TRAINING_APPROVAL.json"),"--run-id",run_id,"--task-id",task["task_id"],
                "--endpoint-id","server149","--hardware-class","SERVER149_CPU"]})
    return commands


def accept_inputs(output,plan):
    """Run the actual trainer's input loaders/validators without model fitting."""
    import torch
    from .training import load_prepared_artifact_from_authorized_handle,_validate_prepared,validate_formal_graph_variant_binding
    from .native_graph_input import load_training_graph_inputs
    from .current_binding_graph import ARMS
    from .orchestration import read_task_manifest,find_task
    commands=read(output/"TRAINING_COMMANDS.json")["commands"]
    if len(commands)!=30 or any(c["status"]!="BLOCKED_CODE_ONLY" for c in commands):raise ValueError("Expected thirty blocked training commands")
    expected={(group,fold) for group in ("G0","G1","G2",*ARMS) for fold in range(5)}
    observed=[(c.get("arm",c.get("graph_variant")),c.get("fold")) for c in commands]
    if len(set(observed))!=30 or set(observed)!=expected:
        raise ValueError("Core commands must cover all six groups and five folds exactly once")
    records=[]
    for command in commands:
        argv=command["argv"]
        if "--allow-training" in argv:raise ValueError("Core workflow cannot dispatch training")
        value=lambda name:argv[argv.index(name)+1]
        arg=lambda name:Path(value(name))
        config_path=arg("--config");manifest=arg("--input-manifest");cfg=yaml.safe_load(config_path.read_text())
        if cfg["execution_control"]["execution_mode"]!="CODE_ONLY" or cfg["execution_control"]["training_authorized"] is not False:
            raise ValueError("Generated training commands are not blocked")
        if cfg["runtime_profile"].get("target_host")!="149" or cfg["runtime_profile"].get("device")!="cpu":raise ValueError("Training command host differs")
        fold=command["fold"]
        task=find_task(read_task_manifest(arg("--task-manifest")),value("--task-id"))
        if (int(task["patient_fold"])!=fold or int(task["seed"])!=20260726 or task["run_id"]!=value("--run-id")
                or task["status"]!="BLOCKED" or task["owner"]!="server149" or task["hardware_class"]!="SERVER149_CPU"
                or value("--endpoint-id")!="server149" or value("--hardware-class")!="SERVER149_CPU"):
            raise ValueError("Core training task differs from its fold, seed or execution control")
        contract=cfg["task_contract"]
        if (contract.get("graph_variant")!=command.get("graph_variant","G2")
                or contract.get("binding_expansion_arm")!=command.get("arm")):
            raise ValueError("Core command config differs from its graph group")
        rows=[r for r in read(manifest)["fold_inputs"] if r["fold"]==fold]
        if len(rows)!=1:raise ValueError("Core input manifest has no unique requested fold")
        row=rows[0];prepared=Path(row["path"])
        payload,digest=load_prepared_artifact_from_authorized_handle(torch,manifest,fold=fold,prepared_path=prepared)
        if payload["formal_graph_authority"]["receipt"]!=record(output/"typed/GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json"):
            raise ValueError("Core input belongs to another typed graph generation")
        payload,_=load_training_graph_inputs(torch,payload=payload,config=cfg,input_manifest_path=manifest,fold=fold,root=output,
            prepared_sha256=digest,task_seed=20260726)
        _validate_prepared(payload,fold=fold,artifact_hashes=payload["artifact_hashes"])
        validate_formal_graph_variant_binding(run_id=argv[argv.index("--run-id")+1],config=cfg,payload=payload,prepared_path=prepared)
        records.extend([record(config_path),record(manifest),record(arg("--task-manifest")),record(prepared)])
        if row.get("binding_expansion"):
            records.extend([record(row["binding_expansion"]["path"]),record(row["binding_expansion"]["source_admission"]["path"])])
        if row.get("node_omics"):
            source=verified_path(row["node_omics"],manifest.parent)
            node_manifest=read(source)
            records.append(record(source))
            for item in [node_manifest["source_config"],node_manifest["private_contributions"],
                         {"path":node_manifest["table_path"],"sha256":node_manifest["table_sha256"]},
                         *node_manifest["source_artifacts"].values()]:
                records.append(record(verified_path(item,source.parent)))
    return list({r["path"]:r for r in records}.values())


def run(manifest,output,*,allow_baseline_fitting=False):
    require_host()
    if allow_baseline_fitting is not True:raise ValueError("Core preparation includes L1 fitting; an explicit --allow-baseline-fitting is required")
    plan=preflight(manifest,output);output=Path(output).resolve();output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",{**plan,"l1_baseline_fitting_authorized":True})
    state={"target_host":"149","neural_training_started":False,"status":"RUNNING","stages":[]};write(output/"STAGE_STATUS.json",state)
    stage="initialization"
    try:
        for command in plan["commands"]:
            stage=command["stage"]
            if stage=="binding_comparison":
                base=output/"prepared/G2/INPUT_MANIFEST.json"
                write(output/"resolved_comparison.json",{"format":"CURRENT_C_BINDING_COMPARISON_INPUT_V1","config":plan["files"]["comparison_config"],
                    "sources":plan["expansion_sources"],"fold_inputs":[{"fold":r["fold"],"graph_input_mode":NATIVE_MODE,
                        "prepared":{"path":r["path"],"sha256":r["sha256"]},"input_manifest":record(base)} for r in read(base)["fold_inputs"]]})
            argv=resolve(command,output);entry={"stage":stage,"argv":argv,"status":"RUNNING"};state["stages"].append(entry);write(output/"STAGE_STATUS.json",state)
            import importlib
            main=importlib.import_module(argv[2]).main
            log=output/(stage+".log")
            with log.open("w",encoding="utf-8") as handle,redirect_stdout(handle),redirect_stderr(handle):result=main(argv[3:])
            entry.update(exit_code=result,log=record(log))
            if result!=0:raise ValueError("Native core stage failed: "+stage)
            entry["status"]="PASS";write(output/"STAGE_STATUS.json",state)
        stage="training_command_acceptance"
        commands=training_commands(output,plan["files"]["comparison_config"],plan["release"])
        commands+=read(output/"comparison/COMPARISON.json")["commands"]
        write(output/"TRAINING_COMMANDS.json",{"target_host":"149","neural_training_started":False,"commands":commands})
        artifacts=accept_inputs(output,plan)
        for source in [*plan["sources"],*plan["implementations"]]:verified_path(source,output)
        if plan["native_cnv_workflow"]:
            from .native_cnv_workflow import validate
            validate(plan["native_cnv_workflow"],output)
        state["status"]="PASS";write(output/"STAGE_STATUS.json",state)
        result={"format":FORMAT,"status":"PASS","target_host":"149","neural_training_started":False,
            "preparation_includes_l1_baseline_fitting":True,"l1_baseline_fitting_authorized":True,
            "preflight":record(output/"LAUNCH_PREFLIGHT.json"),"stage_status":record(output/"STAGE_STATUS.json"),
            "typed_graph":record(output/"typed/GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json"),
            "formal_summary":record(output/"prepared/PREP_INPUT_SUMMARY.json"),"comparison":record(output/"comparison/COMPARISON.json"),
            "training_commands":record(output/"TRAINING_COMMANDS.json"),"accepted_artifacts":artifacts,"neural_checkpoints_created":False}
        path=output/"NATIVE_CORE_BINDING.json";write(path,result)
        write(output/"SUCCESS.json",{"status":"PASS","target_host":"149","neural_training_started":False,"binding":record(path)})
        return path
    except Exception as error:
        if state["stages"] and state["stages"][-1]["status"]=="RUNNING":state["stages"][-1]["status"]="FAILED"
        state.update(status="FAILED",failed_stage=stage,error=str(error),error_type=type(error).__name__);write(output/"STAGE_STATUS.json",state);raise


def validate(item,root):
    require_host();path=verified_path(item,root);binding=read(path);output=path.parent
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or binding.get("neural_training_started") is not False or binding.get("neural_checkpoints_created") is not False
            or read(output/"SUCCESS.json")!={"status":"PASS","target_host":"149","neural_training_started":False,"binding":record(path)}):
        raise ValueError("Incomplete native core workflow")
    for k in ("preflight","stage_status","typed_graph","formal_summary","comparison","training_commands"):verified_path(binding[k],output)
    plan=read(binding["preflight"]["path"]);state=read(binding["stage_status"]["path"])
    if state["status"]!="PASS" or len(state["stages"])!=3:raise ValueError("Core stage closure differs")
    for s,c in zip(state["stages"],plan["commands"]):
        if s["status"]!="PASS" or s["exit_code"]!=0 or s["argv"]!=resolve(c,output):raise ValueError("Core command execution differs")
        verified_path(s["log"],output)
    for source in [*plan["sources"],*plan["implementations"],*binding["accepted_artifacts"]]:verified_path(source,output)
    if plan["native_cnv_workflow"]:
        from .native_cnv_workflow import validate as validate_cnv
        validate_cnv(plan["native_cnv_workflow"],output)
    if accept_inputs(output,plan)!=binding["accepted_artifacts"]:raise ValueError("Core accepted input artifacts differ")
    return path,binding
