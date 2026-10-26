"""Bind current native producers to fusion development and selected publication.

Preparation executes the original input collector. Fitting is recorded as a
separate command and never dispatched. Collection replays immutable completion.
"""
from contextlib import redirect_stdout,redirect_stderr
from dataclasses import asdict
from pathlib import Path
import re
import sys

import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path,sha256
from .formal_expression_sources import read,write,record
from . import multimodal_fusion as fusion
from .fusion_numeric_replay import same_frame,replay_release

ROOT=Path(__file__).resolve().parents[2]
INPUT="NATIVE_FUSION_WORKFLOW_INPUT_V1"
PREPARED="NATIVE_FUSION_WORKFLOW_PREPARED_V1"
FORMAT="NATIVE_FUSION_WORKFLOW_RESULTS_V1"
MODULE="scripts.run_v32_current_multimodal_fusion"
RESULT_REFS={"selected_model","core_run_id","core_prediction","primary_fusion"}
CODE=("cc_hhgt/v32/native_fusion_workflow.py","scripts/run_native_fusion_workflow.py",
    "scripts/run_v32_current_multimodal_fusion.py","cc_hhgt/v32/fusion_numeric_replay.py",
    "cc_hhgt/v32/multimodal_fusion.py","cc_hhgt/v32/current_primary_fusion.py",
    "cc_hhgt/v32/native_auxiliary_fusion.py","cc_hhgt/v32/single_cell_signed_fusion.py",
    "cc_hhgt/v32/selected_score.py","cc_hhgt/v32/current_analysis_workflow.py")


def validate_result_refs(value):
    if isinstance(value,dict):
        if "result_ref" in value and value["result_ref"] not in RESULT_REFS:
            raise ValueError("Fusion publication result reference is unavailable; pin the existing artifact explicitly")
        for item in value.values():validate_result_refs(item)
    elif isinstance(value,list):
        for item in value:validate_result_refs(item)


def plan(manifest,output,*,new=True):
    from .current_primary_fusion import load_primary_fusion_inputs
    from .native_auxiliary_fusion import FORMAT as AUX,load_native_auxiliary
    from .single_cell_signed_fusion import FORMAT as SC,load_signed_sc_fusion
    from .current_analysis_web import COMPONENTS,STATUSES
    from .native_results_workflow import pinned_template
    require_host();manifest=Path(manifest).resolve();output=Path(output).resolve();spec=read(manifest)
    required={"format","fusion_run_id","primary_fusion","experts","selected_score_column","selection_rationale",
              "geneset","subtype","publication_components"}
    if (not required<=set(spec) or set(spec)-required-{"training_config","rule_evidence_binding"} or spec["format"]!=INPUT
            or not isinstance(spec["fusion_run_id"],str) or not re.fullmatch(r"V32-FUSION-[A-Z0-9_-]{8,85}",spec["fusion_run_id"])
            or spec["selected_score_column"] not in {"discovery_adjusted_probability","fused_confidence_probability"}
            or not isinstance(spec["selection_rationale"],str) or not spec["selection_rationale"].strip()):
        raise ValueError("Invalid native fusion workflow input")
    if new and output.exists():raise FileExistsError("Use a new fusion preparation directory")
    config=fusion.ResidualFusionConfig(**spec.get("training_config",{}));config.validate()
    from .selected_genesets import SelectedGeneSetConfig
    from .subtypes import RankedSubtypeConfig
    SelectedGeneSetConfig(**spec["geneset"]).validate()
    RankedSubtypeConfig(**spec["subtype"]).validate()
    sources=[]
    primary_path=verified_path(spec["primary_fusion"],manifest.parent);primary=read(primary_path)
    primary=load_primary_fusion_inputs(primary_path,expected_sha=sha256(primary_path),expected_core_run_id=primary["run_id"])
    sources.append(record(primary_path));experts=[];seen=set()
    if not isinstance(spec["experts"],list) or not spec["experts"]:raise ValueError("Fusion needs explicit native expert bindings")
    for item in spec["experts"]:
        path=verified_path(item,manifest.parent);value=read(path);sources.append(record(path))
        if value.get("format")==AUX:
            _,value=load_native_auxiliary(path,expected_sha=sha256(path),expected_core_run_id=primary["run_id"])
            prediction=value["outputs"]["fusion"]
        elif value.get("format")==SC:
            value,_,_,dependencies=load_signed_sc_fusion(path,expected_sha=sha256(path),replay=False)
            sources.extend(dependencies);prediction=value["outputs"]["single_cell_signed_fusion"]
        else:raise ValueError("Fusion workflow requires native auxiliary or signed SC bindings")
        for definition in value["expert_specs"]:
            name=definition["expert_id"]
            if name in seen:raise ValueError("Duplicate native fusion expert")
            seen.add(name)
            experts.append({"spec":definition,"run_id":value["run_id"],"estimand":value["estimand"],
                "binding_audit":record(path),"prediction":prediction})
    experts.sort(key=lambda r:r["spec"]["expert_id"])
    normalized={"format":"CURRENT_MULTIMODAL_FUSION_MANIFEST_V1","core_run_id":primary["run_id"],
        "fusion_run_id":spec["fusion_run_id"],"primary_prediction":primary["prediction"],"primary_binding_audit":record(primary_path),
        "primary_folds":primary["folds"],"experts":experts,"training_config":asdict(config)}
    if "rule_evidence_binding" in spec:
        normalized["rule_evidence_binding"]=record(verified_path(spec["rule_evidence_binding"],manifest.parent))
        sources.append(normalized["rule_evidence_binding"])
    components=spec["publication_components"];templates={}
    if set(components)!=set(COMPONENTS):raise ValueError("Fusion publication requires every component status")
    components={k:dict(v) for k,v in components.items()}
    for name,item in components.items():
        status=item.get("status")
        if status in {"BUILD","REUSE"}:
            key="input_manifest" if status=="BUILD" else "binding"
            if set(item)!={"status",key}:raise ValueError("Ambiguous fusion publication input")
            path=verified_path(item[key],manifest.parent);sources.append(record(path));item[key]=record(path)
            if status=="BUILD":
                templates[name]=pinned_template(read(path),path.parent,sources)
                validate_result_refs(templates[name])
        elif status not in STATUSES or set(item)!={"status","reason"} or not isinstance(item["reason"],str) or not item["reason"].strip():
            raise ValueError("Missing fusion publication component reason")
    for name in CODE:sources.append(record(ROOT/name))
    sources.append(record(manifest))
    base=[sys.executable,"-m",MODULE,"--manifest",str(output/"FUSION_INPUT.json"),"--expected-core-run-id",primary["run_id"]]
    return {"target_host":"149","training_started":False,"input":record(manifest),"fusion_input":normalized,
        "reference_exact_model":{"model_id":"native-primary","producer_kind":"current_core_release",
            "producer_binding":primary["primary_source"],"run_id":primary["run_id"],"score_column":"association_membership_probability"},
        **{k:spec[k] for k in ("selected_score_column","selection_rationale","geneset","subtype")},
        "publication_components":components,"publication_templates":templates,
        "sources":sorted({r["path"]:r for r in sources}.values(),key=lambda r:r["path"]),
        "prepare_command":[*base,"--output",str(output/"input_preflight")],
        "execution_command":{"argv":[*base,"--output",str(output/"training"),"--fit"],
            "operation":"PAIR_BLOCKED_FUSION_FITTING","automatically_executed":False,"requires_separate_explicit_execution":True}}


def execute_preflight(argv,log):
    from scripts import run_v32_current_multimodal_fusion as runner
    if len(argv)<4 or argv[1:3]!=["-m",MODULE] or "--fit" in argv:raise ValueError("Fusion workflow cannot dispatch fitting")
    with Path(log).open("w",encoding="utf-8") as handle,redirect_stdout(handle),redirect_stderr(handle):
        if runner.main(argv[3:])!=0:raise RuntimeError("Original fusion input collection failed")


def input_state(value,output):
    from scripts import run_v32_current_multimodal_fusion as runner
    output=Path(output);manifest=output/"FUSION_INPUT.json"
    if read(manifest)!=value["fusion_input"]:raise ValueError("Fusion normalized sources differ")
    frame,audit,_,_=runner.collect_inputs(manifest,expected_core_run_id=value["fusion_input"]["core_run_id"],replay_expert_models=False)
    if read(output/"input_preflight/INPUT_AUDIT.json")!=audit:raise ValueError("Fusion prepared input audit differs")
    same_frame(pd.read_parquet(output/"input_preflight/PRIVATE_FUSION_INPUT.parquet"),frame,"prepared input matrix")
    ready=read(output/"input_preflight/PREPARED.json")
    if ready!={"status":"PREPARED_NOT_TRAINED","target_host":"149","training_started":False,
            "core_run_id":audit["core_run_id"],"rows":len(frame),"input_sha256":sha256(output/"input_preflight/PRIVATE_FUSION_INPUT.parquet"),
            "eligible_for_current_fitting":True,"primary_label_authority":"NATIVE_VALIDATION_REPLAY"} or not audit["eligible_for_current_fitting"]:
        raise ValueError("Fusion input preparation is not ready for all five folds")
    return frame,audit


def prepare(manifest,output):
    value=plan(manifest,output);output=Path(output).resolve();output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",value)
    try:
        write(output/"FUSION_INPUT.json",value["fusion_input"])
        execute_preflight(value["prepare_command"],output/"prepare.log");input_state(value,output)
        if value!=plan(manifest,output,new=False):raise ValueError("Fusion sources changed during preparation")
        write(output/"EXECUTION_COMMAND.json",value["execution_command"])
        binding=output/"PREPARED.json"
        write(binding,{"format":PREPARED,"status":"PASS_READY","target_host":"149","training_started":False,
            "plan":record(output/"LAUNCH_PREFLIGHT.json"),"fusion_input":record(output/"FUSION_INPUT.json"),
            "input_audit":record(output/"input_preflight/INPUT_AUDIT.json"),
            "input_frame":record(output/"input_preflight/PRIVATE_FUSION_INPUT.parquet"),
            "ready":record(output/"input_preflight/PREPARED.json"),"execution_command":record(output/"EXECUTION_COMMAND.json")});return binding
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise


def validate(item,root):
    require_host();path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or any(binding.get(k) is not False for k in ("training_started","source_ancestry_complete",
                "independent_predictive_performance_verified","production_deployed"))):
        raise ValueError("Incomplete native fusion collection")
    prepared,_,value,frame,audit=validate_preparation(binding["preparation"],path.parent)
    completed=verified_path(binding["completed"],path.parent)
    if completed!=prepared.parent/"training/release/MULTIMODAL_FUSION_BINDING.json":
        raise ValueError("Fusion collected completion differs from its prepared location")
    trained=read(completed);parent=read(completed.parent.parent/"SUCCESS.json")
    if (parent.get("binding_sha256")!=binding["completed"]["sha256"] or parent.get("target_host")!="149"
            or read(verified_path(trained["artifacts"]["fusion_input_audit"],completed.parent))!=audit):
        raise ValueError("Fusion completed inputs differ from preparation")
    same_frame(pd.read_parquet(verified_path(trained["artifacts"]["fusion_input_private"],completed.parent)),frame,"collected input matrix")
    numeric=verified_path(binding["numeric_replay"],path.parent)
    if read(numeric)!=replay_release(completed,expected_sha=binding["completed"]["sha256"]):
        raise ValueError("Fusion saved numerical replay differs")
    expected={"format":"CURRENT_SELECTED_ANALYSIS_INPUT_V1","selected_model":{
        "model_id":"native-fusion","producer_kind":"fusion_release","producer_binding":binding["completed"],
        "run_id":value["fusion_input"]["fusion_run_id"],"score_column":value["selected_score_column"]},
        "reference_exact_model":value["reference_exact_model"],"selection":{"kind":"EXPLICIT_SOURCE_REGISTRATION",
            "rationale":value["selection_rationale"]},"geneset":value["geneset"],"subtype":value["subtype"]}
    if "rule_evidence_binding" in value["fusion_input"]:expected["rule_evidence_binding"]=value["fusion_input"]["rule_evidence_binding"]
    if read(verified_path(binding["selected_input"],path.parent))!=expected:
        raise ValueError("Fusion collected selected input differs from preparation")
    from .current_analysis_workflow import validate_current_analysis_workflow
    _,publication=validate_current_analysis_workflow(binding["publication"],path.parent)
    if (publication["input_manifest"]!=binding["publication_input"] or publication["selected_analysis"]!=binding["selected_analysis"]
            or publication["web_binding"]!=binding["web_binding"]):raise ValueError("Fusion publication handoff differs")
    source=read(verified_path(binding["publication_input"],path.parent))
    if source["selected"]!={"status":"BUILD","input_manifest":binding["selected_input"]}:
        raise ValueError("Fusion publication did not build its declared selection")
    from .native_results_workflow import result_references
    refs={"selected_model":expected["selected_model"],"core_run_id":value["fusion_input"]["core_run_id"],
        "core_prediction":value["reference_exact_model"]["producer_binding"],"primary_fusion":value["fusion_input"]["primary_binding_audit"]}
    components={k:dict(v) for k,v in value["publication_components"].items()}
    for name,template in value["publication_templates"].items():
        manifest=path.parent/(name+"_input.json")
        if read(manifest)!=result_references(template,refs):raise ValueError("Fusion collected context template differs")
        components[name]={"status":"BUILD","input_manifest":record(manifest)}
    if source["components"]!=components:raise ValueError("Fusion collected publication components differ")
    return path,binding


def validate_preparation(item,root):
    require_host();path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=PREPARED or binding.get("status")!="PASS_READY" or binding.get("target_host")!="149"
            or binding.get("training_started") is not False):raise ValueError("Invalid fusion preparation binding")
    for key in ("plan","fusion_input","input_audit","input_frame","ready","execution_command"):verified_path(binding[key],path.parent)
    value=read(binding["plan"]["path"])
    if value!=plan(verified_path(value["input"],path.parent),path.parent,new=False):raise ValueError("Fusion preparation source or plan differs")
    if read(binding["execution_command"]["path"])!=value["execution_command"]:raise ValueError("Fusion fitting command differs")
    frame,audit=input_state(value,path.parent)
    return path,binding,value,frame,audit


def collect(preparation,completed,output):
    require_host();output=Path(output).resolve()
    if output.exists():raise FileExistsError("Use a new fusion collection directory")
    path,_,value,frame,audit=validate_preparation(preparation,ROOT);result=verified_path(completed,ROOT)
    if result!=path.parent/"training/release/MULTIMODAL_FUSION_BINDING.json":raise ValueError("Fusion completion is outside the prepared output")
    training=path.parent/"training"
    if output==training or training in output.parents:raise ValueError("Fusion collection cannot modify completed results")
    parent=read(training/"SUCCESS.json");binding=read(result)
    if (parent.get("status")!="SUCCESS_SECONDARY_FUSION" or parent.get("binding_sha256")!=sha256(result)
            or parent.get("core_run_id")!=value["fusion_input"]["core_run_id"] or parent.get("target_host")!="149"
            or binding.get("training_run_id")!=value["fusion_input"]["fusion_run_id"]
            or read(verified_path(binding["artifacts"]["fusion_input_audit"],result.parent))!=audit):
        raise ValueError("Fusion completion differs from prepared run/configuration/inputs")
    same_frame(pd.read_parquet(verified_path(binding["artifacts"]["fusion_input_private"],result.parent)),frame,"completed prepared matrix")
    output.mkdir(parents=True);write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,
        "preparation":record(path),"completed":record(result)})
    try:
        replay=replay_release(result,expected_sha=sha256(result));write(output/"NUMERIC_REPLAY.json",replay)
        model={"model_id":"native-fusion","producer_kind":"fusion_release","producer_binding":record(result),
            "run_id":binding["training_run_id"],"score_column":value["selected_score_column"]}
        selected=output/"SELECTED_INPUT.json"
        request={"format":"CURRENT_SELECTED_ANALYSIS_INPUT_V1","selected_model":model,
            "reference_exact_model":value["reference_exact_model"],"selection":{"kind":"EXPLICIT_SOURCE_REGISTRATION",
                "rationale":value["selection_rationale"]},"geneset":value["geneset"],"subtype":value["subtype"]}
        if "rule_evidence_binding" in value["fusion_input"]:request["rule_evidence_binding"]=value["fusion_input"]["rule_evidence_binding"]
        write(selected,request)
        from .native_results_workflow import result_references
        refs={"selected_model":model,"core_run_id":value["fusion_input"]["core_run_id"],
            "core_prediction":value["reference_exact_model"]["producer_binding"],"primary_fusion":value["fusion_input"]["primary_binding_audit"]}
        components={k:dict(v) for k,v in value["publication_components"].items()}
        for name,template in value["publication_templates"].items():
            source=output/(name+"_input.json");write(source,result_references(template,refs))
            components[name]={"status":"BUILD","input_manifest":record(source)}
        publication=output/"PUBLICATION_INPUT.json"
        write(publication,{"format":"CURRENT_ANALYSIS_WORKFLOW_INPUT_V1","release_id":binding["training_run_id"],
            "selected":{"status":"BUILD","input_manifest":record(selected)},"components":components})
        from .current_analysis_workflow import run_current_analysis_workflow
        published=run_current_analysis_workflow(publication,output/"publication")
        verified_path(completed,ROOT);validate_preparation(preparation,ROOT)
        target=output/"NATIVE_FUSION_BINDING.json"
        write(target,{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
            "preparation":record(path),"completed":record(result),"numeric_replay":record(output/"NUMERIC_REPLAY.json"),
            "selected_input":record(selected),"publication_input":record(publication),"publication":record(published["binding_path"]),
            "selected_analysis":read(published["binding_path"])["selected_analysis"],"web_binding":published["web_binding"],
            "source_ancestry_complete":False,"independent_predictive_performance_verified":False,"production_deployed":False});return target
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise
