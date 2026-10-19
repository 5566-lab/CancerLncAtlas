"""Run declared-cohort CNV statistics, independent audits and corrected E1.

The original scientific decision controls whether corrected inputs are built.
This workflow never fits the L1 baseline, prepares core tensors or trains heads.
"""
from pathlib import Path
import re
import subprocess
import sys

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, write, record
from .local_cnv_cohort import from_config
from .local_cnv_core_preparation import validate_local_cnv_audit, validate_validation_a1_authority

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_LOCAL_CNV_WORKFLOW_INPUT_V1"
FORMAT = "NATIVE_LOCAL_CNV_WORKFLOW_RELEASE_V1"
CORRECTION = "LOCAL_CNV_CORE_RETRAIN_REQUIRED"
ASSOCIATION = "scripts.run_v32_local_cnv_confounding_audit_r2"
FINALIZER = "scripts.finalize_and_audit_v32_local_cnv_r3"
VALIDATION = "scripts.materialize_v32_local_cnv_validation_associations"
VALIDATION_AUDIT = "scripts.audit_v32_local_cnv_validation_associations"
GRAPH = "scripts.materialize_v32_local_cnv_corrected_graph_authority"
CODE = ["cc_hhgt/v32/native_cnv_workflow.py", "scripts/run_native_cnv_workflow.py",
    "cc_hhgt/v32/local_cnv_cohort.py", "cc_hhgt/v32/local_cnv_core_preparation.py",
    "cc_hhgt/v32/local_cnv_sources.py", "cc_hhgt/v32/local_cnv_source_replay.py", "cc_hhgt/v32/local_cnv_confounding.py",
    "cc_hhgt/v32/native_directional_cnv.py",
    *[name.replace(".", "/")+".py" for name in (ASSOCIATION, FINALIZER, VALIDATION, VALIDATION_AUDIT, GRAPH)]]


def preflight(manifest, output):
    require_host(); manifest = Path(manifest).resolve(strict=True); output = Path(output).resolve()
    spec = read(manifest)
    required = {"format", "run_id", "candidates", "patient_map", "source_config", "directional_success", "graph_root", "graph_receipt"}
    if (not required <= set(spec) or set(spec)-required-{"expression_workflow"}
            or spec.get("format") != INPUT or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", spec.get("run_id", ""))):
        raise ValueError("Invalid native local-CNV workflow declaration")
    if output.exists(): raise FileExistsError("Use a new native local-CNV workflow output")
    files = {k:record(verified_path(spec[k], manifest.parent)) for k in ("candidates", "patient_map", "source_config", "directional_success", "graph_receipt")}
    configured = read(files["source_config"]["path"])
    cohort = from_config(configured, Path(files["source_config"]["path"]).parent,
                         files["candidates"]["path"], files["patient_map"]["path"])
    if cohort is None: raise ValueError("Native workflow requires an explicit candidate_definition")
    sources = configured["sources"]
    cov = record(verified_path(sources["covariates"], Path(files["source_config"]["path"]).parent))
    # Native workers consume these paths as absolute paths. No relocation or
    # implicit replacement of the author's source config occurs here.
    if any(not Path(sources[k]["path"]).is_absolute() for k in ("covariates", "lnc_expression_root", "gene_expression_root", "pathway_activity_root")):
        raise ValueError("Native source config requires absolute input paths")
    expression_binding = None
    if "expression_workflow" in spec:
        from .native_expression_workflow import validate
        source, binding = validate(spec["expression_workflow"], manifest.parent)
        declared = binding["outputs"]
        if (declared["candidates"] != files["candidates"] or declared["patient_map"] != files["patient_map"]
                or configured["candidate_definition"] != declared["candidate_definition"]
                or any(sources[k] != v for k,v in declared["sources"].items())):
            raise ValueError("Local-CNV inputs differ from the native expression workflow")
        expression_binding = record(source)
    graph_root = Path(spec["graph_root"])
    if not graph_root.is_absolute(): graph_root = manifest.parent/graph_root
    graph_root = graph_root.resolve(strict=True)
    from scripts.materialize_v32_local_cnv_corrected_graph_authority import load_source_receipt
    graph = load_source_receipt(Path(files["graph_receipt"]["path"]), files["graph_receipt"]["sha256"])
    if graph.get("cancer_scope") != cohort["cancers"]: raise ValueError("Graph and local-CNV cancer declarations differ")
    statics = []
    for key in ("signed_membership", "protein_gene_encoding"):
        source = record(graph_root/"static"/(key+".parquet"))
        if record(verified_path(graph["artifacts"][key], Path(files["graph_receipt"]["path"]).parent)) != source:
            raise ValueError("Statistical graph inputs differ from the source graph receipt")
        statics.append(source)
    directional = Path(files["directional_success"]["path"])
    directional_status = read(directional)
    if (directional.name != "SUCCESS.json" or directional_status.get("status") not in {"PASS", "SUCCESS"}
            or set(directional_status.get("cancers", {})) != set(cohort["cancers"])):
        raise ValueError("Directional CNV cancer coverage differs")
    from .native_directional_cnv import FORMAT as DIRECTIONAL_FORMAT, validate_release
    directional_materialization = None
    if directional_status.get("format") == DIRECTIONAL_FORMAT:
        _, native = validate_release(files["directional_success"], manifest.parent, replay=True)
        if native["cohort"] != cohort: raise ValueError("Native directional and association cohort declarations differ")
        directional_materialization = files["directional_success"]
    from .fold_local_expression import expression_path
    from .fold_local_activity import activity_path
    biological = []
    for cancer in cohort["cancers"]:
        compact = directional.parent/("cancer="+cancer)
        from .cnv_input_sources import ARRAYS, IDS
        biological += [record(compact/name) for name in (*IDS, *ARRAYS["cnv"], "SUCCESS.json", "SOURCE_FILES.json")]
        # Pin the original segment bytes, not just the JSON that names them.
        # This is the native directional materializer's SOURCE_FILES schema.
        raw_sources = read(compact/"SOURCE_FILES.json").get("records")
        if not isinstance(raw_sources,list):
            raise ValueError("Native directional CNV source records are missing")
        for item in raw_sources:
            biological.append(record(verified_path(item,compact)))
        for fold in range(5):
            biological += [record(expression_path(sources,key,cancer,fold,files["patient_map"]["path"]))
                           for key in ("lnc_expression_root", "gene_expression_root")]
            biological.append(record(activity_path(sources,cancer,fold,files["patient_map"]["path"])))
    commands = []
    def add(stage, module, args, corrected=False):
        commands.append({"stage":stage, "requires_correction":corrected,
                         "argv":[sys.executable,"-m",module,*[str(a) if not isinstance(a,dict) else a for a in args]]})
    shared = ["--candidates",files["candidates"]["path"],"--patient-map",files["patient_map"]["path"]]
    data = ["--source-config",files["source_config"]["path"],"--directional-root",directional.parent]
    for cancer in cohort["cancers"]:
        add("association:"+cancer, ASSOCIATION, [*shared,*data,"--cancer",cancer,"--covariates",cov["path"],
            "--graph-root",graph_root,"--output-root",output/"association","--run-id",spec["run_id"]])
    add("association_closeout", FINALIZER, [*shared,*data,"--root",output/"association","--covariates",cov["path"],
        "--runner-code",ROOT/(ASSOCIATION.replace(".","/")+".py"),"--run-id",spec["run_id"]])
    dependency = {"sha256_of":"association_success"}
    for cancer in cohort["cancers"]:
        add("validation:"+cancer, VALIDATION, [*shared,*data,"--cancer",cancer,
            "--local-cnv-audit-root",output/"association","--local-cnv-success-sha256",dependency,
            "--output-root",output/"validation","--run-id",spec["run_id"]+".validation"], True)
    add("validation_closeout", VALIDATION_AUDIT, [*shared,"--root",output/"validation",
        "--local-cnv-audit-success",output/"association/SUCCESS.json"], True)
    graph_args = ["--source-graph-receipt",files["graph_receipt"]["path"],"--source-graph-receipt-sha256",files["graph_receipt"]["sha256"],
        "--local-cnv-audit-root",output/"association","--local-cnv-audit-success-sha256",dependency,"--output-root",output/"corrected_graph"]
    add("graph_initialize", GRAPH, [*graph_args,"--phase","initialize"], True)
    for cancer in cohort["cancers"]: add("graph:"+cancer, GRAPH, [*graph_args,"--phase","materialize-cancer","--cancer",cancer], True)
    add("graph_closeout", GRAPH, [*graph_args,"--phase","finalize"], True)
    pins = [record(manifest),*files.values(),cov,*statics,*biological,*([expression_binding] if expression_binding else [])]
    return {"target_host":"149","training_started":False,"baseline_fitting_started":False,"input":record(manifest),
        "cohort":cohort,"sources":list({i["path"]:i for i in pins}.values()),
        "implementations":[record(ROOT/name) for name in CODE], "commands":commands,
        "files":files,"run_id":spec["run_id"],"expression_workflow":expression_binding,
        "directional_materialization":directional_materialization}


def resolved_command(item, output):
    args = []
    for value in item["argv"]:
        if isinstance(value,dict):
            if value != {"sha256_of":"association_success"}: raise ValueError("Unknown command dependency")
            value = record(output/"association/SUCCESS.json")["sha256"]
        args.append(value)
    return args


def collect(output, plan):
    output = Path(output)
    if plan.get("directional_materialization") is not None:
        from .native_directional_cnv import validate_release
        _, directional = validate_release(plan["directional_materialization"], output)
        if directional["cohort"] != plan["cohort"]: raise ValueError("CNV workflow directional cohort changed")
    if plan.get("expression_workflow") is not None:
        from .native_expression_workflow import validate as validate_expression_workflow
        validate_expression_workflow(plan["expression_workflow"], output)
    result = validate_local_cnv_audit(output/"association", require_correction=False)
    if result.get("cohort") != plan["cohort"]: raise ValueError("Workflow closeout cohort differs")
    from .local_cnv_sources import validate as validate_sources
    for cancer in plan["cohort"]["cancers"]:
        receipt = read(output/"association/receipts"/(cancer+".json"))
        if receipt["run_id"] != plan["run_id"]: raise ValueError("Workflow worker run identity differs")
        validate_sources(receipt["consumed_sources"], output/"association/receipts")
    records = {"association_success":record(output/"association/SUCCESS.json"),
        "decision":record(output/"association/LOCAL_CNV_DECISION.json"),
        "independent_audit":record(output/"association/LOCAL_CNV_INDEPENDENT_AUDIT.json")}
    if result["decision"] == CORRECTION:
        valid = validate_validation_a1_authority(output/"validation")
        if valid.get("cohort") != plan["cohort"] or valid["local_cnv_audit_success_sha256"] != result["success_sha256"]:
            raise ValueError("Workflow validation authority differs from association")
        for cancer in plan["cohort"]["cancers"]:
            receipt = read(output/"validation/receipts"/(cancer+".json"))
            if receipt["run_id"] != plan["run_id"]+".validation": raise ValueError("Validation worker identity differs")
            validate_sources(receipt["consumed_sources"], output/"validation/receipts")
        gsuccess = read(output/"corrected_graph/SUCCESS.json")
        graph_path = output/"corrected_graph/GRAPH_INPUT_AUTHORITY_RECEIPT.json"
        if (gsuccess["status"] != "PASS" or gsuccess["graph_receipt_sha256"] != record(graph_path)["sha256"]
                or gsuccess["local_cnv_audit_success_sha256"] != result["success_sha256"]):
            raise ValueError("Workflow corrected graph completion differs")
        from .input_lineage import artifact_sha256
        graph = read(graph_path)
        for fold in range(5):
            item = graph["fold_artifacts"][str(fold)]["train_coexpression"]
            if artifact_sha256(item["path"]) != item["sha256"]: raise ValueError("Corrected graph partition hash differs")
        records.update(validation_success=record(output/"validation/SUCCESS.json"),corrected_graph=record(graph_path),
                       corrected_graph_success=record(output/"corrected_graph/SUCCESS.json"))
    elif (output/"validation").exists() or (output/"corrected_graph").exists():
        raise ValueError("Non-correction decision has corrected analysis outputs")
    return result["decision"],records


def run(manifest, output):
    plan = preflight(manifest, output); output = Path(output).resolve(); output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json", plan)
    state = {"target_host":"149","training_started":False,"status":"RUNNING","stages":[]}
    write(output/"STAGE_STATUS.json", state); decision = None; stage = "initialization"
    try:
        for item in plan["commands"]:
            stage = item["stage"]
            if item["requires_correction"] and decision != CORRECTION:
                state["stages"].append({"stage":stage,"status":"NOT_RUN_BY_DECISION","decision":decision})
                write(output/"STAGE_STATUS.json", state); continue
            argv = resolved_command(item,output)
            current = {"stage":stage,"argv":argv,"status":"RUNNING"}; state["stages"].append(current)
            write(output/"STAGE_STATUS.json", state)
            result = subprocess.run(argv,cwd=ROOT,capture_output=True,text=True,timeout=3600)
            log = output/(stage.replace(":","_")+".log");log.write_text(result.stdout+result.stderr,encoding="utf-8")
            current.update(exit_code=result.returncode,log=record(log))
            if result.returncode: raise ValueError("Native CNV stage failed: "+stage+": "+result.stderr[-2200:])
            current["status"] = "PASS";write(output/"STAGE_STATUS.json",state)
            if stage == "association_closeout":
                decision = validate_local_cnv_audit(output/"association",require_correction=False)["decision"]
        stage = "native_acceptance"
        decision, outputs = collect(output,plan)
        for source in [*plan["sources"],*plan["implementations"]]: verified_path(source,output)
        state["status"] = "PASS";write(output/"STAGE_STATUS.json",state)
        binding = {"format":FORMAT,"target_host":"149","status":"PASS","training_started":False,"baseline_fitting_started":False,
            "preflight":record(output/"LAUNCH_PREFLIGHT.json"),"stage_status":record(output/"STAGE_STATUS.json"),
            "decision":decision,"corrected_inputs_available":decision==CORRECTION,"outputs":outputs,
            "core_preparation_status":"NOT_RUN_INCLUDES_L1_FITTING", "model_training_status":"NOT_RUN_AS_REQUESTED",
            "typed_graph_preparation_executed":False,"comparison_preparation_executed":False}
        path = output/"NATIVE_CNV_BINDING.json";write(path,binding)
        write(output/"SUCCESS.json",{"status":"PASS","target_host":"149","training_started":False,"binding":record(path)})
        return path
    except Exception as error:
        if state["stages"] and state["stages"][-1]["status"] == "RUNNING": state["stages"][-1]["status"] = "FAILED"
        state.update(status="FAILED",failed_stage=stage,error=str(error),error_type=type(error).__name__)
        write(output/"STAGE_STATUS.json",state);raise


def validate(item, root):
    require_host();path = verified_path(item,root);binding = read(path);output = path.parent
    if (binding.get("format") != FORMAT or binding.get("status") != "PASS" or binding.get("target_host") != "149"
            or binding.get("training_started") is not False or binding.get("baseline_fitting_started") is not False
            or read(output/"SUCCESS.json") != {"status":"PASS","target_host":"149","training_started":False,"binding":record(path)}):
        raise ValueError("Incomplete native local-CNV workflow")
    plan = read(verified_path(binding["preflight"],output));state = read(verified_path(binding["stage_status"],output))
    for source in [*plan["sources"],*plan["implementations"]]: verified_path(source,output)
    decision,outputs = collect(output,plan)
    if (binding["decision"] != decision or binding["outputs"] != outputs or binding["corrected_inputs_available"] != (decision==CORRECTION)
            or state["status"] != "PASS" or len(state["stages"]) != len(plan["commands"])):
        raise ValueError("Native local-CNV result/command closure differs")
    for entry,command in zip(state["stages"],plan["commands"]):
        if entry["stage"] != command["stage"]: raise ValueError("CNV workflow stage order differs")
        if command["requires_correction"] and decision != CORRECTION:
            if entry != {"stage":command["stage"],"status":"NOT_RUN_BY_DECISION","decision":decision}:
                raise ValueError("CNV workflow scientific decision was bypassed")
        else:
            if entry["status"] != "PASS" or entry["exit_code"] != 0 or entry["argv"] != resolved_command(command,output):
                raise ValueError("CNV workflow command differs")
            verified_path(entry["log"],output)
    return path,binding
