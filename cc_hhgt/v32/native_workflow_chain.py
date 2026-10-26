"""Resume native scientific workflows without learned-model optimizer training.

Only the closed operation registry below is executable. Registered statistical
workflows can run regressions. Completed neural/L1/auxiliary/fusion fitting is
supplied as pinned external data and checked by the original native consumers.
An OS file lock prevents two coordinators from advancing the same chain.
"""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy
import importlib
import json
from pathlib import Path
import re

from .cnv_input_sources import require_host
from .current_core_export import verified_path, checked_sha
from .formal_expression_sources import read, record
from .current_analysis_workflow import write

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_WORKFLOW_CHAIN_INPUT_V1"
FORMAT = "NATIVE_WORKFLOW_CHAIN_V1"
EXTERNAL = "NATIVE_WORKFLOW_CHAIN_EXTERNAL_V1"
STAGE = "NATIVE_WORKFLOW_CHAIN_STAGE_V1"
SLUG = re.compile(r"[a-z][a-z0-9_-]{0,63}")
MODULES = {name: "cc_hhgt.v32.native_"+name+"_workflow" for name in
           ("expression", "cnv", "core", "results", "auxiliary", "genomic", "state", "clinical", "single_cell", "fusion", "rbp", "methylation", "distal", "distal_atac", "probe", "distal_mutation", "single_cell_mapping", "single_cell_donors")}
RUNS = {"expression.run": "expression", "cnv.run": "cnv", "results.run": "results", "rbp.run":"rbp", "methylation.run":"methylation", "distal.run":"distal", "distal_atac.run":"distal_atac", "probe.run":"probe", "distal_mutation.run":"distal_mutation", "single_cell_mapping.run":"single_cell_mapping", "single_cell_donors.run":"single_cell_donors"}
MODULES["single_cell_static"] = "cc_hhgt.v32.native_single_cell_static_workflow"
RUNS["single_cell_static.run"] = "single_cell_static"
MODULES["single_cell_graph"] = "cc_hhgt.v32.native_single_cell_graph_workflow"
RUNS["single_cell_graph.run"] = "single_cell_graph"
MODULES["single_cell_episodes"] = "cc_hhgt.v32.native_single_cell_episodes_workflow"
MODULES["annotation"] = "cc_hhgt.v32.native_annotation_workflow"
RUNS["annotation.run"] = "annotation"
MODULES["single_cell_annotation"] = "cc_hhgt.v32.native_single_cell_annotation_workflow"
RUNS["single_cell_annotation.run"] = "single_cell_annotation"
MODULES["genomic_annotation"] = "cc_hhgt.v32.native_genomic_annotation_workflow"
RUNS["genomic_annotation.run"] = "genomic_annotation"
MODULES["pathway_membership"] = "cc_hhgt.v32.native_pathway_membership_workflow"
RUNS["pathway_membership.run"] = "pathway_membership"
PAIRED = ("auxiliary", "genomic", "state", "clinical", "single_cell", "fusion", "single_cell_episodes")
OPERATIONS = {*RUNS, "core.accept", "publication.run", *[n+"."+p for n in PAIRED for p in ("prepare", "collect")]}


class Waiting(Exception):
    pass


def module(name):
    return importlib.import_module(MODULES[name])


def code_records():
    names = [__file__, str(ROOT/"scripts/run_native_workflow_chain.py")]
    names += [str(ROOT/(name.replace(".", "/")+".py")) for name in MODULES.values()]
    names += [str(ROOT/"cc_hhgt/v32/current_analysis_workflow.py")]
    return [record(Path(name)) for name in names]


def pointer(value, path):
    if not isinstance(path, str) or (path and not path.startswith("/")):
        raise ValueError("Chain references require an empty or absolute JSON pointer")
    for key in path.split("/")[1:]:
        key = key.replace("~1", "/").replace("~0", "~")
        try: value = value[int(key)] if isinstance(value, list) else value[key]
        except (KeyError, IndexError, ValueError, TypeError) as error:
            raise ValueError("Chain result pointer is unavailable: "+path) from error
    return deepcopy(value)


def input_pin(value,root):
    path=(Path(root)/value["path"]).resolve()
    if path.is_dir():
        # Native producers use different directory membership hash contracts.
        # Preserve the declaration for that producer's actual validator rather
        # than hashing a directory as a file or guessing a replacement scheme.
        return {**value,"path":str(path),"sha256":checked_sha(value.get("sha256"),"Native directory input")}
    return {**value,**record(verified_path(value,root))}


def normalize(value, root, dependencies, externals):
    if isinstance(value, list): return [normalize(v, root, dependencies, externals) for v in value]
    if not isinstance(value, dict): return value
    for key, names in (("stage_ref", dependencies), ("external_ref", externals)):
        if key in value:
            if set(value)-{key, "pointer"} or not isinstance(value[key], str) or not SLUG.fullmatch(value[key]):
                raise ValueError("Invalid chain reference")
            path = value.get("pointer", "")
            if not isinstance(path, str) or (path and not path.startswith("/")): raise ValueError("Invalid JSON pointer")
            names.add(value[key]); return {key: value[key], "pointer": path}
    if "json_file" in value:
        if set(value)!={"json_file"}: raise ValueError("Ambiguous generated JSON input")
        return {"json_file": normalize(value["json_file"], root, dependencies, externals)}
    if "project_path" in value:
        if set(value)!={"project_path"} or not isinstance(value["project_path"], str): raise ValueError("Invalid project path")
        return str((root/value["project_path"]).resolve())
    if {"path", "sha256"} <= set(value):
        return input_pin(value,root)
    return {k: normalize(v, root, dependencies, externals) for k,v in value.items()}


def plan(manifest):
    require_host(); path = Path(manifest).resolve(); spec = read(path)
    if (set(spec)!={"format", "chain_id", "stages"} or spec["format"]!=INPUT
            or not isinstance(spec["chain_id"], str) or not SLUG.fullmatch(spec["chain_id"])
            or not isinstance(spec["stages"], list) or not spec["stages"]):
        raise ValueError("Invalid native workflow chain")
    stages = []; ids = set(); all_externals = set()
    for row in spec["stages"]:
        if (not isinstance(row, dict) or set(row)!={"id", "operation", "inputs"}
                or not isinstance(row["id"], str) or not SLUG.fullmatch(row["id"]) or row["id"] in ids
                or row["operation"] not in OPERATIONS or not isinstance(row["inputs"], dict)):
            raise ValueError("Unknown, duplicate or unsafe native chain operation")
        dependencies = set(); externals = set()
        inputs = normalize(row["inputs"], path.parent, dependencies, externals)
        # Topological order is explicit; never guess dependencies or reorder a
        # scientific workflow. Missing inputs do not prevent independent stages.
        if not dependencies <= ids: raise ValueError("Chain has a forward, cyclic or unknown dependency")
        if row["operation"].endswith(".collect"):
            expected = {"preparation", "risk", "entity"} if row["operation"]=="clinical.collect" else {"preparation", "completed"}
            if set(inputs)!=expected: raise ValueError("Wrong native collection inputs")
        elif row["operation"]=="core.accept" and set(inputs)!={"binding", "cnv_workflow"}:
            raise ValueError("Core registration requires the completed preparation and its native CNV source")
        stages.append({**row, "inputs": inputs, "dependencies": sorted(dependencies), "externals": sorted(externals)})
        ids.add(row["id"]); all_externals.update(externals)
    return {"format": FORMAT, "chain_id": spec["chain_id"], "target_host": "149", "training_started": False,
            "input": record(path), "stages": stages, "external_names": sorted(all_externals), "implementations": code_records(),
            "completion_scope": "DECLARED_NATIVE_WORKFLOW_STAGES_ONLY", "native_stage_preflight_completed":False,
            "production_deployed": False}


@contextmanager
def locked(root):
    import fcntl
    with (root/"COORDINATOR.lock").open("a+") as handle:
        try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error: raise RuntimeError("Another coordinator is advancing this chain") from error
        try: yield
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def initialize(manifest, output):
    value = plan(manifest); root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write(root/"LAUNCH_PREFLIGHT.json", value)
    write(root/"CHAIN.json", {"format": FORMAT, "target_host": "149", "training_started": False,
        "plan": record(root/"LAUNCH_PREFLIGHT.json"), "stages": {}, "externals": {}, "status": "READY",
        "completion_scope": value["completion_scope"], "production_deployed": False})
    return root/"CHAIN.json"


def resolve(value, stages, externals, folder, *, materialize):
    if isinstance(value, list): return [resolve(v, stages, externals, folder, materialize=materialize) for v in value]
    if not isinstance(value, dict): return value
    for key, records in (("stage_ref", stages), ("external_ref", externals)):
        if key in value:
            name = value[key]
            if name not in records: raise Waiting(key+":"+name)
            item = records[name]; source = verified_path(item, folder)
            result = pointer(read(source), value["pointer"]) if value["pointer"] else record(source)
            dependencies=set();extra=set()
            result=normalize(result,source.parent,dependencies,extra)
            if dependencies or extra:raise ValueError("Chain result contains undeclared references")
            return resolve(result, {}, {}, folder, materialize=materialize)
    if "json_file" in value:
        payload = resolve(value["json_file"], stages, externals, folder, materialize=materialize)
        # Content-addressed names are stable between execution and resume.
        import hashlib
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        path = folder/("input_"+digest+".json")
        if materialize:
            if path.exists() and read(path)!=payload: raise ValueError("Generated chain input changed")
            if not path.exists(): write(path, payload)
        elif not path.is_file() or read(path)!=payload: raise ValueError("Generated chain input is missing or changed")
        return record(path)
    if {"path", "sha256"} <= set(value): return input_pin(value,folder)
    return {k: resolve(v, stages, externals, folder, materialize=materialize) for k,v in value.items()}


def dispatch(operation, inputs, folder):
    """Closed dispatcher: no user-supplied command, fit function or shell."""
    if operation not in OPERATIONS: raise ValueError("Unsafe native chain operation")
    output = folder/"output"
    if operation=="core.accept":
        source, value = module("core").validate(inputs["binding"], folder)
        cnv, _ = module("cnv").validate(inputs["cnv_workflow"], folder)
        if read(value["preflight"]["path"])["native_cnv_workflow"]!=record(cnv):
            raise ValueError("Registered core preparation belongs to another native CNV workflow")
        return source
    name, phase = operation.split(".")
    if phase=="collect":
        names = ("preparation", "risk", "entity") if name=="clinical" else ("preparation", "completed")
        return module(name).collect(*[inputs[k] for k in names], output)
    manifest = folder/"INPUT.json"; write(manifest, inputs)
    if phase=="prepare": return module(name).prepare(manifest, output)
    if operation in RUNS: return module(RUNS[operation]).run(manifest, output)
    if operation=="publication.run":
        from .current_analysis_workflow import run_current_analysis_workflow
        return Path(run_current_analysis_workflow(manifest, output)["binding_path"])
    raise ValueError("Unsupported native chain operation")


def validate_result(operation, item, folder):
    name, phase = operation.split(".")
    if phase=="prepare": module(name).validate_preparation(item, folder)
    elif operation in RUNS or name in ("auxiliary", "genomic", "state", "clinical", "core", "single_cell_episodes"):
        module(name).validate(item, folder)
    elif name=="single_cell":
        path = verified_path(item, folder); value = read(path)
        if (value.get("format")!=module(name).FORMAT or value.get("status")!="PASS"
                or value.get("target_host")!="149" or value.get("training_started") is not False
                or value.get("native_model_forwards_replayed") is not True):
            raise ValueError("Incomplete SC collection")
        from .single_cell_signed_fusion import load_signed_sc_fusion
        signed = verified_path(value["signed_fusion"], path.parent)
        bound, _, _, _ = load_signed_sc_fusion(signed, expected_sha=value["signed_fusion"]["sha256"])
        pointer_path=verified_path(value["pointer"],path.parent)
        if verified_path(bound["inputs"]["pointer"],signed.parent)!=pointer_path:
            raise ValueError("SC collection pointer differs")
        # Signed records also carry byte counts; compare verified file identity,
        # then connect the collection envelope to the pointer actually replayed.
        declared=read(pointer_path)
        for key,field in (("preparation","workflow_preparation"),("training_success","training_success"),
                          ("numeric_replay","numeric_replay")):
            if verified_path(value[key],path.parent)!=verified_path(declared[field],pointer_path.parent):
                raise ValueError("SC collection dependency differs: "+key)
    elif name=="fusion":
        module(name).validate(item, folder)
    elif operation=="publication.run":
        from .current_analysis_workflow import validate_current_analysis_workflow
        validate_current_analysis_workflow(item, folder)
    else: raise ValueError("Unverifiable chain result")


def validate_stage(row, receipt, results, externals, root):
    path = verified_path(receipt, root); value = read(path); folder = path.parent
    resolved = resolve(row["inputs"], results, externals, folder, materialize=False)
    if (value.get("format")!=STAGE or value.get("status")!="PASS" or value.get("operation")!=row["operation"]
            or value.get("stage_id")!=row["id"] or value.get("target_host")!="149"
            or value.get("training_started") is not False or value.get("inputs")!=resolved):
        raise ValueError("Completed chain stage differs from its declaration")
    for item in value["implementations"]: verified_path(item, root)
    if value["implementations"]!=code_records(): raise ValueError("Chain implementation changed")
    result = verified_path(value["result"], folder)
    if row["operation"]!="core.accept" and folder/"output" not in result.parents:
        raise ValueError("Chain result is outside its stage output")
    if row["operation"]=="core.accept":
        if record(result)!=resolved["binding"]: raise ValueError("Registered core result differs")
        source = read(read(result)["preflight"]["path"])["native_cnv_workflow"]
        if source!=resolved["cnv_workflow"]: raise ValueError("Registered core source differs")
    elif row["operation"].endswith((".prepare", ".run")) and read(folder/"INPUT.json")!=resolved:
        raise ValueError("Native stage input manifest changed")
    native=read(result);operation=row["operation"]
    if operation.endswith(".prepare") or operation in RUNS:
        key="plan" if operation.endswith(".prepare") else "preflight"
        original=read(verified_path(native[key],result.parent))["input"]
        if original!=record(folder/"INPUT.json"):raise ValueError("Native result belongs to another stage input")
    elif operation=="publication.run":
        if native["input_manifest"]!=record(folder/"INPUT.json"):raise ValueError("Publication belongs to another stage input")
    elif operation.endswith(".collect"):
        keys={"preparation":"preparation","completed":"training_success" if operation=="single_cell.collect" else "completed"}
        if operation=="clinical.collect":keys={"preparation":"preparation","risk":"risk_completed","entity":"entity_completed"}
        if any(native[field]!=resolved[key] for key,field in keys.items()):raise ValueError("Native collection belongs to another completion")
    validate_result(row["operation"], value["result"], folder)
    return record(result)


def advance(output, external_manifest=None):
    require_host(); root = Path(output).resolve(strict=True)
    with locked(root):
        state = read(root/"CHAIN.json"); launch = verified_path(state["plan"], root); value = read(launch)
        if state.get("format")!=FORMAT or state.get("target_host")!="149" or state.get("training_started") is not False:
            raise ValueError("Invalid chain state")
        if value!=plan(verified_path(value["input"], root)): raise ValueError("Chain declaration or implementation changed")
        if set(state["stages"])-{s["id"] for s in value["stages"]}: raise ValueError("Unknown recorded stage")
        externals = dict(state["externals"])
        if external_manifest:
            source = Path(external_manifest).resolve(); declared = read(source)
            if (set(declared)!={"format", "inputs"} or declared["format"]!=EXTERNAL
                    or not isinstance(declared["inputs"], dict) or set(declared["inputs"])-set(value["external_names"])):
                raise ValueError("Invalid external completion registry")
            for name,item in declared["inputs"].items():
                pin = record(verified_path(item, source.parent))
                if name in externals and externals[name]!=pin: raise ValueError("Registered external input cannot be replaced")
                externals[name] = pin
        for item in externals.values(): verified_path(item, root)
        results = {}; waiting = []; failures = []; state["externals"] = externals
        state.update(status="ADVANCING", training_started=False); write(root/"CHAIN.json", state)
        for row in value["stages"]:
            key = row["id"]
            if key in state["stages"]:
                try: results[key] = validate_stage(row, state["stages"][key], results, externals, root)
                except Exception as error:
                    state.update(status="FAILED",failures=[{"stage":key,"phase":"REVALIDATION","error":str(error)}])
                    write(root/"CHAIN.json",state);raise
                continue
            unavailable = ["stage:"+n for n in row["dependencies"] if n not in results]
            unavailable += ["external:"+n for n in row["externals"] if n not in externals]
            if unavailable: waiting.append({"stage": key, "missing": unavailable}); continue
            stage_root = root/"stages"/key; stage_root.mkdir(parents=True, exist_ok=True)
            attempts = [p for p in stage_root.iterdir() if p.is_dir() and p.name.isdigit()]
            if attempts:
                prior=max(attempts,key=lambda p:int(p.name)); completed=prior/"STAGE_BINDING.json"
                if completed.is_file():
                    results[key]=validate_stage(row,record(completed),results,externals,root)
                    state["stages"][key]=record(completed);write(root/"CHAIN.json",state);continue
                if not (prior/"RUN_FAILED.json").is_file():
                    waiting.append({"stage":key,"missing":["INCOMPLETE_ATTEMPT_REQUIRES_INSPECTION"],"attempt":str(prior)})
                    continue
            folder = stage_root/f"{max([int(p.name) for p in attempts], default=0)+1:04d}"
            folder.mkdir()
            write(folder/"LAUNCH_PREFLIGHT.json", {"target_host":"149", "training_started":False,
                "operation":row["operation"], "stage_id":key, "chain_plan":record(launch)})
            try:
                resolved = resolve(row["inputs"], results, externals, folder, materialize=True)
                with (folder/"stage.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                    result = dispatch(row["operation"], resolved, folder)
                result = record(result)
                stage = folder/"STAGE_CANDIDATE.json"
                write(stage, {"format":STAGE, "status":"PASS", "target_host":"149", "training_started":False,
                    "stage_id":key, "operation":row["operation"], "inputs":resolved, "result":result,
                    "implementations":code_records()})
                results[key] = validate_stage(row, record(stage), results, externals, root)
                target=folder/"STAGE_BINDING.json";stage.replace(target);stage=target
                state["stages"][key] = record(stage); write(root/"CHAIN.json", state)
            except Exception as error:
                failure = {"stage":key, "attempt":str(folder), "error_type":type(error).__name__, "error":str(error)}
                write(folder/"RUN_FAILED.json", {**failure, "target_host":"149", "training_started":False})
                failures.append(failure)
        for item in externals.values(): verified_path(item, root)
        if value!=plan(value["input"]["path"]): raise ValueError("Chain inputs changed while advancing")
        inspection=any("INCOMPLETE_ATTEMPT_REQUIRES_INSPECTION" in item["missing"] for item in waiting)
        state.update(status="FAILED" if failures else "NEEDS_INSPECTION" if inspection else "WAITING_EXTERNAL" if waiting else "DECLARED_STAGES_COMPLETE",
            waiting=waiting, failures=failures, results=results)
        write(root/"CHAIN.json", state)
        return root/"CHAIN.json"


def recover(output, stage_id, completed):
    """Adopt a validated finished result after interruption, without rerunning it."""
    require_host();root=Path(output).resolve(strict=True)
    with locked(root):
        state=read(root/"CHAIN.json");value=read(verified_path(state["plan"],root))
        if value!=plan(verified_path(value["input"],root)):raise ValueError("Chain declaration or implementation changed")
        if stage_id in state["stages"]:raise ValueError("Chain stage is already registered")
        rows={r["id"]:r for r in value["stages"]}
        if stage_id not in rows:raise ValueError("Unknown recovery stage")
        results={}
        for row in value["stages"]:
            if row["id"]==stage_id:break
            if row["id"] in state["stages"]:
                results[row["id"]]=validate_stage(row,state["stages"][row["id"]],results,state["externals"],root)
        row=rows[stage_id];stage_root=root/"stages"/stage_id
        attempts=[p for p in stage_root.iterdir() if p.is_dir() and p.name.isdigit()]
        if not attempts:raise ValueError("No interrupted chain attempt to recover")
        folder=max(attempts,key=lambda p:int(p.name));target=folder/"STAGE_BINDING.json"
        if target.exists():raise ValueError("A finished stage receipt exists; use advance to reconcile it")
        resolved=resolve(row["inputs"],results,state["externals"],folder,materialize=False)
        pin=record(verified_path(completed,root));candidate=folder/"RECOVERY_CANDIDATE.json"
        write(candidate,{"format":STAGE,"status":"PASS","target_host":"149","training_started":False,
            "stage_id":stage_id,"operation":row["operation"],"inputs":resolved,"result":pin,"implementations":code_records()})
        validate_stage(row,record(candidate),results,state["externals"],root)
        candidate.replace(target);state["stages"][stage_id]=record(target)
        state.update(status="READY",recovered_stage=stage_id)
        write(root/"CHAIN.json",state);return root/"CHAIN.json"
