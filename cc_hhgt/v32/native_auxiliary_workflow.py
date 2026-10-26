"""Native CNV/ATAC preparation, explicit training commands and completed results.

This orchestrator never invokes a trainer. It executes the original no-fit input
collectors and independent completed-result auditors, then the actual native
fusion adapter. Training remains a separate explicit execution of the generated
command, with the same pinned input and parameter declaration.
"""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import importlib
import math
from pathlib import Path
import re
import sys

from .cnv_input_sources import require_host
from .current_core_export import verified_path, validate_current_export
from .formal_expression_sources import read, write, record
from .native_pathway_membership_workflow import CODE as MEMBERSHIP_CODE

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_AUXILIARY_WORKFLOW_INPUT_V1"
PREPARED = "NATIVE_AUXILIARY_WORKFLOW_PREPARED_V1"
FORMAT = "NATIVE_AUXILIARY_WORKFLOW_RESULTS_V1"
MODULES = {
    "cnv": ("scripts.preflight_v32_directional_cnv_head", "scripts.run_v32_directional_cnv_head_oof",
            "scripts.audit_v32_directional_cnv_head_oof"),
    "atac": ("scripts.preflight_v32_atac_training", "scripts.run_v32_atac_training",
             "scripts.audit_v32_atac_fresh_oof_result"),
}
SAMPLING = {
    "cnv": ("seed", "min_pair_callable", "max_train_rows", "max_validation_rows"),
    "atac": ("seed", "min_pathway_promoter_genes", "min_inner_train_patients", "min_outer_train_patients", "max_training_rows"),
}
CODE = ("cc_hhgt/v32/native_auxiliary_workflow.py", "scripts/run_native_auxiliary_workflow.py",
    "cc_hhgt/v32/cnv_only_training_v2.py", "cc_hhgt/v32/cnv_head_association.py",
    "cc_hhgt/v32/cnv_input_sources.py", "cc_hhgt/v32/genomic_training.py",
    "cc_hhgt/v32/atac_training.py", "cc_hhgt/v32/atac_cohort.py",
    "cc_hhgt/v32/atac_materialization.py", *MEMBERSHIP_CODE,
    "cc_hhgt/v32/native_auxiliary_fusion.py", "cc_hhgt/v32/current_core_export.py",
    *(module.replace(".", "/")+".py" for modules in MODULES.values() for module in modules))


def implementations(): return [record(ROOT/name) for name in CODE]


def load_input(manifest):
    require_host(); manifest = Path(manifest).resolve(strict=True); spec = read(manifest)
    kind = spec.get("producer_kind")
    common = {"format", "producer_kind", "native_run_id", "core_run_id", "candidate_authority", "patient_folds"}
    extra = {"cnv": {"directional_success", "association_success", "core_manifest"},
             "atac": {"materialization_success", "pathway_membership"}}
    if (kind not in extra or spec.get("format") != INPUT or not common|extra.get(kind, set()) <= set(spec)
            or set(spec)-common-extra.get(kind, set())-{"config"}
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,119}", str(spec.get("native_run_id", "")))
            or not isinstance(spec.get("core_run_id"), str) or not spec["core_run_id"].strip()):
        raise ValueError("Invalid native auxiliary workflow manifest")
    names = {"candidate_authority", "patient_folds"}|extra[kind]
    files = {key:record(verified_path(spec[key], manifest.parent)) for key in names}
    if kind == "cnv":
        from .genomic_training import GenomicTrainingConfig
        defaults = GenomicTrainingConfig(); excluded = {"cnv_event_threshold"}
    else:
        from .atac_training import AtacTrainingConfig
        defaults = AtacTrainingConfig(); excluded = {"expected_rows_per_cancer"}
    values = asdict(defaults); override = spec.get("config", {})
    if not isinstance(override, dict) or set(override)-set(values) or set(override)&excluded:
        raise ValueError("Unsupported native auxiliary training configuration")
    for key, value in override.items():
        expected = values[key]
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or (type(expected) is int and type(value) is not int)):
            raise ValueError("Invalid numeric auxiliary configuration: "+key)
        values[key] = value
    settings = type(defaults)(**values); settings.validate()
    return manifest, spec, files, asdict(settings), excluded


def arguments(values):
    return [part for key, value in values.items() for part in ("--"+key.replace("_", "-"), str(value))]


def plan(manifest, output, *, new=True):
    manifest, spec, files, config, excluded = load_input(manifest); output = Path(output).resolve()
    if new and output.exists(): raise FileExistsError("Use a new auxiliary preparation directory")
    kind = spec["producer_kind"]; modules = MODULES[kind]
    base = {"candidates":files["candidate_authority"]["path"], "patient_folds":files["patient_folds"]["path"]}
    if kind == "cnv":
        for key in ("directional_success", "association_success"):
            if Path(files[key]["path"]).name != "SUCCESS.json": raise ValueError("CNV requires explicit aggregate SUCCESS inputs")
        base.update(directional_root=str(Path(files["directional_success"]["path"]).parent),
                    association_root=str(Path(files["association_success"]["path"]).parent))
        prepare_args = {**base, "core_manifest":files["core_manifest"]["path"]}
        train_args = {**base, "core_embedding_manifest":files["core_manifest"]["path"]}
    else:
        base.update(materialization_success=files["materialization_success"]["path"],
                    pathway_membership=files["pathway_membership"]["path"])
        prepare_args = train_args = base
    prepare_args = {**prepare_args, "output":str(output/"HEAD_PREFLIGHT.json"), **{k:config[k] for k in SAMPLING[kind]}}
    train_args = {**train_args, "output_root":str(output/"training"), "training_run_id":spec["native_run_id"],
                  **{k:v for k,v in config.items() if k not in excluded}}
    command = [sys.executable, "-m", modules[1], *arguments(train_args)]
    # Parse with the actual training CLI, without calling its execution function.
    importlib.import_module(modules[1])._parser().parse_args(command[3:])
    return {"target_host":"149", "training_started":False, "input":record(manifest), "kind":kind,
        "native_run_id":spec["native_run_id"], "core_run_id":spec["core_run_id"], "files":files, "config":config,
        "prepare_command":[sys.executable, "-m", modules[0], *arguments(prepare_args)],
        "training_command":{"argv":command, "execution_mode":"CODE_ONLY", "automatically_executed":False,
                            "requires_separate_explicit_execution":True},
        "implementations":implementations()}


def execute(command, log):
    # Allow only the original no-fit preparation/audit programs in this dispatcher.
    module = command[2]
    allowed = {value[i] for value in MODULES.values() for i in (0, 2)}
    if module not in allowed: raise ValueError("Auxiliary workflow cannot dispatch a training command")
    with Path(log).open("w", encoding="utf-8") as handle, redirect_stdout(handle), redirect_stderr(handle):
        result = importlib.import_module(module).main(command[3:])
        if result not in (None, 0): raise RuntimeError("Auxiliary stage returned failure")


def input_state(value, ready):
    """Re-enter the actual trainer input loader and validate every captured file."""
    files = value["files"]; kind = value["kind"]; paths = {k:verified_path(v, ROOT) for k,v in files.items()}
    if ready.get("status") != "PASS_READY" or ready.get("target_host") != "149" or ready.get("training_started") is not False:
        raise ValueError("Auxiliary input collection has not completed without fitting")
    if ready.get("sampling_config") != {k:value["config"][k] for k in SAMPLING[kind]}:
        raise ValueError("Auxiliary sampling parameters differ")
    if kind == "cnv":
        from .cnv_only_training_v2 import prepare_directional_inputs
        validate_current_export(paths["core_manifest"], expected_run_id=value["core_run_id"])
        actual = prepare_directional_inputs(candidates_path=paths["candidate_authority"], folds_path=paths["patient_folds"],
            directional_root=paths["directional_success"].parent, association_root=paths["association_success"].parent,
            core_manifest_path=paths["core_manifest"])
        if ready.get("association_run_id") != actual["authority"]["run_id"]: raise ValueError("CNV association run differs")
        rows = ready.get("core_folds", [])
        if len(rows) != 5 or {r["fold_id"] for r in rows} != set(range(5)): raise ValueError("Incomplete CNV input folds")
        observed = {r["path"]:r["sha256"] for r in ready["consumed_inputs"]}
        for item in actual["capture"].records.values():
            if observed.get(item["path"]) != item["sha256"]: raise ValueError("CNV input collector omitted or changed an input")
        pins = [*ready["consumed_inputs"], *[{"path":p, "sha256":h} for r in rows for p,h in r["input_hashes"].items()]]
    else:
        from .atac_training import prepare_atac_inputs, AtacTrainingConfig
        actual = prepare_atac_inputs(candidates_path=paths["candidate_authority"], patient_folds_path=paths["patient_folds"],
            pathway_membership_path=paths["pathway_membership"], materialization_success_path=paths["materialization_success"],
            config=AtacTrainingConfig(**value["config"]))
        rows = ready.get("folds", [])
        if len(rows) != 5 or {r["fold"] for r in rows} != set(range(5)): raise ValueError("Incomplete ATAC input folds")
        m = actual["materialization"]
        pins = [*m["records"], *m["input_artifacts"],
                {"path":m["input_contract_path"], "sha256":m["input_contract_sha256"]},
                {"path":m["manifest_path"], "sha256":m["manifest_sha256"]},
                {"path":m["r_audit_path"], "sha256":m["r_audit_sha256"]}]
    if not actual["cohort"] or ready.get("cohort") != actual["cohort"] or ready.get("input_hashes") != actual["input_hashes"]:
        raise ValueError("Auxiliary workflow requires the matching native cohort and actual input hashes")
    pins += [actual["cohort"]["candidate_definition"], *files.values()]
    return sorted({str(verified_path(p, ROOT)):record(verified_path(p, ROOT)) for p in pins}.values(), key=lambda p:p["path"])


def prepare(manifest, output):
    value = plan(manifest, output); output = Path(output).resolve(); output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json", value)
    try:
        execute(value["prepare_command"], output/"prepare.log")
        ready = read(output/"HEAD_PREFLIGHT.json"); pins = input_state(value, ready)
        if value != plan(manifest, output, new=False): raise ValueError("Auxiliary inputs/code changed during preparation")
        write(output/"TRAINING_COMMAND.json", value["training_command"])
        write(output/"PREPARED.json", {"format":PREPARED, "status":"PASS_READY", "target_host":"149",
            "training_started":False, "plan":record(output/"LAUNCH_PREFLIGHT.json"), "input":value["input"],
            "head_preflight":record(output/"HEAD_PREFLIGHT.json"), "training_command":record(output/"TRAINING_COMMAND.json"),
            "consumed_sources":pins})
        return output/"PREPARED.json"
    except Exception as error:
        write(output/"RUN_FAILED.json", {"target_host":"149", "training_started":False,
            "stage":"prepare", "error_type":type(error).__name__, "error":str(error)})
        raise


def validate_preparation(item, root):
    require_host(); path = verified_path(item, root); binding = read(path)
    if (binding.get("format") != PREPARED or binding.get("status") != "PASS_READY"
            or binding.get("target_host") != "149" or binding.get("training_started") is not False):
        raise ValueError("Invalid native auxiliary preparation binding")
    manifest = verified_path(binding["input"], path.parent)
    value = read(verified_path(binding["plan"], path.parent))
    if value != plan(manifest, path.parent, new=False): raise ValueError("Auxiliary preparation plan changed")
    command = read(verified_path(binding["training_command"], path.parent))
    if command != value["training_command"]: raise ValueError("Auxiliary training command differs")
    ready = read(verified_path(binding["head_preflight"], path.parent))
    if input_state(value, ready) != binding["consumed_sources"]: raise ValueError("Auxiliary consumed source inventory differs")
    for pin in binding["consumed_sources"]: verified_path(pin, path.parent)
    return path, binding, value


def validate_completed(manifest, context, root, *, require_audit=True):
    """Called by the actual fusion input consumer when workflow ancestry is bound."""
    path, binding, value = validate_preparation(manifest["workflow_preparation"], root)
    kind = value["kind"]; files = value["files"]
    if (manifest["producer_kind"] != kind or manifest["native_run_id"] != value["native_run_id"]
            or manifest["core_run_id"] != value["core_run_id"]
            or any(manifest[k] != files[k] for k in ("candidate_authority", "patient_folds"))):
        raise ValueError("Completed auxiliary result differs from prepared run/authorities")
    expected = path.parent/"training"/("OOF_MANIFEST.json" if kind == "cnv" else "SUCCESS.json")
    if verified_path(manifest["native_receipt"], root) != expected: raise ValueError("Auxiliary result is outside its declared training output")
    sources = context["sources"]
    sources.bind(manifest["workflow_preparation"], root)
    for name in ("input", "plan", "head_preflight", "training_command"): sources.bind(binding[name], path.parent)
    for pin in binding["consumed_sources"]: sources.bind(pin, path.parent)
    if kind == "cnv":
        if manifest.get("core_manifest") != files["core_manifest"] or context.get("consumed") is None:
            raise ValueError("CNV completion lacks its prepared core/consumed inputs")
        observed = {r["role"]:r["sha256"] for r in context["consumed"][1]["records"] if r["role"] in {"materialization_success", "association_success"}}
        for role,key in (("materialization_success","directional_success"), ("association_success","association_success")):
            if observed.get(role) != files[key]["sha256"]: raise ValueError("CNV completion consumed a different prepared input")
        config = context["native"].get("training_config")
    else:
        lineage = context["lineage"]
        input_audit = read(sources.bind({"path":lineage["input_lineage_audit_path"], "sha256":lineage["input_lineage_audit_sha256"]}, expected.parent))
        observed = {r["role"]:r["sha256"] for r in input_audit["input_artifacts"]}
        if (observed.get("materialization_success") != files["materialization_success"]["sha256"]
                or observed.get("membership") != files["pathway_membership"]["sha256"]):
            raise ValueError("ATAC completion consumed a different prepared input")
        run_config = read(sources.bind({"path":lineage["config_path"], "sha256":lineage["config_sha256"]}, expected.parent))
        if run_config.get("training_run_id") != value["native_run_id"]: raise ValueError("ATAC run configuration identity differs")
        config = run_config.get("training")
    if config != value["config"]: raise ValueError("Auxiliary completion configuration differs from preparation")
    if require_audit:
        declared = manifest.get("workflow_audit")
        if not isinstance(declared, dict) or set(declared) != {"audit", "success"}:
            raise ValueError("Workflow auxiliary consumer requires its independent audit binding")
        audit_path = sources.bind(declared["audit"], root); audit = read(audit_path)
        success = read(sources.bind(declared["success"], root))
        audit_format = ("CC_HHGT_V3_2_CNV_OOF_INDEPENDENT_AUDIT_V1" if kind == "cnv" else
                        "CC_HHGT_V3_2_ATAC_FRESH_OOF_INDEPENDENT_AUDIT_V1")
        source_key = "oof_manifest_sha256" if kind == "cnv" else "training_success_sha256"
        audit_key = "independent_audit_sha256" if kind == "cnv" else "audit_sha256"
        run = success.get("training_run_id") if kind == "cnv" else audit.get("training_run_id")
        if (audit.get("format") != audit_format or audit.get("status") != "PASS"
                or success.get("status") != ("SUCCESS" if kind == "cnv" else "PASS")
                or audit.get(source_key) != manifest["native_receipt"]["sha256"]
                or success.get(audit_key) != declared["audit"]["sha256"]
                or run != value["native_run_id"] or audit.get("cohort") != context["native"].get("cohort")):
            raise ValueError("Independent audit belongs to a different auxiliary result")
        if kind == "cnv":
            sources.bind({"path":str(audit_path.parent/"cnv_oof_metrics_global.json"),
                          "sha256":audit["global_metrics_sha256"]}, root)
    return value


def collect(preparation, completed, output):
    path, binding, value = validate_preparation(preparation, ROOT); output = Path(output).resolve()
    if output.exists(): raise FileExistsError("Use a new auxiliary result collection directory")
    result_path = verified_path(completed, ROOT)
    output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json", {"target_host":"149", "training_started":False,
        "preparation":record(path), "completed":record(result_path)})
    try:
        from .native_auxiliary_fusion import INPUT_FORMAT, native_sources, materialize_native_auxiliary, load_native_auxiliary
        kind = value["kind"]; files = value["files"]
        manifest = {"format":INPUT_FORMAT, "producer_kind":kind, "modalities":[kind], "core_run_id":value["core_run_id"],
            "native_run_id":value["native_run_id"], "candidate_authority":files["candidate_authority"],
            "patient_folds":files["patient_folds"], "native_receipt":record(result_path), "workflow_preparation":record(path)}
        if kind == "cnv": manifest["core_manifest"] = files["core_manifest"]
        adapter_input = output/"NATIVE_AUXILIARY_INPUT.json"
        # Validate completion before running the auditor. The final adapter
        # manifest is published only after the actual independent audit exists.
        write(adapter_input, {k:v for k,v in manifest.items() if k != "workflow_preparation"})
        context = native_sources(adapter_input, expected_core_run_id=value["core_run_id"])
        validate_completed(manifest, context, output, require_audit=False)
        if kind == "cnv":
            args = {"root":result_path.parent, "candidates":files["candidate_authority"]["path"],
                "patient_folds":files["patient_folds"]["path"], "association_root":Path(files["association_success"]["path"]).parent,
                "expected_training_run_id":value["native_run_id"], "output_root":output/"independent_audit"}
            audit_name = "CNV_OOF_INDEPENDENT_AUDIT.json"
        else:
            args = {"training_success":result_path, "candidates":files["candidate_authority"]["path"],
                "output_root":output/"independent_audit", "expected_training_run_id":value["native_run_id"]}
            audit_name = "ATAC_FRESH_OOF_AUDIT.json"
        execute([sys.executable, "-m", MODULES[kind][2], *arguments(args)], output/"independent_audit.log")
        audit = output/"independent_audit"/audit_name
        if read(audit).get("status") != "PASS": raise ValueError("Independent auxiliary audit did not pass")
        manifest["workflow_audit"] = {"audit":record(audit), "success":record(audit.parent/"SUCCESS.json")}
        write(adapter_input, manifest)
        adapter = materialize_native_auxiliary(adapter_input, output/"fusion", expected_core_run_id=value["core_run_id"])
        _, info = load_native_auxiliary(adapter, expected_sha=record(adapter)["sha256"], expected_core_run_id=value["core_run_id"])
        for pin in context["sources"].records.values(): verified_path(pin, ROOT)
        verified_path(completed, ROOT); validate_preparation(preparation, ROOT)
        write(output/"NATIVE_AUXILIARY_BINDING.json", {"format":FORMAT, "status":"PASS", "target_host":"149",
            "training_started":False, "model_forwards_executed":False, "producer_kind":kind,
            "core_run_id":value["core_run_id"], "native_run_id":value["native_run_id"], "preparation":record(path),
            "completed":record(result_path), "adapter_input":record(adapter_input), "independent_audit":record(audit),
            "independent_success":record(audit.parent/"SUCCESS.json"), "fusion_binding":record(adapter),
            "fusion_experts":record(adapter.parent/"FUSION_EXPERTS.json"),
            "candidate_rows":info["rows"], "fold_rows":info["fold_rows"],
            "source_ancestry_complete":False, "formal_increment_established":False})
        return output/"NATIVE_AUXILIARY_BINDING.json"
    except Exception as error:
        write(output/"RUN_FAILED.json", {"target_host":"149", "training_started":False,
            "stage":"collect", "error_type":type(error).__name__, "error":str(error)})
        raise


def validate(item, root):
    require_host(); path = verified_path(item, root); value = read(path)
    if (value.get("format") != FORMAT or value.get("status") != "PASS" or value.get("target_host") != "149"
            or value.get("training_started") is not False or value.get("model_forwards_executed") is not False
            or value.get("source_ancestry_complete") is not False or value.get("formal_increment_established") is not False):
        raise ValueError("Invalid auxiliary result workflow binding")
    _, _, plan_value = validate_preparation(value["preparation"], path.parent)
    for key in ("completed", "adapter_input", "independent_audit", "independent_success", "fusion_binding", "fusion_experts"):
        verified_path(value[key], path.parent)
    from .native_auxiliary_fusion import load_native_auxiliary
    _, adapter = load_native_auxiliary(Path(value["fusion_binding"]["path"]),
        expected_sha=value["fusion_binding"]["sha256"], expected_core_run_id=plan_value["core_run_id"])
    manifest = read(value["adapter_input"]["path"])
    audit = read(value["independent_audit"]["path"])
    success = read(value["independent_success"]["path"])
    if (adapter["input_manifest"] != value["adapter_input"] or manifest["native_receipt"] != value["completed"]
            or manifest["workflow_preparation"] != value["preparation"] or audit.get("status") != "PASS"
            or manifest.get("workflow_audit") != {"audit":value["independent_audit"], "success":value["independent_success"]}
            or success.get("status") not in {"SUCCESS", "PASS"}
            or any(value[k] != plan_value[k] for k in ("core_run_id", "native_run_id"))
            or value["producer_kind"] != plan_value["kind"]
            or value["candidate_rows"] != adapter["rows"] or value["fold_rows"] != adapter["fold_rows"]):
        raise ValueError("Auxiliary collection bindings differ")
    key = "oof_manifest_sha256" if value["producer_kind"] == "cnv" else "training_success_sha256"
    audit_key = "independent_audit_sha256" if value["producer_kind"] == "cnv" else "audit_sha256"
    if audit.get(key) != value["completed"]["sha256"] or success.get(audit_key) != value["independent_audit"]["sha256"]:
        raise ValueError("Independent audit belongs to a different auxiliary result")
    experts = [{"spec":s, "prediction":adapter["outputs"]["fusion"], "binding_audit":value["fusion_binding"],
                "run_id":adapter["run_id"], "estimand":adapter["estimand"]} for s in adapter["expert_specs"]]
    if read(value["fusion_experts"]["path"]) != experts: raise ValueError("Auxiliary fusion expert declaration differs")
    return path, value
