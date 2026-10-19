"""Consume completed native core runs through selection and current publication.

No trainer is called. Training configuration copies may change execution
controls and output locations only; scientific inputs stay bound to the
original blocked preparation. All six groups retain their five-fold sources.
"""
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from pathlib import Path
import math
import re

import yaml

from .cnv_input_sources import require_host
from .current_core_export import verified_path, validate_checkpoint, validate_training_receipt
from .formal_expression_sources import read, write, record
from .current_binding_graph import ARMS
from .training_guard import code_tree_sha256

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_CORE_RESULTS_INPUT_V1"
FORMAT = "NATIVE_CORE_RESULTS_RELEASE_V1"
GROUPS = ("G0", "G1", "G2", *ARMS)
RESULT_REFS = {"selected_model", "core_prediction", "core_run_id", "core_export", "primary_fusion"}


def pinned_template(value, root, sources):
    if isinstance(value, dict):
        if "result_ref" in value:
            if set(value) != {"result_ref"} or value["result_ref"] not in RESULT_REFS:
                raise ValueError("Unknown native result reference")
            return dict(value)
        if "path" in value and "sha256" in value:
            item = record(verified_path(value, root)); sources.append(item)
            return {**value, **item}
        return {k: pinned_template(v, root, sources) for k, v in value.items()}
    if isinstance(value, list): return [pinned_template(v, root, sources) for v in value]
    return value


def scientific_config(value):
    value = deepcopy(value)
    value.pop("execution_control", None)
    value.get("training_io", {}).pop("output_root", None)
    return value


def preflight(manifest, output):
    require_host()
    manifest = Path(manifest).resolve(strict=True); output = Path(output).resolve()
    spec = read(manifest)
    if (set(spec) != {"format", "release", "core_preparation", "runs", "geneset", "subtype", "publication_components"}
            or spec.get("format") != INPUT or set(spec["runs"]) != set(GROUPS)
            or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", spec.get("release", ""))):
        raise ValueError("Native results require six explicit completed runs and publication options")
    if output.exists(): raise FileExistsError("Use a new native results output")
    from .native_core_workflow import validate as validate_preparation
    preparation, binding = validate_preparation(spec["core_preparation"], manifest.parent)
    core = read(binding["preflight"]["path"])
    from .selected_genesets import SelectedGeneSetConfig
    from .subtypes import RankedSubtypeConfig
    SelectedGeneSetConfig(**spec["geneset"]).validate()
    if "min_expression_folds" not in spec["geneset"]: raise ValueError("Declare GeneSet expression-fold eligibility")
    RankedSubtypeConfig(**spec["subtype"]).validate()
    from .current_analysis_web import COMPONENTS
    if set(spec["publication_components"]) != set(COMPONENTS):
        raise ValueError("Declare the status of every publication component")
    # This plan verifies completed checkpoints and their actual input loaders;
    # it never constructs a model or runs a scientific producer.
    import torch
    from .orchestration import read_task_manifest, find_task
    from scripts.export_v32_current_core_embeddings import _validated_payload
    from .training import _runtime_chunk_contract
    commands = read(binding["training_commands"]["path"])["commands"]
    sources = [record(manifest), record(preparation)]; runs = {}; inputs = {}; losses = {}
    from .current_analysis_web import STATUSES
    components = {}; templates = {}
    for name, component in spec["publication_components"].items():
        status = component.get("status")
        if status in STATUSES:
            if set(component) != {"status", "reason"} or not isinstance(component["reason"], str) or not component["reason"].strip():
                raise ValueError("Unavailable publication component requires an explicit reason")
            components[name] = dict(component)
        elif status in {"BUILD", "REUSE"}:
            key = "input_manifest" if status == "BUILD" else "binding"
            if set(component) != {"status", key}: raise ValueError("Ambiguous publication component")
            source = verified_path(component[key], manifest.parent); sources.append(record(source))
            components[name] = {"status": status, key: record(source)}
            if status == "BUILD": templates[name] = pinned_template(read(source), source.parent, sources)
        else: raise ValueError("Unknown publication component status")
    code = code_tree_sha256(ROOT)
    for group in GROUPS:
        declaration = spec["runs"][group]
        if set(declaration) != {"config", "input_manifest", "task_manifest", "folds"}:
            raise ValueError("Completed run fields differ")
        paths = {k: verified_path(declaration[k], manifest.parent) for k in ("config", "input_manifest", "task_manifest")}
        registered = {k: record(p) for k, p in paths.items()}; sources.extend(registered.values())
        folds = declaration["folds"]
        if (len(folds) != 5 or any(type(r.get("fold")) is not int for r in folds)
                or {r["fold"] for r in folds} != set(range(5))):
            raise ValueError("Completed run requires all five integer folds")
        cfg = yaml.safe_load(paths["config"].read_text())
        if cfg.get("execution_control", {}).get("execution_mode") != "TRAINING" or cfg["execution_control"].get("training_authorized") is not True:
            raise ValueError("Completed run does not reference its training configuration")
        tasks = read_task_manifest(paths["task_manifest"])
        authorized = read(paths["input_manifest"])
        rows = []; completed = []; values = []; run_ids = set()
        for fold_item in sorted(folds, key=lambda r: r["fold"]):
            if set(fold_item) != {"fold", "checkpoint", "training_receipt"}:
                raise ValueError("Completed fold fields differ")
            fold = fold_item["fold"]
            command = next(c for c in commands if c.get("arm", c.get("graph_variant")) == group and c["fold"] == fold)
            args = command["argv"]; arg = lambda name: args[args.index(name)+1]
            base = yaml.safe_load(Path(arg("--config")).read_text())
            if scientific_config(cfg) != scientific_config(base):
                raise ValueError("Completed run changed the prepared scientific configuration")
            if authorized != read(arg("--input-manifest")):
                raise ValueError("Completed run changed prepared inputs or binding overlays")
            task = find_task(tasks, arg("--task-id"))
            if (len(tasks) != 5 or task["status"] != "PENDING" or task["run_id"] != arg("--run-id")
                    or int(task["patient_fold"]) != fold or int(task["seed"]) != 20260726
                    or task["owner"] != "server149" or task["hardware_class"] != "SERVER149_CPU"
                    or task["task_type"] != "CC_HHGT_PATIENT_FOLD"):
                raise ValueError("Completed training task differs from its registered plan")
            native = next(r for r in authorized["fold_inputs"] if r["fold"] == fold)
            row = {"fold": fold, "config": registered["config"], "input_manifest": registered["input_manifest"],
                   "prepared": record(verified_path(native, paths["input_manifest"].parent))}
            for name in ("checkpoint", "training_receipt"):
                row[name] = record(verified_path(fold_item[name], manifest.parent))
            if "binding_expansion" in native:
                row["binding_expansion"] = record(verified_path(native["binding_expansion"], paths["input_manifest"].parent))
            success = read(row["training_receipt"]["path"])
            variant = group if group.startswith("G") else "G2"
            validate_training_receipt(success, run_id=arg("--run-id"), fold=fold,
                                      checkpoint_sha=row["checkpoint"]["sha256"], variant=variant)
            hashes = {"code_sha256": code, **{k+"_sha256": r["sha256"] for k, r in registered.items()}}
            if (success.get("artifact_hashes") != hashes or success["task_id"] != task["task_id"]
                    or success.get("test_labels_read") is not False or success.get("test_metrics_computed") is not False
                    or success.get("architecture_id") != "HHGT_FORMAL_CORE_EXTERNAL_ROUTER"
                    or success.get("evidence_integration_mode") != "external_router"):
                raise ValueError("Completed training receipt differs from the registered code, task or sources")
            loss = float(success.get("best_validation_loss", math.nan))
            if not math.isfinite(loss) or loss < 0: raise ValueError("Completed run has invalid validation loss")
            checkpoint = torch.load(row["checkpoint"]["path"], map_location="cpu", weights_only=False)
            validate_checkpoint(checkpoint, success, fold=fold, prepared_sha=row["prepared"]["sha256"],
                                config_sha=row["config"]["sha256"], input_manifest_sha=row["input_manifest"]["sha256"])
            payload = _validated_payload({k: Path(v["path"]) for k, v in row.items() if isinstance(v, dict)}, row, fold)
            chunks, weights = _runtime_chunk_contract(payload["bundle"])
            if success["runtime_num_chunks"] != len(chunks) or success["runtime_chunk_weights"] != {str(c): weights[c] for c in chunks}:
                raise ValueError("Completed run changed the runtime chunk estimator")
            sources.extend(v for v in row.values() if isinstance(v, dict))
            rows.append(row); values.append(loss); run_ids.add(success["run_id"])
            completed.append({"fold": fold, **{k: row[k] for k in ("checkpoint", "training_receipt")}})
        if len(run_ids) != 1: raise ValueError("Completed folds belong to different runs")
        runs[group] = {**registered, "folds": completed}
        inputs[group] = {"format": "CURRENT_CORE_PREDICTION_INPUT_V1", "core_run_id": run_ids.pop(),
            "graph_variant": group if group.startswith("G") else "G2", "candidate_authority": core["files"]["candidates"],
            "fold_local_scope": core["files"]["fold_local_scope"], "patient_map": core["typed_files"]["patient_map"],
            "pathway_hierarchy": record(verified_path(read(binding["typed_graph"]["path"])["artifacts"]["pathway_hierarchy"], preparation.parent)),
            "fold_inputs": rows}
        losses[group] = values
    return {"target_host": "149", "training_started": False, "input": record(manifest), "code_sha256": code,
        "core_preparation": record(preparation), "prepared_parent": str(preparation.parent/"prepared"),
        "candidate_authority": core["files"]["candidates"], "runs": runs, "prediction_inputs": inputs,
        "validation_losses": losses, "release": spec["release"], "geneset": spec["geneset"], "subtype": spec["subtype"],
        "publication_components": components, "publication_templates": templates,
        "sources": list({r["path"]: r for r in sources}.values())}


def result_references(value, refs):
    if isinstance(value, dict):
        if "result_ref" in value:
            if set(value) != {"result_ref"} or value["result_ref"] not in refs:
                raise ValueError("Unknown native result reference")
            return deepcopy(refs[value["result_ref"]])
        return {k: result_references(v, refs) for k, v in value.items()}
    if isinstance(value, list): return [result_references(v, refs) for v in value]
    return value


def run(manifest, output):
    plan = preflight(manifest, output); output = Path(output).resolve(); output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json", plan)
    state = {"target_host": "149", "training_started": False, "status": "RUNNING", "stages": []}
    write(output/"STAGE_STATUS.json", state)
    def stage(name, operation):
        entry = {"stage": name, "status": "RUNNING"}; state["stages"].append(entry); write(output/"STAGE_STATUS.json", state)
        log = output/(name+".log")
        try:
            with log.open("w", encoding="utf-8") as handle, redirect_stdout(handle), redirect_stderr(handle):
                result = operation()
            entry.update(status="PASS", log=record(log)); write(output/"STAGE_STATUS.json", state)
            return result
        except Exception as error:
            entry.update(status="FAILED", log=record(log)); state.update(status="FAILED", failed_stage=name, error=str(error))
            write(output/"STAGE_STATUS.json", state); raise
    from .g012_winner_lock import lock_g012_validation_winner
    lock = stage("g012_selection", lambda: lock_g012_validation_winner(repo_root=ROOT,
        prepared_parent=plan["prepared_parent"], run_parent=None, candidate_authority_path=plan["candidate_authority"]["path"],
        output_root=output/"selection", pair_fold_seed=20260826,
        run_declarations={k: plan["runs"][k] for k in GROUPS[:3]}))
    winner = lock["winner"]; predictions = {}; manifests = {}
    from .current_core_prediction import materialize_current_predictions
    for group in GROUPS:
        source = output/(group+"_prediction_input.json"); write(source, plan["prediction_inputs"][group]); manifests[group] = source
        predictions[group] = record(stage("prediction_"+group, lambda g=group, p=source:
            materialize_current_predictions(p, output/("prediction_"+g), expected_run_id=plan["prediction_inputs"][g]["core_run_id"], device="cpu")))
    comparison = {"format": "NATIVE_BINDING_VALIDATION_COMPARISON_V1", "selection_metric": "MEAN_FIVE_FOLD_BEST_VALIDATION_LOSS",
        "rows": [{"arm": a, "fold_validation_losses": plan["validation_losses"][a],
            "mean_validation_loss": sum(plan["validation_losses"][a])/5} for a in ARMS],
        "training_sources": {a: plan["runs"][a] for a in ARMS}, "predictions": {a: predictions[a] for a in ARMS},
        "independent_validation_claimed": False, "changes_g012_winner": False}
    write(output/"BINDING_COMPARISON.json", comparison)
    run_id = plan["prediction_inputs"][winner]["core_run_id"]
    export_input = output/"export_input.json"; write(export_input, {**plan["prediction_inputs"][winner], "format": "CURRENT_CORE_EXPORT_INPUT_V1"})
    from scripts.export_v32_current_core_embeddings import main as export
    def export_core():
        if export(["--manifest", str(export_input), "--expected-core-run-id", run_id,
                   "--output", str(output/"core_export"), "--device", "cpu"]) != 0:
            raise RuntimeError("Native core export command failed")
    stage("selected_core_export", export_core)
    from .current_primary_fusion import materialize_primary_fusion_inputs
    primary = record(stage("primary_fusion_inputs", lambda: materialize_primary_fusion_inputs(predictions[winner]["path"], output/"primary_fusion",
        expected_sha=predictions[winner]["sha256"], expected_core_run_id=run_id)))
    model = {"model_id": "native-primary", "producer_kind": "current_core_release", "producer_binding": predictions[winner],
             "run_id": run_id, "score_column": "association_membership_probability"}
    selected_input = output/"selected_input.json"
    write(selected_input, {"format": "CURRENT_SELECTED_ANALYSIS_INPUT_V1", "selected_model": model, "reference_exact_model": model,
        "selection": {"kind": "G012_VALIDATION_WINNER", "binding": lock["winner_declaration"]},
        "geneset": plan["geneset"], "subtype": plan["subtype"]})
    refs = {"selected_model": model, "core_prediction": predictions[winner], "core_run_id": run_id,
        "core_export": record(output/"core_export/CORE_EMBEDDING_MANIFEST.json"), "primary_fusion": primary}
    publication = output/"publication_input.json"
    components = deepcopy(plan["publication_components"])
    for name, template in plan["publication_templates"].items():
        path = output/(name+"_publication_input.json"); write(path, result_references(template, refs))
        components[name] = {"status": "BUILD", "input_manifest": record(path)}
    write(publication, {"format": "CURRENT_ANALYSIS_WORKFLOW_INPUT_V1", "release_id": plan["release"],
        "selected": {"status": "BUILD", "input_manifest": record(selected_input)},
        "components": components})
    from .current_analysis_workflow import run_current_analysis_workflow
    published = stage("current_publication", lambda: run_current_analysis_workflow(publication, output/"publication"))
    def close():
        for item in plan["sources"]: verified_path(item, output)
        if code_tree_sha256(ROOT) != plan["code_sha256"]: raise ValueError("Results workflow code changed")
        from .native_core_workflow import validate as validate_preparation
        validate_preparation(plan["core_preparation"], output)
    stage("source_recheck", close)
    state["status"] = "PASS"; write(output/"STAGE_STATUS.json", state)
    binding = {"format": FORMAT, "status": "PASS", "target_host": "149", "training_started": False,
        "preflight": record(output/"LAUNCH_PREFLIGHT.json"), "stage_status": record(output/"STAGE_STATUS.json"),
        "selection": lock["winner_declaration"], "winner": winner, "predictions": predictions,
        "binding_comparison": record(output/"BINDING_COMPARISON.json"), "core_export": refs["core_export"],
        "primary_fusion": primary, "publication": record(published["binding_path"]), "production_deployed": False,
        "auxiliary_training_started": False, "fusion_fitting_started": False, "independent_validation_claimed": False}
    path = output/"NATIVE_RESULTS_BINDING.json"; write(path, binding)
    write(output/"SUCCESS.json", {"status": "PASS", "target_host": "149", "training_started": False, "binding": record(path)})
    return path


def validate(item, root):
    require_host(); path = verified_path(item, root); binding = read(path); output = path.parent
    if (binding.get("format") != FORMAT or binding.get("status") != "PASS" or binding.get("target_host") != "149"
            or any(binding.get(k) is not False for k in ("training_started", "auxiliary_training_started", "fusion_fitting_started", "production_deployed", "independent_validation_claimed"))
            or read(output/"SUCCESS.json") != {"status": "PASS", "target_host": "149", "training_started": False, "binding": record(path)}):
        raise ValueError("Incomplete native results workflow")
    plan = read(verified_path(binding["preflight"], output)); state = read(verified_path(binding["stage_status"], output))
    expected_stages = ["g012_selection", *["prediction_"+g for g in GROUPS], "selected_core_export", "primary_fusion_inputs", "current_publication", "source_recheck"]
    if (state["status"] != "PASS" or any(s["status"] != "PASS" for s in state["stages"])
            or [s["stage"] for s in state["stages"]] != expected_stages or set(binding["predictions"]) != set(GROUPS)):
        raise ValueError("Native results stages are incomplete")
    for stage in state["stages"]: verified_path(stage["log"], output)
    for source in plan["sources"]: verified_path(source, output)
    if code_tree_sha256(ROOT) != plan["code_sha256"]: raise ValueError("Results workflow implementation changed")
    from .native_core_workflow import validate as validate_preparation
    validate_preparation(plan["core_preparation"], output)
    from .g012_winner_lock import validate_comparison_sources
    selection = verified_path(binding["selection"], output); validate_comparison_sources(read(selection), selection.parent)
    if binding["winner"] != read(selection)["winner_id"]: raise ValueError("Native result winner differs")
    from .current_core_prediction import audit_current_predictions
    for source in binding["predictions"].values(): audit_current_predictions(source["path"], expected_sha=source["sha256"])
    comparison = read(verified_path(binding["binding_comparison"], output))
    expected_rows = [{"arm": a, "fold_validation_losses": [read(r["training_receipt"]["path"])["best_validation_loss"] for r in plan["runs"][a]["folds"]]} for a in ARMS]
    for row in expected_rows: row["mean_validation_loss"] = sum(row["fold_validation_losses"])/5
    if (comparison["rows"] != expected_rows or comparison["training_sources"] != {a: plan["runs"][a] for a in ARMS}
            or comparison["predictions"] != {a: binding["predictions"][a] for a in ARMS}
            or comparison["changes_g012_winner"] is not False or comparison["independent_validation_claimed"] is not False):
        raise ValueError("Binding comparison differs from actual completed runs")
    from .current_core_export import validate_current_export
    export = verified_path(binding["core_export"], output); export_value, _, _ = validate_current_export(export)
    for fold in export_value["folds"].values():
        for source in fold["exports"].values(): verified_path(source, export.parent)
    from .current_primary_fusion import load_primary_fusion_inputs
    source = binding["primary_fusion"]
    load_primary_fusion_inputs(source["path"], expected_sha=source["sha256"], expected_core_run_id=export_value["core_run_id"])
    publication = read(verified_path(binding["publication"], output))
    from .current_analysis_web import CurrentAnalysisWebStore
    CurrentAnalysisWebStore(publication["web_binding"]["path"], expected_sha=publication["web_binding"]["sha256"])
    return path, binding
