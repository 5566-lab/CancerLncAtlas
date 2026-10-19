"""Cached GDC STAR -> five-fold native expression/activity/candidate inputs.

This bounded workflow calls existing scientific producers. It ends before CNV
association, model fitting and publication; a successful receipt attests only
to the declared preprocessing outputs and their actual native-reader checks.
"""
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, write, record
from . import gdc_star_counts as star
from . import fold_local_expression as expression
from . import fold_local_activity as activity
from . import fold_local_scope as scope

INPUT = "NATIVE_EXPRESSION_WORKFLOW_INPUT_V1"
FORMAT = "NATIVE_EXPRESSION_WORKFLOW_RELEASE_V1"
ROOT = Path(__file__).resolve().parents[2]
CODE = ["cc_hhgt/v32/native_expression_workflow.py", "scripts/run_native_expression_workflow.py",
        "cc_hhgt/v32/gdc_star_counts.py", "scripts/materialize_gdc_star_counts.py", "scripts/assemble_gdc_star_counts.R",
        "scripts/materialize_fold_local_expression.R", "scripts/frozen_tmm.R",
        "scripts/materialize_fold_local_ssgsea.R", "scripts/frozen_ssgsea.R",
        "cc_hhgt/v32/fold_local_expression.py", "scripts/convert_fold_local_expression.py",
        "cc_hhgt/v32/fold_local_activity.py", "scripts/convert_fold_local_activity.py",
        "cc_hhgt/v32/fold_local_scope.py", "scripts/materialize_fold_local_scope.py", "scripts/prepare_v32_formal.py"]


def preflight(manifest, output):
    require_host(); manifest = Path(manifest).resolve(strict=True); output = Path(output).resolve()
    spec = read(manifest)
    if (set(spec) != {"format", "cancers", "gdc_query", "download_root", "patient_map", "annotation", "gmt_manifest", "budget_per_cancer", "seed"}
            or spec.get("format") != INPUT or type(spec["seed"]) is not int or spec["seed"] < 0):
        raise ValueError("Invalid native expression workflow declaration")
    if output.exists(): raise FileExistsError("Use a new native expression workflow output")
    files = {key: record(verified_path(spec[key], manifest.parent)) for key in ("gdc_query", "patient_map", "annotation", "gmt_manifest")}
    query = read(files["gdc_query"]["path"])
    rows = star.replay_pages(query, Path(files["gdc_query"]["path"]).parent)
    if spec["cancers"] != query["cancers"] or query.get("files") != rows:
        raise ValueError("Workflow cancer/file scope differs from the archived GDC query")
    cache = Path(spec["download_root"])
    if not cache.is_absolute(): cache = manifest.parent / cache
    cache = cache.resolve(strict=True)
    raw = [star.acquire(item, cache) for item in rows]  # Cache only: never downloads.
    patients = pd.read_csv(files["patient_map"]["path"], sep="\t", dtype={"patient_id":str, "cancer_id":str})
    if (not {"cancer_id", "patient_id", "patient_fold_id"} <= set(patients)
            or patients[["cancer_id", "patient_id", "patient_fold_id"]].isna().any().any()
            or set(patients.cancer_id) != set(spec["cancers"])
            or patients.patient_id.duplicated().any()
            or not patients.patient_id.str.fullmatch(r"TCGA-[A-Za-z0-9]{2}-[A-Za-z0-9]{4}").all()
            or not patients.patient_fold_id.isin(range(5)).all()):
        raise ValueError("Invalid workflow patient fold map")
    for cancer in spec["cancers"]:
        subset = patients.loc[patients.cancer_id.eq(cancer)]
        available = {r["patient_id"] for r in rows if r["cancer_id"] == cancer and r["sample_id"][13:15] == "01"}
        if set(subset.patient_fold_id) != set(range(5)) or not set(subset.patient_id) <= available:
            raise ValueError("Every declared patient needs a primary tumor and all five folds")
    gmt_path = Path(files["gmt_manifest"]["path"])
    gmt = read(gmt_path)
    if any(not Path(entry["file"]["path"]).is_absolute() for entry in gmt.get("collections", [])):
        raise ValueError("Native GMT collection files require absolute pinned paths")
    # Validate annotation/GMT membership and budget before doing any assembly.
    scope.candidate_frame({**spec, **files}, manifest.parent)
    gmt_sources = [record(verified_path(entry["file"], gmt_path.parent)) for entry in gmt["collections"]]
    # Preserve the original manifest, and write the resolved copy only when
    # the workflow actually runs.
    resolved_gmt = {**gmt, "collections": [{**entry, "file": source} for entry, source in zip(gmt["collections"], gmt_sources)]}
    commands = []
    def command(name, argv): commands.append({"stage":name, "argv":list(map(str, argv))})
    normal = [output / "normalization" / c for c in spec["cancers"]]
    activities = [output / "activity" / c for c in spec["cancers"]]
    for cancer, norm, act in zip(spec["cancers"], normal, activities):
        counts = output / "counts" / cancer
        command("counts:"+cancer, [sys.executable, "-m", "scripts.materialize_gdc_star_counts", "materialize",
            "--manifest", files["gdc_query"]["path"], "--cancer", cancer, "--download-root", cache, "--output", counts])
        command("normalization:"+cancer, ["/usr/bin/Rscript", ROOT/"scripts/materialize_fold_local_expression.R",
            counts/('TCGA-'+cancer+'.RDS'), files["patient_map"]["path"], cancer, norm])
        command("activity:"+cancer, ["/usr/bin/Rscript", ROOT/"scripts/materialize_fold_local_ssgsea.R",
            norm, files["annotation"]["path"], output/"resolved_gmt.json", act])
    command("expression_conversion", [sys.executable, ROOT/"scripts/convert_fold_local_expression.py",
        *[arg for root in normal for arg in ("--source", root)], "--output", output/"expression_tables"])
    command("activity_conversion", [sys.executable, ROOT/"scripts/convert_fold_local_activity.py",
        *[arg for root in activities for arg in ("--source", root)], "--output", output/"activity_tables"])
    command("scope", [sys.executable, ROOT/"scripts/materialize_fold_local_scope.py", "--manifest", output/"resolved_scope.json", "--output", output/"scope"])
    sources = [record(manifest), *files.values(), *query["pages"], *raw, *gmt_sources]
    # Normalize page paths before saving the preflight outside the query tree.
    sources = [record(verified_path(item, Path(files["gdc_query"]["path"]).parent)) for item in sources]
    return {"target_host":"149", "operation":"NATIVE_EXPRESSION_PREPARATION", "training_started":False,
        "network_acquisition_enabled":False, "input":record(manifest), "cancers":spec["cancers"], "files":files,
        "sources":sources, "implementations":[record(ROOT/name) for name in CODE], "commands":commands,
        "resolved_gmt":resolved_gmt, "budget_per_cancer":spec["budget_per_cancer"], "seed":spec["seed"]}


def outputs(root, cancers, patient_map):
    """Use the same readers as association/graph preparation, including scope."""
    root = Path(root)
    source_config = {**read(root/"expression_tables/SOURCE_CONFIG_FRAGMENT.json")["sources"],
                     **read(root/"activity_tables/SOURCE_CONFIG_FRAGMENT.json")["sources"]}
    partitions = []
    for cancer in cancers:
        for fold in range(5):
            for key in ("lnc_expression_root", "gene_expression_root"):
                path = expression.expression_path(source_config, key, cancer, fold, patient_map)
                partitions.append(record(path.parent/"EXPRESSION_SOURCES.json"))
            path = activity.activity_path(source_config, cancer, fold, patient_map)
            partitions.append(record(path.parent/"ACTIVITY_SOURCES.json"))
    candidate = root/"scope/candidates.parquet"; scope_item = record(root/"scope/SCOPE_RELEASE.json")
    scope.release(scope_item, root, candidates=candidate, patient_map=patient_map)
    return {"sources":source_config, "candidate_definition":scope.candidate_binding(candidate),
        "candidates":record(candidate), "fold_local_scope":scope_item, "patient_map":record(patient_map),
        "partitions":partitions}


def run(manifest, output):
    plan = preflight(manifest, output); output = Path(output).resolve(); output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json", plan)
    write(output/"resolved_gmt.json", plan["resolved_gmt"])
    status = {"target_host":"149", "training_started":False, "status":"RUNNING", "stages":[]}
    write(output/"STAGE_STATUS.json", status)
    stage = "initialization"
    try:
        for item in plan["commands"]:
            stage = item["stage"]
            if stage == "scope":
                write(output/"resolved_scope.json", {"format":scope.INPUT, "annotation":plan["files"]["annotation"],
                    "gmt_manifest":record(output/"resolved_gmt.json"), "budget_per_cancer":plan["budget_per_cancer"], "seed":plan["seed"],
                    "normalizations":{c:record(output/"normalization"/c/"SUCCESS.json") for c in plan["cancers"]}})
            state = {**item, "status":"RUNNING"}; status["stages"].append(state); write(output/"STAGE_STATUS.json", status)
            result = subprocess.run(item["argv"], cwd=ROOT, capture_output=True, text=True, timeout=3600)
            log = output/(stage.replace(":", "_")+".log"); log.write_text(result.stdout+result.stderr, encoding="utf-8")
            state.update(exit_code=result.returncode, log=record(log))
            if result.returncode: raise ValueError("Native preprocessing stage failed: "+stage+": "+result.stderr[-2000:])
            state["status"] = "PASS"; write(output/"STAGE_STATUS.json", status)
        stage = "native_reader_acceptance"
        value = outputs(output, plan["cancers"], plan["files"]["patient_map"]["path"])
        fragment = {k:value[k] for k in ("sources", "candidate_definition")}
        write(output/"SOURCE_CONFIG_FRAGMENT.json", fragment)
        for item in [*plan["sources"], *plan["implementations"]]: verified_path(item, output)
        status["status"] = "PASS"; write(output/"STAGE_STATUS.json", status)
        binding = {"format":FORMAT, "target_host":"149", "status":"PASS", "training_started":False,
            "association_analysis_executed":False, "production_deployed":False,
            "preflight":record(output/"LAUNCH_PREFLIGHT.json"), "stage_status":record(output/"STAGE_STATUS.json"),
            "source_config_fragment":record(output/"SOURCE_CONFIG_FRAGMENT.json"), "outputs":value}
        path = output/"NATIVE_EXPRESSION_BINDING.json"; write(path, binding)
        write(output/"SUCCESS.json", {"status":"PASS", "target_host":"149", "training_started":False, "binding":record(path)})
        return path
    except Exception as error:
        if status["stages"] and status["stages"][-1]["status"] == "RUNNING": status["stages"][-1]["status"] = "FAILED"
        status.update(status="FAILED", failed_stage=stage, error_type=type(error).__name__, error=str(error))
        write(output/"STAGE_STATUS.json", status)
        raise


def validate(item, root):
    require_host(); path = verified_path(item, root); binding = read(path)
    success = read(path.parent/"SUCCESS.json")
    if (binding.get("format") != FORMAT or binding.get("status") != "PASS" or binding.get("target_host") != "149"
            or binding.get("training_started") is not False or binding.get("association_analysis_executed") is not False
            or success != {"status":"PASS", "target_host":"149", "training_started":False, "binding":record(path)}):
        raise ValueError("Incomplete native expression workflow")
    plan = read(verified_path(binding["preflight"], path.parent))
    status = read(verified_path(binding["stage_status"], path.parent))
    if (status.get("status") != "PASS" or len(status.get("stages", [])) != len(plan["commands"])
            or any(s.get("status") != "PASS" or s.get("exit_code") != 0 or s["stage"] != c["stage"] or s["argv"] != c["argv"]
                   for s, c in zip(status["stages"], plan["commands"]))):
        raise ValueError("Native expression workflow stage closure differs")
    for source in [*plan["sources"], *plan["implementations"], *[s["log"] for s in status["stages"]]]: verified_path(source, path.parent)
    actual = outputs(path.parent, plan["cancers"], plan["files"]["patient_map"]["path"])
    fragment = read(verified_path(binding["source_config_fragment"], path.parent))
    if binding["outputs"] != actual or fragment != {k:actual[k] for k in ("sources", "candidate_definition")}:
        raise ValueError("Native expression workflow outputs differ")
    return path, binding
