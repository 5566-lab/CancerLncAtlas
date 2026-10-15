"""Run existing selected-analysis/context publishers in dependency order on 149.

Inputs are completed native producers, not training instructions. A workflow
can build a current publication or reuse a pinned release. It never infers a
completed upstream analysis from the mere presence of an input file.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import socket

import pyarrow.parquet as pq

from .current_core_export import sha256,verified_path
from .current_analysis_web import (COMPONENTS,STATUSES,NOTES,INPUT_FORMAT as WEB_INPUT,
    CurrentAnalysisWebStore,materialize_current_analysis_web)

INPUT_FORMAT="CURRENT_ANALYSIS_WORKFLOW_INPUT_V1"
FORMAT="CURRENT_ANALYSIS_WORKFLOW_RELEASE_V1"
ORDER=("evidence","selected","state","clinical","drug","single_cell","patient","genomic","distal","external","web")


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def record(path):return {"path":str(Path(path).resolve()),"sha256":sha256(path)}
def write(path,value):
    path=Path(path);temporary=path.with_name("."+path.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    temporary.replace(path)


def implementation_hashes():
    root=Path(__file__).resolve().parents[2]
    names=["cc_hhgt/v32/current_analysis_workflow.py","cc_hhgt/v32/current_analysis_web.py",
        "cc_hhgt/v32/selected_genesets.py","cc_hhgt/v32/selected_score.py",
        "cc_hhgt/v32/current_state_context.py","cc_hhgt/v32/current_clinical_context.py",
        "cc_hhgt/v32/current_drug_context.py","cc_hhgt/v32/current_single_cell_context.py",
        "cc_hhgt/v32/current_patient_context.py","cc_hhgt/v32/current_omics_context.py",
        "scripts/materialize_v32_current_rule_evidence.py","cc_hhgt/v32/evidence_rule_release.py",
        "cc_hhgt/v32/external_validation_current.py"]
    return {name:sha256(root/name) for name in names}


def validate_manifest(path):
    manifest=read(path)
    if (set(manifest)!={"format","release_id","selected","components"} or manifest.get("format")!=INPUT_FORMAT
            or not isinstance(manifest.get("release_id"),str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",manifest["release_id"])
            or set(manifest.get("components",{}))!=set(COMPONENTS)):
        raise ValueError("Workflow needs one selected analysis and all nine explicit component statuses")
    sources={}
    for name,spec in {"selected":manifest["selected"],**manifest["components"]}.items():
        status=spec.get("status")
        if status in STATUSES and name!="selected":
            if set(spec)!={"status","reason"} or not isinstance(spec["reason"],str) or not spec["reason"].strip():
                raise ValueError("Unavailable workflow component requires a reason and no hidden inputs")
        elif status in {"BUILD","REUSE"}:
            key="input_manifest" if status=="BUILD" else "binding"
            if set(spec)!={"status",key}:raise ValueError("Workflow stage has ambiguous inputs")
            source=verified_path(spec[key],path.parent);sources[name]=record(source)
        else:raise ValueError("Unknown workflow stage status: "+name)
    return manifest,sources


def resolve_template(value,root,references,sources):
    """Only named earlier-stage bindings may be substituted; all files are pinned."""
    if isinstance(value,dict):
        if "workflow_ref" in value:
            name=value.get("workflow_ref")
            if set(value)!={"workflow_ref"} or name not in references:
                raise ValueError("Unknown or unavailable earlier workflow reference: "+str(name))
            return dict(references[name])
        if "path" in value and "sha256" in value:
            path=verified_path(value,root);sources[str(path)]=record(path)
            return {**value,"path":str(path)}
        return {k:resolve_template(v,root,references,sources) for k,v in value.items()}
    if isinstance(value,list):return [resolve_template(v,root,references,sources) for v in value]
    return value


def build_stage(name,path,output):
    # Explicit existing entry points: arbitrary shell commands and trainer
    # callables are not accepted in a workflow input.
    if name=="selected":
        from .selected_genesets import materialize_selected_analysis as build
    elif name=="state":
        from .current_state_context import materialize_current_state_context as build
    elif name=="clinical":
        from .current_clinical_context import materialize_current_clinical_context as build
    elif name=="drug":
        from .current_drug_context import materialize_current_drug_context as build
    elif name=="single_cell":
        from .current_single_cell_context import materialize_current_single_cell_context as build
    elif name=="patient":
        from .current_patient_context import materialize_current_patient_context as build
    elif name in {"genomic","distal"}:
        from .current_omics_context import materialize_current_omics_context as build
        if read(path).get("kind")!=name:raise ValueError("Workflow omics kind differs from its stage")
    elif name=="evidence":
        from scripts.materialize_v32_current_rule_evidence import main
        if main(["--manifest",str(path),"--output",str(output)])!=0:
            raise RuntimeError("Native rule evidence command failed")
        return record(output/"RULE_EVIDENCE_BINDING.json")
    elif name=="external":
        from .external_validation_current import materialize_current_external as build
    else:raise ValueError("No current publisher for stage "+name)
    result=build(path,output)
    return record(Path(result["binding_path"]))


def public_table_catalog(store):
    """Actual public Arrow schema and table semantics, without private paths."""
    tables=[]
    for identity,table in store.binding["tables"].items():
        files=[verified_path(r,store.path.parent) for r in table["files"]]
        fields=[]
        if table["kind"]=="parquet":
            schema=pq.ParquetFile(files[0]).schema_arrow
            for path in files[1:]:
                if not schema.equals(pq.ParquetFile(path).schema_arrow,check_metadata=False):
                    raise ValueError("Workflow public table schemas differ across parts")
            fields=[{"name":field.name,"arrow_type":str(field.type),"nullable":field.nullable} for field in schema]
        tables.append({"table_id":identity,"component":table["component"],"kind":table["kind"],
            "rows":table["rows"],"table_semantics":NOTES[table["component"]],"fields":fields,
            "parts":[{"part":i,"sha256":r["sha256"]} for i,r in enumerate(table["files"])]})
    return {"format":"CURRENT_PUBLIC_TABLE_CATALOG_V1","web_binding_sha256":store.sha,
        "selected_analysis_sha256":store.binding["selected_analysis_sha256"],"tables":tables,
        "scope":"PUBLIC_SCHEMAS_AND_TABLE_SEMANTICS_NOT_COMPLETE_FIELD_DEFINITIONS",
        "independent_external_validation_attested":False}


def run_current_analysis_workflow(manifest_path,output):
    if socket.gethostname()!="149":raise RuntimeError("Current workflow requires target host 149")
    path=Path(manifest_path).resolve();manifest,declared=validate_manifest(path)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    inputs=output/"resolved_inputs";inputs.mkdir()
    sources={str(path):record(path),**{r["path"]:r for r in declared.values()}}
    codes=implementation_hashes()
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,
        "input_manifest":record(path),"implementation_sha256":codes})
    state={"format":"CURRENT_ANALYSIS_WORKFLOW_STATUS_V1","status":"RUNNING","target_host":"149",
        "training_started":False,"production_deployed":False,"stages":[]}
    refs={};components={};stage="preflight"
    try:
        for stage in ORDER[:-1]:
            spec=manifest["selected"] if stage=="selected" else manifest["components"][stage]
            if spec["status"] in STATUSES:
                components[stage]=dict(spec);state["stages"].append({"stage":stage,**spec})
            else:
                if spec["status"]=="BUILD":
                    source=verified_path(declared[stage],path.parent)
                    resolved=resolve_template(read(source),source.parent,refs,sources)
                    resolved_path=inputs/(stage+".json");write(resolved_path,resolved)
                    sources[str(resolved_path)]=record(resolved_path)
                    binding=build_stage(stage,resolved_path,output/stage)
                    item={"stage":stage,"status":"BUILT","resolved_input":record(resolved_path),"binding":binding}
                else:
                    binding=dict(declared[stage]);item={"stage":stage,"status":"REUSED","binding":binding}
                if stage=="selected":
                    from .selected_genesets import SelectedAnalysisQuery
                    selected=SelectedAnalysisQuery(binding["path"],expected_sha=binding["sha256"])
                    refs["selected_analysis"]=binding;refs["selected_score"]=record(selected.score_binding_path)
                else:
                    refs[stage]=binding;components[stage]={"status":"AVAILABLE","binding":binding}
                state["stages"].append(item)
            write(output/"STAGE_STATUS.json",state)
        stage="web"
        web_input=inputs/"web.json"
        write(web_input,{"format":WEB_INPUT,"release_id":manifest["release_id"],
            "selected_analysis":refs["selected_analysis"],"components":components})
        result=materialize_current_analysis_web(web_input,output/"web")
        web=record(Path(result["binding_path"]))
        store=CurrentAnalysisWebStore(web["path"],expected_sha=web["sha256"])
        catalog_path=output/"PUBLIC_TABLE_CATALOG.json";write(catalog_path,public_table_catalog(store))
        state["stages"].append({"stage":stage,"status":"BUILT","resolved_input":record(web_input),"binding":web})
        for source in sources.values():verified_path(source,path.parent)
        if codes!=implementation_hashes():raise ValueError("Workflow implementation changed during execution")
        binding={"format":FORMAT,"status":"PASS","target_host":"149","input_manifest":record(path),
            "release_id":manifest["release_id"],"selected_analysis":refs["selected_analysis"],"web_binding":web,
            "public_table_catalog":record(catalog_path),"implementation_sha256":codes,
            "source_inputs":list(sources.values()),"stages":state["stages"],"training_started":False,
            "upstream_training_executed":False,"production_deployed":False,
            "model_use_inventory_completeness_verified":False,"independent_external_validation_attested":False}
        target=output/"CURRENT_WORKFLOW_BINDING.json";write(target,binding)
        state["status"]="PASS";write(output/"STAGE_STATUS.json",state)
        result={"status":"PASS","target_host":"149","binding_path":str(target),"binding_sha256":sha256(target),
            "web_binding":web,"stage_status_sha256":sha256(output/"STAGE_STATUS.json"),"training_started":False}
        write(output/"SUCCESS.json",result);return result
    except Exception as error:
        state.update(status="FAILED",failed_stage=stage,error_type=type(error).__name__,error=str(error))
        write(output/"STAGE_STATUS.json",state)
        raise
