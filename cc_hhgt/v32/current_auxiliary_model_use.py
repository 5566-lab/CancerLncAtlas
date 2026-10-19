"""Trace native auxiliary producers consumed by an actual fusion snapshot.

Source metadata describes exposure to whole registered input files. It does not
assert every row was fitted, identify a patient, or attest raw-study completeness.
Native prediction aggregation is replayed; no auxiliary optimizer/forward runs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .current_core_export import sha256
from .evidence_rule_summary import KEYS, candidate_aliases, stable_id


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def input_source(inventory,item,root,model,role,fold=-1,*,phase="FEATURE_CONSTRUCTION"):
    """Follow only hash-bound records from the actual native producer lineage."""
    digest=item.get("sha256")
    if not digest:
        inventory.gap(model,role,"AUTHORITY_HAS_NO_FILE_HASH",fold);return
    source=inventory.locations.get(digest,item)
    if digest in inventory.locations:inventory.used_locations.add(digest)
    path=Path(source.get("path",""));path=path if path.is_absolute() else root/path
    if not source.get("path") or not path.exists():
        inventory.authority(source,root,model,role,fold);return
    path=inventory.bind(source,root,model,role,fold)
    if item.get("upstream_expression"):
        from .formal_expression_sources import inspect
        inspect(inventory,item["upstream_expression"],root,path,model,role+".expression",fold,phase=phase)
    # Controls are dependencies, not biological study evidence. Never infer
    # provenance from a filename, a directory name or patient identifiers.
    if item.get("source_role") in {"split_manifest","v32_core_checkpoint"} or item.get("role") in {"folds","materialization_success"}:
        return
    suffixes=set(path.suffixes)
    is_table=bool(suffixes & {".parquet",".tsv",".csv",".maf"}) or item.get("source_kind")=="table"
    if path.is_dir():
        files=[p for p in path.rglob("*") if p.is_file()]
        is_table=bool(files) and all(p.suffix==".parquet" for p in files)
    if not is_table:
        inventory.gap(model,role,"AUXILIARY_NON_TABLE_UPSTREAM_CONTENT_NOT_RECONSTRUCTED",fold,digest);return
    from .preparation_sources import source_members,canonical_sha
    for member in source_members(path,"table"):
        inventory.bind(member,path.parent,model,role+".consumed_file",fold)
        inventory.patient_use.table(member["path"],digest=member["sha256"],model=model,context=inventory.context_id,
            role=role,fold=fold,phase=phase,scope="WHOLE_REGISTERED_NATIVE_INPUT_NOT_ALL_ROWS_FITTED")
        tokens=member["study_tokens"]
        if not tokens:inventory.gap(model,role,"SOURCE_STUDY_METADATA_ABSENT",fold,member["sha256"])
        identity=stable_id("",[model,fold,role,member["sha256"],inventory.context_id,phase])
        inventory.usage.append({"cancer_id":"","lncrna_id":"","pathway_id":"",
            "source_id":"AUXILIARY-SOURCE:"+identity,"source_event_id":identity,
            "source_file_sha256":member["sha256"],"phase":phase,"endpoint":"MODEL_USE",
            "label":np.nan,"is_experimental":False,"is_predicted":False,"mapped":True,
            "assay":"REGISTERED_NATIVE_INPUT_METADATA","biosample":"","study_tokens_json":json.dumps(tokens),
            "raw_record_id":identity,"raw_row_sha256":canonical_sha(member),"raw_identifiers_json":"{}",
            "model_id":model,"patient_fold_id":fold,"use_view":"AUXILIARY_SOURCE_METADATA",
            "candidate_rows":0,"entity_scope":"WHOLE_REGISTERED_FILE_NO_ENTITY_ASSERTION",
            "dependency_context":inventory.context_id})
    inventory.gap(model,role,"RAW_UPSTREAM_TRANSFORMATIONS_NOT_RECONSTRUCTED",fold,digest)


def inspect_fusion_expert(inventory,path,binding,audit,frame,expert,model,endpoint):
    role="fusion.expert."+expert
    declaration=audit.get("expert_bindings",{}).get(expert)
    if declaration is None:
        inventory.gap(model,role,"AUXILIARY_RAW_TRAINING_DEPENDENCIES_NOT_RECONSTRUCTED");return
    from .multimodal_fusion import ExpertSpec
    from dataclasses import asdict
    spec=ExpertSpec(**declaration["spec"]);spec.validate()
    if spec.expert_id!=expert or endpoint not in spec.endpoint_roles:
        raise ValueError("Fusion expert binding endpoint or identity differs")
    source_hashes={r["sha256"] for r in binding["source_inputs"]}
    for name in ("binding_audit","prediction"):
        if declaration[name]["sha256"] not in source_hashes:
            raise ValueError("Fusion expert binding is absent from actual consumed source declarations")
    audit_path=inventory.bind(declaration["binding_audit"],path.parent,model,role+".binding")
    prediction=inventory.bind(declaration["prediction"],path.parent,model,role+".prediction")
    native=read(audit_path)
    if native.get("run_id")!=declaration["run_id"] or native.get("estimand")!=declaration["estimand"]:
        raise ValueError("Fusion expert run or estimand differs from native producer")
    if json.loads(json.dumps(asdict(spec))) not in native.get("expert_specs",[]):
        raise ValueError("Fusion expert specification differs from native producer")
    def compare_snapshot(observed):
        left=frame[KEYS+[expert+"_probability",expert+"_available"]].sort_values(KEYS).reset_index(drop=True)
        right=observed[KEYS+[spec.probability_column,spec.availability_column]].sort_values(KEYS).reset_index(drop=True)
        right.columns=left.columns
        try:pd.testing.assert_frame_equal(left,right,check_dtype=False,rtol=0,atol=1e-12)
        except AssertionError as error:raise ValueError("Fusion snapshot differs from the declared native expert input") from error
    from .single_cell_signed_fusion import FORMAT as SIGNED_FORMAT,load_signed_sc_fusion
    if native.get("format")==SIGNED_FORMAT:
        native,inputs,outputs,sources=load_signed_sc_fusion(audit_path,expected_sha=sha256(audit_path))
        if sha256(outputs["single_cell_signed_fusion"])!=sha256(prediction):
            raise ValueError("Fusion expert consumed prediction differs from signed SC aggregate")
        compare_snapshot(pd.read_parquet(outputs["single_cell_signed_fusion"]))
        for i,item in enumerate(sources):inventory.bind(item,audit_path.parent,model,role+".signed_source."+str(i))
        from .current_single_cell_model_use import inspect_signed_sources
        inspect_signed_sources(inventory,native,inputs,outputs,model,role)
        return
    from .native_auxiliary_fusion import FORMAT,load_native_auxiliary,native_sources
    if native.get("format")!=FORMAT:
        inventory.gap(model,role,"UNSUPPORTED_AUXILIARY_NATIVE_BINDING",digest=sha256(audit_path));return
    observed,native=load_native_auxiliary(audit_path,expected_sha=sha256(audit_path),expected_core_run_id=audit["core_run_id"])
    if native["outputs"]["fusion"]["sha256"]!=sha256(prediction):
        raise ValueError("Fusion expert consumed prediction differs from native aggregate")
    compare_snapshot(observed)
    manifest_path=inventory.bind(native["input_manifest"],audit_path.parent,model,role+".manifest")
    context=native_sources(manifest_path,expected_core_run_id=audit["core_run_id"])
    for i,item in enumerate(context["sources"].records.values()):
        inventory.bind(item,manifest_path.parent,model,role+".native_source."+str(i))
    sources_by_sha={r["sha256"]:r for r in context["sources"].records.values()}
    for (modality,fold),digest in context["checkpoint_map"].items():
        inventory.bind(sources_by_sha[digest],manifest_path.parent,model,role+".checkpoint."+modality,fold)
    from .evidence_rule_sources import read_table
    candidates=read_table(context["candidates_path"])
    candidate_aliases(candidates)
    inventory.entities(candidates,model=model,fold=-1,phase="FEATURE_CONSTRUCTION",
        view="AUXILIARY_CANDIDATE_INPUT",digest=sha256(context["candidates_path"]))
    manifest=context["manifest"];native_root=Path(manifest["native_receipt"]["path"])
    if not native_root.is_absolute():native_root=manifest_path.parent/native_root
    native_root=native_root.parent
    lineage=context["lineage"]
    inputs=[]
    if context["kind"]=="genomic":inputs=lineage["input_artifacts"]
    elif context["kind"]=="atac":
        input_path=inventory.bind({"path":lineage["input_lineage_audit_path"],
            "sha256":lineage["input_lineage_audit_sha256"]},native_root,model,role+".input_audit")
        source_audit=read(input_path)
        inputs=source_audit["input_artifacts"]+source_audit.get("materialization_raw_input_artifacts",[])
    elif context.get("consumed") is not None:
        from .current_cnv_model_use import inspect_consumed_sources
        inspect_consumed_sources(inventory,context,model,role)
    else:
        # Old CNV receipts retain some input digests but not consumed partitions.
        # Resolve exact hashes when supplied; never treat a SUCCESS as its matrix.
        for name,digest in context["native"].items():
            if name.endswith("_sha256") and isinstance(digest,str) and len(digest)==64 and digest not in {r["sha256"] for r in context["sources"].records.values()}:
                input_source(inventory,{"sha256":digest},native_root,model,role+"."+name)
        inventory.gap(model,role,"CNV_CONSUMED_PARTITION_ANCESTRY_NOT_RECONSTRUCTED")
    control_hashes={manifest["patient_folds"]["sha256"],manifest.get("core_manifest",{}).get("sha256")}
    for i,item in enumerate(inputs):
        if item.get("sha256") in control_hashes:item={**item,"source_role":"split_manifest"}
        input_source(inventory,item,native_root,model,role+".input."+str(i))
    if context["core_path"]:
        core_path=context["core_path"];core=read(core_path)
        from .genomic_training import _resolve_manifest_path
        for fold in range(5):
            part=core["folds"][str(fold)]
            inventory.bind(part["training_receipt"],core_path.parent,model,role+".core.training_receipt",fold)
            # Native genomic/CNV heads load exactly these two node exports.
            for node_type in ("lncRNA","pathway"):
                export=part["exports"][node_type]
                actual=_resolve_manifest_path(core_path,export["path"])
                inventory.bind({**export,"path":str(actual)},core_path.parent,model,role+".core.export."+node_type,fold)
                if node_type=="lncRNA":
                    nodes=pd.read_parquet(actual,columns=["node_id"]).rename(columns={"node_id":"lncrna_id"})
                    inventory.entities(nodes,model=model,fold=fold,phase="FEATURE_CONSTRUCTION",view="AUXILIARY_CORE_NODE_INPUT",digest=export["sha256"])
        from .current_core_model_use import inspect_export
        if not inspect_export(inventory,core_path,model,role+".core"):
            inventory.gap(model,role,"AUXILIARY_CORE_TRAINING_ANCESTRY_NOT_FULLY_RECONSTRUCTED")
    inventory.gap(model,role,"AUXILIARY_RAW_STUDY_AND_PATIENT_USAGE_NOT_FULLY_RECONSTRUCTED")
