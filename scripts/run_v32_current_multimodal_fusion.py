#!/usr/bin/env python3
"""Prepare or explicitly fit fusion from a versioned current-source manifest.

The default only assembles private feature/label inputs. Fitting is a separate
explicit option. Paths, run identities and expert semantics come from the
manifest, avoiding the historical runner's hard-coded August predictions.
"""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from cc_hhgt.v32.multimodal_fusion import (ExpertSpec, ResidualFusionConfig, artifact_sha256,
    build_pair_aggregated_fusion_frame_from_parquet, materialize_multimodal_fusion_release,
    train_pair_blocked_crossfit_fusion)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--expected-core-run-id",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fit",action="store_true",help="Explicitly fit five pair-blocked fusion folds and final models")
    args=parser.parse_args(argv)
    if socket.gethostname()!="149":
        raise RuntimeError("Current fusion preparation and execution require target host 149")
    manifest=json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format")!="CURRENT_MULTIMODAL_FUSION_MANIFEST_V1":
        raise ValueError("Unknown current fusion manifest")
    if manifest.get("core_run_id")!=args.expected_core_run_id:
        raise ValueError("Fusion manifest does not select the requested current core")
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({
        "target_host":"149","observed_host":socket.gethostname(),"fit_requested":args.fit,
        "training_started":False,"core_run_id":args.expected_core_run_id,
        "manifest_sha256":artifact_sha256(args.manifest)},indent=2))
    declarations=[]
    verified={}

    def bound(record,role="current_v32_oof_prediction"):
        path=Path(record["path"])
        if not path.is_absolute():path=args.manifest.resolve().parent/path
        path=path.resolve()
        if path not in verified:
            verified[path]=artifact_sha256(path)
        if verified[path]!=record.get("sha256"):
            raise ValueError(f"Current fusion input hash differs: {path}")
        declaration={"path":str(path),"sha256":verified[path],"source_role":role,"generation":"current_v32"}
        if declaration not in declarations:declarations.append(declaration)
        return path

    primary=bound(manifest["primary_prediction"],"provenance_audited_standardized_source")
    core_binding_path=bound(manifest["primary_binding_audit"],"provenance_audited_standardized_source")
    core_binding=json.loads(core_binding_path.read_text(encoding="utf-8"))
    from cc_hhgt.v32.current_primary_fusion import FORMAT, load_primary_fusion_inputs
    native_labels=core_binding.get("format")==FORMAT
    if native_labels:
        core_binding=load_primary_fusion_inputs(core_binding_path,expected_sha=verified[core_binding_path],
                                              expected_core_run_id=args.expected_core_run_id)
    elif args.fit:
        raise ValueError("Fitting requires replayed native primary validation labels; bare label hash lists are unverified")
    if (core_binding.get("run_id")!=args.expected_core_run_id or core_binding.get("status")!="PASS"
            or core_binding.get("prediction",{}).get("sha256")!=verified[primary]):
        raise ValueError("Primary prediction is not bound to the requested current core run")
    fold_records=manifest["primary_folds"]
    if len(fold_records)!=5 or {item["fold_id"] for item in fold_records}!=set(range(5)):
        raise ValueError("Current primary requires all five label views")
    authorized_folds = ({r["fold_id"]:r["artifact"]["sha256"] for r in core_binding["folds"]}
                        if native_labels else None)
    if native_labels and {r["fold_id"]:r["artifact"]["sha256"] for r in fold_records} != authorized_folds:
        raise ValueError("Native validation fold labels differ from the selected core binding")
    if not native_labels and {item["artifact"]["sha256"] for item in fold_records} != set(core_binding.get("fold_prediction_sha256", [])):
        raise ValueError("Primary fold labels differ from the selected core binding")
    folds=[bound(item["artifact"],"task_definition" if native_labels else "current_v32_oof_prediction")
           for item in sorted(fold_records,key=lambda item:item["fold_id"])]
    experts={}
    expert_bindings={}
    native_audits={}
    for item in manifest["experts"]:
        definition=dict(item["spec"])
        definition["endpoint_roles"]=tuple(definition["endpoint_roles"])
        spec=ExpertSpec(**definition)
        spec.validate()
        if not item.get("run_id") or not item.get("estimand"):
            raise ValueError("Every current expert requires a run identity and declared estimand")
        if item.get("evidence_mode") == "RULE_ONLY":
            raise ValueError("Rule evidence is an annotation; use rule_evidence_binding, not a fitted expert")
        if spec.expert_id=="evidence_transformer":
            raise ValueError("Current policy excludes the retired neural Evidence head")
        audit_path=bound(item["binding_audit"],"provenance_audited_standardized_source")
        audit=json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("format")=="NATIVE_AUXILIARY_FUSION_BINDING_V1":
            from cc_hhgt.v32.native_auxiliary_fusion import load_native_auxiliary
            if audit_path not in native_audits:
                _,native_audits[audit_path]=load_native_auxiliary(audit_path,
                    expected_sha=verified[audit_path],expected_core_run_id=args.expected_core_run_id)
            audit=native_audits[audit_path]
        elif audit.get("format")=="SC_HHGT_SIGNED_FUSION_BINDING_V1":
            from cc_hhgt.v32.single_cell_signed_fusion import load_signed_sc_fusion
            if audit_path not in native_audits:
                native_audits[audit_path]=load_signed_sc_fusion(audit_path,expected_sha=verified[audit_path])[0]
            audit=native_audits[audit_path]
        if audit.get("run_id")!=item["run_id"] or audit.get("estimand")!=item["estimand"]:
            raise ValueError(f"{spec.expert_id} run identity or estimand differs from its producer audit")
        dependency=audit.get("core_dependency")
        if dependency=="CURRENT_CORE_EMBEDDINGS":
            if audit.get("core_run_id")!=args.expected_core_run_id:
                raise ValueError(f"{spec.expert_id} was produced from a different core")
            from cc_hhgt.v32.current_core_export import validate_current_export
            source_core=bound(audit["core_manifest"],"provenance_audited_standardized_source")
            validate_current_export(source_core,expected_run_id=args.expected_core_run_id)
        elif dependency not in {"INDEPENDENT_MODEL","CORE_INDEPENDENT_EVIDENCE"}:
            raise ValueError(f"{spec.expert_id} lacks an explicit core dependency")
        approved_specs=audit.get("expert_specs",[])
        # JSON normalization changes tuple endpoint roles into list form.
        from dataclasses import asdict
        if json.loads(json.dumps(asdict(spec))) not in approved_specs:
            raise ValueError(f"{spec.expert_id} semantics differ from its producer declaration")
        record=item["prediction"]
        output_records=list(audit.get("outputs",{}).values()) if isinstance(audit.get("outputs"),dict) else audit.get("outputs",[])
        # The producer audit must bind the consumed artifact, not merely state PASS.
        if not any(isinstance(value,dict) and value.get("sha256")==record.get("sha256") for value in output_records):
            raise ValueError(f"{spec.expert_id} producer audit does not bind its fusion input")
        if audit.get("status") not in {"PASS","SUCCESS","PASS_HASH_BOUND_FUSION_INPUT"}:
            raise ValueError(f"{spec.expert_id} producer audit did not pass")
        if spec in experts or any(existing.expert_id==spec.expert_id for existing in experts):
            raise ValueError("Duplicate current fusion expert")
        experts[spec]=bound(record)
        expert_bindings[spec.expert_id]={"spec":json.loads(json.dumps(asdict(spec))),
            "run_id":item["run_id"],"estimand":item["estimand"],
            "prediction":{"path":str(experts[spec]),"sha256":verified[experts[spec]]},
            "binding_audit":{"path":str(audit_path),"sha256":verified[audit_path]}}
    frame,audit=build_pair_aggregated_fusion_frame_from_parquet(primary,folds,experts,
        temp_directory=args.output/"duckdb_tmp",native_validation_labels=native_labels)
    available_pair_folds=(set(frame.loc[frame.fusion_target_available,"fusion_pair_fold"])
                          if native_labels else set())
    fit_ready=native_labels and available_pair_folds==set(range(5))
    audit.update(primary_label_authority="NATIVE_VALIDATION_REPLAY" if native_labels else "UNVERIFIED_LEGACY_HASH_LIST",
                 native_label_authority_verified=native_labels,eligible_for_current_fitting=fit_ready,
                 primary_label_binding={"path":str(core_binding_path),"sha256":verified[core_binding_path]})
    rule_binding = manifest.get("rule_evidence_binding")
    if rule_binding:
        from cc_hhgt.v32.evidence_rule_release import attach_rule_annotations
        path = bound(rule_binding, "provenance_audited_standardized_source")
        rule_binding = {"path": str(path), "sha256": verified[path]}
        frame = attach_rule_annotations(frame, binding_path=path, expected_sha=verified[path])
        audit.update(rule_evidence_binding=rule_binding, rule_evidence_used_as_model_input=False)
    frame_path=args.output/"PRIVATE_FUSION_INPUT.parquet"
    frame.to_parquet(frame_path,index=False)
    audit.update(core_run_id=args.expected_core_run_id,manifest_sha256=artifact_sha256(args.manifest),
                 training_started=False,primary_source_sha256=verified[primary],
                 expert_bindings=expert_bindings)
    (args.output/"INPUT_AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n")
    if not args.fit:
        receipt={"status":"PREPARED_NOT_TRAINED","target_host":"149","training_started":False,
                 "core_run_id":args.expected_core_run_id,"rows":len(frame),"input_sha256":artifact_sha256(frame_path),
                 "eligible_for_current_fitting":fit_ready,"primary_label_authority":audit["primary_label_authority"]}
        (args.output/"PREPARED.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(receipt))
        return 0
    if not fit_ready:
        raise ValueError("Native available fusion labels do not cover all five pair folds; fitting did not start")
    config=ResidualFusionConfig(**manifest.get("training_config",{}))
    specs=tuple(experts)
    discovery=train_pair_blocked_crossfit_fusion(frame,specs,endpoint="discovery",config=config)
    confidence=train_pair_blocked_crossfit_fusion(frame,specs,endpoint="confidence",config=config)
    result=materialize_multimodal_fusion_release(args.output/"release",frame,discovery,confidence,
        source_declarations=declarations,training_run_id=manifest["fusion_run_id"],input_audit=audit,
        rule_evidence_binding=rule_binding)
    (args.output/"SUCCESS.json").write_text(json.dumps({"status":"SUCCESS_SECONDARY_FUSION",**result,
        "core_run_id":args.expected_core_run_id,"target_host":"149","training_started":True},indent=2)+"\n")
    print(json.dumps(result))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
