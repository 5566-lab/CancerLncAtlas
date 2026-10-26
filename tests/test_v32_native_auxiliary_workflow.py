"""Actual native no-fit collectors -> frozen result containers -> audits/fusion."""
from copy import deepcopy
from pathlib import Path
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import native_auxiliary_workflow as workflow
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_native_cnv_head_scope import native as cnv_native, auditable_package
from test_v32_native_atac_scope import materialized as atac_native, frozen_package as frozen_atac, RUN as ATAC_RUN


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    def forbidden(*a, **k): raise AssertionError("No optimizer, fitting or auxiliary forward allowed")
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    from cc_hhgt.v32 import cnv_only_training_v2 as cnv, atac_training as atac, genomic_training as genomic
    for module in (cnv, genomic):
        monkeypatch.setattr(module, "_fit_head", forbidden); monkeypatch.setattr(module, "_predict_head", forbidden)
    monkeypatch.setattr(atac, "_fit_logistic", forbidden); monkeypatch.setattr(atac, "_predict_logistic", forbidden)
    import scripts.run_v32_current_multimodal_fusion as fusion
    monkeypatch.setattr(fusion, "train_pair_blocked_crossfit_fusion", forbidden)


def specification(kind, fixture, output):
    spec = {"format":workflow.INPUT, "producer_kind":kind, "core_run_id":"current-G2",
            "native_run_id":"v32-cnv-synthetic" if kind == "cnv" else ATAC_RUN}
    if kind == "cnv":
        args, core = fixture
        files = {"candidate_authority":args.candidates, "patient_folds":args.patient_map,
            "core_manifest":core, "directional_success":args.directional_root/"SUCCESS.json",
            "association_success":args.output_root/"SUCCESS.json"}
    else:
        _, paths, success, _ = fixture
        files = {"candidate_authority":paths["candidates"], "patient_folds":paths["patient_folds"],
            "materialization_success":success, "pathway_membership":paths["pathway_membership"]}
    spec.update({k:record(v) for k,v in files.items()}); write(output, spec)
    return output


def complete(kind, fixture, prepared, root):
    _, _, plan = workflow.validate_preparation(record(prepared), root)
    training = prepared.parent/"training"
    if kind == "cnv":
        path = auditable_package(fixture, root/"serialization_fixture")
        receipt_path = Path(read(path)["native_receipt"]["path"]); receipt = read(receipt_path)
        receipt["training_config"] = plan["config"]
        training.mkdir(); out = training/"OOF_MANIFEST.json"; write(out, receipt)
        ready = read(receipt_path.parent/"PACKAGE_READY.json"); ready["oof_manifest_sha256"] = record(out)["sha256"]
        write(training/"PACKAGE_READY.json", ready)
        return out
    _, out = frozen_atac(fixture, training)
    success = read(out); lp = Path(success["lineage_path"]); lineage = read(lp)
    config = training/"RUN_CONFIG.json"
    write(config, {"training_run_id":plan["native_run_id"], "training":plan["config"],
                   "synthetic_serialization_no_optimizer":True})
    lineage.update(config_path=str(config), config_sha256=record(config)["sha256"])
    write(lp, lineage); success["lineage_sha256"] = record(lp)["sha256"]; write(out, success)
    return out


@pytest.fixture(params=["cnv", "atac"])
def branch(request, cnv_native, atac_native):
    return request.param, cnv_native if request.param == "cnv" else atac_native


def test_native_chain_auxiliary_prepare_collect(branch,tmp_path):
    from native_chain_fixture import exercise
    kind,fixture=branch;source=specification(kind,fixture,tmp_path/"input.json")
    result=exercise("auxiliary",source,tmp_path,lambda prepared:complete(kind,fixture,prepared,tmp_path))
    assert result["producer_kind"]==kind and not result["formal_increment_established"]


def test_actual_prepare_collect_cli_independent_audit_and_fusion(branch, tmp_path):
    from scripts.run_native_auxiliary_workflow import main
    from cc_hhgt.v32.native_auxiliary_fusion import load_native_auxiliary
    kind, fixture = branch; manifest = specification(kind, fixture, tmp_path/"input.json")
    output = tmp_path/"prepared"
    args = ["prepare", "--manifest", str(manifest), "--output", str(output)]
    assert main([*args, "--plan"]) == 0 and not output.exists()
    assert main(args) == 0
    prepared = output/"PREPARED.json"
    _, binding, plan = workflow.validate_preparation(record(prepared), tmp_path)
    assert not binding["training_started"] and not (output/"training").exists()
    command = read(output/"TRAINING_COMMAND.json")
    assert command["execution_mode"] == "CODE_ONLY" and not command["automatically_executed"]
    actual_parser = __import__(workflow.MODULES[kind][1], fromlist=["_parser"])._parser()
    parsed = actual_parser.parse_args(command["argv"][3:])
    assert parsed.training_run_id == plan["native_run_id"] and parsed.output_root == str(output/"training")
    with pytest.raises(ValueError, match="cannot dispatch"): workflow.execute(command["argv"], tmp_path/"forbidden.log")
    assert not (tmp_path/"forbidden.log").exists()
    completed = complete(kind, fixture, prepared, tmp_path)
    before = {str(p):record(p)["sha256"] for p in completed.parent.rglob("*") if p.is_file()}
    collected = tmp_path/"collected"
    assert main(["collect", "--preparation", str(prepared), "--preparation-sha256", record(prepared)["sha256"],
        "--completed", str(completed), "--completed-sha256", record(completed)["sha256"], "--output", str(collected)]) == 0
    result = collected/"NATIVE_AUXILIARY_BINDING.json"
    _, final = workflow.validate(record(result), tmp_path)
    assert final["candidate_rows"] == (6 if kind == "cnv" else 240)
    assert final["fold_rows"] == final["candidate_rows"]*5
    assert not final["training_started"] and not final["source_ancestry_complete"]
    assert before == {str(p):record(p)["sha256"] for p in completed.parent.rglob("*") if p.is_file()}
    adapter = Path(final["fusion_binding"]["path"])
    frame, info = load_native_auxiliary(adapter, expected_sha=record(adapter)["sha256"], expected_core_run_id="current-G2")
    assert len(frame) == final["candidate_rows"] and not info["native_model_forwards_replayed"]
    assert record(prepared) in info["sources"]
    assert final["independent_audit"] in info["sources"] and final["independent_success"] in info["sources"]
    # Exercise the actual current fusion preparer. Primary probabilities/labels
    # here are explicitly synthetic legacy hash-list fixtures, so its native
    # label authority remains unverified and fitting eligibility must stay false.
    from cc_hhgt.v32.multimodal_fusion import PRIMARY_COLUMN, ANALYSIS_VERSION
    import scripts.run_v32_current_multimodal_fusion as fusion
    keys = frame[["cancer_id", "lncrna_id", "pathway_id"]]
    primary = tmp_path/"primary.parquet"
    keys.assign(**{PRIMARY_COLUMN:.25, "analysis_version":ANALYSIS_VERSION}).to_parquet(primary,index=False)
    folds = []
    for fold in range(5):
        target = tmp_path/f"labels_{fold}.parquet"
        keys.assign(patient_fold_id=fold, held_out_proxy_label=fold%2).to_parquet(target,index=False)
        folds.append({"fold_id":fold, "artifact":record(target)})
    primary_binding = tmp_path/"primary.json"
    write(primary_binding, {"status":"PASS", "run_id":"current-G2", "prediction":record(primary),
        "fold_prediction_sha256":[r["artifact"]["sha256"] for r in folds], "synthetic_unverified_label_fixture":True})
    fusion_manifest = tmp_path/"fusion.json"
    write(fusion_manifest, {"format":"CURRENT_MULTIMODAL_FUSION_MANIFEST_V1", "core_run_id":"current-G2",
        "primary_prediction":record(primary), "primary_binding_audit":record(primary_binding),
        "primary_folds":folds, "experts":read(final["fusion_experts"]["path"])})
    assert fusion.main(["--manifest",str(fusion_manifest),"--expected-core-run-id","current-G2",
                        "--output",str(tmp_path/"fusion_prepared")]) == 0
    state = read(tmp_path/"fusion_prepared/PREPARED.json")
    assert not state["training_started"] and not state["eligible_for_current_fitting"]
    actual = pd.read_parquet(tmp_path/"fusion_prepared/PRIVATE_FUSION_INPUT.parquet")
    assert len(actual) == len(frame) and actual[kind+"_available"].any()
    with pytest.raises(FileExistsError): workflow.collect(record(prepared), record(completed), collected)


@pytest.mark.parametrize("fault", ["unsupported", "unknown_parameter", "bool", "fractional", "nonfinite", "negative", "native_run", "input_hash"])
def test_invalid_manifest_cannot_create_output(cnv_native, tmp_path, fault):
    manifest = specification("cnv", cnv_native, tmp_path/"input.json"); spec = read(manifest)
    if fault == "unsupported": spec["producer_kind"] = "genomic"
    elif fault == "unknown_parameter": spec["config"] = {"unknown":1}
    elif fault == "bool": spec["config"] = {"epochs":True}
    elif fault == "fractional": spec["config"] = {"epochs":1.5}
    elif fault == "nonfinite": spec["config"] = {"learning_rate":float("nan")}
    elif fault == "negative": spec["config"] = {"learning_rate":-1.}
    elif fault == "native_run": spec["native_run_id"] = "../elsewhere"
    else: spec["candidate_authority"]["sha256"] = "0"*64
    write(manifest, spec)
    with pytest.raises(ValueError): workflow.prepare(manifest, tmp_path/"prepared")
    assert not (tmp_path/"prepared").exists()


def test_actual_preparation_failure_is_not_a_ready_workflow(atac_native, tmp_path):
    manifest = specification("atac", atac_native, tmp_path/"input.json"); spec = read(manifest)
    spec["config"] = {"max_training_rows":10}; write(manifest, spec)
    with pytest.raises(ValueError, match=">=100"): workflow.prepare(manifest, tmp_path/"prepared")
    assert not (tmp_path/"prepared/PREPARED.json").exists()
    assert read(tmp_path/"prepared/RUN_FAILED.json")["stage"] == "prepare"


@pytest.mark.parametrize("fault", ["config", "run", "output_path", "input", "preparation_command"])
def test_completed_results_cannot_drift_from_preparation(branch, tmp_path, fault):
    kind, fixture = branch; manifest = specification(kind, fixture, tmp_path/"input.json")
    prepared = workflow.prepare(manifest, tmp_path/"prepared")
    completed = complete(kind, fixture, prepared, tmp_path)
    result = read(completed)
    if fault == "config":
        if kind == "cnv": result["training_config"]["epochs"] += 1
        else:
            lp = Path(result["lineage_path"]); lineage = read(lp); cp = Path(lineage["config_path"]); config = read(cp)
            config["training"]["epochs"] += 1; write(cp, config)
            lineage["config_sha256"] = record(cp)["sha256"]; write(lp, lineage); result["lineage_sha256"] = record(lp)["sha256"]
        write(completed, result)
    elif fault == "run": result["training_run_id"] += "-different"; write(completed, result)
    elif fault == "output_path":
        alternate = tmp_path/"other.json"; write(alternate, result); completed = alternate
    elif fault == "input":
        spec = read(manifest); spec["native_run_id"] += "-changed"; write(manifest, spec)
    else:
        cp = prepared.parent/"TRAINING_COMMAND.json"; cmd = read(cp); cmd["argv"][-1] = "999"; write(cp, cmd)
        binding = read(prepared); binding["training_command"] = record(cp); write(prepared, binding)
    with pytest.raises((ValueError, FileNotFoundError)):
        workflow.collect(record(prepared), record(completed), tmp_path/"collected")
    assert not (tmp_path/"collected/NATIVE_AUXILIARY_BINDING.json").exists()


def test_rehashed_audit_cannot_be_substituted(branch, tmp_path):
    kind, fixture = branch; manifest = specification(kind, fixture, tmp_path/"input.json")
    prepared = workflow.prepare(manifest, tmp_path/"prepared"); completed = complete(kind, fixture, prepared, tmp_path)
    result = workflow.collect(record(prepared), record(completed), tmp_path/"collected")
    value = read(result); audit_path = Path(value["independent_audit"]["path"]); audit = read(audit_path)
    key = "oof_manifest_sha256" if kind == "cnv" else "training_success_sha256"
    audit[key] = "0"*64; write(audit_path, audit); value["independent_audit"] = record(audit_path); write(result, value)
    with pytest.raises(ValueError): workflow.validate(record(result), tmp_path)


@pytest.mark.parametrize("fault", ["changed_audit", "missing_audit_binding", "wrong_audit_source", "expert_declaration"])
def test_actual_fusion_consumer_rechecks_workflow_audit(branch, tmp_path, fault):
    from cc_hhgt.v32.native_auxiliary_fusion import native_sources
    kind, fixture = branch; manifest = specification(kind, fixture, tmp_path/"input.json")
    prepared = workflow.prepare(manifest, tmp_path/"prepared"); completed = complete(kind, fixture, prepared, tmp_path)
    result = workflow.collect(record(prepared), record(completed), tmp_path/"collected"); value = read(result)
    adapter_input = Path(value["adapter_input"]["path"]); declaration = read(adapter_input)
    if fault == "missing_audit_binding":
        declaration.pop("workflow_audit"); write(adapter_input, declaration)
    elif fault == "expert_declaration":
        path = Path(value["fusion_experts"]["path"]); experts = read(path); experts[0]["estimand"] = "PATIENT_TREATMENT_EFFECT"
        write(path, experts); value["fusion_experts"] = record(path); write(result, value)
        with pytest.raises(ValueError, match="expert declaration"): workflow.validate(record(result), tmp_path)
        return
    else:
        path = Path(value["independent_audit"]["path"]); audit = read(path)
        if fault == "changed_audit": audit["status"] = "FAILED"
        else:
            key = "oof_manifest_sha256" if kind == "cnv" else "training_success_sha256"
            audit[key] = "0"*64
        write(path, audit)
        if fault == "wrong_audit_source":
            success_path = Path(value["independent_success"]["path"]); success = read(success_path)
            success["independent_audit_sha256" if kind == "cnv" else "audit_sha256"] = record(path)["sha256"]
            write(success_path, success)
            declaration["workflow_audit"] = {"audit":record(path), "success":record(success_path)}
            write(adapter_input, declaration)
    with pytest.raises(ValueError): native_sources(adapter_input, expected_core_run_id="current-G2")


def test_wrong_host_cannot_plan_or_dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr(socket, "gethostname", lambda:"windows")
    with pytest.raises(RuntimeError, match="149"): workflow.plan(tmp_path/"absent.json", tmp_path/"output")
