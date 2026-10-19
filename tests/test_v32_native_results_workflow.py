"""Native preparation -> frozen serialized checkpoints -> actual publication.

Checkpoint containers and optimizer telemetry are explicit synthetic fixtures.
No optimizer is invoked; frozen forwards, selection, exports and publishers are
the actual production implementations. Full-cohort identity controls remain
the same declared substitutions as the upstream native-core fixture.
"""
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path
import json

import pandas as pd
import pytest
import torch
import yaml

from test_v32_native_core_workflow import inputs, completed as prepared, controls, normalized, scoped, native_authority, no_fit
from cc_hhgt.v32 import native_results_workflow as workflow
from cc_hhgt.v32.formal_expression_sources import read, write, record


def patched_controls(patch, native_authority, scoped):
    controls(patch, native_authority, scoped)
    import cc_hhgt.v32.g012_winner_lock as lock
    import cc_hhgt.v32.sealed_test_inference as sealed
    for module in (lock, sealed):
        patch.setattr(module, "FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256", record(scoped.parent/"patient_map.tsv")["sha256"])
        patch.setattr(module, "FROZEN_V32_RECEIPT_SHA256", record(native_authority[0]/"patients.json")["sha256"])


@pytest.fixture(scope="module")
def registered(prepared, native_authority, scoped, tmp_path_factory):
    root = tmp_path_factory.mktemp("native_completed_results")
    from cc_hhgt.v32.training import load_prepared_artifact_from_authorized_handle, _prepared_input_authority_hashes, _runtime_chunk_contract
    from cc_hhgt.v32.native_graph_input import load_training_graph_inputs
    from cc_hhgt.gnn import build_model
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual
    from cc_hhgt.v32.current_core_export import CHECKPOINT_FORMAT
    from cc_hhgt.v32.training_guard import code_tree_sha256
    from cc_hhgt.v32.current_analysis_web import COMPONENTS
    commands = read(prepared/"TRAINING_COMMANDS.json")["commands"]; runs = {}
    with pytest.MonkeyPatch.context() as patch:
        patched_controls(patch, native_authority, scoped)
        code = code_tree_sha256(workflow.ROOT)
        for group_index, group in enumerate(workflow.GROUPS):
            folder = root/group; folder.mkdir()
            planned = [c for c in commands if c.get("arm", c.get("graph_variant")) == group]
            first = planned[0]["argv"]; arg = lambda name: first[first.index(name)+1]
            config = yaml.safe_load(Path(arg("--config")).read_text())
            config["execution_control"].update(execution_mode="TRAINING", training_authorized=True)
            config["training_io"]["output_root"] = str(folder/"unused_training_output")
            cp = folder/"config.yaml"; cp.write_text(yaml.safe_dump(config, sort_keys=False))
            ip = folder/"INPUT_MANIFEST.json"; ip.write_bytes(Path(arg("--input-manifest")).read_bytes())
            tasks = pd.read_csv(arg("--task-manifest"), sep="\t").fillna("")
            tasks["status"] = "PENDING"; tasks["blocked_reason"] = ""
            tp = folder/"TASK_MANIFEST.tsv"; tasks.to_csv(tp, sep="\t", index=False)
            declaration = {"config": record(cp), "input_manifest": record(ip), "task_manifest": record(tp), "folds": []}
            hashes = {"code_sha256": code, "config_sha256": record(cp)["sha256"],
                "input_manifest_sha256": record(ip)["sha256"], "task_manifest_sha256": record(tp)["sha256"]}
            for command in planned:
                fold = command["fold"]; args = command["argv"]
                row = next(r for r in read(ip)["fold_inputs"] if r["fold"] == fold)
                payload, digest = load_prepared_artifact_from_authorized_handle(torch, ip, fold=fold, prepared_path=Path(row["path"]))
                payload, _ = load_training_graph_inputs(torch, payload=payload, config=config, input_manifest_path=ip,
                    fold=fold, root=root, prepared_sha256=digest, task_seed=20260726)
                torch.manual_seed(500+group_index*10+fold)
                encoder = build_model("cc_hhgt", payload["bundle"], payload["feature_dim"], payload["legacy_model_config"])
                model = build_v32_cc_hhgt_residual(encoder, hidden_channels=96, context_features=4, dropout=.2)
                checkpoint = {"checkpoint_format": CHECKPOINT_FORMAT, "model_state": model.state_dict(),
                    "patient_fold": fold, "task_id": args[args.index("--task-id")+1],
                    "graph_variant": group if group.startswith("G") else "G2", "artifact_hashes": hashes,
                    "input_authority_hashes": _prepared_input_authority_hashes(payload), "prepared_artifact_sha256": digest,
                    "cycle": 0, "synthetic_serialization_fixture": True, "no_optimizer_actually_run": True}
                ck = folder/(str(fold)+".pt"); torch.save(checkpoint, ck)
                chunks, weights = _runtime_chunk_contract(payload["bundle"])
                receipt = {k: v for k, v in checkpoint.items() if k != "model_state"}
                receipt.update(status="SUCCESS", run_id=args[args.index("--run-id")+1], best_model_state_sha256=record(ck)["sha256"],
                    optimizer_steps=1, parameter_delta_positive=True, validation_full_runtime_coverage=True,
                    runtime_num_chunks=len(chunks), runtime_chunk_weights={str(c): weights[c] for c in chunks},
                    best_validation_loss={"G0": .4, "G1": .2, "G2": .3}.get(group, .25)+fold*.001,
                    architecture_id="HHGT_FORMAL_CORE_EXTERNAL_ROUTER", evidence_integration_mode="external_router",
                    test_labels_read=False, test_metrics_computed=False)
                success = folder/(str(fold)+".json"); write(success, receipt)
                declaration["folds"].append({"fold": fold, "checkpoint": record(ck), "training_receipt": record(success)})
            runs[group] = declaration
    path = root/"results.json"
    candidate = pd.read_parquet(read(prepared/"LAUNCH_PREFLIGHT.json")["files"]["candidates"]["path"])
    pair = candidate.loc[candidate.lncrna_id.eq("LNC:Z")].iloc[0]
    external = root/"external.parquet"
    pd.DataFrame({"record_id": ["synthetic_known_lnc"], "lncrna": ["z"], "cancer": ["breast"], "pathway": ["p"],
        "label": [1], "pmid": ["99119922"], "is_experimental": [True], "is_predicted": [False], "assay": ["Synthetic"]}).to_parquet(external, index=False)
    identities = root/"identities.parquet"
    pd.DataFrame([("lncrna", "z", pair.lncrna_id), ("cancer", "breast", pair.cancer_id), ("pathway", "p", pair.pathway_id)],
        columns=["namespace", "raw_id", "canonical_id"]).to_parquet(identities, index=False)
    external_input = root/"external.json"
    write(external_input, {"format": "CURRENT_EXTERNAL_VALIDATION_INPUT_V2",
        "models": [{"model_id": "selected", "producer_kind": "selected_release", "producer_binding": {"workflow_ref": "selected_score"},
                    "run_id": {"result_ref": "core_run_id"}, "score_column": "selected_score"}],
        "model_use_inventory": {"derive_from_native_models": True},
        "identity_crosswalk": {**record(identities), "path": identities.name},
        "external_sources": [{"kind": "exact_assay", "source_id": "synthetic-external", "table": {**record(external), "path": external.name}}],
        "evaluation": {"primary_model": "selected", "cutoffs": [1], "lncrna_aggregation": "MAX", "bootstrap_replicates": 0, "seed": 0}})
    components = {k: {"status": "NOT_PRODUCED", "reason": "No auxiliary producer in this synthetic core workflow check"} for k in COMPONENTS}
    components["external"] = {"status": "BUILD", "input_manifest": {**record(external_input), "path": external_input.name}}
    write(path, {"format": workflow.INPUT, "release": "synthetic-native-results", "core_preparation": record(prepared/"NATIVE_CORE_BINDING.json"),
        "runs": runs, "geneset": {"min_members": 1, "max_members": 3, "score_min": 0., "min_expression_folds": 1},
        "subtype": {"score_column": "selected_score", "min_members": 1, "min_cancers": 2, "bootstrap_replicates": 2},
        "publication_components": components})
    return root, path


@pytest.fixture(scope="module")
def released(registered, native_authority, scoped):
    root, source = registered
    with pytest.MonkeyPatch.context() as patch:
        patched_controls(patch, native_authority, scoped)
        from scripts.run_native_results_workflow import main
        buffer = StringIO()
        with redirect_stdout(buffer): assert main(["--manifest", str(source), "--output", str(root/"output"), "--plan"]) == 0
        plan = json.loads(buffer.getvalue()); assert set(plan["runs"]) == set(workflow.GROUPS) and not (root/"output").exists()
        before = {f["checkpoint"]["path"]: f["checkpoint"]["sha256"] for r in read(source)["runs"].values() for f in r["folds"]}
        assert main(["--manifest", str(source), "--output", str(root/"output")]) == 0
        assert all(record(p)["sha256"] == digest for p, digest in before.items())
    return root/"output/NATIVE_RESULTS_BINDING.json"


def test_actual_native_selection_predictions_export_labels_and_web(released, native_authority, scoped, monkeypatch):
    patched_controls(monkeypatch, native_authority, scoped)
    _, binding = workflow.validate(record(released), released.parent)
    assert binding["winner"] == "G1" and set(binding["predictions"]) == set(workflow.GROUPS)
    lock = read(binding["selection"]["path"])
    assert set(lock["compared_runs"]) == {"G0", "G1", "G2"}
    for group, source in binding["predictions"].items():
        prediction = read(source["path"]); frame = pd.read_parquet(prediction["prediction"]["path"])
        assert len(frame) == 3 and not prediction["training_started"]
        assert all(r["core_unchanged_after_prediction"] for r in prediction["ensemble_source_records"])
        assert frame.loc[frame.lncrna_id.eq("LNC:LOW"), "shared_or_local_scope"].eq("not_expression_eligible").all()
    export = read(binding["core_export"]["path"])
    assert all(r["core_unchanged_after_export"] for r in export["folds"].values())
    primary = read(binding["primary_fusion"]["path"])
    assert primary["supervision_scope"] == "VALIDATION_REUSED_FOR_FUSION_DEVELOPMENT"
    for item in primary["folds"]:
        labels = pd.read_parquet(item["artifact"]["path"])
        row = next(r for r in lock["compared_runs"]["G1"]["prepared_folds"] if r["patient_fold"] == item["fold_id"])
        payload = torch.load(row["path"], map_location="cpu", weights_only=False)
        inverse = {v: k for k, v in payload["bundle"].node_maps["lncRNA"].items()}
        expected = {inverse[int(index)]: bool(mask) for batch in payload["validation_batches"]
                    for index, mask in zip(batch["candidate_batch"]["l"], batch["association_available"])}
        assert labels.set_index("lncrna_id").validation_label_available.to_dict() == expected
        assert not labels.loc[labels.lncrna_id.eq("LNC:LOW"), "validation_label_available"].any()
        assert labels.loc[~labels.validation_label_available, "validation_proxy_label"].isna().all()
    publication = read(binding["publication"]["path"])
    from cc_hhgt.v32.current_analysis_web import CurrentAnalysisWebStore
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from website.backend.current_analysis_api import current_analysis_router
    store = CurrentAnalysisWebStore(publication["web_binding"]["path"], expected_sha=publication["web_binding"]["sha256"])
    app = FastAPI(); app.include_router(current_analysis_router(store)); client = TestClient(app)
    catalog = client.get("/api/current/catalog").json()
    result = client.post("/api/current/tables/scores.candidates/query", json={"version": catalog["version"]})
    assert result.status_code == 200 and len(result.json()["rows"]) == 3
    isolation = client.post("/api/current/tables/external.isolation/query", json={"version": catalog["version"]}).json()["rows"]
    assert isolation[0]["lncrna_overlap"] and not isolation[0]["provenance_held_out"]
    stages = publication["stages"]
    inventory = read(next(s["binding"]["path"] for s in stages if s["stage"] == "model_use"))
    dependencies = pd.read_parquet(inventory["outputs"]["dependencies"]["path"])
    assert "G012_COMPARISON:G0" in set(dependencies.dependency_context)
    assert not inventory["model_use_inventory_completeness_verified"]
    replayed = pd.read_parquet(inventory["outputs"]["transformation_replays"]["path"])
    from cc_hhgt.v32.prepared_construction import FORMAT
    construction = replayed.loc[replayed.format.eq(FORMAT)]
    assert set(construction.patient_fold_id) == set(range(5))
    assert {"G012_COMPARISON:G0", "G012_COMPARISON:G1", "G012_COMPARISON:G2"} <= set(construction.dependency_context)
    from cc_hhgt.v32.local_cnv_source_replay import FORMAT as LOCAL_CNV_REPLAY
    statistics = replayed.loc[replayed.format.eq(LOCAL_CNV_REPLAY)]
    assert set(statistics.patient_fold_id) == set(range(5))
    assert {"G012_COMPARISON:G0", "G012_COMPARISON:G1", "G012_COMPARISON:G2"} <= set(statistics.dependency_context)
    assert {role.rsplit(".", 1)[-1] for role in statistics.role} == {"pair", "graph", "validation"}
    assert inventory["model_forwards_executed"] and not inventory["neural_model_forwards_executed"]
    gaps = pd.read_parquet(inventory["outputs"]["gaps"]["path"])
    assert "RAW_FEATURE_LABEL_STUDY_LINEAGE_NOT_RECONSTRUCTED" not in set(gaps.reason)
    assert "LOCAL_CNV_NUMERIC_REPLAY_NOT_BOUND" not in set(gaps.reason)
    assert len(gaps) > 0  # Other upstream ancestry limits remain blocking.
    assert not binding["fusion_fitting_started"] and not binding["production_deployed"]


@pytest.mark.parametrize("fault", ["missing_group", "duplicate_fold", "science_config", "task", "receipt", "checkpoint"])
def test_mixed_or_incomplete_completed_runs_fail_before_outputs(registered, native_authority, scoped, monkeypatch, tmp_path, fault):
    patched_controls(monkeypatch, native_authority, scoped)
    spec = deepcopy(read(registered[1])); declaration = spec["runs"]["G0"]
    spec["publication_components"]["external"]["input_manifest"] = record(registered[0]/"external.json")
    if fault == "missing_group": del spec["runs"]["experimental"]
    elif fault == "duplicate_fold": declaration["folds"][1] = declaration["folds"][0]
    elif fault == "science_config":
        cfg = yaml.safe_load(Path(declaration["config"]["path"]).read_text()); cfg["primary_model"]["hidden_channels"] = 32
        path = tmp_path/"config.yaml"; path.write_text(yaml.safe_dump(cfg)); declaration["config"] = record(path)
    elif fault == "task":
        task = pd.read_csv(declaration["task_manifest"]["path"], sep="\t"); task.loc[0, "patient_fold"] = 4
        path = tmp_path/"tasks.tsv"; task.to_csv(path, sep="\t", index=False); declaration["task_manifest"] = record(path)
    elif fault == "receipt":
        fold = declaration["folds"][0]; receipt = read(fold["training_receipt"]["path"]); receipt["optimizer_steps"] = 0
        path = tmp_path/"receipt.json"; write(path, receipt); fold["training_receipt"] = record(path)
    else: declaration["folds"][0]["checkpoint"] = declaration["folds"][1]["checkpoint"]
    source = tmp_path/"bad.json"; write(source, spec)
    with pytest.raises((ValueError, RuntimeError)): workflow.run(source, tmp_path/"output")
    assert not (tmp_path/"output").exists()


def test_changed_losing_arm_checkpoint_invalidates_release(released, registered, native_authority, scoped, monkeypatch):
    patched_controls(monkeypatch, native_authority, scoped)
    path = Path(read(registered[1])["runs"]["G0"]["folds"][0]["checkpoint"]["path"]); original = path.read_bytes()
    try:
        path.write_bytes(original+b"changed losing arm")
        with pytest.raises(ValueError): workflow.validate(record(released), released.parent)
    finally: path.write_bytes(original)


def test_actual_prediction_failure_preserves_failed_stage(registered, native_authority, scoped, monkeypatch, tmp_path):
    patched_controls(monkeypatch, native_authority, scoped)
    spec = deepcopy(read(registered[1])); fold = spec["runs"]["G0"]["folds"][0]
    checkpoint = torch.load(fold["checkpoint"]["path"], map_location="cpu", weights_only=False)
    checkpoint["model_state"].pop(next(iter(checkpoint["model_state"])))
    path = tmp_path/"broken.pt"; torch.save(checkpoint, path); fold["checkpoint"] = record(path)
    receipt = read(fold["training_receipt"]["path"]); receipt["best_model_state_sha256"] = record(path)["sha256"]
    path = tmp_path/"training.json"; write(path, receipt); fold["training_receipt"] = record(path)
    # Preserve the original relative template base when copying this request.
    spec["publication_components"]["external"]["input_manifest"] = record(registered[0]/"external.json")
    source = tmp_path/"broken.json"; write(source, spec)
    with pytest.raises(RuntimeError, match="state_dict"): workflow.run(source, tmp_path/"output")
    state = read(tmp_path/"output/STAGE_STATUS.json")
    assert state["status"] == "FAILED" and state["failed_stage"] == "prediction_G0"
    assert not (tmp_path/"output/SUCCESS.json").exists() and not (tmp_path/"output/publication").exists()
