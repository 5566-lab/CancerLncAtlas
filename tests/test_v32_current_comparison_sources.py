"""All G012 competitors reach native source isolation, without optimization."""
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from cc_hhgt.v32.current_model_use import (read, write, record, INPUT_FORMAT, materialize_current_model_use,
                                        load_current_model_use)
from cc_hhgt.v32.g012_winner_lock import (lock_g012_validation_winner, validate_comparison_sources, G012WinnerLockError)


def forbidden(*args, **kwargs): raise AssertionError("No optimization or inference during selection-source inspection")


@pytest.fixture(autouse=True)
def no_optimization(monkeypatch):
    assert socket.gethostname() == "149"
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)


def current_runs(root):
    from test_v32_g012_winner_lock import _fixture
    from test_v32_current_binding_expansion import verified_payload, comparison_config
    from test_v32_training_objective import _graph_binding
    from test_v32_preparation_sources import adjusted_capture
    from scripts.prepare_v32_formal import attach_replication_labels, make_batches
    from cc_hhgt.v32.formal_graph import build_formal_graph_authority, build_variant_runtime_bundle, materialize_symmetric_ppi
    from cc_hhgt.v32.formal_graph_authority import bind_formal_graph_variant
    from cc_hhgt.v32.preparation_sources import attach_sources
    from cc_hhgt.v32.training import _prepared_input_authority_hashes, _runtime_chunk_contract
    from cc_hhgt.v32.current_core_export import CHECKPOINT_FORMAT
    from cc_hhgt.gnn import build_model
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual
    fixture = _fixture(root)
    candidates = pd.DataFrame({"cancer_id": ["BRCA"], "lncrna_id": ["LNC:Z"], "pathway_id": ["P1"], "pathway_family_id": ["F1"]})
    candidates.to_parquet(fixture["candidate"], index=False)
    scope = fixture["prepared_parent"] / "scope.parquet"
    candidates[["cancer_id", "lncrna_id"]].assign(shared_or_local_scope="shared").to_parquet(scope, index=False)
    manifests = {}
    for variant in ("G0", "G1", "G2"):
        run = fixture["run_parent"] / variant; prepared = fixture["prepared_parent"] / variant
        config = comparison_config(); config["task_contract"]["graph_variant"] = variant
        # These are ordinary formal G012 payloads. They do not consume the
        # historical C overlay used by the shared binding-comparison fixture.
        config["task_contract"].pop("rbp_evidence_mode")
        config_path = run / "config.yaml"; config_path.write_text(yaml.safe_dump(config))
        input_path = run / "authorization/INPUT_MANIFEST.json"
        task_path = run / "authorization/TASK_MANIFEST.tsv"
        inputs = []; payloads = []
        for fold in range(5):
            payload = verified_payload(fold); base = payload["bundle"]
            ppi = materialize_symmetric_ppi(pd.DataFrame({"protein_id_a":["UNIPROT:Z"],"protein_id_b":["UNIPROT:A"],"weight":[.5]}))
            authority = build_formal_graph_authority([base.edges, ppi], outer_fold=fold)
            bundle = build_variant_runtime_bundle(authority, variant, edge_chunk_size=2)
            binding = _graph_binding(build_variant_runtime_bundle(authority, "G2", edge_chunk_size=2)); binding["outer_fold"] = fold
            payload.update(bundle=bundle, formal_graph_variant=variant,
                formal_graph_authority=bind_formal_graph_variant(binding, authority, variant),
                artifact_hashes={"source": "a"*64}, input_authority_hashes={"source": "a"*64})
            payload.pop("graph_overlay_authority", None)
            _, source_receipt, frames = adjusted_capture(root / f"source_{variant}_{fold}", fold,
                train_pmid="11112222" if variant == "G0" else "33334444")
            for frame in frames.values():
                frame["shared_or_local_scope"] = "shared"; frame["detection_rate"] = .8; frame["detected_cancers"] = 5
            frames["validation"] = attach_replication_labels(frames["train"], frames["validation"])
            for split in ("train", "validation"):
                payload[split+"_batches"] = make_batches(frames[split], np.zeros(1), bundle, split=split)
            payload["input_scope"]["association_mode"] = "local-cnv-adjusted"
            payload["label_contract"].update(train_direction_label_source="train_local_cnv_adjusted_effect",
                validation_direction_label_source="validation_local_cnv_adjusted_replication_effect",
                association_target="A1_LOCAL", association_available_mask="association_available")
            attach_sources(payload, source_receipt)
            path = prepared / f"PATIENT_FOLD_{fold}.pt"; torch.save(payload, path)
            inputs.append({"fold": fold, **record(path)}); payloads.append(payload)
        write(input_path, {"fold_inputs": inputs})
        hashes = {"code_sha256": "a"*64, "config_sha256": record(config_path)["sha256"],
            "input_manifest_sha256": record(input_path)["sha256"], "task_manifest_sha256": record(task_path)["sha256"]}
        manifest = {"format": "CURRENT_CORE_PREDICTION_INPUT_V1", "core_run_id": "fresh-"+variant, "graph_variant": variant,
                    "candidate_authority": record(fixture["candidate"]), "lncrna_scope": record(scope), "fold_inputs": []}
        for fold, (item, payload) in enumerate(zip(inputs, payloads)):
            task_id = f"fresh-{variant}|PATIENT_FOLD_{fold}|CC-HHGT|20260726"
            task = run / "training" / task_id.replace("|", "__")
            success_path = task / "SUCCESS.json"; previous = read(success_path)
            torch.manual_seed(fold+50)
            encoder = build_model("cc_hhgt", payload["bundle"], payload["feature_dim"], payload["legacy_model_config"])
            model = build_v32_cc_hhgt_residual(encoder, hidden_channels=8, context_features=4, dropout=0.)
            checkpoint = {"checkpoint_format": CHECKPOINT_FORMAT, "model_state": model.state_dict(), "patient_fold": fold,
                "task_id": task_id, "graph_variant": variant, "artifact_hashes": hashes,
                "input_authority_hashes": _prepared_input_authority_hashes(payload), "prepared_artifact_sha256": item["sha256"], "cycle": 0}
            path = task / "best_model_state.pt"; torch.save(checkpoint, path)
            chunks, weights = _runtime_chunk_contract(payload["bundle"])
            previous.update({k: v for k, v in checkpoint.items() if k != "model_state"})
            previous.update(run_id="fresh-"+variant, best_model_state_sha256=record(path)["sha256"],
                optimizer_steps=1, parameter_delta_positive=True, validation_full_runtime_coverage=True,
                runtime_chunk_weights={str(c): weights[c] for c in chunks}, runtime_num_chunks=len(chunks),
                synthetic_serialization_fixture=True, no_optimizer_actually_run=True)
            write(success_path, previous)
            manifest["fold_inputs"].append({"fold": fold, "prepared": {k:item[k] for k in ("path", "sha256")},
                "checkpoint": record(path), "training_receipt": record(success_path),
                "config": record(config_path), "input_manifest": record(input_path)})
        path = run / "CURRENT_INPUT.json"; write(path, manifest); manifests[variant] = path
    return fixture, manifests


@pytest.fixture(scope="module")
def comparison_release(tmp_path_factory):
    from cc_hhgt.v32.current_core_prediction import materialize_current_predictions
    from cc_hhgt.v32.selected_score import materialize_selected_scores
    root = tmp_path_factory.mktemp("all_comparison_sources")
    with pytest.MonkeyPatch.context() as patch:
        for name in ("Adam", "AdamW", "SGD"): patch.setattr(torch.optim, name, forbidden)
        fixture, manifests = current_runs(root)
        primary = materialize_current_predictions(manifests["G1"], root / "prediction", expected_run_id="fresh-G1")
        import scripts.export_v32_current_core_embeddings as exporter
        patch.setattr(exporter, "_build_model", forbidden)
        result = lock_g012_validation_winner(prepared_parent=fixture["prepared_parent"], run_parent=fixture["run_parent"],
            candidate_authority_path=fixture["candidate"], output_root=root / "selection", pair_fold_seed=13)
        lock = Path(result["winner_declaration"]["path"])
        spec = {"model_id":"primary", "producer_kind":"current_core_release", "producer_binding":record(primary),
                "run_id":"fresh-G1", "score_column":"association_membership_probability"}
        policy = root / "SELECTED_INPUT.json"
        write(policy, {"format":"CURRENT_SELECTED_ANALYSIS_INPUT_V1", "selected_model":spec, "reference_exact_model":spec,
                       "selection":{"kind":"G012_VALIDATION_WINNER", "binding":record(lock)}})
        selected = materialize_selected_scores(policy, root / "selected")
        models = [{"model_id":"selected", "producer_kind":"selected_release", "producer_binding":record(selected),
                   "run_id":"fresh-G1", "score_column":"selected_score"}]
        source = root / "INVENTORY_INPUT.json"; write(source, {"format":INPUT_FORMAT, "models":models})
        inventory = materialize_current_model_use(source, root / "inventory")
    return root, lock, policy, models, Path(inventory["binding_path"])


def test_all_native_competitors_and_their_losing_study_are_retained(comparison_release, monkeypatch):
    import scripts.export_v32_current_core_embeddings as exporter
    monkeypatch.setattr(exporter, "_build_model", forbidden)
    _, lock, _, models, path = comparison_release
    assert read(lock)["winner_id"] == "G1"
    assert set(validate_comparison_sources(read(lock), lock.parent)) == {"G0", "G1", "G2"}
    binding, tables = load_current_model_use(path, expected_sha=record(path)["sha256"], models=models)
    use = tables["usage"]
    assert set(use.dependency_context) == {"SELECTED_SCORE", "G012_COMPARISON:G0", "G012_COMPARISON:G1", "G012_COMPARISON:G2"}
    selected = use.loc[use.dependency_context.eq("SELECTED_SCORE")]
    assert not selected.study_tokens_json.str.contains("11112222").any()
    assert set(use.loc[use.study_tokens_json.str.contains("11112222"), "dependency_context"]) == {"G012_COMPARISON:G0"}
    for context in ("G012_COMPARISON:G0", "G012_COMPARISON:G1", "G012_COMPARISON:G2"):
        dependencies = tables["dependencies"].loc[tables["dependencies"].dependency_context.eq(context)]
        assert set(dependencies.loc[dependencies.role.eq("checkpoint"), "patient_fold_id"]) == set(range(5))
    assert "ALL_COMPETITOR_TRAINING_DEPENDENCIES_NOT_RECONSTRUCTED" not in set(tables["gaps"].reason)
    assert "RAW_FEATURE_LABEL_STUDY_LINEAGE_NOT_RECONSTRUCTED" in set(tables["gaps"].reason)
    assert not binding["model_use_inventory_completeness_verified"]


@pytest.mark.parametrize("fault", ["omitted_arm", "omitted_fold", "loss", "checkpoint_alias", "receipt_hash", "receipt_content"])
def test_comparison_source_reader_rejects_loser_omissions_and_rehashed_losses(comparison_release, tmp_path, fault):
    _, path, _, _, _ = comparison_release
    lock = read(path); run = lock["compared_runs"]["G0"]
    if fault == "omitted_arm": del lock["compared_runs"]["G0"]
    elif fault == "omitted_fold": run["checkpoints"].pop()
    elif fault == "loss": lock["selection_table"][0]["fold_validation_losses"][0] += .1
    elif fault == "checkpoint_alias": lock["checkpoints"][0]["sha256"] = "f"*64
    else:
        receipt = read(run["checkpoints"][0]["training_receipt"]["path"])
        receipt["best_validation_loss"] += .05
        changed = tmp_path / "SUCCESS.json"; write(changed, receipt)
        if fault == "receipt_hash": run["checkpoints"][0]["training_receipt"]["path"] = str(changed)
        else: run["checkpoints"][0]["training_receipt"] = record(changed)
    from cc_hhgt.v32.sealed_test_inference import SealedTestInferenceError
    with pytest.raises((ValueError, G012WinnerLockError, SealedTestInferenceError)):
        validate_comparison_sources(lock, path.parent)


def test_selected_consumer_replays_losing_model_receipts(comparison_release, tmp_path):
    from cc_hhgt.v32.selected_score import prepare_selected_scores
    _, path, policy, _, _ = comparison_release
    lock = read(path); lock["selection_table"][0]["mean_validation_loss"] += .01
    altered = tmp_path / "LOCK.json"; write(altered, lock)
    manifest = read(policy); manifest["selection"]["binding"] = record(altered)
    source = tmp_path / "SELECTED.json"; write(source, manifest)
    with pytest.raises(G012WinnerLockError, match="actual training receipts"):
        prepare_selected_scores(source)


def test_external_v2_excludes_a_study_seen_only_by_the_losing_model(comparison_release, tmp_path):
    from cc_hhgt.v32.external_validation_current import materialize_current_external, CurrentExternalValidationQuery
    _, _, _, models, inventory = comparison_release
    crosswalk = tmp_path / "crosswalk.parquet"
    pd.DataFrame([("lncrna","new","LNC:NEW"),("cancer","breast","BRCA"),("pathway","p","P1")],
                 columns=["namespace","raw_id","canonical_id"]).to_parquet(crosswalk,index=False)
    observed = tmp_path / "observed.parquet"
    pd.DataFrame({"record_id":["synthetic"],"lncrna":["new"],"cancer":["breast"],"pathway":["p"],
        "label":[1],"is_experimental":[True],"is_predicted":[False],"assay":["Synthetic assay"],"pmid":["11112222"]}).to_parquet(observed,index=False)
    source = tmp_path / "INPUT.json"
    write(source,{"format":"CURRENT_EXTERNAL_VALIDATION_INPUT_V2","models":models,"model_use_inventory":record(inventory),
        "identity_crosswalk":record(crosswalk),"external_sources":[{"kind":"exact_assay","source_id":"loser_study","table":record(observed)}],
        "evaluation":{"primary_model":"selected","cutoffs":[1],"lncrna_aggregation":"MAX","bootstrap_replicates":0,"seed":0}})
    result = materialize_current_external(source,tmp_path / "external")
    query = CurrentExternalValidationQuery(result["binding_path"],expected_sha=result["binding_sha256"])
    rows = query.query(table="isolation")["rows"]
    assert rows and all(r["study_overlap"] and not r["lncrna_overlap"] and not r["provenance_held_out"] for r in rows)
    dependencies = query.query(table="model_dependencies",filters={"dependency_context":"G012_COMPARISON:G0","role":"checkpoint"})["rows"]
    assert {r["patient_fold_id"] for r in dependencies} == set(range(5))
    assert all("path" not in r for r in dependencies)
    usage = query.query(table="model_use",filters={"dependency_context":"G012_COMPARISON:G0","use_view":"PREPARATION_SOURCE_METADATA"})["rows"]
    assert usage and any("11112222" in r["study_tokens_json"] for r in usage)


def test_historical_winner_only_lock_remains_explicitly_incomplete(comparison_release, tmp_path):
    from cc_hhgt.v32.current_model_use import Inventory
    _, path, _, _, _ = comparison_release
    legacy = read(path); legacy.pop("comparison_sources_format"); legacy.pop("compared_runs")
    assert validate_comparison_sources(legacy, path.parent) is None
    altered = tmp_path / "OLD_LOCK.json"; write(altered, legacy)
    inventory = Inventory([], tmp_path); inventory.comparison(altered, "selected")
    assert inventory.gaps[0]["reason"] == "ALL_COMPETITOR_TRAINING_DEPENDENCIES_NOT_RECONSTRUCTED"
