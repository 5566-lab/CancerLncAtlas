"""Actual stage/write/seal/reassembly/adapter; synthetic checkpoints, no fitting."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import genomic_resource_staging as stage
from cc_hhgt.v32 import genomic_partition_training as trainer
from cc_hhgt.v32.genomic_training import _split_patients
from cc_hhgt.v32.native_auxiliary_fusion import materialize_native_auxiliary, load_native_auxiliary
from test_v32_native_auxiliary_fusion import fixture, read, write, record, no_fitting


def partitioned(root, existing=None, candidate_counts=None, streaming_success=None):
    path, keys = fixture(root, candidate_counts=candidate_counts) if existing is None else (existing, pd.read_parquet(read(existing)["candidate_authority"]["path"]))
    manifest = read(path); native = root/"native"
    original_inputs = read(native/"LINEAGE.json")["input_artifacts"]
    from cc_hhgt.v32.evidence_rule_sources import read_table
    patients = read_table(Path(manifest["patient_folds"]["path"]))
    if streaming_success is not None:
        patients = patients.loc[patients.cancer_id.isin(read(streaming_success)["cohort"]["cancers"])].copy()
    cnv = streaming_success if streaming_success is not None else write(root/"streaming.json", {"format": stage.CNV_STREAMING_FORMAT, "status": "SUCCESS",
        "cancers": {c: {} for c in keys.cancer_id.unique()}, "required_cancers": list(keys.cancer_id.unique()),
        "patient_folds": 5, "typed_unavailable_is_never_zero": True,
        "synthetic_gate_metadata_only_not_a_raw_cnv_validation": True})
    ordered = pd.concat([stage._canonical_partition(keys.loc[keys.cancer_id.eq(c)], c) for c in keys.cancer_id.unique()])
    digest = hashlib.sha256("".join("\t".join(r)+"\n" for r in ordered.itertuples(index=False, name=None)).encode()).hexdigest()
    counts = {c:len(local) for c,local in keys.groupby("cancer_id", sort=False)}
    scope_counts = {"candidate_rows_per_cancer": next(iter(counts.values()))} if len(set(counts.values())) == 1 else {"candidate_rows_by_cancer": counts}
    budget = write(root/"budget.json", {"format": stage.BUDGET_FORMAT, "resource_budget_id": "synthetic-small",
        "scope": {"formal_cancers": list(counts), "candidate_rows": len(keys), **scope_counts},
        "adapter_budget": {"max_candidate_partition_rows": 80, "max_declared_adapter_peak_rss_bytes": 10**12,
                           "max_memmap_bytes": 10**8, "max_staging_disk_bytes": 10**8},
        "downstream_budget": {"max_declared_process_rss_bytes": 10**12, "minimum_host_available_memory_bytes": 1,
                              "minimum_host_free_disk_bytes": 1, "forbid_launch_without_resource_success": True},
        "memmap_layout": {"modalities": ["mutation", "cnv"], "patient_folds": 5,
                          "probability_dtype": "float32", "available_dtype": "bool", "processed_dtype": "bool"},
        "input_bindings": {"candidate_authority": {**manifest["candidate_authority"], "ordered_exact_candidate_key_sha256": digest},
                           "cnv_streaming_aggregate_success": {**record(cnv), "format": stage.CNV_STREAMING_FORMAT,
                               "status": "SUCCESS", "sha256_policy": "CALLER_MUST_PIN_EXACT_64_HEX_AT_LAUNCH"}}})
    stage.stage_genomic_cancer_memmaps(candidate_authority_path=manifest["candidate_authority"]["path"],
        genomic_training_source_path=Path(trainer.__file__).with_name("genomic_training.py"),
        cnv_streaming_success_path=cnv, cnv_streaming_success_sha256=record(cnv)["sha256"],
        budget_contract_path=budget, output_root=root/"stage", enforce_host_budget=True)
    stage_success = root/"stage/SUCCESS.json"
    for fold in range(5):
        for modality in ("mutation", "cnv"):
            for cancer, local in keys.groupby("cancer_id", sort=False):
                local = stage._canonical_partition(local, cancer)
                available = local.cancer_id.eq("BRCA").to_numpy(copy=True)
                if fold == 4: available &= ~local.lncrna_id.eq("LNC:L1").to_numpy()
                values = np.full(len(local), .2+fold*.15, dtype=np.float32); values[~available] = np.nan
                if fold == 0: values[local.lncrna_id.eq("LNC:L0") & available] = 0.
                stage.write_genomic_memmap_partition(staging_success_path=stage_success, modality=modality,
                    patient_fold=fold, cancer_id=cancer, candidate_keys=local, probability=values, available=available)
    stage.seal_genomic_memmap_predictions(staging_success_path=stage_success)
    ready = root/"stage/PREDICTIONS_READY.json"
    stage.reassemble_genomic_memmap_outputs(predictions_ready_path=ready, output_root=root/"reassembled",
        training_run_id=manifest["native_run_id"])
    checkpoints = read(native/"CHECKPOINT_MANIFEST.json"); traces = []
    for fold in range(5):
        for modality in ("mutation", "cnv"):
            trace = trainer._patient_split_trace({s: _split_patients(patients, fold, s) for s in ("train", "validation", "test")}, fold, modality)
            traces.append(trace)
            item = next(r for r in checkpoints["records"] if (r["modality"], r["patient_fold"]) == (modality, fold))
            cp = Path(item["path"]); value = torch.load(cp, weights_only=False)
            value["split_policy"] = trace; torch.save(value, cp)
            item.update(record(cp), checkpoint_selection_split="validation", scaler_fit_split="train",
                        test_rows_consumed_by_fit=0, source_checkpoint_sha256=None)
    policy = {"format": trainer.SPLIT_POLICY_FORMAT, "outer_test_policy": "PREDICTION_ONLY_AFTER_VALIDATION_CHECKPOINT_SELECTION",
        "scaling_fit": "TRAIN_PATIENTS_ONLY", "optimizer_fit": "TRAIN_PATIENTS_ONLY", "early_stopping": "VALIDATION_PATIENTS_ONLY",
        "checkpoint_choice": "VALIDATION_PATIENTS_ONLY", "test_keys_or_labels_enter_scaling_early_stop_or_checkpoint": False, "traces": traces}
    core_path = Path(manifest["core_manifest"]["path"]); exports = {}
    for fold in read(core_path)["folds"].values():
        for node_type in ("lncRNA", "pathway"):
            item = fold["exports"][node_type]; p = Path(item["path"])
            exports[str(p if p.is_absolute() else core_path.parent/p)] = item["sha256"]
    lineage = {"format": trainer.TRAINING_FORMAT, "status": "TRAINING_COMPLETE_PREDICTIONS_SEALED_NOT_RELEASED",
        "training_run_id": manifest["native_run_id"], "patient_folds": 5, "modalities": ["mutation", "cnv"],
        "checkpoint_count": 10, "global_cross_cancer_heads": True, "core_parameters_before_sha256": exports,
        "core_parameters_after_sha256": exports, "core_manifest_sha256": manifest["core_manifest"]["sha256"],
        "split_policy": policy, "predictions_ready_path": str(ready), "predictions_ready_sha256": record(ready)["sha256"],
        "resource_stage_success_path": str(stage_success), "resource_stage_success_sha256": record(stage_success)["sha256"],
        "ordered_exact_candidate_key_sha256": digest, "fixture_no_optimizer_run": True}
    if streaming_success is not None:
        lineage.update(cohort=read(cnv)["cohort"],core_run_id=manifest["core_run_id"],configuration={"cnv_event_threshold":.30})
    success = {"format": trainer.TRAINING_FORMAT, "status": "TRAINING_SUCCESS_PREDICTIONS_SEALED_NOT_RELEASED",
        "training_run_id": manifest["native_run_id"], "lineage_path": str(native/"LINEAGE.json"),
        "checkpoint_manifest_path": str(native/"CHECKPOINT_MANIFEST.json"), "predictions_ready_path": str(ready),
        "predictions_ready_sha256": record(ready)["sha256"], "checkpoint_count": 10, "release_ready": False,
        "formal_genomic_oof_success_emitted": False, "fixture_no_optimizer_run": True}
    trainer.write_partitioned_completion(native, native, source_audit={"status": "PASS", "artifacts": [
        {**item, "status": "PASS", "source_role": "split_manifest" if item["sha256"]==manifest["patient_folds"]["sha256"] else "standardized_input"}
        for item in original_inputs]}, split_policy=policy, checkpoint_manifest=checkpoints,
        resource_audit={"status": "RESOURCE_BUDGET_PASS", "fixture_no_optimizer_run": True}, lineage=lineage, training_success=success)
    manifest["native_receipt"] = record(native/"TRAINING_SUCCESS.json")
    manifest["reassembly_receipt"] = record(root/"reassembled/SUCCESS.json")
    write(path, manifest)
    return path, keys


def repin_training(path):
    manifest = read(path); success_path = Path(manifest["native_receipt"]["path"]); success = read(success_path)
    lp = Path(success["lineage_path"]); lineage = read(lp)
    for name, item in lineage["completion_records"].items():
        lineage["completion_records"][name] = record(Path(item["path"]))
    write(lp, lineage); success["lineage_sha256"] = record(lp)["sha256"]
    success["checkpoint_manifest_sha256"] = lineage["completion_records"]["CHECKPOINT_MANIFEST.json"]["sha256"]
    write(success_path, success); manifest["native_receipt"] = record(success_path); write(path, manifest)


@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("variable", [False, True])
def test_partitioned_completed_outputs_reach_actual_native_and_fusion_consumers(tmp_path, relative, variable):
    counts = {"BRCA":73, "LAML":19} if variable else None
    path, keys = partitioned(tmp_path, candidate_counts=counts)
    if relative:
        manifest = read(path)
        for key in ("native_receipt", "reassembly_receipt", "candidate_authority", "patient_folds", "core_manifest"):
            manifest[key]["path"] = str(Path(manifest[key]["path"]).relative_to(tmp_path))
        write(path, manifest)
    result = materialize_native_auxiliary(path, tmp_path/"adapter", expected_core_run_id="current-G2")
    frame, audit = load_native_auxiliary(result, expected_sha=record(result)["sha256"], expected_core_run_id="current-G2")
    assert len(frame) == len(keys)
    if variable:
        assert frame.groupby("cancer_id").size().to_dict() == counts
    assert audit["native_status"] == "TRAINING_SUCCESS_PREDICTIONS_SEALED_NOT_RELEASED"
    assert not audit["training_started"] and not audit["native_model_forwards_replayed"]
    assert frame.loc[frame.cancer_id.eq("LAML"), "mutation_context_probability"].isna().all()
    partial = frame.loc[frame.cancer_id.eq("BRCA") & frame.lncrna_id.eq("LNC:L1")].iloc[0]
    assert partial.mutation_patient_folds_with_prediction == 4
    import scripts.run_v32_current_multimodal_fusion as runner
    primary = tmp_path/"primary.parquet"
    from cc_hhgt.v32.multimodal_fusion import PRIMARY_COLUMN, ANALYSIS_VERSION
    keys.assign(**{PRIMARY_COLUMN: .25, "analysis_version": ANALYSIS_VERSION}).to_parquet(primary, index=False)
    folds = []
    for fold in range(5):
        p = tmp_path/f"labels_{fold}.parquet"
        keys.assign(patient_fold_id=fold, held_out_proxy_label=fold%2).to_parquet(p, index=False)
        folds.append({"fold_id": fold, "artifact": record(p)})
    primary_binding = write(tmp_path/"primary.json", {"status": "PASS", "run_id": "current-G2",
        "prediction": record(primary), "fold_prediction_sha256": [f["artifact"]["sha256"] for f in folds]})
    fusion = write(tmp_path/"fusion.json", {"format": "CURRENT_MULTIMODAL_FUSION_MANIFEST_V1", "core_run_id": "current-G2",
        "primary_prediction": record(primary), "primary_binding_audit": record(primary_binding), "primary_folds": folds,
        "experts": read(tmp_path/"adapter/FUSION_EXPERTS.json")})
    assert runner.main(["--manifest", str(fusion), "--expected-core-run-id", "current-G2", "--output", str(tmp_path/"fusion")]) == 0
    assert read(tmp_path/"fusion/PREPARED.json")["status"] == "PREPARED_NOT_TRAINED"
    assert not (tmp_path/"fusion/SUCCESS.json").exists()


def test_partitioned_inputs_remain_visible_in_zero_weight_fusion_model_use(tmp_path):
    from test_v32_auxiliary_model_use import setup
    from cc_hhgt.v32.current_model_use import derive_manifest
    path, raw = setup(tmp_path, "genomic", prepare_native=lambda p: partitioned(tmp_path, p))
    _, _, tables, _ = derive_manifest(path)
    assert tables["usage"].study_tokens_json.str.contains("11112222").any()
    assert record(raw)["sha256"] in set(tables["dependencies"].sha256)
    assert "AUXILIARY_RAW_STUDY_AND_PATIENT_USAGE_NOT_FULLY_RECONSTRUCTED" in set(tables["gaps"].reason)


@pytest.mark.parametrize("fault", ["stage_only", "reassembly_only", "missing_controls", "split", "checkpoint_split", "head_missing", "core_changed", "input_changed", "reassembly_run", "repinned_predictions", "seal_changed"])
def test_partitioned_lifecycle_and_replay_reject_incomplete_or_relabelled_inputs(tmp_path, fault):
    path, _ = partitioned(tmp_path); manifest = read(path)
    native = tmp_path/"native"; lineage = read(native/"LINEAGE.json")
    if fault == "stage_only": manifest["native_receipt"] = record(tmp_path/"stage/SUCCESS.json"); write(path, manifest)
    elif fault == "reassembly_only": manifest["native_receipt"] = manifest["reassembly_receipt"]; write(path, manifest)
    elif fault == "missing_controls":
        lineage.pop("completion_records"); write(native/"LINEAGE.json", lineage)
        success = read(native/"TRAINING_SUCCESS.json"); success["lineage_sha256"] = record(native/"LINEAGE.json")["sha256"]
        write(native/"TRAINING_SUCCESS.json", success); manifest["native_receipt"] = record(native/"TRAINING_SUCCESS.json"); write(path, manifest)
    elif fault == "split":
        policy = read(native/"SPLIT_POLICY.json"); policy["traces"][0]["patient_set_sha256"]["test"] = "0"*64
        write(native/"SPLIT_POLICY.json", policy); lineage["split_policy"] = policy; write(native/"LINEAGE.json", lineage); repin_training(path)
    elif fault in {"checkpoint_split", "head_missing"}:
        cps = read(native/"CHECKPOINT_MANIFEST.json")
        if fault == "head_missing": cps["records"].pop()
        else:
            item = cps["records"][0]; cp = Path(item["path"]); value = torch.load(cp, weights_only=False)
            value["split_policy"]["test_labels_or_features_enter_fit"] = True; torch.save(value, cp); item.update(record(cp))
        write(native/"CHECKPOINT_MANIFEST.json", cps); repin_training(path)
    elif fault == "core_changed":
        lineage["core_parameters_after_sha256"] = {}; write(native/"LINEAGE.json", lineage); repin_training(path)
    elif fault == "input_changed":
        p = Path(manifest["patient_folds"]["path"]); frame = pd.read_parquet(p); frame.loc[0, "patient_id"] += "-changed"; frame.to_parquet(p, index=False)
    elif fault in {"reassembly_run", "repinned_predictions"}:
        lp = tmp_path/"reassembled/LINEAGE.json"; assembled = read(lp)
        if fault == "reassembly_run": assembled["training_run_id"] = "v32-another"
        else:
            for item in [assembled["combined_predictions"], *assembled["patient_fold_predictions"]]:
                p = Path(item["path"]); frame = pd.read_parquet(p)
                for modality in ("mutation", "cnv"):
                    frame.loc[frame[modality+"_available"], modality+"_context_probability"] = np.float32(.9)
                frame.to_parquet(p, index=False); item.update(record(p))
        write(lp, assembled); sp = tmp_path/"reassembled/SUCCESS.json"; success = read(sp)
        success["lineage_sha256"] = record(lp)["sha256"]; success["prediction_sha256"] = assembled["combined_predictions"]["sha256"]
        write(sp, success); manifest["reassembly_receipt"] = record(sp); write(path, manifest)
    elif fault == "seal_changed":
        p = tmp_path/"stage/memmaps/fold_probability.float32.npy"; array = np.load(p, mmap_mode="r+"); array[0, 0, 0] = .99; array.flush(); del array
    with pytest.raises((ValueError, RuntimeError)):
        materialize_native_auxiliary(path, tmp_path/"adapter", expected_core_run_id="current-G2")
