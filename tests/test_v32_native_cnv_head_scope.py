"""Native association closeout -> real CNV features/labels -> consumers, no fit."""
from contextlib import ExitStack
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import cnv_only_training_v2 as head
from cc_hhgt.v32.cnv_head_association import association_contract
from cc_hhgt.v32.cnv_input_sources import validate_consumed_sources
from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.native_auxiliary_fusion import Sources, materialize_native_auxiliary, load_native_auxiliary
from test_v32_local_cnv_cohort import fixture_inputs, close_train, CANCERS, LNCS, PATHWAY
from test_v32_current_core_export import export_fixture
from test_v32_native_auxiliary_fusion import fixture as frozen_native
import scripts.run_v32_local_cnv_confounding_audit_r2 as association

RUN = "SYNTHETIC_NATIVE_CNV_HEAD_SCOPE"


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    def forbidden(*args, **kwargs): raise AssertionError("No optimizer or auxiliary forward permitted")
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    monkeypatch.setattr(head, "_fit_head", forbidden)
    monkeypatch.setattr(head, "_predict_head", forbidden)


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    root = tmp_path_factory.mktemp("native_cnv_head")
    args = fixture_inputs(root/"inputs", True)
    aggregate = read(args.directional_root/"SUCCESS.json")
    for cancer in CANCERS:
        path = args.directional_root/f"cancer={cancer}"/"SUCCESS.json"
        value = read(path); value.update(signed_values_preserved=True,
            amplification_deletion_separate=True, absolute_burden_used_as_continuous=False)
        write(path, value)
        aggregate["cancers"][cancer].update(path=str(path.parent), success_sha256=record(path)["sha256"])
    write(args.directional_root/"SUCCESS.json", aggregate)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(association, "RUN_ID", RUN)
        for cancer in CANCERS:
            args.cancer = cancer; association.run_cancer(args)
        close_train(args)
    core_root = root/"core"; core_root.mkdir()
    core_path, core = export_fixture(core_root)
    for fold in core["folds"].values():
        for kind, ids in (("lncRNA", LNCS), ("pathway", [PATHWAY])):
            p = core_root/fold["exports"][kind]["path"]
            pd.DataFrame({"node_index": range(len(ids)), "node_id": ids,
                "core_feature_000": np.arange(len(ids), dtype=float), "core_feature_001": 1.}).to_parquet(p, index=False)
            fold["exports"][kind] = {**record(p), "rows": len(ids), "features": 2}
    core["synthetic_serialization_no_optimizer"] = True; write(core_path, core)
    return args, core_path


def prepare(native):
    args, core = native
    return head.prepare_directional_inputs(candidates_path=args.candidates, folds_path=args.patient_map,
        directional_root=args.directional_root, association_root=args.output_root, core_manifest_path=core)


def collect_all(prepared):
    for fold in range(5):
        core = head.load_fold_core_embeddings(prepared["core_manifest_path"], prepared["manifest"], fold)
        values = head.collect_directional_examples(prepared, core, fold, head.GenomicTrainingConfig())
        assert set(values[2]) == set(values[5]) == {0., 1.}
        assert values[0].shape[1] == 4 and values[1].shape[1] == len(head.CNV_DOMAIN_FEATURES)
        assert all(np.isfinite(v).all() for v in values)


def test_actual_native_scope_preflight_and_all_five_fold_samples(native, tmp_path):
    args, core = native; prepared = prepare(native)
    assert prepared["cancers"] == tuple(CANCERS) and len(prepared["candidates"]) == 6
    assert prepared["authority"]["run_id"] == RUN != head.EXPECTED_ASSOCIATION_RUN_ID
    collect_all(prepared)
    local = prepared["candidates"].loc[prepared["candidates"].cancer_id.eq("BRCA")]
    for split in ("train", "validation"):
        labels = head._formal_labels(args.output_root, "BRCA", 0, local, split, authority=prepared["authority"])
        assert np.isnan(labels[local.lncrna_id.eq("LNC:LOW").to_numpy()]).all()
    source = prepared["capture"].write(tmp_path/"CONSUMED.json", kind="cnv", run_id="v32-synthetic-native")
    _, consumed = validate_consumed_sources(Sources(), source, tmp_path, kind="cnv", run_id="v32-synthetic-native",
        candidates=record(args.candidates), folds=record(args.patient_map), core=record(core))
    assert sum(r["role"] == "association_control" for r in consumed["records"]) == 5
    from scripts.preflight_v32_directional_cnv_head import main
    output = tmp_path/"HEAD_PREFLIGHT.json"
    assert main(["--candidates", str(args.candidates), "--patient-folds", str(args.patient_map),
        "--directional-root", str(args.directional_root), "--association-root", str(args.output_root),
        "--core-manifest", str(core), "--output", str(output)]) == 0
    report = read(output)
    assert report["candidate_rows"] == 6 and report["cancers"] == CANCERS and len(report["core_folds"]) == 5
    assert report["training_started"] is False and report["target_host"] == "149"
    assert report["sampling_config"]["min_pair_callable"] == head.GenomicTrainingConfig().min_pair_callable


def frozen_package(native, tmp_path):
    args, core = native; prepared = prepare(native); collect_all(prepared)
    path, _ = frozen_native(tmp_path/"frozen", "cnv", core_override=(core, prepared["candidates"], "current-G2"))
    manifest = read(path); receipt_path = Path(manifest["native_receipt"]["path"]); receipt = read(receipt_path)
    manifest["candidate_authority"] = record(args.candidates); manifest["patient_folds"] = record(args.patient_map)
    receipt.update(candidate_authority_sha256=record(args.candidates)["sha256"],
        fold_manifest_sha256=record(args.patient_map)["sha256"], cohort=prepared["cohort"],
        association_run_id=RUN, cancers=list(CANCERS))
    receipt["consumed_sources"] = prepared["capture"].write(tmp_path/"CONSUMED.json", kind="cnv", run_id=manifest["native_run_id"])
    for item in receipt["predictions"]:
        p = Path(item["path"]); frame = pd.read_parquet(p)
        frame.candidate_authority_sha256 = manifest["candidate_authority"]["sha256"]
        frame.fold_manifest_sha256 = manifest["patient_folds"]["sha256"]
        frame.to_parquet(p, index=False); item.update(record(p))
    write(receipt_path, receipt); manifest["native_receipt"] = record(receipt_path); write(path, manifest)
    return path


def test_actual_native_scope_consumed_by_frozen_auxiliary_adapter(native, tmp_path):
    path = frozen_package(native, tmp_path)
    result = materialize_native_auxiliary(path, tmp_path/"adapter", expected_core_run_id="current-G2")
    data, audit = load_native_auxiliary(result, expected_sha=record(result)["sha256"], expected_core_run_id="current-G2")
    assert len(data) == 6 and audit["fold_rows"] == 30 and not audit["training_started"]


def auditable_package(native, tmp_path):
    args, _ = native; path = frozen_package(native, tmp_path)
    manifest = read(path); rp = Path(manifest["native_receipt"]["path"]); receipt = read(rp)
    # The auditor inspects completed checkpoint containers. These are explicit
    # serialization/telemetry fixtures, never evidence that fitting happened.
    checkpoints = {}
    for item in receipt["checkpoints"]:
        cp = Path(item["path"]); value = torch.load(cp, weights_only=False)
        value.update(optimizer_step_count=1, initialization={"initial_parameter_sha256": "a"*64},
            final_parameter_sha256="b"*64, old_checkpoint_loaded=False, mutation_features_used=False,
            synthetic_telemetry_no_optimizer_actually_run=True)
        torch.save(value, cp); item.update(record(cp)); checkpoints[item["fold"]] = item["sha256"]
    patient_map = pd.read_csv(args.patient_map, sep="\t")
    for item in receipt["predictions"]:
        p = Path(item["path"]); frame = pd.read_parquet(p); fold = item["fold"]
        patients = patient_map.loc[patient_map.cancer_id.eq(item["cancer"])]
        probability = frame.cnv_probability.to_numpy()
        clipped = np.clip(probability, 1e-7, 1-1e-7)
        frame["cnv_logit"] = np.log(clipped/(1-clipped))
        frame["checkpoint_sha256"] = checkpoints[fold]
        frame["lncrna_local_cnv_available"] = frame.cnv_available
        frame["pathway_cnv_available"] = frame.cnv_available
        frame["local_cnv_feature_count"] = 1; frame["pathway_cnv_feature_count"] = 1
        frame["train_patient_n"] = int((~patients.patient_fold_id.isin([fold, (fold+1)%5])).sum())
        frame["validation_patient_n"] = int(patients.patient_fold_id.eq((fold+1)%5).sum())
        frame["oof_patient_n"] = int(patients.patient_fold_id.eq(fold).sum())
        frame["model_version"] = "V3.2"; frame["checkpoint_id"] = f"cnv_patient_fold_{fold}"
        frame["directional_cnv_success_sha256"] = frame["cnv_mapping_sha256"] = record(args.directional_root/"SUCCESS.json")["sha256"]
        frame["association_audit_success_sha256"] = record(args.output_root/"SUCCESS.json")["sha256"]
        frame.to_parquet(p, index=False); item.update(record(p))
    receipt.update(sealed_test_read=False, old_checkpoint_loaded=False)
    write(rp, receipt)
    write(rp.parent/"PACKAGE_READY.json", {"status": "READY_FOR_INDEPENDENT_AUDIT",
        "training_run_id": receipt["training_run_id"], "cohort": receipt["cohort"], "oof_manifest_sha256": record(rp)["sha256"]})
    (rp.parent/"PRIVATE_EVALUATION").mkdir()
    return path


def test_independent_auditor_uses_same_declared_scope_and_explicit_run(native, tmp_path):
    from scripts.audit_v32_directional_cnv_head_oof import main
    args, _ = native; path = auditable_package(native, tmp_path)
    rp = Path(read(path)["native_receipt"]["path"]); receipt = read(rp)
    argv = ["--root", str(rp.parent), "--candidates", str(args.candidates), "--patient-folds", str(args.patient_map),
        "--association-root", str(args.output_root)]
    with pytest.raises(ValueError, match="expected-training-run-id"): main(argv)
    assert not (rp.parent/"SUCCESS.json").exists()
    main([*argv, "--expected-training-run-id", receipt["training_run_id"]])
    success = read(rp.parent/"SUCCESS.json"); audit = read(rp.parent/"CNV_OOF_INDEPENDENT_AUDIT.json")
    assert success["prediction_file_count"] == 10 and success["cohort"] == receipt["cohort"]
    assert audit["checks"]["candidate_rows"] == 30
    assert len(pd.read_csv(rp.parent/"PRIVATE_EVALUATION/cnv_oof_metrics_cancer_fold.tsv", sep="\t")) == 10


def test_independent_oof_labels_use_adjacent_validation_and_reject_unknowns(tmp_path):
    from scripts.audit_v32_directional_cnv_head_oof import load_oof_labels
    write(tmp_path/"SUCCESS.json", {"status": "SUCCESS"})
    keys = pd.DataFrame({"lncrna_id": ["A", "B"], "pathway_id": ["P", "P"]})
    for fold in range(5):
        p = tmp_path/"pair_level/cancer=BRCA"/f"fold={fold}"/"part.parquet"; p.parent.mkdir(parents=True)
        keys.assign(label_A0_validation=["strong_positive" if fold%2 else "unlabeled", "unavailable"],
            run_id=head.EXPECTED_ASSOCIATION_RUN_ID).to_parquet(p, index=False)
    for fold in range(5):
        labels = load_oof_labels(tmp_path, fold, "BRCA", keys)
        assert labels[0] == (fold-1)%5%2 and np.isnan(labels[1])
    p = tmp_path/"pair_level/cancer=BRCA/fold=1/part.parquet"
    keys.assign(label_A0_validation=["unexpected", "unavailable"], run_id=head.EXPECTED_ASSOCIATION_RUN_ID).to_parquet(p, index=False)
    with pytest.raises(RuntimeError, match="unknown classes"): load_oof_labels(tmp_path, 2, "BRCA", keys)


@pytest.mark.parametrize("fault", ["removed_scope", "worker_run", "changed_label", "wrong_candidates", "wrong_patients", "directional_scope", "missing_control"])
def test_native_scope_cannot_be_inferred_or_relabelled(native, tmp_path, fault):
    args, core = native
    with ExitStack() as restore:
        def change(path, value):
            original = Path(path).read_bytes(); restore.callback(Path(path).write_bytes, original); write(path, value)
        if fault == "removed_scope":
            p = args.output_root/"SUCCESS.json"; value = read(p); value.pop("cohort"); change(p, value)
        elif fault == "worker_run":
            p = args.output_root/"receipts/BRCA.json"; value = read(p); value["run_id"] = "another"; change(p, value)
            lp = args.output_root/"LINEAGE.json"; lineage = read(lp); lineage["receipt_sha256"]["BRCA"] = record(p)["sha256"]; change(lp, lineage)
            sp = args.output_root/"SUCCESS.json"; success = read(sp); success["lineage_sha256"] = record(lp)["sha256"]; change(sp, success)
        elif fault == "changed_label":
            p = args.output_root/"pair_level/cancer=BRCA/fold=0/part.parquet"
            original = p.read_bytes(); restore.callback(p.write_bytes, original)
            frame = pd.read_parquet(p); frame["label_A0_full"] = "unlabeled"; frame.to_parquet(p, index=False)
        elif fault == "directional_scope":
            p = args.directional_root/"SUCCESS.json"; value = read(p); value["cancers"].pop("LUAD"); change(p, value)
        elif fault == "missing_control":
            p = args.output_root/"LINEAGE.json"; value = read(p); value["receipt_sha256"].pop("BRCA"); change(p, value)
            sp = args.output_root/"SUCCESS.json"; success = read(sp); success["lineage_sha256"] = record(p)["sha256"]; change(sp, success)
        elif fault in {"wrong_candidates", "wrong_patients"}:
            candidate = tmp_path/"candidates.parquet"; candidate.write_bytes(args.candidates.read_bytes())
            patients = tmp_path/"patients.tsv"; patients.write_bytes(args.patient_map.read_bytes())
            with pytest.raises(ValueError, match="input differs"):
                association_contract(args.output_root,
                    candidates=candidate if fault=="wrong_candidates" else args.candidates,
                    patient_map=patients if fault=="wrong_patients" else args.patient_map)
            return
        with pytest.raises((ValueError, RuntimeError)):
            prepare(native)


def test_legacy_run_identity_is_not_inferred_from_arbitrary_success(tmp_path):
    write(tmp_path/"SUCCESS.json", {"status": "SUCCESS", "run_id": "another"})
    result = association_contract(tmp_path)
    assert result["cohort"] is None and result["run_id"] == head.EXPECTED_ASSOCIATION_RUN_ID
