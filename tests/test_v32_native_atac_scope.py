"""Real R materializer and trainer preparation; frozen result fixtures, no fit."""
from contextlib import ExitStack
from pathlib import Path
import socket
import subprocess

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import atac_training as atac
from cc_hhgt.v32.atac_materialization import run_atac_materialization, INPUT_CONTRACT_FORMAT
from cc_hhgt.v32.atac_cohort import replay
from cc_hhgt.v32.fold_local_scope import candidate_frame, CANDIDATES
from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.native_auxiliary_fusion import INPUT_FORMAT, materialize_native_auxiliary, load_native_auxiliary

ROOT = Path(__file__).resolve().parents[1]
RUN = "SYNTHETIC_ATAC_FROZEN_SERIALIZATION_NO_FIT"
CANCERS = ["BRCA", "OV"]


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    def forbidden(*args, **kwargs): raise AssertionError("No optimizer or auxiliary prediction allowed")
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    monkeypatch.setattr(atac, "_fit_logistic", forbidden)
    monkeypatch.setattr(atac, "_predict_logistic", forbidden)


@pytest.fixture(scope="module")
def materialized(tmp_path_factory):
    root = tmp_path_factory.mktemp("native_atac_scope")
    write(root/"LAUNCH_PREFLIGHT.json", {"target_host": "149", "training_started": False,
        "operation": "SYNTHETIC_R_MATERIALIZATION_AND_NO_FIT_PREPARATION"})
    lncs = [f"A{i:03}" for i in range(120)]; genes = ["G1", "G2", "G3"]
    annotation = root/"annotation.tsv"
    pd.DataFrame({"ensembl_gene_id": lncs+genes, "gene_type": ["lncRNA"]*120+["protein_coding"]*3}).to_csv(annotation, sep="\t", index=False)
    gmt = root/"hallmark.gmt"; gmt.write_text("P1\tdescription\tG1\tG2\tG3\n")
    gm = root/"gmt.json"; write(gm, {"format": "SSGSEA_GMT_INPUT_V1", "collections": [{"collection": "HALLMARK", "file": record(gmt)}]})
    definition = {"format": CANDIDATES, "target_host": "149", "training_started": False,
        "measured_detection_used": False, "annotation": record(annotation), "gmt_manifest": record(gm),
        "cancers": CANCERS, "budget_per_cancer": 120, "seed": 20260726,
        "implementations": [record(ROOT/p) for p in ("cc_hhgt/v32/fold_local_scope.py",
            "scripts/materialize_fold_local_scope.py", "scripts/prepare_v32_formal.py")]}
    candidate = root/"candidates.parquet"; candidate_frame(definition, root).to_parquet(candidate, index=False)
    definition["output"] = record(candidate); write(root/"CANDIDATE_SOURCES.json", definition)
    patients = root/"patients.tsv"
    pd.DataFrame([(c, f"TCGA-{c[:2]}-{i:04}", i%5) for c in CANCERS for i in range(20)],
        columns=["cancer_id", "patient_id", "patient_fold_id"]).to_csv(patients, sep="\t", index=False)
    membership = root/"membership.parquet"
    pd.DataFrame({"pathway_id": "MSIGDB:HALLMARK:P1", "gene_id": genes}).to_parquet(membership, index=False)
    mapping = []
    for cancer in atac.RAW_COVERED_CANCERS:
        n = 20 if cancer == "BRCA" else 1
        for i in range(n):
            mapping.append({"bam_prefix": f"{cancer}-{i:03}", "aliquot_id": f"TCGA-{cancer[:2]}-{i:04}-01A",
                "Case_ID": f"TCGA-{cancer[:2]}-{i:04}-01A"})
    for index, aliquot in ((20, "01A"), (21, "02A")):
        mapping.append({"bam_prefix": f"BRCA-{index:03}", "aliquot_id": f"TCGA-BR-0000-{aliquot}", "Case_ID": f"TCGA-BR-0000-{aliquot}"})
    mp = root/"mapping.tsv"
    pd.DataFrame(mapping).assign(stanfordUUID=[f"SYNTHETIC-{i}" for i in range(len(mapping))]).to_csv(mp, sep="\t", index=False)
    promoters = root/"promoters.bed.gz"
    pd.DataFrame({"chrom": "chr1", "start": np.arange(123)*20, "end": np.arange(123)*20+10,
        "gene_id": lncs+genes}).to_csv(promoters, sep="\t", index=False, compression="gzip")
    # Retain the official 562,709-peak and full 23-cancer raw-shape gates.
    # Only the explicit candidate cohort is reduced. No biological data is read.
    generator = root/"make_synthetic.R"
    generator.write_text('''suppressPackageStartupMessages(library(data.table))
args <- commandArgs(TRUE); root <- args[[1]]
m <- fread(file.path(root,"mapping.tsv")); n <- 562709L
x <- data.frame(seqnames=rep("chr1",n),start=(0:(n-1))*20L,end=(0:(n-1))*20L+10L,
                name=paste0("peak",seq_len(n)),score=0,strand="*",extra=0)
for (j in seq_len(nrow(m))) {
  v <- numeric(n); i <- seq_len(123)
  v[i] <- sin(i*.13+j*.29)+cos(i*.07*j)+j*.03
  x[[gsub("-","_",m$bam_prefix[j],fixed=TRUE)]] <- v
}
saveRDS(x,file.path(root,"matrix.rds"),compress=FALSE)
fwrite(x[,c("seqnames","start","end","name")],file.path(root,"peaks.tsv"),sep="\\t")
''')
    subprocess.run(["/usr/bin/Rscript", str(generator), str(root)], check=True, timeout=120)
    paths = {"candidates": candidate, "patient_folds": patients, "pathway_membership": membership,
        "normalized_atac_matrix": root/"matrix.rds", "identifier_mapping": mp,
        "peak_set": root/"peaks.tsv", "promoters": promoters}
    contract = root/"contract.json"
    write(contract, {"format": INPUT_CONTRACT_FORMAT, "contract_id": "SYNTHETIC_NATIVE_ATAC",
        "candidate_definition": record(root/"CANDIDATE_SOURCES.json"),
        "files": [{**record(p), "bytes": p.stat().st_size, "basename": p.name, "role": role} for role,p in paths.items()]})
    result = run_atac_materialization(candidates_path=candidate, patient_folds_path=patients,
        pathway_membership_path=membership, matrix_path=paths["normalized_atac_matrix"], mapping_path=mp,
        peak_set_path=paths["peak_set"], promoters_path=promoters,
        extractor_script_path=ROOT/"scripts/materialize_v32_atac_gene_accessibility.R",
        input_contract_path=contract, output_root=root/"materialized")
    return root, paths, root/"materialized/GENE_ACCESSIBILITY_SUCCESS.json", result


def prepare(native, **kwargs):
    _, paths, success, _ = native
    return atac.prepare_atac_inputs(candidates_path=paths["candidates"], patient_folds_path=paths["patient_folds"],
        pathway_membership_path=paths["pathway_membership"], materialization_success_path=success, **kwargs)


def test_actual_R_promoter_aggregation_cohort_and_trainer_features(materialized, tmp_path):
    root, paths, success, result = materialized
    assert result["covered_cancers"] == ["BRCA"] and result["raw_gap_cancers"] == ["OV"]
    audit = read(result["r_audit_path"])
    assert audit["matrix_rows"] == 562709 and audit["cancers"]["BRCA"]["patients_with_multiple_aliquots"] == 1
    matrix = pd.read_parquet(result["records"][0]["path"])
    mapping = pd.read_csv(paths["identifier_mapping"], sep="\t")
    j = np.flatnonzero(mapping.aliquot_id.eq("TCGA-BR-0000-01A"))+1
    k = np.flatnonzero(mapping.aliquot_id.eq("TCGA-BR-0000-02A"))+1
    signal = lambda t: np.sin(.13+t*.29)+np.cos(.07*t)+t*.03
    expected = (signal(j).mean()+signal(k).mean())/2
    assert matrix.loc[matrix.gene_id.eq("A000"), "TCGA-BR-0000"].iloc[0] == pytest.approx(expected, abs=1e-6)
    prepared = prepare(materialized); assert prepared["cancers"] == tuple(CANCERS) and prepared["total_rows"] == 240
    for fold,(x, score, audit) in atac.collect_atac_examples(prepared).items():
        assert x.shape == (120,7) and len(np.unique(score > np.median(score))) == 2
        assert audit == [{"cancer_id": "BRCA", "status": "USED", "inner_train_patients": 12,
                          "inner_validation_patients": 4, "sampled_pairs": 120}]
    from scripts.preflight_v32_atac_training import main
    out = tmp_path/"preflight.json"
    argv = ["--candidates", str(paths["candidates"]), "--patient-folds", str(paths["patient_folds"]),
        "--pathway-membership", str(paths["pathway_membership"]), "--materialization-success", str(success)]
    assert main([*argv, "--output", str(out)]) == 0
    assert read(out)["status"] == "PASS_READY" and read(out)["training_started"] is False
    failed = tmp_path/"too_few.json"
    with pytest.raises(ValueError, match=">=100"): main([*argv, "--max-training-rows", "10", "--output", str(failed)])
    assert read(failed)["status"] == "FAILED"


def frozen_package(native, root):
    root.mkdir(); _, paths, materialized, result = native
    keys = pd.read_parquet(paths["candidates"], columns=list(atac.TARGET_KEYS)); cohort = result["cohort"]
    cp_records = []; frames = []
    hashes = {"candidates": record(paths["candidates"])["sha256"], "folds": record(paths["patient_folds"])["sha256"]}
    for fold in range(5):
        pieces = []
        for cancer in CANCERS:
            local = keys.loc[keys.cancer_id.eq(cancer)].reset_index(drop=True)
            frame = (atac._gap_fold_frame(local, fold) if cancer == "OV" else
                     atac._fold_frame(local, fold, np.full(len(local), .2+.1*fold), np.full(len(local), ""), 16, 4))
            pieces.append(frame)
        frame = pd.concat(pieces, ignore_index=True); frames.append(frame)
        fp = root/f"fold_{fold}.parquet"; frame.to_parquet(fp, index=False)
        cp = root/f"checkpoint_{fold}.json"
        write(cp, {"checkpoint_format": atac.CHECKPOINT_FORMAT, "patient_fold": fold, "modality": "atac",
            "training_run_id": RUN, "cohort": cohort, "input_hashes": hashes,
            "outer_test_patients_used_for_fit": False, "synthetic_serialization_no_optimizer": True,
            "feature_mean": [0.]*7, "feature_scale": [1.]*7, "weights_with_intercept": [0.]*8})
        fs = root/f"fold_success_{fold}.json"
        write(fs, {"status": "SUCCESS", "patient_fold": fold, "training_run_id": RUN, "cohort": cohort,
            "checkpoint_sha256": record(cp)["sha256"], "prediction_sha256": record(fp)["sha256"],
            "heldout_patients_used_for_fit": False})
        cp_records.append({**record(cp), "patient_fold": fold, "status": "SUCCESS",
            "prediction_path": str(fp), "prediction_sha256": record(fp)["sha256"],
            "success_path": str(fs), "success_sha256": record(fs)["sha256"]})
    cp_manifest = root/"CHECKPOINT_MANIFEST.json"; write(cp_manifest, {"records": cp_records})
    typed = frames[0].drop(columns=["patient_fold_id", "outer_train_patient_count", "outer_test_patient_count"]).copy()
    typed.loc[typed.atac_available, "atac_context_probability"] = .4
    typed["atac_patient_folds_with_prediction"] = typed.atac_available.astype(int)*5
    typed["analysis_version"] = atac.ANALYSIS_VERSION; typed["training_run_id"] = RUN
    typed["module_id"] = atac.MODULE_ID; typed["changes_primary_ranking"] = False
    tp = root/"typed.parquet"; typed.to_parquet(tp, index=False)
    ia = root/"INPUT_LINEAGE_AUDIT.json"
    write(ia, {"materialization_raw_input_artifacts": result["input_artifacts"],
        "input_artifacts": [{**record(p), "role": role} for role,p in
        (("candidates", paths["candidates"]), ("folds", paths["patient_folds"]),
         ("membership", paths["pathway_membership"]), ("materialization_success", materialized))]})
    lp = root/"LINEAGE.json"
    lineage = {"analysis_version": atac.ANALYSIS_VERSION, "modality": "atac", "training_run_id": RUN,
        "training_status": "SUCCESS", "folds": 5, "cohort": cohort, "patient_level_modality_oof": True,
        "patient_fold_oof_predictions_not_fold_averaged": True, "outer_test_patients_used_in_any_upstream_fit": 0,
        "old_checkpoint_loaded": False, "old_predictions_used_as_features": False, "old_rankings_used_as_outputs": False,
        "predictions_sha256": record(tp)["sha256"], "exact_pathway_family_broadcast_used": False,
        "missing_atac_assumed_zero": False, "formal_positive_contribution_claimed": False,
        "raw_covered_cancers": ["BRCA"], "typed_raw_gap_cancers": ["OV"],
        "input_lineage_audit_path": str(ia), "input_lineage_audit_sha256": record(ia)["sha256"],
        "checkpoint_manifest_path": str(cp_manifest), "checkpoint_manifest_sha256": record(cp_manifest)["sha256"]}
    write(lp, lineage)
    sp = root/"SUCCESS.json"
    write(sp, {"status": "SUCCESS", "training_run_id": RUN, "cohort": cohort, "success_written_last": True,
        "old_predictions_or_checkpoints_used": False, "prediction_path": str(tp), "prediction_sha256": record(tp)["sha256"],
        "lineage_path": str(lp), "lineage_sha256": record(lp)["sha256"],
        "checkpoint_manifest_path": str(cp_manifest), "checkpoint_manifest_sha256": record(cp_manifest)["sha256"],
        "synthetic_serialization_no_optimizer": True})
    manifest = root/"manifest.json"
    write(manifest, {"format": INPUT_FORMAT, "producer_kind": "atac", "modalities": ["atac"],
        "core_run_id": "current-G2", "native_run_id": RUN, "candidate_authority": record(paths["candidates"]),
        "patient_folds": record(paths["patient_folds"]), "native_receipt": record(sp)})
    return manifest, sp


def test_frozen_results_independent_audit_and_actual_fusion_adapter(materialized, tmp_path):
    from scripts.audit_v32_atac_fresh_oof_result import main, AuditError
    manifest, success = frozen_package(materialized, tmp_path/"package")
    argv = ["--training-success", str(success), "--candidates", str(materialized[1]["candidates"]), "--output-root", str(tmp_path/"audit")]
    with pytest.raises(AuditError, match="expected-training-run-id"): main(argv)
    assert main([*argv, "--expected-training-run-id", RUN]) == 0
    audit = read(tmp_path/"audit/ATAC_FRESH_OOF_AUDIT.json")
    assert audit["prediction_rows"] == 240 and len(audit["fold_records"]) == 5
    binding = materialize_native_auxiliary(manifest, tmp_path/"adapter", expected_core_run_id="current-G2")
    data, audit = load_native_auxiliary(binding, expected_sha=record(binding)["sha256"], expected_core_run_id="current-G2")
    assert len(data) == 240 and audit["fold_rows"] == 1200 and not audit["training_started"]


def test_native_readiness_cli_accepts_same_candidate_authority(materialized, tmp_path):
    from scripts.audit_v32_atac_readiness import main
    root, paths, _, _ = materialized
    assets = tmp_path/"assets"; assets.mkdir()
    for role, name in (("normalized_atac_matrix", "TCGA-ATAC_PanCan_Log2Norm_Counts.rds"),
                       ("identifier_mapping", "TCGA_identifier_mapping.txt"),
                       ("peak_set", "TCGA-ATAC_PanCancer_PeakSet.txt")):
        (assets/name).symlink_to(paths[role])
    # This readiness script inventories these spreadsheets but does not read
    # their cells. Explicit placeholders exercise that limited inventory only.
    for name in ("TCGA-ATAC_DataS7_PeakToGeneLinks_v2.xlsx", "TCGA-ATAC_DataS1_DonorsAndStats_v4.xlsx"):
        (assets/name).write_text("SYNTHETIC INVENTORY PLACEHOLDER, NOT A VALID WORKBOOK")
    history = tmp_path/"historical.json"; write(history, {"synthetic_inventory_only": True})
    source = tmp_path/"historical_source.txt"; source.write_text("SYNTHETIC SOURCE INVENTORY")
    output = tmp_path/"readiness"
    assert main(["--atac-asset-root", str(assets), "--candidate-universe", str(paths["candidates"]),
        "--candidate-definition", str(root/"CANDIDATE_SOURCES.json"), "--patient-folds", str(paths["patient_folds"]),
        "--promoters", str(paths["promoters"]), "--pathway-membership", str(paths["pathway_membership"]),
        "--historical-pilot-success", str(history), "--historical-ablation-audit", str(history),
        "--historical-r-builder", str(source), "--router-source", str(source), "--router-launcher", str(source),
        "--output-root", str(output)]) == 0
    audit = read(output/"ATAC_READINESS.json")
    assert audit["summary"]["candidate_cancers"] == 2 and audit["summary"]["candidate_rows"] == 240
    assert audit["summary"]["cohort"]["raw_gap_cancers"] == ["OV"]


@pytest.mark.parametrize("fault", ["removed_cohort", "raw_changed", "contract_duplicate", "coverage", "different_candidates", "different_patients"])
def test_native_materialization_scope_cannot_be_downgraded_or_relabelled(materialized, tmp_path, fault):
    root, paths, success, result = materialized
    with ExitStack() as restore:
        def change(p, value):
            p = Path(p); original = p.read_bytes(); restore.callback(p.write_bytes, original); write(p, value)
        if fault == "removed_cohort":
            value = read(success); value.pop("cohort"); change(success, value)
        elif fault == "raw_changed":
            p = paths["identifier_mapping"]; original = p.read_bytes(); restore.callback(p.write_bytes, original); p.write_bytes(original+b"\n")
        elif fault == "contract_duplicate":
            p = Path(result["input_contract_path"]); value = read(p); value["files"].append(value["files"][0]); change(p, value)
            value = read(success); value["input_contract_sha256"] = record(p)["sha256"]; change(success, value)
        elif fault == "coverage":
            value = read(success); value["covered_cancers"] = ["OV"]; change(success, value)
        else:
            role, field = ("candidates", "candidates") if fault == "different_candidates" else ("patient_folds", "patient_map")
            copy = tmp_path/paths[role].name; copy.write_bytes(paths[role].read_bytes())
            with pytest.raises(ValueError, match="input differs"):
                replay(result["cohort"], success.parent, **{field: copy})
            return
        with pytest.raises((ValueError, atac.AtacTrainingError)): prepare(materialized)


@pytest.mark.parametrize("fault", ["typed_mean", "fold_keys", "fold_identity", "checkpoint_cohort", "removed_completion_cohort"])
def test_independent_result_reconciliation_rejects_rehashed_changes(materialized, tmp_path, fault):
    from scripts.audit_v32_atac_fresh_oof_result import main, AuditError
    _, sp = frozen_package(materialized, tmp_path/"package"); success = read(sp)
    lp = Path(success["lineage_path"]); lineage = read(lp)
    cm = Path(success["checkpoint_manifest_path"]); checkpoints = read(cm); cp = checkpoints["records"][0]
    if fault == "typed_mean":
        p = Path(success["prediction_path"]); frame = pd.read_parquet(p); frame.loc[frame.atac_available, "atac_context_probability"] = .9; frame.to_parquet(p, index=False)
        success["prediction_sha256"] = lineage["predictions_sha256"] = record(p)["sha256"]
    elif fault in {"fold_keys", "fold_identity"}:
        p = Path(cp["prediction_path"]); frame = pd.read_parquet(p)
        if fault == "fold_keys": frame.loc[0, "lncrna_id"] = "LNC:NOT_A_CANDIDATE"
        else: frame["patient_fold_id"] = 4
        frame.to_parquet(p, index=False); cp["prediction_sha256"] = record(p)["sha256"]
    elif fault == "checkpoint_cohort":
        p = Path(cp["path"]); value = read(p); value.pop("cohort"); write(p, value); cp["sha256"] = record(p)["sha256"]
    else: success.pop("cohort"); lineage.pop("cohort")
    fs = Path(cp["success_path"]); value = read(fs)
    value.update(checkpoint_sha256=cp["sha256"], prediction_sha256=cp["prediction_sha256"])
    write(fs, value); cp["success_sha256"] = record(fs)["sha256"]
    write(cm, checkpoints); success["checkpoint_manifest_sha256"] = lineage["checkpoint_manifest_sha256"] = record(cm)["sha256"]
    write(lp, lineage); success["lineage_sha256"] = record(lp)["sha256"]; write(sp, success)
    with pytest.raises((ValueError, AuditError)):
        main(["--training-success", str(sp), "--candidates", str(materialized[1]["candidates"]),
              "--output-root", str(tmp_path/"audit"), "--expected-training-run-id", RUN])


@pytest.mark.parametrize("bad", [0.5, None, -1, 5])
def test_fractional_null_or_outside_fold_is_rejected(bad):
    frame = pd.DataFrame({"cancer_id": ["BRCA"]*6, "patient_id": [f"p{i}" for i in range(6)], "patient_fold_id": [0,1,2,3,4,bad]})
    with pytest.raises(atac.AtacTrainingError): atac.normalise_patient_folds(frame)


def test_legacy_33_cancer_preparation_unchanged_without_fit(tmp_path):
    from test_v32_atac_training import _fixture
    paths = _fixture(tmp_path)
    prepared = atac.prepare_atac_inputs(candidates_path=paths["candidates"], patient_folds_path=paths["folds"],
        pathway_membership_path=paths["membership"], materialization_success_path=paths["materialization"],
        config=atac.AtacTrainingConfig(expected_rows_per_cancer=None))
    assert prepared["cohort"] is None and prepared["cancers"] == atac.FORMAL_CANCERS
    assert len(atac.collect_atac_examples(prepared)) == 5
