"""Actual two-cancer workers and independent closeout; no aggregate substitutes."""
from pathlib import Path
from types import SimpleNamespace
import copy
import socket
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.fold_local_scope import candidate_frame, CANDIDATES
from cc_hhgt.v32.local_cnv_cohort import from_config, replay
from cc_hhgt.v32.local_cnv_confounding import DEFAULT_COVARIATES
from cc_hhgt.v32.local_cnv_core_preparation import validate_local_cnv_audit, validate_validation_a1_authority
from test_v32_fold_local_expression import forbidden
import scripts.run_v32_local_cnv_confounding_audit_r2 as association
import scripts.materialize_v32_local_cnv_validation_associations as validation
import scripts.finalize_and_audit_v32_local_cnv_r3 as finalizer
import scripts.audit_v32_local_cnv_validation_associations as validation_finalizer
import scripts.materialize_v32_local_cnv_corrected_graph_authority as corrected

ROOT = Path(__file__).resolve().parents[1]
CANCERS = ["BRCA", "LUAD"]
LNCS = ["LNC:A", "LNC:B", "LNC:LOW"]
PATHWAY = "MSIGDB:HALLMARK:P1"


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)


def fixture_inputs(root, confounded, *, variable_low=False):
    root.mkdir()
    annotation = root / "annotation.tsv"
    pd.DataFrame({"ensembl_gene_id": ["A", "B", "LOW", "G1"],
                  "gene_type": ["lncRNA"]*3 + ["protein_coding"]}).to_csv(annotation, sep="\t", index=False)
    gmt = root / "hallmark.gmt"; gmt.write_text("P1\tdescription\t" + "\t".join(f"G{i}" for i in range(1, 11)) + "\n")
    manifest = root / "gmt.json"
    write(manifest, {"format": "SSGSEA_GMT_INPUT_V1", "collections": [{"collection": "HALLMARK", "file": record(gmt)}]})
    definition = {"format": CANDIDATES, "target_host": "149", "training_started": False, "measured_detection_used": False,
        "annotation": record(annotation), "gmt_manifest": record(manifest), "cancers": CANCERS,
        "budget_per_cancer": 3, "seed": 20260726, "implementations": [record(ROOT / name) for name in (
            "cc_hhgt/v32/fold_local_scope.py", "scripts/materialize_fold_local_scope.py", "scripts/prepare_v32_formal.py")]}
    candidates = root / "candidates.parquet"; candidate_frame(definition, root).to_parquet(candidates, index=False)
    definition["output"] = record(candidates); write(root / "CANDIDATE_SOURCES.json", definition)
    t = np.arange(120, dtype=float)
    signal = np.sin(t*.3) + np.cos(t*.13)
    activity = signal + .35*np.sin(t*1.83)
    lnc_values = [signal + .35*np.sin(t*1.37), np.sin(t*.91) + .3*np.cos(t*.17), np.zeros(120)]
    if variable_low: lnc_values[2] = np.cos(t*.63) + .2*np.sin(t*.83)
    maps = []; covariates = []; coverage = {}
    sources = {k: {"path": str(root / k)} for k in ("lnc_expression_root", "gene_expression_root", "pathway_activity_root")}
    for cancer in CANCERS:
        patients = [f"PRIVATE_{cancer}_{i:03}" for i in range(120)]
        maps.append(pd.DataFrame({"cancer_id": cancer, "patient_id": patients, "patient_fold_id": np.arange(120) % 5}))
        cov = pd.DataFrame({"cancer_id": cancer, "patient_id": patients})
        for name in DEFAULT_COVARIATES: cov[name] = 0.
        cov["purity"] = np.linspace(.2, .8, 120); covariates.append(cov)
        for key, entity, value, items, arrays in (
                ("lnc_expression_root", "lncrna_id", "logcpm", LNCS, lnc_values),
                ("gene_expression_root", "gene_id", "logcpm", ["G1"], [signal + .4*np.cos(t*1.71)]),
                ("pathway_activity_root", "pathway_id", "activity_score", [PATHWAY], [activity])):
            folder = root / key / f"cancer_id={cancer}"; folder.mkdir(parents=True)
            pd.concat([pd.DataFrame({"cancer_id": cancer, "patient_id": patients, "sample_id": patients,
                entity: item, value: values}) for item, values in zip(items, arrays)]).to_parquet(folder / "part-0.parquet", index=False)
        compact = root / "directional" / f"cancer={cancer}"; compact.mkdir(parents=True)
        for key, values in (("patient_ids", patients), ("lncrna_ids", LNCS), ("pathway_ids", [PATHWAY])):
            write(compact / (key + ".json"), values)
        for kind, count in (("lncrna", 3), ("pathway", 1)):
            signed = np.zeros((120, count), dtype=np.float32)
            if confounded and kind == "lncrna": signed[:, 0] = signal
            for feature, array in {"signed_mean": signed, "callable": np.ones_like(signed, dtype=bool),
                                   "amplification": signed > .3, "deletion": signed < -.3}.items():
                np.save(compact / f"{kind}_{feature}.npy", array)
        raw = root / (cancer + "_synthetic_segments.tsv")
        pd.DataFrame({"patient_id": patients, "synthetic_value": signal if confounded else np.zeros(120)}).to_csv(raw, sep="\t", index=False)
        write(compact / "SOURCE_FILES.json", {"records": [record(raw)]})
        write(compact / "SUCCESS.json", {"status": "SUCCESS", "files": {p.name: record(p)["sha256"] for p in compact.iterdir()}})
        coverage[cancer] = {"processed_segment_patients": 120, "typed_unavailable_patients": 0}
    write(root / "directional/SUCCESS.json", {"status": "SUCCESS", "cancers": coverage})
    patient_map = root / "patients.tsv"; pd.concat(maps).to_csv(patient_map, sep="\t", index=False)
    covariate_path = root / "covariates.parquet"; pd.concat(covariates).to_parquet(covariate_path, index=False)
    sources["covariates"] = record(covariate_path)
    config = root / "sources.json"
    write(config, {"sources": sources, "candidate_definition": record(root / "CANDIDATE_SOURCES.json")})
    static = root / "graph/static"; static.mkdir(parents=True)
    for name in ("signed_membership", "protein_gene_encoding"):
        pd.DataFrame({"gene_id": ["G1"]}).to_parquet(static / (name + ".parquet"), index=False)
    return SimpleNamespace(cancer="BRCA", candidates=candidates, patient_map=patient_map, covariates=covariate_path,
        source_config=config, graph_root=static.parent, directional_root=root / "directional", output_root=root / "train")


def close_train(args):
    finalizer.main(["--root", str(args.output_root), "--candidates", str(args.candidates),
        "--patient-map", str(args.patient_map), "--covariates", str(args.covariates),
        "--source-config", str(args.source_config), "--directional-root", str(args.directional_root),
        "--runner-code", str(Path(association.__file__)), "--run-id", association.RUN_ID])


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("local_cnv_real_closeout")
    output = {}
    for name, confounded, variable_low in (("confounded", True, False), ("small", False, True), ("review", False, False)):
        args = fixture_inputs(root / name, confounded, variable_low=variable_low)
        for cancer in CANCERS:
            args.cancer = cancer; association.run_cancer(args)
        close_train(args)
        output[name] = args
    args = output["confounded"]
    vargs = SimpleNamespace(**{**vars(args), "local_cnv_audit_root": args.output_root,
        "local_cnv_success_sha256": record(args.output_root / "SUCCESS.json")["sha256"],
        "run_id": "SYNTHETIC_COHORT_VALIDATION", "output_root": root / "confounded/validation"})
    for cancer in CANCERS:
        vargs.cancer = cancer; validation._run(vargs)
    validation_finalizer.main(["--root", str(vargs.output_root), "--candidates", str(args.candidates),
        "--patient-map", str(args.patient_map), "--local-cnv-audit-success", str(args.output_root / "SUCCESS.json")])
    output["validation"] = vargs
    return output


def test_actual_two_cancer_five_fold_closeout_and_validation(completed):
    args = completed["confounded"]; root = args.output_root
    success = read(root / "SUCCESS.json"); cohort = success["cohort"]
    assert success["decision"] == "LOCAL_CNV_CORE_RETRAIN_REQUIRED"
    assert cohort["cancers"] == CANCERS and cohort["requested_budget_per_cancer"] == 3
    assert cohort["candidate_rows_by_cancer"] == {"BRCA": 3, "LUAD": 3}
    assert cohort["expected_pair_rows_all_folds"] == 30
    summary = read(root / "local_cnv_confounding_global_summary.json")
    assert summary["cancers"] == 2 and summary["folds"] == 10 and summary["candidate_rows"] == 30
    assert len(pd.read_parquet(root / "local_cnv_confounding_pair_level.parquet")) == 30
    train = validate_local_cnv_audit(root)
    valid = validate_validation_a1_authority(completed["validation"].output_root)
    assert train["cohort"] == valid["cohort"] == cohort
    assert valid["local_cnv_audit_success_sha256"] == train["success_sha256"]
    assert read(completed["validation"].output_root / "INDEPENDENT_AUDIT.json")["rows"] == 30
    # Unavailable effects are retained, rather than shrinking the candidate set.
    rows = pd.read_parquet(root / "local_cnv_confounding_pair_level.parquet")
    assert rows.loc[rows.lncrna_id.eq("LNC:LOW"), "availability"].eq("typed_unavailable").all()
    assert rows.loc[rows.lncrna_id.eq("LNC:LOW"), ["rho_A0_full", "rho_A1_local", "p_A1_local"]].isna().all().all()
    valid_rows = pd.read_parquet(completed["validation"].output_root / "validation_associations")
    assert valid_rows.loc[valid_rows.lncrna_id.eq("LNC:LOW"), "availability"].eq("typed_unavailable").all()


def test_actual_small_effect_decision_is_preserved_and_does_not_authorize_correction(completed):
    root = completed["small"].output_root
    assert read(root / "SUCCESS.json")["decision"] == "LOCAL_CNV_IMPACT_SMALL"
    with pytest.raises(RuntimeError, match="does not require corrected"):
        validate_local_cnv_audit(root)


def test_insufficient_rank_comparison_remains_review_not_small(completed):
    root = completed["review"].output_root
    assert read(root / "SUCCESS.json")["decision"] == "LOCAL_CNV_SENSITIVITY_ONLY_REVIEW"
    assert np.isnan(read(root / "local_cnv_confounding_global_summary.json")["median_cancer_pathway_rank_spearman"])
    with pytest.raises(RuntimeError, match="does not require corrected"):
        validate_local_cnv_audit(root)


@pytest.mark.parametrize("kind", ["train", "validation"])
@pytest.mark.parametrize("fault", ["removed_cohort", "audit_hash", "audit_cohort"])
def test_native_closeout_binding_cannot_be_downgraded(completed, tmp_path, kind, fault):
    train = kind == "train"
    source = completed["confounded" if train else "validation"].output_root
    root = tmp_path / "copied_release"
    shutil.copytree(source, root)
    audit_path = root / ("LOCAL_CNV_INDEPENDENT_AUDIT.json" if train else "INDEPENDENT_AUDIT.json")
    success = read(root / "SUCCESS.json")
    if fault == "removed_cohort": success.pop("cohort")
    elif fault == "audit_hash": success["independent_audit_sha256"] = "0" * 64
    else:
        audit = read(audit_path); audit["cohort"]["folds"] = 4; write(audit_path, audit)
        success["independent_audit_sha256"] = record(audit_path)["sha256"]
    write(root / "SUCCESS.json", success)
    with pytest.raises((ValueError, RuntimeError), match="cohort"):
        (validate_local_cnv_audit if train else validate_validation_a1_authority)(root)


def test_actual_corrected_graph_closes_declared_cancers_without_constant_patch(completed, tmp_path):
    from cc_hhgt.v32.formal_graph_authority import GRAPH_INPUT_RECEIPT_FORMAT, GRAPH_INPUT_RECEIPT_STATUS
    args = completed["confounded"]
    source = tmp_path / "source.json"
    write(source, {"format": GRAPH_INPUT_RECEIPT_FORMAT, "status": GRAPH_INPUT_RECEIPT_STATUS,
        "cancer_scope": CANCERS, "counts": {}, "gates": {},
        "fold_artifacts": {str(f): {"train_coexpression": {}} for f in range(5)}})
    settings = SimpleNamespace(output_root=tmp_path / "corrected", source_graph_receipt=source,
        source_graph_receipt_sha256=record(source)["sha256"], local_cnv_audit_root=args.output_root,
        local_cnv_audit_success_sha256=record(args.output_root / "SUCCESS.json")["sha256"], cancer="BRCA")
    assert corrected.initialize(settings) == 0
    for cancer in CANCERS:
        settings.cancer = cancer; assert corrected.materialize_cancer(settings) == 0
    assert corrected.finalize(settings) == 0
    document = read(settings.output_root / "GRAPH_INPUT_AUTHORITY_RECEIPT.json")
    assert document["cancer_scope"] == CANCERS
    assert set(document["local_cnv_adjustment"]["cancer_receipt_sha256"]) == set(CANCERS)
    assert all(len(list((settings.output_root / f"folds/fold_{f}/train_coexpression").glob("part_*.parquet"))) == 2 for f in range(5))


@pytest.mark.parametrize("fault", ["cancer", "counts", "budget", "folds", "missing_patient_fold", "cross_cancer_patient"])
def test_cohort_replay_rejects_repinned_wrong_coverage(completed, tmp_path, fault):
    args = completed["confounded"]; cohort = copy.deepcopy(read(args.output_root / "SUCCESS.json")["cohort"])
    if fault == "cancer": cohort["cancers"].pop()
    elif fault == "counts": cohort["candidate_rows_by_cancer"]["BRCA"] = 2
    elif fault == "budget": cohort["requested_budget_per_cancer"] = 2
    elif fault == "folds": cohort["folds"] = 4
    else:
        frame = pd.read_csv(args.patient_map, sep="\t")
        if fault == "cross_cancer_patient":
            frame.loc[frame.cancer_id.eq("LUAD").idxmax(), "patient_id"] = frame.loc[frame.cancer_id.eq("BRCA"), "patient_id"].iloc[0]
        else:
            frame = frame.loc[~(frame.cancer_id.eq("LUAD") & frame.patient_fold_id.eq(4))]
        path = tmp_path / "patients.tsv"; frame.to_csv(path, sep="\t", index=False); cohort["patient_map"] = record(path)
    with pytest.raises(ValueError, match="cohort|folds"):
        replay(cohort, tmp_path)


@pytest.mark.parametrize("fault", ["missing", "extra", "worker_cohort", "worker_run_id"])
def test_closeout_rejects_incomplete_or_mixed_partitions(tmp_path, fault):
    args = fixture_inputs(tmp_path / "inputs", True)
    for cancer in CANCERS:
        args.cancer = cancer; association.run_cancer(args)
    if fault == "missing":
        (args.output_root / "pair_level/cancer=LUAD/fold=4/part.parquet").unlink()
    elif fault == "extra":
        path = args.output_root / "pair_level/extra.parquet"
        pd.DataFrame({"not_audited": [1]}).to_parquet(path)
    else:
        path = args.output_root / "receipts/LUAD.json"; value = read(path)
        if fault == "worker_cohort": value["cohort"]["candidate_rows_by_cancer"]["LUAD"] = 2
        else: value["run_id"] = "WRONG_RUN"
        write(path, value)
    with pytest.raises((ValueError, RuntimeError), match="coverage|worker cohort|receipt contract"): close_train(args)
    assert not (args.output_root / "SUCCESS.json").exists()


def test_legacy_candidate_budget_is_not_inferred_from_small_observed_inputs(tmp_path):
    args = fixture_inputs(tmp_path / "inputs", True)
    spec = read(args.source_config); spec.pop("candidate_definition"); write(args.source_config, spec)
    assert from_config(spec, tmp_path, args.candidates, args.patient_map) is None
    with pytest.raises(RuntimeError, match="candidate authority drift"): association.run_cancer(args)
