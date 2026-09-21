#!/usr/bin/env python3
"""Fail-closed audit of the materialized V3.1 exact-pathway website release."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.common import file_sha256, load_config, read_table, write_table
from cc_hhgt.formal_gate import validate_formal_training_gate
from cc_hhgt.relocated_lineage import validate_relocated_training_lineage
from cc_hhgt.v30_integrity import atomic_write_json, canonical_json_sha256
from cc_hhgt.v31_exact_materialize import FORBIDDEN_LABEL_COLUMNS


EXPECTED_CANCERS = 33
EXPECTED_SEEDS = 3
MODELS = {"rgcn", "hgt", "cc_hhgt"}
REQUIRED_FORMAL_CANCERS = {"HNSC", "LGG"}
PREDICTION_AUDIT_COLUMNS = [
    "candidate_id",
    "cancer_id",
    "lncrna_id",
    "pathway_id",
    "pathway_family_id",
    "calibrated_probability",
    "seed_probability_std",
    "direction_positive_probability",
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _gmt(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            fields = line.split("\t")
            if len(fields) < 3:
                raise RuntimeError(f"Invalid GMT row in {path}: {line[:100]}")
            rows.append(fields)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-gate", required=True)
    parser.add_argument("--training-lineage-gate", required=True)
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--ensemble-root", required=True)
    parser.add_argument("--materialized-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    gate_path = Path(args.formal_gate).resolve()
    gate_document = _read_json(gate_path)
    training_root = Path(args.training_root).resolve()
    cfg = load_config(
        args.config,
        project_root_override=Path(gate_document["paths"]["run_root"]).resolve(),
        create_dirs=False,
    )
    gate, execution_gate_sha256 = validate_formal_training_gate(
        gate_path,
        cfg,
        verify_assets=True,
        run_id=args.run_id,
        input_root=gate_document["paths"]["input_root"],
        input_manifest=gate_document["paths"]["input_manifest"],
        output_root=training_root,
    )
    _, training_gate_sha256, normalized_gate_contract_sha256 = (
        validate_relocated_training_lineage(args.training_lineage_gate, gate)
    )
    selection_path = Path(args.selection).resolve()
    selection = _read_json(selection_path)
    selected_model = str(selection.get("selected_model"))
    if (
        selection.get("status") != "PASS"
        or selection.get("run_id") != args.run_id
        or selected_model not in MODELS
        or selection.get("selection_split") != "val_only"
        or bool(selection.get("test_metrics_used"))
        or int(selection.get("folds", -1)) != EXPECTED_CANCERS
        or int(selection.get("seed_ensemble_size", -1)) != EXPECTED_SEEDS
        or not bool(selection.get("matrix_audit_release_eligible"))
    ):
        raise RuntimeError("Website audit requires the validation-only formal model selection")

    materialized = Path(args.materialized_root).resolve()
    success_path = materialized / "SUCCESS.json"
    success = _read_json(success_path)
    if (
        success.get("status") != "PASS"
        or success.get("run_id") != args.run_id
        or success.get("selected_model") != selected_model
        or success.get("pathway_target_level") != "exact_pathway"
        or success.get("pathway_family_role") != "auxiliary_hierarchy_only"
        or int(success.get("cancers", -1)) != EXPECTED_CANCERS
        or list(success.get("reference_only_cancers", ["missing"])) != []
        or bool(success.get("heldout_label_columns_in_public_scores"))
        or success.get("training_lineage_gate_sha256") != training_gate_sha256
        or success.get("execution_site_gate_sha256") != execution_gate_sha256
    ):
        raise RuntimeError("Invalid V3.1 exact-pathway materialization SUCCESS contract")

    filter_contract = success.get("pancancer_lncrna_filter", {})
    if (
        not bool(filter_contract.get("enabled"))
        or not bool(filter_contract.get("evidence_cannot_bypass_filter"))
        or int(filter_contract.get("minimum_detected_cancers", -1)) != 3
        or float(filter_contract.get("within_cancer_min_sample_detection_rate", -1))
        != 0.10
    ):
        raise RuntimeError("Website release lacks the frozen pan-cancer lncRNA filter contract")

    tables = Path(gate["paths"]["asset_results"]).resolve() / "tables"
    folds = read_table(tables / "fold_manifest.tsv")
    eligibility = read_table(tables / "pancancer_lncrna_eligibility.parquet")
    eligible = set(
        eligibility.loc[eligibility.eligible.astype(bool), "lncrna_id"].astype(str)
    )
    pathways = set(
        read_table(tables.parent.parent / "standardized" / "dim_pathway.parquet")[
            "pathway_id"
        ].astype(str)
    )
    fold_cancers = set(folds.test_cancer.astype(str))
    if (
        len(folds) != EXPECTED_CANCERS
        or folds.fold_id.astype(str).duplicated().any()
        or len(fold_cancers) != EXPECTED_CANCERS
        or not REQUIRED_FORMAL_CANCERS.issubset(fold_cancers)
        or not eligible
        or not pathways
    ):
        raise RuntimeError("Website audit lacks registered folds, lncRNA eligibility, or pathways")

    fold_success = {str(row["fold_id"]): row for row in success.get("fold_audits", [])}
    if len(fold_success) != EXPECTED_CANCERS:
        raise RuntimeError("Materialization SUCCESS lacks 33 unique fold audits")
    ensemble_root = Path(args.ensemble_root).resolve()
    fold_rows: list[dict] = []
    score_total = selected_total = 0
    for fold in folds.itertuples(index=False):
        fold_id = str(fold.fold_id)
        cancer = str(fold.test_cancer)
        audit = fold_success.get(fold_id)
        if not audit or str(audit.get("cancer_id")) != cancer:
            raise RuntimeError(f"Materialized fold/cancer audit mismatch: {fold_id}")
        inference_success_path = ensemble_root / selected_model / fold_id / "SUCCESS.json"
        inference = _read_json(inference_success_path)
        prediction = inference_success_path.parent / "prediction_exact_pathway_three_seed_ensemble.parquet"
        if (
            inference.get("status") != "PASS"
            or inference.get("run_id") != args.run_id
            or inference.get("selected_model") != selected_model
            or inference.get("formal_gate_sha256") != training_gate_sha256
            or int(inference.get("seed_count", -1)) != EXPECTED_SEEDS
            or bool(inference.get("heldout_label_columns_in_output"))
            or inference.get("prediction_file_sha256") != file_sha256(prediction)
            or audit.get("inference_success_sha256") != file_sha256(inference_success_path)
        ):
            raise RuntimeError(f"Invalid website inference lineage: {fold_id}")

        score_path = materialized / "web_exact_pathway_score" / f"cancer_id={cancer}" / "part-0.parquet"
        selected_path = materialized / "web_exact_pathway_selected" / f"cancer_id={cancer}" / "part-0.parquet"
        if file_sha256(score_path) != audit.get("score_sha256") or file_sha256(selected_path) != audit.get("selected_sha256"):
            raise RuntimeError(f"Materialized partition hash mismatch: {fold_id}")
        score = pd.read_parquet(score_path)
        selected = pd.read_parquet(selected_path)
        for name, frame in (("score", score), ("selected", selected)):
            forbidden = sorted(FORBIDDEN_LABEL_COLUMNS & set(frame.columns))
            if forbidden:
                raise RuntimeError(f"Public {name} partition contains labels {forbidden}: {fold_id}")
            required = {
                "candidate_id", "cancer_id", "lncrna_id", "pathway_id",
                "pathway_family_id", "calibrated_probability", "relationship_class",
            }
            missing = sorted(required - set(frame.columns))
            if missing or frame.candidate_id.astype(str).duplicated().any():
                raise RuntimeError(f"Invalid public {name} identity in {fold_id}: missing={missing}")
            probability = pd.to_numeric(frame.calibrated_probability, errors="coerce").to_numpy(float)
            if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise RuntimeError(f"Invalid public probabilities in {fold_id}/{name}")
            if set(frame.cancer_id.astype(str)) != {cancer}:
                raise RuntimeError(f"Cancer partition drift in {fold_id}/{name}")
            if not set(frame.lncrna_id.astype(str)).issubset(eligible):
                raise RuntimeError(f"Ineligible lncRNA entered website {fold_id}/{name}")
            if not set(frame.pathway_id.astype(str)).issubset(pathways):
                raise RuntimeError(f"Unknown exact pathway entered website {fold_id}/{name}")
        if len(score) != int(inference["rows"]) or len(score) != int(audit["rows"]):
            raise RuntimeError(f"Full-universe website row count mismatch: {fold_id}")
        if len(selected) != int(audit["selected_rows"]):
            raise RuntimeError(f"Selected website row count mismatch: {fold_id}")
        if selected.relationship_class.astype(str).eq("exploratory").any():
            raise RuntimeError(f"Exploratory row entered selected website partition: {fold_id}")
        try:
            prediction_frame = pd.read_parquet(
                prediction, columns=PREDICTION_AUDIT_COLUMNS
            )
        except (KeyError, ValueError) as exc:
            raise RuntimeError(
                f"Inference prediction lacks website audit columns in {fold_id}"
            ) from exc
        if prediction_frame.candidate_id.astype(str).duplicated().any():
            raise RuntimeError(f"Inference prediction has duplicate candidates: {fold_id}")
        public_prediction_columns = [
            column for column in PREDICTION_AUDIT_COLUMNS if column != "candidate_id"
        ]
        joined = prediction_frame.merge(
            score[["candidate_id", *public_prediction_columns]],
            on="candidate_id",
            how="outer",
            validate="one_to_one",
            suffixes=("_inference", "_website"),
            indicator=True,
        )
        if not joined._merge.astype(str).eq("both").all():
            raise RuntimeError(f"Website candidates differ from inference: {fold_id}")
        identity_columns = [
            "cancer_id", "lncrna_id", "pathway_id", "pathway_family_id"
        ]
        for column in identity_columns:
            if not joined[f"{column}_inference"].astype(str).equals(
                joined[f"{column}_website"].astype(str)
            ):
                raise RuntimeError(
                    f"Website exact-pathway identity differs from inference: "
                    f"{fold_id}/{column}"
                )
        probability_columns = [
            "calibrated_probability",
            "seed_probability_std",
            "direction_positive_probability",
        ]
        for column in probability_columns:
            inference_values = pd.to_numeric(
                joined[f"{column}_inference"], errors="coerce"
            ).to_numpy(float)
            website_values = pd.to_numeric(
                joined[f"{column}_website"], errors="coerce"
            ).to_numpy(float)
            if not np.array_equal(inference_values, website_values, equal_nan=False):
                raise RuntimeError(
                    f"Website probability differs from inference: {fold_id}/{column}"
                )
        selected_expected = score.loc[
            score.relationship_class.astype(str).ne("exploratory"), "candidate_id"
        ].astype(str)
        selected_actual = selected.candidate_id.astype(str)
        if set(selected_actual) != set(selected_expected):
            raise RuntimeError(
                f"Selected website rows are not exactly the non-exploratory score rows: {fold_id}"
            )
        score_total += len(score)
        selected_total += len(selected)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "cancer_id": cancer,
                "score_rows": len(score),
                "selected_rows": len(selected),
                "score_sha256": file_sha256(score_path),
                "selected_sha256": file_sha256(selected_path),
                "status": "PASS",
            }
        )
    if score_total != int(success["score_rows"]) or selected_total != int(success["selected_rows"]):
        raise RuntimeError("Website aggregate score/selected row counts drifted")

    master = pd.read_parquet(materialized / "geneset_master_exact_pathway.parquet")
    member = pd.read_parquet(materialized / "geneset_member_exact_pathway.parquet")
    if master.empty or member.empty or master.geneset_id.astype(str).duplicated().any():
        raise RuntimeError("Website exact-pathway gene sets are empty or non-unique")
    if member.duplicated(["geneset_id", "lncrna_id"]).any():
        raise RuntimeError("Website exact-pathway gene-set members are duplicated")
    if set(master.pathway_id.astype(str)) - pathways or set(member.lncrna_id.astype(str)) - eligible:
        raise RuntimeError("Website gene sets violate exact-pathway or lncRNA eligibility")
    if set(member.geneset_id.astype(str)) - set(master.geneset_id.astype(str)):
        raise RuntimeError("Website gene-set member lacks a registered master")
    if set(master.cancer_id.astype(str)) != fold_cancers:
        raise RuntimeError("Website exact-pathway gene sets do not cover all 33 cancers")
    member_contract = member.merge(
        master[["geneset_id", "cancer_id", "pathway_id"]],
        on="geneset_id",
        how="left",
        validate="many_to_one",
        suffixes=("_member", "_master"),
    )
    for column in ("cancer_id", "pathway_id"):
        if not member_contract[f"{column}_member"].astype(str).equals(
            member_contract[f"{column}_master"].astype(str)
        ):
            raise RuntimeError(
                f"Website gene-set member/master {column} contract drifted"
            )
    counts = member.groupby("geneset_id", observed=True).size()
    minimum = int(cfg["materialization"]["min_members"])
    maximum = int(cfg["materialization"]["max_members"])
    if counts.min() < minimum or counts.max() > maximum:
        raise RuntimeError("Website gene-set size violates the frozen materialization thresholds")
    if len(master) != int(success["genesets"]) or len(member) != int(success["geneset_members"]):
        raise RuntimeError("Website gene-set aggregate counts drifted")

    gmt_paths = [
        materialized / "cancer_exact_pathway_lncRNA_symbols.gmt",
        materialized / "cancer_exact_pathway_lncRNA_ids.gmt",
        materialized / "cancer_exact_pathway_lncRNA_symbols_extended.gmt",
    ]
    master_names = set(master.geneset_name.astype(str))
    gmt_rows = {path.name: _gmt(path) for path in gmt_paths}
    for name, rows in gmt_rows.items():
        if not rows or any(row[0] not in master_names or len(row[2:]) < minimum for row in rows):
            raise RuntimeError(f"Invalid or empty exact-pathway GMT: {name}")
    extended_names = {
        row[0] for row in gmt_rows["cancer_exact_pathway_lncRNA_symbols_extended.gmt"]
    }
    if extended_names != master_names:
        raise RuntimeError("Extended exact-pathway GMT does not cover every gene-set master")

    files = []
    for path in sorted(materialized.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "relative_path": str(path.relative_to(materialized)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Website release audit refuses reuse: {output_root}")
    temporary = output_root.parent / f".{output_root.name}.tmp"
    if temporary.exists():
        raise RuntimeError(f"Stale website release audit temporary root: {temporary}")
    temporary.mkdir(parents=True)
    write_table(pd.DataFrame(fold_rows), temporary / "fold_partition_audit.tsv")
    write_table(pd.DataFrame(files), temporary / "release_file_manifest.tsv")
    payload = {
        "status": "PASS",
        "release_eligible": True,
        "run_id": args.run_id,
        "selected_model": selected_model,
        "selection_split": "val_only",
        "test_metrics_used_for_selection": False,
        "pathway_target_level": "exact_pathway",
        "pathway_family_role": "auxiliary_hierarchy_only",
        "reference_only_cancers": [],
        "cancers": EXPECTED_CANCERS,
        "pancancer_eligible_lncRNAs": len(eligible),
        "pancancer_lncrna_filter": filter_contract,
        "score_rows": score_total,
        "selected_rows": selected_total,
        "genesets": len(master),
        "geneset_members": len(member),
        "execution_site_gate_sha256": execution_gate_sha256,
        "training_lineage_gate_sha256": training_gate_sha256,
        "normalized_gate_contract_sha256": normalized_gate_contract_sha256,
        "selection_sha256": file_sha256(selection_path),
        "materialization_success_sha256": file_sha256(success_path),
        "release_files": len(files),
        "release_file_merkle_sha256": canonical_json_sha256(files),
        "gmt_rows": {name: len(rows) for name, rows in gmt_rows.items()},
        "heldout_label_columns_in_public_scores": False,
    }
    atomic_write_json(temporary / "WEBSITE_RELEASE_AUDIT.json", payload)
    atomic_write_json(temporary / "SUCCESS.json", payload)
    os.replace(temporary, output_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
