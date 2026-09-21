#!/usr/bin/env python3
"""Materialize V3.1 exact-pathway website rankings and lncRNA gene sets.

This post-training stage consumes only the validation-selected, three-seed
full-universe scores.  Patient-association labels are excluded from public
score artifacts; frozen evidence is joined only after inference to classify
and explain website members.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from cc_hhgt.common import file_sha256, load_config, read_table, write_table
from cc_hhgt.formal_gate import validate_formal_training_gate
from cc_hhgt.pathway_target import require_exact_pathway_contract
from cc_hhgt.relocated_lineage import validate_relocated_training_lineage
from cc_hhgt.v30_integrity import atomic_write_json
from cc_hhgt.v31_exact_materialize import (
    FORBIDDEN_LABEL_COLUMNS,
    IDENTITY_COLUMNS,
    build_exact_pathway_genesets,
    classify_exact_pathway_scores,
)


EXPECTED_CANCERS = 33
EXPECTED_SEEDS = 3
PURPOSE = "V3.1_EXACT_PATHWAY_WEBSITE_MATERIALIZATION"


def _atomic_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.parquet")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _read_success(
    path: Path,
    *,
    run_id: str,
    selected_model: str,
    training_lineage_gate_sha256: str,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS"
        or payload.get("run_id") != run_id
        or payload.get("selected_model") != selected_model
        or payload.get("pathway_target_level") != "exact_pathway"
        or int(payload.get("seed_count", -1)) != EXPECTED_SEEDS
        or bool(payload.get("heldout_label_columns_in_output"))
        or payload.get("formal_gate_sha256") != training_lineage_gate_sha256
    ):
        raise RuntimeError(f"Invalid exact-pathway ensemble inference: {path}")
    prediction = Path(payload["prediction_file"]).resolve()
    if not prediction.is_file():
        # Preserve portability after the ensemble release is copied from the
        # training host: SUCCESS may contain the original absolute path.
        prediction = path.parent / "prediction_exact_pathway_three_seed_ensemble.parquet"
    if not prediction.is_file() or file_sha256(prediction) != payload.get(
        "prediction_file_sha256"
    ):
        raise RuntimeError(f"Ensemble prediction hash mismatch: {prediction}")
    payload["prediction_file"] = str(prediction)
    return payload


def _dimension(frame: pd.DataFrame, key: str, columns: list[str]) -> pd.DataFrame:
    keep = [column for column in [key, *columns] if column in frame]
    if key not in keep:
        raise RuntimeError(f"Dimension lacks {key}")
    result = frame[keep].drop_duplicates(key)
    if result[key].astype(str).duplicated().any():
        raise RuntimeError(f"Dimension is not unique on {key}")
    return result


def _write_gmt(
    path: Path,
    master: pd.DataFrame,
    member: pd.DataFrame,
    include: set[str],
    gene_column: str,
    min_members: int,
) -> None:
    names = (
        master.set_index("geneset_id").geneset_name.astype(str).to_dict()
        if not master.empty
        else {}
    )
    lines: list[str] = []
    if not member.empty and gene_column in member:
        filtered = member.loc[member.relationship_class.astype(str).isin(include)]
        for geneset_id, group in filtered.groupby(
            "geneset_id", observed=True, sort=True
        ):
            genes = [
                value
                for value in group.sort_values("rank", kind="stable")[gene_column]
                .fillna("")
                .astype(str)
                .tolist()
                if value
            ]
            genes = list(dict.fromkeys(genes))
            if len(genes) >= min_members:
                lines.append(
                    "\t".join(
                        [names[str(geneset_id)], "CancerLncAtlas CC-HHGT V3.1", *genes]
                    )
                )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--formal-gate", required=True)
    parser.add_argument(
        "--training-lineage-gate",
        help="Original training-site gate when materialization runs on a verified relocation.",
    )
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--ensemble-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise RuntimeError(f"Exact-pathway materializer refuses reuse: {output_root}")
    selection_path = Path(args.selection).resolve()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_model = str(selection.get("selected_model", ""))
    if (
        selection.get("status") != "PASS"
        or selection.get("run_id") != args.run_id
        or selection.get("selection_split") != "val_only"
        or bool(selection.get("test_metrics_used"))
        or int(selection.get("folds", -1)) != EXPECTED_CANCERS
        or int(selection.get("seed_ensemble_size", -1)) != EXPECTED_SEEDS
        or not bool(selection.get("matrix_audit_release_eligible"))
        or selected_model not in {"rgcn", "hgt", "cc_hhgt"}
    ):
        raise RuntimeError("Invalid validation-only exact-pathway model selection")

    gate_path = Path(args.formal_gate).resolve()
    gate_document = json.loads(gate_path.read_text(encoding="utf-8"))
    training_root = Path(args.training_root).resolve()
    cfg = load_config(
        args.config,
        project_root_override=Path(gate_document["paths"]["run_root"]).resolve(),
        create_dirs=False,
    )
    require_exact_pathway_contract(cfg)
    gate, execution_gate_sha256 = validate_formal_training_gate(
        gate_path,
        cfg,
        verify_assets=True,
        run_id=args.run_id,
        input_root=gate_document["paths"]["input_root"],
        input_manifest=gate_document["paths"]["input_manifest"],
        output_root=training_root,
    )
    lineage_gate_path = args.training_lineage_gate or args.formal_gate
    _, training_gate_sha256, normalized_gate_contract_sha256 = (
        validate_relocated_training_lineage(lineage_gate_path, gate)
    )
    tables = Path(gate["paths"]["asset_results"]).resolve() / "tables"
    standardized = tables.parent.parent / "standardized"
    eligibility = read_table(tables / "pancancer_lncrna_eligibility.parquet")
    eligible_lncRNAs = int(eligibility.eligible.astype(bool).sum())
    if eligible_lncRNAs <= 0:
        raise RuntimeError("Exact-pathway website release has no pan-cancer eligible lncRNAs")
    filter_contract = cfg["candidate_universe"]["pancancer_lnc_filter"]
    if (
        not bool(filter_contract.get("enabled"))
        or not bool(filter_contract.get("evidence_cannot_bypass_filter"))
        or int(filter_contract.get("minimum_detected_cancers", -1)) != 3
        or float(filter_contract.get("within_cancer_min_sample_detection_rate", -1))
        != 0.10
    ):
        raise RuntimeError("Exact-pathway website release requires the frozen pan-cancer lncRNA filter")
    fold_manifest = read_table(tables / "fold_manifest.tsv")
    if len(fold_manifest) != EXPECTED_CANCERS:
        raise RuntimeError("Exact-pathway website release requires all 33 LOCO folds")

    dim_lnc = _dimension(
        read_table(standardized / "dim_lncRNA.parquet"),
        "lncrna_id",
        ["ensembl_gene_id", "gene_symbol", "gene_name", "gene_type", "chromosome"],
    )
    dim_pathway = _dimension(
        read_table(standardized / "dim_pathway.parquet"),
        "pathway_id",
        [
            "pathway_name",
            "pathway_source",
            "pathway_collection",
            "source_url",
            "gene_count",
            "version",
        ],
    )
    output_root.mkdir(parents=True)
    score_root = output_root / "web_exact_pathway_score"
    selected_root = output_root / "web_exact_pathway_selected"
    score_root.mkdir()
    selected_root.mkdir()
    geneset_candidate_parts: list[pd.DataFrame] = []
    pathway_summary_parts: list[pd.DataFrame] = []
    fold_audits: list[dict] = []
    ensemble_root = Path(args.ensemble_root).resolve()
    for row in fold_manifest.sort_values("test_cancer", kind="stable").itertuples(
        index=False
    ):
        cancer = str(row.test_cancer)
        fold_id = str(row.fold_id)
        success_path = ensemble_root / selected_model / fold_id / "SUCCESS.json"
        success = _read_success(
            success_path,
            run_id=args.run_id,
            selected_model=selected_model,
            training_lineage_gate_sha256=training_gate_sha256,
        )
        if str(success.get("test_cancer")) != cancer:
            raise RuntimeError(f"Fold/cancer mismatch in {success_path}")
        prediction = read_table(Path(success["prediction_file"]))
        forbidden = sorted(FORBIDDEN_LABEL_COLUMNS & set(prediction.columns))
        if forbidden:
            raise RuntimeError(f"Inference score contains held-out labels: {forbidden}")
        candidate_path = tables / "candidate_universe" / f"cancer_id={cancer}"
        annotation_columns = [
            *IDENTITY_COLUMNS,
            "direction",
            "observed_evidence_score",
            "bulk_detection_rate",
            "sc_detection_rate",
            "pancancer_detected_cancers",
            "bulk_support",
            "sc_ssgsea_support",
            "replication_support",
            "direction_consistency",
            "bulk_available",
            "sc_available",
        ]
        annotation = read_table(candidate_path, columns=annotation_columns).rename(
            columns={"direction": "observed_direction"}
        )
        merged = prediction.merge(
            annotation,
            on=IDENTITY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(prediction) or len(merged) != int(success["rows"]):
            raise RuntimeError(f"Candidate annotation coverage failure for {cancer}")
        merged = merged.merge(dim_lnc, on="lncrna_id", how="left", validate="many_to_one")
        merged = merged.merge(
            dim_pathway, on="pathway_id", how="left", validate="many_to_one"
        )
        classified = classify_exact_pathway_scores(merged, cfg["materialization"])
        classified["analysis_version"] = cfg["analysis_version"]
        classified["pathway_target_level"] = "exact_pathway"
        classified["pathway_family_role"] = "auxiliary_hierarchy_only"
        classified["selected_model"] = selected_model
        classified["selection_split"] = "val_only"
        classified["rank_within_cancer_pathway"] = classified.groupby(
            ["cancer_id", "pathway_id"], observed=True
        ).member_weight.rank(method="first", ascending=False)
        classified["rank_within_cancer_lncrna"] = classified.groupby(
            ["cancer_id", "lncrna_id"], observed=True
        ).member_weight.rank(method="first", ascending=False)
        score_path = score_root / f"cancer_id={cancer}" / "part-0.parquet"
        _atomic_table(classified, score_path)
        selected = classified.loc[
            classified.relationship_class.astype(str).ne("exploratory")
        ].copy()
        selected_path = selected_root / f"cancer_id={cancer}" / "part-0.parquet"
        _atomic_table(selected, selected_path)
        pathway_summary_parts.append(
            selected.groupby(["cancer_id", "pathway_id"], observed=True)
            .agg(
                selected_lncRNAs=("lncrna_id", "nunique"),
                max_probability=("calibrated_probability", "max"),
                mean_probability=("calibrated_probability", "mean"),
                max_member_weight=("member_weight", "max"),
            )
            .reset_index()
        )
        # Gene sets are capped per directional exact pathway. Keeping only
        # this deterministic prefix prevents all selected rows from all 33
        # cancers being accumulated in memory.
        max_members = int(cfg["materialization"]["max_members"])
        geneset_candidate_parts.append(
            selected.sort_values(
                [
                    "cancer_id",
                    "pathway_id",
                    "final_direction",
                    "member_weight",
                    "calibrated_probability",
                    "observed_evidence_score",
                    "lncrna_id",
                ],
                ascending=[True, True, True, False, False, False, True],
                kind="stable",
            )
            .groupby(
                ["cancer_id", "pathway_id", "final_direction"],
                observed=True,
                sort=False,
            )
            .head(max_members)
        )
        fold_audits.append(
            {
                "fold_id": fold_id,
                "cancer_id": cancer,
                "rows": int(len(classified)),
                "selected_rows": int(len(selected)),
                "score_sha256": file_sha256(score_path),
                "selected_sha256": file_sha256(selected_path),
                "inference_success_sha256": file_sha256(success_path),
            }
        )

    selected_for_genesets = (
        pd.concat(geneset_candidate_parts, ignore_index=True)
        if geneset_candidate_parts
        else pd.DataFrame()
    )
    master, member = build_exact_pathway_genesets(
        selected_for_genesets,
        analysis_version=cfg["analysis_version"],
        min_members=int(cfg["materialization"]["min_members"]),
        max_members=int(cfg["materialization"]["max_members"]),
    )
    if master.empty or member.empty:
        raise RuntimeError("Exact-pathway website release produced no eligible gene sets")
    if not member.empty:
        member["member_gene"] = member.get("gene_symbol", "").fillna("").astype(str)
        fallback = member.get("ensembl_gene_id", member.lncrna_id).fillna("").astype(str)
        member.loc[member.member_gene.eq(""), "member_gene"] = fallback
    write_table(master, output_root / "geneset_master_exact_pathway.parquet")
    write_table(member, output_root / "geneset_member_exact_pathway.parquet")
    default_classes = set(cfg["materialization"]["default_gmt_include"])
    extended_classes = set(cfg["materialization"]["extended_gmt_include"])
    min_members = int(cfg["materialization"]["min_members"])
    _write_gmt(
        output_root / "cancer_exact_pathway_lncRNA_symbols.gmt",
        master,
        member,
        default_classes,
        "member_gene",
        min_members,
    )
    _write_gmt(
        output_root / "cancer_exact_pathway_lncRNA_ids.gmt",
        master,
        member,
        default_classes,
        "lncrna_id",
        min_members,
    )
    _write_gmt(
        output_root / "cancer_exact_pathway_lncRNA_symbols_extended.gmt",
        master,
        member,
        extended_classes,
        "member_gene",
        min_members,
    )
    pathway_summary = (
        pd.concat(pathway_summary_parts, ignore_index=True)
        if pathway_summary_parts
        else pd.DataFrame(columns=["cancer_id", "pathway_id"])
    )
    write_table(pathway_summary, output_root / "web_cancer_exact_pathway_summary.parquet")
    payload = {
        "status": "PASS",
        "purpose": PURPOSE,
        "run_id": args.run_id,
        "selected_model": selected_model,
        "selection_split": "val_only",
        "test_metrics_used_for_selection": False,
        "pathway_target_level": "exact_pathway",
        "pathway_family_role": "auxiliary_hierarchy_only",
        "cancers": EXPECTED_CANCERS,
        "reference_only_cancers": [],
        "score_rows": int(sum(row["rows"] for row in fold_audits)),
        "selected_rows": int(sum(row["selected_rows"] for row in fold_audits)),
        "genesets": int(len(master)),
        "geneset_members": int(len(member)),
        "pancancer_eligible_lncRNAs": eligible_lncRNAs,
        "pancancer_lncrna_filter": {
            "enabled": True,
            "within_cancer_min_sample_detection_rate": 0.10,
            "minimum_detected_cancers": 3,
            "evidence_cannot_bypass_filter": True,
        },
        "heldout_label_columns_in_public_scores": False,
        "selection_sha256": file_sha256(selection_path),
        "formal_gate_sha256": execution_gate_sha256,
        "execution_site_gate_sha256": execution_gate_sha256,
        "training_lineage_gate_sha256": training_gate_sha256,
        "normalized_gate_contract_sha256": normalized_gate_contract_sha256,
        "relocated_execution": training_gate_sha256 != execution_gate_sha256,
        "fold_audits": fold_audits,
    }
    atomic_write_json(output_root / "SUCCESS.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
