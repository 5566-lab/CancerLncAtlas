#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from cc_hhgt.candidates import build_candidate_universe
from cc_hhgt.common import load_config, read_table, write_table
from cc_hhgt.evidence import build_pair_evidence
from cc_hhgt.feature_builders import build_lnc_gene_coexpression, build_ucell_support
from cc_hhgt.graph_build import build_fold_manifests, build_graph_tables
from cc_hhgt.ids import build_crosswalks
from cc_hhgt.pathways import build_pathway_families, build_pathway_similarity
from cc_hhgt.pathway_target import (
    EXACT_PATHWAY_TARGET,
    pathway_target_level,
    require_exact_pathway_contract,
)
from cc_hhgt.standardize import (
    standardize_dimensions,
    standardize_drug_tables,
    standardize_interactions,
    standardize_pathway_members,
    standardize_single_cell,
)
from cc_hhgt.v30_integrity import (
    atomic_write_json,
    build_file_manifest,
    canonical_json_sha256,
    file_sha256,
    merkle_sha256,
    verify_file_manifest,
)
from cc_hhgt.v30_state_assets import build_v30_state_assets


def _manifest(path: Path) -> list[dict]:
    frame = pd.read_csv(path, sep="\t")
    required = {"relative_path", "size_bytes", "sha256"}
    if not required.issubset(frame):
        raise RuntimeError(f"Invalid file manifest {path}: missing {sorted(required - set(frame))}")
    return frame.to_dict("records")


def _prepare_protein_assets(cfg: dict) -> None:
    input_root = Path(cfg["_formal_input_root"])
    standardized = cfg["_standardized"]
    for name in ("protein_gene_map.parquet", "string_protein_edge.parquet", "drug_protein_target.parquet"):
        source = input_root / "processed" / name
        if not source.exists():
            raise FileNotFoundError(source)
        write_table(read_table(source), standardized / name)

    relation = read_table(standardized / "interaction_relation.parquet")
    protein_gene = read_table(standardized / "protein_gene_map.parquet")
    protein_rows = relation.loc[
        relation.partner_type.astype(str).str.lower().str.contains("protein")
        & relation.lncrna_id.notna()
        & relation.partner_id.notna()
    ].copy()
    by_gene = protein_rows.loc[protein_rows.partner_id.astype(str).str.startswith("GENE:")].merge(
        protein_gene[["gene_id", "protein_id"]].drop_duplicates(),
        left_on="partner_id",
        right_on="gene_id",
        how="inner",
    )[["lncrna_id", "protein_id"]]
    direct = protein_rows.loc[~protein_rows.partner_id.astype(str).str.startswith("GENE:")].copy()
    direct["uniprot_accession"] = (
        direct.partner_id.astype("string")
        .str.replace(r"^UNIPROT:", "", regex=True, case=False)
        .str.split(r"[;,\s|]+", regex=True)
        .str[0]
        .str.strip()
    )
    direct = direct.merge(
        protein_gene[["uniprot_accession", "protein_id"]].drop_duplicates("uniprot_accession"),
        on="uniprot_accession",
        how="inner",
    )[["lncrna_id", "protein_id"]]
    validated = pd.concat([by_gene, direct], ignore_index=True).dropna().drop_duplicates()
    validated["source_database"] = "frozen_interaction_relation_rebuilt"
    write_table(validated, standardized / "lncRNA_protein_relation.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild all V3.0 graph/state assets from one frozen canonical sample universe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument(
        "--repo-root",
        help="Relocated frozen code root; contents must match CODE_MANIFEST.tsv",
    )
    parser.add_argument("--output-root", required=True, help="The locked V3STATE run root")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    input_root = Path(args.input_root).resolve()
    input_manifest = Path(args.input_manifest).resolve()
    lock_path = output_root / "RUN_LOCK.json"
    provenance_path = output_root / "provenance" / "PROVENANCE.json"
    if not lock_path.exists() or not provenance_path.exists():
        raise RuntimeError("Output root is not a frozen V3STATE run")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if args.run_id != lock.get("run_id") or args.run_id != provenance.get("run_id"):
        raise RuntimeError("--run-id does not match the locked run root")
    expected_manifest = (output_root / "provenance" / "INPUT_MANIFEST.tsv").resolve()
    if input_manifest != expected_manifest:
        raise RuntimeError(f"--input-manifest must be the run snapshot: {expected_manifest}")
    input_records = _manifest(input_manifest)
    code_manifest = output_root / "provenance" / "CODE_MANIFEST.tsv"
    code_records = _manifest(code_manifest)
    repo_root = Path(args.repo_root or provenance["repo_root"]).resolve()
    mismatches = verify_file_manifest(input_root, input_records) + verify_file_manifest(repo_root, code_records)
    if mismatches:
        raise RuntimeError(f"Frozen inputs or code changed before asset build: {mismatches[:20]}")

    canonical_root = output_root / "canonical"
    canonical_success = json.loads((canonical_root / "SUCCESS.json").read_text(encoding="utf-8"))
    if canonical_success.get("status") != "PASS" or canonical_success.get("run_id") != args.run_id:
        raise RuntimeError("Canonical sample gate is missing or belongs to another run")
    assets_root = output_root / "assets"
    temporary_root = output_root / ".assets.tmp"
    if assets_root.exists() or temporary_root.exists():
        raise RuntimeError("Refusing to reuse formal assets; start a new run_id")

    cfg = load_config(
        args.config,
        project_root_override=input_root,
        create_dirs=False,
    )
    # The manifest is meaningful only if every relative input is resolved
    # against the exact frozen snapshot that was just verified.  Historical
    # code left ``_root`` at project_root and could therefore validate one
    # tree while reading another.
    cfg["_root"] = input_root
    cfg["project_root"] = str(input_root)
    cfg.pop("input_root", None)
    target_level = pathway_target_level(cfg)
    if target_level == EXACT_PATHWAY_TARGET:
        require_exact_pathway_contract(cfg)
    if bool(cfg.get("graph_contract", {}).get("episodic_pseudoheldout", False)):
        state_loss_weight = float(
            cfg.get("state_training", {}).get("auxiliary_loss_weight", float("nan"))
        )
        if state_loss_weight != 0.50:
            raise RuntimeError(
                "V3.1 episodic fixed-graph assets require "
                f"state_training.auxiliary_loss_weight=0.50, observed {state_loss_weight}"
            )
    temporary_root.mkdir()
    cfg["_results"] = temporary_root / "results"
    cfg["_standardized"] = temporary_root / "standardized"
    cfg["_cache"] = temporary_root / "cache"
    for key in ("_results", "_standardized", "_cache"):
        cfg[key].mkdir(parents=True, exist_ok=True)
    cfg["_canonical_samples_path"] = canonical_root / "canonical_samples.parquet"
    cfg["_canonical_state_complete_cases_path"] = canonical_root / "canonical_state_complete_cases.parquet"
    cfg["_canonical_eligibility_path"] = canonical_root / "state_complete_case_eligibility.tsv"
    cfg["_formal_input_root"] = input_root
    cfg["_sample_universe_sha256"] = canonical_success["sample_universe_sha256"]
    cfg["_formal_lineage"] = {
        "run_id": args.run_id,
        "input_merkle_sha256": provenance["input_merkle_sha256"],
        "code_merkle_sha256": provenance["code_merkle_sha256"],
        "config_merkle_sha256": provenance["config_merkle_sha256"],
        "schema_sha256": provenance["schema_merkle_sha256"],
        "environment_sha256": provenance["environment_sha256"],
    }

    dimensions = standardize_dimensions(cfg)
    standardize_pathway_members(cfg, dimensions["dim_gene"])
    standardize_interactions(cfg)
    standardize_single_cell(cfg)
    standardize_drug_tables(cfg)
    state_long = read_table(cfg["_canonical_state_complete_cases_path"])[
        ["sample_id", "patient_id", "cancer_id", "state_id", "state_value", "sample_universe_sha256", "run_id"]
    ].copy()
    state_long["state_type"] = "continuous"
    write_table(state_long, cfg["_standardized"] / "tumor_state_long.parquet")
    _prepare_protein_assets(cfg)
    crosswalk = build_crosswalks(cfg)
    drug_gene_path = cfg["_standardized"] / "drug_gene_target.parquet"
    drug_gene = read_table(drug_gene_path) if drug_gene_path.exists() else pd.DataFrame()
    if not {"drug_id", "gene_id"}.issubset(drug_gene.columns):
        # The frozen canonical protein target table already carries the
        # audited drug-to-gene mapping.  Some source snapshots lack the raw
        # DrugCentral file used by the legacy crosswalk builder, which used to
        # produce a schema-less empty parquet.
        canonical_targets = read_table(cfg["_standardized"] / "drug_protein_target.parquet")
        drug_gene = canonical_targets[["drug_id", "gene_id"]].dropna().drop_duplicates()
        drug_gene["mapping_method"] = "frozen_canonical_drug_protein_target"
        write_table(drug_gene, drug_gene_path)
        crosswalk["drug_gene_target"] = len(drug_gene)

    build_lnc_gene_coexpression(cfg)
    build_pathway_similarity(cfg)
    families, family_member = build_pathway_families(cfg)
    build_ucell_support(cfg)
    evidence = build_pair_evidence(cfg)
    candidates = build_candidate_universe(cfg)
    if target_level == EXACT_PATHWAY_TARGET:
        required_evidence = {
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            "pathway_family_id",
            "association_proxy_label",
            "strong_association_label",
            "label_semantics",
            "n_positive_association_sources",
        }
        if not required_evidence.issubset(evidence.columns):
            raise RuntimeError(
                "Exact-pathway evidence contract is incomplete: "
                f"{sorted(required_evidence - set(evidence.columns))}"
            )
        if evidence.duplicated(["cancer_id", "lncrna_id", "pathway_id"]).any():
            raise RuntimeError("Exact-pathway evidence identities are not unique")
        if (
            len(candidates) != int(cfg["formal_contract"]["expected_folds"])
            or set(candidates.target_level.astype(str)) != {"exact_pathway"}
            or pd.to_numeric(candidates.n_positive, errors="coerce").fillna(0).le(0).any()
        ):
            raise RuntimeError("Exact-pathway candidate manifest failed the 33-cancer positive gate")
    state_summary = build_v30_state_assets(cfg)
    nodes, edges = build_graph_tables(cfg)
    folds = build_fold_manifests(cfg)

    reference = set(map(str, cfg["analysis_cancers"]["reference_only"]))
    contract = cfg["formal_contract"]
    expected_folds = int(contract["expected_folds"])
    expected_models = int(contract["expected_models"])
    expected_seeds = int(contract["expected_seeds"])
    configured_tasks = int(contract["expected_tasks"])
    observed_test_cancers = set(folds.test_cancer.astype(str))
    required_formal = set(map(str, contract.get("required_formal_cancers", [])))
    if len(folds) != expected_folds or observed_test_cancers & reference:
        raise RuntimeError(
            f"Expected exactly {expected_folds} formal LOCO folds with no reference-only tests, "
            f"observed {len(folds)}"
        )
    if not required_formal.issubset(observed_test_cancers):
        raise RuntimeError(
            f"Required formal cancers are absent from LOCO tests: {sorted(required_formal - observed_test_cancers)}"
        )
    expected_tasks = len(folds) * expected_models * expected_seeds
    if expected_tasks != configured_tasks:
        raise RuntimeError(f"Formal task matrix is not {configured_tasks}: {expected_tasks}")
    if set(nodes.loc[nodes.node_type.astype(str).eq("state"), "canonical_id"].astype(str)) != set(cfg["sample_contract"]["target_states"]):
        raise RuntimeError("Formal graph state nodes differ from selected RNAss/DNAss targets")

    asset_records = build_file_manifest(temporary_root)
    asset_manifest = pd.DataFrame(asset_records)
    asset_manifest.to_csv(temporary_root / "ASSET_MANIFEST.tsv", sep="\t", index=False)
    asset_merkle = merkle_sha256(asset_records)
    success = {
        "status": "PASS",
        "run_id": args.run_id,
        "sample_universe_sha256": canonical_success["sample_universe_sha256"],
        "asset_merkle_sha256": asset_merkle,
        "asset_manifest_sha256": file_sha256(temporary_root / "ASSET_MANIFEST.tsv"),
        "n_asset_files": len(asset_records),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_folds": len(folds),
        "expected_tasks": expected_tasks,
        "reference_only": sorted(reference),
        "runtime_relocation": {
            "frozen_input_root": str(provenance["input_root"]),
            "runtime_input_root": str(input_root),
            "frozen_repo_root": str(provenance["repo_root"]),
            "runtime_repo_root": str(repo_root),
            "identity_basis": "verified content manifests and Merkle hashes",
        },
        "crosswalk": crosswalk,
        "n_pathway_families": len(families),
        "n_pathway_family_members": len(family_member),
        "pathway_target_level": target_level,
        "pathway_target_column": (
            "pathway_id" if target_level == EXACT_PATHWAY_TARGET else "pathway_family_id"
        ),
        "n_exact_pathways_in_evidence": (
            int(evidence.pathway_id.nunique()) if "pathway_id" in evidence else 0
        ),
        "n_evidence_rows": len(evidence),
        "candidate_manifest_rows": len(candidates),
        "pancancer_lncrna_filter": read_table(
            cfg["_results"] / "tables" / "pancancer_lncrna_eligibility.parquet"
        ).eligible.astype(bool).value_counts().to_dict(),
        "state": state_summary,
        "cache_key_sha256": canonical_json_sha256(
            {
                **cfg["_formal_lineage"],
                "sample_universe_sha256": canonical_success["sample_universe_sha256"],
                "fold_manifest_sha256": file_sha256(cfg["_results"] / "tables" / "fold_manifest.tsv"),
            }
        ),
    }
    mismatches = verify_file_manifest(input_root, input_records) + verify_file_manifest(repo_root, code_records)
    if mismatches:
        raise RuntimeError(f"Frozen inputs or code changed during asset build: {mismatches[:20]}")
    atomic_write_json(temporary_root / "SUCCESS.json", success)
    os.replace(temporary_root, assets_root)
    print(json.dumps({**success, "assets_root": str(assets_root)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
