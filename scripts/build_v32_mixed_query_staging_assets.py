#!/usr/bin/env python
"""Build sanitized V3.2 mixed-query assets without touching the production site."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.full_model_contract import validate_module_lineage
from cc_hhgt.v32.release_registry import V32_PREFIX, artifact_sha256


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _canonical_gene(value: Any) -> str:
    text = str(value).strip().upper()
    if text.startswith("GENE:"):
        text = text[5:]
    return text.split(".", 1)[0] if text.startswith("ENSG") else text


def _canonical_lnc(value: Any) -> str:
    text = str(value).strip().upper()
    if text.startswith("GENE:"):
        text = text[5:]
    if not text.startswith("LNC:"):
        text = "LNC:" + text
    prefix, inner = text.split(":", 1)
    if inner.startswith("ENSG"):
        inner = inner.split(".", 1)[0]
    return prefix + ":" + inner


KEY_COVERAGE_FORMAT = "CANCERLNCATLAS_V32_MIXED_QUERY_KEY_COVERAGE_V1"
KEY_COVERAGE_SCOPES = {
    "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_CARTESIAN",
    "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_OBSERVED_ROWS",
}


def _standardize_identifier_maps(
    gene_map: pd.DataFrame,
    lnc_map: pd.DataFrame,
    *,
    eligible_genes: set[str],
    eligible_lnc: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    gene_required = {"gene_id", "ensembl_gene_id", "gene_symbol", "aliases", "entity_type"}
    lnc_required = {"lncrna_id", "ensembl_gene_id", "gene_symbol", "aliases", "entity_type"}
    if not gene_required.issubset(gene_map.columns):
        raise ValueError(f"Static gene ID map lacks columns: {sorted(gene_required-set(gene_map.columns))}")
    if not lnc_required.issubset(lnc_map.columns):
        raise ValueError(f"Static lncRNA ID map lacks columns: {sorted(lnc_required-set(lnc_map.columns))}")
    gene = gene_map.loc[gene_map.entity_type.astype(str).eq("protein_coding_gene")].copy()
    gene["canonical_id"] = gene.ensembl_gene_id.map(_canonical_gene)
    gene = gene.loc[gene.canonical_id.isin(eligible_genes)]
    missing_gene = sorted(eligible_genes - set(gene.canonical_id))
    if missing_gene:
        raise ValueError(
            "Protein membership lacks authoritative protein-coding annotation: "
            f"{missing_gene[:5]}"
        )
    gene["identifier"] = gene.ensembl_gene_id.astype(str)
    gene["entity_type"] = "protein_coding_gene"
    gene["annotation_status"] = "AUTHORITATIVE_PROTEIN_CODING_ANNOTATION"
    lnc = lnc_map.copy()
    lnc["canonical_id"] = lnc.lncrna_id.map(_canonical_lnc)
    lnc = lnc.loc[lnc.canonical_id.isin(eligible_lnc)]
    lnc["identifier"] = lnc.ensembl_gene_id.astype(str)
    lnc["entity_type"] = "lncRNA"
    lnc["annotation_status"] = "STATIC_ANNOTATION_MATCH"
    missing_lnc = sorted(eligible_lnc - set(lnc.canonical_id))
    if missing_lnc:
        fallback = pd.DataFrame(
            {
                "identifier": [value.removeprefix("LNC:") for value in missing_lnc],
                "canonical_id": missing_lnc,
                "entity_type": "lncRNA",
                "gene_symbol": [value.removeprefix("LNC:") for value in missing_lnc],
                "aliases": "",
                "annotation_status": "CANONICAL_V32_PREDICTION_ID_FALLBACK",
            }
        )
        lnc = pd.concat([lnc, fallback], ignore_index=True)
    columns = [
        "identifier",
        "canonical_id",
        "entity_type",
        "gene_symbol",
        "aliases",
        "annotation_status",
    ]
    result = pd.concat([lnc[columns], gene[columns]], ignore_index=True)
    result = result.drop_duplicates(["entity_type", "canonical_id"]).sort_values(
        ["entity_type", "canonical_id"], kind="stable"
    )
    if result.empty or set(result.entity_type) != {"lncRNA", "protein_coding_gene"}:
        raise ValueError("Standardized identifier map lacks one required entity type")
    unreachable_lnc = sorted(eligible_lnc - set(result.loc[result.entity_type.eq("lncRNA"), "canonical_id"]))
    if unreachable_lnc:
        raise ValueError(f"V3.2 predicted lncRNAs remain unreachable: {unreachable_lnc[:5]}")
    audit = {
        "eligible_v32_lncrna_count": len(eligible_lnc),
        "static_annotation_match_count": len(eligible_lnc) - len(missing_lnc),
        "canonical_fallback_count": len(missing_lnc),
        "canonical_fallback_ids": missing_lnc,
        "unreachable_after_restandardization_count": len(unreachable_lnc),
        "silent_drop_count": 0,
    }
    return result.reset_index(drop=True), audit


def _load_key_coverage_receipt(path: Path, source_sha256: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Key-coverage receipt is unreadable") from exc
    required = {
        "format",
        "status",
        "scope",
        "source_exact_association_sha256",
        "eligible_pair_authority_sha256",
        "eligible_pair_count",
        "pathway_count",
        "expected_key_count",
        "observed_key_count",
        "missing_key_count",
        "duplicate_key_count",
        "complete_key_coverage",
        "unscored_key_encoding",
        "independent_of_asset_builder",
        "audit_engine",
    }
    missing = sorted(required - set(value)) if isinstance(value, dict) else sorted(required)
    if missing:
        raise ValueError(f"Key-coverage receipt lacks fields: {missing}")
    try:
        pair_count = int(value["eligible_pair_count"])
        pathway_count = int(value["pathway_count"])
        expected = int(value["expected_key_count"])
        observed = int(value["observed_key_count"])
        missing_count = int(value["missing_key_count"])
        duplicate_count = int(value["duplicate_key_count"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Key-coverage receipt counts are not integers") from exc
    if (
        value.get("format") != KEY_COVERAGE_FORMAT
        or value.get("status") != "PASS"
        or value.get("scope") not in KEY_COVERAGE_SCOPES
        or value.get("source_exact_association_sha256") != source_sha256
        or value.get("eligible_pair_authority_sha256") != source_sha256
        or value.get("complete_key_coverage") is not True
        or value.get("unscored_key_encoding")
        != "NULL_WITH_TYPED_NOT_EVALUATED_REASON"
        or value.get("independent_of_asset_builder") is not True
        or not str(value.get("audit_engine", "")).strip()
        or pair_count < 1
        or pathway_count < 1
        or (
            expected != pair_count * pathway_count
            if value.get("scope") == "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_CARTESIAN"
            else expected != observed
        )
        or observed != expected
        or missing_count != 0
        or duplicate_count != 0
    ):
        raise ValueError("Key-coverage receipt failed the complete Cartesian gate")
    return value


def _stream_exact_association(
    source: Path,
    destination: Path,
    *,
    lineage: dict[str, Any],
    coverage: dict[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """Validate and materialize the association in bounded-memory batches."""

    required = [
        "cancer_id",
        "lncrna_id",
        "pathway_id",
        "association_membership_probability",
        "analysis_version",
        "training_run_id",
        "pathway_target_level",
    ]
    dataset = ds.dataset(source, format="parquet")
    missing = sorted(set(required) - set(dataset.schema.names))
    if missing:
        raise ValueError(f"Exact association lacks columns: {missing}")
    optional = [
        name
        for name in ("availability", "eligible_for_mixed_query", "availability_reason")
        if name in dataset.schema.names
    ]
    output_columns = required + [
        "availability",
        "eligible_for_mixed_query",
        "availability_reason",
    ]
    output_schema = pa.schema(
        [
            ("cancer_id", pa.string()),
            ("lncrna_id", pa.string()),
            ("pathway_id", pa.string()),
            ("association_membership_probability", pa.float64()),
            ("analysis_version", pa.string()),
            ("training_run_id", pa.string()),
            ("pathway_target_level", pa.string()),
            ("availability", pa.bool_()),
            ("eligible_for_mixed_query", pa.bool_()),
            ("availability_reason", pa.string()),
        ]
    )
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    writer = pq.ParquetWriter(temporary, output_schema, compression="zstd")
    row_count = 0
    available_count = 0
    not_evaluated_reasons: dict[str, int] = {}
    eligible_lnc: set[str] = set()
    observed_pathways: set[str] = set()
    observed_pairs: set[tuple[str, str]] = set()
    try:
        scanner = dataset.scanner(columns=required + optional, batch_size=batch_size)
        for batch in scanner.to_batches():
            frame = batch.to_pandas()
            if frame.empty:
                continue
            if set(frame.analysis_version.astype(str)) != {str(lineage["analysis_version"])}:
                raise ValueError("Exact association/lineage analysis version mismatch")
            if set(frame.training_run_id.astype(str)) != {str(lineage["training_run_id"])}:
                raise ValueError("Exact association/lineage run mismatch")
            if not frame.pathway_target_level.astype(str).eq("exact_pathway").all():
                raise ValueError("Exact association contains a non-exact target")
            if not frame.analysis_version.astype(str).str.startswith(V32_PREFIX).all():
                raise ValueError("Exact association is not current V3.2")
            frame["cancer_id"] = frame.cancer_id.astype(str).str.strip().str.upper()
            frame["lncrna_id"] = frame.lncrna_id.map(_canonical_lnc)
            frame["pathway_id"] = frame.pathway_id.astype(str).str.strip()
            if (
                frame.cancer_id.eq("").any()
                or frame.lncrna_id.eq("LNC:").any()
                or frame.pathway_id.eq("").any()
            ):
                raise ValueError("Exact association contains an empty canonical key")
            if "availability" in frame:
                if not pd.api.types.is_bool_dtype(frame.availability.dtype):
                    raise ValueError("Exact association availability must be boolean")
                availability = frame.availability.fillna(False).astype(bool)
            else:
                availability = pd.Series(True, index=frame.index, dtype=bool)
            if "eligible_for_mixed_query" in frame:
                if not pd.api.types.is_bool_dtype(frame.eligible_for_mixed_query.dtype):
                    raise ValueError("eligible_for_mixed_query must be boolean")
                eligibility = frame.eligible_for_mixed_query.fillna(False).astype(bool)
            else:
                eligibility = pd.Series(True, index=frame.index, dtype=bool)
            if not eligibility.all():
                raise ValueError("Staging association contains ineligible mixed-query keys")
            raw_probability = frame.association_membership_probability.copy()
            probability = pd.to_numeric(raw_probability, errors="coerce")
            available_probability = probability.loc[availability]
            if available_probability.isna().any() or not available_probability.between(
                0.0, 1.0
            ).all():
                raise ValueError("Available exact-pathway probabilities must be finite in [0,1]")
            if raw_probability.loc[~availability].notna().any():
                raise ValueError("NOT_EVALUATED exact-pathway probabilities must remain null")
            if "availability_reason" not in frame:
                frame["availability_reason"] = None
            reasons = frame.availability_reason.fillna("").astype(str).str.strip()
            if reasons.loc[~availability].eq("").any():
                raise ValueError("NOT_EVALUATED exact-pathway keys require a typed reason")
            for reason, count in reasons.loc[~availability].value_counts().items():
                not_evaluated_reasons[str(reason)] = (
                    not_evaluated_reasons.get(str(reason), 0) + int(count)
                )
            frame["association_membership_probability"] = probability
            frame["availability"] = availability
            frame["eligible_for_mixed_query"] = eligibility
            frame["availability_reason"] = reasons.where(~availability, None)
            if frame.duplicated(["cancer_id", "lncrna_id", "pathway_id"]).any():
                raise ValueError("Exact association contains duplicate keys within a batch")
            eligible_lnc.update(frame.lncrna_id.astype(str))
            observed_pathways.update(frame.pathway_id.astype(str))
            observed_pairs.update(
                zip(frame.cancer_id.astype(str), frame.lncrna_id.astype(str), strict=True)
            )
            row_count += len(frame)
            available_count += int(availability.sum())
            writer.write_table(
                pa.Table.from_pandas(
                    frame[output_columns], schema=output_schema, preserve_index=False
                )
            )
    finally:
        writer.close()
    if row_count == 0:
        raise ValueError("Exact association is empty")
    if (
        row_count != int(coverage["observed_key_count"])
        or len(observed_pairs) != int(coverage["eligible_pair_count"])
        or len(observed_pathways) != int(coverage["pathway_count"])
    ):
        raise ValueError("Streamed association axes differ from the key-coverage receipt")
    os.replace(temporary, destination)
    return {
        "row_count": row_count,
        "available_key_count": available_count,
        "not_evaluated_key_count": row_count - available_count,
        "not_evaluated_reason_counts": dict(sorted(not_evaluated_reasons.items())),
        "eligible_lnc": eligible_lnc,
        "pathways": observed_pathways,
        "eligible_pair_count": len(observed_pairs),
        "batch_size": batch_size,
        "bounded_memory_streaming": True,
    }


def build_assets(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite staging assets: {output}")
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Stale staging temporary directory exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        lineage = json.loads(args.exact_lineage.read_text(encoding="utf-8"))
        validate_module_lineage("exact_pathway", lineage)
        source_association_sha256 = artifact_sha256(args.exact_association)
        if lineage.get("prediction_sha256") != source_association_sha256:
            raise ValueError("Exact association is not the prediction artifact pinned by lineage")
        coverage = _load_key_coverage_receipt(
            args.key_coverage_receipt, source_association_sha256
        )
        outputs = {
            "mixed_query_identifier_map": temporary / "identifier_map.parquet",
            "mixed_query_exact_pathway_membership": temporary / "exact_pathway_membership.parquet",
            "mixed_query_v32_lnc_exact_association": temporary / "v32_lnc_exact_association.parquet",
            "mixed_query_pathway_metadata": temporary / "pathway_metadata.parquet",
        }
        association_audit = _stream_exact_association(
            args.exact_association,
            outputs["mixed_query_v32_lnc_exact_association"],
            lineage=lineage,
            coverage=coverage,
            batch_size=args.batch_size,
        )

        membership = pd.read_parquet(args.exact_pathway_membership)
        if not {"pathway_id", "gene_id"}.issubset(membership.columns):
            raise ValueError("Exact pathway membership lacks pathway_id/gene_id")
        membership = membership[["pathway_id", "gene_id"]].copy()
        membership["pathway_id"] = membership.pathway_id.astype(str)
        membership["gene_id"] = membership.gene_id.map(_canonical_gene)
        membership = membership.drop_duplicates().sort_values(
            ["pathway_id", "gene_id"], kind="stable"
        )
        current_exact_pathways = set(association_audit["pathways"])
        missing_pathways = sorted(current_exact_pathways - set(membership.pathway_id))
        if missing_pathways:
            raise ValueError(f"Exact associations lack gene membership: {missing_pathways[:5]}")
        # The reusable annotation contains pathways outside the trained V3.2
        # candidate universe. They cannot enter this exact-model query surface.
        membership = membership.loc[
            membership.pathway_id.isin(current_exact_pathways)
        ].reset_index(drop=True)

        gene_map = pd.read_parquet(args.gene_identifier_map)
        gene_required = {"ensembl_gene_id", "entity_type"}
        if not gene_required.issubset(gene_map.columns):
            raise ValueError(
                f"Static gene ID map lacks columns: {sorted(gene_required-set(gene_map.columns))}"
            )
        protein_gene_ids = set(
            gene_map.loc[
                gene_map.entity_type.astype(str).eq("protein_coding_gene"),
                "ensembl_gene_id",
            ].map(_canonical_gene)
        )
        nonprotein_membership = membership.loc[
            ~membership.gene_id.isin(protein_gene_ids)
        ].copy()
        membership = membership.loc[
            membership.gene_id.isin(protein_gene_ids)
        ].reset_index(drop=True)
        pathways_without_protein_members = sorted(
            current_exact_pathways - set(membership.pathway_id)
        )
        if pathways_without_protein_members:
            raise ValueError(
                "Exact pathways lack authoritative protein-coding ORA members after filtering: "
                f"{pathways_without_protein_members[:5]}"
            )
        membership_audit = {
            "retained_protein_coding_membership_rows": int(len(membership)),
            "excluded_nonprotein_or_unregistered_rows": int(len(nonprotein_membership)),
            "excluded_identifier_count": int(nonprotein_membership.gene_id.nunique()),
            "excluded_identifiers": sorted(set(nonprotein_membership.gene_id.astype(str))),
            "affected_pathway_count": int(nonprotein_membership.pathway_id.nunique()),
            "pathways_without_protein_members": pathways_without_protein_members,
        }

        identifier, identifier_audit = _standardize_identifier_maps(
            gene_map,
            pd.read_parquet(args.lncrna_identifier_map),
            eligible_genes=set(membership.gene_id),
            eligible_lnc=set(association_audit["eligible_lnc"]),
        )
        metadata = pd.DataFrame({"pathway_id": sorted(set(membership.pathway_id))})
        metadata["pathway_source"] = metadata.pathway_id.str.split(":", n=1).str[0]
        metadata["pathway_name"] = metadata.pathway_id

        _atomic_parquet(identifier, outputs["mixed_query_identifier_map"])
        _atomic_parquet(membership, outputs["mixed_query_exact_pathway_membership"])
        _atomic_parquet(metadata, outputs["mixed_query_pathway_metadata"])
        manifest = {
            "status": "SUCCESS",
            "analysis_version": lineage["analysis_version"],
            "training_run_id": lineage["training_run_id"],
            "pathway_target_level": "exact_pathway",
            "source_exact_lineage_path": str(args.exact_lineage.resolve()),
            "source_exact_lineage_sha256": artifact_sha256(args.exact_lineage),
            "source_exact_association_sha256": source_association_sha256,
            "key_coverage": {
                key: coverage[key]
                for key in (
                    "format",
                    "scope",
                    "eligible_pair_authority_sha256",
                    "eligible_pair_count",
                    "pathway_count",
                    "expected_key_count",
                    "observed_key_count",
                    "missing_key_count",
                    "duplicate_key_count",
                    "complete_key_coverage",
                    "unscored_key_encoding",
                )
            },
            "key_coverage_receipt": {
                "path": str(args.key_coverage_receipt.resolve()),
                "sha256": artifact_sha256(args.key_coverage_receipt),
                "audit_engine": coverage["audit_engine"],
                "independent_of_asset_builder": True,
            },
            "source_static_annotations": {
                "exact_pathway_membership": {
                    "path": str(args.exact_pathway_membership.resolve()),
                    "sha256": artifact_sha256(args.exact_pathway_membership),
                },
                "gene_identifier_map": {
                    "path": str(args.gene_identifier_map.resolve()),
                    "sha256": artifact_sha256(args.gene_identifier_map),
                },
                "lncrna_identifier_map": {
                    "path": str(args.lncrna_identifier_map.resolve()),
                    "sha256": artifact_sha256(args.lncrna_identifier_map),
                },
            },
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "static_annotation_only_no_predictions_or_rankings": True,
            "pathway_family_broadcast": False,
            "membership_restricted_to_current_v32_exact_pathways": True,
            "historical_static_identifier_annotation_restandardized": True,
            "identifier_completion_audit": identifier_audit,
            "protein_membership_audit": membership_audit,
            "association_streaming_audit": {
                key: value
                for key, value in association_audit.items()
                if key not in {"eligible_lnc", "pathways"}
            },
            "rows": {
                "identifier_map": int(len(identifier)),
                "exact_pathway_membership": int(len(membership)),
                "v32_lnc_exact_association": int(association_audit["row_count"]),
                "pathway_metadata": int(len(metadata)),
            },
            "artifacts": {
                role: {
                    "filename": path.name,
                    "sha256": artifact_sha256(path),
                }
                for role, path in outputs.items()
            },
        }
        manifest_path = temporary / "ASSET_MANIFEST.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
        return {**manifest, "output_root": str(output)}
    except Exception:
        # Leave the temporary directory for forensic inspection; never publish it.
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-association", required=True, type=Path)
    parser.add_argument("--exact-lineage", required=True, type=Path)
    parser.add_argument("--key-coverage-receipt", required=True, type=Path)
    parser.add_argument("--exact-pathway-membership", required=True, type=Path)
    parser.add_argument("--gene-identifier-map", required=True, type=Path)
    parser.add_argument("--lncrna-identifier-map", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return args


def main() -> None:
    print(json.dumps(build_assets(parse_args()), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
