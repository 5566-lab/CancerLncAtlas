"""Partitioned fresh V3.2 single-cell input construction from 10x H5.

Each cancer is rebuilt independently from ``raw_feature_bc_matrix.h5`` and
cell metadata.  Only GENCODE gene annotation plus the hash-pinned current
V3.2 exact membership/candidate authority are admitted.  Historical pathway
associations, predictions, rankings, checkpoints and ``sc_trajectory`` trees
are rejected by path and are never fallback inputs.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .input_lineage import artifact_sha256
from .single_cell_input_builder import (
    ANALYSIS_VERSION,
    EXACT_PATHWAY_COUNT,
    EXPECTED_CANCERS,
    KNOWN_FEATURE_UNIVERSE_LIMITATIONS,
    SingleCellInputBuildError,
    build_lnc_celltype_summary,
    compute_fresh_associations,
)


PARTITION_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_H5_PARTITION_V1"
ANNOTATION_FORMAT = "CC_HHGT_V3_2_GENCODE_GENE_ANNOTATION_V1"
CURRENT_V32_EXACT_MEMBERSHIP_SHA256 = (
    "0ae85904df979046fcbfb7f781977947392e99735b819f4d90803869831871ef"
)
CURRENT_V32_CANDIDATE_SHA256 = (
    "cced638f3d17ab9b1b0ff521169ac27dea3de1fc042adb7edefe6100a2ca071f"
)
CURRENT_V32_CANDIDATE_ROWS = 3_300_000
CURRENT_V32_EXACT_MEMBERSHIP_EDGES = 352_205
LOW_FEATURE_UNIVERSE_THRESHOLD = 1_000

_PRIMARY_CHROMS = frozenset(
    [*(f"chr{index}" for index in range(1, 23)), "chrX", "chrY"]
)
_FORBIDDEN_PATH_TOKENS = (
    "checkpoint",
    "prediction",
    "probability",
    "ranking",
    "ranked",
    "sc_trajectory",
    "pathway_association",
    "lnc_exact_pathway",
    "pair_support",
    "family_support",
)
_SOURCE_TIER_TO_TRAINER = {
    "raw_counts": "primary_raw_count",
    "source_normalized": "approved",
    "legacy_counts": "approved",
}
_ATTR_PATTERN = re.compile(r'(\w+) "([^"]*)"')


class SingleCellPartitionBuildError(RuntimeError):
    """Raised when a cancer partition cannot prove fresh source lineage."""


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _stable_gene_id(value: Any) -> str:
    text = str(value).strip()
    upper = text.upper()
    for prefix in ("GENE:", "PROTEIN:", "LNC:", "LNCRNA:"):
        if upper.startswith(prefix):
            text = text[len(prefix) :]
            break
    return re.sub(r"\.\d+$", "", text)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_fresh_source_path(path: str | Path, role: str) -> Path:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    nearby = "/".join(_token(part) for part in source.parts[-4:])
    found = sorted(token for token in _FORBIDDEN_PATH_TOKENS if token in nearby)
    if found:
        raise SingleCellPartitionBuildError(
            f"{role} path is a forbidden historical/result source: {source}; tokens={found}"
        )
    if source.suffix.lower() in {".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib"}:
        raise SingleCellPartitionBuildError(f"{role} cannot be a model/checkpoint: {source}")
    return source


def _decode(values: Any) -> list[str]:
    return [
        item.decode("utf-8", errors="replace")
        if isinstance(item, (bytes, np.bytes_))
        else str(item)
        for item in values
    ]


def prepare_gencode_annotation(
    *,
    gtf_path: str | Path,
    output_parquet: str | Path,
    provenance_path: str | Path,
) -> dict[str, Any]:
    """Create a stable-ID/symbol annotation cache directly from a GENCODE GTF."""

    source = _assert_fresh_source_path(gtf_path, "GENCODE annotation")
    output = Path(output_parquet).resolve()
    provenance = Path(provenance_path).resolve()
    if output.exists() or provenance.exists():
        raise SingleCellPartitionBuildError("GENCODE annotation cache output reuse is forbidden")
    opener = gzip.open if source.suffix.lower() == ".gz" else open
    rows: list[dict[str, str]] = []
    with opener(source, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene" or fields[0] not in _PRIMARY_CHROMS:
                continue
            attributes = dict(_ATTR_PATTERN.findall(fields[8]))
            gene_id = _stable_gene_id(attributes.get("gene_id", ""))
            symbol = str(attributes.get("gene_name", "")).strip()
            gene_type = _token(attributes.get("gene_type", attributes.get("gene_biotype", "")))
            if gene_type == "lncrna":
                gene_class = "lncRNA"
            elif gene_type == "protein_coding":
                gene_class = "protein_coding"
            else:
                continue
            if gene_id and symbol:
                rows.append(
                    {
                        "gene_id": gene_id,
                        "gene_symbol": symbol,
                        "gene_class": gene_class,
                        "chromosome": fields[0],
                    }
                )
    frame = pd.DataFrame(rows).drop_duplicates()
    if frame.empty or frame.gene_id.duplicated().any():
        duplicates = frame.loc[frame.gene_id.duplicated(False), "gene_id"].head(10).tolist()
        raise SingleCellPartitionBuildError(
            f"GENCODE gene annotation is empty or has conflicting stable IDs: {duplicates}"
        )
    symbol_counts = frame.groupby("gene_symbol", observed=True).gene_id.nunique()
    frame["unique_symbol"] = frame.gene_symbol.map(symbol_counts).eq(1)
    frame = frame.sort_values(["gene_class", "gene_id"], kind="stable").reset_index(drop=True)
    _atomic_parquet(frame, output)
    payload = {
        "format": ANNOTATION_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "source_gtf_path": str(source),
        "source_gtf_sha256": artifact_sha256(source),
        "annotation_path": str(output),
        "annotation_sha256": artifact_sha256(output),
        "rows": int(len(frame)),
        "lncrna_rows": int(frame.gene_class.eq("lncRNA").sum()),
        "protein_coding_rows": int(frame.gene_class.eq("protein_coding").sum()),
        "primary_chromosomes_only": True,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
        "historical_checkpoints_used": False,
    }
    payload["contract_sha256"] = _canonical_sha256(payload)
    _atomic_json(provenance, payload)
    return payload


def load_current_v32_authority(
    membership_path: str | Path,
    candidate_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Hash-bind and restrict the current V3.2 authority to exact 2,135 pathways."""

    membership_source = _assert_fresh_source_path(membership_path, "exact membership")
    candidate_source = _assert_fresh_source_path(candidate_path, "exact candidates")
    membership_sha = artifact_sha256(membership_source)
    candidate_sha = artifact_sha256(candidate_source)
    if membership_sha != CURRENT_V32_EXACT_MEMBERSHIP_SHA256:
        raise SingleCellPartitionBuildError(
            "Exact membership is not the pinned current V3.2 source: "
            f"expected={CURRENT_V32_EXACT_MEMBERSHIP_SHA256}, observed={membership_sha}"
        )
    if candidate_sha != CURRENT_V32_CANDIDATE_SHA256:
        raise SingleCellPartitionBuildError(
            "Candidate universe is not the pinned current V3.2 source: "
            f"expected={CURRENT_V32_CANDIDATE_SHA256}, observed={candidate_sha}"
        )
    candidates = pd.read_parquet(
        candidate_source, columns=["cancer_id", "lncrna_id", "pathway_id"]
    ).dropna()
    candidates["cancer_id"] = candidates.cancer_id.astype(str).str.upper().str.strip()
    candidates["lncrna_id"] = candidates.lncrna_id.astype(str).map(
        lambda value: "LNC:" + _stable_gene_id(value)
    )
    candidates["pathway_id"] = candidates.pathway_id.astype(str).str.strip()
    if len(candidates) != CURRENT_V32_CANDIDATE_ROWS:
        raise SingleCellPartitionBuildError(
            f"Expected {CURRENT_V32_CANDIDATE_ROWS:,} candidate rows; observed={len(candidates):,}"
        )
    if set(candidates.cancer_id) != EXPECTED_CANCERS:
        raise SingleCellPartitionBuildError("Pinned candidate file does not cover exact 33 cancers")
    if candidates.pathway_id.nunique() != EXACT_PATHWAY_COUNT:
        raise SingleCellPartitionBuildError("Pinned candidate file is not exact 2,135 pathways")
    if candidates.duplicated(["cancer_id", "lncrna_id", "pathway_id"]).any():
        raise SingleCellPartitionBuildError("Pinned candidate keys are duplicated")

    raw_membership = pd.read_parquet(membership_source, columns=["pathway_id", "gene_id"])
    raw_membership["pathway_id"] = raw_membership.pathway_id.astype(str).str.strip()
    raw_membership["gene_id"] = raw_membership.gene_id.map(_stable_gene_id)
    exact_ids = set(candidates.pathway_id)
    membership = raw_membership.loc[raw_membership.pathway_id.isin(exact_ids)].copy()
    membership = membership.drop_duplicates(["pathway_id", "gene_id"])
    if (
        membership.pathway_id.nunique() != EXACT_PATHWAY_COUNT
        or len(membership) != CURRENT_V32_EXACT_MEMBERSHIP_EDGES
    ):
        raise SingleCellPartitionBuildError(
            "Current membership filtered by current candidates must yield "
            f"{CURRENT_V32_EXACT_MEMBERSHIP_EDGES:,} edges/2,135 pathways; "
            f"observed={len(membership):,}/"
            f"{membership.pathway_id.nunique():,}"
        )
    membership["membership_weight"] = 1.0
    membership = membership.sort_values(["pathway_id", "gene_id"], kind="stable").reset_index(drop=True)
    audit = {
        "membership_source_path": str(membership_source),
        "membership_source_sha256": membership_sha,
        "membership_source_pathways": int(raw_membership.pathway_id.nunique()),
        "exact_membership_edges": int(len(membership)),
        "exact_membership_pathways": int(membership.pathway_id.nunique()),
        "candidate_path": str(candidate_source),
        "candidate_sha256": candidate_sha,
        "candidate_rows": int(len(candidates)),
        "candidate_cancers": int(candidates.cancer_id.nunique()),
        "candidate_exact_pathways": int(candidates.pathway_id.nunique()),
        "family_to_exact_broadcast": False,
    }
    return membership, candidates, audit


def _manifest_row(
    *,
    dataset_id: str,
    cancer: str,
    source_tier: str,
    measurement_scale: str,
    lnc_feature_count: int,
    donor_metadata_available: bool,
    formal_eligible: bool,
    quality_flags: list[str],
) -> pd.DataFrame:
    limited = (
        cancer in KNOWN_FEATURE_UNIVERSE_LIMITATIONS
        or int(lnc_feature_count) < LOW_FEATURE_UNIVERSE_THRESHOLD
    )
    flags = list(dict.fromkeys(quality_flags))
    if int(lnc_feature_count) < LOW_FEATURE_UNIVERSE_THRESHOLD:
        flags.append("LOW_LNCRNA_FEATURE_UNIVERSE")
    if cancer in KNOWN_FEATURE_UNIVERSE_LIMITATIONS:
        flags.append("KNOWN_SOURCE_FEATURE_UNIVERSE_LIMITATION")
    return pd.DataFrame(
        {
            "dataset_id": [dataset_id],
            "cancer_id": [cancer],
            "expression_source_tier": [source_tier],
            "measurement_scale": [measurement_scale],
            "source_generation": ["V3.2_FRESH_FROM_RAW_FEATURE_BC_MATRIX_H5"],
            "formal_eligible": [bool(formal_eligible and not limited and donor_metadata_available)],
            "quality_status": ["LIMITED" if limited else "PASS"],
            "model_derived": [False],
            "outcome_derived": [False],
            "lncrna_feature_universe_count": [int(lnc_feature_count)],
            "donor_metadata_available": [bool(donor_metadata_available)],
            "feature_universe_status": ["LIMITED" if limited else "PASS"],
            "quality_flags": [";".join(dict.fromkeys(flags)) if flags else "NONE"],
            "source_tier": [_SOURCE_TIER_TO_TRAINER[source_tier]],
            "manifest_authority": ["V3.2_33_CANCER_SINGLE_CELL_SOURCE_MANIFEST"],
        }
    )


def _empty_association() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "lncrna_id", "pathway_id", "rho", "p_value", "fdr", "dataset_id",
            "cancer_id", "cell_population", "analysis_context", "n_patients",
            "n_pseudobulk_groups", "fdr_family_size", "source_tier",
            "expression_source_tier", "association_method", "generation",
        ]
    )


def build_exact_pathway_availability(
    exact_pathway_ids: list[str],
    activity: pd.DataFrame | None,
    association: pd.DataFrame | None,
    *,
    partition_blocking_reason: str | None = None,
) -> pd.DataFrame:
    """Materialise exact-pathway availability without numeric imputation."""

    exact_ids = sorted(set(map(str, exact_pathway_ids)))
    activity_ids = (
        set(activity.pathway_id.dropna().astype(str))
        if activity is not None and not activity.empty
        else set()
    )
    association_ids = (
        set(association.pathway_id.dropna().astype(str))
        if association is not None and not association.empty
        else set()
    )
    rows = []
    for pathway in exact_ids:
        activity_available = pathway in activity_ids
        association_available = pathway in association_ids
        if partition_blocking_reason:
            activity_reason = partition_blocking_reason
            association_reason = partition_blocking_reason
        else:
            activity_reason = None if activity_available else "NO_MEMBER_PROTEIN_IN_MATRIX"
            if association_available:
                association_reason = None
            elif not activity_available:
                association_reason = "NO_MEMBER_PROTEIN_IN_MATRIX"
            else:
                association_reason = "NO_FINITE_DONOR_ASSOCIATION"
        rows.append(
            {
                "pathway_id": pathway,
                "activity_available": activity_available,
                "activity_unavailable_reason": activity_reason,
                "association_available": association_available,
                "association_unavailable_reason": association_reason,
                "numeric_imputation_used": False,
            }
        )
    return pd.DataFrame(rows)


def compute_exact_pathway_activity_from_aggregates(
    *,
    aggregate_expression: np.ndarray,
    gene_order: list[str],
    gene_class: Mapping[str, str],
    group_records: pd.DataFrame,
    group_cell_counts: np.ndarray,
    membership: pd.DataFrame,
    dataset_id: str,
    cancer_id: str,
    source_tier: str,
    expression_source_tier: str,
    quality_flags: str,
    feature_universe_status: str,
) -> pd.DataFrame:
    """Compute protein-only rank-mean activity without an edge-expanded join.

    The previous generic implementation materialised group-by-membership-edge
    rows.  This algebraically identical sparse matrix implementation keeps the
    formal 2,135-way result while making large cancer partitions tractable.
    """

    try:
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - server dependency guard
        raise SingleCellPartitionBuildError("scipy is required for pathway activity") from exc
    protein_positions = [
        index for index, gene in enumerate(gene_order)
        if gene_class[gene] == "protein_coding"
    ]
    if not protein_positions:
        raise SingleCellPartitionBuildError("No protein genes are available for activity")
    protein_ids = [gene_order[index] for index in protein_positions]
    protein_index = {gene: index for index, gene in enumerate(protein_ids)}
    protein_values = np.asarray(aggregate_expression[protein_positions, :], dtype=float)
    ranks = pd.DataFrame(protein_values).rank(
        axis=0, method="average", pct=True
    ).to_numpy(dtype=np.float32)

    pathway_ids = sorted(set(membership.pathway_id.astype(str)))
    pathway_index = {pathway: index for index, pathway in enumerate(pathway_ids)}
    edges = membership.loc[membership.gene_id.astype(str).isin(protein_index)].copy()
    if edges.empty:
        raise SingleCellPartitionBuildError(
            "No GENCODE protein expression overlaps current exact membership"
        )
    row = edges.pathway_id.astype(str).map(pathway_index).to_numpy(np.int64)
    column = edges.gene_id.astype(str).map(protein_index).to_numpy(np.int64)
    weight = pd.to_numeric(edges.membership_weight, errors="raise").to_numpy(np.float32)
    pathway_gene = sparse.csr_matrix(
        (weight, (row, column)), shape=(len(pathway_ids), len(protein_ids))
    )
    weight_sum = np.asarray(pathway_gene.sum(axis=1)).reshape(-1)
    present = weight_sum > 0
    pathway_ids = [pathway for pathway, keep in zip(pathway_ids, present, strict=True) if keep]
    activity_values = np.asarray(pathway_gene[present] @ ranks, dtype=float)
    activity_values /= weight_sum[present, None]
    n_genes = np.asarray(pathway_gene[present].getnnz(axis=1), dtype=int)
    if not np.isfinite(activity_values).all():
        raise SingleCellPartitionBuildError("Fresh exact-pathway activity is not finite")

    number_pathways, number_groups = activity_values.shape
    result = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "cancer_id": cancer_id,
            "donor_id": np.tile(group_records.donor_id.to_numpy(object), number_pathways),
            "cell_type": np.tile(group_records.cell_type.to_numpy(object), number_pathways),
            "cell_state": np.tile(group_records.cell_state.to_numpy(object), number_pathways),
            "pathway_id": np.repeat(np.asarray(pathway_ids, dtype=object), number_groups),
            "source_tier": source_tier,
            "expression_source_tier": expression_source_tier,
            "quality_flags": quality_flags,
            "feature_universe_status": feature_universe_status,
            "n_genes": np.repeat(n_genes, number_groups),
            "n_cells": np.tile(np.asarray(group_cell_counts, dtype=int), number_pathways),
            "activity": activity_values.reshape(number_pathways * number_groups),
            "activity_method": "V3.2_PROTEIN_ONLY_WITHIN_PSEUDOBULK_RANK_MEAN_SPARSE_V2",
            "generation": "V3.2_FRESH_FROM_RAW_FEATURE_BC_MATRIX_H5",
        }
    )
    return result.sort_values(
        ["cancer_id", "dataset_id", "donor_id", "cell_type", "cell_state", "pathway_id"],
        kind="stable",
    ).reset_index(drop=True)


def _write_blocked_partition(
    *,
    cancer: str,
    h5_path: Path,
    metadata_path: Path,
    output_root: Path,
    source_tier: str,
    measurement_scale: str,
    lnc_feature_count: int,
    authority: Mapping[str, Any],
    exact_pathway_ids: list[str],
) -> dict[str, Any]:
    if output_root.exists():
        raise SingleCellPartitionBuildError(f"Partition output reuse is forbidden: {output_root}")
    temporary = output_root.with_name(f".{output_root.name}.{os.getpid()}.tmp")
    temporary.mkdir(parents=True)
    try:
        manifest = _manifest_row(
            dataset_id=f"SC_UNAVAILABLE_{cancer}",
            cancer=cancer,
            source_tier=source_tier,
            measurement_scale=measurement_scale,
            lnc_feature_count=lnc_feature_count,
            donor_metadata_available=False,
            formal_eligible=False,
            quality_flags=["DONOR_CELLTYPE_METADATA_UNAVAILABLE"],
        )
        _atomic_parquet(manifest, temporary / "dataset_manifest_row.parquet")
        availability = build_exact_pathway_availability(
            exact_pathway_ids,
            None,
            None,
            partition_blocking_reason="DONOR_CELLTYPE_METADATA_UNAVAILABLE",
        )
        _atomic_parquet(
            availability, temporary / "exact_pathway_availability.parquet"
        )
        status = {
            "format": PARTITION_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "cancer_id": cancer,
            "status": "BLOCKED_MISSING_CELL_METADATA",
            "formal_eligible": False,
            "blocking_reason": "DONOR_CELLTYPE_METADATA_UNAVAILABLE",
            "source_tier": source_tier,
            "measurement_scale": measurement_scale,
            # Metadata-blocked partitions never open the H5 matrix, so this is
            # only the earlier read-only audit count.  Keep the legacy generic
            # field for manifest compatibility, but make the fresh/direct-ID
            # field explicitly unavailable so no downstream audit can mistake
            # the audit value for a newly measured count.
            "lncrna_feature_universe_count": int(lnc_feature_count),
            "lncrna_feature_universe_count_semantics": (
                "READ_ONLY_AUDIT_COUNT_METADATA_BLOCKED_H5_NOT_SCANNED"
            ),
            "audited_lncrna_feature_universe_count": int(lnc_feature_count),
            "fresh_direct_id_lncrna_feature_count": None,
            "fresh_minus_audit_feature_count": None,
            "feature_count_tolerance": None,
            "feature_universe_status": str(manifest.feature_universe_status.iloc[0]),
            "h5_path": str(h5_path),
            "h5_sha256": artifact_sha256(h5_path),
            "metadata_path": str(metadata_path),
            "metadata_exists": False,
            "authority": dict(authority),
            "exact_pathway_availability_path": str(
                output_root / "exact_pathway_availability.parquet"
            ),
            "exact_pathway_availability_sha256": artifact_sha256(
                temporary / "exact_pathway_availability.parquet"
            ),
            "activity_missing_exact_pathway_ids": sorted(exact_pathway_ids),
            "association_missing_exact_pathway_ids": sorted(exact_pathway_ids),
            "numeric_imputation_used": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "historical_sc_trajectory_used": False,
            "fresh_ucell_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
            "fresh_pseudotime_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
            "single_cell_module_complete": False,
            "release_ready": False,
            "production_deployed": False,
        }
        status["contract_sha256"] = _canonical_sha256(status)
        _atomic_json(temporary / "PARTITION_STATUS.json", status)
        os.replace(temporary, output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return status


def build_single_cell_partition(
    *,
    cancer_id: str,
    h5_path: str | Path,
    metadata_path: str | Path,
    annotation_parquet_path: str | Path,
    annotation_provenance_path: str | Path,
    membership_path: str | Path,
    candidate_path: str | Path,
    output_root: str | Path,
    source_tier: str,
    measurement_scale: str,
    audited_lnc_feature_count: int,
    min_association_observations: int = 5,
    association_chunk_size: int = 100_000,
) -> dict[str, Any]:
    """Build one atomic cancer partition directly from H5 and cell metadata."""

    cancer = str(cancer_id).upper().strip()
    if cancer not in EXPECTED_CANCERS:
        raise SingleCellPartitionBuildError(f"Cancer is outside V3.2 authority: {cancer}")
    source_tier = _token(source_tier)
    if source_tier not in _SOURCE_TIER_TO_TRAINER:
        raise SingleCellPartitionBuildError(f"Unsupported source tier: {source_tier}")
    if int(min_association_observations) < 3:
        raise ValueError("min_association_observations must be at least 3")
    h5 = _assert_fresh_source_path(h5_path, "single-cell H5")
    if h5.name != "raw_feature_bc_matrix.h5" or h5.parent.name.upper() != cancer:
        raise SingleCellPartitionBuildError(
            "H5 must be processed/sc_tool_input/<CANCER>/raw_feature_bc_matrix.h5"
        )
    metadata = Path(metadata_path).resolve()
    output = Path(output_root).resolve()
    membership, candidates, authority = load_current_v32_authority(
        membership_path, candidate_path
    )
    local_candidates = candidates.loc[candidates.cancer_id.eq(cancer)].copy()
    if local_candidates.empty:
        raise SingleCellPartitionBuildError(f"No current V3.2 candidates for {cancer}")
    if not metadata.is_file():
        return _write_blocked_partition(
            cancer=cancer,
            h5_path=h5,
            metadata_path=metadata,
            output_root=output,
            source_tier=source_tier,
            measurement_scale=measurement_scale,
            lnc_feature_count=int(audited_lnc_feature_count),
            authority=authority,
            exact_pathway_ids=sorted(set(membership.pathway_id.astype(str))),
        )
    metadata = _assert_fresh_source_path(metadata, "cell metadata")
    annotation_path = _assert_fresh_source_path(annotation_parquet_path, "GENCODE cache")
    annotation_provenance = _assert_fresh_source_path(
        annotation_provenance_path, "GENCODE provenance"
    )
    provenance_payload = json.loads(annotation_provenance.read_text(encoding="utf-8"))
    if provenance_payload.get("format") != ANNOTATION_FORMAT:
        raise SingleCellPartitionBuildError("GENCODE cache provenance has wrong format")
    if provenance_payload.get("annotation_sha256") != artifact_sha256(annotation_path):
        raise SingleCellPartitionBuildError("GENCODE annotation cache SHA mismatch")
    if output.exists():
        raise SingleCellPartitionBuildError(f"Partition output reuse is forbidden: {output}")

    annotation = pd.read_parquet(annotation_path)
    required_annotation = {"gene_id", "gene_symbol", "gene_class", "unique_symbol"}
    if missing := sorted(required_annotation - set(annotation.columns)):
        raise SingleCellPartitionBuildError(f"GENCODE cache lacks columns: {missing}")
    annotation["gene_id"] = annotation.gene_id.map(_stable_gene_id)
    by_id = annotation.set_index("gene_id")
    by_symbol = annotation.loc[annotation.unique_symbol.astype(bool)].set_index("gene_symbol")
    candidate_lnc = set(local_candidates.lncrna_id.astype(str))
    member_proteins = set(membership.gene_id.astype(str))

    try:
        import h5py
        from scipy import sparse
    except ImportError as exc:  # pragma: no cover - exercised on server environment
        raise SingleCellPartitionBuildError("h5py and scipy are required for H5 build") from exc

    cell_meta = pd.read_parquet(metadata)
    required_meta = {"cell_id", "patient_id", "cell_type_major", "dataset_id", "cancer_id"}
    if missing := sorted(required_meta - set(cell_meta.columns)):
        raise SingleCellPartitionBuildError(f"Cell metadata lacks columns: {missing}")
    if cell_meta.cell_id.astype(str).duplicated().any():
        raise SingleCellPartitionBuildError("Cell metadata cell_id values are duplicated")
    if set(cell_meta.cancer_id.astype(str).str.upper()) != {cancer}:
        raise SingleCellPartitionBuildError("Cell metadata cancer_id disagrees with partition")
    dataset_ids = set(cell_meta.dataset_id.astype(str).str.strip())
    if len(dataset_ids) != 1 or "" in dataset_ids:
        raise SingleCellPartitionBuildError("Cell metadata must declare exactly one dataset_id")
    dataset_id = next(iter(dataset_ids))

    with h5py.File(h5, "r") as handle:
        group = handle["matrix"] if "matrix" in handle else handle
        required_h5 = {"data", "indices", "indptr", "shape", "barcodes", "features"}
        if not required_h5.issubset(group.keys()):
            raise SingleCellPartitionBuildError("H5 does not contain a 10x sparse matrix")
        shape = tuple(int(value) for value in np.asarray(group["shape"]))
        barcodes = _decode(group["barcodes"][:])
        if shape[1] != len(barcodes):
            raise SingleCellPartitionBuildError("H5 matrix/barcode dimensions disagree")
        feature_group = group["features"]
        ids = _decode(feature_group["id"][:])
        names = _decode(feature_group["name"][:])
        if shape[0] != len(ids) or len(ids) != len(names):
            raise SingleCellPartitionBuildError("H5 matrix/feature dimensions disagree")

        indexer = pd.Index(cell_meta.cell_id.astype(str)).get_indexer(barcodes)
        if (indexer < 0).any() or len(cell_meta) != len(barcodes):
            raise SingleCellPartitionBuildError(
                "H5 barcodes and cell metadata must match one-to-one"
            )
        cell_meta = cell_meta.iloc[indexer].reset_index(drop=True)
        include = np.ones(len(cell_meta), dtype=bool)
        if "doublet_flag" in cell_meta:
            value = cell_meta.doublet_flag
            if pd.api.types.is_bool_dtype(value):
                doublet = value.fillna(False).to_numpy(bool)
            else:
                lowered = value.fillna("false").astype(str).str.lower().str.strip()
                if not set(lowered.unique()).issubset({"true", "false", "1", "0", "yes", "no"}):
                    raise SingleCellPartitionBuildError("doublet_flag is not explicit boolean")
                doublet = lowered.isin({"true", "1", "yes"}).to_numpy(bool)
            include &= ~doublet
        donor = cell_meta.patient_id.fillna("").astype(str).str.strip()
        cell_type = cell_meta.cell_type_major.fillna("").astype(str).str.strip()
        state = (
            cell_meta.cell_state.fillna("").astype(str).str.strip()
            if "cell_state" in cell_meta
            else pd.Series("", index=cell_meta.index)
        )
        include &= donor.ne("").to_numpy() & cell_type.ne("").to_numpy()
        if not include.any():
            raise SingleCellPartitionBuildError("No cells retain donor and cell-type metadata")
        group_frame = pd.DataFrame(
            {
                "donor_id": donor[include].to_numpy(),
                "cell_type": cell_type[include].to_numpy(),
                "cell_state": state[include].to_numpy(),
            }
        )
        group_index = pd.MultiIndex.from_frame(group_frame).drop_duplicates()
        group_lookup = {key: index for index, key in enumerate(group_index.tolist())}
        group_codes = np.asarray(
            [group_lookup[key] for key in map(tuple, group_frame.to_numpy(object))],
            dtype=np.int32,
        )
        included_cells = np.flatnonzero(include)
        group_cell_counts = np.bincount(group_codes, minlength=len(group_index)).astype(int)

        mapped: list[tuple[int, str, str]] = []
        mapped_lnc_all: set[str] = set()
        for feature_index, (raw_id, raw_name) in enumerate(zip(ids, names, strict=True)):
            direct = _stable_gene_id(raw_id)
            if direct in by_id.index:
                row = by_id.loc[direct]
                stable = direct
            elif raw_name in by_symbol.index:
                row = by_symbol.loc[raw_name]
                stable = str(row.name) if str(row.name).startswith("ENSG") else str(row.gene_id)
                # ``row.name`` is the symbol index; recover the stable ID explicitly.
                stable = str(annotation.loc[
                    annotation.gene_symbol.eq(raw_name) & annotation.unique_symbol.astype(bool),
                    "gene_id",
                ].iloc[0])
            else:
                continue
            gene_class = str(row.gene_class)
            if gene_class == "lncRNA":
                mapped_lnc_all.add(stable)
                typed = "LNC:" + stable
                if typed in candidate_lnc:
                    mapped.append((feature_index, typed, gene_class))
            elif gene_class == "protein_coding" and stable in member_proteins:
                mapped.append((feature_index, stable, gene_class))
        fresh_lnc_feature_count = len(mapped_lnc_all)
        feature_count_delta = fresh_lnc_feature_count - int(audited_lnc_feature_count)
        feature_count_tolerance = max(
            25, int(np.ceil(0.02 * int(audited_lnc_feature_count)))
        )
        if abs(feature_count_delta) > feature_count_tolerance:
            raise SingleCellPartitionBuildError(
                "Fresh direct-Ensembl H5/GENCODE lncRNA feature count differs materially "
                "from the read-only symbol-first audit: "
                f"audit={audited_lnc_feature_count}, fresh={fresh_lnc_feature_count}, "
                f"tolerance={feature_count_tolerance}"
            )
        if not mapped:
            raise SingleCellPartitionBuildError("No current candidate lncRNA or pathway protein maps to H5")
        gene_order = list(dict.fromkeys(item[1] for item in mapped))
        gene_class = {item[1]: item[2] for item in mapped}
        gene_code = {gene: index for index, gene in enumerate(gene_order)}
        feature_rows = np.asarray([item[0] for item in mapped], dtype=np.int64)
        target_rows = np.asarray([gene_code[item[1]] for item in mapped], dtype=np.int64)

        matrix = sparse.csc_matrix(
            (
                np.asarray(group["data"][:]),
                np.asarray(group["indices"][:]),
                np.asarray(group["indptr"][:]),
            ),
            shape=shape,
        )
        selected = matrix[feature_rows, :]
        collapse = sparse.csr_matrix(
            (np.ones(len(feature_rows), dtype=np.float32), (target_rows, np.arange(len(feature_rows)))),
            shape=(len(gene_order), len(feature_rows)),
        )
        collapsed = (collapse @ selected).tocsc()
        membership_matrix = sparse.csr_matrix(
            (
                np.ones(len(included_cells), dtype=np.float32),
                (included_cells, group_codes),
            ),
            shape=(shape[1], len(group_index)),
        )
        sums = (collapsed @ membership_matrix).toarray()
        detected_matrix = collapsed.copy()
        detected_matrix.data = (detected_matrix.data > 0).astype(np.int32)
        detected_matrix.eliminate_zeros()
        detected = (detected_matrix @ membership_matrix).toarray().astype(int)

    group_records = group_index.to_frame(index=False)
    number_genes = len(gene_order)
    number_groups = len(group_records)
    expression_values = sums.reshape(number_genes * number_groups)
    if source_tier == "source_normalized":
        expression_values = expression_values / np.tile(group_cell_counts, number_genes)
    elif np.nanmin(expression_values) < 0 or not np.allclose(
        expression_values, np.rint(expression_values), atol=1e-8
    ):
        raise SingleCellPartitionBuildError("Count-tier H5 aggregation is negative/non-integer")
    expression = pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "cancer_id": cancer,
            "donor_id": np.tile(group_records.donor_id.to_numpy(object), number_genes),
            "cell_type": np.tile(group_records.cell_type.to_numpy(object), number_genes),
            "cell_state": np.tile(group_records.cell_state.to_numpy(object), number_genes),
            "gene_id": np.repeat(np.asarray(gene_order, dtype=object), number_groups),
            "gene_type": np.repeat(
                np.asarray([gene_class[gene] for gene in gene_order], dtype=object), number_groups
            ),
            "gene_class": np.repeat(
                np.asarray([gene_class[gene] for gene in gene_order], dtype=object), number_groups
            ),
            "expression": expression_values,
            "n_cells": np.tile(group_cell_counts, number_genes),
            "n_detected_cells": detected.reshape(number_genes * number_groups),
            "source_tier": _SOURCE_TIER_TO_TRAINER[source_tier],
            "expression_source_tier": source_tier,
            "quality_flags": (
                "KNOWN_SOURCE_FEATURE_UNIVERSE_LIMITATION"
                if cancer in KNOWN_FEATURE_UNIVERSE_LIMITATIONS
                else "NONE"
            ),
            "feature_universe_status": (
                "LIMITED" if cancer in KNOWN_FEATURE_UNIVERSE_LIMITATIONS else "PASS"
            ),
        }
    )
    lnc_celltype = build_lnc_celltype_summary(expression)
    activity = compute_exact_pathway_activity_from_aggregates(
        aggregate_expression=sums,
        gene_order=gene_order,
        gene_class=gene_class,
        group_records=group_records,
        group_cell_counts=group_cell_counts,
        membership=membership,
        dataset_id=dataset_id,
        cancer_id=cancer,
        source_tier=_SOURCE_TIER_TO_TRAINER[source_tier],
        expression_source_tier=source_tier,
        quality_flags=(
            "KNOWN_SOURCE_FEATURE_UNIVERSE_LIMITATION"
            if cancer in KNOWN_FEATURE_UNIVERSE_LIMITATIONS
            else "NONE"
        ),
        feature_universe_status=(
            "LIMITED" if cancer in KNOWN_FEATURE_UNIVERSE_LIMITATIONS else "PASS"
        ),
    )
    association_reason: str | None = None
    try:
        association = compute_fresh_associations(
            expression,
            activity,
            local_candidates,
            min_observations=int(min_association_observations),
            chunk_size=int(association_chunk_size),
        )
    except SingleCellInputBuildError as exc:
        if "produced no finite" not in str(exc):
            raise
        association = _empty_association()
        association_reason = "NO_CONTEXT_WITH_MINIMUM_DONOR_REPLICATION"
    donor_count = int(group_records.donor_id.nunique())
    feature_limited = (
        cancer in KNOWN_FEATURE_UNIVERSE_LIMITATIONS
        or int(audited_lnc_feature_count) < LOW_FEATURE_UNIVERSE_THRESHOLD
    )
    formal_eligible = bool(
        not feature_limited
        and donor_count >= int(min_association_observations)
        and not association.empty
    )
    quality_flags: list[str] = []
    if feature_count_delta:
        quality_flags.append("FRESH_DIRECT_ID_FEATURE_COUNT_DIFFERS_FROM_READ_ONLY_AUDIT")
    if association_reason:
        quality_flags.append(association_reason)
    manifest = _manifest_row(
        dataset_id=dataset_id,
        cancer=cancer,
        source_tier=source_tier,
        measurement_scale=measurement_scale,
        lnc_feature_count=fresh_lnc_feature_count,
        donor_metadata_available=True,
        formal_eligible=formal_eligible,
        quality_flags=quality_flags,
    )
    availability = build_exact_pathway_availability(
        sorted(set(membership.pathway_id.astype(str))), activity, association
    )

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise SingleCellPartitionBuildError(f"Temporary partition already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        tables = {
            "dataset_manifest_row.parquet": manifest,
            "gene_expression_facts.parquet": expression,
            "lnc_celltype.parquet": lnc_celltype,
            "activity.parquet": activity,
            "single_cell_association.parquet": association,
            "exact_pathway_availability.parquet": availability,
        }
        for name, frame in tables.items():
            _atomic_parquet(frame, temporary / name)
        artifacts = {
            name: {
                "path": str((output / name).resolve()),
                "sha256": artifact_sha256(temporary / name),
                "rows": int(len(frame)),
            }
            for name, frame in tables.items()
        }
        status = {
            "format": PARTITION_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "cancer_id": cancer,
            "dataset_id": dataset_id,
            "status": (
                "SUCCESS_PARTITION_BUILT_FORMAL_ELIGIBLE"
                if formal_eligible
                else "SUCCESS_PARTITION_BUILT_NOT_FORMAL"
            ),
            "formal_eligible": formal_eligible,
            "blocking_reason": (
                None
                if formal_eligible
                else (
                    "LIMITED_LNCRNA_FEATURE_UNIVERSE"
                    if feature_limited
                    else association_reason or "INSUFFICIENT_DONOR_REPLICATION"
                )
            ),
            "source_tier": source_tier,
            "measurement_scale": measurement_scale,
            "feature_universe_status": str(manifest.feature_universe_status.iloc[0]),
            "lncrna_feature_universe_count": fresh_lnc_feature_count,
            "lncrna_feature_universe_count_semantics": (
                "FRESH_DIRECT_STABLE_ID_THEN_UNIQUE_GENCODE_SYMBOL_COUNT"
            ),
            "fresh_direct_id_lncrna_feature_count": fresh_lnc_feature_count,
            "audited_lncrna_feature_universe_count": int(audited_lnc_feature_count),
            "fresh_minus_audit_feature_count": feature_count_delta,
            "feature_count_tolerance": feature_count_tolerance,
            "feature_mapping_policy": "DIRECT_STABLE_ENSEMBL_THEN_UNIQUE_GENCODE_SYMBOL",
            "cells_in_h5": int(len(cell_meta)),
            "cells_used_after_metadata_and_doublet_gate": int(include.sum()),
            "donors": donor_count,
            "donor_celltype_state_groups": number_groups,
            "candidate_lncrnas_in_h5": int(sum(gene.startswith("LNC:") for gene in gene_order)),
            "membership_proteins_in_h5": int(sum(not gene.startswith("LNC:") for gene in gene_order)),
            "activity_pathways": int(activity.pathway_id.nunique()),
            "association_exact_pathways": int(association.pathway_id.nunique()) if len(association) else 0,
            "activity_missing_exact_pathway_ids": availability.loc[
                ~availability.activity_available, "pathway_id"
            ].astype(str).tolist(),
            "association_missing_exact_pathway_ids": availability.loc[
                ~availability.association_available, "pathway_id"
            ].astype(str).tolist(),
            "numeric_imputation_used": False,
            "h5_path": str(h5),
            "h5_sha256": artifact_sha256(h5),
            "metadata_path": str(metadata),
            "metadata_sha256": artifact_sha256(metadata),
            "annotation_path": str(annotation_path),
            "annotation_sha256": artifact_sha256(annotation_path),
            "annotation_provenance_sha256": artifact_sha256(annotation_provenance),
            "authority": authority,
            "artifacts": artifacts,
            "protein_only_exact_pathway_activity": True,
            "family_to_exact_broadcast": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "historical_sc_trajectory_used": False,
            "fresh_ucell_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
            "fresh_pseudotime_status": "PENDING_CELL_LEVEL_FRESH_RECOMPUTE",
            "single_cell_module_complete": False,
            "release_ready": False,
            "production_deployed": False,
        }
        status["contract_sha256"] = _canonical_sha256(status)
        _atomic_json(temporary / "PARTITION_STATUS.json", status)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return status


__all__ = [
    "ANNOTATION_FORMAT",
    "CURRENT_V32_CANDIDATE_ROWS",
    "CURRENT_V32_CANDIDATE_SHA256",
    "CURRENT_V32_EXACT_MEMBERSHIP_EDGES",
    "CURRENT_V32_EXACT_MEMBERSHIP_SHA256",
    "PARTITION_FORMAT",
    "SingleCellPartitionBuildError",
    "build_single_cell_partition",
    "build_exact_pathway_availability",
    "compute_exact_pathway_activity_from_aggregates",
    "load_current_v32_authority",
    "prepare_gencode_annotation",
]
