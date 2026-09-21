"""Fail-closed V3.2 single-cell standard-input construction.

The builder admits only gene-level expression facts.  It never reads a
single-cell pathway association, UCell/trajectory result, model prediction,
ranking, embedding, or checkpoint.  Exact-pathway activity and lncRNA to
pathway association targets are recomputed inside this module from the pinned
2,135-pathway V3.2 membership.

The private-head trainer and this builder share the same explicit 33-cancer
authority.  Source-tier, donor-metadata and feature-universe failures remain
per-cancer gates; expanding the authority never turns a limited data set into
formal evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .associations import benjamini_hochberg
from .input_lineage import artifact_sha256, validate_input_lineage


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BUILD_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_STANDARD_INPUTS_V1"
EXACT_PATHWAY_COUNT = 2_135

EXPECTED_CANCERS = frozenset(
    {
        "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
        "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
        "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
        "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
        "UVM",
    }
)

EXPRESSION_SOURCE_TIERS = frozenset(
    {"raw_counts", "source_normalized", "legacy_counts"}
)
KNOWN_FEATURE_UNIVERSE_LIMITATIONS = frozenset({"CESC", "UCS", "UVM"})

_SOURCE_TIER_ALIASES = {
    "raw": "raw_counts",
    "raw_count": "raw_counts",
    "raw_counts": "raw_counts",
    "primary_raw_count": "raw_counts",
    "source_normalised": "source_normalized",
    "source_normalized": "source_normalized",
    "normalized_by_source": "source_normalized",
    "legacy_count": "legacy_counts",
    "legacy_counts": "legacy_counts",
}
_TRAINER_SOURCE_TIER = {
    "raw_counts": "primary_raw_count",
    "source_normalized": "approved",
    "legacy_counts": "approved",
}
_COUNT_SCALES = frozenset({"count", "counts", "raw_count", "raw_counts"})
_NORMALIZED_SCALES = frozenset(
    {"normalized", "normalised", "log_normalized", "log_normalised", "source_normalized"}
)
_PASS_QUALITY = frozenset({"pass", "qualified", "approved", "formal"})

_FORBIDDEN_EXPRESSION_PATH_TOKENS = (
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
_FORBIDDEN_RESULT_COLUMN_TOKENS = (
    "prediction",
    "probability",
    "ranking",
    "checkpoint",
    "embedding",
    "fold_rank",
    "selection_frequency",
    "pair_support",
    "family_support",
    "oof_",
)
_FORBIDDEN_EXPRESSION_COLUMNS = frozenset(
    {
        "pathway_id",
        "pathway_family_id",
        "rho",
        "correlation",
        "association_effect",
        "p_value",
        "fdr",
        "q_value",
        "padj",
        "label",
        "sample_weight",
        "probability",
        "prediction",
        "rank",
        "ranking",
        "ucell",
        "mean_ucell",
        "pseudotime",
    }
)
_LNC_TYPES = frozenset(
    {
        "lncrna",
        "lnc_rna",
        "long_noncoding_rna",
        "long_non_coding_rna",
        "antisense",
        "processed_transcript",
        "sense_intronic",
        "sense_overlapping",
        "3prime_overlapping_ncrna",
        "macro_lncrna",
        "bidirectional_promoter_lncrna",
    }
)
_PROTEIN_TYPES = frozenset({"protein_coding", "protein_coding_gene", "coding"})


class SingleCellInputBuildError(RuntimeError):
    """Raised when a proposed standard input cannot prove its lineage."""


@dataclass(frozen=True)
class SingleCellInputBuildConfig:
    min_association_observations: int = 5
    low_feature_universe_threshold: int = 1_000
    association_chunk_size: int = 100_000

    def validate(self) -> None:
        if int(self.min_association_observations) < 3:
            raise ValueError("min_association_observations must be at least 3")
        if int(self.low_feature_universe_threshold) < 1:
            raise ValueError("low_feature_universe_threshold must be positive")
        if int(self.association_chunk_size) < 1:
            raise ValueError("association_chunk_size must be positive")


def _token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _canonical_json_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _explicit_bool(series: pd.Series, context: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    observed = set(lowered.loc[series.notna()].unique())
    if not observed.issubset({"true", "false", "1", "0", "yes", "no"}):
        raise SingleCellInputBuildError(
            f"{context} requires explicit booleans; observed={sorted(observed)}"
        )
    return lowered.isin({"true", "1", "yes"})


def _normalise_gene_id(value: Any) -> str:
    text = str(value).strip()
    upper = text.upper()
    for prefix in ("GENE:", "PROTEIN:"):
        if upper.startswith(prefix):
            text = text[len(prefix) :]
            break
    return re.sub(r"\.\d+$", "", text)


def _normalise_lnc_id(value: Any) -> str:
    text = str(value).strip()
    upper = text.upper()
    for prefix in ("LNC:", "LNCRNA:"):
        if upper.startswith(prefix):
            text = text[len(prefix) :]
            break
    return "LNC:" + re.sub(r"\.\d+$", "", text)


def _normalise_gene_class(value: Any) -> str:
    token = _token(value)
    if token in _LNC_TYPES or token.startswith("lncrna"):
        return "lncRNA"
    if token in _PROTEIN_TYPES:
        return "protein_coding"
    return "other"


def _read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir() or source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    suffixes = [suffix.lower() for suffix in source.suffixes]
    logical = suffixes[-2] if suffixes and suffixes[-1] in {".gz", ".bz2", ".xz"} else source.suffix.lower()
    if logical not in {".csv", ".tsv", ".txt"}:
        raise SingleCellInputBuildError(f"Unsupported tabular input: {source}")
    return pd.read_csv(source, sep="\t" if logical in {".tsv", ".txt"} else ",")


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


def _assert_no_result_columns(
    frame: pd.DataFrame,
    context: str,
    *,
    forbidden_exact: frozenset[str] = frozenset(),
) -> None:
    bad: list[str] = []
    for column in map(str, frame.columns):
        token = _token(column)
        if token in forbidden_exact or any(item in token for item in _FORBIDDEN_RESULT_COLUMN_TOKENS):
            bad.append(column)
    if bad:
        raise SingleCellInputBuildError(
            f"{context} contains historical/result-like columns: {sorted(set(bad))}"
        )


def _assert_expression_source_path(path: Path) -> None:
    # Inspect the artifact name and its nearest staging directories.  Scanning
    # the complete absolute path creates false positives because this source
    # repository is itself named ``ranked_subtypes``.
    nearby = [_token(part) for part in path.parts[-3:]]
    filename = _token(path.name)
    found = {
        item for item in _FORBIDDEN_EXPRESSION_PATH_TOKENS if item in filename
    }
    # These directory names identify derived single-cell result trees even
    # when the actual file has a neutral name such as ``part-0.parquet``.
    derived_tree_tokens = {"sc_trajectory", "pathway_association", "lnc_exact_pathway"}
    found.update(
        item
        for item in derived_tree_tokens
        if any(component == item or component.startswith(item + "_") for component in nearby)
    )
    found = sorted(found)
    if found:
        raise SingleCellInputBuildError(
            f"Expression input path looks like historical/model output: {path}; tokens={found}"
        )
    if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".joblib"}:
        raise SingleCellInputBuildError(f"Model/checkpoint input is forbidden: {path}")


def validate_dataset_manifest(
    frame: pd.DataFrame,
    *,
    low_feature_universe_threshold: int = 1_000,
) -> pd.DataFrame:
    """Return the authoritative, quality-annotated 33-cancer manifest."""

    _assert_no_result_columns(frame, "single-cell dataset manifest")
    required = {
        "dataset_id",
        "cancer_id",
        "expression_source_tier",
        "measurement_scale",
        "source_generation",
        "formal_eligible",
        "quality_status",
        "model_derived",
        "outcome_derived",
        "lncrna_feature_universe_count",
        "donor_metadata_available",
    }
    if missing := sorted(required - set(frame.columns)):
        raise SingleCellInputBuildError(f"33-cancer dataset manifest lacks columns: {missing}")

    result = frame[list(required)].copy()
    result["dataset_id"] = result.dataset_id.astype(str).str.strip()
    result["cancer_id"] = result.cancer_id.astype(str).str.upper().str.strip()
    result["expression_source_tier"] = result.expression_source_tier.map(
        lambda value: _SOURCE_TIER_ALIASES.get(_token(value), _token(value))
    )
    unknown_tiers = sorted(set(result.expression_source_tier) - EXPRESSION_SOURCE_TIERS)
    if unknown_tiers:
        raise SingleCellInputBuildError(
            f"Unsupported expression_source_tier values: {unknown_tiers}"
        )
    result["measurement_scale"] = result.measurement_scale.map(_token)
    bad_count_scale = result.expression_source_tier.isin({"raw_counts", "legacy_counts"}) & ~result.measurement_scale.isin(_COUNT_SCALES)
    bad_normalized_scale = result.expression_source_tier.eq("source_normalized") & ~result.measurement_scale.isin(_NORMALIZED_SCALES)
    if bad_count_scale.any() or bad_normalized_scale.any():
        bad = result.loc[bad_count_scale | bad_normalized_scale, [
            "dataset_id", "expression_source_tier", "measurement_scale"
        ]].to_dict("records")
        raise SingleCellInputBuildError(f"Source tier/measurement scale mismatch: {bad}")

    result["formal_eligible"] = _explicit_bool(result.formal_eligible, "formal_eligible")
    result["model_derived"] = _explicit_bool(result.model_derived, "model_derived")
    result["outcome_derived"] = _explicit_bool(result.outcome_derived, "outcome_derived")
    result["donor_metadata_available"] = _explicit_bool(
        result.donor_metadata_available, "donor_metadata_available"
    )
    if result.model_derived.any() or result.outcome_derived.any():
        bad = sorted(result.loc[result.model_derived | result.outcome_derived, "dataset_id"])
        raise SingleCellInputBuildError(
            f"Only non-model-derived, non-outcome-derived expression sources are allowed: {bad}"
        )

    generation = result.source_generation.astype(str).str.strip()
    invalid_generation = generation.eq("") | generation.str.contains(
        r"(?:checkpoint|prediction|ranking|probability|\bv?2[._-]|\bv?3[._-]?[01]\b)",
        case=False,
        regex=True,
    )
    if invalid_generation.any():
        bad = result.loc[invalid_generation, ["dataset_id", "source_generation"]].to_dict("records")
        raise SingleCellInputBuildError(f"Unusable source_generation declarations: {bad}")

    feature_count = pd.to_numeric(result.lncrna_feature_universe_count, errors="coerce")
    if feature_count.isna().any() or (feature_count < 1).any() or not np.equal(feature_count, np.floor(feature_count)).all():
        raise SingleCellInputBuildError(
            "lncrna_feature_universe_count must be a positive integer for every dataset"
        )
    result["lncrna_feature_universe_count"] = feature_count.astype(int)
    result["quality_status"] = result.quality_status.astype(str).map(_token)
    bad_quality_promotion = result.formal_eligible & ~result.quality_status.isin(_PASS_QUALITY)
    if bad_quality_promotion.any():
        names = sorted(result.loc[bad_quality_promotion, "dataset_id"])
        raise SingleCellInputBuildError(
            f"Non-passing datasets cannot be formal_eligible: {names}"
        )

    if result.dataset_id.eq("").any() or result.dataset_id.duplicated().any():
        raise SingleCellInputBuildError("dataset_id must be non-empty and unique")
    conflicts = result.groupby("dataset_id", observed=True).cancer_id.nunique()
    if (conflicts > 1).any():
        raise SingleCellInputBuildError("A dataset_id maps to multiple cancers")
    observed_cancers = set(result.cancer_id)
    if observed_cancers != EXPECTED_CANCERS:
        raise SingleCellInputBuildError(
            "Dataset manifest must cover exactly the 33 authoritative cancers; "
            f"missing={sorted(EXPECTED_CANCERS - observed_cancers)}, "
            f"unexpected={sorted(observed_cancers - EXPECTED_CANCERS)}"
        )

    flags: list[str] = []
    statuses: list[str] = []
    for row in result.itertuples(index=False):
        local: list[str] = []
        low = int(row.lncrna_feature_universe_count) < int(low_feature_universe_threshold)
        if low:
            local.append("LOW_LNCRNA_FEATURE_UNIVERSE")
        if str(row.cancer_id) in KNOWN_FEATURE_UNIVERSE_LIMITATIONS:
            local.append("KNOWN_SOURCE_FEATURE_UNIVERSE_LIMITATION")
        flags.append(";".join(local) if local else "NONE")
        statuses.append("LIMITED" if local else "PASS")
    result["feature_universe_status"] = statuses
    result["quality_flags"] = flags
    limited_promoted = result.formal_eligible & result.feature_universe_status.eq("LIMITED")
    if limited_promoted.any():
        names = sorted(result.loc[limited_promoted, "dataset_id"])
        raise SingleCellInputBuildError(
            f"Limited feature universes cannot be formal_eligible: {names}"
        )
    metadata_promoted = result.formal_eligible & ~result.donor_metadata_available
    if metadata_promoted.any():
        names = sorted(result.loc[metadata_promoted, "dataset_id"])
        raise SingleCellInputBuildError(
            f"Datasets without donor/cell-type metadata cannot be formal_eligible: {names}"
        )
    result["source_tier"] = result.expression_source_tier.map(_TRAINER_SOURCE_TIER)
    result["source_generation"] = generation
    result["manifest_authority"] = "V3.2_33_CANCER_SINGLE_CELL_SOURCE_MANIFEST"
    return result.sort_values(["cancer_id", "dataset_id"], kind="stable").reset_index(drop=True)


def validate_exact_membership(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalise the pinned 2,135 exact-pathway membership."""

    _assert_no_result_columns(frame, "exact-pathway membership")
    required = {"pathway_id", "gene_id"}
    if missing := sorted(required - set(frame.columns)):
        raise SingleCellInputBuildError(f"Exact membership lacks columns: {missing}")
    family_columns = sorted(column for column in frame.columns if "family" in _token(column))
    if family_columns:
        raise SingleCellInputBuildError(
            f"Exact membership cannot contain family mapping columns: {family_columns}"
        )
    result = frame[["pathway_id", "gene_id"]].copy()
    result["pathway_id"] = result.pathway_id.astype(str).str.strip()
    result["gene_id"] = result.gene_id.map(_normalise_gene_id)
    if result.isna().any().any() or result.eq("").any().any():
        raise SingleCellInputBuildError("Exact membership contains missing/empty IDs")
    if result.duplicated(["pathway_id", "gene_id"]).any():
        raise SingleCellInputBuildError("Exact membership contains duplicate pathway/gene edges")
    if result.pathway_id.nunique() != EXACT_PATHWAY_COUNT:
        raise SingleCellInputBuildError(
            f"Expected exactly {EXACT_PATHWAY_COUNT} exact pathways; "
            f"observed={result.pathway_id.nunique()}"
        )
    result["membership_weight"] = 1.0
    return result.sort_values(["pathway_id", "gene_id"], kind="stable").reset_index(drop=True)


def validate_membership_provenance(
    membership_path: str | Path,
    provenance_path: str | Path,
    *,
    membership_rows: int,
) -> dict[str, Any]:
    """Bind membership bytes to an explicit current-V3.2 exact manifest."""

    membership = Path(membership_path).resolve()
    provenance = Path(provenance_path).resolve()
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    observed_sha = artifact_sha256(membership)
    if str(payload.get("analysis_version")) != ANALYSIS_VERSION:
        raise SingleCellInputBuildError("Membership provenance is not the current V3.2 analysis")
    if _token(payload.get("pathway_target_level")) != "exact_pathway":
        raise SingleCellInputBuildError("Membership provenance is not exact-pathway level")
    required_booleans = {
        "membership_restricted_to_current_v32_exact_pathways": True,
        "pathway_family_broadcast": False,
        "static_annotation_only_no_predictions_or_rankings": True,
        "historical_checkpoints_used": False,
        "historical_predictions_used": False,
        "historical_rankings_used": False,
    }
    failures = {
        key: payload.get(key)
        for key, expected in required_booleans.items()
        if payload.get(key) is not expected
    }
    if failures:
        raise SingleCellInputBuildError(
            f"Membership provenance lacks fail-closed V3.2 attestations: {failures}"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise SingleCellInputBuildError("Membership provenance lacks an artifacts mapping")
    records = [
        value for key, value in artifacts.items()
        if "membership" in _token(key) and isinstance(value, Mapping)
    ]
    matches = [
        record for record in records
        if str(record.get("sha256", "")).lower() == observed_sha.lower()
    ]
    if len(matches) != 1:
        raise SingleCellInputBuildError(
            "Membership SHA256 is not uniquely pinned by its V3.2 provenance manifest"
        )
    declared_rows = (payload.get("rows") or {}).get("exact_pathway_membership")
    if int(declared_rows or -1) != int(membership_rows):
        raise SingleCellInputBuildError(
            f"Membership row count differs from provenance: declared={declared_rows}, "
            f"observed={membership_rows}"
        )
    declared_pathways = (payload.get("rows") or {}).get("pathway_metadata")
    if int(declared_pathways or -1) != EXACT_PATHWAY_COUNT:
        raise SingleCellInputBuildError(
            "Membership provenance does not attest the 2,135-pathway namespace"
        )
    return {
        "path": str(provenance),
        "sha256": artifact_sha256(provenance),
        "membership_sha256": observed_sha,
        "exact_pathways": EXACT_PATHWAY_COUNT,
        "status": "PASS",
    }


def validate_exact_candidates(
    frame: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the static current-V3.2 candidate task definition."""

    _assert_no_result_columns(frame, "V3.2 exact candidate universe")
    required = {"cancer_id", "lncrna_id", "pathway_id"}
    if missing := sorted(required - set(frame.columns)):
        raise SingleCellInputBuildError(f"Exact candidate universe lacks columns: {missing}")
    result = frame[["cancer_id", "lncrna_id", "pathway_id"]].dropna().copy()
    result["cancer_id"] = result.cancer_id.astype(str).str.upper().str.strip()
    result["lncrna_id"] = result.lncrna_id.map(_normalise_lnc_id)
    result["pathway_id"] = result.pathway_id.astype(str).str.strip()
    if result.duplicated(list(required)).any():
        raise SingleCellInputBuildError("Exact candidate keys are duplicated")
    observed_cancers = set(result.cancer_id)
    if observed_cancers != EXPECTED_CANCERS:
        raise SingleCellInputBuildError(
            "Exact candidates must cover the same 33-cancer authority; "
            f"missing={sorted(EXPECTED_CANCERS - observed_cancers)}, "
            f"unexpected={sorted(observed_cancers - EXPECTED_CANCERS)}"
        )
    exact_ids = set(membership.pathway_id)
    candidate_ids = set(result.pathway_id)
    if candidate_ids != exact_ids:
        raise SingleCellInputBuildError(
            "Candidate pathways do not exactly match pinned membership; "
            f"missing={len(exact_ids - candidate_ids)}, unexpected={len(candidate_ids - exact_ids)}"
        )
    return result.sort_values(
        ["cancer_id", "lncrna_id", "pathway_id"], kind="stable"
    ).reset_index(drop=True)


def validate_expression_facts(
    frame: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    require_all_manifest_datasets: bool = True,
) -> pd.DataFrame:
    """Canonicalise donor/cell-type pseudobulk gene-expression facts."""

    _assert_no_result_columns(
        frame,
        "single-cell expression facts",
        forbidden_exact=_FORBIDDEN_EXPRESSION_COLUMNS,
    )
    aliases = {
        "dataset_id": ("dataset_id",),
        "cancer_id": ("cancer_id",),
        "donor_id": ("donor_id", "patient_id", "subject_id"),
        "cell_type": ("cell_type", "cell_type_major", "cell_population"),
        "gene_id": ("gene_id",),
        "gene_type": ("gene_type", "gene_biotype", "feature_type"),
        "expression": ("expression", "expression_value", "pseudobulk_expression", "count"),
        "n_cells": ("n_cells",),
        "n_detected_cells": ("n_detected_cells", "n_detected"),
    }
    selected: dict[str, str] = {}
    lookup = {_token(column): str(column) for column in frame.columns}
    for output, candidates in aliases.items():
        source = next((lookup[_token(item)] for item in candidates if _token(item) in lookup), None)
        if source is None:
            raise SingleCellInputBuildError(
                f"Expression facts lack {output}; accepted aliases={list(candidates)}"
            )
        selected[output] = source
    state_column = next(
        (lookup[_token(item)] for item in ("cell_state", "state") if _token(item) in lookup),
        None,
    )
    result = pd.DataFrame({key: frame[column] for key, column in selected.items()})
    result["cell_state"] = (
        frame[state_column].fillna("").astype(str).str.strip()
        if state_column is not None
        else ""
    )
    for column in ("dataset_id", "donor_id", "cell_type"):
        result[column] = result[column].astype(str).str.strip()
    result["cancer_id"] = result.cancer_id.astype(str).str.upper().str.strip()
    result["gene_class"] = result.gene_type.map(_normalise_gene_class)
    result["gene_id"] = result.gene_id.map(_normalise_gene_id)
    lnc_mask = result.gene_class.eq("lncRNA")
    result.loc[lnc_mask, "gene_id"] = result.loc[lnc_mask, "gene_id"].map(_normalise_lnc_id)
    result["expression"] = pd.to_numeric(result.expression, errors="coerce")
    result["n_cells"] = pd.to_numeric(result.n_cells, errors="coerce")
    result["n_detected_cells"] = pd.to_numeric(result.n_detected_cells, errors="coerce")
    if result[["dataset_id", "cancer_id", "donor_id", "cell_type", "gene_id"]].eq("").any().any():
        raise SingleCellInputBuildError("Expression fact keys cannot be empty")
    numeric = result[["expression", "n_cells", "n_detected_cells"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise SingleCellInputBuildError("Expression facts contain non-finite numeric values")
    if (result.n_cells < 1).any() or (result.n_detected_cells < 0).any() or (result.n_detected_cells > result.n_cells).any():
        raise SingleCellInputBuildError("Invalid n_cells/n_detected_cells values")
    if not np.equal(result.n_cells, np.floor(result.n_cells)).all() or not np.equal(result.n_detected_cells, np.floor(result.n_detected_cells)).all():
        raise SingleCellInputBuildError("Cell counts must be integer-valued")
    result[["n_cells", "n_detected_cells"]] = result[["n_cells", "n_detected_cells"]].astype(int)

    manifest_columns = [
        "dataset_id", "cancer_id", "expression_source_tier", "measurement_scale",
        "source_tier", "source_generation", "lncrna_feature_universe_count",
        "quality_status", "quality_flags", "feature_universe_status",
    ]
    result = result.merge(
        manifest[manifest_columns],
        on=["dataset_id", "cancer_id"], how="left", validate="many_to_one",
    )
    if result.expression_source_tier.isna().any():
        bad = result.loc[result.expression_source_tier.isna(), ["dataset_id", "cancer_id"]].drop_duplicates().to_dict("records")
        raise SingleCellInputBuildError(f"Expression rows do not map to the 33-cancer manifest: {bad}")
    observed_datasets = set(result.dataset_id)
    expected_datasets = set(manifest.dataset_id)
    if require_all_manifest_datasets and observed_datasets != expected_datasets:
        raise SingleCellInputBuildError(
            "Expression facts must cover every declared dataset; "
            f"missing={sorted(expected_datasets - observed_datasets)}, "
            f"unexpected={sorted(observed_datasets - expected_datasets)}"
        )

    count_source = result.expression_source_tier.isin({"raw_counts", "legacy_counts"})
    if (result.loc[count_source, "expression"] < 0).any():
        raise SingleCellInputBuildError("Count-tier expression cannot be negative")
    count_values = result.loc[count_source, "expression"].to_numpy(float)
    if count_values.size and not np.allclose(count_values, np.rint(count_values), atol=1e-8):
        raise SingleCellInputBuildError("Count-tier expression must be integer-valued")

    key = ["dataset_id", "donor_id", "cell_type", "cell_state", "gene_id"]
    if result.duplicated(key).any():
        examples = result.loc[result.duplicated(key, keep=False), key].head(10).to_dict("records")
        raise SingleCellInputBuildError(
            f"Expression facts must already be unique donor pseudobulks: {examples}"
        )
    group_key = ["dataset_id", "donor_id", "cell_type", "cell_state"]
    inconsistent_cells = result.groupby(group_key, observed=True).n_cells.nunique()
    if (inconsistent_cells > 1).any():
        raise SingleCellInputBuildError("n_cells is inconsistent within a donor/cell pseudobulk")

    observed_lnc = result.loc[result.gene_class.eq("lncRNA")].groupby(
        "dataset_id", observed=True
    ).gene_id.nunique()
    declared_lnc = manifest.set_index("dataset_id").lncrna_feature_universe_count
    excess = observed_lnc.loc[observed_lnc > observed_lnc.index.to_series().map(declared_lnc)]
    if not excess.empty:
        raise SingleCellInputBuildError(
            f"Observed lncRNAs exceed declared feature universes: {excess.to_dict()}"
        )
    usable = result.loc[result.gene_class.isin({"lncRNA", "protein_coding"})]
    coverage = usable.groupby(["dataset_id", "gene_class"], observed=True).size().unstack(fill_value=0)
    coverage_scope = list(manifest.dataset_id) if require_all_manifest_datasets else sorted(observed_datasets)
    coverage = coverage.reindex(coverage_scope, fill_value=0)
    for required_class in ("lncRNA", "protein_coding"):
        if required_class not in coverage:
            coverage[required_class] = 0
    incomplete = coverage.loc[
        coverage["lncRNA"].eq(0) | coverage["protein_coding"].eq(0)
    ]
    if not incomplete.empty:
        raise SingleCellInputBuildError(
            "Every declared dataset requires lncRNA and protein-coding expression facts; "
            f"incomplete={incomplete.to_dict('index')}"
        )
    return result.sort_values(key, kind="stable").reset_index(drop=True)


def build_lnc_celltype_summary(expression: pd.DataFrame) -> pd.DataFrame:
    """Build lncRNA/cell-type facts directly from expression pseudobulks."""

    lnc = expression.loc[expression.gene_class.eq("lncRNA")].copy()
    if lnc.empty:
        raise SingleCellInputBuildError("No lncRNA expression facts are available")
    count_tier = lnc.expression_source_tier.isin({"raw_counts", "legacy_counts"})
    per_cell = np.where(count_tier, lnc.expression / lnc.n_cells, lnc.expression)
    lnc["analysis_expression"] = np.asarray(per_cell, dtype=float)
    lnc["log_expression"] = np.where(count_tier, np.log1p(lnc.analysis_expression), lnc.analysis_expression)
    lnc["weighted_log_expression"] = lnc.log_expression * lnc.n_cells
    keys = [
        "dataset_id", "cancer_id", "gene_id", "cell_type", "source_tier",
        "expression_source_tier", "quality_flags", "feature_universe_status",
    ]
    summary = lnc.groupby(keys, observed=True, as_index=False).agg(
        n_cells=("n_cells", "sum"),
        n_detected=("n_detected_cells", "sum"),
        weighted_log_expression=("weighted_log_expression", "sum"),
        mean_expression=("analysis_expression", "mean"),
        n_donors=("donor_id", "nunique"),
    )
    summary["detection_rate"] = summary.n_detected / summary.n_cells
    summary["mean_log_expression"] = summary.weighted_log_expression / summary.n_cells
    summary = summary.drop(columns="weighted_log_expression")

    tau: dict[tuple[str, str], float] = {}
    for (dataset, gene), group in summary.groupby(["dataset_id", "gene_id"], observed=True):
        values = group.mean_expression.to_numpy(float)
        maximum = float(np.nanmax(values)) if values.size else 0.0
        value = 0.0 if len(values) <= 1 or maximum <= 0 else float(np.sum(1.0 - values / maximum) / (len(values) - 1))
        tau[(str(dataset), str(gene))] = value
    summary["specificity_tau"] = [
        tau[(str(dataset), str(gene))]
        for dataset, gene in zip(summary.dataset_id, summary.gene_id, strict=True)
    ]
    summary = summary.rename(columns={"gene_id": "lncrna_id", "cell_type": "cell_type_major"})
    summary["generation"] = "V3.2_FRESH_FROM_EXPRESSION_FACTS"
    return summary.sort_values(
        ["cancer_id", "dataset_id", "lncrna_id", "cell_type_major"], kind="stable"
    ).reset_index(drop=True)


def compute_exact_pathway_activity(
    expression: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute rank-mean exact-pathway activity from protein genes only."""

    protein = expression.loc[expression.gene_class.eq("protein_coding")].copy()
    if protein.empty:
        raise SingleCellInputBuildError("No protein-coding expression facts are available")
    member_genes = set(membership.gene_id)
    lnc_members = set(expression.loc[expression.gene_class.eq("lncRNA"), "gene_id"]) & member_genes
    if lnc_members:
        raise SingleCellInputBuildError(
            f"lncRNA IDs leak into exact-pathway membership: {sorted(lnc_members)[:10]}"
        )
    protein["observation_id"] = (
        protein.dataset_id.astype(str)
        + "|" + protein.donor_id.astype(str)
        + "|" + protein.cell_type.astype(str)
        + "|" + protein.cell_state.astype(str)
    )
    protein["rank_percentile"] = protein.groupby(
        "observation_id", observed=True
    ).expression.rank(method="average", pct=True)
    joined = protein.merge(
        membership,
        on="gene_id", how="inner", validate="many_to_many",
    )
    if joined.empty:
        raise SingleCellInputBuildError(
            "No protein expression genes overlap the pinned exact membership"
        )
    joined["weighted_rank"] = joined.rank_percentile * joined.membership_weight
    keys = [
        "dataset_id", "cancer_id", "donor_id", "cell_type", "cell_state",
        "pathway_id", "source_tier", "expression_source_tier", "quality_flags",
        "feature_universe_status",
    ]
    activity = joined.groupby(keys, observed=True, as_index=False).agg(
        weighted_sum=("weighted_rank", "sum"),
        weight_sum=("membership_weight", "sum"),
        n_genes=("gene_id", "nunique"),
        n_cells=("n_cells", "max"),
    )
    activity["activity"] = activity.weighted_sum / activity.weight_sum
    activity = activity.drop(columns=["weighted_sum", "weight_sum"])
    activity["activity_method"] = "V3.2_PROTEIN_ONLY_WITHIN_PSEUDOBULK_RANK_MEAN_V1"
    activity["generation"] = "V3.2_FRESH_FROM_EXPRESSION_FACTS"
    if not np.isfinite(activity.activity.to_numpy(float)).all():
        raise SingleCellInputBuildError("Fresh exact-pathway activity is not finite")
    return activity.sort_values(
        ["cancer_id", "dataset_id", "donor_id", "cell_type", "cell_state", "pathway_id"],
        kind="stable",
    ).reset_index(drop=True)


def _rank_standardise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranked = pd.DataFrame(matrix).rank(axis=0, method="average").to_numpy(float)
    ranked -= ranked.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.square(ranked).sum(axis=0))
    return ranked, norms


def compute_fresh_associations(
    expression: pd.DataFrame,
    activity: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    min_observations: int = 5,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Recompute exact lncRNA/pathway Spearman targets per data context."""

    lnc = expression.loc[expression.gene_class.eq("lncRNA")].copy()
    if lnc.empty or activity.empty:
        raise SingleCellInputBuildError("Fresh associations require lncRNA and pathway activity")
    count_tier = lnc.expression_source_tier.isin({"raw_counts", "legacy_counts"})
    lnc["analysis_expression"] = np.where(
        count_tier, lnc.expression / lnc.n_cells, lnc.expression
    )
    context_keys = ["dataset_id", "cancer_id", "cell_type", "cell_state"]
    rows: list[pd.DataFrame] = []
    candidate_by_cancer = {
        cancer: part[["lncrna_id", "pathway_id"]]
        for cancer, part in candidates.groupby("cancer_id", observed=True)
    }
    for context, lnc_part in lnc.groupby(context_keys, observed=True, sort=True):
        dataset, cancer, cell_type, cell_state = map(str, context)
        activity_part = activity.loc[
            activity.dataset_id.eq(dataset)
            & activity.cancer_id.eq(cancer)
            & activity.cell_type.eq(cell_type)
            & activity.cell_state.astype(str).eq(cell_state)
        ]
        common_donors = sorted(set(lnc_part.donor_id.astype(str)) & set(activity_part.donor_id.astype(str)))
        if len(common_donors) < int(min_observations):
            continue
        left = lnc_part.pivot(index="donor_id", columns="gene_id", values="analysis_expression").reindex(common_donors).fillna(0.0)
        right = activity_part.pivot(index="donor_id", columns="pathway_id", values="activity").reindex(common_donors)
        right = right.dropna(axis=1, how="any")
        if left.empty or right.empty:
            continue
        pair = candidate_by_cancer[cancer]
        pair = pair.loc[pair.lncrna_id.isin(left.columns) & pair.pathway_id.isin(right.columns)]
        if pair.empty:
            continue
        left_index = {value: index for index, value in enumerate(left.columns)}
        right_index = {value: index for index, value in enumerate(right.columns)}
        left_rank, left_norm = _rank_standardise(left.to_numpy(float))
        right_rank, right_norm = _rank_standardise(right.to_numpy(float))
        li_all = pair.lncrna_id.map(left_index).to_numpy(int)
        ri_all = pair.pathway_id.map(right_index).to_numpy(int)
        local_rows: list[pd.DataFrame] = []
        for start in range(0, len(pair), int(chunk_size)):
            stop = min(start + int(chunk_size), len(pair))
            li = li_all[start:stop]
            ri = ri_all[start:stop]
            denominator = left_norm[li] * right_norm[ri]
            valid = denominator > 0
            if not valid.any():
                continue
            rho = np.full(len(li), np.nan, dtype=float)
            rho[valid] = np.einsum(
                "ij,ij->j", left_rank[:, li[valid]], right_rank[:, ri[valid]]
            ) / denominator[valid]
            rho = np.clip(rho, -1.0, 1.0)
            finite = np.isfinite(rho)
            if not finite.any():
                continue
            local = pair.iloc[start:stop].loc[finite].copy()
            local["rho"] = rho[finite]
            local_rows.append(local)
        if not local_rows:
            continue
        local = pd.concat(local_rows, ignore_index=True)
        n = len(common_donors)
        denominator = np.clip(1.0 - np.square(local.rho.to_numpy(float)), 1e-15, None)
        statistic = local.rho.to_numpy(float) * np.sqrt((n - 2) / denominator)
        local["p_value"] = 2.0 * stats.t.sf(np.abs(statistic), df=n - 2)
        local["fdr"] = benjamini_hochberg(local.p_value.to_numpy(float))
        local["dataset_id"] = dataset
        local["cancer_id"] = cancer
        local["cell_population"] = cell_type
        local["analysis_context"] = cell_state if cell_state else "overall"
        local["n_patients"] = n
        local["n_pseudobulk_groups"] = n
        local["fdr_family_size"] = len(local)
        local["source_tier"] = str(activity_part.source_tier.iloc[0])
        local["expression_source_tier"] = str(activity_part.expression_source_tier.iloc[0])
        local["association_method"] = "V3.2_FRESH_DONOR_PSEUDOBULK_SPEARMAN_EXACT_PATHWAY_V1"
        local["generation"] = "V3.2_FRESH_FROM_EXPRESSION_FACTS"
        rows.append(local)
    if not rows:
        raise SingleCellInputBuildError(
            "Fresh association recomputation produced no finite, sufficiently replicated targets"
        )
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(
        ["dataset_id", "cell_population", "analysis_context", "lncrna_id", "pathway_id"]
    ).any():
        raise SingleCellInputBuildError("Fresh association target keys are duplicated")
    return result.sort_values(
        ["cancer_id", "dataset_id", "cell_population", "analysis_context", "lncrna_id", "pathway_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_current_trainer_manifest(authoritative: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the trainer manifest and prove its boundary is exactly 33 cancers."""

    from .single_cell_training import FORMAL_SINGLE_CELL_CANCERS

    if set(FORMAL_SINGLE_CELL_CANCERS) != EXPECTED_CANCERS:
        raise SingleCellInputBuildError(
            "Single-cell trainer boundary is not the authoritative 33-cancer set"
        )
    result = authoritative.copy()
    result["requested_formal_eligible"] = result.formal_eligible.astype(bool)
    result["current_trainer_boundary_reason"] = pd.NA
    report = {
        "current_trainer_formal_boundary": sorted(FORMAL_SINGLE_CELL_CANCERS),
        "current_trainer_boundary_count": len(FORMAL_SINGLE_CELL_CANCERS),
        "authoritative_cancer_count": len(EXPECTED_CANCERS),
        "boundary_expansion_required": False,
        "currently_suppressed_requested_cancers": [],
        "boundary_policy": "V3.2_EXACT_33_CANCER_AUTHORITY",
        "authority_does_not_override_per_dataset_quality_gates": True,
    }
    return result, report


def _output_hashes(
    root: Path,
    names: Sequence[str],
    *,
    published_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    public = published_root or root
    return {
        name: {"path": str((public / name).resolve()), "sha256": artifact_sha256(root / name)}
        for name in names
    }


def build_v32_single_cell_inputs(
    *,
    dataset_manifest_path: str | Path,
    expression_facts_path: str | Path,
    exact_membership_path: str | Path,
    membership_provenance_path: str | Path,
    exact_candidates_path: str | Path,
    output_root: str | Path,
    build_run_id: str,
    config: SingleCellInputBuildConfig | None = None,
) -> dict[str, Any]:
    """Build a fresh, hash-bound single-cell handoff for private-head training."""

    settings = config or SingleCellInputBuildConfig()
    settings.validate()
    if not str(build_run_id).strip():
        raise ValueError("build_run_id is required")
    paths = {
        "dataset_manifest": Path(dataset_manifest_path).resolve(),
        "expression_facts": Path(expression_facts_path).resolve(),
        "exact_membership": Path(exact_membership_path).resolve(),
        "membership_provenance": Path(membership_provenance_path).resolve(),
        "exact_candidates": Path(exact_candidates_path).resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    _assert_expression_source_path(paths["expression_facts"])
    output = Path(output_root).resolve()
    if output.exists():
        raise SingleCellInputBuildError(f"Single-cell input builder refuses output reuse: {output}")

    lineage = validate_input_lineage(
        [
            {
                "artifact_id": "single_cell_33c_source_manifest",
                "path": paths["dataset_manifest"],
                "generation": "V3.2",
                "source_role": "split_manifest",
                "outcome_derived": False,
                "fold_fitted": False,
                "use_role": "split_control",
            },
            {
                "artifact_id": "single_cell_gene_expression_facts",
                "path": paths["expression_facts"],
                "generation": "V3.2_RESTANDARDIZED_FROM_SOURCE_EXPRESSION_FACTS",
                "source_role": "standardized_input",
                "outcome_derived": False,
                "fold_fitted": False,
                "use_role": "core_input",
            },
            {
                "artifact_id": "current_v32_exact_membership",
                "path": paths["exact_membership"],
                "generation": "V3.2",
                "source_role": "static_annotation",
                "outcome_derived": False,
                "fold_fitted": False,
                "use_role": "core_input",
                "source_target_level": "exact_pathway",
                "target_level": "exact_pathway",
                "family_to_exact_broadcast": False,
            },
            {
                "artifact_id": "current_v32_exact_candidates",
                "path": paths["exact_candidates"],
                "generation": "V3.2",
                "source_role": "standardized_input",
                "outcome_derived": False,
                "fold_fitted": False,
                "use_role": "aux_input",
                "source_target_level": "exact_pathway",
                "target_level": "exact_pathway",
                "family_to_exact_broadcast": False,
            },
        ]
    )
    manifest = validate_dataset_manifest(
        _read_table(paths["dataset_manifest"]),
        low_feature_universe_threshold=settings.low_feature_universe_threshold,
    )
    membership = validate_exact_membership(_read_table(paths["exact_membership"]))
    membership_provenance = validate_membership_provenance(
        paths["exact_membership"],
        paths["membership_provenance"],
        membership_rows=len(membership),
    )
    candidates = validate_exact_candidates(_read_table(paths["exact_candidates"]), membership)
    expression = validate_expression_facts(
        _read_table(paths["expression_facts"]), manifest,
        require_all_manifest_datasets=True,
    )
    lnc_celltype = build_lnc_celltype_summary(expression)
    activity = compute_exact_pathway_activity(expression, membership)
    association = compute_fresh_associations(
        expression,
        activity,
        candidates,
        min_observations=settings.min_association_observations,
        chunk_size=settings.association_chunk_size,
    )
    trainer_manifest, boundary_report = build_current_trainer_manifest(manifest)

    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise SingleCellInputBuildError(f"Temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        tables = {
            "dataset_manifest_33c.parquet": manifest,
            "dataset_manifest.current_trainer.parquet": trainer_manifest,
            "candidates.parquet": candidates,
            "exact_pathway_membership.parquet": membership,
            "lnc_celltype.parquet": lnc_celltype,
            "activity.parquet": activity,
            "single_cell_association.parquet": association,
        }
        for name, frame in tables.items():
            _atomic_parquet(frame, temporary / name)
        association_cancers = set(association.cancer_id.astype(str))
        activity_cancers = set(activity.cancer_id.astype(str))
        association_missing_cancers = sorted(EXPECTED_CANCERS - association_cancers)
        activity_missing_cancers = sorted(EXPECTED_CANCERS - activity_cancers)
        recompute = {
            "format": "CC_HHGT_V3_2_SINGLE_CELL_FRESH_RECOMPUTE_REQUIREMENTS_V1",
            "analysis_version": ANALYSIS_VERSION,
            "fresh_outputs_completed": {
                "lnc_celltype": True,
                "exact_pathway_activity": True,
                "lncrna_exact_pathway_association": True,
            },
            "still_requires_cell_level_fresh_recompute": {
                "ucell": {
                    "required": True,
                    "input": "cell-level expression facts plus current exact membership",
                    "historical_sc_trajectory_output_allowed": False,
                },
                "pseudotime": {
                    "required": True,
                    "input": "cell-level expression facts and source cell annotations",
                    "historical_sc_trajectory_output_allowed": False,
                },
            },
            "fresh_association_missing_cancers": association_missing_cancers,
            "fresh_activity_missing_cancers": activity_missing_cancers,
            "association_33c_coverage_complete": not association_missing_cancers,
            "activity_33c_coverage_complete": not activity_missing_cancers,
            "historical_pathway_association_used": False,
            "historical_activity_used": False,
            "historical_prediction_or_ranking_used": False,
        }
        _atomic_json(temporary / "RECOMPUTE_REQUIREMENTS.json", recompute)
        handoff = {
            "format": BUILD_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "build_run_id": str(build_run_id),
            "authoritative_manifest": "dataset_manifest_33c.parquet",
            "trainer_manifest": "dataset_manifest.current_trainer.parquet",
            "candidates": "candidates.parquet",
            "association": "single_cell_association.parquet",
            "lnc_celltype": "lnc_celltype.parquet",
            "activity": "activity.parquet",
            "training_generation_arguments": {
                "association_generation": "V3.2_FRESH_FROM_EXPRESSION_FACTS",
                "lnc_celltype_generation": "V3.2_FRESH_FROM_EXPRESSION_FACTS",
                "activity_generation": "V3.2_FRESH_FROM_EXPRESSION_FACTS",
            },
            "schemas_are_consumable_by_single_cell_training": True,
            "formal_33c_consumer_ready": (
                not boundary_report["boundary_expansion_required"]
                and not association_missing_cancers
                and not activity_missing_cancers
                and bool(manifest.formal_eligible.all())
            ),
            "boundary_report": boundary_report,
            "formal_33c_authority_does_not_override_quality_gates": True,
            "pathway_target_level": "exact_pathway",
            "exact_pathways": EXACT_PATHWAY_COUNT,
            "family_to_exact_broadcast": False,
        }
        _atomic_json(temporary / "TRAINING_HANDOFF.json", handoff)
        input_audit = {
            **lineage,
            "membership_provenance": membership_provenance,
            "expression_fact_contract": "DONOR_X_CELLTYPE_X_CELLSTATE_X_GENE",
            "expression_source_tiers": sorted(EXPRESSION_SOURCE_TIERS),
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "historical_single_cell_pathway_outputs_used": False,
        }
        input_audit["base_lineage_sha256"] = lineage["lineage_sha256"]
        input_audit["input_contract_sha256"] = _canonical_json_sha256(input_audit)
        _atomic_json(temporary / "INPUT_LINEAGE_AUDIT.json", input_audit)
        data_names = [*tables, "RECOMPUTE_REQUIREMENTS.json", "TRAINING_HANDOFF.json", "INPUT_LINEAGE_AUDIT.json"]
        artifacts = _output_hashes(temporary, data_names, published_root=output)
        build = {
            "format": BUILD_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "status": "SUCCESS_STANDARD_INPUTS_BUILT",
            "build_run_id": str(build_run_id),
            "counts": {
                "cancers": int(manifest.cancer_id.nunique()),
                "datasets": int(manifest.dataset_id.nunique()),
                "exact_pathways": int(membership.pathway_id.nunique()),
                "membership_edges": int(len(membership)),
                "candidate_rows": int(len(candidates)),
                "expression_fact_rows": int(len(expression)),
                "lnc_celltype_rows": int(len(lnc_celltype)),
                "activity_rows": int(len(activity)),
                "association_rows": int(len(association)),
            },
            "quality": {
                "known_feature_universe_limitations": sorted(KNOWN_FEATURE_UNIVERSE_LIMITATIONS),
                "known_limitations_recorded": all(
                    manifest.loc[manifest.cancer_id.isin(KNOWN_FEATURE_UNIVERSE_LIMITATIONS), "quality_flags"]
                    .str.contains("KNOWN_SOURCE_FEATURE_UNIVERSE_LIMITATION")
                ),
            },
            "lineage": {
                "input_lineage_sha256": input_audit["base_lineage_sha256"],
                "input_contract_sha256": input_audit["input_contract_sha256"],
                "membership_sha256": membership_provenance["membership_sha256"],
                "membership_provenance_sha256": membership_provenance["sha256"],
            },
            "historical_single_cell_pathway_outputs_used": False,
            "historical_predictions_used": False,
            "historical_rankings_used": False,
            "historical_checkpoints_used": False,
            "fresh_activity_recomputed": True,
            "fresh_association_recomputed": True,
            "family_to_exact_broadcast": False,
            "all_33_cancers_expression_covered": True,
            "association_33c_coverage_complete": not association_missing_cancers,
            "association_missing_cancers": association_missing_cancers,
            "activity_33c_coverage_complete": not activity_missing_cancers,
            "activity_missing_cancers": activity_missing_cancers,
            "formal_33c_training_ready": handoff["formal_33c_consumer_ready"],
            "release_ready": False,
            "production_deployed": False,
            "artifacts": artifacts,
            "config": asdict(settings),
        }
        build["build_contract_sha256"] = _canonical_json_sha256(build)
        _atomic_json(temporary / "BUILD_SUCCESS.json", build)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return json.loads((output / "BUILD_SUCCESS.json").read_text(encoding="utf-8"))


__all__ = [
    "ANALYSIS_VERSION",
    "BUILD_FORMAT",
    "EXACT_PATHWAY_COUNT",
    "EXPECTED_CANCERS",
    "EXPRESSION_SOURCE_TIERS",
    "KNOWN_FEATURE_UNIVERSE_LIMITATIONS",
    "SingleCellInputBuildConfig",
    "SingleCellInputBuildError",
    "build_current_trainer_manifest",
    "build_lnc_celltype_summary",
    "build_v32_single_cell_inputs",
    "compute_exact_pathway_activity",
    "compute_fresh_associations",
    "validate_dataset_manifest",
    "validate_exact_candidates",
    "validate_exact_membership",
    "validate_expression_facts",
    "validate_membership_provenance",
]
