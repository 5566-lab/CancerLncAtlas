"""Leakage-safe V3.2 single-cell private-head training.

This module deliberately rebuilds the single-cell capability from explicit
dataset/donor-level association and activity assets.  Historical single-cell
probabilities, rankings, family support and release tables are not accepted.
Only an explicitly approved, non-staging data set may produce a formal value;
everything else is materialised as ``NULL_WITH_REASON``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .input_lineage import artifact_sha256, audit_input_lineage


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MODULE_ID = "single_cell"
TARGET_LEVEL = "dataset_x_celltype_x_lncrna_x_exact_pathway"
TARGET_KEYS = ("dataset_id", "cell_type", "lncrna_id", "pathway_id")
N_FOLDS = 5
CORE_EXPORT_FORMAT = "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1"
PRIVATE_CHECKPOINT_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_PRIVATE_HEAD_V1"
PREDICTION_FORMAT = "CC_HHGT_V3_2_SINGLE_CELL_TYPED_PREDICTIONS_V1"

# V3.2 uses the complete TCGA 33-cancer authority.  Cancer membership is
# necessary, but never sufficient: source tier, donor/cell-type metadata,
# feature-universe quality and association availability are gated separately.
FORMAL_SINGLE_CELL_CANCERS = frozenset(
    {
        "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
        "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
        "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
        "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
        "UVM",
    }
)
_STAGING_TIERS = frozenset(
    {"staging", "nominal", "development", "dev", "pilot", "legacy_staging"}
)
_FORMAL_SOURCE_TIERS = frozenset(
    {"formal", "approved", "released", "primary", "primary_raw", "primary_raw_count", "raw"}
)
_FORMAL_QUALITY = frozenset({"pass", "qualified", "approved", "formal"})
_FORBIDDEN_PATH_TOKENS = (
    "probability", "prediction", "ranking", "ranked", "pair_support",
    "pair_evidence", "family_support", "release_table", "web_table", "checkpoint",
)
_FORBIDDEN_COLUMN_TOKENS = (
    "historical_probability", "legacy_probability", "oof_probability",
    "oof_prediction", "predicted_probability", "fold_rank",
    "selection_frequency", "pair_support", "family_support",
)
_FORBIDDEN_EXACT_COLUMNS = frozenset(
    {
        "probability", "prediction", "ranking", "rank", "rank_score",
        "pathway_family_id", "source_pathway_family_id",
        "family_to_exact_broadcast", "broadcast_from_family",
    }
)
_DOMAIN_FEATURES = (
    "lnc_detection_rate",
    "lnc_mean_log_expression",
    "lnc_specificity_tau",
    "pathway_activity",
    "pathway_mean_ucell",
    "pathway_mean_pseudotime",
    "log1p_n_cells",
    "lnc_feature_available",
    "pathway_feature_available",
)


class SingleCellTrainingError(RuntimeError):
    """Raised when single-cell source, split or isolation rules are violated."""


@dataclass(frozen=True)
class SingleCellTrainingConfig:
    seed: int = 20260825
    epochs: int = 40
    batch_size: int = 1024
    hidden_features: int = 64
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    max_train_rows: int = 200_000
    max_validation_rows: int = 100_000
    prediction_batch_size: int = 16_384

    def validate(self) -> None:
        positive = (
            self.epochs, self.batch_size, self.hidden_features, self.patience,
            self.max_train_rows, self.max_validation_rows,
            self.prediction_batch_size,
        )
        if any(int(value) < 1 for value in positive):
            raise ValueError("Single-cell integer controls must be positive")
        if not 0 <= float(self.dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0:
            raise ValueError("learning rate/weight decay are invalid")


@dataclass(frozen=True)
class EmbeddingLookup:
    values: np.ndarray
    index: Mapping[str, int]


@dataclass(frozen=True)
class FoldCoreEmbeddings:
    fold: int
    lncrna: EmbeddingLookup
    pathway: EmbeddingLookup
    checkpoint_sha256: str
    parameter_sha256: str
    artifact_hashes: Mapping[str, str]


def embedding_usability(lookup: EmbeddingLookup) -> dict[str, Any]:
    """Audit whether an exported node matrix can distinguish its node IDs."""

    values = np.asarray(lookup.values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise SingleCellTrainingError("Core embedding matrix is empty or malformed")
    varying = int(np.count_nonzero(np.ptp(values, axis=0) > 1.0e-12))
    unique = int(len(np.unique(values, axis=0)))
    # A one-node unit fixture is not evidence of collapse.  Formal lncRNA
    # exports contain thousands of rows and must vary across at least two IDs.
    usable = bool(values.shape[0] == 1 or (unique > 1 and varying > 0))
    return {
        "rows": int(values.shape[0]),
        "features": int(values.shape[1]),
        "unique_embedding_vectors": unique,
        "varying_feature_count": varying,
        "usable_for_node_discrimination": usable,
        "status": "USABLE" if usable else "CONSTANT_EMBEDDING_MASKED",
    }


def _normalise_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


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


def _read_table(path: str | Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    suffixes = [item.lower() for item in source.suffixes]
    logical = suffixes[-2] if suffixes and suffixes[-1] in {".gz", ".bz2", ".xz"} else source.suffix.lower()
    if source.is_dir() or logical == ".parquet":
        return pd.read_parquet(source, columns=list(columns) if columns else None)
    separator = "\t" if logical in {".tsv", ".txt"} else ","
    return pd.read_csv(source, sep=separator, usecols=list(columns) if columns else None)


def _first_column(
    frame: pd.DataFrame,
    candidates: Sequence[str],
    context: str,
    *,
    required: bool = True,
) -> str | None:
    lookup = {_normalise_token(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if _normalise_token(candidate) in lookup:
            return lookup[_normalise_token(candidate)]
    if required:
        raise SingleCellTrainingError(f"{context} lacks any of columns {list(candidates)}")
    return None


def _explicit_bool(series: pd.Series, context: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    observed = set(lowered.loc[series.notna()].unique())
    allowed = {"true", "false", "1", "0", "yes", "no"}
    if not observed.issubset(allowed):
        raise SingleCellTrainingError(f"{context} is not an explicit boolean: {sorted(observed)}")
    return lowered.isin({"true", "1", "yes"})


def _assert_nonpredictive(
    frame: pd.DataFrame,
    context: str,
    *,
    allow_family_annotation: bool = False,
) -> None:
    bad: list[str] = []
    for original in frame.columns:
        column = _normalise_token(original)
        if allow_family_annotation and column == "pathway_family_id":
            continue
        if column in _FORBIDDEN_EXACT_COLUMNS:
            bad.append(str(original))
        elif any(token in column for token in _FORBIDDEN_COLUMN_TOKENS):
            bad.append(str(original))
    if bad:
        raise SingleCellTrainingError(
            f"{context} contains historical/family result columns: {sorted(set(bad))}"
        )


def _assert_source_path(path: str | Path, *, core_manifest: bool = False) -> None:
    if core_manifest:
        return
    name = _normalise_token(Path(path).name)
    found = [token for token in _FORBIDDEN_PATH_TOKENS if token in name]
    if found:
        raise SingleCellTrainingError(
            f"Historical/result-like single-cell input path is forbidden: {path}; tokens={found}"
        )


def _node_id(value: Any, kind: str) -> str:
    text = str(value).strip()
    prefixes = ("LNC:", "LNCRNA:") if kind == "lncrna" else ("PATHWAY:", "PW:")
    for prefix in prefixes:
        if text.upper().startswith(prefix):
            return text[len(prefix) :]
    return text


def normalise_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the V3.2 cancer/lncRNA/exact-pathway candidate namespace."""

    _assert_nonpredictive(
        frame, "V3.2 single-cell candidates", allow_family_annotation=True
    )
    required = {"cancer_id", "lncrna_id", "pathway_id"}
    if missing := sorted(required - set(frame.columns)):
        raise SingleCellTrainingError(f"Candidate table lacks exact-pathway keys: {missing}")
    result = frame[["cancer_id", "lncrna_id", "pathway_id"]].dropna().astype(str)
    result["cancer_id"] = result.cancer_id.str.upper()
    result = result.drop_duplicates().sort_values(
        ["cancer_id", "lncrna_id", "pathway_id"], kind="stable"
    ).reset_index(drop=True)
    if result.empty:
        raise SingleCellTrainingError("V3.2 single-cell candidate universe is empty")
    return result


def normalise_dataset_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the explicit formal-coverage and blocking declaration.

    A staging/nominal data set cannot be marked formal.  The V3.2 cancer
    boundary is all 33 cancers; data-quality gates remain fail-closed per
    dataset.
    """

    _assert_nonpredictive(frame, "single-cell dataset manifest")
    dataset = _first_column(frame, ("dataset_id",), "dataset manifest")
    cancer = _first_column(frame, ("cancer_id",), "dataset manifest")
    eligible = _first_column(frame, ("formal_eligible", "eligible_for_formal"), "dataset manifest")
    tier = _first_column(frame, ("source_tier", "release_tier"), "dataset manifest")
    quality = _first_column(frame, ("quality_status", "status"), "dataset manifest")
    feature_status = _first_column(
        frame, ("feature_universe_status",), "dataset manifest", required=False
    )
    quality_flags = _first_column(
        frame, ("quality_flags",), "dataset manifest", required=False
    )
    donor_metadata = _first_column(
        frame,
        ("donor_metadata_available", "cell_metadata_available"),
        "dataset manifest",
        required=False,
    )
    result = pd.DataFrame(
        {
            "dataset_id": frame[dataset].astype(str).str.strip(),
            "cancer_id": frame[cancer].astype(str).str.upper().str.strip(),
            "formal_eligible": _explicit_bool(frame[eligible], "formal_eligible"),
            "source_tier": frame[tier].astype(str).map(_normalise_token),
            "quality_status": frame[quality].astype(str).map(_normalise_token),
            "feature_universe_status": (
                frame[feature_status].astype(str).map(_normalise_token)
                if feature_status is not None
                else "pass"
            ),
            "quality_flags": (
                frame[quality_flags].fillna("NONE").astype(str)
                if quality_flags is not None
                else "NONE"
            ),
            "donor_metadata_available": (
                _explicit_bool(frame[donor_metadata], "donor_metadata_available")
                if donor_metadata is not None
                else True
            ),
        }
    )
    if result.dataset_id.eq("").any():
        raise SingleCellTrainingError("dataset_id cannot be empty")
    conflicts = result.groupby("dataset_id", observed=True).cancer_id.nunique()
    if (conflicts > 1).any():
        raise SingleCellTrainingError("A single-cell dataset maps to multiple cancers")
    result = result.drop_duplicates("dataset_id", keep="last")
    bad_staging = result.formal_eligible & result.source_tier.isin(_STAGING_TIERS)
    if bad_staging.any():
        names = sorted(result.loc[bad_staging, "dataset_id"].astype(str))
        raise SingleCellTrainingError(
            f"Staging/nominal datasets cannot be promoted to formal coverage: {names}"
        )
    undeclared_formal = result.formal_eligible & ~result.source_tier.isin(_FORMAL_SOURCE_TIERS)
    if undeclared_formal.any():
        names = sorted(result.loc[undeclared_formal, "dataset_id"].astype(str))
        raise SingleCellTrainingError(
            f"Formal datasets require an approved non-staging source tier: {names}"
        )
    outside = result.formal_eligible & ~result.cancer_id.isin(FORMAL_SINGLE_CELL_CANCERS)
    if outside.any():
        names = sorted(result.loc[outside, "cancer_id"].unique())
        raise SingleCellTrainingError(
            f"Formal single-cell coverage is restricted to the V3.2 33-cancer authority: {names}"
        )
    limited_promoted = result.formal_eligible & ~result.feature_universe_status.eq("pass")
    if limited_promoted.any():
        names = sorted(result.loc[limited_promoted, "dataset_id"].astype(str))
        raise SingleCellTrainingError(
            "Limited feature universes cannot be promoted to formal coverage: "
            f"{names}"
        )
    metadata_promoted = result.formal_eligible & ~result.donor_metadata_available
    if metadata_promoted.any():
        names = sorted(result.loc[metadata_promoted, "dataset_id"].astype(str))
        raise SingleCellTrainingError(
            "Datasets without donor/cell-type metadata cannot be promoted to formal "
            f"coverage: {names}"
        )
    result["qualified"] = (
        result.formal_eligible
        & result.cancer_id.isin(FORMAL_SINGLE_CELL_CANCERS)
        & result.quality_status.isin(_FORMAL_QUALITY)
        & result.source_tier.isin(_FORMAL_SOURCE_TIERS)
        & result.feature_universe_status.eq("pass")
        & result.donor_metadata_available
    )
    return result.sort_values(["cancer_id", "dataset_id"], kind="stable").reset_index(drop=True)


def _dataset_lookup(manifest: pd.DataFrame) -> Mapping[str, str]:
    eligible = manifest.loc[manifest.qualified]
    counts = eligible.groupby("cancer_id", observed=True).dataset_id.nunique()
    unique_cancers = set(counts.loc[counts.eq(1)].index.astype(str))
    return {
        str(row.cancer_id): str(row.dataset_id)
        for row in eligible.itertuples(index=False)
        if str(row.cancer_id) in unique_cancers
    }


def normalise_single_cell_associations(
    frame: pd.DataFrame,
    dataset_manifest: pd.DataFrame,
) -> pd.DataFrame:
    """Normalise raw/non-predictive association targets without family mapping."""

    _assert_nonpredictive(frame, "single-cell association target")
    cancer = _first_column(frame, ("cancer_id",), "single-cell association")
    dataset = _first_column(frame, ("dataset_id",), "single-cell association", required=False)
    donor = _first_column(
        frame, ("donor_id", "patient_id", "subject_id"), "single-cell association", required=False
    )
    cell = _first_column(
        frame, ("cell_type", "cell_type_major", "cell_population", "compartment"),
        "single-cell association",
    )
    state = _first_column(
        frame, ("cell_state", "analysis_context", "state"),
        "single-cell association", required=False,
    )
    lnc = _first_column(frame, ("lncrna_id", "lncRNA_id"), "single-cell association")
    pathway = _first_column(frame, ("pathway_id",), "single-cell association")
    effect = _first_column(
        frame, ("rho", "effect", "association_effect", "correlation", "beta"),
        "single-cell association",
    )
    fdr = _first_column(
        frame, ("fdr", "q_value", "padj", "p_value"), "single-cell association"
    )
    tier = _first_column(
        frame, ("source_tier", "analysis_tier"), "single-cell association", required=False
    )
    n_obs = _first_column(
        frame, ("n_cells", "n_patients", "n_pseudobulk_groups", "n_observations"),
        "single-cell association", required=False,
    )

    cancer_values = frame[cancer].astype(str).str.upper().str.strip()
    if dataset is None:
        lookup = _dataset_lookup(dataset_manifest)
        dataset_values = cancer_values.map(lookup)
    else:
        dataset_values = frame[dataset].astype(str).str.strip()
    cell_major = frame[cell].astype(str).str.strip()
    state_values = (
        frame[state].fillna("").astype(str).str.strip() if state else pd.Series("", index=frame.index)
    )
    combined = np.where(
        state_values.eq("") | state_values.str.casefold().eq(cell_major.str.casefold()),
        cell_major,
        cell_major + "::" + state_values,
    )
    result = pd.DataFrame(
        {
            "dataset_id": dataset_values,
            "cancer_id": cancer_values,
            "donor_id": (
                frame[donor].fillna("").astype(str).str.strip()
                if donor else pd.Series("", index=frame.index)
            ),
            "cell_type": combined,
            "cell_type_major": cell_major,
            "cell_state": state_values.replace("", pd.NA),
            "lncrna_id": frame[lnc].astype(str).str.strip(),
            "pathway_id": frame[pathway].astype(str).str.strip(),
            "association_effect": pd.to_numeric(frame[effect], errors="coerce"),
            "association_fdr": pd.to_numeric(frame[fdr], errors="coerce"),
            "n_observations": (
                pd.to_numeric(frame[n_obs], errors="coerce") if n_obs else 1.0
            ),
            "association_source_tier": (
                frame[tier].astype(str).map(_normalise_token)
                if tier else "undeclared"
            ),
        }
    )
    result = result.dropna(subset=["dataset_id", "association_effect", "association_fdr"])
    result = result.loc[
        result.dataset_id.astype(str).ne("")
        & result.lncrna_id.ne("")
        & result.pathway_id.ne("")
        & result.cell_type.astype(str).ne("")
    ].copy()
    result["association_fdr"] = result.association_fdr.clip(0.0, 1.0)
    # A bounded, direction-agnostic replication-strength target.  Association
    # estimates and q-values are labels only; neither enters domain features.
    result["association_target"] = (
        result.association_effect.abs().clip(0.0, 1.0)
        * (1.0 - result.association_fdr)
    ).clip(0.0, 1.0)
    result = result.merge(
        dataset_manifest[
            ["dataset_id", "cancer_id", "qualified", "source_tier", "quality_status"]
        ],
        on=["dataset_id", "cancer_id"], how="left", validate="many_to_one",
    )
    result["qualified"] = result.qualified.fillna(False).astype(bool)
    result["formal_row"] = (
        result.qualified
        & result.association_source_tier.isin(_FORMAL_SOURCE_TIERS)
        & result.cancer_id.isin(FORMAL_SINGLE_CELL_CANCERS)
    )
    return result.reset_index(drop=True)


def build_blocked_folds(frame: pd.DataFrame, *, seed: int = 20260825) -> pd.DataFrame:
    """Assign deterministic five-fold blocks with donor-first isolation."""

    required = {"dataset_id", "cancer_id"}
    if missing := sorted(required - set(frame.columns)):
        raise SingleCellTrainingError(f"Blocked-fold input lacks {missing}")
    donor = frame["donor_id"].fillna("").astype(str) if "donor_id" in frame else pd.Series("", index=frame.index)
    dataset = frame.dataset_id.astype(str)
    cancer = frame.cancer_id.astype(str)
    block = np.where(
        donor.str.strip().ne(""),
        cancer + "|donor:" + donor,
        cancer + "|dataset:" + dataset,
    )
    blocks = pd.DataFrame({"block_id": pd.Series(block).drop_duplicates()})
    blocks["_order"] = blocks.block_id.map(
        lambda value: hashlib.sha256(f"{int(seed)}|{value}".encode("utf-8")).hexdigest()
    )
    blocks = blocks.sort_values(["_order", "block_id"], kind="stable").reset_index(drop=True)
    blocks["single_cell_fold_id"] = np.arange(len(blocks), dtype=int) % N_FOLDS
    mapping = dict(zip(blocks.block_id, blocks.single_cell_fold_id, strict=True))
    result = frame.copy()
    result["block_id"] = block
    result["single_cell_fold_id"] = result.block_id.map(mapping).astype(int)
    validate_fold_isolation(result)
    return result


def validate_fold_isolation(frame: pd.DataFrame) -> None:
    required = {"block_id", "single_cell_fold_id"}
    if missing := sorted(required - set(frame.columns)):
        raise SingleCellTrainingError(f"Fold-isolation table lacks {missing}")
    counts = frame.groupby("block_id", observed=True).single_cell_fold_id.nunique()
    if (counts > 1).any():
        raise SingleCellTrainingError("A donor/dataset block appears in multiple folds")
    observed = set(pd.to_numeric(frame.single_cell_fold_id, errors="raise").astype(int))
    if not observed.issubset(set(range(N_FOLDS))):
        raise SingleCellTrainingError("Single-cell fold IDs must be in 0..4")


def split_block_ids(frame: pd.DataFrame, fold: int) -> dict[str, frozenset[str]]:
    validation = (int(fold) + 1) % N_FOLDS
    mapping = frame[["block_id", "single_cell_fold_id"]].drop_duplicates()
    result = {
        "test": frozenset(mapping.loc[mapping.single_cell_fold_id.eq(fold), "block_id"]),
        "validation": frozenset(mapping.loc[mapping.single_cell_fold_id.eq(validation), "block_id"]),
        "train": frozenset(
            mapping.loc[~mapping.single_cell_fold_id.isin([fold, validation]), "block_id"]
        ),
    }
    if (result["train"] & result["validation"]) or (result["train"] & result["test"]) or (result["validation"] & result["test"]):
        raise SingleCellTrainingError("Donor/dataset split leakage detected")
    return result


def exact_candidate_join(
    associations: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict associations by exact ``pathway_id``; never broadcast families."""

    _assert_nonpredictive(associations, "single-cell exact join source")
    candidates = normalise_candidates(candidates)
    keys = ["cancer_id", "lncrna_id", "pathway_id"]
    missing = sorted(set(keys) - set(associations.columns))
    if missing:
        raise SingleCellTrainingError(f"Single-cell associations lack exact join keys: {missing}")
    result = associations.merge(
        candidates.assign(_v32_exact_candidate=True),
        on=keys, how="inner", validate="many_to_one",
    )
    result["exact_pathway_join"] = True
    return result


def _normalise_lnc_celltype(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_nonpredictive(frame, "lncRNA-cell-type activity")
    dataset = _first_column(frame, ("dataset_id",), "lncRNA-cell-type activity")
    cancer = _first_column(frame, ("cancer_id",), "lncRNA-cell-type activity")
    lnc = _first_column(frame, ("lncrna_id", "lncRNA_id"), "lncRNA-cell-type activity")
    cell = _first_column(
        frame, ("cell_type", "cell_type_major", "cell_population"), "lncRNA-cell-type activity"
    )
    result = pd.DataFrame(
        {
            "dataset_id": frame[dataset].astype(str),
            "cancer_id": frame[cancer].astype(str).str.upper(),
            "lncrna_id": frame[lnc].astype(str),
            "cell_type": frame[cell].astype(str).str.strip(),
            "cell_key": frame[cell].astype(str).map(_normalise_token),
        }
    )
    for output, aliases in {
        "lnc_detection_rate": ("detection_rate",),
        "lnc_mean_log_expression": ("mean_log_expression", "mean_expression"),
        "lnc_specificity_tau": ("specificity_tau",),
        "lnc_n_cells": ("n_cells",),
    }.items():
        source = _first_column(frame, aliases, "lncRNA-cell-type activity", required=False)
        result[output] = pd.to_numeric(frame[source], errors="coerce") if source else np.nan
    return result.groupby(
        ["dataset_id", "cancer_id", "lncrna_id", "cell_type", "cell_key"],
        observed=True, as_index=False,
    ).agg(
        lnc_detection_rate=("lnc_detection_rate", "mean"),
        lnc_mean_log_expression=("lnc_mean_log_expression", "mean"),
        lnc_specificity_tau=("lnc_specificity_tau", "mean"),
        lnc_n_cells=("lnc_n_cells", "max"),
    )


def _normalise_activity(
    frame: pd.DataFrame,
    *,
    kind: str,
    dataset_manifest: pd.DataFrame,
) -> pd.DataFrame:
    _assert_nonpredictive(frame, f"single-cell {kind} activity")
    cancer = _first_column(frame, ("cancer_id",), f"{kind} activity")
    pathway = _first_column(frame, ("pathway_id",), f"{kind} activity")
    dataset = _first_column(frame, ("dataset_id",), f"{kind} activity", required=False)
    donor = _first_column(frame, ("donor_id", "patient_id"), f"{kind} activity", required=False)
    cell = _first_column(
        frame, ("cell_type", "cell_type_major", "compartment", "immune_subtype", "contrast"),
        f"{kind} activity", required=False,
    )
    value_aliases = {
        "ucell": ("mean_ucell", "ucell", "activity"),
        "pseudotime": ("effect", "standardized_effect", "mean_pseudotime"),
        "activity": ("activity", "effect", "score", "value"),
    }
    value = _first_column(frame, value_aliases[kind], f"{kind} activity")
    n_cells = _first_column(frame, ("n_cells", "n_observations"), f"{kind} activity", required=False)
    pseudotime = _first_column(
        frame, ("mean_pseudotime", "pseudotime"), f"{kind} activity", required=False
    )
    tier = _first_column(
        frame, ("source_tier", "release_tier"), f"{kind} activity", required=False
    )
    source_path = _first_column(
        frame, ("_source_path", "source_path"), f"{kind} activity", required=False
    )
    cancer_values = frame[cancer].astype(str).str.upper()
    donor_values = frame[donor].fillna("").astype(str) if donor else pd.Series("", index=frame.index)
    if dataset:
        dataset_values = frame[dataset].astype(str)
    elif donor:
        dataset_values = donor_values.str.split(":", n=1).str[0]
    else:
        dataset_values = cancer_values.map(_dataset_lookup(dataset_manifest))
    activity_tier = (
        frame[tier].fillna("").astype(str).map(_normalise_token)
        if tier else pd.Series("undeclared", index=frame.index)
    )
    if source_path:
        staging_path = frame[source_path].fillna("").astype(str).str.contains(
            r"(?:^|[/\\_.-])staging(?:$|[/\\_.-])", case=False, regex=True
        )
        activity_tier = activity_tier.mask(staging_path, "staging")
    cell_values = frame[cell].fillna("").astype(str).str.strip() if cell else pd.Series("", index=frame.index)
    result = pd.DataFrame(
        {
            "dataset_id": dataset_values,
            "cancer_id": cancer_values,
            "donor_id": donor_values,
            "cell_type": cell_values,
            "cell_key": cell_values.map(_normalise_token),
            "pathway_id": frame[pathway].astype(str),
            "activity_kind": kind,
            "activity_source_tier": activity_tier,
            "activity_value": pd.to_numeric(frame[value], errors="coerce"),
            "mean_pseudotime": (
                pd.to_numeric(frame[pseudotime], errors="coerce") if pseudotime else np.nan
            ),
            "n_cells": pd.to_numeric(frame[n_cells], errors="coerce") if n_cells else np.nan,
        }
    )
    return result.dropna(subset=["dataset_id", "pathway_id", "activity_value"]).reset_index(drop=True)


def build_domain_features(
    associations: pd.DataFrame,
    lnc_celltype: pd.DataFrame,
    activity: pd.DataFrame,
) -> np.ndarray:
    """Join lnc/cell and exact-pathway activity without target-derived values."""

    base = associations.reset_index(drop=True).copy()
    base["_row_id"] = np.arange(len(base), dtype=int)
    base["cell_key"] = base.cell_type_major.astype(str).map(_normalise_token)
    lnc = lnc_celltype.copy()
    merged = base.merge(
        lnc,
        on=["dataset_id", "cancer_id", "lncrna_id", "cell_key"],
        how="left", validate="many_to_one",
    )
    if activity.empty:
        activity_wide = pd.DataFrame(columns=["dataset_id", "pathway_id"])
    else:
        local = activity.copy()
        local["weighted"] = local.activity_value * local.n_cells.fillna(1.0).clip(lower=1.0)
        grouped = local.groupby(
            ["dataset_id", "pathway_id", "activity_kind"], observed=True, as_index=False
        ).agg(
            activity_value=("activity_value", "mean"),
            mean_pseudotime=("mean_pseudotime", "mean"),
            n_cells=("n_cells", "sum"),
        )
        values = grouped.pivot_table(
            index=["dataset_id", "pathway_id"], columns="activity_kind",
            values="activity_value", aggfunc="mean",
        ).reset_index()
        values.columns.name = None
        auxiliary = grouped.groupby(
            ["dataset_id", "pathway_id"], observed=True, as_index=False
        ).agg(mean_pseudotime=("mean_pseudotime", "mean"), pathway_n_cells=("n_cells", "sum"))
        activity_wide = values.merge(auxiliary, on=["dataset_id", "pathway_id"], how="outer")
    merged = merged.merge(
        activity_wide, on=["dataset_id", "pathway_id"], how="left", validate="many_to_one"
    )
    def numeric_column(name: str) -> pd.Series:
        if name not in merged:
            return pd.Series(np.nan, index=merged.index, dtype=float)
        return pd.to_numeric(merged[name], errors="coerce")

    merged["pathway_activity"] = numeric_column("activity")
    merged["pathway_mean_ucell"] = numeric_column("ucell")
    merged["pathway_mean_pseudotime"] = numeric_column("mean_pseudotime")
    merged["log1p_n_cells"] = np.log1p(
        pd.concat(
            [
                numeric_column("lnc_n_cells"),
                numeric_column("pathway_n_cells"),
            ], axis=1,
        ).max(axis=1, skipna=True).fillna(0.0).clip(lower=0.0)
    )
    merged["lnc_feature_available"] = merged[
        ["lnc_detection_rate", "lnc_mean_log_expression", "lnc_specificity_tau"]
    ].notna().any(axis=1).astype(float)
    merged["pathway_feature_available"] = merged[
        ["pathway_activity", "pathway_mean_ucell", "pathway_mean_pseudotime"]
    ].notna().any(axis=1).astype(float)
    for column in _DOMAIN_FEATURES:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
    merged = merged.sort_values("_row_id", kind="stable")
    return merged[list(_DOMAIN_FEATURES)].to_numpy(np.float32)


def _embedding_lookup(frame: pd.DataFrame, kind: str) -> EmbeddingLookup:
    if "node_id" not in frame:
        raise SingleCellTrainingError(f"{kind} core embedding lacks node_id")
    features = sorted(column for column in frame if str(column).startswith("core_feature_"))
    if not features:
        raise SingleCellTrainingError(f"{kind} core embedding lacks core_feature_* columns")
    ids = frame.node_id.map(lambda value: _node_id(value, kind)).astype(str)
    if ids.duplicated().any():
        raise SingleCellTrainingError(f"{kind} core embedding IDs are duplicated")
    values = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise SingleCellTrainingError(f"{kind} core embeddings are not finite")
    return EmbeddingLookup(values=values, index={value: index for index, value in enumerate(ids)})


def _resolve_manifest_path(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() and path.is_file():
        return path.resolve()
    for parent in (manifest_path.parent, *manifest_path.parents):
        candidate = parent / path
        if candidate.is_file():
            return candidate.resolve()
    raise SingleCellTrainingError(f"Core embedding export cannot be resolved: {value}")


def validate_core_manifest(path: str | Path) -> tuple[dict[str, Any], str, str]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("export_format") != CORE_EXPORT_FORMAT:
        raise SingleCellTrainingError("Core embeddings are not a registered V3.2 export")
    if not str(payload.get("analysis_version", "")).startswith("CancerLncAtlas_V3.2"):
        raise SingleCellTrainingError("Single-cell parent core is not V3.2")
    if payload.get("all_embeddings_from_newly_trained_v32_core") is not True:
        raise SingleCellTrainingError("Core export lacks new-V3.2 training attestation")
    if payload.get("historical_checkpoint_loaded") is not False:
        raise SingleCellTrainingError("Core export loaded a historical checkpoint")
    if payload.get("historical_prediction_loaded") is not False:
        raise SingleCellTrainingError("Core export loaded historical predictions")
    folds = payload.get("folds")
    if not isinstance(folds, Mapping) or set(map(str, folds)) != set(map(str, range(N_FOLDS))):
        raise SingleCellTrainingError("Core export requires exactly folds 0..4")
    parameter_hashes: list[str] = []
    for fold in range(N_FOLDS):
        item = folds[str(fold)]
        if int(item.get("patient_fold", -1)) != fold:
            raise SingleCellTrainingError(f"Core fold ID drift at fold {fold}")
        if item.get("old_checkpoint_loaded") is not False or item.get("trained_from_scratch") is not True:
            raise SingleCellTrainingError(f"Core fold {fold} is not newly trained")
        value = str(item.get("core_parameter_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise SingleCellTrainingError(f"Core fold {fold} lacks a parameter hash")
        parameter_hashes.append(value)
    return payload, artifact_sha256(source), _canonical_sha256(parameter_hashes)


def load_fold_core_embeddings(
    manifest_path: str | Path,
    manifest: Mapping[str, Any],
    fold: int,
    *,
    lncrna_node_type: str = "lncRNA",
    pathway_node_type: str = "pathway",
) -> FoldCoreEmbeddings:
    source = Path(manifest_path).resolve()
    item = manifest["folds"][str(fold)]
    exports = item.get("exports", {})
    if lncrna_node_type not in exports or pathway_node_type not in exports:
        raise SingleCellTrainingError(f"Core fold {fold} lacks lncRNA/pathway exports")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for kind, node_type in (("lncrna", lncrna_node_type), ("pathway", pathway_node_type)):
        declaration = exports[node_type]
        resolved = _resolve_manifest_path(source, str(declaration["path"]))
        observed = artifact_sha256(resolved)
        if observed != str(declaration.get("sha256", "")):
            raise SingleCellTrainingError(f"Core fold {fold} {kind} export hash mismatch")
        paths[kind] = resolved
        hashes[str(resolved)] = observed
    lnc = _embedding_lookup(pd.read_parquet(paths["lncrna"]), "lncrna")
    pathway = _embedding_lookup(pd.read_parquet(paths["pathway"]), "pathway")
    if lnc.values.shape[1] != pathway.values.shape[1]:
        raise SingleCellTrainingError("lncRNA/pathway core widths differ")
    return FoldCoreEmbeddings(
        fold=fold, lncrna=lnc, pathway=pathway,
        checkpoint_sha256=str(item["checkpoint_sha256"]),
        parameter_sha256=str(item["core_parameter_sha256"]),
        artifact_hashes=hashes,
    )


def candidate_core(
    frame: pd.DataFrame,
    core: FoldCoreEmbeddings,
) -> tuple[np.ndarray, np.ndarray]:
    lpos, ppos = _candidate_core_positions(frame, core)
    available = (lpos >= 0) & (ppos >= 0)
    width = core.lncrna.values.shape[1]
    values = np.zeros((len(frame), width * 2), dtype=np.float32)
    # The current formal V3.2 export has a constant lncRNA vector in every
    # fold.  Treating that as an informative lncRNA feature would be false.
    # Keep its branch at zero and let fresh single-cell lncRNA measurements
    # provide lncRNA identity/context; pathway embeddings remain usable.
    if embedding_usability(core.lncrna)["usable_for_node_discrimination"]:
        values[available, :width] = core.lncrna.values[lpos[available]]
    values[available, width:] = core.pathway.values[ppos[available]]
    return values, available


def _candidate_core_positions(
    frame: pd.DataFrame,
    core: FoldCoreEmbeddings,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve core rows without allocating the dense embedding matrix."""

    lpos = np.fromiter(
        (
            core.lncrna.index.get(_node_id(value, "lncrna"), -1)
            for value in frame.lncrna_id
        ),
        dtype=np.int64,
        count=len(frame),
    )
    ppos = np.fromiter(
        (
            core.pathway.index.get(_node_id(value, "pathway"), -1)
            for value in frame.pathway_id
        ),
        dtype=np.int64,
        count=len(frame),
    )
    return lpos, ppos


def candidate_core_availability(
    frame: pd.DataFrame,
    core: FoldCoreEmbeddings,
) -> np.ndarray:
    """Return the exact core-coverage mask without dense feature materialisation."""

    lpos, ppos = _candidate_core_positions(frame, core)
    return (lpos >= 0) & (ppos >= 0)


def _sample_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    if len(labels) <= int(maximum):
        return np.arange(len(labels), dtype=int)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(len(labels), size=int(maximum), replace=False))


def fit_private_head(
    train_core: np.ndarray,
    train_domain: np.ndarray,
    train_labels: np.ndarray,
    validation_core: np.ndarray,
    validation_domain: np.ndarray,
    validation_labels: np.ndarray,
    *,
    fold: int,
    config: SingleCellTrainingConfig,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Train a fresh detached private head; no checkpoint loader exists here."""

    import torch
    from torch.nn import functional as F

    from .integrated_model import build_private_auxiliary_head

    if len(train_labels) < 2 or np.nanstd(train_labels) < 1e-8:
        raise SingleCellTrainingError(f"Single-cell fold {fold} lacks target variation")
    mean = train_domain.mean(axis=0).astype(np.float32)
    scale = train_domain.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train_scaled = (train_domain - mean) / scale
    validation_scaled = (validation_domain - mean) / scale if len(validation_domain) else validation_domain
    head_seed = int(config.seed + fold * 101)
    head, initialization = build_private_auxiliary_head(
        MODULE_ID,
        core_features=int(train_core.shape[1]),
        domain_features=int(train_domain.shape[1]),
        hidden_features=int(config.hidden_features),
        dropout=float(config.dropout),
        seed=head_seed,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay)
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(head_seed)
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(int(config.epochs)):
        head.train()
        order = torch.randperm(len(train_labels), generator=generator).numpy()
        losses: list[float] = []
        for start in range(0, len(order), int(config.batch_size)):
            index = order[start : start + int(config.batch_size)]
            core_tensor = torch.tensor(train_core[index], dtype=torch.float32, requires_grad=True)
            domain_tensor = torch.tensor(train_scaled[index], dtype=torch.float32)
            target_tensor = torch.tensor(train_labels[index], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = head(core_tensor, domain_tensor).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, target_tensor)
            loss.backward()
            if core_tensor.grad is not None:
                raise SingleCellTrainingError("Single-cell private head leaked gradients into core")
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.no_grad():
            if len(validation_labels):
                logits = head(
                    torch.tensor(validation_core, dtype=torch.float32),
                    torch.tensor(validation_scaled, dtype=torch.float32),
                ).squeeze(-1)
                validation_loss = float(
                    F.binary_cross_entropy_with_logits(
                        logits, torch.tensor(validation_labels, dtype=torch.float32)
                    )
                )
            else:
                validation_loss = float(np.mean(losses))
        history.append(
            {"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {name: value.detach().clone() for name, value in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience):
            break
    if best_state is None:
        raise SingleCellTrainingError("Single-cell private head produced no checkpoint state")
    head.load_state_dict(best_state, strict=True)
    metadata = initialization.as_dict()
    metadata.update(
        {
            "single_cell_fold": int(fold),
            "train_rows": int(len(train_labels)),
            "validation_rows": int(len(validation_labels)),
            "best_validation_loss": float(best_loss),
            "core_embedding_detached": True,
            "core_parameters_frozen": True,
        }
    )
    return head, metadata, mean, scale, history


def _predict(
    head: Any,
    core: np.ndarray,
    domain: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    import torch

    result = np.empty(len(core), dtype=np.float32)
    head.eval()
    with torch.no_grad():
        for start in range(0, len(result), int(batch_size)):
            stop = min(start + int(batch_size), len(result))
            logits = head(
                torch.tensor(core[start:stop], dtype=torch.float32),
                torch.tensor((domain[start:stop] - mean) / scale, dtype=torch.float32),
            ).squeeze(-1)
            result[start:stop] = torch.sigmoid(logits).cpu().numpy()
    return result


def coverage_null_scaffold(
    candidates: pd.DataFrame,
    dataset_manifest: pd.DataFrame,
    associations: pd.DataFrame,
) -> pd.DataFrame:
    """Materialise candidate-level null rows for cancers without formal data."""

    candidates = normalise_candidates(candidates)
    qualified = dataset_manifest.loc[dataset_manifest.qualified]
    formal_rows = associations.loc[associations.formal_row] if "formal_row" in associations else associations.iloc[0:0]
    rows: list[pd.DataFrame] = []
    for cancer, local in candidates.groupby("cancer_id", observed=True, sort=False):
        if cancer not in FORMAL_SINGLE_CELL_CANCERS:
            reason = "CANCER_NOT_IN_V32_33_CANCER_AUTHORITY"
        elif not qualified.cancer_id.eq(cancer).any():
            declared = dataset_manifest.loc[dataset_manifest.cancer_id.eq(cancer)]
            if declared.empty:
                reason = "NO_DECLARED_SINGLE_CELL_DATASET"
            elif (~declared.donor_metadata_available).any():
                reason = "DONOR_CELLTYPE_METADATA_UNAVAILABLE"
            elif (~declared.feature_universe_status.eq("pass")).any():
                reason = "LIMITED_LNCRNA_FEATURE_UNIVERSE"
            elif declared.source_tier.isin(_STAGING_TIERS).any():
                reason = "STAGING_OR_NOMINAL_SINGLE_CELL_SOURCE_NOT_FORMAL"
            else:
                reason = "NO_FORMAL_QUALIFIED_SINGLE_CELL_DATASET"
        elif not formal_rows.cancer_id.eq(cancer).any():
            reason = "NO_NON_STAGING_DONOR_OR_DATASET_ASSOCIATION"
        else:
            continue
        part = local.copy()
        part["dataset_id"] = "UNAVAILABLE::" + str(cancer)
        part["cell_type"] = "UNAVAILABLE"
        part["cell_type_major"] = pd.NA
        part["cell_state"] = pd.NA
        part["single_cell_replication_probability"] = np.nan
        part["single_cell_available"] = False
        part["single_cell_unavailable_reason"] = reason
        rows.append(part)
    if not rows:
        return pd.DataFrame(
            columns=[*TARGET_KEYS, "cancer_id", "single_cell_replication_probability",
                     "single_cell_available", "single_cell_unavailable_reason"]
        )
    return pd.concat(rows, ignore_index=True)


def _figure_manifest(
    roots: Sequence[str | Path],
    dataset_manifest: pd.DataFrame,
    candidate_cancers: Sequence[str],
    output: Path,
) -> dict[str, Any]:
    allowed = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    entries: list[dict[str, Any]] = []
    dataset_ids = sorted(dataset_manifest.dataset_id.astype(str), key=len, reverse=True)
    cancers = sorted(set(dataset_manifest.cancer_id.astype(str)) | set(FORMAL_SINGLE_CELL_CANCERS))
    for root_value in roots:
        root = Path(root_value).resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        for path in sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in allowed):
            if path.is_symlink():
                raise SingleCellTrainingError(f"Figure symlink is forbidden: {path}")
            text = path.as_posix()
            dataset = next((value for value in dataset_ids if value in text), None)
            cancer = next((value for value in cancers if re.search(rf"(?:^|[/_.-]){re.escape(value)}(?:$|[/_.-])", text, re.I)), None)
            entries.append(
                {
                    "status": "AVAILABLE",
                    "dataset_id": dataset,
                    "cancer_id": cancer,
                    "path": str(path),
                    "sha256": artifact_sha256(path),
                    "extension": path.suffix.lower(),
                }
            )
    covered_datasets = {entry["dataset_id"] for entry in entries if entry["dataset_id"]}
    for row in dataset_manifest.itertuples(index=False):
        if row.dataset_id not in covered_datasets:
            entries.append(
                {
                    "status": "NULL_WITH_REASON",
                    "dataset_id": row.dataset_id,
                    "cancer_id": row.cancer_id,
                    "path": None,
                    "sha256": None,
                    "reason": "NO_REGISTERED_SINGLE_CELL_FIGURE",
                }
            )
    represented_cancers = {entry["cancer_id"] for entry in entries if entry["cancer_id"]}
    for cancer in sorted(set(map(str, candidate_cancers)) - represented_cancers):
        entries.append(
            {
                "status": "NULL_WITH_REASON",
                "dataset_id": None,
                "cancer_id": cancer,
                "path": None,
                "sha256": None,
                "reason": "NO_FORMAL_QUALIFIED_SINGLE_CELL_DATASET_OR_FIGURE",
            }
        )
    payload = {
        "format": "CC_HHGT_V3_2_SINGLE_CELL_FIGURE_MANIFEST_V1",
        "analysis_version": ANALYSIS_VERSION,
        "entries": entries,
        "figure_files_are_outputs_not_training_features": True,
    }
    _atomic_json(output, payload)
    return payload


def _source_audit(
    *,
    candidates: Path,
    dataset_manifest: Path,
    association: Path,
    association_generation: str,
    lnc_celltype: Path,
    lnc_celltype_generation: str,
    activities: Sequence[tuple[str, Path, str]],
) -> dict[str, Any]:
    specs: list[dict[str, Any]] = [
        {
            "artifact_id": "v32_candidates", "path": candidates,
            "generation": "V3.2", "source_role": "standardized_input",
            "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input",
            "source_target_level": "exact_pathway", "target_level": "exact_pathway",
            "family_to_exact_broadcast": False,
        },
        {
            "artifact_id": "single_cell_dataset_manifest", "path": dataset_manifest,
            "generation": "V3.2", "source_role": "split_manifest",
            "outcome_derived": False, "fold_fitted": False, "use_role": "split_control",
        },
        {
            "artifact_id": "single_cell_association_target", "path": association,
            # The association table is an outcome-derived label materialised by
            # the V3.2 single-cell pipeline.  Calling it raw data made a legacy
            # association table eligible for use as a target, because raw assay
            # targets are intentionally allowed by the generic lineage policy.
            # Single-cell labels therefore use the stricter training_label role:
            # the shared lineage gate will reject every non-V3.2 generation.
            "generation": association_generation, "source_role": "training_label",
            "outcome_derived": True, "fold_fitted": False, "use_role": "training_target",
            "source_target_level": "exact_pathway", "target_level": "exact_pathway",
            "family_to_exact_broadcast": False,
        },
        {
            "artifact_id": "single_cell_lnc_celltype", "path": lnc_celltype,
            "generation": lnc_celltype_generation, "source_role": "raw_data",
            "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input",
        },
    ]
    for kind, path, generation in activities:
        specs.append(
            {
                "artifact_id": f"single_cell_{kind}", "path": path,
                "generation": generation, "source_role": "raw_data",
                "outcome_derived": False, "fold_fitted": False, "use_role": "aux_input",
                "source_target_level": "exact_pathway", "target_level": "exact_pathway",
                "family_to_exact_broadcast": False,
            }
        )
    audit = audit_input_lineage(specs)
    if audit["status"] != "PASS":
        failures = {
            row["artifact_id"]: row["reasons"]
            for row in audit["artifacts"] if row["status"] != "PASS"
        }
        raise SingleCellTrainingError(f"Single-cell source lineage rejected: {failures}")
    return audit


def run_single_cell_training(
    *,
    candidates_path: str | Path,
    dataset_manifest_path: str | Path,
    association_path: str | Path,
    lnc_celltype_path: str | Path,
    core_embedding_manifest_path: str | Path,
    output_root: str | Path,
    training_run_id: str,
    activity_path: str | Path | None = None,
    pseudotime_path: str | Path | None = None,
    ucell_path: str | Path | None = None,
    figure_roots: Sequence[str | Path] = (),
    association_generation: str = "DECLARED_SINGLE_CELL_ASSOCIATION_SOURCE",
    lnc_celltype_generation: str = "DECLARED_SINGLE_CELL_NONPREDICTIVE_SOURCE",
    activity_generation: str = "DECLARED_SINGLE_CELL_NONPREDICTIVE_SOURCE",
    pseudotime_generation: str = "DECLARED_SINGLE_CELL_NONPREDICTIVE_SOURCE",
    ucell_generation: str = "DECLARED_SINGLE_CELL_NONPREDICTIVE_SOURCE",
    config: SingleCellTrainingConfig | None = None,
    lncrna_node_type: str = "lncRNA",
    pathway_node_type: str = "pathway",
) -> dict[str, Any]:
    """Run new V3.2 single-cell training and materialise typed outputs."""

    settings = config or SingleCellTrainingConfig()
    settings.validate()
    if not str(training_run_id).strip():
        raise ValueError("training_run_id is required")
    paths = {
        "candidates": Path(candidates_path).resolve(),
        "dataset_manifest": Path(dataset_manifest_path).resolve(),
        "association": Path(association_path).resolve(),
        "lnc_celltype": Path(lnc_celltype_path).resolve(),
        "core_manifest": Path(core_embedding_manifest_path).resolve(),
    }
    optional = {
        "activity": Path(activity_path).resolve() if activity_path else None,
        "pseudotime": Path(pseudotime_path).resolve() if pseudotime_path else None,
        "ucell": Path(ucell_path).resolve() if ucell_path else None,
    }
    for name, path in paths.items():
        _assert_source_path(path, core_manifest=name == "core_manifest")
    for path in optional.values():
        if path is not None:
            _assert_source_path(path)
    output = Path(output_root).resolve()
    if output.exists():
        raise SingleCellTrainingError(f"Single-cell training refuses output reuse: {output}")

    generations = {
        "activity": str(activity_generation),
        "pseudotime": str(pseudotime_generation),
        "ucell": str(ucell_generation),
    }
    declared_generations = {
        "association": str(association_generation),
        "lnc_celltype": str(lnc_celltype_generation),
        **generations,
    }
    if any(not value.strip() for value in declared_generations.values()):
        raise ValueError("Every supplied single-cell source requires a generation declaration")
    activity_specs = [
        (name, path, generations[name]) for name, path in optional.items() if path is not None
    ]
    audit = _source_audit(
        candidates=paths["candidates"], dataset_manifest=paths["dataset_manifest"],
        association=paths["association"], association_generation=association_generation,
        lnc_celltype=paths["lnc_celltype"],
        lnc_celltype_generation=lnc_celltype_generation,
        activities=activity_specs,
    )
    core_manifest, core_manifest_sha, core_parameter_composite = validate_core_manifest(
        paths["core_manifest"]
    )
    candidates = normalise_candidates(_read_table(paths["candidates"]))
    dataset_manifest = normalise_dataset_manifest(_read_table(paths["dataset_manifest"]))
    association_all = normalise_single_cell_associations(
        _read_table(paths["association"]), dataset_manifest
    )
    association = exact_candidate_join(
        association_all.loc[association_all.formal_row].copy(), candidates
    )
    lnc_celltype = _normalise_lnc_celltype(_read_table(paths["lnc_celltype"]))
    lnc_celltype = lnc_celltype.merge(
        dataset_manifest[["dataset_id", "cancer_id", "qualified"]],
        on=["dataset_id", "cancer_id"], how="left", validate="many_to_one",
    )
    lnc_celltype = lnc_celltype.loc[lnc_celltype.qualified.fillna(False)].drop(columns="qualified")
    activity_parts: list[pd.DataFrame] = []
    for kind, path, _generation in activity_specs:
        activity_parts.append(
            _normalise_activity(_read_table(path), kind=kind, dataset_manifest=dataset_manifest)
        )
    activity = (
        pd.concat(activity_parts, ignore_index=True)
        if activity_parts else pd.DataFrame(
            columns=["dataset_id", "cancer_id", "donor_id", "cell_type", "cell_key",
                     "pathway_id", "activity_kind", "activity_source_tier",
                     "activity_value", "mean_pseudotime", "n_cells"]
        )
    )
    activity = activity.merge(
        dataset_manifest[["dataset_id", "cancer_id", "qualified", "source_tier"]].rename(
            columns={"source_tier": "dataset_source_tier"}
        ),
        on=["dataset_id", "cancer_id"], how="left", validate="many_to_one",
    )
    activity["effective_source_tier"] = activity.activity_source_tier.where(
        activity.activity_source_tier.ne("undeclared"), activity.dataset_source_tier
    )
    activity = activity.loc[
        activity.qualified.fillna(False)
        & activity.effective_source_tier.isin(_FORMAL_SOURCE_TIERS)
    ].drop(columns=["qualified", "dataset_source_tier"])

    output.mkdir(parents=True)
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir()
    folds = build_blocked_folds(association, seed=settings.seed) if len(association) else association.assign(
        block_id=pd.Series(dtype=str), single_cell_fold_id=pd.Series(dtype=int)
    )
    domain = build_domain_features(folds, lnc_celltype, activity) if len(folds) else np.empty((0, len(_DOMAIN_FEATURES)), np.float32)
    prediction_sum = np.zeros(len(folds), dtype=np.float64)
    prediction_count = np.zeros(len(folds), dtype=np.int16)
    checkpoint_rows: list[dict[str, Any]] = []
    core_hashes_before: dict[str, str] = {}
    core_embedding_usability: list[dict[str, Any]] = []

    enough_blocks = folds.block_id.nunique() >= N_FOLDS if len(folds) else False
    for fold in range(N_FOLDS):
        core = load_fold_core_embeddings(
            paths["core_manifest"], core_manifest, fold,
            lncrna_node_type=lncrna_node_type, pathway_node_type=pathway_node_type,
        )
        core_hashes_before.update(core.artifact_hashes)
        lnc_core_usability = embedding_usability(core.lncrna)
        pathway_core_usability = embedding_usability(core.pathway)
        if not pathway_core_usability["usable_for_node_discrimination"]:
            raise SingleCellTrainingError(
                f"Core fold {fold} pathway embeddings cannot distinguish exact pathways"
            )
        core_embedding_usability.append(
            {
                "single_cell_fold": fold,
                "lncrna": lnc_core_usability,
                "pathway": pathway_core_usability,
                "lncrna_fallback": (
                    None
                    if lnc_core_usability["usable_for_node_discrimination"]
                    else "MASK_CONSTANT_CORE_AND_USE_FRESH_SINGLE_CELL_LNCRNA_FEATURES"
                ),
            }
        )
        # Resolve coverage over the full formal table, but materialise dense
        # core features only for selected rows.  The source has millions of
        # associations, so full expansion would allocate several gigabytes
        # for a lncRNA branch that is explicitly masked as constant.
        core_available = (
            candidate_core_availability(folds, core)
            if len(folds)
            else np.empty(0, bool)
        )
        split = split_block_ids(folds, fold) if len(folds) else {
            "train": frozenset(), "validation": frozenset(), "test": frozenset()
        }
        train_mask = folds.block_id.isin(split["train"]).to_numpy() & core_available if len(folds) else np.empty(0, bool)
        validation_mask = folds.block_id.isin(split["validation"]).to_numpy() & core_available if len(folds) else np.empty(0, bool)
        test_mask = folds.block_id.isin(split["test"]).to_numpy() & core_available if len(folds) else np.empty(0, bool)
        train_index = np.flatnonzero(train_mask)
        validation_index = np.flatnonzero(validation_mask)
        train_index = train_index[_sample_indices(folds.association_target.to_numpy(float)[train_index], settings.max_train_rows, settings.seed + fold)]
        validation_index = validation_index[_sample_indices(folds.association_target.to_numpy(float)[validation_index], settings.max_validation_rows, settings.seed + 10_000 + fold)]
        reason = None
        if not enough_blocks:
            reason = "INSUFFICIENT_INDEPENDENT_DONOR_OR_DATASET_BLOCKS"
        elif len(train_index) < 2 or np.nanstd(folds.association_target.to_numpy(float)[train_index]) < 1e-8:
            reason = "FOLD_TARGET_NOT_TRAINABLE"
        if reason:
            checkpoint_rows.append(
                {
                    "single_cell_fold": fold, "status": "NULL_WITH_REASON", "reason": reason,
                    "source_checkpoint_sha256": None, "fresh_random_initialization": False,
                    "initial_parameter_sha256": None,
                    "initialization_status": "NOT_INSTANTIATED_WITHOUT_TRAINABLE_DATA",
                    "core_checkpoint_sha256": core.checkpoint_sha256,
                    "core_parameter_sha256": core.parameter_sha256,
                    "core_lncrna_embedding_usability": lnc_core_usability,
                    "core_pathway_embedding_usability": pathway_core_usability,
                    "train_rows": int(len(train_index)), "validation_rows": int(len(validation_index)),
                }
            )
            continue
        train_core, train_core_available = candidate_core(
            folds.iloc[train_index], core
        )
        validation_core, validation_core_available = candidate_core(
            folds.iloc[validation_index], core
        )
        if not train_core_available.all() or not validation_core_available.all():
            raise SingleCellTrainingError(
                f"Core fold {fold} selected rows changed availability during materialisation"
            )
        head, metadata, mean, scale, history = fit_private_head(
            train_core, domain[train_index],
            folds.association_target.to_numpy(np.float32)[train_index],
            validation_core, domain[validation_index],
            folds.association_target.to_numpy(np.float32)[validation_index],
            fold=fold, config=settings,
        )
        import torch

        checkpoint_path = checkpoint_root / f"single_cell_fold_{fold}.pt"
        torch.save(
            {
                "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
                "analysis_version": ANALYSIS_VERSION,
                "module_id": MODULE_ID,
                "single_cell_fold": fold,
                "initialization": metadata,
                "model_state": head.state_dict(),
                "domain_features": list(_DOMAIN_FEATURES),
                "domain_mean": mean,
                "domain_scale": scale,
                "history": history,
                "core_checkpoint_sha256": core.checkpoint_sha256,
                "core_parameter_sha256": core.parameter_sha256,
                "old_checkpoint_loaded": False,
                "old_predictions_used": False,
            },
            checkpoint_path,
        )
        checkpoint_rows.append(
            {
                "single_cell_fold": fold, "status": "SUCCESS", "reason": None,
                "path": str(checkpoint_path), "sha256": artifact_sha256(checkpoint_path),
                "source_checkpoint_sha256": None, "fresh_random_initialization": True,
                "initial_parameter_sha256": metadata["initial_parameter_sha256"],
                "core_checkpoint_sha256": core.checkpoint_sha256,
                "core_parameter_sha256": core.parameter_sha256,
                "core_lncrna_embedding_usability": lnc_core_usability,
                "core_pathway_embedding_usability": pathway_core_usability,
                "train_rows": int(len(train_index)), "validation_rows": int(len(validation_index)),
            }
        )
        if test_mask.any():
            indices = np.flatnonzero(test_mask)
            for start in range(0, len(indices), int(settings.prediction_batch_size)):
                batch_indices = indices[
                    start : start + int(settings.prediction_batch_size)
                ]
                batch_core, batch_available = candidate_core(
                    folds.iloc[batch_indices], core
                )
                if not batch_available.all():
                    raise SingleCellTrainingError(
                        f"Core fold {fold} test rows changed availability during streaming"
                    )
                values = _predict(
                    head,
                    batch_core,
                    domain[batch_indices],
                    mean,
                    scale,
                    settings.prediction_batch_size,
                )
                prediction_sum[batch_indices] += values
                prediction_count[batch_indices] += 1

    for path_text, expected in core_hashes_before.items():
        if artifact_sha256(path_text) != expected:
            raise SingleCellTrainingError(f"Frozen V3.2 core export changed: {path_text}")
    if artifact_sha256(paths["core_manifest"]) != core_manifest_sha:
        raise SingleCellTrainingError("Frozen V3.2 core manifest changed during training")

    if len(folds):
        row_prediction = np.divide(
            prediction_sum, prediction_count,
            out=np.full(len(folds), np.nan, dtype=float), where=prediction_count > 0,
        )
        predicted_rows = folds[[
            "dataset_id", "cancer_id", "cell_type", "cell_type_major", "cell_state",
            "lncrna_id", "pathway_id",
        ]].copy()
        predicted_rows["single_cell_replication_probability"] = row_prediction
        grouped = predicted_rows.groupby(
            ["dataset_id", "cancer_id", "cell_type", "cell_type_major", "cell_state",
             "lncrna_id", "pathway_id"],
            observed=True, dropna=False, as_index=False,
        ).agg(single_cell_replication_probability=("single_cell_replication_probability", "mean"))
        grouped["single_cell_available"] = grouped.single_cell_replication_probability.notna()
        grouped["single_cell_unavailable_reason"] = np.where(
            grouped.single_cell_available, pd.NA,
            "NO_FRESH_FIVE_FOLD_PRIVATE_HEAD_PREDICTION",
        )
    else:
        grouped = pd.DataFrame(columns=[
            "dataset_id", "cancer_id", "cell_type", "cell_type_major", "cell_state",
            "lncrna_id", "pathway_id", "single_cell_replication_probability",
            "single_cell_available", "single_cell_unavailable_reason",
        ])
    nulls = coverage_null_scaffold(candidates, dataset_manifest, association_all)
    typed = pd.concat([grouped, nulls], ignore_index=True, sort=False)
    typed["analysis_version"] = ANALYSIS_VERSION
    typed["training_run_id"] = str(training_run_id)
    typed["module_id"] = MODULE_ID
    typed["target_level"] = TARGET_LEVEL
    typed["prediction_format"] = PREDICTION_FORMAT
    typed["changes_primary_ranking"] = False
    if typed.duplicated(list(TARGET_KEYS)).any():
        raise SingleCellTrainingError("Single-cell typed target keys are duplicated")
    finite = pd.to_numeric(typed.single_cell_replication_probability, errors="coerce").dropna()
    if not finite.between(0, 1).all():
        raise SingleCellTrainingError("Single-cell probability is outside [0, 1]")
    prediction_path = output / "single_cell_typed_predictions.parquet"
    _atomic_parquet(typed, prediction_path)

    # Public companion: lncRNA x cell type activity/detection.
    lnc_output = lnc_celltype.copy()
    lnc_output["lnc_celltype_available"] = True
    lnc_output["lnc_celltype_unavailable_reason"] = pd.NA
    if len(nulls):
        lnc_null = nulls[
            ["cancer_id", "lncrna_id", "single_cell_unavailable_reason"]
        ].drop_duplicates(["cancer_id", "lncrna_id"])
        lnc_null["dataset_id"] = "UNAVAILABLE::" + lnc_null.cancer_id.astype(str)
        lnc_null["cell_type"] = "UNAVAILABLE"
        lnc_null["cell_key"] = "unavailable"
        for column in (
            "lnc_detection_rate", "lnc_mean_log_expression", "lnc_specificity_tau", "lnc_n_cells"
        ):
            lnc_null[column] = np.nan
        lnc_null["lnc_celltype_available"] = False
        lnc_null["lnc_celltype_unavailable_reason"] = lnc_null.pop(
            "single_cell_unavailable_reason"
        )
        lnc_output = pd.concat([lnc_output, lnc_null], ignore_index=True, sort=False)
    lnc_output["analysis_version"] = ANALYSIS_VERSION
    lnc_output["source_is_nonpredictive"] = True
    lnc_path = output / "lnc_celltype_summary.parquet"
    _atomic_parquet(lnc_output, lnc_path)

    # Public companion: exact-pathway aggregation only (no family ID exists).
    exact = typed.groupby(
        ["cancer_id", "lncrna_id", "pathway_id"], observed=True, as_index=False
    ).agg(
        single_cell_replication_probability=("single_cell_replication_probability", "mean"),
        single_cell_available=("single_cell_available", "max"),
        single_cell_unavailable_reason=("single_cell_unavailable_reason", "first"),
        n_dataset_celltype_targets=("dataset_id", "nunique"),
    )
    exact["analysis_version"] = ANALYSIS_VERSION
    exact["exact_pathway_only"] = True
    exact_path = output / "lnc_exact_pathway.parquet"
    _atomic_parquet(exact, exact_path)

    activity_output = activity.copy()
    activity_output["activity_available"] = True
    activity_output["activity_unavailable_reason"] = pd.NA
    if len(nulls):
        activity_null = nulls[
            ["cancer_id", "pathway_id", "single_cell_unavailable_reason"]
        ].drop_duplicates(["cancer_id", "pathway_id"])
        activity_null["dataset_id"] = "UNAVAILABLE::" + activity_null.cancer_id.astype(str)
        activity_null["donor_id"] = pd.NA
        activity_null["cell_type"] = "UNAVAILABLE"
        activity_null["cell_key"] = "unavailable"
        activity_null["activity_kind"] = "NULL_WITH_REASON"
        activity_null["activity_source_tier"] = "unavailable"
        activity_null["effective_source_tier"] = "unavailable"
        activity_null["activity_value"] = np.nan
        activity_null["mean_pseudotime"] = np.nan
        activity_null["n_cells"] = np.nan
        activity_null["activity_available"] = False
        activity_null["activity_unavailable_reason"] = activity_null.pop(
            "single_cell_unavailable_reason"
        )
        activity_output = pd.concat(
            [activity_output, activity_null], ignore_index=True, sort=False
        )
    activity_output["analysis_version"] = ANALYSIS_VERSION
    activity_output["source_is_nonpredictive"] = True
    activity_output_path = output / "activity.parquet"
    _atomic_parquet(activity_output, activity_output_path)

    figure_manifest_path = output / "FIGURE_MANIFEST.json"
    figure_payload = _figure_manifest(
        figure_roots, dataset_manifest, sorted(candidates.cancer_id.unique()),
        figure_manifest_path,
    )
    successful_checkpoints = [row for row in checkpoint_rows if row["status"] == "SUCCESS"]
    checkpoint_manifest = {
        "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "folds": N_FOLDS,
        "records": checkpoint_rows,
        "all_private_heads_random_initialization": all(
            row.get("source_checkpoint_sha256") is None
            and row.get("fresh_random_initialization") is True
            for row in successful_checkpoints
        ),
        "trained_private_head_count": len(successful_checkpoints),
        "checkpoint_file_count": len(successful_checkpoints),
        "all_five_fold_private_heads_trained": len(successful_checkpoints) == N_FOLDS,
        "core_detached_and_frozen": True,
        "core_embedding_usability": core_embedding_usability,
        "constant_lncrna_core_masked": any(
            not row["lncrna"]["usable_for_node_discrimination"]
            for row in core_embedding_usability
        ),
        "core_materialization_policy": "STREAM_SELECTED_ROWS_AND_TEST_BATCHES_ONLY",
    }
    checkpoint_manifest_path = output / "CHECKPOINT_MANIFEST.json"
    _atomic_json(checkpoint_manifest_path, checkpoint_manifest)
    config_path = output / "RUN_CONFIG.json"
    _atomic_json(
        config_path,
        {
            "analysis_version": ANALYSIS_VERSION,
            "training_run_id": training_run_id,
            "training": asdict(settings),
            "formal_single_cell_cancers": sorted(FORMAL_SINGLE_CELL_CANCERS),
            "formal_cancer_boundary_policy": "V3.2_EXACT_33_CANCER_AUTHORITY",
            "split_policy": "DONOR_IF_PRESENT_ELSE_DATASET_BLOCKED_FIVE_FOLD",
            "target_policy": "ABS_ASSOCIATION_EFFECT_X_ONE_MINUS_FDR_LABEL_ONLY",
            "exact_pathway_policy": "INNER_JOIN_ON_LITERAL_V32_PATHWAY_ID_NO_FAMILY_BROADCAST",
            "staging_policy": "NEVER_FORMAL_NULL_WITH_REASON",
            "constant_lncrna_core_policy": (
                "MASK_CONSTANT_CORE_AND_USE_FRESH_SINGLE_CELL_LNCRNA_FEATURES"
            ),
            "core_materialization_policy": "STREAM_SELECTED_ROWS_AND_TEST_BATCHES_ONLY",
        },
    )
    audit_path = output / "INPUT_LINEAGE_AUDIT.json"
    _atomic_json(audit_path, audit)
    code_files = {"single_cell_training.py": artifact_sha256(Path(__file__).resolve())}
    runner = Path(__file__).resolve().parents[2] / "scripts" / "run_v32_single_cell_training.py"
    if runner.is_file():
        code_files["run_v32_single_cell_training.py"] = artifact_sha256(runner)
    input_artifacts = [
        {
            "path": row["path"], "sha256": row["sha256"],
            "artifact_kind": row["source_role"], "generation": row["generation"],
            "source_role": row["source_role"], "outcome_derived": row["outcome_derived"],
            "fold_fitted": row["fold_fitted"], "use_role": row["use_role"],
        }
        for row in audit["artifacts"]
    ]
    input_artifacts.append(
        {
            "path": str(paths["core_manifest"]), "sha256": core_manifest_sha,
            "artifact_kind": "v32_core_checkpoint", "generation": "V3.2",
            "source_role": "v32_core_checkpoint", "outcome_derived": True,
            "fold_fitted": True, "use_role": "aux_parent",
        }
    )
    trained_records = successful_checkpoints
    if len(trained_records) == N_FOLDS:
        training_status = "SUCCESS"
    elif len(trained_records) == 0:
        training_status = "AUDITED_UNAVAILABLE"
    else:
        training_status = "INCOMPLETE"
    available_mask = typed.single_cell_available.fillna(False).astype(bool)
    all_probabilities_null = bool(
        pd.to_numeric(typed.single_cell_replication_probability, errors="coerce").isna().all()
    )
    all_unavailable_rows_have_reason = bool(
        typed.loc[~available_mask, "single_cell_unavailable_reason"].notna().all()
    )
    lineage = {
        "module_id": MODULE_ID,
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": training_run_id,
        "training_status": training_status,
        "initialization_policy": "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        "folds": N_FOLDS,
        "seeds": [int(settings.seed + fold * 101) for fold in range(N_FOLDS)],
        "code_sha256": _canonical_sha256(code_files),
        "config_sha256": artifact_sha256(config_path),
        "input_manifest_sha256": audit["lineage_sha256"],
        "checkpoint_manifest_sha256": artifact_sha256(checkpoint_manifest_path),
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "old_family_support_used": False,
        "private_head_trained_from_scratch": len(trained_records) == N_FOLDS,
        "all_five_fold_private_heads_trained": len(trained_records) == N_FOLDS,
        "trained_folds": len(trained_records),
        "checkpoint_files": len(trained_records),
        "release_ready": (
            len(trained_records) == N_FOLDS and int(available_mask.sum()) > 0
        ),
        "all_probabilities_null": all_probabilities_null,
        "all_unavailable_rows_have_reason": all_unavailable_rows_have_reason,
        "core_parameters_frozen": True,
        "core_embedding_usability": core_embedding_usability,
        "constant_lncrna_core_masked": any(
            not row["lncrna"]["usable_for_node_discrimination"]
            for row in core_embedding_usability
        ),
        "core_materialization_policy": "STREAM_SELECTED_ROWS_AND_TEST_BATCHES_ONLY",
        "lncrna_identity_feature_source": (
            "fresh_single_cell_detection_expression_specificity"
        ),
        "v32_core_checkpoint_sha256": core_manifest_sha,
        "core_parameters_before_sha256": core_parameter_composite,
        "core_parameters_after_sha256": core_parameter_composite,
        "split_unit": "dataset_or_donor",
        "donor_dataset_overlap_across_splits": False,
        "formal_cancer_boundary": sorted(FORMAL_SINGLE_CELL_CANCERS),
        "formal_cancer_boundary_policy": "V3.2_EXACT_33_CANCER_AUTHORITY",
        "formal_datasets": sorted(dataset_manifest.loc[dataset_manifest.qualified, "dataset_id"]),
        "staging_datasets_not_promoted": sorted(
            dataset_manifest.loc[dataset_manifest.source_tier.isin(_STAGING_TIERS), "dataset_id"]
        ),
        "input_artifacts": input_artifacts,
        "prediction_path": str(prediction_path),
        "prediction_sha256": artifact_sha256(prediction_path),
        "lnc_celltype_path": str(lnc_path),
        "lnc_celltype_sha256": artifact_sha256(lnc_path),
        "lnc_exact_pathway_path": str(exact_path),
        "lnc_exact_pathway_sha256": artifact_sha256(exact_path),
        "activity_path": str(activity_output_path),
        "activity_sha256": artifact_sha256(activity_output_path),
        "figure_manifest_path": str(figure_manifest_path),
        "figure_manifest_sha256": artifact_sha256(figure_manifest_path),
        "figure_manifest_entries": len(figure_payload["entries"]),
        "prediction_rows": int(len(typed)),
        "available_rows": int(available_mask.sum()),
        "null_rows": int((~available_mask).sum()),
    }
    lineage_path = output / "LINEAGE.json"
    _atomic_json(lineage_path, lineage)
    contract_validation_path = output / "FULL_MODEL_CONTRACT_VALIDATION.json"
    try:
        from .full_model_contract import validate_module_lineage

        validate_module_lineage(MODULE_ID, lineage)
        contract_validation = {
            "status": "PASS",
            "module_id": MODULE_ID,
            "training_status": training_status,
            "validator": "cc_hhgt.v32.full_model_contract.validate_module_lineage",
            "fail_closed": True,
        }
    except Exception as exc:
        contract_validation = {
            "status": "REJECTED_FAIL_CLOSED",
            "module_id": MODULE_ID,
            "training_status": training_status,
            "validator": "cc_hhgt.v32.full_model_contract.validate_module_lineage",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "fail_closed": True,
        }
    _atomic_json(contract_validation_path, contract_validation)
    success = {
        "status": training_status,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "training_run_id": training_run_id,
        "prediction_path": str(prediction_path),
        "prediction_sha256": lineage["prediction_sha256"],
        "lineage_path": str(lineage_path),
        "lineage_sha256": artifact_sha256(lineage_path),
        "formal_datasets": len(lineage["formal_datasets"]),
        "trained_folds": len(trained_records),
        "available_rows": lineage["available_rows"],
        "null_rows": lineage["null_rows"],
        "staging_never_promoted": True,
        "exact_pathway_only": True,
        "release_ready": len(trained_records) == N_FOLDS and lineage["available_rows"] > 0,
        "all_probabilities_null": all_probabilities_null,
        "all_unavailable_rows_have_reason": all_unavailable_rows_have_reason,
        "contract_validation_path": str(contract_validation_path),
        "contract_validation_status": contract_validation["status"],
    }
    _atomic_json(output / "SUCCESS.json", success)
    return success


__all__ = [
    "ANALYSIS_VERSION",
    "FORMAL_SINGLE_CELL_CANCERS",
    "SingleCellTrainingConfig",
    "SingleCellTrainingError",
    "build_blocked_folds",
    "build_domain_features",
    "candidate_core",
    "candidate_core_availability",
    "coverage_null_scaffold",
    "embedding_usability",
    "exact_candidate_join",
    "fit_private_head",
    "normalise_candidates",
    "normalise_dataset_manifest",
    "normalise_single_cell_associations",
    "run_single_cell_training",
    "split_block_ids",
    "validate_core_manifest",
    "validate_fold_isolation",
]
