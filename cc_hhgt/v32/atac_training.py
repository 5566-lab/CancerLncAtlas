"""Fresh, patient-fold OOF ATAC co-accessibility expert for V3.2.

The expert is intentionally independent from every historical model result.  It
uses only the official TCGA ATAC accessibility matrix after a separately
audited promoter aggregation, the canonical V3.2 patient folds, exact pathway
membership, and the exact candidate universe.

For each outer patient fold a small logistic calibration head is trained from
scratch.  Inner-training patients provide co-accessibility features and the
next non-test fold provides a persistence target; outer-test patients never
enter normalisation, labels, fitting, or prediction features.  Consequently a
fold prediction asks whether an ATAC association learned in the other patients
is expected to persist in the held-out patients.  It is evidence of
co-accessibility, not direction, mediation, perturbation, or causality.

The implementation streams one cancer and one candidate batch at a time.  It
never creates a 3.3M x patient dense matrix and it writes all Parquet outputs
incrementally.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MODULE_ID = "atac_coaccessibility"
N_FOLDS = 5
TARGET_KEYS = ("cancer_id", "lncrna_id", "pathway_id")
PREDICTION_FORMAT = "CC_HHGT_V3_2_ATAC_TYPED_PREDICTIONS_V1"
CHECKPOINT_FORMAT = "CC_HHGT_V3_2_ATAC_FOLD_CALIBRATOR_V1"
MATERIALIZATION_FORMAT = "CC_HHGT_V3_2_ATAC_GENE_ACCESSIBILITY_SUCCESS_V1"

FORMAL_CANCERS = (
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
    "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
    "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS",
    "UVM",
)
RAW_GAP_CANCERS = (
    "DLBC", "KICH", "LAML", "OV", "PAAD", "READ", "SARC", "THYM",
    "UCS", "UVM",
)
RAW_COVERED_CANCERS = tuple(
    cancer for cancer in FORMAL_CANCERS if cancer not in RAW_GAP_CANCERS
)
FEATURE_NAMES = (
    "lncrna_open_mean",
    "pathway_open_mean",
    "joint_open_mean",
    "state_concordance",
    "signed_covariance",
    "log_patient_count",
    "log_pathway_promoter_genes",
)

_FORBIDDEN_INPUT_NAME_TOKENS = (
    "checkpoint", "oof_prediction", "prediction", "probability", "ranking",
    "ranked",
)
_FORBIDDEN_INPUT_COLUMN_TOKENS = (
    "adjusted_probability", "historical_probability", "oof_prediction",
    "pair_support", "prediction", "probability", "ranking", "sample_weight",
)


class AtacTrainingError(RuntimeError):
    """Raised when the fresh ATAC training contract is violated."""


@dataclass(frozen=True)
class AtacTrainingConfig:
    seed: int = 20260726
    min_pathway_promoter_genes: int = 3
    min_inner_train_patients: int = 2
    min_outer_train_patients: int = 2
    max_training_rows: int = 150_000
    prediction_pair_batch_size: int = 20_000
    epochs: int = 60
    learning_rate: float = 0.04
    l2: float = 1e-3
    expected_rows_per_cancer: int | None = 100_000

    def validate(self) -> None:
        positive = (
            self.min_pathway_promoter_genes,
            self.min_inner_train_patients,
            self.min_outer_train_patients,
            self.max_training_rows,
            self.prediction_pair_batch_size,
            self.epochs,
        )
        if any(int(value) < 1 for value in positive):
            raise ValueError("ATAC integer controls must be positive")
        if self.learning_rate <= 0 or self.l2 < 0:
            raise ValueError("ATAC optimiser controls are invalid")
        if self.expected_rows_per_cancer is not None and self.expected_rows_per_cancer < 1:
            raise ValueError("expected_rows_per_cancer must be positive or None")


@dataclass(frozen=True)
class CancerAccessibility:
    cancer_id: str
    gene_ids: tuple[str, ...]
    values: np.ndarray
    patient_ids: tuple[str, ...]
    patient_folds: np.ndarray
    gene_index: Mapping[str, int]


@dataclass(frozen=True)
class CandidateMap:
    lncrna_gene_index: np.ndarray
    pathway_index: np.ndarray
    pathway_gene_count: np.ndarray
    structural_reason: np.ndarray


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normalise_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _gene_id(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"^(?:LNC:|LNCRNA:|GENE:)", "", text, flags=re.IGNORECASE)
    return text.split(".", 1)[0]


def _patient_id(value: Any) -> str:
    text = str(value).strip().replace(".", "-")
    return text[:12] if text.upper().startswith("TCGA-") and len(text) >= 12 else text


def _assert_nonpredictive_path(path: str | Path, context: str) -> None:
    token = _normalise_token(Path(path).name)
    found = [item for item in _FORBIDDEN_INPUT_NAME_TOKENS if item in token]
    if found:
        raise AtacTrainingError(
            f"{context} resembles a historical/result input: {path}; tokens={found}"
        )


def _assert_nonpredictive_columns(columns: Iterable[Any], context: str) -> None:
    bad = []
    for column in columns:
        token = _normalise_token(column)
        if any(item in token for item in _FORBIDDEN_INPUT_COLUMN_TOKENS):
            bad.append(str(column))
    if bad:
        raise AtacTrainingError(f"{context} contains result-like columns: {sorted(bad)}")


def normalise_patient_folds(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = {_normalise_token(column): str(column) for column in frame.columns}
    cancer_col = lookup.get("cancer_id")
    patient_col = lookup.get("patient_id")
    fold_col = (
        lookup.get("patient_fold_id") or lookup.get("fold_id") or lookup.get("fold")
    )
    if not cancer_col or not patient_col or not fold_col:
        raise AtacTrainingError("Patient-fold manifest lacks cancer/patient/fold columns")
    result = frame[[cancer_col, patient_col, fold_col]].rename(
        columns={
            cancer_col: "cancer_id", patient_col: "patient_id",
            fold_col: "patient_fold_id",
        }
    )
    result["cancer_id"] = result.cancer_id.astype(str).str.upper()
    if result[["cancer_id", "patient_id"]].isna().any().any():
        raise AtacTrainingError("Patient-fold manifest contains null cancer/patient IDs")
    result["patient_id"] = result.patient_id.astype(str).str.strip()
    if result.patient_id.eq("").any():
        raise AtacTrainingError("Patient-fold manifest contains empty patient_id")
    result["patient_fold_id"] = pd.to_numeric(
        result.patient_fold_id, errors="raise"
    ).astype(int)
    result = result.drop_duplicates()
    conflict = result.groupby(
        ["cancer_id", "patient_id"], observed=True
    ).patient_fold_id.nunique()
    if (conflict > 1).any():
        raise AtacTrainingError("A patient occurs in multiple canonical folds")
    result = result.drop_duplicates(["cancer_id", "patient_id"])
    observed_by_cancer = result.groupby("cancer_id", observed=True).patient_fold_id.agg(
        lambda value: frozenset(map(int, value))
    )
    bad = observed_by_cancer.loc[observed_by_cancer.ne(frozenset(range(N_FOLDS)))]
    if not bad.empty:
        raise AtacTrainingError(f"ATAC requires patient folds 0..4 in every cancer: {bad.to_dict()}")
    return result.sort_values(
        ["cancer_id", "patient_id"], kind="stable"
    ).reset_index(drop=True)


def normalise_membership(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_nonpredictive_columns(frame.columns, "Exact pathway membership")
    if not {"pathway_id", "gene_id"}.issubset(frame.columns):
        if "pathway_family_id" in frame.columns:
            raise AtacTrainingError("Family-level pathway membership is forbidden")
        raise AtacTrainingError("Exact membership requires pathway_id and gene_id")
    result = frame[["pathway_id", "gene_id"]].dropna().astype(str)
    result["gene_id"] = result.gene_id.map(_gene_id)
    result = result.drop_duplicates()
    if result.empty:
        raise AtacTrainingError("Exact pathway membership is empty")
    return result


def _candidate_table(path: Path, cancer_id: str) -> pd.DataFrame:
    import pyarrow.parquet as pq

    table = pq.read_table(
        path,
        columns=list(TARGET_KEYS),
        filters=[("cancer_id", "=", str(cancer_id))],
    )
    frame = table.to_pandas()
    _assert_nonpredictive_columns(frame.columns, f"{cancer_id} candidates")
    if set(frame.columns) != set(TARGET_KEYS):
        raise AtacTrainingError(f"{cancer_id} candidate keys are malformed")
    frame = frame.loc[:, list(TARGET_KEYS)].astype(str).reset_index(drop=True)
    frame["cancer_id"] = frame.cancer_id.str.upper()
    if frame.empty or not frame.cancer_id.eq(cancer_id).all():
        raise AtacTrainingError(f"Candidate universe lacks cancer {cancer_id}")
    if frame.duplicated(list(TARGET_KEYS)).any():
        raise AtacTrainingError(f"Candidate universe duplicates exact keys for {cancer_id}")
    return frame


def _validate_materialization(success_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(success_path.read_text(encoding="utf-8"))
    if payload.get("status") != "SUCCESS" or payload.get("format") != MATERIALIZATION_FORMAT:
        raise AtacTrainingError("ATAC gene-accessibility materialization is not immutable SUCCESS")
    if payload.get("old_predictions_used") is not False:
        raise AtacTrainingError("ATAC materialization does not prohibit old predictions")
    if payload.get("old_checkpoints_used") is not False:
        raise AtacTrainingError("ATAC materialization does not prohibit old checkpoints")
    declared_covered = tuple(sorted(map(str, payload.get("covered_cancers", []))))
    declared_gaps = tuple(sorted(map(str, payload.get("raw_gap_cancers", []))))
    if declared_covered != tuple(sorted(RAW_COVERED_CANCERS)):
        raise AtacTrainingError("ATAC materialization does not contain the exact 23 raw-covered cancers")
    if declared_gaps != tuple(sorted(RAW_GAP_CANCERS)):
        raise AtacTrainingError("ATAC materialization raw-gap list is not the exact typed 10")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != len(RAW_COVERED_CANCERS):
        raise AtacTrainingError("ATAC materialization lacks 23 cancer records")
    by_cancer: dict[str, dict[str, Any]] = {}
    for record in records:
        cancer = str(record.get("cancer_id", "")).upper()
        path = Path(str(record.get("path", "")))
        if cancer in by_cancer or cancer not in RAW_COVERED_CANCERS:
            raise AtacTrainingError("ATAC materialization cancer records are invalid")
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
        observed_sha = file_sha256(path)
        if observed_sha != record.get("sha256"):
            raise AtacTrainingError(f"ATAC materialization hash mismatch for {cancer}")
        by_cancer[cancer] = {**record, "path": str(path), "sha256": observed_sha}
    return payload, by_cancer


def _load_accessibility(
    record: Mapping[str, Any], cancer_id: str, folds: pd.DataFrame
) -> CancerAccessibility:
    path = Path(str(record["path"]))
    frame = pd.read_parquet(path)
    if "gene_id" not in frame.columns:
        raise AtacTrainingError(f"{cancer_id} accessibility lacks gene_id")
    frame["gene_id"] = frame.gene_id.map(_gene_id)
    if frame.gene_id.duplicated().any() or frame.empty:
        raise AtacTrainingError(f"{cancer_id} accessibility genes are empty/duplicated")
    patient_columns = [str(column) for column in frame.columns if str(column) != "gene_id"]
    normalised = [_patient_id(column) for column in patient_columns]
    if len(set(normalised)) != len(normalised):
        raise AtacTrainingError(f"{cancer_id} accessibility patients collapse after normalisation")
    fold_local = folds.loc[folds.cancer_id.eq(cancer_id)].set_index("patient_id")
    unknown = sorted(set(normalised) - set(fold_local.index.astype(str)))
    if unknown:
        raise AtacTrainingError(
            f"{cancer_id} accessibility contains non-canonical patients: {unknown[:5]}"
        )
    values = frame[patient_columns].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float32
    )
    if not np.isfinite(values).all():
        raise AtacTrainingError(f"{cancer_id} accessibility contains non-finite values")
    patient_folds = fold_local.loc[normalised, "patient_fold_id"].to_numpy(dtype=np.int8)
    genes = tuple(frame.gene_id.astype(str))
    return CancerAccessibility(
        cancer_id=cancer_id,
        gene_ids=genes,
        values=values,
        patient_ids=tuple(normalised),
        patient_folds=patient_folds,
        gene_index={gene: index for index, gene in enumerate(genes)},
    )


def _pathway_projection(
    candidates: pd.DataFrame,
    accessibility: CancerAccessibility,
    membership: pd.DataFrame,
    min_pathway_genes: int,
) -> tuple[tuple[str, ...], Any, np.ndarray, CandidateMap]:
    from scipy import sparse

    pathways = tuple(pd.unique(candidates.pathway_id.astype(str)))
    pathway_index = {value: index for index, value in enumerate(pathways)}
    local = membership.loc[
        membership.pathway_id.astype(str).isin(pathway_index)
        & membership.gene_id.astype(str).isin(accessibility.gene_index)
    ].copy()
    local["pathway_index"] = local.pathway_id.astype(str).map(pathway_index)
    local["gene_index"] = local.gene_id.astype(str).map(accessibility.gene_index)
    local = local.dropna(subset=["pathway_index", "gene_index"]).drop_duplicates(
        ["pathway_index", "gene_index"]
    )
    counts = np.zeros(len(pathways), dtype=np.int32)
    if not local.empty:
        row = local.pathway_index.to_numpy(dtype=np.int64)
        col = local.gene_index.to_numpy(dtype=np.int64)
        counts += np.bincount(row, minlength=len(pathways)).astype(np.int32)
        weight = sparse.csr_matrix(
            (np.ones(len(local), dtype=np.float32), (row, col)),
            shape=(len(pathways), len(accessibility.gene_ids)),
        )
    else:
        weight = sparse.csr_matrix(
            (len(pathways), len(accessibility.gene_ids)), dtype=np.float32
        )
    inverse = np.zeros(len(pathways), dtype=np.float32)
    valid = counts >= int(min_pathway_genes)
    inverse[valid] = 1.0 / counts[valid]
    weight = sparse.diags(inverse) @ weight

    lnc_indices = np.asarray(
        [accessibility.gene_index.get(_gene_id(value), -1) for value in candidates.lncrna_id],
        dtype=np.int64,
    )
    path_indices = candidates.pathway_id.astype(str).map(pathway_index).to_numpy(dtype=np.int64)
    path_counts = counts[path_indices]
    reason = np.full(len(candidates), "", dtype=object)
    reason[lnc_indices < 0] = "ATAC_LNCRNA_PROMOTER_NOT_MAPPED"
    reason[(lnc_indices >= 0) & (path_counts < int(min_pathway_genes))] = (
        "ATAC_EXACT_PATHWAY_PROMOTER_COVERAGE_LT_MINIMUM"
    )
    return pathways, weight.tocsr(), counts, CandidateMap(
        lncrna_gene_index=lnc_indices,
        pathway_index=path_indices,
        pathway_gene_count=path_counts,
        structural_reason=reason,
    )


def _standardised_matrices(
    accessibility: CancerAccessibility,
    pathway_weight: Any,
    fit_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if int(fit_mask.sum()) < 1:
        raise AtacTrainingError("Cannot standardise ATAC with zero fit patients")
    fit = accessibility.values[:, fit_mask]
    mean = fit.mean(axis=1, dtype=np.float64).astype(np.float32)
    if fit.shape[1] >= 2:
        scale = fit.std(axis=1, ddof=1, dtype=np.float64).astype(np.float32)
    else:
        scale = np.ones(len(mean), dtype=np.float32)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    z = np.clip((accessibility.values - mean[:, None]) / scale[:, None], -8.0, 8.0)
    pathway_z = np.asarray(pathway_weight @ z, dtype=np.float32)
    return z.astype(np.float32, copy=False), pathway_z


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def _pair_features(
    candidates: pd.DataFrame,
    mapping: CandidateMap,
    lnc_z: np.ndarray,
    pathway_z: np.ndarray,
    patient_mask: np.ndarray,
    *,
    start: int = 0,
    stop: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stop = len(candidates) if stop is None else min(int(stop), len(candidates))
    position = np.arange(int(start), stop, dtype=np.int64)
    structural = mapping.structural_reason[position] == ""
    available_position = position[structural]
    features = np.full((len(position), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    if not len(available_position) or int(patient_mask.sum()) < 1:
        return features, structural
    lnc = lnc_z[
        mapping.lncrna_gene_index[available_position]
    ][:, patient_mask]
    pathway = pathway_z[
        mapping.pathway_index[available_position]
    ][:, patient_mask]
    a = _sigmoid(lnc.astype(np.float64))
    b = _sigmoid(pathway.astype(np.float64))
    joint = a * b
    concordance = joint + (1.0 - a) * (1.0 - b)
    centered_a = a - a.mean(axis=1, keepdims=True)
    centered_b = b - b.mean(axis=1, keepdims=True)
    local = np.column_stack(
        [
            a.mean(axis=1),
            b.mean(axis=1),
            joint.mean(axis=1),
            concordance.mean(axis=1),
            (centered_a * centered_b).mean(axis=1),
            np.full(len(available_position), math.log1p(int(patient_mask.sum()))),
            np.log1p(mapping.pathway_gene_count[available_position]),
        ]
    ).astype(np.float32)
    features[np.flatnonzero(structural)] = local
    return features, structural


def _persistence_score(
    mapping: CandidateMap,
    lnc_z: np.ndarray,
    pathway_z: np.ndarray,
    patient_mask: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    lnc = lnc_z[mapping.lncrna_gene_index[positions]][:, patient_mask]
    pathway = pathway_z[mapping.pathway_index[positions]][:, patient_mask]
    a = _sigmoid(lnc.astype(np.float64))
    b = _sigmoid(pathway.astype(np.float64))
    return ((2.0 * a - 1.0) * (2.0 * b - 1.0)).mean(axis=1).astype(np.float32)


def _fit_logistic(
    features: np.ndarray,
    persistence: np.ndarray,
    *,
    fold: int,
    config: AtacTrainingConfig,
) -> dict[str, Any]:
    valid = np.isfinite(features).all(axis=1) & np.isfinite(persistence)
    x = features[valid].astype(np.float64)
    target_score = persistence[valid].astype(np.float64)
    if len(x) < 100:
        raise AtacTrainingError(f"ATAC fold {fold} has fewer than 100 training examples")
    threshold = float(np.median(target_score))
    y = (target_score > threshold).astype(np.float64)
    if len(np.unique(y)) != 2:
        raise AtacTrainingError(f"ATAC fold {fold} persistence target has one class")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    x = (x - mean) / scale
    design = np.column_stack([np.ones(len(x)), x])
    weight = np.zeros(design.shape[1], dtype=np.float64)
    positive = float(y.mean())
    sample_weight = np.where(y > 0.5, 0.5 / positive, 0.5 / (1.0 - positive))
    history: list[float] = []
    for _ in range(int(config.epochs)):
        probability = _sigmoid(design @ weight)
        gradient = design.T @ ((probability - y) * sample_weight) / len(y)
        gradient[1:] += float(config.l2) * weight[1:]
        weight -= float(config.learning_rate) * gradient
        loss = -np.mean(
            sample_weight
            * (y * np.log(np.clip(probability, 1e-8, 1.0))
               + (1.0 - y) * np.log(np.clip(1.0 - probability, 1e-8, 1.0)))
        ) + 0.5 * float(config.l2) * float(np.square(weight[1:]).sum())
        history.append(float(loss))
    if not np.isfinite(weight).all() or not np.isfinite(history).all():
        raise AtacTrainingError(f"ATAC fold {fold} logistic fit is non-finite")
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "modality": "atac",
        "patient_fold": int(fold),
        "initialization": "ALL_ZERO_FRESH_NO_SOURCE_CHECKPOINT",
        "source_checkpoint_sha256": None,
        "old_checkpoint_loaded": False,
        "old_predictions_used": False,
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights_with_intercept": weight.tolist(),
        "persistence_target_threshold": threshold,
        "persistence_target_prevalence": positive,
        "train_rows": int(len(x)),
        "epochs": int(config.epochs),
        "loss_first": history[0],
        "loss_last": history[-1],
    }


def _predict_logistic(model: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weight = np.asarray(model["weights_with_intercept"], dtype=np.float64)
    design = np.column_stack(
        [np.ones(len(features)), (features.astype(np.float64) - mean) / scale]
    )
    result = _sigmoid(design @ weight).astype(np.float32)
    if not np.isfinite(result).all() or not ((result >= 0) & (result <= 1)).all():
        raise AtacTrainingError("ATAC calibrated probabilities are invalid")
    return result


def _sample_positions(
    eligible: np.ndarray, maximum: int, *, seed: int
) -> np.ndarray:
    positions = np.flatnonzero(eligible)
    if len(positions) <= maximum:
        return positions
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(positions, size=int(maximum), replace=False))


def _arrow_table(frame: pd.DataFrame) -> Any:
    import pyarrow as pa

    return pa.Table.from_pandas(frame, preserve_index=False)


def _fold_frame(
    candidates: pd.DataFrame,
    fold: int,
    probability: np.ndarray,
    reason: np.ndarray,
    outer_train_count: int,
    outer_test_count: int,
) -> pd.DataFrame:
    available = np.isfinite(probability)
    final_reason = np.asarray(reason, dtype=object).copy()
    final_reason[available] = None
    final_reason[(~available) & (final_reason == "")] = "ATAC_NO_FOLD_OOF_PREDICTION"
    return candidates.assign(
        patient_fold_id=np.int8(fold),
        atac_context_probability=pd.Series(probability, dtype="float32"),
        atac_available=pd.Series(available, dtype=bool),
        atac_unavailable_reason=pd.Series(final_reason, dtype="string"),
        outer_train_patient_count=np.int16(outer_train_count),
        outer_test_patient_count=np.int16(outer_test_count),
    )


def _gap_fold_frame(candidates: pd.DataFrame, fold: int) -> pd.DataFrame:
    count = len(candidates)
    return _fold_frame(
        candidates,
        fold,
        np.full(count, np.nan, dtype=np.float32),
        np.full(count, "ATAC_RAW_NOT_AVAILABLE_FOR_CANCER", dtype=object),
        0,
        0,
    )


def run_atac_training(
    *,
    candidates_path: str | Path,
    patient_folds_path: str | Path,
    pathway_membership_path: str | Path,
    materialization_success_path: str | Path,
    output_root: str | Path,
    training_run_id: str,
    config: AtacTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train five fresh calibrators and stream unified typed predictions."""

    settings = config or AtacTrainingConfig()
    settings.validate()
    paths = {
        "candidates": Path(candidates_path).resolve(),
        "folds": Path(patient_folds_path).resolve(),
        "membership": Path(pathway_membership_path).resolve(),
        "materialization_success": Path(materialization_success_path).resolve(),
    }
    for name, path in paths.items():
        _assert_nonpredictive_path(path, name)
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    output = Path(output_root).resolve()
    if output.exists():
        raise AtacTrainingError(f"ATAC trainer refuses output reuse: {output}")

    input_hashes = {name: file_sha256(path) for name, path in paths.items()}
    materialization, records = _validate_materialization(paths["materialization_success"])
    folds = normalise_patient_folds(pd.read_csv(paths["folds"], sep="\t"))
    membership = normalise_membership(pd.read_parquet(paths["membership"]))

    candidate_counts: dict[str, int] = {}
    total_rows = 0
    for cancer in FORMAL_CANCERS:
        frame = _candidate_table(paths["candidates"], cancer)
        if (
            settings.expected_rows_per_cancer is not None
            and len(frame) != int(settings.expected_rows_per_cancer)
        ):
            raise AtacTrainingError(
                f"{cancer} has {len(frame)} candidates; expected "
                f"{settings.expected_rows_per_cancer}"
            )
        candidate_counts[cancer] = len(frame)
        total_rows += len(frame)
        del frame
    if set(folds.cancer_id.unique()) != set(FORMAL_CANCERS):
        raise AtacTrainingError("Patient-fold manifest is not the exact 33-cancer scope")

    output.mkdir(parents=True)
    checkpoint_root = output / "checkpoints"
    fold_prediction_root = output / "patient_fold_oof_predictions"
    checkpoint_root.mkdir()
    fold_prediction_root.mkdir()

    # First pass: bounded, deterministic training samples for each outer fold.
    quota = max(1, math.ceil(settings.max_training_rows / len(RAW_COVERED_CANCERS)))
    feature_parts: dict[int, list[np.ndarray]] = {fold: [] for fold in range(N_FOLDS)}
    score_parts: dict[int, list[np.ndarray]] = {fold: [] for fold in range(N_FOLDS)}
    sample_audit: dict[int, list[dict[str, Any]]] = {fold: [] for fold in range(N_FOLDS)}
    for cancer_offset, cancer in enumerate(RAW_COVERED_CANCERS):
        candidates = _candidate_table(paths["candidates"], cancer)
        accessibility = _load_accessibility(records[cancer], cancer, folds)
        _, pathway_weight, _, mapping = _pathway_projection(
            candidates, accessibility, membership, settings.min_pathway_promoter_genes
        )
        structurally_eligible = mapping.structural_reason == ""
        for fold in range(N_FOLDS):
            validation_fold = (fold + 1) % N_FOLDS
            inner_mask = (accessibility.patient_folds != fold) & (
                accessibility.patient_folds != validation_fold
            )
            validation_mask = accessibility.patient_folds == validation_fold
            if (
                int(inner_mask.sum()) < settings.min_inner_train_patients
                or int(validation_mask.sum()) < 1
            ):
                sample_audit[fold].append(
                    {
                        "cancer_id": cancer,
                        "status": "SKIPPED_INSUFFICIENT_INNER_PATIENTS",
                        "inner_train_patients": int(inner_mask.sum()),
                        "inner_validation_patients": int(validation_mask.sum()),
                    }
                )
                continue
            positions = _sample_positions(
                structurally_eligible,
                quota,
                seed=settings.seed + fold * 10_003 + cancer_offset * 101,
            )
            z, pathway_z = _standardised_matrices(
                accessibility, pathway_weight, inner_mask
            )
            # Compute only sampled rows.  _pair_features indexes a contiguous
            # range, so a tiny sampled candidate view/mapping keeps memory fixed.
            sample_candidates = candidates.iloc[positions].reset_index(drop=True)
            sample_mapping = CandidateMap(
                lncrna_gene_index=mapping.lncrna_gene_index[positions],
                pathway_index=mapping.pathway_index[positions],
                pathway_gene_count=mapping.pathway_gene_count[positions],
                structural_reason=mapping.structural_reason[positions],
            )
            features, available = _pair_features(
                sample_candidates, sample_mapping, z, pathway_z, inner_mask
            )
            valid_positions = np.flatnonzero(available)
            persistence = _persistence_score(
                sample_mapping, z, pathway_z, validation_mask, valid_positions
            )
            local_features = features[valid_positions]
            feature_parts[fold].append(local_features)
            score_parts[fold].append(persistence)
            sample_audit[fold].append(
                {
                    "cancer_id": cancer,
                    "status": "USED",
                    "inner_train_patients": int(inner_mask.sum()),
                    "inner_validation_patients": int(validation_mask.sum()),
                    "sampled_pairs": int(len(local_features)),
                }
            )

    models: dict[int, dict[str, Any]] = {}
    checkpoint_records: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        if not feature_parts[fold]:
            raise AtacTrainingError(f"ATAC fold {fold} has no training cancer")
        x = np.concatenate(feature_parts[fold], axis=0)
        score = np.concatenate(score_parts[fold], axis=0)
        if len(x) > settings.max_training_rows:
            positions = _sample_positions(
                np.ones(len(x), dtype=bool),
                settings.max_training_rows,
                seed=settings.seed + fold * 1_000_003,
            )
            x = x[positions]
            score = score[positions]
        model = _fit_logistic(x, score, fold=fold, config=settings)
        model.update(
            {
                "training_run_id": str(training_run_id),
                "outer_test_fold": fold,
                "inner_validation_fold": (fold + 1) % N_FOLDS,
                "outer_test_patients_used_for_fit": False,
                "input_hashes": input_hashes,
                "materialization_manifest_sha256": materialization.get(
                    "manifest_sha256"
                ),
                "sample_audit": sample_audit[fold],
            }
        )
        checkpoint_path = checkpoint_root / f"atac_patient_fold_{fold}.json"
        _atomic_json(checkpoint_path, model)
        checkpoint_sha = file_sha256(checkpoint_path)
        models[fold] = model
        checkpoint_records.append(
            {
                "patient_fold": fold,
                "status": "TRAINED_PREDICTION_PENDING",
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
                "train_rows": int(model["train_rows"]),
            }
        )

    # Second pass: one cancer and one pair batch at a time, writing five OOF
    # partitions plus the fold-averaged typed table.  The five unaveraged
    # partitions are retained for routing and fair-comparison audits.
    import pyarrow.parquet as pq

    fold_paths: dict[int, Path] = {}
    fold_writers: dict[int, Any] = {}
    for fold in range(N_FOLDS):
        partition = fold_prediction_root / f"patient_fold={fold}"
        partition.mkdir()
        fold_paths[fold] = partition / "part-0.parquet"
    typed_path = output / "atac_typed_predictions.parquet"
    typed_writer = None
    cancer_audit: list[dict[str, Any]] = []
    available_cancers: list[str] = []
    try:
        for cancer in FORMAL_CANCERS:
            candidates = _candidate_table(paths["candidates"], cancer)
            fold_probabilities: list[np.ndarray] = []
            fold_reasons: list[np.ndarray] = []
            fold_train_counts: list[int] = []
            fold_test_counts: list[int] = []
            if cancer in RAW_GAP_CANCERS:
                for fold in range(N_FOLDS):
                    frame = _gap_fold_frame(candidates, fold)
                    table = _arrow_table(frame)
                    if fold not in fold_writers:
                        fold_writers[fold] = pq.ParquetWriter(
                            fold_paths[fold], table.schema, compression="zstd"
                        )
                    fold_writers[fold].write_table(table)
                    fold_probabilities.append(
                        np.full(len(candidates), np.nan, dtype=np.float32)
                    )
                    fold_reasons.append(
                        np.full(
                            len(candidates),
                            "ATAC_RAW_NOT_AVAILABLE_FOR_CANCER",
                            dtype=object,
                        )
                    )
                    fold_train_counts.append(0)
                    fold_test_counts.append(0)
                structural_reason = fold_reasons[0]
                mapped_lncrnas = 0
                covered_pathways = 0
                aligned_patients = 0
            else:
                accessibility = _load_accessibility(records[cancer], cancer, folds)
                pathways, pathway_weight, pathway_counts, mapping = _pathway_projection(
                    candidates,
                    accessibility,
                    membership,
                    settings.min_pathway_promoter_genes,
                )
                structural_reason = mapping.structural_reason.copy()
                aligned_patients = len(accessibility.patient_ids)
                mapped_lncrnas = int(
                    len(
                        {
                            str(value)
                            for value, index in zip(
                                candidates.lncrna_id,
                                mapping.lncrna_gene_index,
                                strict=True,
                            )
                            if index >= 0
                        }
                    )
                )
                covered_pathways = int(
                    sum(pathway_counts >= settings.min_pathway_promoter_genes)
                )
                for fold in range(N_FOLDS):
                    outer_train = accessibility.patient_folds != fold
                    outer_test = accessibility.patient_folds == fold
                    probability = np.full(len(candidates), np.nan, dtype=np.float32)
                    reason = structural_reason.copy()
                    if int(outer_train.sum()) < settings.min_outer_train_patients:
                        reason[reason == ""] = "ATAC_INSUFFICIENT_OUTER_TRAIN_PATIENTS"
                    else:
                        z, pathway_z = _standardised_matrices(
                            accessibility, pathway_weight, outer_train
                        )
                        for start in range(0, len(candidates), settings.prediction_pair_batch_size):
                            stop = min(
                                start + settings.prediction_pair_batch_size,
                                len(candidates),
                            )
                            features, structural = _pair_features(
                                candidates,
                                mapping,
                                z,
                                pathway_z,
                                outer_train,
                                start=start,
                                stop=stop,
                            )
                            local_position = np.flatnonzero(structural)
                            if len(local_position):
                                probability[start + local_position] = _predict_logistic(
                                    models[fold], features[local_position]
                                )
                        reason[np.isfinite(probability)] = ""
                        reason[(~np.isfinite(probability)) & (reason == "")] = (
                            "ATAC_NO_FOLD_OOF_PREDICTION"
                        )
                    frame = _fold_frame(
                        candidates,
                        fold,
                        probability,
                        reason,
                        int(outer_train.sum()),
                        int(outer_test.sum()),
                    )
                    table = _arrow_table(frame)
                    if fold not in fold_writers:
                        fold_writers[fold] = pq.ParquetWriter(
                            fold_paths[fold], table.schema, compression="zstd"
                        )
                    fold_writers[fold].write_table(table)
                    fold_probabilities.append(probability)
                    fold_reasons.append(reason)
                    fold_train_counts.append(int(outer_train.sum()))
                    fold_test_counts.append(int(outer_test.sum()))

            matrix = np.vstack(fold_probabilities)
            count = np.isfinite(matrix).sum(axis=0).astype(np.int8)
            probability = np.divide(
                np.nansum(matrix, axis=0),
                count,
                out=np.full(len(candidates), np.nan, dtype=np.float32),
                where=count > 0,
            ).astype(np.float32)
            reason = np.asarray(structural_reason, dtype=object).copy()
            reason[(count == 0) & (reason == "")] = "ATAC_NO_OOF_PREDICTION"
            reason[count > 0] = None
            available = count > 0
            typed = candidates.assign(
                atac_context_probability=pd.Series(probability, dtype="float32"),
                atac_available=pd.Series(available, dtype=bool),
                atac_unavailable_reason=pd.Series(reason, dtype="string"),
                atac_patient_folds_with_prediction=pd.Series(count, dtype="int8"),
                analysis_version=ANALYSIS_VERSION,
                training_run_id=str(training_run_id),
                module_id=MODULE_ID,
                target_level="cancer_x_lncrna_x_exact_pathway_atac_coaccessibility",
                prediction_format=PREDICTION_FORMAT,
                changes_primary_ranking=False,
            )
            table = _arrow_table(typed)
            if typed_writer is None:
                typed_writer = pq.ParquetWriter(typed_path, table.schema, compression="zstd")
            typed_writer.write_table(table)
            if available.any():
                available_cancers.append(cancer)
            cancer_audit.append(
                {
                    "cancer_id": cancer,
                    "raw_atac_available": cancer not in RAW_GAP_CANCERS,
                    "aligned_patients": aligned_patients,
                    "mapped_candidate_lncrnas": mapped_lncrnas,
                    "covered_exact_pathways": covered_pathways,
                    "available_rows": int(available.sum()),
                    "unavailable_rows": int((~available).sum()),
                    "fold_train_patient_counts": fold_train_counts,
                    "fold_test_patient_counts": fold_test_counts,
                }
            )
    finally:
        for writer in fold_writers.values():
            writer.close()
        if typed_writer is not None:
            typed_writer.close()

    if tuple(sorted(available_cancers)) != tuple(sorted(RAW_COVERED_CANCERS)):
        raise AtacTrainingError(
            "Fresh ATAC predictions were not generated for all 23 raw-covered cancers"
        )
    if not typed_path.is_file() or pq.ParquetFile(typed_path).metadata.num_rows != total_rows:
        raise AtacTrainingError("ATAC typed prediction output did not preserve candidate rows")

    fold_status: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        prediction_path = fold_paths[fold]
        if pq.ParquetFile(prediction_path).metadata.num_rows != total_rows:
            raise AtacTrainingError(f"ATAC fold {fold} did not preserve candidate rows")
        checkpoint = checkpoint_records[fold]
        success = {
            "format": "CC_HHGT_V3_2_ATAC_FOLD_SUCCESS_V1",
            "status": "SUCCESS",
            "analysis_version": ANALYSIS_VERSION,
            "modality": "atac",
            "patient_fold": fold,
            "checkpoint_path": checkpoint["path"],
            "checkpoint_sha256": checkpoint["sha256"],
            "prediction_path": str(prediction_path),
            "prediction_sha256": file_sha256(prediction_path),
            "prediction_rows": int(total_rows),
            "heldout_patients_used_for_fit": False,
            "old_checkpoint_loaded": False,
            "old_predictions_used": False,
        }
        success_path = checkpoint_root / f"atac_patient_fold_{fold}.SUCCESS.json"
        _atomic_json(success_path, success)
        checkpoint["status"] = "SUCCESS"
        checkpoint["success_path"] = str(success_path)
        checkpoint["success_sha256"] = file_sha256(success_path)
        checkpoint["prediction_path"] = str(prediction_path)
        checkpoint["prediction_sha256"] = success["prediction_sha256"]
        fold_status.append(
            {
                "patient_fold": fold,
                "status": "SUCCESS",
                "heldout_patients_used_for_fit": False,
                "checkpoint_sha256": checkpoint["sha256"],
                "prediction_sha256": success["prediction_sha256"],
            }
        )

    config_path = output / "RUN_CONFIG.json"
    _atomic_json(
        config_path,
        {
            "analysis_version": ANALYSIS_VERSION,
            "module_id": MODULE_ID,
            "training_run_id": str(training_run_id),
            "training": asdict(settings),
            "scientific_semantics": (
                "FOLD_LOCAL_PROMOTER_COACCESSIBILITY_PERSISTENCE_PROBABILITY_"
                "NOT_CAUSAL_NOT_DIRECTIONAL"
            ),
            "technical_replicate_policy": materialization.get(
                "technical_replicate_policy"
            ),
            "multiple_aliquot_policy": materialization.get("multiple_aliquot_policy"),
            "missing_policy": "TYPED_NULL_NEVER_ZERO",
            "pathway_policy": "EXACT_PATHWAY_ID_ONLY_NO_FAMILY_BROADCAST",
        },
    )
    checkpoint_manifest_path = output / "CHECKPOINT_MANIFEST.json"
    _atomic_json(
        checkpoint_manifest_path,
        {
            "format": CHECKPOINT_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "folds": N_FOLDS,
            "records": checkpoint_records,
            "all_heads_fresh_initialization": True,
            "old_checkpoints_loaded": False,
        },
    )
    input_audit_path = output / "INPUT_LINEAGE_AUDIT.json"
    _atomic_json(
        input_audit_path,
        {
            "status": "PASS",
            "input_artifacts": [
                {
                    "role": name,
                    "path": str(paths[name]),
                    "sha256": input_hashes[name],
                    "outcome_derived": False,
                    "historical_prediction": False,
                    "historical_checkpoint": False,
                }
                for name in paths
            ],
            "materialization_raw_input_artifacts": materialization.get(
                "input_artifacts", []
            ),
            "forbidden_result_inputs_read": [],
        },
    )
    prediction_sha = file_sha256(typed_path)
    code_paths = [Path(__file__).resolve()]
    runner = Path(__file__).resolve().parents[2] / "scripts" / "run_v32_atac_training.py"
    if runner.is_file():
        code_paths.append(runner)
    code_hashes = {str(path): file_sha256(path) for path in code_paths}
    lineage = {
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "modality": "atac",
        "training_run_id": str(training_run_id),
        "training_status": "SUCCESS",
        "folds": N_FOLDS,
        "patient_level_modality_oof": True,
        "patient_fold_oof_predictions_not_fold_averaged": True,
        "outer_test_patients_used_in_any_upstream_fit": 0,
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "fresh_private_heads_trained_from_scratch": True,
        "fold_status": fold_status,
        "predictions_path": str(typed_path),
        "predictions_sha256": prediction_sha,
        "prediction_rows": int(total_rows),
        "prediction_format": PREDICTION_FORMAT,
        "raw_covered_cancers": list(RAW_COVERED_CANCERS),
        "typed_raw_gap_cancers": list(RAW_GAP_CANCERS),
        "cancers_with_any_prediction": sorted(available_cancers),
        "cancer_audit": cancer_audit,
        "checkpoint_manifest_path": str(checkpoint_manifest_path),
        "checkpoint_manifest_sha256": file_sha256(checkpoint_manifest_path),
        "input_lineage_audit_path": str(input_audit_path),
        "input_lineage_audit_sha256": file_sha256(input_audit_path),
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "code_files": code_hashes,
        "code_sha256": _canonical_json_sha256(code_hashes),
        "exact_pathway_family_broadcast_used": False,
        "missing_atac_assumed_zero": False,
        "formal_positive_contribution_claimed": False,
        "eligible_for_router_fair_compare": True,
        "scientific_semantics": (
            "ATAC_PROMOTER_COACCESSIBILITY_PERSISTENCE_EVIDENCE_NOT_CAUSAL_"
            "NOT_DIRECTIONAL_NOT_PERTURBATIONAL"
        ),
    }
    lineage_path = output / "LINEAGE.json"
    _atomic_json(lineage_path, lineage)
    success = {
        "status": "SUCCESS",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "training_run_id": str(training_run_id),
        "prediction_path": str(typed_path),
        "prediction_sha256": prediction_sha,
        "prediction_rows": int(total_rows),
        "lineage_path": str(lineage_path),
        "lineage_sha256": file_sha256(lineage_path),
        "checkpoint_manifest_path": str(checkpoint_manifest_path),
        "checkpoint_manifest_sha256": lineage["checkpoint_manifest_sha256"],
        "covered_cancers_with_predictions": sorted(available_cancers),
        "typed_raw_gap_cancers": list(RAW_GAP_CANCERS),
        "candidate_universe_preserved": True,
        "old_predictions_or_checkpoints_used": False,
        "missing_is_typed_null_never_zero": True,
        "success_written_last": True,
        "release_ready": False,
        "release_reason": "REQUIRES_FAIR_ROUTER_VS_HIERARCHICAL_VALIDATION",
    }
    _atomic_json(output / "SUCCESS.json", success)
    return success


__all__ = [
    "ANALYSIS_VERSION",
    "AtacTrainingConfig",
    "AtacTrainingError",
    "CHECKPOINT_FORMAT",
    "FORMAL_CANCERS",
    "MATERIALIZATION_FORMAT",
    "RAW_COVERED_CANCERS",
    "RAW_GAP_CANCERS",
    "file_sha256",
    "normalise_membership",
    "normalise_patient_folds",
    "run_atac_training",
]
