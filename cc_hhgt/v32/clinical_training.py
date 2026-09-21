"""V3.2 pooled clinical head trained from raw TCGA-CDR outcomes.

This module defines the data and loss contract only.  It never loads a V3.0
checkpoint, risk score, OOF prediction, or ranking.  Every fold receives a
frozen V3.2 core representation and a newly initialised private survival head.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .integrated_model import build_private_auxiliary_head


CLINICAL_ENDPOINTS = ("OS", "DSS", "PFI", "PFS", "DFI", "DFS")
CLINICAL_CHECKPOINT_FORMAT = "CC_HHGT_V3_2_CLINICAL_PRIVATE_HEAD_V1"


class ClinicalTrainingError(RuntimeError):
    """Raised when a clinical fold violates the V3.2 training contract."""


def _explicit_patient_series(value: pd.Series, *, context: str) -> pd.Series:
    result = value.astype("string")
    if result.isna().any():
        raise ClinicalTrainingError(f"{context} contains null explicit patient_id")
    result = result.str.strip()
    if result.eq("").any():
        raise ClinicalTrainingError(f"{context} contains empty explicit patient_id")
    return result.astype(str)


def _valid_event(value: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce")
    return numeric.where(numeric.isin([0.0, 1.0])).astype("float64")


def _valid_time(value: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(value, errors="coerce").astype("float64")
    return numeric.where(numeric > 0)


def standardize_tcga_cdr_workbook(
    workbook: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build all six endpoint rows without silently equating DFI and DFS."""

    source = Path(workbook)
    if not source.is_file():
        raise ClinicalTrainingError(f"TCGA-CDR workbook is missing: {source}")
    main = pd.read_excel(source, sheet_name="TCGA-CDR")
    extra = pd.read_excel(source, sheet_name="ExtraEndpoints")
    required = {"bcr_patient_barcode", "type", "OS", "OS.time"}
    if missing := sorted(required - set(main.columns)):
        raise ClinicalTrainingError(f"TCGA-CDR main sheet lacks columns: {missing}")
    main = main.rename(columns={"bcr_patient_barcode": "patient_id", "type": "cancer_id"})
    extra = extra.rename(columns={"bcr_patient_barcode": "patient_id", "type": "cancer_id"})
    main["patient_id"] = _explicit_patient_series(
        main.patient_id, context="TCGA-CDR main sheet"
    )
    extra["patient_id"] = _explicit_patient_series(
        extra.patient_id, context="TCGA-CDR ExtraEndpoints sheet"
    )
    base = main[["cancer_id", "patient_id"]].drop_duplicates().copy()
    base["cancer_id"] = base.cancer_id.astype(str).str.replace("TCGA-", "", regex=False)

    sources: dict[str, tuple[pd.DataFrame, str | None, str | None, str]] = {
        "OS": (main, "OS.time", "OS", "TCGA-CDR"),
        "DSS": (main, "DSS.time", "DSS", "TCGA-CDR"),
        "PFI": (main, "PFI.time", "PFI", "TCGA-CDR"),
        "PFS": (extra, "PFS.time", "PFS", "ExtraEndpoints"),
        "DFI": (main, "DFI.time", "DFI", "TCGA-CDR"),
        # TCGA-CDR contains DFI but no scientifically distinct DFS column.
        "DFS": (main, None, None, "NO_DISTINCT_DFS_SOURCE"),
    }
    rows: list[pd.DataFrame] = []
    for endpoint in CLINICAL_ENDPOINTS:
        table, time_column, event_column, source_sheet = sources[endpoint]
        part = base.copy()
        if time_column is None or event_column is None:
            part["time_days"] = np.nan
            part["event"] = np.nan
            part["endpoint_available"] = False
            part["failure_reason"] = source_sheet
        else:
            if time_column not in table or event_column not in table:
                raise ClinicalTrainingError(
                    f"Endpoint {endpoint} source lacks {time_column}/{event_column}"
                )
            values = table[["cancer_id", "patient_id", time_column, event_column]].copy()
            values["cancer_id"] = values.cancer_id.astype(str).str.replace(
                "TCGA-", "", regex=False
            )
            values["patient_id"] = _explicit_patient_series(
                values.patient_id, context=f"TCGA-CDR {endpoint} endpoint"
            )
            values = values.drop_duplicates(["cancer_id", "patient_id"], keep="first")
            values["time_days"] = _valid_time(values[time_column])
            values["event"] = _valid_event(values[event_column])
            part = part.merge(
                values[["cancer_id", "patient_id", "time_days", "event"]],
                on=["cancer_id", "patient_id"],
                how="left",
                validate="one_to_one",
            )
            part["endpoint_available"] = part.time_days.notna() & part.event.notna()
            part["failure_reason"] = np.where(
                part.endpoint_available, "", "ENDPOINT_NOT_RECORDED_FOR_PATIENT"
            )
        part["clinical_endpoint"] = endpoint
        part["source_sheet"] = source_sheet
        rows.append(part)
    endpoints = pd.concat(rows, ignore_index=True)
    endpoints = endpoints.sort_values(
        ["cancer_id", "patient_id", "clinical_endpoint"], kind="stable"
    ).reset_index(drop=True)

    covariates = main[["cancer_id", "patient_id"]].copy()
    aliases = {
        "age_years": "age_at_initial_pathologic_diagnosis",
        "sex": "gender",
        "pathologic_stage": "ajcc_pathologic_tumor_stage",
        "clinical_stage": "clinical_stage",
    }
    for target, column in aliases.items():
        covariates[target] = main[column] if column in main else pd.NA
    covariates["cancer_id"] = covariates.cancer_id.astype(str).str.replace(
        "TCGA-", "", regex=False
    )
    covariates["patient_id"] = _explicit_patient_series(
        covariates.patient_id, context="TCGA-CDR covariates"
    )
    covariates = covariates.drop_duplicates(["cancer_id", "patient_id"], keep="first")
    return endpoints, covariates.reset_index(drop=True)


def patient_fold_manifest(sample_folds: pd.DataFrame) -> pd.DataFrame:
    required = {"cancer_id", "sample_id", "patient_id", "patient_fold_id"}
    if missing := sorted(required - set(sample_folds)):
        raise ClinicalTrainingError(f"Patient fold manifest lacks columns: {missing}")
    folds = sample_folds[list(required)].copy()
    if folds[list(required)].isna().any().any():
        raise ClinicalTrainingError("Patient fold manifest contains null contract fields")
    for column in ("cancer_id", "sample_id", "patient_id"):
        folds[column] = folds[column].astype(str).str.strip()
        if folds[column].eq("").any():
            raise ClinicalTrainingError(f"Patient fold manifest contains empty {column}")
    sample_mapping = folds.groupby(["cancer_id", "sample_id"], observed=True).patient_id.nunique()
    if sample_mapping.gt(1).any():
        raise ClinicalTrainingError("A sample maps to multiple explicit patients")
    conflicts = folds.groupby(["cancer_id", "patient_id"], observed=True).patient_fold_id.nunique()
    if (conflicts > 1).any():
        raise ClinicalTrainingError("A patient appears in multiple V3.2 folds")
    folds = folds.drop_duplicates(["cancer_id", "patient_id"])
    folds["patient_fold_id"] = pd.to_numeric(
        folds.patient_fold_id, errors="raise"
    ).astype("int8")
    if not folds.patient_fold_id.between(0, 4).all():
        raise ClinicalTrainingError("patient_fold_id must be in 0..4")
    return folds[["cancer_id", "patient_id", "patient_fold_id"]].reset_index(drop=True)


def core_feature_columns(frame: pd.DataFrame) -> list[str]:
    columns = sorted(column for column in frame if str(column).startswith("core_feature_"))
    if not columns:
        raise ClinicalTrainingError("V3.2 core embedding has no core_feature columns")
    return columns


def project_measurements_to_core(
    measurements: pd.DataFrame,
    embeddings: pd.DataFrame,
    *,
    entity_column: str,
    value_column: str,
    train_patient_ids: Sequence[str],
    output_prefix: str,
) -> pd.DataFrame:
    """Train-fold standardize a molecular matrix and project onto frozen core."""

    required = {"patient_id", entity_column, value_column}
    if missing := sorted(required - set(measurements)):
        raise ClinicalTrainingError(f"Molecular measurements lack columns: {missing}")
    features = core_feature_columns(embeddings)
    embedding = embeddings.set_index("node_id")
    common = sorted(set(measurements[entity_column].astype(str)).intersection(embedding.index.astype(str)))
    if not common:
        raise ClinicalTrainingError(f"No {entity_column} IDs align to the V3.2 core")
    subset = measurements.loc[measurements[entity_column].astype(str).isin(common)].copy()
    matrix = subset.pivot_table(
        index="patient_id", columns=entity_column, values=value_column, aggfunc="mean"
    ).reindex(columns=common)
    values = matrix.to_numpy(np.float32)
    train_mask = matrix.index.astype(str).isin(set(map(str, train_patient_ids)))
    if int(train_mask.sum()) < 2:
        raise ClinicalTrainingError("Fold-local projection has fewer than two training patients")
    train_values = values[train_mask]
    mean = np.nanmean(train_values, axis=0)
    scale = np.nanstd(train_values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    values = np.where(np.isfinite(values), values, mean)
    standardized = (values - mean) / scale
    embedding_values = embedding.loc[common, features].to_numpy(np.float32)
    projected = standardized @ embedding_values / math.sqrt(len(common))
    output = pd.DataFrame(
        projected,
        index=matrix.index,
        columns=[f"{output_prefix}_{index:03d}" for index in range(projected.shape[1])],
    )
    output.index.name = "patient_id"
    return output.reset_index()


def stage_ordinal(value: pd.Series) -> pd.Series:
    text = value.fillna("").astype(str).str.upper()
    result = pd.Series(np.nan, index=value.index, dtype="float64")
    for number, roman in ((4, "IV"), (3, "III"), (2, "II"), (1, "I")):
        result = result.mask(result.isna() & text.str.contains(fr"\b{roman}\b", regex=True), number)
    return result


def encode_fold_covariates(
    covariates: pd.DataFrame, train_patient_ids: Sequence[str]
) -> pd.DataFrame:
    frame = covariates.copy()
    train = frame.patient_id.astype(str).isin(set(map(str, train_patient_ids)))
    age = pd.to_numeric(frame.get("age_years"), errors="coerce")
    age_mean = float(age.loc[train].mean()) if age.loc[train].notna().any() else 0.0
    age_sd = float(age.loc[train].std()) if age.loc[train].notna().sum() > 1 else 1.0
    if not np.isfinite(age_sd) or age_sd <= 1e-6:
        age_sd = 1.0
    stage = stage_ordinal(
        frame.get("pathologic_stage", pd.Series(pd.NA, index=frame.index)).where(
            frame.get("pathologic_stage", pd.Series(pd.NA, index=frame.index)).notna(),
            frame.get("clinical_stage", pd.Series(pd.NA, index=frame.index)),
        )
    )
    stage_mean = float(stage.loc[train].mean()) if stage.loc[train].notna().any() else 0.0
    stage_sd = float(stage.loc[train].std()) if stage.loc[train].notna().sum() > 1 else 1.0
    if not np.isfinite(stage_sd) or stage_sd <= 1e-6:
        stage_sd = 1.0
    sex = frame.get("sex", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    return pd.DataFrame(
        {
            "patient_id": frame.patient_id.astype(str),
            "domain_age": age.fillna(age_mean).sub(age_mean).div(age_sd).astype("float32"),
            "domain_age_missing": age.isna().astype("float32"),
            "domain_stage": stage.fillna(stage_mean).sub(stage_mean).div(stage_sd).astype("float32"),
            "domain_stage_missing": stage.isna().astype("float32"),
            "domain_sex_female": sex.eq("FEMALE").astype("float32"),
            "domain_sex_male": sex.eq("MALE").astype("float32"),
            "domain_sex_unknown": (~sex.isin(["FEMALE", "MALE"])).astype("float32"),
        }
    )


def endpoint_time_bins(time: np.ndarray, event: np.ndarray, n_bins: int) -> np.ndarray:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=float)
    observed = time[(event == 1) & np.isfinite(time)]
    if len(observed) < max(5, n_bins):
        observed = time[np.isfinite(time)]
    if not len(observed):
        raise ClinicalTrainingError("Cannot fit time bins without observed follow-up")
    edges = np.unique(np.quantile(observed, np.linspace(0, 1, n_bins + 1)[1:]))
    if len(edges) < n_bins:
        edges = np.linspace(max(float(np.nanmin(observed)), 1.0), float(np.nanmax(observed)) + 1e-6, n_bins)
    return edges.astype("float64")


def time_to_bin(time: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, np.asarray(time, float), side="left"), 0, len(edges) - 1)


def discrete_time_nll(logits, time_bin, event, available):
    import torch

    hazard = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    index = torch.arange(hazard.shape[1], device=hazard.device).unsqueeze(0)
    bins = time_bin.long().unsqueeze(1)
    mask = available.float()
    bins = bins.masked_fill(mask.unsqueeze(1) == 0, 0)
    event = event.float().masked_fill(mask == 0, 0)
    log_survival = torch.log1p(-hazard)
    event_ll = (log_survival * (index < bins)).sum(1) + (torch.log(hazard) * (index == bins)).sum(1)
    censor_ll = (log_survival * (index <= bins)).sum(1)
    likelihood = event * event_ll + (1 - event) * censor_ll
    return -(likelihood * mask).sum() / mask.sum().clamp_min(1.0)


def survival_risk_probability(logits: np.ndarray) -> np.ndarray:
    hazard = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, float), -30, 30)))
    return 1.0 - np.prod(1.0 - hazard, axis=1)


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    time, event, risk = map(lambda value: np.asarray(value), (time, event, risk))
    valid = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    time, event, risk = time[valid], event[valid], risk[valid]
    comparable = concordant = 0.0
    for index in range(len(time)):
        if event[index] != 1:
            continue
        later = time > time[index]
        comparable += float(later.sum())
        concordant += float((risk[index] > risk[later]).sum())
        concordant += 0.5 * float((risk[index] == risk[later]).sum())
    return concordant / comparable if comparable else math.nan


def build_clinical_head(
    *, core_features: int, domain_features: int, n_bins: int, seed: int
):
    return build_private_auxiliary_head(
        "clinical",
        core_features=core_features,
        domain_features=domain_features,
        output_features=len(CLINICAL_ENDPOINTS) * int(n_bins),
        hidden_features=192,
        dropout=0.20,
        seed=int(seed),
    )


def reshape_endpoint_logits(logits, n_bins: int):
    return logits.reshape(logits.shape[0], len(CLINICAL_ENDPOINTS), int(n_bins))


def train_private_clinical_head(
    core,
    domain,
    endpoint_arrays: Mapping[str, Mapping[str, np.ndarray]],
    *,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    n_bins: int,
    seed: int,
    epochs: int = 100,
    patience: int = 12,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
) -> tuple[Any, dict[str, Any], list[dict[str, float]]]:
    """Train one pooled fold; only the private head has trainable parameters."""

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if not endpoint_arrays:
        raise ClinicalTrainingError("No clinical endpoint meets the fold training threshold")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    core_tensor = torch.as_tensor(core, dtype=torch.float32, device=device)
    domain_tensor = torch.as_tensor(domain, dtype=torch.float32, device=device)
    head, initialization = build_clinical_head(
        core_features=core_tensor.shape[1],
        domain_features=domain_tensor.shape[1],
        n_bins=n_bins,
        seed=seed,
    )
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.as_tensor(train_indices, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )

    def loss_for(indices: np.ndarray):
        index = torch.as_tensor(indices, dtype=torch.long, device=device)
        logits = reshape_endpoint_logits(
            head(core_tensor[index], domain_tensor[index]), n_bins
        )
        losses = []
        for endpoint_index, endpoint in enumerate(CLINICAL_ENDPOINTS):
            if endpoint not in endpoint_arrays:
                continue
            arrays = endpoint_arrays[endpoint]
            losses.append(
                discrete_time_nll(
                    logits[:, endpoint_index],
                    torch.as_tensor(arrays["bin"][indices], dtype=torch.long, device=device),
                    torch.as_tensor(arrays["event"][indices], dtype=torch.float32, device=device),
                    torch.as_tensor(arrays["available"][indices], dtype=torch.float32, device=device),
                )
            )
        if not losses:
            raise ClinicalTrainingError("Selected split has no available clinical loss")
        return torch.stack(losses).mean()

    best_state = None
    best_validation = math.inf
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        head.train()
        batch_losses = []
        for (index,) in loader:
            numpy_index = index.numpy()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(numpy_index)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        head.eval()
        with torch.no_grad():
            validation = float(loss_for(validation_indices).detach().cpu())
        history.append(
            {
                "epoch": float(epoch + 1),
                "training_loss": float(np.mean(batch_losses)),
                "validation_loss": validation,
            }
        )
        if validation < best_validation - 1e-5:
            best_validation = validation
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    if best_state is None:
        raise ClinicalTrainingError("Clinical head produced no valid checkpoint")
    head.load_state_dict(best_state)
    head.eval()
    return head, {**initialization.as_dict(), "best_validation_loss": best_validation}, history


__all__ = [
    "CLINICAL_CHECKPOINT_FORMAT",
    "CLINICAL_ENDPOINTS",
    "ClinicalTrainingError",
    "build_clinical_head",
    "core_feature_columns",
    "discrete_time_nll",
    "encode_fold_covariates",
    "endpoint_time_bins",
    "harrell_c_index",
    "patient_fold_manifest",
    "project_measurements_to_core",
    "reshape_endpoint_logits",
    "standardize_tcga_cdr_workbook",
    "survival_risk_probability",
    "time_to_bin",
    "train_private_clinical_head",
]
