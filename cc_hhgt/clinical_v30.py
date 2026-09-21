"""CancerLncAtlas V3.0 clinical and survival utilities.

The clinical extension is deliberately downstream of the V2.9 biological
relationship model. Survival endpoints never enter the discovery branch.
They are used for two independent tasks:

1. patient-level multi-endpoint risk prediction (OS/DSS/PFI/PFS/DFI/DFS), and
2. candidate-level reproducibility of lncRNA, lncRNA-pathway and
   lncRNA-state clinical associations.

All preprocessing is fit on train patients only within the existing patient
fold manifest. Missing endpoints are represented by availability masks rather
than negative labels.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.duration.hazard_regression import PHReg

from .common import first_existing_column, read_table, require_columns, write_json, write_table
from .stats import bh_fdr

ENDPOINTS = ("OS", "DSS", "PFI", "PFS", "DFI", "DFS")

PATIENT_ALIASES = ["patient_id", "case_id", "submitter_id", "bcr_patient_barcode", "participant_id"]
SAMPLE_ALIASES = ["sample_id", "sample_submitter_id", "aliquot_id", "barcode"]
CANCER_ALIASES = ["cancer_id", "project_id", "cohort", "cancer", "type"]


def _normalise_colname(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _find_alias(frame: pd.DataFrame, aliases: Sequence[str]) -> str | None:
    lookup = {_normalise_colname(c): c for c in frame.columns}
    for alias in aliases:
        key = _normalise_colname(alias)
        if key in lookup:
            return lookup[key]
    return None


def tcga_patient_id(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper().startswith("TCGA-") and len(text) >= 12:
        return text[:12]
    return text



def _coalesce_alias_series(frame: pd.DataFrame, aliases: Sequence[str]) -> pd.Series:
    lookup = {_normalise_colname(c): c for c in frame.columns}
    columns = [lookup[_normalise_colname(alias)] for alias in aliases if _normalise_colname(alias) in lookup]
    if not columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    out = frame[columns[0]].copy()
    for column in columns[1:]:
        out = out.where(out.notna(), frame[column])
    return out

def normalise_event(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        return values.where(values.isin([0, 1]))
    mapping = {
        "1": 1,
        "1.0": 1,
        "dead": 1,
        "deceased": 1,
        "death": 1,
        "event": 1,
        "yes": 1,
        "true": 1,
        "progressed": 1,
        "progression": 1,
        "recurred": 1,
        "recurrence": 1,
        "0": 0,
        "0.0": 0,
        "alive": 0,
        "living": 0,
        "censored": 0,
        "no": 0,
        "false": 0,
        "diseasefree": 0,
        "disease_free": 0,
    }
    return series.astype(str).str.strip().str.lower().map(mapping).astype("float64")


def normalise_time(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    values = values.where(values > 0)
    return values


def load_all_existing(paths: Sequence[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        if path is None or not Path(path).exists():
            continue
        path = Path(path)
        if "".join(path.suffixes).lower().endswith(".xlsx"):
            frame = pd.read_excel(path)
        else:
            frame = read_table(path)
        frame["_source_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    # union_by_name semantics
    columns = sorted({column for frame in frames for column in frame.columns})
    return pd.concat([frame.reindex(columns=columns) for frame in frames], ignore_index=True)


def standardise_endpoint_table(
    sources: pd.DataFrame,
    endpoint_aliases: Mapping[str, Mapping[str, Sequence[str]]],
    required_endpoints: Sequence[str] = ("OS",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sources.empty:
        raise RuntimeError("No clinical survival source is available")

    patient_col = _find_alias(sources, PATIENT_ALIASES)
    sample_col = _find_alias(sources, SAMPLE_ALIASES)
    cancer_col = _find_alias(sources, CANCER_ALIASES)
    if patient_col is None and sample_col is None:
        raise ValueError("Clinical source has neither patient nor sample identifier")
    if cancer_col is None:
        raise ValueError("Clinical source has no cancer_id/project/cohort column")

    base = pd.DataFrame(index=sources.index)
    cancer_series = _coalesce_alias_series(sources, CANCER_ALIASES)
    base["cancer_id"] = cancer_series.astype(str).str.replace("TCGA-", "", regex=False)
    patient_series = _coalesce_alias_series(sources, PATIENT_ALIASES)
    if sample_col is not None:
        patient_series = patient_series.where(
            patient_series.notna(), sources[sample_col]
        )
    base["patient_id"] = patient_series.map(tcga_patient_id)
    base["sample_id"] = sources[sample_col].astype(str) if sample_col else pd.NA
    base["source_path"] = sources.get("_source_path", pd.Series("unknown", index=sources.index)).astype(str)

    rows: list[pd.DataFrame] = []
    summary: list[dict[str, Any]] = []
    for endpoint in ENDPOINTS:
        aliases = endpoint_aliases.get(endpoint, {})
        time_alias_list = list(aliases.get("time", []))
        event_alias_list = list(aliases.get("event", []))
        time_col = _find_alias(sources, time_alias_list)
        event_col = _find_alias(sources, event_alias_list)
        time_raw = _coalesce_alias_series(sources, time_alias_list)
        event_raw = _coalesce_alias_series(sources, event_alias_list)

        # Robust OS fallback from standard TCGA fields.
        if endpoint == "OS":
            death = normalise_time(_coalesce_alias_series(sources, ["days_to_death"]))
            follow = normalise_time(_coalesce_alias_series(sources, ["days_to_last_follow_up", "days_to_last_followup"]))
            time = normalise_time(time_raw).fillna(death.fillna(follow))
            vital_raw = _coalesce_alias_series(sources, ["vital_status"])
            event = normalise_event(event_raw).fillna(normalise_event(vital_raw))
        else:
            time = normalise_time(time_raw)
            event = normalise_event(event_raw)

        part = base.copy()
        part["endpoint"] = endpoint
        part["time_days"] = time
        part["event"] = event
        part["endpoint_available"] = (part.time_days.notna() & part.event.notna()).astype("int8")
        part = part.loc[part.patient_id.notna()].copy()
        rows.append(part)
        summary.append({
            "endpoint": endpoint,
            "time_column": time_col,
            "event_column": event_col,
            "n_rows": int(len(part)),
            "n_available": int(part.endpoint_available.sum()),
            "n_events": int(part.loc[part.endpoint_available.eq(1), "event"].sum()),
        })

    endpoints = pd.concat(rows, ignore_index=True)
    endpoints = endpoints.sort_values(["cancer_id", "patient_id", "endpoint", "endpoint_available", "source_path"])
    # Prefer available rows, then de-duplicate across multiple sources.
    endpoints = endpoints.drop_duplicates(["cancer_id", "patient_id", "endpoint"], keep="last").reset_index(drop=True)
    summary_df = pd.DataFrame(summary)
    for endpoint in required_endpoints:
        n = int(endpoints.loc[(endpoints.endpoint.eq(endpoint)) & endpoints.endpoint_available.eq(1)].shape[0])
        if n == 0:
            raise RuntimeError(f"Required endpoint {endpoint} has no valid patients")
    return endpoints, summary_df


def build_clinical_covariates(
    sources: pd.DataFrame,
    endpoint_table: pd.DataFrame,
    covariate_aliases: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    patient_col = _find_alias(sources, PATIENT_ALIASES)
    sample_col = _find_alias(sources, SAMPLE_ALIASES)
    cancer_col = _find_alias(sources, CANCER_ALIASES)
    if patient_col is None and sample_col is None:
        raise ValueError("No patient identifier for clinical covariates")
    if cancer_col is None:
        raise ValueError("No cancer identifier for clinical covariates")
    frame = sources.copy()
    patient_series = _coalesce_alias_series(frame, PATIENT_ALIASES)
    if sample_col is not None:
        patient_series = patient_series.where(
            patient_series.notna(), frame[sample_col]
        )
    frame["patient_id"] = patient_series.map(tcga_patient_id)
    frame["cancer_id"] = _coalesce_alias_series(frame, CANCER_ALIASES).astype(str).str.replace("TCGA-", "", regex=False)
    for canonical, aliases in (covariate_aliases or {}).items():
        frame[canonical] = _coalesce_alias_series(frame, aliases)
    keep = [c for c in frame.columns if c != "_source_path"]
    frame = frame[keep].dropna(subset=["patient_id", "cancer_id"])
    # Keep the most complete row per patient.
    frame["_nonmissing"] = frame.notna().sum(axis=1)
    frame = frame.sort_values("_nonmissing").drop_duplicates(["cancer_id", "patient_id"], keep="last").drop(columns="_nonmissing")
    eligible = endpoint_table[["cancer_id", "patient_id"]].drop_duplicates()
    return eligible.merge(frame, on=["cancer_id", "patient_id"], how="left")


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    risk = np.asarray(risk, float)
    valid = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    time, event, risk = time[valid], event[valid], risk[valid]
    concordant = 0.0
    comparable = 0.0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        mask = time > time[i]
        if not mask.any():
            continue
        comparable += float(mask.sum())
        concordant += float((risk[i] > risk[mask]).sum())
        concordant += 0.5 * float((risk[i] == risk[mask]).sum())
    return concordant / comparable if comparable > 0 else math.nan


def fit_phreg(
    time: np.ndarray,
    event: np.ndarray,
    design: np.ndarray,
    column_names: Sequence[str],
    ties: str = "breslow",
    maxiter: int = 100,
) -> dict[str, Any]:
    valid = np.isfinite(time) & np.isfinite(event) & np.all(np.isfinite(design), axis=1)
    time = np.asarray(time, float)[valid]
    event = np.asarray(event, float)[valid]
    design = np.asarray(design, float)[valid]
    if len(time) < max(20, design.shape[1] + 8) or event.sum() < 5:
        return {"status": "INSUFFICIENT", "n": len(time), "events": int(event.sum())}
    try:
        model = PHReg(time, design, status=event, ties=ties)
        result = model.fit(disp=0, maxiter=maxiter)
        params = np.asarray(result.params, float)
        bse = np.asarray(result.bse, float)
        pvalues = np.asarray(result.pvalues, float)
        return {
            "status": "PASS",
            "n": int(len(time)),
            "events": int(event.sum()),
            "params": dict(zip(column_names, params, strict=True)),
            "bse": dict(zip(column_names, bse, strict=True)),
            "pvalues": dict(zip(column_names, pvalues, strict=True)),
            "risk": design @ params,
        }
    except Exception as exc:
        return {"status": "FAILED", "n": int(len(time)), "events": int(event.sum()), "error": repr(exc)}


def make_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, list(numeric_columns)),
        ("categorical", categorical, list(categorical_columns)),
    ], remainder="drop", verbose_feature_names_out=False)


def clean_na_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Replace pandas extension-NA values that break sklearn imputers."""
    frame = frame.copy()
    for column in frame.columns:
        series = frame[column]
        if isinstance(series.dtype, pd.StringDtype) or series.dtype == object:
            frame[column] = series.astype(object).where(series.notna(), np.nan)
    return frame


def endpoint_time_bins(time: np.ndarray, event: np.ndarray, n_bins: int) -> np.ndarray:
    time = np.asarray(time, float)
    event = np.asarray(event, float)
    observed = time[(event == 1) & np.isfinite(time)]
    if len(observed) < max(5, n_bins):
        observed = time[np.isfinite(time)]
    if len(observed) == 0:
        return np.arange(1, n_bins + 1, dtype=float)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:]
    edges = np.unique(np.quantile(observed, quantiles))
    if len(edges) < n_bins:
        edges = np.linspace(max(float(np.nanmin(observed)), 1.0), float(np.nanmax(observed)) + 1e-6, n_bins)
    return edges.astype("float64")


def time_to_bin(time: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, np.asarray(time, float), side="left"), 0, len(edges) - 1)


def discrete_time_nll(logits, time_bin, event, available):
    """Negative log likelihood for right-censored discrete hazards."""
    import torch

    hazard = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    batch, n_bins = hazard.shape
    index = torch.arange(n_bins, device=hazard.device).unsqueeze(0).expand(batch, -1)
    time_bin = time_bin.long().unsqueeze(1)
    event = event.float().unsqueeze(1)
    available = available.float()
    # Unavailable rows must not propagate NaN through the likelihood.
    time_bin = time_bin.masked_fill(available.unsqueeze(1) == 0, 0)
    event = event.masked_fill(available.unsqueeze(1) == 0, 0)
    before = index < time_bin
    at = index == time_bin
    through = index <= time_bin
    log_surv = torch.log1p(-hazard)
    log_haz = torch.log(hazard)
    event_ll = (log_surv * before).sum(1) + (log_haz * at).sum(1)
    censor_ll = (log_surv * through).sum(1)
    ll = event.squeeze(1) * event_ll + (1 - event.squeeze(1)) * censor_ll
    denom = available.sum().clamp_min(1.0)
    return -(ll * available).sum() / denom


class MultiEndpointSurvivalMLP:
    """Factory wrapper to avoid importing torch at module import time."""

    @staticmethod
    def build(input_dim: int, endpoints: Sequence[str], n_bins: int, hidden_dim: int, dropout: float):
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                self.heads = nn.ModuleDict({endpoint: nn.Linear(hidden_dim, n_bins) for endpoint in endpoints})

            def forward(self, x):
                z = self.encoder(x)
                return {endpoint: head(z) for endpoint, head in self.heads.items()}, z

        return _Model()


def survival_risk_from_logits(logits: np.ndarray) -> np.ndarray:
    hazard = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    return 1.0 - np.prod(1.0 - hazard, axis=1)


def gradient_input_importance(model, x, endpoint: str) -> np.ndarray:
    import torch

    model.eval()
    tensor = torch.tensor(x, dtype=torch.float32, requires_grad=True)
    outputs, _ = model(tensor)
    risk = torch.sigmoid(outputs[endpoint]).sum(dim=1).mean()
    risk.backward()
    return np.mean(np.abs(tensor.grad.detach().cpu().numpy() * x), axis=0)


def clinical_replication_label(test_beta: float, test_p: float, test_cindex: float, train_beta: float, settings: Mapping[str, Any]) -> float:
    valid = all(np.isfinite(x) for x in [test_beta, test_p, train_beta])
    if not valid:
        return math.nan
    direction = np.sign(test_beta) == np.sign(train_beta) and np.sign(test_beta) != 0
    beta_ok = abs(test_beta) >= float(settings.get("replication_min_abs_beta", 0.10))
    p_ok = test_p <= float(settings.get("replication_max_test_p", 0.20))
    c_ok = (not np.isfinite(test_cindex)) or test_cindex >= float(settings.get("replication_min_test_cindex", 0.52))
    return float(direction and beta_ok and p_ok and c_ok)


class ClinicalReplicationMLP:
    @staticmethod
    def build(input_dim: int, hidden_dim: int = 96, dropout: float = 0.15):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )


def random_effects_log_hazard(beta: np.ndarray, se: np.ndarray) -> dict[str, float]:
    beta = np.asarray(beta, float)
    se = np.asarray(se, float)
    valid = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    beta, se = beta[valid], se[valid]
    if len(beta) == 0:
        return {"beta": math.nan, "se": math.nan, "p_value": math.nan, "ci_lower": math.nan, "ci_upper": math.nan, "tau2": math.nan, "i2": math.nan, "k": 0}
    var = se**2
    w_fixed = 1.0 / var
    mean_fixed = np.sum(w_fixed * beta) / np.sum(w_fixed)
    q = float(np.sum(w_fixed * (beta - mean_fixed) ** 2))
    c = float(np.sum(w_fixed) - np.sum(w_fixed**2) / np.sum(w_fixed))
    tau2 = max((q - (len(beta) - 1)) / c, 0.0) if c > 0 and len(beta) > 1 else 0.0
    w = 1.0 / (var + tau2)
    mean = float(np.sum(w * beta) / np.sum(w))
    meta_se = float(np.sqrt(1.0 / np.sum(w)))
    z = mean / meta_se if meta_se > 0 else math.nan
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else math.nan
    i2 = max((q - (len(beta) - 1)) / q, 0.0) * 100 if q > 0 and len(beta) > 1 else 0.0
    return {
        "beta": mean,
        "se": meta_se,
        "p_value": p,
        "ci_lower": mean - 1.96 * meta_se,
        "ci_upper": mean + 1.96 * meta_se,
        "tau2": tau2,
        "i2": i2,
        "k": int(len(beta)),
    }


def geometric_priority(functional: pd.Series, clinical: pd.Series) -> pd.Series:
    f = pd.to_numeric(functional, errors="coerce")
    c = pd.to_numeric(clinical, errors="coerce")
    out = pd.Series(np.nan, index=f.index, dtype="float64")
    both = f.notna() & c.notna()
    out.loc[both] = np.sqrt(np.clip(f.loc[both], 0, 1) * np.clip(c.loc[both], 0, 1))
    out.loc[f.notna() & c.isna()] = f.loc[f.notna() & c.isna()]
    out.loc[f.isna() & c.notna()] = c.loc[f.isna() & c.notna()]
    return out
