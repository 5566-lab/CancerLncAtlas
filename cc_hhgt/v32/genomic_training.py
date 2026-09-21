"""Fresh V3.2 Mutation/CNV private-head training from non-predictive assets.

The module never reads an earlier model result.  Mutation pathway calls are
rebuilt from patient-gene calls and the current exact-pathway membership.
Missing calls remain missing: only an observed event or an explicit callable
wild-type/neutral record may participate in a label or prediction.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .input_lineage import audit_input_lineage, artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
MODULE_ID = "mutation_cnv"
ASSAY_CALLABLE_SENTINEL = "__MC3_WES_ASSAY_CALLABLE__"
TARGET_LEVEL = "cancer_x_lncrna_x_exact_pathway_genomic_context"
TARGET_KEYS = ("cancer_id", "lncrna_id", "pathway_id")
N_FOLDS = 5
CORE_EXPORT_FORMAT = "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1"
PRIVATE_CHECKPOINT_FORMAT = "CC_HHGT_V3_2_GENOMIC_PRIVATE_HEAD_V1"
PREDICTION_FORMAT = "CC_HHGT_V3_2_GENOMIC_TYPED_PREDICTIONS_V1"

_FORBIDDEN_INPUT_NAME_TOKENS = (
    "adjusted_probability",
    "pair_support",
    "pair_evidence",
    "oof_prediction",
    "prediction",
    "probability",
    "ranking",
    "ranked",
    "checkpoint",
)
_FORBIDDEN_INPUT_COLUMN_TOKENS = (
    "adjusted_probability",
    "pair_support",
    "pair_evidence",
    "oof_prediction",
    "historical_probability",
    "legacy_probability",
    "sample_weight",
)
_DOMAIN_FEATURES = (
    "log_pair_callable",
    "pair_callable_fraction",
    "lncrna_event_fraction",
    "pathway_event_fraction",
    "lncrna_mean_absolute_burden",
    "pathway_mean_absolute_burden",
)


class GenomicTrainingError(RuntimeError):
    """Raised when the fresh-training or missingness contract is violated."""


@dataclass(frozen=True)
class GenomicTrainingConfig:
    seed: int = 20260726
    epochs: int = 40
    batch_size: int = 1024
    hidden_features: int = 64
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    min_pair_callable: int = 12
    max_train_rows: int = 200_000
    max_validation_rows: int = 100_000
    prediction_batch_size: int = 16_384
    cnv_event_threshold: float = 0.30

    def validate(self) -> None:
        integer_positive = (
            self.epochs,
            self.batch_size,
            self.hidden_features,
            self.patience,
            self.min_pair_callable,
            self.max_train_rows,
            self.max_validation_rows,
            self.prediction_batch_size,
        )
        if any(int(value) < 1 for value in integer_positive):
            raise ValueError("Genomic training integer controls must be positive")
        if not 0 <= float(self.dropout) < 1:
            raise ValueError("dropout must be in [0, 1)")
        if float(self.learning_rate) <= 0 or float(self.weight_decay) < 0:
            raise ValueError("learning rate/weight decay are invalid")
        if float(self.cnv_event_threshold) <= 0:
            raise ValueError("CNV event threshold must be positive")


@dataclass(frozen=True)
class EmbeddingLookup:
    ids: tuple[str, ...]
    values: np.ndarray
    index: Mapping[str, int]


@dataclass(frozen=True)
class FoldCoreEmbeddings:
    fold: int
    lncrna: EmbeddingLookup
    pathway: EmbeddingLookup
    checkpoint_sha256: str
    core_parameter_sha256: str
    input_hashes: Mapping[str, str]


@dataclass(frozen=True)
class CandidateStatistics:
    domain: np.ndarray
    labels: np.ndarray
    available: np.ndarray
    reasons: np.ndarray


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return artifact_sha256(path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
    suffixes = [suffix.lower() for suffix in source.suffixes]
    logical = suffixes[-2] if suffixes and suffixes[-1] in {".gz", ".bz2", ".xz"} else source.suffix.lower()
    if source.is_dir() or logical == ".parquet":
        return pd.read_parquet(source, columns=list(columns) if columns else None)
    separator = "\t" if logical in {".tsv", ".txt", ".maf", ".seg"} else ","
    return pd.read_csv(source, sep=separator, usecols=list(columns) if columns else None)


def _normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _patient_id(value: Any) -> str:
    text = str(value).strip().replace(".", "-")
    return text[:12] if text.upper().startswith("TCGA-") and len(text) >= 12 else text


def _node_id(value: Any, kind: str) -> str:
    text = str(value).strip()
    prefixes = ("LNC:", "LNCRNA:") if kind == "lncrna" else ("PATHWAY:", "PW:")
    upper = text.upper()
    for prefix in prefixes:
        if upper.startswith(prefix):
            return text[len(prefix) :]
    return text


def _first_column(frame: pd.DataFrame, candidates: Sequence[str], context: str) -> str:
    lookup = {_normalise_name(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if _normalise_name(candidate) in lookup:
            return lookup[_normalise_name(candidate)]
    raise GenomicTrainingError(f"{context} lacks any of columns {list(candidates)}")


def _bool_series(value: pd.Series, context: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(value):
        return value.fillna(False).astype(bool)
    lowered = value.astype(str).str.strip().str.lower()
    allowed = {"true", "false", "1", "0", "yes", "no"}
    observed = set(lowered.loc[value.notna()].unique())
    if not observed.issubset(allowed):
        raise GenomicTrainingError(f"{context} is not an explicit boolean: {sorted(observed)}")
    return lowered.isin({"true", "1", "yes"})


def _assert_nonpredictive(frame: pd.DataFrame, context: str) -> None:
    bad: list[str] = []
    for column in frame.columns:
        token = _normalise_name(column)
        if any(forbidden in token for forbidden in _FORBIDDEN_INPUT_COLUMN_TOKENS):
            bad.append(str(column))
        elif token in {"label", "proxy_label", "prediction", "probability", "ranking", "rank"}:
            bad.append(str(column))
    if bad:
        raise GenomicTrainingError(f"{context} contains forbidden result columns: {sorted(set(bad))}")


def _assert_source_name(path: str | Path, *, core_parent: bool = False) -> None:
    name = _normalise_name(Path(path).name)
    if core_parent:
        return
    found = [token for token in _FORBIDDEN_INPUT_NAME_TOKENS if token in name]
    if found:
        raise GenomicTrainingError(f"Forbidden historical/result-like input path: {path}; tokens={found}")


def normalise_patient_folds(frame: pd.DataFrame) -> pd.DataFrame:
    cancer = _first_column(frame, ("cancer_id",), "patient folds")
    patient = _first_column(frame, ("patient_id",), "patient folds")
    fold = _first_column(frame, ("patient_fold_id", "fold_id", "fold"), "patient folds")
    result = frame[[cancer, patient, fold]].rename(
        columns={cancer: "cancer_id", patient: "patient_id", fold: "patient_fold_id"}
    )
    result["cancer_id"] = result.cancer_id.astype(str).str.upper()
    if result[["cancer_id", "patient_id"]].isna().any().any():
        raise GenomicTrainingError("Patient folds contain null cancer/patient IDs")
    result["patient_id"] = result.patient_id.astype(str).str.strip()
    if result.patient_id.eq("").any():
        raise GenomicTrainingError("Patient folds contain empty patient_id")
    result["patient_fold_id"] = pd.to_numeric(result.patient_fold_id, errors="raise").astype(int)
    result = result.drop_duplicates()
    conflicts = result.groupby(["cancer_id", "patient_id"], observed=True).patient_fold_id.nunique()
    if (conflicts > 1).any():
        raise GenomicTrainingError("A patient appears in multiple V3.2 folds")
    result = result.drop_duplicates(["cancer_id", "patient_id"])
    observed_by_cancer = result.groupby("cancer_id", observed=True).patient_fold_id.agg(
        lambda value: frozenset(map(int, value))
    )
    bad = observed_by_cancer.loc[observed_by_cancer.ne(frozenset(range(N_FOLDS)))]
    if not bad.empty:
        raise GenomicTrainingError(
            "V3.2 genomic training requires folds 0..4 in every cancer: "
            f"{bad.to_dict()}"
        )
    return result.sort_values(["cancer_id", "patient_id"], kind="stable").reset_index(drop=True)


def normalise_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_nonpredictive(frame, "V3.2 candidates")
    missing = sorted(set(TARGET_KEYS) - set(frame.columns))
    if missing:
        raise GenomicTrainingError(f"Candidate table lacks exact-pathway keys: {missing}")
    result = frame.loc[:, list(TARGET_KEYS)].astype(str).drop_duplicates()
    result["cancer_id"] = result.cancer_id.str.upper()
    if result.duplicated(list(TARGET_KEYS)).any() or result.empty:
        raise GenomicTrainingError("Candidate table is empty or duplicated")
    return result.sort_values(list(TARGET_KEYS), kind="stable").reset_index(drop=True)


def normalise_exact_membership(frame: pd.DataFrame) -> pd.DataFrame:
    _assert_nonpredictive(frame, "exact pathway-gene membership")
    if "pathway_id" not in frame or "gene_id" not in frame:
        if "pathway_family_id" in frame:
            raise GenomicTrainingError("Family-level pathway statistics are forbidden; exact pathway_id is required")
        raise GenomicTrainingError("Membership requires pathway_id and gene_id")
    result = frame[["pathway_id", "gene_id"]].dropna().astype(str).drop_duplicates()
    if result.empty:
        raise GenomicTrainingError("Exact pathway membership is empty")
    return result


def _normalise_mutation_calls(frame: pd.DataFrame, entity_kind: str) -> pd.DataFrame:
    _assert_nonpredictive(frame, f"sample-{entity_kind} mutation")
    cancer_col = _first_column(frame, ("cancer_id",), "mutation calls")
    patient_col = _first_column(frame, ("patient_id", "sample_id"), "mutation calls")
    if entity_kind == "gene":
        entity_col = _first_column(frame, ("gene_id", "ensembl_gene_id"), "gene mutation calls")
        event_col = _first_column(frame, ("is_mutated", "gene_any_mutation"), "gene mutation calls")
        availability_candidates = ("mutation_callable", "callable")
        burden_candidates = ("mutation_count", "n_mutations")
    else:
        entity_col = _first_column(frame, ("lncrna_id", "lncRNA_id"), "lncRNA mutation calls")
        event_col = _first_column(frame, ("lncrna_any_mutation", "is_mutated"), "lncRNA mutation calls")
        availability_candidates = ("lncrna_mutation_available", "mutation_callable", "callable")
        burden_candidates = (
            "exonic_variant_count",
            "mutation_count",
            "length_normalized_mutation_burden",
        )
    lookup = {_normalise_name(column): str(column) for column in frame.columns}
    availability_col = next(
        (lookup[_normalise_name(candidate)] for candidate in availability_candidates if _normalise_name(candidate) in lookup),
        None,
    )
    absence_col = lookup.get("absence_is_wildtype")
    burden_col = next(
        (lookup[_normalise_name(candidate)] for candidate in burden_candidates if _normalise_name(candidate) in lookup),
        None,
    )
    local = pd.DataFrame(
        {
            "cancer_id": frame[cancer_col].astype(str).str.upper(),
            "patient_id": frame[patient_col].map(_patient_id),
            "entity_id": frame[entity_col].astype(str),
            "event": _bool_series(frame[event_col], f"{entity_kind} mutation event"),
        }
    )
    explicit_callable = (
        _bool_series(frame[availability_col], f"{entity_kind} mutation callability")
        if availability_col
        else pd.Series(False, index=frame.index)
    )
    explicit_wildtype = (
        _bool_series(frame[absence_col], f"{entity_kind} absence_is_wildtype")
        if absence_col
        else pd.Series(False, index=frame.index)
    )
    # An observed event is callable.  A non-event is callable only when the
    # source explicitly supplied callability or wild-type status.
    local["callable"] = local.event | explicit_callable | explicit_wildtype
    if burden_col:
        local["burden"] = pd.to_numeric(frame[burden_col], errors="coerce").abs()
    else:
        local["burden"] = local.event.astype(float)
    local.loc[~local.callable, ["event", "burden"]] = [False, np.nan]
    grouped = local.groupby(
        ["cancer_id", "patient_id", "entity_id"], observed=True, sort=False
    ).agg(event=("event", "max"), callable=("callable", "max"), burden=("burden", "sum"))
    result = grouped.reset_index()
    result.loc[~result.callable, "burden"] = np.nan
    return result


def normalise_gene_mutation_calls(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalise_mutation_calls(frame, "gene")


def normalise_lncrna_mutation_calls(frame: pd.DataFrame) -> pd.DataFrame:
    return _normalise_mutation_calls(frame, "lncrna")


def build_exact_pathway_calls(
    gene_calls: pd.DataFrame,
    exact_membership: pd.DataFrame,
) -> pd.DataFrame:
    """Rebuild patient exact-pathway calls without family-level statistics."""

    required = {"cancer_id", "patient_id", "entity_id", "event", "callable", "burden"}
    if missing := sorted(required - set(gene_calls.columns)):
        raise GenomicTrainingError(f"Gene call table lacks {missing}")
    from scipy import sparse

    membership = normalise_exact_membership(exact_membership)
    member_totals = membership.groupby("pathway_id", observed=True).gene_id.nunique()
    calls = gene_calls.copy()
    assay_callable = calls.loc[
        calls.entity_id.astype(str).eq(ASSAY_CALLABLE_SENTINEL)
        & calls.callable.astype(bool),
        ["cancer_id", "patient_id"],
    ].drop_duplicates()
    calls["event"] = calls.event.astype(bool) & calls.callable.astype(bool)
    calls["burden"] = pd.to_numeric(calls.burden, errors="coerce")
    calls = calls.groupby(
        ["cancer_id", "patient_id", "entity_id"], observed=True, sort=False
    ).agg(
        event=("event", "max"),
        callable=("callable", "max"),
        burden=("burden", "sum"),
    ).reset_index()
    result_parts: list[pd.DataFrame] = []
    for cancer, local_calls in calls.groupby("cancer_id", observed=True, sort=False):
        local_membership = membership.loc[
            membership.gene_id.astype(str).isin(local_calls.entity_id.astype(str).unique())
        ]
        if local_membership.empty:
            continue
        patients = tuple(local_calls.patient_id.astype(str).unique())
        genes = tuple(local_membership.gene_id.astype(str).unique())
        pathways = tuple(local_membership.pathway_id.astype(str).unique())
        patient_index = {value: index for index, value in enumerate(patients)}
        gene_index = {value: index for index, value in enumerate(genes)}
        pathway_index = {value: index for index, value in enumerate(pathways)}
        local_calls = local_calls.loc[local_calls.entity_id.astype(str).isin(gene_index)].copy()
        call_rows = local_calls.patient_id.astype(str).map(patient_index).to_numpy(int)
        call_columns = local_calls.entity_id.astype(str).map(gene_index).to_numpy(int)
        shape = (len(patients), len(genes))
        observed_matrix = sparse.csr_matrix(
            (np.ones(len(local_calls), dtype=np.float32), (call_rows, call_columns)), shape=shape
        )
        callable_matrix = sparse.csr_matrix(
            (local_calls.callable.astype(np.float32), (call_rows, call_columns)), shape=shape
        )
        event_matrix = sparse.csr_matrix(
            (local_calls.event.astype(np.float32), (call_rows, call_columns)), shape=shape
        )
        burden_matrix = sparse.csr_matrix(
            (
                local_calls.burden.fillna(0.0).to_numpy(np.float32),
                (call_rows, call_columns),
            ),
            shape=shape,
        )
        membership_rows = local_membership.gene_id.astype(str).map(gene_index).to_numpy(int)
        membership_columns = local_membership.pathway_id.astype(str).map(pathway_index).to_numpy(int)
        member_matrix = sparse.csr_matrix(
            (
                np.ones(len(local_membership), dtype=np.float32),
                (membership_rows, membership_columns),
            ),
            shape=(len(genes), len(pathways)),
        )
        observed_count = (observed_matrix @ member_matrix).tocsr()
        callable_count = (callable_matrix @ member_matrix).tocsr()
        event_count = (event_matrix @ member_matrix).tocsr()
        burden_sum = (burden_matrix @ member_matrix).tocsr()
        row_index, column_index = observed_count.nonzero()
        if not len(row_index):
            continue
        observed_events = np.asarray(event_count[row_index, column_index]).ravel() > 0
        callable_genes = np.asarray(callable_count[row_index, column_index]).ravel()
        pathway_member_count = np.asarray(
            [int(member_totals[pathways[index]]) for index in column_index], dtype=int
        )
        callable_pathway = observed_events | np.isclose(callable_genes, pathway_member_count)
        burden = np.asarray(burden_sum[row_index, column_index]).ravel().astype(float)
        burden[~callable_pathway] = np.nan
        result_parts.append(
            pd.DataFrame(
                {
                    "cancer_id": str(cancer),
                    "patient_id": [patients[index] for index in row_index],
                    "entity_id": [pathways[index] for index in column_index],
                    "event": observed_events,
                    "callable": callable_pathway,
                    "burden": burden,
                }
            )
        )
    if assay_callable.empty is False:
        sentinel = assay_callable.assign(
            entity_id=ASSAY_CALLABLE_SENTINEL,
            event=False,
            callable=True,
            burden=0.0,
        )
        result_parts.append(sentinel)
    if not result_parts:
        return pd.DataFrame(
            columns=["cancer_id", "patient_id", "entity_id", "event", "callable", "burden"]
        )
    return pd.concat(result_parts, ignore_index=True)


def normalise_cnv_calls(
    frame: pd.DataFrame,
    *,
    entity_kind: str,
    event_threshold: float,
) -> pd.DataFrame:
    """Normalize explicit long-format GISTIC/segment-derived CNV calls."""

    _assert_nonpredictive(frame, f"{entity_kind} CNV calls")
    cancer_col = _first_column(frame, ("cancer_id",), "CNV calls")
    patient_col = _first_column(frame, ("patient_id", "sample_id"), "CNV calls")
    entity_candidates = ("gene_id", "entity_id") if entity_kind == "gene" else ("lncrna_id", "entity_id")
    entity_col = _first_column(frame, entity_candidates, f"{entity_kind} CNV calls")
    value_col = _first_column(
        frame,
        ("cnv_value", "segment_mean", "Segment_Mean", "gistic_call", "copy_number_value"),
        "CNV calls",
    )
    lookup = {_normalise_name(column): str(column) for column in frame.columns}
    callable_col = next(
        (lookup[key] for key in ("cnv_callable", "callable") if key in lookup),
        None,
    )
    value = pd.to_numeric(frame[value_col], errors="coerce")
    callable_value = _bool_series(frame[callable_col], "CNV callability") if callable_col else value.notna()
    local = pd.DataFrame(
        {
            "cancer_id": frame[cancer_col].astype(str).str.upper(),
            "patient_id": frame[patient_col].map(_patient_id),
            "entity_id": frame[entity_col].astype(str),
            "callable": callable_value & value.notna(),
            "event": value.abs().ge(float(event_threshold)) & callable_value,
            "burden": value.abs(),
        }
    )
    local.loc[~local.callable, ["event", "burden"]] = [False, np.nan]
    result = local.groupby(
        ["cancer_id", "patient_id", "entity_id"], observed=True, sort=False
    ).agg(event=("event", "max"), callable=("callable", "max"), burden=("burden", "mean")).reset_index()
    result.loc[~result.callable, "burden"] = np.nan
    return result


def _normalise_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    kind = _first_column(frame, ("entity_type", "feature_type"), "CNV entity intervals")
    entity = _first_column(frame, ("entity_id", "gene_id", "lncrna_id"), "CNV entity intervals")
    chromosome = _first_column(frame, ("chromosome", "chrom", "seqname"), "CNV entity intervals")
    start = _first_column(frame, ("start", "start_position"), "CNV entity intervals")
    end = _first_column(frame, ("end", "end_position"), "CNV entity intervals")
    result = frame[[kind, entity, chromosome, start, end]].rename(
        columns={kind: "entity_type", entity: "entity_id", chromosome: "chromosome", start: "start", end: "end"}
    )
    result["entity_type"] = result.entity_type.astype(str).str.lower().replace({"lncrna": "lncrna", "lncrna_gene": "lncrna", "protein_coding": "gene"})
    result = result.loc[result.entity_type.isin(["gene", "lncrna"])].copy()
    result["chromosome"] = result.chromosome.astype(str).str.replace(r"^chr", "", regex=True)
    result["start"] = pd.to_numeric(result.start, errors="raise").astype(np.int64)
    result["end"] = pd.to_numeric(result.end, errors="raise").astype(np.int64)
    if (result.end < result.start).any():
        raise GenomicTrainingError("CNV entity interval has end < start")
    return result.drop_duplicates(["entity_type", "entity_id", "chromosome", "start", "end"])


def segment_calls_from_raw(
    segment_root: str | Path,
    intervals: pd.DataFrame,
    patient_folds: pd.DataFrame,
    *,
    manifest_tsv: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Map raw GDC segments to gene/lncRNA midpoints without zero filling."""

    root = Path(segment_root)
    if not root.is_dir():
        raise GenomicTrainingError(f"CNV segment root is missing: {root}")
    iv = _normalise_intervals(intervals)
    allowed = set(
        zip(patient_folds.cancer_id.astype(str), patient_folds.patient_id.astype(str))
    )
    records: list[pd.DataFrame] = []
    unmatched_files = 0
    if manifest_tsv is None:
        files = sorted(root.rglob("*.seg*.txt"))
    else:
        import csv
        from .gdc_segment_cnv import selected_segment_target

        with Path(manifest_tsv).open("r", encoding="utf-8", newline="") as handle:
            selected = [
                dict(row)
                for row in csv.DictReader(handle, delimiter="\t")
                if str(row.get("selected_for_patient", "")).lower() in {"true", "1"}
            ]
        files = [selected_segment_target(row, root) for row in selected]
        if any(not path.is_file() for path in files):
            raise GenomicTrainingError("Manifest-bound CNV segment file disappeared after preflight")
    for path in files:
        cancer = path.parent.name.upper()
        patient = _patient_id(path.name.split("__", 1)[0])
        if (cancer, patient) not in allowed:
            unmatched_files += 1
            continue
        segment = pd.read_csv(path, sep="\t")
        chrom_col = _first_column(segment, ("Chromosome", "chromosome", "chrom"), "CNV segment")
        start_col = _first_column(segment, ("Start", "start"), "CNV segment")
        end_col = _first_column(segment, ("End", "end"), "CNV segment")
        value_col = _first_column(segment, ("Segment_Mean", "segment_mean"), "CNV segment")
        segment = segment[[chrom_col, start_col, end_col, value_col]].copy()
        segment.columns = ["chromosome", "start", "end", "cnv_value"]
        segment["chromosome"] = segment.chromosome.astype(str).str.replace(r"^chr", "", regex=True)
        for chromosome, local_iv in iv.groupby("chromosome", observed=True, sort=False):
            local_seg = segment.loc[segment.chromosome.eq(chromosome)].sort_values("start")
            if local_seg.empty:
                continue
            starts = pd.to_numeric(local_seg.start, errors="raise").to_numpy(np.int64)
            ends = pd.to_numeric(local_seg.end, errors="raise").to_numpy(np.int64)
            values = pd.to_numeric(local_seg.cnv_value, errors="coerce").to_numpy(float)
            midpoint = ((local_iv.start.to_numpy(np.int64) + local_iv.end.to_numpy(np.int64)) // 2)
            position = np.searchsorted(starts, midpoint, side="right") - 1
            valid = (position >= 0) & (midpoint <= ends[np.maximum(position, 0)])
            if not valid.any():
                continue
            mapped = local_iv.loc[valid, ["entity_type", "entity_id"]].copy()
            mapped["cancer_id"] = cancer
            mapped["patient_id"] = patient
            mapped["cnv_value"] = values[position[valid]]
            mapped["cnv_callable"] = np.isfinite(mapped.cnv_value)
            records.append(mapped)
    combined = pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["entity_type", "entity_id", "cancer_id", "patient_id", "cnv_value", "cnv_callable"]
    )
    gene = combined.loc[combined.entity_type.eq("gene")].rename(columns={"entity_id": "gene_id"})
    lnc = combined.loc[combined.entity_type.eq("lncrna")].rename(columns={"entity_id": "lncrna_id"})
    audit = {
        "segment_files": len(files),
        "matched_segment_files": len(files) - unmatched_files,
        "unmatched_segment_files": unmatched_files,
        "manifest_bound_files": manifest_tsv is not None,
        "gene_call_rows": int(len(gene)),
        "lncrna_call_rows": int(len(lnc)),
    }
    return gene, lnc, audit


def _call_matrix(
    calls: pd.DataFrame,
    patients: Sequence[str],
    entities: Sequence[str],
    callable_entities: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    patient_index = {str(value): index for index, value in enumerate(patients)}
    entity_values = tuple(dict.fromkeys(map(str, entities)))
    entity_index = {value: index for index, value in enumerate(entity_values)}
    event = np.full((len(patients), len(entity_values)), np.nan, dtype=np.float32)
    burden = np.full_like(event, np.nan)
    if not calls.empty and entity_values:
        assay_callable = calls.loc[
            calls.patient_id.astype(str).isin(patient_index)
            & calls.entity_id.astype(str).eq(ASSAY_CALLABLE_SENTINEL)
            & calls.callable.astype(bool)
        ]
        for patient_id in assay_callable.patient_id.astype(str).unique():
            position = patient_index[patient_id]
            if callable_entities is None:
                event[position, :] = 0.0
                burden[position, :] = 0.0
            else:
                callable_positions = [
                    entity_index[value]
                    for value in entity_values
                    if value in callable_entities
                ]
                event[position, callable_positions] = 0.0
                burden[position, callable_positions] = 0.0
        local = calls.loc[
            calls.patient_id.astype(str).isin(patient_index)
            & calls.entity_id.astype(str).isin(entity_index)
            & calls.callable.astype(bool)
        ]
        for row in local.itertuples(index=False):
            pi = patient_index[str(row.patient_id)]
            ei = entity_index[str(row.entity_id)]
            event[pi, ei] = float(bool(row.event))
            burden[pi, ei] = float(row.burden) if pd.notna(row.burden) else float(bool(row.event))
    return event, burden, entity_index


def candidate_statistics(
    candidates: pd.DataFrame,
    lncrna_calls: pd.DataFrame,
    pathway_calls: pd.DataFrame,
    patients: Sequence[str],
    *,
    modality: str,
    min_pair_callable: int,
    lncrna_callable_entities: set[str] | None = None,
    pathway_callable_entities: set[str] | None = None,
) -> CandidateStatistics:
    """Build availability-aware candidate labels/features for one cancer/split."""

    size = len(candidates)
    prefix = modality.upper()
    if size == 0:
        return CandidateStatistics(
            np.empty((0, len(_DOMAIN_FEATURES)), np.float32),
            np.empty(0, np.float32),
            np.empty(0, bool),
            np.empty(0, object),
        )
    if hasattr(lncrna_calls, "candidate_statistics_arrays"):
        domain, labels, available, reasons = lncrna_calls.candidate_statistics_arrays(
            candidates, patients, min_pair_callable
        )
        return CandidateStatistics(domain, labels, available, reasons)
    if not patients or lncrna_calls.empty or pathway_calls.empty:
        reason = f"{prefix}_NOT_AVAILABLE_FOR_CANCER"
        return CandidateStatistics(
            np.zeros((size, len(_DOMAIN_FEATURES)), np.float32),
            np.full(size, np.nan, np.float32),
            np.zeros(size, bool),
            np.full(size, reason, object),
        )
    l_event, l_burden, l_index = _call_matrix(
        lncrna_calls,
        patients,
        candidates.lncrna_id.astype(str).unique(),
        lncrna_callable_entities,
    )
    p_event, p_burden, p_index = _call_matrix(
        pathway_calls,
        patients,
        candidates.pathway_id.astype(str).unique(),
        pathway_callable_entities,
    )
    l_positions = np.asarray([l_index.get(str(value), -1) for value in candidates.lncrna_id], dtype=int)
    p_positions = np.asarray([p_index.get(str(value), -1) for value in candidates.pathway_id], dtype=int)
    domain = np.zeros((size, len(_DOMAIN_FEATURES)), dtype=np.float32)
    labels = np.full(size, np.nan, dtype=np.float32)
    available = np.zeros(size, dtype=bool)
    reasons = np.full(size, f"{prefix}_NO_EXPLICIT_PAIR_CALLABILITY", dtype=object)
    chunk_size = 4096
    for start in range(0, size, chunk_size):
        stop = min(start + chunk_size, size)
        li = l_positions[start:stop]
        pi = p_positions[start:stop]
        valid_entity = (li >= 0) & (pi >= 0)
        if not valid_entity.any():
            reasons[start:stop][li < 0] = f"{prefix}_LNCRNA_UNAVAILABLE"
            reasons[start:stop][pi < 0] = f"{prefix}_PATHWAY_UNAVAILABLE"
            continue
        local_columns = np.flatnonzero(valid_entity)
        lx = l_event[:, li[valid_entity]]
        py = p_event[:, pi[valid_entity]]
        lb = l_burden[:, li[valid_entity]]
        pb = p_burden[:, pi[valid_entity]]
        pair = np.isfinite(lx) & np.isfinite(py)
        count = pair.sum(axis=0).astype(float)
        safe_count = np.maximum(count, 1.0)
        x = np.where(pair, lx, 0.0)
        y = np.where(pair, py, 0.0)
        x_sum = x.sum(axis=0)
        y_sum = y.sum(axis=0)
        xy_sum = (x * y).sum(axis=0)
        x_rate = x_sum / safe_count
        y_rate = y_sum / safe_count
        n11 = xy_sum
        n10 = x_sum - n11
        n01 = y_sum - n11
        n00 = count - n11 - n10 - n01
        denominator = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
        phi = np.divide(
            n11 * n00 - n10 * n01,
            denominator,
            out=np.full_like(denominator, np.nan, dtype=float),
            where=denominator > 0,
        )
        l_mean = np.divide(
            np.where(pair, np.abs(lb), 0.0).sum(axis=0),
            safe_count,
            out=np.zeros_like(count),
        )
        p_mean = np.divide(
            np.where(pair, np.abs(pb), 0.0).sum(axis=0),
            safe_count,
            out=np.zeros_like(count),
        )
        rows = start + local_columns
        domain[rows] = np.column_stack(
            [
                np.log1p(count),
                count / max(len(patients), 1),
                x_rate,
                y_rate,
                l_mean,
                p_mean,
            ]
        ).astype(np.float32)
        eligible = (
            (count >= int(min_pair_callable))
            & (x_sum > 0)
            & (x_sum < count)
            & (y_sum > 0)
            & (y_sum < count)
            & np.isfinite(phi)
        )
        labels[rows[eligible]] = (phi[eligible] > 0).astype(np.float32)
        available[rows[eligible]] = True
        reasons[rows[count < int(min_pair_callable)]] = f"{prefix}_INSUFFICIENT_PAIR_CALLABILITY"
        no_variation = (count >= int(min_pair_callable)) & ~eligible
        reasons[rows[no_variation]] = f"{prefix}_NO_EXPLICIT_WT_EVENT_VARIATION"
        reasons[rows[eligible]] = ""
    return CandidateStatistics(domain, labels, available, reasons)


def _embedding_lookup(frame: pd.DataFrame, kind: str) -> EmbeddingLookup:
    if "node_id" not in frame:
        raise GenomicTrainingError(f"{kind} core embedding lacks node_id")
    features = sorted(column for column in frame if str(column).startswith("core_feature_"))
    if not features:
        raise GenomicTrainingError(f"{kind} core embedding lacks core_feature_* columns")
    ids = tuple(frame.node_id.map(lambda value: _node_id(value, kind)).astype(str))
    if len(set(ids)) != len(ids):
        raise GenomicTrainingError(f"{kind} core embedding IDs are duplicated")
    values = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise GenomicTrainingError(f"{kind} core embeddings are not finite")
    return EmbeddingLookup(ids, values, {value: index for index, value in enumerate(ids)})


def _resolve_manifest_path(manifest_path: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() and value.is_file():
        return value
    for ancestor in (manifest_path.parent, *manifest_path.parents):
        candidate = ancestor / value
        if candidate.is_file():
            return candidate.resolve()
    raise GenomicTrainingError(f"Core embedding export cannot be resolved: {relative}")


def _validate_core_manifest(path: str | Path) -> tuple[dict[str, Any], str, str]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("export_format") != CORE_EXPORT_FORMAT:
        raise GenomicTrainingError("Core embeddings are not the registered V3.2 export format")
    if not str(payload.get("analysis_version", "")).startswith("CancerLncAtlas_V3.2"):
        raise GenomicTrainingError("Core embedding parent is not V3.2")
    if payload.get("all_embeddings_from_newly_trained_v32_core") is not True:
        raise GenomicTrainingError("Core embeddings lack newly-trained V3.2 attestation")
    if payload.get("historical_checkpoint_loaded") is not False:
        raise GenomicTrainingError("Core embedding lineage loaded a historical checkpoint")
    if payload.get("historical_prediction_loaded") is not False:
        raise GenomicTrainingError("Core embedding lineage loaded historical predictions")
    folds = payload.get("folds")
    if not isinstance(folds, Mapping) or set(map(str, folds)) != set(map(str, range(N_FOLDS))):
        raise GenomicTrainingError("Core embedding manifest requires exactly five folds")
    parameter_hashes: list[str] = []
    for fold in range(N_FOLDS):
        item = folds[str(fold)]
        if int(item.get("patient_fold", -1)) != fold:
            raise GenomicTrainingError(f"Core embedding fold ID drift: {fold}")
        if item.get("old_checkpoint_loaded") is not False or item.get("trained_from_scratch") is not True:
            raise GenomicTrainingError(f"Fold {fold} core is not newly trained from scratch")
        parameter_hashes.append(str(item.get("core_parameter_sha256", "")))
        if not re.fullmatch(r"[0-9a-f]{64}", parameter_hashes[-1]):
            raise GenomicTrainingError(f"Fold {fold} lacks a core parameter SHA256")
    parameter_composite = _canonical_json_sha256(parameter_hashes)
    return payload, _file_sha256(source), parameter_composite


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
        raise GenomicTrainingError(f"Fold {fold} lacks lncRNA/pathway core exports")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for kind, node_type in (("lncrna", lncrna_node_type), ("pathway", pathway_node_type)):
        declaration = exports[node_type]
        path = _resolve_manifest_path(source, str(declaration["path"]))
        observed = _file_sha256(path)
        if observed != str(declaration.get("sha256", "")):
            raise GenomicTrainingError(f"Fold {fold} {kind} core embedding SHA256 mismatch")
        paths[kind] = path
        hashes[str(path)] = observed
    lnc = _embedding_lookup(pd.read_parquet(paths["lncrna"]), "lncrna")
    pathway = _embedding_lookup(pd.read_parquet(paths["pathway"]), "pathway")
    if lnc.values.shape[1] != pathway.values.shape[1]:
        raise GenomicTrainingError("lncRNA/pathway core embedding widths differ")
    return FoldCoreEmbeddings(
        fold=fold,
        lncrna=lnc,
        pathway=pathway,
        checkpoint_sha256=str(item["checkpoint_sha256"]),
        core_parameter_sha256=str(item["core_parameter_sha256"]),
        input_hashes=hashes,
    )


def _candidate_core(candidates: pd.DataFrame, core: FoldCoreEmbeddings) -> tuple[np.ndarray, np.ndarray]:
    # The frozen graph stores typed node IDs (for example ``LNC:ENSG...``),
    # while the embedding lookup intentionally removes that type prefix.  Apply
    # the identical canonicalization to candidate IDs before joining; otherwise
    # every real V3.2 lncRNA silently misses its same-generation embedding.
    lpos = np.asarray(
        [core.lncrna.index.get(_node_id(value, "lncrna"), -1) for value in candidates.lncrna_id],
        int,
    )
    ppos = np.asarray(
        [core.pathway.index.get(_node_id(value, "pathway"), -1) for value in candidates.pathway_id],
        int,
    )
    available = (lpos >= 0) & (ppos >= 0)
    width = core.lncrna.values.shape[1]
    values = np.zeros((len(candidates), width * 2), dtype=np.float32)
    if available.any():
        values[available, :width] = core.lncrna.values[lpos[available]]
        values[available, width:] = core.pathway.values[ppos[available]]
    return values, available


def _split_patients(folds: pd.DataFrame, fold: int, split: str) -> dict[str, list[str]]:
    validation = (fold + 1) % N_FOLDS
    if split == "test":
        selected = folds.patient_fold_id.eq(fold)
    elif split == "validation":
        selected = folds.patient_fold_id.eq(validation)
    elif split == "train":
        selected = ~folds.patient_fold_id.isin([fold, validation])
    else:
        raise ValueError(split)
    return {
        cancer: group.patient_id.astype(str).tolist()
        for cancer, group in folds.loc[selected].groupby("cancer_id", observed=True, sort=False)
    }


def _group_calls(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(cancer): group.reset_index(drop=True)
        for cancer, group in frame.groupby("cancer_id", observed=True, sort=False)
    }


def normalise_cnv_entity_coverage(frame: pd.DataFrame) -> dict[str, dict[str, set[str]]]:
    """Normalize a sparse-GISTIC entity-callability sidecar."""

    _assert_nonpredictive(frame, "CNV entity callability")
    required = {"cancer_id", "entity_type", "entity_id"}
    if missing := sorted(required - set(frame.columns)):
        raise GenomicTrainingError(f"CNV entity callability lacks columns: {missing}")
    local = frame[["cancer_id", "entity_type", "entity_id"]].dropna().astype(str).drop_duplicates()
    local["cancer_id"] = local.cancer_id.str.upper()
    local["entity_type"] = local.entity_type.str.lower().replace(
        {"lncrna_gene": "lncrna", "protein_coding": "gene"}
    )
    if invalid := sorted(set(local.entity_type) - {"gene", "lncrna"}):
        raise GenomicTrainingError(f"Unknown CNV entity callability types: {invalid}")
    result: dict[str, dict[str, set[str]]] = {"gene": {}, "lncrna": {}}
    for (kind, cancer), group in local.groupby(
        ["entity_type", "cancer_id"], observed=True, sort=False
    ):
        result[str(kind)][str(cancer)] = set(group.entity_id.astype(str))
    return result


def exact_pathway_callable_entities(
    exact_membership: pd.DataFrame,
    gene_callable_by_cancer: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    """Return pathways whose complete exact gene membership is represented."""

    membership = normalise_exact_membership(exact_membership)
    members = {
        str(pathway): set(group.gene_id.astype(str))
        for pathway, group in membership.groupby("pathway_id", observed=True, sort=False)
    }
    return {
        str(cancer): {
            pathway for pathway, genes in members.items() if genes.issubset(callable_genes)
        }
        for cancer, callable_genes in gene_callable_by_cancer.items()
    }


def _balanced_indices(labels: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(labels == value) for value in (0.0, 1.0)]
    if any(len(index) == 0 for index in classes):
        return np.empty(0, dtype=int)
    per_class = max(1, maximum // 2)
    selected = [
        rng.choice(index, size=min(len(index), per_class), replace=False) for index in classes
    ]
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result


def _collect_examples(
    candidates: pd.DataFrame,
    lncrna_by_cancer: Mapping[str, pd.DataFrame],
    pathway_by_cancer: Mapping[str, pd.DataFrame],
    patients_by_cancer: Mapping[str, Sequence[str]],
    core: FoldCoreEmbeddings,
    *,
    modality: str,
    min_pair_callable: int,
    maximum: int,
    seed: int,
    lncrna_callable_by_cancer: Mapping[str, set[str]] | None = None,
    pathway_callable_by_cancer: Mapping[str, set[str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cancers = tuple(candidates.cancer_id.unique())
    quota = max(2, int(math.ceil(maximum / max(len(cancers), 1))))
    core_parts: list[np.ndarray] = []
    domain_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for offset, (cancer, local) in enumerate(candidates.groupby("cancer_id", observed=True, sort=False)):
        stats = candidate_statistics(
            local,
            lncrna_by_cancer.get(str(cancer), pd.DataFrame()),
            pathway_by_cancer.get(str(cancer), pd.DataFrame()),
            patients_by_cancer.get(str(cancer), ()),
            modality=modality,
            min_pair_callable=min_pair_callable,
            lncrna_callable_entities=(
                lncrna_callable_by_cancer.get(str(cancer), set())
                if lncrna_callable_by_cancer is not None else None
            ),
            pathway_callable_entities=(
                pathway_callable_by_cancer.get(str(cancer), set())
                if pathway_callable_by_cancer is not None else None
            ),
        )
        eligible = np.flatnonzero(stats.available)
        if not len(eligible):
            continue
        rng = np.random.default_rng(seed + offset)
        chosen = rng.choice(eligible, size=min(len(eligible), quota), replace=False)
        core_values, core_available = _candidate_core(local.iloc[chosen], core)
        chosen_available = np.flatnonzero(core_available)
        if not len(chosen_available):
            continue
        core_parts.append(core_values[chosen_available])
        domain_parts.append(stats.domain[chosen][chosen_available])
        label_parts.append(stats.labels[chosen][chosen_available])
    if not label_parts:
        width = core.lncrna.values.shape[1] * 2
        return (
            np.empty((0, width), np.float32),
            np.empty((0, len(_DOMAIN_FEATURES)), np.float32),
            np.empty(0, np.float32),
        )
    core_values = np.concatenate(core_parts)
    domain_values = np.concatenate(domain_parts)
    labels = np.concatenate(label_parts)
    choice = _balanced_indices(labels, min(maximum, len(labels)), seed + 100_000)
    if not len(choice):
        width = core.lncrna.values.shape[1] * 2
        return (
            np.empty((0, width), np.float32),
            np.empty((0, len(_DOMAIN_FEATURES)), np.float32),
            np.empty(0, np.float32),
        )
    core_values, domain_values, labels = core_values[choice], domain_values[choice], labels[choice]
    return core_values, domain_values, labels


def _fit_head(
    train_core: np.ndarray,
    train_domain: np.ndarray,
    train_labels: np.ndarray,
    validation_core: np.ndarray,
    validation_domain: np.ndarray,
    validation_labels: np.ndarray,
    *,
    modality: str,
    fold: int,
    config: GenomicTrainingConfig,
) -> tuple[Any, dict[str, Any], np.ndarray, np.ndarray, list[dict[str, float]]]:
    import torch
    from torch.nn import functional as F

    from .integrated_model import build_private_auxiliary_head

    if len(np.unique(train_labels)) != 2:
        raise GenomicTrainingError(f"{modality} fold {fold} lacks two explicit training classes")
    mean = train_domain.mean(axis=0).astype(np.float32)
    scale = train_domain.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    train_scaled = (train_domain - mean) / scale
    validation_scaled = (validation_domain - mean) / scale if len(validation_domain) else validation_domain
    head_seed = int(config.seed + fold * 101 + (0 if modality == "mutation" else 10_000))
    head, initialization = build_private_auxiliary_head(
        f"mutation_cnv_{modality}",
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
            # requires_grad=True makes the structural detach test active even
            # though these are exported, frozen core representations.
            core_tensor = torch.tensor(train_core[index], dtype=torch.float32, requires_grad=True)
            domain_tensor = torch.tensor(train_scaled[index], dtype=torch.float32)
            label_tensor = torch.tensor(train_labels[index], dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = head(core_tensor, domain_tensor).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, label_tensor)
            loss.backward()
            if core_tensor.grad is not None:
                raise GenomicTrainingError("Private genomic head leaked gradients into core embeddings")
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.no_grad():
            if len(validation_labels) and len(np.unique(validation_labels)) == 2:
                val_logits = head(
                    torch.tensor(validation_core, dtype=torch.float32),
                    torch.tensor(validation_scaled, dtype=torch.float32),
                ).squeeze(-1)
                validation_loss = float(
                    F.binary_cross_entropy_with_logits(
                        val_logits, torch.tensor(validation_labels, dtype=torch.float32)
                    )
                )
            else:
                validation_loss = float(np.mean(losses))
        history.append(
            {"epoch": float(epoch), "train_loss": float(np.mean(losses)), "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-7:
            best_loss = validation_loss
            best_state = {name: tensor.detach().clone() for name, tensor in head.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(config.patience):
            break
    if best_state is None:
        raise GenomicTrainingError("Private genomic head produced no checkpoint state")
    head.load_state_dict(best_state, strict=True)
    metadata = initialization.as_dict()
    metadata.update(
        {
            "modality": modality,
            "patient_fold": int(fold),
            "train_rows": int(len(train_labels)),
            "validation_rows": int(len(validation_labels)),
            "best_validation_loss": float(best_loss),
            "core_embedding_detached": True,
            "core_parameters_frozen": True,
        }
    )
    return head, metadata, mean, scale, history


def _predict_head(
    head: Any,
    core_values: np.ndarray,
    domain_values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    import torch

    result = np.empty(len(core_values), dtype=np.float32)
    head.eval()
    with torch.no_grad():
        for start in range(0, len(result), int(batch_size)):
            stop = min(start + int(batch_size), len(result))
            logits = head(
                torch.tensor(core_values[start:stop], dtype=torch.float32),
                torch.tensor((domain_values[start:stop] - mean) / scale, dtype=torch.float32),
            ).squeeze(-1)
            result[start:stop] = torch.sigmoid(logits).cpu().numpy()
    return result


def _audit_sources(
    *,
    candidates_path: Path,
    patient_folds_path: Path,
    membership_path: Path,
    sample_gene_mutation_path: Path,
    sample_lncrna_mutation_path: Path,
    mc3_path: Path,
    cnv_gene_calls_path: Path | None,
    cnv_lncrna_calls_path: Path | None,
    cnv_entity_coverage_path: Path | None,
    cnv_segment_root: Path | None,
    cnv_download_complete_path: Path | None,
    entity_intervals_path: Path | None,
    gistic_source_path: Path | None,
    cnv_streaming_success_path: Path | None,
) -> dict[str, Any]:
    specs: list[dict[str, Any]] = [
        {"path": candidates_path, "generation": "V3.2", "source_role": "standardized_input", "outcome_derived": False, "fold_fitted": False},
        {"path": patient_folds_path, "generation": "V3.2", "source_role": "split_manifest", "outcome_derived": False, "fold_fitted": False, "use_role": "split_control"},
        {"path": membership_path, "generation": "V3.2", "source_role": "static_annotation", "outcome_derived": False, "fold_fitted": False},
        {"path": sample_gene_mutation_path, "generation": "V2.8", "source_role": "standardized_input", "outcome_derived": False, "fold_fitted": False},
        {"path": sample_lncrna_mutation_path, "generation": "V2.8", "source_role": "standardized_input", "outcome_derived": False, "fold_fitted": False},
        {"path": mc3_path, "generation": "MC3-v0.2.8", "source_role": "raw_data", "outcome_derived": False, "fold_fitted": False},
    ]
    for path, role, generation in (
        (cnv_gene_calls_path, "standardized_input", "CNV_RESTANDARDIZED"),
        (cnv_lncrna_calls_path, "standardized_input", "CNV_RESTANDARDIZED"),
        (cnv_entity_coverage_path, "static_annotation", "CNV_EXPLICIT_ENTITY_CALLABILITY"),
        (cnv_segment_root, "raw_data", "GDC_SEGMENT_GRCH38"),
        (cnv_download_complete_path, "standardized_input", "GDC_SEGMENT_DOWNLOAD_GATE"),
        (entity_intervals_path, "static_annotation", "GENCODE_INTERVALS"),
        (gistic_source_path, "raw_data", "GISTIC_RAW"),
        (cnv_streaming_success_path, "standardized_input", "V3.2-CNV_STREAMING_COMPACT_V1"),
    ):
        if path is not None:
            specs.append(
                {"path": path, "generation": generation, "source_role": role, "outcome_derived": False, "fold_fitted": False}
            )
    audit = audit_input_lineage(specs)
    if audit["status"] != "PASS":
        failures = {
            item["artifact_id"]: item["reasons"]
            for item in audit["artifacts"]
            if item["status"] != "PASS"
        }
        raise GenomicTrainingError(f"Genomic source lineage rejected: {failures}")
    return audit


def run_genomic_training(
    *,
    candidates_path: str | Path,
    patient_folds_path: str | Path,
    pathway_gene_membership_path: str | Path,
    sample_gene_mutation_path: str | Path,
    sample_lncrna_mutation_path: str | Path,
    mc3_path: str | Path,
    core_embedding_manifest_path: str | Path,
    output_root: str | Path,
    training_run_id: str,
    cnv_gene_calls_path: str | Path | None = None,
    cnv_lncrna_calls_path: str | Path | None = None,
    cnv_entity_coverage_path: str | Path | None = None,
    cnv_segment_root: str | Path | None = None,
    cnv_download_complete_path: str | Path | None = None,
    entity_intervals_path: str | Path | None = None,
    gistic_source_path: str | Path | None = None,
    cnv_streaming_success_path: str | Path | None = None,
    lncrna_node_type: str = "lncRNA",
    pathway_node_type: str = "pathway",
    config: GenomicTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train five-fold Mutation/CNV heads and emit typed V3.2 predictions."""

    settings = config or GenomicTrainingConfig()
    settings.validate()
    if not training_run_id or not str(training_run_id).startswith("v32-"):
        raise GenomicTrainingError("training_run_id must be a non-empty v32-* identifier")
    paths = {
        "candidates": Path(candidates_path).resolve(),
        "patient_folds": Path(patient_folds_path).resolve(),
        "membership": Path(pathway_gene_membership_path).resolve(),
        "sample_gene_mutation": Path(sample_gene_mutation_path).resolve(),
        "sample_lncrna_mutation": Path(sample_lncrna_mutation_path).resolve(),
        "mc3": Path(mc3_path).resolve(),
        "core_manifest": Path(core_embedding_manifest_path).resolve(),
    }
    optional_paths = {
        "cnv_gene_calls": Path(cnv_gene_calls_path).resolve() if cnv_gene_calls_path else None,
        "cnv_lncrna_calls": Path(cnv_lncrna_calls_path).resolve() if cnv_lncrna_calls_path else None,
        "cnv_entity_coverage": Path(cnv_entity_coverage_path).resolve() if cnv_entity_coverage_path else None,
        "cnv_segment_root": Path(cnv_segment_root).resolve() if cnv_segment_root else None,
        "cnv_download_complete": Path(cnv_download_complete_path).resolve() if cnv_download_complete_path else None,
        "entity_intervals": Path(entity_intervals_path).resolve() if entity_intervals_path else None,
        "gistic_source": Path(gistic_source_path).resolve() if gistic_source_path else None,
        "cnv_streaming_success": Path(cnv_streaming_success_path).resolve() if cnv_streaming_success_path else None,
    }
    for path in (*paths.values(), *(value for value in optional_paths.values() if value is not None)):
        _assert_source_name(path, core_parent=path == paths["core_manifest"])
    if (optional_paths["cnv_gene_calls"] is None) != (optional_paths["cnv_lncrna_calls"] is None):
        raise GenomicTrainingError("Long CNV mode requires both gene and lncRNA call tables")
    if optional_paths["cnv_entity_coverage"] is not None and optional_paths["cnv_gene_calls"] is None:
        raise GenomicTrainingError("CNV entity callability requires paired long/sparse CNV calls")
    cnv_modes = sum(
        value is not None
        for value in (
            optional_paths["cnv_gene_calls"], optional_paths["cnv_segment_root"],
            optional_paths["cnv_streaming_success"],
        )
    )
    if cnv_modes > 1:
        raise GenomicTrainingError("Choose exactly one CNV input mode")
    if optional_paths["cnv_segment_root"] is not None and optional_paths["entity_intervals"] is None:
        raise GenomicTrainingError("Raw segment mode requires entity intervals")
    if (optional_paths["cnv_segment_root"] is None) != (optional_paths["cnv_download_complete"] is None):
        raise GenomicTrainingError("Formal raw segment mode requires a DOWNLOAD_COMPLETE gate")
    segment_gate: dict[str, Any] | None = None
    streaming_payload: dict[str, Any] | None = None
    if optional_paths["cnv_segment_root"] is not None:
        from .gdc_segment_cnv import validate_segment_download_gate

        segment_gate = validate_segment_download_gate(
            optional_paths["cnv_download_complete"],
            staging_root=optional_paths["cnv_segment_root"],
            bindings={
                "candidates": paths["candidates"],
                "patient_folds": paths["patient_folds"],
                "entity_intervals": optional_paths["entity_intervals"],
                "pathway_membership": paths["membership"],
                "sample_gene_mutation": paths["sample_gene_mutation"],
                "sample_lncrna_mutation": paths["sample_lncrna_mutation"],
                "mc3": paths["mc3"],
                "core_embedding_manifest": paths["core_manifest"],
            },
        )
        raise GenomicTrainingError(
            "RAW_GDC_SEGMENT_GLOBAL_LONG_FORM_NOT_FORMAL: the current mapper accumulates "
            "all patient-by-entity rows in memory; use a future hash-bound streaming "
            "materializer/trainer instead"
        )
    if optional_paths["cnv_streaming_success"] is not None:
        from .segment_cnv_streaming import validate_streaming_store

        streaming_payload = validate_streaming_store(
            optional_paths["cnv_streaming_success"],
            expected_bindings={
                "candidates": paths["candidates"], "patient_folds": paths["patient_folds"],
                "pathway_membership": paths["membership"],
                "sample_gene_mutation": paths["sample_gene_mutation"],
                "sample_lncrna_mutation": paths["sample_lncrna_mutation"],
                "mc3": paths["mc3"], "core_embedding_manifest": paths["core_manifest"],
            },
        )
    output = Path(output_root).resolve()
    if output.exists():
        raise GenomicTrainingError(f"Genomic training refuses output reuse: {output}")

    source_audit = _audit_sources(
        candidates_path=paths["candidates"],
        patient_folds_path=paths["patient_folds"],
        membership_path=paths["membership"],
        sample_gene_mutation_path=paths["sample_gene_mutation"],
        sample_lncrna_mutation_path=paths["sample_lncrna_mutation"],
        mc3_path=paths["mc3"],
        cnv_gene_calls_path=optional_paths["cnv_gene_calls"],
        cnv_lncrna_calls_path=optional_paths["cnv_lncrna_calls"],
        cnv_entity_coverage_path=optional_paths["cnv_entity_coverage"],
        cnv_segment_root=optional_paths["cnv_segment_root"],
        cnv_download_complete_path=optional_paths["cnv_download_complete"],
        entity_intervals_path=optional_paths["entity_intervals"],
        gistic_source_path=optional_paths["gistic_source"],
        cnv_streaming_success_path=optional_paths["cnv_streaming_success"],
    )
    core_manifest, core_manifest_sha, core_parameter_composite = _validate_core_manifest(
        paths["core_manifest"]
    )

    candidates = normalise_candidates(_read_table(paths["candidates"]))
    folds = normalise_patient_folds(_read_table(paths["patient_folds"]))
    membership = normalise_exact_membership(_read_table(paths["membership"]))
    cnv_static_coverage: dict[str, dict[str, set[str]]] | None = None
    cnv_pathway_coverage: dict[str, set[str]] | None = None
    if optional_paths["cnv_entity_coverage"] is not None:
        cnv_static_coverage = normalise_cnv_entity_coverage(
            _read_table(optional_paths["cnv_entity_coverage"])
        )
        cnv_pathway_coverage = exact_pathway_callable_entities(
            membership, cnv_static_coverage["gene"]
        )
    missing_cancers = sorted(set(candidates.cancer_id) - set(folds.cancer_id))
    if missing_cancers:
        raise GenomicTrainingError(f"Candidate cancers missing from patient folds: {missing_cancers}")
    gene_mutation = normalise_gene_mutation_calls(_read_table(paths["sample_gene_mutation"]))
    lnc_mutation = normalise_lncrna_mutation_calls(_read_table(paths["sample_lncrna_mutation"]))
    gene_mutation = gene_mutation.merge(
        folds[["cancer_id", "patient_id"]], on=["cancer_id", "patient_id"], how="inner", validate="many_to_one"
    )
    lnc_mutation = lnc_mutation.merge(
        folds[["cancer_id", "patient_id"]], on=["cancer_id", "patient_id"], how="inner", validate="many_to_one"
    )
    pathway_mutation = build_exact_pathway_calls(gene_mutation, membership)

    segment_audit: dict[str, Any] = {"mode": "UNAVAILABLE"}
    compact_cnv = None
    if optional_paths["cnv_streaming_success"] is not None:
        from .segment_cnv_streaming import open_streaming_partitions

        compact_cnv = open_streaming_partitions(optional_paths["cnv_streaming_success"])
        raw_gene_cnv = pd.DataFrame()
        raw_lnc_cnv = pd.DataFrame()
        segment_audit = {
            "mode": "STREAMING_COMPACT_GDC_SEGMENT",
            "streaming_success_path": str(optional_paths["cnv_streaming_success"]),
            "streaming_context_sha256": streaming_payload["context_sha256"],
            "cancers": len(compact_cnv),
            "typed_unavailable_is_never_zero": True,
        }
    elif optional_paths["cnv_gene_calls"] is not None:
        raw_gene_cnv = _read_table(optional_paths["cnv_gene_calls"])
        raw_lnc_cnv = _read_table(optional_paths["cnv_lncrna_calls"])
        segment_audit = {"mode": "LONG_GISTIC_OR_SEGMENT_DERIVED"}
    elif optional_paths["cnv_segment_root"] is not None:
        raw_gene_cnv, raw_lnc_cnv, segment_details = segment_calls_from_raw(
            optional_paths["cnv_segment_root"],
            _read_table(optional_paths["entity_intervals"]),
            folds,
            manifest_tsv=segment_gate["manifest_tsv"],
        )
        segment_audit = {
            "mode": "RAW_GDC_SEGMENT_MIDPOINT",
            "download_gate_path": str(optional_paths["cnv_download_complete"]),
            "download_inventory_sha256": segment_gate["inventory_sha256"],
            "selected_manifest_files": segment_gate["selected_files"],
            **segment_details,
        }
    else:
        raw_gene_cnv = pd.DataFrame(columns=["cancer_id", "patient_id", "gene_id", "cnv_value", "cnv_callable"])
        raw_lnc_cnv = pd.DataFrame(columns=["cancer_id", "patient_id", "lncrna_id", "cnv_value", "cnv_callable"])
    if compact_cnv is not None:
        gene_cnv = pd.DataFrame()
        lnc_cnv = pd.DataFrame()
        pathway_cnv = pd.DataFrame()
    elif len(raw_gene_cnv) and len(raw_lnc_cnv):
        gene_cnv = normalise_cnv_calls(
            raw_gene_cnv, entity_kind="gene", event_threshold=settings.cnv_event_threshold
        )
        lnc_cnv = normalise_cnv_calls(
            raw_lnc_cnv, entity_kind="lncrna", event_threshold=settings.cnv_event_threshold
        )
        gene_cnv = gene_cnv.merge(
            folds[["cancer_id", "patient_id"]], on=["cancer_id", "patient_id"], how="inner", validate="many_to_one"
        )
        lnc_cnv = lnc_cnv.merge(
            folds[["cancer_id", "patient_id"]], on=["cancer_id", "patient_id"], how="inner", validate="many_to_one"
        )
        pathway_cnv = build_exact_pathway_calls(gene_cnv, membership)
    else:
        empty = pd.DataFrame(columns=["cancer_id", "patient_id", "entity_id", "event", "callable", "burden"])
        gene_cnv = empty.copy()
        lnc_cnv = empty.copy()
        pathway_cnv = empty.copy()

    sources = {
        "mutation": (_group_calls(lnc_mutation), _group_calls(pathway_mutation), None, None),
        "cnv": (
            _group_calls(lnc_cnv),
            _group_calls(pathway_cnv),
            cnv_static_coverage["lncrna"] if cnv_static_coverage is not None else None,
            cnv_pathway_coverage,
        ),
    }
    if compact_cnv is not None:
        sources["cnv"] = (compact_cnv, compact_cnv, None, None)
    probability_sums = {modality: np.zeros(len(candidates), np.float64) for modality in sources}
    probability_counts = {modality: np.zeros(len(candidates), np.int16) for modality in sources}
    probability_by_patient_fold = {
        modality: np.full((N_FOLDS, len(candidates)), np.nan, np.float32)
        for modality in sources
    }
    checkpoint_rows: list[dict[str, Any]] = []
    modality_fold_status: dict[str, list[dict[str, Any]]] = {"mutation": [], "cnv": []}
    core_hashes_before: dict[str, str] = {}

    output.mkdir(parents=True)
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir()
    for fold in range(N_FOLDS):
        core = load_fold_core_embeddings(
            paths["core_manifest"], core_manifest, fold,
            lncrna_node_type=lncrna_node_type, pathway_node_type=pathway_node_type,
        )
        core_hashes_before.update(core.input_hashes)
        split_maps = {
            split: _split_patients(folds, fold, split)
            for split in ("train", "validation", "test")
        }
        for modality, (
            lnc_by_cancer,
            path_by_cancer,
            lnc_callable_by_cancer,
            path_callable_by_cancer,
        ) in sources.items():
            train_core, train_domain, train_labels = _collect_examples(
                candidates, lnc_by_cancer, path_by_cancer, split_maps["train"], core,
                modality=modality, min_pair_callable=settings.min_pair_callable,
                maximum=settings.max_train_rows, seed=settings.seed + fold,
                lncrna_callable_by_cancer=lnc_callable_by_cancer,
                pathway_callable_by_cancer=path_callable_by_cancer,
            )
            validation_core, validation_domain, validation_labels = _collect_examples(
                candidates, lnc_by_cancer, path_by_cancer, split_maps["validation"], core,
                modality=modality, min_pair_callable=settings.min_pair_callable,
                maximum=settings.max_validation_rows, seed=settings.seed + 50_000 + fold,
                lncrna_callable_by_cancer=lnc_callable_by_cancer,
                pathway_callable_by_cancer=path_callable_by_cancer,
            )
            if len(train_labels) == 0 or len(np.unique(train_labels)) < 2:
                modality_fold_status[modality].append(
                    {
                        "patient_fold": fold,
                        "status": "NULL_WITH_REASON",
                        "reason": f"{modality.upper()}_HEAD_NOT_TRAINABLE_WITH_EXPLICIT_CALLS",
                        "train_rows": int(len(train_labels)),
                    }
                )
                continue
            head, metadata, mean, scale, history = _fit_head(
                train_core, train_domain, train_labels,
                validation_core, validation_domain, validation_labels,
                modality=modality, fold=fold, config=settings,
            )
            import torch

            checkpoint_path = checkpoint_root / f"{modality}_patient_fold_{fold}.pt"
            checkpoint_partial = checkpoint_root / f".{modality}_patient_fold_{fold}.partial.pt"
            torch.save(
                {
                    "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
                    "analysis_version": ANALYSIS_VERSION,
                    "module_id": MODULE_ID,
                    "modality": modality,
                    "patient_fold": fold,
                    "initialization": metadata,
                    "model_state": head.state_dict(),
                    "domain_features": list(_DOMAIN_FEATURES),
                    "domain_mean": mean,
                    "domain_scale": scale,
                    "history": history,
                    "core_checkpoint_sha256": core.checkpoint_sha256,
                    "core_parameter_sha256": core.core_parameter_sha256,
                    "old_checkpoint_loaded": False,
                    "old_predictions_used": False,
                },
                checkpoint_partial,
            )
            os.replace(checkpoint_partial, checkpoint_path)
            checkpoint_sha = _file_sha256(checkpoint_path)
            _atomic_json(
                checkpoint_root / f"{modality}_patient_fold_{fold}.SUCCESS.json",
                {
                    "format": "CC_HHGT_V3_2_GENOMIC_PRIVATE_HEAD_FOLD_SUCCESS_V1",
                    "status": "SUCCESS", "modality": modality, "patient_fold": fold,
                    "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
                    "core_checkpoint_sha256": core.checkpoint_sha256,
                    "core_parameter_sha256": core.core_parameter_sha256,
                    "train_rows": int(len(train_labels)), "validation_rows": int(len(validation_labels)),
                },
            )
            checkpoint_rows.append(
                {
                    "modality": modality,
                    "patient_fold": fold,
                    "path": str(checkpoint_path),
                    "sha256": checkpoint_sha,
                    "initial_parameter_sha256": metadata["initial_parameter_sha256"],
                    "source_checkpoint_sha256": None,
                    "core_checkpoint_sha256": core.checkpoint_sha256,
                    "core_parameter_sha256": core.core_parameter_sha256,
                    "train_rows": int(len(train_labels)),
                    "validation_rows": int(len(validation_labels)),
                    "status": "SUCCESS",
                }
            )
            modality_fold_status[modality].append(
                {"patient_fold": fold, "status": "SUCCESS", "train_rows": int(len(train_labels))}
            )
            for cancer, local in candidates.groupby("cancer_id", observed=True, sort=False):
                stats = candidate_statistics(
                    local,
                    lnc_by_cancer.get(str(cancer), pd.DataFrame()),
                    path_by_cancer.get(str(cancer), pd.DataFrame()),
                    split_maps["test"].get(str(cancer), ()),
                    modality=modality,
                    min_pair_callable=settings.min_pair_callable,
                    lncrna_callable_entities=(
                        lnc_callable_by_cancer.get(str(cancer), set())
                        if lnc_callable_by_cancer is not None else None
                    ),
                    pathway_callable_entities=(
                        path_callable_by_cancer.get(str(cancer), set())
                        if path_callable_by_cancer is not None else None
                    ),
                )
                if not stats.available.any():
                    continue
                core_values, core_available = _candidate_core(local, core)
                predict_mask = stats.available & core_available
                if not predict_mask.any():
                    continue
                probability = _predict_head(
                    head, core_values[predict_mask], stats.domain[predict_mask], mean, scale,
                    settings.prediction_batch_size,
                )
                indices = local.index.to_numpy()[predict_mask]
                probability_sums[modality][indices] += probability
                probability_counts[modality][indices] += 1
                probability_by_patient_fold[modality][fold, indices] = probability

    # Re-hash every frozen representation after all optimizer steps.
    for path_text, before in core_hashes_before.items():
        after = _file_sha256(path_text)
        if after != before:
            raise GenomicTrainingError(f"Frozen core embedding changed during private training: {path_text}")
    if _file_sha256(paths["core_manifest"]) != core_manifest_sha:
        raise GenomicTrainingError("Core embedding manifest changed during private training")

    typed = candidates.copy()
    modality_trained = {
        modality: any(row["status"] == "SUCCESS" for row in rows)
        for modality, rows in modality_fold_status.items()
    }
    full_patients = {
        cancer: group.patient_id.astype(str).tolist()
        for cancer, group in folds.groupby("cancer_id", observed=True, sort=False)
    }
    for modality, (
        lnc_by_cancer,
        path_by_cancer,
        lnc_callable_by_cancer,
        path_callable_by_cancer,
    ) in sources.items():
        count = probability_counts[modality]
        probability = np.divide(
            probability_sums[modality], count,
            out=np.full(len(candidates), np.nan, dtype=float), where=count > 0,
        )
        reason = np.full(len(candidates), f"{modality.upper()}_NO_OOF_PREDICTION", object)
        if not modality_trained[modality]:
            reason[:] = f"{modality.upper()}_PRIVATE_HEAD_NOT_TRAINABLE"
        else:
            for cancer, local in candidates.groupby("cancer_id", observed=True, sort=False):
                if str(cancer) not in lnc_by_cancer or str(cancer) not in path_by_cancer:
                    reason[local.index] = f"{modality.upper()}_NOT_AVAILABLE_FOR_CANCER"
                    continue
                stats = candidate_statistics(
                    local,
                    lnc_by_cancer[str(cancer)],
                    path_by_cancer[str(cancer)],
                    full_patients.get(str(cancer), ()),
                    modality=modality,
                    min_pair_callable=settings.min_pair_callable,
                    lncrna_callable_entities=(
                        lnc_callable_by_cancer.get(str(cancer), set())
                        if lnc_callable_by_cancer is not None else None
                    ),
                    pathway_callable_entities=(
                        path_callable_by_cancer.get(str(cancer), set())
                        if path_callable_by_cancer is not None else None
                    ),
                )
                reason[local.index] = stats.reasons
        reason[(count == 0) & (reason == "")] = (
            f"{modality.upper()}_CORE_EMBEDDING_OR_FOLD_PREDICTION_UNAVAILABLE"
        )
        reason[count > 0] = ""
        typed[f"{modality}_context_probability"] = probability
        typed[f"{modality}_available"] = count > 0
        typed[f"{modality}_unavailable_reason"] = pd.Series(reason).replace("", pd.NA)
        typed[f"{modality}_patient_folds_with_prediction"] = count.astype(int)
    values = typed[["mutation_context_probability", "cnv_context_probability"]].to_numpy(float)
    available_count = np.isfinite(values).sum(axis=1)
    typed["mutation_cnv_context_probability"] = np.divide(
        np.nansum(values, axis=1), available_count,
        out=np.full(len(typed), np.nan, dtype=float), where=available_count > 0,
    )
    typed["genomic_available"] = available_count > 0
    typed["genomic_unavailable_reason"] = np.where(
        available_count > 0, pd.NA, "NO_AVAILABLE_NEWLY_TRAINED_GENOMIC_HEAD"
    )
    typed["analysis_version"] = ANALYSIS_VERSION
    typed["training_run_id"] = str(training_run_id)
    typed["module_id"] = MODULE_ID
    typed["target_level"] = TARGET_LEVEL
    typed["prediction_format"] = PREDICTION_FORMAT
    typed["changes_primary_ranking"] = False
    if len(typed) != len(candidates) or typed[list(TARGET_KEYS)].duplicated().any():
        raise GenomicTrainingError("Typed genomic output changed the exact candidate universe")
    for column in ("mutation_context_probability", "cnv_context_probability", "mutation_cnv_context_probability"):
        finite = pd.to_numeric(typed[column], errors="coerce").dropna()
        if not finite.between(0, 1).all():
            raise GenomicTrainingError(f"{column} is outside [0, 1]")

    prediction_path = output / "mutation_cnv_typed_predictions.parquet"
    _atomic_parquet(typed, prediction_path)
    fold_prediction_root = output / "patient_fold_oof_predictions"
    fold_prediction_paths: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        fold_frame = candidates.copy()
        fold_frame["patient_fold_id"] = fold
        for modality in ("mutation", "cnv"):
            values = probability_by_patient_fold[modality][fold].astype(float)
            fold_frame[f"{modality}_context_probability"] = values
            fold_frame[f"{modality}_available"] = np.isfinite(values)
        partition = fold_prediction_root / f"patient_fold={fold}"
        partition.mkdir(parents=True, exist_ok=False)
        fold_path = partition / "part-0.parquet"
        _atomic_parquet(fold_frame, fold_path)
        fold_prediction_paths.append(
            {
                "patient_fold_id": fold,
                "path": str(fold_path),
                "sha256": _file_sha256(fold_path),
                "rows": len(fold_frame),
            }
        )
    expert_prediction_paths: dict[str, Path] = {}
    for modality in ("mutation", "cnv"):
        expert_path = output / f"{modality}_typed_predictions.parquet"
        expert_columns = list(TARGET_KEYS) + [
            f"{modality}_context_probability",
            f"{modality}_available",
            f"{modality}_unavailable_reason",
            f"{modality}_patient_folds_with_prediction",
            "analysis_version",
            "training_run_id",
            "module_id",
            "target_level",
            "prediction_format",
            "changes_primary_ranking",
        ]
        _atomic_parquet(typed[expert_columns], expert_path)
        expert_prediction_paths[modality] = expert_path
    checkpoint_manifest = {
        "checkpoint_format": PRIVATE_CHECKPOINT_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "folds": N_FOLDS,
        "records": checkpoint_rows,
        "modality_fold_status": modality_fold_status,
        "all_private_heads_random_initialization": all(
            row.get("source_checkpoint_sha256") is None for row in checkpoint_rows
        ),
        "core_detached_and_frozen": True,
    }
    checkpoint_manifest_path = output / "CHECKPOINT_MANIFEST.json"
    _atomic_json(checkpoint_manifest_path, checkpoint_manifest)
    config_path = output / "RUN_CONFIG.json"
    run_config = {
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": training_run_id,
        "training": asdict(settings),
        "lncrna_node_type": lncrna_node_type,
        "pathway_node_type": pathway_node_type,
        "mutation_absence_policy": "MISSING_IS_UNAVAILABLE_NEVER_WILDTYPE_OR_ZERO",
        "cnv_absence_policy": "MISSING_IS_UNAVAILABLE_NEVER_NEUTRAL_OR_ZERO",
        "cnv_sparse_callability_policy": (
            "SPARSE_EVENTS_PLUS_EXPLICIT_ENTITY_COVERAGE"
            if cnv_static_coverage is not None else "NOT_APPLICABLE"
        ),
        "exact_pathway_policy": "REBUILT_FROM_SAMPLE_GENE_X_V32_EXACT_MEMBERSHIP",
    }
    _atomic_json(config_path, run_config)
    input_audit_path = output / "INPUT_LINEAGE_AUDIT.json"
    _atomic_json(input_audit_path, source_audit)

    input_artifacts = [
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "artifact_kind": (
                "raw_data" if item["source_role"] == "raw_data"
                else "annotation" if item["source_role"] == "static_annotation"
                else "split_manifest" if item["source_role"] == "split_manifest"
                else "standardized_input"
            ),
            "generation": item["generation"],
            "source_role": item["source_role"],
            "outcome_derived": item["outcome_derived"],
            "fold_fitted": item["fold_fitted"],
        }
        for item in source_audit["artifacts"]
    ]
    input_artifacts.append(
        {
            "path": str(paths["core_manifest"]),
            "sha256": core_manifest_sha,
            "artifact_kind": "v32_core_checkpoint",
            "generation": "V3.2",
            "source_role": "v32_core_checkpoint",
            "outcome_derived": True,
            "fold_fitted": True,
            "use_role": "aux_parent",
        }
    )
    code_files = {"genomic_training.py": _file_sha256(Path(__file__).resolve())}
    runner_path = Path(__file__).resolve().parents[2] / "scripts" / "run_v32_genomic_training.py"
    if runner_path.is_file():
        code_files["run_v32_genomic_training.py"] = _file_sha256(runner_path)
    code_sha = _canonical_json_sha256(code_files)
    lineage = {
        "module_id": MODULE_ID,
        "analysis_version": ANALYSIS_VERSION,
        "training_run_id": training_run_id,
        "training_status": "SUCCESS",
        "initialization_policy": "FROZEN_V32_CORE_PRIVATE_HEAD_FROM_SCRATCH",
        "folds": N_FOLDS,
        "seeds": sorted(
            {
                int(settings.seed + fold * 101 + offset)
                for fold in range(N_FOLDS)
                for offset in (0, 10_000)
            }
        ),
        "code_sha256": code_sha,
        "config_sha256": _file_sha256(config_path),
        "input_manifest_sha256": source_audit["lineage_sha256"],
        "checkpoint_manifest_sha256": _file_sha256(checkpoint_manifest_path),
        "old_checkpoint_loaded": False,
        "old_predictions_used_as_features": False,
        "old_rankings_used_as_outputs": False,
        "private_head_trained_from_scratch": bool(checkpoint_rows),
        "core_parameters_frozen": True,
        "v32_core_checkpoint_sha256": core_manifest_sha,
        "core_parameters_before_sha256": core_parameter_composite,
        "core_parameters_after_sha256": core_parameter_composite,
        "input_artifacts": input_artifacts,
        "modalities": modality_fold_status,
        "mutation_exact_pathway_rebuilt_from_gene_calls": True,
        "family_pathway_statistics_loaded": False,
        "missing_mutation_assumed_wildtype": False,
        "missing_cnv_assumed_neutral": False,
        "cnv_cancers_with_calls": sorted(set(lnc_cnv.cancer_id.astype(str)) & set(pathway_cnv.cancer_id.astype(str))),
        "cnv_cancers_null_with_reason": sorted(set(candidates.cancer_id.astype(str)) - set(lnc_cnv.cancer_id.astype(str))),
        "cnv_explicit_entity_callability_used": cnv_static_coverage is not None,
        "cnv_raw_segment_callability_used": segment_gate is not None or streaming_payload is not None,
        "cnv_streaming_compact_used": streaming_payload is not None,
        "cnv_streaming_context_sha256": (
            streaming_payload.get("context_sha256") if streaming_payload is not None else None
        ),
        "cnv_callable_gene_entities_by_cancer": (
            {key: len(value) for key, value in cnv_static_coverage["gene"].items()}
            if cnv_static_coverage is not None else {}
        ),
        "cnv_callable_lncrna_entities_by_cancer": (
            {key: len(value) for key, value in cnv_static_coverage["lncrna"].items()}
            if cnv_static_coverage is not None else {}
        ),
        "cnv_fully_callable_exact_pathways_by_cancer": (
            {key: len(value) for key, value in cnv_pathway_coverage.items()}
            if cnv_pathway_coverage is not None else {}
        ),
        "segment_ingestion": segment_audit,
        "prediction_path": str(prediction_path),
        "prediction_sha256": _file_sha256(prediction_path),
        "prediction_rows": int(len(typed)),
        "expert_prediction_paths": {
            modality: {
                "path": str(path),
                "sha256": _file_sha256(path),
            }
            for modality, path in expert_prediction_paths.items()
        },
        "patient_fold_oof_prediction_paths": fold_prediction_paths,
        "patient_fold_oof_predictions_not_fold_averaged": True,
        "mutation_available_rows": int(typed.mutation_available.sum()),
        "cnv_available_rows": int(typed.cnv_available.sum()),
    }
    lineage_path = output / "LINEAGE.json"
    _atomic_json(lineage_path, lineage)
    success = {
        "status": "SUCCESS",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "training_run_id": training_run_id,
        "prediction_path": str(prediction_path),
        "prediction_sha256": lineage["prediction_sha256"],
        "lineage_path": str(lineage_path),
        "lineage_sha256": _file_sha256(lineage_path),
        "checkpoint_manifest_sha256": lineage["checkpoint_manifest_sha256"],
        "candidate_rows_preserved": int(len(typed)),
        "mutation_available_rows": lineage["mutation_available_rows"],
        "cnv_available_rows": lineage["cnv_available_rows"],
        "expert_prediction_paths": lineage["expert_prediction_paths"],
        "patient_fold_oof_prediction_paths": lineage["patient_fold_oof_prediction_paths"],
        "null_is_never_zero_or_wildtype": True,
    }
    _atomic_json(output / "SUCCESS.json", success)
    return success


__all__ = [
    "ANALYSIS_VERSION",
    "GenomicTrainingConfig",
    "GenomicTrainingError",
    "build_exact_pathway_calls",
    "candidate_statistics",
    "normalise_cnv_calls",
    "normalise_exact_membership",
    "normalise_gene_mutation_calls",
    "normalise_lncrna_mutation_calls",
    "run_genomic_training",
    "segment_calls_from_raw",
]
