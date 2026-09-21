"""Prepare a leakage-safe pan-cancer GISTIC CNV pilot bundle.

The public PanCan GISTIC matrix is wide and contains both measured zero calls
and non-zero events.  Expanding every measured zero into long form creates
hundreds of millions of rows.  This module therefore emits a sparse event
table plus an explicit entity-callability table.  Downstream code must use
both tables; absence from the sparse event table alone never means neutral.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO

import pandas as pd

from .genomic_training import ASSAY_CALLABLE_SENTINEL, GenomicTrainingError
from .multimodal_fusion import artifact_sha256


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_ROUTED_CNV_CANDIDATE"
MODULE_ID = "pancancer_gistic_cnv_pilot"
_TARGET_KEYS = ("cancer_id", "lncrna_id", "pathway_id")
_GTF_ATTRIBUTE = re.compile(r'(\w+) "([^"]+)";')
_CALL_SCHEMA = {
    "cancer_id": "string",
    "patient_id": "string",
    "entity_id": "string",
    "cnv_value": "float64",
    "cnv_callable": "bool",
}


class PanCancerCNVError(GenomicTrainingError):
    """Raised when a GISTIC pilot input violates the typed CNV contract."""


@dataclass(frozen=True)
class GisticPreparationConfig:
    min_patients_per_cancer: int = 12
    parquet_row_group_size: int = 100_000
    require_all_fold_cancers: bool = True
    allowed_calls: tuple[int, ...] = (-2, -1, 0, 1, 2)

    def validate(self) -> None:
        if self.min_patients_per_cancer < 1:
            raise ValueError("min_patients_per_cancer must be positive")
        if self.parquet_row_group_size < 1:
            raise ValueError("parquet_row_group_size must be positive")
        if not self.allowed_calls or 0 not in self.allowed_calls:
            raise ValueError("allowed_calls must be non-empty and include zero")


def _open_text(path: Path) -> TextIO:
    return gzip.open(path, "rt", newline="") if path.suffix.lower() == ".gz" else path.open("r", encoding="utf-8", newline="")


def _patient_id(value: object) -> str:
    text = str(value).strip()
    parts = text.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 and parts[0].upper() == "TCGA" else text


def _sample_barcode(value: object) -> str:
    return "-".join(str(value).strip().split("-")[:4])


def _sample_type(value: object) -> int:
    parts = str(value).strip().split("-")
    if len(parts) < 4:
        return 99
    match = re.match(r"(\d{2})", parts[3])
    return int(match.group(1)) if match else 99


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_table(path: Path) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith((".tsv", ".tsv.gz", ".txt", ".txt.gz")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _normalise_folds(frame: pd.DataFrame) -> pd.DataFrame:
    if "cancer_id" not in frame:
        raise PanCancerCNVError("Patient fold manifest lacks cancer_id")
    if "patient_id" not in frame or "sample_id" not in frame:
        raise PanCancerCNVError("Patient fold manifest requires explicit patient_id and sample_id")
    if frame[["cancer_id", "patient_id", "sample_id"]].isna().any().any():
        raise PanCancerCNVError("Patient fold manifest contains null identity fields")
    result = pd.DataFrame(
        {
            "cancer_id": frame.cancer_id.astype(str).str.upper(),
            "patient_id": frame.patient_id.astype(str).str.strip(),
            "fold_sample_id": frame.sample_id.astype(str).str.strip(),
        }
    ).drop_duplicates()
    if result[["patient_id", "fold_sample_id"]].eq("").any().any():
        raise PanCancerCNVError("Patient fold manifest contains empty identity fields")
    sample_mapping = result.groupby(["cancer_id", "fold_sample_id"], observed=True).patient_id.nunique()
    if sample_mapping.gt(1).any():
        raise PanCancerCNVError("A fold sample maps to multiple explicit patients")
    conflicts = result.groupby("patient_id", observed=True).cancer_id.nunique()
    if conflicts.gt(1).any():
        raise PanCancerCNVError("A TCGA patient maps to multiple cancer IDs")
    # Multiple RNA aliquots from one patient are allowed, but the desired
    # sample barcode is deterministic and is used only to rank CNV aliquots.
    result = result.sort_values(["patient_id", "fold_sample_id"], kind="stable")
    return result.drop_duplicates("patient_id", keep="first").reset_index(drop=True)


def _target_entities(
    intervals: pd.DataFrame,
    membership: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, str], set[str], set[str]]:
    required_iv = {"entity_type", "entity_id"}
    if missing := sorted(required_iv - set(intervals.columns)):
        raise PanCancerCNVError(f"Entity intervals lack columns: {missing}")
    if not {"pathway_id", "gene_id"}.issubset(membership.columns):
        raise PanCancerCNVError("Exact pathway membership lacks pathway_id/gene_id")
    if missing := sorted(set(_TARGET_KEYS) - set(candidates.columns)):
        raise PanCancerCNVError(f"Candidate universe lacks keys: {missing}")
    target_genes = set(membership.gene_id.dropna().astype(str).str.split(".").str[0])
    target_lncs = set(candidates.lncrna_id.dropna().astype(str))
    gene_by_ensembl: dict[str, str] = {}
    lnc_by_ensembl: dict[str, str] = {}
    for row in intervals[["entity_type", "entity_id"]].drop_duplicates().itertuples(index=False):
        entity_id = str(row.entity_id)
        ensembl = entity_id.removeprefix("LNC:").split(".")[0]
        kind = str(row.entity_type).lower()
        if kind == "gene" and entity_id.split(".")[0] in target_genes:
            gene_by_ensembl[ensembl] = entity_id.split(".")[0]
        elif kind in {"lncrna", "lncrna_gene"} and entity_id in target_lncs:
            lnc_by_ensembl[ensembl] = entity_id
    if not gene_by_ensembl or not lnc_by_ensembl:
        raise PanCancerCNVError("No target gene or lncRNA entity could be resolved from intervals")
    return gene_by_ensembl, lnc_by_ensembl, target_genes, target_lncs


def _symbol_entity_maps(
    gtf_path: Path,
    gene_by_ensembl: Mapping[str, str],
    lnc_by_ensembl: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str], int]:
    symbol_candidates: dict[str, set[str]] = {}
    with _open_text(gtf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attributes = dict(_GTF_ATTRIBUTE.findall(fields[8]))
            gene_id = attributes.get("gene_id", "").split(".")[0]
            symbol = attributes.get("gene_name")
            if gene_id and symbol:
                symbol_candidates.setdefault(symbol, set()).add(gene_id)
    gene_map: dict[str, str] = {}
    lnc_map: dict[str, str] = {}
    ambiguous = 0
    for symbol, ids in symbol_candidates.items():
        gene_hits = {gene_by_ensembl[value] for value in ids if value in gene_by_ensembl}
        lnc_hits = {lnc_by_ensembl[value] for value in ids if value in lnc_by_ensembl}
        if len(gene_hits) == 1:
            gene_map[symbol] = next(iter(gene_hits))
        elif len(gene_hits) > 1:
            ambiguous += 1
        if len(lnc_hits) == 1:
            lnc_map[symbol] = next(iter(lnc_hits))
        elif len(lnc_hits) > 1:
            ambiguous += 1
    return gene_map, lnc_map, ambiguous


def _choose_gistic_columns(
    sample_names: Iterable[str], folds: pd.DataFrame
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_lookup = folds.set_index("patient_id").to_dict("index")
    by_patient: dict[str, list[tuple[int, str]]] = {}
    all_samples = list(map(str, sample_names))
    for index, sample in enumerate(all_samples):
        patient = _patient_id(sample)
        if patient in fold_lookup:
            by_patient.setdefault(patient, []).append((index, sample))
    selected: list[dict[str, Any]] = []
    duplicates = 0
    for patient, candidates in sorted(by_patient.items()):
        fold_row = fold_lookup[patient]
        desired_barcode = _sample_barcode(fold_row["fold_sample_id"])
        ranked = sorted(
            candidates,
            key=lambda item: (
                _sample_barcode(item[1]) != desired_barcode,
                _sample_type(item[1]) != 1,
                _sample_type(item[1]),
                item[1],
            ),
        )
        duplicates += max(len(ranked) - 1, 0)
        index, sample = ranked[0]
        selected.append(
            {
                "source_column_index": int(index),
                "source_sample_id": sample,
                "patient_id": patient,
                "cancer_id": str(fold_row["cancer_id"]),
                "fold_sample_id": str(fold_row["fold_sample_id"]),
                "duplicate_columns_discarded": len(ranked) - 1,
                "selection_policy": "FOLD_SAMPLE_BARCODE_THEN_PRIMARY_TUMOR_THEN_LEXICAL",
            }
        )
    cancers = sorted({str(row["cancer_id"]) for row in selected})
    audit = {
        "gistic_sample_columns": len(all_samples),
        "fold_patients": int(folds.patient_id.nunique()),
        "matched_unique_patients": len(selected),
        "unmatched_gistic_columns": len(all_samples) - sum(len(value) for value in by_patient.values()),
        "discarded_duplicate_patient_columns": duplicates,
        "covered_cancers": cancers,
    }
    return selected, audit


class _ParquetSink:
    def __init__(self, path: Path, row_group_size: int):
        import pyarrow as pa

        self.path = path
        self.row_group_size = int(row_group_size)
        self.schema = pa.schema(
            [
                pa.field("cancer_id", pa.string()),
                pa.field("patient_id", pa.string()),
                pa.field("entity_id", pa.string()),
                pa.field("cnv_value", pa.float64()),
                pa.field("cnv_callable", pa.bool_()),
            ]
        )
        self.writer = None
        self.rows: list[dict[str, Any]] = []
        self.count = 0

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.row_group_size:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, self.schema, compression="zstd")
        self.writer.write_table(table)
        self.count += len(self.rows)
        self.rows.clear()

    def close(self) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.flush()
        if self.writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=self.schema), self.path, compression="zstd")
        else:
            self.writer.close()


def prepare_pancancer_gistic_bundle(
    *,
    gistic_path: str | Path,
    gencode_gtf_path: str | Path,
    patient_folds_path: str | Path,
    entity_intervals_path: str | Path,
    pathway_membership_path: str | Path,
    candidates_path: str | Path,
    output_root: str | Path,
    run_id: str,
    config: GisticPreparationConfig | None = None,
    audit_only: bool = False,
) -> dict[str, Any]:
    """Create sparse pan-cancer CNV calls and explicit static callability.

    The output is a pre-formal routed candidate and never overwrites V3.2
    formal artifacts.  ``audit_only`` performs the complete coverage scan but
    intentionally does not produce trainable call tables.
    """

    settings = config or GisticPreparationConfig()
    settings.validate()
    if not str(run_id).startswith("v32-routed-cnv-candidate-"):
        raise PanCancerCNVError("run_id must start with v32-routed-cnv-candidate-")
    paths = {
        "gistic": Path(gistic_path).resolve(),
        "gtf": Path(gencode_gtf_path).resolve(),
        "folds": Path(patient_folds_path).resolve(),
        "intervals": Path(entity_intervals_path).resolve(),
        "membership": Path(pathway_membership_path).resolve(),
        "candidates": Path(candidates_path).resolve(),
    }
    for label, path in paths.items():
        if not path.is_file():
            raise PanCancerCNVError(f"Missing {label} input: {path}")
    output = Path(output_root).resolve()
    if output.exists():
        raise PanCancerCNVError(f"CNV candidate preparation refuses output reuse: {output}")
    output.mkdir(parents=True)

    folds = _normalise_folds(_read_table(paths["folds"]))
    intervals = _read_table(paths["intervals"])
    membership = _read_table(paths["membership"])
    candidates = _read_table(paths["candidates"])
    gene_by_ensembl, lnc_by_ensembl, target_genes, target_lncs = _target_entities(
        intervals, membership, candidates
    )
    gene_symbol_map, lnc_symbol_map, ambiguous_symbols = _symbol_entity_maps(
        paths["gtf"], gene_by_ensembl, lnc_by_ensembl
    )

    with _open_text(paths["gistic"]) as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise PanCancerCNVError("GISTIC input is empty") from exc
    if len(header) < 4 or header[:3] != ["Gene Symbol", "Locus ID", "Cytoband"]:
        raise PanCancerCNVError("GISTIC header does not match thresholded-by-genes format")
    selections, sample_audit = _choose_gistic_columns(header[3:], folds)
    required_cancers = sorted(folds.cancer_id.unique())
    covered_cancers = sample_audit["covered_cancers"]
    if settings.require_all_fold_cancers and covered_cancers != required_cancers:
        missing = sorted(set(required_cancers) - set(covered_cancers))
        raise PanCancerCNVError(f"GISTIC lacks fold-matched patients for cancers: {missing}")
    per_cancer = pd.DataFrame(selections).groupby("cancer_id", observed=True).patient_id.nunique()
    insufficient = {
        str(cancer): int(count)
        for cancer, count in per_cancer.items()
        if int(count) < settings.min_patients_per_cancer
    }
    if insufficient:
        raise PanCancerCNVError(f"Cancers below minimum matched-patient gate: {insufficient}")

    selection_path = output / "PATIENT_SELECTION.parquet"
    pd.DataFrame(selections).to_parquet(selection_path, index=False)
    gene_path = output / "cnv_gene_sparse_calls.parquet"
    lnc_path = output / "cnv_lncrna_sparse_calls.parquet"
    gene_sink = None if audit_only else _ParquetSink(gene_path, settings.parquet_row_group_size)
    lnc_sink = None if audit_only else _ParquetSink(lnc_path, settings.parquet_row_group_size)
    selected_positions = [3 + int(item["source_column_index"]) for item in selections]
    allowed = set(map(int, settings.allowed_calls))
    gene_coverage: set[str] = set()
    lnc_coverage: set[str] = set()
    source_rows = mapped_rows = invalid_values = zero_values = nonzero_values = 0
    event_counts = {"gene": 0, "lncrna": 0}
    with _open_text(paths["gistic"]) as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader)
        for fields in reader:
            source_rows += 1
            if len(fields) != len(header):
                raise PanCancerCNVError(
                    f"GISTIC row {source_rows + 1} has {len(fields)} fields; expected {len(header)}"
                )
            symbol = fields[0]
            entities: list[tuple[str, str, _ParquetSink | None]] = []
            if symbol in gene_symbol_map:
                entity = gene_symbol_map[symbol]
                gene_coverage.add(entity)
                entities.append(("gene", entity, gene_sink))
            if symbol in lnc_symbol_map:
                entity = lnc_symbol_map[symbol]
                lnc_coverage.add(entity)
                entities.append(("lncrna", entity, lnc_sink))
            if not entities:
                continue
            mapped_rows += 1
            parsed_values: list[float | None] = []
            for position in selected_positions:
                raw = fields[position].strip()
                if raw in {"", "NA", "NaN", "nan", "."}:
                    parsed_values.append(None)
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise PanCancerCNVError(f"Invalid GISTIC value {raw!r}") from exc
                if not value.is_integer() or int(value) not in allowed:
                    invalid_values += 1
                    raise PanCancerCNVError(f"Out-of-contract thresholded GISTIC call: {value}")
                parsed_values.append(value)
            for (kind, entity, sink) in entities:
                for selection, value in zip(selections, parsed_values):
                    if value is None:
                        continue
                    if value == 0:
                        zero_values += 1
                        continue
                    nonzero_values += 1
                    event_counts[kind] += 1
                    if sink is not None:
                        sink.append(
                            {
                                "cancer_id": selection["cancer_id"],
                                "patient_id": selection["patient_id"],
                                "entity_id": entity,
                                "cnv_value": float(value),
                                "cnv_callable": True,
                            }
                        )
    # A per-sample sentinel declares that the selected patient was assayed.  It
    # only expands over entities present in cnv_entity_coverage.parquet.
    if not audit_only:
        assert gene_sink is not None and lnc_sink is not None
        for selection in selections:
            sentinel = {
                "cancer_id": selection["cancer_id"],
                "patient_id": selection["patient_id"],
                "entity_id": ASSAY_CALLABLE_SENTINEL,
                "cnv_value": 0.0,
                "cnv_callable": True,
            }
            gene_sink.append(sentinel.copy())
            lnc_sink.append(sentinel.copy())
        gene_sink.close()
        lnc_sink.close()

    coverage_rows = [
        {"cancer_id": cancer, "entity_type": kind, "entity_id": entity}
        for cancer in covered_cancers
        for kind, entities in (("gene", sorted(gene_coverage)), ("lncrna", sorted(lnc_coverage)))
        for entity in entities
    ]
    coverage_path = output / "cnv_entity_coverage.parquet"
    if not audit_only:
        pd.DataFrame(coverage_rows).to_parquet(coverage_path, index=False)
    audit = {
        "status": "AUDIT_ONLY" if audit_only else "SUCCESS",
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "run_id": run_id,
        "contract": "SPARSE_NONZERO_EVENTS_PLUS_EXPLICIT_ENTITY_CALLABILITY",
        "neutral_policy": "ZERO_ONLY_WHEN_SAMPLE_AND_ENTITY_ARE_EXPLICITLY_CALLABLE",
        "sample_selection": sample_audit,
        "patients_per_cancer": {str(key): int(value) for key, value in per_cancer.sort_index().items()},
        "required_cancers": required_cancers,
        "all_required_cancers_present": covered_cancers == required_cancers,
        "source_gene_rows": source_rows,
        "mapped_source_rows": mapped_rows,
        "ambiguous_gencode_symbols_excluded": ambiguous_symbols,
        "target_gene_entities": len(target_genes),
        "target_lncrna_entities": len(target_lncs),
        "covered_gene_entities": len(gene_coverage),
        "covered_lncrna_entities": len(lnc_coverage),
        "gene_entity_coverage_fraction": len(gene_coverage) / max(len(target_genes), 1),
        "lncrna_entity_coverage_fraction": len(lnc_coverage) / max(len(target_lncs), 1),
        "gene_nonzero_event_rows": event_counts["gene"],
        "lncrna_nonzero_event_rows": event_counts["lncrna"],
        "zero_values_omitted_from_sparse_events": zero_values,
        "nonzero_values": nonzero_values,
        "invalid_values": invalid_values,
        "config": asdict(settings),
        "inputs": {label: {"path": str(path), "sha256": artifact_sha256(path)} for label, path in paths.items()},
        "old_predictions_used": False,
        "formal_v32_artifacts_overwritten": False,
        "pilot_only": True,
    }
    audit_path = output / "AUDIT.json"
    _atomic_json(audit_path, audit)
    outputs: dict[str, Any] = {
        "audit": {"path": str(audit_path), "sha256": artifact_sha256(audit_path)},
        "patient_selection": {"path": str(selection_path), "sha256": artifact_sha256(selection_path)},
    }
    if not audit_only:
        outputs.update(
            {
                "gene_calls": {"path": str(gene_path), "sha256": artifact_sha256(gene_path)},
                "lncrna_calls": {"path": str(lnc_path), "sha256": artifact_sha256(lnc_path)},
                "entity_coverage": {"path": str(coverage_path), "sha256": artifact_sha256(coverage_path)},
            }
        )
    success = {
        "status": audit["status"],
        "analysis_version": ANALYSIS_VERSION,
        "module_id": MODULE_ID,
        "run_id": run_id,
        "outputs": outputs,
        "all_required_cancers_present": audit["all_required_cancers_present"],
        "trainable_bundle": not audit_only,
        "pilot_only": True,
        "formal_v32_primary_unchanged": True,
    }
    success_path = output / "SUCCESS.json"
    _atomic_json(success_path, success)
    return success


__all__ = [
    "ANALYSIS_VERSION",
    "GisticPreparationConfig",
    "MODULE_ID",
    "PanCancerCNVError",
    "prepare_pancancer_gistic_bundle",
]
