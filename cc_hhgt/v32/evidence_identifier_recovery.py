"""Fail-closed recovery audit for raw lncRNA identifiers in Evidence sources.

The historical interaction builder retained raw source tokens in
``source_record.parquet`` but wrote an empty canonical lncRNA identifier when
its small symbol/alias dictionary did not resolve a record.  This module
replays only the identifier decision from the immutable raw TSV authorities.
It never invents an identifier: source Ensembl IDs are authoritative, while
names/aliases are accepted only when every matched token identifies the same
lncRNA.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence


AUDIT_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_RAW_IDENTIFIER_RECOVERY_AUDIT_V1"
DEFAULT_CHUNK_ROWS = 250_000
_ENSEMBL_RE = re.compile(r"ENSG\d{6,}", flags=re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")
_TOKEN_SPLIT_RE = re.compile(r"[|;,]+")


class EvidenceIdentifierRecoveryError(RuntimeError):
    """Raised when a raw Evidence authority or recovery invariant fails."""


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = _SPACE_RE.sub(" ", str(value)).strip()
    return "" if text.lower() in {"nan", "na", "n/a", "none", "null", "-"} else text


def normalized_name(value: object) -> str:
    return _NON_ALNUM_RE.sub("", clean_text(value).upper())


def unversion_ensembl(value: object) -> str:
    text = clean_text(value).upper()
    match = _ENSEMBL_RE.search(text)
    return match.group(0).upper() if match else ""


def stable_source_record_id(source_database: object, original_record_id: object) -> str:
    payload = "\x1f".join((clean_text(source_database), clean_text(original_record_id)))
    return "SRC:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _index_add(index: dict[str, set[str]], key: object, identifier: object) -> None:
    normalized = normalized_name(key)
    canonical = clean_text(identifier)
    if normalized and canonical:
        index[normalized].add(canonical)


@dataclass(frozen=True)
class LncRNAAuthority:
    canonical_ids: frozenset[str]
    ensembl: Mapping[str, frozenset[str]]
    names: Mapping[str, frozenset[str]]
    historical_symbol: Mapping[str, str]
    historical_ensembl: Mapping[str, str]
    ambiguous_name_count: int
    rows: int


def load_lncrna_authority(path: str | Path) -> LncRNAAuthority:
    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise EvidenceIdentifierRecoveryError(f"Unsafe lncRNA dimension: {source}")
    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"lncrna_id", "ensembl_gene_id", "gene_symbol", "gene_name", "aliases"}
        if reader.fieldnames is None or (missing := sorted(required - set(reader.fieldnames))):
            raise EvidenceIdentifierRecoveryError(f"lncRNA dimension lacks columns: {missing}")
        rows = list(reader)
    canonical_ids: set[str] = set()
    ensembl_sets: dict[str, set[str]] = defaultdict(set)
    name_sets: dict[str, set[str]] = defaultdict(set)
    historical_symbol: dict[str, str] = {}
    historical_ensembl: dict[str, str] = {}
    for row in rows:
        identifier = clean_text(row.get("lncrna_id"))
        if not identifier:
            raise EvidenceIdentifierRecoveryError("lncRNA dimension contains an empty canonical ID")
        canonical_ids.add(identifier)
        ensembl = unversion_ensembl(row.get("ensembl_gene_id"))
        if ensembl:
            ensembl_sets[ensembl].add(identifier)
            historical_ensembl[ensembl] = identifier
        symbol = normalized_name(row.get("gene_symbol"))
        if symbol:
            name_sets[symbol].add(identifier)
            # Exact reproduction of the historical dict-comprehension policy:
            # duplicate primary symbols silently kept the last row.
            historical_symbol[symbol] = identifier
        _index_add(name_sets, row.get("gene_name"), identifier)
        aliases = clean_text(row.get("aliases"))
        for alias in aliases.split(";") if aliases else ():
            normalized = normalized_name(alias)
            if normalized:
                name_sets[normalized].add(identifier)
                # Historical aliases used setdefault and could never override a
                # primary-symbol mapping.
                historical_symbol.setdefault(normalized, identifier)
    if len(canonical_ids) != len(rows):
        raise EvidenceIdentifierRecoveryError("lncRNA dimension duplicates canonical IDs")
    ensembl = {key: frozenset(value) for key, value in ensembl_sets.items()}
    names = {key: frozenset(value) for key, value in name_sets.items()}
    return LncRNAAuthority(
        canonical_ids=frozenset(canonical_ids),
        ensembl=ensembl,
        names=names,
        historical_symbol=historical_symbol,
        historical_ensembl=historical_ensembl,
        ambiguous_name_count=sum(len(value) > 1 for value in names.values()),
        rows=len(rows),
    )


@dataclass(frozen=True)
class Resolution:
    identifier: str
    route: str
    candidate_count: int
    source_ensembl: str


def historical_resolution(
    authority: LncRNAAuthority, *, raw_name: object, source_ensembl: object = ""
) -> Resolution:
    ensembl = unversion_ensembl(source_ensembl)
    identifier = authority.historical_ensembl.get(ensembl, "") if ensembl else ""
    if identifier:
        return Resolution(identifier, "historical_source_ensembl", 1, ensembl)
    identifier = authority.historical_symbol.get(normalized_name(raw_name), "")
    return Resolution(
        identifier,
        "historical_symbol_or_alias" if identifier else "historical_unresolved",
        1 if identifier else 0,
        ensembl,
    )


def _name_candidates(authority: LncRNAAuthority, values: Iterable[object]) -> set[str]:
    candidates: set[str] = set()
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        normalized = normalized_name(text)
        candidates.update(authority.names.get(normalized, ()))
        for token in _TOKEN_SPLIT_RE.split(text):
            token_normalized = normalized_name(token)
            if token_normalized and token_normalized != normalized:
                candidates.update(authority.names.get(token_normalized, ()))
    return candidates


def fail_closed_resolution(
    authority: LncRNAAuthority,
    *,
    raw_name: object,
    source_ensembl: object = "",
    source_aliases: Sequence[object] = (),
) -> Resolution:
    ensembl = unversion_ensembl(source_ensembl)
    if ensembl:
        candidates = set(authority.ensembl.get(ensembl, ()))
        if len(candidates) == 1:
            return Resolution(next(iter(candidates)), "authoritative_source_ensembl", 1, ensembl)
        if len(candidates) > 1:
            return Resolution("", "ambiguous_source_ensembl", len(candidates), ensembl)
    raw = clean_text(raw_name)
    if raw in authority.canonical_ids:
        return Resolution(raw, "canonical_lncrna_id", 1, ensembl)
    candidates = _name_candidates(authority, (raw_name, *source_aliases))
    if len(candidates) == 1:
        return Resolution(next(iter(candidates)), "unique_name_or_alias", 1, ensembl)
    if len(candidates) > 1:
        return Resolution("", "ambiguous_name_or_alias", len(candidates), ensembl)
    return Resolution("", "unresolved", 0, ensembl)


@dataclass(frozen=True)
class SourceSpec:
    database: str
    relative_path: str
    id_column: str
    name_columns: tuple[str, ...]
    ensembl_columns: tuple[str, ...] = ()
    alias_columns: tuple[str, ...] = ()
    category_columns: tuple[str, ...] = ()
    partner_name_columns: tuple[str, ...] = ()
    raw_id_columns: tuple[str, ...] = ()
    pmid_columns: tuple[str, ...] = ()
    experimental: bool = True
    rnainter_orientation: bool = False
    historical_ensembl_used: bool = True


SOURCE_SPECS = (
    SourceSpec(
        "NPInter",
        "input/npinter5_human_lncRNA_interactions.comp.tsv",
        "ncRI_ID",
        ("RNA_name",),
        alias_columns=("RNA_NONCODE_ID",),
        pmid_columns=("PMIDs",),
    ),
    SourceSpec(
        "RNAInter",
        "input/rnainter4_human_experimental.tsv",
        "RNAInterID",
        ("Interactor1.Symbol", "Interactor2.Symbol"),
        category_columns=("Category1", "Category2"),
        raw_id_columns=("Raw_ID1", "Raw_ID2"),
        experimental=True,
        rnainter_orientation=True,
        historical_ensembl_used=False,
    ),
    SourceSpec(
        "RNAInter",
        "input/rnainter4_human_predicted.tsv",
        "RNAInterID",
        ("Interactor1.Symbol", "Interactor2.Symbol"),
        category_columns=("Category1", "Category2"),
        raw_id_columns=("Raw_ID1", "Raw_ID2"),
        experimental=False,
        rnainter_orientation=True,
        historical_ensembl_used=False,
    ),
    SourceSpec(
        "LncRNA2Target",
        "input/lncrna2target2_human_low_throughput.tsv",
        "__ROW_NUMBER__",
        ("LncRNA_official_symbol", "lncRNA_name_from_paper"),
        ensembl_columns=("Ensembl_ID",),
        pmid_columns=("PMID",),
    ),
    SourceSpec(
        "LncTarD",
        "input/lnctard2_curated_human.tsv",
        "RID",
        ("Regulator",),
        ensembl_columns=("RegulatorEnsembleID",),
        alias_columns=("RegulatorAliases",),
        pmid_columns=("PubMedID",),
    ),
    SourceSpec(
        "LncACTdb",
        "input/lncactdb4_human_curated.tsv",
        "__FIRST_COLUMN__",
        ("lncrna",),
        pmid_columns=("pubmed",),
    ),
)


def _first_nonempty(row: Mapping[str, object], columns: Sequence[str]) -> str:
    for column in columns:
        value = clean_text(row.get(column))
        if value:
            return value
    return ""


def _source_fields(
    spec: SourceSpec, row: Mapping[str, object], *, row_number: int, first_column: str
) -> tuple[str, str, str, tuple[str, ...], str]:
    if spec.id_column == "__ROW_NUMBER__":
        record_id = f"LOW:{row_number}"
    elif spec.id_column == "__FIRST_COLUMN__":
        record_id = clean_text(row.get(first_column))
    else:
        record_id = clean_text(row.get(spec.id_column))
    if spec.rnainter_orientation:
        categories = [clean_text(row.get(column)) for column in spec.category_columns]
        lnc_index = 0 if categories and categories[0] == "lncRNA" else 1
        raw_name = clean_text(row.get(spec.name_columns[lnc_index]))
        raw_id = clean_text(row.get(spec.raw_id_columns[lnc_index]))
        source_ensembl = raw_id
        aliases = (raw_id,)
    else:
        raw_name = _first_nonempty(row, spec.name_columns)
        source_ensembl = _first_nonempty(row, spec.ensembl_columns)
        aliases = tuple(clean_text(row.get(column)) for column in spec.alias_columns)
    pmid = _first_nonempty(row, spec.pmid_columns)
    return record_id, raw_name, source_ensembl, aliases, pmid


def _artifact(path: Path, *, with_sha256: bool) -> dict[str, Any]:
    stat = path.stat()
    value: dict[str, Any] = {
        "path": str(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if with_sha256:
        value["sha256"] = sha256_file(path)
    return value


def audit_raw_identifier_recovery(
    *,
    project_root: str | Path,
    output_root: str | Path,
    dimension_relative_path: str = "processed/dimensions/dim_lncRNA.tsv",
    source_specs: Sequence[SourceSpec] = SOURCE_SPECS,
    top_unresolved: int = 250,
    hash_inputs: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = Path(output_root).resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if not root.is_dir() or root.is_symlink():
        raise EvidenceIdentifierRecoveryError(f"Unsafe project root: {root}")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Recovery audit refuses output reuse: {output}")
    dimension = root / dimension_relative_path
    authority = load_lncrna_authority(dimension)
    per_source: list[dict[str, Any]] = []
    unresolved_counter: Counter[tuple[str, str, str]] = Counter()
    recovered_counter: Counter[tuple[str, str, str, str]] = Counter()
    all_source_ids: set[str] = set()
    input_records = [_artifact(dimension, with_sha256=hash_inputs)]
    totals: Counter[str] = Counter()

    for spec in source_specs:
        source = root / spec.relative_path
        if not source.is_file() or source.is_symlink():
            raise EvidenceIdentifierRecoveryError(f"Missing/unsafe raw source: {source}")
        input_records.append(_artifact(source, with_sha256=hash_inputs))
        counts: Counter[str] = Counter()
        for key in (
            "rows",
            "empty_original_record_id",
            "duplicate_source_record_id",
            "raw_token_present",
            "experimental_rows",
            "experimental_rows_with_pmid",
            "historical_mapped",
            "historical_unresolved",
            "fail_closed_mapped",
            "newly_recoverable",
            "historical_mapping_changed_or_rejected",
            "still_unresolved_or_ambiguous",
        ):
            counts[key] = 0
        route_counts: Counter[str] = Counter()
        with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames:
                raise EvidenceIdentifierRecoveryError(f"Raw source has no header: {source}")
            first_column = reader.fieldnames[0]
            for row_number, row in enumerate(reader, start=1):
                counts["rows"] += 1
                record_id, raw_name, source_ensembl, aliases, pmid = _source_fields(
                    spec, row, row_number=row_number, first_column=first_column
                )
                if not record_id:
                    counts["empty_original_record_id"] += 1
                    continue
                source_record_id = stable_source_record_id(spec.database, record_id)
                if source_record_id in all_source_ids:
                    counts["duplicate_source_record_id"] += 1
                all_source_ids.add(source_record_id)
                counts["raw_token_present"] += bool(raw_name)
                counts["experimental_rows"] += bool(spec.experimental)
                counts["experimental_rows_with_pmid"] += bool(
                    spec.experimental and re.search(r"\d{6,9}", pmid)
                )
                historical = historical_resolution(
                    authority,
                    raw_name=raw_name,
                    source_ensembl=(source_ensembl if spec.historical_ensembl_used else ""),
                )
                recovered = fail_closed_resolution(
                    authority,
                    raw_name=raw_name,
                    source_ensembl=source_ensembl,
                    source_aliases=aliases,
                )
                counts["historical_mapped"] += bool(historical.identifier)
                counts["historical_unresolved"] += not historical.identifier
                counts["fail_closed_mapped"] += bool(recovered.identifier)
                route_counts[recovered.route] += 1
                if not historical.identifier and recovered.identifier:
                    counts["newly_recoverable"] += 1
                    recovered_counter[(spec.database, raw_name, source_ensembl, recovered.route)] += 1
                elif historical.identifier and recovered.identifier != historical.identifier:
                    counts["historical_mapping_changed_or_rejected"] += 1
                if not recovered.identifier:
                    counts["still_unresolved_or_ambiguous"] += 1
                    unresolved_counter[(spec.database, raw_name, source_ensembl)] += 1
        per_source.append(
            {
                "source_database": spec.database,
                "relative_path": spec.relative_path,
                "experimental": spec.experimental,
                "counts": dict(sorted(counts.items())),
                "resolution_routes": dict(sorted(route_counts.items())),
            }
        )
        totals.update(counts)

    totals["unique_source_record_ids"] = len(all_source_ids)
    temporary.mkdir(parents=True)
    unresolved_path = temporary / "TOP_UNRESOLVED_RAW_TOKENS.tsv"
    with unresolved_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("source_database", "lncrna_raw", "source_identifier", "rows"))
        for (database, raw_name, source_id), count in unresolved_counter.most_common(top_unresolved):
            writer.writerow((database, raw_name, source_id, count))
    recovered_path = temporary / "TOP_NEWLY_RECOVERABLE_RAW_TOKENS.tsv"
    with recovered_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ("source_database", "lncrna_raw", "source_identifier", "recovery_route", "rows")
        )
        for (database, raw_name, source_id, route), count in recovered_counter.most_common(
            top_unresolved
        ):
            writer.writerow((database, raw_name, source_id, route, count))
    report: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "status": "PASS_RAW_TOKENS_RETAINED_FAIL_CLOSED_RECOVERY_QUANTIFIED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "formal_training_started": False,
        "production_deployed": False,
        "production_port_8260_touched": False,
        "contract": {
            "raw_token_required": True,
            "blank_identifier_imputation_forbidden": True,
            "ambiguous_name_mapping_forbidden": True,
            "source_ensembl_has_priority": True,
            "family_to_exact_pathway_broadcast": False,
            "historical_output_reuse": False,
            "duplicate_source_record_ids_are_reported_not_silently_overwritten": True,
        },
        "lncrna_authority": {
            "rows": authority.rows,
            "canonical_ids": len(authority.canonical_ids),
            "unique_ensembl_keys": sum(len(value) == 1 for value in authority.ensembl.values()),
            "unique_name_or_alias_keys": sum(len(value) == 1 for value in authority.names.values()),
            "ambiguous_name_or_alias_keys": authority.ambiguous_name_count,
        },
        "inputs": input_records,
        "counts": dict(sorted(totals.items())),
        "per_source": per_source,
        "artifacts": {
            "top_unresolved": _artifact(unresolved_path, with_sha256=True),
            "top_newly_recoverable": _artifact(recovered_path, with_sha256=True),
        },
        "interpretation": {
            "raw_data_missing": False,
            "historical_empty_ids_are_processing_output": True,
            "newly_recoverable_rows_may_be_rematerialized": totals["newly_recoverable"] > 0,
            "still_unresolved_rows_must_remain_typed_unavailable": True,
            "duplicate_source_record_ids_require_collision_safe_rematerialization": (
                totals["duplicate_source_record_id"] > 0
            ),
            "audit_is_rebuild_or_training": False,
        },
    }
    report_path = temporary / "AUDIT.json"
    _atomic_json(report_path, report)
    success = {
        "format": "CANCERLNCATLAS_V32_EVIDENCE_RAW_IDENTIFIER_RECOVERY_SUCCESS_V1",
        "status": "SUCCESS_AUDIT_ONLY_NOT_REBUILT",
        "audit": _artifact(report_path, with_sha256=True),
        "formal_training_started": False,
        "production_deployed": False,
        "production_port_8260_touched": False,
    }
    _atomic_json(temporary / "SUCCESS.json", success)
    os.replace(temporary, output)
    return {**report, "output_root": str(output)}


__all__ = [
    "AUDIT_FORMAT",
    "EvidenceIdentifierRecoveryError",
    "LncRNAAuthority",
    "Resolution",
    "SOURCE_SPECS",
    "SourceSpec",
    "audit_raw_identifier_recovery",
    "clean_text",
    "fail_closed_resolution",
    "historical_resolution",
    "load_lncrna_authority",
    "normalized_name",
    "sha256_file",
    "stable_source_record_id",
    "unversion_ensembl",
]
