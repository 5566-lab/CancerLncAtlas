"""Collision-safe rematerialization of the V3.2 Evidence interaction authority.

The historical interaction builder discarded source-side Ensembl identifiers
for RNAInter and used many-to-one symbol dictionaries.  It also derived an
``interaction_id`` only from ``source_record_id`` even though the raw sources
contain duplicate record identifiers.  This module rebuilds the standardized
interaction table directly from immutable raw TSVs while preserving the old
record identifier for lineage.

No identifier is guessed.  Source Ensembl IDs have priority and names/aliases
are accepted only when they resolve to exactly one canonical entity.  Every
raw row receives a literal, collision-free positional identifier of the form
``INT32:<source-file-key>:<one-based-row-number>``.  A header-only Evidence
event table is emitted by contract so the same experimental occurrence is not
fed to the Evidence head twice.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .evidence_identifier_recovery import (
    EvidenceIdentifierRecoveryError,
    LncRNAAuthority,
    clean_text,
    fail_closed_resolution,
    load_lncrna_authority,
    normalized_name,
    sha256_file,
    stable_source_record_id,
    unversion_ensembl,
)


FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_INTERACTION_REMATERIALIZATION_V1"
EMPTY_EVENT_FORMAT = "CANCERLNCATLAS_V32_EMPTY_EVIDENCE_EVENT_AUTHORITY_V1"
_PMID_RE = re.compile(r"\d{6,9}")
_TOKEN_SPLIT_RE = re.compile(r"[|;,]+")


class EvidenceInteractionRematerializationError(RuntimeError):
    """Raised when a rematerialization authority or invariant fails."""


@dataclass(frozen=True)
class EntityResolution:
    identifier: str
    route: str
    candidate_count: int
    source_ensembl: str


@dataclass(frozen=True)
class GeneAuthority:
    canonical_ids: frozenset[str]
    ensembl: Mapping[str, frozenset[str]]
    names: Mapping[str, frozenset[str]]
    rows: int
    ambiguous_name_count: int


def _index_add(index: dict[str, set[str]], key: object, identifier: str) -> None:
    normalized = normalized_name(key)
    if normalized and identifier:
        index[normalized].add(identifier)


def load_gene_authority(path: str | Path) -> GeneAuthority:
    """Load a fail-closed gene/protein/miRNA identifier authority."""

    source = Path(path).resolve()
    if not source.is_file() or source.is_symlink():
        raise EvidenceInteractionRematerializationError(
            f"Missing or unsafe gene dimension: {source}"
        )
    with source.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"gene_id", "ensembl_gene_id", "gene_symbol", "gene_name", "aliases"}
        if reader.fieldnames is None or (missing := sorted(required - set(reader.fieldnames))):
            raise EvidenceInteractionRematerializationError(
                f"Gene dimension lacks required columns: {missing}"
            )
        canonical_ids: set[str] = set()
        ensembl_sets: dict[str, set[str]] = defaultdict(set)
        name_sets: dict[str, set[str]] = defaultdict(set)
        rows = 0
        for row in reader:
            rows += 1
            identifier = clean_text(row.get("gene_id"))
            if not identifier:
                raise EvidenceInteractionRematerializationError(
                    "Gene dimension contains an empty canonical ID"
                )
            if identifier in canonical_ids:
                raise EvidenceInteractionRematerializationError(
                    f"Gene dimension duplicates canonical ID: {identifier}"
                )
            canonical_ids.add(identifier)
            ensembl = unversion_ensembl(row.get("ensembl_gene_id"))
            if ensembl:
                ensembl_sets[ensembl].add(identifier)
            _index_add(name_sets, row.get("gene_symbol"), identifier)
            _index_add(name_sets, row.get("gene_name"), identifier)
            aliases = clean_text(row.get("aliases"))
            for alias in aliases.split(";") if aliases else ():
                _index_add(name_sets, alias, identifier)
    ensembl_map = {key: frozenset(value) for key, value in ensembl_sets.items()}
    name_map = {key: frozenset(value) for key, value in name_sets.items()}
    return GeneAuthority(
        canonical_ids=frozenset(canonical_ids),
        ensembl=ensembl_map,
        names=name_map,
        rows=rows,
        ambiguous_name_count=sum(len(value) > 1 for value in name_map.values()),
    )


def _candidate_union(
    authority: GeneAuthority, values: Iterable[object]
) -> set[str]:
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


def resolve_gene(
    authority: GeneAuthority,
    *,
    raw_name: object,
    source_identifier: object = "",
    source_aliases: Sequence[object] = (),
) -> EntityResolution:
    """Resolve a partner only when an authoritative route is unique."""

    source_ensembl = unversion_ensembl(source_identifier)
    if source_ensembl:
        candidates = set(authority.ensembl.get(source_ensembl, ()))
        if len(candidates) == 1:
            return EntityResolution(
                next(iter(candidates)), "authoritative_source_ensembl", 1, source_ensembl
            )
        if len(candidates) > 1:
            return EntityResolution(
                "", "ambiguous_source_ensembl", len(candidates), source_ensembl
            )
    raw = clean_text(raw_name)
    if raw in authority.canonical_ids:
        return EntityResolution(raw, "canonical_gene_id", 1, source_ensembl)
    candidates = _candidate_union(authority, (raw_name, *source_aliases))
    if len(candidates) == 1:
        return EntityResolution(
            next(iter(candidates)), "unique_name_or_alias", 1, source_ensembl
        )
    if len(candidates) > 1:
        return EntityResolution("", "ambiguous_name_or_alias", len(candidates), source_ensembl)
    return EntityResolution("", "unresolved", 0, source_ensembl)


@dataclass(frozen=True)
class RawSource:
    source_key: str
    database: str
    version: str
    relative_path: str
    experimental: bool
    predicted: bool
    kind: str


RAW_SOURCES = (
    RawSource(
        "NPINTER5",
        "NPInter",
        "5.0",
        "input/npinter5_human_lncRNA_interactions.comp.tsv",
        True,
        False,
        "npinter",
    ),
    RawSource(
        "RNAINTER4_EXPERIMENTAL",
        "RNAInter",
        "4.0-experimental",
        "input/rnainter4_human_experimental.tsv",
        True,
        False,
        "rnainter",
    ),
    RawSource(
        "RNAINTER4_PREDICTED",
        "RNAInter",
        "4.0-predicted",
        "input/rnainter4_human_predicted.tsv",
        False,
        True,
        "rnainter",
    ),
    RawSource(
        "LNCRNA2TARGET2_LOW",
        "LncRNA2Target",
        "2.0-low-throughput",
        "input/lncrna2target2_human_low_throughput.tsv",
        True,
        False,
        "lncrna2target",
    ),
    RawSource(
        "LNCTARD2",
        "LncTarD",
        "2.0",
        "input/lnctard2_curated_human.tsv",
        True,
        False,
        "lnctard",
    ),
    RawSource(
        "LNCACTDB4",
        "LncACTdb",
        "4.0",
        "input/lncactdb4_human_curated.tsv",
        True,
        False,
        "lncactdb",
    ),
)


OUTPUT_COLUMNS = (
    "interaction_id",
    "source_row_id",
    "source_row_index",
    "source_row_sha256",
    "source_sha256",
    "source_record_id",
    "historical_interaction_id",
    "original_record_id",
    "source_database",
    "source_dataset",
    "source_relative_path",
    "cancer_id",
    "lncrna_id",
    "lncrna_raw",
    "lncrna_source_identifier",
    "lncrna_source_aliases",
    "lncrna_mapping_route",
    "lncrna_mapping_candidate_count",
    "partner_id",
    "partner_raw",
    "partner_source_identifier",
    "partner_source_aliases",
    "partner_type",
    "partner_mapping_route",
    "partner_mapping_candidate_count",
    "mapping_status",
    "relation_type",
    "relation_raw",
    "direction",
    "experiment_family",
    "experiment_raw",
    "interaction_type",
    "throughput",
    "is_experimental",
    "is_predicted",
    "pmid",
    "species",
    "disease_raw",
    "cell_line",
    "tissue",
    "drug_raw",
    "orientation_status",
)


EMPTY_EVENT_COLUMNS = (
    "evidence_event_id",
    "lncrna_id",
    "partner_id",
    "pathway_id",
    "pathway_family_id",
    "cancer_id",
    "source_database",
    "source_dataset",
    "source_record_id",
    "pmid",
    "experiment_family",
    "relation_type",
    "direction",
    "is_experimental",
    "is_predicted",
    "source_row_sha256",
)


def relation_type(value: object) -> str:
    text = clean_text(value).lower()
    if "bind" in text or "interaction" in text or "sponge" in text or "cerna" in text:
        return "binding_or_interaction"
    if "regulat" in text or "target" in text:
        return "regulation"
    return "association"


def experiment_family(value: object) -> str:
    text = clean_text(value).lower()
    if re.search(r"clip|chirp|chart|rap|rip|pull.?down|immunoprecip|\bip\b|emsa", text):
        return "physical_binding"
    if re.search(r"knock|sirna|shrna|crispr|overexpress|transfect|deplet", text):
        return "functional_perturbation"
    if "luciferase" in text or "reporter" in text:
        return "reporter_assay"
    if re.search(r"qpcr|rt-pcr|rna-seq|microarray|western", text):
        return "expression_or_abundance"
    if not text:
        return "unspecified"
    return "other_experimental"


def _first(row: Mapping[str, object], columns: Sequence[str]) -> str:
    for column in columns:
        value = clean_text(row.get(column))
        if value:
            return value
    return ""


def _canonical_row_sha256(
    *, relative_path: str, row_number: int, row: Mapping[str, object]
) -> str:
    payload = {
        "relative_path": relative_path,
        "row_number": row_number,
        "raw_fields": {str(key): clean_text(value) for key, value in row.items()},
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _historical_interaction_id(source_record_id: str) -> str:
    return "INT:" + hashlib.sha1(source_record_id.encode("utf-8")).hexdigest()


def _orientation(
    row: Mapping[str, object],
    lnc_authority: LncRNAAuthority,
) -> tuple[int | None, str]:
    categories = [clean_text(row.get("Category1")).lower(), clean_text(row.get("Category2")).lower()]
    category_hits = [index for index, value in enumerate(categories) if value == "lncrna"]
    if len(category_hits) == 1:
        return category_hits[0], "explicit_single_lncrna_category"
    resolutions = [
        fail_closed_resolution(
            lnc_authority,
            raw_name=row.get(f"Interactor{index + 1}.Symbol"),
            source_ensembl=row.get(f"Raw_ID{index + 1}"),
            source_aliases=(row.get(f"Raw_ID{index + 1}"),),
        )
        for index in range(2)
    ]
    mapped = [index for index, resolution in enumerate(resolutions) if resolution.identifier]
    if len(mapped) == 1:
        return mapped[0], "inferred_unique_fail_closed_lncrna_mapping"
    if len(category_hits) > 1:
        return None, "ambiguous_multiple_lncrna_categories"
    if len(mapped) > 1:
        return None, "ambiguous_both_entities_map_to_lncrna"
    return None, "unresolved_no_lncrna_orientation"


def _standardize_raw(
    spec: RawSource,
    row: Mapping[str, object],
    *,
    row_number: int,
    first_column: str,
    lnc_authority: LncRNAAuthority,
) -> dict[str, Any]:
    """Extract raw semantics without performing canonical identifier mapping."""

    if spec.kind == "npinter":
        noncode_id = clean_text(row.get("RNA_NONCODE_ID"))
        return {
            "original_record_id": clean_text(row.get("ncRI_ID")),
            "lncrna_raw": clean_text(row.get("RNA_name")),
            "lncrna_source_identifier": noncode_id,
            "lncrna_aliases": (noncode_id,) if noncode_id else (),
            "partner_raw": clean_text(row.get("partner_name")),
            "partner_source_identifier": clean_text(row.get("partner_ID")),
            "partner_aliases": (),
            "partner_type": clean_text(row.get("partner_type")),
            "relation_raw": _first(row, ("interaction_type", "interaction_class")),
            "experiment_raw": clean_text(row.get("methods")),
            "disease_raw": "",
            "cell_line": clean_text(row.get("tissue_cell")),
            "tissue": clean_text(row.get("tissue_cell")),
            "drug_raw": "",
            "direction": "",
            "pmid": clean_text(row.get("PMIDs")),
            "throughput": clean_text(row.get("throughput_flag")),
            "species": clean_text(row.get("species")),
            "orientation_status": "explicit_npinter_lncrna_column",
        }
    if spec.kind == "rnainter":
        lnc_index, orientation_status = _orientation(row, lnc_authority)
        if lnc_index is None:
            # Preserve both raw entities in typed quarantine fields.  Empty lnc
            # input forces the canonical mapping to remain unavailable.
            lnc_raw = "|".join(
                filter(None, (clean_text(row.get("Interactor1.Symbol")), clean_text(row.get("Interactor2.Symbol"))))
            )
            lnc_source = "|".join(
                filter(None, (clean_text(row.get("Raw_ID1")), clean_text(row.get("Raw_ID2"))))
            )
            partner_index = None
            partner_raw = ""
            partner_source = ""
            partner_type = ""
            lnc_aliases: tuple[str, ...] = ()
        else:
            partner_index = 1 - lnc_index
            lnc_raw = clean_text(row.get(f"Interactor{lnc_index + 1}.Symbol"))
            lnc_source = clean_text(row.get(f"Raw_ID{lnc_index + 1}"))
            partner_raw = clean_text(row.get(f"Interactor{partner_index + 1}.Symbol"))
            partner_source = clean_text(row.get(f"Raw_ID{partner_index + 1}"))
            partner_type = clean_text(row.get(f"Category{partner_index + 1}"))
            lnc_aliases = (lnc_source,)
        method = (
            clean_text(row.get("predict"))
            if spec.predicted
            else _first(row, ("strong", "weak"))
        )
        species_column = f"Species{lnc_index + 1}" if lnc_index is not None else "Species1"
        return {
            "original_record_id": clean_text(row.get("RNAInterID")),
            "lncrna_raw": lnc_raw,
            "lncrna_source_identifier": lnc_source,
            "lncrna_aliases": lnc_aliases,
            "partner_raw": partner_raw,
            "partner_source_identifier": partner_source,
            "partner_aliases": (partner_source,) if partner_source else (),
            "partner_type": partner_type,
            "relation_raw": "predicted_interaction" if spec.predicted else "experimental_interaction",
            "experiment_raw": method,
            "disease_raw": "",
            "cell_line": "",
            "tissue": "",
            "drug_raw": "",
            "direction": "",
            "pmid": "",
            "throughput": "unspecified",
            "species": clean_text(row.get(species_column)),
            "orientation_status": orientation_status,
        }
    if spec.kind == "lncrna2target":
        return {
            "original_record_id": f"LOW:{row_number}",
            "lncrna_raw": _first(row, ("LncRNA_official_symbol", "lncRNA_name_from_paper")),
            "lncrna_source_identifier": clean_text(row.get("Ensembl_ID")),
            "lncrna_aliases": (row.get("lncRNA_name_from_paper"),),
            "partner_raw": _first(row, ("Target_official_symbol", "Target_symbol_from_paper")),
            "partner_source_identifier": clean_text(row.get("Target_entrez_gene_ID")),
            "partner_aliases": (row.get("Target_symbol_from_paper"),),
            "partner_type": "gene",
            "relation_raw": "regulation",
            "experiment_raw": clean_text(row.get("LncRNA_experiment")),
            "disease_raw": clean_text(row.get("Disease_state")),
            "cell_line": clean_text(row.get("Cell_Line")),
            "tissue": clean_text(row.get("Tissue_Origin")),
            "drug_raw": "",
            "direction": "",
            "pmid": clean_text(row.get("PMID")),
            "throughput": "low-throughput",
            "species": clean_text(row.get("Species")),
            "orientation_status": "explicit_lncrna2target_lncrna_column",
        }
    if spec.kind == "lnctard":
        return {
            "original_record_id": clean_text(row.get("RID")),
            "lncrna_raw": clean_text(row.get("Regulator")),
            "lncrna_source_identifier": clean_text(row.get("RegulatorEnsembleID")),
            "lncrna_aliases": (row.get("RegulatorAliases"),),
            "partner_raw": clean_text(row.get("Target")),
            "partner_source_identifier": _first(row, ("TargetEnsembleID", "TargetEntrezID")),
            "partner_aliases": (row.get("TargetAliases"),),
            "partner_type": clean_text(row.get("TargetType")),
            "relation_raw": _first(row, ("regulatoryType", "regulatoryMechanism")),
            "experiment_raw": clean_text(row.get("Experimental.method.for.lncRNA.target")),
            "disease_raw": _first(row, ("DiseaseName2", "DiseaseName")),
            "cell_line": "",
            "tissue": "",
            "drug_raw": clean_text(row.get("Drugs")),
            "direction": clean_text(row.get("RegulationDiretion")),
            "pmid": clean_text(row.get("PubMedID")),
            "throughput": "curated",
            "species": "Homo sapiens",
            "orientation_status": "explicit_lnctard_regulator_type",
        }
    if spec.kind == "lncactdb":
        gene = clean_text(row.get("gene"))
        mirna = clean_text(row.get("mir"))
        return {
            "original_record_id": clean_text(row.get(first_column)),
            "lncrna_raw": clean_text(row.get("lncrna")),
            "lncrna_source_identifier": "",
            "lncrna_aliases": (),
            "partner_raw": gene or mirna,
            "partner_source_identifier": clean_text(row.get("mirid")) if not gene else "",
            "partner_aliases": (),
            "partner_type": "gene" if gene else "miRNA",
            "relation_raw": "ceRNA",
            "experiment_raw": clean_text(row.get("experimental")),
            "disease_raw": clean_text(row.get("disease")),
            "cell_line": clean_text(row.get("tissue/cells")),
            "tissue": clean_text(row.get("tissue/cells")),
            "drug_raw": "",
            "direction": "",
            "pmid": clean_text(row.get("pubmed")),
            "throughput": "curated",
            "species": clean_text(row.get("species")),
            "orientation_status": "explicit_lncactdb_lncrna_column",
        }
    raise EvidenceInteractionRematerializationError(f"Unsupported source kind: {spec.kind}")


def _mapping_status(lnc_route: str, lnc_id: str, partner_route: str, partner_id: str) -> str:
    if not lnc_id:
        return (
            "ambiguous_lncrna"
            if lnc_route.startswith("ambiguous_")
            else "unresolved_lncrna"
        )
    if partner_id:
        return "lncrna_and_partner_mapped"
    return (
        "lncrna_mapped_partner_ambiguous"
        if partner_route.startswith("ambiguous_")
        else "lncrna_mapped_partner_unresolved"
    )


def _input_hash_authority(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS_RAW_TOKENS_RETAINED_FAIL_CLOSED_RECOVERY_QUANTIFIED":
        raise EvidenceInteractionRematerializationError(
            f"Recovery audit is not a passing authority: {path}"
        )
    authority: dict[str, str] = {}
    for item in payload.get("inputs", []):
        item_path = clean_text(item.get("path"))
        digest = clean_text(item.get("sha256")).lower()
        if item_path and digest:
            authority[str(Path(item_path).resolve()).lower()] = digest
    if not authority:
        raise EvidenceInteractionRematerializationError(
            f"Recovery audit contains no SHA-pinned inputs: {path}"
        )
    return authority


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def rematerialize_evidence_interactions(
    *,
    project_root: str | Path,
    output_root: str | Path,
    recovery_audit_path: str | Path | None = None,
    lnc_dimension_relative_path: str = "processed/dimensions/dim_lncRNA.tsv",
    gene_dimension_relative_path: str = "processed/dimensions/dim_gene.tsv",
    raw_sources: Sequence[RawSource] = RAW_SOURCES,
    compression_level: int = 1,
    progress_every: int = 250_000,
) -> dict[str, Any]:
    """Build a new, immutable, collision-safe interaction TSV authority."""

    root = Path(project_root).resolve()
    output = Path(output_root).resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if not root.is_dir() or root.is_symlink():
        raise EvidenceInteractionRematerializationError(f"Unsafe project root: {root}")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Rematerialization refuses output reuse: {output}")
    if not 0 <= compression_level <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    source_keys = [spec.source_key for spec in raw_sources]
    relative_paths = [spec.relative_path for spec in raw_sources]
    if len(source_keys) != len(set(source_keys)):
        raise EvidenceInteractionRematerializationError(
            "Raw source keys must be unique for collision-free positional IDs"
        )
    if len(relative_paths) != len(set(relative_paths)):
        raise EvidenceInteractionRematerializationError(
            "Raw source paths must be unique"
        )

    audit_path = Path(recovery_audit_path).resolve() if recovery_audit_path else None
    pinned_hashes = _input_hash_authority(audit_path)
    lnc_path = root / lnc_dimension_relative_path
    gene_path = root / gene_dimension_relative_path
    source_paths = [root / spec.relative_path for spec in raw_sources]
    for path in (lnc_path, gene_path, *source_paths):
        if not path.is_file() or path.is_symlink():
            raise EvidenceInteractionRematerializationError(f"Missing or unsafe input: {path}")

    input_rows: list[dict[str, Any]] = []
    for path in (lnc_path, gene_path, *source_paths):
        observed = sha256_file(path)
        expected = pinned_hashes.get(str(path.resolve()).lower())
        if pinned_hashes and expected is None and path != gene_path:
            raise EvidenceInteractionRematerializationError(
                f"Input is not declared by the recovery audit: {path}"
            )
        if expected is not None and observed != expected:
            raise EvidenceInteractionRematerializationError(
                f"Input SHA-256 drift: {observed} != {expected}: {path}"
            )
        input_rows.append(
            {
                "path": str(path),
                "bytes": int(path.stat().st_size),
                "sha256": observed,
                "expected_sha256": expected,
                "sha256_pinned": expected is not None,
            }
        )

    lnc_authority = load_lncrna_authority(lnc_path)
    gene_authority = load_gene_authority(gene_path)
    temporary.mkdir(parents=True)
    relation_path = temporary / "interaction_relation_recovered.tsv.gz"
    empty_event_path = temporary / "evidence_event_empty_by_contract.tsv"
    progress_path = temporary / "PROGRESS.json"
    global_counts: Counter[str] = Counter()
    per_source: list[dict[str, Any]] = []

    with gzip.open(
        relation_path,
        "wt",
        encoding="utf-8",
        newline="",
        compresslevel=compression_level,
    ) as relation_handle:
        writer = csv.DictWriter(
            relation_handle,
            fieldnames=list(OUTPUT_COLUMNS),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for spec in raw_sources:
            source_path = root / spec.relative_path
            source_sha = next(
                row["sha256"] for row in input_rows if row["path"] == str(source_path)
            )
            counts: Counter[str] = Counter()
            lnc_routes: Counter[str] = Counter()
            partner_routes: Counter[str] = Counter()
            orientation_routes: Counter[str] = Counter()
            with source_path.open(
                "r", encoding="utf-8-sig", errors="replace", newline=""
            ) as source_handle:
                reader = csv.DictReader(source_handle, delimiter="\t")
                if not reader.fieldnames:
                    raise EvidenceInteractionRematerializationError(
                        f"Raw source has no header: {source_path}"
                    )
                first_column = reader.fieldnames[0]
                for row_number, raw_row in enumerate(reader, start=1):
                    extracted = _standardize_raw(
                        spec,
                        raw_row,
                        row_number=row_number,
                        first_column=first_column,
                        lnc_authority=lnc_authority,
                    )
                    original_record_id = clean_text(extracted["original_record_id"])
                    source_record_id = stable_source_record_id(
                        spec.database, original_record_id
                    )
                    interaction_id = f"INT32:{spec.source_key}:{row_number}"
                    source_row_id = f"SRCROW32:{spec.source_key}:{row_number}"
                    row_sha = _canonical_row_sha256(
                        relative_path=spec.relative_path,
                        row_number=row_number,
                        row=raw_row,
                    )
                    if extracted["orientation_status"].startswith(("ambiguous_", "unresolved_")):
                        lnc_resolution = fail_closed_resolution(
                            lnc_authority, raw_name="", source_ensembl=""
                        )
                        lnc_route = extracted["orientation_status"]
                    else:
                        lnc_resolution = fail_closed_resolution(
                            lnc_authority,
                            raw_name=extracted["lncrna_raw"],
                            source_ensembl=extracted["lncrna_source_identifier"],
                            source_aliases=tuple(extracted["lncrna_aliases"]),
                        )
                        lnc_route = lnc_resolution.route
                    partner_resolution = resolve_gene(
                        gene_authority,
                        raw_name=extracted["partner_raw"],
                        source_identifier=extracted["partner_source_identifier"],
                        source_aliases=tuple(extracted["partner_aliases"]),
                    )
                    status = _mapping_status(
                        lnc_route,
                        lnc_resolution.identifier,
                        partner_resolution.route,
                        partner_resolution.identifier,
                    )
                    normalized_relation = relation_type(extracted["relation_raw"])
                    normalized_experiment = experiment_family(extracted["experiment_raw"])
                    output_row = {
                        "interaction_id": interaction_id,
                        "source_row_id": source_row_id,
                        "source_row_index": row_number,
                        "source_row_sha256": row_sha,
                        "source_sha256": source_sha,
                        "source_record_id": source_record_id,
                        "historical_interaction_id": _historical_interaction_id(source_record_id),
                        "original_record_id": original_record_id,
                        "source_database": spec.database,
                        "source_dataset": spec.version,
                        "source_relative_path": spec.relative_path,
                        "cancer_id": "PAN_CANCER",
                        "lncrna_id": lnc_resolution.identifier,
                        "lncrna_raw": extracted["lncrna_raw"],
                        "lncrna_source_identifier": extracted["lncrna_source_identifier"],
                        "lncrna_source_aliases": ";".join(
                            clean_text(value)
                            for value in extracted["lncrna_aliases"]
                            if clean_text(value)
                        ),
                        "lncrna_mapping_route": lnc_route,
                        "lncrna_mapping_candidate_count": lnc_resolution.candidate_count,
                        "partner_id": partner_resolution.identifier,
                        "partner_raw": extracted["partner_raw"],
                        "partner_source_identifier": extracted["partner_source_identifier"],
                        "partner_source_aliases": ";".join(
                            clean_text(value)
                            for value in extracted["partner_aliases"]
                            if clean_text(value)
                        ),
                        "partner_type": extracted["partner_type"],
                        "partner_mapping_route": partner_resolution.route,
                        "partner_mapping_candidate_count": partner_resolution.candidate_count,
                        "mapping_status": status,
                        "relation_type": normalized_relation,
                        "relation_raw": extracted["relation_raw"],
                        "direction": extracted["direction"],
                        "experiment_family": normalized_experiment,
                        "experiment_raw": extracted["experiment_raw"],
                        "interaction_type": normalized_relation,
                        "throughput": extracted["throughput"],
                        "is_experimental": str(bool(spec.experimental)).lower(),
                        "is_predicted": str(bool(spec.predicted)).lower(),
                        "pmid": extracted["pmid"],
                        "species": extracted["species"],
                        "disease_raw": extracted["disease_raw"],
                        "cell_line": extracted["cell_line"],
                        "tissue": extracted["tissue"],
                        "drug_raw": extracted["drug_raw"],
                        "orientation_status": extracted["orientation_status"],
                    }
                    writer.writerow(output_row)
                    counts["rows"] += 1
                    counts["empty_original_record_id"] += not bool(original_record_id)
                    counts["lncrna_mapped"] += bool(lnc_resolution.identifier)
                    counts["lncrna_unresolved_or_ambiguous"] += not bool(
                        lnc_resolution.identifier
                    )
                    counts["partner_mapped"] += bool(partner_resolution.identifier)
                    counts["partner_unresolved_or_ambiguous"] += not bool(
                        partner_resolution.identifier
                    )
                    counts["lncrna_and_partner_mapped"] += bool(
                        lnc_resolution.identifier and partner_resolution.identifier
                    )
                    counts["experimental_rows"] += bool(spec.experimental)
                    counts["predicted_rows"] += bool(spec.predicted)
                    counts["rows_with_pmid"] += bool(_PMID_RE.search(extracted["pmid"]))
                    lnc_routes[lnc_route] += 1
                    partner_routes[partner_resolution.route] += 1
                    orientation_routes[extracted["orientation_status"]] += 1
                    if progress_every > 0 and counts["rows"] % progress_every == 0:
                        _write_json(
                            progress_path,
                            {
                                "status": "RUNNING",
                                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                                "source_key": spec.source_key,
                                "source_rows": counts["rows"],
                                "global_rows_before_current_source": global_counts["rows"],
                            },
                        )
            global_counts.update(counts)
            per_source.append(
                {
                    "source_key": spec.source_key,
                    "source_database": spec.database,
                    "source_dataset": spec.version,
                    "relative_path": spec.relative_path,
                    "counts": dict(sorted(counts.items())),
                    "lncrna_mapping_routes": dict(sorted(lnc_routes.items())),
                    "partner_mapping_routes": dict(sorted(partner_routes.items())),
                    "orientation_routes": dict(sorted(orientation_routes.items())),
                }
            )

    with empty_event_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerow(
            EMPTY_EVENT_COLUMNS
        )
    if progress_path.exists():
        progress_path.unlink()

    relation_artifact = _artifact(relation_path)
    relation_artifact["path"] = str(output / relation_path.name)
    event_artifact = _artifact(empty_event_path)
    event_artifact["path"] = str(output / empty_event_path.name)
    report: dict[str, Any] = {
        "format": FORMAT,
        "status": "PASS_REMATERIALIZED_REQUIRES_INDEPENDENT_VALIDATION",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "output_root": str(output),
        "formal_training_started": False,
        "production_deployed": False,
        "production_port_8260_touched": False,
        "contract": {
            "raw_source_and_lncrna_sha256_pinned": bool(pinned_hashes),
            "gene_authority_sha256_recorded_in_manifest": True,
            "historical_source_record_id_preserved": True,
            "collision_free_positional_interaction_id": True,
            "source_row_content_sha256_present": True,
            "ambiguous_identifier_mapping_forbidden": True,
            "blank_identifier_imputation_forbidden": True,
            "family_to_exact_pathway_broadcast": False,
            "evidence_event_is_header_only_by_contract": True,
            "interaction_relation_is_single_occurrence_authority": True,
            "historical_outputs_overwritten": False,
            "independent_validation_complete": False,
        },
        "authorities": {
            "lncrna_rows": lnc_authority.rows,
            "lncrna_ambiguous_name_keys": lnc_authority.ambiguous_name_count,
            "gene_rows": gene_authority.rows,
            "gene_ambiguous_name_keys": gene_authority.ambiguous_name_count,
            "recovery_audit_path": str(audit_path) if audit_path else None,
            "recovery_audit_sha256": sha256_file(audit_path) if audit_path else None,
        },
        "inputs": input_rows,
        "counts": dict(sorted(global_counts.items())),
        "per_source": per_source,
        "outputs": {
            "interaction_relation": relation_artifact,
            "evidence_event": event_artifact,
        },
    }
    manifest_path = temporary / "MANIFEST.json"
    _write_json(manifest_path, report)
    manifest_artifact = _artifact(manifest_path)
    manifest_artifact["path"] = str(output / manifest_path.name)
    success = {
        "format": FORMAT,
        "status": report["status"],
        "generated_at_utc": report["generated_at_utc"],
        "manifest": manifest_artifact,
        "row_count": int(global_counts["rows"]),
        "formal_training_started": False,
        "production_deployed": False,
        "production_port_8260_touched": False,
    }
    _write_json(temporary / "SUCCESS.json", success)
    os.replace(temporary, output)
    report["output_root"] = str(output)
    return report


__all__ = [
    "EMPTY_EVENT_COLUMNS",
    "EMPTY_EVENT_FORMAT",
    "EntityResolution",
    "EvidenceInteractionRematerializationError",
    "FORMAT",
    "GeneAuthority",
    "OUTPUT_COLUMNS",
    "RAW_SOURCES",
    "RawSource",
    "experiment_family",
    "load_gene_authority",
    "relation_type",
    "rematerialize_evidence_interactions",
    "resolve_gene",
]
