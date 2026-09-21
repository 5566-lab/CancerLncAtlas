#!/usr/bin/env python3
"""Independently validate a recovered V3.2 Evidence interaction authority.

This validator deliberately does not import the production rematerialization
module.  It re-reads every immutable raw row, independently reconstructs the
raw semantic fields and fail-closed identifier decision, and checks the
collision-safe positional IDs and raw-row hashes one by one.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_INTERACTION_REMATERIALIZATION_V1"
VALIDATION_FORMAT = "CANCERLNCATLAS_V32_EVIDENCE_INTERACTION_INDEPENDENT_VALIDATION_V1"
PASS_STATUS = "PASS_INDEPENDENT_REMATERIALIZATION_VALIDATION"
_SPACE = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^A-Z0-9]+")
_ENSEMBL = re.compile(r"ENSG\d{6,}", re.IGNORECASE)
_SPLIT = re.compile(r"[|;,]+")
_PMID = re.compile(r"\d{6,9}")


class ValidationError(RuntimeError):
    pass


def clean(value: object) -> str:
    if value is None:
        return ""
    text = _SPACE.sub(" ", str(value)).strip()
    return "" if text.lower() in {"nan", "na", "n/a", "none", "null", "-"} else text


def norm(value: object) -> str:
    return _NON_ALNUM.sub("", clean(value).upper())


def ens(value: object) -> str:
    match = _ENSEMBL.search(clean(value).upper())
    return match.group(0).upper() if match else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_record_id(database: str, original_record_id: str) -> str:
    payload = "\x1f".join((clean(database), clean(original_record_id)))
    return "SRC:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def raw_row_sha(relative_path: str, row_number: int, row: Mapping[str, object]) -> str:
    payload = {
        "relative_path": relative_path,
        "row_number": row_number,
        "raw_fields": {str(key): clean(value) for key, value in row.items()},
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Authority:
    canonical: frozenset[str]
    ensembl: Mapping[str, frozenset[str]]
    names: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class Resolution:
    identifier: str
    route: str
    candidates: int


def load_authority(path: Path, id_column: str) -> Authority:
    canonical: set[str] = set()
    ensembl_map: dict[str, set[str]] = defaultdict(set)
    name_map: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {id_column, "ensembl_gene_id", "gene_symbol", "gene_name", "aliases"}
        if reader.fieldnames is None or (missing := sorted(required - set(reader.fieldnames))):
            raise ValidationError(f"Dimension lacks columns {missing}: {path}")
        for row in reader:
            identifier = clean(row.get(id_column))
            if not identifier or identifier in canonical:
                raise ValidationError(f"Empty/duplicate canonical ID in {path}: {identifier!r}")
            canonical.add(identifier)
            ensembl = ens(row.get("ensembl_gene_id"))
            if ensembl:
                ensembl_map[ensembl].add(identifier)
            for value in (row.get("gene_symbol"), row.get("gene_name")):
                key = norm(value)
                if key:
                    name_map[key].add(identifier)
            aliases = clean(row.get("aliases"))
            for alias in aliases.split(";") if aliases else ():
                key = norm(alias)
                if key:
                    name_map[key].add(identifier)
    return Authority(
        frozenset(canonical),
        {key: frozenset(value) for key, value in ensembl_map.items()},
        {key: frozenset(value) for key, value in name_map.items()},
    )


def resolve(
    authority: Authority,
    *,
    raw_name: object,
    source_identifier: object,
    aliases: Sequence[object],
    canonical_route: str,
) -> Resolution:
    source_ensembl = ens(source_identifier)
    if source_ensembl:
        candidates = set(authority.ensembl.get(source_ensembl, ()))
        if len(candidates) == 1:
            return Resolution(next(iter(candidates)), "authoritative_source_ensembl", 1)
        if len(candidates) > 1:
            return Resolution("", "ambiguous_source_ensembl", len(candidates))
    raw = clean(raw_name)
    if raw in authority.canonical:
        return Resolution(raw, canonical_route, 1)
    candidates: set[str] = set()
    for value in (raw_name, *aliases):
        text = clean(value)
        if not text:
            continue
        candidates.update(authority.names.get(norm(text), ()))
        for token in _SPLIT.split(text):
            token_key = norm(token)
            if token_key and token_key != norm(text):
                candidates.update(authority.names.get(token_key, ()))
    if len(candidates) == 1:
        return Resolution(next(iter(candidates)), "unique_name_or_alias", 1)
    if len(candidates) > 1:
        return Resolution("", "ambiguous_name_or_alias", len(candidates))
    return Resolution("", "unresolved", 0)


def first(row: Mapping[str, object], columns: Sequence[str]) -> str:
    for column in columns:
        value = clean(row.get(column))
        if value:
            return value
    return ""


def normalized_relation(value: object) -> str:
    text = clean(value).lower()
    if "bind" in text or "interaction" in text or "sponge" in text or "cerna" in text:
        return "binding_or_interaction"
    if "regulat" in text or "target" in text:
        return "regulation"
    return "association"


def normalized_experiment(value: object) -> str:
    text = clean(value).lower()
    if re.search(r"clip|chirp|chart|rap|rip|pull.?down|immunoprecip|\bip\b|emsa", text):
        return "physical_binding"
    if re.search(r"knock|sirna|shrna|crispr|overexpress|transfect|deplet", text):
        return "functional_perturbation"
    if "luciferase" in text or "reporter" in text:
        return "reporter_assay"
    if re.search(r"qpcr|rt-pcr|rna-seq|microarray|western", text):
        return "expression_or_abundance"
    return "unspecified" if not text else "other_experimental"


@dataclass(frozen=True)
class Source:
    key: str
    database: str
    dataset: str
    relative_path: str
    experimental: bool
    predicted: bool
    kind: str


SOURCES = (
    Source("NPINTER5", "NPInter", "5.0", "input/npinter5_human_lncRNA_interactions.comp.tsv", True, False, "npinter"),
    Source("RNAINTER4_EXPERIMENTAL", "RNAInter", "4.0-experimental", "input/rnainter4_human_experimental.tsv", True, False, "rnainter"),
    Source("RNAINTER4_PREDICTED", "RNAInter", "4.0-predicted", "input/rnainter4_human_predicted.tsv", False, True, "rnainter"),
    Source("LNCRNA2TARGET2_LOW", "LncRNA2Target", "2.0-low-throughput", "input/lncrna2target2_human_low_throughput.tsv", True, False, "lncrna2target"),
    Source("LNCTARD2", "LncTarD", "2.0", "input/lnctard2_curated_human.tsv", True, False, "lnctard"),
    Source("LNCACTDB4", "LncACTdb", "4.0", "input/lncactdb4_human_curated.tsv", True, False, "lncactdb"),
)


EXPECTED_COLUMNS = (
    "interaction_id", "source_row_id", "source_row_index", "source_row_sha256",
    "source_sha256", "source_record_id", "historical_interaction_id",
    "original_record_id", "source_database", "source_dataset", "source_relative_path",
    "cancer_id", "lncrna_id", "lncrna_raw", "lncrna_source_identifier",
    "lncrna_source_aliases", "lncrna_mapping_route", "lncrna_mapping_candidate_count",
    "partner_id", "partner_raw", "partner_source_identifier", "partner_source_aliases",
    "partner_type", "partner_mapping_route", "partner_mapping_candidate_count",
    "mapping_status", "relation_type", "relation_raw", "direction", "experiment_family",
    "experiment_raw", "interaction_type", "throughput", "is_experimental",
    "is_predicted", "pmid", "species", "disease_raw", "cell_line", "tissue",
    "drug_raw", "orientation_status",
)

EMPTY_EVENT_COLUMNS = (
    "evidence_event_id", "lncrna_id", "partner_id", "pathway_id",
    "pathway_family_id", "cancer_id", "source_database", "source_dataset",
    "source_record_id", "pmid", "experiment_family", "relation_type",
    "direction", "is_experimental", "is_predicted", "source_row_sha256",
)


def orientation(row: Mapping[str, object], lnc: Authority) -> tuple[int | None, str]:
    categories = [clean(row.get("Category1")).lower(), clean(row.get("Category2")).lower()]
    explicit = [index for index, value in enumerate(categories) if value == "lncrna"]
    if len(explicit) == 1:
        return explicit[0], "explicit_single_lncrna_category"
    resolutions = [
        resolve(
            lnc,
            raw_name=row.get(f"Interactor{index + 1}.Symbol"),
            source_identifier=row.get(f"Raw_ID{index + 1}"),
            aliases=(row.get(f"Raw_ID{index + 1}"),),
            canonical_route="canonical_lncrna_id",
        )
        for index in range(2)
    ]
    mapped = [index for index, value in enumerate(resolutions) if value.identifier]
    if len(mapped) == 1:
        return mapped[0], "inferred_unique_fail_closed_lncrna_mapping"
    if len(explicit) > 1:
        return None, "ambiguous_multiple_lncrna_categories"
    if len(mapped) > 1:
        return None, "ambiguous_both_entities_map_to_lncrna"
    return None, "unresolved_no_lncrna_orientation"


def extract(
    source: Source,
    row: Mapping[str, object],
    row_number: int,
    first_column: str,
    lnc: Authority,
) -> dict[str, Any]:
    if source.kind == "npinter":
        noncode = clean(row.get("RNA_NONCODE_ID"))
        return dict(
            original_record_id=clean(row.get("ncRI_ID")), lncrna_raw=clean(row.get("RNA_name")),
            lncrna_source_identifier=noncode, lncrna_aliases=(noncode,) if noncode else (),
            partner_raw=clean(row.get("partner_name")), partner_source_identifier=clean(row.get("partner_ID")),
            partner_aliases=(), partner_type=clean(row.get("partner_type")),
            relation_raw=first(row, ("interaction_type", "interaction_class")), experiment_raw=clean(row.get("methods")),
            disease_raw="", cell_line=clean(row.get("tissue_cell")), tissue=clean(row.get("tissue_cell")),
            drug_raw="", direction="", pmid=clean(row.get("PMIDs")), throughput=clean(row.get("throughput_flag")),
            species=clean(row.get("species")), orientation_status="explicit_npinter_lncrna_column",
        )
    if source.kind == "rnainter":
        index, orientation_status = orientation(row, lnc)
        if index is None:
            lnc_raw = "|".join(filter(None, (clean(row.get("Interactor1.Symbol")), clean(row.get("Interactor2.Symbol")))))
            lnc_source = "|".join(filter(None, (clean(row.get("Raw_ID1")), clean(row.get("Raw_ID2")))))
            partner_raw = partner_source = partner_type = ""
            aliases: tuple[str, ...] = ()
        else:
            partner_index = 1 - index
            lnc_raw = clean(row.get(f"Interactor{index + 1}.Symbol"))
            lnc_source = clean(row.get(f"Raw_ID{index + 1}"))
            aliases = (lnc_source,)
            partner_raw = clean(row.get(f"Interactor{partner_index + 1}.Symbol"))
            partner_source = clean(row.get(f"Raw_ID{partner_index + 1}"))
            partner_type = clean(row.get(f"Category{partner_index + 1}"))
        method = clean(row.get("predict")) if source.predicted else first(row, ("strong", "weak"))
        species_column = f"Species{index + 1}" if index is not None else "Species1"
        return dict(
            original_record_id=clean(row.get("RNAInterID")), lncrna_raw=lnc_raw,
            lncrna_source_identifier=lnc_source, lncrna_aliases=aliases,
            partner_raw=partner_raw, partner_source_identifier=partner_source,
            partner_aliases=(partner_source,) if partner_source else (), partner_type=partner_type,
            relation_raw="predicted_interaction" if source.predicted else "experimental_interaction",
            experiment_raw=method, disease_raw="", cell_line="", tissue="", drug_raw="", direction="", pmid="",
            throughput="unspecified", species=clean(row.get(species_column)), orientation_status=orientation_status,
        )
    if source.kind == "lncrna2target":
        return dict(
            original_record_id=f"LOW:{row_number}", lncrna_raw=first(row, ("LncRNA_official_symbol", "lncRNA_name_from_paper")),
            lncrna_source_identifier=clean(row.get("Ensembl_ID")), lncrna_aliases=(row.get("lncRNA_name_from_paper"),),
            partner_raw=first(row, ("Target_official_symbol", "Target_symbol_from_paper")),
            partner_source_identifier=clean(row.get("Target_entrez_gene_ID")), partner_aliases=(row.get("Target_symbol_from_paper"),),
            partner_type="gene", relation_raw="regulation", experiment_raw=clean(row.get("LncRNA_experiment")),
            disease_raw=clean(row.get("Disease_state")), cell_line=clean(row.get("Cell_Line")), tissue=clean(row.get("Tissue_Origin")),
            drug_raw="", direction="", pmid=clean(row.get("PMID")), throughput="low-throughput", species=clean(row.get("Species")),
            orientation_status="explicit_lncrna2target_lncrna_column",
        )
    if source.kind == "lnctard":
        return dict(
            original_record_id=clean(row.get("RID")), lncrna_raw=clean(row.get("Regulator")),
            lncrna_source_identifier=clean(row.get("RegulatorEnsembleID")), lncrna_aliases=(row.get("RegulatorAliases"),),
            partner_raw=clean(row.get("Target")), partner_source_identifier=first(row, ("TargetEnsembleID", "TargetEntrezID")),
            partner_aliases=(row.get("TargetAliases"),), partner_type=clean(row.get("TargetType")),
            relation_raw=first(row, ("regulatoryType", "regulatoryMechanism")), experiment_raw=clean(row.get("Experimental.method.for.lncRNA.target")),
            disease_raw=first(row, ("DiseaseName2", "DiseaseName")), cell_line="", tissue="", drug_raw=clean(row.get("Drugs")),
            direction=clean(row.get("RegulationDiretion")), pmid=clean(row.get("PubMedID")), throughput="curated",
            species="Homo sapiens", orientation_status="explicit_lnctard_regulator_type",
        )
    if source.kind == "lncactdb":
        gene = clean(row.get("gene")); mirna = clean(row.get("mir"))
        return dict(
            original_record_id=clean(row.get(first_column)), lncrna_raw=clean(row.get("lncrna")), lncrna_source_identifier="",
            lncrna_aliases=(), partner_raw=gene or mirna, partner_source_identifier=clean(row.get("mirid")) if not gene else "",
            partner_aliases=(), partner_type="gene" if gene else "miRNA", relation_raw="ceRNA",
            experiment_raw=clean(row.get("experimental")), disease_raw=clean(row.get("disease")),
            cell_line=clean(row.get("tissue/cells")), tissue=clean(row.get("tissue/cells")), drug_raw="", direction="",
            pmid=clean(row.get("pubmed")), throughput="curated", species=clean(row.get("species")),
            orientation_status="explicit_lncactdb_lncrna_column",
        )
    raise ValidationError(f"Unknown source kind: {source.kind}")


def expected_mapping_status(lnc_route: str, lnc_id: str, partner_route: str, partner_id: str) -> str:
    if not lnc_id:
        return "ambiguous_lncrna" if lnc_route.startswith("ambiguous_") else "unresolved_lncrna"
    if partner_id:
        return "lncrna_and_partner_mapped"
    return "lncrna_mapped_partner_ambiguous" if partner_route.startswith("ambiguous_") else "lncrna_mapped_partner_unresolved"


def assert_equal(observed: object, expected: object, *, source: Source, row_number: int, field: str) -> None:
    if str(observed) != str(expected):
        raise ValidationError(
            f"{source.key} row {row_number} field {field}: {observed!r} != {expected!r}"
        )


def validate(
    *,
    project_root: Path,
    rematerialization_root: Path,
    output_root: Path,
    progress_every: int,
    sources: Sequence[Source] = SOURCES,
) -> dict[str, Any]:
    project = project_root.resolve(); remat = rematerialization_root.resolve(); output = output_root.resolve()
    temporary = output.with_name(f".{output.name}.tmp")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"Validation refuses output reuse: {output}")
    if not project.is_dir() or project.is_symlink() or not remat.is_dir() or remat.is_symlink():
        raise ValidationError("Project/rematerialization root is missing or unsafe")
    manifest_path = remat / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != EXPECTED_FORMAT or manifest.get("status") != "PASS_REMATERIALIZED_REQUIRES_INDEPENDENT_VALIDATION":
        raise ValidationError("Rematerialization manifest format/status is not admissible")
    if manifest.get("contract", {}).get("independent_validation_complete") is not False:
        raise ValidationError("Input manifest must declare independent validation incomplete")
    relation = remat / "interaction_relation_recovered.tsv.gz"
    empty_event = remat / "evidence_event_empty_by_contract.tsv"
    for key, path in (("interaction_relation", relation), ("evidence_event", empty_event)):
        declaration = manifest["outputs"][key]
        if Path(declaration["path"]).resolve() != path or sha256_file(path) != declaration["sha256"]:
            raise ValidationError(f"Output artifact path/hash mismatch: {key}")
    with empty_event.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    if len(rows) != 1 or tuple(rows[0]) != EMPTY_EVENT_COLUMNS:
        raise ValidationError(
            "Evidence event authority must contain exactly the canonical header"
        )

    input_by_path = {str(Path(item["path"]).resolve()).lower(): item for item in manifest["inputs"]}
    for item in manifest["inputs"]:
        path = Path(item["path"]).resolve()
        try:
            path.relative_to(project)
        except ValueError as exc:
            raise ValidationError(f"Manifest input escapes project root: {path}") from exc
        if sha256_file(path) != item["sha256"]:
            raise ValidationError(f"Manifest input SHA drift: {path}")
    lnc_path = project / "processed/dimensions/dim_lncRNA.tsv"
    gene_path = project / "processed/dimensions/dim_gene.tsv"
    lnc = load_authority(lnc_path, "lncrna_id")
    gene = load_authority(gene_path, "gene_id")
    temporary.mkdir(parents=True)
    progress_path = temporary / "PROGRESS.json"
    global_counts: Counter[str] = Counter(); per_source: list[dict[str, Any]] = []
    with gzip.open(relation, "rt", encoding="utf-8", newline="") as relation_handle:
        relation_reader = csv.DictReader(relation_handle, delimiter="\t")
        if tuple(relation_reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValidationError("Recovered interaction header drift")
        for source in sources:
            raw_path = project / source.relative_path
            source_manifest = input_by_path.get(str(raw_path.resolve()).lower())
            if source_manifest is None:
                raise ValidationError(f"Raw source absent from manifest: {raw_path}")
            raw_sha = source_manifest["sha256"]
            counts: Counter[str] = Counter(); lnc_routes: Counter[str] = Counter(); partner_routes: Counter[str] = Counter(); orientations: Counter[str] = Counter()
            with raw_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as raw_handle:
                raw_reader = csv.DictReader(raw_handle, delimiter="\t")
                if not raw_reader.fieldnames:
                    raise ValidationError(f"Raw source has no header: {raw_path}")
                first_column = raw_reader.fieldnames[0]
                for row_number, raw_row in enumerate(raw_reader, start=1):
                    observed = next(relation_reader, None)
                    if observed is None:
                        raise ValidationError(f"Recovered table ended before {source.key} row {row_number}")
                    extracted = extract(source, raw_row, row_number, first_column, lnc)
                    expected_source_record = source_record_id(source.database, extracted["original_record_id"])
                    fixed = {
                        "interaction_id": f"INT32:{source.key}:{row_number}",
                        "source_row_id": f"SRCROW32:{source.key}:{row_number}",
                        "source_row_index": str(row_number),
                        "source_row_sha256": raw_row_sha(source.relative_path, row_number, raw_row),
                        "source_sha256": raw_sha,
                        "source_record_id": expected_source_record,
                        "historical_interaction_id": "INT:" + hashlib.sha1(expected_source_record.encode("utf-8")).hexdigest(),
                        "original_record_id": extracted["original_record_id"],
                        "source_database": source.database,
                        "source_dataset": source.dataset,
                        "source_relative_path": source.relative_path,
                        "cancer_id": "PAN_CANCER",
                        "lncrna_raw": extracted["lncrna_raw"],
                        "lncrna_source_identifier": extracted["lncrna_source_identifier"],
                        "lncrna_source_aliases": ";".join(clean(v) for v in extracted["lncrna_aliases"] if clean(v)),
                        "partner_raw": extracted["partner_raw"],
                        "partner_source_identifier": extracted["partner_source_identifier"],
                        "partner_source_aliases": ";".join(clean(v) for v in extracted["partner_aliases"] if clean(v)),
                        "partner_type": extracted["partner_type"],
                        "relation_type": normalized_relation(extracted["relation_raw"]),
                        "relation_raw": extracted["relation_raw"],
                        "direction": extracted["direction"],
                        "experiment_family": normalized_experiment(extracted["experiment_raw"]),
                        "experiment_raw": extracted["experiment_raw"],
                        "interaction_type": normalized_relation(extracted["relation_raw"]),
                        "throughput": extracted["throughput"],
                        "is_experimental": str(source.experimental).lower(),
                        "is_predicted": str(source.predicted).lower(),
                        "pmid": extracted["pmid"], "species": extracted["species"],
                        "disease_raw": extracted["disease_raw"], "cell_line": extracted["cell_line"],
                        "tissue": extracted["tissue"], "drug_raw": extracted["drug_raw"],
                        "orientation_status": extracted["orientation_status"],
                    }
                    for field, expected in fixed.items():
                        assert_equal(observed.get(field, ""), expected, source=source, row_number=row_number, field=field)
                    if extracted["orientation_status"].startswith(("ambiguous_", "unresolved_")):
                        lnc_resolution = Resolution("", extracted["orientation_status"], 0)
                    else:
                        lnc_resolution = resolve(
                            lnc, raw_name=extracted["lncrna_raw"], source_identifier=extracted["lncrna_source_identifier"],
                            aliases=tuple(extracted["lncrna_aliases"]), canonical_route="canonical_lncrna_id",
                        )
                    partner_resolution = resolve(
                        gene, raw_name=extracted["partner_raw"], source_identifier=extracted["partner_source_identifier"],
                        aliases=tuple(extracted["partner_aliases"]), canonical_route="canonical_gene_id",
                    )
                    mapping = {
                        "lncrna_id": lnc_resolution.identifier, "lncrna_mapping_route": lnc_resolution.route,
                        "lncrna_mapping_candidate_count": str(lnc_resolution.candidates),
                        "partner_id": partner_resolution.identifier, "partner_mapping_route": partner_resolution.route,
                        "partner_mapping_candidate_count": str(partner_resolution.candidates),
                        "mapping_status": expected_mapping_status(lnc_resolution.route, lnc_resolution.identifier, partner_resolution.route, partner_resolution.identifier),
                    }
                    for field, expected in mapping.items():
                        assert_equal(observed.get(field, ""), expected, source=source, row_number=row_number, field=field)
                    counts["rows"] += 1; counts["empty_original_record_id"] += not bool(extracted["original_record_id"])
                    counts["lncrna_mapped"] += bool(lnc_resolution.identifier); counts["lncrna_unresolved_or_ambiguous"] += not bool(lnc_resolution.identifier)
                    counts["partner_mapped"] += bool(partner_resolution.identifier); counts["partner_unresolved_or_ambiguous"] += not bool(partner_resolution.identifier)
                    counts["lncrna_and_partner_mapped"] += bool(lnc_resolution.identifier and partner_resolution.identifier)
                    counts["experimental_rows"] += source.experimental; counts["predicted_rows"] += source.predicted
                    counts["rows_with_pmid"] += bool(_PMID.search(extracted["pmid"]))
                    lnc_routes[lnc_resolution.route] += 1; partner_routes[partner_resolution.route] += 1; orientations[extracted["orientation_status"]] += 1
                    if progress_every > 0 and counts["rows"] % progress_every == 0:
                        progress_path.write_text(json.dumps({"status": "RUNNING", "source_key": source.key, "source_rows": counts["rows"], "generated_at_utc": datetime.now(timezone.utc).isoformat()}, sort_keys=True) + "\n", encoding="utf-8")
            observed_source = next(item for item in manifest["per_source"] if item["source_key"] == source.key)
            expected_source = {
                "counts": dict(sorted(counts.items())), "lncrna_mapping_routes": dict(sorted(lnc_routes.items())),
                "partner_mapping_routes": dict(sorted(partner_routes.items())), "orientation_routes": dict(sorted(orientations.items())),
            }
            for key, expected in expected_source.items():
                if observed_source[key] != expected:
                    raise ValidationError(f"Manifest per-source {key} drift for {source.key}")
            global_counts.update(counts)
            per_source.append({"source_key": source.key, **expected_source})
        if next(relation_reader, None) is not None:
            raise ValidationError("Recovered interaction table has rows beyond raw authorities")
    if dict(sorted(global_counts.items())) != manifest["counts"]:
        raise ValidationError("Global rematerialization counts do not match independent recount")
    if progress_path.exists():
        progress_path.unlink()
    generated = datetime.now(timezone.utc).isoformat()
    report = {
        "format": VALIDATION_FORMAT, "status": PASS_STATUS, "generated_at_utc": generated,
        "project_root": str(project), "rematerialization_root": str(remat), "output_root": str(output),
        "formal_training_started": False, "production_deployed": False, "production_port_8260_touched": False,
        "contract": {
            "production_module_imported": False, "every_raw_row_replayed": True,
            "every_source_row_sha256_recomputed": True, "every_identifier_mapping_recomputed_fail_closed": True,
            "every_positional_id_verified": True, "header_only_evidence_event_verified": True,
            "single_occurrence_authority_verified": True, "input_and_output_sha256_verified": True,
        },
        "counts": dict(sorted(global_counts.items())), "per_source": per_source,
        "inputs": {
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "interaction_relation": {"path": str(relation), "sha256": sha256_file(relation)},
            "evidence_event": {"path": str(empty_event), "sha256": sha256_file(empty_event)},
        },
    }
    validation_path = temporary / "VALIDATION.json"
    validation_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    validation_sha = sha256_file(validation_path)
    success = {
        "format": VALIDATION_FORMAT, "status": PASS_STATUS, "generated_at_utc": generated,
        "validation": {"path": str(output / "VALIDATION.json"), "sha256": validation_sha},
        "rows": int(global_counts["rows"]), "formal_training_started": False,
        "production_deployed": False, "production_port_8260_touched": False,
    }
    (temporary / "SUCCESS.json").write_text(json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--rematerialization-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--progress-every", type=int, default=250_000)
    args = parser.parse_args()
    report = validate(
        project_root=args.project_root,
        rematerialization_root=args.rematerialization_root,
        output_root=args.output_root,
        progress_every=args.progress_every,
    )
    print(json.dumps({"status": report["status"], "output_root": report["output_root"], "counts": report["counts"]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
