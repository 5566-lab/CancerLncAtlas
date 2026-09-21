"""Traceable assay-detail enrichment for V3.2 perturbation evidence.

The authoritative ``evidence_event`` table intentionally contains only an
experiment family.  This module never overwrites that table.  It creates a
sidecar keyed by ``evidence_event_id`` using source records and, optionally,
PubMed title/abstract metadata.  Every populated value carries its match
route, evidence strength and manual-review requirement; an unavailable value
is kept unavailable rather than guessed.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


DETAIL_FORMAT = "CANCERLNCATLAS_V32_EXPERIMENT_ASSAY_DETAIL_V1"
DETAIL_FILENAME = "v32_experiment_assay_detail.parquet"
MANIFEST_FILENAME = "EXPERIMENT_ASSAY_DETAIL_MANIFEST.json"
PUBMED_CACHE_FORMAT = "CANCERLNCATLAS_PUBMED_TITLE_ABSTRACT_JSONL_V1"


class ExperimentAssayDetailError(RuntimeError):
    """Raised when assay-detail lineage or schema validation fails."""


@dataclass(frozen=True)
class AssayDetailResult:
    output_dir: Path
    detail_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "-", "na", "n/a", "none", "nan"} else text


def _normalise_symbol(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", _clean(value).upper())


def _split_pmids(value: object) -> list[str]:
    return re.findall(r"\d+", _clean(value))


def _deduplicated_join(values: Iterable[object]) -> str:
    cleaned = sorted({_clean(value) for value in values if _clean(value)})
    return ";".join(cleaned)


_PERTURBATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("CRISPR_INTERFERENCE", r"\bcrispr\s*[-_]?\s*i\b|crispr interference"),
    ("CRISPR_ACTIVATION", r"\bcrispr\s*[-_]?\s*a\b|crispr activation"),
    ("CRISPR_KNOCKOUT_OR_EDITING", r"\bcrispr\b|\bcas9\b|gene edit|knock[- ]?out"),
    ("SHRNA_KNOCKDOWN", r"\bsh\s*rna\b|short hairpin"),
    ("SIRNA_KNOCKDOWN", r"\bsi\s*rna\b|small interfering rna"),
    ("ANTISENSE_OLIGONUCLEOTIDE", r"antisense oligo|\baso\b|gapmer"),
    ("RNA_INTERFERENCE", r"rna interference|\brnai\b"),
    ("KNOCKDOWN_OR_SILENCING", r"knock[- ]?down|silenc(?:e|ed|ing)"),
    ("OVEREXPRESSION", r"over[- ]?express|ectopic expression"),
    ("RESCUE", r"\brescue\b|re-expression"),
)

_READOUT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("REPORTER_OR_LUCIFERASE", r"luciferase|reporter assay"),
    ("QPCR_OR_RT_PCR", r"q(?:rt)?[- ]?pcr|rt[- ]?pcr|real[- ]time pcr"),
    ("RNA_SEQ", r"rna[- ]?seq|transcriptom"),
    ("MICROARRAY", r"microarray"),
    ("WESTERN_BLOT", r"western blot|immunoblot"),
    ("RIP", r"rna immunoprecipitation|\brip\b"),
    ("CHIP", r"chromatin immunoprecipitation|chip[- ]?seq|\bchip\b"),
    ("PULLDOWN", r"pull[- ]?down|pulldown"),
    ("PHENOTYPE_VIABILITY", r"viability|proliferation|colony formation"),
    ("PHENOTYPE_APOPTOSIS", r"apoptosis|caspase"),
    ("PHENOTYPE_MIGRATION_INVASION", r"migration|invasion|wound healing"),
)


def classify_assay_text(value: object) -> dict[str, object]:
    """Return controlled perturbation and readout terms without guessing."""

    text = _clean(value)
    lowered = text.lower()
    perturbations = [
        label for label, pattern in _PERTURBATION_PATTERNS if re.search(pattern, lowered)
    ]
    readouts = [label for label, pattern in _READOUT_PATTERNS if re.search(pattern, lowered)]
    return {
        "perturbation_methods": ";".join(dict.fromkeys(perturbations)),
        "perturbation_method_available": bool(perturbations),
        "readout_assays": ";".join(dict.fromkeys(readouts)),
        "readout_assay_available": bool(readouts),
    }


def _dimension_maps(
    dim_lncrna: pd.DataFrame, dim_gene: pd.DataFrame
) -> tuple[dict[str, str], dict[str, str]]:
    required_lnc = {"lncrna_id", "gene_symbol"}
    required_gene = {"gene_id", "gene_symbol"}
    if not required_lnc.issubset(dim_lncrna.columns):
        raise ExperimentAssayDetailError(
            f"dim_lncRNA missing columns: {sorted(required_lnc - set(dim_lncrna.columns))}"
        )
    if not required_gene.issubset(dim_gene.columns):
        raise ExperimentAssayDetailError(
            f"dim_gene missing columns: {sorted(required_gene - set(dim_gene.columns))}"
        )
    lnc = (
        dim_lncrna[["lncrna_id", "gene_symbol"]]
        .dropna()
        .drop_duplicates("lncrna_id")
        .set_index("lncrna_id")
        .gene_symbol.map(_clean)
        .to_dict()
    )
    gene = (
        dim_gene[["gene_id", "gene_symbol"]]
        .dropna()
        .drop_duplicates("gene_id")
        .set_index("gene_id")
        .gene_symbol.map(_clean)
        .to_dict()
    )
    return lnc, gene


def prepare_ncpath_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype="string", compression="infer")
    required = {
        "interID",
        "ncName",
        "tarName",
        "experiment",
        "reference",
        "organism",
        "interDescription",
        "tissueOrCell",
        "datasource",
    }
    if not required.issubset(frame.columns):
        raise ExperimentAssayDetailError(
            f"NcPath source missing columns: {sorted(required - set(frame.columns))}"
        )
    frame = frame.loc[frame.organism.astype(str).eq("Homo sapiens")].copy()
    frame["pmid"] = frame.reference.map(_split_pmids)
    frame = frame.explode("pmid")
    frame = frame.loc[frame.pmid.astype(str).str.fullmatch(r"\d+")].copy()
    frame["lnc_symbol_key"] = frame.ncName.map(_normalise_symbol)
    frame["gene_symbol_key"] = frame.tarName.map(_normalise_symbol)
    frame["assay_detail_source"] = frame.experiment.map(_clean)
    frame["source_statement"] = frame.interDescription.map(_clean)
    frame["source_context"] = frame.tissueOrCell.map(_clean)
    return frame.reset_index(drop=True)


def parse_pubmed_xml(payload: bytes) -> dict[str, dict[str, str]]:
    root = ET.fromstring(payload)
    records: dict[str, dict[str, str]] = {}
    for article in root.findall(".//PubmedArticle"):
        pmid = "".join(article.findtext(".//PMID", default="")).strip()
        if not pmid:
            continue
        title_node = article.find(".//ArticleTitle")
        title = "" if title_node is None else "".join(title_node.itertext()).strip()
        abstract_parts = [
            "".join(node.itertext()).strip()
            for node in article.findall(".//Abstract/AbstractText")
        ]
        abstract = " ".join(part for part in abstract_parts if part)
        records[pmid] = {"title": title, "abstract": abstract}
    return records


def fetch_pubmed_records(
    pmids: Sequence[str],
    *,
    batch_size: int = 150,
    pause_seconds: float = 0.34,
    email: str | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, dict[str, str]]:
    """Fetch public PubMed title/abstract metadata in rate-limited batches."""

    unique = sorted({str(pmid) for pmid in pmids if str(pmid).isdigit()})
    records: dict[str, dict[str, str]] = {}
    for start in range(0, len(unique), batch_size):
        batch = unique[start : start + batch_size]
        query: dict[str, str] = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
        }
        if email:
            query["email"] = email
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, headers={"User-Agent": "CancerLncAtlas/3.2 assay-audit"})
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            records.update(parse_pubmed_xml(response.read()))
        if start + batch_size < len(unique):
            time.sleep(max(0.0, pause_seconds))
    return records


def write_pubmed_cache(records: Mapping[str, Mapping[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as handle:
        for pmid in sorted(records, key=lambda value: int(value)):
            payload = {
                "cache_format": PUBMED_CACHE_FORMAT,
                "pmid": pmid,
                "title": _clean(records[pmid].get("title", "")),
                "abstract": _clean(records[pmid].get("abstract", "")),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    partial.replace(path)


def read_pubmed_cache(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    records: dict[str, dict[str, str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("cache_format") != PUBMED_CACHE_FORMAT:
                raise ExperimentAssayDetailError(
                    f"Unexpected PubMed cache format at line {line_number}"
                )
            pmid = str(payload.get("pmid", ""))
            if not pmid.isdigit() or pmid in records:
                raise ExperimentAssayDetailError(
                    f"Invalid or duplicate PMID at cache line {line_number}: {pmid}"
                )
            records[pmid] = {
                "title": _clean(payload.get("title", "")),
                "abstract": _clean(payload.get("abstract", "")),
            }
    return records


def build_assay_detail(
    evidence: pd.DataFrame,
    ncpath: pd.DataFrame,
    dim_lncrna: pd.DataFrame,
    dim_gene: pd.DataFrame,
    *,
    pubmed_records: Mapping[str, Mapping[str, str]] | None = None,
) -> pd.DataFrame:
    required = {
        "evidence_event_id",
        "pmid",
        "lncrna_id",
        "partner_id",
        "experiment_family",
    }
    if not required.issubset(evidence.columns):
        raise ExperimentAssayDetailError(
            f"evidence_event missing columns: {sorted(required - set(evidence.columns))}"
        )
    selected = evidence.loc[
        evidence.experiment_family.astype(str).str.lower().str.contains("perturb")
    ].copy()
    if selected.evidence_event_id.duplicated().any():
        raise ExperimentAssayDetailError("evidence_event_id must be unique")
    lnc_map, gene_map = _dimension_maps(dim_lncrna, dim_gene)
    selected["pmid"] = selected.pmid.map(_clean)
    selected["lnc_symbol"] = selected.lncrna_id.map(lnc_map).map(_clean)
    selected["gene_symbol"] = selected.partner_id.map(gene_map).map(_clean)
    selected["lnc_symbol_key"] = selected.lnc_symbol.map(_normalise_symbol)
    selected["gene_symbol_key"] = selected.gene_symbol.map(_normalise_symbol)

    exact_groups = {
        key: group
        for key, group in ncpath.groupby(
            ["pmid", "lnc_symbol_key", "gene_symbol_key"], sort=False
        )
    }
    pmid_groups = {key: group for key, group in ncpath.groupby("pmid", sort=False)}
    pubmed = dict(pubmed_records or {})
    rows: list[dict[str, object]] = []
    for row in selected.itertuples(index=False):
        exact_key = (row.pmid, row.lnc_symbol_key, row.gene_symbol_key)
        source = exact_groups.get(exact_key)
        route = ""
        strength = ""
        manual_review_required = True
        source_record_ids = ""
        source_database = ""
        assay_detail = ""
        source_context = ""
        unavailable_reason = ""
        if source is not None and bool(row.lnc_symbol_key) and bool(row.gene_symbol_key):
            assay_detail = _deduplicated_join(source.assay_detail_source)
            if assay_detail:
                route = "PMID_LNCRNA_GENE_EXACT"
                strength = "HIGH"
                manual_review_required = False
                source_record_ids = _deduplicated_join(source.interID)
                source_database = _deduplicated_join(source.datasource)
                source_context = _deduplicated_join(source.source_context)
        if not assay_detail:
            source = pmid_groups.get(row.pmid)
            if source is not None:
                details = sorted(
                    {_clean(value) for value in source.assay_detail_source if _clean(value)}
                )
                if len(details) == 1:
                    assay_detail = details[0]
                    route = "PMID_UNIQUE_SOURCE_ASSAY"
                    strength = "MEDIUM"
                    source_record_ids = _deduplicated_join(source.interID)
                    source_database = _deduplicated_join(source.datasource)
                    source_context = _deduplicated_join(source.source_context)
        if not assay_detail and row.pmid in pubmed:
            record = pubmed[row.pmid]
            abstract_text = " ".join(
                filter(None, [_clean(record.get("title", "")), _clean(record.get("abstract", ""))])
            )
            classified = classify_assay_text(abstract_text)
            controlled = ";".join(
                filter(
                    None,
                    [
                        str(classified["perturbation_methods"]),
                        str(classified["readout_assays"]),
                    ],
                )
            )
            if controlled:
                assay_detail = controlled
                route = "PUBMED_TITLE_ABSTRACT_KEYWORDS"
                strength = "LOW"
                source_database = "PubMed"
        if not assay_detail:
            unavailable_reason = (
                "NO_TRACEABLE_ASSAY_DETAIL_IN_BOUND_SOURCES"
                if row.pmid
                else "PMID_UNAVAILABLE"
            )
        classified = classify_assay_text(assay_detail)
        rows.append(
            {
                "evidence_event_id": row.evidence_event_id,
                "pmid": row.pmid or pd.NA,
                "lncrna_id": row.lncrna_id,
                "partner_id": row.partner_id,
                "lncrna_symbol": row.lnc_symbol or pd.NA,
                "partner_symbol": row.gene_symbol or pd.NA,
                "assay_detail": assay_detail or pd.NA,
                "assay_detail_available": bool(assay_detail),
                "assay_detail_unavailable_reason": unavailable_reason or pd.NA,
                "perturbation_methods": classified["perturbation_methods"] or pd.NA,
                "perturbation_method_available": classified[
                    "perturbation_method_available"
                ],
                "readout_assays": classified["readout_assays"] or pd.NA,
                "readout_assay_available": classified["readout_assay_available"],
                "match_route": route or "UNAVAILABLE_TYPED",
                "evidence_strength": strength or "UNAVAILABLE",
                "manual_review_required": bool(manual_review_required),
                "source_database": source_database or pd.NA,
                "source_record_ids": source_record_ids or pd.NA,
                "source_context": source_context or pd.NA,
                "changes_primary_ranking": False,
                "changes_discovery_ranking": False,
            }
        )
    result = pd.DataFrame(rows).sort_values("evidence_event_id").reset_index(drop=True)
    if len(result) != len(selected) or result.evidence_event_id.duplicated().any():
        raise ExperimentAssayDetailError("Assay-detail output lost or duplicated source events")
    for column in (
        "assay_detail_available",
        "perturbation_method_available",
        "readout_assay_available",
        "manual_review_required",
        "changes_primary_ranking",
        "changes_discovery_ranking",
    ):
        result[column] = result[column].astype(bool)
    return result


def materialise_assay_detail(
    *,
    evidence_event_path: Path,
    ncpath_path: Path,
    dim_lncrna_path: Path,
    dim_gene_path: Path,
    output_dir: Path,
    pubmed_cache_path: Path | None = None,
) -> AssayDetailResult:
    paths = {
        "evidence_event": Path(evidence_event_path).resolve(),
        "ncpath": Path(ncpath_path).resolve(),
        "dim_lncrna": Path(dim_lncrna_path).resolve(),
        "dim_gene": Path(dim_gene_path).resolve(),
    }
    if pubmed_cache_path is not None:
        paths["pubmed_cache"] = Path(pubmed_cache_path).resolve()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ExperimentAssayDetailError(f"Missing assay-detail inputs: {missing}")
    output = Path(output_dir).resolve()
    if output.exists():
        raise ExperimentAssayDetailError(f"Output already exists: {output}")

    evidence = pd.read_parquet(paths["evidence_event"])
    ncpath = prepare_ncpath_source(paths["ncpath"])
    dim_lncrna = pd.read_parquet(paths["dim_lncrna"])
    dim_gene = pd.read_parquet(paths["dim_gene"])
    pubmed = read_pubmed_cache(paths.get("pubmed_cache"))
    detail = build_assay_detail(
        evidence,
        ncpath,
        dim_lncrna,
        dim_gene,
        pubmed_records=pubmed,
    )

    output.mkdir(parents=True, exist_ok=False)
    detail_path = output / DETAIL_FILENAME
    partial = detail_path.with_suffix(".parquet.partial")
    detail.to_parquet(partial, index=False)
    partial.replace(detail_path)
    route_counts = {
        str(key): int(value) for key, value in detail.match_route.value_counts().items()
    }
    manifest: dict[str, Any] = {
        "format": DETAIL_FORMAT,
        "status": "MATERIALISATION_SUCCESS",
        "rows": int(len(detail)),
        "assay_detail_available_rows": int(detail.assay_detail_available.sum()),
        "perturbation_method_available_rows": int(
            detail.perturbation_method_available.sum()
        ),
        "manual_review_required_rows": int(detail.manual_review_required.sum()),
        "match_route_counts": route_counts,
        "changes_primary_ranking": False,
        "changes_discovery_ranking": False,
        "input_artifacts": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "output_artifact": {
            "path": str(detail_path),
            "sha256": file_sha256(detail_path),
        },
        "lineage_policy": {
            "authoritative_evidence_overwritten": False,
            "historical_predictions_used": False,
            "unmatched_detail_inferred": False,
            "pubmed_keyword_matches_require_manual_review": True,
        },
    }
    manifest_path = output / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return AssayDetailResult(output, detail_path, manifest_path, manifest)


__all__ = [
    "AssayDetailResult",
    "ExperimentAssayDetailError",
    "build_assay_detail",
    "classify_assay_text",
    "fetch_pubmed_records",
    "materialise_assay_detail",
    "parse_pubmed_xml",
    "prepare_ncpath_source",
    "read_pubmed_cache",
    "write_pubmed_cache",
]
