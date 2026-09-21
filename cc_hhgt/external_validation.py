from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .common import input_path, read_table, stable_id, utc_now, write_json, write_table


PRIMARY_SOURCES = {"Lnc2Cancer", "LncRNADisease", "RNADisease", "GSE85011"}
GSE85011_PMID = "27980086"
GSE85011_ACCESSION = "GSE85011"

# Longest/specific phrases win. Generic contexts that map to more than one TCGA
# cohort are retained as ambiguous instead of being duplicated into several
# cancer-specific validation sets.
CANCER_PHRASES: dict[str, tuple[str, ...]] = {
    "ACC": ("adrenocortical carcinoma", "adrenal cortical carcinoma"),
    "BLCA": ("bladder urothelial carcinoma", "urothelial carcinoma", "bladder cancer", "bladder carcinoma"),
    "BRCA": ("triple negative breast cancer", "breast cancer", "breast carcinoma", "mammary carcinoma"),
    "CESC": ("cervical squamous cell carcinoma", "cervical cancer", "cervical carcinoma", "cervix cancer"),
    "CHOL": ("cholangiocarcinoma", "bile duct cancer", "biliary tract cancer"),
    "COAD": ("colon adenocarcinoma", "colon cancer", "colon carcinoma"),
    "DLBC": ("diffuse large b cell lymphoma", "diffuse large b-cell lymphoma"),
    "ESCA": ("esophageal carcinoma", "esophageal cancer", "oesophageal cancer"),
    "GBM": ("glioblastoma multiforme", "glioblastoma"),
    "HNSC": ("head and neck squamous cell carcinoma", "head and neck cancer"),
    "KICH": ("kidney chromophobe", "chromophobe renal cell carcinoma"),
    "KIRC": ("kidney renal clear cell carcinoma", "clear cell renal cell carcinoma"),
    "KIRP": ("kidney renal papillary cell carcinoma", "papillary renal cell carcinoma"),
    "LAML": ("acute myeloid leukemia", "acute myelogenous leukemia", "aml"),
    "LGG": ("lower grade glioma", "low grade glioma", "brain lower grade glioma"),
    "LIHC": ("hepatocellular carcinoma", "liver cancer", "liver carcinoma", "hepatoma", "hcc"),
    "LUAD": ("lung adenocarcinoma",),
    "LUSC": ("lung squamous cell carcinoma", "squamous cell lung carcinoma"),
    "MESO": ("mesothelioma",),
    "OV": ("ovarian serous cystadenocarcinoma", "ovarian serous carcinoma", "ovarian cancer", "ovarian carcinoma"),
    "PAAD": ("pancreatic adenocarcinoma", "pancreatic ductal adenocarcinoma", "pancreatic cancer", "pdac"),
    "PCPG": ("pheochromocytoma", "paraganglioma"),
    "PRAD": ("prostate adenocarcinoma", "prostate cancer", "prostatic carcinoma"),
    "READ": ("rectum adenocarcinoma", "rectal cancer", "rectal carcinoma"),
    "SARC": ("soft tissue sarcoma", "sarcoma"),
    "SKCM": ("skin cutaneous melanoma", "cutaneous melanoma", "skin melanoma"),
    "STAD": ("stomach adenocarcinoma", "gastric adenocarcinoma", "gastric cancer", "stomach cancer"),
    "TGCT": ("testicular germ cell tumor", "testicular germ cell cancer", "testicular cancer"),
    "THCA": ("papillary thyroid carcinoma", "thyroid carcinoma", "thyroid cancer"),
    "THYM": ("thymoma",),
    "UCEC": ("uterine corpus endometrial carcinoma", "endometrial carcinoma", "endometrial cancer"),
    "UCS": ("uterine carcinosarcoma",),
    "UVM": ("uveal melanoma", "ocular melanoma"),
}

AMBIGUOUS_PHRASES: dict[str, tuple[str, ...]] = {
    "COAD;READ": ("colorectal cancer", "colorectal carcinoma", "colorectal adenocarcinoma"),
    "LUAD;LUSC": ("non small cell lung cancer", "non-small cell lung cancer", "nsclc", "lung cancer", "lung carcinoma"),
    "KICH;KIRC;KIRP": ("renal cell carcinoma", "kidney cancer", "renal cancer"),
    "GBM;LGG": ("glioma", "brain tumor", "brain tumour"),
    "UCEC;UCS": ("uterine cancer", "uterine carcinoma"),
}

PREDICTION_METHOD_TOKENS = (
    "prediction", "predicted", "computational", "machine learning", "random walk",
    "tanimoto", "tam", "ldap", "lrlslda", "rwrlnd", "rwrlncd", "algorithm",
)
EXPERIMENT_METHOD_TOKENS = (
    "pcr", "rna-seq", "rna sequencing", "microarray", "western", "immunoblot",
    "luciferase", "knockdown", "knock-out", "knockout", "crispr", "chip",
    "rip", "fish", "northern", "tissue", "patient", "cell culture", "transwell",
    "colony formation", "cck-8", "flow cytometry", "immunohistochemistry",
)


def normalize_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]+", "", str(value).strip().upper())


def _safe_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def normalize_pmid(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    match = re.search(r"\b(\d{6,9})\b", text)
    return match.group(1) if match else ""


def normalize_direction(value: object) -> str:
    text = _safe_text(value).lower()
    if re.search(r"\b(up|increase|overexpress|promot|oncogen|high)\w*", text):
        return "positive"
    if re.search(r"\b(down|decrease|underexpress|suppress|inhibit|low)\w*", text):
        return "negative"
    return "unknown"


def build_lnc_lookup(dim_lnc: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    candidates: dict[str, set[str]] = defaultdict(set)
    for row in dim_lnc.itertuples(index=False):
        lnc_id = str(row.lncrna_id)
        values: list[object] = [
            lnc_id,
            getattr(row, "ensembl_gene_id", None),
            getattr(row, "ensembl_gene_id_versioned", None),
            getattr(row, "gene_symbol", None),
            getattr(row, "gene_name", None),
            getattr(row, "hgnc_id", None),
        ]
        aliases = getattr(row, "aliases", None)
        if aliases is not None and not pd.isna(aliases):
            values.extend(re.split(r"[;,|]", str(aliases)))
        for value in values:
            key = normalize_name(value)
            if key:
                candidates[key].add(lnc_id)
    unique = {key: next(iter(ids)) for key, ids in candidates.items() if len(ids) == 1}
    ambiguous = {key for key, ids in candidates.items() if len(ids) > 1}
    return unique, ambiguous


def map_lnc_values(
    values: pd.Series,
    lookup: dict[str, str],
    ambiguous: set[str],
) -> tuple[pd.Series, pd.Series]:
    normalized = values.fillna("").map(normalize_name)
    mapped = normalized.map(lookup).astype("string")
    status = np.select(
        [mapped.notna(), normalized.isin(ambiguous)],
        ["mapped_unique", "ambiguous_identifier"],
        default="unmapped",
    )
    return mapped, pd.Series(status, index=values.index, dtype="string")


def build_cancer_aliases(dim_cancer: pd.DataFrame) -> dict[str, str]:
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in dim_cancer.itertuples(index=False):
        cancer_id = str(row.cancer_id)
        values: list[object] = [
            cancer_id,
            getattr(row, "tcga_code", None),
            getattr(row, "english_name", None),
            getattr(row, "chinese_name", None),
        ]
        synonyms = getattr(row, "synonyms", None)
        if synonyms is not None and not pd.isna(synonyms):
            values.extend(re.split(r"[;|]", str(synonyms)))
        for value in values:
            key = normalize_name(value)
            if key:
                aliases[key].add(cancer_id)
    return {key: next(iter(ids)) for key, ids in aliases.items() if len(ids) == 1}


def map_cancer_value(
    disease_raw: object,
    aliases: dict[str, str],
    cancer_hint: object = None,
) -> tuple[object, str, str]:
    hint = _safe_text(cancer_hint).strip().upper()
    if hint in set(CANCER_PHRASES):
        return hint, hint, "mapped_tcga_hint"
    raw = "" if disease_raw is None or pd.isna(disease_raw) else str(disease_raw).strip()
    exact = aliases.get(normalize_name(raw))
    if exact:
        return exact, exact, "mapped_exact"
    normalized_text = re.sub(r"[^a-z0-9]+", " ", raw.lower()).strip()
    matches: list[tuple[int, str]] = []
    for cancer_id, phrases in CANCER_PHRASES.items():
        for phrase in phrases:
            phrase_text = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
            if re.search(rf"(?<![a-z0-9]){re.escape(phrase_text)}(?![a-z0-9])", normalized_text):
                matches.append((len(phrase_text), cancer_id))
    if matches:
        longest = max(length for length, _ in matches)
        ids = sorted({cancer_id for length, cancer_id in matches if length == longest})
        if len(ids) == 1:
            return ids[0], ids[0], "mapped_phrase"
        joined = ";".join(ids)
        return pd.NA, joined, "ambiguous_multiple_tcga"
    for candidate_ids, phrases in AMBIGUOUS_PHRASES.items():
        for phrase in phrases:
            phrase_text = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
            if re.search(rf"(?<![a-z0-9]){re.escape(phrase_text)}(?![a-z0-9])", normalized_text):
                return pd.NA, candidate_ids, "ambiguous_multiple_tcga"
    return pd.NA, "", "unmapped_non_tcga_disease"


def classify_lncrnadisease_method(value: object) -> tuple[str, bool, bool]:
    text = _safe_text(value).lower()
    has_experiment = any(token in text for token in EXPERIMENT_METHOD_TOKENS)
    has_prediction = any(token in text for token in PREDICTION_METHOD_TOKENS)
    if has_experiment:
        return "experimental_or_clinical", True, False
    if has_prediction:
        return "computational_prediction", False, True
    return "curated_unspecified", False, False


def _read_external_input(cfg: dict[str, Any], key: str) -> pd.DataFrame:
    path = input_path(cfg, key)
    if not path or not path.exists():
        raise FileNotFoundError(f"Required external-validation input is missing: {key} -> {path}")
    return read_table(path)


def _base_frame(
    source_database: str,
    source_row_id: pd.Series,
    lncrna_raw: pd.Series,
    disease_raw: pd.Series,
    method_raw: pd.Series,
    description: pd.Series,
    pmid: pd.Series,
    direction_raw: pd.Series,
    evidence_tier: pd.Series | str,
    is_experimental: pd.Series | bool,
    is_predicted: pd.Series | bool,
    cancer_hint: pd.Series | None = None,
) -> pd.DataFrame:
    n = len(lncrna_raw)
    def column(value: pd.Series | object, dtype: str | None = None) -> pd.Series:
        if isinstance(value, pd.Series):
            result = value.reset_index(drop=True)
        else:
            result = pd.Series([value] * n)
        return result.astype(dtype) if dtype else result

    return pd.DataFrame({
        "source_database": column(source_database, "string"),
        "source_row_id": column(source_row_id, "string"),
        "lncrna_raw": column(lncrna_raw, "string"),
        "disease_raw": column(disease_raw, "string"),
        "cancer_hint_raw": column(pd.NA if cancer_hint is None else cancer_hint, "string"),
        "method_raw": column(method_raw, "string"),
        "description": column(description, "string"),
        "pmid": column(pmid).map(normalize_pmid).astype("string"),
        "direction_raw": column(direction_raw, "string"),
        "direction": column(direction_raw).map(normalize_direction).astype("string"),
        "evidence_tier": column(evidence_tier, "string"),
        "is_experimental": column(is_experimental).astype(bool),
        "is_predicted": column(is_predicted).astype(bool),
        "dataset_accession": pd.Series([pd.NA] * n, dtype="string"),
        "cell_line": pd.Series([pd.NA] * n, dtype="string"),
        "growth_modifier_hit": pd.Series([False] * n, dtype=bool),
        "growth_effect_direction": pd.Series(["unknown"] * n, dtype="string"),
        "web_display": pd.Series([True] * n, dtype=bool),
    })


def read_lnc2cancer(cfg: dict[str, Any]) -> pd.DataFrame:
    raw = _read_external_input(cfg, "lnc2cancer_external")
    row_id = raw.index.to_series().map(lambda x: f"LNC2CANCER:{x + 1}")
    return _base_frame(
        "Lnc2Cancer",
        row_id,
        raw["name"],
        raw["cancer type"],
        raw["methods"],
        raw["function description"],
        raw["pubmed id"],
        raw["regulated"],
        "experimentally_supported_literature",
        True,
        False,
        raw.get("target_cancer_flag"),
    )


def read_lncrnadisease(cfg: dict[str, Any]) -> pd.DataFrame:
    raw = _read_external_input(cfg, "lncrnadisease_external")
    method = raw["Validated Method//Prediction Method"].fillna("")
    classified = method.map(classify_lncrnadisease_method)
    tier = classified.map(lambda x: x[0])
    experimental = classified.map(lambda x: x[1])
    predicted = classified.map(lambda x: x[2])
    row_id = raw.index.to_series().map(lambda x: f"LNCRNADISEASE:{x + 1}")
    return _base_frame(
        "LncRNADisease",
        row_id,
        raw["ncRNA Symbol"],
        raw["Disease Name"],
        method,
        raw["Description"],
        raw["PubMed ID"],
        raw["Dysfunction Pattern"],
        tier,
        experimental,
        predicted,
    )


def read_rnadisease_experimental(cfg: dict[str, Any]) -> pd.DataFrame:
    raw = _read_external_input(cfg, "rnadisease_experimental_external")
    empty = pd.Series([""] * len(raw), index=raw.index, dtype="string")
    return _base_frame(
        "RNADisease",
        raw["RDID"],
        raw["RNA Symbol"],
        raw["Disease Name"],
        pd.Series(["RNADisease experimental"] * len(raw), index=raw.index),
        pd.Series([""] * len(raw), index=raw.index),
        raw["PMID"],
        empty,
        "experimental_database_record",
        True,
        False,
    )


def read_rnadisease_predicted(cfg: dict[str, Any]) -> pd.DataFrame:
    raw = _read_external_input(cfg, "rnadisease_predicted_external")
    empty = pd.Series([""] * len(raw), index=raw.index, dtype="string")
    method = raw.get("method_name", empty).fillna("")
    return _base_frame(
        "RNADisease",
        raw["RDID"],
        raw["RNA_symbol"],
        raw["disease_name"],
        method,
        pd.Series([""] * len(raw), index=raw.index),
        empty,
        empty,
        "computational_prediction",
        False,
        True,
    )


def read_gse85011(cfg: dict[str, Any]) -> pd.DataFrame:
    raw = _read_external_input(cfg, "gse85011_sample_metadata")
    selected = raw.loc[
        raw["processed_data_file"].astype(str).str.contains("rnaseq_tpm", case=False, na=False)
        & ~raw["is_control"].fillna(False).astype(bool)
    ].copy()
    selected = selected.drop_duplicates(["cell_line", "target_raw"])
    cancer_map = {
        "HELA": "CESC",
        "MCF7": "BRCA",
        "MDAMB231": "BRCA",
        "MDA-MB-231": "BRCA",
        "U87": "GBM",
    }
    selected["cancer_hint"] = selected.cell_line.astype(str).str.upper().map(cancer_map).fillna("")
    frame = _base_frame(
        "GSE85011",
        selected["gsm"],
        selected["target_raw"],
        selected["cell_line"].map(lambda x: f"CRISPRi cell-line growth screen: {x}"),
        pd.Series(["CRISPRi followed by RNA-seq"] * len(selected), index=selected.index),
        pd.Series(
            ["Growth-modifier lncRNA hit selected for follow-up RNA-seq; per-locus screen score is not present in GEO."]
            * len(selected),
            index=selected.index,
        ),
        pd.Series([GSE85011_PMID] * len(selected), index=selected.index),
        pd.Series([""] * len(selected), index=selected.index),
        "experimental_crispri_growth_modifier",
        True,
        False,
        selected["cancer_hint"],
    )
    frame["dataset_accession"] = GSE85011_ACCESSION
    frame["cell_line"] = selected["cell_line"].astype("string").reset_index(drop=True)
    frame["growth_modifier_hit"] = True
    frame["validation_scope"] = np.where(
        selected["cancer_hint"].astype(str).reset_index(drop=True).ne(""),
        "matched_tcga_context",
        "web_only_unmatched_cell_context",
    )
    return frame


def training_pmids(cfg: dict[str, Any]) -> set[str]:
    pmids: set[str] = set()
    for name in ("interaction_relation.parquet", "evidence_event.parquet"):
        path = cfg["_standardized"] / name
        if not path.exists():
            continue
        frame = read_table(path, columns=["pmid"])
        pmids.update(value for value in frame.pmid.map(normalize_pmid) if value)
    return pmids


def _source_values(path: Path) -> set[str]:
    if not path.exists():
        return set()
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    values: set[str] = set()
    scanner = ds.dataset(path, format="parquet").scanner(
        columns=["source_database"], batch_size=1_000_000
    )
    for batch in scanner.to_batches():
        values.update(
            str(value)
            for value in pc.unique(batch.column("source_database")).to_pylist()
            if value is not None
        )
    return values


def standardize_external_evidence(cfg: dict[str, Any]) -> dict[str, Any]:
    dim_lnc = read_table(cfg["_standardized"] / "dim_lncRNA.parquet")
    dim_cancer = read_table(cfg["_standardized"] / "dim_cancer.parquet")
    lnc_lookup, ambiguous_lnc = build_lnc_lookup(dim_lnc)
    cancer_aliases = build_cancer_aliases(dim_cancer)
    train_pmids = training_pmids(cfg)

    disease = pd.concat(
        [
            read_lnc2cancer(cfg),
            read_lncrnadisease(cfg),
            read_rnadisease_experimental(cfg),
            read_rnadisease_predicted(cfg),
        ],
        ignore_index=True,
    )
    gse = read_gse85011(cfg)
    combined = pd.concat([disease, gse], ignore_index=True)

    combined["lncrna_id"], combined["lncrna_mapping_status"] = map_lnc_values(
        combined.lncrna_raw, lnc_lookup, ambiguous_lnc
    )
    # Old CRiNCL names sometimes carry guide-location suffixes rather than a
    # distinct lncRNA symbol. Retry only unresolved GSE rows after stripping
    # these known suffixes.
    gse_unmapped = combined.source_database.eq("GSE85011") & combined.lncrna_id.isna()
    if gse_unmapped.any():
        retry = (
            combined.loc[gse_unmapped, "lncrna_raw"]
            .astype(str)
            .str.replace(r"(?:_x|up)$", "", regex=True, case=False)
        )
        retry_id, retry_status = map_lnc_values(retry, lnc_lookup, ambiguous_lnc)
        combined.loc[gse_unmapped, "lncrna_id"] = retry_id
        combined.loc[gse_unmapped, "lncrna_mapping_status"] = retry_status

    cancer_cache: dict[tuple[str, str], tuple[object, str, str]] = {}
    for disease_raw, cancer_hint in combined[["disease_raw", "cancer_hint_raw"]].drop_duplicates().itertuples(index=False):
        key = (_safe_text(disease_raw), _safe_text(cancer_hint))
        cancer_cache[key] = map_cancer_value(disease_raw, cancer_aliases, cancer_hint)
    mapped = [
        cancer_cache[(_safe_text(disease_raw), _safe_text(cancer_hint))]
        for disease_raw, cancer_hint in combined[["disease_raw", "cancer_hint_raw"]].itertuples(index=False)
    ]
    combined["cancer_id"] = pd.Series([item[0] for item in mapped], dtype="string")
    combined["candidate_cancer_ids"] = pd.Series([item[1] for item in mapped], dtype="string")
    combined["cancer_mapping_status"] = pd.Series([item[2] for item in mapped], dtype="string")
    combined["training_pmid_overlap"] = combined.pmid.astype(str).isin(train_pmids) & combined.pmid.astype(str).ne("")
    combined["validation_role"] = np.select(
        [
            combined.source_database.eq("GSE85011"),
            combined.is_predicted.fillna(False).astype(bool),
        ],
        ["functional_web_and_external", "secondary_consistency"],
        default="primary_cancer_relevance",
    )
    mapped_context = combined.lncrna_id.notna() & combined.cancer_id.notna()
    combined["primary_validation_eligible"] = (
        mapped_context
        & ~combined.training_pmid_overlap
        & combined.is_experimental.fillna(False).astype(bool)
        & ~combined.is_predicted.fillna(False).astype(bool)
    )
    combined["consistency_validation_eligible"] = (
        mapped_context & combined.is_predicted.fillna(False).astype(bool)
    )
    combined["external_evidence_id"] = [
        stable_id("EXT", source, row_id)
        for source, row_id in zip(combined.source_database, combined.source_row_id)
    ]
    combined["independent_event_hash"] = [
        stable_id(
            "EXTEVT",
            lnc if not pd.isna(lnc) else raw,
            cancer if not pd.isna(cancer) else disease_raw,
            pmid,
            tier,
        )
        for lnc, raw, cancer, disease_raw, pmid, tier in zip(
            combined.lncrna_id,
            combined.lncrna_raw,
            combined.cancer_id,
            combined.disease_raw,
            combined.pmid,
            combined.evidence_tier,
        )
    ]
    combined["analysis_version"] = cfg["analysis_version"]

    std = cfg["_standardized"]
    out_dir = cfg["_results"] / "tables" / "external_validation"
    write_table(combined, std / "external_validation_evidence.parquet")
    write_table(
        combined.loc[combined.source_database.ne("GSE85011")].copy(),
        std / "external_lncRNA_disease_evidence.parquet",
    )
    write_table(
        combined.loc[combined.source_database.eq("GSE85011")].copy(),
        std / "gse85011_growth_modifier_evidence.parquet",
    )
    write_table(
        combined.loc[
            combined.primary_validation_eligible | combined.consistency_validation_eligible
        ].copy(),
        out_dir / "external_validation_eligible.parquet",
    )
    write_table(
        combined.loc[combined.web_display.fillna(False).astype(bool)].copy(),
        out_dir / "web_external_evidence.parquet",
    )

    source_summary = (
        combined.groupby(["source_database", "evidence_tier"], dropna=False, as_index=False)
        .agg(
            records=("external_evidence_id", "size"),
            mapped_lncRNA_records=("lncrna_id", lambda x: int(x.notna().sum())),
            mapped_cancer_records=("cancer_id", lambda x: int(x.notna().sum())),
            unique_lncRNAs=("lncrna_id", "nunique"),
            unique_cancers=("cancer_id", "nunique"),
            independent_pmids=("pmid", lambda x: int(pd.Series(x)[pd.Series(x).astype(str).ne("")].nunique())),
            training_pmid_overlap_records=("training_pmid_overlap", "sum"),
            primary_validation_records=("primary_validation_eligible", "sum"),
            consistency_validation_records=("consistency_validation_eligible", "sum"),
        )
    )
    write_table(source_summary, out_dir / "external_validation_source_summary.tsv")

    gse_rows = combined.loc[combined.source_database.eq("GSE85011")]
    gse_name_keys = set(gse_rows.lncrna_raw.map(normalize_name)) - {""}
    gse_ids = set(gse_rows.lncrna_id.dropna().astype(str))
    overlap_rows: list[dict[str, object]] = []
    for source, group in combined.loc[
        combined.source_database.ne("GSE85011")
    ].groupby("source_database", observed=True):
        source_name_keys = set(group.lncrna_raw.map(normalize_name)) - {""}
        source_ids = set(group.lncrna_id.dropna().astype(str))
        overlap_rows.append(
            {
                "source_database": source,
                "gse85011_pmid_record_overlap": int(
                    group.pmid.astype(str).eq(GSE85011_PMID).sum()
                ),
                "unique_normalized_name_overlap": len(source_name_keys & gse_name_keys),
                "unique_mapped_lncRNA_id_overlap": len(source_ids & gse_ids),
                "interpretation": (
                    "Identifier overlap is expected target-space overlap, not evidence "
                    "independence. Direct PMID overlap is separately reported."
                ),
            }
        )
    overlap = pd.DataFrame(overlap_rows)
    write_table(overlap, out_dir / "gse85011_external_database_overlap.tsv")

    isolated_sources = {"Lnc2Cancer", "LncRNADisease", "RNADisease", "GSE85011"}
    graph_sources = _source_values(cfg["_results"] / "tables" / "graph_edge.parquet")
    pair_sources = _source_values(
        cfg["_results"] / "tables" / "pair_evidence_source_contribution.parquet"
    )
    leaked_graph_sources = sorted(graph_sources & isolated_sources)
    leaked_pair_sources = sorted(pair_sources & isolated_sources)
    leakage = pd.DataFrame(
        [
            {
                "check": "external_sources_absent_from_training_graph_and_pair_labels",
                "status": (
                    "PASS"
                    if not leaked_graph_sources and not leaked_pair_sources
                    else "FAIL"
                ),
                "detail": (
                    f"graph_source_overlap={leaked_graph_sources}; "
                    f"pair_source_overlap={leaked_pair_sources}. External sources "
                    "are not appended to labels, calibration or early stopping."
                ),
            },
            {
                "check": "training_pmid_overlap_flagged",
                "status": "PASS" if int(combined.training_pmid_overlap.sum()) >= 0 else "FAIL",
                "detail": f"overlap_records={int(combined.training_pmid_overlap.sum())}; overlapping records are excluded from primary validation",
            },
            {
                "check": "rnadisease_predicted_not_primary_truth",
                "status": "PASS" if not combined.loc[combined.is_predicted, "primary_validation_eligible"].any() else "FAIL",
                "detail": "Predicted records are secondary consistency evidence only.",
            },
            {
                "check": "gse85011_direct_pmid_overlap_reported",
                "status": "PASS",
                "detail": (
                    f"PMID {GSE85011_PMID} record overlap: "
                    + ", ".join(
                        f"{row.source_database}={row.gse85011_pmid_record_overlap}"
                        for row in overlap.itertuples(index=False)
                    )
                    + ". Identifier overlap is retained as target-space overlap only."
                ),
            },
            {
                "check": "gse85011_growth_direction_not_inferred",
                "status": "PASS" if combined.loc[combined.source_database.eq("GSE85011"), "growth_effect_direction"].eq("unknown").all() else "FAIL",
                "detail": "GEO identifies selected growth-modifier hits but does not publish per-locus screen direction in the downloaded Series tables.",
            },
        ]
    )
    write_table(leakage, out_dir / "external_validation_leakage_audit.tsv")
    study = pd.DataFrame(
        [
            {
                "dataset_accession": GSE85011_ACCESSION,
                "pmid": GSE85011_PMID,
                "screened_lncRNA_loci": 16401,
                "reported_growth_modifier_hits": 499,
                "reported_cell_type_specific_fraction": 0.89,
                "selected_followup_loci_in_geo": int(
                    combined.loc[combined.source_database.eq("GSE85011"), "lncrna_raw"].nunique()
                ),
                "per_locus_growth_score_available_in_geo": False,
                "web_interpretation": "Experimental growth-modifier hit selected for follow-up RNA-seq; do not infer direction or pathway from GEO alone.",
                "source_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE85011",
            }
        ]
    )
    write_table(study, out_dir / "gse85011_study_summary.tsv")
    manifest = {
        "generated_at": utc_now(),
        "analysis_version": cfg["analysis_version"],
        "records": int(len(combined)),
        "primary_validation_records": int(combined.primary_validation_eligible.sum()),
        "secondary_consistency_records": int(combined.consistency_validation_eligible.sum()),
        "training_pmid_overlap_records": int(combined.training_pmid_overlap.sum()),
        "sources": sorted(combined.source_database.unique().tolist()),
        "training_isolation_policy": (
            "External evidence is never appended to interaction_relation, graph_edge, "
            "pair_evidence, labels, calibration, or early stopping."
        ),
    }
    write_json(manifest, out_dir / "external_validation_manifest.json")
    return manifest


def _prediction_files(cfg: dict[str, Any]) -> list[Path]:
    root = cfg["_results"] / "tables" / "all_candidate_prediction"
    return sorted(root.glob("cancer_id=*/part-0.parquet"))


def build_lnc_prediction_ranks(cfg: dict[str, Any]) -> pd.DataFrame:
    top_n = int(cfg.get("external_validation", {}).get("top_pathways_per_lncRNA", 5))
    frames: list[pd.DataFrame] = []
    for path in _prediction_files(cfg):
        prediction = read_table(path)
        score_col = (
            "calibrated_probability"
            if "calibrated_probability" in prediction
            else "raw_probability"
        )
        required = {"cancer_id", "lncrna_id", "pathway_family_id", score_col}
        missing = required - set(prediction)
        if missing:
            raise ValueError(f"{path} is missing external-validation columns: {sorted(missing)}")
        prediction = prediction.sort_values(score_col, ascending=False)
        top = prediction.groupby(["cancer_id", "lncrna_id"], observed=True, sort=False).head(top_n)
        rank = (
            top.groupby(["cancer_id", "lncrna_id"], as_index=False, observed=True)
            .agg(
                max_pathway_probability=(score_col, "max"),
                mean_top_pathway_probability=(score_col, "mean"),
                n_top_pathways=(score_col, "size"),
                top_pathway_family_id=("pathway_family_id", "first"),
                model_name=("model_name", "first"),
                fold_id=("fold_id", "first"),
            )
        )
        rank["lncrna_rank_score"] = (
            0.7 * rank.max_pathway_probability + 0.3 * rank.mean_top_pathway_probability
        )
        rank = rank.sort_values(
            ["cancer_id", "lncrna_rank_score"], ascending=[True, False]
        )
        rank["rank"] = rank.groupby("cancer_id", observed=True).cumcount() + 1
        rank["n_ranked_lncRNAs"] = rank.groupby("cancer_id", observed=True).lncrna_id.transform("size")
        rank["percentile"] = 1.0 - (rank["rank"] - 1) / rank.n_ranked_lncRNAs.clip(lower=1)
        frames.append(rank)
    if not frames:
        raise FileNotFoundError(
            f"No all-candidate prediction partitions found under {cfg['_results'] / 'tables' / 'all_candidate_prediction'}"
        )
    return pd.concat(frames, ignore_index=True)


def _evaluate_role(
    ranks: pd.DataFrame,
    evidence: pd.DataFrame,
    role: str,
    ks: Iterable[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if role == "primary":
        positives = evidence.loc[evidence.primary_validation_eligible].copy()
    else:
        positives = evidence.loc[evidence.consistency_validation_eligible].copy()
    positives = positives.dropna(subset=["lncrna_id", "cancer_id"])
    positives = positives.drop_duplicates(["source_database", "cancer_id", "lncrna_id"])
    matches = positives.merge(
        ranks,
        on=["cancer_id", "lncrna_id"],
        how="inner",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for (source, cancer), group in positives.groupby(
        ["source_database", "cancer_id"], observed=True
    ):
        ranked = ranks.loc[ranks.cancer_id.astype(str).eq(str(cancer))]
        matched = matches.loc[
            matches.source_database.eq(source)
            & matches.cancer_id.astype(str).eq(str(cancer))
        ]
        n_candidates = int(ranked.n_ranked_lncRNAs.max()) if len(ranked) else 0
        n_positive = int(group.lncrna_id.nunique())
        n_matched = int(matched.lncrna_id.nunique())
        base = {
            "validation_role": role,
            "source_database": source,
            "cancer_id": cancer,
            "n_external_positive_lncRNAs": n_positive,
            "n_external_positive_in_candidate_universe": n_matched,
            "n_ranked_lncRNAs": n_candidates,
            "median_positive_percentile": float(matched.percentile.median()) if len(matched) else np.nan,
        }
        for k in ks:
            effective_k = min(int(k), n_candidates)
            hits = int(matched.loc[matched["rank"] <= effective_k, "lncrna_id"].nunique())
            expected = effective_k * n_matched / n_candidates if n_candidates else np.nan
            rows.append({
                **base,
                "k": int(k),
                "effective_k": effective_k,
                "hits_at_k": hits,
                "annotation_hit_rate_at_k": hits / effective_k if effective_k else np.nan,
                "recall_at_k": hits / n_matched if n_matched else np.nan,
                "fold_enrichment_at_k": hits / expected if expected and expected > 0 else np.nan,
            })
    return pd.DataFrame(rows), matches


def run_external_validation(cfg: dict[str, Any]) -> dict[str, Any]:
    evidence_path = cfg["_standardized"] / "external_validation_evidence.parquet"
    if not evidence_path.exists():
        raise FileNotFoundError(
            f"{evidence_path} is missing; run 01b_standardize_external_validation.py first"
        )
    evidence = read_table(evidence_path)
    ranks = build_lnc_prediction_ranks(cfg)
    settings = cfg.get("external_validation", {})
    ks = settings.get("rank_cutoffs", [10, 25, 50, 100])
    primary_metrics, primary_matches = _evaluate_role(ranks, evidence, "primary", ks)
    secondary_metrics, secondary_matches = _evaluate_role(ranks, evidence, "secondary_consistency", ks)
    metrics = pd.concat([primary_metrics, secondary_metrics], ignore_index=True)
    matches = pd.concat([primary_matches, secondary_matches], ignore_index=True)
    out_dir = cfg["_results"] / "tables" / "external_validation"
    write_table(ranks, out_dir / "lncrna_prediction_rank.parquet")
    write_table(metrics, out_dir / "external_validation_metrics.tsv")
    write_table(matches, out_dir / "external_validation_prediction_match.parquet")
    summary = {
        "generated_at": utc_now(),
        "analysis_version": cfg["analysis_version"],
        "ranked_lncRNA_contexts": int(len(ranks)),
        "primary_external_matches": int(len(primary_matches)),
        "secondary_consistency_matches": int(len(secondary_matches)),
        "metric_rows": int(len(metrics)),
        "interpretation": (
            "External databases contain incomplete positive annotations. "
            "annotation_hit_rate_at_k is not classical precision against verified negatives."
        ),
    }
    write_json(summary, out_dir / "external_validation_result_summary.json")
    return summary


def external_report_markdown(cfg: dict[str, Any]) -> str:
    out_dir = cfg["_results"] / "tables" / "external_validation"
    source_path = out_dir / "external_validation_source_summary.tsv"
    metric_path = out_dir / "external_validation_metrics.tsv"
    leakage_path = out_dir / "external_validation_leakage_audit.tsv"
    sections = ["## Independent external validation"]
    sections.append(
        "Lnc2Cancer, LncRNADisease, experimental RNADisease and GSE85011 are "
        "kept outside the training graph, pair evidence, labels, calibration and early stopping. "
        "RNADisease predicted records are secondary consistency evidence only."
    )
    if source_path.exists():
        source = read_table(source_path)
        sections.extend(["### External evidence standardization", source.to_markdown(index=False)])
    if leakage_path.exists():
        leakage = read_table(leakage_path)
        sections.extend(["### Leakage audit", leakage.to_markdown(index=False)])
    if metric_path.exists():
        metrics = read_table(metric_path)
        display = metrics.loc[
            metrics.k.isin([25, 100]),
            [
                "validation_role",
                "source_database",
                "cancer_id",
                "k",
                "hits_at_k",
                "recall_at_k",
                "fold_enrichment_at_k",
            ],
        ]
        sections.extend(["### Frozen-model ranking validation", display.to_markdown(index=False)])
    sections.append(
        "GSE85011 web rows represent experimental growth-modifier hits selected for "
        "follow-up RNA-seq. GEO does not provide the per-locus screen direction in its "
        "Series tables, so direction remains `unknown` rather than being inferred."
    )
    return "\n\n".join(sections)
