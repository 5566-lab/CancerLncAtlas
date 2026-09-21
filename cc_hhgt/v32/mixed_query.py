"""Exact-pathway enrichment for mixed lncRNA and protein-coding gene lists.

Protein-coding genes use classical hypergeometric over-representation analysis
against the registered exact-pathway membership universe.  lncRNAs use only a
hash-registered, newly trained V3.2 exact-pathway association table.  The two
channels are reported separately and combined only as a descriptive ranking
score; the combination is neither a probability nor a joint p-value.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .release_registry import (
    V32_PREFIX,
    ValidatedReleaseRegistry,
    load_release_registry,
)


ASSOCIATION_PROBABILITY = "association_membership_probability"
KEY_COVERAGE_FORMAT = "CANCERLNCATLAS_V32_MIXED_QUERY_KEY_COVERAGE_V1"
KEY_COVERAGE_SCOPES = frozenset(
    {
        "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_CARTESIAN",
        "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_OBSERVED_ROWS",
    }
)
NOT_EVALUATED_REASON = "NOT_EVALUATED"
MAX_MEMBERS = 100
MAX_TOP_K = 100
_ALIAS_SPLIT = re.compile(r"[;,|]")
_ENSEMBL_VERSION = re.compile(r"^(ENSG\d+)\.\d+$", re.IGNORECASE)
_STATIC_RESULT_COLUMN = re.compile(
    r"label|prediction|probabilit|ranking|(^|_)score($|_)|checkpoint|oof",
    re.IGNORECASE,
)
_OLD_ASSOCIATION_COLUMN = re.compile(
    r"legacy|historical|(?:^|_)v2(?:_|$)|(?:^|_)v3[_\-.]?[01](?:_|$)",
    re.IGNORECASE,
)
_PRIVATE_ASSOCIATION_COLUMNS = frozenset(
    {
        "label",
        "label_class",
        "proxy_label",
        "association_proxy_label",
        "held_out_proxy_label",
        "heldout_proxy_label",
        "sample_weight",
        "patient_fold_id",
        "fold_id",
        "l1_probability",
        "ridge_probability",
        "family_probability",
        "pathway_family_probability",
    }
)


class MixedQueryError(RuntimeError):
    """Base error for a rejected mixed exact-pathway query or asset."""


class MixedQueryAssetError(MixedQueryError):
    """Raised when a registered query asset violates public V3.2 semantics."""


class MixedQueryInputError(MixedQueryError):
    """Raised when a user list cannot be queried unambiguously."""


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise MixedQueryAssetError(f"Unsupported registered table format: {path}")


def _normal_identifier(value: Any) -> str:
    text = str(value).strip().upper()
    match = _ENSEMBL_VERSION.fullmatch(text)
    return match.group(1) if match else text


def _canonical_gene(value: Any) -> str:
    text = _normal_identifier(value)
    if text.startswith("GENE:"):
        text = text[5:]
    match = _ENSEMBL_VERSION.fullmatch(text)
    return match.group(1) if match else text


def _canonical_lnc(value: Any) -> str:
    text = _normal_identifier(value)
    if text.startswith("GENE:"):
        text = text[5:]
    if not text.startswith("LNC:"):
        text = "LNC:" + text
    inner = text[4:]
    match = _ENSEMBL_VERSION.fullmatch(inner)
    return "LNC:" + (match.group(1) if match else inner)


def _normal_entity_type(value: Any) -> str | None:
    text = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    if text in {"lncrna", "lnc_rna", "long_noncoding_rna", "long_non_coding_rna"}:
        return "lncRNA"
    if text in {"protein_coding_gene", "protein_coding", "gene", "hgnc_gene"}:
        return "protein_coding_gene"
    return None


def _safe_value(row: Mapping[str, Any], name: str) -> Any | None:
    value = row.get(name)
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    return value


def _iter_aliases(value: Any) -> Iterable[str]:
    if value is None or pd.isna(value):
        return ()
    return (item.strip() for item in _ALIAS_SPLIT.split(str(value)) if item.strip())


def _bh_adjust(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment preserving the input index."""

    result = pd.Series(np.nan, index=values.index, dtype=float)
    finite = pd.to_numeric(values, errors="coerce").dropna()
    if finite.empty:
        return result
    ordered = finite.sort_values(kind="mergesort")
    count = len(ordered)
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    raw = ordered.to_numpy(float)
    for offset in range(count - 1, -1, -1):
        rank = offset + 1
        running = min(running, raw[offset] * count / rank)
        adjusted[offset] = min(1.0, running)
    result.loc[ordered.index] = adjusted
    return result


def hypergeometric_overrepresentation(
    *,
    overlap: int,
    pathway_size: int,
    query_size: int,
    universe_size: int,
) -> float:
    """Return P[X >= overlap] for a classical one-sided ORA test."""

    values = (overlap, pathway_size, query_size, universe_size)
    if any(int(value) != value or value < 0 for value in values):
        raise ValueError("ORA counts must be non-negative integers")
    overlap = int(overlap)
    pathway_size = int(pathway_size)
    query_size = int(query_size)
    universe_size = int(universe_size)
    if pathway_size > universe_size or query_size > universe_size:
        raise ValueError("ORA pathway/query size exceeds the background universe")
    if overlap > min(pathway_size, query_size):
        raise ValueError("ORA overlap exceeds the pathway or query size")
    if query_size == 0 or overlap == 0:
        return 1.0
    return float(hypergeom.sf(overlap - 1, universe_size, pathway_size, query_size))


def _json_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if value is None or (isinstance(value, float) and not math.isfinite(value)) or bool(pd.isna(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


class MixedExactPathwayQuery:
    """In-memory query store created only from a validated release registry."""

    def __init__(self, registry: ValidatedReleaseRegistry) -> None:
        capability = registry.manifest["capabilities"]["mixed_exact_pathway_query"]
        if capability.get("enabled") is not True:
            raise MixedQueryAssetError("The registered mixed exact-pathway query is disabled")
        self.registry = registry
        self.identifier_map = _read_table(registry.artifact("mixed_query_identifier_map"))
        self.membership = _read_table(
            registry.artifact("mixed_query_exact_pathway_membership")
        )
        self.association = _read_table(
            registry.artifact("mixed_query_v32_lnc_exact_association")
        )
        metadata_path = registry.artifacts.get("mixed_query_pathway_metadata")
        self.pathway_metadata = _read_table(metadata_path) if metadata_path else pd.DataFrame()
        manifest_path = registry.artifacts.get("mixed_query_asset_manifest")
        if manifest_path is None:
            raise MixedQueryAssetError(
                "The enabled mixed query lacks its hash-registered asset manifest"
            )
        try:
            self.asset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MixedQueryAssetError("The mixed-query asset manifest is unreadable") from exc
        self._validate_static_assets()
        self._identifier_index = self._build_identifier_index()
        self._lncrna_identifier_ids = frozenset(
            canonical
            for matches in self._identifier_index.values()
            for entity, canonical in matches
            if entity == "lncRNA"
        )
        self._protein_identifier_ids = frozenset(
            canonical
            for matches in self._identifier_index.values()
            for entity, canonical in matches
            if entity == "protein_coding_gene"
        )
        self._validate_and_normalize_membership()
        self._validate_and_normalize_association()
        self._validate_key_coverage_contract()
        self._build_pathway_index()

    @classmethod
    def from_registry(cls, path: str | Path) -> "MixedExactPathwayQuery":
        return cls(load_release_registry(path, require_staging=True))

    def _validate_static_assets(self) -> None:
        if self.identifier_map.empty:
            raise MixedQueryAssetError("The identifier map is empty")
        if self.membership.empty:
            raise MixedQueryAssetError("The exact-pathway membership table is empty")
        for name, frame in (
            ("identifier map", self.identifier_map),
            ("exact-pathway membership", self.membership),
        ):
            result_columns = sorted(
                column for column in frame.columns if _STATIC_RESULT_COLUMN.search(str(column))
            )
            if result_columns:
                raise MixedQueryAssetError(f"{name} contains model-result columns: {result_columns}")

    def _add_identifier(
        self,
        index: dict[str, set[tuple[str, str]]],
        value: Any,
        entity_type: str,
        canonical_id: str,
    ) -> None:
        if value is None or pd.isna(value) or not str(value).strip():
            return
        key = _normal_identifier(value)
        index[key].add((entity_type, canonical_id))
        if entity_type == "lncRNA" and canonical_id.startswith("LNC:"):
            index[_normal_identifier(canonical_id[4:])].add((entity_type, canonical_id))
        if entity_type == "protein_coding_gene":
            index[_normal_identifier("GENE:" + canonical_id)].add((entity_type, canonical_id))

    def _build_identifier_index(self) -> dict[str, set[tuple[str, str]]]:
        columns = set(self.identifier_map.columns)
        long_form = {"identifier", "canonical_id", "entity_type"}.issubset(columns)
        wide_form = "entity_type" in columns and bool(
            columns & {"gene_id", "lncrna_id", "ensembl_gene_id", "gene_symbol"}
        )
        if not long_form and not wide_form:
            raise MixedQueryAssetError(
                "Identifier map requires long columns identifier/canonical_id/entity_type "
                "or the static wide identifier schema"
            )
        index: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for raw_row in self.identifier_map.to_dict("records"):
            entity_type = _normal_entity_type(raw_row.get("entity_type"))
            if entity_type is None:
                continue
            if long_form:
                raw_canonical = _safe_value(raw_row, "canonical_id")
            elif entity_type == "lncRNA":
                raw_canonical = (
                    _safe_value(raw_row, "lncrna_id")
                    or _safe_value(raw_row, "ensembl_gene_id")
                    or _safe_value(raw_row, "gene_id")
                )
            else:
                raw_canonical = (
                    _safe_value(raw_row, "ensembl_gene_id")
                    or _safe_value(raw_row, "gene_id")
                )
            if raw_canonical is None:
                continue
            canonical = (
                _canonical_lnc(raw_canonical)
                if entity_type == "lncRNA"
                else _canonical_gene(raw_canonical)
            )
            values: list[Any] = [raw_canonical, canonical]
            for column in ("identifier", "gene_id", "lncrna_id", "ensembl_gene_id", "gene_symbol"):
                value = _safe_value(raw_row, column)
                if value is not None:
                    values.append(value)
            values.extend(_iter_aliases(raw_row.get("aliases")))
            for value in values:
                self._add_identifier(index, value, entity_type, canonical)
        if not any(entity == "lncRNA" for matches in index.values() for entity, _ in matches):
            raise MixedQueryAssetError("Identifier map contains no lncRNA identifiers")
        if not any(
            entity == "protein_coding_gene" for matches in index.values() for entity, _ in matches
        ):
            raise MixedQueryAssetError("Identifier map contains no protein-coding gene identifiers")
        return dict(index)

    def _resolve_static_gene(self, value: Any) -> str:
        key = _normal_identifier(value)
        matches = {
            canonical
            for entity, canonical in self._identifier_index.get(key, set())
            if entity == "protein_coding_gene"
        }
        return next(iter(matches)) if len(matches) == 1 else _canonical_gene(value)

    def _validate_and_normalize_membership(self) -> None:
        required = {"pathway_id", "gene_id"}
        missing = sorted(required - set(self.membership.columns))
        if missing:
            raise MixedQueryAssetError(f"Exact-pathway membership lacks columns: {missing}")
        frame = self.membership.copy()
        frame["pathway_id"] = frame["pathway_id"].astype(str).str.strip()
        frame["gene_id"] = frame["gene_id"].map(self._resolve_static_gene)
        frame = frame.loc[frame.pathway_id.ne("") & frame.gene_id.ne("")].copy()
        if frame.empty:
            raise MixedQueryAssetError("Exact-pathway membership has no usable rows")
        self.membership = frame.drop_duplicates(["pathway_id", "gene_id"]).reset_index(drop=True)
        non_protein = sorted(set(self.membership.gene_id) - set(self._protein_identifier_ids))
        if non_protein:
            raise MixedQueryAssetError(
                "Protein ORA membership contains identifiers that are not registered "
                f"protein-coding genes: {non_protein[:5]}"
            )

    def _validate_and_normalize_association(self) -> None:
        required = {
            "cancer_id",
            "lncrna_id",
            "pathway_id",
            ASSOCIATION_PROBABILITY,
            "analysis_version",
            "training_run_id",
            "pathway_target_level",
            "availability",
            "eligible_for_mixed_query",
        }
        missing = sorted(required - set(self.association.columns))
        if missing:
            raise MixedQueryAssetError(f"V3.2 lnc exact-pathway association lacks columns: {missing}")
        lower_columns = {str(column).casefold(): str(column) for column in self.association.columns}
        private = sorted(set(lower_columns) & _PRIVATE_ASSOCIATION_COLUMNS)
        historical = sorted(
            column for column in self.association.columns if _OLD_ASSOCIATION_COLUMN.search(str(column))
        )
        extra_probability = sorted(
            column
            for column in self.association.columns
            if "probability" in str(column).casefold() and column != ASSOCIATION_PROBABILITY
        )
        if private or historical or extra_probability:
            raise MixedQueryAssetError(
                "The public lnc association asset contains private/old/extra result columns: "
                f"private={private}, historical={historical}, extra_probability={extra_probability}"
            )
        if "family_to_exact_broadcast" in self.association and self.association[
            "family_to_exact_broadcast"
        ].fillna(False).astype(bool).any():
            raise MixedQueryAssetError("Family-to-exact score broadcasting is forbidden")
        for column in ("source_target_level", "prediction_target_level"):
            if column in self.association and not self.association[column].astype(str).eq(
                "exact_pathway"
            ).all():
                raise MixedQueryAssetError(f"{column} contains a non-exact source target")
        if not self.association["pathway_target_level"].astype(str).eq("exact_pathway").all():
            raise MixedQueryAssetError("lnc associations are not uniformly exact-pathway predictions")
        if not self.association["analysis_version"].astype(str).str.startswith(V32_PREFIX).all():
            raise MixedQueryAssetError("lnc associations are not uniformly current V3.2 results")

        exact_module = self.registry.manifest["modules"]["exact_pathway"]
        expected_version = str(exact_module["analysis_version"])
        expected_run = str(exact_module["training_run_id"])
        if set(self.association.analysis_version.astype(str)) != {expected_version}:
            raise MixedQueryAssetError("lnc association version does not match the registered exact model")
        if set(self.association.training_run_id.astype(str)) != {expected_run}:
            raise MixedQueryAssetError("lnc association run does not match the registered exact model")

        frame = self.association.copy()
        frame["cancer_id"] = frame["cancer_id"].astype(str).str.strip().str.upper()
        frame["lncrna_id"] = frame["lncrna_id"].map(_canonical_lnc)
        frame["pathway_id"] = frame["pathway_id"].astype(str).str.strip()
        frame[ASSOCIATION_PROBABILITY] = pd.to_numeric(
            frame[ASSOCIATION_PROBABILITY], errors="coerce"
        )
        unknown_lnc = sorted(set(frame.lncrna_id) - set(self._lncrna_identifier_ids))
        if unknown_lnc:
            raise MixedQueryAssetError(
                "V3.2 predictions contain lncRNAs that cannot be reached through the "
                f"registered identifier map: {unknown_lnc[:5]}"
            )
        availability = frame["availability"].fillna(False).astype(bool)
        eligibility = frame["eligible_for_mixed_query"].fillna(False).astype(bool)
        if not eligibility.all():
            raise MixedQueryAssetError(
                "The public mixed-query association must contain only eligible keys"
            )
        values = frame.loc[availability, ASSOCIATION_PROBABILITY]
        if values.isna().any() or not values.between(0.0, 1.0).all():
            raise MixedQueryAssetError("Available lnc exact-pathway probabilities must be finite in [0,1]")
        unavailable = ~availability
        if frame.loc[unavailable, ASSOCIATION_PROBABILITY].notna().any():
            raise MixedQueryAssetError(
                "Unavailable lnc exact-pathway keys must preserve a null probability"
            )
        if unavailable.any():
            if "availability_reason" not in frame:
                raise MixedQueryAssetError(
                    "Unavailable lnc exact-pathway keys require availability_reason"
                )
            reasons = frame.loc[unavailable, "availability_reason"].fillna("").astype(str).str.strip()
            if reasons.eq("").any():
                raise MixedQueryAssetError(
                    "Unavailable lnc exact-pathway keys require a non-empty reason"
                )
        if not availability.any():
            raise MixedQueryAssetError("The registered lnc association table has no eligible rows")
        keys = ["cancer_id", "lncrna_id", "pathway_id"]
        if frame.duplicated(keys).any():
            raise MixedQueryAssetError(
                "lnc association keys are duplicated; fold outputs must be consolidated before staging"
            )
        known_pathways = set(self.membership.pathway_id)
        association_pathways = set(frame.pathway_id)
        unknown = sorted(association_pathways - known_pathways)
        if unknown:
            raise MixedQueryAssetError(
                f"lnc association contains exact pathways without static gene membership: {unknown[:5]}"
            )
        extra = sorted(known_pathways - association_pathways)
        if extra:
            raise MixedQueryAssetError(
                "Static membership exceeds the current V3.2 exact-pathway prediction universe: "
                f"{extra[:5]}"
            )
        self.association_all = frame.reset_index(drop=True)
        self.association = frame.loc[availability].reset_index(drop=True)

    def _validate_key_coverage_contract(self) -> None:
        coverage = self.asset_manifest.get("key_coverage")
        if not isinstance(coverage, Mapping):
            raise MixedQueryAssetError("The mixed-query asset manifest lacks key_coverage")
        required = {
            "format",
            "scope",
            "eligible_pair_authority_sha256",
            "eligible_pair_count",
            "pathway_count",
            "expected_key_count",
            "observed_key_count",
            "missing_key_count",
            "duplicate_key_count",
            "complete_key_coverage",
            "unscored_key_encoding",
        }
        missing = sorted(required - set(coverage))
        if missing:
            raise MixedQueryAssetError(
                f"The mixed-query key-coverage contract lacks fields: {missing}"
            )
        try:
            pair_count = int(coverage["eligible_pair_count"])
            pathway_count = int(coverage["pathway_count"])
            expected = int(coverage["expected_key_count"])
            observed = int(coverage["observed_key_count"])
            missing_count = int(coverage["missing_key_count"])
            duplicate_count = int(coverage["duplicate_key_count"])
        except (TypeError, ValueError) as exc:
            raise MixedQueryAssetError("The mixed-query key counts are not integers") from exc
        source_hash = str(self.asset_manifest.get("source_exact_association_sha256", ""))
        observed_pairs = self.association_all[["cancer_id", "lncrna_id"]].drop_duplicates()
        if (
            coverage.get("format") != KEY_COVERAGE_FORMAT
            or coverage.get("scope") not in KEY_COVERAGE_SCOPES
            or coverage.get("eligible_pair_authority_sha256") != source_hash
            or coverage.get("complete_key_coverage") is not True
            or coverage.get("unscored_key_encoding")
            != "NULL_WITH_TYPED_NOT_EVALUATED_REASON"
            or pair_count < 1
            or pathway_count < 1
            or (
                expected != pair_count * pathway_count
                if coverage.get("scope")
                == "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_CARTESIAN"
                else expected != observed
            )
            or observed != expected
            or observed != len(self.association_all)
            or missing_count != 0
            or duplicate_count != 0
            or pathway_count != self.membership.pathway_id.nunique()
            or pair_count != len(observed_pairs)
        ):
            raise MixedQueryAssetError(
                "The mixed-query association failed complete eligible-pair x pathway key coverage"
            )

    def _build_pathway_index(self) -> None:
        self.pathway_to_genes = {
            str(pathway_id): frozenset(group.gene_id.astype(str))
            for pathway_id, group in self.membership.groupby("pathway_id", observed=True)
        }
        self.gene_universe = frozenset(self.membership.gene_id.astype(str))
        metadata = self.membership.copy()
        if not self.pathway_metadata.empty:
            if "pathway_id" not in self.pathway_metadata:
                raise MixedQueryAssetError("Pathway metadata lacks pathway_id")
            metadata = self.pathway_metadata.copy()
        keep = [column for column in ("pathway_id", "pathway_name", "pathway_source") if column in metadata]
        metadata = metadata[keep].drop_duplicates("pathway_id")
        self.pathway_metadata_by_id = metadata.set_index("pathway_id").to_dict("index")
        self.cancers = tuple(sorted(self.association.cancer_id.astype(str).unique()))

    def classify(self, members: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {
            "lncRNA": [],
            "protein_coding_gene": [],
            "ambiguous": [],
            "unmapped": [],
            "duplicate": [],
        }
        seen: set[tuple[str, str]] = set()
        for raw_value in members:
            raw = str(raw_value).strip()
            if not raw:
                continue
            key = _normal_identifier(raw)
            explicit_lnc = key.startswith("LNC:")
            candidates = set(self._identifier_index.get(key, set()))
            if explicit_lnc:
                inner = _normal_identifier(key[4:])
                candidates |= self._identifier_index.get(inner, set())
                candidates = {item for item in candidates if item[0] == "lncRNA"}
            if not candidates:
                result["unmapped"].append({"input": raw, "canonical_id": None})
                continue
            canonical_pairs = sorted(candidates)
            if len(canonical_pairs) != 1:
                result["ambiguous"].append(
                    {
                        "input": raw,
                        "candidates": [
                            {"entity_type": entity, "canonical_id": canonical}
                            for entity, canonical in canonical_pairs
                        ],
                    }
                )
                continue
            entity_type, canonical = canonical_pairs[0]
            pair = (entity_type, canonical)
            item = {"input": raw, "canonical_id": canonical}
            if pair in seen:
                item["entity_type"] = entity_type
                result["duplicate"].append(item)
                continue
            seen.add(pair)
            result[entity_type].append(item)
        return result

    def _protein_ora(self, genes: set[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
        in_background = genes & self.gene_universe
        rows: list[dict[str, Any]] = []
        universe_size = len(self.gene_universe)
        query_size = len(in_background)
        for pathway_id, pathway_genes in self.pathway_to_genes.items():
            overlap = sorted(in_background & pathway_genes)
            p_value = (
                hypergeometric_overrepresentation(
                    overlap=len(overlap),
                    pathway_size=len(pathway_genes),
                    query_size=query_size,
                    universe_size=universe_size,
                )
                if query_size
                else np.nan
            )
            rows.append(
                {
                    "pathway_id": pathway_id,
                    "protein_pathway_size": len(pathway_genes),
                    "protein_query_size_in_background": query_size,
                    "protein_overlap_count": len(overlap),
                    "protein_overlap_gene_ids": overlap,
                    "protein_ora_p_value": p_value,
                }
            )
        frame = pd.DataFrame(rows)
        frame["protein_ora_q_value"] = _bh_adjust(frame["protein_ora_p_value"])
        return frame, {
            "mapped_protein_gene_count": len(genes),
            "protein_gene_count_in_background": query_size,
            "protein_gene_background_size": universe_size,
            "protein_gene_ids_outside_background": sorted(genes - self.gene_universe),
        }

    def _lnc_summary(self, lnc_ids: set[str], cancer_id: str | None) -> pd.DataFrame:
        columns = [
            "pathway_id",
            "lncrna_support_count",
            "lncrna_cancer_support_count",
            "lncrna_association_probability_mean",
            "lncrna_association_probability_max",
        ]
        if not lnc_ids:
            return pd.DataFrame(columns=columns)
        frame = self.association.loc[self.association.lncrna_id.isin(lnc_ids)].copy()
        if cancer_id is not None:
            frame = frame.loc[frame.cancer_id.eq(cancer_id)]
        if frame.empty:
            return pd.DataFrame(columns=columns)
        summary = (
            frame.groupby("pathway_id", observed=True)
            .agg(
                lncrna_support_count=("lncrna_id", "nunique"),
                lncrna_cancer_support_count=("cancer_id", "nunique"),
                lncrna_association_probability_mean=(ASSOCIATION_PROBABILITY, "mean"),
                lncrna_association_probability_max=(ASSOCIATION_PROBABILITY, "max"),
            )
            .reset_index()
        )
        return summary

    def _lnc_coverage(self, lnc_ids: set[str], cancer_id: str | None) -> dict[str, Any]:
        frame = self.association_all.loc[self.association_all.lncrna_id.isin(lnc_ids)].copy()
        if cancer_id is not None:
            frame = frame.loc[frame.cancer_id.eq(cancer_id)]
        if frame.empty:
            return {
                "eligible_lncrna_cancer_pair_count": 0,
                "lncrna_expected_key_count": 0,
                "lncrna_evaluated_key_count": 0,
                "lncrna_not_evaluated_key_count": 0,
                "lncrna_not_evaluated_reason_counts": {},
            }
        available = frame.availability.fillna(False).astype(bool)
        reason_counts = (
            frame.loc[~available, "availability_reason"]
            .astype(str)
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
            if (~available).any()
            else {}
        )
        return {
            "eligible_lncrna_cancer_pair_count": int(
                frame[["cancer_id", "lncrna_id"]].drop_duplicates().shape[0]
            ),
            "lncrna_expected_key_count": int(len(frame)),
            "lncrna_evaluated_key_count": int(available.sum()),
            "lncrna_not_evaluated_key_count": int((~available).sum()),
            "lncrna_not_evaluated_reason_counts": reason_counts,
        }

    def query(
        self,
        members: Iterable[str],
        *,
        cancer_id: str | None = None,
        top_k: int = 20,
    ) -> dict[str, Any]:
        raw_members = [str(value).strip() for value in members if str(value).strip()]
        if not raw_members:
            raise MixedQueryInputError("At least one non-empty member is required")
        if len(raw_members) > MAX_MEMBERS:
            raise MixedQueryInputError(f"At most {MAX_MEMBERS} input members are allowed")
        if not 1 <= int(top_k) <= MAX_TOP_K:
            raise MixedQueryInputError(f"top_k must be between 1 and {MAX_TOP_K}")
        cancer = cancer_id.strip().upper() if cancer_id else None
        if cancer is not None and cancer not in self.cancers:
            raise MixedQueryInputError(f"Cancer is absent from the V3.2 association asset: {cancer}")

        classification = self.classify(raw_members)
        lnc_ids = {item["canonical_id"] for item in classification["lncRNA"]}
        gene_ids = {item["canonical_id"] for item in classification["protein_coding_gene"]}
        if not lnc_ids and not gene_ids:
            raise MixedQueryInputError(
                "No uniquely mapped lncRNA or protein-coding gene remained after classification"
            )

        protein, protein_coverage = self._protein_ora(gene_ids)
        lnc = self._lnc_summary(lnc_ids, cancer)
        lnc_coverage = self._lnc_coverage(lnc_ids, cancer)
        result = protein.merge(lnc, on="pathway_id", how="left")
        for column in ("lncrna_support_count", "lncrna_cancer_support_count"):
            result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
        gene_support = result.protein_overlap_count.gt(0)
        lnc_support = result.lncrna_support_count.gt(0)
        gene_component = (1.0 - result.protein_ora_q_value.clip(0.0, 1.0)).where(gene_support)
        lnc_component = result.lncrna_association_probability_mean.where(lnc_support)
        combined = pd.Series(np.nan, index=result.index, dtype=float)
        both = gene_support & lnc_support
        gene_only = gene_support & ~lnc_support
        lnc_only = lnc_support & ~gene_support
        if both.any():
            combined.loc[both] = np.sqrt(
                gene_component.loc[both].astype(float).clip(lower=0.0)
                * lnc_component.loc[both].astype(float).clip(lower=0.0)
            )
        if gene_only.any():
            combined.loc[gene_only] = gene_component.loc[gene_only].astype(float)
        if lnc_only.any():
            combined.loc[lnc_only] = lnc_component.loc[lnc_only].astype(float)
        result["protein_ora_evidence_component"] = gene_component
        result["v32_lncrna_evidence_component"] = lnc_component
        result["combined_evidence_score"] = combined
        result["evidence_channels"] = np.select(
            [both, gene_support, lnc_support],
            ["protein_ora+v32_lncrna_model", "protein_ora", "v32_lncrna_model"],
            default="none",
        )
        result = result.loc[gene_support | lnc_support].copy()
        if result.empty:
            ranked = result
        else:
            ranked = result.sort_values(
                [
                    "combined_evidence_score",
                    "protein_ora_p_value",
                    "lncrna_association_probability_mean",
                    "pathway_id",
                ],
                ascending=[False, True, False, True],
                na_position="last",
                kind="mergesort",
            ).head(int(top_k))
        records: list[dict[str, Any]] = []
        for row in ranked.to_dict("records"):
            metadata = self.pathway_metadata_by_id.get(str(row["pathway_id"]), {})
            row["pathway_name"] = metadata.get("pathway_name")
            row["pathway_source"] = metadata.get("pathway_source") or str(row["pathway_id"]).split(":", 1)[0]
            records.append({key: _json_value(value) for key, value in row.items()})

        normalized_request = {
            "members": sorted(_normal_identifier(value) for value in raw_members),
            "cancer_id": cancer,
            "top_k": int(top_k),
            "release_registry_sha256": self.registry.registry_sha256,
        }
        query_sha256 = hashlib.sha256(
            json.dumps(normalized_request, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        exact_module = self.registry.manifest["modules"]["exact_pathway"]
        return {
            "analysis_version": self.registry.analysis_version,
            "release_id": self.registry.release_id,
            "query_sha256": query_sha256,
            "query_scope": cancer or "PAN_CANCER_MEAN",
            "classification": classification,
            "coverage": {
                **protein_coverage,
                "mapped_lncrna_count": len(lnc_ids),
                **lnc_coverage,
                "ambiguous_input_count": len(classification["ambiguous"]),
                "unmapped_input_count": len(classification["unmapped"]),
                "duplicate_input_count": len(classification["duplicate"]),
            },
            "results": records,
            "semantics": {
                "target_level": "exact_pathway_only",
                "protein_gene_channel": (
                    "One-sided hypergeometric over-representation analysis with BH correction; "
                    "the background is the registered union of protein-coding genes in exact pathways."
                ),
                "lncrna_channel": (
                    "Mean available association_membership_probability from the newly trained V3.2 "
                    "exact-pathway model; pan-cancer means exclude typed NOT_EVALUATED rows and report "
                    "their denominator in coverage. A scored row is not itself thresholded biological support."
                ),
                "combined_evidence_score": (
                    "Balanced geometric mean of (1 - protein ORA q-value) and the V3.2 lncRNA mean "
                    "association probability when both channels support a pathway; otherwise the available "
                    "channel. It is a descriptive ranking score, not a probability or joint p-value."
                ),
                "native_kegg_lncrna_annotation": False,
                "lncrna_is_not_treated_as_classical_pathway_member": True,
                "pathway_family_score_broadcast": False,
                "historical_prediction_reuse": False,
                "complete_key_coverage_required": True,
                "unscored_key_encoding": "NULL_WITH_TYPED_NOT_EVALUATED_REASON",
                "lncrna_support_count_semantics": "number_of_scored_lncrnas_not_thresholded_support",
            },
            "provenance": {
                "exact_pathway_training_run_id": exact_module["training_run_id"],
                "release_registry_sha256": self.registry.registry_sha256,
                "mounted_artifact_sha256": dict(self.registry.artifact_hashes),
                "environment": self.registry.manifest["environment"],
                "production_deployed": self.registry.manifest["production_deployed"],
            },
        }


__all__ = [
    "ASSOCIATION_PROBABILITY",
    "MAX_MEMBERS",
    "MAX_TOP_K",
    "MixedExactPathwayQuery",
    "MixedQueryAssetError",
    "MixedQueryError",
    "MixedQueryInputError",
    "hypergeometric_overrepresentation",
]
