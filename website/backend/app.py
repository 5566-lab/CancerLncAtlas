"""CancerLncAtlas production web application.

The site and the v2.6 model API share one FastAPI process so the browser never
needs cross-origin access. Heavy model assets remain lazy-loaded and the full
graph is never loaded online.
"""

from __future__ import annotations

import os
import re
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import pyarrow.dataset as ds
from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:  # Works both as a package import and as a directly launched app module.
    from .v31_exact_store import (
        ExactPathwayReleaseStore,
        is_finalized_exact_pathway_release,
    )
except ImportError:  # pragma: no cover - direct uvicorn file launch
    from v31_exact_store import (
        ExactPathwayReleaseStore,
        is_finalized_exact_pathway_release,
    )


MODEL_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = MODEL_ROOT / "website" / "frontend"
DATA_ROOT = Path(
    os.getenv("CANCERLNCATLAS_DATA_ROOT", "./data/CancerLncAtlas")
)
SC_UMAP_MANIFEST = Path(
    os.getenv(
        "CANCERLNCATLAS_SC_UMAP_MANIFEST",
        str(DATA_ROOT / "results/model/cancerlncatlas_web_derived/umap_asset_manifest.parquet"),
    )
)

CANCER_SEARCH_ALIASES = {
    "ACC": "肾上腺皮质癌", "BLCA": "膀胱尿路上皮癌 膀胱癌",
    "BRCA": "乳腺浸润癌 乳腺癌", "CESC": "宫颈鳞癌 宫颈腺癌 宫颈癌",
    "CHOL": "胆管癌", "COAD": "结肠腺癌 结肠癌",
    "DLBC": "弥漫大B细胞淋巴瘤 淋巴瘤", "ESCA": "食管癌",
    "GBM": "胶质母细胞瘤 脑胶质瘤", "HNSC": "头颈鳞癌",
    "KICH": "肾嫌色细胞癌 肾癌", "KIRC": "肾透明细胞癌 肾癌",
    "KIRP": "肾乳头状细胞癌 肾癌", "LAML": "急性髓系白血病 白血病",
    "LGG": "低级别胶质瘤 脑胶质瘤", "LIHC": "肝细胞癌 肝癌",
    "LUAD": "肺腺癌 肺癌", "LUSC": "肺鳞癌 肺癌", "MESO": "间皮瘤",
    "OV": "卵巢浆液性囊腺癌 卵巢癌", "PAAD": "胰腺腺癌 胰腺癌",
    "PCPG": "嗜铬细胞瘤 副神经节瘤", "PRAD": "前列腺腺癌 前列腺癌",
    "READ": "直肠腺癌 直肠癌", "SARC": "肉瘤",
    "SKCM": "皮肤黑色素瘤 黑色素瘤", "STAD": "胃腺癌 胃癌",
    "TGCT": "睾丸生殖细胞肿瘤 睾丸癌", "THCA": "甲状腺癌",
    "THYM": "胸腺瘤", "UCEC": "子宫内膜癌", "UCS": "子宫癌肉瘤",
    "UVM": "葡萄膜黑色素瘤",
}


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _discover_v29_release() -> Path:
    configured = os.getenv("CANCERLNCATLAS_RELEASE_ROOT")
    if configured:
        return Path(configured)
    release_base = DATA_ROOT / "results/model/cc_hhgt_v2_9_state_graph/releases"
    candidates = sorted(
        [path / "final_release" for path in release_base.glob("*") if (path / "final_release/SUCCESS.json").exists()],
        reverse=True,
    )
    if candidates:
        return candidates[0]
    # Development fallback before immutable release finalization.
    return MODEL_ROOT / "results/model/cc_hhgt_v2_9_state_graph"



def _discover_v30_release() -> Path | None:
    configured = os.getenv("CANCERLNCATLAS_CLINICAL_RELEASE_ROOT")
    if configured:
        candidate = Path(configured)
        return candidate if (candidate / "SUCCESS.json").exists() else None
    release_base = DATA_ROOT / "results/model/cc_hhgt_v3_0_clinical/releases"
    candidates = sorted(
        [path / "final_release" for path in release_base.glob("*") if (path / "final_release/SUCCESS.json").exists()],
        reverse=True,
    )
    return candidates[0] if candidates else None


def _discover_v31_release() -> Path | None:
    configured = os.getenv("CANCERLNCATLAS_V31_RELEASE_ROOT")
    if configured:
        candidate = Path(configured)
        return candidate if is_finalized_exact_pathway_release(candidate) else None
    release_base = DATA_ROOT / "results/model/cc_hhgt_v3_1_exact_pathway/releases"
    candidates = sorted(
        [
            path / "final_release"
            for path in release_base.glob("*")
            if is_finalized_exact_pathway_release(path / "final_release")
        ],
        reverse=True,
    )
    return candidates[0] if candidates else None

FORMAL_RELEASE = _discover_v29_release()
RELEASE = FORMAL_RELEASE
WEB_TABLES = FORMAL_RELEASE / "web_tables"
CLINICAL_RELEASE = _discover_v30_release()
CLINICAL_WEB_TABLES = CLINICAL_RELEASE / "web_tables" if CLINICAL_RELEASE else None
V31_RELEASE = _discover_v31_release()
if _environment_flag("CANCERLNCATLAS_REQUIRE_V31_EXACT_PATHWAY", default=True) and V31_RELEASE is None:
    raise RuntimeError(
        "Production website requires an audited V3.1 exact-pathway release; "
        "refusing silent fallback to the historical V2.9 reference-only presentation. "
        "Only an explicitly non-production historical development process may set "
        "CANCERLNCATLAS_REQUIRE_V31_EXACT_PATHWAY=0."
    )
os.environ.setdefault("CC_HHGT_QUERY_RELEASE_ROOT", str(FORMAL_RELEASE))

from cc_hhgt_v26.api import app, engine  # noqa: E402


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


class SiteStore:
    def __init__(self) -> None:
        self.query = engine()
        self.final = self.query.final
        self.exact = ExactPathwayReleaseStore(V31_RELEASE) if V31_RELEASE else None
        self.cancers = self.exact.cancers if self.exact else sorted(self.final["cancer_id"].unique())
        self.family_member = self.query.member
        self.pathway_gene = self.query.pathway_gene
        self.lnc_lookup = pd.read_parquet(
            MODEL_ROOT / "query_assets" / "lncrna_identifier_index.parquet"
        )
        self.gene_lookup = pd.read_parquet(
            MODEL_ROOT / "query_assets" / "gene_identifier_index.parquet"
        )
        self.lnc_symbol = self.lnc_lookup.set_index("lncrna_id")[
            "gene_symbol"
        ].to_dict()
        self._web_cache: dict[str, pd.DataFrame] = {}
        self._search_rows = self._build_search_rows()
        self._single_cell_dataset: ds.Dataset | None = None

    def _build_search_rows(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        observed_lnc = (
            set(self.lnc_lookup.lncrna_id.astype(str))
            if self.exact
            else set(self.final["lncrna_id"])
        )
        for row in self.lnc_lookup.itertuples(index=False):
            if row.lncrna_id in observed_lnc:
                rows.append(
                    {
                        "type": "lncRNA",
                        "id": str(row.lncrna_id),
                        "label": str(row.gene_symbol or row.lncrna_id),
                        "subtitle": str(
                            getattr(row, "gene_name", "")
                            or getattr(row, "gene_type", "lncRNA")
                        ),
                        "href": f"/lncrna?id={row.lncrna_id}",
                    }
                )
        for row in self.gene_lookup.head(40000).itertuples(index=False):
            if row.entity_type == "protein_coding_gene":
                rows.append(
                    {
                        "type": "Gene",
                        "id": str(row.gene_id),
                        "label": str(row.gene_symbol or row.gene_id),
                        "subtitle": str(row.gene_type),
                        "href": f"/smart/custom?members={row.gene_symbol}",
                    }
                )
        if self.exact:
            rows.extend(self.exact.search_rows())
        else:
            family_sizes = (
                self.family_member.groupby("pathway_family_id", observed=True)
                .size()
                .to_dict()
            )
            for family_id, size in family_sizes.items():
                rows.append(
                    {
                        "type": "Pathway family",
                        "id": str(family_id),
                        "label": str(family_id),
                        "subtitle": f"{size} exact pathways",
                        "href": f"/gene-set?family={family_id}",
                    }
                )
        cancer_table = self.cancer_overview()
        for row in cancer_table.itertuples(index=False):
            rows.append(
                    {
                        "type": "Cancer",
                        "id": str(row.cancer_id),
                        "label": str(row.english_name),
                        "search_aliases": CANCER_SEARCH_ALIASES.get(
                            str(row.cancer_id), ""
                        ),
                    "subtitle": (
                        f"{row.cancer_id} · "
                            f"{'reference only' if row.reference_only else ('V3.1 exact-pathway scored' if self.exact else 'V2.9 scored')}"
                    ),
                    "href": f"/cancer?id={row.cancer_id}",
                }
            )
        return rows

    def cancer_overview(self) -> pd.DataFrame:
        legacy = self.web_table("web_cancer_overview")
        return self.exact.cancer_overview(legacy) if self.exact else legacy

    def web_table(self, name: str) -> pd.DataFrame:
        if name not in self._web_cache:
            path = WEB_TABLES / f"{name}.parquet"
            if not path.exists():
                raise HTTPException(
                    status_code=503,
                    detail=f"Materialized website table unavailable: {name}",
                )
            self._web_cache[name] = pd.read_parquet(path)
        return self._web_cache[name]

    def cancer_component(
        self, name: str, cancer_id: str
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load an optional cancer-page component from the formal release."""

        table_name = name.removesuffix(".parquet")
        source_path = WEB_TABLES / f"{table_name}.parquet"
        if not source_path.is_file():
            return pd.DataFrame(), {
                "status": "source_unavailable",
                "table": table_name,
                "source": "formal_release",
                "rows": 0,
                "reason": "formal_table_missing",
            }
        try:
            table = self.web_table(table_name)
        except Exception as exc:
            if isinstance(exc, HTTPException) and exc.status_code != 503:
                raise
            return pd.DataFrame(), {
                "status": "source_unavailable",
                "table": table_name,
                "source": "formal_release",
                "source_path": str(source_path),
                "rows": 0,
                "reason": "table_read_failed",
            }
        if "cancer_id" not in table.columns:
            return pd.DataFrame(), {
                "status": "source_unavailable",
                "table": table_name,
                "source": "formal_release",
                "source_path": str(source_path),
                "rows": 0,
                "reason": "required_column_missing:cancer_id",
            }
        selected = table.loc[
            table["cancer_id"]
            .astype(str)
            .str.strip()
            .str.upper()
            .eq(cancer_id.strip().upper())
        ].copy()
        return selected, {
            "status": "available" if not selected.empty else "no_evidence",
            "table": table_name,
            "source": "formal_release",
            "source_path": str(source_path),
            "rows": int(len(selected)),
            "reason": None,
        }


    def clinical_web_table(self, name: str) -> pd.DataFrame:
        cache_key = f"clinical::{name}"
        if cache_key not in self._web_cache:
            if CLINICAL_WEB_TABLES is None:
                raise HTTPException(status_code=503, detail="V3.0 clinical release is not available")
            path = CLINICAL_WEB_TABLES / f"{name}.parquet"
            if not path.exists():
                raise HTTPException(status_code=503, detail=f"Clinical web table unavailable: {name}")
            self._web_cache[cache_key] = pd.read_parquet(path)
        return self._web_cache[cache_key]

    def clinical_endpoint_counts(
        self, cancer: str | None = None
    ) -> pd.DataFrame:
        """Count events once per patient, not once per repeated OOF row."""

        path = (
            CLINICAL_RELEASE / "clinical/patient_survival_oof_prediction.parquet"
            if CLINICAL_RELEASE
            else Path("__missing__")
        )
        if not path.is_file():
            try:
                summary = self.clinical_web_table(
                    "web_patient_risk_oof_summary"
                ).copy()
            except HTTPException as exc:
                if exc.status_code != 503:
                    raise
                return pd.DataFrame(
                    columns=["cancer_id", "endpoint", "n_patients", "n_events"]
                )
            required_summary = {
                "cancer_id", "endpoint", "n_patients", "n_events"
            }
            if not required_summary.issubset(summary.columns):
                return pd.DataFrame(columns=sorted(required_summary))
            if cancer:
                summary = summary.loc[
                    summary.cancer_id.astype(str).str.upper().eq(
                        cancer.strip().upper()
                    )
                ]
            for column in ("n_patients", "n_events"):
                summary[column] = pd.to_numeric(
                    summary[column], errors="coerce"
                )
            if (summary["n_events"] > summary["n_patients"]).any():
                return pd.DataFrame(columns=sorted(required_summary))
            return summary[list(sorted(required_summary))]
        dataset = ds.dataset(path, format="parquet")
        required = {
            "cancer_id", "endpoint", "patient_id", "event", "endpoint_available"
        }
        if not required.issubset(dataset.schema.names):
            return pd.DataFrame(
                columns=["cancer_id", "endpoint", "n_patients", "n_events"]
            )
        expression = None
        if cancer:
            expression = ds.field("cancer_id") == cancer.strip().upper()
        frame = dataset.to_table(
            filter=expression, columns=sorted(required)
        ).to_pandas()
        frame = frame.loc[pd.to_numeric(frame.endpoint_available, errors="coerce").eq(1)]
        frame["event"] = (
            pd.to_numeric(frame.event, errors="coerce")
            .fillna(0)
            .gt(0)
            .astype("int8")
        )
        patient = (
            frame.groupby(
                ["cancer_id", "endpoint", "patient_id"], observed=True
            )["event"]
            .max()
            .reset_index()
        )
        summary = (
            patient.groupby(["cancer_id", "endpoint"], observed=True)
            .agg(n_patients=("patient_id", "nunique"), n_events=("event", "sum"))
            .reset_index()
        )
        if (summary["n_events"] > summary["n_patients"]).any():
            return pd.DataFrame(
                columns=["cancer_id", "endpoint", "n_patients", "n_events"]
            )
        return summary

    def clinical_visuals(
        self, cancer: str, endpoint: str | None = None
    ) -> dict[str, Any]:
        cancer = cancer.strip().upper()
        endpoints = self.clinical_web_table("web_clinical_endpoint_summary")
        endpoints = endpoints.loc[endpoints.cancer_id.astype(str).eq(cancer)].copy()
        if endpoints.empty:
            return {
                "status": "no_evidence",
                "reason_code": "NO_ELIGIBLE_CLINICAL_ENDPOINT",
                "reason": (
                    "No endpoint met the clinical release eligibility requirements "
                    "for this cancer"
                ),
                "cancer_id": cancer,
                "selected_endpoint": None,
                "endpoints": [],
                "event_count_semantics": "unique patients with an event",
                "patient_level_deduplication": True,
                "event_count_status": "no_evidence",
            }
        counts = self.clinical_endpoint_counts(cancer)
        endpoints = endpoints.drop(
            columns=["n_patients", "n_events"], errors="ignore"
        )
        if not counts.empty:
            endpoints = endpoints.merge(
                counts,
                on=["cancer_id", "endpoint"],
                how="left",
                validate="one_to_one",
            )
        else:
            endpoints["n_patients"] = None
            endpoints["n_events"] = None
        choices = endpoints.endpoint.astype(str).tolist()
        selected = endpoint.strip().upper() if endpoint else None
        if selected not in choices:
            selected = "OS" if "OS" in choices else choices[0]
        return {
            "status": "available",
            "cancer_id": cancer,
            "selected_endpoint": selected,
            "endpoints": records(endpoints),
            "event_count_semantics": "unique patients with an event",
            "patient_level_deduplication": True,
            "event_count_status": (
                "available" if not counts.empty else "source_unavailable"
            ),
        }

    def clinical_profile(self, lnc: str, cancer: str | None = None) -> dict[str, Any]:
        lnc_id = self.resolve_lnc(lnc)
        lnc_table = self.clinical_web_table("web_lncRNA_survival_association")
        lnc_table = lnc_table[lnc_table.subject_id.astype(str).eq(lnc_id)]
        pair = self.clinical_web_table("web_lncRNA_pathway_clinical_association")
        pair = pair[pair.subject_id.astype(str).eq(lnc_id)]
        state = self.clinical_web_table("web_lncRNA_state_clinical_association")
        state = state[state.subject_id.astype(str).eq(lnc_id)]
        priority = self.clinical_web_table("web_translational_priority")
        priority = priority[priority.lncrna_id.astype(str).eq(lnc_id)]
        if cancer:
            cancer = cancer.upper()
            lnc_table = lnc_table[lnc_table.cancer_id.eq(cancer)]
            pair = pair[pair.cancer_id.eq(cancer)]
            state = state[state.cancer_id.eq(cancer)]
            priority = priority[priority.cancer_id.eq(cancer)]
        return {
            "lncrna_id": lnc_id,
            "survival_associations": records(lnc_table),
            "pathway_clinical_associations": records(pair),
            "state_clinical_associations": records(state),
            "translational_priority": records(priority.sort_values("translational_priority_score", ascending=False).head(200)),
            "semantics": {
                "clinical_relevance_score": "Held-out clinical association replication score",
                "translational_priority_score": "Functional-confidence and clinical-relevance prioritization; not a survival probability",
                "discovery_ranking_probability": "Frozen V2.9 discovery score; survival is excluded",
            },
        }

    def resolve_lnc(self, value: str) -> str:
        normalized = value.strip().upper()
        if normalized.startswith("LNC:"):
            return normalized
        matches = self.lnc_lookup[
            self.lnc_lookup["gene_symbol"].str.upper().eq(normalized)
            | self.lnc_lookup["ensembl_gene_id"].str.upper().eq(normalized)
        ]
        if matches.empty:
            raise HTTPException(status_code=404, detail=f"lncRNA not found: {value}")
        canonical_ids = sorted(set(matches["lncrna_id"].astype(str)))
        if len(canonical_ids) != 1:
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "AMBIGUOUS_LNCRNA_IDENTIFIER",
                    "input": value,
                    "candidate_lncrna_ids": canonical_ids,
                },
            )
        return canonical_ids[0]

    def search(self, query: str, limit: int) -> list[dict[str, str]]:
        needle = query.strip().upper()
        if len(needle) < 2:
            return []
        prefix = []
        contains = []
        for row in self._search_rows:
            haystack = (
                f"{row['label']} {row['id']} {row['subtitle']} "
                f"{row.get('search_aliases', '')}"
            ).upper()
            if row["label"].upper().startswith(needle) or row["id"].upper().startswith(
                needle
            ):
                prefix.append(row)
            elif needle in haystack:
                contains.append(row)
            if len(prefix) >= limit:
                break
        return (prefix + contains)[:limit]

    def stats(self) -> dict[str, Any]:
        if self.exact:
            exact = self.exact.stats()
            exact.update(
                {
                    "physical_interactions": int(len(self.query.interaction)),
                    "clinical_release_available": CLINICAL_RELEASE is not None,
                    "state_graph_release_available": True,
                }
            )
            return exact
        return {
            "cancers": 33,
            "scored_cancers": int(self.final["cancer_id"].nunique()),
            "reference_only_cancers": 0,
            "lncrnas": int(self.final["lncrna_id"].nunique()),
            "pathway_families": int(self.final["pathway_family_id"].nunique()),
            "predicted_relations": int(len(self.final)),
            "physical_interactions": int(len(self.query.interaction)),
            "patient_oof_predictions": 659025,
            "strict_tasks": 297,
            "adapter_tasks": 495,
            "version": "3.0-clinical" if CLINICAL_RELEASE else "2.9-state-graph",
            "clinical_release_available": CLINICAL_RELEASE is not None,
        }

    def lnc_profile(self, value: str, cancer: str | None) -> dict[str, Any]:
        lnc_id = self.resolve_lnc(value)
        cancer = cancer.strip().upper() if cancer else None
        if self.exact:
            try:
                profile = self.exact.lnc_profile(lnc_id, cancer)
            except KeyError:
                raise HTTPException(status_code=404, detail="No V3.1 exact-pathway relationship rows")
            interaction = self.query.interaction[
                self.query.interaction["lncrna_id"].eq(lnc_id)
            ].sort_values("n_evidence_events", ascending=False)
            return {
                "lncrna_id": lnc_id,
                "symbol": self.lnc_symbol.get(lnc_id, lnc_id),
                "cancer_id": cancer,
                **profile,
                "physical_interactions": records(interaction.head(20)),
                "model_version": "V3.1-exact-pathway",
                "pathway_target_level": "exact_pathway",
            }
        table = self.final[self.final["lncrna_id"].eq(lnc_id)].copy()
        if cancer:
            table = table[table["cancer_id"].eq(cancer)]
        if table.empty:
            raise HTTPException(status_code=404, detail="No v2.6 relationship rows")
        if "fused_confidence_probability" in table:
            table["combined_probability"] = table["fused_confidence_probability"]
        elif "discovery_ranking_probability" in table:
            table["combined_probability"] = table["discovery_ranking_probability"]
        else:
            fallback = [
                column
                for column in [
                    "cross_cancer_probability",
                    "cancer_native_probability",
                    "evidence_integrated_probability",
                ]
                if column in table
            ]
            table["combined_probability"] = table[fallback].mean(axis=1)
        top = table.sort_values("combined_probability", ascending=False).head(30)
        across = (
            table.groupby("cancer_id", observed=True)
            .agg(
                relationship_count=("pathway_family_id", "size"),
                mean_cross_cancer=("cross_cancer_probability", "mean"),
                mean_cancer_specific=("cancer_specific_probability", "mean"),
                mean_evidence=("evidence_integrated_probability", "mean"),
                max_combined=("combined_probability", "max"),
            )
            .reset_index()
            .sort_values("max_combined", ascending=False)
        )
        interaction = self.query.interaction[
            self.query.interaction["lncrna_id"].eq(lnc_id)
        ].sort_values("n_evidence_events", ascending=False)
        return {
            "lncrna_id": lnc_id,
            "symbol": self.lnc_symbol.get(lnc_id, lnc_id),
            "cancer_id": cancer,
            "cold_start_status": str(top.iloc[0]["cold_start_status"]),
            "top_relationships": records(
                top[
                    [
                        "cancer_id",
                        "pathway_family_id",
                        "cross_cancer_probability",
                        "cancer_native_probability",
                        "cancer_specific_probability",
                        "evidence_integrated_probability",
                        "discovery_ranking_probability",
                        "fused_confidence_probability",
                        "combined_probability",
                        "specificity_score",
                        "direction",
                        "cold_start_status",
                    ]
                ]
            ),
            "cancer_landscape": records(across),
            "physical_interactions": records(interaction.head(20)),
        }

    def lnc_visuals(self, value: str, cancer: str | None) -> dict[str, Any]:
        """Compose visuals without failing the whole page on optional sources."""

        profile = self.lnc_profile(value, cancer)
        relationships = list(profile.get("top_relationships", []))
        selected_cancer = cancer.strip().upper() if cancer else None
        if selected_cancer is None and relationships:
            selected_cancer = str(relationships[0].get("cancer_id") or "") or None
        single_cell: dict[str, Any]
        try:
            single_cell = self.single_cell(
                str(profile["lncrna_id"]), selected_cancer, 200
            )
            single_cell["status"] = (
                "available" if single_cell.get("returned") else "no_evidence"
            )
        except HTTPException as exc:
            if exc.status_code != 503:
                raise
            single_cell = {
                "status": "source_unavailable",
                "reason": "single_cell_association_table_unavailable",
                "rows": [],
            }
        return {
            "status": "available",
            "lncrna_id": profile["lncrna_id"],
            "symbol": profile.get("symbol"),
            "selected_cancer": selected_cancer,
            "pathway_heatmap": {
                "status": "available" if relationships else "no_evidence",
                "cells": relationships,
                "value_semantics": "exact-pathway model relationship score",
            },
            "cancer_landscape": profile.get("cancer_landscape", []),
            "single_cell": single_cell,
            "physical_interactions": {
                "status": "available"
                if profile.get("physical_interactions")
                else "no_evidence",
                "rows": profile.get("physical_interactions", []),
            },
            "survival_curve": {
                "status": "source_unavailable",
                "reason": "patient_expression_stratified_curve_not_in_current_release",
                "groups": [],
            },
            "drug_evidence": {
                "status": "source_unavailable",
                "reason": "drug_head_is_exposed_through_the_v3_2_staging_contract",
                "rows": [],
            },
            "partial_components_do_not_fail_request": True,
        }

    def genesets(
        self,
        search: str | None,
        family: str | None,
        pathway: str | None,
        cancer: str | None,
        limit: int,
    ) -> dict[str, Any]:
        if self.exact:
            # ``family`` remains a compatibility alias for old bookmarked URLs;
            # it is interpreted as a free-text exact-pathway query, never as
            # the V3.1 target.
            return self.exact.genesets(
                search=search or family,
                pathway=pathway,
                cancer=cancer,
                limit=limit,
            )
        member = self.family_member
        if family:
            member = member[member["pathway_family_id"].eq(family)]
        if search:
            member = member[
                member["pathway_id"].str.contains(
                    search, case=False, regex=False, na=False
                )
                | member["pathway_family_id"].str.contains(
                    search, case=False, regex=False, na=False
                )
            ]
        summary = (
            member.groupby("pathway_family_id", observed=True)
            .agg(
                n_pathways=("pathway_id", "nunique"),
                representative_pathway=(
                    "pathway_id",
                    lambda values: sorted(values)[0],
                ),
            )
            .reset_index()
            .head(limit)
        )
        return {
            "families": records(summary),
            "members": records(member.head(500)),
            "total_families": int(member["pathway_family_id"].nunique()),
        }

    def _get_sc_dataset(self) -> ds.Dataset:
        if self._single_cell_dataset is None:
            path = DATA_ROOT / "parquet" / "sc_lncRNA_pathway_association"
            if not path.exists():
                raise HTTPException(
                    status_code=503, detail="Single-cell dataset is unavailable"
                )
            self._single_cell_dataset = ds.dataset(path, format="parquet")
        return self._single_cell_dataset

    def single_cell(
        self,
        lnc: str | None,
        cancer: str | None,
        limit: int,
        pathway_id: str | None = None,
    ) -> dict[str, Any]:
        dataset = self._get_sc_dataset()
        names = set(dataset.schema.names)
        lnc_column = next(
            (name for name in ["lncrna_id", "lncRNA_id"] if name in names), None
        )
        cancer_column = next(
            (name for name in ["cancer_id", "cancer_type"] if name in names), None
        )
        cell_column = next(
            (
                name
                for name in [
                    "cell_type",
                    "cell_type_standard",
                    "celltype",
                    "cell_state",
                ]
                if name in names
            ),
            None,
        )
        pathway_column = next(
            (name for name in ["pathway_id", "pathway_family_id"] if name in names),
            None,
        )
        preferred = [
            lnc_column,
            cancer_column,
            cell_column,
            pathway_column,
            "effect",
            "rho",
            "correlation",
            "fdr",
            "q_value",
            "direction",
            "n_cells",
            "dataset_id",
        ]
        columns = [name for name in preferred if name and name in names]
        expression = None
        resolved_lnc = None
        if lnc and lnc_column:
            resolved_lnc = self.resolve_lnc(lnc)
            expression = ds.field(lnc_column) == resolved_lnc
        if cancer and cancer_column:
            cancer = cancer.strip().upper()
            cancer_expression = ds.field(cancer_column) == cancer
            expression = (
                cancer_expression
                if expression is None
                else expression & cancer_expression
            )
        if pathway_id and pathway_column:
            pathway_expression = ds.field(pathway_column) == pathway_id
            expression = (
                pathway_expression
                if expression is None
                else expression & pathway_expression
            )
        # Apply every selector before the row cap and stop scanning as soon as
        # ``limit`` matching rows are found.  Materializing the whole filtered
        # dataset before ``slice`` made sparse evidence look absent and could
        # exhaust web-worker memory on unfiltered requests.
        table = dataset.head(limit, filter=expression, columns=columns)
        frame = table.to_pandas()
        return {
            "lncrna_id": resolved_lnc,
            "cancer_id": cancer,
            "pathway_id": pathway_id,
            "rows": records(frame),
            "columns": columns,
            "returned": len(frame),
            "data_scope": "single_cell_association_evidence",
        }

    def sc_summary(
        self,
        cancer: str,
        lnc: str | None = None,
        pathway_id: str | None = None,
    ) -> dict[str, Any]:
        cancer = cancer.strip().upper()
        if cancer not in self.cancers:
            raise HTTPException(status_code=404, detail="Cancer not found")
        try:
            result = self.single_cell(
                lnc, cancer, 500, pathway_id=pathway_id
            )
        except HTTPException as exc:
            if exc.status_code != 503:
                raise
            return {
                "status": "source_unavailable",
                "cancer_id": cancer,
                "reason": "single_cell_association_release_not_mounted",
                "rows": [],
            }
        rows = result.get("rows", [])
        if pathway_id:
            rows = [
                row
                for row in rows
                if str(row.get("pathway_id") or row.get("pathway_family_id"))
                == pathway_id
            ]
        return {
            "status": "available" if rows else "no_evidence",
            "cancer_id": cancer,
            "lncrna_id": result.get("lncrna_id"),
            "pathway_id": pathway_id,
            "rows": rows,
            "returned": len(rows),
            "count_semantics": "association rows; not unique cells or lncRNAs",
        }

    def sc_umap(self, cancer: str) -> dict[str, Any]:
        cancer = cancer.strip().upper()
        if cancer not in self.cancers:
            raise HTTPException(status_code=404, detail="Cancer not found")
        if not SC_UMAP_MANIFEST.is_file():
            return {
                "status": "source_unavailable",
                "cancer_id": cancer,
                "reason": "umap_asset_manifest_not_mounted",
                "figures": [],
            }
        manifest = pd.read_parquet(SC_UMAP_MANIFEST)
        if "cancer_id" not in manifest:
            return {
                "status": "failed_qc",
                "cancer_id": cancer,
                "reason": "umap_manifest_missing_cancer_id",
                "figures": [],
            }
        selected = manifest.loc[
            manifest.cancer_id.astype(str).str.upper().eq(cancer)
        ]
        return {
            "status": "available" if not selected.empty else "no_evidence",
            "cancer_id": cancer,
            "figures": records(selected),
        }

    def network(self, lnc: str, cancer: str | None) -> dict[str, Any]:
        profile = self.lnc_profile(lnc, cancer)
        lnc_id = profile["lncrna_id"]
        nodes: dict[str, dict[str, Any]] = {
            lnc_id: {
                "id": lnc_id,
                "label": profile["symbol"],
                "type": "lncRNA",
            }
        }
        edges = []
        relationships = profile["top_relationships"][:8]
        if self.exact:
            for relation in relationships:
                pathway = str(relation["pathway_id"])
                nodes[pathway] = {
                    "id": pathway,
                    "label": relation.get("pathway_name") or pathway,
                    "type": "exact_pathway",
                    "score": relation["calibrated_probability"],
                }
                edges.append(
                    {
                        "source": lnc_id,
                        "target": pathway,
                        "type": "V3.1_exact_pathway_prediction",
                        "style": "dashed",
                        "weight": relation["calibrated_probability"],
                    }
                )
                genes = self.pathway_gene[
                    self.pathway_gene["pathway_id"].eq(pathway)
                ].head(3)
                for gene in genes["gene_symbol"]:
                    gene_id = f"GENE:{gene}"
                    nodes[gene_id] = {"id": gene_id, "label": gene, "type": "gene"}
                    edges.append(
                        {
                            "source": gene_id,
                            "target": pathway,
                            "type": "exact_membership",
                            "style": "solid",
                            "weight": 0.7,
                        }
                    )
        else:
            for relation in relationships:
                family_id = relation["pathway_family_id"]
                nodes[family_id] = {
                    "id": family_id,
                    "label": family_id.replace("SPF:", "Family "),
                    "type": "pathway_family",
                    "score": relation["combined_probability"],
                }
                edges.append(
                    {
                        "source": lnc_id,
                        "target": family_id,
                        "type": "prediction",
                        "style": "dashed",
                        "weight": relation["combined_probability"],
                    }
                )
                family_paths = self.family_member[
                    self.family_member["pathway_family_id"].eq(family_id)
                ].head(1)
                for pathway in family_paths["pathway_id"]:
                    nodes[pathway] = {
                        "id": pathway,
                        "label": pathway.split(":")[-1][:34],
                        "type": "exact_pathway",
                    }
                    edges.append(
                        {
                            "source": pathway,
                            "target": family_id,
                            "type": "family_membership",
                            "style": "solid",
                            "weight": 0.5,
                        }
                    )
                    genes = self.pathway_gene[
                        self.pathway_gene["pathway_id"].eq(pathway)
                    ].head(3)
                    for gene in genes["gene_symbol"]:
                        gene_id = f"GENE:{gene}"
                        nodes[gene_id] = {
                            "id": gene_id,
                            "label": gene,
                            "type": "gene",
                        }
                        edges.append(
                            {
                                "source": gene_id,
                                "target": pathway,
                                "type": "exact_membership",
                                "style": "solid",
                                "weight": 0.7,
                            }
                        )
        for interaction in profile["physical_interactions"][:8]:
            partner = interaction["partner_id"]
            nodes[partner] = {
                "id": partner,
                "label": partner.replace("UNIPROT:", ""),
                "type": "protein",
            }
            edges.append(
                {
                    "source": lnc_id,
                    "target": partner,
                    "type": "experimental_physical",
                    "style": "thick_solid",
                    "weight": min(
                        1.0, float(interaction["n_evidence_events"]) / 5
                    ),
                    "pmids": interaction.get("supporting_pmids"),
                }
            )
        return {
            "nodes": list(nodes.values())[:200],
            "edges": edges[:300],
            "legend": {
                "exact_membership": "solid",
                "prediction": "dashed",
                "experimental_physical": "thick_solid",
                "PPI": "thin",
                "drug_target": "directed",
            },
        }

    def datasets(self) -> list[dict[str, Any]]:
        definitions = [
            (
                "TCGA bulk expression",
                "Bulk RNA expression and covariate-adjusted pathway association",
                DATA_ROOT / "parquet" / "bulk_lncRNA_expression",
                "bulk",
            ),
            (
                "Single-cell atlas",
                "Cell-type and malignant-state lncRNA/pathway evidence",
                DATA_ROOT / "parquet" / "sc_lncRNA_pathway_association",
                "single-cell",
            ),
            (
                "Physical interactions",
                "Curated experimental lncRNA–protein evidence events",
                DATA_ROOT / "parquet" / "interaction_relation.parquet",
                "interaction",
            ),
            (
                "CC-HHGT V2.9 state-graph probabilities",
                "Graph ensemble, patient-native, event evidence, state and fused scores",
                FORMAL_RELEASE
                / "predictions"
                / "final_expert_fusion_table.parquet",
                "model",
            ),
        ]
        if self.exact:
            definitions.append(
                (
                    "CC-HHGT V3.1 exact-pathway website release",
                    "Validation-selected three-seed exact-pathway scores and lncRNA gene sets",
                    self.exact.root,
                    "model",
                )
            )
        output = []
        for name, description, path, category in definitions:
            files = [path] if path.is_file() else list(path.rglob("*.parquet"))
            size = sum(file.stat().st_size for file in files if file.exists())
            output.append(
                {
                    "name": name,
                    "description": description,
                    "category": category,
                    "files": len(files),
                    "size_bytes": size,
                    "available": path.exists(),
                }
            )
        return output

    def cancer_profile(self, cancer_id: str, limit: int) -> dict[str, Any]:
        cancer_id = cancer_id.strip().upper()
        overview = self.cancer_overview()
        row = overview[overview["cancer_id"].eq(cancer_id)]
        if row.empty:
            raise HTTPException(status_code=404, detail="Cancer not found")
        exact_profile = self.exact.cancer_profile(cancer_id, limit) if self.exact else None
        if exact_profile:
            ranking = pd.DataFrame(exact_profile["lncrna_ranking"])
            pathway = pd.DataFrame(exact_profile["pathway_ranking"])
        else:
            ranking = self.web_table("web_cancer_lncRNA_ranking")
            ranking = ranking[ranking["cancer_id"].eq(cancer_id)].sort_values(
                "rank_within_cancer"
            )
            pathway = self.web_table("web_cancer_pathway_ranking")
            pathway = pathway[pathway["cancer_id"].eq(cancer_id)].sort_values(
                "rank_within_cancer"
            )
        component_status: dict[str, dict[str, Any]] = {}
        celltype, component_status["celltype"] = self.cancer_component(
            "web_cancer_celltype_summary", cancer_id
        )
        state, component_status["state"] = self.cancer_component(
            "web_cancer_state_summary", cancer_id
        )
        drug, component_status["drug"] = self.cancer_component(
            "web_cancer_drug_summary", cancer_id
        )
        if exact_profile:
            exact_genesets = exact_profile.get("geneset_summary")
            geneset = pd.DataFrame(exact_genesets or [])
            component_status["geneset"] = {
                "status": (
                    "source_unavailable"
                    if exact_genesets is None
                    else "available"
                    if not geneset.empty
                    else "no_evidence"
                ),
                "table": "exact_profile:geneset_summary",
                "source": "exact_pathway_release",
                "rows": int(len(geneset)),
                "reason": (
                    "exact_component_missing" if exact_genesets is None else None
                ),
            }
        else:
            geneset, component_status["geneset"] = self.cancer_component(
                "web_cancer_geneset_summary", cancer_id
            )
        return {
            "overview": records(row)[0],
            "lncrna_ranking": records(ranking.head(limit)),
            "pathway_ranking": records(pathway.head(limit)),
            "celltype_summary": records(celltype),
            "state_summary": records(state),
            "drug_summary": records(drug),
            "geneset_summary": records(geneset),
            "component_status": component_status,
            "probability_semantics": {
                "calibrated_probability": "V3.1 validation-selected deep architecture, three-seed mean calibrated probability for an exact pathway",
                "member_weight": "Deterministic exact-pathway gene-set member ranking weight",
                "graph_ensemble_probability": "OOF stacking of V2.9 R-GCN, HGT and CC-HHGT-Strict",
                "cancer_native_probability": "Patient-cross-fitted cancer-native expert",
                "discovery_ranking_probability": "Direct target evidence masked; used for discovery ranking",
                "fused_confidence_probability": "Graph, patient, indirect and direct evidence confidence fusion",
                "lncrna_state_probability": "Patient and V2.9 state-node graph experts for lncRNA-state association",
            },
        }


@lru_cache(maxsize=1)
def store() -> SiteStore:
    return SiteStore()


@app.get("/api/site/stats")
def site_stats() -> dict[str, Any]:
    return store().stats()


@app.get("/api/site/search")
def site_search(
    q: str = Query(min_length=2, max_length=100),
    limit: int = Query(default=12, ge=1, le=50),
) -> dict[str, Any]:
    return {"query": q, "results": store().search(q, limit)}


@app.get("/api/site/lncrna/{lncrna}")
def site_lncrna(lncrna: str, cancer: str | None = None) -> dict[str, Any]:
    return store().lnc_profile(lncrna, cancer)


@app.get("/api/site/lncrna/{lncrna}/visuals")
def site_lncrna_visuals(
    lncrna: str, cancer: str | None = None
) -> dict[str, Any]:
    return store().lnc_visuals(lncrna, cancer)


@app.get("/api/site/genesets")
def site_genesets(
    search: str | None = None,
    family: str | None = None,
    pathway: str | None = None,
    cancer: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    return store().genesets(search, family, pathway, cancer, limit)


@app.get("/api/site/genesets/{geneset_id}")
def site_geneset_detail(geneset_id: str) -> dict[str, Any]:
    exact = store().exact
    if exact is None:
        raise HTTPException(status_code=503, detail="Exact-pathway release unavailable")
    try:
        return exact.geneset_detail(geneset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Gene set not found") from exc


@app.get("/api/site/single-cell")
def site_single_cell(
    lnc: str | None = None,
    cancer: str | None = None,
    limit: int = Query(default=200, ge=1, le=500),
) -> dict[str, Any]:
    return store().single_cell(lnc, cancer, limit)


@app.get("/api/site/sc-summary/{cancer}")
def site_sc_summary(
    cancer: str,
    lnc: str | None = None,
    pathway_id: str | None = None,
) -> dict[str, Any]:
    return store().sc_summary(cancer, lnc, pathway_id)


@app.get("/api/site/sc-umap/{cancer}")
def site_sc_umap(cancer: str) -> dict[str, Any]:
    return store().sc_umap(cancer)


@app.get("/api/site/network")
def site_network(lnc: str, cancer: str | None = None) -> dict[str, Any]:
    return store().network(lnc, cancer)


@app.get("/api/site/datasets")
def site_datasets() -> dict[str, Any]:
    return {"datasets": store().datasets()}


@app.get("/api/site/cancers")
def site_cancers() -> dict[str, Any]:
    table = store().cancer_overview().sort_values("cancer_id")
    return {"cancers": records(table)}


@app.get("/api/site/cancer/{cancer_id}")
def site_cancer(
    cancer_id: str,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    return store().cancer_profile(cancer_id, limit)



@app.get("/api/site/clinical/endpoints")
def site_clinical_endpoints() -> dict[str, Any]:
    site_store = store()
    endpoints = site_store.clinical_web_table("web_clinical_endpoint_summary").copy()
    counts = site_store.clinical_endpoint_counts()
    endpoints = endpoints.drop(
        columns=["n_patients", "n_events"], errors="ignore"
    )
    if not counts.empty:
        endpoints = endpoints.merge(
            counts,
            on=["cancer_id", "endpoint"],
            how="left",
            validate="one_to_one",
        )
    else:
        endpoints["n_patients"] = None
        endpoints["n_events"] = None
    return {
        "endpoints": records(endpoints),
        "event_count_semantics": "unique patients with an event",
        "event_count_status": (
            "available" if not counts.empty else "source_unavailable"
        ),
    }


@app.get("/api/site/clinical/visuals")
def site_clinical_visuals(
    cancer: str = Query(min_length=2, max_length=12),
    endpoint: str | None = None,
) -> dict[str, Any]:
    return store().clinical_visuals(cancer, endpoint)


@app.get("/api/site/clinical/lncrna/{lncrna}")
def site_clinical_lncrna(lncrna: str, cancer: str | None = None) -> dict[str, Any]:
    return store().clinical_profile(lncrna, cancer)


@app.get("/api/site/clinical/priority")
def site_clinical_priority(
    cancer: str | None = None,
    endpoint: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    table = store().clinical_web_table("web_translational_priority")
    if cancer:
        table = table[table.cancer_id.eq(cancer.upper())]
    if endpoint and "clinical_best_endpoint" in table:
        table = table[table.clinical_best_endpoint.eq(endpoint.upper())]
    table = table.sort_values("translational_priority_score", ascending=False).head(limit)
    return {"results": records(table)}


DOWNLOADS = {
    "model-card": MODEL_ROOT / "V2_6_MODEL_CARD.md",
    "api-doc": MODEL_ROOT / "V2_6_QUERY_API.md",
    "leakage-audit": MODEL_ROOT / "V2_6_LEAKAGE_AUDIT.md",
    "probabilities": FORMAL_RELEASE
    / "predictions/final_three_probability_table.parquet",
    "selected-candidates": FORMAL_RELEASE
    / "predictions/candidate_prediction_selected.parquet",
    "genesets": FORMAL_RELEASE / "genesets/default_genesets.gmt",
    "qa-report": FORMAL_RELEASE / "reports/SERVER_POSTTRAIN_QA.tsv",
    "manifest": MODEL_ROOT / "V2_6_RETURN_CONTENT_MANIFEST.tsv",
    "clinical-model-card": (CLINICAL_RELEASE / "V3_0_MODEL_CARD.md") if CLINICAL_RELEASE else Path("__missing__"),
    "clinical-priority": (CLINICAL_RELEASE / "predictions/v3_0_functional_clinical_table.parquet") if CLINICAL_RELEASE else Path("__missing__"),
    "clinical-associations": (CLINICAL_RELEASE / "clinical/clinical_association_summary.parquet") if CLINICAL_RELEASE else Path("__missing__"),
    "v31-genesets": (V31_RELEASE / "cancer_exact_pathway_lncRNA_symbols.gmt") if V31_RELEASE else Path("__missing__"),
    "v31-genesets-ids": (V31_RELEASE / "cancer_exact_pathway_lncRNA_ids.gmt") if V31_RELEASE else Path("__missing__"),
    "v31-geneset-master": (V31_RELEASE / "geneset_master_exact_pathway.parquet") if V31_RELEASE else Path("__missing__"),
    "v31-geneset-members": (V31_RELEASE / "geneset_member_exact_pathway.parquet") if V31_RELEASE else Path("__missing__"),
}


def available_downloads() -> list[dict[str, Any]]:
    """Build the catalog from current filesystem truth on every request."""

    output = []
    for key, path in sorted(DOWNLOADS.items()):
        if not path.is_file():
            continue
        stat = path.stat()
        output.append(
            {
                "key": key,
                "file_name": path.name,
                "file_size": int(stat.st_size),
                "format": path.suffix.lstrip(".").upper(),
                "download_url": f"/api/site/download/{key}",
            }
        )
    return output


@app.get("/api/site/downloads")
def site_downloads() -> dict[str, Any]:
    files = available_downloads()
    return {"files": files, "only_existing_files": True, "count": len(files)}


@app.get("/api/site/download/{key}")
def site_download(key: str) -> FileResponse:
    path = DOWNLOADS.get(key)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Download not found")
    return FileResponse(path, filename=path.name)


@app.get("/api/search")
def compatibility_search(
    q: str = Query(min_length=2, max_length=100),
    limit: int = Query(default=12, ge=1, le=50),
) -> dict[str, Any]:
    return site_search(q=q, limit=limit)


@app.get("/api/cancers")
def compatibility_cancers() -> dict[str, Any]:
    return site_cancers()


@app.get("/api/cancers/{cancer_id}")
def compatibility_cancer(
    cancer_id: str,
    limit: int = Query(default=50, ge=1, le=200),
) -> dict[str, Any]:
    return site_cancer(cancer_id=cancer_id, limit=limit)


@app.get("/api/downloads")
def compatibility_downloads() -> dict[str, Any]:
    return site_downloads()


@app.get("/api/site/docs/{key}")
def site_docs(key: str) -> dict[str, str]:
    path = DOWNLOADS.get(key)
    if path is None or path.suffix.lower() != ".md" or not path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    return {"key": key, "markdown": path.read_text(encoding="utf-8")}


@app.middleware("http")
async def request_metrics(request: Request, call_next):
    started = time.perf_counter()
    response = await call_next(request)
    if request.url.path.startswith(("/v2.6/", "/api/site/")):
        event = {
            "timestamp": time.time(),
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "rss_mb": round(psutil.Process().memory_info().rss / 1024**2, 2),
        }
        log_path = FORMAL_RELEASE / "reports/query_api_requests.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    return response


app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")


def _attach_v32_staging_application() -> None:
    """Attach the hash-bound V3.2 application only under an explicit flag.

    This is a local/staging integration gate, not a production-deployment
    declaration.  The unified loader independently verifies every configured
    binding and its pinned SHA-256 before any V3.2 route is registered.
    """

    app.state.v32_staging_enabled = False
    if not _environment_flag("CANCERLNCATLAS_ENABLE_V32_STAGING", default=False):
        return

    from cc_hhgt.v32.unified_staging_bindings import (
        create_app_from_unified_bindings,
    )

    unified_path = MODEL_ROOT / "config" / "v32_unified_staging_bindings.json"
    v32_application = create_app_from_unified_bindings(
        unified_path,
        repo_root=MODEL_ROOT,
    )
    # Include the already-prefixed V3.2 routes rather than mounting at '/',
    # which would shadow the historical site catch-all below.
    app.include_router(v32_application.router)
    app.state.v32_staging_enabled = True
    app.state.v32_unified_binding_path = str(unified_path)

    @app.get("/v32-staging.html", include_in_schema=False)
    def v32_staging_frontend() -> FileResponse:
        return FileResponse(FRONTEND / "v32-staging.html")

    @app.get("/v32-capability-catalog.json", include_in_schema=False)
    def v32_capability_catalog() -> FileResponse:
        return FileResponse(
            FRONTEND / "v32-capability-catalog.json",
            media_type="application/json",
        )


_attach_v32_staging_application()


@app.get("/{page_path:path}", include_in_schema=False)
def frontend(page_path: str) -> FileResponse:
    if page_path in {"v32-staging.html", "v32-capability-catalog.json"} or page_path.startswith(
        ("api/", "v2.6/", "v3.2-staging/")
    ):
        raise HTTPException(status_code=404)
    return FileResponse(FRONTEND / "index.html")
