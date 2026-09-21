"""Lazy query layer for the immutable V3.1 exact-pathway website release."""
from __future__ import annotations

import json
import hashlib
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a release file without depending on a repository-only package."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


FORBIDDEN_PUBLIC_LABELS = {
    "label",
    "label_class",
    "proxy_label",
    "association_proxy_label",
    "direction_label",
    "sample_weight",
}
REQUIRED_FORMAL_CANCERS = {"HNSC", "LGG"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def is_finalized_exact_pathway_release(release_root: str | Path) -> bool:
    """Return true only for an atomically finalized, internally linked release."""

    root = Path(release_root).resolve()
    success_path = root / "SUCCESS.json"
    final_path = root / "FINAL_RELEASE.json"
    marker_path = root.parent / "FINALIZATION_SUCCESS.json"
    try:
        success = json.loads(success_path.read_text(encoding="utf-8"))
        final = json.loads(final_path.read_text(encoding="utf-8"))
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    shared = (
        "release_id",
        "run_id",
        "selected_model",
        "selection_split",
        "pathway_target_level",
        "pathway_family_role",
        "cancers",
        "reference_only_cancers",
        "pancancer_eligible_lncRNAs",
        "materialization_success_sha256",
        "website_release_audit_sha256",
        "selection_sha256",
        "site_source_success_sha256",
        "materialized_file_merkle_sha256",
    )
    return (
        success.get("status") == "PASS"
        and final.get("status") == "PASS"
        and marker.get("status") == "PASS"
        and bool(final.get("release_eligible"))
        and bool(marker.get("release_eligible"))
        and final.get("release_id") == root.parent.name
        and all(final.get(key) == marker.get(key) for key in shared)
        and final.get("run_id") == success.get("run_id")
        and final.get("selected_model") == success.get("selected_model")
        and final.get("pathway_target_level") == "exact_pathway"
        and final.get("pathway_family_role") == "auxiliary_hierarchy_only"
        and int(final.get("cancers", -1)) == 33
        and list(final.get("reference_only_cancers", ["missing"])) == []
        and final.get("materialization_success_sha256")
        == file_sha256(success_path)
        and int(marker.get("final_release_files", 0)) > 0
        and bool(
            SHA256_PATTERN.fullmatch(
                str(marker.get("final_release_file_merkle_sha256", ""))
            )
        )
    )


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return frame.astype(object).where(pd.notna(frame), None).to_dict("records")


class ExactPathwayReleaseStore:
    """Read exact-pathway partitions without loading the 33-cancer universe."""

    def __init__(self, release_root: str | Path) -> None:
        self.root = Path(release_root).resolve()
        success_path = self.root / "SUCCESS.json"
        if not success_path.is_file():
            raise RuntimeError(f"V3.1 exact-pathway release lacks SUCCESS.json: {self.root}")
        self.success = json.loads(success_path.read_text(encoding="utf-8"))
        if (
            self.success.get("status") != "PASS"
            or self.success.get("pathway_target_level") != "exact_pathway"
            or self.success.get("pathway_family_role") != "auxiliary_hierarchy_only"
            or int(self.success.get("cancers", -1)) != 33
            or list(self.success.get("reference_only_cancers", ["missing"])) != []
            or bool(self.success.get("heldout_label_columns_in_public_scores"))
        ):
            raise RuntimeError("Invalid V3.1 exact-pathway website release contract")
        self.score_root = self.root / "web_exact_pathway_score"
        self.selected_root = self.root / "web_exact_pathway_selected"
        self.master_path = self.root / "geneset_master_exact_pathway.parquet"
        self.member_path = self.root / "geneset_member_exact_pathway.parquet"
        self.summary_path = self.root / "web_cancer_exact_pathway_summary.parquet"
        required = [
            self.score_root,
            self.selected_root,
            self.master_path,
            self.member_path,
            self.summary_path,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"Incomplete V3.1 exact-pathway website release: {missing}")
        self.fold_audits = pd.DataFrame(self.success.get("fold_audits", []))
        filter_contract = self.success.get("pancancer_lncrna_filter", {})
        if (
            not bool(filter_contract.get("enabled"))
            or not bool(filter_contract.get("evidence_cannot_bypass_filter"))
            or int(filter_contract.get("minimum_detected_cancers", -1)) != 3
            or float(filter_contract.get("within_cancer_min_sample_detection_rate", -1))
            != 0.10
            or int(self.success.get("pancancer_eligible_lncRNAs", 0)) <= 0
        ):
            raise RuntimeError("V3.1 website release lacks the frozen pan-cancer lncRNA filter")
        if (
            len(self.fold_audits) != 33
            or self.fold_audits.cancer_id.astype(str).nunique() != 33
            or not REQUIRED_FORMAL_CANCERS.issubset(
                set(self.fold_audits.cancer_id.astype(str))
            )
        ):
            raise RuntimeError("V3.1 website release lacks 33 unique cancer fold audits")
        self.cancers = sorted(self.fold_audits.cancer_id.astype(str).unique())
        self.master = pd.read_parquet(self.master_path)
        if self.master.empty:
            raise RuntimeError("V3.1 exact-pathway release contains no gene sets")
        if self.master.duplicated("geneset_id").any():
            raise RuntimeError("V3.1 exact-pathway gene-set master is not unique")

    def _partition_path(self, cancer: str, *, selected: bool) -> Path:
        root = self.selected_root if selected else self.score_root
        path = root / f"cancer_id={cancer}" / "part-0.parquet"
        if not path.is_file():
            raise RuntimeError(f"V3.1 exact-pathway partition is missing: {path}")
        return path

    @lru_cache(maxsize=6)
    def cancer_rows(self, cancer: str, selected: bool = True) -> pd.DataFrame:
        cancer = cancer.strip().upper()
        if cancer not in self.cancers:
            raise KeyError(cancer)
        frame = pd.read_parquet(self._partition_path(cancer, selected=selected))
        forbidden = sorted(FORBIDDEN_PUBLIC_LABELS & set(frame.columns))
        if forbidden:
            raise RuntimeError(f"Public V3.1 score partition contains labels: {forbidden}")
        return frame

    def search_rows(self) -> list[dict[str, str]]:
        if self.master.empty:
            return []
        columns = [
            column
            for column in ["pathway_id", "pathway_name", "pathway_source"]
            if column in self.master
        ]
        pathways = self.master[columns].drop_duplicates("pathway_id")
        rows = []
        for row in pathways.itertuples(index=False):
            pathway_id = str(row.pathway_id)
            name = str(getattr(row, "pathway_name", "") or pathway_id)
            source = str(getattr(row, "pathway_source", "") or "exact pathway")
            rows.append(
                {
                    "type": "Exact pathway",
                    "id": pathway_id,
                    "label": name,
                    "subtitle": source,
                    "href": f"/gene-set?pathway={pathway_id}",
                }
            )
        return rows

    def cancer_overview(self, legacy: pd.DataFrame | None = None) -> pd.DataFrame:
        audit = self.fold_audits.rename(
            columns={"rows": "scored_relationship_count", "selected_rows": "significant_lncRNA_pathway_relations"}
        ).copy()
        keep = [
            "cancer_id",
            "scored_relationship_count",
            "significant_lncRNA_pathway_relations",
        ]
        overview = audit[keep].copy()
        if legacy is not None and not legacy.empty:
            descriptive = [
                column
                for column in ["cancer_id", "english_name", "chinese_name"]
                if column in legacy
            ]
            overview = overview.merge(
                legacy[descriptive].drop_duplicates("cancer_id"),
                on="cancer_id",
                how="left",
                validate="one_to_one",
            )
        if "english_name" not in overview:
            overview["english_name"] = overview.cancer_id
        overview["english_name"] = overview.english_name.fillna(overview.cancer_id)
        overview["reference_only"] = False
        overview["detectable_lncRNAs"] = int(
            self.success.get("pancancer_eligible_lncRNAs", 0)
        )
        overview["model_version"] = "V3.1-exact-pathway"
        return overview.sort_values("cancer_id").reset_index(drop=True)

    def cancer_profile(self, cancer: str, limit: int) -> dict[str, Any]:
        cancer = cancer.strip().upper()
        rows = self.cancer_rows(cancer, selected=True).copy()
        if rows.empty:
            lnc = pd.DataFrame()
            pathway = pd.DataFrame()
        else:
            lnc = (
                rows.sort_values(
                    ["member_weight", "calibrated_probability", "lncrna_id"],
                    ascending=[False, False, True],
                    kind="stable",
                )
                .drop_duplicates("lncrna_id")
                .head(limit)
                .copy()
            )
            lnc["rank_within_cancer"] = range(1, len(lnc) + 1)
            pathway = (
                rows.groupby(
                    ["cancer_id", "pathway_id", "pathway_name", "pathway_family_id"],
                    observed=True,
                    dropna=False,
                )
                .agg(
                    lncRNA_count=("lncrna_id", "nunique"),
                    max_calibrated_probability=("calibrated_probability", "max"),
                    mean_calibrated_probability=("calibrated_probability", "mean"),
                    max_member_weight=("member_weight", "max"),
                )
                .reset_index()
                .sort_values(
                    ["max_member_weight", "max_calibrated_probability", "pathway_id"],
                    ascending=[False, False, True],
                    kind="stable",
                )
                .head(limit)
            )
            pathway["rank_within_cancer"] = range(1, len(pathway) + 1)
        lnc_columns = [
            column
            for column in [
                "rank_within_cancer", "gene_symbol", "lncrna_id", "pathway_name",
                "pathway_id", "pathway_family_id", "final_direction",
                "calibrated_probability", "seed_probability_std", "observed_evidence_score",
                "relationship_class", "member_weight",
            ]
            if column in lnc
        ]
        return {
            "lncrna_ranking": _records(lnc[lnc_columns] if lnc_columns else lnc),
            "pathway_ranking": _records(pathway),
            "geneset_summary": _records(
                self.master.loc[self.master.cancer_id.astype(str).eq(cancer)].head(limit)
                if "cancer_id" in self.master
                else pd.DataFrame()
            ),
        }

    def lnc_profile(self, lnc_id: str, cancer: str | None, limit: int = 100) -> dict[str, Any]:
        cancers = [cancer.strip().upper()] if cancer else self.cancers
        parts = []
        for current in cancers:
            path = self._partition_path(current, selected=True)
            frame = pd.read_parquet(path, filters=[("lncrna_id", "==", lnc_id)])
            if len(frame):
                parts.append(frame)
        table = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if table.empty:
            raise KeyError(lnc_id)
        table = table.sort_values(
            ["member_weight", "calibrated_probability", "cancer_id", "pathway_id"],
            ascending=[False, False, True, True],
            kind="stable",
        )
        landscape = (
            table.groupby("cancer_id", observed=True)
            .agg(
                relationship_count=("pathway_id", "size"),
                exact_pathways=("pathway_id", "nunique"),
                max_calibrated_probability=("calibrated_probability", "max"),
                mean_calibrated_probability=("calibrated_probability", "mean"),
                max_member_weight=("member_weight", "max"),
            )
            .reset_index()
            .sort_values("max_member_weight", ascending=False)
        )
        columns = [
            column
            for column in [
                "cancer_id", "pathway_id", "pathway_name", "pathway_family_id",
                "final_direction", "calibrated_probability", "seed_probability_std",
                "observed_evidence_score", "relationship_class", "member_weight",
            ]
            if column in table
        ]
        return {
            "top_relationships": _records(table[columns].head(limit)),
            "cancer_landscape": _records(landscape),
        }

    def genesets(
        self,
        *,
        search: str | None,
        pathway: str | None,
        cancer: str | None,
        limit: int,
    ) -> dict[str, Any]:
        master = self.master.copy()
        if pathway:
            master = master.loc[master.pathway_id.astype(str).eq(pathway)]
        if cancer:
            master = master.loc[master.cancer_id.astype(str).eq(cancer.upper())]
        if search:
            needle = str(search)
            mask = master.pathway_id.astype(str).str.contains(needle, case=False, regex=False, na=False)
            if "pathway_name" in master:
                mask |= master.pathway_name.astype(str).str.contains(needle, case=False, regex=False, na=False)
            if "gene_symbol" in master:
                mask |= master.gene_symbol.astype(str).str.contains(needle, case=False, regex=False, na=False)
            master = master.loc[mask]
        total = int(len(master))
        selected_master = master.head(limit).copy()
        ids = selected_master.geneset_id.astype(str).tolist() if len(selected_master) else []
        if ids:
            try:
                member = pd.read_parquet(
                    self.member_path,
                    filters=[("geneset_id", "in", ids)],
                )
            except (TypeError, ValueError):
                member = pd.read_parquet(self.member_path)
                member = member.loc[member.geneset_id.astype(str).isin(ids)]
        else:
            member = pd.DataFrame()
        return {
            "genesets": _records(selected_master),
            "members": _records(member.head(2000)),
            "total_genesets": total,
            "pathway_target_level": "exact_pathway",
        }

    def geneset_detail(self, geneset_id: str) -> dict[str, Any]:
        """Return an exact gene set, with family/pathway IDs as legacy aliases."""

        requested = str(geneset_id).strip()
        master = self.master.copy()
        selected = master.loc[master.geneset_id.astype(str).eq(requested)]
        resolution = "geneset_id"
        if selected.empty and "pathway_family_id" in master:
            selected = master.loc[
                master.pathway_family_id.astype(str).eq(requested)
            ]
            resolution = "pathway_family_id"
        if selected.empty and "pathway_id" in master:
            selected = master.loc[master.pathway_id.astype(str).eq(requested)]
            resolution = "pathway_id"
        if selected.empty:
            raise KeyError(requested)

        ids = selected.geneset_id.astype(str).tolist()
        try:
            members = pd.read_parquet(
                self.member_path, filters=[("geneset_id", "in", ids)]
            )
        except (TypeError, ValueError):
            members = pd.read_parquet(self.member_path)
            members = members.loc[members.geneset_id.astype(str).isin(ids)]
        sort_columns = [
            column
            for column in ("geneset_id", "rank", "member_weight", "lncrna_id")
            if column in members
        ]
        if sort_columns:
            ascending = [column not in {"member_weight"} for column in sort_columns]
            members = members.sort_values(
                sort_columns, ascending=ascending, kind="stable"
            )
        selected_records = _records(selected)
        if resolution == "geneset_id":
            overview: dict[str, Any] = selected_records[0]
        else:
            overview = {
                "requested_id": requested,
                "resolution": resolution,
                "geneset_count": int(len(selected)),
                "cancer_count": int(selected.cancer_id.nunique())
                if "cancer_id" in selected
                else None,
                "pathway_count": int(selected.pathway_id.nunique())
                if "pathway_id" in selected
                else None,
                "member_count": int(len(members)),
            }
        return {
            "status": "available",
            "requested_id": requested,
            "resolution": resolution,
            "overview": overview,
            "genesets": selected_records,
            "members": _records(members.head(5000)),
            "pathway_target_level": "exact_pathway",
            "legacy_family_broadcast": False,
        }

    def stats(self) -> dict[str, Any]:
        pathways = int(self.master.pathway_id.nunique()) if "pathway_id" in self.master else 0
        return {
            "cancers": 33,
            "scored_cancers": 33,
            "reference_only_cancers": 0,
            "lncrnas": int(self.success.get("pancancer_eligible_lncRNAs", 0)),
            "exact_pathways": pathways,
            "predicted_relations": int(self.success["score_rows"]),
            "selected_relations": int(self.success["selected_rows"]),
            "genesets": int(self.success["genesets"]),
            "geneset_members": int(self.success["geneset_members"]),
            "selected_model": self.success["selected_model"],
            "version": "3.1-exact-pathway",
            "pathway_target_level": "exact_pathway",
            "pathway_family_role": "auxiliary_hierarchy_only",
        }
