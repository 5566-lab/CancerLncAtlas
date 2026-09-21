"""CPU-only lightweight query engine for the CC-HHGT V2.9 release."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .query_models import AttentionSetEncoder, ProteinInteractionDecoder


ROOT = Path(__file__).resolve().parents[2]
ASSET = Path(os.getenv("CC_HHGT_QUERY_ASSET_ROOT", str(ROOT / "query_assets")))
MODEL = Path(os.getenv("CC_HHGT_QUERY_MODEL_ROOT", str(ROOT / "models")))
RELEASE = Path(os.getenv("CC_HHGT_QUERY_RELEASE_ROOT", str(ROOT / "results/model/cc_hhgt_v2_9_state_graph")))
CACHE = Path(os.getenv("CC_HHGT_QUERY_CACHE_ROOT", str(ROOT / "query_cache")))


def _final_prediction_path() -> Path:
    configured = os.getenv("CC_HHGT_QUERY_FINAL_TABLE")
    candidates = [
        Path(configured) if configured else None,
        RELEASE / "predictions" / "final_expert_fusion_table.parquet",
        RELEASE / "final_expert_fusion_table.parquet",
        RELEASE / "v2_9_downstream" / "v2_9_release" / "final_expert_fusion_table.parquet",
        RELEASE / "final_three_probability_table.parquet",
    ]
    for path in candidates:
        if path is not None and path.exists():
            return path
    raise FileNotFoundError(
        "V2.9 final prediction table is unavailable under " + str(RELEASE)
    )
MAX_MEMBERS = 100
MAX_TOP_K = 100
MAX_NETWORK_NODES = 200


def _normal(value: str) -> str:
    return value.strip().upper()


class QueryEngine:
    def __init__(self) -> None:
        torch.set_num_threads(max(1, min(torch.get_num_threads(), 8)))
        self.device = torch.device("cpu")
        self.gene = pd.read_parquet(ASSET / "gene_embedding.parquet")
        self.protein = pd.read_parquet(ASSET / "protein_embedding.parquet")
        self.protein_gene_map = pd.read_parquet(
            ASSET / "protein_gene_symbol_map.parquet"
        )
        self.lnc = pd.read_parquet(ASSET / "global_lncrna_embedding.parquet")
        self.family = pd.read_parquet(
            ASSET / "global_pathway_family_embedding.parquet"
        )
        self.member = pd.read_parquet(
            ASSET / "static_pathway_family_member.parquet"
        )
        self.pathway_gene = pd.read_parquet(
            ASSET / "known_pathway_gene_member.parquet"
        )
        self.gene_family = pd.read_parquet(ASSET / "gene_family_weight.parquet")
        self.final = pd.read_parquet(_final_prediction_path())
        self.interaction = pd.read_parquet(
            ASSET / "physical_interaction_evidence.parquet"
        )
        self.embedding_columns = [
            column for column in self.gene if column.startswith("embedding_")
        ]
        self.gene_vectors = self.gene.set_index("gene_symbol")[
            self.embedding_columns
        ]
        self.protein_vectors = self.protein.set_index("protein_id")[
            self.embedding_columns
        ]
        self.lnc_vectors = self.lnc.set_index("canonical_id")[
            self.embedding_columns
        ]
        self.family_vectors = self.family.set_index("canonical_id")[
            self.embedding_columns
        ]
        self.family_matrix = F.normalize(
            torch.as_tensor(
                self.family_vectors.to_numpy(dtype=np.float32, copy=True)
            ),
            dim=-1,
        )
        self.family_ids = self.family_vectors.index.to_numpy()
        self._build_identifier_index()
        self._load_models()
        self.pathway_to_genes = {
            pathway: set(group["gene_symbol"])
            for pathway, group in self.pathway_gene.groupby(
                "pathway_id", observed=True
            )
        }
        self.gene_to_pathways = {
            gene: sorted(group["pathway_id"].unique())
            for gene, group in self.pathway_gene.groupby(
                "gene_symbol", observed=True
            )
        }
        self.gene_to_family = {
            gene: group.sort_values("combined_weight", ascending=False)
            for gene, group in self.gene_family.groupby("gene_symbol", observed=True)
        }
        self.interaction_index = {
            (row.lncrna_id, row.partner_id): row
            for row in self.interaction.itertuples(index=False)
        }
        self.protein_to_symbols = {
            protein_id: [
                value
                for value in group["gene_symbol"].dropna().unique()
                if value in self.gene_vectors.index
            ]
            for protein_id, group in self.protein_gene_map.groupby(
                "protein_id", observed=True
            )
        }
        CACHE.mkdir(parents=True, exist_ok=True)

    def _build_identifier_index(self) -> None:
        gene_lookup = pd.read_parquet(ASSET / "gene_identifier_index.parquet")
        lnc_lookup = pd.read_parquet(ASSET / "lncrna_identifier_index.parquet")
        self.identifiers: dict[str, list[tuple[str, str]]] = {}

        def add(token: Any, entity_type: str, canonical_id: str) -> None:
            if token is None or pd.isna(token) or not str(token).strip():
                return
            normalized = _normal(str(token))
            self.identifiers.setdefault(normalized, []).append(
                (entity_type, canonical_id)
            )

        for row in gene_lookup.itertuples(index=False):
            entity_type = str(row.entity_type)
            canonical = str(row.gene_id)
            for value in [row.gene_id, row.ensembl_gene_id, row.gene_symbol]:
                add(value, entity_type, canonical)
            if isinstance(row.aliases, str):
                for alias in row.aliases.replace(";", "|").replace(",", "|").split("|"):
                    add(alias, entity_type, canonical)
        for row in lnc_lookup.itertuples(index=False):
            canonical = str(row.lncrna_id)
            for value in [row.lncrna_id, row.ensembl_gene_id, row.gene_symbol]:
                add(value, "lncRNA", canonical)
            if isinstance(row.aliases, str):
                for alias in row.aliases.replace(";", "|").replace(",", "|").split("|"):
                    add(alias, "lncRNA", canonical)
        for protein_id in self.protein_vectors.index:
            add(protein_id, "protein", str(protein_id))
            add(str(protein_id).replace("UNIPROT:", ""), "protein", str(protein_id))

        dim_gene = gene_lookup.set_index("gene_id")
        self.gene_id_to_symbol = dim_gene["gene_symbol"].to_dict()
        lnc_symbol = lnc_lookup.set_index("lncrna_id")["gene_symbol"]
        self.lnc_id_to_symbol = lnc_symbol.to_dict()

    def _load_models(self) -> None:
        custom_checkpoint = torch.load(
            MODEL / "custom_geneset_encoder" / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.custom_model = AttentionSetEncoder(
            custom_checkpoint["embedding_dim"], custom_checkpoint["hidden"]
        )
        self.custom_model.load_state_dict(custom_checkpoint["model_state"])
        self.custom_model.eval()
        self.ood_threshold = float(custom_checkpoint["ood_threshold"])
        protein_checkpoint = torch.load(
            MODEL / "protein_set_encoder" / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.protein_model = ProteinInteractionDecoder(
            protein_checkpoint["embedding_dim"], protein_checkpoint["hidden"]
        )
        self.protein_model.load_state_dict(protein_checkpoint["model_state"])
        self.protein_model.eval()

    def classify(self, members: list[str]) -> dict[str, Any]:
        result: dict[str, list[dict[str, str]]] = {
            "lncRNA": [],
            "protein_coding_gene": [],
            "protein": [],
            "ambiguous": [],
            "unmapped": [],
        }
        for raw in members:
            matches = list(set(self.identifiers.get(_normal(raw), [])))
            if not matches:
                result["unmapped"].append({"input": raw, "canonical_id": ""})
                continue
            types = {value[0] for value in matches}
            if len(matches) > 1 and len(types) > 1:
                result["ambiguous"].append(
                    {
                        "input": raw,
                        "canonical_id": "|".join(
                            sorted(value[1] for value in matches)
                        ),
                    }
                )
                continue
            entity_type, canonical = sorted(matches)[0]
            item = {"input": raw, "canonical_id": canonical}
            if entity_type == "lncRNA":
                item["gene_symbol"] = self.lnc_id_to_symbol.get(canonical)
            result[entity_type].append(item)
        return result

    def _cache_key(self, endpoint: str, payload: dict[str, Any]) -> str:
        normalized = json.dumps(
            {"response_schema": "lncrna-symbol-v1", "endpoint": endpoint, **payload},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _cached(
        self, endpoint: str, payload: dict[str, Any], function: Any
    ) -> dict[str, Any]:
        key = self._cache_key(endpoint, payload)
        path = CACHE / f"{key}.json"
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            value["cache"] = {"sha256": key, "hit": True}
            return value
        value = function()
        value["cache"] = {"sha256": key, "hit": False}
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return value

    def _gene_symbols(
        self, classified: dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        symbols = []
        canonical = []
        for item in classified["protein_coding_gene"]:
            gene_id = item["canonical_id"]
            symbol = self.gene_id_to_symbol.get(gene_id)
            if symbol in self.gene_vectors.index:
                symbols.append(symbol)
                canonical.append(gene_id)
        return symbols, canonical

    def _encode_gene_set(
        self, symbols: list[str]
    ) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        mapped = [symbol for symbol in symbols if symbol in self.gene_vectors.index]
        if not mapped:
            return None, 0.0, None
        values = torch.as_tensor(
            self.gene_vectors.loc[mapped].to_numpy(dtype=np.float32, copy=True)[
                None, :, :
            ]
        )
        mask = torch.ones((1, len(mapped)), dtype=torch.bool)
        with torch.no_grad():
            encoded = self.custom_model(values, mask)
            score = (
                F.normalize(encoded, dim=-1) @ self.family_matrix.T
            ).squeeze(0).numpy()
        return encoded.squeeze(0).numpy(), len(mapped) / max(len(symbols), 1), score

    def _rank_lnc(
        self,
        family_ids: list[str],
        cancer_id: str | None,
        top_k: int,
    ) -> pd.DataFrame:
        table = self.final[self.final["pathway_family_id"].isin(family_ids)].copy()
        if cancer_id:
            table = table[table["cancer_id"].eq(cancer_id)]
        if table.empty:
            return table
        preferred_score = (
            "fused_confidence_probability"
            if "fused_confidence_probability" in table
            else "discovery_ranking_probability"
            if "discovery_ranking_probability" in table
            else None
        )
        if not cancer_id:
            numeric = [
                column
                for column in [
                    "cross_cancer_probability",
                    "cancer_native_probability",
                    "cancer_specific_probability",
                    "evidence_integrated_probability",
                    "discovery_ranking_probability",
                    "fused_confidence_probability",
                    "specificity_score",
                ]
                if column in table
            ]
            table = (
                table.groupby(
                    ["lncrna_id", "pathway_family_id"], observed=True
                )[numeric]
                .mean()
                .reset_index()
            )
            table["cancer_id"] = "PAN_CANCER_AGGREGATE"
        if preferred_score and preferred_score in table:
            table["functional_association_probability"] = table[preferred_score]
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
            table["functional_association_probability"] = table[fallback].mean(axis=1)
        return table.sort_values(
            "functional_association_probability", ascending=False
        ).head(top_k)

    def custom_gene_set(
        self,
        members: list[str],
        cancer_id: str | None,
        top_k: int,
    ) -> dict[str, Any]:
        payload = {"members": members, "cancer_id": cancer_id, "top_k": top_k}

        def compute() -> dict[str, Any]:
            classified = self.classify(members)
            symbols, _ = self._gene_symbols(classified)
            encoded, coverage, score = self._encode_gene_set(symbols)
            if score is None:
                return {
                    "prediction_scope": "custom_virtual_pathway",
                    "classification": classified,
                    "nearest_known_pathways": [],
                    "nearest_pathway_families": [],
                    "predicted_lncRNAs": [],
                    "coverage": 0.0,
                    "OOD": True,
                    "uncertainty": 1.0,
                }
            order = np.argsort(score)[::-1][: min(top_k, 20)]
            families = [
                {
                    "pathway_family_id": str(self.family_ids[index]),
                    "similarity": float(score[index]),
                }
                for index in order
            ]
            input_set = set(symbols)
            pathway_rank = sorted(
                (
                    (
                        pathway,
                        len(input_set & genes)
                        / max(len(input_set | genes), 1),
                    )
                    for pathway, genes in self.pathway_to_genes.items()
                    if input_set & genes
                ),
                key=lambda value: value[1],
                reverse=True,
            )[: min(top_k, 20)]
            lnc = self._rank_lnc(
                [str(self.family_ids[index]) for index in order[:5]],
                cancer_id,
                top_k,
            )
            max_score = float(score[order[0]])
            return {
                "prediction_scope": "custom_virtual_pathway",
                "classification": classified,
                "nearest_known_pathways": [
                    {"pathway_id": pathway, "gene_jaccard": float(value)}
                    for pathway, value in pathway_rank
                ],
                "nearest_pathway_families": families,
                "predicted_lncRNAs": self._records(lnc),
                "cancer_id": cancer_id or "PAN_CANCER_AGGREGATE",
                "cancer_specific_probability": (
                    float(lnc["cancer_specific_probability"].mean())
                    if len(lnc)
                    else None
                ),
                "functional_association_probability": (
                    float(lnc["functional_association_probability"].mean())
                    if len(lnc)
                    else None
                ),
                "coverage": float(coverage),
                "OOD": bool(max_score < self.ood_threshold or coverage < 0.2),
                "OOD_score": float(1 - max_score),
                "uncertainty": float(
                    min(1.0, (1 - max_score) + (1 - coverage) * 0.5)
                ),
            }

        return self._cached("custom-gene-set-lncrna", payload, compute)

    def _records(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        frame = frame.copy()
        if "lncrna_id" in frame:
            mapped = frame["lncrna_id"].astype(str).map(self.lnc_id_to_symbol)
            if "gene_symbol" not in frame:
                frame["gene_symbol"] = mapped
            else:
                existing = frame["gene_symbol"]
                usable = existing.notna() & existing.astype(str).str.strip().ne("")
                frame["gene_symbol"] = existing.where(usable, mapped)
        clean = frame.astype(object).where(pd.notna(frame), None)
        return clean.to_dict(orient="records")

    def mixed_gene_set(
        self,
        members: list[str],
        cancer_id: str | None,
        top_k: int,
        network_nodes: int,
    ) -> dict[str, Any]:
        payload = {
            "members": members,
            "cancer_id": cancer_id,
            "top_k": top_k,
            "network_nodes": network_nodes,
        }

        def compute() -> dict[str, Any]:
            classified = self.classify(members)
            symbols, canonical_genes = self._gene_symbols(classified)
            _, gene_coverage, score = self._encode_gene_set(symbols)
            family_ids = (
                [str(self.family_ids[index]) for index in np.argsort(score)[::-1][:5]]
                if score is not None
                else []
            )
            input_lnc = [
                item["canonical_id"] for item in classified["lncRNA"]
            ]
            ranked = self._rank_lnc(family_ids, cancer_id, top_k)
            if input_lnc and not ranked.empty:
                lnc_match = ranked[ranked["lncrna_id"].isin(input_lnc)]
            else:
                lnc_match = ranked
            coding_score = float(np.max(score)) if score is not None else 0.0
            lnc_score = (
                float(lnc_match["functional_association_probability"].mean())
                if len(lnc_match)
                else 0.0
            )
            concordance = (
                float(1 - abs(coding_score - lnc_score))
                if coding_score and lnc_score
                else 0.0
            )
            combined = (
                coding_score * max(len(symbols), 1)
                + lnc_score * max(len(input_lnc), 1)
            ) / max(len(symbols) + len(input_lnc), 1)
            edges = []
            for symbol in symbols:
                if symbol not in self.gene_to_pathways:
                    continue
                for pathway_id in self.gene_to_pathways[symbol][:3]:
                    edges.append(
                        {
                            "source": symbol,
                            "target": pathway_id,
                            "edge_type": "gene_to_exact_pathway",
                            "style": "solid",
                        }
                    )
            for lnc_id in input_lnc:
                for family_id in family_ids[:3]:
                    edges.append(
                        {
                            "source": lnc_id,
                            "target": family_id,
                            "edge_type": "lncRNA_to_pathway_family_prediction",
                            "style": "dashed",
                        }
                    )
            return {
                "classification": classified,
                "cancer_id": cancer_id or "PAN_CANCER_AGGREGATE",
                "coding_gene_enrichment_score": coding_score,
                "lncrna_association_score": lnc_score,
                "cross_type_concordance": concordance,
                "combined_mixed_set_score": float(combined),
                "coverage": float(gene_coverage),
                "ranked_lncRNAs": self._records(ranked),
                "network": {
                    "edges": edges[:network_nodes],
                    "node_limit": network_nodes,
                    "legend": {
                        "gene_to_exact_pathway": "solid",
                        "lncRNA_to_pathway_family_prediction": "dashed",
                        "experimental_relation": "thick_solid",
                        "PPI": "thin",
                        "drug_target": "directed",
                    },
                },
            }

        return self._cached("mixed-gene-set", payload, compute)

    def protein_set(
        self,
        members: list[str],
        cancer_id: str | None,
        top_k: int,
    ) -> dict[str, Any]:
        payload = {"members": members, "cancer_id": cancer_id, "top_k": top_k}

        def compute() -> dict[str, Any]:
            classified = self.classify(members)
            protein_ids = [
                item["canonical_id"] for item in classified["protein"]
            ]
            symbols, gene_ids = self._gene_symbols(classified)
            functional_symbols = list(symbols)
            vectors = []
            supporting_ids = []
            for symbol, gene_id in zip(symbols, gene_ids, strict=True):
                vectors.append(self.gene_vectors.loc[symbol].to_numpy(np.float32))
                supporting_ids.append(gene_id)
            for protein_id in protein_ids:
                if protein_id in self.protein_vectors.index:
                    vectors.append(
                        self.protein_vectors.loc[protein_id].to_numpy(np.float32)
                    )
                    supporting_ids.append(protein_id)
                    functional_symbols.extend(
                        self.protein_to_symbols.get(protein_id, [])
                    )
            if not vectors:
                return {
                    "classification": classified,
                    "ranked_lncRNAs": [],
                    "coverage": 0.0,
                    "OOD": True,
                    "uncertainty": 1.0,
                    "physical_interaction_probability": None,
                    "functional_association_probability": None,
                }
            protein_tensor = torch.as_tensor(
                np.stack(vectors)[None, :, :], dtype=torch.float32
            )
            protein_mask = torch.ones(
                (1, len(vectors)), dtype=torch.bool
            )
            with torch.no_grad():
                pooled = self.protein_model.protein_encoder(
                    protein_tensor, protein_mask
                ).squeeze(0)
                probabilities = []
                lnc_array = self.lnc_vectors.to_numpy(
                    dtype=np.float32, copy=True
                )
                for start in range(0, len(lnc_array), 4096):
                    lnc = torch.as_tensor(lnc_array[start : start + 4096])
                    repeated = pooled[None, :].expand(len(lnc), -1)
                    features = torch.cat(
                        [
                            lnc,
                            repeated,
                            lnc * repeated,
                            torch.abs(lnc - repeated),
                        ],
                        dim=-1,
                    )
                    probabilities.extend(
                        torch.sigmoid(
                            self.protein_model.decoder(features).squeeze(-1)
                        ).tolist()
                    )
            ranking = pd.DataFrame(
                {
                    "lncrna_id": self.lnc_vectors.index,
                    "physical_interaction_probability": probabilities,
                }
            ).sort_values(
                "physical_interaction_probability", ascending=False
            ).head(top_k)
            _, coverage, family_score = self._encode_gene_set(
                functional_symbols
            )
            if family_score is not None:
                family_ids = [
                    str(self.family_ids[index])
                    for index in np.argsort(family_score)[::-1][:5]
                ]
                functional = self._rank_lnc(family_ids, cancer_id, top_k * 5)
                if (
                    not functional.empty
                    and "functional_association_probability" in functional
                ):
                    functional_map = functional.groupby("lncrna_id")[
                        "functional_association_probability"
                    ].mean()
                    ranking["functional_association_probability"] = ranking[
                        "lncrna_id"
                    ].map(functional_map)
                else:
                    ranking["functional_association_probability"] = np.nan
            else:
                ranking["functional_association_probability"] = np.nan
            evidence_records = []
            for row in ranking.itertuples(index=False):
                known = []
                for partner_id in supporting_ids:
                    event = self.interaction_index.get(
                        (row.lncrna_id, partner_id)
                    )
                    if event is not None:
                        known.append(
                            {
                                "partner_id": partner_id,
                                "supporting_PMIDs": event.supporting_pmids,
                                "source_databases": event.source_databases,
                            }
                        )
                evidence_records.append(known)
            ranking["known_interactions"] = evidence_records
            ranking["predicted_physical_interaction"] = (
                ranking["physical_interaction_probability"] >= 0.5
            )
            return {
                "classification": classified,
                "cancer_id": cancer_id or "PAN_CANCER_AGGREGATE",
                "ranked_lncRNAs": self._records(ranking),
                "supporting_proteins": supporting_ids,
                "cold_start_input": bool(
                    any(
                        int(
                            hashlib.sha256(value.encode()).hexdigest()[:8],
                            16,
                        )
                        % 100
                        >= 95
                        for value in supporting_ids
                    )
                ),
                "physical_interaction_probability": float(
                    ranking["physical_interaction_probability"].mean()
                ),
                "functional_association_probability": (
                    float(
                        ranking["functional_association_probability"].dropna().mean()
                    )
                    if ranking["functional_association_probability"].notna().any()
                    else None
                ),
                "coverage": len(vectors) / max(len(members), 1),
                "OOD": bool(len(vectors) / max(len(members), 1) < 0.2),
                "uncertainty": float(
                    ranking["physical_interaction_probability"].std()
                ),
                "prediction_scope": "physical_and_functional_are_separate",
            }

        return self._cached("protein-set-lncrna", payload, compute)
