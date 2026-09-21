from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .common import input_path, read_table, write_table

KNOWN_RESOURCES = [
    "NPInter", "RNAInter", "LncTarD", "LncRNA2Target", "LncACTdb",
    "Lnc2Cancer", "LncRNADisease", "RNADisease", "ncRNADrug",
    "GSE85011", "STRING", "DrugCentral", "PRISM", "GDSC",
    "Cell Model Passports", "DepMap 24Q2",
]

EXTERNAL_INPUTS = {
    "Lnc2Cancer": "lnc2cancer_external",
    "LncRNADisease": "lncrnadisease_external",
    "RNADisease": "rnadisease_experimental_external",
    "GSE85011": "gse85011_sample_metadata",
}
ISOLATED_EXTERNAL_RESOURCES = set(EXTERNAL_INPUTS)


def _norm_source(value: str) -> str:
    text = str(value).lower()
    mapping = {
        "npinter": "NPInter", "rnainter": "RNAInter", "lnctard": "LncTarD",
        "lncrna2target": "LncRNA2Target", "lncact": "LncACTdb", "lnc2cancer": "Lnc2Cancer",
        "lncrnadisease": "LncRNADisease", "rnadisease": "RNADisease", "ncrnadrug": "ncRNADrug",
        "string": "STRING", "drugcentral": "DrugCentral", "prism": "PRISM", "gdsc": "GDSC",
        "gse85011": "GSE85011", "cell model": "Cell Model Passports", "depmap": "DepMap 24Q2",
    }
    for key, name in mapping.items():
        if key in text:
            return name
    return str(value)


def build_resource_usage_manifest(cfg: dict[str, Any]) -> pd.DataFrame:
    rows = {name: {"resource_name": name} for name in KNOWN_RESOURCES}
    relation_path = cfg["_standardized"] / "interaction_relation.parquet"
    if relation_path.exists():
        rel = read_table(relation_path)
        rel["resource_name"] = rel.source_database.map(_norm_source)
        summary = rel.groupby("resource_name", observed=True).agg(
            standardized_relation_count=("interaction_row_id", "nunique"),
            mapped_lncRNA_count=("lncrna_id", "nunique"),
            mapped_partner_count=("partner_id", "nunique"),
            experimental_relation_count=("is_experimental", "sum"),
            predicted_relation_count=("is_predicted", "sum"),
        ).reset_index()
        for r in summary.itertuples(index=False):
            rows.setdefault(r.resource_name, {"resource_name": r.resource_name}).update(r._asdict())
    curated_path = cfg["_standardized"] / "lncRNA_drug_curated_evidence.parquet"
    if curated_path.exists():
        cur = read_table(curated_path)
        cur["resource_name"] = cur.source_database.map(_norm_source)
        for name, group in cur.groupby("resource_name", observed=True):
            rows.setdefault(name, {"resource_name": name})["curated_drug_evidence_count"] = len(group)
    for key, name in [("prism_lnc_drug", "PRISM"), ("gdsc_lnc_drug", "GDSC"), ("cell_model_passports_model", "Cell Model Passports"), ("depmap_protein_expression", "DepMap 24Q2")]:
        path = input_path(cfg, key)
        rows[name]["downloaded"] = bool(path and path.exists())
        rows[name]["input_path"] = str(path) if path else None
    for name, key in EXTERNAL_INPUTS.items():
        path = input_path(cfg, key)
        rows[name]["downloaded"] = bool(path and path.exists())
        rows[name]["input_path"] = str(path) if path else None
    # RNADisease is supplied as separate experimental and predicted snapshots.
    predicted_path = input_path(cfg, "rnadisease_predicted_external")
    rows["RNADisease"]["predicted_input_path"] = str(predicted_path) if predicted_path else None
    rows["RNADisease"]["downloaded"] = bool(
        rows["RNADisease"].get("downloaded") and predicted_path and predicted_path.exists()
    )

    external_path = cfg["_standardized"] / "external_validation_evidence.parquet"
    if external_path.exists():
        external = read_table(external_path)
        summary = external.groupby("source_database", observed=True).agg(
            standardized_external_record_count=("external_evidence_id", "nunique"),
            external_mapped_lncRNA_count=("lncrna_id", "nunique"),
            external_mapped_cancer_count=("cancer_id", "nunique"),
            primary_external_validation_count=("primary_validation_eligible", "sum"),
            secondary_consistency_count=("consistency_validation_eligible", "sum"),
            training_pmid_overlap_count=("training_pmid_overlap", "sum"),
        )
        for name, values in summary.iterrows():
            resource_name = _norm_source(name)
            rows.setdefault(resource_name, {"resource_name": resource_name}).update(
                values.to_dict()
            )
    for row in rows.values():
        row.setdefault("downloaded", True if row.get("standardized_relation_count", 0) or row.get("curated_drug_evidence_count", 0) else False)
        row["used_in_annotation"] = row["resource_name"] in {"Cell Model Passports", "DepMap 24Q2"}
        row["used_in_graph"] = row["resource_name"] in {"NPInter", "RNAInter", "LncTarD", "LncRNA2Target", "LncACTdb", "STRING", "DrugCentral", "ncRNADrug"}
        row["used_in_pair_evidence"] = row["resource_name"] in {"NPInter", "RNAInter", "LncTarD", "LncRNA2Target", "LncACTdb", "ncRNADrug", "PRISM", "GDSC", "DrugCentral"}
        row["used_in_training_label"] = False
        row["used_as_weak_supervision"] = row["used_in_pair_evidence"]
        row["used_in_external_validation"] = bool(
            row["resource_name"] in ISOLATED_EXTERNAL_RESOURCES
            and row.get("standardized_external_record_count", 0)
        )
        if row.get("standardized_external_record_count", 0):
            row["provenance_status"] = "TRACEABLE_TO_ISOLATED_EXTERNAL_EVIDENCE"
        elif row.get("standardized_relation_count") is not None:
            row["provenance_status"] = "TRACEABLE_TO_STANDARDIZED_RELATION"
        else:
            row["provenance_status"] = "RESOURCE_LEVEL_ONLY"
    out = pd.DataFrame(rows.values()).fillna(0)
    write_table(out, cfg["_results"] / "tables" / "resource_usage_manifest.tsv")
    return out


def refresh_resource_usage_after_model(cfg: dict[str, Any]) -> pd.DataFrame:
    manifest_path = cfg["_results"] / "tables" / "resource_usage_manifest.tsv"
    manifest = read_table(manifest_path) if manifest_path.exists() else build_resource_usage_manifest(cfg)
    graph_path = cfg["_results"] / "tables" / "graph_edge.parquet"
    if graph_path.exists():
        graph = read_table(graph_path, columns=["source_database", "edge_id"])
        graph["resource_name"] = graph.source_database.map(_norm_source)
        counts = graph.groupby("resource_name").edge_id.nunique()
        manifest["graph_edge_count"] = manifest.resource_name.map(counts).fillna(0).astype(int)
    else:
        manifest["graph_edge_count"] = 0
    pair_path = cfg["_results"] / "tables" / "pair_evidence_source_contribution.parquet"
    if pair_path.exists():
        pair = read_table(pair_path, columns=["source_database", "pair_id"])
        pair["resource_name"] = pair.source_database.map(_norm_source)
        counts = pair.groupby("resource_name").pair_id.nunique()
        manifest["pair_evidence_count"] = manifest.resource_name.map(counts).fillna(0).astype(int)
    else:
        manifest["pair_evidence_count"] = 0
    manifest["used_in_graph"] = manifest.graph_edge_count > 0
    manifest["used_in_pair_evidence"] = manifest.pair_evidence_count > 0
    manifest["used_as_weak_supervision"] = manifest.used_in_pair_evidence
    manifest["used_in_training_label"] = False
    external_counts = (
        pd.to_numeric(
            manifest["standardized_external_record_count"], errors="coerce"
        ).fillna(0)
        if "standardized_external_record_count" in manifest
        else pd.Series(0, index=manifest.index, dtype=float)
    )
    manifest["used_in_external_validation"] = (
        manifest.resource_name.isin(ISOLATED_EXTERNAL_RESOURCES)
        & external_counts.gt(0)
    )
    external_mask = manifest.resource_name.isin(ISOLATED_EXTERNAL_RESOURCES)
    manifest["external_training_isolation_status"] = "NOT_APPLICABLE"
    manifest.loc[
        external_mask
        & ~manifest.used_in_graph
        & ~manifest.used_in_pair_evidence
        & ~manifest.used_in_training_label,
        "external_training_isolation_status",
    ] = "PASS"
    manifest.loc[
        external_mask
        & (manifest.used_in_graph | manifest.used_in_pair_evidence | manifest.used_in_training_label),
        "external_training_isolation_status",
    ] = "FAIL"
    manifest["usage_status"] = "NOT_USED"
    manifest.loc[manifest.downloaded.astype(bool), "usage_status"] = "DOWNLOADED_ONLY"
    manifest.loc[manifest.get("standardized_relation_count", 0).fillna(0).astype(float) > 0, "usage_status"] = "STANDARDIZED"
    manifest.loc[manifest.used_in_graph, "usage_status"] = "USED_IN_GRAPH"
    manifest.loc[manifest.used_in_pair_evidence, "usage_status"] = "USED_IN_PAIR_EVIDENCE"
    manifest.loc[manifest.used_in_graph & manifest.used_in_pair_evidence, "usage_status"] = "USED_IN_GRAPH_AND_PAIR_EVIDENCE"
    manifest.loc[
        external_counts.gt(0),
        "usage_status",
    ] = "USED_IN_ISOLATED_EXTERNAL_VALIDATION"
    write_table(manifest, manifest_path)
    return manifest
