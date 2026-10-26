"""Outcome-independent distal candidate regions and their analysis lineage.

Coordinate proximity is a candidate hypothesis, not a validated regulatory
link. Legacy Data S7 links remain useful evidence but their cohort-selected
membership cannot be used as outcome-independent expression predictors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


POLICY = "GRCH38_PROMOTER_WINDOW_CANDIDATES_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gene_core(values: pd.Series) -> pd.Series:
    if values.isna().any():
        raise ValueError("Missing gene identifier")
    result = values.astype(str).str.removeprefix("LNC:").str.split(".", regex=False).str[0]
    if result.str.strip().eq("").any():
        raise ValueError("Empty gene identifier")
    return result


def _intervals(frame, chromosome, start, end, *, one_based=False):
    result = frame.copy()
    required = [chromosome, start, end]
    if not set(required).issubset(result) or result[required].isna().any().any():
        raise ValueError("Incomplete coordinate table")
    result[chromosome] = result[chromosome].astype(str).str.removeprefix("chr")
    for column in [start, end]:
        number = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(number).all() or not number.eq(np.floor(number)).all():
            raise ValueError("Coordinates must be finite integers")
        result[column] = number.astype(np.int64)
    if one_based:
        result[start] -= 1
    if result[start].lt(0).any() or result[end].le(result[start]).any():
        raise ValueError("Invalid half-open genomic interval")
    return result


def coordinate_distal_candidates(promoters: pd.DataFrame, peaks: pd.DataFrame,
                                required_genes: pd.Series, *, max_distance_bp: int = 1_000_000):
    """Return all peaks within a fixed window of any transcript promoter.

    Promoters are BED0 half-open; official ATAC peaks are 1-based inclusive.
    Peaks intersecting ANY promoter of the same gene are excluded. No expression,
    activity, mutation frequency, p-value or held-out patient data are consumed.
    """
    if isinstance(max_distance_bp, bool) or int(max_distance_bp) != max_distance_bp or max_distance_bp < 1:
        raise ValueError("max_distance_bp must be a positive integer")
    required = set(gene_core(required_genes))
    if not required:
        raise ValueError("Empty candidate gene universe")
    p = _intervals(promoters.rename(columns={"#chrom": "chromosome"}), "chromosome", "start", "end")
    p["gene_id"] = gene_core(p.gene_id)
    p = p.loc[p.gene_id.isin(required), ["gene_id", "chromosome", "start", "end"]].drop_duplicates()
    q = _intervals(peaks, "seqnames", "start", "end", one_based=True)
    if q.name.isna().any() or q.name.duplicated().any():
        raise ValueError("Peak identities must be complete and unique")
    chromosomes = {chrom: g.sort_values("start") for chrom, g in q.groupby("seqnames", sort=True)}
    rows = []
    for (gene, chrom), group in p.groupby(["gene_id", "chromosome"], sort=True):
        local = chromosomes.get(chrom)
        if local is None:
            continue
        starts, ends = local.start.to_numpy(), local.end.to_numpy()
        distance = np.full(len(local), np.inf)
        intersects = np.zeros(len(local), dtype=bool)
        for promoter in group.itertuples(index=False):
            intersects |= (starts < promoter.end) & (ends > promoter.start)
            distance = np.minimum(distance, np.maximum(0, np.maximum(promoter.start - ends, starts - promoter.end)))
        selected = (~intersects) & (distance <= max_distance_bp)
        for peak, gap in zip(local.loc[selected].itertuples(index=False), distance[selected]):
            rows.append((f"LNC:{gene}", str(peak.name), f"chr{chrom}", int(peak.start),
                         int(peak.end), int(gap)))
    result = pd.DataFrame(rows, columns=["lncrna_id", "peak_name", "chromosome", "peak_start0",
                                         "peak_end0", "distance_to_promoter_bp"])
    if result.duplicated(["lncrna_id", "peak_name"]).any():
        raise ValueError("Gene coordinates map to conflicting chromosomes")
    return result.sort_values(["lncrna_id", "peak_name"]).reset_index(drop=True)


def topology_lineage(links_path: Path, audit_path: Path | None = None) -> dict:
    """Bind an actual topology file; unproven/legacy topology stays evidence-only."""
    digest = sha256(links_path)
    audit_path = audit_path or Path(str(links_path) + ".audit.json")
    result = {"links_sha256": digest, "policy": "UNVERIFIED_OR_COHORT_SELECTED",
              "predictive_evaluation_eligible": False, "coordinate_build": None}
    if not audit_path.is_file():
        return result
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("output_sha256") != digest:
        raise ValueError("Distal topology audit does not match the topology file")
    eligible = (audit.get("status") == "PASS" and audit.get("policy") == POLICY
                and audit.get("cohort_expression_used") is False
                and audit.get("cohort_pathway_activity_used") is False
                and audit.get("coordinate_build") == "GRCh38")
    return {**result, "policy": audit.get("policy", result["policy"]),
            "predictive_evaluation_eligible": eligible,
            "coordinate_build": audit.get("coordinate_build"), "audit_sha256": sha256(audit_path)}


def methylation_probe_lineage(probe_map: Path) -> dict:
    audit_path = probe_map.parent / "AUDIT.json"
    if not audit_path.is_file():
        return {"predictive_evaluation_eligible": False, "policy": "UNVERIFIED_PROBE_MAP"}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("output", {}).get("sha256") != sha256(probe_map):
        raise ValueError("Probe-map audit does not match the methylation mapping")
    return audit.get("topology_lineage", {"predictive_evaluation_eligible": False,
                                         "policy": "LEGACY_COHORT_SELECTED_PROBE_MAP"})


def require_shared_predictive_topology(*lineages: dict) -> dict:
    if not lineages or any(x.get("predictive_evaluation_eligible") is not True for x in lineages):
        raise ValueError("Formal distal prediction requires outcome-independent topology in every modality")
    hashes = {x.get("links_sha256") for x in lineages}
    if None in hashes or len(hashes) != 1:
        raise ValueError("Mutation, ATAC and methylation must use the same distal topology")
    return dict(lineages[0])


def feature_topology_lineage(path: Path, *, kind: str) -> dict:
    """Check that an audit's topology belongs to the consumed feature artifact."""
    audit_path = path.parent / ("DISTAL_ATAC_AUDIT.json" if kind == "atac" else "AUDIT.json")
    if not audit_path.is_file():
        return {"predictive_evaluation_eligible": False, "policy": "FEATURE_AUDIT_ABSENT"}
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if kind == "atac" and "native_distal_atac_preparation" in audit:
        from .native_distal_atac_workflow import validate_features
        validate_features({"path":str(audit_path),"sha256":sha256(audit_path)},audit_path.parent)
    lineage = audit.get("topology_lineage", {"predictive_evaluation_eligible": False,
                                            "policy": "LEGACY_FEATURES"})
    if lineage.get("predictive_evaluation_eligible") is not True:
        return lineage
    if kind == "atac":
        outputs = audit.get("output_md5", {})
        expected = [value for key, value in outputs.items() if Path(key).name == path.name]
        actual = hashlib.md5(path.read_bytes()).hexdigest()
    else:
        outputs = audit.get("outputs", {})
        expected = [value["sha256"] for value in outputs.values() if isinstance(value, dict)
                    and Path(value.get("path", "")).name == path.name and "sha256" in value]
        actual = sha256(path)
    if expected != [actual]:
        raise ValueError(f"{kind} feature differs from its topology-bound audit")
    return lineage
