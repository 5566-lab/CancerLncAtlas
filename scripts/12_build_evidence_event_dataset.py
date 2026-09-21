#!/usr/bin/env python3
"""Build candidate-level direct and indirect evidence events.

The builder is deliberately conservative:
- direct events require an explicit pathway/pathway-family assertion;
- lncRNA-partner-pathway routes are indirect mechanism events;
- TCGA expression correlations are not converted into literature events;
- source rows and PMIDs remain traceable.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.getenv("CC_HHGT_EVIDENCE_EVENT_ROOT", str(ROOT / "results" / "evidence_event_model")))
KEYS = ["cancer_id", "lncrna_id", "pathway_family_id"]


def first_existing(env_name: str, candidates: list[Path], required: bool = True) -> Path | None:
    explicit = os.getenv(env_name)
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
        raise FileNotFoundError(f"{env_name}={path} does not exist")
    for path in candidates:
        if path.exists():
            return path
    if required:
        raise FileNotFoundError(f"None of the candidate paths exists for {env_name}: {candidates}")
    return None


def discover_direct_event_inputs() -> list[Path]:
    """Locate explicitly curated lncRNA-pathway assertions without broad scans."""
    patterns = [value.strip() for value in os.getenv("CC_HHGT_DIRECT_EVENT_GLOBS", "").split(os.pathsep) if value.strip()]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend([Path(pattern)] if Path(pattern).exists() else [Path(value) for value in sorted(glob.glob(pattern))])
    if paths:
        return sorted(set(path.resolve() for path in paths))
    exact_names = [
        "direct_literature_events.parquet", "direct_literature_event.parquet",
        "direct_lncrna_pathway_events.parquet", "direct_pathway_events.parquet",
        "direct_literature_events.tsv", "direct_literature_events.csv",
    ]
    roots = [
        ROOT / "input_snapshot" / "processed", ROOT / "results" / "evidence_fusion",
        ROOT.parent / "processed", ROOT.parent / "CC_HHGT_v2_8_evidence_fusion" / "results",
        ROOT.parent / "CC_HHGT_v2_8_evidence_fusion" / "processed",
    ]
    for base in roots:
        for name in exact_names:
            candidate = base / name
            if candidate.exists():
                paths.append(candidate.resolve())
    sibling = ROOT.parent / "CC_HHGT_v2_8_evidence_fusion"
    if sibling.exists():
        for pattern in ["direct_literature_event*.parquet", "direct_lncrna_pathway*.parquet"]:
            paths.extend(path.resolve() for path in sibling.rglob(pattern))
    return sorted(set(paths))


def stable_id(*parts: object) -> str:
    text = "\x1f".join("" if p is None else str(p) for p in parts)
    return "EVT:" + hashlib.sha1(text.encode("utf-8")).hexdigest()


def clean_text(series: pd.Series, default: str = "unknown") -> pd.Series:
    value = series.astype("string").fillna("").str.strip()
    return value.mask(value.isin(["", "-", "NA", "None", "nan"]), default)


def normalize_cancer(series: pd.Series) -> pd.Series:
    value = clean_text(series, "PAN_CANCER")
    return value.replace({"ALL": "PAN_CANCER", "GLOBAL": "PAN_CANCER"})


def normalize_direct_table(path: Path, family_member: pd.DataFrame) -> pd.DataFrame:
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path, sep=None, engine="python")
    aliases = {
        "lncrna_id": ["lncrna_id", "lncRNA_id", "rna_id", "subject_id"],
        "pathway_family_id": ["pathway_family_id", "family_id"],
        "pathway_id": ["pathway_id", "pathway"],
        "cancer_id": ["cancer_id", "cancer", "tcga_code"],
        "pmid": ["pmid", "PMID", "pubmed_id"],
        "source_database": ["source_database", "source", "database"],
        "experiment_family": ["experiment_family", "assay_type", "experiment_type"],
        "relation_type": ["relation_type", "assertion_type"],
        "direction": ["direction", "effect_direction"],
        "tissue": ["tissue", "organ"],
        "cell_line": ["cell_line", "cellline"],
        "species": ["species", "organism"],
        "manual_review_status": ["manual_review_status", "review_status"],
        "evidence_level": ["evidence_level", "evidence_tier"],
        "event_weight": ["event_weight", "weight", "confidence"],
    }
    rename = {}
    for target, options in aliases.items():
        for option in options:
            if option in frame.columns:
                rename[option] = target
                break
    frame = frame.rename(columns=rename)
    if "lncrna_id" not in frame:
        raise RuntimeError(f"Direct event input {path} lacks lncRNA identifier")
    if "pathway_family_id" not in frame:
        if "pathway_id" not in frame:
            raise RuntimeError(f"Direct event input {path} lacks pathway_id/pathway_family_id")
        frame = frame.merge(family_member, on="pathway_id", how="inner")
    for column, default in {
        "cancer_id": "PAN_CANCER", "pmid": "", "source_database": path.stem,
        "experiment_family": "direct_literature", "relation_type": "direct_pathway_assertion",
        "direction": "unknown", "tissue": "unknown", "cell_line": "unknown",
        "species": "human", "manual_review_status": "unreviewed",
        "evidence_level": "direct", "event_weight": 1.0,
    }.items():
        if column not in frame:
            frame[column] = default
    frame["cancer_id"] = normalize_cancer(frame["cancer_id"])
    for column in ["pmid", "source_database", "experiment_family", "relation_type", "direction", "tissue", "cell_line", "species", "manual_review_status", "evidence_level"]:
        frame[column] = clean_text(frame[column], "unknown")
    frame["event_weight"] = pd.to_numeric(frame["event_weight"], errors="coerce").fillna(1.0).clip(0, 2)
    frame["route_type"] = "direct_literature"
    frame["direct_target_evidence"] = 1
    frame["partner_id"] = frame.get("partner_id", "")
    frame["partner_type"] = frame.get("partner_type", "pathway")
    frame["fdr"] = np.nan
    frame["n_independent_pmids"] = frame["pmid"].ne("unknown").astype(int)
    frame["n_independent_events"] = 1
    frame["is_experimental"] = 1
    frame["is_predicted"] = 0
    frame["source_file"] = str(path)
    return frame


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    project = ROOT.parent
    processed_candidates = [
        ROOT / "input_snapshot" / "processed",
        project / "processed",
        ROOT / "results" / "standardized",
    ]
    support_path = first_existing("CC_HHGT_INTERACTION_PATHWAY_SUPPORT", [p / "interaction_pathway_support.parquet" for p in processed_candidates])
    event_path = first_existing("CC_HHGT_EVIDENCE_EVENT_PATH", [p / "evidence_event.parquet" for p in processed_candidates], required=False)
    family_path = first_existing("CC_HHGT_PATHWAY_FAMILY_MEMBER", [
        ROOT / "results" / "static_pathway" / "static_pathway_family_member.parquet",
        ROOT / "results" / "tables" / "pathway_family_member.parquet",
    ])

    family = pd.read_parquet(family_path)
    weight_col = "membership_weight" if "membership_weight" in family else None
    family = family[["pathway_id", "pathway_family_id"] + ([weight_col] if weight_col else [])].drop_duplicates()
    if weight_col is None:
        family["membership_weight"] = 1.0
    else:
        family = family.rename(columns={weight_col: "membership_weight"})

    support = pd.read_parquet(support_path)
    support = support.merge(family, on="pathway_id", how="inner")
    support["cancer_id"] = normalize_cancer(support.get("cancer_id", pd.Series("PAN_CANCER", index=support.index)))
    support["lncrna_id"] = clean_text(support["lncrna_id"], "")
    support = support.loc[support["lncrna_id"].ne("")].copy()
    support["support_type"] = clean_text(support.get("support_type", pd.Series("mechanism", index=support.index)))
    support["highest_experiment_level"] = clean_text(support.get("highest_experiment_level", pd.Series("unknown", index=support.index)))
    support["direction"] = clean_text(support.get("direction", pd.Series("unknown", index=support.index)))
    support["fdr"] = pd.to_numeric(support.get("fdr"), errors="coerce")
    support["n_independent_pmids"] = pd.to_numeric(support.get("n_independent_pmids"), errors="coerce").fillna(0)
    support["n_independent_events"] = pd.to_numeric(support.get("n_independent_events"), errors="coerce").fillna(1)
    support["event_weight"] = (
        np.clip(-np.log10(support["fdr"].clip(lower=1e-12)) / 8, 0, 1)
        * np.clip(np.log1p(support["n_independent_events"]) / np.log(11), 0.1, 1)
        * support["membership_weight"].fillna(1)
    ).clip(0.05, 1.5)
    support["route_type"] = np.select(
        [
            support["support_type"].str.contains("perturb|functional|knock|overexpress|rescue", case=False, regex=True),
            support["highest_experiment_level"].str.contains("binding|RIP|CLIP|ChIRP|pull", case=False, regex=True),
        ],
        ["indirect_perturbation_route", "indirect_interaction_route"],
        default="indirect_pathway_overlap",
    )
    predicted_mask = (
        support["support_type"].str.contains("predict|computational|in.silico", case=False, regex=True)
        | support["highest_experiment_level"].str.contains("predict|computational|in.silico", case=False, regex=True)
    )
    experimental_mask = support["highest_experiment_level"].str.contains(
        "RIP|CLIP|ChIRP|pull|binding|knock|overexpress|rescue|CRISPR|qPCR|western|luciferase|assay",
        case=False, regex=True,
    ) & ~predicted_mask

    aggregate = pd.DataFrame({
        "cancer_id": support["cancer_id"],
        "lncrna_id": support["lncrna_id"],
        "pathway_family_id": support["pathway_family_id"],
        "route_type": support["route_type"],
        "source_database": "interaction_pathway_support",
        "pmid": "unknown",
        "experiment_family": support["highest_experiment_level"],
        "relation_type": support["support_type"],
        "direction": support["direction"],
        "tissue": "unknown",
        "cell_line": "unknown",
        "species": "human",
        "manual_review_status": "derived_from_curated_events",
        "evidence_level": "indirect",
        "event_weight": support["event_weight"],
        "partner_id": support.get("target_ids", ""),
        "partner_type": "gene_or_protein_set",
        "fdr": support["fdr"],
        "n_independent_pmids": support["n_independent_pmids"],
        "n_independent_events": support["n_independent_events"],
        "is_experimental": experimental_mask.astype("int8"),
        "is_predicted": predicted_mask.astype("int8"),
        "direct_target_evidence": 0,
        "source_file": str(support_path),
    })

    granular = pd.DataFrame()
    if event_path is not None:
        events = pd.read_parquet(event_path)
        required = {"evidence_event_id", "lncrna_id", "partner_id"}
        if not required.issubset(events.columns):
            raise RuntimeError(f"{event_path} lacks {sorted(required - set(events.columns))}")
        events["lncrna_id"] = clean_text(events["lncrna_id"], "")
        events["partner_id"] = clean_text(events["partner_id"], "")
        events = events.loc[events["lncrna_id"].ne("") & events["partner_id"].ne("")].copy()
        target = support[["cancer_id", "lncrna_id", "pathway_family_id", "target_ids", "membership_weight"]].copy()
        target["partner_id"] = target["target_ids"].fillna("").astype(str).str.split(r"[;,|]")
        target = target.explode("partner_id")
        target["partner_id"] = target["partner_id"].astype(str).str.strip()
        target = target.loc[target["partner_id"].ne("")].drop_duplicates(["cancer_id", "lncrna_id", "pathway_family_id", "partner_id"])
        # DuckDB handles the potentially large relation efficiently.
        con = duckdb.connect()
        con.register("events", events)
        con.register("targets", target)
        granular = con.execute("""
            SELECT t.cancer_id, e.lncrna_id, t.pathway_family_id,
                   e.evidence_event_id, e.pmid, e.partner_id,
                   e.relation_type, e.experiment_family, e.cell_line, e.tissue,
                   e.direction, e.manual_review_status, t.membership_weight
            FROM events e JOIN targets t
              ON e.lncrna_id=t.lncrna_id AND e.partner_id=t.partner_id
        """).df()
        con.close()
        if not granular.empty:
            granular["route_type"] = np.where(
                clean_text(granular["experiment_family"]).str.contains("perturb|knock|overexpress|rescue|CRISPR", case=False, regex=True),
                "indirect_perturbation_route",
                "indirect_interaction_route",
            )
            granular["source_database"] = "evidence_event"
            granular["species"] = "human"
            granular["evidence_level"] = "indirect"
            granular["event_weight"] = pd.to_numeric(granular["membership_weight"], errors="coerce").fillna(1.0)
            granular["partner_type"] = np.where(granular["partner_id"].astype(str).str.startswith("GENE:"), "gene_or_protein", "other")
            granular["fdr"] = np.nan
            granular["n_independent_pmids"] = granular["pmid"].astype(str).str.strip().ne("").astype(int)
            granular["n_independent_events"] = 1
            granular_predicted = clean_text(granular["experiment_family"]).str.contains(
                "predict|computational|in.silico", case=False, regex=True
            )
            granular["is_experimental"] = (~granular_predicted).astype("int8")
            granular["is_predicted"] = granular_predicted.astype("int8")
            granular["direct_target_evidence"] = 0
            granular["source_file"] = str(event_path)

    direct_frames = []
    direct_paths = discover_direct_event_inputs()
    require_direct = os.getenv("CC_HHGT_REQUIRE_DIRECT_EVENTS", "1") == "1"
    if require_direct and not direct_paths:
        raise FileNotFoundError(
            "V2.9 requires at least one structured direct lncRNA-pathway event file. "
            "Set CC_HHGT_DIRECT_EVENT_GLOBS to the curated parquet/TSV path."
        )
    for path in direct_paths:
        direct_frames.append(normalize_direct_table(path, family))

    frames = [aggregate]
    if not granular.empty:
        frames.append(granular)
    frames.extend(direct_frames)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column, default in {
        "cancer_id": "PAN_CANCER", "pmid": "unknown", "source_database": "unknown",
        "experiment_family": "unknown", "relation_type": "unknown", "direction": "unknown",
        "tissue": "unknown", "cell_line": "unknown", "species": "human",
        "manual_review_status": "unreviewed", "evidence_level": "indirect",
        "partner_id": "unknown", "partner_type": "unknown", "event_weight": 0.5,
        "fdr": np.nan, "n_independent_pmids": 0, "n_independent_events": 1,
        "is_experimental": 0, "is_predicted": 0, "direct_target_evidence": 0,
        "source_file": "unknown",
    }.items():
        if column not in combined:
            combined[column] = default
    combined["cancer_id"] = normalize_cancer(combined["cancer_id"])
    for column in ["lncrna_id", "pathway_family_id", "route_type", "source_database", "pmid", "experiment_family", "relation_type", "direction", "tissue", "cell_line", "species", "manual_review_status", "evidence_level", "partner_id", "partner_type"]:
        combined[column] = clean_text(combined[column])
    combined["event_weight"] = pd.to_numeric(combined["event_weight"], errors="coerce").fillna(0.5).clip(0, 2)
    combined["fdr_score"] = np.clip(-np.log10(pd.to_numeric(combined["fdr"], errors="coerce").clip(lower=1e-12)) / 12, 0, 1).fillna(0)
    combined["event_id"] = [
        stable_id(*row)
        for row in combined[["cancer_id", "lncrna_id", "pathway_family_id", "route_type", "source_database", "pmid", "partner_id", "experiment_family", "direction"]].astype(str).itertuples(index=False, name=None)
    ]
    combined = combined.drop_duplicates("event_id")
    columns = [
        "event_id", "cancer_id", "lncrna_id", "pathway_family_id", "route_type",
        "source_database", "pmid", "experiment_family", "relation_type", "direction",
        "tissue", "cell_line", "species", "manual_review_status", "evidence_level",
        "partner_id", "partner_type", "event_weight", "fdr_score",
        "n_independent_pmids", "n_independent_events", "is_experimental", "is_predicted",
        "direct_target_evidence", "source_file",
    ]
    combined = combined[columns].sort_values(KEYS + ["route_type", "event_id"]).reset_index(drop=True)
    output = OUTPUT / "evidence_event_dataset.parquet"
    combined.to_parquet(output, index=False, compression="zstd")
    source_audit = combined.groupby(["route_type", "source_database"], observed=True, as_index=False).agg(
        n_events=("event_id", "nunique"), n_candidates=("lncrna_id", "count"), n_pmids=("pmid", lambda x: x[x.ne("unknown")].nunique()),
    )
    source_audit.to_csv(OUTPUT / "evidence_event_source_audit.tsv", sep="\t", index=False)
    summary = {
        "status": "COMPLETED",
        "events": int(len(combined)),
        "direct_events": int(combined["direct_target_evidence"].eq(1).sum()),
        "indirect_events": int(combined["direct_target_evidence"].eq(0).sum()),
        "unique_pmids": int(combined.loc[combined["pmid"].ne("unknown"), "pmid"].nunique()),
        "direct_event_inputs": [str(path) for path in direct_paths],
        "direct_events_require_explicit_pathway_assertion": True,
        "tcga_correlation_not_imported_as_literature": True,
    }
    (OUTPUT / "EVENT_BUILD_SUCCESS.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
