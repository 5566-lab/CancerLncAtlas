"""Observed evidence summaries without a learned score or a causal label.

Study identity uses connected provenance tokens, not database row counts.
Different contexts and opposing directions remain visible in the event ledger.
"""
from __future__ import annotations
import hashlib
import json
import re

import numpy as np
import pandas as pd

KEYS = ["cancer_id", "lncrna_id", "pathway_id"]
MEASURED = {"DIRECT_EXACT_ASSERTION", "PARTNER_EXACT_MEMBER", "RBP_PERTURBATION_PATHWAY_BRIDGE"}
PREDICTED = {"SEQUENCE_PREDICTED_MEMBER", "COMPUTATIONAL_SUPPORT"}
ROUTES = MEASURED | PREDICTED | {"UNVERIFIED_ASSERTION"}
GLOBAL = {"", "PAN_CANCER", "GLOBAL", "GLOBAL_NOT_CANCER_SPECIFIC"}
CONTEXT = ["study_id", "route_type", "partner_id", "assay", "biosample", "tissue", "cell_line",
           "species", "experiment_id", "intervention", "readout", "mediator_id", "source_cancer_id"]
EVENT_COLUMNS = [*KEYS, "source_event_id", "source_id", "source_cancer_id", "route_type", "partner_id", "assay",
    "biosample", "tissue", "cell_line", "species", "experiment_id", "intervention", "readout", "mediator_id",
    "direction", "review_value", "raw_occurrences", "study_tokens_json", "provenance_json", "detail_json"]
COUNT_COLUMNS = ["raw_records", "deduplicated_context_events", "independent_studies", "direct_studies",
    "network_support_studies", "functional_bridge_studies", "prediction_sources", "unverified_provenance_records",
    "cancer_specific_studies", "global_studies", "reviewed_direct_studies"]
SUMMARY_COLUMNS = [*KEYS, *COUNT_COLUMNS, "observed_direct_direction", "direction_conflict",
    "review_conflict", "reviewed_study_fraction_accepted", "evidence_available", "evidence_unavailable_reason",
    "neural_score_available", "changes_primary_ranking", "independent_external_validation", "evidence_summary_semantics"]


def stable_id(prefix, values):
    return prefix+hashlib.sha256(json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def text(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def review_value(value):
    value = re.sub(r"[ _-]+", " ", text(value).lower())
    if value in {"accepted", "validated", "confirmed", "curated", "approved", "pass", "passed"}: return 1.
    if value in {"rejected", "refuted", "invalid", "invalidated", "failed", "false", "not approved"}: return 0.
    return np.nan  # Includes NOT validated; experimental status is not review.


def observed_direction(value):
    value = re.sub(r"[ _-]+", " ", text(value).lower())
    if value in {"positive", "up", "upregulated", "activation", "activates", "increase", "increased"}: return "positive"
    if value in {"negative", "down", "downregulated", "inhibition", "inhibits", "decrease", "decreased"}: return "negative"
    if value in {"neutral", "no effect", "unchanged"}: return "neutral"
    if value in {"mixed", "conflicting", "positive and negative"}: return "conflicting"
    return "unknown"


def provenance_tokens(row):
    tokens = set()
    for value in re.split(r"[|;,\s]+", text(row.get("pmid"))):
        value = re.sub(r"^PMID:", "", value, flags=re.I)
        if re.fullmatch(r"[0-9]{4,10}", value): tokens.add("PMID:"+value)
    doi = re.sub(r"^(?:https?://(?:dx\.)?doi.org/|doi:)", "", text(row.get("doi")), flags=re.I).lower()
    if re.fullmatch(r"10\.\d{4,9}/\S+", doi): tokens.add("DOI:"+doi)
    dataset = text(row.get("study_dataset_id", row.get("source_dataset"))).upper()
    tokens.update("DATASET:"+x for x in re.findall(r"\b(?:GSE\d+|SRP\d+|PRJ[A-Z]{2}\d+|E-MTAB-\d+|EGAS\d+)\b", dataset))
    experiment = text(row.get("experiment_id", row.get("experiment_accession"))).upper()
    if re.fullmatch(r"ENCSR[A-Z0-9]{6}", experiment): tokens.add("EXPERIMENT:"+experiment)
    return sorted(tokens)


def assign_study_components(events):
    """Conservatively collapse overlapping multi-PMID/DOI/dataset provenance."""
    frame = events.copy(); parent = {}
    if frame.empty:
        return frame.assign(study_id=pd.Series(dtype=str), provenance_verified=pd.Series(dtype=bool))
    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]; value = parent[value]
        return value
    for raw in frame.study_tokens_json.unique():
        tokens = json.loads(raw)
        for token in tokens: find(token)
        for token in tokens[1:]: parent[find(token)] = find(tokens[0])
    components = {}
    for token in parent: components.setdefault(find(token), []).append(token)
    identities = {token: stable_id("STUDY:", sorted(component)) for component in components.values() for token in component}
    frame["study_id"] = [identities[json.loads(row.study_tokens_json)[0]] if json.loads(row.study_tokens_json)
        else stable_id("UNVERIFIED:", [row.source_id, row.source_event_id]) for row in frame.itertuples()]
    frame["provenance_verified"] = ~frame.study_id.str.startswith("UNVERIFIED:")
    return frame


def candidate_aliases(candidates):
    """Allow only bijective prefix/case aliases; never strip gene versions."""
    if candidates.empty or not set(KEYS).issubset(candidates) or candidates[KEYS].isna().any().any() or candidates.duplicated(KEYS).any():
        raise ValueError("Evidence requires a complete unique exact candidate universe")
    maps = {}
    for column in KEYS:
        if not candidates[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError("Candidate identifiers must be strings")
        values = candidates[column].astype(str)
        if values.str.strip().ne(values).any() or values.eq("").any(): raise ValueError("Candidate identifiers must be canonical nonempty strings")
        mapping = {}
        for value in values.unique():
            alias = value.upper()
            if column == "lncrna_id": alias = re.sub(r"^(LNC|LNCRNA):", "", alias)
            if alias in mapping and mapping[alias] != value: raise ValueError("Ambiguous candidate identity alias")
            mapping[alias] = value
        maps[column] = mapping
    return maps


def attach_candidates(events, candidates):
    maps = candidate_aliases(candidates)
    frame = events.copy()
    for column in ("lncrna_id", "pathway_id"):
        aliases = frame[column].astype(str).str.upper()
        if column == "lncrna_id": aliases = aliases.str.replace(r"^(LNC|LNCRNA):", "", regex=True)
        frame[column] = aliases.map(maps[column])
    frame = frame.dropna(subset=["lncrna_id", "pathway_id"])
    global_mask = frame.source_cancer_id.isin(GLOBAL)
    local = frame.loc[~global_mask].copy()
    local["cancer_id"] = local.source_cancer_id.str.upper().map(maps["cancer_id"])
    local = local.merge(candidates[KEYS], on=KEYS, how="inner", validate="many_to_one")
    global_rows = candidates[KEYS].merge(frame.loc[global_mask].drop(columns="cancer_id"),
        on=["lncrna_id", "pathway_id"], how="inner", validate="many_to_many")
    result = pd.concat([local, global_rows], ignore_index=True)
    result["evidence_record_id"] = [stable_id("EVRULE:", [*row[:3], row[3], row[4]]) for row in
        result[[*KEYS, "source_id", "source_event_id"]].itertuples(index=False, name=None)]
    if result.evidence_record_id.duplicated().any(): raise ValueError("Duplicate evidence source event for one exact candidate")
    return result


def validate_events(events):
    if not set(EVENT_COLUMNS).issubset(events): raise ValueError("Incomplete rule event ledger")
    if events.duplicated(["source_id", "source_event_id", *KEYS]).any(): raise ValueError("Duplicate source event identity")
    if not events.route_type.isin(ROUTES).all(): raise ValueError("Unknown rule evidence route")
    if not events.direction.isin(["positive", "negative", "neutral", "conflicting", "unknown"]).all():
        raise ValueError("Unrecognized observed direction")
    for column in ("source_event_id", "source_id", "lncrna_id", "pathway_id"):
        if events[column].isna().any() or events[column].astype(str).eq("").any(): raise ValueError("Missing rule source identity")
    n = pd.to_numeric(events.raw_occurrences, errors="coerce")
    if not np.isfinite(n).all() or not n.ge(1).all() or not n.eq(n.astype(int)).all(): raise ValueError("Invalid raw occurrence counts")
    q = pd.to_numeric(events.review_value, errors="coerce").dropna()
    if not q.isin([0., 1.]).all(): raise ValueError("Review is accepted/rejected/unknown, not a model probability")
    nondirect = events.route_type.ne("DIRECT_EXACT_ASSERTION")
    if events.loc[nondirect, "direction"].ne("unknown").any() or events.loc[nondirect, "review_value"].notna().any():
        raise ValueError("Indirect/predicted evidence cannot set direct direction or review")


def update_source_matches(matches, ledger):
    """Track source events that actually reach candidates across cancer parts."""
    for source_id, group in ledger.groupby("source_id", sort=False):
        record = matches.setdefault(source_id, {"source_events": set(), "candidate_rows": 0})
        record["source_events"].update(group.source_event_id)
        record["candidate_rows"] += len(group)


def complete_source_coverage(coverage, matches):
    rows = []
    for source in coverage:
        matched = matches.get(source["source_id"], {"source_events": set(), "candidate_rows": 0})
        count = len(matched["source_events"])
        rows.append({**source, "matched_source_events": count, "candidate_evidence_rows": matched["candidate_rows"],
            "outside_candidate_universe_events": source["normalized_events"]-count})
    return pd.DataFrame(rows)


def summarize_events(events, candidates):
    """Aggregate contexts, then independent studies; retain every candidate."""
    validate_events(events); candidate_aliases(candidates)
    contexts = []; rows = []
    for keys, group in events.groupby(KEYS, sort=False):
        local = []
        for context, copies in group.groupby(CONTEXT, dropna=False, sort=False):
            row = dict(zip(CONTEXT, context)); direct = row["route_type"] == "DIRECT_EXACT_ASSERTION"
            verified = not row["study_id"].startswith("UNVERIFIED:")
            directions = set(copies.direction)-{"unknown"} if direct and verified else set()
            conflict = "conflicting" in directions or len(directions) > 1
            reviews = set(copies.review_value.dropna()) if direct and verified else set()
            row.update(zip(KEYS, keys))
            row.update(context_id=stable_id("EVCONTEXT:", [*keys, *context]),
                raw_records=int(copies.raw_occurrences.sum()), source_records=len(copies),
                provenance_verified=verified, direction_conflict=conflict,
                observed_direct_direction="conflicting" if conflict else next(iter(directions), "unknown"),
                review_conflict=len(reviews)>1, review_value=next(iter(reviews)) if len(reviews)==1 else np.nan,
                source_record_ids_json=json.dumps(sorted(copies.evidence_record_id)),
                source_ids_json=json.dumps(sorted(set(copies.source_id))))
            local.append(row)
        contexts.extend(local); clean = pd.DataFrame(local)
        measured = clean.provenance_verified & clean.route_type.isin(MEASURED)
        direct = clean.provenance_verified & clean.route_type.eq("DIRECT_EXACT_ASSERTION")
        directions = set(clean.loc[direct, "observed_direct_direction"])-{"unknown"}
        conflict = "conflicting" in directions or len(directions)>1
        quality = []; quality_conflict = bool(clean.review_conflict.any())
        for _, study in clean.loc[direct].groupby("study_id"):
            reviews = set(study.review_value.dropna())
            if len(reviews)>1 or study.review_conflict.any(): quality_conflict = True
            elif len(reviews)==1: quality.append(next(iter(reviews)))
        rows.append({**dict(zip(KEYS, keys)), "raw_records": int(group.raw_occurrences.sum()),
            "deduplicated_context_events": len(clean), "independent_studies": clean.loc[measured, "study_id"].nunique(),
            "direct_studies": clean.loc[direct, "study_id"].nunique(),
            "network_support_studies": clean.loc[measured & clean.route_type.eq("PARTNER_EXACT_MEMBER"), "study_id"].nunique(),
            "functional_bridge_studies": clean.loc[measured & clean.route_type.eq("RBP_PERTURBATION_PATHWAY_BRIDGE"), "study_id"].nunique(),
            "prediction_sources": clean.loc[clean.route_type.isin(PREDICTED), "study_id"].nunique(),
            "unverified_provenance_records": int((~clean.provenance_verified).sum()),
            "cancer_specific_studies": clean.loc[measured & ~clean.source_cancer_id.isin(GLOBAL), "study_id"].nunique(),
            "global_studies": clean.loc[measured & clean.source_cancer_id.isin(GLOBAL), "study_id"].nunique(),
            "reviewed_direct_studies": len(quality), "observed_direct_direction": "conflicting" if conflict else next(iter(directions), "unknown"),
            "direction_conflict": conflict, "review_conflict": quality_conflict,
            "reviewed_study_fraction_accepted": float(np.mean(quality)) if quality else np.nan})
    summary = candidates[KEYS].merge(pd.DataFrame(rows, columns=[*KEYS, *COUNT_COLUMNS,
        "observed_direct_direction", "direction_conflict", "review_conflict", "reviewed_study_fraction_accepted"]),
        on=KEYS, how="left", validate="one_to_one")
    for column in COUNT_COLUMNS: summary[column] = summary[column].fillna(0).astype("int64")
    summary["observed_direct_direction"] = summary.observed_direct_direction.fillna("unknown")
    for column in ("direction_conflict", "review_conflict"): summary[column] = summary[column].fillna(False).astype(bool)
    summary["evidence_available"] = summary.raw_records.gt(0)
    summary["evidence_unavailable_reason"] = np.where(summary.evidence_available, None, "NO_REGISTERED_EVIDENCE_NOT_A_NEGATIVE")
    summary["neural_score_available"] = False; summary["changes_primary_ranking"] = False
    summary["independent_external_validation"] = False
    summary["evidence_summary_semantics"] = "OBSERVED_PROVENANCE_NOT_PREDICTED_CAUSALITY"
    return summary[SUMMARY_COLUMNS], pd.DataFrame(contexts)
