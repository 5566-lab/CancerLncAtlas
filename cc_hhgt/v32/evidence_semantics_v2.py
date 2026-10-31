"""Evidence role separation and independent-study aggregation.

A physical partner is network support, not a direct whole-pathway outcome.
Unreviewed records are unknown, not gold negatives.  Database copies do not
become independent studies.  Conflicting direct directions remain unknown.
"""
from __future__ import annotations
import hashlib, math, re
import numpy as np
import pandas as pd

EVIDENCE_SEMANTICS="V32_CONTEXT_PRESERVING_REVIEW_QUALITY_R6"
REVIEW_POSITIVE={"accepted","validated","confirmed","curated","approved","pass","passed"}
REVIEW_NEGATIVE={"rejected","refuted","invalid","invalidated","failed","false","not approved"}
REVIEW_UNKNOWN={"","unknown","pending","not validated","unvalidated","not yet validated","not confirmed","not reviewed"}


def review_quality_target(value):
    if value is None or pd.isna(value):return math.nan
    text=re.sub(r"[ _-]+"," ",str(value).strip().lower())
    if text in REVIEW_NEGATIVE:return 0.
    if text in REVIEW_POSITIVE:return 1.
    # Unknown extensions are not matched by a permissive substring search.
    return math.nan


def _text(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def independent_study_key(row):
    pmids=sorted(set(re.findall(r"(?<!\d)\d{4,10}(?!\d)",_text(row.get("pmid")))))
    doi=_text(row.get("doi")).lower()
    dataset=_text(row.get("study_dataset_id",row.get("source_dataset"))).upper()
    accession=re.findall(r"(?:GSE\d+|SRP\d+|PRJ[A-Z]{2}\d+|E-MTAB-\d+|EGAS\d+)",dataset)
    if pmids:return "PMID:"+";".join(pmids)
    if doi:return "DOI:"+doi
    if accession:return "DATASET:"+";".join(sorted(set(accession)))
    return "UNVERIFIED:"+_text(row.get("event_id"))


def independent_events(frame:pd.DataFrame)->pd.DataFrame:
    """One biological study/context/route unit per query; preserve conflict flags."""
    if frame.empty:return frame.copy()
    data=frame.copy();data["independent_study_id"]=[independent_study_key(r) for r in data.to_dict("records")]
    # Source/database identity is excluded from the biological deduplication key.
    fields=["independent_study_id","route_type","partner_id","experiment_type","tissue","cell_line","species","experiment_id","intervention","readout","mediator_id"]
    for col in fields:
        if col not in data:data[col]=""
    rows=[]
    for _,group in data.groupby(fields,dropna=False,sort=True):
        row=group.sort_values("event_id").iloc[0].copy()
        row["source_copy_count"]=len(group)
        row["source_databases"]=";".join(sorted(set(group.get("source_database",pd.Series(dtype=str)).dropna().astype(str))))
        direct=str(row.get("route_type"))=="DIRECT_EXACT_ASSERTION"
        provenance=not str(row.independent_study_id).startswith("UNVERIFIED:")
        dirs=pd.to_numeric(group.get("direction_target",pd.Series(dtype=float)),errors="coerce")
        dirs=sorted(set(dirs[dirs.isin([0,1,2])].astype(int))) if direct and provenance else []
        # No voting that can conceal opposing experiments.
        row["direction_target"]=dirs[0] if len(dirs)==1 else -1
        row["direction_conflict"]=len(dirs)>1
        confidence=pd.to_numeric(group.get("confidence_target",pd.Series(dtype=float)),errors="coerce")
        quality=sorted(set(confidence[np.isfinite(confidence)].astype(float))) if direct and provenance else []
        row["confidence_target"]=quality[0] if len(quality)==1 else np.nan
        row["quality_review_conflict"]=len(quality)>1 or bool(group.get("review_conflict",pd.Series(False,index=group.index)).fillna(False).any())
        if row["quality_review_conflict"]:row["confidence_target"]=np.nan
        row["supervision_provenance_verified"]=provenance
        row["evidence_semantics"]=EVIDENCE_SEMANTICS
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)


def bag_targets(independent:pd.DataFrame)->tuple[float,int]:
    if independent.empty:return math.nan,-1
    # First aggregate within a study, then between studies: database coverage is not replication.
    quality=[];directions=[];conflict=False
    for _,study in independent.groupby("independent_study_id",sort=True):
        q=pd.to_numeric(study.confidence_target,errors="coerce");q=q[np.isfinite(q)]
        if len(q):quality.append(float(q.mean()))
        d=pd.to_numeric(study.direction_target,errors="coerce");d=sorted(set(d[d.isin([0,1,2])].astype(int)))
        conflict |= bool(study.get("direction_conflict",pd.Series(False,index=study.index)).any()) or len(d)>1
        directions.extend(d)
    distinct=sorted(set(directions))
    direction=distinct[0] if len(distinct)==1 and not conflict else -1
    # Target is fraction of explicitly reviewed direct study records passing review,
    # NOT probability that a lncRNA causally controls a pathway.
    return (float(np.mean(quality)) if quality else math.nan),direction


def summarize_evidence_bag(events: pd.DataFrame) -> dict:
    """Observed evidence, available even when a neural model cannot be trained.

    Counts and directions are observations, never a model-discovery probability.
    """
    clean=independent_events(events)
    if clean.empty:
        return {'raw_records':0,'independent_studies':0,'direct_studies':0,'network_support_studies':0,
                'observed_direct_direction':'unknown','direction_conflict':False,
                'reviewed_study_quality':math.nan,'neural_score_available':False,
                'changes_primary_ranking':False,'deduplicated_context_events':0,
                'unverified_provenance_records':0,
                'evidence_summary_semantics':'OBSERVED_PROVENANCE_NOT_PREDICTED_CAUSALITY'}
    direct=clean.route_type.eq('DIRECT_EXACT_ASSERTION')
    verified=~clean.independent_study_id.str.startswith('UNVERIFIED:')
    quality,direction=bag_targets(clean)
    directions=pd.to_numeric(clean.loc[direct,'direction_target'],errors='coerce')
    d=set(directions[directions.isin([0,1,2])].astype(int))
    conflict=bool(clean.get('direction_conflict',pd.Series(False,index=clean.index)).any()) or len(d)>1
    return {'raw_records':len(events),'deduplicated_context_events':len(clean),
        'independent_studies':int(clean.loc[verified,'independent_study_id'].nunique()),
        'direct_studies':int(clean.loc[verified&direct,'independent_study_id'].nunique()),
        'network_support_studies':int(clean.loc[verified&~direct,'independent_study_id'].nunique()),
        'unverified_provenance_records':int((~verified).sum()),
        'observed_direct_direction':{0:'negative',1:'neutral',2:'positive'}.get(direction,'unknown'),
        'direction_conflict':conflict,'reviewed_study_quality':quality,
        'neural_score_available':False,'changes_primary_ranking':False,
        'evidence_summary_semantics':'OBSERVED_PROVENANCE_NOT_PREDICTED_CAUSALITY'}
