"""Independent SQL aggregation of the source-bound rule event ledger.

Source normalization is separately recomputed with the declared adapters.
The SQL summary does not call the production pandas summary implementation.
This is a computational audit, not independent biological validation.
"""
import numpy as np
import pandas as pd
import duckdb

from .evidence_rule_summary import KEYS, CONTEXT, COUNT_COLUMNS, SUMMARY_COLUMNS


def sql_summary(events, candidates):
    connection = duckdb.connect()
    connection.register("input_events", events); connection.register("candidate_universe", candidates[KEYS])
    context_keys = ",".join('"'+column+'"' for column in [*KEYS, *CONTEXT])
    keys = ",".join(KEYS)
    join = " AND ".join(f"c.{k}=s.{k}" for k in KEYS)
    qjoin = " AND ".join(f"c.{k}=q.{k}" for k in KEYS)
    counts = ",".join(f"coalesce(s.{k},0)::BIGINT AS {k}" for k in COUNT_COLUMNS if k != "reviewed_direct_studies")
    try:
        return connection.execute(f"""
        WITH scoped AS (
          SELECT *, route_type='DIRECT_EXACT_ASSERTION' AND provenance_verified AS direct,
            route_type IN ('DIRECT_EXACT_ASSERTION','PARTNER_EXACT_MEMBER','RBP_PERTURBATION_PATHWAY_BRIDGE')
              AND provenance_verified AS measured
          FROM input_events
        ), contexts AS (
          SELECT {context_keys}, bool_or(provenance_verified) AS verified,
            bool_or(direct) AS direct, bool_or(measured) AS measured,
            sum(CAST(raw_occurrences AS BIGINT)) AS raw_records,
            count(DISTINCT direction) FILTER (WHERE direct AND direction<>'unknown') AS n_direction,
            min(direction) FILTER (WHERE direct AND direction<>'unknown') AS any_direction,
            coalesce(bool_or(direction='conflicting') FILTER (WHERE direct),false) AS direction_conflict,
            count(DISTINCT CAST(review_value AS DOUBLE)) FILTER (WHERE direct) AS n_review,
            min(CAST(review_value AS DOUBLE)) FILTER (WHERE direct) AS review_value
          FROM scoped GROUP BY {context_keys}
        ), study_review AS (
          SELECT {keys}, study_id, count(DISTINCT review_value) FILTER (WHERE n_review=1) AS n_review,
            min(review_value) FILTER (WHERE n_review=1) AS review_value,
            bool_or(n_review>1) AS context_conflict
          FROM contexts WHERE direct GROUP BY {keys},study_id
        ), q AS (
          SELECT {keys}, count(*) FILTER (WHERE n_review=1 AND NOT context_conflict) AS reviewed_direct_studies,
            avg(review_value) FILTER (WHERE n_review=1 AND NOT context_conflict) AS reviewed_study_fraction_accepted,
            bool_or(n_review>1 OR context_conflict) AS review_conflict
          FROM study_review GROUP BY {keys}
        ), s AS (
          SELECT {keys}, sum(raw_records) AS raw_records, count(*) AS deduplicated_context_events,
            count(DISTINCT study_id) FILTER (WHERE measured) AS independent_studies,
            count(DISTINCT study_id) FILTER (WHERE direct) AS direct_studies,
            count(DISTINCT study_id) FILTER (WHERE measured AND route_type='PARTNER_EXACT_MEMBER') AS network_support_studies,
            count(DISTINCT study_id) FILTER (WHERE measured AND route_type='RBP_PERTURBATION_PATHWAY_BRIDGE') AS functional_bridge_studies,
            count(DISTINCT study_id) FILTER (WHERE route_type IN ('SEQUENCE_PREDICTED_MEMBER','COMPUTATIONAL_SUPPORT')) AS prediction_sources,
            count(*) FILTER (WHERE NOT verified) AS unverified_provenance_records,
            count(DISTINCT study_id) FILTER (WHERE measured AND source_cancer_id NOT IN ('','PAN_CANCER','GLOBAL','GLOBAL_NOT_CANCER_SPECIFIC')) AS cancer_specific_studies,
            count(DISTINCT study_id) FILTER (WHERE measured AND source_cancer_id IN ('','PAN_CANCER','GLOBAL','GLOBAL_NOT_CANCER_SPECIFIC')) AS global_studies,
            bool_or(direction_conflict OR n_direction>1) OR count(DISTINCT any_direction)>1 AS direction_conflict,
            min(any_direction) AS any_direction
          FROM contexts GROUP BY {keys}
        ) SELECT c.*, {counts}, coalesce(q.reviewed_direct_studies,0)::BIGINT AS reviewed_direct_studies,
          CASE WHEN s.direction_conflict THEN 'conflicting' ELSE coalesce(s.any_direction,'unknown') END AS observed_direct_direction,
          coalesce(s.direction_conflict,false) AS direction_conflict, coalesce(q.review_conflict,false) AS review_conflict,
          q.reviewed_study_fraction_accepted,
          coalesce(s.raw_records,0)>0 AS evidence_available,
          CASE WHEN coalesce(s.raw_records,0)>0 THEN NULL ELSE 'NO_REGISTERED_EVIDENCE_NOT_A_NEGATIVE' END AS evidence_unavailable_reason,
          false AS neural_score_available, false AS changes_primary_ranking, false AS independent_external_validation,
          'OBSERVED_PROVENANCE_NOT_PREDICTED_CAUSALITY' AS evidence_summary_semantics
        FROM candidate_universe c LEFT JOIN s ON {join} LEFT JOIN q ON {qjoin}
        ORDER BY {','.join('c.'+k for k in KEYS)}
        """).df()[SUMMARY_COLUMNS]
    finally: connection.close()


def assert_summary_equal(observed, expected):
    if list(observed.columns) != SUMMARY_COLUMNS or observed.duplicated(KEYS).any(): raise ValueError("Noncanonical rule summary schema/keys")
    left = observed.sort_values(KEYS).reset_index(drop=True); right = expected.sort_values(KEYS).reset_index(drop=True)
    if not left[KEYS].equals(right[KEYS]): raise ValueError("Independent rule audit found a different candidate universe")
    for column in SUMMARY_COLUMNS[len(KEYS):]:
        if column == "reviewed_study_fraction_accepted":
            same = np.allclose(left[column].to_numpy(float), right[column].to_numpy(float), equal_nan=True, rtol=0, atol=1e-12)
        else: same = left[column].astype("string").fillna("<NULL>").equals(right[column].astype("string").fillna("<NULL>"))
        if not same: raise ValueError(f"Independent SQL rule recomputation disagrees: {column}")
