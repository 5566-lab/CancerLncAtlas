"""Read-only MC3 exact-site recurrence audit by cancer.

The input ``variant_id`` is the stable chromosome/start/end/ref/alt allele ID
materialized by the V2.8 normalizer.  Rows are first collapsed to one
patient/site event so transcript or aliquot duplicates cannot inflate
recurrence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


QUERY = r"""
WITH events AS (
    SELECT upper(cancer_id) AS cancer_id, patient_id, variant_id
    FROM read_parquet(?)
    WHERE cancer_id IS NOT NULL
      AND upper(cancer_id) <> 'NONE'
      AND patient_id IS NOT NULL
      AND variant_id IS NOT NULL
    GROUP BY ALL
), sites AS (
    SELECT cancer_id, variant_id, count(DISTINCT patient_id) AS n_patients
    FROM events
    GROUP BY cancer_id, variant_id
), site_stats AS (
    SELECT cancer_id,
           count(*) AS unique_sites,
           sum(n_patients) AS patient_site_events,
           count(*) FILTER (WHERE n_patients = 1) AS singleton_sites,
           count(*) FILTER (WHERE n_patients >= 2) AS repeated_2plus_sites,
           count(*) FILTER (WHERE n_patients >= 3) AS recurrent_3plus_sites,
           max(n_patients) AS max_patients_one_site
    FROM sites
    GROUP BY cancer_id
), patient_stats AS (
    SELECT cancer_id,
           count(*) AS patients,
           median(n_sites) AS median_sites_per_patient,
           quantile_cont(n_sites, 0.25) AS q25_sites_per_patient,
           quantile_cont(n_sites, 0.75) AS q75_sites_per_patient,
           max(n_sites) AS max_sites_per_patient
    FROM (
        SELECT cancer_id, patient_id, count(*) AS n_sites
        FROM events
        GROUP BY cancer_id, patient_id
    )
    GROUP BY cancer_id
)
SELECT s.cancer_id, p.patients, s.unique_sites, s.patient_site_events,
       s.singleton_sites, s.repeated_2plus_sites, s.recurrent_3plus_sites,
       round(100.0 * s.singleton_sites / s.unique_sites, 2) AS singleton_site_pct,
       round(100.0 * s.repeated_2plus_sites / s.unique_sites, 2) AS repeated_2plus_site_pct,
       s.max_patients_one_site, p.median_sites_per_patient,
       p.q25_sites_per_patient, p.q75_sites_per_patient, p.max_sites_per_patient
FROM site_stats AS s
JOIN patient_stats AS p USING (cancer_id)
ORDER BY s.cancer_id
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-event", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.variant_event.resolve(strict=True)
    frame = duckdb.connect(database=":memory:").execute(QUERY, [str(source)]).df()
    rendered = frame.to_csv(sep="\t", index=False)
    if args.output is None:
        print(rendered, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8", newline="")


if __name__ == "__main__":
    main()
