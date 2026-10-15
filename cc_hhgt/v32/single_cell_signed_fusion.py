"""Donor-balanced, signed SC-HHGT features for exact-pathway fusion.

Positive and negative association strengths are separate bounded measurements,
not probabilities. The full donor/cell-context table and context summaries are
retained; pooled summaries never replace context-specific interpretation.
"""
from __future__ import annotations

import json
import socket
from dataclasses import asdict
from pathlib import Path

import duckdb

from .multimodal_fusion import ANALYSIS_VERSION, ExpertSpec, artifact_sha256


SIGNED_SC_EXPERTS = tuple(ExpertSpec(
    expert_id=f"single_cell_{direction}", probability_column=f"single_cell_{direction}_strength",
    availability_column="single_cell_available", endpoint_roles=("discovery", "confidence"),
    split_unit="canonical_donor", source_role="current_v32_oof_prediction", value_transform="identity_unit",
) for direction in ("positive", "negative"))


def _literal(path: Path) -> str:
    return "'" + str(path.resolve()).replace("'", "''") + "'"


def _record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": artifact_sha256(path), "bytes": path.stat().st_size}


def _bound_record(value: dict, label: str) -> Path:
    path = Path(value["path"])
    if not path.is_file() or artifact_sha256(path) != value.get("sha256"):
        raise ValueError(f"SC-HHGT {label} does not match its pointer")
    return path


def materialize_signed_sc_fusion(pointer_path: Path, candidates_path: Path, donor_crosswalk: Path,
                                 output: Path, *, min_donors: int = 3, arm: str | None = None) -> dict:
    """Stream the activated model's actual OOF rows to auditable fusion inputs.

    The crosswalk has dataset_id, query_donor_id, canonical_donor_id. It must
    reconcile reused patients across studies; canonical donors cannot cross
    outer folds. No label/rho/residual column enters these feature summaries.
    """
    if socket.gethostname() != "149":
        raise RuntimeError("Single-cell fusion preparation must execute on target host 149")
    if isinstance(min_donors, bool) or int(min_donors) != min_donors or min_donors < 2:
        raise ValueError("At least two independent donors are required")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if (pointer.get("status") != "PASS_ACTIVATED_SC_HHGT"
            or pointer.get("estimand") != "signed_within_donor_metacell_spearman_rho"
            or pointer.get("split_policy") != "donor_blocked_outer_oof_with_nested_inner_validation"
            or pointer.get("training_complete") is not True or pointer.get("outer_folds") != 5):
        raise ValueError("An activated, donor-OOF signed SC-HHGT pointer is required")
    selected_arm = arm or pointer.get("primary_arm")
    if selected_arm not in {"with_lasso", "without_lasso"}:
        raise ValueError("Unknown single-cell arm")
    if selected_arm != pointer.get("primary_arm"):
        raise ValueError("Ablation arm cannot replace the activated primary arm implicitly")
    output.mkdir(parents=True, exist_ok=False)
    (output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149",
        "training_started":False,"stage":"SIGNED_SC_FUSION_FEATURES"},indent=2))
    source = _bound_record(pointer["oof_predictions"], "OOF predictions")
    validation_path = _bound_record(pointer["independent_validation"], "independent validation")
    if json.loads(validation_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise ValueError("Single-cell independent validation did not pass")
    inputs = {"pointer": _record(pointer_path), "predictions": pointer["oof_predictions"],
              "candidates": _record(candidates_path), "donor_crosswalk": _record(donor_crosswalk)}
    (output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host":"149",
        "training_started":False,"stage":"SIGNED_SC_FUSION_FEATURES","inputs":inputs},indent=2))
    con=duckdb.connect()
    try:
        con.execute("SET threads=8")
        con.execute("SET memory_limit='8GB'")
        con.execute("SET temp_directory=?",[str(output/"duckdb_tmp")])
        con.execute(f"CREATE VIEW candidates AS SELECT cancer_id,lncrna_id,pathway_id FROM read_parquet({_literal(candidates_path)})")
        con.execute(f"CREATE VIEW crosswalk AS SELECT dataset_id,query_donor_id,canonical_donor_id FROM read_parquet({_literal(donor_crosswalk)})")
        for table,keys in [("candidates",["cancer_id","lncrna_id","pathway_id"]),
                           ("crosswalk",["dataset_id","query_donor_id"])]:
            if con.execute(f"SELECT count(*) FROM (SELECT {','.join(keys)} FROM {table} GROUP BY ALL HAVING count(*)>1)").fetchone()[0]:
                raise ValueError(f"Duplicate {table} identifiers")
            all_keys=keys+(["canonical_donor_id"] if table=="crosswalk" else [])
            if con.execute(f"SELECT count(*) FROM {table} WHERE "+" OR ".join(f"{k} IS NULL OR trim({k})=''" for k in all_keys)).fetchone()[0]:
                raise ValueError(f"Missing {table} identifiers")
        con.execute(f"""CREATE VIEW selected AS
            SELECT dataset_id,cancer_id,cell_key,query_donor_id,lncrna_id,pathway_id,outer_fold_id,predicted_rho
            FROM read_parquet({_literal(source)}) WHERE arm='{selected_arm}'""")
        if con.execute("SELECT count(*) FROM selected").fetchone()[0]==0:
            raise ValueError("Selected SC-HHGT arm has no predictions")
        if {row[0] for row in con.execute("SELECT DISTINCT outer_fold_id FROM selected").fetchall()} != set(range(5)):
            raise ValueError("Signed SC-HHGT predictions do not cover all five outer folds")
        errors=con.execute("""SELECT count(*) FROM selected s LEFT JOIN crosswalk d USING(dataset_id,query_donor_id)
            WHERE d.canonical_donor_id IS NULL OR s.cell_key IS NULL OR trim(s.cell_key)=''
              OR s.cancer_id IS NULL OR s.lncrna_id IS NULL OR s.pathway_id IS NULL
              OR s.predicted_rho IS NULL OR NOT isfinite(s.predicted_rho) OR abs(s.predicted_rho)>1
              OR s.outer_fold_id IS NULL OR s.outer_fold_id NOT IN (0,1,2,3,4)""").fetchone()[0]
        if errors:
            raise ValueError("Invalid signed predictions or incomplete donor crosswalk")
        duplicates=con.execute("""SELECT count(*) FROM (SELECT dataset_id,cancer_id,cell_key,query_donor_id,lncrna_id,pathway_id
            FROM selected GROUP BY ALL HAVING count(*)>1)""").fetchone()[0]
        if duplicates:
            raise ValueError("Duplicate donor/cell-context prediction rows")
        leaked=con.execute("""SELECT count(*) FROM (SELECT canonical_donor_id FROM selected JOIN crosswalk
            USING(dataset_id,query_donor_id) GROUP BY canonical_donor_id HAVING count(DISTINCT outer_fold_id)>1)""").fetchone()[0]
        if leaked:
            raise ValueError("One canonical donor appears in multiple outer folds")
        # A donor duplicated across datasets receives one contribution per
        # cell context. Its cross-dataset provenance remains in the raw table.
        con.execute("""CREATE TEMP TABLE donor_context AS SELECT s.*,d.canonical_donor_id
            FROM selected s JOIN crosswalk d USING(dataset_id,query_donor_id)
            JOIN candidates c USING(cancer_id,lncrna_id,pathway_id)""")
        con.execute("""CREATE TEMP TABLE canonical_context AS
            SELECT cancer_id,lncrna_id,pathway_id,cell_key,canonical_donor_id,
                   avg(predicted_rho) AS predicted_rho, min(outer_fold_id) AS outer_fold_id
            FROM donor_context GROUP BY ALL""")
        con.execute("""CREATE TEMP TABLE context_summary AS SELECT cancer_id,lncrna_id,pathway_id,cell_key,
            count(*) AS donors,avg(predicted_rho) AS signed_rho_mean,
            avg(abs(predicted_rho)) AS absolute_rho_mean,stddev_samp(predicted_rho) AS donor_rho_sd,
            avg(cast(predicted_rho>0 AS double)) AS positive_donor_fraction,
            count(DISTINCT outer_fold_id) AS donor_folds
            FROM canonical_context GROUP BY ALL""")
        con.execute("""CREATE TEMP TABLE donor_summary AS
            SELECT cancer_id,lncrna_id,pathway_id,canonical_donor_id,
                avg(predicted_rho) AS signed_rho_mean,
                avg(greatest(predicted_rho,0)) AS positive_strength,
                avg(greatest(-predicted_rho,0)) AS negative_strength,
                count(*) AS cell_contexts
            FROM canonical_context GROUP BY ALL""")
        con.execute("""CREATE TEMP TABLE pooled AS SELECT cancer_id,lncrna_id,pathway_id,
            count(*) AS donors,avg(signed_rho_mean) AS signed_rho_mean,
            avg(positive_strength) AS positive_strength,avg(negative_strength) AS negative_strength,
            stddev_samp(signed_rho_mean) AS donor_rho_sd
            FROM donor_summary GROUP BY ALL""")
        con.execute(f"""CREATE TEMP TABLE fusion AS SELECT c.*,
            p.donors>={min_donors} AND p.donors IS NOT NULL AS single_cell_available,
            CASE WHEN p.donors>={min_donors} THEN p.positive_strength END AS single_cell_positive_strength,
            CASE WHEN p.donors>={min_donors} THEN p.negative_strength END AS single_cell_negative_strength,
            CASE WHEN p.donors>={min_donors} THEN p.signed_rho_mean END AS single_cell_signed_rho_mean,
            CASE WHEN p.donors>={min_donors} THEN p.donor_rho_sd END AS single_cell_donor_rho_sd,
            coalesce(p.donors,0) AS single_cell_donors,
            CASE WHEN p.donors>={min_donors} THEN NULL WHEN p.donors IS NULL THEN 'NO_DONOR_OOF_ROWS'
                 ELSE 'INSUFFICIENT_INDEPENDENT_DONORS' END AS single_cell_unavailable_reason,
            '{ANALYSIS_VERSION}' AS analysis_version
            FROM candidates c LEFT JOIN pooled p USING(cancer_id,lncrna_id,pathway_id)""")
        for table,name in [("donor_context","DONOR_CELL_CONTEXT.parquet"),("context_summary","CELL_CONTEXT_SUMMARY.parquet"),
                           ("fusion","SINGLE_CELL_SIGNED_FUSION.parquet")]:
            con.execute(f"COPY {table} TO {_literal(output/name)} (FORMAT PARQUET,COMPRESSION ZSTD)")
        counts=con.execute("SELECT count(*),count(*) FILTER(WHERE single_cell_available) FROM fusion").fetchone()
        context_rows=con.execute("SELECT count(*) FROM donor_context").fetchone()[0]
    finally:
        con.close()
    artifacts={p.stem.lower():_record(p) for p in sorted(output.glob("*.parquet"))}
    # An adapter run is identified by its actual inputs and aggregation policy,
    # not by an arbitrary label supplied later in a fusion manifest.
    import hashlib
    identity={"inputs":inputs,"arm":selected_arm,"min_donors":min_donors}
    run_id="SC_SIGNED_"+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    audit={"format":"SC_HHGT_SIGNED_FUSION_BINDING_V1","status":"PASS","target_host":"149",
        "run_id":run_id,"source_model_run_id":pointer.get("run_id"),
        "estimand":"DONOR_BALANCED_SIGNED_ASSOCIATION_STRENGTH","core_dependency":"INDEPENDENT_MODEL",
        "training_started":False,"inputs":inputs,"outputs":artifacts,"arm":selected_arm,
        "rows":counts[0],"available_rows":counts[1],"donor_context_rows":context_rows,
        "min_donors":min_donors,"aggregation":"EQUAL_CONTEXT_WEIGHT_WITHIN_CANONICAL_DONOR_THEN_EQUAL_DONOR_WEIGHT",
        "direction_preserved":True,"raw_rho_treated_as_probability":False,"observed_rho_used_as_feature":False,
        "donor_sd_semantics":"DESCRIPTIVE_BETWEEN_DONOR_SPREAD_NOT_MODEL_UNCERTAINTY",
        "expert_specs":[asdict(spec) for spec in SIGNED_SC_EXPERTS],"production_deployed":False}
    (output/"AUDIT.json").write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8")
    (output/"SUCCESS.json").write_text(json.dumps({"status":"PASS","audit_sha256":artifact_sha256(output/"AUDIT.json")}))
    return audit
