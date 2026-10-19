"""Native Mutation/CNV/ATAC fold outputs consumed without fitting or relabelling."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import socket

import duckdb
import pandas as pd

from .current_core_export import sha256, verified_path, validate_current_export
from .evidence_rule_sources import read_table
from .evidence_rule_summary import KEYS, candidate_aliases
from .multimodal_fusion import ANALYSIS_VERSION, ExpertSpec, _duckdb_literal as literal

INPUT_FORMAT = "NATIVE_AUXILIARY_FUSION_INPUT_V1"
FORMAT = "NATIVE_AUXILIARY_FUSION_BINDING_V1"
SEMANTICS = {
    "genomic": "POSITIVE_WITHIN_MODALITY_EVENT_PHI_CLASSIFIER",
    "cnv": "FOLD_LOCAL_UNADJUSTED_EXPRESSION_ASSOCIATION_PROXY_FROM_CNV",
    "cnv_event": "POSITIVE_CNV_EVENT_COOCCURRENCE_PHI_CLASSIFIER",
    "atac": "PROMOTER_COACCESSIBILITY_PERSISTENCE_NOT_CAUSAL_NOT_DIRECTIONAL",
}
COUNTS = ("callable_patient_n", "train_patient_n", "validation_patient_n", "oof_patient_n",
          "outer_train_patient_count", "outer_test_patient_count")


def _json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path, data): Path(path).write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8")


class Sources:
    def __init__(self): self.records = {}

    def bind(self, record, root):
        path = verified_path(record, root)
        self.records[str(path)] = {"path": str(path), "sha256": record["sha256"]}
        return path

    def document(self, record, root):
        path = self.bind(record, root)
        return path, _json(path)


def _folds(path):
    from .local_cnv_confounding import canonical_patient
    data = read_table(path)
    column = "patient_id" if "patient_id" in data else "sample_id"
    data = data[["cancer_id", column, "patient_fold_id"]].copy()
    data.columns = ["cancer_id", "patient_id", "patient_fold_id"]
    if data.isna().any().any(): raise ValueError("Patient fold authority contains nulls")
    values = pd.to_numeric(data.patient_fold_id, errors="raise")
    if not values.isin(range(5)).all() or pd.api.types.is_bool_dtype(values):
        raise ValueError("Patient fold IDs must be integers 0..4")
    data.patient_fold_id = values.astype(int)
    data.patient_id = data.patient_id.map(canonical_patient)
    if data.patient_id.eq("").any() or data.cancer_id.astype(str).str.strip().eq("").any():
        raise ValueError("Missing canonical patient/cancer identity")
    if data.groupby("patient_id").patient_fold_id.nunique().gt(1).any():
        raise ValueError("One canonical patient crosses native folds")
    if set(data.patient_fold_id) != set(range(5)): raise ValueError("Patient authority lacks five folds")
    return data.drop_duplicates().reset_index(drop=True)


def _authority_in(artifacts, candidates, folds):
    hashes = {r.get("sha256") for r in artifacts}
    if not {candidates["sha256"], folds["sha256"]}.issubset(hashes):
        raise ValueError("Native producer did not bind the selected candidate and patient authorities")


def native_sources(manifest_path, *, expected_core_run_id):
    """Read native producer formats, including actual checkpoints and split sources."""
    path = Path(manifest_path).resolve(); manifest = _json(path); root = path.parent
    kind = manifest.get("producer_kind"); modalities = manifest.get("modalities", [])
    allowed = {"genomic": {"mutation", "cnv"}, "cnv": {"cnv"}, "cnv_event": {"cnv"}, "atac": {"atac"}}
    if (manifest.get("format") != INPUT_FORMAT or kind not in allowed
            or not modalities or len(set(modalities)) != len(modalities)
            or not set(modalities).issubset(allowed[kind])
            or manifest.get("core_run_id") != expected_core_run_id):
        raise ValueError("Invalid native auxiliary manifest or selected current core")
    sources = Sources()
    candidates_path = sources.bind(manifest["candidate_authority"], root)
    patient_path = sources.bind(manifest["patient_folds"], root)
    patients = _folds(patient_path)
    native_path, native = sources.document(manifest["native_receipt"], root)
    if not manifest.get("native_run_id") or native.get("training_run_id") != manifest["native_run_id"]:
        raise ValueError("Native auxiliary run identity differs")
    lineage = native; typed_path = None; core = None; core_path = None; consumed = None
    if kind == "genomic" and native.get("format") == "CC_HHGT_V3_2_PARTITIONED_GLOBAL_GENOMIC_TRAINING_V1":
        from .native_partitioned_genomic import load_partitioned
        lineage, typed_path, checkpoints, fold_records = load_partitioned(sources, manifest, native_path, native,
            patient_path, manifest_root=root)
    elif kind in {"genomic", "atac"}:
        if native.get("status") != "SUCCESS": raise ValueError("Native auxiliary training is incomplete")
        lineage_path, lineage = sources.document({"path": native["lineage_path"], "sha256": native["lineage_sha256"]}, native_path.parent)
        if (lineage.get("training_status") != "SUCCESS" or lineage.get("training_run_id") != manifest["native_run_id"]
                or lineage.get("folds") != 5 or lineage.get("patient_fold_oof_predictions_not_fold_averaged") is not True):
            raise ValueError("Native lineage lacks five unaveraged OOF folds")
        typed_path = sources.bind({"path": native["prediction_path"], "sha256": native["prediction_sha256"]}, native_path.parent)
        typed_key = "prediction_sha256" if kind == "genomic" else "predictions_sha256"
        if lineage.get(typed_key) != native["prediction_sha256"]: raise ValueError("Native typed prediction identity drift")
        cp_path = (lineage.get("checkpoint_manifest_path") or str(lineage_path.parent/"CHECKPOINT_MANIFEST.json"))
        _, checkpoints = sources.document({"path": cp_path, "sha256": lineage["checkpoint_manifest_sha256"]}, lineage_path.parent)
        if kind == "genomic":
            _authority_in(lineage["input_artifacts"], manifest["candidate_authority"], manifest["patient_folds"])
            fold_records = lineage["patient_fold_oof_prediction_paths"]
            if native["patient_fold_oof_prediction_paths"] != fold_records: raise ValueError("Genomic fold sources drifted")
        else:
            if lineage.get("outer_test_patients_used_in_any_upstream_fit") != 0:
                raise ValueError("ATAC reports upstream test-patient fitting")
            _, input_audit = sources.document({"path": lineage["input_lineage_audit_path"],
                "sha256": lineage["input_lineage_audit_sha256"]}, lineage_path.parent)
            _authority_in(input_audit["input_artifacts"], manifest["candidate_authority"], manifest["patient_folds"])
            fold_records = [{"patient_fold_id": r["patient_fold"], "path": r["prediction_path"],
                "sha256": r["prediction_sha256"]} for r in checkpoints["records"]]
    else:
        expected_format = "CC_HHGT_V3_2_CNV_HEAD_OOF_V2" if kind=="cnv_event" else "CC_HHGT_V3_2_DIRECTIONAL_CNV_HEAD_OOF_V2"
        expected_target = "positive_cnv_event_cooccurrence_phi" if kind=="cnv_event" else "current_fold_local_unadjusted_lncrna_exact_pathway_association"
        if (native.get("format") != expected_format
                or native.get("status") not in {"READY_FOR_INDEPENDENT_AUDIT", "SUCCESS"}
                or native.get("cnv_only") is not True or native.get("five_fold_oof") is not True
                or native.get("mutation_features_used") is not False
                or native.get("target") != expected_target):
            raise ValueError("Invalid native directional CNV producer")
        if kind=="cnv_event":
            _, precheck = sources.document(native["input_precheck"],native_path.parent)
            if (precheck.get("status") != "PASS" or precheck.get("candidates_sha256") != manifest["candidate_authority"]["sha256"]
                    or precheck.get("folds_sha256") != manifest["patient_folds"]["sha256"]):
                raise ValueError("CNV event precheck source identity differs")
        if (native.get("candidate_authority_sha256") != manifest["candidate_authority"]["sha256"]
                or native.get("fold_manifest_sha256") != manifest["patient_folds"]["sha256"]):
            raise ValueError("CNV candidate/patient authority mismatch")
        if native.get("core_parameters_before_sha256") != native.get("core_parameters_after_sha256"):
            raise ValueError("Native CNV core was changed")
        if "consumed_sources" in native:
            from .cnv_input_sources import validate_consumed_sources
            consumed = validate_consumed_sources(sources, native["consumed_sources"], native_path.parent,
                kind=kind, run_id=manifest["native_run_id"], candidates=manifest["candidate_authority"],
                folds=manifest["patient_folds"], core=manifest["core_manifest"])
            if kind == "cnv":
                item = next(r for r in consumed[1]["records"] if r["role"] == "association_success")
                association = _json(sources.bind(item, consumed[0].parent))
                if (native.get("cohort") != association.get("cohort") or
                        (association.get("cohort") is not None and
                         (native.get("association_run_id") != association.get("run_id")
                          or native.get("cancers") != association["cohort"]["cancers"]))):
                    raise ValueError("Native CNV completion differs from its consumed association cohort")
        elif native.get("cohort") is not None:
            raise ValueError("Native CNV cohort requires actual consumed source records")
        checkpoints = {"records": native["checkpoints"]}
        fold_records = [{**r, "patient_fold_id": r["fold"]} for r in native["predictions"]]
        if len(fold_records) != native["prediction_file_count"]: raise ValueError("CNV partition count mismatch")
    if kind != "atac":
        core_path = sources.bind(manifest["core_manifest"], root)
        core, _, _ = validate_current_export(core_path, expected_run_id=expected_core_run_id)
        native_core_sha = lineage.get("v32_core_checkpoint_sha256") if kind == "genomic" else native.get("core_manifest_sha256")
        if native_core_sha != manifest["core_manifest"]["sha256"]:
            raise ValueError("Native auxiliary was trained against a different core export")
        if kind == "genomic" and "completion_records" in lineage:
            expected_exports = {}
            for fold in core["folds"].values():
                for node_type in ("lncRNA", "pathway"):
                    item = fold["exports"][node_type]
                    path = sources.bind(item, core_path.parent)
                    expected_exports[str(path)] = item["sha256"]
            if lineage["core_parameters_before_sha256"] != expected_exports:
                raise ValueError("Partitioned frozen core files differ from actual selected exports")
    checkpoint_map = {}
    for record in checkpoints["records"]:
        modality = record.get("modality", "cnv" if kind=="cnv_event" else kind)
        if modality not in modalities: continue
        fold = record.get("patient_fold", record.get("fold"))
        if type(fold) is not int or fold not in range(5) or (modality, fold) in checkpoint_map:
            raise ValueError("Duplicate or invalid native checkpoint fold")
        cp_path = sources.bind(record, native_path.parent)
        if kind == "atac":
            checkpoint = _json(cp_path)
            if (checkpoint.get("checkpoint_format") != "CC_HHGT_V3_2_ATAC_FOLD_CALIBRATOR_V1"
                    or checkpoint.get("training_run_id") != manifest["native_run_id"]
                    or checkpoint.get("outer_test_patients_used_for_fit") is not False
                    or checkpoint.get("input_hashes", {}).get("candidates") != manifest["candidate_authority"]["sha256"]
                    or checkpoint.get("input_hashes", {}).get("folds") != manifest["patient_folds"]["sha256"]):
                raise ValueError("ATAC checkpoint does not bind the native split/input identity")
            success_path, success = sources.document({"path": record["success_path"], "sha256": record["success_sha256"]}, native_path.parent)
            if (success.get("status") != "SUCCESS" or success.get("checkpoint_sha256") != record["sha256"]
                    or success.get("prediction_sha256") != record["prediction_sha256"]
                    or success.get("heldout_patients_used_for_fit") is not False):
                raise ValueError("ATAC checkpoint/prediction success binding drift")
        else:
            import torch
            checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)
            if checkpoint.get("checkpoint_format") != "CC_HHGT_V3_2_GENOMIC_PRIVATE_HEAD_V1":
                raise ValueError("Unsupported native genomic checkpoint")
            parent = core["folds"][str(fold)]
            if (checkpoint.get("core_parameter_sha256") != parent["core_parameter_sha256"]
                    or checkpoint.get("core_checkpoint_sha256") != parent["checkpoint_sha256"]
                    or not checkpoint.get("model_state") or not checkpoint.get("history")):
                raise ValueError("Native head lacks its actual state/history or selected fold core")
            if not all(torch.is_tensor(v) and torch.isfinite(v).all() for v in checkpoint["model_state"].values()):
                raise ValueError("Native head contains invalid state tensors")
        if checkpoint.get("patient_fold") != fold or checkpoint.get("modality") != modality:
            raise ValueError("Native checkpoint modality/fold mismatch")
        checkpoint_map[(modality, fold)] = record["sha256"]
    if kind == "genomic":
        for modality in modalities:
            states = lineage["modalities"][modality]
            if len(states) != 5 or {r["patient_fold"] for r in states} != set(range(5)):
                raise ValueError("Genomic modality lacks explicit status for all folds")
            for state in states:
                exists = (modality, state["patient_fold"]) in checkpoint_map
                if state["status"] == "SUCCESS" and not exists: raise ValueError("Genomic trained fold lacks checkpoint")
                if state["status"] not in {"SUCCESS", "NULL_WITH_REASON"} or (state["status"] == "NULL_WITH_REASON" and (exists or not state.get("reason"))):
                    raise ValueError("Genomic untrained fold lacks an explicit reason")
    elif set(checkpoint_map) != {(m,f) for m in modalities for f in range(5)}:
        raise ValueError("Native auxiliary lacks all five checkpoints")
    if {r["patient_fold_id"] for r in fold_records} != set(range(5)):
        raise ValueError("Native auxiliary lacks five prediction folds")
    bound_folds = []
    for record in fold_records:
        fold = record["patient_fold_id"]
        if type(fold) is not int or fold not in range(5): raise ValueError("Invalid native fold index")
        bound_folds.append({**record, "path": str(sources.bind(record, native_path.parent))})
    specs = [ExpertSpec("cnv_event" if m=="cnv" and kind in {"genomic","cnv_event"} else m,
        f"{m}_context_probability", f"{m}_available", ("discovery", "confidence"),
        "patient", source_role="current_v32_oof_prediction") for m in modalities]
    return dict(manifest=manifest, sources=sources, candidates_path=candidates_path, patients=patients,
        native=native, lineage=lineage, kind=kind, modalities=modalities, fold_records=bound_folds,
        typed_path=typed_path, core_path=core_path, checkpoint_map=checkpoint_map, specs=specs, consumed=consumed)


def _analyse(context, output=None, verify_fold_path=None):
    """Stream original partitions; preserve each fold and recompute equal-fold means."""
    c = context; kind = c["kind"]; con = duckdb.connect()
    try:
        con.execute("SET threads=4"); con.execute("SET memory_limit='4GB'")
        if output: con.execute("SET temp_directory=?", [str(output/"duckdb_tmp")])
        con.execute(f"CREATE VIEW candidates AS SELECT cancer_id,lncrna_id,pathway_id FROM read_parquet({literal(c['candidates_path'])})")
        candidates = con.execute("SELECT * FROM candidates").df(); candidate_aliases(candidates)
        if not set(candidates.cancer_id).issubset(set(c["patients"].cancer_id)):
            raise ValueError("Candidates lack a bound cancer patient authority")
        fragments = []
        for i, record in enumerate(c["fold_records"]):
            fold = record["patient_fold_id"]; raw = f"raw_{i}"
            con.execute(f"CREATE VIEW {raw} AS SELECT * FROM read_parquet({literal(record['path'])}, hive_partitioning=false)")
            schema = {r[0]:r[1] for r in con.execute(f"DESCRIBE {raw}").fetchall()}
            fold_column = "fold_id" if kind == "cnv" else "patient_fold" if kind=="cnv_event" else "patient_fold_id"
            if con.execute(f"SELECT count(*) FROM {raw} WHERE {fold_column} IS NULL OR {fold_column}<>{fold}").fetchone()[0]:
                raise ValueError("Native prediction file contains a different patient fold")
            if record.get("cancer") is not None and con.execute(f"SELECT count(*) FROM {raw} WHERE cancer_id<>? OR cancer_id IS NULL",[record["cancer"]]).fetchone()[0]:
                raise ValueError("Native CNV cancer partition identity differs")
            for modality in c["modalities"]:
                probability = "cnv_probability" if kind in {"cnv","cnv_event"} else f"{modality}_context_probability"
                available = f"{modality}_available"
                if schema.get(available) != "BOOLEAN" or schema.get(probability) not in {"FLOAT", "DOUBLE"}:
                    raise ValueError("Native probabilities/availability must retain numeric/boolean types")
                checkpoint_sha = c["checkpoint_map"].get((modality, fold))
                if checkpoint_sha is None and con.execute(f"SELECT count(*) FROM {raw} WHERE {available}").fetchone()[0]:
                    raise ValueError("Untrained native fold contains available predictions")
                if kind == "cnv":
                    checks = {"run_id": c["manifest"]["native_run_id"], "checkpoint_sha256": checkpoint_sha,
                        "core_manifest_sha256": c["manifest"]["core_manifest"]["sha256"],
                        "candidate_authority_sha256": c["manifest"]["candidate_authority"]["sha256"],
                        "fold_manifest_sha256": c["manifest"]["patient_folds"]["sha256"]}
                    for field, value in checks.items():
                        if con.execute(f"SELECT count(*) FROM {raw} WHERE {field} IS DISTINCT FROM ?",[value]).fetchone()[0]:
                            raise ValueError("CNV row provenance differs from its native manifest: "+field)
                if kind == "cnv_event":
                    if con.execute(f"SELECT count(*) FROM {raw} WHERE training_run_id IS DISTINCT FROM ? OR checkpoint_sha256 IS DISTINCT FROM ?",
                            [c["manifest"]["native_run_id"],checkpoint_sha]).fetchone()[0]:
                        raise ValueError("CNV event row run/checkpoint identity differs")
                reason_column = "unavailable_reason" if kind == "cnv" else f"{modality}_unavailable_reason"
                reason = f"NULLIF(trim(CAST({reason_column} AS VARCHAR)),'')" if reason_column in schema else f"CASE WHEN {available} THEN NULL ELSE '{modality.upper()}_NO_NATIVE_OOF_PREDICTION' END"
                if checkpoint_sha is None:
                    state = next(r for r in c["lineage"]["modalities"][modality] if r["patient_fold"] == fold)
                    reason = "'"+state["reason"].replace("'", "''")+"'"
                counts = ",".join(f"CAST({name} AS DOUBLE) AS {name}" if name in schema else f"NULL::DOUBLE AS {name}" for name in COUNTS)
                fragments.append(f"SELECT cancer_id,lncrna_id,pathway_id,{fold} AS patient_fold_id,'{modality}' AS modality,"
                    f"CAST({probability} AS DOUBLE) AS probability,{available} AS available,{reason} AS unavailable_reason,{counts} FROM {raw}")
        con.execute("CREATE VIEW native_folds AS "+" UNION ALL ".join(fragments))
        if con.execute("SELECT count(*) FROM native_folds WHERE available IS NULL OR (available AND (probability IS NULL OR NOT isfinite(probability) OR probability<0 OR probability>1 OR unavailable_reason IS NOT NULL)) OR (NOT available AND (probability IS NOT NULL OR unavailable_reason IS NULL))").fetchone()[0]:
            raise ValueError("Native prediction violates probability/typed-missingness semantics")
        if con.execute("SELECT count(*) FROM (SELECT cancer_id,lncrna_id,pathway_id,modality,patient_fold_id FROM native_folds GROUP BY ALL HAVING count(*)<>1)").fetchone()[0]:
            raise ValueError("Duplicate native candidate/modality/fold")
        if con.execute("SELECT count(*) FROM native_folds n ANTI JOIN candidates c USING(cancer_id,lncrna_id,pathway_id)").fetchone()[0]:
            raise ValueError("Native predictions leave the candidate authority")
        coverage = con.execute("SELECT modality,patient_fold_id,count(*) AS rows,count_if(available) AS available_rows FROM native_folds GROUP BY ALL ORDER BY ALL").df()
        if len(coverage) != 5*len(c["modalities"]) or not coverage.rows.eq(len(candidates)).all():
            raise ValueError("Native folds do not preserve the complete candidate universe")
        for field in COUNTS:
            if con.execute(f"SELECT count(*) FROM native_folds WHERE {field} IS NOT NULL AND (NOT isfinite({field}) OR {field}<0 OR {field}<>floor({field}))").fetchone()[0]:
                raise ValueError("Invalid native patient count: "+field)
        patient_counts = c["patients"].groupby(["cancer_id", "patient_fold_id"]).size().rename("authority_patient_n").reset_index()
        con.register("patient_counts", patient_counts)
        count_field = "oof_patient_n" if kind == "cnv" else "outer_test_patient_count" if kind == "atac" else None
        if count_field:
            # Raw-gap ATAC rows legitimately have zero observed patients.
            if con.execute(f"SELECT count(*) FROM native_folds n LEFT JOIN patient_counts p USING(cancer_id,patient_fold_id) WHERE {count_field}>coalesce(authority_patient_n,0) OR (available AND ({count_field} IS NULL OR {count_field}=0))").fetchone()[0]:
                raise ValueError("Native OOF patient count exceeds the registered held-out fold")
        if kind == "cnv" and con.execute("SELECT count(*) FROM native_folds WHERE callable_patient_n>oof_patient_n").fetchone()[0]:
            raise ValueError("CNV callable patients exceed its OOF fold")
        if verify_fold_path:
            con.execute(f"CREATE VIEW recorded_folds AS SELECT * FROM read_parquet({literal(verify_fold_path)})")
            for left, right in (("native_folds","recorded_folds"),("recorded_folds","native_folds")):
                if con.execute(f"SELECT count(*) FROM (SELECT * FROM {left} EXCEPT ALL SELECT * FROM {right})").fetchone()[0]:
                    raise ValueError("Native fold-context replay differs")
        con.execute("""CREATE TEMP TABLE native_summary AS SELECT cancer_id,lncrna_id,pathway_id,modality,avg(probability) AS probability,
            count(probability) AS folds_available,stddev_samp(probability) AS fold_sd,
            string_agg(DISTINCT unavailable_reason,'; ' ORDER BY unavailable_reason) AS missing_fold_reasons
            FROM native_folds GROUP BY ALL""")
        if c["typed_path"]:
            con.execute(f"CREATE VIEW native_typed AS SELECT * FROM read_parquet({literal(c['typed_path'])})")
            if con.execute("SELECT count(*) FROM native_typed WHERE training_run_id IS DISTINCT FROM ? OR analysis_version IS DISTINCT FROM ?",
                    [c["manifest"]["native_run_id"],ANALYSIS_VERSION]).fetchone()[0]:
                raise ValueError("Native aggregate run/version identity differs")
            if (con.execute("SELECT count(*) FROM native_typed").fetchone()[0] != len(candidates)
                    or con.execute("SELECT count(*) FROM (SELECT cancer_id,lncrna_id,pathway_id FROM native_typed GROUP BY ALL HAVING count(*)<>1)").fetchone()[0]
                    or con.execute("SELECT count(*) FROM native_typed ANTI JOIN candidates USING(cancer_id,lncrna_id,pathway_id)").fetchone()[0]):
                raise ValueError("Native aggregate candidate universe differs")
        result = candidates.sort_values(KEYS).reset_index(drop=True)
        for modality in c["modalities"]:
            reason = "CASE WHEN s.folds_available>0 THEN NULL ELSE s.missing_fold_reasons END"
            join = ""
            if c["typed_path"]:
                value = f"t.{modality}_context_probability"; available = f"t.{modality}_available"
                reason = f"NULLIF(trim(CAST(t.{modality}_unavailable_reason AS VARCHAR)),'')"
                join = " JOIN native_typed t USING(cancer_id,lncrna_id,pathway_id) "
                invalid = con.execute(f"""SELECT count(*) FROM native_summary s {join} WHERE modality=? AND (
                    {available} IS DISTINCT FROM (s.folds_available>0)
                    OR t.{modality}_patient_folds_with_prediction IS DISTINCT FROM s.folds_available
                    OR ((s.folds_available>0) AND ({value} IS NULL OR NOT isfinite({value}) OR abs({value}-s.probability)>1e-6 OR {reason} IS NOT NULL))
                    OR ((s.folds_available=0) AND ({value} IS NOT NULL OR {reason} IS NULL)))""",[modality]).fetchone()[0]
                if invalid:
                    raise ValueError("Native averaged probabilities disagree with their original five folds")
            local = con.execute(f"SELECT s.*,{reason} AS aggregate_unavailable_reason FROM native_summary s {join} WHERE modality=? ORDER BY cancer_id,lncrna_id,pathway_id",[modality]).df()
            result[f"{modality}_context_probability"] = local.probability.to_numpy()
            result[f"{modality}_available"] = local.folds_available.gt(0).to_numpy()
            result[f"{modality}_patient_folds_with_prediction"] = local.folds_available.to_numpy()
            result[f"{modality}_fold_sd"] = local.fold_sd.to_numpy()
            result[f"{modality}_missing_fold_reasons"] = local.missing_fold_reasons.to_numpy()
            result[f"{modality}_unavailable_reason"] = local.aggregate_unavailable_reason.to_numpy()
        result["analysis_version"] = ANALYSIS_VERSION; result["training_run_id"] = c["manifest"]["native_run_id"]
        result["native_estimand"] = SEMANTICS[kind]
        if output:
            con.execute(f"COPY (SELECT * FROM native_folds ORDER BY cancer_id,lncrna_id,pathway_id,modality,patient_fold_id) TO {literal(output/'NATIVE_FOLD_CONTEXT.parquet')} (FORMAT PARQUET, COMPRESSION ZSTD)")
            result.to_parquet(output/"AUXILIARY_FUSION.parquet",index=False,compression="zstd")
            coverage.to_parquet(output/"FOLD_COVERAGE.parquet",index=False)
        return result, coverage
    finally: con.close()


def materialize_native_auxiliary(manifest_path, output, *, expected_core_run_id):
    if socket.gethostname() != "149": raise RuntimeError("Native auxiliary preparation requires host 149")
    path = Path(manifest_path).resolve(); digest = sha256(path); output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    _write(output/"LAUNCH_PREFLIGHT.json", {"target_host":"149","training_started":False,"input_sha256":digest})
    context = native_sources(path,expected_core_run_id=expected_core_run_id)
    result, coverage = _analyse(context,output)
    if sha256(path) != digest: raise ValueError("Native auxiliary input manifest changed")
    record = lambda p:{"path":str(p),"sha256":sha256(p)}
    audit = {"format":FORMAT,"status":"PASS","target_host":"149","run_id":context["manifest"]["native_run_id"],
        "estimand":SEMANTICS[context["kind"]],"core_run_id":expected_core_run_id,
        "core_dependency":"INDEPENDENT_MODEL" if context["kind"]=="atac" else "CURRENT_CORE_EMBEDDINGS",
        "expert_specs":[asdict(s) for s in context["specs"]],"input_manifest":record(path),
        "sources":list(context["sources"].records.values()),
        "outputs":{name:record(output/file) for name,file in (("fusion","AUXILIARY_FUSION.parquet"),("fold_context","NATIVE_FOLD_CONTEXT.parquet"),("coverage","FOLD_COVERAGE.parquet"))},
        "native_status":context["native"]["status"],"rows":len(result),"fold_rows":int(coverage.rows.sum()),
        "aggregation":"EQUAL_MEAN_OF_AVAILABLE_NATIVE_FOLDS_NULL_IF_NONE","native_checkpoints_inspected":True,
        "aggregate_replay_absolute_tolerance":1e-12,"native_float32_mean_comparison_tolerance":1e-6,
        "native_model_forwards_replayed":False,"patient_split_authority_verified":True,
        "oof_scope":"NATIVE_MODALITY_PATIENT_FOLDS_NOT_NEW_PAIR_HELDOUT_VALIDATION",
        "independent_biological_validation":False,"scientific_gain_claimed":False,
        "training_started":False,"causal_or_treatment_effect_claimed":False,"production_deployed":False}
    if context["core_path"]: audit["core_manifest"] = record(context["core_path"])
    _write(output/"AUDIT.json",audit)
    _write(output/"FUSION_EXPERTS.json", [{"spec":asdict(spec),"prediction":audit["outputs"]["fusion"],
        "binding_audit":record(output/"AUDIT.json"),"run_id":audit["run_id"],"estimand":audit["estimand"]} for spec in context["specs"]])
    _write(output/"SUCCESS.json",{"status":"PASS","binding_sha256":sha256(output/"AUDIT.json"),"training_started":False})
    return output/"AUDIT.json"


def load_native_auxiliary(binding_path, *, expected_sha, expected_core_run_id):
    """Verify source artifacts and reconstruct the actual native-fold aggregate."""
    path = verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd()); audit = _json(path)
    success = _json(path.parent/"SUCCESS.json")
    if (audit.get("format") != FORMAT or audit.get("status") != "PASS" or audit.get("training_started") is not False
            or success.get("status") != "PASS" or success.get("binding_sha256") != expected_sha):
        raise ValueError("Incomplete native auxiliary binding")
    manifest_path = verified_path(audit["input_manifest"],path.parent)
    context = native_sources(manifest_path,expected_core_run_id=expected_core_run_id)
    if (audit["run_id"] != context["manifest"]["native_run_id"] or audit["estimand"] != SEMANTICS[context["kind"]]
            or audit["expert_specs"] != json.loads(json.dumps([asdict(s) for s in context["specs"]]))
            or audit["sources"] != list(context["sources"].records.values())
            or audit["core_run_id"] != expected_core_run_id
            or audit["core_dependency"] != ("INDEPENDENT_MODEL" if context["kind"]=="atac" else "CURRENT_CORE_EMBEDDINGS")):
        raise ValueError("Native auxiliary producer semantics or source identity changed")
    if context["core_path"] and audit.get("core_manifest") != {"path":str(context["core_path"]),"sha256":sha256(context["core_path"])}:
        raise ValueError("Native auxiliary core alias differs from its actual input")
    for record in audit["outputs"].values(): verified_path(record,path.parent)
    expected, coverage = _analyse(context,verify_fold_path=audit["outputs"]["fold_context"]["path"])
    if audit["rows"] != len(expected) or audit["fold_rows"] != int(coverage.rows.sum()):
        raise ValueError("Native auxiliary declared coverage differs")
    observed = pd.read_parquet(audit["outputs"]["fusion"]["path"])
    try:
        pd.testing.assert_frame_equal(observed,expected,check_dtype=False,rtol=0,atol=1e-12)
        pd.testing.assert_frame_equal(pd.read_parquet(audit["outputs"]["coverage"]["path"]),coverage,check_dtype=False,check_exact=True)
    except AssertionError as error: raise ValueError("Native auxiliary aggregation replay differs") from error
    return observed,audit
