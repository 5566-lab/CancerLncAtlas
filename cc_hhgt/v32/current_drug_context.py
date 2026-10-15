"""Current Drug context: sparse predictions, measurements and frozen ablation.

No dense selected-member x all-drug universe is generated. Missing predictions
are resolved from the native factors on demand and are never replaced by zero.
Weighted GeneSet summaries describe available members; they are not predictions
of a joint GeneSet response, treatment efficacy or biological causality.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile

import pandas as pd

from .current_core_export import sha256, verified_path
from .current_drug_sources import (current_drug_sources, replay_native_reports, same_relation, read_json, KEYS)
from .drug_sparse_query import (DrugSparseQueryBundle, _bounded_duckdb, _parquet_relation,
    _parquet_file_relation, PROBABILITY_COLUMN)
from .multimodal_fusion import _duckdb_literal as literal

FORMAT = "CURRENT_DRUG_CONTEXT_RELEASE_V1"
MEMBERS = ["geneset_id","geneset_rank","cancer_id","lncrna_id","pathway_id","selected_score","association_direction"]
PROFILE_KEYS = ["geneset_id","cancer_id","pathway_id","association_direction","drug_id"]
SEMANTICS = {
    "model_probability":"ASSOCIATION_STRENGTH_NOT_SENSITIVITY_DIRECTION_OR_PATIENT_BENEFIT",
    "measured_response":"DESCRIPTIVE_CELL_MODEL_EXPRESSION_SENSITIVITY_ASSOCIATION",
    "ablation":"FROZEN_MODEL_INPUT_REMOVAL_NONADDITIVE_NOT_BIOLOGICAL_CAUSALITY",
    "geneset_profiles":"SELECTED_SCORE_WEIGHTED_AVAILABLE_MEMBER_SUMMARY_NOT_JOINT_GENESET_RESPONSE",
    "missing_predictions":"NATIVE_SPARSE_FACTORS_RESOLVE_TYPED_ABSENCE_NOT_ZERO",
    "attribution_coverage":"ONLY_EXPLICIT_NATIVE_ATTRIBUTION_QUERIES",
}


def write_json(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def implementation_hashes():
    root = Path(__file__).resolve().parent
    names = ["current_drug_sources.py","current_drug_context.py","drug_sparse_query.py","drug_attribution.py",
        "drug_response_analysis.py","drug_dose_response.py","drug_pharmacology.py","drug_pharmacology_native.py",
        "drug_training.py","drug_training_streaming.py","integrated_model.py"]
    return {name:sha256(root/name) for name in names}


def _emit_tables(context, output):
    """Use bounded SQL joins/aggregates; keep native sparse factors separate."""
    selected = context["selected"]; manifest = context["manifest"]; outputs = {}
    selected.download("geneset_members")
    relation = _parquet_relation(selected.paths["geneset_members"])
    with _bounded_duckdb() as con:
        con.execute(f"CREATE VIEW members AS SELECT {','.join(MEMBERS)} FROM {relation}")
        con.execute("CREATE VIEW selected_keys AS SELECT DISTINCT cancer_id,lncrna_id FROM members")
        con.execute("""CREATE VIEW totals AS SELECT geneset_id,count(*) AS member_count,
            sum(selected_score) AS total_score_weight FROM members GROUP BY geneset_id""")

        def emit(name, query):
            target = output/(name+".parquet")
            common = f"""{literal(selected.binding_sha256)} AS selected_analysis_sha256,
                {literal(selected.binding['selected_score_run_id'])} AS selected_score_run_id,
                {literal(manifest['drug_run_id'])} AS drug_training_run_id,
                false AS changes_primary_ranking, false AS tcga_patient_response_claimed"""
            con.execute(f"COPY (SELECT t.*, {common} FROM ({query}) t) TO {literal(str(target))} (FORMAT PARQUET,COMPRESSION ZSTD)")
            count = con.execute(f"SELECT count(*) FROM {_parquet_relation(target)}").fetchone()[0]
            outputs[name] = {"path":str(target),"sha256":sha256(target),"rows":int(count),"public_download_eligible":True}
            con.execute(f"CREATE VIEW \"{name}\" AS {query}")

        emit("selected_members", "SELECT * FROM members")
        probability = PROBABILITY_COLUMN
        pred = _parquet_relation(context["sparse"].paths["available_predictions"])
        emit("selected_drug_predictions", f"""SELECT p.cancer_id,p.lncrna_id,p.drug_id,p.{probability},
            p.availability,p.prediction_fold_mask,p.cell_line_folds_with_prediction,
            {literal(SEMANTICS['model_probability'])} AS prediction_semantics
            FROM {pred} p SEMI JOIN selected_keys s USING(cancer_id,lncrna_id)""")
        groups = ','.join('m.'+k if k!='drug_id' else 'p.drug_id' for k in PROFILE_KEYS)
        emit("selected_drug_prediction_profiles", f"""WITH stats AS (SELECT {groups},
            count(*) AS available_members, sum(m.selected_score) AS available_score_weight,
            sum(m.selected_score*p.{probability})/nullif(sum(m.selected_score),0) AS mean_association_strength_probability
            FROM members m JOIN selected_drug_predictions p USING(cancer_id,lncrna_id) GROUP BY ALL)
            SELECT stats.*,t.member_count,t.total_score_weight, available_members::DOUBLE/t.member_count AS member_coverage,
            available_score_weight/nullif(t.total_score_weight,0) AS selected_weight_coverage,
            available_score_weight>0 AS summary_available,
            CASE WHEN available_score_weight=0 THEN 'ZERO_AVAILABLE_SELECTED_SCORE_WEIGHT' END AS unavailable_reason,
            {literal(SEMANTICS['geneset_profiles'])} AS summary_semantics FROM stats JOIN totals t USING(geneset_id)""")

        for group,paths in context["scan_tables"].items():
            source = _parquet_file_relation(paths)
            if group.endswith("coverage"):
                emit(group, "SELECT * FROM "+source)
                continue
            name = "selected_"+group
            emit(name,f"SELECT p.* FROM {source} p SEMI JOIN selected_keys s USING(cancer_id,lncrna_id)")
            condition = ["subtype_system","subtype_id"]
            if group != "associations":
                condition += ["response_metric","response_unit","exposure_hours_key"]
                condition += ["dose_molar_key"] if group=="dose_associations" else ["low_dose_molar_key","high_dose_molar_key"]
            selected_group = ','.join([groups,*('p.'+key for key in condition)])
            # Correlation magnitudes and p-values are not pooled into a putative
            # GeneSet effect. The directions retain their native condition.
            emit(name+"_direction_profiles",f"""WITH stats AS (SELECT {selected_group},
                count(*) FILTER(WHERE p.association_available) AS available_members,
                coalesce(sum(m.selected_score) FILTER(WHERE p.association_available),0) AS available_score_weight,
                coalesce(sum(m.selected_score) FILTER(WHERE p.association_available AND p.signed_expression_sensitivity_rho>0),0) AS positive_weight,
                coalesce(sum(m.selected_score) FILTER(WHERE p.association_available AND p.signed_expression_sensitivity_rho<0),0) AS negative_weight,
                coalesce(sum(m.selected_score) FILTER(WHERE p.association_available AND p.signed_expression_sensitivity_rho=0),0) AS neutral_weight
                FROM members m JOIN {name} p USING(cancer_id,lncrna_id) GROUP BY ALL)
                SELECT stats.*,t.member_count,t.total_score_weight,
                available_members::DOUBLE/t.member_count AS member_coverage,
                available_score_weight/nullif(t.total_score_weight,0) AS selected_weight_coverage,
                positive_weight/nullif(available_score_weight,0) AS positive_available_weight_fraction,
                negative_weight/nullif(available_score_weight,0) AS negative_available_weight_fraction,
                neutral_weight/nullif(available_score_weight,0) AS neutral_available_weight_fraction,
                available_score_weight>0 AS summary_available,
                CASE WHEN available_members=0 THEN 'NO_AVAILABLE_MEMBER_ASSOCIATION'
                     WHEN available_score_weight=0 THEN 'ZERO_AVAILABLE_SELECTED_SCORE_WEIGHT' END AS unavailable_reason,
                false AS joint_geneset_inference_performed,{literal(SEMANTICS['geneset_profiles'])} AS summary_semantics
                FROM stats JOIN totals t USING(geneset_id)""")

        for group,paths in context["attribution_tables"].items():
            emit("selected_"+group,f"SELECT p.* FROM {_parquet_file_relation(paths)} p SEMI JOIN selected_keys s USING(cancer_id,lncrna_id)")
        if "attribution" in context["attribution_tables"]:
            # TARGET rows stay drug/lncRNA-specific. Only native exact-pathway
            # removal experiments can be joined to a selected pathway's set.
            emit("selected_pathway_attribution_profiles",f"""WITH effects AS (
                SELECT cancer_id,lncrna_id,drug_id,ablation_entity_id AS pathway_id,
                avg(delta_probability) AS mean_fold_probability_change,
                count(DISTINCT cell_line_fold) AS folds_available
                FROM selected_attribution WHERE ablation_kind='EXACT_PATHWAY_TARGET_SET' GROUP BY ALL),
                stats AS (SELECT {groups},count(*) AS attributed_members,
                    sum(m.selected_score) AS available_score_weight,
                    sum(m.selected_score*p.mean_fold_probability_change)/nullif(sum(m.selected_score),0) AS mean_member_probability_change,
                    min(p.folds_available) AS minimum_member_folds, max(p.folds_available) AS maximum_member_folds
                    FROM members m JOIN effects p USING(cancer_id,lncrna_id,pathway_id) GROUP BY ALL)
                SELECT stats.*,t.member_count,t.total_score_weight,
                    attributed_members::DOUBLE/t.member_count AS member_coverage,
                    available_score_weight/nullif(t.total_score_weight,0) AS selected_weight_coverage,
                    available_score_weight>0 AS summary_available,
                    false AS additive_attribution_claimed,false AS causal_claimed,
                    {literal(SEMANTICS['ablation'])} AS attribution_semantics,
                    {literal(SEMANTICS['geneset_profiles'])} AS summary_semantics
                FROM stats JOIN totals t USING(geneset_id)""")
        pharma = context["pharmacology"]
        if pharma is not None:
            # Chemical metadata and target annotations contain no cell IDs or
            # response measurements. Fingerprint bits remain model covariates.
            static = pharma.static_table()
            static = static[[c for c in static if not c.startswith("morgan_r2_chiral_bit_")]].copy()
            static = static.rename(columns={c:c.replace("_exact_p_","_exact_negative_log10_molar_") for c in static if "_exact_p_" in c})
            targets = pharma.targets.rename(columns={"p_value_molar":"negative_log10_molar_activity", "p_relation":"negative_log10_molar_relation"})
            con.register("pharma_static",static); con.register("pharma_targets",targets)
            emit("drug_pharmacology", "SELECT * FROM pharma_static")
            emit("drug_target_annotations", "SELECT * FROM pharma_targets")
    return outputs


def materialize_current_drug_context(manifest_path, output):
    if socket.gethostname() != "149": raise RuntimeError("Current Drug context requires target host 149")
    path = Path(manifest_path).resolve(); output = Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    write_json(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"input_manifest_sha256":sha256(path)})
    context = current_drug_sources(path); replay_native_reports(context)
    outputs = _emit_tables(context,output)
    context["sources"].verify()
    if context["pharmacology"] is not None: context["pharmacology"].verify_sources()
    binding = {"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
        "input_manifest":{"path":str(path),"sha256":sha256(path)},
        "selected_analysis":context["manifest"]["selected_analysis"],"core_run_id":context["manifest"]["core_run_id"],
        "drug_run_id":context["manifest"]["drug_run_id"],"selected_score_run_id":context["selected"].binding["selected_score_run_id"],
        "sparse_query_manifest":context["manifest"]["sparse_query_manifest"],"outputs":outputs,
        "source_artifacts":list(context["sources"].records.values()),"semantics":SEMANTICS,
        "native_response_statistics_replayed":True,"native_attributions_replayed":True,
        "all_public_predictions_replayed":False,"context_changes_ranking":False,"production_deployed":False,
        "dose_analysis_available":bool(context["scan"].get("dose_response_analysis")),
        "pharmacology_features_consumed":context["pharmacology"] is not None,
        "implementation_sha256":implementation_hashes()}
    target = output/"CURRENT_DRUG_CONTEXT_BINDING.json"; write_json(target,binding)
    # Independently replay aggregation SQL from the bound upstream outputs;
    # native statistical/frozen-forward replay already ran above.
    audit = _audit_tables(target,sha256(target),context)
    audit_path = output/"RECOMPUTATION_AUDIT.json"; write_json(audit_path,audit)
    result = {"status":"PASS","target_host":"149","training_started":False,"binding_path":str(target),
        "binding_sha256":sha256(target),"audit_sha256":sha256(audit_path)}
    write_json(output/"SUCCESS.json",result)
    return result


def _load(binding_path, expected_sha):
    path = verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd()); binding = read_json(path)
    if (binding.get("format") != FORMAT or binding.get("status") != "PASS" or binding.get("target_host") != "149"
            or binding.get("semantics") != SEMANTICS or binding.get("training_started") is not False
            or binding.get("native_response_statistics_replayed") is not True or binding.get("native_attributions_replayed") is not True
            or binding.get("all_public_predictions_replayed") is not False or binding.get("context_changes_ranking") is not False
            or binding.get("production_deployed") is not False):
        raise ValueError("Invalid current Drug context binding")
    outputs = {name:verified_path(r,path.parent) for name,r in binding["outputs"].items()}
    if any(r.get("public_download_eligible") is not True for r in binding["outputs"].values()):
        raise ValueError("Drug context includes private downloads")
    return path,binding,outputs


def _audit_tables(binding_path, expected_sha, context):
    path,binding,outputs = _load(binding_path,expected_sha)
    if binding["implementation_sha256"] != implementation_hashes(): raise ValueError("Drug context implementation changed")
    for key in ("selected_analysis","core_run_id","drug_run_id","sparse_query_manifest"):
        if binding[key] != context["manifest"][key]: raise ValueError("Drug context source identity changed")
    if binding["source_artifacts"] != list(context["sources"].records.values()): raise ValueError("Drug context source inventory changed")
    with tempfile.TemporaryDirectory(prefix="current-drug-tables-") as temp:
        replayed = _emit_tables(context,Path(temp))
        if set(replayed) != set(outputs): raise ValueError("Drug context output coverage differs")
        for name,record in replayed.items():
            if record["rows"] != binding["outputs"][name]["rows"]: raise ValueError("Drug context output count differs")
            same_relation("SELECT * FROM "+_parquet_relation(outputs[name]), "SELECT * FROM "+_parquet_relation(Path(record["path"])))
    return {"status":"PASS","binding_sha256":expected_sha,"target_host":"149","training_started":False,
        "native_reports_replayed":True,"selected_tables_recomputed":True,"tables":list(outputs)}


def audit_current_drug_context(binding_path, *, expected_sha):
    if socket.gethostname() != "149": raise RuntimeError("Drug replay audit requires target host 149")
    path,binding,_ = _load(binding_path,expected_sha)
    manifest_path = verified_path(binding["input_manifest"],path.parent)
    context = current_drug_sources(manifest_path); replay_native_reports(context)
    return _audit_tables(path,expected_sha,context)


class CurrentDrugContextQuery:
    def __init__(self,binding_path,*,expected_sha,selected_analysis_sha256):
        path,self.binding,self.paths = _load(binding_path,expected_sha)
        if self.binding["selected_analysis"]["sha256"] != selected_analysis_sha256:
            raise ValueError("Drug context belongs to another selected analysis")
        success = read_json(path.parent/"SUCCESS.json"); audit_path = path.parent/"RECOMPUTATION_AUDIT.json"
        audit = read_json(audit_path)
        if (success.get("status") != "PASS" or success.get("binding_sha256") != expected_sha
                or success.get("audit_sha256") != sha256(audit_path) or audit.get("status") != "PASS"
                or audit.get("binding_sha256") != expected_sha or audit.get("native_reports_replayed") is not True):
            raise ValueError("Current Drug context lacks completed recomputation")
        self.sha = expected_sha; self.root = path.parent

    def download(self,name):
        if name not in self.paths: raise ValueError("Unknown public Drug download")
        record = self.binding["outputs"][name]; verified_path(record,self.root)
        return {**record,"binding_sha256":self.sha,"selected_analysis_sha256":self.binding["selected_analysis"]["sha256"]}

    def query(self,*,table="selected_drug_prediction_profiles",filters=None,limit=100,offset=0):
        if table not in self.paths: raise ValueError("Unknown public Drug table")
        if type(limit) is not int or type(offset) is not int or not 1<=limit<=1000 or not 0<=offset<=1000000:
            raise ValueError("Invalid Drug query pagination")
        filters = filters or {}
        allowed = {*KEYS,"pathway_id","geneset_id","association_direction","subtype_system","subtype_id","ablation_kind",
            "ablation_entity_id","gene_id","response_metric","response_unit","exposure_hours_key",
            "dose_molar_key","low_dose_molar_key","high_dose_molar_key"}
        if not set(filters).issubset(allowed) or not all(isinstance(v,str) for v in filters.values()):
            raise ValueError("Drug filters require exact string identities")
        path = self.download(table)["path"]
        with _bounded_duckdb() as con:
            columns = [r[0] for r in con.execute("DESCRIBE SELECT * FROM read_parquet(?)",[path]).fetchall()]
            if not set(filters).issubset(columns): raise ValueError("Drug filter is not a column of this table")
            where = " AND ".join('"'+k+'"=?' for k in filters) or "true"
            # Include all columns to give rows tied on one identity a stable
            # pagination order (e.g. censored target assays and fold effects).
            order = ','.join('"'+c.replace('"','""')+'"' for c in columns)
            data = con.execute(f"SELECT * FROM read_parquet(?) WHERE {where} ORDER BY {order} LIMIT ? OFFSET ?",
                [path,*filters.values(),limit,offset]).df()
        return {"binding_sha256":self.sha,"selected_analysis_sha256":self.binding["selected_analysis"]["sha256"],
            "semantics":SEMANTICS,"rows":data.astype(object).where(data.notna(),None).to_dict("records")}

    def member_predictions(self,*,geneset_id,drug_id,limit=100,offset=0):
        """Bounded on-demand missingness, without storing a dense drug product."""
        if not isinstance(drug_id,str) or not drug_id.strip(): raise ValueError("A drug identity is required")
        result = self.query(table="selected_members",filters={"geneset_id":geneset_id},limit=limit,offset=offset)
        record = self.binding["sparse_query_manifest"]
        # Validate the bound factors for each request; caching an unchecked
        # mutable source directory would allow a later cross-version query.
        spec_path = verified_path(self.binding["input_manifest"],self.root)
        sparse_path = verified_path(record,spec_path.parent)
        sparse = DrugSparseQueryBundle(sparse_path,expected_manifest_sha256=record["sha256"])
        for row in result["rows"]:
            native = sparse.resolve(row["cancer_id"],row["lncrna_id"],drug_id)
            row.update(native)
        return result
