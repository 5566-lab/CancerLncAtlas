"""Current selected GeneSets consume the actual signed donor-OOF SC outputs.

Public summaries retain cell and dataset context, donor counts and spread.
Private donor rows remain bound to the release, not exposed by its downloads.
Preparation replays aggregation and any bound new-workflow frozen models.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket
import tempfile

import pyarrow.parquet as pq

from .current_core_export import sha256, verified_path
from .drug_sparse_query import _bounded_duckdb
from .context_table_replay import relation, compare_tables
from .current_drug_sources import same_relation
from .selected_genesets import SelectedAnalysisQuery
from .single_cell_signed_fusion import load_signed_sc_fusion

INPUT_FORMAT = "CURRENT_SINGLE_CELL_CONTEXT_INPUT_V1"
FORMAT = "CURRENT_SINGLE_CELL_CONTEXT_RELEASE_V1"
KEYS = ["cancer_id", "lncrna_id", "pathway_id"]


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path,value): Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def record(path): return {"path":str(Path(path).resolve()),"sha256":sha256(path)}


def implementation_hashes():
    root=Path(__file__).resolve().parent
    return {name:sha256(root/name) for name in ("current_single_cell_context.py","single_cell_signed_fusion.py","context_table_replay.py")}


def sources(manifest_path,*,replay):
    path=Path(manifest_path).resolve(); manifest=read(path); bound={str(path):record(path)}
    if manifest.get("format")!=INPUT_FORMAT: raise ValueError("Invalid current single-cell context manifest")
    def bind(item,root=path.parent):
        result=verified_path(item,root); bound[str(result)]=record(result); return result
    selected_path=bind(manifest["selected_analysis"])
    selected=SelectedAnalysisQuery(selected_path,expected_sha=manifest["selected_analysis"]["sha256"])
    score_path=bind(selected.binding["selected_score_binding"],selected_path.parent); score=read(score_path)
    primary_path=bind(score["sources"]["reference_exact_producer"]["producer_binding"],score_path.parent)
    primary=read(primary_path); candidates=bind(primary["input_artifacts"][0],primary_path.parent)
    native_path=bind(manifest["native_binding"]); native=read(native_path)
    success_path=native_path.parent/"SUCCESS.json"; success=read(bind(record(success_path)))
    if (native.get("format")!="SC_HHGT_SIGNED_FUSION_BINDING_V1" or native.get("status")!="PASS"
            or native.get("target_host")!="149" or native.get("training_started") is not False
            or native.get("core_dependency")!="INDEPENDENT_MODEL" or native.get("direction_preserved") is not True
            or native.get("raw_rho_treated_as_probability") is not False or native.get("observed_rho_used_as_feature") is not False
            or success.get("status")!="PASS" or success.get("audit_sha256")!=sha256(native_path)):
        raise ValueError("Current single-cell context requires completed signed donor-OOF aggregation")
    inputs={k:bind(v,native_path.parent) for k,v in native["inputs"].items()}
    outputs={k:bind(v,native_path.parent) for k,v in native["outputs"].items()}
    if set(outputs)!={"donor_cell_context","cell_context_summary","single_cell_signed_fusion"}:
        raise ValueError("Unknown native single-cell output inventory")
    try:
        same_relation(f"SELECT {','.join(KEYS)} FROM {relation(candidates)}",
                      f"SELECT {','.join(KEYS)} FROM {relation(inputs['candidates'])}")
    except ValueError as error:raise ValueError("Single-cell candidate universe differs from selected analysis") from error
    native,inputs,outputs,dependencies=load_signed_sc_fusion(native_path,expected_sha=sha256(native_path),replay=replay)
    for item in dependencies:bind(item,native_path.parent)
    # If SC actually contributes to the selected fusion, the webpage cannot
    # show another SC run with an otherwise identical candidate universe.
    chosen_path=bind(score["sources"]["selected_producer"]["producer_binding"],score_path.parent)
    chosen=read(chosen_path); used=False
    if chosen.get("format")=="CC_HHGT_V3_2_MULTIMODAL_FUSION_BINDING_V1":
        private=bind(chosen["artifacts"]["fusion_input_private"],chosen_path.parent)
        names=pq.ParquetFile(private).schema_arrow.names
        used=any(name+"_probability" in names for name in ("single_cell_positive","single_cell_negative"))
        if used and sha256(native_path) not in {r["sha256"] for r in chosen["source_inputs"]}:
            raise ValueError("Single-cell context differs from the actual selected fusion input")
    for item in selected.binding["outputs"].values():bind(item,selected_path.parent)
    return {"manifest":manifest,"selected":selected,"native":native,"paths":outputs,"sources":bound,
            "used_in_selected_score":used}


def context_tables(context,output):
    """Derive public context views using bounded SQL and complete denominators."""
    native=context["native"]; selected=context["selected"]; output=Path(output)
    minimum=int(native["min_donors"]); paths=context["paths"]
    with _bounded_duckdb() as con:
        for name,path in paths.items():con.execute(f"CREATE VIEW {name} AS SELECT * FROM {relation(path)}")
        con.execute(f"CREATE VIEW members AS SELECT * FROM {relation(selected.paths['geneset_members'])}")
        con.execute("""CREATE VIEW dataset_context AS SELECT cancer_id,lncrna_id,pathway_id,dataset_id,cell_key,
            count(*) AS donors,avg(predicted_rho) AS signed_rho_mean,avg(abs(predicted_rho)) AS absolute_rho_mean,
            stddev_samp(predicted_rho) AS donor_rho_sd,avg(cast(predicted_rho>0 AS DOUBLE)) AS positive_donor_fraction,
            count(DISTINCT outer_fold_id) AS donor_folds FROM (
                SELECT cancer_id,lncrna_id,pathway_id,dataset_id,cell_key,canonical_donor_id,
                    avg(predicted_rho) AS predicted_rho,min(outer_fold_id) AS outer_fold_id
                FROM donor_cell_context GROUP BY ALL) GROUP BY ALL""")
        queries={"native_pooled_support":"SELECT * FROM single_cell_signed_fusion"}
        for name,view in (("native_cell_context","cell_context_summary"),("native_dataset_context","dataset_context")):
            queries[name]=f"""SELECT *,donors>={minimum} AS context_available,
                CASE WHEN donors<{minimum} THEN 'INSUFFICIENT_INDEPENDENT_DONORS' END AS context_unavailable_reason
                FROM {view}"""
        queries["selected_member_support"]="SELECT m.*,s.* EXCLUDE(cancer_id,lncrna_id,pathway_id,analysis_version) FROM members m JOIN single_cell_signed_fusion s USING(cancer_id,lncrna_id,pathway_id)"
        queries["selected_member_context"]=f"""SELECT m.*,c.* EXCLUDE(cancer_id,lncrna_id,pathway_id),
            coalesce(c.donors>={minimum},false) AS context_available,
            CASE WHEN c.donors IS NULL THEN 'NO_DONOR_OOF_ROWS' WHEN c.donors<{minimum}
                THEN 'INSUFFICIENT_INDEPENDENT_DONORS' END AS context_unavailable_reason
            FROM members m LEFT JOIN cell_context_summary c USING(cancer_id,lncrna_id,pathway_id)"""
        con.execute("CREATE VIEW member_context AS "+queries["selected_member_context"])
        queries["selected_geneset_context"]=f"""WITH totals AS (
            SELECT geneset_id,count(*) AS selected_members,sum(selected_score) AS selected_score_weight FROM members GROUP BY geneset_id),
            summaries AS (SELECT geneset_id,cancer_id,pathway_id,cell_key,
                count(*) FILTER(WHERE cell_key IS NOT NULL) AS observed_context_members,
                count(*) FILTER(WHERE context_available) AS available_members,
                coalesce(sum(selected_score) FILTER(WHERE context_available),0) AS available_score_weight,
                sum(selected_score*signed_rho_mean) FILTER(WHERE context_available)
                    /nullif(sum(selected_score) FILTER(WHERE context_available),0) AS mean_member_signed_rho,
                sum(selected_score*positive_donor_fraction) FILTER(WHERE context_available)
                    /nullif(sum(selected_score) FILTER(WHERE context_available),0) AS mean_member_positive_donor_fraction
                FROM member_context GROUP BY ALL)
            SELECT s.*,t.selected_members,s.available_members::DOUBLE/t.selected_members AS member_coverage,
                s.available_score_weight/nullif(t.selected_score_weight,0) AS selected_weight_coverage,
                s.available_score_weight>0 AS summary_available,
                CASE WHEN cell_key IS NULL THEN 'NO_DONOR_OOF_ROWS' WHEN s.available_score_weight<=0 THEN 'NO_SUFFICIENT_DONOR_MEMBER' END AS unavailable_reason
                FROM summaries s JOIN totals t USING(geneset_id)
                WHERE cell_key IS NOT NULL OR NOT EXISTS (SELECT 1 FROM summaries q WHERE q.geneset_id=s.geneset_id AND q.cell_key IS NOT NULL)"""
        results={}
        for name,query in queries.items():
            destination=output/(name+".parquet")
            # Source identities are metadata columns, not interpolation of user filters.
            digest=selected.binding_sha256; run=native["run_id"].replace("'","''")
            annotated=f"""SELECT q.*,'{digest}' AS context_selected_analysis_sha256,'{run}' AS single_cell_run_id,
                false AS joint_geneset_effect_estimated,false AS donor_sd_is_confidence_interval,
                {str(context['used_in_selected_score']).lower()} AS used_in_selected_score FROM ({query}) q"""
            con.execute("COPY ("+annotated+") TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",[str(destination)])
            results[name]={**record(destination),"rows":pq.ParquetFile(destination).metadata.num_rows}
    return results


def materialize_current_single_cell_context(manifest_path,output):
    if socket.gethostname()!="149":raise RuntimeError("Single-cell context preparation requires target host 149")
    path=Path(manifest_path).resolve(); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"input_manifest":record(path)})
    context=sources(path,replay=True); outputs=context_tables(context,output)
    for item in context["sources"].values():verified_path(item,path.parent)
    binding={"format":FORMAT,"status":"PASS","target_host":"149","input_manifest":record(path),
        "selected_analysis":context["manifest"]["selected_analysis"],"native_binding":context["manifest"]["native_binding"],
        "single_cell_run_id":context["native"]["run_id"],"min_donors":context["native"]["min_donors"],
        "source_artifacts":list(context["sources"].values()),"outputs":outputs,"implementation_sha256":implementation_hashes(),
        "native_aggregation_replayed":True,"native_model_forwards_replayed":context["native"].get("native_numeric_replay_bound",False),
        "used_in_selected_score":context["used_in_selected_score"],"joint_geneset_effect_estimated":False,
        "donor_sd_is_confidence_interval":False,"donor_identifiers_in_public_tables":False,
        "training_started":False,"production_deployed":False}
    bp=output/"CURRENT_SINGLE_CELL_CONTEXT_BINDING.json";write(bp,binding)
    audit=audit_current_single_cell_context(bp,expected_sha=sha256(bp),replay_native=False)
    ap=output/"RECOMPUTATION_AUDIT.json";write(ap,audit)
    result={"status":"PASS","target_host":"149","training_started":False,"binding_path":str(bp),
        "binding_sha256":sha256(bp),"audit_sha256":sha256(ap)}
    write(output/"SUCCESS.json",result);return result


def load(binding_path,expected_sha,selected_analysis_sha):
    path=verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd()); binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or binding["selected_analysis"]["sha256"]!=selected_analysis_sha):raise ValueError("Invalid or different selected single-cell context")
    for field in ("joint_geneset_effect_estimated","donor_sd_is_confidence_interval",
                  "donor_identifiers_in_public_tables","training_started","production_deployed"):
        if binding.get(field) is not False:raise ValueError("Invalid single-cell context semantics: "+field)
    if binding.get("native_aggregation_replayed") is not True:raise ValueError("Native single-cell aggregation was not verified")
    native=read(verified_path(binding["native_binding"],path.parent))
    if binding.get("native_model_forwards_replayed") is not native.get("native_numeric_replay_bound",False):
        raise ValueError("Single-cell frozen forward semantics differ from native binding")
    paths={name:verified_path(item,path.parent) for name,item in binding["outputs"].items()}
    return path,binding,paths


def audit_current_single_cell_context(binding_path,*,expected_sha,replay_native=True):
    if socket.gethostname()!="149":raise RuntimeError("Single-cell context audit requires target host 149")
    original=read(verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd()))
    path,binding,paths=load(binding_path,expected_sha,original["selected_analysis"]["sha256"])
    if binding["implementation_sha256"]!=implementation_hashes():raise ValueError("Single-cell context implementation changed")
    context=sources(verified_path(binding["input_manifest"],path.parent),replay=replay_native)
    for key in ("selected_analysis","native_binding"):
        if binding[key]!=context["manifest"][key]:raise ValueError("Single-cell context source identity changed")
    if (binding["source_artifacts"]!=list(context["sources"].values()) or binding["single_cell_run_id"]!=context["native"]["run_id"]
            or binding["min_donors"]!=context["native"]["min_donors"] or binding["used_in_selected_score"]!=context["used_in_selected_score"]):
        raise ValueError("Single-cell context provenance or settings changed")
    with tempfile.TemporaryDirectory(prefix="single_cell_public_replay_") as tmp:
        expected=context_tables(context,Path(tmp))
        if set(expected)!=set(paths):raise ValueError("Single-cell public inventory changed")
        for name,item in expected.items():
            keys=KEYS.copy()
            if "geneset" in name:keys=["geneset_id","cell_key"]
            elif "member" in name:keys+=["geneset_id"]
            if "context" in name and "cell_key" not in keys:keys+=["cell_key"]
            if "dataset" in name:keys+=["dataset_id"]
            compare_tables(paths[name],item["path"],keys)
            if item["rows"]!=binding["outputs"][name]["rows"]:raise ValueError("Single-cell context row count changed")
    return {"status":"PASS","binding_sha256":expected_sha,"native_aggregation_replayed_in_this_audit":replay_native,
        "native_model_forwards_replayed_in_this_audit":replay_native and binding["native_model_forwards_replayed"],
        "public_tables_recomputed":True,"training_started":False,"independent_biological_validation":False}


class CurrentSingleCellContextQuery:
    def __init__(self,binding_path,*,expected_sha,selected_analysis_sha):
        self.path,self.binding,self.paths=load(binding_path,expected_sha,selected_analysis_sha)
        audit_path=self.path.parent/"RECOMPUTATION_AUDIT.json";audit=read(audit_path);success=read(self.path.parent/"SUCCESS.json")
        if (success.get("status")!="PASS" or success.get("binding_sha256")!=expected_sha
                or success.get("audit_sha256")!=sha256(audit_path) or audit.get("status")!="PASS"
                or audit.get("binding_sha256")!=expected_sha or audit.get("public_tables_recomputed") is not True):
            raise ValueError("Incomplete current single-cell context")

    def download(self,name):
        if name not in self.paths:raise ValueError("Unknown public single-cell table")
        item=self.binding["outputs"][name];verified_path(item,self.path.parent);return item
