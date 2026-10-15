"""Publish native patient-subgroup statistics for the same selected candidates.

HC3 intervals describe full-cohort associations; OOF prediction errors are
separate. The selected view retains native family-wide FDR and never pools
member slopes into a fabricated joint GeneSet or individual causal effect.
"""
from __future__ import annotations
import json
from pathlib import Path
import re
import socket
import tempfile

import pyarrow.parquet as pq

from .current_core_export import sha256,verified_path
from .context_table_replay import compare_tables,relation
from .current_drug_sources import same_relation
from .drug_sparse_query import _bounded_duckdb
from .selected_genesets import SelectedAnalysisQuery

INPUT_FORMAT="CURRENT_PATIENT_CONTEXT_INPUT_V1"
FORMAT="CURRENT_PATIENT_CONTEXT_RELEASE_V1"
KEYS=["cancer_id","lncrna_id","pathway_id"]


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def record(path):return {"path":str(Path(path).resolve()),"sha256":sha256(path)}


def implementation_hashes():
    root=Path(__file__).resolve().parents[2]
    return {name:sha256(root/name) for name in ("cc_hhgt/v32/current_patient_context.py","cc_hhgt/v32/context_table_replay.py",
        "cc_hhgt/v32/patient_heterogeneity.py","scripts/analyze_v32_patient_heterogeneity.py")}


def verify_native(audit_path,*,replay,bound):
    from scripts.analyze_v32_patient_heterogeneity import _source_record,main
    native=read(audit_path);success_path=audit_path.parent/"SUCCESS.json";success=read(success_path)
    bound[str(success_path)]=record(success_path)
    if (native.get("format")!="PATIENT_SUBGROUP_HETEROGENEITY_V1" or native.get("status")!="PASS"
            or native.get("target_host")!="149" or native.get("training_started") is not False
            or native.get("statistical_analysis_executed") is not True or native.get("neural_optimizer_started") is not False
            or native.get("heldout_outcomes_used_for_own_fit") is not False or native.get("missing_mutation_assumed_wildtype") is not False
            or native.get("causal_claimed") is not False or native.get("external_validation_claimed") is not False
            or success.get("status")!="PASS" or success.get("audit_sha256")!=sha256(audit_path)):
        raise ValueError("Patient context requires completed native subgroup analysis")
    policy=native.get("group_source_policy",{})
    if policy.get("kind") not in {"MUTATION_CALLS","DECLARED_GROUPS"}:
        raise ValueError("Native patient analysis lacks its actual group-source policy; rematerialize from sources")
    implementations=implementation_hashes()
    expected_code={k:implementations[k] for k in ("cc_hhgt/v32/patient_heterogeneity.py","scripts/analyze_v32_patient_heterogeneity.py")}
    if native.get("implementation_sha256")!=expected_code:raise ValueError("Patient native replay requires its recorded implementation")
    if set(native["sources"])!={"expression","activity","groups","covariates","folds","candidates"}:
        raise ValueError("Patient native source inventory differs")
    if set(native["outputs"])!={"INTERACTION_SUMMARY.parquet","SUBGROUP_OOF_METRICS.parquet","FIT_AUDIT.jsonl","patient_oof"}:
        raise ValueError("Patient native output inventory differs")
    for item in [*native["sources"].values(),*native["outputs"].values()]:
        source=Path(item["path"])
        if "sha256" in item:verified_path(item,audit_path.parent)
        elif _source_record(source)!=item:raise ValueError("Patient source dataset changed")
        bound[str(source)]=item
    if replay:
        args=[]
        for name,item in native["sources"].items():args += ["--"+name,item["path"]]
        settings=native["config"]
        args += ["--group-id",native["group_id"],"--candidate-selection",native["candidate_selection"],
            "--nuisance-columns",",".join(settings["covariates"])]
        for field in ("min_train_patients","min_group_patients","confidence","ridge_alpha"):
            args += ["--"+field.replace("_","-"),str(settings[field])]
        if policy["kind"]=="MUTATION_CALLS":args += ["--groups-are-mutation-calls"]
        for name in ("mutation_count_column","mutation_assay_column","locus_callable_column"):
            args += ["--"+name.replace("_","-"),policy[name]]
        with tempfile.TemporaryDirectory(prefix="patient_context_native_replay_") as tmp:
            output=Path(tmp)/"native"
            if main(args+["--output",str(output)])!=0:raise ValueError("Patient native replay failed")
            expected=read(output/"AUDIT.json")
            for name,value in expected.items():
                if name not in {"outputs","patient_predictions_root"} and native.get(name)!=value:
                    raise ValueError("Patient native policy or provenance differs: "+name)
            for name,keys in (("INTERACTION_SUMMARY.parquet",KEYS+["group_id"]),
                               ("SUBGROUP_OOF_METRICS.parquet",KEYS+["group_id","group_value"])):
                compare_tables(native["outputs"][name]["path"],output/name,keys,label="Patient")
            if sha256(output/"FIT_AUDIT.jsonl")!=native["outputs"]["FIT_AUDIT.jsonl"]["sha256"]:
                raise ValueError("Patient OOF fit lineage differs from source replay")
            original=native["outputs"]["patient_oof"];actual=expected["outputs"]["patient_oof"]
            if [v["path"] for v in original["files"]]!=[v["path"] for v in actual["files"]]:
                raise ValueError("Patient OOF partition inventory differs")
            for item in original["files"]:
                compare_tables(Path(original["path"])/item["path"],output/"patient_oof"/item["path"],
                    ["patient_id","lncrna_id","pathway_id","group_id"],label="Patient OOF")
    return native


def sources(manifest_path,*,replay):
    path=Path(manifest_path).resolve();manifest=read(path);bound={str(path):record(path)}
    items=manifest.get("analyses")
    if manifest.get("format")!=INPUT_FORMAT or not isinstance(items,list) or not items:
        raise ValueError("Patient context needs an explicit nonempty native analysis inventory")
    def bind(item,root=path.parent):
        result=verified_path(item,root);bound[str(result)]=record(result);return result
    selected_path=bind(manifest["selected_analysis"])
    selected=SelectedAnalysisQuery(selected_path,expected_sha=manifest["selected_analysis"]["sha256"])
    score_path=bind(selected.binding["selected_score_binding"],selected_path.parent);score=read(score_path)
    primary_path=bind(score["sources"]["reference_exact_producer"]["producer_binding"],score_path.parent)
    candidates=bind(read(primary_path)["input_artifacts"][0],primary_path.parent)
    native={}
    for item in items:
        identity=item.get("source_id")
        if not isinstance(identity,str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}",identity) or identity in native:
            raise ValueError("Patient analysis sources need unique stable identifiers")
        source=bind(item["native_binding"]);report=verify_native(source,replay=replay,bound=bound)
        try:
            same_relation(f"SELECT {','.join(KEYS)} FROM {relation(candidates)}",
                f"SELECT {','.join(KEYS)} FROM {relation(report['sources']['candidates']['path'])}")
        except ValueError as error:raise ValueError("Patient analysis candidate universe differs from selected analysis") from error
        native[identity]=report
    for item in selected.binding["outputs"].values():bind(item,selected_path.parent)
    return {"manifest":manifest,"selected":selected,"native":native,"sources":bound}


def context_tables(context,output):
    selected=context["selected"];output=Path(output);tables={}
    with _bounded_duckdb() as con:
        for table,file in (("native_interactions","INTERACTION_SUMMARY.parquet"),("native_oof_metrics","SUBGROUP_OOF_METRICS.parquet")):
            parts=[f"SELECT *, '{identity}' AS analysis_source_id FROM {relation(native['outputs'][file]['path'])}"
                for identity,native in context["native"].items()]
            con.execute(f"CREATE VIEW {table} AS "+" UNION ALL BY NAME ".join(parts))
        con.execute(f"CREATE VIEW members AS SELECT * FROM {relation(selected.paths['geneset_members'])}")
        queries={name:f"SELECT * FROM {name}" for name in ("native_interactions","native_oof_metrics")}
        queries["selected_member_interactions"]="SELECT m.*,n.* EXCLUDE(cancer_id,lncrna_id,pathway_id) FROM members m JOIN native_interactions n USING(cancer_id,lncrna_id,pathway_id)"
        queries["selected_member_oof_metrics"]="SELECT m.*,n.* EXCLUDE(cancer_id,lncrna_id,pathway_id) FROM members m JOIN native_oof_metrics n USING(cancer_id,lncrna_id,pathway_id)"
        # Counts/coverage are legitimate GeneSet summaries. A mean of member
        # slopes is not an estimated joint GeneSet interaction and is not made.
        queries["selected_geneset_coverage"]="""SELECT m.geneset_id,m.cancer_id,m.pathway_id,n.group_id,n.analysis_source_id,
            count(*) AS selected_members,count(*) FILTER(WHERE n.inference_available) AS inference_available_members,
            count(*) FILTER(WHERE n.inference_available)::DOUBLE/count(*) AS inference_member_coverage,
            coalesce(sum(m.selected_score) FILTER(WHERE n.inference_available),0)/nullif(sum(m.selected_score),0) AS inference_weight_coverage,
            count(*) FILTER(WHERE n.inference_available AND n.group0_slope*n.group1_slope<0) AS opposite_group_direction_members,
            count(*) FILTER(WHERE n.inference_available AND n.interaction_fdr<=0.05) AS native_fdr_05_supported_members,
            count(*) FILTER(WHERE n.inference_available)>0 AS summary_available,
            CASE WHEN count(*) FILTER(WHERE n.inference_available)=0 THEN 'NO_AVAILABLE_MEMBER_INTERACTION' END AS unavailable_reason
            FROM members m JOIN native_interactions n USING(cancer_id,lncrna_id,pathway_id) GROUP BY ALL"""
        for name,query in queries.items():
            target=output/(name+".parquet")
            annotated=f"""SELECT q.*,'{selected.binding_sha256}' AS context_selected_analysis_sha256,
                false AS joint_geneset_interaction_estimated,false AS context_changes_selected_ranking,
                'DESCRIPTIVE_FULL_COHORT_HC3_AND_SEPARATE_PATIENT_OOF_METRICS' AS interpretation FROM ({query}) q"""
            con.execute("COPY ("+annotated+") TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",[str(target)])
            tables[name]={**record(target),"rows":pq.ParquetFile(target).metadata.num_rows}
    return tables


def verify_sources(bound):
    from scripts.analyze_v32_patient_heterogeneity import _source_record
    for item in bound:
        if "sha256" in item:verified_path(item,Path.cwd())
        elif _source_record(Path(item["path"]))!=item:raise ValueError("Patient source dataset changed")


def materialize_current_patient_context(manifest_path,output):
    if socket.gethostname()!="149":raise RuntimeError("Patient context preparation requires target host 149")
    path=Path(manifest_path).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"input_manifest":record(path)})
    context=sources(path,replay=True);outputs=context_tables(context,output);verify_sources(context["sources"].values())
    binding={"format":FORMAT,"status":"PASS","target_host":"149","input_manifest":record(path),
        "selected_analysis":context["manifest"]["selected_analysis"],"analyses":context["manifest"]["analyses"],
        "source_artifacts":list(context["sources"].values()),"outputs":outputs,"implementation_sha256":implementation_hashes(),
        "native_statistics_and_patient_oof_replayed":True,"joint_geneset_interaction_estimated":False,
        "context_changes_selected_ranking":False,"patient_identifiers_in_public_tables":False,
        "independent_external_validation":False,"training_started":False,"production_deployed":False}
    bp=output/"CURRENT_PATIENT_CONTEXT_BINDING.json";write(bp,binding)
    ap=output/"RECOMPUTATION_AUDIT.json";write(ap,audit_current_patient_context(bp,expected_sha=sha256(bp),replay_native=False))
    result={"status":"PASS","target_host":"149","training_started":False,"binding_path":str(bp),"binding_sha256":sha256(bp),"audit_sha256":sha256(ap)}
    write(output/"SUCCESS.json",result);return result


def load(binding_path,expected_sha,selected_analysis_sha):
    path=verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd());binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or binding["selected_analysis"]["sha256"]!=selected_analysis_sha):raise ValueError("Invalid or different selected patient context")
    for field in ("joint_geneset_interaction_estimated","context_changes_selected_ranking","patient_identifiers_in_public_tables",
                  "independent_external_validation","training_started","production_deployed"):
        if binding.get(field) is not False:raise ValueError("Invalid patient context semantics: "+field)
    if binding.get("native_statistics_and_patient_oof_replayed") is not True:raise ValueError("Patient native replay missing")
    paths={name:verified_path(item,path.parent) for name,item in binding["outputs"].items()}
    return path,binding,paths


def audit_current_patient_context(binding_path,*,expected_sha,replay_native=True):
    if socket.gethostname()!="149":raise RuntimeError("Patient context audit requires target host 149")
    original=read(verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd()))
    path,binding,paths=load(binding_path,expected_sha,original["selected_analysis"]["sha256"])
    if binding["implementation_sha256"]!=implementation_hashes():raise ValueError("Patient context implementation changed")
    context=sources(verified_path(binding["input_manifest"],path.parent),replay=replay_native)
    if (binding["selected_analysis"]!=context["manifest"]["selected_analysis"] or binding["analyses"]!=context["manifest"]["analyses"]
            or binding["source_artifacts"]!=list(context["sources"].values())):raise ValueError("Patient context provenance changed")
    with tempfile.TemporaryDirectory(prefix="patient_context_public_replay_") as tmp:
        expected=context_tables(context,Path(tmp))
        if set(expected)!=set(paths):raise ValueError("Patient public inventory changed")
        for name,item in expected.items():
            keys=KEYS+["group_id","analysis_source_id"]
            if "geneset" in name:keys=["geneset_id","group_id","analysis_source_id"]
            elif "member" in name:keys+=["geneset_id"]
            if "metrics" in name:keys+=["group_value"]
            compare_tables(paths[name],item["path"],keys,label="Patient public")
            if item["rows"]!=binding["outputs"][name]["rows"]:raise ValueError("Patient context row count changed")
    return {"status":"PASS","binding_sha256":expected_sha,"native_replayed_in_this_audit":replay_native,
        "public_tables_recomputed":True,"training_started":False}


class CurrentPatientContextQuery:
    def __init__(self,binding_path,*,expected_sha,selected_analysis_sha):
        self.path,self.binding,self.paths=load(binding_path,expected_sha,selected_analysis_sha)
        ap=self.path.parent/"RECOMPUTATION_AUDIT.json";audit=read(ap);success=read(self.path.parent/"SUCCESS.json")
        if (success.get("status")!="PASS" or success.get("binding_sha256")!=expected_sha or success.get("audit_sha256")!=sha256(ap)
                or audit.get("status")!="PASS" or audit.get("binding_sha256")!=expected_sha or audit.get("public_tables_recomputed") is not True):
            raise ValueError("Incomplete current patient context")

    def download(self,name):
        if name not in self.paths:raise ValueError("Unknown public patient context table")
        item=self.binding["outputs"][name];verified_path(item,self.path.parent);return item
