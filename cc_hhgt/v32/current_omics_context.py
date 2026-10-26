"""Selected public genomic/distal contexts from actual native producers.

Genomic publication checks native heads/folds and replays aggregation, not
model forwards. Distal publication replays the original nested statistical
command. Neither context fits a neural model or changes selected ranking.
"""
from __future__ import annotations
import json
from pathlib import Path
import socket
import tempfile

import pandas as pd
import pyarrow.parquet as pq

from .current_core_export import sha256,verified_path,validate_current_export
from .context_table_replay import compare_tables,relation
from .current_drug_sources import same_relation
from .drug_sparse_query import _bounded_duckdb
from .native_auxiliary_fusion import load_native_auxiliary,SEMANTICS
from .selected_genesets import SelectedAnalysisQuery

INPUT_FORMAT="CURRENT_OMICS_CONTEXT_INPUT_V1"
FORMAT="CURRENT_OMICS_CONTEXT_RELEASE_V1"
KEYS=["cancer_id","lncrna_id","pathway_id"]
KINDS=("genomic","cnv_event","cnv","atac")
FEATURES=("distal_mutation_burden","atac_distal_accessibility","promoter_methylation_beta",
          "distal_methylation_beta","local_cnv_log2","tumor_purity")


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
def record(path):return {"path":str(Path(path).resolve()),"sha256":sha256(path)}
def sql_text(value):return "'"+str(value).replace("'","''")+"'"


class Sources:
    def __init__(self):self.records={}
    def bind(self,item,root):
        path=verified_path(item,root);self.records[str(path)]=record(path);return path
    def verify(self):
        for item in self.records.values():verified_path(item,Path.cwd())


def implementations(kind):
    root=Path(__file__).resolve().parents[2]
    names=["cc_hhgt/v32/current_omics_context.py","cc_hhgt/v32/context_table_replay.py"]
    names += (["cc_hhgt/v32/native_auxiliary_fusion.py"] if kind=="genomic" else
        ["scripts/run_v32_distal_regulatory_mutation_head.py","cc_hhgt/v32/epigenetic_expression_head.py",
         "cc_hhgt/v32/distal_regulatory_mutation_head.py","cc_hhgt/v32/distal_topology.py"])
    return {name:sha256(root/name) for name in names}


def same_candidates(left,right):
    try:same_relation(f"SELECT {','.join(KEYS)} FROM {relation(left)}",f"SELECT {','.join(KEYS)} FROM {relation(right)}")
    except ValueError as error:raise ValueError("Omics candidate universe differs from selected analysis") from error


def genomic_sources(context):
    manifest=context["manifest"];root=context["root"];bound=context["sources"];primary=context["primary"]
    specs=manifest.get("native_bindings",{})
    if set(specs)!=set(KINDS):raise ValueError("Genomic context requires every native producer status")
    chosen=context["chosen"];chosen_root=context["chosen_root"];active=set();chosen_hashes=set()
    if chosen.get("format")=="CC_HHGT_V3_2_MULTIMODAL_FUSION_BINDING_V1":
        private=bound.bind(chosen["artifacts"]["fusion_input_private"],chosen_root)
        names=pq.ParquetFile(private).schema_arrow.names
        active={e for e in ("mutation","cnv_event","cnv","atac") if e+"_probability" in names}
        chosen_hashes={r["sha256"] for r in chosen["source_inputs"]}
    native={};catalog=[];covered=set()
    for kind,spec in specs.items():
        if spec.get("status") in {"NOT_PRODUCED","SOURCE_UNAVAILABLE","NOT_APPLICABLE"}:
            if set(spec)!={"status","reason"} or not isinstance(spec["reason"],str) or not spec["reason"].strip():
                raise ValueError("Unavailable genomic producer needs an explicit reason")
            catalog.append({"producer_kind":kind,"status":spec["status"],"reason":spec["reason"],
                "native_run_id":None,"native_estimand":SEMANTICS[kind],"source_binding_sha256":None})
            continue
        if set(spec)!={"status","binding"} or spec.get("status")!="AVAILABLE":raise ValueError("Invalid genomic producer binding")
        path=bound.bind(spec["binding"],root)
        _,audit=load_native_auxiliary(path,expected_sha=spec["binding"]["sha256"],expected_core_run_id=primary["training_run_id"])
        input_path=bound.bind(audit["input_manifest"],path.parent);source=read(input_path)
        if source["producer_kind"]!=kind:raise ValueError("Genomic context relabelled its native producer")
        candidates=bound.bind(source["candidate_authority"],input_path.parent);same_candidates(context["candidates"],candidates)
        if audit.get("core_manifest"):
            core_path=bound.bind(audit["core_manifest"],path.parent)
            core,_,_=validate_current_export(core_path,expected_run_id=primary["training_run_id"])
            folds={r["patient_fold"]:r for r in primary["ensemble_source_records"]}
            if set(folds)!=set(range(5)):raise ValueError("Selected core requires five folds")
            for fold,item in core["folds"].items():
                if any(item[k]!=folds[int(fold)][k] for k in ("checkpoint_sha256","core_parameter_sha256")):
                    raise ValueError("Genomic context and selected scores use different core checkpoints")
        for item in audit["sources"]:bound.bind(item,path.parent)
        paths={name:bound.bind(item,path.parent) for name,item in audit["outputs"].items()}
        if set(paths)!={"fusion","fold_context","coverage"}:raise ValueError("Genomic native output inventory differs")
        experts={s["probability_column"].removesuffix("_context_probability"):s["expert_id"] for s in audit["expert_specs"]}
        used={m:expert in active and spec["binding"]["sha256"] in chosen_hashes for m,expert in experts.items()}
        covered.update(experts[m] for m in used if used[m])
        native[kind]={"audit":audit,"paths":paths,"experts":experts,"used":used}
        catalog.append({"producer_kind":kind,"status":"AVAILABLE","reason":None,"native_run_id":audit["run_id"],
            "native_estimand":audit["estimand"],"source_binding_sha256":spec["binding"]["sha256"]})
    if not native:raise ValueError("An available genomic context needs at least one native analysis")
    if active-covered:raise ValueError("Selected fusion genomic inputs are missing from the current context inventory")
    return {"native":native,"catalog":pd.DataFrame(catalog)}


def distal_sources(context,*,replay):
    from scripts.run_v32_distal_regulatory_mutation_head import main
    from .distal_topology import require_shared_predictive_topology
    bound=context["sources"];path=bound.bind(context["manifest"]["native_binding"],context["root"]);native=read(path)
    success_path=path.parent/"SUCCESS.json";success=read(bound.bind(record(success_path),path.parent))
    if (native.get("format")!="CANCERLNCATLAS_V32_DISTAL_REGULATORY_MUTATION_HEAD_NESTED_OOF_V2"
            or native.get("status")!="PASS" or native.get("target_host")!="149" or success.get("status")!="PASS"
            or success.get("audit_sha256")!=sha256(path) or native.get("neural_optimizer_started") is not False
            or native.get("statistical_analysis_executed") is not True):raise ValueError("Incomplete or historical distal source binding")
    for key in ("old_model_prediction_or_score_reused","heldout_lncrna_expression_used_for_own_fit",
                "outer_heldout_lncrna_expression_used_for_any_training_component_fit","heldout_pathway_activity_used_for_own_fit",
                "missing_assay_assumed_zero","causal_claimed"):
        if native.get(key) is not False:raise ValueError("Invalid distal source semantics: "+key)
    if native.get("outer_training_expression_components_are_inner_oof") is not True or native.get("patient_folds")!=5:
        raise ValueError("Distal source lacks nested five-fold isolation")
    codes=implementations("distal");expected={k:codes[k] for k in ("scripts/run_v32_distal_regulatory_mutation_head.py",
        "cc_hhgt/v32/epigenetic_expression_head.py","cc_hhgt/v32/distal_regulatory_mutation_head.py")}
    if native.get("implementation_sha256")!=expected:raise ValueError("Distal replay requires its actual recorded implementation")
    inputs={name:bound.bind(item,path.parent) for name,item in native["sources"].items()}
    if set(inputs)!={"head_input_audit","candidates"}:raise ValueError("Distal source inventory differs")
    same_candidates(context["candidates"],inputs["candidates"])
    prepared=read(inputs["head_input_audit"])
    from .native_distal_mutation_workflow import validate_distal_input
    for item in validate_distal_input(prepared,inputs["head_input_audit"].parent):bound.bind(item,path.parent)
    if prepared.get("distal_atac_root"):
        atac_audit=Path(prepared["distal_atac_root"])/"DISTAL_ATAC_AUDIT.json"
        if atac_audit.is_file() and "native_distal_atac_preparation" in read(atac_audit):
            from .native_distal_atac_workflow import validate_features
            for item in validate_features(record(atac_audit),path.parent):bound.bind(item,path.parent)
    if "native_distal_preparation" in prepared:
        from .native_distal_workflow import validate_prepared
        for item in validate_prepared(record(inputs["head_input_audit"]),path.parent):bound.bind(item,path.parent)
    elif "methylation_workflow" in prepared:
        from .native_methylation_workflow import validate_distal_input
        for item in validate_distal_input(prepared,inputs["head_input_audit"].parent):bound.bind(item,path.parent)
    lineage=require_shared_predictive_topology(prepared.get("topology_lineage",{}),native.get("topology_lineage",{}))
    if (sha256(inputs["head_input_audit"])!=native["head_input_audit_sha256"]
            or sha256(inputs["candidates"])!=native["candidate_universe_sha256"]):raise ValueError("Distal source aliases differ")
    partitions=[]
    for item in native["partition_sources"]:
        partitions.append({"cancer_id":item["cancer_id"],**{key:bound.bind(item[key],path.parent) for key in ("head_input","activity")}})
    if not partitions or len({p["cancer_id"] for p in partitions})!=len(partitions):raise ValueError("Invalid distal source partitions")
    groups={"candidate_oof_pathway_head":[],"patient_oof_expression_component":[],"nested_outer_expression_components":[]}
    for item in native["artifacts"]:
        target=bound.bind(item,path.parent)
        relative=target.relative_to(path.parent)
        if len(relative.parts)!=3 or relative.parts[0] not in groups:raise ValueError("Unknown distal native artifact")
        groups[relative.parts[0]].append(target)
    for paths in groups.values():
        if len(paths)!=len(partitions) or {p.parent.name for p in paths}!={"cancer_id="+p["cancer_id"] for p in partitions}:
            raise ValueError("Distal public/private output partition inventory differs")
    if replay:
        head_roots={p["head_input"].parent.parent for p in partitions};activity_roots={p["activity"].parent.parent for p in partitions}
        if head_roots!={inputs["head_input_audit"].parent} or len(activity_roots)!=1:raise ValueError("Distal input roots differ from native layout")
        ec=native["expression_config"];pc=native["pathway_config"]
        with tempfile.TemporaryDirectory(prefix="current_distal_native_replay_") as tmp:
            output=Path(tmp)/"native"
            if main(["--head-input-root",str(inputs["head_input_audit"].parent),"--pathway-activity-root",str(next(iter(activity_roots))),
                "--candidate-universe",str(inputs["candidates"]),"--output",str(output),
                "--ridge-alpha-expression",str(ec["ridge_alpha"]),"--min-expression-train-patients",str(ec["min_train_patients"]),
                "--ridge-alpha-pathway",str(pc["ridge_alpha"]),"--min-pathway-train-patients",str(pc["min_train_patients"])])!=0:
                raise ValueError("Distal native replay failed")
            other=read(output/"AUDIT.json")
            for key,value in other.items():
                if key not in {"artifacts","generated_at_utc"} and native.get(key)!=value:raise ValueError("Distal source policy or provenance differs: "+key)
            for name,paths in groups.items():
                keys=KEYS if name=="candidate_oof_pathway_head" else ["cancer_id","patient_id","lncrna_id"]
                if name=="nested_outer_expression_components":keys=keys+["outer_fold_id"]
                for target in paths:compare_tables(target,output/target.relative_to(path.parent),keys,label="Distal")
    return {"native":native,"paths":groups,"partitions":partitions,"topology":lineage}


def sources(manifest_path,*,replay=True):
    path=Path(manifest_path).resolve();manifest=read(path);bound=Sources();bound.bind(record(path),path.parent)
    kind=manifest.get("kind")
    if manifest.get("format")!=INPUT_FORMAT or kind not in {"genomic","distal"}:raise ValueError("Invalid omics context input")
    selected_path=bound.bind(manifest["selected_analysis"],path.parent)
    selected=SelectedAnalysisQuery(selected_path,expected_sha=manifest["selected_analysis"]["sha256"])
    score_path=bound.bind(selected.binding["selected_score_binding"],selected_path.parent);score=read(score_path)
    primary_path=bound.bind(score["sources"]["reference_exact_producer"]["producer_binding"],score_path.parent);primary=read(primary_path)
    from .current_core_prediction import FORMAT as current_format
    if primary.get("format")!=current_format:raise ValueError("Omics context requires the native current-core reference")
    candidates=bound.bind(primary["input_artifacts"][0],primary_path.parent)
    chosen_path=bound.bind(score["sources"]["selected_producer"]["producer_binding"],score_path.parent)
    for item in selected.binding["outputs"].values():bound.bind(item,selected_path.parent)
    context={"manifest":manifest,"root":path.parent,"sources":bound,"selected":selected,"primary":primary,
        "candidates":candidates,"chosen":read(chosen_path),"chosen_root":chosen_path.parent,"kind":kind}
    context.update(genomic_sources(context) if kind=="genomic" else distal_sources(context,replay=replay))
    return context


def genomic_queries(context,con):
    con.register("producer_catalog",context["catalog"]);folds=[];aggregate=[]
    for kind,value in context["native"].items():
        audit=value["audit"];estimand=sql_text(audit["estimand"]);run=sql_text(audit["run_id"])
        for modality,expert in value["experts"].items():
            extra=f"'{kind}' AS producer_kind,'{modality}' AS modality,'{expert}' AS expert_id,{estimand} AS native_estimand,{run} AS native_run_id,{str(value['used'][modality]).lower()} AS used_in_selected_score"
            aggregate.append(f"""SELECT cancer_id,lncrna_id,pathway_id,{modality}_context_probability AS probability,
                {modality}_available AS available,{modality}_patient_folds_with_prediction AS folds_available,
                {modality}_fold_sd AS fold_sd,{modality}_unavailable_reason AS unavailable_reason,
                {modality}_missing_fold_reasons AS missing_fold_reasons,{extra} FROM {relation(value['paths']['fusion'])}""")
            folds.append(f"SELECT * EXCLUDE(modality),{extra} FROM {relation(value['paths']['fold_context'])} WHERE modality='{modality}'")
    con.execute("CREATE VIEW native_aggregate AS "+" UNION ALL BY NAME ".join(aggregate))
    con.execute("CREATE VIEW native_folds AS "+" UNION ALL BY NAME ".join(folds))
    return {"producer_catalog":"SELECT * FROM producer_catalog","native_aggregate":"SELECT * FROM native_aggregate",
        "native_folds":"SELECT * FROM native_folds",
        "selected_member_genomic":"SELECT m.*,n.* EXCLUDE(cancer_id,lncrna_id,pathway_id) FROM members m JOIN native_aggregate n USING(cancer_id,lncrna_id,pathway_id)",
        "selected_geneset_genomic":"""SELECT m.geneset_id,m.cancer_id,m.pathway_id,n.producer_kind,n.modality,n.expert_id,n.native_estimand,n.native_run_id,n.used_in_selected_score,
            count(*) AS selected_members,count(*) FILTER(WHERE n.available) AS available_members,
            count(*) FILTER(WHERE n.available)::DOUBLE/count(*) AS member_coverage,
            coalesce(sum(m.selected_score) FILTER(WHERE n.available),0)/nullif(sum(m.selected_score),0) AS selected_weight_coverage,
            sum(m.selected_score*n.probability) FILTER(WHERE n.available)/nullif(sum(m.selected_score) FILTER(WHERE n.available),0) AS mean_member_probability,
            coalesce(sum(m.selected_score) FILTER(WHERE n.available),0)>0 AS summary_available,
            CASE WHEN coalesce(sum(m.selected_score) FILTER(WHERE n.available),0)<=0 THEN 'NO_AVAILABLE_MEMBER_PREDICTION' END AS unavailable_reason
            FROM members m JOIN native_aggregate n USING(cancer_id,lncrna_id,pathway_id) GROUP BY ALL"""}


def distal_queries(context,con):
    from .drug_sparse_query import _parquet_file_relation
    con.execute("CREATE VIEW native_candidates AS SELECT * FROM "+_parquet_file_relation(context["paths"]["candidate_oof_pathway_head"]))
    con.execute("CREATE VIEW private_components AS SELECT * FROM "+_parquet_file_relation(context["paths"]["patient_oof_expression_component"]))
    con.execute("CREATE VIEW private_inputs AS SELECT * FROM "+_parquet_file_relation([p["head_input"] for p in context["partitions"]]))
    features=" UNION ALL ".join(f"SELECT cancer_id,lncrna_id,'{feature}' AS feature,count(*) AS patients,count(*) FILTER(WHERE {feature}__available) AS available_patients,count(DISTINCT patient_fold_id) AS patient_folds FROM private_inputs GROUP BY cancer_id,lncrna_id" for feature in FEATURES)
    flag="distal_regulatory_mutation_pathway_available"
    return {"native_candidates":"SELECT * FROM native_candidates","native_feature_coverage":features,
        "native_component_coverage":"""SELECT cancer_id,lncrna_id,patient_fold_id,count(*) AS patients,
            count(*) FILTER(WHERE mutation_regulatory_expression_component_available) AS available_patients,
            avg(mutation_regulatory_expression_component_z) AS mean_expression_component_z,
            stddev_samp(mutation_regulatory_expression_component_z) AS expression_component_sd,
            string_agg(DISTINCT mutation_regulatory_expression_component_unavailable_reason,'; ' ORDER BY mutation_regulatory_expression_component_unavailable_reason) AS missing_reasons
            FROM private_components GROUP BY ALL""",
        "selected_member_distal":"SELECT m.*,n.* EXCLUDE(cancer_id,lncrna_id,pathway_id) FROM members m JOIN native_candidates n USING(cancer_id,lncrna_id,pathway_id)",
        "selected_geneset_distal_coverage":f"""SELECT m.geneset_id,m.cancer_id,m.pathway_id,count(*) AS selected_members,
            count(*) FILTER(WHERE n.{flag}) AS available_members,count(*) FILTER(WHERE n.{flag})::DOUBLE/count(*) AS member_coverage,
            coalesce(sum(m.selected_score) FILTER(WHERE n.{flag}),0)/nullif(sum(m.selected_score),0) AS selected_weight_coverage,
            count(*) FILTER(WHERE n.{flag} AND n.mean_oof_slope>0) AS positive_slope_members,
            count(*) FILTER(WHERE n.{flag} AND n.mean_oof_slope<0) AS negative_slope_members,
            count(*) FILTER(WHERE n.{flag})>0 AS summary_available,
            CASE WHEN count(*) FILTER(WHERE n.{flag})=0 THEN 'NO_AVAILABLE_DISTAL_MEMBER' END AS unavailable_reason
            FROM members m JOIN native_candidates n USING(cancer_id,lncrna_id,pathway_id) GROUP BY ALL"""}


def context_tables(context,output):
    output=Path(output);selected=context["selected"];tables={}
    with _bounded_duckdb() as con:
        con.execute(f"CREATE VIEW members AS SELECT * FROM {relation(selected.paths['geneset_members'])}")
        queries=genomic_queries(context,con) if context["kind"]=="genomic" else distal_queries(context,con)
        for name,query in queries.items():
            target=output/(name+".parquet")
            statement=f"SELECT q.*,'{selected.binding_sha256}' AS context_selected_analysis_sha256,false AS joint_geneset_effect_estimated,false AS causal_effect_claimed,false AS context_changes_selected_ranking FROM ({query}) q"
            con.execute("COPY ("+statement+") TO ? (FORMAT PARQUET,COMPRESSION ZSTD)",[str(target)])
            tables[name]={**record(target),"rows":pq.ParquetFile(target).metadata.num_rows}
    return tables


def materialize_current_omics_context(manifest_path,output):
    if socket.gethostname()!="149":raise RuntimeError("Omics context preparation requires target host 149")
    path=Path(manifest_path).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,"input_manifest":record(path)})
    context=sources(path);tables=context_tables(context,output);context["sources"].verify()
    binding={"format":FORMAT,"status":"PASS","target_host":"149","kind":context["kind"],"input_manifest":record(path),
        "selected_analysis":context["manifest"]["selected_analysis"],"outputs":tables,"source_artifacts":list(context["sources"].records.values()),
        "implementation_sha256":implementations(context["kind"]),"native_replay_scope":"NATIVE_FOLD_AGGREGATION_CHECKPOINT_INSPECTION" if context["kind"]=="genomic" else "NESTED_STATISTICAL_COMMAND_AND_PRIVATE_COMPONENTS",
        "native_neural_forwards_replayed":False,"joint_geneset_effect_estimated":False,"causal_effect_claimed":False,
        "context_changes_selected_ranking":False,"patient_identifiers_in_public_tables":False,"training_started":False,"production_deployed":False}
    bp=output/"CURRENT_OMICS_CONTEXT_BINDING.json";write(bp,binding)
    ap=output/"RECOMPUTATION_AUDIT.json";write(ap,audit_current_omics_context(bp,expected_sha=sha256(bp),replay_native=False))
    result={"status":"PASS","target_host":"149","training_started":False,"binding_path":str(bp),"binding_sha256":sha256(bp),"audit_sha256":sha256(ap)}
    write(output/"SUCCESS.json",result);return result


def load(binding_path,expected_sha,selected_analysis_sha,kind):
    path=verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd());binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or binding.get("kind")!=kind or kind not in {"genomic","distal"}
            or binding["selected_analysis"]["sha256"]!=selected_analysis_sha):raise ValueError("Invalid or different selected omics context")
    for field in ("native_neural_forwards_replayed","joint_geneset_effect_estimated","causal_effect_claimed","context_changes_selected_ranking",
                  "patient_identifiers_in_public_tables","training_started","production_deployed"):
        if binding.get(field) is not False:raise ValueError("Invalid omics context semantics: "+field)
    return path,binding,{name:verified_path(item,path.parent) for name,item in binding["outputs"].items()}


def audit_current_omics_context(binding_path,*,expected_sha,replay_native=True):
    if socket.gethostname()!="149":raise RuntimeError("Omics context audit requires target host 149")
    original=read(verified_path({"path":str(binding_path),"sha256":expected_sha},Path.cwd()))
    path,binding,paths=load(binding_path,expected_sha,original["selected_analysis"]["sha256"],original["kind"])
    if binding["implementation_sha256"]!=implementations(binding["kind"]):raise ValueError("Omics replay implementation changed")
    context=sources(verified_path(binding["input_manifest"],path.parent),replay=replay_native)
    if binding["selected_analysis"]!=context["manifest"]["selected_analysis"] or binding["source_artifacts"]!=list(context["sources"].records.values()):raise ValueError("Omics provenance changed")
    with tempfile.TemporaryDirectory(prefix="current_omics_public_replay_") as tmp:
        expected=context_tables(context,Path(tmp))
        if set(expected)!=set(paths):raise ValueError("Omics public inventory differs")
        for name,item in expected.items():
            columns=pq.ParquetFile(paths[name]).schema_arrow.names
            keys=[k for k in ("cancer_id","lncrna_id","pathway_id","geneset_id","producer_kind","modality","patient_fold_id","feature") if k in columns]
            compare_tables(paths[name],item["path"],keys,label="Omics")
            if item["rows"]!=binding["outputs"][name]["rows"]:raise ValueError("Omics row count changed")
    context["sources"].verify()
    return {"status":"PASS","binding_sha256":expected_sha,"public_tables_recomputed":True,
        "native_replayed_in_this_audit":replay_native or context["kind"]=="genomic","training_started":False}


class CurrentOmicsContextQuery:
    def __init__(self,binding_path,*,expected_sha,selected_analysis_sha,kind):
        self.path,self.binding,self.paths=load(binding_path,expected_sha,selected_analysis_sha,kind)
        ap=self.path.parent/"RECOMPUTATION_AUDIT.json";audit=read(ap);success=read(self.path.parent/"SUCCESS.json")
        if (success.get("status")!="PASS" or success.get("binding_sha256")!=expected_sha or success.get("audit_sha256")!=sha256(ap)
                or audit.get("status")!="PASS" or audit.get("binding_sha256")!=expected_sha or audit.get("public_tables_recomputed") is not True):raise ValueError("Incomplete omics context")
    def download(self,name):
        if name not in self.paths:raise ValueError("Unknown public omics table")
        item=self.binding["outputs"][name];verified_path(item,self.path.parent);return item
