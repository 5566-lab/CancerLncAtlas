"""Assemble native patient covariates, run nested statistics and close out.

Upstream feature materialization remains explicit. This workflow never fits a
neural model or deletes raw data; its statistical regressions are declared.
"""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import math
import tempfile

import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, record
from .current_analysis_workflow import write
from .input_lineage import artifact_sha256
from .context_table_replay import compare_tables

ROOT=Path(__file__).resolve().parents[2]
INPUT="NATIVE_DISTAL_WORKFLOW_INPUT_V1"
FORMAT="NATIVE_DISTAL_WORKFLOW_RESULTS_V1"
FILES=("distal_mutation","local_cnv","purity_covariates","candidate_universe")
DIRECTORIES=("formal_expression_root","distal_atac_root","pathway_activity_root")
DEFAULTS={"ridge_alpha_expression":1.0,"ridge_alpha_pathway":1.0,
          "min_expression_train_patients":20,"min_pathway_train_patients":20}
CODE=("cc_hhgt/v32/native_distal_workflow.py","scripts/run_native_distal_workflow.py",
      "scripts/materialize_v32_distal_regulatory_head_input.py","scripts/run_v32_distal_regulatory_mutation_head.py",
      "scripts/audit_v32_distal_regulatory_mutation_head.py","cc_hhgt/v32/epigenetic_expression_head.py",
      "cc_hhgt/v32/distal_regulatory_mutation_head.py","cc_hhgt/v32/current_omics_context.py",
      "cc_hhgt/v32/distal_topology.py","cc_hhgt/v32/context_table_replay.py")


def plan(manifest,output,*,new=True):
    require_host();manifest=Path(manifest).resolve();spec=read(manifest);output=Path(output).resolve()
    required={"format","methylation_workflow",*FILES,*DIRECTORIES}
    if not required<=set(spec) or set(spec)-required-{"parameters"} or spec["format"]!=INPUT:
        raise ValueError("Invalid native distal workflow manifest")
    if new and output.exists():raise FileExistsError("Use a new distal workflow directory")
    parameters=spec.get("parameters",{})
    if not isinstance(parameters,dict) or set(parameters)-set(DEFAULTS):raise ValueError("Unknown distal parameters")
    parameters={**DEFAULTS,**parameters}
    for key,value in parameters.items():
        if key.startswith("min_"):
            if type(value) is not int or value<5:raise ValueError("Invalid distal patient-count setting")
        elif type(value) not in (int,float) or not math.isfinite(value) or value<0:
            raise ValueError("Invalid distal ridge setting")
    files={k:record(verified_path(spec[k],manifest.parent)) for k in FILES}
    sources={v["path"]:v for v in files.values()};directories={}
    for key in DIRECTORIES:
        path=(manifest.parent/spec[key]["path"]).resolve()
        if not path.is_dir() or artifact_sha256(path)!=spec[key]["sha256"]:raise ValueError("Distal directory checksum differs: "+key)
        if path==output or path in output.parents:raise ValueError("Distal output must be outside source directories")
        directories[key]={"path":str(path),"sha256":artifact_sha256(path)}
        for child in sorted(path.rglob("*")):
            if child.is_file():sources[str(child)]=record(child)
    from .native_methylation_workflow import validate as validate_methylation
    atac_audit=Path(directories["distal_atac_root"]["path"])/"DISTAL_ATAC_AUDIT.json"
    if atac_audit.is_file() and "native_distal_atac_preparation" in read(atac_audit):
        from .native_distal_atac_workflow import validate_features
        for item in validate_features(record(atac_audit),manifest.parent):sources[item["path"]]=item
    methylation_path,methylation=validate_methylation(spec["methylation_workflow"],manifest.parent)
    methylation_features=record(verified_path(methylation["outputs"]["patient_features"],methylation_path.parent))
    mutation_audit=Path(files["distal_mutation"]["path"]).parent/"AUDIT.json"
    sources[str(mutation_audit)]=record(mutation_audit)
    mutation=read(mutation_audit)
    if (mutation.get("status")!="PASS_DISTAL_MUTATION_FEATURES" or mutation.get("locus_callability_claimed") is not False
            or mutation.get("missing_assay_assumed_zero") is not False):raise ValueError("Invalid native distal mutation semantics")
    from .distal_topology import feature_topology_lineage,require_shared_predictive_topology
    lineage=feature_topology_lineage(Path(files["distal_mutation"]["path"]),kind="mutation")
    methylation_audit=read(verified_path(methylation["audit"],methylation_path.parent))
    require_shared_predictive_topology(lineage,methylation_audit.get("topology_lineage",{}))
    candidates=pd.read_parquet(files["candidate_universe"]["path"]);keys=["cancer_id","lncrna_id","pathway_id"]
    from scripts.materialize_v32_distal_regulatory_head_input import FORMAL_CANCERS
    if (candidates.empty or not set(keys)<=set(candidates) or candidates[keys].isna().any().any()
            or candidates.duplicated(keys).any() or not set(candidates.cancer_id)<=set(FORMAL_CANCERS)):
        raise ValueError("Distal workflow needs unique canonical candidates")
    cancers=sorted(candidates.cancer_id.unique())
    for cancer in cancers:
        for key in ("formal_expression_root","pathway_activity_root"):
            if not (Path(directories[key]["path"])/f"cancer_id={cancer}/part-0.parquet").is_file():
                raise FileNotFoundError("Missing distal cancer partition: "+key+"/"+cancer)
        atac=Path(directories["distal_atac_root"]["path"])/(cancer+".distal_accessibility.tsv.gz")
        if atac.is_file():require_shared_predictive_topology(lineage,feature_topology_lineage(atac,kind="atac"))
    return {"input":record(manifest),"target_host":"149","training_started":False,
        "statistical_regressions_planned":True,"files":files,"directories":directories,
        "methylation_workflow":record(methylation_path),"methylation_features":methylation_features,
        "cancers":cancers,"parameters":parameters,"topology_lineage":lineage,
        "source_files":sorted(sources.values(),key=lambda r:r["path"]),"implementations":[record(ROOT/n) for n in CODE],
        "source_ancestry_complete":False,"raw_deletion_executed":False}


def assemble(value,output):
    from scripts.materialize_v32_distal_regulatory_head_input import main
    roles={**value["files"],**value["directories"]}
    keys=("formal_expression_root","distal_mutation","distal_atac_root","local_cnv","purity_covariates")
    args=[part for key in keys for part in ("--"+key.replace("_","-"),roles[key]["path"])]
    args += ["--methylation",value["methylation_features"]["path"],
        "--methylation-workflow",value["methylation_workflow"]["path"],
        "--methylation-workflow-sha256",value["methylation_workflow"]["sha256"],
        "--cancers",",".join(value["cancers"]),"--output",str(output)]
    if main(args)!=0:raise RuntimeError("Distal input assembly failed")


def validate_prepared(item,root):
    """Reconstruct all assembled covariates, not only their recorded hashes."""
    require_host();path=verified_path(item,root);actual=read(path)
    preflight=verified_path(actual["native_distal_preparation"],path.parent);value=read(preflight)
    if path!=preflight.parent/"head_input/AUDIT.json":raise ValueError("Distal prepared location differs")
    if value!=plan(verified_path(value["input"],preflight.parent),preflight.parent,new=False):
        raise ValueError("Distal workflow input or implementation differs")
    with tempfile.TemporaryDirectory(prefix="native_distal_input_replay_") as tmp:
        target=Path(tmp)/"head_input"
        with (Path(tmp)/"replay.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):assemble(value,target)
        expected=read(target/"AUDIT.json");excluded={"outputs","generated_at_utc","native_distal_preparation"}
        if {k:v for k,v in actual.items() if k not in excluded}!={k:v for k,v in expected.items() if k not in excluded}:
            raise ValueError("Distal preparation metadata differs from source reconstruction")
        if len(actual["outputs"])!=len(expected["outputs"]):raise ValueError("Distal prepared partition count differs")
        for left,right in zip(actual["outputs"],expected["outputs"]):
            file=verified_path(left,path.parent);reference=Path(right["path"])
            if file!=path.parent/reference.parent.name/reference.name:raise ValueError("Distal prepared partition identity differs")
            compare_tables(file,reference,["cancer_id","patient_id","lncrna_id"],label="Distal prepared")
    from .native_methylation_workflow import validate_distal_input
    dependencies=validate_distal_input(actual,path.parent)
    return [record(preflight),value["input"],*value["source_files"],*value["implementations"],*dependencies]


def statistics(value,output):
    from scripts.run_v32_distal_regulatory_mutation_head import main
    args=["--head-input-root",str(output/"head_input"),"--pathway-activity-root",value["directories"]["pathway_activity_root"]["path"],
        "--candidate-universe",value["files"]["candidate_universe"]["path"],"--output",str(output/"statistics")]
    args += [part for key,v in value["parameters"].items() for part in ("--"+key.replace("_","-"),str(v))]
    if main(args)!=0:raise RuntimeError("Distal nested statistics failed")


def closure(value,output,destination):
    from scripts.audit_v32_distal_regulatory_mutation_head import main
    methylation=read(value["methylation_workflow"]["path"])
    args=["--head-root",str(output/"statistics"),"--head-input-root",str(output/"head_input"),
        "--methylation-root",str(Path(methylation["audit"]["path"]).parent),
        "--distal-mutation-root",str(Path(value["files"]["distal_mutation"]["path"]).parent),
        "--candidate-universe",value["files"]["candidate_universe"]["path"],
        "--candidate-scope-sha256",value["files"]["candidate_universe"]["sha256"],"--output",str(destination)]
    if main(args)!=0:raise RuntimeError("Distal closure failed")


def check_results(value,output):
    from .current_omics_context import Sources,distal_sources
    context={"manifest":{"native_binding":record(output/"statistics/AUDIT.json")},"root":output,
        "sources":Sources(),"candidates":Path(value["files"]["candidate_universe"]["path"])}
    native=distal_sources(context,replay=True)["native"]
    if native["sources"]!={"head_input_audit":record(output/"head_input/AUDIT.json"),"candidates":value["files"]["candidate_universe"]}:
        raise ValueError("Distal statistics source identity differs from preparation")
    expected_partitions=[{"cancer_id":c,"head_input":record(output/f"head_input/cancer_id={c}/part-0.parquet"),
        "activity":record(Path(value["directories"]["pathway_activity_root"]["path"])/f"cancer_id={c}/part-0.parquet")} for c in value["cancers"]]
    if native["partition_sources"]!=expected_partitions:raise ValueError("Distal statistics partitions differ from preparation")
    ec=native["expression_config"];pc=native["pathway_config"]
    parameters={"ridge_alpha_expression":ec["ridge_alpha"],"ridge_alpha_pathway":pc["ridge_alpha"],
        "min_expression_train_patients":ec["min_train_patients"],"min_pathway_train_patients":pc["min_train_patients"]}
    if parameters!=value["parameters"]:raise ValueError("Distal statistics settings differ from preparation")
    actual=read(output/"closure/CLOSURE_AUDIT.json")
    if read(output/"closure/SUCCESS.json")!={"status":"PASS","closure_audit_sha256":record(output/"closure/CLOSURE_AUDIT.json")["sha256"]}:
        raise ValueError("Distal closure completion differs")
    with tempfile.TemporaryDirectory(prefix="native_distal_closure_replay_") as tmp:
        closure(value,output,Path(tmp)/"closure");expected=read(Path(tmp)/"closure/CLOSURE_AUDIT.json")
        if {k:v for k,v in actual.items() if k!="completed_at_utc"}!={k:v for k,v in expected.items() if k!="completed_at_utc"}:
            raise ValueError("Distal closure differs from actual results")
    context["sources"].verify()
    return native


def run(manifest,output):
    value=plan(manifest,output);output=Path(output).resolve();output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",value)
    try:
        with (output/"stage.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):
            assemble(value,output/"head_input")
            prepared=output/"head_input/AUDIT.json";audit=read(prepared)
            audit["native_distal_preparation"]=record(output/"LAUNCH_PREFLIGHT.json");write(prepared,audit)
            statistics(value,output);closure(value,output,output/"closure");native=check_results(value,output)
        if value!=plan(manifest,output,new=False):raise ValueError("Distal workflow inputs changed during execution")
        binding=output/"NATIVE_DISTAL_BINDING.json"
        write(binding,{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
            "statistical_analysis_executed":True,"prepared_sources_numerically_replayed":True,
            "neural_optimizer_started":False,"raw_deletion_executed":False,"source_ancestry_complete":False,
            "production_deployed":False,"preflight":record(output/"LAUNCH_PREFLIGHT.json"),"prepared":record(prepared),
            "audit":record(output/"statistics/AUDIT.json"),"closure":record(output/"closure/CLOSURE_AUDIT.json"),
            "outputs":native["artifacts"]})
        return binding
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise


def validate(item,root):
    require_host();path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or any(binding.get(k) is not False for k in ("training_started","neural_optimizer_started","raw_deletion_executed","source_ancestry_complete","production_deployed"))
            or any(binding.get(k) is not True for k in ("statistical_analysis_executed","prepared_sources_numerically_replayed"))):
        raise ValueError("Incomplete native distal workflow")
    for key,name in (("preflight","LAUNCH_PREFLIGHT.json"),("prepared","head_input/AUDIT.json"),
                     ("audit","statistics/AUDIT.json"),("closure","closure/CLOSURE_AUDIT.json")):
        if verified_path(binding[key],path.parent)!=path.parent/name:raise ValueError("Distal workflow result identity differs")
    value=read(binding["preflight"]["path"])
    if value!=plan(verified_path(value["input"],path.parent),path.parent,new=False):raise ValueError("Distal workflow sources changed")
    native=check_results(value,path.parent)
    if binding["outputs"]!=native["artifacts"]:raise ValueError("Distal workflow output inventory differs")
    return path,binding
