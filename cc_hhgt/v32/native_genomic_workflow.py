"""Native genomic preparation and completed-result collection; never dispatch fit.

Resource prediction arrays are intentionally mutable between the two phases.
Their immutable stage contract is pinned at preparation; completed arrays are
checked through the actual trainer seal and reassembler at collection.
"""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import re
import sys

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, write, record
from . import native_genomic_cnv as cnv
from . import genomic_resource_staging as resource
from .genomic_training import GenomicTrainingConfig

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_GENOMIC_WORKFLOW_INPUT_V1"
PREPARED = "NATIVE_GENOMIC_WORKFLOW_PREPARED_V1"
FORMAT = "NATIVE_GENOMIC_WORKFLOW_RESULTS_V1"
AUDIT = "NATIVE_GENOMIC_COLLECTION_AUDIT_V1"
TRAINER = "scripts.train_v32_partitioned_genomic_heads"
CODE = tuple(sorted(set((*cnv.CODE, "cc_hhgt/v32/native_genomic_workflow.py", "scripts/run_native_genomic_workflow.py",
    "cc_hhgt/v32/genomic_training.py", "cc_hhgt/v32/genomic_partition_training.py", "cc_hhgt/v32/genomic_resource_staging.py",
    "cc_hhgt/v32/native_partitioned_genomic.py", "cc_hhgt/v32/native_auxiliary_fusion.py", TRAINER.replace(".","/")+".py"))))
LIMITS = {"adapter_budget": {"max_declared_adapter_peak_rss_bytes", "max_memmap_bytes", "max_staging_disk_bytes"},
          "downstream_budget": {"max_declared_process_rss_bytes", "minimum_host_available_memory_bytes", "minimum_host_free_disk_bytes"}}


def arguments(values): return [p for k,v in values.items() for p in ("--"+k.replace("_","-"),str(v))]


def plan(manifest, output, *, new=True):
    require_host(); manifest=Path(manifest).resolve(); spec=read(manifest); output=Path(output).resolve()
    required={"format","native_run_id","genomic_input","resource_limits"}
    if (not required <= set(spec) or set(spec)-required-{"config"} or spec["format"]!=INPUT
            or not re.fullmatch(r"v32-[A-Za-z0-9][A-Za-z0-9_.-]{0,115}",str(spec["native_run_id"]))):
        raise ValueError("Invalid native genomic workflow input")
    if new and output.exists(): raise FileExistsError("Use a new genomic preparation directory")
    _,source,_,context=cnv.load(spec["genomic_input"],manifest.parent)
    limits=spec["resource_limits"]
    if not isinstance(limits,dict) or set(limits)!=set(LIMITS): raise ValueError("Missing genomic resource limits")
    for name,keys in LIMITS.items():
        if (not isinstance(limits[name],dict) or set(limits[name])!=keys
                or any(type(v) is not int or v<=0 for v in limits[name].values())):
            raise ValueError("Invalid genomic resource limits: "+name)
    config=asdict(GenomicTrainingConfig()); config["cnv_event_threshold"]=context["event_threshold"]
    override=spec.get("config",{})
    if not isinstance(override,dict) or set(override)-set(config): raise ValueError("Unknown genomic training configuration")
    for name,value in override.items():
        if (isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value)
                or (type(config[name]) is int and type(value) is not int)):
            raise ValueError("Invalid genomic training configuration: "+name)
        config[name]=value
    GenomicTrainingConfig(**config).validate()
    if config["cnv_event_threshold"]!=context["event_threshold"]: raise ValueError("Genomic event threshold differs from raw preparation")
    files=context["bindings"]; cohort=context["cohort"]
    values={"repo_root":ROOT,"resource_staging_success":output/"resources/SUCCESS.json",
        "patient_folds":files["patient_folds"]["path"],"candidate_definition":cohort["candidate_definition"]["path"],
        "candidate_definition_sha256":cohort["candidate_definition"]["sha256"],"expected_core_run_id":source["core_run_id"],
        "pathway_gene_membership":files["pathway_membership"]["path"],
        **{k:files[k]["path"] for k in ("sample_gene_mutation","sample_lncrna_mutation","mc3","core_embedding_manifest")},
        "cnv_streaming_success":output/"streaming/SUCCESS.json","output_root":output/"training",
        "training_run_id":spec["native_run_id"],**config}
    command=[sys.executable,"-m",TRAINER,*arguments(values)]
    from scripts.train_v32_partitioned_genomic_heads import _parser
    _parser().parse_args(command[3:])
    prepare=[sys.executable,"-m",TRAINER,*arguments({**values,"output_root":output/"head_preflight"}),"--prepare-only"]
    _parser().parse_args(prepare[3:])
    return {"target_host":"149","training_started":False,"input":record(manifest),"genomic_input":context["native_input"],
        "native_run_id":spec["native_run_id"],"core_run_id":source["core_run_id"],"files":files,"cohort":cohort,
        "config":config,"resource_limits":limits,"raw_context":context,"prepare_command":prepare,
        "training_command":{"argv":command,"automatically_executed":False,"requires_separate_explicit_execution":True,
                            "execution_mode":"CODE_ONLY"},"implementations":[record(ROOT/name) for name in CODE]}


def budget(value, output):
    output=Path(output); files=value["files"]; cohort=value["cohort"]; digest=hashlib.sha256()
    for cancer in cohort["cancers"]:
        frame=resource._canonical_partition(resource._read_candidate_cancer(Path(files["candidates"]["path"]),cancer),cancer)
        for row in frame.itertuples(index=False,name=None): digest.update(("\t".join(row)+"\n").encode())
    return {"format":resource.BUDGET_FORMAT,"resource_budget_id":value["native_run_id"]+"-resources",
        "scope":{"formal_cancers":cohort["cancers"],"candidate_rows":cohort["candidate_rows"],
                 "candidate_rows_by_cancer":cohort["candidate_rows_by_cancer"]},
        "adapter_budget":{**value["resource_limits"]["adapter_budget"],"max_candidate_partition_rows":max(cohort["candidate_rows_by_cancer"].values())},
        "downstream_budget":{**value["resource_limits"]["downstream_budget"],"forbid_launch_without_resource_success":True},
        "memmap_layout":{"modalities":["mutation","cnv"],"patient_folds":5,"probability_dtype":"float32","available_dtype":"bool","processed_dtype":"bool"},
        "input_bindings":{"candidate_authority":{**files["candidates"],"ordered_exact_candidate_key_sha256":digest.hexdigest()},
            "cnv_streaming_aggregate_success":{**record(output/"streaming/SUCCESS.json"),"format":resource.CNV_STREAMING_FORMAT,
                "status":"SUCCESS","sha256_policy":"CALLER_MUST_PIN_EXACT_64_HEX_AT_LAUNCH"}}}


def execute_preflight(command, log):
    if command[2]!=TRAINER or command.count("--prepare-only")!=1:
        raise ValueError("Genomic workflow cannot dispatch fitting")
    from scripts.train_v32_partitioned_genomic_heads import main
    with Path(log).open("w",encoding="utf-8") as handle,redirect_stdout(handle),redirect_stderr(handle):
        if main(command[3:])!=0: raise RuntimeError("Genomic input preflight failed")


def input_state(value, output):
    """Check historical preparation evidence without treating mutable arrays as inputs."""
    output=Path(output); files=value["files"]; cohort=value["cohort"]
    success=output/"streaming/SUCCESS.json"; release=cnv.validate(success,expected_bindings={k:r["path"] for k,r in files.items()})
    if release["native_input"]!=value["genomic_input"] or release["cohort"]!=cohort: raise ValueError("Prepared genomic raw release differs")
    audit=read(output/"RAW_CNV_AUDIT.json")
    expected={"format":"NATIVE_GENOMIC_CNV_NUMERICAL_AUDIT_V1","status":"PASS","target_host":"149","training_started":False,
        "source":record(success),"arrays_compared":6*len(cohort["cancers"]),"source_ancestry_complete":False}
    if audit!=expected: raise ValueError("Prepared genomic raw audit differs")
    if read(output/"RESOURCE_BUDGET.json")!=budget(value,output): raise ValueError("Prepared genomic resource budget differs")
    _,stage=resource._load_stage(output/"resources/SUCCESS.json",verify_external_hashes=True)
    if (stage["candidate_authority"]["sha256"]!=files["candidates"]["sha256"]
            or stage["resource_budget"]["sha256"]!=record(output/"RESOURCE_BUDGET.json")["sha256"]
            or stage["cnv_streaming_aggregate_success"]["sha256"]!=record(success)["sha256"]):
        raise ValueError("Prepared genomic stage input differs")
    ready=read(output/"head_preflight/HEAD_PREFLIGHT.json")
    for key,expected_value in {"status":"PASS_READY","target_host":"149","training_started":False,"model_forwards_executed":False,
        "training_run_id":value["native_run_id"],"core_run_id":value["core_run_id"],"configuration":value["config"],
        "cohort":cohort,"candidate_rows":cohort["candidate_rows"],"candidate_rows_by_cancer":cohort["candidate_rows_by_cancer"],
        "resource_staging_success":record(output/"resources/SUCCESS.json"),"core_manifest":files["core_embedding_manifest"],
        "host_checks_enforced":True,"predictions_written":False,"training_success_emitted":False,"source_ancestry_complete":False}.items():
        if ready.get(key)!=expected_value: raise ValueError("Genomic input preflight differs: "+key)
    if not ready.get("host_checks") or not all(ready["host_checks"].values()): raise ValueError("Genomic host checks did not pass")
    required_inputs={files[k]["path"]:files[k]["sha256"] for k in ("candidates","patient_folds","pathway_membership","sample_gene_mutation","sample_lncrna_mutation","mc3")}
    required_inputs[str(success)]=record(success)["sha256"]
    if (ready["input_lineage_audit"].get("status")!="PASS"
            or {r["path"]:r["sha256"] for r in ready["input_lineage_audit"]["artifacts"]}!=required_inputs):
        raise ValueError("Genomic preflight consumed inputs differ")
    from .genomic_training import normalise_patient_folds,_read_table,_split_patients,load_fold_core_embeddings
    from .genomic_partition_training import _patient_split_trace
    patients=normalise_patient_folds(_read_table(files["patient_folds"]["path"]))
    patients=patients.loc[patients.cancer_id.isin(cohort["cancers"])]
    rows=ready["folds"]; expected_folds={(m,f) for m in ("mutation","cnv") for f in range(5)}
    if len(rows)!=10 or {(r["modality"],r["patient_fold"]) for r in rows}!=expected_folds: raise ValueError("Genomic preflight lacks ten groups")
    core_inputs={}; core_path=Path(files["core_embedding_manifest"]["path"]); core_manifest=read(core_path)
    for fold in range(5): core_inputs.update(load_fold_core_embeddings(core_path,core_manifest,fold).input_hashes)
    if ready["core_input_hashes"]!=core_inputs: raise ValueError("Genomic prepared core exports differ")
    for row in rows:
        f=row["patient_fold"]; expected_trace=_patient_split_trace({s:_split_patients(patients,f,s) for s in ("train","validation","test")},f,row["modality"])
        if row["split_policy"]!=expected_trace: raise ValueError("Genomic prepared patient split differs")
        for split in ("train","validation"):
            n,p=row[split+"_rows"],row[split+"_positive_rows"]
            if type(n) is not int or type(p) is not int or not 0<p<n: raise ValueError("Genomic preflight lacks two-class samples")
    pins=[value["input"],value["genomic_input"],value["raw_context"]["segment_manifest"],cohort["candidate_definition"],
        *files.values(),*value["implementations"],*[{"path":p,"sha256":h} for p,h in core_inputs.items()],
        *[r for rows in value["raw_context"]["raw_segments"].values() for r in rows]]
    paths=[output/"RESOURCE_BUDGET.json",output/"RAW_CNV_AUDIT.json",output/"RAW_CNV_AUDIT.json.preflight.json",
        output/"resources/SUCCESS.json",output/"resources/RESOURCE_MANIFEST.json",output/"head_preflight/HEAD_PREFLIGHT.json",
        output/"head_preflight/LAUNCH_PREFLIGHT.json",*[p for p in (output/"streaming").rglob("*") if p.is_file()],
        *[Path(r["path"]) for r in stage["candidate_partitions"]]]
    pins.extend(record(p) for p in paths)
    observed={str(verified_path(r,ROOT)):record(verified_path(r,ROOT)) for r in pins}
    return [observed[p] for p in sorted(observed)]


def prepare(manifest, output):
    value=plan(manifest,output); output=Path(output).resolve(); output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",value)
    try:
        cnv.materialize(value["genomic_input"],ROOT,output/"streaming")
        from scripts.materialize_native_genomic_cnv import main as audit
        success=output/"streaming/SUCCESS.json"
        audit(["audit","--success",str(success),"--expected-sha256",record(success)["sha256"],"--output",str(output/"RAW_CNV_AUDIT.json")])
        write(output/"RESOURCE_BUDGET.json",budget(value,output))
        resource.stage_genomic_cancer_memmaps(candidate_authority_path=value["files"]["candidates"]["path"],
            genomic_training_source_path=ROOT/"cc_hhgt/v32/genomic_training.py",cnv_streaming_success_path=success,
            cnv_streaming_success_sha256=record(success)["sha256"],budget_contract_path=output/"RESOURCE_BUDGET.json",output_root=output/"resources")
        execute_preflight(value["prepare_command"],output/"prepare.log")
        pins=input_state(value,output)
        if value!=plan(manifest,output,new=False): raise ValueError("Genomic inputs or code changed during preparation")
        write(output/"TRAINING_COMMAND.json",value["training_command"])
        write(output/"PREPARED.json",{"format":PREPARED,"status":"PASS_READY","target_host":"149","training_started":False,
            "input":value["input"],"plan":record(output/"LAUNCH_PREFLIGHT.json"),"training_command":record(output/"TRAINING_COMMAND.json"),
            "consumed_sources":pins})
        return output/"PREPARED.json"
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"stage":"prepare","error":str(error)})
        raise


def validate_preparation(item, root):
    require_host(); path=verified_path(item,root); binding=read(path)
    if (binding.get("format")!=PREPARED or binding.get("status")!="PASS_READY"
            or binding.get("target_host")!="149" or binding.get("training_started") is not False):
        raise ValueError("Invalid genomic preparation binding")
    manifest=verified_path(binding["input"],path.parent); value=read(verified_path(binding["plan"],path.parent))
    if value!=plan(manifest,path.parent,new=False): raise ValueError("Genomic preparation plan changed")
    if read(verified_path(binding["training_command"],path.parent))!=value["training_command"]: raise ValueError("Genomic training command differs")
    if binding["consumed_sources"]!=input_state(value,path.parent): raise ValueError("Genomic prepared source inventory differs")
    return path,binding,value


def completion(preparation, completed, root):
    path,binding,value=validate_preparation(preparation,root)
    result=verified_path(completed,root)
    if result!=path.parent/"training/TRAINING_SUCCESS.json": raise ValueError("Genomic result is outside its prepared training output")
    from .native_auxiliary_fusion import Sources
    sources=Sources(); success=read(result)
    from .genomic_partition_training import TRAINING_FORMAT
    if (success.get("format")!=TRAINING_FORMAT or success.get("status")!="TRAINING_SUCCESS_PREDICTIONS_SEALED_NOT_RELEASED"
            or success.get("checkpoint_count")!=10):
        raise ValueError("Genomic resource preparation is not completed training")
    lp,lineage=sources.document({"path":success["lineage_path"],"sha256":success["lineage_sha256"]},result.parent)
    if (lp!=result.parent/"LINEAGE.json" or lineage.get("training_run_id")!=value["native_run_id"]
            or lineage.get("configuration")!=value["config"] or lineage.get("cohort")!=value["cohort"]
            or lineage.get("core_run_id")!=value["core_run_id"]
            or lineage.get("resource_stage_success_path")!=str(path.parent/"resources/SUCCESS.json")
            or lineage.get("resource_stage_success_sha256")!=record(path.parent/"resources/SUCCESS.json")["sha256"]):
        raise ValueError("Genomic completion differs from prepared run/configuration/resources")
    ready=read(path.parent/"head_preflight/HEAD_PREFLIGHT.json")
    _,audit=sources.document(lineage["completion_records"]["INPUT_LINEAGE_AUDIT.json"],result.parent)
    expected={r["path"]:r["sha256"] for r in ready["input_lineage_audit"]["artifacts"]}
    if {r["path"]:r["sha256"] for r in audit["artifacts"]}!=expected: raise ValueError("Genomic completion consumed unprepared inputs")
    _,checkpoints=sources.document(lineage["completion_records"]["CHECKPOINT_MANIFEST.json"],result.parent)
    rows={(r["modality"],r["patient_fold"]):r for r in ready["folds"]}
    for checkpoint in checkpoints["records"]:
        prepared=rows[(checkpoint["modality"],checkpoint["patient_fold"])]
        if any(checkpoint.get(k)!=prepared[k] for k in ("train_rows","validation_rows")):
            raise ValueError("Genomic completion sample counts differ from preparation")
    return path,binding,value,lineage


def expected_audit(manifest, context, prepared, value):
    from .native_auxiliary_fusion import _analyse
    result,coverage=_analyse(context)
    return {"format":AUDIT,"status":"PASS","target_host":"149","training_started":False,"model_forwards_executed":False,
        "preparation":record(prepared),"completed":manifest["native_receipt"],"reassembly":manifest["reassembly_receipt"],
        "native_run_id":value["native_run_id"],"core_run_id":value["core_run_id"],"configuration":value["config"],
        "candidate_rows":len(result),"fold_rows":int(coverage.rows.sum()),"coverage":coverage.to_dict("records"),
        "source_ancestry_complete":False,"formal_increment_established":False,
        "scope":"CHECKPOINT_SPLITS_PREPARED_INPUTS_SEALED_MEMMAP_REASSEMBLY_AND_NATIVE_FOLD_AGGREGATION"}


def validate_completed(manifest, context, root, *, require_audit=True):
    prepared,binding,value,_=completion(manifest["workflow_preparation"],manifest["native_receipt"],root)
    expected={"producer_kind":"genomic","modalities":["mutation","cnv"],"native_run_id":value["native_run_id"],
        "core_run_id":value["core_run_id"],"candidate_authority":value["files"]["candidates"],
        "patient_folds":value["files"]["patient_folds"],"core_manifest":value["files"]["core_embedding_manifest"]}
    if any(manifest.get(k)!=v for k,v in expected.items()): raise ValueError("Genomic consumer differs from workflow authorities")
    sources=context["sources"];sources.bind(manifest["workflow_preparation"],root)
    for item in [binding["input"],binding["plan"],binding["training_command"],*binding["consumed_sources"]]: sources.bind(item,prepared.parent)
    if require_audit:
        if not isinstance(manifest.get("workflow_audit"),dict): raise ValueError("Genomic workflow consumer requires its collection audit")
        audit=read(sources.bind(manifest["workflow_audit"],root))
        if audit!=expected_audit(manifest,context,prepared,value): raise ValueError("Genomic collection audit differs from actual completed sources")
    return value


def collect(preparation, completed, output):
    prepared,_,value,lineage=completion(preparation,completed,ROOT); output=Path(output).resolve()
    if output.exists(): raise FileExistsError("Use a new genomic collection directory")
    output.mkdir(parents=True); write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,
        "preparation":record(prepared),"completed":completed})
    try:
        ready=verified_path({"path":lineage["predictions_ready_path"],"sha256":lineage["predictions_ready_sha256"]},prepared.parent)
        resource.reassemble_genomic_memmap_outputs(predictions_ready_path=ready,output_root=output/"reassembled",training_run_id=value["native_run_id"])
        from .native_auxiliary_fusion import INPUT_FORMAT,native_sources,materialize_native_auxiliary,load_native_auxiliary
        manifest={"format":INPUT_FORMAT,"producer_kind":"genomic","modalities":["mutation","cnv"],
            "native_run_id":value["native_run_id"],"core_run_id":value["core_run_id"],"candidate_authority":value["files"]["candidates"],
            "patient_folds":value["files"]["patient_folds"],"core_manifest":value["files"]["core_embedding_manifest"],
            "native_receipt":record(verified_path(completed,ROOT)),"reassembly_receipt":record(output/"reassembled/SUCCESS.json")}
        adapter_input=output/"NATIVE_AUXILIARY_INPUT.json";write(adapter_input,manifest)
        context=native_sources(adapter_input,expected_core_run_id=value["core_run_id"])
        manifest["workflow_preparation"]=record(prepared); validate_completed(manifest,context,output,require_audit=False)
        audit=output/"COLLECTION_AUDIT.json";write(audit,expected_audit(manifest,context,prepared,value))
        manifest["workflow_audit"]=record(audit);write(adapter_input,manifest)
        adapter=materialize_native_auxiliary(adapter_input,output/"fusion",expected_core_run_id=value["core_run_id"])
        _,info=load_native_auxiliary(adapter,expected_sha=record(adapter)["sha256"],expected_core_run_id=value["core_run_id"])
        write(output/"NATIVE_GENOMIC_BINDING.json",{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
            "model_forwards_executed":False,"native_run_id":value["native_run_id"],"core_run_id":value["core_run_id"],
            "preparation":record(prepared),"completed":manifest["native_receipt"],"reassembly":manifest["reassembly_receipt"],
            "adapter_input":record(adapter_input),"collection_audit":record(audit),"fusion_binding":record(adapter),
            "fusion_experts":record(adapter.parent/"FUSION_EXPERTS.json"),"candidate_rows":info["rows"],"fold_rows":info["fold_rows"],
            "source_ancestry_complete":False,"formal_increment_established":False})
        return output/"NATIVE_GENOMIC_BINDING.json"
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"stage":"collect","error":str(error)})
        raise


def validate(item, root):
    require_host(); path=verified_path(item,root); value=read(path)
    if (value.get("format")!=FORMAT or value.get("status")!="PASS" or value.get("target_host")!="149"
            or any(value.get(k) is not False for k in ("training_started","model_forwards_executed","source_ancestry_complete","formal_increment_established"))):
        raise ValueError("Invalid genomic workflow result")
    _,_,prepared=validate_preparation(value["preparation"],path.parent)
    for name in ("completed","reassembly","adapter_input","collection_audit","fusion_binding","fusion_experts"):verified_path(value[name],path.parent)
    from .native_auxiliary_fusion import load_native_auxiliary
    _,adapter=load_native_auxiliary(value["fusion_binding"]["path"],expected_sha=value["fusion_binding"]["sha256"],expected_core_run_id=prepared["core_run_id"])
    manifest=read(value["adapter_input"]["path"])
    if (adapter["input_manifest"]!=value["adapter_input"] or manifest["native_receipt"]!=value["completed"]
            or manifest["reassembly_receipt"]!=value["reassembly"] or manifest["workflow_preparation"]!=value["preparation"]
            or manifest["workflow_audit"]!=value["collection_audit"] or any(value[k]!=prepared[k] for k in ("native_run_id","core_run_id"))
            or value["candidate_rows"]!=adapter["rows"] or value["fold_rows"]!=adapter["fold_rows"]):
        raise ValueError("Genomic workflow result bindings differ")
    experts=[{"spec":s,"prediction":adapter["outputs"]["fusion"],"binding_audit":value["fusion_binding"],
              "run_id":adapter["run_id"],"estimand":adapter["estimand"]} for s in adapter["expert_specs"]]
    if read(value["fusion_experts"]["path"])!=experts: raise ValueError("Genomic fusion experts differ")
    return path,value
