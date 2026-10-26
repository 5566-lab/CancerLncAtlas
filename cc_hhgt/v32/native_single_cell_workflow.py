"""Prepare native SC inputs and bind completed frozen outputs to signed fusion.

This workflow never dispatches fitting or promotes a global/current pointer.
Its source boundary is the existing native episode/static graph manifests.
"""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from pathlib import Path
import re
import sys

from .cnv_input_sources import require_host
from .current_core_export import verified_path, sha256
from .formal_expression_sources import read, write, record
from . import single_cell_numeric_replay as numeric

ROOT=Path(__file__).resolve().parents[2]
INPUT="NATIVE_SINGLE_CELL_WORKFLOW_INPUT_V1"
PREPARED="NATIVE_SINGLE_CELL_WORKFLOW_PREPARED_V1"
FORMAT="NATIVE_SINGLE_CELL_WORKFLOW_RESULTS_V1"
POINTER="NATIVE_SINGLE_CELL_WORKFLOW_POINTER_V1"
STATUS="PASS_SC_HHGT_WORKFLOW"
MODULE="scripts.train_single_cell_hhgt_149"
FILES={"episode_manifest","graph_asset_manifest","candidates","donor_crosswalk"}
CODE=(*numeric.CODE,"cc_hhgt/v32/native_single_cell_workflow.py",
      "scripts/run_native_single_cell_workflow.py","scripts/validate_single_cell_hhgt_training_149.py",
      "cc_hhgt/v32/single_cell_signed_fusion.py")


def donor_partitions(descriptors,crosswalk):
    """Unqualified episode donor arrays require unambiguous cross-study IDs."""
    from scripts import train_single_cell_hhgt_149 as native
    mappings={}
    for donor,canonical in crosswalk:mappings.setdefault(donor,set()).add(canonical)
    used=set();outer={};episodes=0
    for fold,items in descriptors.items():
        for descriptor in items:
            payload=native._load_weights_only(descriptor.path);episodes+=1;groups={}
            for key in ("outer_train_donors","outer_query_donors","source_donors","excluded_donors"):
                donors=set(payload[key]);used.update(donors)
                if any(len(mappings.get(d,()))!=1 for d in donors):
                    raise ValueError("SC episode donors lack complete unambiguous canonical mapping")
                groups[key]={next(iter(mappings[d])) for d in donors}
            if (groups["outer_train_donors"] & groups["outer_query_donors"]
                    or groups["source_donors"] & groups["excluded_donors"]):
                raise ValueError("SC canonical donor aliases cross an episode partition")
            for donor in groups["outer_query_donors"]:
                if donor in outer and outer[donor]!=fold:raise ValueError("SC canonical query donor crosses outer folds")
                outer[donor]=fold
            del payload
    return {"episode_files":episodes,"mapped_native_donor_ids":len(used),"canonical_outer_query_donors":len(outer),
        "complete_episode_donor_mapping":True,"canonical_episode_partitions_disjoint":True,
        "cross_module_patient_closure_verified":False}


def command(settings):
    return [sys.executable,"-m",MODULE,*[part for key,value in asdict(settings).items()
        if key!="prepare_only" and value is not None for part in ("--"+key.replace("_","-"),str(value))]]


def plan(manifest,output,*,new=True):
    from scripts import train_single_cell_hhgt_149 as native
    from .drug_sparse_query import _bounded_duckdb
    from .context_table_replay import relation
    require_host();manifest=Path(manifest).resolve();output=Path(output).resolve();spec=read(manifest)
    required={"format","run_id",*FILES}
    if (set(spec)-required-{"training_config","min_donors"} or not required<=set(spec)
            or spec["format"]!=INPUT or not isinstance(spec["run_id"],str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",spec["run_id"])):
        raise ValueError("Invalid native single-cell workflow input")
    if new and output.exists():raise FileExistsError("Use a new single-cell preparation directory")
    minimum=spec.get("min_donors",3)
    if type(minimum) is not int or minimum<2:raise ValueError("SC minimum donors must be an integer at least two")
    paths={k:verified_path(spec[k],manifest.parent) for k in sorted(FILES)}
    for key,name in (("episode_manifest","SC_HHGT_EPISODE_MANIFEST.json"),("graph_asset_manifest","GRAPH_ASSET_MANIFEST.json")):
        if paths[key].name!=name:raise ValueError("Native SC CLI requires "+name)
        if output==paths[key].parent or paths[key].parent in output.parents:
            raise ValueError("SC output cannot be inside an input dataset")
    fields=set(native.TrainingSettings.__dataclass_fields__)-{"episode_root","graph_assets_root","output_root","prepare_only"}
    overrides=spec.get("training_config",{})
    if not isinstance(overrides,dict) or set(overrides)-fields:raise ValueError("Unknown single-cell configuration")
    settings=native.TrainingSettings(episode_root=paths["episode_manifest"].parent,
        graph_assets_root=paths["graph_asset_manifest"].parent,output_root=output/"training",**overrides)
    native._validate_settings(settings)
    descriptors,lineage=native.load_episode_manifest(settings)
    sources={item["path"]:record(item["path"]) for item in numeric.source_inventory(lineage)}
    with _bounded_duckdb() as con:
        for key,columns in (("candidates",["cancer_id","lncrna_id","pathway_id"]),
                            ("donor_crosswalk",["dataset_id","query_donor_id","canonical_donor_id"])):
            con.execute(f"CREATE VIEW {key} AS SELECT {','.join(columns)} FROM {relation(paths[key])}")
            if con.execute(f"SELECT count(*) FROM {key}").fetchone()[0]==0:raise ValueError("Empty SC "+key)
            if con.execute(f"SELECT count(*) FROM {key} WHERE "+" OR ".join(f"{c} IS NULL OR trim({c})=''" for c in columns)).fetchone()[0]:
                raise ValueError("Missing SC identifiers")
            keys=columns if key=="candidates" else columns[:2]
            if con.execute(f"SELECT count(*) FROM (SELECT {','.join(keys)} FROM {key} GROUP BY ALL HAVING count(*)>1)").fetchone()[0]:
                raise ValueError("Duplicate SC identifiers")
        donor_check=donor_partitions(descriptors,con.execute("SELECT query_donor_id,canonical_donor_id FROM donor_crosswalk").fetchall())
    for path in [manifest,*paths.values(),*(ROOT/name for name in CODE)]:sources[str(path)]=record(path)
    from dataclasses import replace
    fit=command(settings);prepare=command(replace(settings,output_root=output/"input_preflight"))+["--prepare-only"]
    native.parse_args(fit[3:]);native.parse_args(prepare[3:])
    return {"target_host":"149","training_started":False,"input":record(manifest),"run_id":spec["run_id"],
        "files":{k:record(v) for k,v in paths.items()},"min_donors":minimum,"run_config":native._run_config(settings),
        "donor_partitions":donor_check,
        "sources":sorted(sources.values(),key=lambda r:r["path"]),"prepare_command":prepare,
        "execution_command":{"argv":fit,"operation":"SC_HHGT_FITTING","automatically_executed":False,
            "requires_separate_explicit_execution":True},"source_boundary":"PREMATERIALIZED_NATIVE_EPISODES_AND_STATIC_GRAPHS"}


def expected_preflight(value):
    from scripts import train_single_cell_hhgt_149 as native
    settings=native.parse_args(value["prepare_command"][3:]);descriptors,lineage=native.load_episode_manifest(settings)
    return numeric.collect_input_preflight(settings,descriptors,lineage)


def execute_preflight(argv,log):
    from scripts import train_single_cell_hhgt_149 as native
    if len(argv)<4 or argv[1:3]!=["-m",MODULE] or argv.count("--prepare-only")!=1:
        raise ValueError("SC workflow cannot dispatch fitting")
    with Path(log).open("w",encoding="utf-8") as handle,redirect_stdout(handle),redirect_stderr(handle):
        if native.main(argv[3:])!=0:raise RuntimeError("Native SC input collection failed")


def prepare(manifest,output):
    value=plan(manifest,output);output=Path(output).resolve();output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",value)
    try:
        execute_preflight(value["prepare_command"],output/"prepare.log")
        ready=output/"input_preflight/INPUT_PREFLIGHT.json"
        if read(ready)!=expected_preflight(value):raise ValueError("SC preparation numerical inputs differ")
        if value!=plan(manifest,output,new=False):raise ValueError("SC sources changed during preparation")
        write(output/"EXECUTION_COMMAND.json",value["execution_command"])
        path=output/"PREPARED.json"
        write(path,{"format":PREPARED,"status":"PASS_READY","target_host":"149","training_started":False,
            "plan":record(output/"LAUNCH_PREFLIGHT.json"),"input_preflight":record(ready),
            "execution_command":record(output/"EXECUTION_COMMAND.json")});return path
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise


def validate_preparation(item,root):
    require_host();path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=PREPARED or binding.get("status")!="PASS_READY"
            or binding.get("target_host")!="149" or binding.get("training_started") is not False):
        raise ValueError("Invalid single-cell preparation binding")
    value=read(verified_path(binding["plan"],path.parent))
    if value!=plan(verified_path(value["input"],path.parent),path.parent,new=False):raise ValueError("SC preparation source or command differs")
    ready=read(verified_path(binding["input_preflight"],path.parent))
    if ready!=expected_preflight(value):raise ValueError("SC prepared graph/row inputs differ")
    if read(verified_path(binding["execution_command"],path.parent))!=value["execution_command"]:
        raise ValueError("SC execution command differs")
    return path,binding,value,ready


def completed_matches(path,value,ready,success_path):
    if success_path!=path.parent/"training/SUCCESS.json":raise ValueError("SC completion is outside the prepared output")
    success=read(success_path)
    if (read(verified_path(success["files"]["input_preflight"],success_path.parent))!=ready
            or read(verified_path(success["files"]["run_config"],success_path.parent))!=value["run_config"]):
        raise ValueError("SC completion inputs or configuration differ from preparation")
    return success


def pointer_value(prepared,value,success_path,success,numerical,validation):
    return {"format":POINTER,"status":STATUS,"run_id":value["run_id"],"target_host":"149",
        "estimand":"signed_within_donor_metacell_spearman_rho",
        "split_policy":"donor_blocked_outer_oof_with_nested_inner_validation",
        "training_complete":True,"outer_folds":5,"primary_arm":"with_lasso","ablation_arm":"without_lasso",
        "workflow_preparation":record(prepared),"training_success":record(success_path),
        "numeric_replay":record(numerical),"independent_validation":record(validation),
        "oof_predictions":success["files"]["oof_predictions"],"input_manifest":value["files"]["episode_manifest"],
        "source_acquisition_ancestry_verified":False,"cross_module_patient_closure_verified":False,
        "optimizer_execution_independently_verified":False,"independent_predictive_performance_verified":False,
        "global_pointer_updated":False,"production_deployed":False}


def validate_pointer(pointer_path,*,replay=True):
    """Verify every dependency; optionally repeat the frozen neural forwards."""
    require_host();pointer_path=Path(pointer_path).resolve();pointer=read(pointer_path);bound={}
    def bind(item,root=pointer_path.parent):
        path=verified_path(item,root);bound[str(path)]=record(path);return path
    prepared,binding,value,ready=validate_preparation(pointer["workflow_preparation"],pointer_path.parent)
    success_path=bind(pointer["training_success"]);success=completed_matches(prepared,value,ready,success_path)
    numeric_path=bind(pointer["numeric_replay"]);result=read(numeric_path)
    validation=bind(pointer["independent_validation"])
    if pointer!=pointer_value(prepared,value,success_path,success,numeric_path,validation):raise ValueError("SC workflow pointer differs")
    if (result.get("format")!=numeric.FORMAT or result.get("status")!="PASS" or result.get("target_host")!="149"
            or result.get("training_success")!=record(success_path) or result.get("input_preflight")!=record(bind(success["files"]["input_preflight"]))
            or result.get("model_forwards_executed") is not True or result.get("parameters_unchanged") is not True
            or result.get("training_started") is not False or read(validation)!=result.get("native_validation")
            or result.get("implementation_sha256")!={name:sha256(ROOT/name) for name in (*numeric.CODE,"scripts/validate_single_cell_hhgt_training_149.py")}):
        raise ValueError("SC workflow numerical receipt differs")
    if replay and result!=numeric.replay_completed(success_path.parent,expected_success_sha256=sha256(success_path)):
        raise ValueError("SC saved numerical replay differs from actual frozen outputs")
    bind(record(prepared));bind(record(pointer_path))
    for key in ("plan","input_preflight","execution_command"):bind(binding[key],prepared.parent)
    for item in value["sources"]:bind(item,prepared.parent)
    for item in success["files"].values():bind(item,success_path.parent)
    manifest=read(bind(success["files"]["checkpoint_manifest"],success_path.parent))
    for model in manifest["models"]:
        bind(model["success"],success_path.parent)
        for item in model["files"].values():bind(item,success_path.parent)
    return value,list(bound.values())


def collect(preparation,completed,output):
    from .single_cell_signed_fusion import materialize_signed_sc_fusion
    require_host();output=Path(output).resolve()
    if output.exists():raise FileExistsError("Use a new single-cell collection directory")
    path,_,value,ready=validate_preparation(preparation,ROOT);success_path=verified_path(completed,ROOT)
    success=completed_matches(path,value,ready,success_path)
    if success_path.parent==output or success_path.parent in output.parents:
        raise ValueError("SC collection must be outside immutable training results")
    output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",{"target_host":"149","training_started":False,
        "preparation":record(path),"training_success":record(success_path)})
    try:
        numerical=numeric.audit(success_path.parent,expected_success_sha256=sha256(success_path),output=output/"numeric")
        validation=output/"INDEPENDENT_VALIDATION.json";write(validation,read(numerical)["native_validation"])
        pointer=output/"SC_WORKFLOW_POINTER.json";write(pointer,pointer_value(path,value,success_path,success,numerical,validation))
        materialize_signed_sc_fusion(pointer,Path(value["files"]["candidates"]["path"]),
            Path(value["files"]["donor_crosswalk"]["path"]),output/"signed",min_donors=value["min_donors"])
        binding=output/"NATIVE_SINGLE_CELL_BINDING.json"
        write(binding,{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
            "preparation":record(path),"training_success":record(success_path),"pointer":record(pointer),
            "numeric_replay":record(numerical),"signed_fusion":record(output/"signed/AUDIT.json"),
            "native_model_forwards_replayed":True,"source_acquisition_ancestry_verified":False,
            "cross_module_patient_closure_verified":False,"independent_predictive_performance_verified":False,
            "global_pointer_updated":False,"production_deployed":False});return binding
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise
