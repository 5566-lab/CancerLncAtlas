"""Raw MC3/chain to coordinate-bound distal mutation and checked consumers."""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import tempfile
import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read,record
from .current_analysis_workflow import write
from .context_table_replay import compare_tables
from . import mc3_sources

ROOT=Path(__file__).resolve().parents[2]
INPUT="NATIVE_DISTAL_MUTATION_WORKFLOW_INPUT_V1"
FORMAT="NATIVE_DISTAL_MUTATION_WORKFLOW_RESULTS_V1"
FEATURE="patient_lncrna_distal_mutation.parquet"
CODE=("cc_hhgt/v32/native_distal_mutation_workflow.py","scripts/run_native_distal_mutation_workflow.py",
      "cc_hhgt/v32/mc3_sources.py","scripts/materialize_v32_distal_mutation_lncrna.py",
      "cc_hhgt/v32/distal_topology.py","cc_hhgt/v32/methylation_recovery.py","cc_hhgt/v32/context_table_replay.py")
_REPLAY_CACHE={}


def plan(manifest,output,*,new=True):
    require_host();manifest=Path(manifest).resolve();output=Path(output).resolve();spec=read(manifest)
    names={"mc3","chain","patient_folds","peak_set","topology_links"}
    if set(spec)!=names|{"format"} or spec["format"]!=INPUT:raise ValueError("Invalid native distal mutation manifest")
    if new and output.exists():raise FileExistsError("Use a new distal mutation output")
    files={key:record(verified_path(spec[key],manifest.parent)) for key in sorted(names)}
    from scripts.materialize_v32_distal_mutation_lncrna import EXPECTED_MC3_SHA256,EXPECTED_CHAIN_SHA256
    if files["mc3"]["sha256"]!=EXPECTED_MC3_SHA256:raise ValueError("MC3 source SHA256 differs from pinned authority")
    if files["chain"]["sha256"]!=EXPECTED_CHAIN_SHA256:raise ValueError("hg19ToHg38 chain SHA256 differs from pinned authority")
    mc3_sources.patient_authority(files["patient_folds"]["path"])
    from .distal_topology import topology_lineage
    lineage=topology_lineage(Path(files["topology_links"]["path"]))
    if lineage.get("predictive_evaluation_eligible") is not True:raise ValueError("Independent mutation topology required")
    sources=[*files.values(),record(Path(files["topology_links"]["path"]+".audit.json"))]
    if any(output==Path(p["path"]) or output in Path(p["path"]).parents for p in sources):
        raise ValueError("Mutation output must not contain inputs")
    return {"input":record(manifest),"target_host":"149","training_started":False,"files":files,
            "topology_lineage":lineage,"source_files":sources,"implementations":[record(ROOT/n) for n in CODE],
            "source_ancestry_complete":False,"raw_deletion_executed":False}


def invoke(value,output):
    from scripts.materialize_v32_distal_mutation_lncrna import main
    files=value["files"];output.mkdir(parents=True,exist_ok=True)
    stats=mc3_sources.prepare(files["mc3"]["path"],files["chain"]["path"],files["patient_folds"]["path"],output/"inputs")
    write(output/"inputs/SOURCE_AUDIT.json",{"target_host":"149","training_started":False,**stats})
    roles={k:pin["path"] for k,pin in files.items()}
    roles.update(variant_events=str(output/"inputs/variant_event.parquet"),
                 lifted_coordinates=str(output/"inputs/variant_coordinate_grch38.parquet"),
                 sample_crosswalk=str(output/"inputs/mutation_sample_crosswalk.parquet"))
    args=[part for key,path in roles.items() for part in ("--"+key.replace("_","-"),path)]
    if main(args+["--output",str(output/"features")])!=0:raise RuntimeError("Distal mutation producer failed")


def replay(value,output):
    audit=read(output/"features/AUDIT.json")
    with tempfile.TemporaryDirectory(prefix="native_distal_mutation_replay_") as temporary:
        expected_root=Path(temporary)
        with (expected_root/"replay.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):invoke(value,expected_root)
        expected=read(expected_root/"features/AUDIT.json")
        if read(output/"inputs/SOURCE_AUDIT.json")!=read(expected_root/"inputs/SOURCE_AUDIT.json"):
            raise ValueError("MC3 source metadata differs from replay")
        for key,name,keys in (("variant_events","variant_event.parquet",["source_row"]),
                              ("lifted_coordinates","variant_coordinate_grch38.parquet",["variant_id"]),
                              ("sample_crosswalk","mutation_sample_crosswalk.parquet",["cancer_id","patient_id"])):
            actual=output/"inputs"/name
            if verified_path(audit["inputs"][key],output)!=actual:raise ValueError("MC3 generated input location differs")
            compare_tables(actual,expected_root/"inputs"/name,keys,label="MC3 input")
            expected["inputs"][key]={**record(actual)}
        omit={"outputs","native_distal_mutation_preparation"}
        if {k:v for k,v in audit.items() if k not in omit}!={k:v for k,v in expected.items() if k not in omit}:
            raise ValueError("Distal mutation metadata differs from replay")
        keys={FEATURE:["cancer_id","patient_id","lncrna_id"],"POSITIVE_DISTAL_MUTATION_EVENTS.parquet":["cancer_id","patient_id","gene_id"],
              "PATIENT_MUTATION_ASSAY.parquet":["cancer_id","patient_id"],"DISTAL_LNCRNA_CATALOG.parquet":["gene_id"]}
        if set(audit["outputs"])!=set(keys):raise ValueError("Distal mutation output inventory differs")
        for name,columns in keys.items():
            actual=verified_path(audit["outputs"][name],output)
            if actual!=output/"features"/name:raise ValueError("Distal mutation output location differs")
            compare_tables(actual,expected_root/"features"/name,columns,label="Distal mutation")
    return audit


def validate_features(path):
    require_host();path=Path(path).resolve();audit_path=path.parent/"AUDIT.json";audit=read(audit_path)
    preflight=verified_path(audit["native_distal_mutation_preparation"],path.parent);output=preflight.parent;value=read(preflight)
    if path!=output/"features"/FEATURE or preflight!=output/"LAUNCH_PREFLIGHT.json":raise ValueError("Mutation preparation location differs")
    if value!=plan(verified_path(value["input"],output),output,new=False):raise ValueError("Mutation sources or implementation differ")
    produced=[output/"inputs"/name for name in ("variant_event.parquet","variant_coordinate_grch38.parquet","mutation_sample_crosswalk.parquet","SOURCE_AUDIT.json")]
    produced+=[verified_path(pin,output) for pin in audit["outputs"].values()]
    sources=[record(preflight),value["input"],*value["source_files"],*value["implementations"],record(audit_path),*[record(p) for p in produced]]
    signature={"value":value,"sources":sources}
    if _REPLAY_CACHE.get(str(output))!=signature:
        replay(value,output);_REPLAY_CACHE[str(output)]=signature
    return sources


def run(manifest,output):
    value=plan(manifest,output);output=Path(output).resolve();output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",value)
    try:
        with (output/"stage.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):
            invoke(value,output);audit=replay(value,output)
        if value!=plan(manifest,output,new=False):raise ValueError("Mutation sources changed during execution")
        audit["native_distal_mutation_preparation"]=record(output/"LAUNCH_PREFLIGHT.json");write(output/"features/AUDIT.json",audit)
        binding=output/"NATIVE_DISTAL_MUTATION_BINDING.json"
        write(binding,{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
            "preflight":record(output/"LAUNCH_PREFLIGHT.json"),"audit":record(output/"features/AUDIT.json"),
            "distal_mutation":record(output/"features"/FEATURE),"native_numerical_replay_verified":True,
            "source_ancestry_complete":False,"raw_deletion_executed":False,"production_deployed":False})
        return binding
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise


def validate_distal_input(prepared,root):
    item=prepared.get("sources",{}).get("distal_mutation")
    if item is None:return []
    path=verified_path(item,root);audit_path=path.parent/"AUDIT.json"
    if not audit_path.is_file() or "native_distal_mutation_preparation" not in read(audit_path):return []
    dependencies=validate_features(path);keys=["cancer_id","patient_id","lncrna_id"]
    columns=keys+["patient_fold_id","distal_mutation_burden","distal_mutation_burden__available"]
    features=pd.read_parquet(path,columns=columns)
    for pin in prepared["outputs"]:
        actual=pd.read_parquet(verified_path(pin,root),columns=columns)
        expected=actual[keys].merge(features,on=keys,how="left",validate="one_to_one")
        try:pd.testing.assert_frame_equal(actual.reset_index(drop=True),expected,check_dtype=False,rtol=0,atol=1e-12)
        except AssertionError as error:raise ValueError("Distal mutation projection differs from raw replay") from error
    return dependencies


def validate(item,root):
    require_host();path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or binding.get("native_numerical_replay_verified") is not True
            or any(binding.get(k) is not False for k in ("training_started","source_ancestry_complete","raw_deletion_executed","production_deployed"))):
        raise ValueError("Incomplete native distal mutation workflow")
    for key,target in (("preflight","LAUNCH_PREFLIGHT.json"),("audit","features/AUDIT.json"),("distal_mutation","features/"+FEATURE)):
        if verified_path(binding[key],path.parent)!=path.parent/target:raise ValueError("Mutation handoff location differs")
    if read(path.parent/"features/AUDIT.json")["native_distal_mutation_preparation"]!=binding["preflight"]:
        raise ValueError("Mutation preflight differs")
    validate_features(binding["distal_mutation"]["path"])
    return path,binding
