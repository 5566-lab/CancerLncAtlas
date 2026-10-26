"""Map supplied single-cell H5 into native metacell inputs without fitting.

Reuses the original r7/r11 input contract and matrix mapper. No raw acquisition,
annotation/QC inference, PCA, LASSO, neural optimization or deletion is run.
"""
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from scipy import sparse

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read,record
from .current_analysis_workflow import write
from .context_table_replay import compare_tables
from .input_lineage import artifact_sha256

ROOT=Path(__file__).resolve().parents[2]
INPUT="NATIVE_SINGLE_CELL_MAPPING_WORKFLOW_INPUT_V1"
FORMAT="NATIVE_SINGLE_CELL_MAPPING_WORKFLOW_RESULTS_V1"
CODE=("cc_hhgt/v32/native_single_cell_mapping_workflow.py","scripts/run_native_single_cell_mapping_workflow.py",
      "scripts/materialize_single_cell_metacell_149.py","scripts/run_v32_single_cell_verified_streaming.py",
      "cc_hhgt/v32/single_cell_r7_streaming.py","cc_hhgt/v32/single_cell_cell_level.py",
      "cc_hhgt/v32/single_cell_preprocessing.py","cc_hhgt/v32/single_cell_verified_metadata.py",
      "cc_hhgt/v32/context_table_replay.py")
SPARSE=("CODING_MEASURED.npz","LNC_MEASURED.npz","CODING_LOG_FOR_GROUPING.npz","EXACT_SIGNATURE_MATRIX.npz")
ARRAYS=("ALL_GENE_LIBRARY.npy","SIGNATURE_GENE_COUNTS.npy","MISSING_GENE_COUNTS.npy")
FILES={*SPARSE,*ARRAYS,"IDS.json","CELL_METADATA.parquet"}
_REPLAY_CACHE={}


def upstream(value):
    from scripts import run_v32_single_cell_verified_streaming as native
    args=native.build_parser().parse_args(["--mode","plan","--r7-root",value["r7_root"],
        "--run-status-json",value["run_status"]["path"],"--expected-run-status-sha256",value["run_status"]["sha256"],
        "--server-preflight-json",value["server_preflight"]["path"],"--expected-server-preflight-sha256",value["server_preflight"]["sha256"],
        "--cancer-id",value["cancer_id"],"--max-rank",str(value["max_rank"])])
    return native,native._load_contract(args),Path(value["r7_root"])


def plan(manifest,output,*,new=True):
    require_host();manifest=Path(manifest).resolve();output=Path(output).resolve();spec=read(manifest)
    required={"format","cancer_id","r7_root","run_status","server_preflight"}
    if not required<=set(spec) or set(spec)-required-{"max_rank"} or spec["format"]!=INPUT:
        raise ValueError("Invalid single-cell mapping manifest")
    if new and output.exists():raise FileExistsError("Use a new single-cell mapping output")
    maximum=spec.get("max_rank",1500)
    if type(maximum) is not int or maximum<1:raise ValueError("Single-cell rank limit must be a positive integer")
    root=(manifest.parent/spec["r7_root"]).resolve()
    if root==output or root in output.parents:raise ValueError("Single-cell output must be outside the input root")
    value={"cancer_id":spec["cancer_id"],"r7_root":str(root),"max_rank":maximum,
           **{key:record(verified_path(spec[key],manifest.parent)) for key in ("run_status","server_preflight")}}
    if read(value["server_preflight"]["path"]).get("target_host")!="149":raise ValueError("Upstream preflight must target 149")
    _,bundle,_=upstream(value)
    value["cancer_id"]=bundle["cancer_id"]
    sources=[*bundle["input_paths"].values(),root/"TRAINING_HANDOFF.json",root/"dataset_manifest_33c.parquet"]
    if any(output==Path(p).resolve() or output in Path(p).resolve().parents for p in sources):
        raise ValueError("Single-cell output must not contain raw sources")
    return {**value,"input":record(manifest),"target_host":"149","training_started":False,
        "contract":bundle["contract"],"contract_sha256":bundle["contract_sha256"],
        "source_files":[record(p) for p in sorted(set(sources),key=str)],
        "implementations":[record(ROOT/name) for name in CODE],"source_ancestry_complete":False,
        "raw_deletion_executed":False,"measurement_scale":bundle["measurement_scale"]}


def invoke(value,output):
    from scripts.materialize_single_cell_metacell_149 import _materialize_mapped_matrices
    target=output/"mapped"/value["cancer_id"]
    return _materialize_mapped_matrices(value["cancer_id"],target,Path(value["r7_root"]),float("inf"),upstream=upstream(value))


def replay(value,output):
    actual_root=output/"mapped"/value["cancer_id"];actual=read(actual_root/"MAPPED_SUCCESS.json")
    if set(actual["files"])!=FILES:raise ValueError("Single-cell mapped file inventory differs")
    for name,pin in actual["files"].items():
        target=verified_path(pin,actual_root)
        if target!=actual_root/name or pin.get("bytes")!=target.stat().st_size:raise ValueError("Single-cell mapped file identity differs")
    with tempfile.TemporaryDirectory(prefix="native_sc_mapping_replay_") as temporary:
        other=Path(temporary)
        with (other/"replay.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):expected=invoke(value,other)
        expected_root=other/"mapped"/value["cancer_id"]
        omit={"files","finished_unix","native_mapping_preparation"}
        if {k:v for k,v in actual.items() if k not in omit}!={k:v for k,v in expected.items() if k not in omit}:
            raise ValueError("Single-cell mapping metadata differs from raw replay")
        if read(actual_root/"IDS.json")!=read(expected_root/"IDS.json"):raise ValueError("Single-cell mapped identities differ")
        try:
            pd.testing.assert_frame_equal(pd.read_parquet(actual_root/"CELL_METADATA.parquet"),
                pd.read_parquet(expected_root/"CELL_METADATA.parquet"),check_exact=True)
        except AssertionError as error:raise ValueError("Single-cell cell order/metadata differs from raw replay") from error
        for name in SPARSE:
            left=sparse.load_npz(actual_root/name);right=sparse.load_npz(expected_root/name)
            if left.shape!=right.shape or left.dtype!=right.dtype or (left!=right).nnz:
                raise ValueError("Single-cell sparse mapping replay differs: "+name)
        for name in ARRAYS:
            left=np.load(actual_root/name,allow_pickle=False);right=np.load(expected_root/name,allow_pickle=False)
            if left.dtype!=right.dtype or not np.array_equal(left,right):raise ValueError("Single-cell array mapping replay differs: "+name)
    return actual


def validate_mapped(path):
    require_host();path=Path(path).resolve();audit=read(path)
    preflight=verified_path(audit["native_mapping_preparation"],path.parent);value=read(preflight);output=preflight.parent
    if path!=output/"mapped"/value["cancer_id"]/"MAPPED_SUCCESS.json" or preflight!=output/"LAUNCH_PREFLIGHT.json":
        raise ValueError("Single-cell mapping preparation location differs")
    if value!=plan(verified_path(value["input"],output),output,new=False):raise ValueError("Single-cell mapping source or implementation differs")
    sources=[record(path),record(preflight),value["input"],value["run_status"],value["server_preflight"],
             *value["source_files"],*value["implementations"],*[record(verified_path(pin,path.parent)) for pin in audit["files"].values()]]
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
        if value!=plan(manifest,output,new=False):raise ValueError("Single-cell mapping source changed during execution")
        audit["native_mapping_preparation"]=record(output/"LAUNCH_PREFLIGHT.json")
        mapped=output/"mapped"/value["cancer_id"]/"MAPPED_SUCCESS.json";write(mapped,audit)
        path=output/"NATIVE_SINGLE_CELL_MAPPING_BINDING.json"
        write(path,{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
            "preflight":record(output/"LAUNCH_PREFLIGHT.json"),"mapped_success":record(mapped),
            "mapped_root":{"path":str(output/"mapped"),"sha256":artifact_sha256(output/"mapped")},
            "native_numerical_replay_verified":True,"source_ancestry_complete":False,
            "raw_deletion_executed":False,"production_deployed":False})
        return path
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise


def validate(item,root):
    require_host();path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or binding.get("native_numerical_replay_verified") is not True
            or any(binding.get(k) is not False for k in ("training_started","source_ancestry_complete","raw_deletion_executed","production_deployed"))):
        raise ValueError("Incomplete native single-cell mapping workflow")
    preflight=verified_path(binding["preflight"],path.parent);value=read(preflight)
    mapped=verified_path(binding["mapped_success"],path.parent)
    if (preflight!=path.parent/"LAUNCH_PREFLIGHT.json" or mapped!=path.parent/"mapped"/value["cancer_id"]/"MAPPED_SUCCESS.json"
            or read(mapped)["native_mapping_preparation"]!=binding["preflight"]
            or binding["mapped_root"]!={"path":str(path.parent/"mapped"),"sha256":artifact_sha256(path.parent/"mapped")}):
        raise ValueError("Single-cell mapping handoff differs")
    validate_mapped(mapped);return path,binding
