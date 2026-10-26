"""Run cached native KD/RBNS analyses and replay their actual raw inputs.

These analyses are independent of the core model. Numerical reconstruction is
not an independent biological validation and never admits predicted binding as
an experimental edge. This workflow does not download data or fit a model.
"""
from contextlib import redirect_stdout, redirect_stderr
import importlib
from pathlib import Path
import tempfile

import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, record
from .current_analysis_workflow import write
from .input_lineage import artifact_sha256

ROOT=Path(__file__).resolve().parents[2]
INPUT="NATIVE_RBP_WORKFLOW_INPUT_V1"
FORMAT="NATIVE_RBP_WORKFLOW_RESULTS_V1"
CONTRACT="CACHED_RAW_RBP_ANALYSIS_V1"
KINDS={"encode_functional":("analyze_v32_encode_functional","ENCODE_FUNCTIONAL_ANALYSIS_V1"),
       "rbns_sequence":("analyze_v32_rbns_sequences","RBNS_SEQUENCE_ANALYSIS_V1")}
ROLES={"encode_functional":("inventory","gene_crosswalk","rbp_crosswalk","pathway_membership","physical_binding","exact_candidates"),
       "rbns_sequence":("inventory","rbp_crosswalk","transcript_fasta","transcript_crosswalk","lncrnas","protein_gene_map")}
CODE=("cc_hhgt/v32/native_rbp_workflow.py","scripts/run_native_rbp_workflow.py",
      "scripts/analyze_v32_encode_functional.py","scripts/analyze_v32_rbns_sequences.py",
      "cc_hhgt/v32/encode_functional.py","cc_hhgt/v32/rbns_sequence.py",
      "cc_hhgt/v32/rbp_typed_binding.py","cc_hhgt/v32/rbp_assay.py","cc_hhgt/v32/input_lineage.py")


def sources(kind,manifest,cache):
    from .encode_functional import verified_encode_file
    from scripts.analyze_v32_encode_functional import read_table
    manifest=Path(manifest).resolve();cache=Path(cache).resolve();spec=read(manifest)
    if kind not in KINDS or spec.get("format")!=KINDS[kind][1]:raise ValueError("RBP analysis kind differs from its manifest")
    if not cache.is_dir():raise FileNotFoundError("RBP workflow requires an existing verified cache")
    artifacts={}
    for key in ROLES[kind]:
        pin=spec["artifacts"][key];path=(manifest.parent/pin["path"]).resolve()
        digest=artifact_sha256(path)
        if digest!=pin["sha256"]:raise ValueError("RBP artifact checksum differs: "+key)
        artifacts[key]={"path":str(path),"sha256":digest}
    inventory=read_table(Path(artifacts["inventory"]["path"]))
    columns=["file_accession","experiment_accession","rbp","output_type","href","md5sum","file_size","status"]
    if kind=="encode_functional":columns.append("biosample")
    inventory=inventory[columns].drop_duplicates()
    if inventory.file_accession.duplicated().any():raise ValueError("Conflicting native RBP inventory")
    inventory=inventory.set_index("file_accession",drop=False)
    selected=spec["contrasts" if kind=="encode_functional" else "profiles"]
    if not selected:raise ValueError("No native RBP analyses selected")
    raw={}
    for accession in sorted({s["file_accession"] for s in selected}):
        if accession not in inventory.index:raise ValueError("Selected RBP source is absent from inventory")
        row=inventory.loc[accession].to_dict()
        path=verified_encode_file(row,cache,download=False)
        raw[accession]={**record(path),"md5":row["md5sum"]}
    return {"analysis":record(manifest),"artifacts":artifacts,"raw":raw,"cache":str(cache)}


def invoke(kind,manifest,cache,output,chunk_rows):
    args=["--manifest",str(manifest),"--cache",str(cache),"--output",str(output)]
    if kind=="rbns_sequence":args += ["--chunk-rows",str(chunk_rows)]
    if importlib.import_module("scripts."+KINDS[kind][0]).main(args)!=0:
        raise RuntimeError("Native RBP analysis failed")


def replay_audit(item,root):
    """Compare every native output and metadata field to a raw-cache replay."""
    require_host();path=verified_path(item,root);audit=read(path)
    kinds={"ENCODE_FUNCTIONAL_ANALYSIS_AUDIT_V1":"encode_functional","RBNS_SEQUENCE_ANALYSIS_AUDIT_V1":"rbns_sequence"}
    kind=kinds.get(audit.get("format"))
    if (kind is None or audit.get("source_contract")!=CONTRACT or audit.get("status")!="PASS"
            or audit.get("target_host")!="149" or audit.get("training_started") is not False):
        raise ValueError("RBP audit lacks native raw-source reconstruction")
    manifest=verified_path(audit["input_manifest"],path.parent)
    before=sources(kind,manifest,audit["cache_root"])
    if audit["source_files"]!=before["raw"]:raise ValueError("RBP audited raw files differ from selected sources")
    if read(path.parent/"SUCCESS.json")!={"status":"PASS","audit_sha256":record(path)["sha256"]}:
        raise ValueError("RBP completion differs from native audit")
    chunk=audit.get("chunk_rows",50000)
    if type(chunk) is not int or chunk<1:raise ValueError("Invalid RBNS replay chunk size")
    with tempfile.TemporaryDirectory(prefix="rbp_native_replay_") as temporary:
        output=Path(temporary)/"analysis"
        invoke(kind,manifest,before["cache"],output,chunk)
        expected=read(output/"AUDIT.json")
        if {k:v for k,v in audit.items() if k!="outputs"}!={k:v for k,v in expected.items() if k!="outputs"}:
            raise ValueError("RBP native analysis metadata differs from raw replay")
        if set(audit["outputs"])!=set(expected["outputs"]):raise ValueError("RBP output inventory differs from raw replay")
        for name,pin in audit["outputs"].items():
            target=verified_path(pin,path.parent)
            if target!=path.parent/name:raise ValueError("RBP output is outside its native analysis")
            actual=pd.read_parquet(target);reference=pd.read_parquet(output/name)
            if pin.get("rows")!=len(actual):raise ValueError("RBP native output row count differs")
            columns=list(reference.columns)
            if list(actual.columns)!=columns:raise ValueError("RBP native output schema differs")
            if columns:
                actual=actual.sort_values(columns,na_position="last",kind="stable").reset_index(drop=True)
                reference=reference.sort_values(columns,na_position="last",kind="stable").reset_index(drop=True)
            try:pd.testing.assert_frame_equal(actual,reference,rtol=1e-10,atol=1e-12)
            except AssertionError as error:raise ValueError("RBP native output differs from raw replay: "+name) from error
    if sources(kind,manifest,audit["cache_root"])!=before or record(path)["sha256"]!=item["sha256"]:
        raise ValueError("RBP sources changed during replay")
    return path,audit


def plan(manifest,output,*,new=True):
    require_host();manifest=Path(manifest).resolve();spec=read(manifest);output=Path(output).resolve()
    required={"format","kind","analysis","cache"}
    if (not required<=set(spec) or set(spec)-required-{"chunk_rows"} or spec["format"]!=INPUT
            or spec["kind"] not in KINDS):raise ValueError("Invalid native RBP workflow")
    if new and output.exists():raise FileExistsError("Use a new RBP analysis directory")
    chunk=spec.get("chunk_rows",50000)
    if type(chunk) is not int or chunk<1 or (spec["kind"]!="rbns_sequence" and "chunk_rows" in spec):
        raise ValueError("Invalid native RBP chunk setting")
    analysis=verified_path(spec["analysis"],manifest.parent)
    context=sources(spec["kind"],analysis,manifest.parent/spec["cache"])
    for pin in [context["analysis"],*context["artifacts"].values(),*context["raw"].values()]:
        source=Path(pin["path"])
        if output==source or source.is_dir() and source in output.parents:
            raise ValueError("RBP output must be outside its input datasets")
    if output==Path(context["cache"]) or Path(context["cache"]) in output.parents:
        raise ValueError("RBP output must be outside the raw cache")
    return {"input":record(manifest),"target_host":"149","training_started":False,"kind":spec["kind"],
        "sources":context,"chunk_rows":chunk,"download_started":False,"implementations":[record(ROOT/n) for n in CODE]}


def run(manifest,output):
    value=plan(manifest,output);output=Path(output).resolve();output.mkdir(parents=True)
    write(output/"LAUNCH_PREFLIGHT.json",value)
    try:
        context=value["sources"]
        with (output/"analysis.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):
            invoke(value["kind"],context["analysis"]["path"],context["cache"],output/"analysis",value["chunk_rows"])
        native=output/"analysis/AUDIT.json";_,audit=replay_audit(record(native),output)
        if value!=plan(manifest,output,new=False):raise ValueError("RBP workflow sources changed during execution")
        binding=output/"NATIVE_RBP_BINDING.json"
        write(binding,{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
            "preflight":record(output/"LAUNCH_PREFLIGHT.json"),"kind":value["kind"],"audit":record(native),
            "outputs":audit["outputs"],"raw_numerical_replay_verified":True,"download_started":False,
            "independent_predictive_performance_verified":False,"production_deployed":False})
        return binding
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise


def validate(item,root):
    require_host();path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or binding.get("raw_numerical_replay_verified") is not True
            or any(binding.get(k) is not False for k in ("training_started","download_started",
                "independent_predictive_performance_verified","production_deployed"))):
        raise ValueError("Incomplete native RBP workflow")
    value=read(verified_path(binding["preflight"],path.parent))
    if value!=plan(verified_path(value["input"],path.parent),path.parent,new=False):
        raise ValueError("RBP workflow inputs or implementation differ")
    native,audit=replay_audit(binding["audit"],path.parent)
    if (native!=path.parent/"analysis/AUDIT.json" or binding["kind"]!=value["kind"]
            or audit["input_manifest"]!=value["sources"]["analysis"] or binding["outputs"]!=audit["outputs"]):
        raise ValueError("RBP workflow native handoff differs")
    return path,binding
