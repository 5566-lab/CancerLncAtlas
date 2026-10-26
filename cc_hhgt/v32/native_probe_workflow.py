"""Official HM450 reference to independent coordinate topology and probe map.

Supplied promoter/peak coordinates are hypotheses; their biological annotation
ancestry is not established by this workflow. No patient data are consumed.
"""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, record
from .current_analysis_workflow import write
from .context_table_replay import compare_tables

ROOT=Path(__file__).resolve().parents[2]
INPUT="NATIVE_PROBE_WORKFLOW_INPUT_V1"
FORMAT="NATIVE_PROBE_WORKFLOW_RESULTS_V1"
LINKS="INDEPENDENT_DISTAL_LINKS.tsv.gz"
MAP="HM450_LNCRNA_REGION_PROBES.parquet"
CODE=("cc_hhgt/v32/native_probe_workflow.py","scripts/run_native_probe_workflow.py",
      "scripts/build_v32_independent_distal_topology.py","scripts/materialize_v32_hm450_lncrna_probe_map.py",
      "cc_hhgt/v32/distal_topology.py","cc_hhgt/v32/context_table_replay.py")
_REPLAY_CACHE={}


def plan(manifest,output,*,new=True):
    require_host();manifest=Path(manifest).resolve();output=Path(output).resolve();spec=read(manifest)
    names={"hm450_manifest","promoters","required_genes","peak_set"}
    if set(spec)-names-{"format","max_distance_bp"} or not names<=set(spec) or spec.get("format")!=INPUT:
        raise ValueError("Invalid native probe workflow")
    if new and output.exists():raise FileExistsError("Use a new probe workflow directory")
    distance=spec.get("max_distance_bp",1_000_000)
    if type(distance) is not int or distance<1:raise ValueError("Probe distance must be a positive integer")
    files={key:record(verified_path(spec[key],manifest.parent)) for key in sorted(names)}
    from scripts.materialize_v32_hm450_lncrna_probe_map import checksum,EXPECTED_GDC_HM450_MD5
    if checksum(Path(files["hm450_manifest"]["path"]),"md5")!=EXPECTED_GDC_HM450_MD5:
        raise ValueError("Official GDC HM450 manifest MD5 mismatch")
    if any(output==Path(pin["path"]) or output in Path(pin["path"]).parents for pin in files.values()):
        raise ValueError("Probe output must not contain input files")
    return {"input":record(manifest),"target_host":"149","training_started":False,
            "files":files,"max_distance_bp":distance,"coordinate_build":"GRCh38",
            "promoter_definition":"INPUT_BED0_PROMOTERS_SOURCE_BOUND",
            "implementations":[record(ROOT/name) for name in CODE],
            "source_ancestry_complete":False,"raw_deletion_executed":False}


def invoke(value,output):
    from scripts.build_v32_independent_distal_topology import main as build
    from scripts.materialize_v32_hm450_lncrna_probe_map import main as map_probes
    shared=[part for key in ("promoters","required_genes","peak_set")
            for part in ("--"+key.replace("_","-"),value["files"][key]["path"])]
    if build(shared+["--reference-build","GRCh38","--max-distance-bp",str(value["max_distance_bp"]),
                     "--output",str(output/"topology")])!=0:raise RuntimeError("Independent topology failed")
    if map_probes(shared+["--hm450-manifest",value["files"]["hm450_manifest"]["path"],
                         "--topology-links",str(output/"topology"/LINKS),
                         "--promoter-definition",value["promoter_definition"],
                         "--output",str(output/"features")])!=0:raise RuntimeError("Probe mapping failed")


def replay(value,output):
    """Reconstruct both producers, including zero-link coverage and audit metadata."""
    audit=read(output/"features/AUDIT.json")
    actual=verified_path(audit["output"],output/"features")
    if actual!=output/"features"/MAP:raise ValueError("Probe output location differs")
    with tempfile.TemporaryDirectory(prefix="native_probe_replay_") as temporary:
        expected_root=Path(temporary)
        with (expected_root/"replay.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):
            invoke(value,expected_root)
        for name in (LINKS,"GENE_COVERAGE.tsv"):
            if (output/"topology"/name).read_bytes()!=(expected_root/"topology"/name).read_bytes():
                raise ValueError("Probe topology replay differs")
        actual_topology=read(output/"topology"/(LINKS+".audit.json"))
        expected_topology=read(expected_root/"topology"/(LINKS+".audit.json"))
        expected_topology["output"]=str(output/"topology"/LINKS)
        if actual_topology!=expected_topology:raise ValueError("Probe topology metadata differs from replay")
        expected=read(expected_root/"features/AUDIT.json")
        expected["inputs"]["datas7_links"]["path"]=str(output/"topology"/LINKS)
        expected["topology_lineage"]["audit_sha256"]=record(output/"topology"/(LINKS+".audit.json"))["sha256"]
        omit={"output","native_probe_preparation"}
        if {k:v for k,v in audit.items() if k not in omit}!={k:v for k,v in expected.items() if k not in omit}:
            raise ValueError("Probe metadata differs from native replay")
        compare_tables(actual,expected_root/"features"/MAP,["probe_id","lncrna_id","region_type"],label="Probe")
    return audit


def validate_map(probe_map):
    require_host();probe_map=Path(probe_map).resolve();audit_path=probe_map.parent/"AUDIT.json";audit=read(audit_path)
    preflight=verified_path(audit["native_probe_preparation"],audit_path.parent);output=preflight.parent
    if preflight!=output/"LAUNCH_PREFLIGHT.json" or probe_map!=output/"features"/MAP:
        raise ValueError("Native probe preparation location differs")
    value=read(preflight)
    if value!=plan(verified_path(value["input"],output),output,new=False):
        raise ValueError("Probe sources or implementation differ")
    sources=[record(preflight),value["input"],*value["files"].values(),*value["implementations"],
            record(audit_path),record(probe_map),record(output/"topology"/LINKS),
            record(output/"topology"/(LINKS+".audit.json")),record(output/"topology/GENE_COVERAGE.tsv")]
    # Always re-read every input/output/code byte pin before reusing an in-process
    # proof. Never persist a cache token that could replace reconstruction.
    signature={"value":value,"sources":sources}
    if _REPLAY_CACHE.get(str(output))!=signature:
        replay(value,output);_REPLAY_CACHE[str(output)]=signature
    return sources


def run(manifest,output):
    value=plan(manifest,output);output=Path(output).resolve();output.mkdir(parents=True)
    preflight=output/"LAUNCH_PREFLIGHT.json";write(preflight,value)
    try:
        with (output/"stage.log").open("w",encoding="utf-8") as log,redirect_stdout(log),redirect_stderr(log):
            invoke(value,output);audit=replay(value,output)
        if value!=plan(manifest,output,new=False):raise ValueError("Probe sources changed during execution")
        audit["native_probe_preparation"]=record(preflight);write(output/"features/AUDIT.json",audit)
        binding=output/"NATIVE_PROBE_BINDING.json"
        write(binding,{"format":FORMAT,"status":"PASS","target_host":"149","training_started":False,
                       "preflight":record(preflight),"audit":record(output/"features/AUDIT.json"),
                       "probe_map":record(output/"features"/MAP),"topology_links":record(output/"topology"/LINKS),
                       "native_numerical_replay_verified":True,"source_ancestry_complete":False,
                       "raw_deletion_executed":False,"production_deployed":False})
        return binding
    except Exception as error:
        write(output/"RUN_FAILED.json",{"target_host":"149","training_started":False,"error":str(error)});raise


def validate(item,root):
    require_host();path=verified_path(item,root);binding=read(path)
    if (binding.get("format")!=FORMAT or binding.get("status")!="PASS" or binding.get("target_host")!="149"
            or binding.get("native_numerical_replay_verified") is not True
            or any(binding.get(k) is not False for k in ("training_started","source_ancestry_complete","raw_deletion_executed","production_deployed"))):
        raise ValueError("Incomplete native probe workflow")
    expected={"preflight":path.parent/"LAUNCH_PREFLIGHT.json","audit":path.parent/"features/AUDIT.json",
              "probe_map":path.parent/"features"/MAP,"topology_links":path.parent/"topology"/LINKS}
    for key,location in expected.items():
        if verified_path(binding[key],path.parent)!=location:raise ValueError("Probe handoff location differs")
    if read(expected["audit"])["native_probe_preparation"]!=binding["preflight"]:
        raise ValueError("Probe preflight differs")
    validate_map(expected["probe_map"])
    return path,binding
