"""Declared raw segment inputs through the original compact genomic CNV producer.

The signed local-CNV input manifest is reused for file/patient identity only.
This producer retains the genomic event/absolute-burden estimand and six-array
layout. It does not substitute signed local-CNV statistics or fit any model.
"""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path, validate_current_export
from .formal_expression_sources import read, write, record
from . import native_directional_cnv as raw
from . import segment_cnv_streaming as streaming

INPUT = "NATIVE_GENOMIC_CNV_INPUT_V1"
REPLAY = "NATIVE_GENOMIC_CNV_NUMERIC_REPLAY_V1"
ROOT = Path(__file__).resolve().parents[2]
CODE = ("cc_hhgt/v32/native_genomic_cnv.py", "scripts/materialize_native_genomic_cnv.py",
        "cc_hhgt/v32/segment_cnv_streaming.py", "cc_hhgt/v32/current_core_export.py", *raw.CODE)
ARRAYS = tuple(f"{kind}_{feature}.npy" for kind in ("lncrna","pathway") for feature in ("event","callable","burden"))


def load(item, root):
    require_host(); path = verified_path(item, root); spec = read(path)
    required = {"format","segment_manifest","sample_gene_mutation","sample_lncrna_mutation","mc3","core_manifest","core_run_id"}
    if (not required <= set(spec) or set(spec)-required-{"max_memory_bytes"} or spec["format"] != INPUT
            or not isinstance(spec["core_run_id"], str) or not spec["core_run_id"].strip()):
        raise ValueError("Invalid native genomic CNV input")
    files = {k:verified_path(spec[k], path.parent) for k in required-{"format","core_run_id"}}
    loaded = raw.load_manifest(record(files["segment_manifest"]), path.parent)
    validate_current_export(files["core_manifest"], expected_run_id=spec["core_run_id"])
    memory = spec.get("max_memory_bytes", 536_870_912)
    if type(memory) is not int or memory <= 0: raise ValueError("Invalid native genomic CNV memory budget")
    bindings = {"candidates":loaded[2]["candidates"], "patient_folds":loaded[2]["patient_map"],
        "entity_intervals":loaded[2]["intervals"], "pathway_membership":loaded[2]["membership"],
        **{k:files[k] for k in ("sample_gene_mutation","sample_lncrna_mutation","mc3")},
        "core_embedding_manifest":files["core_manifest"]}
    sources = {c:raw.sources(loaded,c) for c in loaded[3]["cancers"]}
    context = {"format":streaming.FORMAT, "native_input":record(path), "segment_manifest":record(loaded[0]),
        "cohort":loaded[3], "bindings":{k:record(p) for k,p in bindings.items()},
        "implementation_sha256":{n:record(ROOT/n)["sha256"] for n in sorted(set(CODE))},
        "event_threshold":loaded[1].get("event_threshold",.30), "max_memory_bytes":memory,
        "max_segment_rows":loaded[1].get("max_segment_rows",2_000_000),
        "raw_segments":sources, "candidate_loading":"CANCER_FILTERED_NO_GLOBAL_OBJECT_TABLE",
        "observed_process_memory_gate_enforced":True, "source_ancestry_complete":False}
    annotation = raw.reference_context(loaded)
    if annotation is not None: context["interval_reference"] = annotation
    membership = raw.membership_context(loaded)
    if membership is not None: context["membership_reference"] = membership
    context["context_sha256"] = streaming._json_sha(context)
    return path, spec, loaded, context


def reader(loaded):
    def consume(cancer, patient):
        row = loaded[7][(cancer,patient)]
        frame = raw.read_segment(row, loaded[0].parent, loaded[1].get("max_segment_rows",2_000_000),
                                 coordinate_system=raw.coordinate_system(loaded[1]))
        if frame is None: return None, None
        return frame, {"patient_id":patient, **record(verified_path(row["file"],loaded[0].parent)),
                       "study_metadata":row.get("study_metadata",{})}
    return consume


def partition(loaded, context, output, cancer):
    candidates = streaming._read_cancer_candidates(loaded[2]["candidates"], cancer)
    return streaming.materialize_cancer_partition(cancer_id=cancer, staging_root=loaded[0].parent,
        output_root=output, candidates=candidates, folds=loaded[4], intervals=loaded[5], membership=loaded[6],
        selected_rows={}, context_sha256=context["context_sha256"], event_threshold=context["event_threshold"],
        max_memory_bytes=context["max_memory_bytes"], max_segment_rows=context["max_segment_rows"], segment_reader=reader(loaded))


def materialize(item, root, output, *, cancers=None):
    require_host(); item = record(verified_path(item, root)); output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    launch = {"target_host":"149", "training_started":False,"input":item,"operation":INPUT}
    preflight = output/"LAUNCH_PREFLIGHT.json"
    if preflight.exists() and read(preflight) != launch: raise ValueError("Native genomic output belongs to another input")
    write(preflight,launch)
    _, _, loaded, context = load(item,root)
    context_path = output/"RUN_CONTEXT.json"
    if context_path.exists() and read(context_path) != context: raise ValueError("Native genomic CNV context drift")
    write(context_path, context)
    declared = loaded[3]["cancers"]; requested = declared if cancers is None else list(cancers)
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(declared):
        raise ValueError("Requested native genomic cancers differ from declared cohort")
    for cancer in requested: partition(loaded,context,output,cancer)
    missing = [c for c in declared if not (output/f"cancer={c}/SUCCESS.json").is_file()]
    if missing: return {"status":"INCOMPLETE", "missing_cancers":missing,"training_started":False}
    if load(item,root)[3] != context: raise ValueError("Native genomic inputs changed while materializing")
    records = {}
    for cancer in declared:
        part = output/f"cancer={cancer}"
        payload = streaming.validate_cancer_partition(part,context_sha256=context["context_sha256"])
        records[cancer] = {"path":str(part),"success_sha256":record(part/"SUCCESS.json")["sha256"],
                          "files_composite_sha256":streaming._json_sha(payload["files"])}
    success = {"format":streaming.FORMAT, "status":"SUCCESS", "context_sha256":context["context_sha256"],
        "run_context_sha256":record(context_path)["sha256"], "cancers":records,"patient_folds":5,
        "required_cancers":declared,"typed_unavailable_is_never_zero":True,"native_input":item,
        "cohort":loaded[3], "training_started":False, "source_ancestry_complete":False}
    path = output/"SUCCESS.json"
    if path.exists() and read(path) != success: raise ValueError("Native genomic aggregate drift")
    write(path,success)
    validate(path)
    return success


def validate(success_path, *, expected_bindings=None, replay=False):
    require_host(); success_path = Path(success_path).resolve(); root = success_path.parent
    payload = read(success_path); context_path = root/"RUN_CONTEXT.json"; context = read(context_path)
    _, _, loaded, actual = load(context["native_input"],root)
    if (context != actual or payload.get("format") != streaming.FORMAT or payload.get("status") != "SUCCESS"
            or payload.get("run_context_sha256") != record(context_path)["sha256"]
            or payload.get("context_sha256") != context["context_sha256"]
            or payload.get("native_input") != context["native_input"] or payload.get("cohort") != loaded[3]
            or payload.get("training_started") is not False or payload.get("patient_folds") != 5
            or payload.get("typed_unavailable_is_never_zero") is not True
            or payload.get("required_cancers") != loaded[3]["cancers"]
            or set(payload.get("cancers",{})) != set(loaded[3]["cancers"])):
        raise ValueError("Native genomic CNV aggregate/context differs")
    for name,path in (expected_bindings or {}).items():
        if context["bindings"].get(name) != record(Path(path).resolve()): raise ValueError("Native genomic input binding differs: "+name)
    for cancer,item in payload["cancers"].items():
        part = root/f"cancer={cancer}"
        if Path(item["path"]).resolve() != part: raise ValueError("Native genomic partition path differs")
        info = streaming.validate_cancer_partition(part,context_sha256=context["context_sha256"])
        if (record(part/"SUCCESS.json")["sha256"] != item["success_sha256"]
                or streaming._json_sha(info["files"]) != item["files_composite_sha256"]
                or read(part/"SOURCE_FILES.json")["records"] != context["raw_segments"][cancer]):
            raise ValueError("Native genomic partition sources differ")
        shape = raw.layout(loaded,cancer)
        for name,ids in zip(raw.IDS,shape[:3]):
            if read(part/name) != ids: raise ValueError("Native genomic partition entity/patient identity differs")
        local = loaded[4].loc[loaded[4].cancer_id.eq(cancer)].sort_values("patient_id")
        for fold in range(5):
            rows = np.load(part/f"fold={fold}/test_patient_rows.npy",allow_pickle=False)
            fold_info = read(part/f"fold={fold}/SUCCESS.json")
            expected_fold = {"format":streaming.FOLD_FORMAT,"status":"SUCCESS","cancer_id":cancer,
                "patient_fold":fold,"context_sha256":context["context_sha256"],
                "test_patient_rows_sha256":record(part/f"fold={fold}/test_patient_rows.npy")["sha256"],"test_patients":len(rows)}
            if (rows.dtype != np.int32 or not np.array_equal(rows,np.flatnonzero(local.patient_fold_id.to_numpy(int)==fold))
                    or fold_info != expected_fold or info["folds"][fold] != expected_fold):
                raise ValueError("Native genomic partition fold patient rows differ")
        for name in ARRAYS:
            array = np.load(part/name,mmap_mode="r",allow_pickle=False)
            width = len(shape[1] if name.startswith("lncrna_") else shape[2])
            if array.shape != (len(shape[0]),width) or array.dtype != (np.float32 if "burden" in name else np.uint8):
                raise ValueError("Native genomic partition array shape/dtype differs")
        if replay:
            with tempfile.TemporaryDirectory(prefix="native-genomic-cnv-replay-",dir=root) as temp:
                partition(loaded,context,Path(temp),cancer)
                for name in ARRAYS:
                    a=np.load(part/name,mmap_mode="r"); b=np.load(Path(temp)/f"cancer={cancer}"/name,mmap_mode="r")
                    for start in range(0,len(a),256):
                        if not np.array_equal(a[start:start+256],b[start:start+256],equal_nan=True):
                            raise ValueError("Native genomic CNV numerical replay differs: "+name)
    return payload


def inspect(inventory, item, root, model, role, fold=-1, *, phase="FEATURE_CONSTRUCTION"):
    """Use the raw-source reconstruction in actual auxiliary model-use traversal."""
    path=inventory.bind(item,root,model,role,fold)
    key=(REPLAY,item["sha256"])
    value=validate(path,replay=key not in inventory.visited);inventory.visited.add(key)
    context_path=path.parent/"RUN_CONTEXT.json";context=read(context_path)
    _,_,loaded,_=load(context["native_input"],path.parent)
    for name,pin in (("context",record(context_path)),("input",context["native_input"]),("segments",context["segment_manifest"])):
        inventory.bind(pin,path.parent,model,role+"."+name,fold)
    for name,digest in context["implementation_sha256"].items():
        inventory.bind({"path":str(ROOT/name),"sha256":digest},ROOT,model,role+".implementation."+name,fold)
    from .current_auxiliary_model_use import input_source,source_metadata
    from .preparation_sources import metadata_tokens
    from .fold_local_scope import inspect_candidates
    inspect_candidates(inventory,loaded[1]["candidate_definition"],loaded[0].parent,loaded[2]["candidates"],model,role+".candidates",fold)
    inventory.bind(loaded[1]["patient_map"],loaded[0].parent,model,role+".patient_map",fold)
    for name in ("intervals","membership"):
        input_source(inventory,loaded[1][name],loaded[0].parent,model,role+"."+name,fold,phase=phase)
    for cancer in value["required_cancers"]:
        part=path.parent/f"cancer={cancer}";info=read(part/"SUCCESS.json")
        inventory.bind(record(part/"SUCCESS.json"),part,model,role+".partition."+cancer,fold)
        for name,digest in info["files"].items():
            inventory.bind({"path":str(part/name),"sha256":digest},part,model,role+".output."+cancer+"."+name,fold)
        for i,source in enumerate(context["raw_segments"][cancer]):
            segment=inventory.bind(source,path.parent,model,role+".segment."+cancer+"."+str(i),fold)
            source_metadata(inventory,segment,model,role+".segment."+cancer+"."+str(i),fold,phase=phase,
                            upstream_tokens=metadata_tokens(source["study_metadata"]))
            inventory.patient_use.observe([pd.DataFrame({"patient_id":[source["patient_id"]]})],digest=source["sha256"],
                model=model,context=inventory.context_id,role=role+".segment_patient",fold=fold,phase=phase,
                scope="DECLARED_SEGMENT_FILE_PATIENT_NOT_PER_PAIR_FITTED_USAGE")
    inventory.transform_replays.append({"model_id":model,"dependency_context":inventory.context_id,
        "patient_fold_id":fold,"role":role,"format":REPLAY,"source_release_sha256":item["sha256"],
        "normalization_sha256":"","activity_sha256":"","folds_recomputed":0,
        "outputs_compared":len(ARRAYS)*len(value["cancers"]),"model_training_started":False})
    inventory.gap(model,role,"SEGMENT_ACQUISITION_AND_REFERENCE_ANCESTRY_UNVERIFIED",fold,item["sha256"])
