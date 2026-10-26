"""Declared-cohort segment-to-directional CNV preparation and reconstruction.

Uses the existing streaming callability and signed mapping algorithms. Raw
segment acquisition, annotation provenance and biological identity are not
independently attested by a numerical reconstruction.
"""
from pathlib import Path
import os
import re

import numpy as np
import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, write, record
from .native_genomic_annotation_workflow import CODE as ANNOTATION_CODE
from .native_pathway_membership_workflow import CODE as MEMBERSHIP_CODE

INPUT = "NATIVE_DIRECTIONAL_CNV_INPUT_V1"
PARTITION = "NATIVE_DIRECTIONAL_CNV_PARTITION_V1"
FORMAT = "NATIVE_DIRECTIONAL_CNV_RELEASE_V1"
REPLAY = "NATIVE_DIRECTIONAL_CNV_NUMERIC_REPLAY_V1"
ROOT = Path(__file__).resolve().parents[2]
CODE = ("cc_hhgt/v32/native_directional_cnv.py", "scripts/materialize_v32_directional_cnv.py",
        "cc_hhgt/v32/directional_cnv.py", "cc_hhgt/v32/segment_cnv_streaming.py",
        "cc_hhgt/v32/local_cnv_cohort.py", "cc_hhgt/v32/fold_local_scope.py",
        "scripts/prepare_v32_formal.py", *ANNOTATION_CODE, *MEMBERSHIP_CODE)
ARRAYS = tuple(f"{kind}_{feature}.npy" for kind in ("lncrna", "pathway")
    for feature in (("signed_mean", "callable", "amplification", "deletion", "neutral")
        + (("amplification_fraction", "deletion_fraction") if kind == "pathway" else ())))
IDS = ("patient_ids.json", "lncrna_ids.json", "pathway_ids.json")


def implementations(): return {name:record(ROOT/name) for name in CODE}


def load_manifest(item, root):
    require_host(); path = verified_path(item, root); value = read(path)
    required = {"format", "candidate_definition", "candidates", "patient_map", "intervals", "membership", "segments"}
    if (not required <= set(value) or set(value)-required-{"event_threshold", "max_segment_rows", "segment_reference"}
            or value["format"] != INPUT or not isinstance(value["segments"], list)):
        raise ValueError("Invalid native directional CNV manifest")
    files = {key:verified_path(value[key], path.parent) for key in required-{"format", "segments"}}
    from .local_cnv_cohort import from_definition
    cohort = from_definition(value["candidate_definition"], path.parent, files["candidates"], files["patient_map"])
    patients = pd.read_csv(files["patient_map"], sep="\t", dtype={"cancer_id":str, "patient_id":str})
    patients = patients.loc[patients.cancer_id.isin(cohort["cancers"])].copy()
    from .local_cnv_confounding import canonical_patient
    if patients.patient_id.map(canonical_patient).ne(patients.patient_id).any():
        raise ValueError("Native directional patient IDs must already be canonical")
    expected = set(map(tuple, patients[["cancer_id", "patient_id"]].to_numpy()))
    records = {}; file_owners = set()
    for row in value["segments"]:
        key = (row.get("cancer_id"), row.get("patient_id"))
        if key not in expected or key in records: raise ValueError("Native segment patient scope is duplicated or outside the cohort")
        base = {"cancer_id", "patient_id", "status"}
        if row.get("status") == "available":
            if set(row)-base-{"file", "study_metadata"} or "file" not in row or not isinstance(row.get("study_metadata", {}), dict):
                raise ValueError("Available segment requires an explicit file binding")
            item = row["file"]
            if (not isinstance(item, dict) or set(item) != {"path", "sha256"} or not isinstance(item.get("path"), str)
                    or not item["path"] or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
                raise ValueError("Native segment file binding is invalid")
            source = Path(item["path"]); source = (path.parent/source).resolve() if not source.is_absolute() else source.resolve()
            if source in file_owners: raise ValueError("One native segment file cannot represent multiple patients")
            file_owners.add(source)
        elif row.get("status") == "typed_unavailable":
            if set(row) != base|{"unavailable_reason"} or row["unavailable_reason"] != "NO_SELECTED_SEGMENT_FILE":
                raise ValueError("Unavailable segment requires the explicit missing-file reason")
        else: raise ValueError("Unknown native segment availability")
        records[key] = row
    if set(records) != expected: raise ValueError("Native segment manifest omits declared patients")
    threshold = value.get("event_threshold", .30); limit = value.get("max_segment_rows", 2_000_000)
    if (isinstance(threshold, bool) or not isinstance(threshold, (int,float)) or not np.isfinite(threshold) or threshold <= 0
            or type(limit) is not int or limit <= 0): raise ValueError("Invalid native segment threshold or row limit")
    intervals = pd.read_parquet(files["intervals"]); membership = pd.read_parquet(files["membership"])
    for name in ("start", "end"):
        if name not in intervals: raise ValueError("Native CNV interval coordinates are missing")
        values = pd.to_numeric(intervals[name], errors="raise")
        if not np.isfinite(values).all() or values.ne(np.floor(values)).any():
            raise ValueError("Native CNV interval coordinates must be finite integers")
    from .segment_cnv_streaming import _normalise_shared_inputs
    patients, intervals, membership = _normalise_shared_inputs(patients, intervals, membership)
    if intervals.start.lt(0).any() or intervals.end.lt(intervals.start).any():
        raise ValueError("Native CNV intervals have invalid coordinates")
    loaded = path, value, files, cohort, patients, intervals, membership, records
    reference_context(loaded)
    membership_context(loaded)
    return loaded


def reference_context(loaded):
    from .native_genomic_annotation_workflow import consumer
    return consumer(loaded[2]["intervals"], loaded[1]["candidate_definition"], loaded[6], loaded[0].parent,
                    loaded[1].get("segment_reference"))


def coordinate_system(value):
    return (value.get("segment_reference") or {}).get("coordinate_system")


def membership_context(loaded):
    from .native_pathway_membership_workflow import consumer
    return consumer(loaded[2]["membership"], loaded[1]["candidate_definition"], loaded[0].parent)


def layout(loaded, cancer):
    path, value, files, cohort, patients, intervals, membership, records = loaded
    if cancer not in cohort["cancers"]: raise ValueError("Directional cancer is outside the declared cohort")
    pairs = pd.read_parquet(files["candidates"], columns=["cancer_id", "lncrna_id", "pathway_id"], filters=[("cancer_id", "==", cancer)])
    patient_ids = sorted(patients.loc[patients.cancer_id.eq(cancer), "patient_id"])
    lncs = sorted(pairs.lncrna_id.unique()); pathways = sorted(pairs.pathway_id.unique())
    members = membership.loc[membership.pathway_id.isin(pathways)]
    genes = sorted(members.gene_id.unique()); gmap = {v:i for i,v in enumerate(genes)}
    positions = [np.array([gmap[g] for g in members.loc[members.pathway_id.eq(p), "gene_id"]], dtype=int) for p in pathways]
    lnc_intervals = intervals.loc[intervals.entity_type.eq("lncrna") & intervals.entity_id.isin(lncs)]
    gene_intervals = intervals.loc[intervals.entity_type.eq("gene") & intervals.entity_id.isin(genes)]
    return patient_ids, lncs, pathways, genes, positions, lnc_intervals, gene_intervals


def read_segment(row, base, limit, *, coordinate_system=None):
    if row["status"] == "typed_unavailable": return None
    path = verified_path(row["file"], base)
    frame = pd.read_csv(path, sep="\t", nrows=limit+1)
    if len(frame) > limit: raise ValueError("Native segment row limit exceeded")
    columns = {str(c).lower():c for c in frame}
    from .local_cnv_confounding import canonical_patient
    for name in ("sample", "sample_id", "patient_id", "case_submitter_id"):
        if name in columns and (frame[columns[name]].isna().any()
                or frame[columns[name]].map(canonical_patient).ne(row["patient_id"]).any()):
            raise ValueError("Native segment content belongs to a different patient")
    names = ("chromosome", "start", "end", "segment_mean")
    if not set(names) <= set(columns): raise ValueError("Native segment columns are incomplete")
    frame = frame[[columns[c] for c in names]].copy(); frame.columns = ["chromosome", "start", "end", "value"]
    if frame.chromosome.isna().any(): raise ValueError("Native segment chromosome is missing")
    frame["chromosome"] = frame.chromosome.astype(str).str.replace(r"^chr", "", regex=True)
    for c in ("start", "end"):
        number = pd.to_numeric(frame[c], errors="raise")
        if not np.isfinite(number).all() or number.lt(0).any() or number.ne(np.floor(number)).any():
            raise ValueError("Native segment coordinates must be finite nonnegative integers")
        frame[c] = number.astype(np.int64)
    if frame.end.lt(frame.start).any(): raise ValueError("Native segment endpoints are reversed")
    if coordinate_system not in (None, "1_BASED_CLOSED"):
        raise ValueError("Unsupported native segment coordinate convention")
    if coordinate_system == "1_BASED_CLOSED" and frame.start.lt(1).any():
        raise ValueError("Declared one-based segments contain a zero start")
    for _, group in frame.sort_values(["chromosome", "start", "end"]).groupby("chromosome"):
        if (group.start.to_numpy()[1:] <= group.end.to_numpy()[:-1]).any():
            raise ValueError("Native segment intervals overlap within a chromosome")
    frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    verified_path(row["file"], base)
    return frame


def row_arrays(segment, shape, threshold):
    _, lncs, pathways, genes, positions, lnc_intervals, gene_intervals = shape
    output = {name:np.full(len(lncs) if name.startswith("lncrna_") else len(pathways), np.nan, dtype=np.float32)
        if "mean" in name or "fraction" in name else np.zeros(len(lncs) if name.startswith("lncrna_") else len(pathways), dtype=np.uint8)
        for name in ARRAYS}
    if segment is None: return output
    from .directional_cnv import map_segments_directional, aggregate_pathway_directional
    from .segment_cnv_streaming import _mapped_entities
    lnc = map_segments_directional(segment, lnc_intervals, lncs, event_threshold=threshold)
    gene = map_segments_directional(segment, gene_intervals, genes, event_threshold=threshold)
    # Preserve the original event-or-all-members canonical pathway callability.
    # Direction remains based on signed means and separate amplification/deletion.
    _, lc, _ = _mapped_entities(segment, lnc_intervals, lncs, threshold)
    ge, gc, _ = _mapped_entities(segment, gene_intervals, genes, threshold)
    if not np.array_equal(lc.astype(bool), lnc.callable): raise ValueError("Native local callability algorithms disagree")
    pc = np.array([bool(len(p) and (ge[p].any() or gc[p].all())) for p in positions])
    pathway = aggregate_pathway_directional(gene, positions, canonical_callable=pc)
    for name in ("signed_mean", "callable", "amplification", "deletion", "neutral"):
        output["lncrna_"+name+".npy"][:] = getattr(lnc, name)
    for name, value in pathway.items(): output["pathway_"+name+".npy"][:] = value
    return output


def sources(loaded, cancer):
    path, _, _, _, _, _, _, records = loaded
    result = []
    for (c, patient), row in sorted(records.items()):
        if c == cancer and row["status"] == "available":
            file = verified_path(row["file"], path.parent)
            result.append({"patient_id":patient, **record(file), "study_metadata":row.get("study_metadata", {})})
    return result


def preflight(item, root, output):
    require_host(); output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    receipt = {"target_host":"149", "training_started":False, "operation":FORMAT,
               "input":record(verified_path(item, root))}
    path = output/"LAUNCH_PREFLIGHT.json"
    if path.exists() and read(path) != receipt: raise ValueError("Directional output belongs to a different input manifest")
    if not path.exists(): write(path, receipt)
    return output


def materialize(item, root, output, cancer):
    output = preflight(item, root, output)
    loaded = load_manifest(item, root); path, value, files, cohort, _, _, _, records = loaded
    final = output/("cancer="+cancer)
    if final.exists(): raise FileExistsError("Native directional partition is immutable")
    shape = layout(loaded, cancer); patients, lncs, pathways, genes = shape[:4]
    initial = sources(loaded, cancer); codes = implementations()
    building = output/(".cancer="+cancer+".building."+str(os.getpid())); building.mkdir()
    write(building/"INCOMPLETE.json", {"target_host":"149", "status":"BUILDING", "training_started":False})
    try:
        arrays = {}
        for name in ARRAYS:
            size = (len(patients), len(lncs) if name.startswith("lncrna_") else len(pathways))
            dtype = np.float32 if "mean" in name or "fraction" in name else np.uint8
            arrays[name] = np.lib.format.open_memmap(building/name, mode="w+", dtype=dtype, shape=size)
        for i, patient in enumerate(patients):
            segment = read_segment(records[(cancer, patient)], path.parent, value.get("max_segment_rows", 2_000_000),
                                   coordinate_system=coordinate_system(value))
            for name, row in row_arrays(segment, shape, value.get("event_threshold", .30)).items(): arrays[name][i] = row
        for array in arrays.values(): array.flush()
        del array, arrays
        for name, ids in zip(IDS, (patients, lncs, pathways)): write(building/name, ids)
        write(building/"SOURCE_FILES.json", {"records":initial})
        if sources(loaded, cancer) != initial: raise ValueError("Native segment inputs changed during preparation")
        verified_path(item, root); load_manifest(item, root)
        if implementations() != codes: raise ValueError("Native directional implementation changed")
        receipt = {"format":PARTITION, "status":"SUCCESS", "target_host":"149", "training_started":False,
            "input":record(path), "cohort":cohort, "cancer_id":cancer, "implementations":codes,
            "patients":len(patients), "lncrnas":len(lncs), "pathways":len(pathways), "genes":len(genes),
            "processed_segment_patients":len(initial), "typed_unavailable_patients":len(patients)-len(initial),
            "event_threshold":value.get("event_threshold", .30), "signed_values_preserved":True,
            "absolute_burden_used_as_continuous":False, "amplification_deletion_separate":True,
            "raw_study_ancestry_complete":False,
            "files":{name:record(building/name)["sha256"] for name in (*IDS, *ARRAYS, "SOURCE_FILES.json")}}
        annotation = reference_context(loaded)
        if annotation is not None: receipt["interval_reference"] = annotation
        membership = membership_context(loaded)
        if membership is not None: receipt["membership_reference"] = membership
        write(building/"SUCCESS.json", receipt); (building/"INCOMPLETE.json").unlink()
        from .gdc_segment_cnv import _rename_directory_noreplace
        _rename_directory_noreplace(building, final)
        return final/"SUCCESS.json"
    except Exception as error:
        write(building/"RUN_FAILED.json", {"target_host":"149", "status":"FAILED", "training_started":False,
            "error_type":type(error).__name__, "error":str(error)})
        raise


def validate_partition(item, root, *, replay=False, cache=None):
    require_host(); path = verified_path(item, root); value = read(path)
    if (value.get("format") != PARTITION or value.get("status") != "SUCCESS" or value.get("target_host") != "149"
            or value.get("training_started") is not False or value.get("raw_study_ancestry_complete") is not False
            or value.get("signed_values_preserved") is not True or value.get("absolute_burden_used_as_continuous") is not False
            or value.get("amplification_deletion_separate") is not True or value.get("implementations") != implementations()
            or set(value.get("files", {})) != set((*IDS, *ARRAYS, "SOURCE_FILES.json"))):
        raise ValueError("Invalid native directional partition contract")
    loaded = load_manifest(value["input"], path.parent)
    if value["cohort"] != loaded[3]: raise ValueError("Directional partition cohort differs")
    if value.get("interval_reference") != reference_context(loaded): raise ValueError("Directional interval reference differs")
    if value.get("membership_reference") != membership_context(loaded): raise ValueError("Directional membership reference differs")
    shape = layout(loaded, value["cancer_id"]); patients, lncs, pathways, genes = shape[:4]
    raw = sources(loaded, value["cancer_id"])
    for name, digest in value["files"].items(): verified_path({"path":str(path.parent/name), "sha256":digest}, path.parent)
    if read(path.parent/"SOURCE_FILES.json") != {"records":raw}: raise ValueError("Directional consumed segments differ")
    for name, ids in zip(IDS, (patients, lncs, pathways)):
        if read(path.parent/name) != ids: raise ValueError("Directional patient/entity ordering differs")
    counts = {"patients":len(patients), "lncrnas":len(lncs), "pathways":len(pathways), "genes":len(genes),
        "processed_segment_patients":len(raw), "typed_unavailable_patients":len(patients)-len(raw),
        "event_threshold":loaded[1].get("event_threshold", .30)}
    if any(value.get(k) != v for k,v in counts.items()): raise ValueError("Directional counts/threshold differ")
    arrays = {name:np.load(path.parent/name, mmap_mode="r", allow_pickle=False) for name in ARRAYS}
    for name, array in arrays.items():
        dtype = np.dtype("float32") if "mean" in name or "fraction" in name else np.dtype("uint8")
        if array.shape != (len(patients), len(lncs) if name.startswith("lncrna_") else len(pathways)) or array.dtype != dtype:
            raise ValueError("Directional array shape/dtype differs")
    key = (REPLAY, item["sha256"])
    if replay and (cache is None or key not in cache):
        for i, patient in enumerate(patients):
            row = loaded[7][(value["cancer_id"], patient)]
            segment = read_segment(row, loaded[0].parent, loaded[1].get("max_segment_rows", 2_000_000),
                                   coordinate_system=coordinate_system(loaded[1]))
            expected = row_arrays(segment, shape, value["event_threshold"])
            for name in ARRAYS:
                if not np.array_equal(arrays[name][i], expected[name], equal_nan=True):
                    raise ValueError("Directional numerical replay differs: "+name)
        load_manifest(value["input"], path.parent); verified_path(item, root)
        for file in raw: verified_path(file, path.parent)
        for name, digest in value["files"].items(): verified_path({"path":str(path.parent/name), "sha256":digest}, path.parent)
        if implementations() != value["implementations"]: raise ValueError("Directional code changed during replay")
        if cache is not None: cache.add(key)
    return path, value, loaded


def finalize(item, root, output):
    output = preflight(item, root, output); loaded = load_manifest(item, root)
    if (output/"SUCCESS.json").exists(): raise FileExistsError("Native directional release is immutable")
    if {p.name for p in output.glob("cancer=*")} != {"cancer="+c for c in loaded[3]["cancers"]}:
        raise ValueError("Native directional release cancer coverage differs")
    records = {}
    for cancer in loaded[3]["cancers"]:
        path, value, _ = validate_partition(record(output/("cancer="+cancer)/"SUCCESS.json"), output)
        if value["input"]["sha256"] != item["sha256"] or value["cancer_id"] != cancer:
            raise ValueError("Native directional partition belongs to a different release")
        records[cancer] = {"path":str(path.parent), "success_sha256":record(path)["sha256"],
            "processed_segment_patients":value["processed_segment_patients"], "typed_unavailable_patients":value["typed_unavailable_patients"]}
    write(output/"SUCCESS.json", {"format":FORMAT, "status":"SUCCESS", "target_host":"149", "training_started":False,
        "input":record(loaded[0]), "cohort":loaded[3], "cancers":records, "cancer_count":len(records)})
    return output/"SUCCESS.json"


def validate_release(item, root, *, replay=False):
    path = verified_path(item, root); value = read(path); loaded = load_manifest(value["input"], path.parent)
    if (value.get("format") != FORMAT or value.get("status") != "SUCCESS" or value.get("target_host") != "149"
            or value.get("training_started") is not False or value.get("cohort") != loaded[3]
            or set(value.get("cancers", {})) != set(loaded[3]["cancers"]) or value.get("cancer_count") != len(loaded[3]["cancers"])):
        raise ValueError("Invalid native directional release")
    if {p.name for p in path.parent.glob("cancer=*")} != {"cancer="+c for c in value["cancers"]}:
        raise ValueError("Native directional physical cancer coverage differs")
    for cancer, entry in value["cancers"].items():
        if Path(entry["path"]).resolve() != path.parent/("cancer="+cancer):
            raise ValueError("Native directional partition path differs from its consumed layout")
        part = {"path":str(Path(entry["path"])/"SUCCESS.json"), "sha256":entry["success_sha256"]}
        _, info, _ = validate_partition(part, path.parent, replay=replay)
        if (info["cancer_id"] != cancer or info["input"] != value["input"]
                or any(entry[k] != info[k] for k in ("processed_segment_patients", "typed_unavailable_patients"))):
            raise ValueError("Native directional release partition differs")
    verified_path(item, root)
    return path, value


def inspect(inventory, item, root, model, role, fold, *, phase):
    path, value, loaded = validate_partition(item, root, replay=True, cache=inventory.visited)
    inventory.bind(value["input"], path.parent, model, role+".manifest", fold)
    for name, source in value["implementations"].items(): inventory.bind(source, path.parent, model, role+".implementation."+name, fold)
    for name, digest in value["files"].items(): inventory.bind({"path":str(path.parent/name), "sha256":digest}, path.parent, model, role+".output."+name, fold)
    from .current_auxiliary_model_use import input_source, source_metadata
    from .preparation_sources import metadata_tokens
    for name in ("intervals", "membership"):
        input_source(inventory, loaded[1][name], loaded[0].parent, model, role+"."+name, fold, phase=phase)
    inventory.bind(loaded[1]["patient_map"], loaded[0].parent, model, role+".patient_map", fold)
    from .fold_local_scope import inspect_candidates
    inspect_candidates(inventory, loaded[1]["candidate_definition"], loaded[0].parent, loaded[2]["candidates"], model, role+".candidates", fold)
    for i, source in enumerate(sources(loaded, value["cancer_id"])):
        raw = inventory.bind(source, path.parent, model, role+".segment."+str(i), fold)
        source_metadata(inventory, raw, model, role+".segment."+str(i), fold, phase=phase,
            upstream_tokens=metadata_tokens(source["study_metadata"]))
        inventory.patient_use.observe([pd.DataFrame({"patient_id":[source["patient_id"]]})], digest=source["sha256"],
            model=model, context=inventory.context_id, role=role+".segment_patient", fold=fold, phase=phase,
            scope="DECLARED_SEGMENT_FILE_PATIENT_NOT_PER_PAIR_FITTED_USAGE")
    inventory.transform_replays.append({"model_id":model, "dependency_context":inventory.context_id,
        "patient_fold_id":fold, "role":role, "format":REPLAY, "source_release_sha256":item["sha256"],
        "normalization_sha256":"", "activity_sha256":"", "folds_recomputed":0,
        "outputs_compared":len(ARRAYS), "model_training_started":False})
    inventory.gap(model, role, "SEGMENT_ACQUISITION_AND_REFERENCE_ANCESTRY_UNVERIFIED", fold, item["sha256"])
