"""Bound GENCODE gene coordinates for the existing midpoint CNV producers.

This entry does not infer symbol aliases, pathway membership, segment assembly
or alignment authenticity. It preserves one-based closed GTF coordinates.
"""
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, write, record
from . import native_annotation_workflow as reference

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_GENOMIC_ANNOTATION_INPUT_V1"
FORMAT = "NATIVE_GENOMIC_ANNOTATION_RESULTS_V1"
TABLE = "entity_intervals.parquet"
BINDING = "NATIVE_GENOMIC_ANNOTATION_BINDING.json"
MARKER = "native_genomic_annotation_preflight"
CODE = ("cc_hhgt/v32/native_genomic_annotation_workflow.py", "scripts/run_native_genomic_annotation_workflow.py", *reference.CODE)
_REPLAY_CACHE = {}


def plan(manifest, output, *, new=True):
    require_host(); manifest = Path(manifest).resolve(); output = Path(output).resolve(); spec = read(manifest)
    if set(spec) != {"format", "reference_annotation"} or spec.get("format") != INPUT:
        raise ValueError("Invalid native genomic annotation declaration")
    if new and output.exists(): raise FileExistsError("Use a new genomic annotation directory")
    path, value = reference.validate(spec["reference_annotation"], manifest.parent)
    launch = read(value["preflight"]["path"])
    if any(output == p or output in p.parents for p in (manifest, path, Path(launch["gtf"]["path"]))):
        raise ValueError("Genomic annotation output must not contain source files")
    return {"input": record(manifest), "reference_annotation": record(path), "annotation": value["annotation"],
            "gtf": launch["gtf"], "genome_build": launch["genome_build"], "gencode_release": launch["gencode_release"],
            "coordinate_system": "1_BASED_CLOSED", "target_host": "149", "training_started": False,
            "implementations": [record(ROOT / name) for name in CODE]}


def derive(value, preflight):
    raw = reference.annotation_table(value["annotation"], preflight.parent)
    selected = raw.loc[raw.gene_type.isin(["lncRNA", "protein_coding"])].copy()
    if selected.empty: raise ValueError("GENCODE has no supported CNV entities")
    selected["entity_type"] = selected.gene_type.map({"lncRNA": "lncrna", "protein_coding": "gene"})
    selected["entity_id"] = selected.gene_type.map({"lncRNA": "LNC:", "protein_coding": "GENE:"}) + selected.ensembl_gene_id
    for name in ("start", "end", "gtf_line"): selected[name] = selected[name].astype("int64")
    names = ["entity_type", "entity_id", "chromosome", "start", "end", "ensembl_gene_id", "versioned_gene_id",
             "gene_symbol", "gene_type", "genome_build", "coordinate_system", "gtf_line"]
    result = selected[names].sort_values(["entity_type", "entity_id"], kind="stable").reset_index(drop=True)
    result.attrs[MARKER] = record(preflight)
    metadata = {"reference_gene_rows": len(raw), "interval_rows": len(result),
                "lncrna_rows": int(result.entity_type.eq("lncrna").sum()),
                "protein_coding_rows": int(result.entity_type.eq("gene").sum()),
                "excluded_other_biotype_rows": len(raw) - len(result), "coordinate_system": "1_BASED_CLOSED",
                "gene_id_policy": "GENE_OR_LNC_PREFIX_PLUS_GENCODE_STABLE_ID_PAR_Y_PRESERVED",
                "contig_filter_applied": False, "liftover_performed": False}
    return result, metadata


def run(manifest, output):
    value = plan(manifest, output); output = Path(output).resolve(); output.mkdir(parents=True)
    preflight = output / "LAUNCH_PREFLIGHT.json"; write(preflight, value)
    try:
        frame, metadata = derive(value, preflight); frame.to_parquet(output / TABLE, index=False)
        if value != plan(manifest, output, new=False): raise ValueError("Genomic annotation source changed")
        binding = output / BINDING
        write(binding, {"format": FORMAT, "status": "PASS", "target_host": "149", "training_started": False,
            "preflight": record(preflight), "intervals": record(output / TABLE), "metadata": metadata,
            "original_coordinates_reconstructed": True, "reference_authenticity_verified": False,
            "source_ancestry_complete": False, "production_deployed": False})
        validate(record(binding), output); return binding
    except Exception as error:
        write(output / "RUN_FAILED.json", {"target_host": "149", "training_started": False, "error": str(error)})
        raise


def validate(item, root):
    require_host(); path = verified_path(item, root); output = path.parent; value = read(path)
    if (path.name != BINDING or value.get("format") != FORMAT or value.get("status") != "PASS"
            or value.get("target_host") != "149" or value.get("original_coordinates_reconstructed") is not True
            or any(value.get(k) is not False for k in ("training_started", "reference_authenticity_verified",
                                                       "source_ancestry_complete", "production_deployed"))):
        raise ValueError("Incomplete native genomic annotation")
    preflight = verified_path(value["preflight"], output); table = verified_path(value["intervals"], output)
    if preflight != output / "LAUNCH_PREFLIGHT.json" or table != output / TABLE:
        raise ValueError("Genomic annotation handoff differs")
    launch = read(preflight)
    if launch != plan(verified_path(launch["input"], output), output, new=False):
        raise ValueError("Genomic annotation sources or implementation differ")
    signature = (record(path), launch, value)
    if _REPLAY_CACHE.get(str(output)) != signature:
        expected, metadata = derive(launch, preflight); actual = pd.read_parquet(table)
        try: pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        except AssertionError as error: raise ValueError("CNV intervals differ from GTF reconstruction") from error
        if actual.attrs != expected.attrs or metadata != value["metadata"]:
            raise ValueError("Genomic annotation metadata differs from reconstruction")
        _REPLAY_CACHE[str(output)] = signature
    return path, value


def discover(path):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != ".parquet": return None
    attrs = json.loads((pq.read_schema(path).metadata or {}).get(b"PANDAS_ATTRS", b"{}"))
    pointer = attrs.get(MARKER)
    if pointer is None:
        if path.name == TABLE and (path.parent / BINDING).exists(): raise ValueError("Genomic interval native marker missing")
        return None
    if verified_path(pointer, path.parent) != path.parent / "LAUNCH_PREFLIGHT.json":
        raise ValueError("Genomic interval marker points outside consumed output")
    _, value = validate(record(path.parent / BINDING), path.parent)
    if verified_path(value["intervals"], path.parent) != path or value["preflight"] != pointer:
        raise ValueError("Genomic annotation binds another interval table")
    return value


def consumer(intervals, definition, membership, root, segment_reference):
    """Match actual candidate annotation and declared coordinate conventions.

    Membership is consumed unchanged. Missing gene coordinates remain explicit
    unavailable members under the original event-or-all-members CNV estimator.
    """
    value = discover(intervals)
    if segment_reference is not None and (not isinstance(segment_reference, dict)
            or set(segment_reference) != {"genome_build", "coordinate_system"}
            or segment_reference["coordinate_system"] != "1_BASED_CLOSED"
            or segment_reference["genome_build"] not in {"GRCh37", "GRCh38"}):
        raise ValueError("Unsupported declared segment reference/coordinate convention")
    if value is None: return None
    launch = read(value["preflight"]["path"])
    if segment_reference != {"genome_build": launch["genome_build"], "coordinate_system": launch["coordinate_system"]}:
        raise ValueError("Native intervals require matching declared segment build and coordinates")
    definition_path = verified_path(definition, root); candidate = read(definition_path)
    candidate_annotation = record(verified_path(candidate["annotation"], definition_path.parent))
    if candidate_annotation["sha256"] != launch["annotation"]["sha256"]:
        raise ValueError("CNV intervals and candidates use different gene annotations")
    frame = pd.read_parquet(intervals, columns=["entity_type", "entity_id"])
    genes = set(frame.loc[frame.entity_type.eq("gene"), "entity_id"])
    members = set(membership.gene_id.dropna().astype(str))
    return {"binding": record(Path(intervals).parent / BINDING), "candidate_annotation": candidate_annotation,
            "segment_reference": segment_reference, "segment_reference_independently_verified": False,
            "membership_genes": len(members), "membership_genes_with_coordinates": len(members & genes),
            "membership_genes_without_coordinates": len(members - genes),
            "missing_members_dropped": False, "reference_authenticity_verified": False}


def inspect(inventory, path, model, role, fold=-1):
    value = discover(path)
    if value is None: return False
    root = Path(path).parent; launch = read(value["preflight"]["path"])
    ref_path, ref = reference.validate(launch["reference_annotation"], root); ref_launch = read(ref["preflight"]["path"])
    pins = [record(root / BINDING), value["preflight"], value["intervals"], launch["input"], launch["reference_annotation"],
            ref["preflight"], ref["annotation"], ref_launch["input"], launch["gtf"], *launch["implementations"]]
    for index, pin in enumerate(pins): inventory.bind(pin, root, model, role + ".annotation_source." + str(index), fold)
    inventory.gap(model, role, "GENCODE_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED", fold, launch["gtf"]["sha256"])
    inventory.transform_replays.append({"model_id": model, "dependency_context": inventory.context_id,
        "patient_fold_id": fold, "role": role, "format": FORMAT,
        "source_release_sha256": record(root / BINDING)["sha256"], "normalization_sha256": "", "activity_sha256": "",
        "folds_recomputed": 0, "outputs_compared": 1, "model_training_started": False})
    return True
