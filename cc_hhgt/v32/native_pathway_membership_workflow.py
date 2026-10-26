"""Reconstruct native CNV pathway members from exact GMT symbols and GENCODE.

Reuse the original coding-preferred, many-to-many mapping. Preserve unmatched
symbols as explicitly unavailable members instead of shrinking the denominator.
Reference reconstruction does not attest resource origin or biological truth.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, write, record
from . import native_annotation_workflow as reference

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_PATHWAY_MEMBERSHIP_INPUT_V1"
FORMAT = "NATIVE_PATHWAY_MEMBERSHIP_RESULTS_V1"
TABLE = "pathway_membership.parquet"
BINDING = "NATIVE_PATHWAY_MEMBERSHIP_BINDING.json"
MARKER = "native_pathway_membership_preflight"
CODE = ("cc_hhgt/v32/native_pathway_membership_workflow.py", "scripts/run_native_pathway_membership_workflow.py",
        "scripts/build_v32_genomic_static_assets.py", "cc_hhgt/v32/fold_local_activity.py", *reference.CODE)
_REPLAY_CACHE = {}


def gmt(item, root):
    from .fold_local_activity import gene_sets
    path = verified_path(item, root); manifest = read(path)
    sources = {entry["collection"]: entry["file"] for entry in manifest.get("collections", [])}
    sets, _ = gene_sets({"inputs": {"gmt_manifest": record(path)}, "gmt_sources": sources}, path.parent)
    return sets, {name: record(verified_path(pin, path.parent)) for name, pin in sources.items()}


def plan(manifest, output, *, new=True):
    require_host(); manifest = Path(manifest).resolve(); output = Path(output).resolve(); spec = read(manifest)
    if set(spec) != {"format", "reference_annotation", "gmt_manifest"} or spec.get("format") != INPUT:
        raise ValueError("Invalid native pathway membership declaration")
    if new and output.exists(): raise FileExistsError("Use a new pathway membership directory")
    path, value = reference.validate(spec["reference_annotation"], manifest.parent)
    launch = read(value["preflight"]["path"])
    gmt_path = verified_path(spec["gmt_manifest"], manifest.parent)
    _, sources = gmt(record(gmt_path), manifest.parent)
    pins = [record(manifest), record(path), value["annotation"], launch["gtf"], record(gmt_path), *sources.values()]
    if any(output == Path(p["path"]) or output in Path(p["path"]).parents for p in pins):
        raise ValueError("Membership output must not contain source files")
    return {"input": record(manifest), "reference_annotation": record(path), "annotation": value["annotation"],
            "gtf": launch["gtf"], "gmt_manifest": record(gmt_path), "gmt_sources": sources,
            "target_host": "149", "training_started": False, "implementations": [record(ROOT / n) for n in CODE]}


def derive(value, preflight):
    from scripts.build_v32_genomic_static_assets import symbol_gene_map
    sets, sources = gmt(value["gmt_manifest"], preflight.parent)
    if sources != value["gmt_sources"]: raise ValueError("GMT source pins differ")
    annotation = reference.annotation_table(value["annotation"], preflight.parent)
    genes = symbol_gene_map(annotation)
    raw = pd.DataFrame([(p, s) for p, symbols in sets.items() for s in symbols], columns=["pathway_id", "gene_symbol"])
    mapped = raw.merge(genes[["gene_symbol", "ensembl_gene_id", "gene_type"]], on="gene_symbol", how="left", validate="many_to_many")
    missing = mapped.ensembl_gene_id.isna()
    mapped["gene_id"] = "GENE:" + mapped.ensembl_gene_id.fillna("")
    # This reserved namespace denotes an unavailable source symbol, never an
    # invented Ensembl gene or a measured neutral value. No alias guessing.
    mapped.loc[missing, "gene_id"] = mapped.loc[missing, "gene_symbol"].map(
        lambda s: "UNMAPPED_GMT:" + hashlib.sha256(s.encode("utf-8")).hexdigest())
    mapped["mapping_status"] = "MAPPED"
    mapped.loc[missing, "mapping_status"] = "UNMAPPED_SYMBOL"
    mapped = mapped.fillna("")
    columns = ["pathway_id", "gene_id", "gene_symbol", "ensembl_gene_id", "gene_type", "mapping_status"]
    frame = mapped[columns].sort_values(["pathway_id", "gene_symbol", "gene_id"], kind="stable").reset_index(drop=True)
    frame.attrs[MARKER] = record(preflight)
    multiplicity = frame.groupby(["pathway_id", "gene_symbol"], sort=False).size()
    metadata = {"pathways": len(sets), "source_symbol_memberships": len(raw), "mapping_rows": len(frame),
                "consumer_gene_memberships": len(frame[["pathway_id", "gene_id"]].drop_duplicates()),
                "unmapped_symbol_memberships": int(missing.sum()), "one_to_many_symbol_memberships": int(multiplicity.gt(1).sum()),
                "noncoding_fallback_rows": int((~missing & mapped.gene_type.ne("protein_coding")).sum()),
                "symbol_policy": "EXACT_CASE_SENSITIVE_PREFER_CODING_KEEP_ALL_MATCHES",
                "unmapped_policy": "PRESERVE_RESERVED_UNAVAILABLE_MEMBER", "measured_expression_filter_applied": False,
                "member_weight_policy": "EXISTING_CNV_UNWEIGHTED_UNIQUE_GENE_MEMBERS"}
    return frame, metadata


def run(manifest, output):
    value = plan(manifest, output); output = Path(output).resolve(); output.mkdir(parents=True)
    preflight = output / "LAUNCH_PREFLIGHT.json"; write(preflight, value)
    try:
        frame, metadata = derive(value, preflight); frame.to_parquet(output / TABLE, index=False)
        if value != plan(manifest, output, new=False): raise ValueError("Membership source changed")
        binding = output / BINDING
        write(binding, {"format": FORMAT, "status": "PASS", "target_host": "149", "training_started": False,
            "preflight": record(preflight), "membership": record(output / TABLE), "metadata": metadata,
            "symbol_mapping_reconstructed": True, "reference_authenticity_verified": False,
            "source_ancestry_complete": False, "production_deployed": False})
        validate(record(binding), output); return binding
    except Exception as error:
        write(output / "RUN_FAILED.json", {"target_host": "149", "training_started": False, "error": str(error)})
        raise


def validate(item, root):
    require_host(); path = verified_path(item, root); output = path.parent; value = read(path)
    if (path.name != BINDING or value.get("format") != FORMAT or value.get("status") != "PASS"
            or value.get("target_host") != "149" or value.get("symbol_mapping_reconstructed") is not True
            or any(value.get(k) is not False for k in ("training_started", "reference_authenticity_verified",
                                                       "source_ancestry_complete", "production_deployed"))):
        raise ValueError("Incomplete native pathway membership")
    preflight = verified_path(value["preflight"], output); table = verified_path(value["membership"], output)
    if preflight != output / "LAUNCH_PREFLIGHT.json" or table != output / TABLE:
        raise ValueError("Membership handoff differs")
    launch = read(preflight)
    if launch != plan(verified_path(launch["input"], output), output, new=False):
        raise ValueError("Membership sources or implementation differ")
    signature = (record(path), launch, value)
    if _REPLAY_CACHE.get(str(output)) != signature:
        expected, metadata = derive(launch, preflight); actual = pd.read_parquet(table)
        try: pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        except AssertionError as error: raise ValueError("Membership differs from GMT/GENCODE reconstruction") from error
        if actual.attrs != expected.attrs or metadata != value["metadata"]:
            raise ValueError("Membership metadata differs from reconstruction")
        _REPLAY_CACHE[str(output)] = signature
    return path, value


def discover(path):
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != ".parquet": return None
    attrs = json.loads((pq.read_schema(path).metadata or {}).get(b"PANDAS_ATTRS", b"{}"))
    pointer = attrs.get(MARKER)
    if pointer is None:
        if path.name == TABLE and (path.parent / BINDING).exists(): raise ValueError("Membership native marker missing")
        return None
    if verified_path(pointer, path.parent) != path.parent / "LAUNCH_PREFLIGHT.json":
        raise ValueError("Membership marker points outside consumed output")
    _, value = validate(record(path.parent / BINDING), path.parent)
    if verified_path(value["membership"], path.parent) != path or value["preflight"] != pointer:
        raise ValueError("Native binding describes another membership table")
    return value


def consumer(path, definition, root):
    value = discover(path)
    if value is None: return None
    launch = read(value["preflight"]["path"])
    candidate_path = verified_path(definition, root); candidate = read(candidate_path)
    candidate_annotation = record(verified_path(candidate["annotation"], candidate_path.parent))
    if candidate_annotation["sha256"] != launch["annotation"]["sha256"]:
        raise ValueError("Membership and candidates use different gene annotations")
    _, sources = gmt(candidate["gmt_manifest"], candidate_path.parent)
    if {k: v["sha256"] for k, v in sources.items()} != {k: v["sha256"] for k, v in launch["gmt_sources"].items()}:
        raise ValueError("Membership and candidates use different GMT collections")
    return {"binding": record(Path(path).parent / BINDING), "candidate_annotation": candidate_annotation,
            "candidate_gmt_manifest": record(verified_path(candidate["gmt_manifest"], candidate_path.parent)),
            "metadata": value["metadata"], "missing_members_dropped": False, "reference_authenticity_verified": False}


def inspect(inventory, path, model, role, fold=-1):
    value = discover(path)
    if value is None: return False
    root = Path(path).parent; launch = read(value["preflight"]["path"])
    _, ref = reference.validate(launch["reference_annotation"], root); ref_launch = read(ref["preflight"]["path"])
    pins = [record(root / BINDING), value["preflight"], value["membership"], launch["input"], launch["reference_annotation"],
            ref["preflight"], ref["annotation"], ref_launch["input"], launch["gtf"], launch["gmt_manifest"],
            *launch["gmt_sources"].values(), *launch["implementations"]]
    for index, pin in enumerate(pins): inventory.bind(pin, root, model, role + ".membership_source." + str(index), fold)
    inventory.gap(model, role, "GENCODE_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED", fold, launch["gtf"]["sha256"])
    inventory.gap(model, role, "GMT_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED", fold, launch["gmt_manifest"]["sha256"])
    inventory.transform_replays.append({"model_id": model, "dependency_context": inventory.context_id,
        "patient_fold_id": fold, "role": role, "format": FORMAT,
        "source_release_sha256": record(root / BINDING)["sha256"], "normalization_sha256": "", "activity_sha256": "",
        "folds_recomputed": 0, "outputs_compared": 1, "model_training_started": False})
    return True
