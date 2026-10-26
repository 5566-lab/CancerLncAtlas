"""Original GTF-to-single-cell gene cache, bound to the native reference entry.

Preserves the original primary-contig/biotype/symbol and ID mapping rules. It
does not perform cell QC, acquire a reference or fit any representation/model.
"""
import json
from pathlib import Path
import tempfile

import pandas as pd
import pyarrow.parquet as pq

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, write, record
from . import native_annotation_workflow as reference
from . import single_cell_partition_builder as original

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_SINGLE_CELL_ANNOTATION_INPUT_V1"
FORMAT = "NATIVE_SINGLE_CELL_ANNOTATION_RESULTS_V1"
TABLE = "gene_annotation.parquet"
PROVENANCE = "GENCODE_PROVENANCE.json"
BINDING = "NATIVE_SINGLE_CELL_ANNOTATION_BINDING.json"
MARKER = "native_gene_annotation_preflight"
CODE = ("cc_hhgt/v32/native_single_cell_annotation_workflow.py", "scripts/run_native_single_cell_annotation_workflow.py",
        "cc_hhgt/v32/single_cell_partition_builder.py", "cc_hhgt/v32/single_cell_input_builder.py", *reference.CODE)
_REPLAY_CACHE = {}


def plan(manifest, output, *, new=True):
    require_host(); manifest = Path(manifest).resolve(); output = Path(output).resolve(); spec = read(manifest)
    if set(spec) != {"format", "reference_annotation"} or spec.get("format") != INPUT:
        raise ValueError("Invalid native single-cell annotation declaration")
    if new and output.exists(): raise FileExistsError("Use a new single-cell annotation directory")
    source, binding = reference.validate(spec["reference_annotation"], manifest.parent)
    launch = read(verified_path(binding["preflight"], source.parent))
    if any(output == p or output in p.parents for p in (manifest, source, Path(launch["gtf"]["path"]))):
        raise ValueError("Single-cell annotation output must not contain its sources")
    return {"input": record(manifest), "reference_annotation": record(source), "gtf": launch["gtf"],
            "gencode_release": launch["gencode_release"], "genome_build": launch["genome_build"],
            "target_host": "149", "training_started": False,
            "implementations": [record(ROOT / name) for name in CODE]}


def invoke(value, output, preflight):
    return original.prepare_gencode_annotation(gtf_path=value["gtf"]["path"],
        output_parquet=output / TABLE, provenance_path=output / PROVENANCE,
        native_annotation_preflight=record(preflight))


def replay(value, output, preflight):
    actual = read(output / PROVENANCE)
    with tempfile.TemporaryDirectory(prefix="native_sc_annotation_replay_") as temporary:
        other = Path(temporary); expected = invoke(value, other, preflight)
        left = pd.read_parquet(output / TABLE); right = pd.read_parquet(other / TABLE)
        try: pd.testing.assert_frame_equal(left, right, check_exact=True)
        except AssertionError as error: raise ValueError("Single-cell annotation differs from original GTF reconstruction") from error
        if left.attrs != right.attrs: raise ValueError("Single-cell annotation source marker differs")
        # Reconstruct the original provenance fields, allowing only the actual
        # serialization's path/hash to differ from the temporary replay output.
        expected["annotation_path"] = str(output / TABLE)
        expected["annotation_sha256"] = record(output / TABLE)["sha256"]
        expected.pop("contract_sha256")
        expected["contract_sha256"] = original._canonical_sha256(expected)
        if actual != expected: raise ValueError("Single-cell GTF provenance differs from reconstruction")


def run(manifest, output):
    value = plan(manifest, output); output = Path(output).resolve(); output.mkdir(parents=True)
    preflight = output / "LAUNCH_PREFLIGHT.json"; write(preflight, value)
    try:
        invoke(value, output, preflight); replay(value, output, preflight)
        if value != plan(manifest, output, new=False): raise ValueError("Single-cell annotation sources changed")
        binding = output / BINDING
        write(binding, {"format": FORMAT, "status": "PASS", "target_host": "149", "training_started": False,
            "preflight": record(preflight), "annotation": record(output / TABLE), "provenance": record(output / PROVENANCE),
            "original_annotation_replayed": True, "source_ancestry_complete": False,
            "reference_authenticity_verified": False, "production_deployed": False})
        validate(record(binding), output)
        return binding
    except Exception as error:
        write(output / "RUN_FAILED.json", {"target_host": "149", "training_started": False, "error": str(error)})
        raise


def validate(item, root):
    require_host(); path = verified_path(item, root); output = path.parent; value = read(path)
    if (path.name != BINDING or value.get("format") != FORMAT or value.get("status") != "PASS"
            or value.get("target_host") != "149" or value.get("original_annotation_replayed") is not True
            or any(value.get(k) is not False for k in ("training_started", "source_ancestry_complete",
                                                       "reference_authenticity_verified", "production_deployed"))):
        raise ValueError("Incomplete native single-cell annotation")
    for key, name in (("preflight", "LAUNCH_PREFLIGHT.json"), ("annotation", TABLE), ("provenance", PROVENANCE)):
        if verified_path(value[key], output) != output / name: raise ValueError("Single-cell annotation handoff differs")
    preflight = output / "LAUNCH_PREFLIGHT.json"; launch = read(preflight)
    if launch != plan(verified_path(launch["input"], output), output, new=False):
        raise ValueError("Single-cell annotation sources or implementation differ")
    signature = (record(path), launch, value)
    if _REPLAY_CACHE.get(str(output)) != signature:
        replay(launch, output, preflight); _REPLAY_CACHE[str(output)] = signature
    return path, value


def discover(path):
    """Read only Parquet metadata to identify marked native reference tables."""
    path = Path(path).resolve()
    if not path.is_file() or path.suffix != ".parquet": return None
    metadata = pq.read_schema(path).metadata or {}
    attrs = json.loads(metadata.get(b"PANDAS_ATTRS", b"{}"))
    pointer = attrs.get(MARKER)
    if pointer is None:
        if path.name == TABLE and (path.parent / BINDING).is_file():
            raise ValueError("Single-cell annotation lost its native source marker")
        return None
    preflight = verified_path(pointer, path.parent)
    if preflight != path.parent / "LAUNCH_PREFLIGHT.json":
        raise ValueError("Single-cell annotation marker points outside the consumed output")
    _, value = validate(record(path.parent / BINDING), path.parent)
    if verified_path(value["annotation"], path.parent) != path or value["preflight"] != pointer:
        raise ValueError("Single-cell annotation binding differs from consumed source")
    return value


def source_files(path):
    value = discover(path)
    if value is None: return None, []
    root = Path(path).parent; launch = read(verified_path(value["preflight"], root))
    ref_path, ref = reference.validate(launch["reference_annotation"], root)
    ref_launch = read(verified_path(ref["preflight"], ref_path.parent))
    pins = [record(root / BINDING), value["preflight"], value["annotation"], value["provenance"],
            launch["input"], launch["reference_annotation"], launch["gtf"], *launch["implementations"],
            ref["preflight"], ref["annotation"], ref_launch["input"], *ref_launch["implementations"]]
    return value, list({pin["path"]: pin for pin in pins}.values())


def inspect(inventory, path, model, role, fold=-1):
    value, pins = source_files(path)
    if value is None: return False
    for index, pin in enumerate(pins): inventory.bind(pin, Path(path).parent, model, role + ".annotation_source." + str(index), fold)
    launch = read(value["preflight"]["path"])
    inventory.gap(model, role, "GENCODE_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED", fold, launch["gtf"]["sha256"])
    inventory.transform_replays.append({"model_id": model, "dependency_context": inventory.context_id,
        "patient_fold_id": fold, "role": role, "format": FORMAT,
        "source_release_sha256": record(Path(path).parent / BINDING)["sha256"],
        "normalization_sha256": "", "activity_sha256": "", "folds_recomputed": 0,
        "outputs_compared": 2, "model_training_started": False})
    return True
