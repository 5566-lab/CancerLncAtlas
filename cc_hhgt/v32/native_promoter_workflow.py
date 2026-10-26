"""Original transcript promoter geometry with bound stable GENCODE identities."""
import base64
import gzip
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, write, record
from . import native_annotation_workflow as reference

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_PROMOTER_INPUT_V1"
FORMAT = "NATIVE_PROMOTER_RESULTS_V1"
TABLE = "native_transcript_promoters.bed.gz"
ORIGINAL = "original_transcript_promoters.bed.gz"
BINDING = "NATIVE_PROMOTER_BINDING.json"
MARKER = "native_promoter_preflight="
CODE = ("cc_hhgt/v32/native_promoter_workflow.py", "scripts/run_native_promoter_workflow.py",
        "scripts/derive_gencode_promoters.py", *reference.CODE)
_REPLAY_CACHE = {}


def plan(manifest, output, *, new=True):
    require_host(); manifest = Path(manifest).resolve(); output = Path(output).resolve(); spec = read(manifest)
    if set(spec) != {"format", "reference_annotation"} or spec.get("format") != INPUT:
        raise ValueError("Invalid native transcript promoter declaration")
    if new and output.exists(): raise FileExistsError("Use a new promoter directory")
    path, value = reference.validate(spec["reference_annotation"], manifest.parent)
    launch = read(value["preflight"]["path"])
    pins = [record(manifest), record(path), value["annotation"], launch["gtf"]]
    if any(output == Path(p["path"]) or output in Path(p["path"]).parents for p in pins):
        raise ValueError("Promoter output must not contain its inputs")
    return {"input": record(manifest), "reference_annotation": record(path), "annotation": value["annotation"],
            "gtf": launch["gtf"], "genome_build": launch["genome_build"], "gencode_release": launch["gencode_release"],
            "target_host": "149", "training_started": False, "implementations": [record(ROOT/n) for n in CODE]}


def transcripts(value):
    """Validate original selected transcript rows against the actual parent genes."""
    from scripts.derive_gencode_promoters import CHROM_ORDER, parse_attrs
    annotation = reference.annotation_table(value["annotation"], ROOT).set_index("versioned_gene_id")
    path = Path(value["gtf"]["path"]); opener = gzip.open if path.suffix == ".gz" else open
    seen = set(); selected = {}; total = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.startswith("#") or not line.strip(): continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9: raise ValueError("Malformed GTF record")
            if fields[2] != "transcript": continue
            total += 1; attrs = {}; position = 0
            while position < len(fields[8]) and fields[8][position:].strip():
                match = reference.ATTRIBUTE.match(fields[8], position)
                if match is None: raise ValueError("Malformed transcript attributes")
                attrs.setdefault(match[1], []).append(match[2] if match[2] is not None else match[3]); position = match.end()
            if any(len(attrs.get(k, [])) != 1 or not attrs[k][0] for k in ("gene_id", "transcript_id", "gene_name")):
                raise ValueError("Missing/ambiguous transcript identity")
            gene, transcript = attrs["gene_id"][0], attrs["transcript_id"][0]
            if gene not in annotation.index or transcript in seen:
                raise ValueError("Transcript has unknown parent or duplicate identity")
            seen.add(transcript); parent = annotation.loc[gene]
            kind = attrs.get("gene_type", attrs.get("gene_biotype", []))
            if (len(kind) != 1 or kind[0] != parent.gene_type or attrs["gene_name"][0] != parent.gene_symbol
                    or fields[0] != parent.chromosome or fields[6] != parent.strand
                    or not fields[3].isdigit() or not fields[4].isdigit()
                    or not int(parent.start) <= int(fields[3]) <= int(fields[4]) <= int(parent.end)):
                raise ValueError("Transcript coordinates or identity differ from parent gene")
            original = parse_attrs(fields[8])
            if any(original.get(k) != attrs[k][0] for k in ("gene_id", "transcript_id", "gene_name")):
                raise ValueError("Transcript attributes cannot be read by the original promoter producer")
            if original.get("gene_type", original.get("gene_biotype")) != kind[0]:
                raise ValueError("Original transcript biotype parsing differs")
            if fields[0] in CHROM_ORDER:
                selected[transcript] = {"gene_id": parent.ensembl_gene_id, "versioned_gene_id": gene, "gtf_line": number}
    if not selected: raise ValueError("No reference-chromosome transcript promoters")
    return selected, total


def compressed(data):
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as stream: stream.write(data)
    return buffer.getvalue()


def derive(value, preflight):
    selected, total = transcripts(value)
    with tempfile.TemporaryDirectory(prefix="native_promoter_") as temporary:
        folder = Path(temporary); source = Path(value["gtf"]["path"])
        if source.suffix != ".gz":
            staged = folder/"source.gtf.gz"; staged.write_bytes(compressed(source.read_bytes())); source = staged
        bed = folder/ORIGINAL; metadata = folder/"metadata.json"
        subprocess.run([sys.executable, str(ROOT/"scripts/derive_gencode_promoters.py"), "--gtf", str(source),
            "--output", str(bed), "--metadata", str(metadata)], check=True, capture_output=True, text=True, timeout=300)
        original = bed.read_bytes(); info = read(metadata)
        frame = pd.read_csv(bed, sep="\t", dtype=str, keep_default_na=False).iloc[:, :8]
    if len(frame) != len(selected) or set(frame.transcript_id) != set(selected):
        raise ValueError("Original producer transcript coverage differs")
    frame["versioned_gene_id"] = frame.gene_id
    expected = [selected[t] for t in frame.transcript_id]
    if frame.gene_id.tolist() != [r["versioned_gene_id"] for r in expected]:
        raise ValueError("Original producer gene identities differ")
    frame["gene_id"] = [r["gene_id"] for r in expected]
    frame["gtf_line"] = [r["gtf_line"] for r in expected]
    marker = MARKER + base64.urlsafe_b64encode(json.dumps(record(preflight), sort_keys=True).encode()).decode()
    frame[marker] = ""
    data = compressed(frame.to_csv(sep="\t", index=False, lineterminator="\n").encode())
    info = {k: v for k,v in info.items() if k not in {"source_gtf", "source_gtf_sha256", "output", "output_sha256"}}
    info.update(transcript_records=total, excluded_nonreference_transcripts=total-len(frame),
        gene_id_policy="GENCODE_STABLE_ID_PAR_Y_PRESERVED_WITH_ORIGINAL_VERSIONED_ID",
        transcript_rows_collapsed=False, chromosome_end_clipping_applied=False)
    return data, original, info


def run(manifest, output):
    value = plan(manifest, output); transcripts(value)
    output = Path(output).resolve(); output.mkdir(parents=True); preflight = output/"LAUNCH_PREFLIGHT.json"; write(preflight, value)
    try:
        data, original, metadata = derive(value, preflight)
        (output/TABLE).write_bytes(data); (output/ORIGINAL).write_bytes(original)
        if value != plan(manifest, output, new=False): raise ValueError("Promoter source changed")
        binding = output/BINDING
        write(binding, {"format": FORMAT, "status": "PASS", "target_host": "149", "training_started": False,
            "preflight": record(preflight), "promoters": record(output/TABLE), "original_promoters": record(output/ORIGINAL),
            "metadata": metadata, "original_geometry_reconstructed": True, "reference_authenticity_verified": False,
            "source_ancestry_complete": False, "production_deployed": False})
        validate(record(binding), output); return binding
    except Exception as error:
        write(output/"RUN_FAILED.json", {"target_host": "149", "training_started": False, "error": str(error)}); raise


def validate(item, root):
    require_host(); path = verified_path(item, root); output = path.parent; value = read(path)
    if (path.name != BINDING or value.get("format") != FORMAT or value.get("status") != "PASS"
            or value.get("target_host") != "149" or value.get("original_geometry_reconstructed") is not True
            or any(value.get(k) is not False for k in ("training_started", "reference_authenticity_verified", "source_ancestry_complete", "production_deployed"))):
        raise ValueError("Incomplete native promoter workflow")
    for key, name in (("preflight", "LAUNCH_PREFLIGHT.json"), ("promoters", TABLE), ("original_promoters", ORIGINAL)):
        if verified_path(value[key], output) != output/name: raise ValueError("Promoter handoff location differs")
    launch = read(output/"LAUNCH_PREFLIGHT.json")
    if launch != plan(verified_path(launch["input"], output), output, new=False):
        raise ValueError("Promoter source or implementation differs")
    signature = (record(path), launch, value)
    if _REPLAY_CACHE.get(str(output)) != signature:
        data, original, metadata = derive(launch, output/"LAUNCH_PREFLIGHT.json")
        if data != (output/TABLE).read_bytes() or original != (output/ORIGINAL).read_bytes() or metadata != value["metadata"]:
            raise ValueError("Promoters differ from original GTF reconstruction")
        _REPLAY_CACHE[str(output)] = signature
    return path, value


def discover(path):
    path = Path(path).resolve()
    if not path.is_file() or not path.name.endswith(".bed.gz"): return None
    with gzip.open(path, "rt", encoding="utf-8") as handle: columns = handle.readline().rstrip("\r\n").split("\t")
    markers = [c[len(MARKER):] for c in columns if c.startswith(MARKER)]
    if not markers:
        if path.name == TABLE and (path.parent/BINDING).exists(): raise ValueError("Native promoter marker missing")
        return None
    if len(markers) != 1: raise ValueError("Ambiguous promoter marker")
    pointer = json.loads(base64.urlsafe_b64decode(markers[0]))
    if verified_path(pointer, path.parent) != path.parent/"LAUNCH_PREFLIGHT.json":
        raise ValueError("Promoter marker points outside consumed output")
    _, value = validate(record(path.parent/BINDING), path.parent)
    if verified_path(value["promoters"], path.parent) != path or value["preflight"] != pointer:
        raise ValueError("Promoter binding describes another table")
    return value


def consumer(path, definition=None, root=None):
    value = discover(path)
    if value is None: return None
    launch = read(value["preflight"]["path"])
    result = {"binding": record(Path(path).resolve().parent/BINDING), "metadata": value["metadata"],
              "genome_build": launch["genome_build"], "reference_authenticity_verified": False}
    if definition is not None:
        candidate_path = verified_path(definition, root); candidate = read(candidate_path)
        annotation = record(verified_path(candidate["annotation"], candidate_path.parent))
        if annotation["sha256"] != launch["annotation"]["sha256"]:
            raise ValueError("Promoters and candidates use different gene annotations")
        result["candidate_annotation"] = annotation
    return result


def source_files(path):
    value = discover(path)
    if value is None: return []
    root = Path(path).resolve().parent; launch = read(value["preflight"]["path"])
    _, ref = reference.validate(launch["reference_annotation"], root); ref_launch = read(ref["preflight"]["path"])
    return [record(root/BINDING), value["preflight"], value["promoters"], value["original_promoters"], launch["input"],
        launch["reference_annotation"], ref["preflight"], ref["annotation"], ref_launch["input"], launch["gtf"], *launch["implementations"]]


def inspect(inventory, path, model, role, fold=-1):
    pins = source_files(path)
    if not pins: return False
    for index, pin in enumerate(pins): inventory.bind(pin, Path(path).parent, model, role+".promoter_source."+str(index), fold)
    inventory.gap(model, role, "GENCODE_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED", fold, pins[9]["sha256"])
    inventory.transform_replays.append({"model_id": model, "dependency_context": inventory.context_id,
        "patient_fold_id": fold, "role": role, "format": FORMAT, "source_release_sha256": pins[0]["sha256"],
        "normalization_sha256": "", "activity_sha256": "", "folds_recomputed": 0, "outputs_compared": 2, "model_training_started": False})
    return True
