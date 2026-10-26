"""Pinned GRCh38 GENCODE gene records to the native bulk annotation table.

No download, expression selection or fitting occurs. Parsing and byte replay
prove this transformation, not the authenticity of the supplied reference.
"""
import csv
import gzip
from pathlib import Path
import re

import pandas as pd

from .cnv_input_sources import require_host
from .current_core_export import verified_path
from .formal_expression_sources import read, write, record

ROOT = Path(__file__).resolve().parents[2]
INPUT = "NATIVE_ANNOTATION_WORKFLOW_INPUT_V1"
FORMAT = "NATIVE_ANNOTATION_WORKFLOW_RESULTS_V1"
TABLE = "GENCODE_GENE_ANNOTATION_V1"
OUTPUT = "gene_annotation.tsv"
BINDING = "ANNOTATION_SOURCES.json"
CODE = ("cc_hhgt/v32/native_annotation_workflow.py", "scripts/run_native_annotation_workflow.py")
COLUMNS = ["ensembl_gene_id", "gene_symbol", "gene_type", "versioned_gene_id", "chromosome",
           "start", "end", "strand", "gencode_release", "genome_build", "coordinate_system",
           "gtf_line", "native_annotation_format"]
GENE = re.compile(r"(ENSG[0-9]+)\.([0-9]+)(_PAR_Y)?")
ATTRIBUTE = re.compile(r'\s*([A-Za-z_][A-Za-z_0-9]*)\s+(?:"([^"\r\n]*)"|([^\s;"\r\n]+))\s*;')
_REPLAY_CACHE = {}


def plan(manifest, output, *, new=True):
    require_host(); manifest = Path(manifest).resolve(); output = Path(output).resolve(); spec = read(manifest)
    if (set(spec) != {"format", "gtf", "gencode_release", "genome_build"} or spec.get("format") != INPUT
            or type(spec["gencode_release"]) is not int or spec["gencode_release"] < 25
            or spec["genome_build"] != "GRCh38"):
        raise ValueError("Annotation requires modern GENCODE release >=25 on GRCh38")
    if new and output.exists(): raise FileExistsError("Use a new annotation workflow directory")
    raw = record(verified_path(spec["gtf"], manifest.parent))
    if any(output == p or output in p.parents for p in (manifest, Path(raw["path"]))):
        raise ValueError("Annotation output must not contain its inputs")
    return {"input": record(manifest), "target_host": "149", "training_started": False,
            "gtf": raw, "gencode_release": spec["gencode_release"], "genome_build": spec["genome_build"],
            "implementations": [record(ROOT / name) for name in CODE], "raw_annotation_parsed": False}


def parse_gtf(path, release, build):
    """Retain every gene/contig/biotype and preserve PAR-Y stable identities."""
    require_host(); rows = []; descriptions = []; identities = set()
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.startswith("##description:"): descriptions.append(line.strip())
            if line.startswith("#") or not line.strip(): continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9: raise ValueError(f"Malformed GTF record at line {number}")
            if fields[2] != "gene": continue
            attributes = {}; position = 0
            while position < len(fields[8]) and fields[8][position:].strip():
                match = ATTRIBUTE.match(fields[8], position)
                if match is None: raise ValueError(f"Malformed GTF gene attributes at line {number}")
                key = match[1]; attributes.setdefault(key, []).append(match[2] if match[2] is not None else match[3])
                position = match.end()
            if any(len(attributes.get(key, [])) != 1 or not attributes[key][0].strip()
                   for key in ("gene_id", "gene_name", "gene_type")):
                raise ValueError(f"Missing or ambiguous GTF gene attributes at line {number}")
            gene = attributes["gene_id"][0]; identity = GENE.fullmatch(gene)
            if identity is None: raise ValueError("Unsupported GENCODE gene ID: " + gene)
            stable = identity[1] + (identity[3] or "")
            if stable in identities: raise ValueError("Duplicate stable GENCODE gene ID: " + stable)
            identities.add(stable)
            if (not fields[0] or not fields[3].isdigit() or not fields[4].isdigit()
                    or not 1 <= int(fields[3]) <= int(fields[4]) or fields[6] not in {"+", "-"}):
                raise ValueError(f"Invalid GTF gene coordinates at line {number}")
            rows.append([stable, attributes["gene_name"][0], attributes["gene_type"][0], gene,
                         fields[0], str(int(fields[3])), str(int(fields[4])), fields[6], str(release), build,
                         "1_BASED_CLOSED", str(number), TABLE])
    # An explicit header declaration is required; filename and caller guesses
    # cannot establish a reference build or release. Patch-level build is unknown.
    header_build = re.search(r"\((GRCh38(?:\.p[0-9]+)?)\)", descriptions[0]) if len(descriptions) == 1 else None
    if (header_build is None
            or re.search(r"\bversion\s+(\d+)\b", descriptions[0]) is None
            or int(re.search(r"\bversion\s+(\d+)\b", descriptions[0])[1]) != release):
        raise ValueError("GTF description does not match declared build/release")
    if not rows: raise ValueError("GTF contains no gene records")
    frame = pd.DataFrame(rows, columns=COLUMNS).sort_values("ensembl_gene_id", kind="stable").reset_index(drop=True)
    return frame, {"description": descriptions[0], "gene_count": len(frame), "header_genome_build": header_build[1],
                   "coordinate_system": "1_BASED_CLOSED", "genome_build_patch_verified": False}


def table_bytes(frame):
    return frame.to_csv(sep="\t", index=False, lineterminator="\n").encode("utf-8")


def run(manifest, output):
    value = plan(manifest, output); output = Path(output).resolve()
    frame, metadata = parse_gtf(value["gtf"]["path"], value["gencode_release"], value["genome_build"])
    # Reject malformed annotation before creating an output or launching a consumer.
    if value != plan(manifest, output): raise ValueError("Annotation sources changed during parsing")
    output.mkdir(parents=True); preflight = output / "LAUNCH_PREFLIGHT.json"; write(preflight, value)
    (output / OUTPUT).write_bytes(table_bytes(frame))
    binding = output / BINDING
    write(binding, {"format": FORMAT, "status": "PASS", "target_host": "149", "training_started": False,
                    "preflight": record(preflight), "annotation": record(output / OUTPUT), "metadata": metadata,
                    "raw_to_annotation_replayed": True, "reference_authenticity_verified": False,
                    "source_ancestry_complete": False, "production_deployed": False})
    validate(record(binding), output)
    return binding


def validate(item, root):
    require_host(); path = verified_path(item, root); value = read(path); output = path.parent
    if (path.name != BINDING or value.get("format") != FORMAT or value.get("status") != "PASS"
            or value.get("target_host") != "149" or value.get("raw_to_annotation_replayed") is not True
            or any(value.get(k) is not False for k in ("training_started", "reference_authenticity_verified",
                                                       "source_ancestry_complete", "production_deployed"))):
        raise ValueError("Incomplete native annotation workflow")
    preflight = verified_path(value["preflight"], output)
    annotation = verified_path(value["annotation"], output)
    if preflight != output / "LAUNCH_PREFLIGHT.json" or annotation != output / OUTPUT:
        raise ValueError("Native annotation handoff location differs")
    launch = read(preflight)
    if launch != plan(verified_path(launch["input"], output), output, new=False):
        raise ValueError("Annotation sources or implementation differ")
    # Re-hash raw input, outputs, declaration and implementations above on every
    # invocation. Reuse a parse only for precisely the same verified bytes.
    signature = (record(path), launch, value)
    if _REPLAY_CACHE.get(str(output)) != signature:
        frame, metadata = parse_gtf(launch["gtf"]["path"], launch["gencode_release"], launch["genome_build"])
        if annotation.read_bytes() != table_bytes(frame) or metadata != value["metadata"]:
            raise ValueError("Annotation differs from original GTF replay")
        _REPLAY_CACHE[str(output)] = signature
    return path, value


def discover(path, item=None, root=None):
    """Recognize native tables after consumers normalize away extra pin fields."""
    path = Path(path).resolve(); item = item or {}; sidecar = path.parent / BINDING
    explicit = item.get("upstream_annotation")
    marked = False
    if path.is_file() and path.suffix == ".tsv":
        with path.open(encoding="utf-8") as handle:
            marked = "native_annotation_format" in handle.readline().rstrip("\r\n").split("\t")
    if not explicit and not marked and not (path.name == OUTPUT and sidecar.exists()): return None
    if explicit and verified_path(explicit, root or path.parent) != sidecar:
        raise ValueError("Native annotation pointer differs from the consumed table")
    _, value = validate(record(sidecar), path.parent)
    if verified_path(value["annotation"], path.parent) != path:
        raise ValueError("Native annotation receipt binds another table")
    return value


def annotation_table(item, root):
    path = verified_path(item, root); discover(path, item, root)
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def compare_star(annotation, raw):
    """Check every counted gene's version/type/name, without inferring release."""
    expected = annotation.set_index("versioned_gene_id"); observed = []
    with Path(raw).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("gene_id\t"): break
        else: raise ValueError("STAR gene metadata header missing")
        reader = csv.DictReader(handle, fieldnames=line.rstrip("\r\n").split("\t"), delimiter="\t")
        if not {"gene_id", "gene_name", "gene_type"} <= set(reader.fieldnames):
            raise ValueError("STAR gene metadata columns missing")
        for row in reader:
            gene = row["gene_id"]
            if gene in {"N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous"}: continue
            if gene not in expected.index: raise ValueError("STAR gene/version absent from GTF: " + str(gene))
            known = expected.loc[gene]
            if row["gene_name"] != known.gene_symbol or row["gene_type"] != known.gene_type:
                raise ValueError("STAR gene name/type differs from GTF: " + gene)
            observed.append(gene)
    if not observed or len(observed) != len(set(observed)): raise ValueError("Empty/duplicate STAR gene metadata")
    return {"counted_genes_checked": len(observed), "annotation_genes": len(expected),
            "uncounted_annotation_genes": len(expected) - len(observed), "star_release_independently_verified": False}


def inspect(inventory, path, item, root, model, role, fold):
    value = discover(path, item, root)
    if value is None: return False
    sidecar = Path(path).parent / BINDING
    launch = read(verified_path(value["preflight"], sidecar.parent))
    for name, pin in [("binding", record(sidecar)), ("preflight", value["preflight"]),
                      ("input", launch["input"]), ("raw_gtf", launch["gtf"]),
                      *[("implementation." + str(i), p) for i, p in enumerate(launch["implementations"])]]:
        inventory.bind(pin, sidecar.parent, model, role + ".annotation." + name, fold)
    # A reference is not a patient study. Keep acquisition/authenticity unknown
    # instead of fabricating exposure tokens or claiming independent validation.
    inventory.gap(model, role, "GENCODE_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED", fold, launch["gtf"]["sha256"])
    inventory.transform_replays.append({"model_id": model, "dependency_context": inventory.context_id,
        "patient_fold_id": fold, "role": role, "format": FORMAT, "source_release_sha256": record(sidecar)["sha256"],
        "normalization_sha256": "", "activity_sha256": "", "folds_recomputed": 0,
        "outputs_compared": 1, "model_training_started": False})
    return True
