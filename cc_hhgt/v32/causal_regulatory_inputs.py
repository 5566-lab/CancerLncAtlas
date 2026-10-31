"""Verified, relocatable indexes for experimental RNA-response networks.

Source-native measurements stay on disk.  The index retains all experiments,
including ineligible ones, and joins known study/sample identities across
sources.  It does not invent a cross-study train/test split.
"""
from __future__ import annotations

import hashlib
import json
import re
import socket
from pathlib import Path

import pandas as pd

FORMAT = "CAUSAL_REGULATORY_INPUTS_V1"
SOURCES = ("PerturbAtlas", "GPSAdb2", "LINCS_iLINCS_OE", "L2S2_KO",
           "GEO_rescue_GSE154812", "LncRNA2Target3", "LncTarD2", "CasRx2024")
MOLECULAR_CLASSES = {"lncRNA", "protein_coding"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def truth(value):
    return str(value).strip().lower() in {"true", "1", "1.0"}


def text(value):
    return "" if value is None or pd.isna(value) else str(value).strip()


def stable_gene(value):
    value = re.sub(r"^(GENE:|LNC:)", "", text(value))
    return re.sub(r"\.\d+$", "", value) if value.startswith("ENSG") else value


class SourceFiles:
    """Every consumed file must be declared and its actual bytes verified."""

    def __init__(self, root, expected_manifest_sha256=None):
        self.root = Path(root).resolve()
        path = self.root / "input_manifest.json"
        self.manifest_sha256 = sha256(path)
        if expected_manifest_sha256 and self.manifest_sha256 != expected_manifest_sha256:
            raise ValueError("Causal input manifest hash mismatch")
        self.manifest = json.loads(path.read_text(encoding="utf-8"))
        self.files = {}
        self.verified = set()
        for row in self.manifest["files"]:
            name = row["path"]
            path = (self.root / name).resolve()
            if (Path(name).is_absolute() or not path.is_relative_to(self.root)
                    or name in self.files or not re.fullmatch(r"[a-f0-9]{64}", row["sha256"])):
                raise ValueError("Unsafe, duplicate or unpinned source file")
            self.files[name] = row

    def path(self, name):
        if name not in self.files:
            raise ValueError("Undeclared causal input: " + name)
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Source path escaped its root")
        row = self.files[name]
        if name not in self.verified:
            if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
                raise ValueError("Causal source bytes changed: " + name)
            self.verified.add(name)
        return path

    def table(self, name):
        return pd.read_csv(self.path(name), sep="\t", dtype=str, keep_default_na=False)

    def verify_all(self):
        for name in self.files:
            self.path(name)
        if sha256(self.root / "input_manifest.json") != self.manifest_sha256:
            raise ValueError("Source manifest changed during verification")


def entity_catalog(files):
    rows = []
    for source in ("PerturbAtlas", "GPSAdb2", "GEO_rescue_GSE154812"):
        table = files.table("lncRNA_edge_audit/" + source + "_readout_annotation.tsv")
        id_col = "ensembl_gene_id" if source.startswith("GEO_") else "gene_id"
        for row in table.to_dict("records"):
            gene = stable_gene(row[id_col])
            if gene.startswith("ENSG"):
                rows.append((gene, row["hgnc_readout_symbol"], row["hgnc_readout_class"]))
    frame = pd.DataFrame(rows, columns=["gene_id", "symbol", "gene_class"]).drop_duplicates()
    # Ambiguous IDs/symbols are never resolved by row order.
    ambiguous = frame.groupby("gene_id").gene_class.nunique()
    frame = frame.loc[~frame.gene_id.isin(ambiguous[ambiguous > 1].index)]
    return frame.drop_duplicates("gene_id").reset_index(drop=True)


def _tokens(*values):
    return set(re.findall(r"\b(?:GSE|GSM|SRP|SRX|SRR|ERP|ERX|ERR|PRJNA|PRJEB)\d+\b",
                          " ".join(text(x) for x in values).upper()))


def _pmid(value):
    value = text(value)
    return "PMID:" + value.split(".")[0] if re.fullmatch(r"\d+(?:\.0)?", value) else ""


def _base(source, identifier, **values):
    row = dict(source=source, experiment_id=str(identifier), intervention_type="UNKNOWN",
               target_id="", target_symbol="", target_class="", cell_context="",
               study_ids="", sample_ids="", source_split="", row_index=-1,
               source_method="", time="", dose="", n_control="", n_treatment="",
               network_eligible=False, review_reason="", mechanism_holdout=False,
               background="", readout="RNA_abundance", usage="RNA_response",
               source_record_id=str(identifier))
    row.update(values)
    return row


def experiment_index(files):
    rows = []
    specifications = (
        ("PerturbAtlas", "contrasts.tsv", "perturb_id", "condition", "tissue_name", "study_accession", "split", "row_index"),
        ("GPSAdb2", "contrasts.tsv", "dataset_id", "intervention", "cell_line", "study_gse", "split", None),
        ("LINCS_iLINCS_OE", "signature_metadata.tsv", "signatureid", None, "cellline", "plate", "split_by_plate", "matrix_signature_index"),
        ("L2S2_KO", "signatures.tsv", "signature_id", "intervention", "cell_line", "plate", "split", "signature_index"),
    )
    for source, filename, id_col, method_col, context_col, study_col, split_col, index_col in specifications:
        table = files.table(source + "/" + filename)
        if table[id_col].duplicated().any():
            raise ValueError("Duplicate experiment identities in " + source)
        if index_col and set(pd.to_numeric(table[index_col], errors="raise")) != set(range(len(table))):
            raise ValueError("Non-bijective matrix row index in " + source)
        sample_tokens = {}
        if source in {"PerturbAtlas", "GPSAdb2"}:
            samples = files.table(source + "/samples.tsv")
            key = "contrast_id" if source == "PerturbAtlas" else "dataset_id"
            for sample in samples.to_dict("records"):
                sample_tokens.setdefault(sample[key], set()).update(_tokens(*sample.values()))
        for item in table.to_dict("records"):
            original_method = item.get(method_col, "OE") if method_col else "OE"
            method = {"CRISPR_KO": "KO", "shRNA": "KD", "CRISPRi": "KD"}.get(original_method, original_method)
            if original_method == "shRNA_label_requires_source_review_NOT_KO":
                method = "KD"
            target = stable_gene(item["canonical_target"])
            symbol = item["target_hgnc_symbol"]
            eligible = (truth(item.get("candidate_primary", "True"))
                        and item["target_class"] in MOLECULAR_CLASSES
                        and target.startswith("ENSG") and method in {"KO", "KD", "OE"})
            review = []
            if not eligible:
                review.append("SOURCE_OR_TARGET_OR_INTERVENTION_REVIEW")
            if truth(item.get("lncRNA_target_requires_construct_or_guide_review", "False")):
                eligible = False
                review.append("CONSTRUCT_OR_GUIDE_REVIEW")
            tokens = _tokens(item.get(study_col), item.get("controls"), item.get("perturbs"))
            tokens.update(sample_tokens.get(item[id_col], set()))
            if source in {"LINCS_iLINCS_OE", "L2S2_KO"}:
                # Plate namespace is shared across the L1000 sources.
                tokens.add("L1000_PLATE:" + item[study_col])
            if not tokens:
                tokens.add(source + ":EXPERIMENT:" + item[id_col])
            rows.append(_base(source, item[id_col], intervention_type=method,
                target_id=target, target_symbol=symbol, target_class=item["target_class"],
                cell_context=item[context_col] or item.get("tissue", ""),
                study_ids=";".join(sorted(tokens)), source_split=item[split_col],
                row_index=int(item[index_col]) if index_col else -1,
                source_method=item.get("source_method", original_method),
                time=item.get("time", ""), dose=item.get("concentration", item.get("source_dose_label", "")),
                n_control=item.get("n_control", item.get("n_control_samples", item.get("nCtrSamples", ""))),
                n_treatment=item.get("n_treatment", item.get("n_treatment_samples", item.get("nTrtSamples", ""))),
                network_eligible=eligible, review_reason=";".join(review),
                mechanism_holdout=symbol == "NORAD" or "mechanism" in item[split_col].lower()))
    for item in files.table("LncRNA2Target3/experiments_human.tsv").to_dict("records"):
        method = {"Knockdown": "KD", "Overexpression": "OE", "Overexpressed":"OE"}.get(item["intervention_source"], "UNKNOWN")
        tokens = _tokens(item["gse"])
        if pmid := _pmid(item["pmid"]):
            tokens.add(pmid)
        eligible = (not truth(item["suspected_source_column_misalignment"])
                    and item["perturbed_hgnc_class"] in MOLECULAR_CLASSES
                    and method in {"KD", "OE"} and stable_gene(item["perturbed_entity_id"]).startswith("ENSG"))
        rows.append(_base("LncRNA2Target3", item["experiment_id"], intervention_type=method,
            target_id=stable_gene(item["perturbed_entity_id"]), target_symbol=item["perturbed_hgnc_symbol"],
            target_class=item["perturbed_hgnc_class"], cell_context=item["cell_line"] or item["tissue"],
            study_ids=";".join(sorted(tokens)) or "LncRNA2Target3:" + item["experiment_id"],
            source_method=item["intervention_source"], network_eligible=eligible,
            review_reason="" if eligible else "SOURCE_NUMERIC_OR_TARGET_REVIEW",
            mechanism_holdout=item["perturbed_hgnc_symbol"] == "NORAD", usage="RNA_response_direction_only"))
    rescue_tokens = {"PMID:34108682"}
    for sample in files.table("GEO_rescue_GSE154812/samples.tsv").to_dict("records"):
        rescue_tokens.update(_tokens(*sample.values()))
    for item in files.table("GEO_rescue_GSE154812/contrasts.tsv").to_dict("records"):
        identifier = item["contrast_id"]
        method = {"NORAD_CRISPRi": "KD", "circPRE2": "RESCUE_CONTROL", "circPRE8": "RESCUE"}[identifier]
        rows.append(_base("GEO_rescue_GSE154812", identifier, intervention_type=method,
            target_id="SYMBOL:NORAD" if method == "KD" else "CONSTRUCT:" + identifier,
            target_symbol="NORAD" if method == "KD" else identifier,
            target_class="lncRNA" if method == "KD" else "synthetic_RNA",
            cell_context="HCT116", study_ids=";".join(sorted(rescue_tokens)),
            source_method=item["kind"], network_eligible=True, mechanism_holdout=True,
            background=item["control"], usage=item["rescue_type"], n_control="2", n_treatment="2"))
    # Curated relations retain their own semantics, not guessed KO/OE labels.
    curated = files.table("LncTarD2/curated_relations_all.tsv.gz")
    for item in curated.to_dict("records"):
        rows.append(_base("LncTarD2", item["record_id"], target_id=stable_gene(item["regulator_entity_id"]),
            target_symbol=item["regulator_hgnc_symbol"], target_class=item["regulator_hgnc_class"],
            cell_context=item["DiseaseName"], study_ids=_pmid(item["PubMedID"]) or item["study_group"],
            source_method=item["intervention_type"], usage=item["evidence_layer"],
            readout=item["levelOfRegulation"], review_reason="INTERVENTION_NOT_STRUCTURED",
            mechanism_holdout=item["regulator_hgnc_symbol"] == "NORAD"))
    rows.append(_base("CasRx2024", "fitness", study_ids="PRJEB60776",
                     readout="cell_fitness", usage="AUXILIARY_FITNESS",
                     review_reason="FAMILY_FITNESS_IS_NOT_MOLECULAR_EDGE"))
    result = pd.DataFrame(rows)
    if result.duplicated(["source", "experiment_id"]).any():
        raise ValueError("Duplicate source/experiment identity")
    return group_known_studies(result)


def group_known_studies(frame):
    """Connected known identifiers; any mechanism holdout embargoes its group."""
    frame = frame.copy()
    parents = list(range(len(frame)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    first = {}
    for i, value in enumerate(frame.study_ids):
        for token in filter(None, value.split(";")):
            if token in first:
                parents[find(i)] = find(first[token])
            else:
                first[token] = i
    components = {}
    for i in range(len(frame)):
        components.setdefault(find(i), []).append(i)
    group_ids = {}; embargo = {}
    for indexes in components.values():
        tokens = sorted({t for i in indexes for t in frame.iloc[i].study_ids.split(";") if t})
        group = "STUDY:" + hashlib.sha256("\n".join(tokens).encode()).hexdigest()[:24]
        holdout = any(bool(frame.iloc[i].mechanism_holdout) for i in indexes)
        for i in indexes:
            group_ids[i] = group
            embargo[i] = holdout
    frame["known_study_group"] = [group_ids[i] for i in range(len(frame))]
    frame["mechanism_holdout"] = [embargo[i] for i in range(len(frame))]
    frame["split"] = ["mechanism_validation" if embargo[i] else "unassigned" for i in range(len(frame))]
    return frame


def build_inputs(source_root, output, *, expected_manifest_sha256):
    if socket.gethostname() != "149":
        raise RuntimeError("Causal network preparation must execute on host 149")
    output = Path(output).resolve()
    source_root = Path(source_root).resolve()
    if output == source_root or output.is_relative_to(source_root):
        raise ValueError("Keep prepared outputs outside the immutable source directory")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "LAUNCH_PREFLIGHT.json", dict(target_host="149", operation="INDEX_CAUSAL_NETWORKS",
        source_manifest_sha256=expected_manifest_sha256, training_started=False))
    files = SourceFiles(source_root, expected_manifest_sha256)
    files.verify_all()
    index = experiment_index(files)
    entities = entity_catalog(files)
    index.to_parquet(output / "EXPERIMENTS.parquet", index=False)
    entities.to_parquet(output / "ENTITIES.parquet", index=False)
    manifest = dict(format=FORMAT, target_host="149", status="PASS",
        source_manifest_sha256=files.manifest_sha256, source_root=str(source_root),
        sources=list(SOURCES), source_files_verified=len(files.verified),
        experiments=len(index), experiments_by_source=index.source.value_counts().to_dict(),
        index={"path":"EXPERIMENTS.parquet", "sha256":sha256(output / "EXPERIMENTS.parquet")},
        entities={"path":"ENTITIES.parquet", "sha256":sha256(output / "ENTITIES.parquet")},
        implementation_sha256=sha256(__file__), training_started=False,
        split_policy="KNOWN_STUDY_COMPONENTS_WITH_MECHANISM_EMBARGO_NO_AUTOMATIC_TRAIN_TEST_ASSIGNMENT",
        cross_source_split_ready=False,
        scope="Experimental response networks; no physical-binding or patient causal-effect claim")
    write_json(output / "COMPLETE.json", manifest)
    return manifest
