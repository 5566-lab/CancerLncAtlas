#!/usr/bin/env python3
"""Fetch/parse selected ENCODE depletion contrasts and consume their evidence.

ENCODE_FUNCTIONAL_ANALYSIS_V1 manifest: artifacts inventory, gene_crosswalk,
rbp_crosswalk, pathway_membership, physical_binding, exact_candidates, each
{path,sha256}; reference_annotation_id; contrasts (FunctionalContrast fields
plus orientation_source). --download is explicit; the default reads verified
existing cache files. It never starts training or changes a graph/release.
"""
from __future__ import annotations
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import socket

import pandas as pd

from cc_hhgt.v32.encode_functional import (FunctionalContrast, validate_contrast_selection, verified_encode_file, read_contrast_table,
    parse_functional_table, gene_level_tests, pathway_support, contextual_functional_evidence)
from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32 import encode_functional


def read_table(path):
    return pd.read_parquet(path) if path.is_dir() or path.suffix == ".parquet" else pd.read_csv(path, sep="\t", dtype=str)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--download", action="store_true")
    p.add_argument("--proxy")
    args = p.parse_args(argv)
    if socket.gethostname() != "149": raise RuntimeError("ENCODE functional analysis requires host 149")
    manifest_sha = artifact_sha256(args.manifest)
    code_paths = [Path(__file__), Path(encode_functional.__file__)]
    code_hashes = {str(path.resolve()): artifact_sha256(path) for path in code_paths}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "ENCODE_FUNCTIONAL_ANALYSIS_V1" or not manifest.get("reference_annotation_id"):
        raise ValueError("Functional analysis requires a versioned source manifest and reference annotation")
    paths = {}
    required = ["inventory", "gene_crosswalk", "rbp_crosswalk", "pathway_membership", "physical_binding", "exact_candidates"]
    for role in required:
        record = manifest["artifacts"][role]; path = Path(record["path"])
        if not path.is_absolute(): path = args.manifest.resolve().parent/path
        if artifact_sha256(path) != record["sha256"]: raise ValueError(f"Functional input hash mismatch: {role}")
        paths[role] = path
    tables = {role: read_table(path) for role, path in paths.items()}
    inventory = tables["inventory"]
    identity_columns = ["file_accession", "experiment_accession", "rbp", "biosample", "status", "href", "md5sum", "file_size", "output_type"]
    inventory = inventory[identity_columns].drop_duplicates()
    if inventory.file_accession.duplicated().any(): raise ValueError("Conflicting file metadata in ENCODE inventory")
    inventory = inventory.set_index("file_accession", drop=False)
    rbps = tables["rbp_crosswalk"][["source_rbp_label", "rbp_id"]].drop_duplicates()
    if rbps.isna().any().any() or rbps.source_rbp_label.duplicated().any(): raise ValueError("Ambiguous source RBP crosswalk")
    rbps = rbps.set_index("source_rbp_label").rbp_id.to_dict()
    specs = manifest["contrasts"]
    contrasts = [FunctionalContrast(**{k: v for k, v in spec.items() if k in {f.name for f in fields(FunctionalContrast)}}) for spec in specs]
    validate_contrast_selection(contrasts)
    if not contrasts or len({c.contrast_id for c in contrasts}) != len(contrasts): raise ValueError("Select distinct functional contrasts")
    if len({(c.file_accession, c.member) for c in contrasts}) != len(contrasts): raise ValueError("The same native table cannot be counted twice")
    for spec, contrast in zip(specs, contrasts):
        contrast.validate()
        if not spec.get("orientation_source"): raise ValueError("Native contrast orientation requires source provenance")
        row = inventory.loc[contrast.file_accession]
        if (row.experiment_accession != contrast.experiment_accession or row.biosample != contrast.biosample
                or rbps.get(row.rbp) != contrast.rbp_id): raise ValueError("Selected contrast disagrees with native experiment/RBP/biosample")
        expected_type = "differential splicing quantifications" if contrast.schema == "rmats" else "differential expression quantifications"
        if row.output_type != expected_type: raise ValueError("Wrong ENCODE output type for the selected parser")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host": "149", "training_started": False,
        "manifest_sha256": manifest_sha, "reference_annotation_id": manifest["reference_annotation_id"]}, indent=2))
    output_records = {}; source_records = {}; counts = []; pathway_parts = []; bridge_parts = []
    settings = manifest.get("analysis", {})
    allowed = {"alpha", "min_abs_log2fc", "min_abs_delta_psi", "min_tested_genes"}
    if set(settings)-allowed: raise ValueError("Unknown functional analysis settings")
    def save(frame, name):
        path = args.output/name; frame.to_parquet(path, index=False, compression="zstd")
        output_records[name] = {"path": str(path.resolve()), "sha256": artifact_sha256(path), "rows": len(frame)}
    for contrast in contrasts:
        record = inventory.loc[contrast.file_accession].to_dict()
        source = verified_encode_file(record, args.cache, download=args.download, proxy=args.proxy)
        source_sha = artifact_sha256(source)
        frame = read_contrast_table(source, contrast)
        records = parse_functional_table(frame, contrast, tables["gene_crosswalk"])
        genes = gene_level_tests(records, **{k: v for k, v in settings.items() if k != "min_tested_genes"})
        context = {key: getattr(contrast, key) for key in ("contrast_id", "file_accession", "experiment_accession", "rbp_id", "biosample")}
        context["assay"] = "RBP_DEPLETION_SPLICING" if contrast.schema == "rmats" else "RBP_DEPLETION_EXPRESSION"
        pathways = pathway_support(genes, tables["pathway_membership"], min_tested_genes=settings.get("min_tested_genes", 3),
                                   contexts=pd.DataFrame([context]))
        bridge = contextual_functional_evidence(pathways, tables["physical_binding"], tables["exact_candidates"], alpha=settings.get("alpha", .05))
        key = hashlib.sha256(contrast.contrast_id.encode()).hexdigest()[:20]
        save(records, f"tests_{key}.parquet"); save(genes, f"genes_{key}.parquet")
        if not pathways.empty: pathway_parts.append(pathways)
        if not bridge.empty: bridge_parts.append(bridge)
        if artifact_sha256(source) != source_sha: raise ValueError("ENCODE source changed during parsing")
        source_records[contrast.file_accession] = {"path": str(source.resolve()), "sha256": source_sha, "md5": record["md5sum"]}
        counts.append({"contrast_id": contrast.contrast_id, "native_tests": len(records), "available_tests": int(records.available.sum()),
            "unmapped_tests": int(records.gene_id.isna().sum()), "tested_genes": int(genes.available.sum()),
            "altered_genes": int(genes.altered.sum()), "pathway_tests": len(pathways), "evidence_bridges": len(bridge)})
    save(pd.concat(pathway_parts, ignore_index=True) if pathway_parts else pd.DataFrame(), "RBP_PATHWAY_SUPPORT.parquet")
    save(pd.concat(bridge_parts, ignore_index=True) if bridge_parts else pd.DataFrame(), "FUNCTIONAL_EVIDENCE.parquet")
    save(pd.DataFrame(counts), "COVERAGE.parquet")
    for role, path in paths.items():
        if artifact_sha256(path) != manifest["artifacts"][role]["sha256"]: raise ValueError(f"Functional input changed: {role}")
    if artifact_sha256(args.manifest) != manifest_sha: raise ValueError("Functional manifest changed")
    for path, digest in code_hashes.items():
        if artifact_sha256(path) != digest: raise ValueError("Functional analysis implementation changed during execution")
    run_id = "ENCODE-KD-"+hashlib.sha256(json.dumps([manifest_sha, code_hashes], sort_keys=True).encode()).hexdigest()[:20]
    audit = {"format": "ENCODE_FUNCTIONAL_ANALYSIS_AUDIT_V1", "status": "PASS", "target_host": "149",
        "run_id": run_id, "implementation_sha256": code_hashes, "core_dependency": "CORE_INDEPENDENT_EVIDENCE",
        "estimand": "RBP_DEPLETION_PATHWAY_OVERREPRESENTATION_AND_CONTEXT_MATCHED_LNCRNA_BINDING_BRIDGE",
        "training_started": False, "manifest_sha256": manifest_sha, "source_files": source_records,
        "source_contract": "CACHED_RAW_RBP_ANALYSIS_V1",
        "input_manifest": {"path": str(args.manifest.resolve()), "sha256": manifest_sha}, "cache_root": str(args.cache.resolve()),
        "outputs": output_records, "contrasts": specs, "reference_annotation_id": manifest["reference_annotation_id"],
        "analysis": settings, "primary_graph_changed": False, "lncrna_perturbation_claimed": False,
        "independent_external_validation": False, "biological_experiments": len({c.experiment_accession for c in contrasts}),
        "multiple_source_files_counted_as_independent_replicates": False}
    (args.output/"AUDIT.json").write_text(json.dumps(audit, indent=2)+"\n")
    (args.output/"SUCCESS.json").write_text(json.dumps({"status": "PASS", "audit_sha256": artifact_sha256(args.output/"AUDIT.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
