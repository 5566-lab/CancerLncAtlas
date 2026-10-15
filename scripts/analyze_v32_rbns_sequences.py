#!/usr/bin/env python3
"""Score version-matched lncRNA transcripts and materialize predicted RBP edges.

Manifest RBNS_SEQUENCE_ANALYSIS_V1 declares annotation_id; artifacts inventory,
rbp_crosswalk, transcript_fasta, transcript_crosswalk, lncrnas, protein_gene_map
as {path,sha256}; profiles with profile_id/file_accession/rbp_id/concentration;
and optional scoring thresholds. Native concentration is prespecified, never
selected per lncRNA. The default uses verified existing files without download.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import socket

import duckdb
import pandas as pd

from cc_hhgt.v32.encode_functional import verified_encode_file
from cc_hhgt.v32.input_lineage import artifact_sha256
from cc_hhgt.v32.rbns_sequence import (parse_rbns_profile, iter_transcript_scores, aggregate_gene_scores, predicted_binding_inputs)
from scripts.analyze_v32_encode_functional import read_table
from cc_hhgt.v32 import rbns_sequence, encode_functional, rbp_typed_binding, rbp_assay


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--proxy")
    parser.add_argument("--chunk-rows", type=int, default=50000)
    args = parser.parse_args(argv)
    if socket.gethostname() != "149": raise RuntimeError("RBNS analysis requires target host 149")
    if args.chunk_rows < 1: raise ValueError("RBNS output chunk size must be positive")
    manifest_sha = artifact_sha256(args.manifest)
    code_paths = [Path(__file__), Path(rbns_sequence.__file__), Path(encode_functional.__file__),
                  Path(rbp_typed_binding.__file__), Path(rbp_assay.__file__)]
    code_hashes = {str(path.resolve()): artifact_sha256(path) for path in code_paths}
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "RBNS_SEQUENCE_ANALYSIS_V1" or not manifest.get("annotation_id"):
        raise ValueError("RBNS requires a versioned input manifest and sequence annotation")
    roles = ["inventory", "rbp_crosswalk", "transcript_fasta", "transcript_crosswalk", "lncrnas", "protein_gene_map"]
    paths = {}
    for role in roles:
        record = manifest["artifacts"][role]; path = Path(record["path"])
        if not path.is_absolute(): path = args.manifest.resolve().parent/path
        if artifact_sha256(path) != record["sha256"]: raise ValueError(f"RBNS input checksum mismatch: {role}")
        paths[role] = path
    inventory = read_table(paths["inventory"])
    columns = ["file_accession", "experiment_accession", "rbp", "output_type", "href", "md5sum", "file_size", "status"]
    inventory = inventory[columns].drop_duplicates()
    if inventory.file_accession.duplicated().any(): raise ValueError("Conflicting RBNS native source metadata")
    inventory = inventory.set_index("file_accession", drop=False)
    rbps = read_table(paths["rbp_crosswalk"])[["source_rbp_label", "rbp_id"]].drop_duplicates()
    if rbps.isna().any().any() or rbps.source_rbp_label.duplicated().any(): raise ValueError("Ambiguous RBP identity map")
    rbps = rbps.set_index("source_rbp_label").rbp_id.to_dict()
    settings = {"min_motif_enrichment": 2., "zero_floor": 1e-6, "min_valid_windows": 50, "min_window_fraction": .9,
                "min_transcript_coverage": 1., "min_excess": .25, "min_mean_log2_enrichment": 0.}
    if set(manifest.get("scoring", {}))-set(settings): raise ValueError("Unknown RBNS scoring settings")
    settings.update(manifest.get("scoring", {}))
    if not manifest["profiles"]: raise ValueError("No selected RBNS profiles")
    selected_keys = [(p["file_accession"], p["concentration"]) for p in manifest["profiles"]]
    if len(set(selected_keys)) != len(selected_keys): raise ValueError("Duplicate RBNS concentration profile")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host": "149", "training_started": False,
        "manifest_sha256": manifest_sha, "annotation_id": manifest["annotation_id"]}, indent=2))
    profiles = []; sources = {}
    for spec in manifest["profiles"]:
        row = inventory.loc[spec["file_accession"]]
        if row.output_type != "enrichment" or rbps.get(row.rbp) != spec["rbp_id"]:
            raise ValueError("Selected RBNS profile does not match native RBP enrichment source")
        source = verified_encode_file(row.to_dict(), args.cache, download=args.download, proxy=args.proxy)
        digest = artifact_sha256(source)
        profile = parse_rbns_profile(pd.read_csv(source, sep="\t"), **spec,
            experiment_accession=row.experiment_accession, source_rbp_label=row.rbp,
            min_motif_enrichment=settings["min_motif_enrichment"], zero_floor=settings["zero_floor"])
        profiles.append(profile)
        sources[row.file_accession] = {"path": str(source.resolve()), "sha256": digest, "md5": row.md5sum}
    crosswalk = read_table(paths["transcript_crosswalk"])
    lncrnas = read_table(paths["lncrnas"])
    if not set(crosswalk.lncrna_id).issubset(set(lncrnas.lncrna_id)):
        raise ValueError("Transcript crosswalk contains lncRNAs outside the registered analysis universe")
    outputs = {}; output_paths = []; pending = []; total_rows = 0
    def save(table, name):
        path = args.output/name; table.to_parquet(path, index=False, compression="zstd")
        outputs[name] = {"path": str(path.resolve()), "sha256": artifact_sha256(path), "rows": len(table)}
        return path
    def flush():
        nonlocal pending
        output_paths.append(str(save(pd.DataFrame(pending), f"transcript_scores_{len(output_paths):06d}.parquet")))
        pending = []
    for row in iter_transcript_scores(paths["transcript_fasta"], crosswalk, profiles, annotation_id=manifest["annotation_id"],
            min_valid_windows=settings["min_valid_windows"], min_window_fraction=settings["min_window_fraction"]):
        pending.append(row); total_rows += 1
        if len(pending) >= args.chunk_rows: flush()
    if pending: flush()
    summary_columns = ["lncrna_id", "profile_id", "observed_transcripts", "scorable_transcripts", "mean_log2_enrichment",
                       "mean_composition_excess", "min_isoform_excess", "max_isoform_excess", "mean_enriched_window_fraction"]
    if output_paths:
        with duckdb.connect() as db:
            db.execute("SET threads=2")
            summary = db.execute("""SELECT lncrna_id, profile_id, count(*) AS observed_transcripts,
                sum(CASE WHEN available THEN 1 ELSE 0 END) AS scorable_transcripts,
                avg(mean_log2_enrichment) AS mean_log2_enrichment,
                avg(composition_excess_log2_enrichment) AS mean_composition_excess,
                min(composition_excess_log2_enrichment) AS min_isoform_excess,
                max(composition_excess_log2_enrichment) AS max_isoform_excess,
                avg(enriched_window_fraction) AS mean_enriched_window_fraction
                FROM read_parquet(?, hive_partitioning=false) GROUP BY lncrna_id, profile_id""", [output_paths]).fetchdf()
    else:
        summary = pd.DataFrame(columns=summary_columns)
    genes = aggregate_gene_scores(summary, crosswalk, lncrnas, profiles, min_transcript_coverage=settings["min_transcript_coverage"])
    relation, binding, binding_audit = predicted_binding_inputs(genes, read_table(paths["protein_gene_map"]), lncrnas,
        min_excess=settings["min_excess"], min_mean_log2_enrichment=settings["min_mean_log2_enrichment"])
    save(genes, "GENE_SEQUENCE_PREFERENCE.parquet")
    save(relation, "PREDICTED_INTERACTION_RELATION.parquet")
    save(binding, "RBNS_TYPED_BINDING.parquet")
    for role, path in paths.items():
        if artifact_sha256(path) != manifest["artifacts"][role]["sha256"]: raise ValueError(f"RBNS input changed during scoring: {role}")
    for record in sources.values():
        if artifact_sha256(record["path"]) != record["sha256"]: raise ValueError("RBNS source changed during scoring")
    if artifact_sha256(args.manifest) != manifest_sha: raise ValueError("RBNS manifest changed during scoring")
    for path, digest in code_hashes.items():
        if artifact_sha256(path) != digest: raise ValueError("RBNS implementation changed during execution")
    run_id = "RBNS-SEQ-"+hashlib.sha256(json.dumps([manifest_sha, code_hashes], sort_keys=True).encode()).hexdigest()[:20]
    audit = {"format": "RBNS_SEQUENCE_ANALYSIS_AUDIT_V1", "status": "PASS", "target_host": "149", "training_started": False,
        "run_id": run_id, "implementation_sha256": code_hashes, "core_dependency": "CORE_INDEPENDENT_EVIDENCE",
        "estimand": "TRANSCRIPT_SEQUENCE_RBNS_KMER_PREFERENCE_PREDICTION",
        "manifest_sha256": manifest_sha, "annotation_id": manifest["annotation_id"], "source_files": sources,
        "outputs": outputs, "scoring": settings, "binding_materialization": binding_audit, "transcript_profile_rows": total_rows,
        "available_gene_profiles": int(genes.available.sum()), "unavailable_gene_profiles": int((~genes.available).sum()),
        "isoform_aggregation": "EQUAL_WEIGHT_REGISTERED_ISOFORMS_NO_EXPRESSION_WEIGHTING",
        "composition_comparator": "INDEPENDENT_BASE_FREQUENCIES_NOT_DINUCLEOTIDE_OR_STRUCTURE_MODEL",
        "in_vitro_rbp_preference_measured": True, "lncrna_binding_measured": False,
        "score_is_probability": False, "pvalues_computed": False, "main_training_graph_modified": False}
    (args.output/"AUDIT.json").write_text(json.dumps(audit, indent=2)+"\n")
    (args.output/"SUCCESS.json").write_text(json.dumps({"status": "PASS", "audit_sha256": artifact_sha256(args.output/"AUDIT.json")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
