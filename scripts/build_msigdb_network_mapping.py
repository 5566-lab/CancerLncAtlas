#!/usr/bin/env python3
"""Materialize a traceable bridge without changing the formal target catalog."""
import argparse
import csv
import datetime
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.msigdb_network import (build_mapping, read_tsv, sha256_file, write_mapping,
                                      read_msigdb_archive, audit_archive_membership)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-json", type=Path)
    parser.add_argument("--archive-url")
    parser.add_argument("--archive-version")
    parser.add_argument("--source-catalog", type=Path)
    parser.add_argument("--hgnc", type=Path)
    args = parser.parse_args()
    host = socket.gethostname()
    if host != "149":
        raise SystemExit("target_host must be 149; actual=" + host)
    root = args.root
    catalog = args.catalog or root / "canonical_targets_r7"
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {
        "registry": root / "catalog_inventory/PATHWAY_SOURCE_ID_REGISTRY_DRAFT.tsv",
        "metadata": root / "CROSSDATABASE_DEFINITION_SOURCES.json",
        "bindings": catalog / "SOURCE_ID_BINDINGS.tsv",
        "definitions": catalog / "DEFINITION_TO_IDENTITY.tsv",
        "redirects": catalog / "ID_REDIRECTS.tsv",
        "overrides": root / "sources/KEGG_CURATED_NETWORK_XREFS.json",
    }
    if args.archive_json:
        if not args.archive_url or not args.archive_version:
            parser.error("--archive-json requires --archive-url and --archive-version")
        paths["archive"] = args.archive_json
    if args.source_catalog:
        if not args.archive_json or not args.hgnc:
            parser.error("--source-catalog requires --archive-json and --hgnc")
        paths["source_catalog"] = args.source_catalog
        paths["hgnc"] = args.hgnc
    receipt = {
        "target_host": host,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "operation": "build_msigdb_network_mapping",
        "input_sha256": {key: {"path": str(path), "sha256": sha256_file(path)}
                         for key, path in paths.items()},
        "formal_catalog_changed": False, "training_started": False,
    }
    preflight = args.output / "MSIGDB_MAPPING_PREFLIGHT.json"
    preflight.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    overrides = json.loads(paths["overrides"].read_text())["records"]
    for row in overrides:
        row["metadata_sha256"] = receipt["input_sha256"]["overrides"]["sha256"]
    registry = read_tsv(paths["registry"])
    metadata = json.loads(paths["metadata"].read_text())["rows"]
    archive_rows = []
    if args.archive_json:
        archive_rows = read_msigdb_archive(
            args.archive_json, args.archive_url, args.archive_version)
    rows, summary = build_mapping(
        registry, metadata + archive_rows,
        read_tsv(paths["bindings"]), read_tsv(paths["definitions"]),
        read_tsv(paths["redirects"]), overrides, verify_cache=True,
    )
    if args.source_catalog:
        members = audit_archive_membership(
            registry, archive_rows,
            json.loads(args.source_catalog.read_text())["gene_sets"],
            read_tsv(args.hgnc), metadata)
        audit_path = args.output / "ARCHIVE_MEMBER_COMPARISON.tsv"
        with audit_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(members[0]))
            writer.writeheader()
            writer.writerows(members)
        receipt["membership_audit_path"] = str(audit_path)
        receipt["membership_audit_sha256"] = sha256_file(audit_path)
        receipt["original_members_preserved"] = True
    output = args.output / "MSIGDB_NETWORK_MAPPING.tsv"
    write_mapping(output, rows)
    receipt.update(summary)
    receipt["output_path"] = str(output)
    receipt["output_sha256"] = sha256_file(output)
    (args.output / "MSIGDB_MAPPING_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
