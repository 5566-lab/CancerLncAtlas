#!/usr/bin/env python3
"""Freeze the explicitly selected historical 381 definitions on server 149."""
import argparse
import datetime
import json
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.kegg381_backbone import (
    build_backbone, file_sha256, load_backbone, materialize_backbone, read_tsv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path, required=True)
    args = parser.parse_args()
    host = socket.gethostname()
    if host != "149":
        raise SystemExit("target_host must be 149; actual=" + host)
    args.output.mkdir(parents=True, exist_ok=True)
    R = args.root
    paths = {
        "selected_source_list": R / "project_log/trace_381_kegg_20261111_r1/KEGG_MEDICUS_381.txt",
        "historical_trace": R / "project_log/trace_381_kegg_20261111_r1/RESULT.json",
        "source_network_mapping": R / "network_cancer_scope_20261111_r1/archival_identity_r2/MSIGDB_NETWORK_MAPPING.tsv",
        "network_scope_annotations": R / "network_cancer_scope_20261111_r1/scope_r1/SOURCE_SCOPE_DECISIONS.tsv",
        "current_catalog": R / "canonical_targets_r7/TARGET_MASTER.tsv",
        "original_gene_catalog": args.source_catalog,
    }
    preflight = {
        "target_host": host,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "operation": "freeze_user_fixed_kegg381_backbone",
        "canonical_pointer_changed": False, "training_started": False,
    }
    (args.output / "LAUNCH_PREFLIGHT.json").write_text(json.dumps(preflight, indent=2) + "\n")
    trace = json.loads(paths["historical_trace"].read_text(encoding="utf-8"))
    paths["historical_coverage"] = Path(trace["source"])
    input_receipts = {key: {"path": str(path), "sha256": file_sha256(path)}
                      for key, path in paths.items()}
    if input_receipts["historical_coverage"]["sha256"] != trace["sha256"]:
        raise SystemExit("historical_coverage_checksum_changed")
    source_ids = paths["selected_source_list"].read_text(encoding="utf-8").splitlines()
    coverage = read_tsv(paths["historical_coverage"])
    usable = {row["pathway_id"] for row in coverage
              if row["pathway_id"].startswith("MSIGDB:KEGG_MEDICUS:")
              and row["usable"].lower() == "true"}
    if usable != set(source_ids):
        raise SystemExit("selected_381_not_equal_to_historical_usable_union")
    original = json.loads(args.source_catalog.read_text(encoding="utf-8"))["gene_sets"]
    rows, members, summary = build_backbone(
        source_ids, read_tsv(paths["source_network_mapping"]),
        read_tsv(paths["network_scope_annotations"]), original,
        read_tsv(paths["current_catalog"]), expected_count=381,
    )
    manifest = dict(preflight, **summary, input_sha256=input_receipts)
    manifest["historical_coverage_cancer_count"] = len({r["cancer_id"] for r in coverage})
    manifest["historical_coverage_rule"] = trace["rule"]
    manifest["historical_coverage_union_verified"] = True
    manifest["code_sha256"] = {
        rel: file_sha256(Path(__file__).resolve().parents[1] / rel)
        for rel in ("cla_engine/kegg381_backbone.py", "scripts/build_kegg381_backbone.py")}
    receipt = materialize_backbone(args.output, rows, members, manifest)
    verified = load_backbone(args.output)
    result = {"target_host": host, "output": str(args.output),
              "verified_count": len(verified["rows"]),
              "manifest_sha256": file_sha256(args.output / "MANIFEST.json"), **summary}
    (args.output / "BUILD_RECEIPT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
