"""Inspect recorded gene-matrix/prepared completion on 149 without restarting jobs.

This checks small JSON receipts and their cross-references. It does not load
matrices, PTs, models or patient rows, repeat scientific validation, or prove
that large artifacts have remained unchanged since the recorded validation.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import socket


CHANNELS = ["cnv_signed", "methylation_promoter_beta"]
SHA = re.compile(r"[0-9a-f]{64}")
MAX_JSON_BYTES = 2 * 1024 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def fold_rows(rows, role):
    require(isinstance(rows, list) and len(rows) == 5, role + ": five rows required")
    require(all(isinstance(r, dict) and integer(r.get("fold")) for r in rows), role + ": invalid fold")
    require({r["fold"] for r in rows} == set(range(5)), role + ": folds must be exactly 0..4")
    return {r["fold"]: r for r in rows}


def validate_receipts(product, binding, complete, manifest, config, receipts, evidence, *, expected_cancers):
    """Pure JSON bookkeeping; synthetic dictionaries may be checked anywhere.

    File hashes and the supplied product identity are checked by inspect_run.
    A successful result describes recorded validation, never a new replay.
    """
    require(integer(expected_cancers, 1), "expected_cancers must be positive")
    for role, row in (("product", product), ("binding", binding), ("complete", complete), ("manifest", manifest)):
        require(row.get("target_host") == "149", role + ": target host differs")
        require(row.get("training_started") is False, role + ": no-training declaration required")
    require(product.get("format") == "GENE_PATIENT_ASSAYS_VERIFIED_V1" and product.get("status") == "PASS",
            "matrix completion did not pass")
    require(product.get("node_type") == "gene" and product.get("gene_type") == "protein_coding"
            and product.get("genome_build") == "GRCh38" and product.get("matrix_orientation") == "patient_by_gene",
            "matrix gene/build/orientation contract differs")
    require(product.get("units") == dict(zip(CHANNELS, ["log2_copy_ratio", "beta"])), "matrix units differ")
    fold_pin = product.get("source_artifacts", {}).get("patient_folds", {})
    binding_fold_pin = binding.get("current_patient_fold_source", {})
    require(isinstance(fold_pin.get("sha256"), str) and SHA.fullmatch(fold_pin["sha256"])
            and binding_fold_pin.get("sha256") == fold_pin["sha256"], "binding patient-fold authority differs")
    require(all(integer(binding.get(k)) for k in ("cnv_files", "methylation_files")), "binding file counts invalid")
    expected_files = binding["cnv_files"] + binding["methylation_files"]
    require(expected_files > 0 and integer(product.get("raw_file_count")) and product["raw_file_count"] == expected_files,
            "raw file total differs from binding")
    parts = product.get("partitions")
    require(isinstance(parts, list) and len(parts) == expected_cancers, "cancer partition count differs")
    require(all(isinstance(p, dict) and isinstance(p.get("cancer_id"), str) and p["cancer_id"] for p in parts),
            "cancer identities missing")
    cancers = {p["cancer_id"] for p in parts}
    require(len(cancers) == expected_cancers, "duplicate cancer partition")
    require(all(set(p.get("channels", {})) == set(CHANNELS) for p in parts), "matrix channel missing or unexpected")
    require(complete.get("status") == "PASS" and complete.get("learned_embeddings_generated") is False,
            "formal completion status or embedding declaration differs")
    require(manifest.get("format") == "GENE_NODE_PREPARED_INPUT_MANIFEST_V1", "formal manifest format differs")
    require(config.get("task_contract", {}).get("node_omics_channels") == CHANNELS, "model channel order differs")
    checks = fold_rows(complete.get("folds"), "completion")
    inputs = fold_rows(manifest.get("fold_inputs"), "input manifest")
    require(set(receipts) == set(evidence) == set(range(5)), "per-fold receipts incomplete")
    gene_counts = set()
    summaries = []
    for fold, check in sorted(checks.items()):
        require(receipts[fold] == check, f"fold {fold}: receipt differs from completion")
        require(check.get("prepared_validator") == "PASS" and check.get("training_started") is False,
                f"fold {fold}: prepared validation did not pass")
        require(integer(check.get("node_input_dim")) and check["node_input_dim"] == 6,
                f"fold {fold}: input width must be six")
        require(integer(check.get("gene_nodes"), 1), f"fold {fold}: gene node count invalid")
        gene_counts.add(check["gene_nodes"])
        observed = check.get("observed_gene_nodes_by_channel", {})
        require(set(observed) == set(CHANNELS) and all(integer(n) and n <= check["gene_nodes"] for n in observed.values()),
                f"fold {fold}: channel coverage invalid")
        node = evidence[fold]
        require(node.get("producer_format") == "CANCERLNCATLAS_NODE_INPUT_PREPARATION_V1", f"fold {fold}: node producer differs")
        require(type(node.get("outer_fold")) is int and node["outer_fold"] == fold
                and node.get("scope") == "cohort" and node.get("channels") == CHANNELS,
                f"fold {fold}: evidence fold/scope/channels differ")
        require(node.get("cancer_id") == "PANCAN_TRAINING_POOL"
                and node.get("cohort_eligibility") == "TRAINING_TUMORS_ONLY",
                f"fold {fold}: pooled training-patient eligibility differs")
        require(node.get("training_started") is False, f"fold {fold}: node no-training declaration missing")
        node_cancers = node.get("cancer_ids")
        require(isinstance(node_cancers, list) and len(node_cancers) == len(cancers) and set(node_cancers) == cancers,
                f"fold {fold}: cancer scope differs")
        authority = check.get("input_authority_hashes", {})
        node_pin = inputs[fold].get("node_omics", {})
        require(isinstance(node_pin.get("sha256"), str) and SHA.fullmatch(node_pin["sha256"])
                and authority.get("node_omics_manifest_sha256") == node_pin["sha256"],
                f"fold {fold}: node manifest authority differs")
        table_sha = node.get("table_sha256")
        require(isinstance(table_sha, str) and SHA.fullmatch(table_sha)
                and authority.get("node_omics_table_sha256") == table_sha,
                f"fold {fold}: node table authority differs")
        require(integer(inputs[fold].get("bytes"), 1) and isinstance(inputs[fold].get("sha256"), str)
                and SHA.fullmatch(inputs[fold]["sha256"]), f"fold {fold}: prepared artifact pin invalid")
        summaries.append({"fold": fold, "gene_nodes": check["gene_nodes"], "node_input_dim": 6,
                          "observed_gene_nodes_by_channel": observed})
    require(len(gene_counts) == 1, "gene universe differs between folds")
    return {"status": "RECORDED_PASS", "raw_file_count": expected_files,
            "cancer_count": len(cancers), "folds": summaries}


class JsonReader:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        path = Path(path).resolve()
        with path.open("rb") as handle:
            raw = handle.read(MAX_JSON_BYTES + 1)
        require(len(raw) <= MAX_JSON_BYTES, "JSON receipt exceeds size limit: " + str(path))
        digest = hashlib.sha256(raw).hexdigest()
        require(path not in self.hashes or self.hashes[path] == digest, "Receipt changed during inspection")
        self.hashes[path] = digest
        value = json.loads(raw.decode("utf-8"))
        require(isinstance(value, dict), "Receipt must be a JSON object: " + str(path))
        return value

    def pinned(self, item, base):
        require(isinstance(item, dict) and isinstance(item.get("path"), str) and item["path"]
                and isinstance(item.get("sha256"), str) and SHA.fullmatch(item["sha256"]), "Malformed receipt pin")
        path = Path(item["path"])
        path = (path if path.is_absolute() else Path(base) / path).resolve()
        value = self.read(path)
        require(self.hashes[path] == item["sha256"], "Receipt SHA256 mismatch: " + str(path))
        return path, value

    def unchanged(self):
        for path in tuple(self.hashes):
            self.read(path)


def inspect_run(product_path, folds_dir, binding_path, *, expected_cancers):
    require(socket.gethostname() == "149", "Real receipt inspection requires host 149")
    product_path, folds_dir, binding_path = (Path(x).resolve() for x in (product_path, folds_dir, binding_path))
    base = {"format": "GENE_NODE_RECORDED_COMPLETION_INSPECTION_V1", "target_host": "149",
            "observed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "verification_scope": "SMALL_JSON_RECEIPTS_ONLY", "scientific_validation_repeated": False,
            "large_artifact_bytes_reverified": False, "training_started": False, "jobs_restarted": False}
    complete_path = folds_dir / "COMPLETE.json"
    missing = [str(p) for p in (product_path, complete_path, binding_path) if not p.is_file()]
    if missing:
        return {**base, "status": "INCOMPLETE", "missing_receipts": missing,
                "process_state": "NOT_INFERRED_FROM_PARTIAL_FILES"}
    reader = JsonReader()
    try:
        product, binding, complete = (reader.read(p) for p in (product_path, binding_path, complete_path))
        manifest_path, manifest = reader.pinned(complete["input_manifest"], folds_dir)
        require(manifest_path == folds_dir / "INPUT_MANIFEST.json", "Final input manifest required; pending manifest is not accepted")
        _, config = reader.pinned(complete["model_config_patch"], folds_dir)
        bound_product, _ = reader.pinned(manifest["gene_product"], manifest_path.parent)
        require(bound_product == product_path, "Formal folds refer to a different matrix product")
        receipts, evidence = {}, {}
        for fold, row in fold_rows(manifest.get("fold_inputs"), "input manifest").items():
            receipts[fold] = reader.read(folds_dir / f"FOLD_{fold}_PREPARED_VALIDATION.json")
            _, evidence[fold] = reader.pinned(row["node_omics"], manifest_path.parent)
        result = validate_receipts(product, binding, complete, manifest, config, receipts, evidence,
                                   expected_cancers=expected_cancers)
        reader.unchanged()
        return {**base, **result, "receipt_hashes": [
            {"path": str(p), "sha256": h} for p, h in sorted(reader.hashes.items())]}
    except (ValueError, KeyError, TypeError, AttributeError, OSError) as error:
        return {**base, "status": "INVALID_RECEIPTS", "error": str(error)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-product", type=Path, required=True, help="Matrix COMPLETE.json")
    parser.add_argument("--node-folds", type=Path, required=True, help="Directory with five prepared validation receipts")
    parser.add_argument("--source-binding", type=Path, required=True, help="BINDING_REPORT.json")
    parser.add_argument("--expected-cancers", type=int, required=True)
    args = parser.parse_args(argv)
    result = inspect_run(args.gene_product, args.node_folds, args.source_binding, expected_cancers=args.expected_cancers)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return {"RECORDED_PASS": 0, "INCOMPLETE": 2}.get(result["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
