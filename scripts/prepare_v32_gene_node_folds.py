"""Attach measured gene matrices to five existing prepared folds, without fitting.

Publishes a new input manifest after the actual training loader and prepared
validator accept every fold. Historical PTs are read-only. No checkpoint is
upgraded, no optimizer constructed, and no model is trained.
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cc_hhgt.v32.formal_expression_sources import read, write, record
from cc_hhgt.v32.current_core_export import verified_path
from cc_hhgt.v32.gene_patient_omics import host, CHANNELS, OUTPUT
from cc_hhgt.v32 import omics_node_inputs as inputs


def prepare(product, prepared_root, output):
    host(); product, prepared_root, output = map(lambda p: Path(p).resolve(), (product, prepared_root, output))
    output.mkdir(parents=True, exist_ok=False)
    write(output / "LAUNCH_PREFLIGHT.json", {"target_host": "149", "training_started": False,
        "gene_product": record(product), "prepared_root": str(prepared_root), "operation": "GENE_NODE_FIVE_FOLD_PREPARATION"})
    receipt = read(product)
    if receipt.get("format") != OUTPUT or receipt.get("status") != "PASS":
        raise ValueError("Completed verified gene product required")
    original_fold_map = verified_path(receipt["source_artifacts"]["patient_folds"], product.parent)
    parent_fold_map = prepared_root / "SAMPLE_PATIENT_FOLD_MAP.tsv"
    if record(original_fold_map)["sha256"] != record(parent_fold_map)["sha256"]:
        raise ValueError("Gene assay and prepared patient-fold authorities differ")
    patients = pd.read_parquet(verified_path(receipt["patients"], product.parent))
    patch = {"task_contract": {"node_omics_channels": list(CHANNELS)}}
    write(output / "MODEL_CONFIG_PATCH.json", patch)
    rows = []
    for fold in range(5):
        directory = output / f"fold_{fold}"; directory.mkdir()
        authority = patients.copy()
        authority["sample_id"] = authority.patient_id  # Explicit logical patient rows, not new aliquots.
        authority["is_tumor"] = True  # Source materializer admits only GDC-declared tumor files.
        authority["partition"] = np.where(authority.patient_fold_id.eq(fold), "test",
            np.where(authority.patient_fold_id.eq((fold+1) % 5), "validation", "train"))
        authority.to_parquet(directory / "sample_authority.parquet", index=False)
        cfg = {"format": inputs.FORMAT, "target_host": "149", "outer_fold": fold, "scope": "cohort",
            "cancer_id": inputs.POOL, "sample_authority": record(directory / "sample_authority.parquet"),
            "assays": [{"channel": c, "unit": u, "layout": "verified_gene_patient", "measurements": record(product)} for c, u in CHANNELS.items()]}
        write(directory / "inputs.json", cfg)
        evidence = inputs.produce(directory / "inputs.json", directory / "node_omics")
        parent = prepared_root / f"PATIENT_FOLD_{fold}.pt"
        rows.append({"fold": fold, **record(parent), "bytes": parent.stat().st_size, "node_omics": record(evidence)})
        print(f"FOLD_{fold}_EVIDENCE_READY", flush=True)
    draft = output / "INPUT_MANIFEST.pending.json"
    manifest = {"format": "GENE_NODE_PREPARED_INPUT_MANIFEST_V1", "target_host": "149", "training_started": False,
                "gene_product": record(product), "fold_inputs": rows}
    write(draft, manifest)
    from cc_hhgt.v32.native_graph_input import load_training_graph_inputs
    from cc_hhgt.v32.training import _validate_prepared
    checks = []
    for row in rows:
        fold = row["fold"]
        print(f"FOLD_{fold}_LOADING_PREPARED", flush=True)
        parent = verified_path({"path": row["path"], "sha256": row["sha256"]}, prepared_root)
        payload = torch.load(parent, map_location="cpu", weights_only=False, mmap=True)
        loaded, overlay = load_training_graph_inputs(torch, payload=payload, config=patch,
            input_manifest_path=draft, fold=fold, root=prepared_root, prepared_sha256=row["sha256"])
        _validate_prepared(loaded, fold=fold, artifact_hashes=payload["artifact_hashes"], authorized_graph_overlay=overlay)
        block = loaded["bundle"].node_omics["gene"]
        check = {"fold": fold, "prepared_validator": "PASS", "gene_nodes": len(block),
            "node_input_dim": loaded["bundle"].node_input_dim,
            "observed_gene_nodes_by_channel": {c: int(block[:, i*2+1].sum()) for i, c in enumerate(CHANNELS)},
            "input_authority_hashes": loaded["input_authority_hashes"], "training_started": False}
        checks.append(check); write(output / f"FOLD_{fold}_PREPARED_VALIDATION.json", check)
        print(f"FOLD_{fold}_PREPARED_PASS", flush=True)
        del loaded, payload, block; gc.collect()
    write(output / "INPUT_MANIFEST.json", manifest)
    write(output / "COMPLETE.json", {"target_host": "149", "status": "PASS", "training_started": False,
        "input_manifest": record(output / "INPUT_MANIFEST.json"), "folds": checks,
        "model_config_patch": record(output / "MODEL_CONFIG_PATCH.json"), "learned_embeddings_generated": False})
    return output / "COMPLETE.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gene-product", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True, help="Existing G0/G1/G2 directory containing the five PTs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.gene_product, args.prepared_root, args.output))
