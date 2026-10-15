#!/usr/bin/env python3
"""Export all five selected V4 core folds on host 149; inference only, no training.

Input CURRENT_CORE_EXPORT_INPUT_V1 contains core_run_id, graph_variant and
fold_inputs. Each fold declares fold plus hash-bound config, input_manifest,
prepared, checkpoint, training_receipt and optional graph_overlay artifacts.
Paths may be relative to this manifest. Relocated sources retain original
training hashes. An optional global_binding_source locates a C-overlay source.
"""
from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

from cc_hhgt.v32.current_core_export import (EXPORT_FORMAT, CHECKPOINT_FORMAT, ESTIMAND,
    sha256, verified_path, stream_embeddings, node_frame, validate_checkpoint,
    validate_training_receipt, validate_current_export)


def _validated_payload(paths, record, fold):
    import torch
    from cc_hhgt.v32.c_graph_overlay import apply_c_graph_overlay
    from cc_hhgt.v32.training import _validate_prepared, _prepared_input_authority_hashes

    payload = torch.load(paths["prepared"], map_location="cpu", weights_only=False)
    authority = json.loads(paths["input_manifest"].read_text(encoding="utf-8"))
    matches = [r for r in authority["fold_inputs"] if r["fold"] == fold]
    if len(matches) != 1 or matches[0].get("sha256") != record["prepared"]["sha256"]:
        raise ValueError("Prepared artifact is not the training-authorized fold")
    authorized = matches[0]
    overlay_record = authorized.get("graph_overlay")
    if bool(overlay_record) != ("graph_overlay" in paths):
        raise ValueError("Export must use exactly the training-authorized graph overlay")
    if overlay_record:
        size_only = overlay_record.get("verification_mode") == "SIZE_AND_GRAPH_SEMANTICS_V1"
        if size_only:
            if paths["graph_overlay"].stat().st_size != overlay_record.get("bytes"):
                raise ValueError("Training-authorized overlay size mismatch")
        elif record["graph_overlay"]["sha256"] != overlay_record.get("sha256"):
            raise ValueError("Training-authorized overlay hash mismatch")
        overlay = torch.load(paths["graph_overlay"], map_location="cpu", weights_only=False)
        payload = apply_c_graph_overlay(payload, overlay, fold=fold,
            source_sha256=record["prepared"]["sha256"],
            overlay_sha256=None if size_only else record["graph_overlay"]["sha256"],
            overlay_bytes=paths["graph_overlay"].stat().st_size,
            receipt_sha256=overlay_record["graph_authority_receipt_sha256"],
            global_binding_source_path=paths.get("global_binding_source"))
    receipt = json.loads(paths["training_receipt"].read_text(encoding="utf-8"))
    _validate_prepared(payload, fold=fold, artifact_hashes=receipt["artifact_hashes"],
                       authorized_graph_overlay=bool(overlay_record))
    if _prepared_input_authority_hashes(payload) != receipt["input_authority_hashes"]:
        raise ValueError("Loaded graph/data authority differs from the selected training run")
    return payload


def _build_model(payload, config, checkpoint, device):
    from cc_hhgt.gnn import build_model
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual
    primary = config.get("primary_model", {})
    encoder = build_model("cc_hhgt", payload["bundle"], int(payload["feature_dim"]),
                          dict(payload["legacy_model_config"]))
    model = build_v32_cc_hhgt_residual(encoder,
        hidden_channels=int(primary.get("hidden_channels", 96)),
        context_features=int(payload.get("conservation_context_features", 4)),
        dropout=float(primary.get("dropout", .2)))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model.to(device)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-core-run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)
    if socket.gethostname() != "149":
        raise RuntimeError("Core inference must execute on target host 149")
    import torch
    import yaml
    from cc_hhgt.v32.training import _runtime_chunk_contract, _runtime_graph_for_chunk

    inputs = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (inputs.get("format") != "CURRENT_CORE_EXPORT_INPUT_V1"
            or inputs.get("core_run_id") != args.expected_core_run_id
            or inputs.get("graph_variant") not in {"G0", "G1", "G2"}):
        raise ValueError("Invalid selected core export inputs")
    records = inputs["fold_inputs"]
    if (len(records) != 5 or any(type(r.get("fold")) is not int for r in records)
            or {r["fold"] for r in records} != set(range(5))):
        raise ValueError("Exactly five unique integer patient folds are required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "LAUNCH_PREFLIGHT.json").write_text(json.dumps({"target_host": "149",
        "training_started": False, "stage": "FROZEN_CORE_INFERENCE", "device": args.device,
        "core_run_id": args.expected_core_run_id, "source_manifest_sha256": sha256(args.manifest)}, indent=2))
    manifest = {"export_format": EXPORT_FORMAT, "status": "INCOMPLETE",
        "analysis_version": "CancerLncAtlas_V3.2_CURRENT_CORE", "training_generation": "V3.2",
        "core_run_id": args.expected_core_run_id, "graph_variant": inputs["graph_variant"],
        "embedding_estimand": ESTIMAND, "all_registered_chunks_exported": False,
        "labels_used_for_export": False, "optimizer_started_during_export": False,
        "export_precision": "FP32_ENCODER_FLOAT64_WEIGHTED_SUM", "folds": {}}
    for record in sorted(records, key=lambda r: r["fold"]):
        fold = record["fold"]
        roles = ["config", "input_manifest", "prepared", "checkpoint", "training_receipt"]
        roles += [key for key in ["graph_overlay", "global_binding_source"] if key in record]
        paths = {key: verified_path(record[key], args.manifest.resolve().parent) for key in roles}
        receipt = json.loads(paths["training_receipt"].read_text(encoding="utf-8"))
        validate_training_receipt(receipt, run_id=args.expected_core_run_id, fold=fold,
                                  checkpoint_sha=record["checkpoint"]["sha256"], variant=inputs["graph_variant"])
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, receipt, fold=fold, prepared_sha=record["prepared"]["sha256"],
            config_sha=record["config"]["sha256"], input_manifest_sha=record["input_manifest"]["sha256"])
        payload = _validated_payload(paths, record, fold)
        model = _build_model(payload, yaml.safe_load(paths["config"].read_text()), checkpoint, args.device)
        chunks, weights = _runtime_chunk_contract(payload["bundle"])
        if ({str(c): weights[c] for c in chunks} != receipt.get("runtime_chunk_weights")
                or len(chunks) != receipt.get("runtime_num_chunks")):
            raise ValueError("Reconstructed chunk estimator differs from training")
        values, core_sha = stream_embeddings(model, payload["bundle"], chunks, weights,
                                             _runtime_graph_for_chunk, device=args.device)
        fold_root = args.output / f"patient_fold={fold}"
        fold_root.mkdir()
        receipt_copy = fold_root / "TRAINING_SUCCESS.json"
        receipt_copy.write_bytes(paths["training_receipt"].read_bytes())
        exports = {}; usability = {}
        for kind, value in values.items():
            if not kind.replace("_", "").isalnum():
                raise ValueError("Unsafe node-type file name")
            frame, usability[kind] = node_frame(value, payload["bundle"].node_maps[kind])
            path = fold_root / f"{kind}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            exports[kind] = {"path": str(path.relative_to(args.output)), "sha256": sha256(path),
                             "rows": len(frame), "features": len(frame.columns) - 2}
        item = {"patient_fold": fold, "checkpoint_format": CHECKPOINT_FORMAT,
            "checkpoint_sha256": record["checkpoint"]["sha256"], "core_parameter_sha256": core_sha,
            "checkpoint_cycle": checkpoint.get("cycle"), "core_unchanged_after_export": True,
            "runtime_chunks_total": len(chunks), "chunk_weights": {str(c): weights[c] for c in chunks},
            "training_receipt": {"path": str(receipt_copy.relative_to(args.output)), "sha256": sha256(receipt_copy)},
            "exports": exports, "embedding_usability": usability, "sources": {k: record[k] for k in roles}}
        manifest["folds"][str(fold)] = item
        (args.output / "PARTIAL_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps({"fold": fold, "status": "EXPORTED", "chunks": len(chunks)}), flush=True)
        del payload, model, checkpoint, values
    manifest.update(status="PASS", all_registered_chunks_exported=True)
    manifest_path = args.output / "CORE_EMBEDDING_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    validate_current_export(manifest_path, expected_run_id=args.expected_core_run_id)
    (args.output / "SUCCESS.json").write_text(json.dumps({"status": "PASS", "training_started": False,
        "manifest_sha256": sha256(manifest_path), "core_run_id": args.expected_core_run_id}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
