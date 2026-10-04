#!/usr/bin/env python3
"""Export V3.2 core node embeddings for the auxiliary private heads (GPU required).

Why this is a new script rather than a flag on `scripts/export_v32_core_embeddings.py`:

  1. that exporter pins `CHECKPOINT_FORMAT = "CC_HHGT_V3_2_FULL_TRAINING_STATE_V1"`,
     while the corrected core emits `..._V4_GROUP_SCHEDULE` (training.py:24);
  2. it encodes one graph, but the V4 core is defined over a runtime-chunk estimand
     (26-29 chunks per fold), so a node has no single chunk-independent embedding;
  3. this run's graph is the C global-binding overlay, which that exporter never loads.

Embedding estimand (decision, recorded in the emitted manifest):

    core_embedding(node) = sum_c w_c * encoder(chunk_c)[node]

with `w_c` taken unchanged from `cc_hhgt.v32.training._runtime_chunk_contract` -- the same
registered weights the formal validation/test logit aggregation uses.  This is the
encoder-space analogue of the registered estimator.  It is NOT claimed to equal
`encoder(union_graph)`; the union-graph encode is audited separately and reported as a
diagnostic only.

Runs OUTSIDE the hashed training tree (`fold_N/code/live` is imported, never modified), so
`code_tree_sha256` stays `bad3135ba858bf36fedd4a102eb509ddb110872844d5bf1def90a93274c0ba58`.

Usage: python3 export_core_embeddings_instance.py FOLD [FOLD ...]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path("/root/CancerLncAtlas_C_GLOBAL_G2_20260928")
OUT_ROOT = ROOT / "core_embeddings"
T0 = time.time()

CORE_EXPORT_FORMAT = "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1"
CHECKPOINT_FORMAT = "CC_HHGT_V3_2_FULL_TRAINING_STATE_V4_GROUP_SCHEDULE"
TASK_TEMPLATE = "v32-g012-g2-c-global-binding-20260928-r1__PATIENT_FOLD_{fold}__CC-HHGT__20260726"
ESTIMAND = "REGISTERED_CHUNK_WEIGHTED_MEAN_OF_ENCODER_OUTPUTS"


def stamp(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:7.1f}s] {msg}", flush=True)


def sha256_file(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def node_frame(node_type, values, node_map):
    import numpy as np
    import pandas as pd

    array = values.detach().cpu().numpy().astype(np.float32)
    if array.ndim != 2:
        raise RuntimeError(f"{node_type} embedding is not a matrix")
    ordered = sorted(((int(index), str(node_id)) for node_id, index in node_map.items()))
    indices = np.asarray([index for index, _ in ordered], dtype=np.int64)
    if len(indices) != array.shape[0] or not np.array_equal(indices, np.arange(len(indices))):
        raise RuntimeError(f"{node_type} node map is not a complete zero-based alignment")
    frame = pd.DataFrame(
        array, columns=[f"core_feature_{i:03d}" for i in range(array.shape[1])]
    )
    frame.insert(0, "node_id", [node_id for _, node_id in ordered])
    frame.insert(0, "node_index", indices)
    return frame


def usability(frame) -> dict:
    import numpy as np

    values = frame.drop(columns=["node_index", "node_id"]).to_numpy(dtype=np.float32)
    unique = int(len(np.unique(values, axis=0)))
    varying = int(np.count_nonzero(np.ptp(values, axis=0) > 1.0e-12))
    usable = bool(values.shape[0] == 1 or (unique > 1 and varying > 0))
    return {
        "rows": int(values.shape[0]),
        "features": int(values.shape[1]),
        "unique_embedding_vectors": unique,
        "varying_feature_count": varying,
        "usable_for_node_discrimination": usable,
        "status": "USABLE" if usable else "CONSTANT_EMBEDDING_MASKED",
    }


def main() -> int:
    folds = [int(value) for value in sys.argv[1:]]
    if not folds:
        raise SystemExit("usage: export_core_embeddings_instance.py FOLD [FOLD ...]")

    import torch
    import yaml

    LIVE = ROOT / "fold_0" / "code" / "live"
    sys.path.insert(0, str(LIVE))

    from cc_hhgt.gnn import build_model, move_graph
    from cc_hhgt.v32.c_graph_overlay import load_c_graph_overlay
    from cc_hhgt.v32.integrated_model import freeze_v32_core
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual
    from cc_hhgt.v32.training import _runtime_chunk_contract, _runtime_graph_for_chunk

    import os

    forced = os.environ.get("CORE_EXPORT_DEVICE", "").strip().lower()
    if forced:
        device = forced
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("V3.2 core embedding export requires CUDA")
        device = "cuda"
    max_chunks = int(os.environ.get("CORE_EXPORT_MAX_CHUNKS", "0") or 0)
    if device != "cuda":
        stamp(f"WARNING: running core export on device={device!r} (validation run only)")

    # NOTE: this must be a LOCAL name.  Assigning to the module-level OUT_ROOT inside main()
    # makes Python treat OUT_ROOT as function-local for the whole body, so on a real run
    # (max_chunks == 0) the assignment below never executes and the following line raised
    # UnboundLocalError.  The chunk-limited smoke paths happened to take the assignment branch
    # and so never exposed it.
    out_root = ROOT / "core_embeddings_smoke" if max_chunks else OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "CORE_EMBEDDING_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("export_format") != CORE_EXPORT_FORMAT:
            raise RuntimeError("existing core embedding manifest format mismatch")
    else:
        manifest = {
            "export_format": CORE_EXPORT_FORMAT,
            "analysis_version": "CancerLncAtlas_V3.2_C_GLOBAL_BINDING_G2",
            "module_id": "exact_pathway",
            "training_generation": "V3.2",
            "all_embeddings_from_newly_trained_v32_core": True,
            "historical_checkpoint_loaded": False,
            "historical_prediction_loaded": False,
            "embedding_estimand": ESTIMAND,
            "embedding_estimand_note": (
                "sum_c w_c * encoder(chunk_c), w_c from _runtime_chunk_contract; the "
                "encoder-space analogue of the registered logit aggregation. Not claimed "
                "to equal encoder(union_graph)."
            ),
            "graph_variant": "G2",
            "rbp_evidence_mode": "C_GLOBAL_PHYSICAL_BINDING",
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "folds": {},
        }

    for fold in folds:
        fold_dir = ROOT / f"fold_{fold}"
        cfg = yaml.safe_load((fold_dir / "auth" / "config.yaml").read_text(encoding="utf-8"))
        primary = cfg.get("primary_model", {})
        manifest_json = fold_dir / "auth" / "INPUT_MANIFEST.json"
        inputs = json.loads(manifest_json.read_text(encoding="utf-8"))
        record = [r for r in inputs["fold_inputs"] if int(r["fold"]) == fold][0]
        ovl = record["graph_overlay"]
        prepared_path = Path(record["path"])
        overlay_path = Path(ovl["path"])

        stamp(f"fold {fold}: loading A payload ({prepared_path.stat().st_size} B)")
        payload = torch.load(str(prepared_path), map_location="cpu", weights_only=False)
        payload, _ = load_c_graph_overlay(
            torch,
            input_manifest_path=manifest_json,
            overlay_path=overlay_path,
            fold=fold,
            parent_sha256=str(record["sha256"]),
            old_payload=payload,
        )
        bundle = payload["bundle"]

        checkpoint_path = fold_dir / "results" / TASK_TEMPLATE.format(fold=fold) / "best_model_state.pt"
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT:
            raise RuntimeError(f"fold {fold} checkpoint format drift: {checkpoint.get('checkpoint_format')}")
        if checkpoint.get("graph_variant") != "G2":
            raise RuntimeError(f"fold {fold} checkpoint graph variant drift")

        encoder = build_model(
            "cc_hhgt", bundle, int(payload["feature_dim"]), dict(payload["legacy_model_config"])
        )
        model = build_v32_cc_hhgt_residual(
            encoder,
            hidden_channels=int(primary.get("hidden_channels", 96)),
            context_features=int(payload.get("conservation_context_features", 4)),
            dropout=float(primary.get("dropout", 0.20)),
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device)
        core_parameter_sha256 = freeze_v32_core(model)
        model.eval()

        chunks, weights = _runtime_chunk_contract(bundle)
        full_chunk_count = len(chunks)
        if max_chunks:
            chunks = chunks[:max_chunks]
            weights = {c: weights[c] for c in chunks}
        stamp(f"fold {fold}: {len(chunks)}/{full_chunk_count} chunks, device={device}, "
              f"checkpoint cycle={checkpoint.get('cycle')}")

        per_chunk: dict[str, list] = {}
        with torch.no_grad():
            for chunk in chunks:
                graph = _runtime_graph_for_chunk(bundle, chunk, device=device)
                encoded = model.encoder.encode(graph)
                for node_type, value in encoded.items():
                    per_chunk.setdefault(node_type, []).append(
                        value.detach().to(device="cpu", dtype=torch.float64)
                    )
                del graph, encoded
            union_audit = {}
            if max_chunks == 0:
                union_graph = move_graph(bundle, device, "cc_hhgt")
                union_encoded = model.encoder.encode(union_graph)
                for node_type, value in union_encoded.items():
                    frame = node_frame(node_type, value.detach().to(device="cpu", dtype=torch.float32),
                                       bundle.node_maps[node_type])
                    union_audit[node_type] = usability(frame)
                del union_graph, union_encoded

        fold_root = out_root / f"patient_fold={fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        exported, audits, spreads = {}, {}, {}
        for node_type in sorted(per_chunk):
            if node_type not in bundle.node_maps:
                raise RuntimeError(f"encoded node type lacks a node map: {node_type}")
            stack = per_chunk[node_type]
            mean = torch.zeros_like(stack[0])
            for chunk, value in zip(chunks, stack):
                mean += value * float(weights[chunk])
            deviation = max(float((value - mean).abs().max()) for value in stack)
            spreads[node_type] = deviation
            frame = node_frame(node_type, mean.to(torch.float32), bundle.node_maps[node_type])
            path = fold_root / f"{node_type}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            exported[node_type] = {
                "path": str(path.relative_to(out_root)),
                "sha256": sha256_file(path),
                "rows": int(len(frame)),
                "features": int(len(frame.columns) - 2),
            }
            audits[node_type] = usability(frame)
            stamp(f"fold {fold}: {node_type} rows={len(frame)} {audits[node_type]['status']} "
                  f"max_chunk_deviation={deviation:.4g}")

        fold_manifest = {
            "patient_fold": fold,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_format": checkpoint["checkpoint_format"],
            "checkpoint_cycle": int(checkpoint.get("cycle", -1)),
            "core_parameter_sha256": core_parameter_sha256,
            "prepared_path": str(prepared_path),
            "prepared_sha256": str(record["sha256"]),
            "overlay_path": str(overlay_path),
            "artifact_hashes": checkpoint.get("artifact_hashes"),
            "old_checkpoint_loaded": False,
            "trained_from_scratch": True,
            "runtime_chunks": len(chunks),
            "runtime_chunks_total": full_chunk_count,
            "partial_chunk_smoke": bool(max_chunks),
            "chunk_weights": {str(c): float(weights[c]) for c in chunks},
            "embedding_estimand": ESTIMAND,
            "exports": exported,
            "embedding_usability": audits,
            "max_abs_chunk_deviation_from_weighted_mean": spreads,
            "union_graph_encode_diagnostic": union_audit,
        }
        (fold_root / "LINEAGE.json").write_text(
            json.dumps(fold_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest["folds"][str(fold)] = fold_manifest
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        del per_chunk, payload, bundle, model, encoder, checkpoint
        torch.cuda.empty_cache()

    print(json.dumps({"status": "SUCCESS", "folds": folds, "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
