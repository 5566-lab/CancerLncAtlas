#!/usr/bin/env python3
"""Export fold-specific node embeddings from the newly trained V3.2 core.

The exporter never reads historical checkpoints or predictions.  It accepts
only the V3.2 checkpoint/prepared formats and records content hashes for every
parent artifact so auxiliary heads can prove which current-generation core
they froze.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
CHECKPOINT_FORMAT = "CC_HHGT_V3_2_FULL_TRAINING_STATE_V1"
PREPARED_FORMAT = "CC_HHGT_V3_2_PREPARED_FOLD_V1"
EXPORT_FORMAT = "CC_HHGT_V3_2_FROZEN_CORE_EMBEDDINGS_V1"


def _file_sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_for_fold(root: Path, fold: int) -> Path:
    matches = sorted(root.glob(f"*PATIENT_FOLD_{fold}*/*best_model_state.pt"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one V3.2 core checkpoint for fold {fold}, observed {len(matches)}"
        )
    return matches[0]


def _node_frame(node_type: str, embedding, node_map: dict[str, int]) -> pd.DataFrame:
    values = embedding.detach().cpu().to(dtype=embedding.dtype).numpy().astype(np.float32)
    if values.ndim != 2:
        raise RuntimeError(f"{node_type} embedding is not a matrix")
    ordered = sorted(((int(index), str(node_id)) for node_id, index in node_map.items()))
    indices = np.asarray([index for index, _ in ordered], dtype=np.int64)
    if len(indices) != values.shape[0] or not np.array_equal(indices, np.arange(len(indices))):
        raise RuntimeError(f"{node_type} node map is not a complete zero-based alignment")
    frame = pd.DataFrame(
        values,
        columns=[f"core_feature_{index:03d}" for index in range(values.shape[1])],
    )
    frame.insert(0, "node_id", [node_id for _, node_id in ordered])
    frame.insert(0, "node_index", indices)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--prepared-root", default="artifacts/formal_release_prepared_1seed"
    )
    parser.add_argument(
        "--checkpoint-root", default="artifacts/formal_release_training_1seed"
    )
    parser.add_argument(
        "--output-root", default="artifacts/v32_full_multitask/core_embeddings"
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(5)))
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    import torch

    from cc_hhgt.gnn import build_model, move_graph
    from cc_hhgt.v32.integrated_model import freeze_v32_core
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual

    if not torch.cuda.is_available():
        raise RuntimeError("V3.2 core embedding export requires CUDA")
    folds = sorted(set(int(fold) for fold in args.folds))
    if not folds or any(fold not in range(5) for fold in folds):
        raise RuntimeError("folds must be a non-empty subset of 0..4")
    prepared_root = root / args.prepared_root
    checkpoint_root = root / args.checkpoint_root
    output_root = root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "CORE_EMBEDDING_MANIFEST.json"
    if manifest_path.is_file():
        release_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if release_manifest.get("export_format") != EXPORT_FORMAT:
            raise RuntimeError("Existing core embedding manifest format mismatch")
    else:
        release_manifest: dict[str, object] = {
            "export_format": EXPORT_FORMAT,
            "analysis_version": ANALYSIS_VERSION,
            "module_id": "exact_pathway",
            "training_generation": "V3.2",
            "all_embeddings_from_newly_trained_v32_core": True,
            "historical_checkpoint_loaded": False,
            "historical_prediction_loaded": False,
            "folds": {},
        }

    for fold in folds:
        prepared_path = prepared_root / f"PATIENT_FOLD_{fold}.pt"
        checkpoint_path = _checkpoint_for_fold(checkpoint_root, fold)
        if not prepared_path.is_file():
            raise RuntimeError(f"Missing prepared fold: {prepared_path}")
        prepared = torch.load(prepared_path, map_location="cpu", weights_only=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if prepared.get("prepared_format") != PREPARED_FORMAT:
            raise RuntimeError(f"Fold {fold} is not a V3.2 prepared artifact")
        if checkpoint.get("checkpoint_format") != CHECKPOINT_FORMAT:
            raise RuntimeError(f"Fold {fold} is not a V3.2 checkpoint")
        if int(prepared.get("patient_fold", -1)) != fold:
            raise RuntimeError(f"Prepared fold ID drift for fold {fold}")
        if prepared.get("artifact_hashes") != checkpoint.get("artifact_hashes"):
            raise RuntimeError(f"Prepared/checkpoint authorization hashes differ for fold {fold}")
        forbidden_checkpoint_keys = {
            "source_checkpoint",
            "source_checkpoint_sha256",
            "historical_checkpoint",
            "historical_model_state",
        }
        if forbidden_checkpoint_keys.intersection(checkpoint):
            raise RuntimeError(f"Fold {fold} checkpoint declares a parent/old checkpoint")

        bundle = prepared["bundle"]
        encoder = build_model(
            "cc_hhgt", bundle, int(prepared["feature_dim"]), prepared["legacy_model_config"]
        )
        model = build_v32_cc_hhgt_residual(
            encoder,
            hidden_channels=96,
            context_features=int(prepared.get("conservation_context_features", 4)),
            dropout=0.20,
        )
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to("cuda")
        core_parameter_sha256 = freeze_v32_core(model)
        graph = prepared.get("graph")
        graph = (
            move_graph(graph, "cuda", "cc_hhgt")
            if graph is not None
            else move_graph(bundle, "cuda", "cc_hhgt")
        )
        with torch.no_grad():
            encoded = model.encoder.encode(graph)

        fold_root = output_root / f"patient_fold={fold}"
        fold_root.mkdir(parents=True, exist_ok=True)
        exported: dict[str, object] = {}
        for node_type in sorted(encoded):
            if node_type not in bundle.node_maps:
                raise RuntimeError(f"Encoded node type lacks a node map: {node_type}")
            frame = _node_frame(node_type, encoded[node_type], bundle.node_maps[node_type])
            path = fold_root / f"{node_type}.parquet"
            frame.to_parquet(path, index=False, compression="zstd")
            exported[node_type] = {
                "path": path.relative_to(root).as_posix(),
                "sha256": _file_sha256(path),
                "rows": int(len(frame)),
                "features": int(len(frame.columns) - 2),
            }

        fold_manifest = {
            "patient_fold": fold,
            "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "prepared_path": prepared_path.relative_to(root).as_posix(),
            "prepared_sha256": _file_sha256(prepared_path),
            "core_parameter_sha256": core_parameter_sha256,
            "checkpoint_format": checkpoint["checkpoint_format"],
            "checkpoint_cycle": int(checkpoint.get("cycle", -1)),
            "artifact_hashes": checkpoint.get("artifact_hashes"),
            "old_checkpoint_loaded": False,
            "trained_from_scratch": True,
            "exports": exported,
        }
        fold_lineage_path = fold_root / "LINEAGE.json"
        fold_lineage_path.write_text(
            json.dumps(fold_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        release_manifest["folds"][str(fold)] = fold_manifest
        del encoded, graph, model, encoder, checkpoint, prepared
        torch.cuda.empty_cache()

    manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "SUCCESS", "manifest": str(manifest_path), "folds": folds}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
