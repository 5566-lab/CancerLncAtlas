#!/usr/bin/env python3
"""Evaluate a trained V3.2 fold on its untouched test batches.

The prepared fold stores the same candidate rows for L1 and CC-HHGT.  This
script only loads a checkpoint and runs inference; it never performs an
optimizer step or changes the prepared data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _load_yaml(path: Path) -> dict:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Configuration must be a mapping")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import torch
    from sklearn.metrics import average_precision_score, roc_auc_score, log_loss

    config = _load_yaml(Path(args.config))
    payload = torch.load(Path(args.prepared), map_location="cpu", weights_only=False)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("prepared_format") != "CC_HHGT_V3_2_PREPARED_FOLD_V1":
        raise RuntimeError("Prepared fold format mismatch")
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise RuntimeError("Checkpoint does not contain model_state")
    if "test_batches" not in payload:
        raise RuntimeError("Prepared fold has no test_batches; regenerate formal preparation")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal evaluation requires the local CUDA device")

    from cc_hhgt.gnn import build_model, move_graph
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual, probability_from_logit

    device = "cuda"
    primary = config.get("primary_model", {})
    encoder = build_model("cc_hhgt", payload["bundle"], int(payload["feature_dim"]), payload["legacy_model_config"])
    model = build_v32_cc_hhgt_residual(
        encoder,
        hidden_channels=int(primary.get("hidden_channels", 96)),
        context_features=int(payload.get("conservation_context_features", 4)),
        dropout=float(primary.get("dropout", 0.20)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    graph = payload.get("graph")
    graph = move_graph(graph, device, "cc_hhgt") if graph is not None else move_graph(payload["bundle"], device, "cc_hhgt")

    base_logits: list[np.ndarray] = []
    final_logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    with torch.no_grad():
        encoded = model.encoder.encode(graph)
        for raw in payload["test_batches"]:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in raw.items()}
            candidate = {k: v.to(device) for k, v in batch["candidate_batch"].items()}
            output = model(
                graph,
                candidate,
                batch["base_logit"],
                batch["conservation_context"],
                batch.get("graph_available"),
                admitted=True,
                encoded=encoded,
            )
            base_logits.append(batch["base_logit"].detach().cpu().numpy())
            final_logits.append(output["final_logit"].detach().cpu().numpy())
            labels.append(batch["proxy_label"].detach().cpu().numpy())

    base = np.concatenate(base_logits)
    final = np.concatenate(final_logits)
    y = np.concatenate(labels).astype(int)
    base_prob = probability_from_logit(base)
    final_prob = probability_from_logit(final)

    def metrics(prob: np.ndarray) -> dict:
        out = {
            "auprc": float(average_precision_score(y, prob)),
            "log_loss": float(log_loss(y, np.clip(prob, 1e-7, 1 - 1e-7), labels=[0, 1])),
        }
        out["auroc"] = float(roc_auc_score(y, prob)) if np.unique(y).size == 2 else None
        return out

    result = {
        "evaluation": "held_out_test",
        "prepared": str(Path(args.prepared)),
        "checkpoint": str(Path(args.checkpoint)),
        "rows": int(y.size),
        "positive_rate": float(y.mean()) if y.size else None,
        "l1": metrics(base_prob),
        "cc_hhgt": metrics(final_prob),
        "delta_cc_hhgt_minus_l1": {
            key: (metrics(final_prob)[key] - metrics(base_prob)[key])
            for key in ("auprc", "auroc", "log_loss")
        },
        "checkpoint_cycle": int(checkpoint.get("cycle", -1)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
