#!/usr/bin/env python3
"""Run one frozen V3.2 fold over the complete 3.3M-candidate universe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _yaml(path: Path) -> dict:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("Config must be a mapping")
    return value


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | None]:
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    labels = np.asarray(labels, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "auprc": float(average_precision_score(labels, probability)),
        "auroc": float(roc_auc_score(labels, probability)) if np.unique(labels).size == 2 else None,
        "log_loss": float(log_loss(labels, probability, labels=[0, 1])),
        "prevalence": float(labels.mean()),
        "rows": int(labels.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--companion", required=True)
    parser.add_argument("--static-root", default="artifacts/formal_prepared")
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics", required=True)
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(root))
    import torch
    from cc_hhgt.gnn import build_model, move_graph
    from cc_hhgt.v32.model import build_v32_cc_hhgt_residual

    if not torch.cuda.is_available():
        raise RuntimeError("Full-candidate inference requires CUDA")
    config = _yaml(Path(args.config))
    payload = torch.load(Path(args.prepared), map_location="cpu", weights_only=False)
    checkpoint = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False)
    if int(payload.get("patient_fold", -1)) != args.fold:
        raise RuntimeError("Prepared fold mismatch")
    if payload.get("artifact_hashes") != checkpoint.get("artifact_hashes"):
        raise RuntimeError("Prepared/checkpoint authorization hashes differ")

    static_root = root / args.static_root
    candidates = pd.read_parquet(static_root / "FORMAL_CANDIDATE_UNIVERSE.parquet")
    candidates = candidates.sort_values(["cancer_id", "lncrna_id", "pathway_id"], kind="stable").reset_index(drop=True)
    scope = pd.read_parquet(static_root / "FORMAL_LNCRNA_SCOPE.parquet", columns=["cancer_id", "lncrna_id", "shared_or_local_scope"])
    candidates = candidates.merge(scope, on=["cancer_id", "lncrna_id"], how="left", validate="many_to_one")
    companion = pd.read_parquet(args.companion).sort_values("candidate_row_index", kind="stable")
    expected_index = np.arange(len(candidates), dtype=np.int64)
    if not np.array_equal(companion.candidate_row_index.to_numpy(np.int64), expected_index):
        raise RuntimeError("Companion candidate row index mismatch")

    device = "cuda"
    primary = config["primary_model"]
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
    maps = payload["bundle"].node_maps

    final_probability: list[np.ndarray] = []
    graph_residual: list[np.ndarray] = []
    graph_gate: list[np.ndarray] = []
    direction_probability: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    offset = 0
    with torch.no_grad():
        encoded = model.encoder.encode(graph)
        for raw in payload["test_batches"]:
            size = int(len(raw["base_logit"]))
            sub = candidates.iloc[offset : offset + size]
            expected = {
                "l": sub.lncrna_id.astype(str).map(maps["lncRNA"]).to_numpy(np.int64),
                "p": sub.pathway_id.astype(str).map(maps["pathway"]).to_numpy(np.int64),
                "c": sub.cancer_id.astype(str).map(maps["cancer"]).to_numpy(np.int64),
            }
            for key, value in expected.items():
                observed = raw["candidate_batch"][key].detach().cpu().numpy()
                if not np.array_equal(value, observed):
                    raise RuntimeError(f"Candidate/node alignment failure fold={args.fold} offset={offset} key={key}")
            batch = {key: (value.to(device) if hasattr(value, "to") else value) for key, value in raw.items()}
            candidate_batch = {key: value.to(device) for key, value in batch["candidate_batch"].items()}
            result = model(
                graph,
                candidate_batch,
                batch["base_logit"],
                batch["conservation_context"],
                batch.get("graph_available"),
                admitted=True,
                encoded=encoded,
            )
            final_probability.append(torch.sigmoid(result["final_logit"]).cpu().numpy().astype("float32"))
            graph_residual.append(result["graph_residual"].cpu().numpy().astype("float32"))
            graph_gate.append(result["graph_gate"].cpu().numpy().astype("float32"))
            direction_probability.append(torch.sigmoid(result["direction_logit"]).cpu().numpy().astype("float32"))
            labels.append(batch["proxy_label"].cpu().numpy().astype("int8"))
            offset += size
    if offset != len(candidates):
        raise RuntimeError("Inference did not cover the complete candidate universe")

    final = np.concatenate(final_probability)
    direction = np.concatenate(direction_probability)
    y = np.concatenate(labels)
    out = candidates.copy()
    out["patient_fold_id"] = str(args.fold)
    out["association_membership_probability"] = final
    out["association_direction"] = np.where(direction >= 0.5, "positive", "negative")
    out["association_direction_probability"] = direction
    out["l1_probability"] = companion.l1_probability.to_numpy("float32")
    out["ridge_probability"] = companion.ridge_probability.to_numpy("float32")
    out["graph_residual"] = np.concatenate(graph_residual)
    out["graph_gate"] = np.concatenate(graph_gate)
    out["regulatory_evidence_confidence"] = companion.regulatory_evidence_confidence.to_numpy("float32")
    out["regulatory_evidence_available"] = companion.regulatory_evidence_available.to_numpy(bool)
    out["pathway_target_level"] = "exact_pathway"
    out["held_out_proxy_label"] = y
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False, compression="zstd", row_group_size=100000)

    summary = {
        "fold": args.fold,
        "rows": int(len(out)),
        "checkpoint_cycle": int(checkpoint.get("cycle", -1)),
        "artifact_hashes": checkpoint.get("artifact_hashes"),
        "candidate_alignment": "FULL_ROW_EXACT",
        "l1": _metrics(y, out.l1_probability.to_numpy()),
        "ridge": _metrics(y, out.ridge_probability.to_numpy()),
        "cc_hhgt": _metrics(y, final),
        "regulatory_evidence": "UNAVAILABLE_NEUTRAL_0P5_NOT_USED_FOR_RANKING",
    }
    summary["delta_cc_hhgt_minus_l1"] = {
        key: summary["cc_hhgt"][key] - summary["l1"][key]
        for key in ("auprc", "auroc", "log_loss")
    }
    summary["delta_cc_hhgt_minus_ridge"] = {
        key: summary["cc_hhgt"][key] - summary["ridge"][key]
        for key in ("auprc", "auroc", "log_loss")
    }
    Path(args.metrics).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
