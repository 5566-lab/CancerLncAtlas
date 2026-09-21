"""Leakage-controlled LOCO training for the v2.6 static graph."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, RGCNConv


ROOT = Path(__file__).resolve().parents[2]
GRAPH_ROOT = ROOT / "results" / "strict_graph"
PAIR_ROOT = ROOT / "results" / "strict_pairs" / "all_cancers"
MODEL_ROOT = ROOT / "models" / "strict_global"
RESULT_ROOT = ROOT / "results" / "strict_global"
PRIMARY_CANCERS = [
    "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
    "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC", "LUAD", "LUSC",
    "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ", "SARC", "SKCM",
    "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
]
SAFE_PAIR_FEATURES = [
    "detection_rate", "median_logcpm", "expression_specificity",
    "n_cancers_detected",
]
FORBIDDEN_PAIR_FEATURES = {
    "bulk_fdr", "bulk_effect", "robustness_pass_count", "direction",
    "label_class", "proxy_label", "direction_label",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fold_row(test_cancer: str, seed: int) -> dict[str, Any]:
    if test_cancer not in PRIMARY_CANCERS:
        raise ValueError(f"Unknown primary LOCO cancer: {test_cancer}")
    index = PRIMARY_CANCERS.index(test_cancer)
    validation = PRIMARY_CANCERS[(index + 1) % len(PRIMARY_CANCERS)]
    train = [value for value in PRIMARY_CANCERS if value not in {test_cancer, validation}]
    return {
        "fold_id": f"LOCO_{test_cancer}",
        "test_cancer": test_cancer,
        "validation_cancer": validation,
        "train_cancers": train,
        "seed": int(seed),
    }


def read_cancer_pairs(cancer: str) -> pd.DataFrame:
    path = PAIR_ROOT / f"cancer_id={cancer}"
    files = list(path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Strict pair partition is missing: {path}")
    frame = pd.concat([pd.read_parquet(item) for item in files], ignore_index=True)
    if "cancer_id" not in frame:
        frame["cancer_id"] = cancer
    return frame


def stratified_sample(frame: pd.DataFrame, maximum: int, seed: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    positive = frame.loc[frame.label_class.ne("unlabeled")]
    unlabeled = frame.loc[frame.label_class.eq("unlabeled")]
    positive_limit = min(len(positive), max(maximum // 2, 1))
    positive = positive.sample(n=positive_limit, random_state=seed) if len(positive) > positive_limit else positive
    remaining = maximum - len(positive)
    unlabeled = unlabeled.sample(n=min(remaining, len(unlabeled)), random_state=seed + 1)
    return pd.concat([positive, unlabeled], ignore_index=True).sample(
        frac=1, random_state=seed + 2
    ).reset_index(drop=True)


def load_fold_pairs(
    fold: dict[str, Any],
    maximum_train: int = 100_000,
    maximum_eval: int = 50_000,
) -> dict[str, pd.DataFrame]:
    per_cancer = max(maximum_train // len(fold["train_cancers"]), 1000)
    train_parts = [
        stratified_sample(read_cancer_pairs(cancer), per_cancer, fold["seed"] + index)
        for index, cancer in enumerate(fold["train_cancers"])
    ]
    frames = {
        "train": stratified_sample(
            pd.concat(train_parts, ignore_index=True), maximum_train, fold["seed"]
        ),
        "validation": stratified_sample(
            read_cancer_pairs(fold["validation_cancer"]), maximum_eval, fold["seed"] + 101
        ),
        "test": stratified_sample(
            read_cancer_pairs(fold["test_cancer"]), maximum_eval, fold["seed"] + 202
        ),
    }
    context = pd.read_parquet(GRAPH_ROOT / "lncrna_cancer_context_raw.parquet")
    context["cancer_id"] = context.cancer_id.astype(str)
    context["lncrna_id"] = context.lncrna_id.astype(str)
    safe = ["cancer_id", "lncrna_id", *SAFE_PAIR_FEATURES, "availability"]
    context = context[safe].drop_duplicates(["cancer_id", "lncrna_id"])
    for split, frame in frames.items():
        frames[split] = frame.merge(context, on=["cancer_id", "lncrna_id"], how="left")
        frames[split][SAFE_PAIR_FEATURES] = frames[split][SAFE_PAIR_FEATURES].fillna(0.0)
        frames[split]["availability"] = frames[split].availability.fillna(False).astype(float)
    if FORBIDDEN_PAIR_FEATURES & set(SAFE_PAIR_FEATURES):
        raise RuntimeError("Strict pair feature allowlist contains a label-derived column")
    return frames


def standardized_cancer_context(train_cancers: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(GRAPH_ROOT / "cancer_context_raw.parquet")
    feature_columns = [column for column in frame if column != "cancer_id"]
    frame[feature_columns] = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    fit = frame.loc[frame.cancer_id.astype(str).isin(train_cancers), feature_columns]
    median = fit.median()
    mean = fit.fillna(median).mean()
    scale = fit.fillna(median).std(ddof=0).replace(0, 1).fillna(1)
    frame[feature_columns] = (frame[feature_columns].fillna(median) - mean) / scale
    frame[feature_columns] = frame[feature_columns].clip(-8, 8).fillna(0)
    return frame


@dataclass
class GraphBundle:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    node_maps: dict[str, dict[str, int]]
    hetero: HeteroData
    homogeneous: dict[str, Any]
    cancer_feature_dim: int


def build_graph(fold: dict[str, Any], edge_cap: int = 300_000) -> GraphBundle:
    nodes = pd.read_parquet(GRAPH_ROOT / "graph_node.parquet")
    edges = pd.read_parquet(GRAPH_ROOT / "graph_edge.parquet")
    if edges.is_context_specific.fillna(False).any():
        raise RuntimeError("Strict graph contains context-specific edges")
    sampled = []
    rng = np.random.default_rng(fold["seed"])
    for _, group in edges.groupby(
        ["source_type", "relation_type", "target_type"], sort=True, observed=True
    ):
        if len(group) > edge_cap:
            weight = (
                pd.to_numeric(group.weight, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(1.0)
                .clip(lower=1e-6)
                .to_numpy(float)
            )
            weight = weight / weight.sum()
            index = rng.choice(len(group), edge_cap, replace=False, p=weight)
            group = group.iloc[np.sort(index)]
        sampled.append(group)
    edges = pd.concat(sampled, ignore_index=True)

    cancer_context = standardized_cancer_context(fold["train_cancers"]).set_index("cancer_id")
    cancer_columns = cancer_context.columns.tolist()
    node_maps: dict[str, dict[str, int]] = {}
    hetero = HeteroData()
    ordered_groups: dict[str, pd.DataFrame] = {}
    for node_type, group in nodes.groupby("node_type", sort=False):
        group = group.sort_values("node_index_within_type").reset_index(drop=True)
        ordered_groups[str(node_type)] = group
        identifiers = group.canonical_id.astype(str).tolist()
        node_maps[str(node_type)] = {value: index for index, value in enumerate(identifiers)}
        source = edges.loc[edges.source_type.astype(str).eq(str(node_type))]
        target = edges.loc[edges.target_type.astype(str).eq(str(node_type))]
        out_degree = source.groupby(source.source_canonical_id.astype(str)).size()
        in_degree = target.groupby(target.target_canonical_id.astype(str)).size()
        degree = (
            group.canonical_id.astype(str).map(out_degree).fillna(0).to_numpy(float)
            + group.canonical_id.astype(str).map(in_degree).fillna(0).to_numpy(float)
        )
        features = np.column_stack([np.log1p(degree), np.ones(len(group))])
        if str(node_type) == "cancer":
            context = cancer_context.reindex(identifiers)[cancer_columns].fillna(0).to_numpy(float)
            features = np.column_stack([features, context])
        hetero[str(node_type)].x = torch.tensor(features, dtype=torch.float32)
        hetero[str(node_type)].canonical_ids = identifiers

    global_index = {
        node_id: index for index, node_id in enumerate(nodes.node_id.astype(str))
    }
    relation_id = 0
    homogeneous_parts: list[tuple[int, np.ndarray, np.ndarray]] = []
    for (source_type, relation, target_type), group in edges.groupby(
        ["source_type", "relation_type", "target_type"], sort=True, observed=True
    ):
        source_map = node_maps[str(source_type)]
        target_map = node_maps[str(target_type)]
        source_index = group.source_canonical_id.astype(str).map(source_map)
        target_index = group.target_canonical_id.astype(str).map(target_map)
        valid = source_index.notna() & target_index.notna()
        source_array = source_index.loc[valid].to_numpy(np.int64)
        target_array = target_index.loc[valid].to_numpy(np.int64)
        index = torch.tensor(np.vstack([source_array, target_array]), dtype=torch.long)
        forward = (str(source_type), str(relation), str(target_type))
        reverse = (str(target_type), f"rev_{relation}", str(source_type))
        hetero[forward].edge_index = index
        hetero[reverse].edge_index = index.flip(0)
        global_source = group.loc[valid, "source_node_id"].astype(str).map(global_index).to_numpy(np.int64)
        global_target = group.loc[valid, "target_node_id"].astype(str).map(global_index).to_numpy(np.int64)
        homogeneous_parts.append((relation_id, global_source, global_target))
        relation_id += 1
        homogeneous_parts.append((relation_id, global_target, global_source))
        relation_id += 1

    h_source = np.concatenate([part[1] for part in homogeneous_parts])
    h_target = np.concatenate([part[2] for part in homogeneous_parts])
    h_relation = np.concatenate([
        np.full(len(part[1]), part[0], dtype=np.int64) for part in homogeneous_parts
    ])
    cancer_rows = nodes.index[nodes.node_type.astype(str).eq("cancer")].to_numpy(np.int64)
    cancer_by_global = nodes.loc[cancer_rows, "canonical_id"].astype(str).tolist()
    cancer_features = cancer_context.reindex(cancer_by_global)[cancer_columns].fillna(0).to_numpy(np.float32)
    homogeneous = {
        "edge_index": torch.tensor(np.vstack([h_source, h_target]), dtype=torch.long),
        "edge_type": torch.tensor(h_relation, dtype=torch.long),
        "num_nodes": len(nodes),
        "num_relations": relation_id,
        "global_map": {
            (row.node_type, str(row.canonical_id)): index
            for index, row in enumerate(nodes.itertuples(index=False))
        },
        "cancer_rows": torch.tensor(cancer_rows, dtype=torch.long),
        "cancer_features": torch.tensor(cancer_features, dtype=torch.float32),
    }
    return GraphBundle(
        nodes=nodes, edges=edges, node_maps=node_maps, hetero=hetero,
        homogeneous=homogeneous, cancer_feature_dim=len(cancer_columns),
    )


def candidate_tensors(
    frame: pd.DataFrame, bundle: GraphBundle, kind: str, device: torch.device
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    valid = (
        frame.lncrna_id.astype(str).isin(bundle.node_maps["lncRNA"])
        & frame.pathway_family_id.astype(str).isin(bundle.node_maps["pathway_family"])
        & frame.cancer_id.astype(str).isin(bundle.node_maps["cancer"])
    )
    frame = frame.loc[valid].reset_index(drop=True)
    if kind == "rgcn":
        mapping = bundle.homogeneous["global_map"]
        lnc = [mapping[("lncRNA", str(value))] for value in frame.lncrna_id]
        pathway = [mapping[("pathway_family", str(value))] for value in frame.pathway_family_id]
        cancer = [mapping[("cancer", str(value))] for value in frame.cancer_id]
    else:
        lnc = [bundle.node_maps["lncRNA"][str(value)] for value in frame.lncrna_id]
        pathway = [
            bundle.node_maps["pathway_family"][str(value)]
            for value in frame.pathway_family_id
        ]
        cancer = [bundle.node_maps["cancer"][str(value)] for value in frame.cancer_id]
    return frame, {
        "l": torch.tensor(lnc, dtype=torch.long, device=device),
        "p": torch.tensor(pathway, dtype=torch.long, device=device),
        "c": torch.tensor(cancer, dtype=torch.long, device=device),
        "x": torch.tensor(
            frame[SAFE_PAIR_FEATURES].to_numpy(np.float32),
            dtype=torch.float32, device=device,
        ),
        "availability": torch.tensor(
            frame[["availability"]].to_numpy(np.float32),
            dtype=torch.float32, device=device,
        ),
        "y": torch.tensor(frame.proxy_label.to_numpy(np.float32), device=device),
        "weak": torch.tensor(
            frame.label_class.eq("weak_positive").to_numpy(),
            dtype=torch.bool, device=device,
        ),
        "direction": torch.tensor(
            frame.direction_label.fillna(-1).to_numpy(np.float32), device=device
        ),
    }


class Decoder(nn.Module):
    def __init__(self, hidden: int, dropout: float, use_safe_pair: bool):
        super().__init__()
        self.use_safe_pair = use_safe_pair
        self.membership = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.direction = nn.Sequential(
            nn.Linear(hidden * 3, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        if use_safe_pair:
            self.safe_pair = nn.Sequential(
                nn.Linear(len(SAFE_PAIR_FEATURES), hidden),
                nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
            )
            self.safe_membership = nn.Linear(hidden, 1)
            self.safe_direction = nn.Linear(hidden, 1)
            self.gate = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 1), nn.Sigmoid())

    def forward(
        self, lnc: torch.Tensor, pathway: torch.Tensor, cancer: torch.Tensor,
        pair: torch.Tensor, availability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        graph = torch.cat([lnc, pathway, cancer], dim=-1)
        membership = self.membership(graph).squeeze(-1)
        direction = self.direction(graph).squeeze(-1)
        if self.use_safe_pair:
            safe = self.safe_pair(pair)
            gate = self.gate(availability).squeeze(-1) * availability.gt(0).any(dim=1)
            membership = membership + gate * self.safe_membership(safe).squeeze(-1)
            direction = direction + gate * self.safe_direction(safe).squeeze(-1)
        return membership, direction


class RGCNStrict(nn.Module):
    def __init__(self, bundle: GraphBundle, hidden: int, layers: int, dropout: float):
        super().__init__()
        graph = bundle.homogeneous
        self.embedding = nn.Embedding(graph["num_nodes"], hidden)
        self.cancer_projection = nn.Linear(bundle.cancer_feature_dim, hidden)
        self.layers = nn.ModuleList([
            RGCNConv(
                hidden, hidden, graph["num_relations"],
                num_bases=min(32, graph["num_relations"]),
            )
            for _ in range(layers)
        ])
        self.dropout = dropout
        self.decoder = Decoder(hidden, dropout, False)

    def encode(self, graph: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = self.embedding.weight.clone()
        projected_cancer = F.relu(
            self.cancer_projection(graph["cancer_features"])
        ).to(dtype=hidden.dtype)
        hidden[graph["cancer_rows"]] = projected_cancer
        for convolution in self.layers:
            hidden = F.relu(convolution(hidden, graph["edge_index"], graph["edge_type"]))
            hidden = F.dropout(hidden, self.dropout, self.training)
        return hidden


class HGTStrict(nn.Module):
    def __init__(
        self, bundle: GraphBundle, hidden: int, layers: int, heads: int,
        dropout: float, use_safe_pair: bool,
    ):
        super().__init__()
        self.projection = nn.ModuleDict({
            node_type: nn.Linear(bundle.hetero[node_type].x.shape[1], hidden)
            for node_type in bundle.hetero.node_types
        })
        self.layers = nn.ModuleList([
            HGTConv(hidden, hidden, bundle.hetero.metadata(), heads=heads)
            for _ in range(layers)
        ])
        self.dropout = dropout
        self.decoder = Decoder(hidden, dropout, use_safe_pair)

    def encode(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        hidden = {
            node_type: F.relu(self.projection[node_type](graph[node_type].x))
            for node_type in graph.node_types
        }
        for convolution in self.layers:
            updated = convolution(hidden, graph.edge_index_dict)
            hidden = {
                node_type: F.dropout(
                    F.relu(updated.get(node_type, value)), self.dropout, self.training
                )
                for node_type, value in hidden.items()
            }
        return hidden


def decode(
    model: nn.Module, kind: str, encoded: Any, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    if kind == "rgcn":
        return model.decoder(
            encoded[batch["l"]], encoded[batch["p"]], encoded[batch["c"]],
            batch["x"], batch["availability"],
        )
    return model.decoder(
        encoded["lncRNA"][batch["l"]],
        encoded["pathway_family"][batch["p"]],
        encoded["cancer"][batch["c"]],
        batch["x"], batch["availability"],
    )


def nnpu_loss(
    logits: torch.Tensor, labels: torch.Tensor, weak: torch.Tensor,
    prior: float = 0.10, unlabeled_weight: float = 0.12,
    weak_weight: float = 0.35,
) -> torch.Tensor:
    positive = labels.gt(0.5)
    unlabeled = ~positive
    if positive.any():
        weights = torch.where(
            weak[positive],
            torch.full_like(logits[positive], weak_weight),
            torch.ones_like(logits[positive]),
        )
        positive_loss = F.binary_cross_entropy_with_logits(
            logits[positive], torch.ones_like(logits[positive]), reduction="none"
        )
        positive_as_negative = F.binary_cross_entropy_with_logits(
            logits[positive], torch.zeros_like(logits[positive]), reduction="none"
        )
        denominator = weights.sum().clamp_min(1e-8)
        positive_risk = prior * (positive_loss * weights).sum() / denominator
        positive_negative_risk = prior * (positive_as_negative * weights).sum() / denominator
    else:
        positive_risk = logits.sum() * 0
        positive_negative_risk = logits.sum() * 0
    if unlabeled.any():
        unlabeled_risk = F.binary_cross_entropy_with_logits(
            logits[unlabeled], torch.zeros_like(logits[unlabeled])
        )
    else:
        unlabeled_risk = logits.sum() * 0
    return positive_risk + unlabeled_weight * torch.clamp(
        unlabeled_risk - positive_negative_risk, min=0
    )


def move_graph(bundle: GraphBundle, kind: str, device: torch.device) -> Any:
    if kind == "rgcn":
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in bundle.homogeneous.items()
        }
    return bundle.hetero.to(device)


def slice_batch(tensors: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[index] for key, value in tensors.items()}


@torch.no_grad()
def predict_logits(
    model: nn.Module, kind: str, graph: Any,
    tensors: dict[str, torch.Tensor], batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    encoded = model.encode(graph)
    membership, direction = [], []
    for start in range(0, len(tensors["l"]), batch_size):
        index = torch.arange(
            start, min(start + batch_size, len(tensors["l"])),
            device=tensors["l"].device,
        )
        first, second = decode(model, kind, encoded, slice_batch(tensors, index))
        membership.append(first.float().cpu())
        direction.append(second.float().cpu())
    return torch.cat(membership).numpy(), torch.cat(direction).numpy()


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (probability < high if high < 1 else probability <= high)
        if mask.any():
            value += mask.mean() * abs(labels[mask].mean() - probability[mask].mean())
    return float(value)


def metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, float]:
    labels = frame.proxy_label.to_numpy(int)
    if len(np.unique(labels)) < 2:
        return {"auprc": math.nan, "auroc": math.nan, "brier": math.nan, "ece": math.nan}
    return {
        "auprc": float(average_precision_score(labels, probability)),
        "auroc": float(roc_auc_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "ece": expected_calibration_error(labels, probability),
    }


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200))
    losses = []
    for temperature in candidates:
        probability = 1 / (1 + np.exp(-np.clip(logits / temperature, -30, 30)))
        probability = np.clip(probability, 1e-7, 1 - 1e-7)
        losses.append(-np.mean(labels * np.log(probability) + (1 - labels) * np.log(1 - probability)))
    return float(candidates[int(np.argmin(losses))])


def train_fold(
    test_cancer: str,
    kind: str,
    seed: int,
    epochs: int = 250,
    patience: int = 15,
    hidden: int = 96,
    pair_hidden: int = 96,
    heads: int = 2,
    layers: int = 2,
    edge_cap: int = 300_000,
    batch_size: int = 16_384,
    accumulation_steps: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del pair_hidden  # Decoder uses the frozen v2.6 hidden width by contract.
    if kind not in {"rgcn", "hgt", "cc_hhgt_strict"}:
        raise ValueError(kind)
    seed_everything(seed)
    fold = fold_row(test_cancer, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_graph(fold, edge_cap=edge_cap)
    graph = move_graph(bundle, kind, device)
    frames = load_fold_pairs(fold)
    tensor_frames: dict[str, pd.DataFrame] = {}
    tensors: dict[str, dict[str, torch.Tensor]] = {}
    for split, frame in frames.items():
        tensor_frames[split], tensors[split] = candidate_tensors(frame, bundle, kind, device)
        if tensor_frames[split].empty:
            raise RuntimeError(f"No graph-mappable pairs for {fold['fold_id']}/{split}")

    if kind == "rgcn":
        model: nn.Module = RGCNStrict(bundle, hidden, layers, 0.20)
    else:
        model = HGTStrict(
            bundle, hidden, layers, heads, 0.20,
            use_safe_pair=kind == "cc_hhgt_strict",
        )
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model_dir = MODEL_ROOT / kind / fold["fold_id"] / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch = -np.inf, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(len(tensors["train"]["l"]), device=device)
        batches = [
            permutation[start:start + batch_size]
            for start in range(0, len(permutation), batch_size)
        ]
        loss_sum, n_pairs = 0.0, 0
        for group_start in range(0, len(batches), accumulation_steps):
            group = batches[group_start:group_start + accumulation_steps]
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                encoded = model.encode(graph)
            for position, index in enumerate(group):
                batch = slice_batch(tensors["train"], index)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                    logits, direction_logits = decode(model, kind, encoded, batch)
                    loss = nnpu_loss(logits, batch["y"], batch["weak"])
                    known = batch["direction"].ge(0) & batch["y"].gt(0.5)
                    if known.any():
                        loss = loss + 0.25 * F.binary_cross_entropy_with_logits(
                            direction_logits[known], batch["direction"][known]
                        )
                    scaled = loss / len(group)
                scaler.scale(scaled).backward(retain_graph=position < len(group) - 1)
                loss_sum += float(loss.detach().cpu()) * len(index)
                n_pairs += len(index)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        validation_logits, _ = predict_logits(
            model, kind, graph, tensors["validation"], batch_size
        )
        validation_probability = 1 / (1 + np.exp(-np.clip(validation_logits, -30, 30)))
        validation_metrics = metrics(tensor_frames["validation"], validation_probability)
        score = validation_metrics["auprc"]
        history.append({
            "epoch": epoch,
            "loss": loss_sum / max(n_pairs, 1),
            **validation_metrics,
        })
        print(
            f"[{fold['fold_id']} {kind} seed={seed}] "
            f"epoch={epoch} loss={history[-1]['loss']:.6f} "
            f"val_auprc={validation_metrics['auprc']:.6f} "
            f"val_auroc={validation_metrics['auroc']:.6f}",
            flush=True,
        )
        if np.isfinite(score) and score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({
                "model_state": model.state_dict(),
                "kind": kind,
                "fold": fold,
                "safe_pair_features": SAFE_PAIR_FEATURES if kind == "cc_hhgt_strict" else [],
                "forbidden_pair_features": sorted(FORBIDDEN_PAIR_FEATURES),
                "node_feature_policy": "fold_degree_plus_train-fold-standardized-cancer-context",
                "edge_policy": "static_only_relation_sample_then_degree_recompute",
                "analysis_version": "cc_hhgt_v2.6.0-strict",
            }, model_dir / "best.pt")
        elif epoch - best_epoch >= patience:
            break
    checkpoint = torch.load(model_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    validation_logits, _ = predict_logits(
        model, kind, graph, tensors["validation"], batch_size
    )
    temperature = temperature_scale(
        validation_logits, tensor_frames["validation"].proxy_label.to_numpy(int)
    )
    prediction_parts, metric_rows = [], []
    for split in ["train", "validation", "test"]:
        logits, direction_logits = predict_logits(model, kind, graph, tensors[split], batch_size)
        raw_probability = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
        calibrated = 1 / (1 + np.exp(-np.clip(logits / temperature, -30, 30)))
        direction_probability = 1 / (1 + np.exp(-np.clip(direction_logits, -30, 30)))
        output = tensor_frames[split][[
            "cancer_id", "lncrna_id", "pathway_family_id", "label_class",
            "proxy_label", "direction_label", "label_source",
        ]].copy()
        output["fold_id"] = fold["fold_id"]
        output["split"] = split
        output["model_name"] = kind
        output["seed"] = seed
        output["raw_logit"] = logits
        output["raw_probability"] = raw_probability
        output["calibrated_probability"] = calibrated
        output["direction_probability"] = direction_probability
        prediction_parts.append(output)
        metric_rows.append({
            "fold_id": fold["fold_id"], "split": split,
            "model_name": kind, "seed": seed, "temperature": temperature,
            **metrics(tensor_frames[split], calibrated),
        })

    model.eval()
    with torch.no_grad():
        encoded = model.encode(graph)
    if kind == "cc_hhgt_strict":
        embedding_root = model_dir / "embeddings"
        embedding_root.mkdir(exist_ok=True)
        for node_type in ["lncRNA", "pathway_family", "cancer"]:
            values = encoded[node_type].float().cpu().numpy()
            ids = bundle.hetero[node_type].canonical_ids
            embedding = pd.DataFrame(values, columns=[f"embedding_{i}" for i in range(values.shape[1])])
            embedding.insert(0, "canonical_id", ids)
            embedding.insert(1, "node_type", node_type)
            embedding.to_parquet(embedding_root / f"{node_type}.parquet", index=False)

    pd.DataFrame(history).to_csv(model_dir / "training_history.tsv", sep="\t", index=False)
    (model_dir / "calibration.json").write_text(
        json.dumps({"temperature": temperature, "validation_cancer": fold["validation_cancer"]}, indent=2),
        encoding="utf-8",
    )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    metric_frame = pd.DataFrame(metric_rows)
    result_dir = RESULT_ROOT / kind / fold["fold_id"] / f"seed_{seed}"
    result_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(result_dir / "prediction.parquet", index=False)
    metric_frame.to_csv(result_dir / "metrics.tsv", sep="\t", index=False)
    return predictions, metric_frame
