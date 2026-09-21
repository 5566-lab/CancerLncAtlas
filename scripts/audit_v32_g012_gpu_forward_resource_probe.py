#!/usr/bin/env python3
"""Forward-only GPU resource probe for the revision-3 G2 runtime chunk.

The probe uses a synthetic typed graph with the registered node/edge counts.
It creates no optimizer, performs no backward pass, and writes no model,
checkpoint, embedding, candidate score, prediction, or training marker.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


NODE_COUNTS = {
    "cancer": 33,
    "gene": 21_981,
    "lncRNA": 8_541,
    "pathway": 2_135,
    "pathway_family": 96,
    # Conservative design-time ceiling; the fresh materializer must replace
    # this with the exact frozen union and rerun the probe before training.
    "protein": 25_000,
}

EDGE_SPECS = (
    ("lncRNA", "expressed_in", "cancer", 76_734),
    ("cancer", "rev_expressed_in", "lncRNA", 76_734),
    ("lncRNA", "coexpressed_positive", "gene", 125_000),
    ("gene", "rev_coexpressed_positive", "lncRNA", 125_000),
    ("lncRNA", "coexpressed_negative", "gene", 125_000),
    ("gene", "rev_coexpressed_negative", "lncRNA", 125_000),
    ("gene", "member_of_positive", "pathway", 224_414),
    ("pathway", "rev_member_of_positive", "gene", 224_414),
    ("gene", "member_of_negative", "pathway", 125_194),
    ("pathway", "rev_member_of_negative", "gene", 125_194),
    ("pathway", "member_of_family", "pathway_family", 2_114),
    ("pathway_family", "rev_member_of_family", "pathway", 2_114),
    ("lncRNA", "binds_protein", "protein", 623_207),
    ("protein", "rev_binds_protein", "lncRNA", 623_207),
    ("protein", "encoded_by", "gene", 20_008),
    ("gene", "rev_encoded_by", "protein", 20_008),
    # Exactly two reciprocal messages under one self-type relation.  There is
    # deliberately no rev_physical_interaction relation.
    ("protein", "physical_interaction", "protein", 154_852),
)

STATIC_SOURCE_ROWS = 1_149_097
VARIABLE_COEXPRESSION_SOURCE_ROWS = 250_000
EXPECTED_DIRECTED_MESSAGES = 2_798_194


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260726)
    return parser.parse_args()


def _edge_index(torch, count: int, source_nodes: int, target_nodes: int, salt: int):
    position = torch.arange(count, dtype=torch.int64)
    source = (position * 104_729 + salt * 7_919) % source_nodes
    target = (position * 130_363 + salt * 15_407) % target_nodes
    return torch.stack((source, target), dim=0)


def main() -> int:
    args = _args()
    output = args.output_json.resolve()
    allowed = Path(
        r"D:\model\CC_HHGT_v3_2_ranked_subtypes_dev\artifacts\v32_g012_gpu_forward_resource_probe_20260829_r1"
    ).resolve()
    if output.parent != allowed:
        raise RuntimeError(f"Output must be directly under {allowed}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite probe output: {output}")

    import torch
    import torch.nn.functional as functional
    from torch import nn
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HGTConv

    if not torch.cuda.is_available() or torch.device(args.device).type != "cuda":
        raise RuntimeError("A visible CUDA device is required")
    if sum(spec[3] for spec in EDGE_SPECS) != EXPECTED_DIRECTED_MESSAGES:
        raise RuntimeError("Registered directed-message count drift")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    free_before, total_memory = torch.cuda.mem_get_info(device)
    data = HeteroData()
    for node_type, count in NODE_COUNTS.items():
        data[node_type].x = torch.ones((count, 2), dtype=torch.float32)
    for salt, (source_type, relation, target_type, count) in enumerate(EDGE_SPECS, 1):
        edge_type = (source_type, relation, target_type)
        data[edge_type].edge_index = _edge_index(
            torch,
            count,
            NODE_COUNTS[source_type],
            NODE_COUNTS[target_type],
            salt,
        )
        data[edge_type].edge_weight = torch.ones(count, dtype=torch.float32)

    class ForwardProbe(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.ModuleDict(
                {node_type: nn.Linear(2, args.hidden) for node_type in data.node_types}
            )
            self.layers = nn.ModuleList(
                [
                    HGTConv(
                        args.hidden,
                        args.hidden,
                        data.metadata(),
                        heads=args.heads,
                    )
                    for _ in range(args.layers)
                ]
            )
            self.weighted_projection = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            node_type: nn.Linear(args.hidden, args.hidden, bias=False)
                            for node_type in data.node_types
                        }
                    )
                    for _ in range(args.layers)
                ]
            )

        @staticmethod
        def weighted_aggregate(hidden, graph, projection):
            aggregate = {
                node_type: torch.zeros_like(value) for node_type, value in hidden.items()
            }
            denominator = {
                node_type: torch.zeros(
                    (value.shape[0], 1), dtype=value.dtype, device=value.device
                )
                for node_type, value in hidden.items()
            }
            for edge_type in graph.edge_types:
                source_type, _, target_type = edge_type
                source, target = graph[edge_type].edge_index
                weight = graph[edge_type].edge_weight.to(dtype=hidden[source_type].dtype)
                aggregate[target_type].index_add_(
                    0,
                    target,
                    hidden[source_type][source] * weight.unsqueeze(-1),
                )
                denominator[target_type].index_add_(
                    0, target, weight.abs().unsqueeze(-1)
                )
            return {
                node_type: projection[node_type](
                    aggregate[node_type] / denominator[node_type].clamp_min(1e-8)
                )
                for node_type in hidden
            }

        def forward(self, graph):
            hidden = {
                node_type: functional.relu(self.projection[node_type](graph[node_type].x))
                for node_type in graph.node_types
            }
            for convolution, projection in zip(
                self.layers, self.weighted_projection, strict=True
            ):
                updated = convolution(hidden, graph.edge_index_dict)
                weighted = self.weighted_aggregate(hidden, graph, projection)
                hidden = {
                    node_type: functional.relu(
                        updated.get(node_type, hidden[node_type]) + weighted[node_type]
                    )
                    for node_type in hidden
                }
            return hidden

    model = ForwardProbe().eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    status = "PASS_FORWARD_ONLY_RESOURCE_PROBE_NOT_AN_AUTHORITY"
    error = None
    checksum = None
    output_shapes = None
    graph = None
    model_on_device = None
    try:
        graph = data.to(device)
        model_on_device = model.to(device)
        torch.cuda.synchronize(device)
        forward_started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            encoded = model_on_device(graph)
        torch.cuda.synchronize(device)
        forward_seconds = time.perf_counter() - forward_started
        output_shapes = {
            node_type: list(value.shape) for node_type, value in encoded.items()
        }
        checksum = float(
            sum(value[:1, :1].float().cpu().item() for value in encoded.values())
        )
        del encoded
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        status = "CUDA_OOM_FORWARD_ONLY_RESOURCE_PROBE"
        error = str(exc)
        forward_seconds = None
    finally:
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        free_after, _ = torch.cuda.mem_get_info(device)
        del graph, model_on_device, model, data
        torch.cuda.empty_cache()

    report = {
        "status": status,
        "design_only": True,
        "training_started": False,
        "optimizer_created": False,
        "backward_called": False,
        "checkpoint_written": False,
        "embedding_written": False,
        "prediction_written": False,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "node_counts": NODE_COUNTS,
        "edge_type_counts": {
            "|".join(spec[:3]): spec[3] for spec in EDGE_SPECS
        },
        "static_source_rows": STATIC_SOURCE_ROWS,
        "static_directed_messages": 2 * STATIC_SOURCE_ROWS,
        "variable_coexpression_source_rows": VARIABLE_COEXPRESSION_SOURCE_ROWS,
        "variable_coexpression_directed_messages": 2
        * VARIABLE_COEXPRESSION_SOURCE_ROWS,
        "total_directed_messages": EXPECTED_DIRECTED_MESSAGES,
        "ppi_relation_types": 1,
        "ppi_directed_messages": 154_852,
        "model": {
            "hidden": args.hidden,
            "heads": args.heads,
            "layers": args.layers,
            "parameter_count": int(parameter_count),
            "mode": "EVAL_INFERENCE_MODE_BF16_FORWARD_ONLY",
        },
        "memory": {
            "device_total_bytes": int(total_memory),
            "device_free_before_bytes": int(free_before),
            "device_free_after_forward_bytes": int(free_after),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
        },
        "wall_seconds": {
            "forward": forward_seconds,
            "total_after_model_construction": time.perf_counter() - started,
        },
        "output_shapes_not_persisted": output_shapes,
        "synthetic_output_checksum_not_a_prediction": checksum,
        "error": error,
        "environment_threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "limitations": [
            "Synthetic endpoints preserve registered counts, not biological topology.",
            "Protein node count is a conservative 25,000 ceiling pending fresh freeze.",
            "Forward-only memory excludes gradients, optimizer, candidate decoder batches, and training activations.",
            "The COMPUTE_HOST login node has no visible CUDA; this uses the original local 4070 Ti SUPER training environment.",
        ],
    }
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, output)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
