#!/usr/bin/env python3
"""Minimal CPU smoke test for the pinned PyTorch/PyG G012 runtime."""
from __future__ import annotations

import json


def main() -> int:
    import torch
    import torch_geometric
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HGTConv, RGCNConv

    data = HeteroData()
    data["lncrna"].x = torch.randn(4, 8, requires_grad=True)
    data["pathway"].x = torch.randn(3, 8, requires_grad=True)
    data["lncrna", "regulates", "pathway"].edge_index = torch.tensor(
        [[0, 1, 2, 3], [0, 1, 2, 0]], dtype=torch.long
    )
    data["pathway", "rev_regulates", "lncrna"].edge_index = torch.tensor(
        [[0, 1, 2, 0], [0, 1, 2, 3]], dtype=torch.long
    )
    hgt = HGTConv(-1, 8, data.metadata(), heads=2)
    output = hgt(data.x_dict, data.edge_index_dict)
    loss = sum(value.square().mean() for value in output.values())
    loss.backward()

    rgcn = RGCNConv(8, 8, num_relations=2)
    homogeneous_x = torch.randn(4, 8, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_type = torch.tensor([0, 1, 0], dtype=torch.long)
    rgcn_output = rgcn(homogeneous_x, edge_index, edge_type)
    rgcn_output.square().mean().backward()
    payload = {
        "status": "PASS",
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "heterodata_node_types": data.node_types,
        "heterodata_edge_types": [list(value) for value in data.edge_types],
        "hgt_output_shapes": {key: list(value.shape) for key, value in output.items()},
        "rgcn_output_shape": list(rgcn_output.shape),
        "cpu_backward": True,
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
