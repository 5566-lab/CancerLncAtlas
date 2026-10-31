"""Index verified causal inputs or prepare one context-specific response graph."""
from __future__ import annotations

import argparse
from pathlib import Path
import socket

from cc_hhgt.v32.causal_regulatory_inputs import build_inputs, sha256, write_json
from cc_hhgt.v32.causal_regulatory_network import CausalRegulatoryNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    index = sub.add_parser("index")
    index.add_argument("--source-root", required=True, type=Path)
    index.add_argument("--source-manifest-sha256", required=True)
    index.add_argument("--output", required=True, type=Path)
    graph = sub.add_parser("network")
    graph.add_argument("--input-manifest", required=True, type=Path)
    graph.add_argument("--source", required=True)
    graph.add_argument("--experiment", required=True)
    graph.add_argument("--output", required=True, type=Path)
    graph.add_argument("--max-fdr", type=float)
    graph.add_argument("--cross-type-only", action="store_true")
    graph.add_argument("--purpose", choices=("exploration", "mechanism_validation"), default="exploration")
    args = parser.parse_args()
    if socket.gethostname() != "149":
        raise RuntimeError("Causal network preparation must execute on host 149")
    if args.operation == "index":
        result = build_inputs(args.source_root, args.output,
                              expected_manifest_sha256=args.source_manifest_sha256)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "LAUNCH_PREFLIGHT.json", dict(target_host="149",
            operation="PREPARE_EXPERIMENTAL_REGULATORY_GRAPH", training_started=False,
            input_manifest_sha256=sha256(args.input_manifest), source=args.source,
            experiment_id=args.experiment))
        network = CausalRegulatoryNetwork(args.input_manifest)
        responses = network.responses(args.source, args.experiment)
        bundle = network.graph(args.source, args.experiment, max_fdr=args.max_fdr,
                               cross_type_only=args.cross_type_only, purpose=args.purpose)
        responses.to_parquet(args.output / "RESPONSES.parquet", index=False)
        bundle.nodes.to_parquet(args.output / "NODES.parquet", index=False)
        bundle.edges.to_parquet(args.output / "EDGES.parquet", index=False)
        import torch
        torch.save(dict(format="EXPERIMENTAL_REGULATORY_GRAPH_V1", bundle=bundle,
                        authority=bundle.experimental_regulatory_authority,
                        contains_trained_parameters=False, contains_optimizer_state=False),
                   args.output / "GRAPH.pt")
        result = dict(target_host="149", status="PASS", training_started=False,
                      authority=bundle.experimental_regulatory_authority,
                      files={name:{"path":name, "sha256":sha256(args.output/name)} for name in
                             ("RESPONSES.parquet", "NODES.parquet", "EDGES.parquet", "GRAPH.pt")})
        write_json(args.output / "COMPLETE.json", result)
    import json
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
