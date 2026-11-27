#!/usr/bin/env python3
"""Export one exact pathway's patient observations from a frozen Y registry."""
from pathlib import Path
import argparse
import json
import socket
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.measured_y_registry import load_target
from cla_engine.common import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registry', type=Path, required=True,
                        help='Per-cancer registry directory containing COMPLETE.json')
    parser.add_argument('--target', required=True, help='Exact canonical pathway or State ID')
    parser.add_argument('--cohort', required=True, choices=['TCGA', 'CPTAC'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if socket.gethostname().split('.')[0] != '149':
        raise RuntimeError('CancerLncAtlas preparation must run on host 149')
    args.output.mkdir(parents=True, exist_ok=False)
    preflight = dict(target_host='149', operation='export fixed Y observations',
                     registry=str(args.registry), target=args.target, cohort=args.cohort,
                     training=False, lncrna_association=False)
    (args.output/'LAUNCH_PREFLIGHT.json').write_text(json.dumps(preflight, indent=2)+'\n')
    values, observed, applicable, metadata = load_target(args.registry, args.target, args.cohort)
    values.to_parquet(args.output/'VALUES.parquet')
    observed.to_parquet(args.output/'OBSERVED.parquet')
    applicable.to_frame('applicability').to_parquet(args.output/'APPLICABILITY.parquet')
    metadata.to_csv(args.output/'OBSERVATIONS.tsv', sep='\t', index=False)
    receipt = dict(preflight, patients=len(values), observations=values.shape[1],
                   registry_receipt_sha256=sha(args.registry/'COMPLETE.json'),
                   files={p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    (args.output/'COMPLETE.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(dict(patients=len(values), observations=values.shape[1],
                         output=str(args.output))))


if __name__ == '__main__':
    main()
