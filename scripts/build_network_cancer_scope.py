#!/usr/bin/env python3
"""Materialize exact-N cancer scope and typed cross-database alignment on host 149."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cla_engine.network_scope import materialize_scope

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mapping',required=True)
    p.add_argument('--snapshot-root',required=True)
    p.add_argument('--catalog-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--snapshot-manifest')
    p.add_argument('--protein-pairs',help='Existing frozen similarity candidates; never grants admission')
    a=p.parse_args()
    print(json.dumps(materialize_scope(a.mapping,a.snapshot_root,a.catalog_root,a.out,
          snapshot_manifest=a.snapshot_manifest,protein_pairs=a.protein_pairs),indent=2))
if __name__=='__main__':
    main()
