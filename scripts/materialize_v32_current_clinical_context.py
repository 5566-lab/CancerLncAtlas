#!/usr/bin/env python3
"""Join native Clinical results to the current selected analysis; no training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def main(argv=None):
    from cc_hhgt.v32.current_clinical_context import materialize_current_clinical_context
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(materialize_current_clinical_context(args.manifest,args.output),ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
