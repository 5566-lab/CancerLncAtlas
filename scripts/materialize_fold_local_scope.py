#!/usr/bin/env python3
"""Create annotation-defined candidates and fold-local detection scope on 149."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cc_hhgt.v32.fold_local_scope import materialize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args();materialize(args.manifest,args.output,runner=__file__)


if __name__=="__main__":main()
