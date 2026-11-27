#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cla_engine.functional_target_view import build_target_view
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True);p.add_argument('--catalog',required=True);p.add_argument('--out',required=True)
    p.add_argument('--cancers',nargs='+',default=['BRCA','LUAD','COAD'])
    a=p.parse_args()
    print(json.dumps(build_target_view(a.source,a.catalog,a.out,a.cancers),indent=2))
