#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cla_engine.relation_catalog import materialize_catalog
p=argparse.ArgumentParser()
p.add_argument('--catalog',required=True);p.add_argument('--backbone',required=True);p.add_argument('--out',required=True);p.add_argument('--protein-annotation')
if __name__=='__main__':
    a=p.parse_args();r=materialize_catalog(a.catalog,a.backbone,a.out,a.protein_annotation)
    print(json.dumps({k:v for k,v in r.items() if k!='files'},indent=2))
