#!/usr/bin/env python3
"""Fit matched Pathway and State numeric Lasso components on host 149."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cla_engine.relation_baseline import materialize_relation_baseline

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
    result=materialize_relation_baseline(plan['components'],plan['policy'],plan['state_definitions'],a.out,
        cfg=plan.get('config'),progress=lambda row:print(json.dumps(row),flush=True))
    print(json.dumps(result['tasks'],indent=2))

if __name__=='__main__':main()
