#!/usr/bin/env python3
"""Build source-only native molecular graph inputs on host 149."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cla_engine.relation_view import materialize_relation_graph_view

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('components','static','bindings','out'):p.add_argument('--'+arg,required=True)
    a=p.parse_args()
    result=materialize_relation_graph_view(a.components,a.static,json.loads(Path(a.bindings).read_text()),a.out,
        progress=lambda row:print(json.dumps(row),flush=True))
    print(result)

if __name__=='__main__':main()
