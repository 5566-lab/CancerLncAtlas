#!/usr/bin/env python3
"""Freeze evidence-defined KEGG cancer mechanisms, independently of label coverage."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cla_engine.kegg_cancer_scope import materialize_scope
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['source_table','raw_records','curation','mapping','catalog','gene_catalog','out']:
        p.add_argument('--'+name.replace('_','-'),required=True)
    a=p.parse_args()
    result=materialize_scope(a.source_table,a.raw_records,a.curation,a.mapping,a.catalog,a.gene_catalog,a.out)
    print(json.dumps({k:v for k,v in result.items() if k not in {'files','inputs'}},indent=2))
