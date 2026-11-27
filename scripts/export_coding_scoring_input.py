"""Build coding-only RNA scoring inputs on149 from a fixed development split."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cla_engine.protein_coding_scoring import materialize_coding_input

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['aligned','annotation','split','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args()
    r=materialize_coding_input(a.aligned,a.annotation,a.split,a.output)
    print('CODING_INPUT_COMPLETE',r['patients'],r['coding_gene_ids'],r['coding_symbols'],flush=True)
