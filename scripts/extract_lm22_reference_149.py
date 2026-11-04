"""Extract the original publication's LM22 reference and DEG indicator sheets."""
import argparse, socket, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cla_engine.common import sha, atomic_json

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
if socket.gethostname().split('.')[0]!='149':raise RuntimeError('Run on host 149')
source=Path(a.source)
if sha(source)!='a6216f8571ee056c5460a85f28ab5d2bfe5aca665cdb90a2bf3e30a38ac648f1':
    raise ValueError('The original LM22 supplementary workbook differs from the pinned publication file')
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
atomic_json(out/'LAUNCH_PREFLIGHT.json',{'target_host':'149','action':'EXTRACT_ORIGINAL_LM22_REFERENCE'})
f=pd.read_excel(source,sheet_name='SuppTable1_GEP_Matrix',header=13).iloc[:,2:].dropna(subset=['Gene symbol'])
d=pd.read_excel(source,sheet_name='SuppTable1_DEGs',header=2)
assert f.shape==d.shape==(547,23)
f.columns=['gene_symbol']+list(f.columns[1:]);d.columns=['gene_symbol']+list(d.columns[1:])
assert f.gene_symbol.is_unique and f.gene_symbol.tolist()==d.gene_symbol.tolist()
f.to_csv(out/'LM22_OFFICIAL_MATRIX.tsv',sep='\t',index=False,float_format='%.17g')
d.to_csv(out/'LM22_OFFICIAL_DEGS.tsv',sep='\t',index=False)
atomic_json(out/'LM22_EXTRACTION.json',{'target_host':'149','source_sha256':sha(source),
 'genes':547,'cell_types':22,'reference_is_not_regression_coefficients':True,
 'files':{q.name:sha(q) for q in out.glob('*.tsv')}})
