import json,socket,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cla_engine.common import sha
from scripts.prepare_graph_ridge import dump

def audit(networks,output):
    assert socket.gethostname()=='149';root=Path(networks);out=Path(output);out.mkdir(parents=True,exist_ok=True);dump(out/'PREFLIGHT.json',dict(target_host='149',stage='READ_ONLY_NETWORK_STRUCTURE_AUDIT',code=sha(Path(__file__))))
    rows=[];hubs=[];nulls=[]
    for c in ['BRCA','LUAD','COAD']:
        folder=root/c;nodes=pd.read_csv(folder/'NODES.tsv',sep='\t').node_id.to_numpy();types=np.array([v.split(':')[0] for v in nodes])
        for name in ['ppi','rpi','union','random_101','random_202','random_303','random_404','random_505']:
            e=pd.read_parquet(folder/('TYPED_'+name+'.parquet'));a=sparse.coo_matrix((np.ones(len(e)),(e.lo,e.hi)),shape=(len(nodes),len(nodes))).tocsr();a=a+a.T;active=np.asarray(a.sum(1)).ravel()>0;degree=np.asarray(a.sum(1)).ravel();nc,cc=connected_components(a[active][:,active]);sizes=np.bincount(cc)
            rows.append(dict(cancer_id=c,mode=name,edges=len(e),lncrna_nodes=int((active&(types=='L')).sum()),gene_product_nodes=int((active&(types=='G')).sum()),components=nc,largest_component=int(sizes.max()),largest_component_fraction=float(sizes.max()/active.sum()),weight_min=float(e.weight.min()),weight_median=float(e.weight.median()),weight_max=float(e.weight.max()),distinct_weights=e.weight.nunique(),unweighted_degree_max=float(degree.max())))
            if name in ['ppi','rpi','union']:
                for typ in ['L','G']:
                    ix=np.flatnonzero(active&(types==typ));top=ix[np.argsort(-degree[ix])[:20]]
                    for rank,i in enumerate(top,1):hubs.append(dict(cancer_id=c,mode=name,node_type=typ,rank=rank,node_id=nodes[i],degree=int(degree[i]),share_of_type_edge_ends=float(degree[i]/degree[ix].sum())))
        for row in json.loads((folder/'NULL_AUDIT.json').read_text()):nulls.append(dict(cancer_id=c,**row))
    pd.DataFrame(rows).to_csv(out/'STRUCTURE.tsv',sep='\t',index=False);pd.DataFrame(hubs).to_csv(out/'HUBS.tsv',sep='\t',index=False);pd.DataFrame(nulls).to_csv(out/'NULL_CONTROLS.tsv',sep='\t',index=False);dump(out/'COMPLETE.json',dict(target_host='149',status='COMPLETE',files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json'}));print(pd.DataFrame(rows).query("mode in ['ppi','rpi','union']").to_string(index=False),flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--networks',required=True);p.add_argument('--output',required=True);a=p.parse_args();audit(a.networks,a.output)
