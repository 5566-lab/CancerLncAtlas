"""Prepare typed PPI/RPI partitions and degree-preserving nulls on host 149."""
from __future__ import annotations
import argparse,json,re,socket,time,sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cla_engine.common import sha
from scripts.prepare_graph_ridge import dump,network
from cc_hhgt.network_evidence import physical_rpi_mask
BLOCKS=('RNA','CNV','MUTATION','PROTEIN','PHOSPHO')
SEEDS=(101,202,303,404,505)

def rewire(edges,n_nodes,seed,bipartite=False,swaps_per_edge=5):
    e=edges[['lo','hi','weight']].copy().reset_index(drop=True)
    if len(e)<2:return e,dict(successful_swaps=0,edge_overlap=1.)
    a=e.lo.to_numpy(copy=True);b=e.hi.to_numpy(copy=True);rng=np.random.default_rng(seed)
    current=set(zip(a.tolist(),b.tolist()));original=set(current);degree=np.bincount(np.r_[a,b],minlength=n_nodes)
    target=swaps_per_edge*len(e);done=0;attempt=0;limit=100*max(1,target)
    while done<target and attempt<limit:
        proposals=rng.integers(len(e),size=(min(10000,limit-attempt),2));flips=rng.integers(2,size=len(proposals))
        for z,flip in zip(proposals,flips):
            attempt+=1;i,j=map(int,z)
            if i==j:continue
            u,v=int(a[i]),int(b[i]);x,y=int(a[j]),int(b[j])
            if not bipartite and flip:x,y=y,x
            if len({u,v,x,y})<4:continue
            p=(u,y) if bipartite else tuple(sorted((u,y)));q=(x,v) if bipartite else tuple(sorted((x,v)))
            if p in current or q in current:continue
            current.remove((int(a[i]),int(b[i])));current.remove((int(a[j]),int(b[j])))
            current.add(p);current.add(q);a[i],b[i]=p;a[j],b[j]=q;done+=1
            if done>=target:break
    if done<target:raise RuntimeError('INSUFFICIENT_NULL_MIXING')
    if not np.array_equal(degree,np.bincount(np.r_[a,b],minlength=n_nodes)):raise ValueError('NULL_DEGREE_DRIFT')
    if len(current)!=len(e) or (a==b).any():raise ValueError('NULL_EDGE_INVALID')
    e['lo']=a;e['hi']=b
    return e,dict(successful_swaps=done,attempts=attempt,edge_overlap=len(current&original)/len(current),degree_exact=True,weight_multiset_exact=True,weighted_node_strength_preserved=False)

def laplacian(edges,n):
    a=sparse.coo_matrix((edges.weight,(edges.lo,edges.hi)),shape=(n,n)).tocsr();a=a+a.T
    d=np.asarray(a.sum(axis=1)).ravel();inv=np.divide(1,np.sqrt(d),out=np.zeros_like(d),where=d>0)
    return (sparse.diags((d>0).astype(float))-sparse.diags(inv)@a@sparse.diags(inv)).tocsr()

def partition(edges,nodes,seed=1979):
    g=nx.Graph()
    for kind,ee in edges.groupby('kind',sort=True):
        w=ee.weight.to_numpy(float);w=w/w.sum()
        g.add_weighted_edges_from(zip(ee.lo.astype(int),ee.hi.astype(int),w))
    if not len(g):return {},dict(communities=0,connected_nodes=0,modularity=None)
    comm=nx.community.louvain_communities(g,weight='weight',resolution=1.,seed=seed)
    comm=sorted(comm,key=lambda z:min(z));assignment={nodes[i]:j for j,group in enumerate(comm) for i in group}
    return assignment,dict(communities=len(comm),connected_nodes=len(g),modularity=nx.community.modularity(g,comm,weight='weight'),min_nodes=min(map(len,comm)),max_nodes=max(map(len,comm)))

def feature_map(source,annotation):
    ft=pd.read_csv(source/'RNA_FEATURES.tsv',sep='\t');lnc_genes=set(ft.loc[ft.feature_type.eq('lncrna'),'gene_id'])
    ann=pd.read_csv(annotation,sep='\t');good=ann.dropna(subset=['reference_symbol']).copy();counts=good.groupby('reference_symbol').gene_id.nunique();good=good[good.reference_symbol.isin(counts[counts.eq(1)].index)].drop_duplicates('reference_symbol');sym=good.set_index('reference_symbol').gene_id.to_dict()
    rows=[]
    for block in BLOCKS:
        import pyarrow.parquet as pq
        # Read the stored pandas axis; no feature order is inferred from names.
        schema=pq.read_schema(source/(block+'.parquet'));meta=json.loads(schema.metadata[b'pandas']);index_names={x for x in meta['index_columns'] if isinstance(x,str)};cols=[x for x in schema.names if x not in index_names]
        for i,col in enumerate(cols):
            arm=None;gene=None;status='EXACT_GENE_ID'
            if block=='RNA':
                row=ft.iloc[i]
                if col!=row.feature_id:raise ValueError('RNA_FEATURE_ORDER')
                gene=row.gene_id;arm=row.feature_type;anchor=('L:' if arm=='lncrna' else 'G:')+gene
            elif block in ['PROTEIN','PHOSPHO']:
                gene=col.split(':',1)[1].split('|')[0].split('.')[0]
                if not re.fullmatch(r'ENSG[0-9]+',gene):gene=None;status='UNMAPPED_FEATURE_ID'
                anchor='G:'+gene if gene else None
            else:
                gene=sym.get(col.split(':',1)[1]);status='UNAMBIGUOUS_REFERENCE_SYMBOL' if gene else 'UNMAPPED_OR_AMBIGUOUS_SYMBOL';anchor=(('L:' if gene in lnc_genes else 'G:')+gene) if gene else None
            rows.append(dict(block=block,position=i,feature_id=col,gene_id=gene,anchor=anchor,rna_arm=arm,mapping_status=status))
    return pd.DataFrame(rows)

def prepare(inputs,output,cancer):
    if socket.gethostname()!='149':raise RuntimeError('HOST_149_REQUIRED')
    src=Path(inputs)/cancer;out=Path(output)/cancer;out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE.json').exists():
        for name,h in json.loads((out/'COMPLETE.json').read_text())['files'].items():
            if sha(out/name)!=h:raise ValueError('PREPARED_OUTPUT_DRIFT')
        print('PREPARE_ALREADY_COMPLETE',cancer,flush=True);return
    dump(out/'PREFLIGHT.json',dict(target_host='149',stage='PREPARE_NETWORK_CONTROLS',source=str(src),code_sha256=sha(Path(__file__)),physical_evidence_gate_sha256=sha(ROOT/'cc_hhgt/network_evidence.py'),started_epoch=time.time()))
    rec=json.loads((src/'COMPLETE.json').read_text())
    for name,h in rec['files'].items():
        if sha(src/name)!=h:raise ValueError('INPUT_HASH_DRIFT')
    ann=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/aligned_r1')/cancer/'GENE_ANNOTATION.tsv'
    fmap=feature_map(src,ann);nodes=sorted(set(fmap.anchor.dropna()));axis={x:i for i,x in enumerate(nodes)}
    authority=Path('/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/outputs/phase7_typed_authority/GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json');ar=json.loads(authority.read_text());tables={}
    for k in ['ppi','protein_gene_encoding','lnc_protein_binding']:
        p=Path(ar['artifacts'][k]['path'])
        if sha(p)!=ar['artifacts'][k]['sha256']:raise ValueError('NETWORK_AUTHORITY_DRIFT')
        tables[k]=pd.read_parquet(p)
    pg=tables['protein_gene_encoding'];n=pg.groupby('protein_id').gene_id.nunique();pg=pg[pg.protein_id.isin(n[n.eq(1)].index)].drop_duplicates('protein_id');gm=pg.set_index('protein_id').gene_id.str.replace('GENE:','G:',regex=False)
    pp=tables['ppi'];ppi=pd.DataFrame(dict(a=pp.protein_id_a.map(gm),b=pp.protein_id_b.map(gm),weight=pp.weight,kind='PPI'))
    rr=tables['lnc_protein_binding'];physical=physical_rpi_mask(rr)
    rr.assign(physical_rpi_admitted=physical).groupby(['physical_rpi_admitted','source_database','graph_assay_class','experiment_family','assay_subtype'],dropna=False).size().rename('rows').reset_index().to_csv(out/'RPI_EVIDENCE_AUDIT.tsv',sep='\t',index=False)
    rr=rr[physical&(~rr.is_context_specific|rr.cancer_id.eq(cancer))]
    rpi=pd.DataFrame(dict(a=rr.lncrna_id.str.replace('LNC:','L:',regex=False),b=rr.protein_id.map(gm),weight=rr.weight,kind='RPI'))
    typed={}
    for name,frame in [('PPI',ppi),('RPI',rpi)]:
        frame=frame[frame.a.isin(axis)&frame.b.isin(axis)&frame.a.ne(frame.b)].copy();a=frame.a.map(axis).to_numpy();b=frame.b.map(axis).to_numpy()
        if name=='PPI':a,b=np.minimum(a,b),np.maximum(a,b)
        e=pd.DataFrame(dict(lo=a,hi=b,weight=frame.weight.to_numpy(),kind=name));e=e.groupby(['lo','hi','kind'],as_index=False).weight.max();typed[name]=e
        e.to_parquet(out/(name+'_EDGES.parquet'),index=False)
    pd.DataFrame(dict(node_id=nodes)).to_csv(out/'NODES.tsv',sep='\t',index=False)
    historical=pd.read_parquet(src/'NETWORK_EDGES.parquet');rna=pd.read_csv(src/'RNA_FEATURES.tsv',sep='\t');corrected_lap,old,corrected_audit=network(authority,rna,cancer);legacy={k:e.copy() for k,e in old.groupby('kind')}
    old.to_parquet(out/'CORRECTED_RNA_NETWORK_EDGES.parquet',index=False);dump(out/'CORRECTED_RNA_NETWORK_AUDIT.json',corrected_audit)
    dump(out/'HISTORICAL_NETWORK_COMPARISON.json',dict(historical_edge_counts=historical.kind.value_counts().to_dict(),corrected_edge_counts=old.kind.value_counts().to_dict(),historical_network_not_used_for_fits=True,reason='Exclude nonphysical evidence and enforce RNA-proxy node types'))
    # Legacy RPI swaps must respect actual lncRNA/protein-side labels, not numeric ordering.
    ri=legacy['RPI'];a=ri.lo.to_numpy();b=ri.hi.to_numpy();types=rna.feature_type.to_numpy();left=types[a]=='lncrna';valid=(types[a]=='lncrna')!=(types[b]=='lncrna')
    # Non-bipartite legacy aliases are retained unchanged, and reported explicitly.
    legacy_fixed=ri.loc[~valid].copy();ri=ri.loc[valid].copy();aa=a[valid];bb=b[valid];ll=left[valid];ri['lo']=np.where(ll,aa,bb);ri['hi']=np.where(ll,bb,aa);legacy['RPI_SWAPPABLE']=ri
    audits=[];allmodes={};legmodes={}
    for mode,kinds in [('ppi',['PPI']),('rpi',['RPI']),('union',['PPI','RPI'])]:
        allmodes[mode]=pd.concat([typed[k] for k in kinds],ignore_index=True);legmodes[mode]=pd.concat([legacy[k] for k in kinds],ignore_index=True)
    for seed in SEEDS:
        newer=[];older=[]
        for kind in ['PPI','RPI']:
            e,audit=rewire(typed[kind],len(nodes),seed+(11 if kind=='PPI' else 29),bipartite=kind=='RPI');e['kind']=kind;newer.append(e);audits.append(dict(graph='typed',kind=kind,seed=seed,**audit))
            e,audit=rewire(legacy[kind] if kind=='PPI' else legacy['RPI_SWAPPABLE'],len(rna),seed+(37 if kind=='PPI' else 53),bipartite=kind=='RPI');e['kind']=kind;older.append(e);audits.append(dict(graph='legacy',kind=kind,seed=seed,fixed_alias_edges=len(legacy_fixed) if kind=='RPI' else 0,**audit))
        allmodes['random_'+str(seed)]=pd.concat(newer,ignore_index=True);legmodes['random_'+str(seed)]=pd.concat(older+[legacy_fixed],ignore_index=True)
        print('REWIRED',cancer,seed,flush=True)
    partitions={};stats=[]
    for mode,e in allmodes.items():
        e.to_parquet(out/('TYPED_'+mode+'.parquet'),index=False);assignment,stat=partition(e,nodes);partitions[mode]=assignment;stats.append(dict(mode=mode,**stat));fmap['module_'+mode]=fmap.anchor.map(assignment).fillna(-1).astype(int)
        pd.DataFrame(assignment.items(),columns=['node_id','module']).to_csv(out/('MODULES_'+mode+'.tsv'),sep='\t',index=False)
        l=laplacian(legmodes[mode],len(rna));sparse.save_npz(out/('LEGACY_'+mode+'.npz'),l)
        if mode=='union':
            original=corrected_lap;diff=l-original
            if diff.nnz and np.max(np.abs(diff.data))>1e-12:raise ValueError('LEGACY_LAPLACIAN_REPRODUCTION_FAILED')
        print('PARTITIONED',cancer,mode,stat,flush=True)
    fmap.to_parquet(out/'FEATURE_MAP.parquet',index=False);pd.DataFrame(stats).to_csv(out/'PARTITION_AUDIT.tsv',sep='\t',index=False);dump(out/'NULL_AUDIT.json',audits)
    fmap.groupby(['block','mapping_status']).size().rename('features').reset_index().to_csv(out/'MAPPING_AUDIT.tsv',sep='\t',index=False)
    dump(out/'BOUND_SOURCES.json',dict(input_receipt=str(src/'COMPLETE.json'),input_receipt_sha256=sha(src/'COMPLETE.json'),authority=str(authority),authority_sha256=sha(authority),annotation=str(ann),annotation_sha256=sha(ann),networkx_version=nx.__version__,legacy_nonbipartite_rpi_alias_edges=len(legacy_fixed)))
    dump(out/'COMPLETE.json',dict(target_host='149',status='COMPLETE',source=str(src),files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json'}));print('PREPARED_NETWORK_CONTROLS',cancer,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True);p.add_argument('--output',required=True);p.add_argument('--cancer',required=True,choices=['BRCA','LUAD','COAD']);a=p.parse_args();prepare(a.inputs,a.output,a.cancer)
