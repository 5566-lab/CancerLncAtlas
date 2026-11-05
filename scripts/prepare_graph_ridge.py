"""Prepare measured multiomics and outcome-independent PPI/RPI priors on 149."""
import argparse,hashlib,json,re,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cc_hhgt.m0_three_arm import feature_axes,restrict_split
from cla_engine.common import sha

def dump(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True,default=str))
def verify(p,h):
    if sha(p)!=h:raise ValueError('SOURCE_HASH_CHANGED: '+str(p))
def patient_mapping(columns,patients,cancer):
    known={p.split(':',2)[2]:p for p in patients};aliases={}
    for case,pid in known.items():
        for a in {case,case.replace('-','.'),'X'+case,'X'+case.replace('-','.')}:
            aliases.setdefault(a,set()).add(pid)
    mapping={c:next(iter(aliases[str(c)])) for c in columns if str(c) in aliases and len(aliases[str(c)])==1}
    if len(set(mapping.values()))!=len(mapping):raise ValueError('PATIENT_ALIAS_COLLISION')
    return mapping

def network(authority,features,cancer):
    d=json.loads(authority.read_text());tables={}
    for key in ['ppi','protein_gene_encoding','lnc_protein_binding']:
        e=d['artifacts'][key];verify(Path(e['path']),e['sha256']);tables[key]=pd.read_parquet(e['path'])
    pg=tables['protein_gene_encoding'];pg=pg.assign(gene_id=pg.gene_id.str.removeprefix('GENE:'))
    # Reject ambiguous protein -> gene assignments; never create a guessed link.
    unique=pg.groupby('protein_id').gene_id.nunique();pg=pg[pg.protein_id.isin(unique[unique.eq(1)].index)].drop_duplicates('protein_id')
    gm=pg.set_index('protein_id').gene_id;axis={g:i for i,g in enumerate(features.gene_id)}
    pp=tables['ppi'];a=pp.protein_id_a.map(gm);b=pp.protein_id_b.map(gm)
    pp=pd.DataFrame(dict(a=a,b=b,weight=pp.weight,kind='PPI'))
    ri=tables['lnc_protein_binding'];original=len(ri)
    allowed=['eclip','other_clip','rip','rna_capture','other_physical','experimental_unspecified']
    ri=ri[ri.graph_assay_class.isin(allowed)&(~ri.is_context_specific | ri.cancer_id.eq(cancer))].copy()
    ri=pd.DataFrame(dict(a=ri.lncrna_id.str.removeprefix('LNC:'),b=ri.protein_id.map(gm),weight=ri.weight,kind='RPI'))
    e=pd.concat([pp,ri],ignore_index=True);e=e[e.a.isin(axis)&e.b.isin(axis)&e.a.ne(e.b)]
    if not np.isfinite(e.weight).all() or e.weight.le(0).any():raise ValueError('INVALID_NETWORK_WEIGHT')
    e['i']=e.a.map(axis);e['j']=e.b.map(axis)
    e['lo']=np.minimum(e.i,e.j);e['hi']=np.maximum(e.i,e.j)
    provenance=e.groupby(['lo','hi','kind'],as_index=False).weight.max()
    collapsed=provenance.groupby(['lo','hi'],as_index=False).weight.max()
    adj=sparse.coo_matrix((collapsed.weight,(collapsed.lo,collapsed.hi)),shape=(len(axis),len(axis))).tocsr();adj=adj+adj.T
    degree=np.asarray(adj.sum(axis=1)).ravel();inv=np.divide(1,np.sqrt(degree),out=np.zeros_like(degree),where=degree>0)
    lap=sparse.diags((degree>0).astype(float))-sparse.diags(inv)@adj@sparse.diags(inv)
    audit=dict(nodes=len(axis),undirected_edges=len(collapsed),by_kind=provenance.kind.value_counts().to_dict(),connected_nodes=int((degree>0).sum()),connected_lncrna=int(((degree>0)&features.feature_type.eq('lncrna')).sum()),connected_mrna=int(((degree>0)&features.feature_type.eq('mrna')).sum()),rpi_source_rows=original,rpi_policy='EXPERIMENTAL_CLASSES_ONLY; GLOBAL_OR_MATCHING_CANCER; UNKNOWN_AND_PREDICTED_EXCLUDED',ppi_policy='EXISTING_STRING_CONFIDENCE_PRIOR; ASSOCIATION_NOT_ASSUMED_DIRECT_PHYSICAL',projection='PPI maps protein pairs to unambiguous encoding genes; RPI maps lncRNA to partner encoding gene',latent_bridge='Unmeasured RNA-arm genes remain topology-only coefficients in I+eta*L; no expression from excluded arm is supplied')
    return lap.tocsr(),provenance,audit

def prepare(work,output):
    if socket.gethostname()!='149':raise RuntimeError('HOST_149_REQUIRED')
    work=Path(work);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    dump(out/'LAUNCH_PREFLIGHT.json',dict(target_host='149',stage='PREPARE',code_sha256=sha(Path(__file__))))
    downloads=json.loads((work/'DOWNLOAD_COMPLETE.json').read_text())
    authority=Path('/dell_2/DSC/CancerLncAtlas/rbp_encode_v33_20260921_r1/outputs/phase7_typed_authority/GRAPH_INPUT_AUTHORITY_RECEIPT_TYPED.json')
    base=Path('/dell_2/DSC/CancerLncAtlas/inputs/cptac_20261104_r1/aligned_r4');assayrec=json.loads((base/'COMPLETE.json').read_text())
    joint=Path('/dsk2/user/dengsc/cla_m0_joint_20261105_r1');coverage=[];bindings=[]
    for c in ['BRCA','LUAD','COAD']:
        d=out/c;d.mkdir(exist_ok=True);src=joint/'aligned_r1'/c
        rec=json.loads((src/'COMPLETE.json').read_text())
        for name in ['GENE_LOGTPM.parquet','GENE_ANNOTATION.tsv','PATIENTS.tsv','NATIVE_TARGETS.tsv','CPTAC_NATIVE_Y.parquet']:verify(src/name,rec['files'][name])
        pat=pd.read_csv(src/'PATIENTS.tsv',sep='\t').rename(columns={'index':'patient_id'}).set_index('patient_id')
        pids=pat.index[pat.cohort.eq('CPTAC')];features,axes=feature_axes(pd.read_csv(src/'GENE_ANNOTATION.tsv',sep='\t'))
        x=pd.read_parquet(src/'GENE_LOGTPM.parquet',columns=features.gene_id.to_list()).reindex(pids);x.columns=features.feature_id
        x.to_parquet(d/'RNA.parquet');features.to_csv(d/'RNA_FEATURES.tsv',sep='\t',index=False)
        cv=joint/'covariates_r1'/c;cr=json.loads((cv/'COMPLETE.json').read_text());verify(cv/'COVARIATES.parquet',cr['files']['COVARIATES.parquet'])
        pd.read_parquet(cv/'COVARIATES.parquet').reindex(pids).to_parquet(d/'COVARIATES.parquet')
        meta=pd.read_csv(src/'NATIVE_TARGETS.tsv',sep='\t');meta=meta[meta.cohort.eq('CPTAC')].copy()
        y=pd.read_parquet(src/'CPTAC_NATIVE_Y.parquet',columns=meta.target_id.to_list()).reindex(pids)
        y.to_parquet(d/'Y.parquet');meta.to_csv(d/'TARGETS.tsv',sep='\t',index=False)
        splitpath=joint/'score_splits_r2'/c/'SPLITS.json';split=restrict_split(json.loads(splitpath.read_text()),pat.index.to_list(),pids)
        dump(d/'SPLITS.json',split)
        bindings.extend([dict(path=str(src/'COMPLETE.json'),sha256=sha(src/'COMPLETE.json')),dict(path=str(cv/'COMPLETE.json'),sha256=sha(cv/'COMPLETE.json')),dict(path=str(splitpath),sha256=sha(splitpath))])
        for e in [z for z in downloads['files'] if z['cancer']==c]:
            path=Path(e['path']);verify(path,e['sha256']);f=pd.read_csv(path,sep='\t',index_col=0);f.index=f.index.astype(str)
            if not f.index.is_unique or not f.columns.is_unique:raise ValueError('DUPLICATE_GENE_OR_PATIENT')
            mapping=patient_mapping(f.columns,pids,c);missing=sorted(set(f.columns)-set(mapping))
            f=f[list(mapping)].rename(columns=mapping).T.reindex(pids).astype(float)
            if np.isinf(f.to_numpy()).any():raise ValueError('INFINITE_ASSAY')
            if e['layer']=='MUTATION' and not set(np.unique(f.to_numpy()[np.isfinite(f.to_numpy())])).issubset({0.,1.}):raise ValueError('NON_BINARY_MUTATION')
            f.columns=[e['layer']+':'+x for x in f.columns];f.to_parquet(d/(e['layer']+'.parquet'))
            coverage.append(dict(cancer=c,layer=e['layer'],features=f.shape[1],patients=int(f.notna().any(axis=1).sum()),total=len(pids),measured_cells=int(f.notna().sum().sum()),unmatched_source_columns=';'.join(missing)))
            bindings.append(e)
        for layer,source in [('PROTEIN','protein'),('PHOSPHO','phosphorylation')]:
            rel=f'matrices/{c}/{source}_Tumor.parquet';verify(base/rel,assayrec['files'][rel]['sha256'])
            f=pd.read_parquet(base/rel).set_index('feature_id').T;f.index=f.index.str.removesuffix('::T');f=f.reindex(pids).astype(float)
            if not f.columns.is_unique or np.isinf(f.to_numpy()).any():raise ValueError('ASSAY_AXIS_INVALID')
            f.columns=[layer+':'+x for x in f.columns];f.to_parquet(d/(layer+'.parquet'))
            coverage.append(dict(cancer=c,layer=layer,features=f.shape[1],patients=int(f.notna().any(axis=1).sum()),total=len(pids),measured_cells=int(f.notna().sum().sum()),unmatched_source_columns=''))
            bindings.append(dict(path=str(base/rel),sha256=sha(base/rel)))
        lap,edges,na=network(authority,features,c);sparse.save_npz(d/'RNA_LAPLACIAN.npz',lap);edges.to_parquet(d/'NETWORK_EDGES.parquet',index=False);dump(d/'NETWORK_AUDIT.json',na)
        dump(d/'COMPLETE.json',dict(target_host='149',status='COMPLETE',files={p.name:sha(p) for p in d.iterdir() if p.is_file() and p.name!='COMPLETE.json'}))
        print('PREPARED',c,'network',na,'coverage',[e for e in coverage if e['cancer']==c],flush=True)
    pd.DataFrame(coverage).to_csv(out/'COVERAGE.tsv',sep='\t',index=False);dump(out/'BOUND_SOURCES.json',bindings+[dict(path=str(authority),sha256=sha(authority))])
    dump(out/'COMPLETE.json',dict(target_host='149',status='COMPLETE',files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'COMPLETE.json'}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--output',required=True);a=p.parse_args();prepare(a.work,a.output)
