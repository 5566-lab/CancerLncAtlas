from pathlib import Path
import sys,json,socket,hashlib,argparse,time,gc
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;F=R/'functional_relation_y_fused_r1';assert socket.gethostname()=='149';sys.path.insert(0,str(R/'source_integration/repo'))
from cla_engine.functional_y import load_functional_context
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(c,context,independent=False):
 start=time.monotonic();base=F/c/context;d=load_functional_context(F,c,context);assert len(d.lncrna_ids)==16889 and len(d.canonical_ids)==3456
 result={'target_host':'149','cancer_id':c,'context':context,'status':'PASS','counts':d.counts(),'complete_sha256':sha(base/'COMPLETE.json'),'y_definition_sha256':sha(F/'Y_DEFINITION.json'),'independent_p_checks':[]}
 if independent:
  wanted=['CLA-P-3423B99760CD551B','CLA-P-366F48557175A084','CLA-P-97C440D7FBA8755B','CLA-S-0342E8CA85E06419'];md=pd.read_csv(base/'READOUT_INDEX.tsv',sep='\t');refs={x['canonical_id']:x for x in json.loads((base/'DEPENDENCE_REFERENCES.json').read_text())};sources={}
  relevant=md.loc[md.canonical_id.isin(wanted)]
  for directory,frame in relevant.groupby('directory'):
   cols=sorted(set(frame.column_index));path=Path(directory)/'ASSOCIATIONS.npz'
   with np.load(path,allow_pickle=False) as z:small={name:z[name][:,cols] for name in ['p','r','available']}
   sources[directory]=(cols,small)
  for cid in wanted:
   if cid not in refs:continue
   frame=md.loc[md.canonical_id.eq(cid)];vectors=[];availability=[]
   for row in frame.itertuples():
    cols,st=sources[row.directory];j=cols.index(row.column_index);vectors.append(st['p'][:,j]);availability.append(st['available'][:,j]&np.isfinite(st['p'][:,j])&np.isfinite(st['r'][:,j]))
   p=np.column_stack(vectors);av=np.column_stack(availability);ref=refs[cid];corr=np.asarray(ref['label_residual_gram_correlation']);recs={tuple(x['indices']):x for x in ref['calibration_references']};expected=np.full(len(p),np.nan);methods=np.zeros(len(p),np.uint8)
   patterns,labels=np.unique(av,axis=0,return_inverse=True)
   for a,pattern in enumerate(patterns):
    rows=np.flatnonzero(labels==a);ix=np.flatnonzero(pattern)
    if len(ix)==0:continue
    if len(ix)==1:expected[rows]=p[rows,ix[0]];methods[rows]=1;continue
    sub=corr[np.ix_(ix,ix)];weights=np.array([1/sum(j in g for j in ix) for i in ix for g in ref['duplicate_groups'] if i in g])
    if not np.isfinite(sub).all():expected[rows]=np.minimum(1,len(ix)*p[np.ix_(rows,ix)].min(1));methods[rows]=3;continue
    if np.all(np.abs(sub)>=1-1e-12):expected[rows]=np.exp(np.log(np.clip(p[np.ix_(rows,ix)],np.finfo(float).tiny,1))@weights/weights.sum());methods[rows]=4;continue
    rr=recs[tuple(ix)];key=rr['null_cache_key'];info=json.loads((base/'null_references'/(key+'.json')).read_text());assert hashlib.sha256(json.dumps(info,sort_keys=True).encode()).hexdigest()==key
    null=np.load(base/'null_references'/(key+'.npy'));assert len(null)==100000 and np.all(np.diff(null)>=0);assert np.array_equal(weights,np.asarray(info['weights']))
    stat=np.sum(-2*np.log(np.clip(p[np.ix_(rows,ix)],np.finfo(float).tiny,1))*weights,axis=1)
    from bisect import bisect_left
    expected[rows]=[(len(null)-bisect_left(null,float(t))+1)/(len(null)+1) for t in stat];methods[rows]=2
   col=d.canonical_ids.index(cid);actual=d.arrays['combined_p'][:,col];assert np.array_equal(np.isfinite(actual),np.isfinite(expected));diff=float(np.nanmax(abs(actual-expected))) if np.isfinite(expected).any() else 0.;assert diff<1e-12,(c,cid,diff);assert np.array_equal(methods,d.arrays['calibration_method'][:,col]);result['independent_p_checks'].append({'canonical_id':cid,'finite_pairs':int(np.isfinite(expected).sum()),'maximum_absolute_error':diff})
 result['seconds']=time.monotonic()-start;out=F/'validation';out.mkdir(exist_ok=True);(out/(c+'__'+context+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--cancer',required=True);ap.add_argument('--context',required=True);ap.add_argument('--independent',action='store_true');args=ap.parse_args();run(args.cancer,args.context,args.independent)
