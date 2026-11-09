from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ProcessPoolExecutor,as_completed
import socket,json,hashlib,gzip,re,os,io
import pandas as pd,numpy as np
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'full_lnc_local_cnv_r1';BASE=Path('/dell_2/DSC/CancerLncAtlas')
OLD=BASE/'results/v32_directional_segment_cnv_20260902_r1'
GTF=Path('/public8/DSC/CancerLncAtlas/input/v3_1_target_context_assets_e0b44122/gencode/gencode.v36.annotation.gtf.gz')
IV=BASE/'inputs/v32_routed_candidate_20260829_r1/grch38_gene_lncrna_intervals.parquet'
AUTH=BASE/'inputs/v32_routed_candidate_20260829_r1/patient_fold_authority_20260829_r1/SAMPLE_PATIENT_FOLD_MAP.tsv'
ANN=Path('/dsk2/user/dengsc/cla_m0_lasso_20261104_r1/tcga_full_tpm_r1/BRCA/GENE_ANNOTATION.tsv')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def specs(intervals,ids):
 ix={v:i for i,v in enumerate(ids)};out={}
 for c,d in intervals.groupby('chromosome',sort=False):
  take=d.entity_id.map(ix);ok=take.notna();d=d.loc[ok];out[str(c).removeprefix('chr')]=(take[ok].to_numpy(int),((d.start.to_numpy(np.int64)+d.end.to_numpy(np.int64))//2))
 return out
def map_array(seg,iv,size):
 total=np.zeros(size,float);count=np.zeros(size,np.int32)
 seg=seg.copy();seg.columns=[str(c).lower() for c in seg.columns];seg['chromosome']=seg.chromosome.astype(str).str.replace(r'^chr','',regex=True)
 for c,(dest,mid) in iv.items():
  d=seg.loc[seg.chromosome.eq(c)].sort_values('start',kind='stable')
  if d.empty:continue
  starts=d.start.to_numpy(np.int64);ends=d.end.to_numpy(np.int64);v=pd.to_numeric(d.segment_mean,errors='coerce').to_numpy(float);ix=np.searchsorted(starts,mid,side='right')-1;safe=np.maximum(ix,0);good=(ix>=0)&(mid<=ends[safe])&np.isfinite(v[safe]);np.add.at(total,dest[good],v[ix[good]]);np.add.at(count,dest[good],1)
 call=count>0;value=np.full(size,np.nan,np.float32);value[call]=(total[call]/count[call]).astype(np.float32);return value,call
def worker(cancer):
 assert socket.gethostname()=='149';out=O/cancer;out.mkdir(exist_ok=False)
 auth=pd.read_csv(AUTH,sep='\t');patients=auth.loc[auth.cancer_id.eq(cancer),'patient_id'].tolist();ids=json.loads((O/'lncrna_ids.json').read_text());v36=pd.read_parquet(O/'GENCODE_V36_LNCRNA_INTERVALS.parquet');newiv=specs(v36,ids)
 prior=OLD/('cancer='+cancer);oldids=json.loads((prior/'lncrna_ids.json').read_text());oldpatients=json.loads((prior/'patient_ids.json').read_text());oldindex={p:i for i,p in enumerate(oldpatients)}
 oldiv=specs(pd.read_parquet(IV),oldids);oldcn=np.load(prior/'lncrna_signed_mean.npy',mmap_mode='r');oldcall=np.load(prior/'lncrna_callable.npy',mmap_mode='r')
 source_records=json.loads((prior/'SOURCE_FILES.json').read_text())['records'];sources={z['patient_id']:Path(z['path']) for z in source_records};assert len(sources)==len(source_records)
 cn=np.lib.format.open_memmap(out/'lncrna_signed_mean.npy',mode='w+',dtype=np.float32,shape=(len(patients),len(ids)));cn[:]=np.nan
 call=np.lib.format.open_memmap(out/'lncrna_callable.npy',mode='w+',dtype=np.uint8,shape=cn.shape);call[:]=0
 shared=[(j,ids.index(x)) for j,x in enumerate(oldids) if x in ids];a=np.array([x[0] for x in shared]);b=np.array([x[1] for x in shared]);oldreproduction=0;coordinate_delta_cells=0;coordinate_call_delta=0;receipts=[];missing=[]
 for i,patient in enumerate(patients):
  f=sources.get(patient)
  if f is None:
   missing.append(patient)
   if patient in oldindex:assert not oldcall[oldindex[patient]].any(), 'Missing source but callable old row'
   continue
  raw=f.read_bytes();seg=pd.read_csv(io.BytesIO(raw),sep='\t');value,available=map_array(seg,newiv,len(ids));cn[i]=value;call[i]=available
  oi=oldindex[patient];ov,oc=map_array(seg,oldiv,len(oldids))
  assert np.array_equal(oc,oldcall[oi].astype(bool)), 'Historical callability mismatch '+cancer+' '+patient
  assert np.array_equal(ov,oldcn[oi],equal_nan=True), 'Historical signed values mismatch '+cancer+' '+patient
  oldreproduction+=1
  valid=oc[a]&available[b];coordinate_delta_cells+=int(np.sum(ov[a][valid]!=value[b][valid]));coordinate_call_delta+=int(np.sum(oc[a]!=available[b]))
  receipts.append({'patient_id':patient,'path':str(f),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
  if (i+1)%200==0:print('CNV_PROGRESS',cancer,i+1,len(patients),flush=True)
 cn.flush();call.flush();(out/'patient_ids.json').write_text(json.dumps(patients));(out/'lncrna_ids.json').write_text(json.dumps(ids));(out/'SOURCE_FILES.json').write_text(json.dumps({'records':receipts},indent=2))
 result={'target_host':'149','status':'COMPLETE','cancer_id':cancer,'patients':len(patients),'lncrnas':len(ids),'annotation':'GENCODE v36 GRCh38, same RNA definition','mapping':'Signed segment mean at gene interval midpoint; average multiple intervals','missing_patients':missing,'patients_exactly_reproducing_old_coordinate_store':oldreproduction,'historical_lncrnas':len(oldids),'shared_lncrnas':len(shared),'coordinate_update_signed_value_differences':coordinate_delta_cells,'coordinate_update_callability_differences':coordinate_call_delta,'files':{x.name:{'sha256':sha(x),'bytes':x.stat().st_size} for x in out.iterdir() if x.is_file()}}
 (out/'COMPLETE.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 O.mkdir(exist_ok=True);assert not (O/'COMPLETE.json').exists()
 assert sha(GTF)=='81359865bab048361cbeaf91c9441aac4c700b63978e6691d579d78bf2322d30'
 ann=pd.read_csv(ANN,sep='\t');ids=['LNC:'+x for x in ann.loc[ann.gene_type.eq('lncRNA'),'gene_id']];wanted=set(ids);assert len(ids)==16889
 rows=[]
 with gzip.open(GTF,'rt') as f:
  for line in f:
   if line.startswith('#'):continue
   z=line.rstrip().split('\t')
   if len(z)!=9 or z[2]!='gene':continue
   attrs=dict(re.findall(r'(\w+) "([^"]*)"',z[8]));gid=attrs.get('gene_id','')
   if '_PAR_Y' in gid:continue
   core='LNC:'+gid.split('.')[0]
   if core in wanted:rows.append({'entity_id':core,'chromosome':z[0].removeprefix('chr'),'start':int(z[3]),'end':int(z[4])})
 iv=pd.DataFrame(rows).drop_duplicates();assert set(iv.entity_id)==wanted;assert not iv.entity_id.duplicated().any();iv.to_parquet(O/'GENCODE_V36_LNCRNA_INTERVALS.parquet',index=False)
 (O/'lncrna_ids.json').write_text(json.dumps(ids));auth=pd.read_csv(AUTH,sep='\t');cancers=sorted(auth.cancer_id.unique())
 pre={'target_host':'149','action':'EXPAND_LOCAL_CNV_ALL_RNA_LNCRNAS','training_started':False,'cancers':cancers,'new_annotation_sha256':sha(GTF),'old_interval_source_sha256':sha(IV),'rna_annotation_sha256':sha(ANN),'patient_map_sha256':sha(AUTH),'old_mapping_algorithm_reproduced_per_patient':True}
 (O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps(pre,indent=2));records={};failures={}
 with ProcessPoolExecutor(max_workers=4) as ex:
  futs={ex.submit(worker,c):c for c in cancers}
  for f in as_completed(futs):
   c=futs[f]
   try:records[c]=f.result();print('COMPLETE_CNV',c,records[c]['patients'],flush=True)
   except Exception as e:failures[c]=repr(e);print('CNV_FAILED',c,repr(e),flush=True)
   (O/'PROGRESS.json').write_text(json.dumps({'target_host':'149','completed':sorted(records),'failures':failures},indent=2))
 result={'target_host':'149','status':'COMPLETE' if not failures else 'INCOMPLETE','records':records,'failures':failures,'created_utc':datetime.now(timezone.utc).isoformat()};(O/('COMPLETE.json' if not failures else 'INCOMPLETE.json')).write_text(json.dumps(result,indent=2));print('FINISHED',len(records),failures,flush=True)
