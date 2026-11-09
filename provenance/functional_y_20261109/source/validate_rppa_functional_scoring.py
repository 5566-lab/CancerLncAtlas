"""Check frozen references and actual saved RPPA score reconstruction."""
from pathlib import Path
import sys,json,socket
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from materialize_rppa_functional_evidence import fit_z,score_matrix,RAW,O
from cla_engine.common import sha
rng=np.random.default_rng(404);x=rng.normal(size=(31,9));train=np.arange(18);alter=x.copy();alter[18:]=rng.normal(100,20,size=(13,9));z,ref=fit_z(x,train);za,ra=fit_z(alter,train)
for k in ref:assert np.array_equal(ref[k],ra[k],equal_nan=True)
assert np.array_equal(z[:18],za[:18],equal_nan=True)
W=np.array([[1,0],[1,0],[1,0],[0,1],[0,-1],[0,1],[0,0],[0,0],[0,0]],float);s,n=score_matrix(z,W);ss,_=score_matrix(za,W);assert np.array_equal(s[:18],ss[:18],equal_nan=True)
missing=z.copy();missing[0,:2]=np.nan;ms,mn=score_matrix(missing,W);assert mn[0,0]==1 and np.isnan(ms[0,0]);assert np.isfinite(ms[0,1])
root=O/'BRCA/train_label_fold2';rec=json.loads((root/'SCORING_REFERENCE.json').read_text());saved=pd.read_parquet(root/'PROTEIN/PATIENT_S.parquet');patients=saved.index.tolist();protein=pd.read_parquet(RAW/'PROTEIN.parquet').set_index('patient_id').reindex(patients)[rec['protein_feature_ids']];si=np.array([patients.index(p) for p in rec['train_patients']]);z,fit=fit_z(protein.to_numpy(float),si)
for k in fit:assert np.allclose(fit[k],np.array(rec['protein_standardization'][k]),equal_nan=True,rtol=0,atol=1e-12)
groups=json.loads((O/'PROTEIN_GROUPS.json').read_text());pos={f:i for i,f in enumerate(protein.columns)};gz=np.column_stack([pd.DataFrame(z[:,[pos[f] for f in vv]]).mean(axis=1).to_numpy() for vv in groups.values()]);pred,count=score_matrix(gz,np.load(O/'PROTEIN_MEMBERSHIP.npy'));actual=saved.to_numpy();assert np.array_equal(np.isnan(pred),np.isnan(actual));err=float(np.nanmax(np.abs(pred-actual)));assert err<1e-12
phos=pd.read_parquet(RAW/'PHOSPHORYLATION.parquet').set_index('patient_id').reindex(patients);hf=rec['phospho_feature_ids'];res=np.full((len(patients),len(hf)),np.nan);key=list(groups);res_errors=[]
for j,r in enumerate(rec['phospho_regression']):
 t=gz[:,key.index(r['total_gene_group'])];v=phos[r['feature_id']].to_numpy(float);ix=si[np.isfinite(t[si])&np.isfinite(v[si])];assert len(ix)==r['n_train']
 if 'slope' in r:
  coef=np.linalg.lstsq(np.column_stack([np.ones(len(ix)),t[ix]]),v[ix],rcond=None)[0];assert np.allclose(coef,[r['intercept'],r['slope']],rtol=0,atol=1e-12);ok=np.isfinite(t)&np.isfinite(v);res[ok,j]=v[ok]-coef[0]-coef[1]*t[ok];res_errors.append(float(abs(np.mean(res[ix,j]))))
hz,hr=fit_z(res,si)
for k in hr:assert np.allclose(hr[k],np.array(rec['phospho_residual_standardization'][k]),equal_nan=True,rtol=0,atol=1e-12)
hm=pd.read_csv(O/'PHOSPHO_SIGNATURE_MEMBERS.tsv',sep='\t');ht=pd.read_csv(O/'PHOSPHO_TARGETS.tsv',sep='\t');ids=ht.source_readout_id.tolist();ww=np.zeros((len(hf),len(ids)))
for r in hm.itertuples():ww[hf.index(r.feature_id),ids.index(r.source_readout_id)]=r.weight
pred,count=score_matrix(hz,ww);actual=pd.read_parquet(root/'PHOSPHO/PATIENT_S.parquet').to_numpy();assert np.array_equal(np.isnan(pred),np.isnan(actual));herr=float(np.nanmax(np.abs(pred-actual)));assert herr<1e-12
split=json.loads((root/'SPLIT.json').read_text());assert not set(split['source_patients'])&set(split['label_patients']);assert split['source_patients']==rec['train_patients']
result={'target_host':'149','status':'PASS','train_reference_invariant_to_heldout_changes':True,'insufficient_components_stay_missing':True,'real_BRCA_protein_max_error':err,'real_BRCA_phospho_max_error':herr,'regression_train_residual_max_mean':max(res_errors),'source_label_patients_disjoint':True,'source_script_sha256':sha(R/'materialize_rppa_functional_evidence.py')};(R/'RPPA_SCORING_VALIDATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
