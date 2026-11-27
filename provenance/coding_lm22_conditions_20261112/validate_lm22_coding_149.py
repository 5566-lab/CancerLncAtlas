import os
os.environ['OPENBLAS_NUM_THREADS']='2'
import json,socket,subprocess,sys
from pathlib import Path
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');WORK=ROOT/'kegg_cancer_redefinition_20261111_r1';REPO=ROOT/'source_integration/repo'
OUT=WORK/'measured_y_definition_r1/lm22_coding_r1';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(REPO))
from cla_engine.relation_evidence_bindings import file_sha
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
save(OUT/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'verify_coding_LM22_reference_and_deconvolution','lncrna_predictor_training':False,'lncrna_association':False})
r=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_coding_lm22.py'],cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(r.stdout,flush=True);save(OUT/'UNIT_TESTS.json',{'target_host':'149','exit_code':r.returncode,'output':r.stdout})
if r.returncode:raise SystemExit(r.returncode)
import numpy as np,pandas as pd
from cla_engine.coding_lm22 import coding_reference,relative_nusvr
sources=Path('/dsk2/user/dengsc/cla_state_signature_update_20261104_r1/sources')
lm=sources/'LM22_OFFICIAL_MATRIX.tsv';hg=sources/'hgnc_complete_set.txt';an=ROOT/'common_rna_scores_r1/COMMON_GENE_ANNOTATION.tsv'
assert file_sha(lm)=='2e625d2911a1336517dc0e02205a4e8ee8794af31cf1a7a5c597f84fdc185a33'
ref,audit=coding_reference(pd.read_csv(lm,sep='\t',index_col=0),pd.read_csv(an,sep='\t',keep_default_na=False),pd.read_csv(hg,sep='\t',keep_default_na=False))
ref.to_csv(OUT/'LM22_CODING_REFERENCE.tsv',sep='\t');audit.to_csv(OUT/'LM22_MEMBER_MAP.tsv',sep='\t',index=False)
cross=OUT/'r_crosscheck';cross.mkdir(exist_ok=True)
rng=np.random.default_rng(20261112);B=rng.lognormal(size=(150,4));mix=np.column_stack([B@p for p in [[.1,.2,.3,.4],[.4,.1,.1,.4],[.25,.25,.25,.25]]])
pd.DataFrame(B,index=['g'+str(i) for i in range(len(B))]).to_csv(cross/'REFERENCE.csv')
pd.DataFrame(mix,index=['g'+str(i) for i in range(len(B))]).to_csv(cross/'MIXTURES.csv')
r=subprocess.run(['/usr/bin/Rscript',str(WORK/'verify_lm22_r_reference.R'),str(cross)],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
(cross/'R_RUN.log').write_text(r.stdout)
if r.returncode:raise RuntimeError(r.stdout)
rr=pd.read_csv(cross/'R_FRACTIONS.csv',index_col=0).to_numpy(float)
py=np.stack([relative_nusvr(B,mix[:,i])[0] for i in range(mix.shape[1])])
delta=float(np.max(abs(rr-py)))
assert delta<1e-6,delta
report={'target_host':'149','signature_members':len(audit),'retained_reference_genes':len(ref),'cells':ref.shape[1],
    'member_decisions':audit.decision.value_counts().to_dict(),'reference_sha256':file_sha(OUT/'LM22_CODING_REFERENCE.tsv'),
    'reference_inputs':{str(p):file_sha(p) for p in [lm,hg,an]},'R_vs_python_max_absolute_error':delta,
    'e1071_version':(cross/'E1071_VERSION.txt').read_text().strip(),
    'method_source':'https://pmc.ncbi.nlm.nih.gov/articles/PMC4739640/',
    'biological_accuracy_validation':False,'published_CIBERSORT_numerical_equivalence':False}
save(OUT/'REFERENCE_VALIDATION.json',report);print(json.dumps(report,indent=2),flush=True)
