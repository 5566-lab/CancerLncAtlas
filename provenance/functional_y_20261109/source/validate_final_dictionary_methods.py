from pathlib import Path
import socket,json,time,sys,hashlib
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;L=R/'project_log';D=R/'reader_dictionary_r4';receipt=L/'DICTIONARY_V4_RECEIPT.json';deadline=time.monotonic()+7200
while not receipt.exists():
 if time.monotonic()>deadline:raise TimeoutError('Dictionary receipt')
 time.sleep(10)
record=json.loads(receipt.read_text());path=Path(record['file']);digest=hashlib.sha256(path.read_bytes()).hexdigest();assert digest==record['sha256']
sys.path.insert(0,'/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps')
from openpyxl import load_workbook
wb=load_workbook(path,read_only=True,data_only=True)
def table(name):
 it=wb[name].iter_rows(values_only=True);return pd.DataFrame(it,columns=next(it)).fillna('')
checks=[]
for name,file in [('Target_master',D/'TARGET_MASTER.tsv'),('Structural_contexts',D/'STRUCTURAL_CONTEXTS.tsv'),('State_scores',D/'STATE_SCORE_DEFINITIONS.tsv'),('Score_definitions',D/'OBSERVED_SCORE_DEFINITIONS.tsv'),('CPTAC_state_methods',R/'cptac_method_index_r1/CPTAC_STATE_METHODS.tsv'),('CPTAC_covariates',R/'cptac_method_index_r1/CPTAC_COVARIATE_COVERAGE.tsv')]:
 actual=table(name);expected=pd.read_csv(file,sep='\t').fillna('');pd.testing.assert_frame_equal(actual,expected,check_dtype=False);checks.append({'sheet':name,'rows':len(actual),'matches_source':str(file)})
coverage=table('Score_coverage');definitions=table('Score_definitions');cptac=table('CPTAC_state_methods');wb.close()
joined=coverage.merge(definitions[['readout_id','method','reference','score_unit']],on='readout_id',how='left',validate='many_to_one')
joined=joined.loc[joined.cohort.str.startswith('CPTAC:')]
cols=['canonical_id','source_readout_id','cohort','cancer_id','measurement_layer','method','reference','score_unit'];keys=['cohort','source_readout_id']
expected=cptac[cols].sort_values(keys).reset_index(drop=True)
state_keys=pd.MultiIndex.from_frame(cptac[keys])
state_rows=joined.loc[pd.MultiIndex.from_frame(joined[keys]).isin(state_keys)]
actual=state_rows.merge(cptac[keys],on=keys,how='inner',validate='one_to_one')[cols].sort_values(keys).reset_index(drop=True)
assert len(actual)==4130;pd.testing.assert_frame_equal(actual,expected,check_dtype=False)
proof={'target_host':'149','status':'PASS','dictionary_sha256':digest,'workbook_csv_consistency':checks,'cptac_readout_methods_matched':len(actual),'checks':['CPTAC score definitions, references and units match each cohort method record','Reader CSV tables match corresponding workbook sheets']}
(L/'FINAL_DICTIONARY_METHODS_VALIDATION.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
