from pathlib import Path
import sys,socket,json,zipfile,hashlib
import pandas as pd
R=Path(__file__).resolve().parent;F=R/'functional_relation_y_fused_r1';D=F/'dictionary_data';O=F/'reports';assert socket.gethostname()=='149';sys.path[:0]=[str(R/'python_deps'),'/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps']
sys.path.insert(0,str(R/'source_integration/repo'))
from openpyxl import load_workbook
rec=json.loads((O/'DICTIONARY_V5_RECEIPT.json').read_text());path=Path(rec['file']);assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256']
forbidden=[b'MOGSA',b'PathFA',b'identity_merge',b'independent_flux_claimed',b'independent_functional_branch_claimed',b'legacy_compatibility',b'LEGACY_WEIGHTED_ZSCORE_UNCHANGED',b'legacy_R07']
with zipfile.ZipFile(path) as z:
 assert z.testzip() is None
 for name in z.namelist():
  if name.endswith('.xml'):
   with z.open(name) as fp:
    tail=b''
    while True:
     block=fp.read(1024*1024)
     if not block:break
     assert not any(x in tail+block for x in forbidden),name
     tail=block[-200:]
wb=load_workbook(path,read_only=True,data_only=True);assert set(wb.sheetnames)==set(rec['sheets'])
def frame(name):
 it=wb[name].values;header=next(it);return pd.DataFrame(it,columns=header)
master=frame('Target_master');assert len(master)==3456 and master.canonical_id.is_unique
assert master.roles.str.contains('Pathway').sum()==2595 and master.roles.str.contains('State').sum()==864
for sheet,fn in [('Y_by_cancer','Y_BY_CANCER.tsv'),('Y_by_context','Y_BY_CONTEXT.tsv'),('Y_target_counts','Y_TARGET_COUNTS.tsv')]:
 a=frame(sheet);b=pd.read_csv(D/fn,sep='\t').fillna('');pd.testing.assert_frame_equal(a.fillna('').reset_index(drop=True),b.reset_index(drop=True),check_dtype=False);assert set(a.cancer_id)=={'BRCA','LUAD','COAD'}
assert len(frame('Y_by_context'))==15
from cla_engine.functional_fusion import ARRAY_NAMES
assert set(frame('Y_fields').field)==set(ARRAY_NAMES)
sd=frame('Score_definitions');cv=frame('Score_coverage');assert len(sd)==len(cv)==44859
expected_sd=pd.read_csv(D/'SCORE_DEFINITIONS.tsv',sep='\t').fillna('');pd.testing.assert_frame_equal(sd.fillna('').reset_index(drop=True),expected_sd.reset_index(drop=True),check_dtype=False)
for col in ['applied_method','definition_database','source_resource','score_matrix','score_unit','data_form','measurement_layer']:assert sd[col].fillna('').astype(str).str.len().gt(0).all(),col
p=sd.loc[sd.dataset.eq('CPTAC')&sd.source_readout_id.eq('ABSOLUTE_purity')];assert len(p)==3 and p.applied_method.str.contains('WES-derived').all() and ~p.source_resource.str.contains('api.gdc.cancer.gov').any()
assert set(sd.cancer_id)=={'BRCA','LUAD','COAD'}
for name in ['Parent_child','Pathway_branches','Reaction_extensions','Reaction_roles','Complex_components','State_components','Phospho_sites']:
 assert next(wb[name].values) and rec['sheets'][name]>0
wb.close();result={'target_host':'149','status':'PASS','dictionary_sha256':rec['sha256'],'sheets':len(rec['sheets']),'scored_identities':3456,'source_bindings':3656,'three_cancer_contexts':15,'source_readout_rows':44859,'checks':['Workbook zip integrity','Required identity/composition/mapping sheets','Exact Y table equality with frozen results','Three-cancer scope','CPTAC WES purity source and units','No internal execution or unused-method flags']};(O/'DICTIONARY_VALIDATION.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
