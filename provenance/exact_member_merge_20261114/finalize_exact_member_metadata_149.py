from pathlib import Path
import json, hashlib, socket
import pandas as pd

assert socket.gethostname().split('.')[0] == '149'
R = Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
O = R / 'kegg_cancer_redefinition_20261111_r1/exact_member_merge_20261114_r1'
N = R / 'canonical_targets_r10_exact_members_20261114_r1'
G = O / 'observation_registry_r3'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')

save(O / 'METADATA_PREFLIGHT.json', {'target_host': '149', 'operation': 'finalize grouped catalog counts and alias redirect'})
red = pd.read_csv(N / 'ID_REDIRECTS.tsv', sep='\t', keep_default_na=False)
drop, keep = 'CLA-P-549730F42F0B148D', 'CLA-P-08A1A41F05A3E9ED'
existing = red.loc[red.previous_internal_id.eq(drop)]
if existing.empty:
    red = pd.concat([red, pd.DataFrame([dict(previous_internal_id=drop, canonical_id=keep, relation='same_observation_group', evidence='user_authorized_identical_protein_members_J1')])], ignore_index=True)
    red.to_csv(N / 'ID_REDIRECTS.tsv', sep='\t', index=False)
else:
    assert existing.canonical_id.eq(keep).all()
pol = pd.read_csv(N / 'TARGET_POLICY.tsv', sep='\t', keep_default_na=False)
official = int((pol.analysis_role.eq('primary_pathway') & pol.official_cancer_supported.astype(str).str.lower().eq('true')).sum())
rec = json.loads((N / 'COMPLETE.json').read_text())
rec['source_backbone_official_cancer_supported'] = 199
rec['backbone_official_cancer_supported'] = official
rec['files'] = {p.name: sha(p) for p in N.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}
save(N / 'COMPLETE.json', rec)
collection = json.loads((G / 'COMPLETE.json').read_text())
cases = []
for cancer in ['BRCA', 'LUAD', 'COAD']:
    case = G / cancer
    refs = json.loads((case / 'INPUT_REFERENCES.json').read_text())
    refs['group_catalog']['sha256'] = sha(N / 'COMPLETE.json')
    save(case / 'INPUT_REFERENCES.json', refs)
    receipt = json.loads((case / 'COMPLETE.json').read_text())
    receipt['files'] = {p.name: sha(p) for p in case.iterdir() if p.is_file() and p.name != 'COMPLETE.json'}
    save(case / 'COMPLETE.json', receipt)
    cases.append(receipt)
collection.update(cases=cases, receipt_hashes={c: sha(G / c / 'COMPLETE.json') for c in ['BRCA', 'LUAD', 'COAD']})
save(G / 'COMPLETE.json', collection)
definition = json.loads((O / 'FIXED_Y_DEFINITION.json').read_text())
definition['registry_receipt_sha256'] = sha(G / 'COMPLETE.json')
save(O / 'FIXED_Y_DEFINITION.json', definition)
save(O / 'FINAL_METADATA.json', dict(target_host='149', primary_pathways=476, official_source_definitions=199, official_groups=official, redirect={drop: keep}, catalog_sha256=sha(N / 'COMPLETE.json')))
print('FINAL_METADATA_VERIFIED', official, 'official groups')
