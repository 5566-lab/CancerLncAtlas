from pathlib import Path
import socket,json,hashlib
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;O=R/'reports/story_revision_20261110_r1';assert json.loads((O/'LAUNCH_PREFLIGHT.json').read_text())['target_host']=='149'
y=pd.read_csv(R/'supervision_summary_r2/Y_SUPERVISION_BY_CANCER.tsv',sep='\t')
m=pd.read_csv(R/'canonical_targets_r6/TARGET_MASTER.tsv',sep='\t')
rows=[]
for c in y.loc[y.strong_positive.gt(0),'cancer_id']:
 rows.append(pd.read_parquet(R/'functional_relation_y_catalog_r6'/c/'sealed_test/POSITIVE_RELATIONS.parquet',filters=[('membership','==','strong_positive')]))
d=pd.concat(rows).merge(m[['canonical_id','display_name','roles']],on='canonical_id')
assert len(d)==495 and d.display_name.eq('Tumor cell fraction').sum()==476
assert d.loc[d.roles.eq('Pathway'),'canonical_id'].nunique()==17
d.to_csv(O/'STRICT_MULTILAYER_EXAMPLES.tsv',sep='\t',index=False)
e=d[(d.cancer_id=='BRCA')&d.display_name.eq('MTORC1 SIGNALING')].iloc[0].to_dict();(O/'REAL_EXAMPLE.json').write_text(json.dumps(e,indent=2))
rows=[]
for layer in ['RNA','PROTEIN']:
 p=R/'numerical_relation_evidence_r1/BRCA/sealed_test'/layer/'SUPPORTED_RELATIONS.parquet'
 a=pd.read_parquet(p,filters=[('lncrna_id','==',e['lncrna_id']),('canonical_id','==',e['canonical_id'])]);assert len(a)==1
 rows.append(a)
f=pd.concat(rows);f.to_csv(O/'REAL_EXAMPLE_EVIDENCE.tsv',sep='\t',index=False)
assert f.statistical_support.eq('strict').all() and f.rho.lt(0).all()
print('EVIDENCE_VERIFIED',len(d),f[['evidence_layer','rho','q','n']].to_dict('records'))
