from pathlib import Path
import socket,json,pandas as pd
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'fixed_y_remake_20261114_r1';F=O/'figure_data';N=R/'canonical_targets_r11_state_groups_20261114_r1'
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
examples=json.loads((F/'EXAMPLES.json').read_text());align=rd(N/'BACKBONE_ALIGNMENT.tsv')
for key in ['WNT','PI3K']:
    rows=align[align.target_canonical_id.eq(examples[key]['canonical_id'])]
    if key=='WNT':rows=rows[rows.source_name.str.contains('Wnt|beta.catenin',case=False)]
    else:rows=rows[rows.source_name.str.contains('PI3K|AKT',case=False)]
    examples[key]['context']=json.loads(rows.sort_values(['jaccard','shared_proteins'],ascending=False).head(5).to_json(orient='records'))
(F/'EXAMPLES.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
summary=rd(O/'Y_bundle/COHORT_SUMMARY.tsv');print(summary.to_string(index=False))
coverage=rd(O/'Y_bundle/TARGET_COVERAGE.tsv');zero=coverage.groupby(['canonical_id','name']).patients_measured.max();print('NO_MEASURED_TARGETS',zero[zero.eq(0)].to_dict())
for key in ['MMR','Hedgehog']:
    e=examples[key];print('EXAMPLE',key,[(s,len(v)) for s,v in e['source_lists'].items()])
for key in ['PI3K','cell_composition','sbs_spectrum']:
    for co in ['TCGA','CPTAC']:
        meta=rd(F/(key+'_'+co+'_BRCA_observations.tsv'))
        if key=='PI3K':meta=meta[meta.kind.eq('member_measurement') & meta.layer.isin(['PROTEIN','PHOSPHO'])]
        elif key=='cell_composition':meta=meta[meta.source_target_id.str.contains('CD8|Cytotoxic',case=False)]
        print('PLOT_METADATA',key,co,meta[['layer','feature_id','source_target_id','patients_measured','definition_json']].head(12).to_json(orient='records'))
