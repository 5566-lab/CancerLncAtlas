import json, socket, sys
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
BASE=ROOT/'kegg_cancer_redefinition_20261111_r1/measured_y_definition_r1'
assert socket.gethostname().split('.')[0]=='149'
sys.path.insert(0,str(ROOT/'source_integration/repo'))
from cla_engine.relation_evidence_bindings import file_sha

out=BASE/'measurement_coverage_r3'
out.mkdir(exist_ok=True)
(out/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'count measured readouts by structural pathway binding','training':False,'lncrna_association':False,'time_utc':datetime.now(timezone.utc).isoformat()}))
import numpy as np
import pandas as pd

detail=[];summary=[];receipts={}
for cancer in ('BRCA','LUAD','COAD'):
    d=BASE/'measurements_coding_r3'/cancer/'validation'
    rec=json.loads((d/'COMPLETE.json').read_text())
    assert rec['target_host']=='149'
    for name,digest in rec['files'].items():assert file_sha(d/name)==digest
    receipts[cancer]=file_sha(d/'COMPLETE.json')
    targets=pd.read_csv(d/'TARGETS.tsv',sep='\t',keep_default_na=False)
    bindings=pd.read_csv(d/'READOUT_BINDINGS.tsv',sep='\t',keep_default_na=False)
    primary=set(targets.loc[targets.analysis_role.eq('primary_pathway'),'canonical_id'])
    assert len(primary)==477
    for cohort in ('TCGA','CPTAC'):
        b=bindings[bindings.cohort.eq(cohort)&bindings.output_canonical_id.isin(primary)]
        masks={p.parent.name:pd.read_parquet(p) for p in (d/cohort).glob('*/OBSERVED.parquet')}
        index=next(iter(masks.values())).index
        for mask in masks.values():assert mask.index.equals(index)
        for layer in ('RNA','STATE','PROTEIN','PHOSPHO','METHYLATION','ALL'):
            selected=b if layer=='ALL' else b[b.layer.eq(layer)]
            rows=[]
            for cid in sorted(primary):
                group=selected[selected.output_canonical_id.eq(cid)]
                record={'cancer':cancer,'cohort':cohort,'layer':layer,'canonical_id':cid,'patients':len(index),'bound_readouts':len(group)}
                for scope,roles in [('identity',{'equivalent_pathway_readout'}),('member',{'molecular_component_readout'}),('any',{'equivalent_pathway_readout','molecular_component_readout'})]:
                    vectors=[masks[row.layer][row.target_id].to_numpy(bool) for row in group.itertuples() if row.evidence_role in roles]
                    measured=np.logical_or.reduce(vectors) if vectors else np.zeros(len(index),dtype=bool)
                    record['patients_with_'+scope+'_measurement']=int(measured.sum())
                rows.append(record)
            detail.extend(rows)
            summary.append({'cancer':cancer,'cohort':cohort,'layer':layer,'patients':len(index),'fixed_primary_pathways':477,
                **{'pathways_with_'+scope+'_measurement':sum(row['patients_with_'+scope+'_measurement']>0 for row in rows) for scope in ('identity','member','any')}})

pd.DataFrame(detail).to_csv(out/'PATHWAY_MEASURED_COVERAGE.tsv',sep='\t',index=False)
pd.DataFrame(summary).to_csv(out/'SUMMARY.tsv',sep='\t',index=False)
(out/'COMPLETE.json').write_text(json.dumps({'target_host':'149','definition':'At least one finite patient readout bound by identity or explicit molecular membership; this is not an activation call or lncRNA association.','input_receipts':receipts,'files':{p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='COMPLETE.json'}},indent=2))
print(pd.DataFrame(summary).to_string(index=False),flush=True)
