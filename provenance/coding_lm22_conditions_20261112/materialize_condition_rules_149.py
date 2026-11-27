import json,socket,subprocess,sys
from pathlib import Path
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');REPO=ROOT/'source_integration/repo'
BASE=ROOT/'kegg_cancer_redefinition_20261111_r1/measured_y_definition_r1';OUT=BASE/'condition_definitions_r1'
OUT.mkdir(exist_ok=True);sys.path.insert(0,str(REPO))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
save(OUT/'LAUNCH_PREFLIGHT.json',{'target_host':'149','operation':'fixed_pathway_context_requirements','training':False,'lncrna_association':False})
r=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_pathway_conditions.py'],cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
print(r.stdout,flush=True);save(OUT/'TEST_RESULTS.json',{'target_host':'149','exit_code':r.returncode,'output':r.stdout})
if r.returncode:raise SystemExit(r.returncode)
import pandas as pd
from cla_engine.pathway_conditions import condition_definition,EVIDENCE
from cla_engine.relation_evidence_bindings import file_sha
back=Path(json.loads((ROOT/'CANONICAL_TARGETS_CURRENT.json').read_text())['backbone'])
primary=pd.read_csv(back/'KEGG_CANCER_BACKBONE.tsv',sep='\t',keep_default_na=False)
scope=pd.read_csv(back/'SOURCE_SCOPE_DECISIONS.tsv',sep='\t',keep_default_na=False).set_index('source_target_id')
source=scope.loc[primary.source_target_id].copy()
assert len(source)==477 and source.canonical_id.is_unique
rules=pd.DataFrame([condition_definition(row) for row in source.to_dict('records')])
assert len(rules)==477 and rules.condition_required.sum()==143
rules.to_csv(OUT/'PATHWAY_CONDITIONS.tsv',sep='\t',index=False)
save(OUT/'OBSERVATION_SCHEMA.json',{'columns':['patient_id','network_id','status','evidence_type','source_id'],
    'status_values':['present','absent'],'missing':'no qualified observation; retained as unknown',
    'conflicting_observations':'unknown; source records retained','accepted_evidence_types_by_condition':EVIDENCE,
    'generic_gene_mutation_or_host_expression_is_not_condition_evidence':True,
    'reference_context_does_not_assert_pathway_activation':True})
report={'target_host':'149','primary_pathways':477,'unconditional_reference_contexts':334,'conditional_contexts':143,
    'condition_kind_counts':rules.condition_kind.value_counts().to_dict(),
    'source_classification_corrections':rules[rules.curation_note.ne('')].to_dict('records'),
    'clinical_evidence_status':'Current aligned covariates do not include qualified event-specific assays; gene-level mutation flags and signatures are not substituted.',
    'patient_condition_measurements_materialized':False,
    'source_hashes':{str(back/name):file_sha(back/name) for name in ['KEGG_CANCER_BACKBONE.tsv','SOURCE_SCOPE_DECISIONS.tsv']},
    'files':{p.name:file_sha(p) for p in OUT.iterdir() if p.is_file()}}
save(OUT/'COMPLETE.json',report);print(json.dumps(report,indent=2),flush=True)
