import json,shutil,socket,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone

assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
WORK=ROOT/'kegg_cancer_redefinition_20261111_r1'
BASE=WORK/'measured_y_definition_r1';REPO=ROOT/'source_integration/repo'
sys.path.insert(0,str(REPO))
from cla_engine.relation_evidence_bindings import file_sha

def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
save(BASE/'CODING_R3_FINALIZE_PREFLIGHT.json',{'target_host':'149','operation':'publish measured definitions and corrected documentation','training':False,'lncrna_association':False,'time_utc':datetime.now(timezone.utc).isoformat()})
tests=json.loads((BASE/'CODING_R3_INTEGRATION_TESTS.json').read_text());assert tests['exit_code']==0
dic=json.loads((BASE/'CODING_R3_DICTIONARY_UPDATE.json').read_text());assert file_sha(dic['dictionary'])==dic['sha256']
integration=json.loads((BASE/'CODING_R3_MEASUREMENT_INTEGRATION.json').read_text())
assert len(integration['cases'])==3
lm=json.loads((BASE/'lm22_coding_r1/MATERIALIZATION_PROGRESS.json').read_text())
assert len(lm['completed'])==6 and not lm['errors']
coverage=BASE/'measurement_coverage_r3'
crec=json.loads((coverage/'COMPLETE.json').read_text())
for name,digest in crec['files'].items():assert file_sha(coverage/name)==digest
for name,key in [('CANONICAL_TARGETS_CURRENT.json','reader_dictionary_sha256'),('CURRENT_Y_DELIVERY.json','dictionary_sha256')]:
    ptr=json.loads((ROOT/name).read_text());assert ptr[key]==dic['sha256']
    assert not ptr.get('accepted_Y') and not ptr.get('functional_y')
reader=json.loads((ROOT/'CURRENT_READER_DELIVERY.json').read_text())
assert reader['dictionary_sha256']==dic['sha256']
assert reader['verified_files'][Path(dic['dictionary']).name]==dic['sha256']

readme=REPO/'README.md';text=readme.read_text()
start=text.index('The main labels are cancer × lncRNA × exact-pathway/State association labels,')
end=text.index('The TCGA protein/phosphorylation input is the hash-pinned official TCPA',start)
replacement='''The biological output axis contains 477 exact KEGG mechanisms and 197 State
identities. Patient measurements are linked to this axis by verified pathway
identity or explicit molecular membership. Source-specific readouts, protein
members and modification sites retain their provenance and observed masks.
lncRNA correlations, their significance and their RMS are not the Y values.

TF, kinase and PROGENy readouts belong to Pathway. They describe regulatory
programs or member states; adding an independent supervised loss requires an
explicit task definition. Display category and database record count do not
set loss weights.

Recomputed RNA readouts use protein-coding gene IDs before symbol aggregation.
The fixed LM22 coding reference contains 517 genes and supplies 22 relative
cell-composition readouts per patient, with its own method version. Published
TLS12 and RNA-derived purity remain separately identified source readouts.
See [measurement definitions](docs/MEASURED_READOUT_DEFINITIONS.rst) and
[the patient-input M0 specification](docs/M0_RELATION_BASELINE.rst).

'''
text=text[:start]+replacement+text[end:]
needle='`prepare` creates patient-disjoint input/label views and fits the matched\nbaseline.'
assert text.count(needle)==1
text=text.replace(needle,'These model CLI commands retain the existing association-label workflow and\nare not an entry point for the current measured-Y definition. The measured\nexport is `scripts/export_measured_readouts.py`; its output does not authorize\na model run.\n\n`prepare` creates patient-disjoint input/label views and fits the matched\nbaseline.')
text=text.replace('For a two-cancer demo, set these fields in your bound configuration:', 'The requested demo uses BRCA, LUAD and COAD. For an existing model CLI configuration:')
text=text.replace('"cancer_list": ["BRCA", "LUAD"]','"cancer_list": ["BRCA", "LUAD", "COAD"]')
text=text.replace('--cancer_list BRCA LUAD --output','--cancer_list BRCA LUAD COAD --output')
text=text.replace('from two demo cancers.','from the selected demo cancers.')
text=text.replace('Per-assay Pathway scores and the current\n485-State axis','The initial per-assay Pathway scores and\n485-State source axis')
readme.write_text(text)

(REPO/'docs/MEASURED_READOUT_DEFINITIONS.rst').write_text('''Patient measurement definitions
===============================

The biological task is cancer x lncRNA x exact pathway. Pathway identities,
members, measured readouts and missingness are defined before lncRNA modeling.
The output catalog has 477 KEGG NETWORK mechanisms and 197 State identities.
Multiple assays can describe one identity without becoming additional pathway
identities. A member-state readout and a whole-pathway readout retain different
structural roles.

Protein-coding RNA basis
-----------------------

The expression basis selects 19,912 GENCODE protein-coding gene IDs before
aggregating matching symbols in linear TPM, yielding 19,907 symbols. Missing
measurements remain missing. Methods using reference distributions estimate
their parameters in the designated reference patients, then apply the same
parameters to query patients. The sealed outer-test fold is not read.

The PC_GENE_ID_BACKGROUND_R1 method version identifies the recomputed pathway,
TF, PROGENy and State readouts. Restricting an original model's members changes
its measurement definition; original member lists and the applied version
are both retained.

LM22 relative cell composition
-----------------------------

LM22_PC_LINEAR_NUSVR_RELATIVE_V1 uses the fixed external LM22 reference and
linear TPM. Of 547 source symbols, 498 directly match coding genes and 19 are
resolved through approved HGNC-to-GENCODE gene identities. The resulting
reference contains 517 genes and 22 cell categories. Twenty-four directly
noncoding symbols and six unmatched symbols do not enter this version.

For each patient independently, the reference and mixture are centered and
scaled, and linear nu-SVR is evaluated at nu=0.25, 0.5 and 0.75. Coefficients
are truncated at zero and normalized to sum to one. The candidate with the
lowest reference reconstruction RMSE supplies the relative composition.
Missing required measurements yield missing readouts. No cross-patient model
or lncRNA association is fitted. This is a coding-member adaptation of the
published algorithm, not the original published CIBERSORT score.

Method: Newman et al., Nature Methods (2015),
https://pmc.ncbi.nlm.nih.gov/articles/PMC4739640/ .
The implementation matches R e1071 on three synthetic mixtures to a maximum
absolute difference of 6.66e-16. This numerical check does not measure accuracy
against independently measured cell fractions.

Mechanism-specific conditions
-----------------------------

The 334 Reference mechanisms do not require an additional event condition.
The other 143 require matching evidence: 78 variant/expression/receptor
conditions, 46 pathogen/effector conditions and 19 exposures. An unspecified
gene mutation does not establish an activating variant; a continuous copy
number value does not by itself establish an amplification call.

The dictionary records the required evidence separately from measured member
states. Unmeasured conditions are unknown, qualified negatives are absent,
and discordant qualified observations remain unknown. A measured member does
not establish that its upstream mutation, pathogen or exposure is present.

N01392 describes arsenic-to-complex-II chemistry. Its cached source record's
Pathogen category is preserved, while its patient condition is arsenic
exposure, based on the mechanism name and C20296 chemical entry.

Model comparison
----------------

M0 uses the fixed patient readouts, their masks and the same patients as the
graph comparison. It compares lncRNA, mRNA and combined RNA inputs with
unpenalized clinical and independent-purity covariates. The M0 specification
defines the input exclusions and distinguishes RNA score reconstruction from
prediction of independently measured outcomes. TF/kinase/PROGENy display
classification does not create independent losses. Pathway and State weights
are 1.0 and 0.5.
''')

prov=REPO/'provenance/coding_lm22_conditions_20261112';prov.mkdir(exist_ok=True)
for source in [BASE/'CODING_R3_INTEGRATION_TESTS.json',BASE/'CODING_R3_MEASUREMENT_INTEGRATION.json',BASE/'CODING_R3_DICTIONARY_UPDATE.json',BASE/'lm22_coding_r1/REFERENCE_VALIDATION.json',BASE/'condition_definitions_r1/PATHWAY_CONDITIONS.tsv',BASE/'condition_definitions_r1/OBSERVATION_SCHEMA.json',coverage/'SUMMARY.tsv']:
    shutil.copyfile(source,prov/source.name)
for name in ['integrate_measured_r3_149.py','publish_measured_r3_dictionary_149.py','measurements_coverage_r3_149.py','materialize_lm22_coding_149.py','materialize_condition_rules_149.py','validate_lm22_coding_149.py','verify_lm22_r_reference.R','finalize_measured_r3_149.py']:
    shutil.copyfile(WORK/name,prov/name)
report={'target_host':'149','measurement_version':'measurements_coding_r3','cases':6,'patients':sum(x['patients'] for x in lm['completed']),
    'primary_pathways':477,'state_identities':197,'new_coding_cell_readouts':22,'lm22_coding_genes':517,
    'dictionary':dic['dictionary'],'dictionary_sha256':dic['sha256'],'condition_requirements':143,
    'model_training':False,'lncrna_association_computation':False,'accepted_Y':None,
    'remaining':['Final Y representation contract','complete member-level measurement and applicable-condition integration','published TLS12 and RNA-purity lineage','reader report alignment']}
save(prov/'DELIVERY.json',report)
staged=subprocess.check_output(['git','diff','--cached','--name-only'],cwd=REPO,text=True).strip()
if staged:raise RuntimeError('unexpected staged files: '+staged)
subprocess.run(['git','diff','--check'],cwd=REPO,check=True)
selected=['cla_engine/coding_lm22.py','cla_engine/pathway_conditions.py','tests/test_coding_lm22.py','tests/test_pathway_conditions.py','docs/MEASURED_READOUT_DEFINITIONS.rst','README.md',str(prov.relative_to(REPO))]
subprocess.run(['git','add','--',*selected],cwd=REPO,check=True)
subprocess.run(['git','commit','-m','Bind coding immune readouts and explicit pathway conditions before lncRNA modeling'],cwd=REPO,check=True)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
bundle=WORK/'coding_lm22_conditions_r3.bundle'
subprocess.run(['git','bundle','create',str(bundle),'8ca5bfa6f788f6253d69e9dc3bf86a6da796217d..HEAD'],cwd=REPO,check=True)
subprocess.run(['git','bundle','verify',str(bundle)],cwd=REPO,check=True)
save(BASE/'CODING_R3_HANDOFF.json',dict(report,code_commit=head,bundle=str(bundle),bundle_sha256=file_sha(bundle)))
print(json.dumps(dict(report,code_commit=head,bundle=str(bundle),bundle_sha256=file_sha(bundle)),indent=2),flush=True)
