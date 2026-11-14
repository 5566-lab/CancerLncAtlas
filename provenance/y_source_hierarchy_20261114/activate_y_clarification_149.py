from pathlib import Path
import json,socket,hashlib,shutil
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'ppt_y_clarification_20261114_r2';D=O/'deliverables';repo=R/'source_integration/repo'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
sj(O/'ACTIVATION_PREFLIGHT.json',{'target_host':'149','operation':'Activate reviewed 28-slide report and explicit supervised target view'})
ppt=json.loads((O/'PPT_MANIFEST.json').read_text());render=json.loads((O/'RENDER_RECEIPT.json').read_text());validation=json.loads((O/'PACKAGE_VALIDATION.json').read_text())
assert ppt['slides']==render['rendered_pages']==28
assert validation['supervised_pathways']==335 and validation['event_annotation_pathways']==141
review={'target_host':'149','all_pages_contact_sheet_reviewed':28,'full_size_pages_reviewed':[4,7,21,24],'source_counts_checked':True,'event_all_cancer_absence_claim_corrected':True,'model_design_distinguished_from_implementation':True}
sj(O/'VISUAL_REVIEW.json',review)
files={p.name:sha(p) for p in D.iterdir() if p.is_file()};assert len(files)==4
manifest={'target_host':'149','files':files,'slides':28,'catalog_pathways':476,'state_entries':48,'supervised_pathway_candidates':335,'event_annotation_pathways':141,'fixed_y_values_unchanged':True,'training':False}
sj(O/'DELIVERY_MANIFEST.json',manifest)
dest=R/'protein_similarity_integration_20261111_r1/deliverables'
for name,h in files.items():shutil.copy2(D/name,dest/name);assert sha(dest/name)==h
backup=O/'previous_current_pointers';backup.mkdir(exist_ok=True);receipts=[]
for name in ['CURRENT_READER_DELIVERY.json','CURRENT_Y_DELIVERY.json','CANONICAL_TARGETS_CURRENT.json','CURRENT_CANONICAL_REFERENCE.json']:
    p=R/name
    if not (backup/name).exists():shutil.copy2(p,backup/name)
    d=json.loads(p.read_text());d.update(training_y_scope=str(O/'TRAINING_Y_SCOPE.json'),current_supervision_pathway_count=335,event_annotation_pathway_count=141,source_and_condition_audit=str(O))
    if name=='CURRENT_READER_DELIVERY.json':d.update(source=str(D),source_receipt_sha256=sha(O/'DELIVERY_MANIFEST.json'),destination=str(dest),directory=str(dest),pptx=str(dest/Path(ppt['pptx']).name),pdf=str(dest/Path(ppt['pptx']).with_suffix('.pdf').name),delivery_manifest=str(O/'DELIVERY_MANIFEST.json'),verified_files=files,status='Y_SOURCE_HIERARCHY_AND_MODEL_REPORT',slides=28)
    tmp=p.with_suffix('.pending.json');sj(tmp,d);tmp.replace(p);receipts.append({'path':str(p),'sha256':sha(p)})
sj(O/'ACTIVATION_RECEIPT.json',{'target_host':'149','pointers':receipts,'fixed_y_changed':False,'reader_ppt_slides':28,'supervised_pathway_candidates':335,'event_annotation_pathways':141,'training':False})
log={'target_host':'149','request':'Supplement database counts, exact WNT hierarchy, matched M0/HHGT design and audit 141 event mechanisms','reader_delivery':str(O/'DELIVERY_MANIFEST.json'),'slides':28,'shared_composition_resources':4,'composition_definitions':2740,'rna_scoring_source_families':14,'rna_definitions':411,'ptmsigdb_definitions':624,'event_condition_root_cause':'Original materializer passed an empty patient-event table; all-cancer absence was not established','supervised_pathway_candidates':335,'event_annotation_pathways':141,'no_permanent_dictionary_deletion':True,'fixed_y_values_and_masks_unchanged':True,'WNT_primary_members':46,'WNT_context_does_not_expand_primary_Y':True,'M0_new_Y_fit_adapter_implemented':False,'HHGT_patient_observation_decoder_implemented':False,'training':False,'lncrna_association':False}
sj(repo/'project_log/Y_SOURCE_HIERARCHY_CONDITIONS_20261114.json',log)
prov=repo/'provenance/y_source_hierarchy_20261114';prov.mkdir(exist_ok=True)
for rel in ['extend_y_ppt_149.py','slide_content.py','package_y_clarification_149.py','render_y_clarification_149.py','activate_y_clarification_149.py','PPT_MANIFEST.json','PACKAGE_VALIDATION.json','TRAINING_Y_SCOPE.json','DELIVERY_MANIFEST.json','VISUAL_REVIEW.json','ACTIVATION_RECEIPT.json']:
    shutil.copy2(O/rel,prov/rel)
for folder,patterns in [('source_inventory',['*.tsv','SUMMARY.json','AUDIT_RECEIPT.json']),('event_conditions',['EVENT_CATEGORY_COUNTS.tsv','EVENT_TARGET_AUDIT_141.tsv','EVENT_AUDIT_SUMMARY.json','CURRENT_APPLICABILITY_CHECK.json']),('y_contract',['*.tsv','Y_MODEL_CONTRACT_REPORT.json'])]:
    q=prov/folder;q.mkdir(exist_ok=True)
    for pattern in patterns:
        for p in (O/folder).glob(pattern):shutil.copy2(p,q/p.name)
doc='''Exact backbone Y, source hierarchy and current supervision scope
===============================================================

The fixed patient Y values and masks remain unchanged. Its catalogue has 476
Pathway identities and 48 State entries. An explicit supervised view selects
335 pathways with existing nonempty eligibility/applicability masks in each of
the six demo cohorts. The 141 event-conditioned definitions remain mechanism
annotations. The original construction passed an empty patient-event table;
that fact does not establish absence of events across cancers. Existing CNV,
mutation and public event inventories require appropriate patient-level mapping
and event interpretation before those targets can return to supervision.

An exact backbone owns member measurements, equivalent pathway readouts and
specified TF/kinase component readouts. A similar subpathway is structural
context, not an equivalent identity. WNT N00056 retains 46 members. Reactome
WNT ligand biogenesis and trafficking shares 18 members; its 8 additional
members and whole-program score do not expand this WNT Y. The BRCA WNT vector
has 155 registered dimensions for TCGA and 260 for CPTAC, with original masks.

The model specification uses the same observed outputs and patient folds for
M0 and HHGT. M0 fits source/platform-specific linear Lasso outputs with agreed
unpenalized clinical and independent-purity covariates. HHGT requires patient
conditioned graph encoding and matching observation decoders. Platform-specific
outputs preserve measurement identities; a cohort covariate is not numerical
calibration. Non-RNA target layers and derivatives are excluded from inputs.
RNA reconstruction hides the target and its direct mRNA constituents, while
lncRNA input remains available. Shared observation columns are counted once;
the prescribed Pathway/State weights remain 1.0/0.5. Model-comparison results are
then aggregated to the frozen backbone to assess predictive lncRNA associations.

Implementation audit: the fixed-Y accessor exists, but the new M0 fitting adapter
and HHGT patient-observation decoder are not yet wired into training. The old
HHGT relation-logit head and RMS-effect preparation are not this task. The
28-slide report therefore presents these models as a subsequent design, with
no new model training or lncRNA-derived Y calculation.

Resource counts are frozen project inventory: four shared composition resources
with 2740 definitions; 14 RNA scoring-resource families with 411 definitions;
PTMsigDB with 624 directional signatures. Source databases, scoring methods and
patient projects use separate denominators. The inventory workbook lists input,
bound and measured dimensions separately.
'''
(repo/'docs/EXACT_Y_SOURCE_HIERARCHY.rst').write_text(doc,encoding='utf8')
print('CLARIFIED_REPORT_AND_SCOPE_ACTIVATED',flush=True)
