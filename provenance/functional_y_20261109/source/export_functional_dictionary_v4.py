"""Export target identities, score definitions and molecular composition for readers."""
from pathlib import Path
import socket,sys,json,hashlib,zipfile
import pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;C=R/'canonical_targets_r6';O=R/'reports';sys.path.insert(0,'/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps')
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.utils import get_column_letter
D=R/'reader_dictionary_r4';D.mkdir(exist_ok=False)
(D/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'SCIENTIFIC_DICTIONARY_EXPORT'}))
M=pd.read_csv(C/'TARGET_MASTER.tsv',sep='\t').fillna('');B=pd.read_csv(C/'SOURCE_ID_BINDINGS.tsv',sep='\t').fillna('');S=pd.read_csv(R/'catalog_inventory/STATE_SOURCE_ID_REGISTRY_DRAFT.tsv',sep='\t').fillna('');lookup=B.set_index('source_target_id').canonical_id.to_dict()
CONTEXT=M.loc[~M.has_scored_readout.astype(str).str.lower().eq('true')].copy();M=M.loc[M.has_scored_readout.astype(str).str.lower().eq('true')].copy()
source=Path('/dell_2/DSC/CancerLncAtlas/inputs/state_protein_map_20261104_r1');rec=json.loads((source/'COMPLETE.json').read_text());mp=source/'STATE_PROTEIN_MAP.tsv.gz';assert hashlib.sha256(mp.read_bytes()).hexdigest()==rec['files'][mp.name]['sha256']
sm=pd.read_csv(mp,sep='\t').fillna('');sm.insert(0,'canonical_id',sm.state_id.map(lookup));assert sm.canonical_id.notna().all()
science_cols=['canonical_id','state_id','signature_component','source_symbol','source_gene_id','source_probe_id','source_weight','score_direction','relation_type','source_resource','gene_id','reference_symbol','gene_biotype','uniprot_accessions','current_graph_protein_ids','mapping_status','rppa500_feature_ids','measurement_layer']
sm[science_cols].to_csv(D/'STATE_MOLECULAR_COMPOSITION.tsv.gz',sep='\t',index=False)
state=[]
for row in S.itertuples():
 t=json.loads(row.original_definition);state.append({'canonical_id':lookup[row.source_target_id],'state_id':row.source_target_id,'state_family':row.family,'measurement_layer':t.get('measurement',''),'score_method':t.get('method',''),'score_unit':t.get('unit',''),'source_resource':t.get('source',''),'data_form':'Patient by score matrix'})
sd=pd.DataFrame(state);sd.to_csv(D/'STATE_SCORE_DEFINITIONS.tsv',sep='\t',index=False)
Q=R/'supervision_summary_r2';T=R/'reaction_scope_relations_r1'
for ready in [Q/'COMPLETE.json',T/'COMPLETE.json',R/'functional_y_validation_r2/COMPLETE.json']:
 record=json.loads(ready.read_text());assert record.get('status') in ['PASS','COMPLETE'],str(ready)
cov=pd.read_csv(Q/'CURRENT_SCORE_COVERAGE.tsv.gz',sep='\t');cov=cov.loc[cov.patients_with_score.gt(0)].copy()
method_names={'GSVA_ssGSEA_alpha0.25_raw_ES_NA_aware':'ssGSEA raw enrichment score, alpha 0.25','BROAD_PTM_SEA_DIRECTIONAL_ES_SITE_COHORT_CENTERED':'Directional PTM-SEA enrichment, cohort-centered site reference'}
cov.method=cov.method.replace(method_names);cov['platform']=cov.cohort.str.split(':').str[0];cov['data_form']='Patient by measurement readout'
CM=R/'cptac_method_index_r1';assert json.loads((CM/'COMPLETE.json').read_text())['status']=='COMPLETE'
cptac_methods=pd.read_csv(CM/'CPTAC_STATE_METHODS.tsv',sep='\t');specific=cptac_methods[['cohort','source_readout_id','method','reference','score_unit']].rename(columns={'method':'actual_method','reference':'actual_reference','score_unit':'native_score_unit'})
cov=cov.merge(specific,on=['cohort','source_readout_id'],how='left',validate='many_to_one');has_native=cov.actual_method.notna();assert int(has_native.sum())==len(cptac_methods)
cov.loc[has_native,'method']=cov.loc[has_native,'actual_method'];cov.loc[has_native,'reference']=cov.loc[has_native,'actual_reference']
# Biological units follow the actual measurement and scoring definition.
def unit(row):
 if pd.notna(row.native_score_unit):return row.native_score_unit
 if row.method=='Common-background ssGSEA raw ES':return 'Raw enrichment score'
 if row.platform=='TCPA_RPPA500':return 'Standardized abundance or phosphosite residual'
 if 'METHYLATION' in row.measurement_layer and row.method not in ['Signature-specific State score','Source-specific State score']:return 'Mean promoter methylation beta value'
 if 'ssGSEA' in row.method or 'PTM-SEA' in row.method:return 'Raw enrichment score'
 return 'Source-specific State units; see State_scores'
cov['score_unit']=[unit(r) for r in cov.itertuples()]
sc=['canonical_id','source_readout_id','platform','measurement_layer','method','reference','score_unit','data_form']
cov['readout_id']=['CLA-R-'+hashlib.sha256(json.dumps(list(row),ensure_ascii=False).encode()).hexdigest()[:16].upper() for row in cov[sc].fillna('').itertuples(index=False,name=None)]
scoredefs=cov[['readout_id']+sc].drop_duplicates();coverage=cov[['readout_id','canonical_id','source_readout_id','cohort','cancer_id','measurement_layer','patients_in_scope','patients_with_score','coverage_fraction','variable']].drop_duplicates()
scoredefs.to_csv(D/'OBSERVED_SCORE_DEFINITIONS.tsv',sep='\t',index=False);coverage.to_csv(D/'SCORE_COVERAGE.tsv.gz',sep='\t',index=False)
yt=pd.concat([pd.read_csv(R/'functional_relation_y_catalog_r6'/c/'sealed_test/TARGET_COUNTS.tsv',sep='\t').assign(cancer_id=c) for c in pd.read_csv(Q/'Y_SUPERVISION_BY_CANCER.tsv',sep='\t').cancer_id],ignore_index=True)
assert len(yt)==33*len(M)
for name,frame in [('tcga_cancers_with_scores',cov.loc[cov.platform.isin(['TCGA','TCPA_RPPA500'])]),('cptac_cancers_with_scores',cov.loc[cov.platform.eq('CPTAC')]),('tcga_cancers_with_estimable_relations',yt.loc[yt.observable_relations.gt(0)]),('tcga_cancers_with_supported_relations',yt.loc[(yt.strong_positive+yt.weak_positive).gt(0)])]:
 count=frame.groupby('canonical_id').cancer_id.nunique();M[name]=M.canonical_id.map(count).fillna(0).astype(int)
sd['definition_source']='TCGA source State catalog'
sd.score_method=sd.score_method.replace({'LEGACY_WEIGHTED_ZSCORE_UNCHANGED':'Weighted gene-expression z score'});sd.score_unit=sd.score_unit.replace({'SOURCE_DEFINED_NO_RESCALE':'Published score units'})
book=Workbook(write_only=True);counts={}
def sheet(title,frame):
 ws=book.create_sheet(title);ws.freeze_panes='B2';cols=list(frame.columns)
 for i,c in enumerate(cols,1):ws.column_dimensions[get_column_letter(i)].width=45 if any(w in c.lower() for w in ['name','source','method','member','alias']) else 26
 header=[]
 for c in cols:
  cell=WriteOnlyCell(ws,value=c);cell.font=Font(name='Calibri',bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='24476B');cell.alignment=Alignment(wrap_text=True);header.append(cell)
 ws.append(header);count=0
 for row in frame.fillna('').itertuples(index=False,name=None):
  rr=[]
  for v in row:
   if hasattr(v,'item'):v=v.item()
   if isinstance(v,str) and v[:1] in ['=','+','-','@']:v="'"+v
   rr.append(v)
  ws.append(rr);count+=1
 ws.auto_filter.ref=f'A1:{get_column_letter(len(cols))}{count+1}';counts[title]=count;print('SHEET',title,count,flush=True)
notes=pd.DataFrame([
 ('Title','Pathway and State identities, score definitions and molecular composition'),('Version','2026-11-09'),('Biological unit','Cancer, lncRNA and canonical pathway or State'),('Patient score S','Patient-level measurement for a specified pathway or State'),('Relationship Y','Cancer-specific lncRNA association with a canonical pathway or State'),('Canonical ID','CLA-P / CLA-S with a stable authority-derived suffix'),('Roles','A biological identity may be used as both a Pathway and a State'),('Source binding','Maps source identifiers to the corresponding canonical identity'),('Components','Molecular members, signature weights, reactions and annotated subpathways'),('Phospho readout','A directional phosphosite signature linked to its source resource'),('Parent-child relation','Annotated inclusion of a subpathway within a parent pathway'),('Shared-parent relation','Two definitions have an annotated common parent'),('Reaction branch','An annotated reaction-order fork'),('Score coverage','Observed source-readout coverage by cohort, cancer and measurement layer'),('Current scoring','Functional Y scoring uses the protocols listed in Current_scoring; source readouts retain their published definitions'),('State scope','Population denominator and molecular phenotype are part of the State definition'),('Y counts','Y_by_cancer uses the independent label-patient fold; Y_by_context lists each of the five separate contexts'),('Score coverage denominator','All eligible TCGA RNA patients or all patients in the corresponding CPTAC clinical cohort'),('Reaction extensions','Combine reaction_ids chunks by extension_id; Reaction_catalog provides reaction names')],columns=['Field','Description'])
sheet('Read_me',notes);sheet('Target_master',M);sheet('Structural_contexts',CONTEXT);sheet('Source_IDs',B.drop(columns=['identity_evidence']));sheet('State_scores',sd);sheet('Score_definitions',scoredefs);sheet('Score_coverage',coverage)
adjust=pd.read_csv(R/'adjustment_coverage_r1/ADJUSTMENT_SUPPORT_BY_CANCER.tsv',sep='\t').drop(columns=['covariate_design_admitted']);sheet('Covariate_support',adjust);sheet('Covariate_coverage',pd.read_csv(R/'adjustment_coverage_r1/SOURCE_COVARIATE_COVERAGE.tsv',sep='\t'))
sheet('CPTAC_state_methods',cptac_methods);sheet('CPTAC_covariates',pd.read_csv(CM/'CPTAC_COVARIATE_COVERAGE.tsv',sep='\t'))
sheet('Y_by_cancer',pd.read_csv(Q/'Y_SUPERVISION_BY_CANCER.tsv',sep='\t'));sheet('Y_by_context',pd.read_csv(Q/'Y_SUPERVISION_BY_CANCER_CONTEXT.tsv',sep='\t'));sheet('Y_target_counts',yt);sheet('S_by_layer',pd.read_csv(Q/'SCORE_LAYER_SUMMARY.tsv',sep='\t'))
CI=R/'target_composition_index_r1';assert json.loads((CI/'VALIDATION.json').read_text())['status']=='PASS'
sheet('Composition_counts',pd.read_csv(CI/'TARGET_COMPOSITION_COUNTS.tsv',sep='\t'));sheet('Pathway_branches',pd.read_csv(CI/'PATHWAY_REACTION_BRANCHES.tsv.gz',sep='\t'))
sheet('Structure_nodes',pd.read_csv(T/'STRUCTURE_NODES.tsv',sep='\t'));sheet('Reaction_scope',pd.read_csv(T/'REACTION_SCOPE_RELATIONS.tsv.gz',sep='\t'));sheet('Reaction_core_members',pd.read_csv(T/'REACTION_CORE_MEMBERS.tsv.gz',sep='\t'))
ext=pd.read_csv(T/'REACTION_EXTENSION_MEMBERS.tsv.gz',sep='\t');chunks=[]
for eid,frame in ext.groupby('extension_id',sort=True):
 members=sorted(set(frame.reaction_id));assert len(members)==len(frame)
 for start_at in range(0,len(members),1000):
  part=members[start_at:start_at+1000];text=' | '.join(part);assert len(text)<32000
  chunks.append({'extension_id':eid,'total_reactions':len(members),'chunk_index':start_at//1000+1,'reactions_in_chunk':len(part),'reaction_ids':text,'source_version':'Reactome98'})
sheet('Reaction_extensions',pd.DataFrame(chunks));ri=pd.read_csv(C/'PATHWAY_REACTION_COMPOSITION.tsv.gz',sep='\t')[['reaction_stable_id','reaction_name']].drop_duplicates();sheet('Reaction_catalog',ri)
sheet('Gene_components',pd.read_csv(C/'PATHWAY_GENE_COMPOSITION.tsv.gz',sep='\t').drop(columns=['source_definition_sha256']));sheet('State_components',sm[science_cols]);sheet('Reaction_components',pd.read_csv(C/'PATHWAY_REACTION_COMPOSITION.tsv.gz',sep='\t')[['canonical_id','pathway_stable_id','pathway_name','reaction_stable_id','reaction_name','membership_scope']])
ptmb=pd.read_csv(C/'PTM_READOUT_BINDINGS.tsv',sep='\t')[['source_readout_id','canonical_id']];sheet('Phospho_readouts',pd.read_csv(C/'PHOSPHO_READOUT_DEFINITIONS.tsv',sep='\t').merge(ptmb,on='source_readout_id',validate='one_to_one'));sheet('Phospho_sites',pd.read_csv(C/'PHOSPHO_READOUT_COMPOSITION.tsv.gz',sep='\t').merge(ptmb,on='source_readout_id',validate='many_to_one'))
sheet('ID_redirects',pd.read_csv(C/'ID_REDIRECTS.tsv',sep='\t'));sheet('Source_aliases',pd.read_csv(C/'SOURCE_ALIASES.tsv',sep='\t'))
choices={'Official_synonyms':('REACTOME_SYNONYMS.tsv',['canonical_id','reactome_id','alias','relation','evidence']),'Parent_child':('REACTOME_PARENT_CHILD.tsv',['parent_id','parent_reactome_id','child_id','child_reactome_id','relation','source_version']),'Shared_parent':('REACTOME_HIERARCHY_SIBLINGS.tsv.gz',['parent_id','left_id','right_id','relation','evidence']),'Reaction_branches':('REACTION_ORDER_BRANCHES.tsv.gz',['fork_step','upstream_processes','downstream_step','downstream_processes','relation','n_next_steps'])}
for title,(fn,cols) in choices.items():sheet(title,pd.read_csv(R/'catalog_inventory'/fn,sep='\t')[cols])

for title,filename in [('Kinase_components','KINASE_MOLECULAR_COMPONENTS.tsv'),('Kinase_constructs','KINASE_CONSTRUCT_RELATIONS.tsv'),('State_scope','STATE_SCOPE_EQUIVALENCES.tsv'),('State_relations','STATE_COMPONENT_RELATIONS.tsv'),('Functional_scope','FUNCTIONAL_SCOPE_RELATIONS.tsv')]:sheet(title,pd.read_csv(C/filename,sep='\t'))
sheet('Shared_core',pd.read_csv(R/'semantic_relations_r1/CROSS_DATABASE_CURATED_RELATIONS.tsv',sep='\t'));sheet('Core_reactions',pd.read_csv(R/'semantic_relations_r1/CURATED_TYROSINE_CORE_REACTIONS.tsv',sep='\t'))
nodes=pd.read_csv(R/'structure_reactome98/BIOPAX_NODES.tsv.gz',sep='\t');names=nodes.set_index('node_id').name.to_dict()
roles=pd.read_csv(R/'structure_reactome98/REACTION_ENTITY_ROLES.tsv.gz',sep='\t');roles['reaction_name']=roles.reaction_node.map(names);sheet('Reaction_roles',roles)
complexes=pd.read_csv(R/'structure_reactome98/COMPLEX_AND_ENTITY_SET_MEMBERS.tsv.gz',sep='\t');complexes['container_name']=complexes.container_node.map(names);complexes['member_name']=complexes.member_node.map(names);sheet('Complex_components',complexes)
sheet('Protein_states',pd.read_csv(R/'structure_reactome98/PROTEIN_STATES.tsv.gz',sep='\t'));sheet('Protein_modifications',pd.read_csv(R/'structure_reactome98/PROTEIN_MODIFICATIONS.tsv.gz',sep='\t'))
protocols=pd.DataFrame([
 ['RNA pathway','TCGA / CPTAC','Fixed-background ssGSEA, alpha 0.25; raw enrichment score','Common protein-coding background; sample-wise ranks','Patient by pathway'],
 ['RNA-derived State','TCGA','Signature-specific recomputation; training-fitted transformations','Frozen source-patient reference for scaling, program weights and regressions','Patient by State'],
 ['Native State','TCGA / CPTAC','Published assay-specific phenotype or score','Source measurement definition and units','Patient by State'],
 ['Protein pathway','RPPA500','Mean standardized antibody gene-group abundance; minimum 3 observed groups','Source-patient antibody means and standard deviations','Patient by pathway'],
 ['Phosphosite signature','RPPA500','Signed mean standardized total-protein-adjusted phosphosite residuals; minimum 3 sites','Exact sequence match; source-patient regression and scaling','Patient by phosphosite signature'],
 ['Protein pathway','CPTAC','Native ssGSEA protein abundance enrichment','Measured mass-spectrometry protein matrix','Patient by pathway'],
 ['Phosphosite signature','CPTAC','Native directional PTM-SEA enrichment','Measured site-specific mass-spectrometry matrix','Patient by phosphosite signature'],
 ['Promoter methylation pathway','CPTAC','Mean measured promoter-associated gene beta values','Measured promoter CpG-to-gene mapping','Patient by pathway']
],columns=['measurement','platform','score_definition','reference','data_form']);sheet('Current_scoring',protocols)
sheet('Y_fields',pd.DataFrame([['membership','0 / 1 / 2','Unlabeled / one-layer or additional support / at least two strict assay families'],['direction','0 / 1 / 2 / 3','Unknown / positive / negative / discordant measured-score association'],['observed_assay_bits','Bit mask','Assay families with estimable lncRNA-target relations'],['strict_assay_bits','Bit mask','Assay families meeting absolute rho >= 0.20 and BH q <= 0.05'],['supported_assay_bits','Bit mask','Strict or additional evidence: absolute rho >= 0.15 and BH q <= 0.10'],['patient_split','Source and label patient IDs','Disjoint patient roles for graph inputs and relationship labels']],columns=['field','values','definition']))

path=O/'Functional_pathway_State_dictionary_v4.xlsx';book.save(path)
forbidden=[b'MOGSA',b'PathFA',b'NOT_FINAL',b'NOT complete',b'integration_status',b'score_merge_authorized',b'not adopted',b'remains retired',b'independent_flux_claimed',b'independent_functional_branch_claimed',b'identity_merge']
with zipfile.ZipFile(path) as z:
 assert z.testzip() is None
 for name in z.namelist():
  if name.endswith('.xml'):
   with z.open(name) as f:
    tail=b''
    while True:
     block=f.read(1024*1024)
     if not block:break
     check=tail+block
     assert not any(v in check for v in forbidden),(name,'process metadata in scientific workbook')
     tail=block[-100:]
receipt={'target_host':'149','file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'sheets':counts,'state_mapping_source_sha256':rec['files'][mp.name]['sha256']}
(R/'project_log/DICTIONARY_V4_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print('WORKBOOK_COMPLETE',json.dumps(receipt),flush=True)
