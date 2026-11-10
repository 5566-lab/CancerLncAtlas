"""Reader dictionary for the three-cancer fused relationship Y."""
from pathlib import Path
import sys,json,socket,hashlib,argparse
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent;F=R/'functional_relation_y_fused_r1';O=R/'shared_relation_demo_r1';C=R/'canonical_targets_r6';D=F/'dictionary_data';D.mkdir(exist_ok=True)
assert socket.gethostname()=='149'
sys.path[:0]=[str(R/'python_deps'),'/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps']
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.utils import get_column_letter
from materialize_shared_tcga_relations import canon
CANCERS=['BRCA','LUAD','COAD'];CONTEXTS=['train_label_fold2','train_label_fold3','train_label_fold4','validation','sealed_test']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return pd.read_csv(p,sep='\t').fillna('')
def out(name,df):df.to_csv(D/(name+'.tsv.gz' if len(df)>100000 else name+'.tsv'),sep='\t',index=False)
def metadata():
 identity=read(C/'TARGET_MASTER.tsv').set_index('canonical_id').to_dict('index');bindings=read(C/'SOURCE_ID_BINDINGS.tsv').set_index('source_target_id').to_dict('index')
 old=read(R/'reader_dictionary_r4/STATE_SCORE_DEFINITIONS.tsv')
 old.score_method=old.score_method.replace({'legacy_R07_weighted_gene_zscores_then_program_zscore':'Weighted gene-expression z score with program standardization'})
 units={'LEGACY_WEIGHTED_ZSCORE_UNCHANGED':'Standardized weighted expression score','EXISTING_GSVA_SSGSEA_NO_RESCALE':'ssGSEA enrichment score','PUBLISHED_PANIMMUNE_SCORE_NO_RESCALE':'Published immune phenotype score','ALTERED_CHROMOSOME_ARMS':'Altered chromosome-arm count','CELL_ABUNDANCE_SCORE':'Cell abundance score','CELL_ENRICHMENT_SCORE':'Cell enrichment score','NATIVE_SCAR_COUNT':'Genomic scar count','NON_SILENT_MUTATIONS_PER_MB':'Non-silent mutations per Mb','RELATIVE_CELL_FRACTION':'Relative cell fraction','RELATIVE_MUTATION_CONTRIBUTION':'Relative mutation contribution'}
 old.score_unit=old.score_unit.replace(units);state=old.set_index('state_id').to_dict('index');cm=read(R/'cptac_method_index_r1/CPTAC_STATE_METHODS.tsv');native_methods={(x.cancer_id,x.source_readout_id):x._asdict() for x in cm.itertuples()};rec=[];cov=[];layers=[];inputs=[]
 method_map={'GSVA_ssGSEA_ALPHA025_COMMON_PC_BACKGROUND_SOURCE_RANGE':'ssGSEA, alpha 0.25, common protein-coding background, source-fitted range','GSVA_ssGSEA_ALPHA025_SAMPLEWISE_RAW_ES':'ssGSEA, alpha 0.25, sample-wise raw ES','BROAD_PTM_SEA_DIRECTIONAL_ES_SOURCE_PATIENT_SITE_MEDIAN':'Directional PTM-SEA, source-patient site median reference','OBSERVED_PROMOTER_GENE_MEAN_BETA':'Mean measured promoter-associated gene methylation beta','SOURCE_FITTED_COMMON_RNA_DEFINITION':'Common RNA State definition, source-fitted transformations','RPPA_TRAIN_Z_GENE_GROUP_MEAN':'Mean source-standardized antibody gene-group abundance'}
 for c in CANCERS:
  for ds,ls in [('TCGA',['RNA','STATE','PROTEIN','PHOSPHO']),('CPTAC',['RNA','STATE','PROTEIN','PHOSPHO','METHYLATION'])]:
   src='CPTAC_DEDUPLICATED' if c=='BRCA' and ds=='CPTAC' else ds;base=O/src/c/'sealed_test';sp=json.loads((base/'SPLIT.json').read_text());ax=json.loads((base/'AXES.json').read_text());patients=ax['patient_ids'];label=sp['label_patients'];source=sp['source_patients']
   for layer in ls:
    p=base/layer;md=read(p/'TARGETS.tsv');s=pd.read_parquet(p/'PATIENT_S.parquet').reindex(patients);assert list(s)==md.target_id.tolist();stats=read(p/'TARGET_COUNTS.tsv');receipt=json.loads((p/'COMPLETE.json').read_text());inputs.append({'dataset':ds,'cancer_id':c,'layer':layer,'directory':str(p),'receipt_sha256':sha(p/'COMPLETE.json')})
    for row in md.to_dict('records'):
     sid=row['source_target_id'];definition=state.get(sid,{});target=row['target_id'];cid=canon(row['canonical_id']);method=row.get('method') or row.get('score_method') or ('ssGSEA, alpha 0.25, common protein-coding background' if layer=='RNA' else definition.get('score_method',''))
     method=method_map.get(method,method);unit='Raw enrichment score';reference='Source-patient reference fitted separately for each context'
     if layer=='STATE':
      unit=definition.get('score_unit','Source-defined State units');method=method or 'Source-specific State definition'
      if sid.startswith('PROGENY:'):method='Official PROGENY top100 weighted linear model';unit='Source-standardized model response'
      elif row.get('native_reference') in [True,'True','true']:reference='Source-cohort measurement definition and reference'
     elif layer=='PROTEIN' and ds=='TCGA':unit='Source-standardized abundance';method=method_map['RPPA_TRAIN_Z_GENE_GROUP_MEAN']
     elif layer=='PHOSPHO' and ds=='TCGA':unit='Source-standardized phosphosite residual';method='Signed mean of exact-site, total-protein-adjusted RPPA residuals'
     elif layer=='METHYLATION':unit='Methylation beta value'
     if layer=='RNA' and ds=='CPTAC':unit='ssGSEA enrichment divided by the fitted source-patient range'
     resource=definition.get('source_resource',row.get('source_database',''))
     if layer=='STATE' and ds=='CPTAC' and row.get('native_reference') in [True,'True','true']:
      native_key=target.split('|',1)[-1];info=native_methods.get((c,native_key),{})
      method=info.get('method') or 'Source-cohort phenotype definition';unit=info.get('score_unit') or units.get(unit,unit);reference=info.get('reference') or 'Source-cohort measurement definition'
      resource='CPTAC State matrix: cptac_20261104_r1/bundle_r1/STATE_PATIENT.parquet; column '+native_key
      if sid=='ABSOLUTE_purity':method='Published WES-derived tumor purity from CPTAC phenotype';unit='Tumor cell fraction';reference='Published source-cohort WES_purity value'
      definition=dict(definition);definition['score_method']=method

     identity_info=identity[cid];binding_info=bindings.get(sid,{})
     definition_database=binding_info.get('source_database') or identity_info['source_database']
     if layer=='STATE' and ds=='CPTAC' and row.get('native_reference') in [True,'True','true']:definition_database='CPTAC published phenotype'
     if not resource:
      if definition_database=='PTMsigDB':resource='/dsk2/user/dengsc/cla_cptac_integration_20261104_r1/sources/PTMSIGDB_PATH_KINASE_V2.gmt'
      else:resource=identity_info.get('source_url') or definition_database
     score_matrix=str(Path('S')/ds/c/'sealed_test'/layer/'PATIENT_S.parquet')
     assert resource and definition_database and (Path('/dsk2/user/dengsc/CancerLncAtlas/processed/functional_y_20261110_r2')/score_matrix).exists()
     platform='TCPA RPPA500' if ds=='TCGA' and layer in ['PROTEIN','PHOSPHO'] else ds
     x={'canonical_id':cid,'dataset':ds,'platform':platform,'cancer_id':c,'source_readout_id':sid,'source_column':target,'measurement_layer':row['measurement_layer'],'assay_group':layer,'applied_method':method,'source_definition_method':definition.get('score_method',''),'definition_database':definition_database,'source_resource':resource,'score_matrix':score_matrix,'score_unit':unit,'reference':reference,'data_form':'Patient by readout matrix'}
     rec.append(x);v=s[target];finite=np.isfinite(v);cov.append({k:x[k] for k in ['canonical_id','dataset','platform','cancer_id','source_readout_id','measurement_layer','assay_group']}|{'context':'sealed_test','patients_in_cohort':len(patients),'patients_with_score':int(finite.sum()),'patient_coverage_fraction':float(finite.mean()),'source_patients':len(source),'source_patients_with_score':int(np.isfinite(v.reindex(source)).sum()),'label_patients':len(label),'label_patients_with_score':int(np.isfinite(v.reindex(label)).sum())})
    vv=np.isfinite(s.to_numpy(float));layers.append({'dataset':ds,'platform':platform,'cancer_id':c,'assay_group':layer,'patients_in_cohort':len(s),'patients_with_any_score':int(vv.any(1).sum()),'score_coverage_fraction':float(vv.any(1).mean()),'registered_readouts':len(md),'readouts_with_score':int(vv.any(0).sum()),'canonical_targets_with_score':len(set(md.loc[vv.any(0),'canonical_id'].map(canon))),'eligible_readout_relations':receipt['eligible_relations']})
 frame=pd.DataFrame(rec);out('SCORE_DEFINITIONS',frame);out('SCORE_COVERAGE',pd.DataFrame(cov));out('LAYER_COVERAGE',pd.DataFrame(layers));out('STATE_SCORE_DEFINITIONS',old);(D/'SCORE_INPUT_REFERENCES.json').write_text(json.dumps(inputs,indent=2));(D/'METADATA_COMPLETE.json').write_text(json.dumps({'target_host':'149','readouts':len(rec),'datasets':CANCERS,'source_receipts':len(inputs)},indent=2));return frame,pd.DataFrame(cov),pd.DataFrame(layers)
def export():
 for c in CANCERS:
  for context in CONTEXTS:assert (F/c/context/'COMPLETE.json').exists(),(c,context)
 scoredefs,cov,layers=metadata();master=read(C/'TARGET_MASTER.tsv');contextnodes=master.loc[~master.has_scored_readout.astype(str).str.lower().eq('true')];master=master.loc[master.has_scored_readout.astype(str).str.lower().eq('true')].copy();assert len(master)==3456
 allcounts=[];summary=[]
 for c in CANCERS:
  for context in CONTEXTS:
   base=F/c/context;t=read(base/'TARGET_COUNTS.tsv');t.insert(0,'cancer_id',c);t.insert(1,'context',context);allcounts.append(t);sp=json.loads((base/'SPLIT.json').read_text());r=json.loads((base/'COMPLETE.json').read_text());n=int(t.strong_relations.sum()+t.weak_relations.sum());non=int(t.supported_with_individually_supported_nonRNA.sum());rna=int(t.supported_with_only_individually_supported_RNA.sum())
   summary.append({k:r[k] for k in ['cancer_id','context','canonical_targets','observed_targets','supported_targets','observed_relations','strong_relations','weak_relations','joint_cohort_relations','mixed_direction_supported_relations']}|{'supported_relations':n,'supported_with_RNA_only_individual_support':rna,'supported_with_nonRNA_individual_support':non,'supported_without_individual_readout_support':n-rna-non,'TCGA_label_patients':len(sp['TCGA']['label_patients']),'CPTAC_label_patients':len(sp['CPTAC']['label_patients']),'TCGA_source_patients':len(sp['TCGA']['source_patients']),'CPTAC_source_patients':len(sp['CPTAC']['source_patients'])})
 targets=pd.concat(allcounts,ignore_index=True);summary=pd.DataFrame(summary);final=targets.loc[targets.context.eq('sealed_test')].copy();sc=summary.loc[summary.context.eq('sealed_test')].copy();assert len(final)==3456*3
 for col,subset in [('demo_cancers_with_patient_scores',cov.loc[cov.patients_with_score.gt(0)]),('demo_cancers_with_estimable_relations',final.loc[final.observed_relations.gt(0)]),('demo_cancers_with_supported_relations',final.loc[final.supported_relations.gt(0)])]:master[col]=master.canonical_id.map(subset.groupby('canonical_id').cancer_id.nunique()).fillna(0).astype(int)
 for name,df in [('TARGET_MASTER',master),('Y_BY_CANCER',sc),('Y_BY_CONTEXT',summary),('Y_TARGET_COUNTS',final),('Y_TARGET_COUNTS_ALL_CONTEXTS',targets)]:out(name,df)
 b=read(C/'SOURCE_ID_BINDINGS.tsv');b=b.loc[b.canonical_id.isin(master.canonical_id)];counts={'canonical_identities':len(master),'source_binding_rows':len(b),'distinct_source_identifiers':b.source_target_id.nunique(),'Pathway_roles':int(master.roles.str.contains('Pathway').sum()),'State_roles':int(master.roles.str.contains('State').sum()),'shared_roles':int(master.roles.eq('Pathway|State').sum()),'structural_contexts':len(contextnodes),'demo_cancers':CANCERS};(D/'IDENTITY_COUNTS.json').write_text(json.dumps(counts,indent=2))
 wb=Workbook(write_only=True);sheets={}
 def sheet(name,df):
  ws=wb.create_sheet(name);ws.freeze_panes='B2';cols=list(df.columns)
  for i,c in enumerate(cols,1):ws.column_dimensions[get_column_letter(i)].width=48 if any(x in c.lower() for x in ['name','method','reference','definition','source']) else 25
  h=[]
  for c in cols:
   cell=WriteOnlyCell(ws,value=c);cell.font=Font(name='Calibri',bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='24476B');cell.alignment=Alignment(wrap_text=True);h.append(cell)
  ws.append(h)
  for row in df.fillna('').itertuples(index=False,name=None):
   vals=[]
   for v in row:
    if hasattr(v,'item'):v=v.item()
    if isinstance(v,str) and v[:1] in ['=','+','-','@']:v="'"+v
    if isinstance(v,str) and len(v)>32767:raise ValueError('EXCEL_CELL_TOO_LONG '+name)
    vals.append(v)
   ws.append(vals)
  ws.auto_filter.ref=f'A1:{get_column_letter(len(cols))}{len(df)+1}';sheets[name]=len(df);print('SHEET',name,len(df),flush=True)
 notes=[('主任务','癌种 × lncRNA × 精确通路/State 的关联关系 Y'),('研究范围','BRCA、LUAD、COAD；各癌种独立计算'),('患者读数 S','不同平台的实测分子读数与评分保留原定义；在关系层融合'),('主结果计数','Y_by_cancer 和 Y_target_counts 仅使用 sealed_test 标签患者；不累加分折'),('统一身份','同义或功能范围等价共享 canonical_id；父子、分支、共同核心和扩展保留关系'),('命名','固定 CLA-P / CLA-S ID；规范生物学名称和来源别名分别保存'),('读数融合','Fisher 统计量；用读数残差相关结构模拟条件零分布，校正重复和依赖'),('强支持','融合 BH q≤0.05 且至少一项调整后 |rho|≥0.20'),('弱支持','融合 BH q≤0.10 且至少一项 |rho|≥0.15，排除强支持'),('无支持标签','未标注；不代表已证实无关系'),('方向','各读数的关联方向及显著方向冲突另存'),('缺失','缺测或无法估计保留缺失，不赋零分'),('协变量','年龄、性别、分期、DNA来源纯度、巨噬细胞比例；目标特异性调整，局部CNV作用于lncRNA残差'),('参考患者与标签患者','参考患者拟合评分和协变量编码；不与对应标签患者重叠'),('权重','Pathway 1.0，State 0.5；双重角色身份计一次'),('数据形式','患者×评分矩阵；lncRNA×统一身份关系矩阵；组成及映射长表'),('非RNA支持','某个非RNA读数单独达到关系支持门槛；不同于融合显著性，也不代表因果')]
 sheet('Read_me',pd.DataFrame(notes,columns=['项目','说明']));sheet('Target_master',master);sheet('Identity_counts',pd.DataFrame([(k,str(v)) for k,v in counts.items()],columns=['field','value']));sheet('Structural_contexts',contextnodes);sheet('Source_IDs',b.drop(columns=['identity_evidence']));sheet('Score_definitions',scoredefs);sheet('Score_coverage',cov);sheet('Layer_coverage',layers);sheet('State_scores',read(D/'STATE_SCORE_DEFINITIONS.tsv'));sheet('Covariate_coverage',read(O/'COVARIATE_COVERAGE.tsv').drop(columns=['source_file_sha256']));sheet('Y_by_cancer',sc);sheet('Y_by_context',summary);sheet('Y_target_counts',final)
 CI=R/'target_composition_index_r1';T=R/'reaction_scope_relations_r1'
 frames=[('Composition_counts',CI/'TARGET_COMPOSITION_COUNTS.tsv'),('Pathway_branches',CI/'PATHWAY_REACTION_BRANCHES.tsv.gz'),('Gene_components',C/'PATHWAY_GENE_COMPOSITION.tsv.gz'),('State_components',R/'reader_dictionary_r4/STATE_MOLECULAR_COMPOSITION.tsv.gz'),('Reaction_components',C/'PATHWAY_REACTION_COMPOSITION.tsv.gz'),('ID_redirects',C/'ID_REDIRECTS.tsv'),('Source_aliases',C/'SOURCE_ALIASES.tsv'),('Official_synonyms',R/'catalog_inventory/REACTOME_SYNONYMS.tsv'),('Parent_child',R/'catalog_inventory/REACTOME_PARENT_CHILD.tsv'),('Shared_parent',R/'catalog_inventory/REACTOME_HIERARCHY_SIBLINGS.tsv.gz'),('Reaction_branches',R/'catalog_inventory/REACTION_ORDER_BRANCHES.tsv.gz'),('Kinase_components',C/'KINASE_MOLECULAR_COMPONENTS.tsv'),('Kinase_constructs',C/'KINASE_CONSTRUCT_RELATIONS.tsv'),('State_scope',C/'STATE_SCOPE_EQUIVALENCES.tsv'),('State_relations',C/'STATE_COMPONENT_RELATIONS.tsv'),('Functional_scope',C/'FUNCTIONAL_SCOPE_RELATIONS.tsv'),('External_mappings',C/'CURATED_EXTERNAL_RELATIONS.tsv'),('Shared_core',R/'semantic_relations_r1/CROSS_DATABASE_CURATED_RELATIONS.tsv'),('Core_reactions',R/'semantic_relations_r1/CURATED_TYROSINE_CORE_REACTIONS.tsv'),('Structure_nodes',T/'STRUCTURE_NODES.tsv'),('Reaction_scope',T/'REACTION_SCOPE_RELATIONS.tsv.gz'),('Reaction_core_members',T/'REACTION_CORE_MEMBERS.tsv.gz')]
 for name,p in frames:
  df=read(p);df=df.drop(columns=[x for x in ['source_definition_sha256','identity_evidence','identity_result','identity_merge','independent_functional_branch_claimed','independent_flux_claimed'] if x in df]);sheet(name,df)
 ext=read(T/'REACTION_EXTENSION_MEMBERS.tsv.gz');rows=[]
 for eid,frame in ext.groupby('extension_id',sort=True):
  members=sorted(set(frame.reaction_id))
  for start in range(0,len(members),1000):rows.append({'extension_id':eid,'total_reactions':len(members),'chunk_index':start//1000+1,'reaction_ids':' | '.join(members[start:start+1000])})
 sheet('Reaction_extensions',pd.DataFrame(rows));sheet('Reaction_catalog',read(C/'PATHWAY_REACTION_COMPOSITION.tsv.gz')[['reaction_stable_id','reaction_name']].drop_duplicates())
 ptmb=read(C/'PTM_READOUT_BINDINGS.tsv')[['source_readout_id','canonical_id']]
 sheet('Phospho_readouts',read(C/'PHOSPHO_READOUT_DEFINITIONS.tsv').merge(ptmb,on='source_readout_id',validate='one_to_one'));sheet('Phospho_sites',read(C/'PHOSPHO_READOUT_COMPOSITION.tsv.gz').merge(ptmb,on='source_readout_id',validate='many_to_one'))
 nodes=read(R/'structure_reactome98/BIOPAX_NODES.tsv.gz');names=nodes.set_index('node_id').name.to_dict();roles=read(R/'structure_reactome98/REACTION_ENTITY_ROLES.tsv.gz');roles['reaction_name']=roles.reaction_node.map(names);sheet('Reaction_roles',roles);complexes=read(R/'structure_reactome98/COMPLEX_AND_ENTITY_SET_MEMBERS.tsv.gz');complexes['container_name']=complexes.container_node.map(names);complexes['member_name']=complexes.member_node.map(names);sheet('Complex_components',complexes)
 for name,fn in [('Protein_states','PROTEIN_STATES.tsv.gz'),('Protein_modifications','PROTEIN_MODIFICATIONS.tsv.gz')]:sheet(name,read(R/'structure_reactome98'/fn))
 fields=[('combined_p','Combined conditional-null association p'),('combined_q','BH over estimable lncRNA × target pairs within cancer/context/primary task'),('membership','0 unlabelled; 1 weak support; 2 strong support'),('direction','0 unknown; 1 positive; 2 negative; 3 conflicting readouts'),('observed','1 if association estimable; 0 if unmeasured or unestimable'),('max_abs_effect','Largest observed absolute adjusted rho'),('rms_effect','Root mean squared adjusted rho after duplicate weighting'),('observed_assay_bits','RNA=1; protein=2; phospho=4; methylation=8; CNV=16; variant=32; other=64'),('individually_supported_assay_bits','Source-readout support, separate from combined significance'),('cohort_count','Number of cohorts with an estimable association'),('readout_count','Number of observed readouts'),('distinct_readout_count','Number after exact duplicate grouping')];fields.extend([('individually_strict_assay_bits', 'Assay families with source-readout |rho| >= 0.20 and BH q <= 0.05'), ('individually_supported_cohort_count', 'Number of cohorts containing at least one individually supported readout'), ('calibration_method', '0 missing; 1 exact single readout; 2 correlated-null Monte Carlo; 3 unknown-dependence Bonferroni; 4 exact perfect dependence'), ('positive_evidence_count', 'Observed positive readouts with unadjusted p <= 0.05 and |rho| >= 0.15'), ('negative_evidence_count', 'Observed negative readouts with unadjusted p <= 0.05 and |rho| >= 0.15')]);sheet('Y_fields',pd.DataFrame(fields,columns=['field','definition']))
 path=F/'reports/Functional_pathway_State_dictionary_v5.xlsx';wb.save(path);receipt={'target_host':'149','file':str(path),'sha256':sha(path),'sheets':sheets,'identity_counts':counts,'contexts':15,'data_root':str(D)};(F/'reports/DICTIONARY_V5_RECEIPT.json').write_text(json.dumps(receipt,indent=2));print('WORKBOOK_COMPLETE',json.dumps(receipt),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--metadata-only',action='store_true');args=ap.parse_args();(D/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','action':'scientific dictionary export'}));metadata() if args.metadata_only else export()
