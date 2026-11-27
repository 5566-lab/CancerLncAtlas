from pathlib import Path
import json,hashlib,socket,sys
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'ppt_y_clarification_20261114_r2';B=W/'fixed_y_remake_20261114_r1/Y_bundle';D=O/'deliverables';D.mkdir(exist_ok=True)
sys.path.append('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps')
import pandas as pd
import openpyxl
from openpyxl.styles import Font,PatternFill,Alignment
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
sj(O/'PACKAGE_PREFLIGHT.json',{'target_host':'149','operation':'Export source inventories and current supervised target view without changing patient Y'})
receipt=json.loads((B/'COMPLETE.json').read_text())
for f,h in receipt['files'].items():assert sha(B/f)==h,f
t=rd(B/'TARGETS.tsv');cov=rd(B/'TARGET_COVERAGE.tsv');ev=rd(O/'event_conditions/EVENT_TARGET_AUDIT_141.tsv')
event_ids=set(ev.canonical_id);assert len(event_ids)==141
p=t[t.analysis_role.eq('primary_pathway')];eligible=p[~p.canonical_id.isin(event_ids)].copy();assert len(p)==476 and len(eligible)==335
for case,cc in cov.groupby('case'):
    pc=cc[cc.analysis_role.eq('primary_pathway')];actual=set(pc.loc[pc.patients_supervision.astype(int)>0,'canonical_id'])
    assert actual==set(eligible.canonical_id),case
    assert pc[pc.canonical_id.isin(event_ids)].patients_supervision.astype(int).eq(0).all(),case
eligible.to_csv(D/'本轮可监督主干335条.tsv',sep='\t',index=False)
scope={'target_host':'149','version':'CLA_FIXED_Y_SUPERVISION_VIEW_20261114_R2','catalog_pathways':476,'state_top_level_entries':48,'pathway_supervision_candidates':335,'event_annotation_only':141,'eligible_pathway_ids':eligible.canonical_id.tolist(),'event_annotation_ids':sorted(event_ids),'fixed_y_bundle':str(B),'fixed_y_bundle_receipt_sha256':sha(B/'COMPLETE.json'),'original_values_and_masks_unchanged':True,'selection_basis':'Existing nonempty observation-and-eligibility-and-applicability mask, identical across six demo cohorts','patient_scope':'Existing development patients only; sealed outer test remains excluded','restoration_rule':'Require verified patient-specific event evidence before restoring an event-conditioned target to supervised scope','permanent_deletion_of_141':False,'training':False}
sj(O/'TRAINING_Y_SCOPE.json',scope)
wb=openpyxl.Workbook();wb.remove(wb.active)
tables=[('通路库',O/'source_inventory/COMPOSITION_RESOURCE_INVENTORY.tsv'),('各组学来源',O/'source_inventory/ASSAY_RESOURCE_COUNTS.tsv'),('State与调控来源',O/'source_inventory/STATE_AND_REGULATOR_RESOURCE_INVENTORY.tsv'),('磷酸化签名',O/'source_inventory/PTMSIGDB_INVENTORY.tsv'),('患者项目与平台',O/'source_inventory/PATIENT_PLATFORM_RESOURCES.tsv'),('实际评分方法',O/'source_inventory/ORIGINAL_ASSAY_SCORE_METHODS.tsv'),('六队列来源覆盖',O/'source_inventory/Y_SOURCE_DIMENSIONS_BY_COHORT.tsv'),('WNT来源范围',O/'y_contract/WNT_CONTEXT_MEMBERS.tsv'),('WNT实际维度',O/'y_contract/WNT_OBSERVATION_SUMMARY.tsv'),('WNT来源读数',O/'y_contract/WNT_READOUT_COMPACT.tsv'),('事件分类',O/'event_conditions/EVENT_CATEGORY_COUNTS.tsv'),('335可监督主干',D/'本轮可监督主干335条.tsv')]
def add_sheet(name,df):
    ws=wb.create_sheet(name);ws.append(list(df.columns))
    for row in df.itertuples(index=False,name=None):ws.append([str(v) if isinstance(v,(list,dict)) else v for v in row])
    ws.freeze_panes='A2';ws.auto_filter.ref=ws.dimensions
    for c in ws[1]:c.font=Font(name='Arial',bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='127F88');c.alignment=Alignment(wrap_text=True)
    for col in ws.columns:
        width=min(55,max(16,max(len(str(c.value or '')) for c in list(col)[:100])+2));ws.column_dimensions[col[0].column_letter].width=width
for name,path in tables:add_sheet(name,rd(path))
cols=['canonical_id','network_id','display_name','specific_event','event_family_zh','accepted_evidence_types','present_evidence_status','available_evidence','required_next_step']
add_sheet('141事件机制逐条',ev[cols])
readme=pd.DataFrame([['数据库口径','项目冻结库存；定义数据库、方法来源、患者项目分别计数。'],['主干Y','精确主干成员的实测值、等价来源读数、明确成员的TF/激酶读数；相似子通路整体评分不自动加入。'],['监督范围','当前335条主干存在非空监督掩码；141条事件机制保留注释。'],['数据原值','原始固定Y文件与掩码校验不变，未计算lncRNA关联或训练模型。'],['未知事件','此前事件条件全NA来自未接入患者事件表，不能解释为所有癌症都没有事件。']],columns=['项目','说明']);add_sheet('阅读说明',readme)
wb.save(D/'数据库来源_WNT与141事件明细_20261114_r2.xlsx')
sj(O/'PACKAGE_VALIDATION.json',{'target_host':'149','bundle_files_verified':len(receipt['files']),'original_bundle_receipt_sha256':sha(B/'COMPLETE.json'),'supervised_pathways':335,'event_annotation_pathways':141,'cohorts_verified':6,'no_patient_values_or_masks_changed':True,'workbook_sheets':len(wb.sheetnames),'workbook':str(D/'数据库来源_WNT与141事件明细_20261114_r2.xlsx')})
print('CLARIFICATION_PACKAGE_COMPLETE',flush=True)
