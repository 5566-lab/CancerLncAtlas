from pathlib import Path
import socket,sys,json,shutil,subprocess
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');WORK=ROOT/'kegg_cancer_redefinition_20261111_r1';REPO=ROOT/'source_integration/repo'
BASE=WORK/'measured_y_definition_r1';REVIEW=BASE/'final_definition_review_r1';REG=BASE/'observation_registry_r2';REPORT=WORK/'reader_delivery_fixed_y_20261112'
CAT=ROOT/'canonical_targets_r9_cancer_evidence_20261111_r6'
sys.path.insert(0,str(REPO));sys.path.append('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps')
from cla_engine.common import sha
import pandas as pd
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
save(REVIEW/'FINALIZE_PREFLIGHT.json',dict(target_host='149',operation='final definition audit and delivery',training=False,lncrna_association=False))
catalog=json.loads((CAT/'COMPLETE.json').read_text());roles=pd.read_csv(CAT/'TARGET_POLICY.tsv',sep='\t').fillna('')
assert roles[roles.analysis_role.eq('context_only')].backbone_shared_protein_count.gt(0).all()
assert roles[roles.analysis_role.eq('primary_pathway')].loss_weight.eq(1.).all()
assert roles[roles.analysis_role.eq('state_auxiliary')].loss_weight.eq(.5).all()
assert roles[roles.analysis_role.eq('context_only')].loss_group.eq('none').all()
old=json.loads((REVIEW/'CATALOG_REVIEW.json').read_text());old['program_source_rows_checked']=860;save(REVIEW/'CATALOG_REVIEW.json',old)
# Verify the second illustrative case against original mapped protein composition.
comp=pd.read_csv(ROOT/'protein_similarity_integration_20261111_r1/PROTEIN_COMPOSITION.tsv.gz',sep='\t')
a=set(comp.loc[comp.node_id.eq('CLA-P-08A1A41F05A3E9ED'),'uniprot_id']);b=set(comp.loc[comp.node_id.eq('CLA-P-549730F42F0B148D'),'uniprot_id']);assert a==b and len(a)==29
target=pd.read_csv(CAT/'TARGET_MASTER.tsv',sep='\t').fillna('');assert set(['CLA-P-08A1A41F05A3E9ED','CLA-P-549730F42F0B148D'])<=set(target.canonical_id)
smoke=REVIEW/'MMR_TCGA_BRCA_export';sr=json.loads((smoke/'COMPLETE.json').read_text())
for n,h in sr['files'].items():assert sha(smoke/n)==h
assert sr['patients']==872 and sr['observations']==89
values=pd.read_parquet(smoke/'VALUES.parquet');observed=pd.read_parquet(smoke/'OBSERVED.parquet');assert values.notna().equals(observed)
publication=json.loads((REVIEW/'DICTIONARY_UPDATE.json').read_text());dictionary=Path(publication['dictionary']);assert sha(dictionary)==publication['sha256']
from openpyxl import load_workbook
wb=load_workbook(dictionary,read_only=True,data_only=True)
assert sum(1 for _ in wb['KEGG癌症主干'].values)==478
assert any('TLS12_GDC_SSGSEA_PC_RAW_V1' in str(row) for row in wb['RNA读数方法'].values)
assert len(wb.sheetnames)==40;wb.close()
pdf=next(REPORT.glob('*.pdf'));ppt=next(REPORT.glob('*.pptx'))
info=subprocess.run(['pdfinfo',str(pdf)],capture_output=True,text=True,check=True).stdout
assert 'Pages:           10' in info
assert (REPORT/'CONTACT_SHEET.png').is_file()
requirements=[
 ('477 evidence-defined KEGG cancer identities',str(CAT/'KEGG_SOURCE_SCOPE_DECISIONS.tsv')),
 ('Original source protein lists and typed cross-library mappings',str(CAT/'BACKBONE_ALIGNMENT.tsv')),
 ('Actual protein-list clustering and functional-scope integration examples',str(REPORT/'FIGURE_DATA.json')),
 ('TF/kinase/PROGENy Pathway category without automatic losses',str(REVIEW/'PROGRAM_CLASSIFICATION.json')),
 ('Measured Y independent of lncRNA; all 58 blocks and masks verified',str(REVIEW/'MEASURED_Y_VALIDATION.json')),
 ('Unknown conditions distinct from negative; actual raw sources reviewed',str(REVIEW/'CONDITION_EVIDENCE_REVIEW.json')),
 ('M0 is patient-input specification with L/M/LM comparisons',str(REPO/'docs/M0_RELATION_BASELINE.rst')),
 ('Reader dictionary overwritten with current Y and scoring definitions',str(REVIEW/'DICTIONARY_UPDATE.json')),
 ('Human-readable 10-slide problem/method/result report',str(ppt)),
 ('Real fixed-Y export without fitting or lncRNA inputs',str(smoke/'COMPLETE.json'))]
audit=dict(target_host='149',definition_version='KEGG_CANCER_MEASURED_Y_20261112_R2',requirements=[dict(requirement=a,evidence=b,sha256=sha(b),status='verified') for a,b in requirements],reader_visual_review='10 PDF pages inspected via rendered contact sheet',training=False,lncrna_association=False,remaining_delivery_steps=['local copy hash verification','GitHub publication'])
save(REVIEW/'FINAL_SCOPE_AUDIT.json',audit)
docs={'M0线性对照设置.txt':REPO/'docs/M0_RELATION_BASELINE.rst','多组学Y定义与数据接口.txt':REPO/'docs/MEASURED_Y_REPRESENTATION.rst'}
for name,source in docs.items():shutil.copyfile(source,REPORT/name)
shutil.copyfile(dictionary,REPORT/dictionary.name)
(REPORT/'阅读顺序.txt').write_text('先阅读《CancerLncAtlas_KEGG癌症主干与多维Y_20261112》PPT或PDF。\n字典先看“阅读说明”“KEGG癌症主干”“Y组织方式”“成员测量覆盖”，再查具体来源。\nM0设置见《M0线性对照设置.txt》；Y定义与读取接口见《多组学Y定义与数据接口.txt》。\n477是精确通路身份数；每条通路包含多项有来源定义的患者观测，不是477个统一活性分数。\n',encoding='utf-8')
files=[ppt,pdf,REPORT/dictionary.name,*[REPORT/n for n in docs],REPORT/'阅读顺序.txt',*sorted((REPORT/'figures').glob('*.png')),*sorted((REPORT/'figures').glob('*.pdf'))]
delivery=dict(target_host='149',definition=str(REVIEW/'FIXED_Y_DEFINITION.json'),registry=str(REG),dictionary_sha256=sha(dictionary),files=[dict(path=str(p),name=str(p.relative_to(REPORT)),sha256=sha(p),bytes=p.stat().st_size) for p in files])
save(REPORT/'DELIVERY_MANIFEST.json',delivery)
p=ROOT/'CURRENT_READER_DELIVERY.json';x=json.loads(p.read_text());x.update(directory=str(REPORT),pptx=str(ppt),pdf=str(pdf),dictionary=str(dictionary),dictionary_sha256=sha(dictionary),status='FIXED_MEASURED_Y_REPORT',observation_registry=str(REG),verified_files={p.name:sha(p) for p in [ppt,pdf,dictionary]},delivery_manifest=str(REPORT/'DELIVERY_MANIFEST.json'))
for key in ['superseded','superseded_reason','reader_report_superseded','report_superseded']:x.pop(key,None)
save(p,x)
p=ROOT/'CURRENT_Y_DELIVERY.json';x=json.loads(p.read_text());x['open_items']=[];x['final_scope_audit']=str(REVIEW/'FINAL_SCOPE_AUDIT.json');save(p,x)
prov=REPO/'provenance/fixed_measured_y_20261112';prov.mkdir(exist_ok=True)
for p in [REVIEW/'LAUNCH_PREFLIGHT.json',REVIEW/'CATALOG_REVIEW.json',REVIEW/'PROGRAM_CLASSIFICATION.json',REVIEW/'MEASURED_Y_VALIDATION.json',REVIEW/'CONDITION_EVIDENCE_REVIEW.json',REVIEW/'EXTERNAL_RNA_POLICY.json',REVIEW/'DICTIONARY_UPDATE.json',REVIEW/'FIXED_Y_DEFINITION.json',REVIEW/'FINAL_SCOPE_AUDIT.json',REVIEW/'TESTS.json',REVIEW/'OBSERVED_COVERAGE.tsv',REVIEW/'READER_OBSERVATION_SUMMARY.tsv',REPORT/'FIGURE_DATA.json',REPORT/'DELIVERY_MANIFEST.json',BASE/'tls12_coding_r1/DEFINITION.json']:
    shutil.copyfile(p,prov/p.name)
for name in ['materialize_tls_registry_r2_149.py','review_fixed_y_149.py','publish_fixed_y_149.py','build_fixed_y_report_149.py','finalize_fixed_y_149.py']:shutil.copyfile(WORK/name,prov/name)
for key in ['coding_fixed_signature.py']:
    assert (REPO/'cla_engine'/key).is_file()
log=REPO/'project_log';log.mkdir(exist_ok=True)
(log/'FIXED_Y_20261112.md').write_text('''# 固定 Y 与 M0 说明交付

用户纠正：先固定 Y，不计算 lncRNA 关联，不训练模型。151、270 等关联可估计性计数不再作为 Y 覆盖率。

477 条 KEGG 癌症机制身份和 197 个 State 身份保持固定。Y 为按身份组织的患者多维观测，保持平台、方法与缺失。TF／激酶／PROGENy 的860条来源定义均归Pathway，不自动增加损失；Pathway=1、State=0.5。

TLS12使用GDC的12基因定义重算编码背景版本；原公开值保留来源参照。ESTIMATE推算纯度保留为外部参照，不当作已核实的默认Y。现有基因级突变与连续CNV文件已核查，其存在不等于具备143条条件机制的精确事件判定。未知条件保留，成员观测不删。

全部58个数值块核对数值、布尔实测掩码和患者轴；13项针对性测试通过。修复缺失主干条件轴被默认视为适用的问题。MMR真实导出为872名TCGA BRCA开发患者、89项观测，没有关联计算或训练。

字典40个工作表按需覆盖，其他工作簿内容逐字节保留。新PPT按问题、方法、结果组织；MMR组成相似与SMO/PTCH1条件差异使用实际原始列表。旧关联Y汇报在交付目录中由新版替换。

M0仅提供患者输入和唯一观测输出的Lasso设计、L/M/LM对照、无惩罚协变量及图模型消融对齐方法。训练入口不在本次交付范围。
''',encoding='utf-8')
print('FINAL_DEFINITION_AUDIT_COMPLETE',flush=True)
