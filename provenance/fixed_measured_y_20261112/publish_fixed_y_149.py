from pathlib import Path
import sys, socket, json, zipfile, shutil, xml.etree.ElementTree as ET
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');WORK=ROOT/'kegg_cancer_redefinition_20261111_r1'
BASE=WORK/'measured_y_definition_r1';REG=BASE/'observation_registry_r2';REVIEW=BASE/'final_definition_review_r1';REPO=ROOT/'source_integration/repo'
CAT=ROOT/'canonical_targets_r9_cancer_evidence_20261111_r6'
sys.path.insert(0,str(REPO));sys.path.append('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps')
from cla_engine.common import sha
import pandas as pd
from openpyxl import load_workbook,Workbook
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
save(REVIEW/'PUBLISH_PREFLIGHT.json',dict(target_host='149',operation='publish fixed molecular Y and reader dictionary',training=False,lncrna_association=False))
validated=json.loads((REVIEW/'MEASURED_Y_VALIDATION.json').read_text());assert validated['all_numeric_blocks_verified'] and validated['registry_sha256']==sha(REG/'COMPLETE.json')
roles=pd.read_csv(CAT/'SOURCE_TASK_ROLES.tsv',sep='\t').fillna('');program=roles[roles.source_target_id.str.startswith(('PROGENY:','DOROTHEA:','KINASE-'))]
assert len(program)==860 and program.display_category.eq('Pathway').all()
save(REVIEW/'PROGRAM_CLASSIFICATION.json',dict(target_host='149',source_rows=860,TF=271,kinase=575,PROGENy=14,all_category='Pathway',independent_losses_enabled=False))
pointer=json.loads((ROOT/'CANONICAL_TARGETS_CURRENT.json').read_text());old=Path(pointer['reader_dictionary']);before=sha(old)
assert before=='9bec9976dd203e7ef5fd943ce90e10e7739458cfa51497905743d584fab879b3'
w=load_workbook(old,read_only=True,data_only=False)
intro=list(w['阅读说明'].values);organization=list(w['Y组织方式'].values);m0=list(w['M0对照设置'].values)
w.close()
seen=set();intro=[row for row in intro if not (row[0] in seen or seen.add(row[0]))]
intro.append(('TLS12 方法','TLS12_GDC_SSGSEA_PC_RAW_V1：GDC PanImmune 的12个趋化因子；固定19907个蛋白编码符号背景，逐患者计算未归一化 ssGSEA。公开原分数另列。'))
intro.append(('来源纯度','ESTIMATE 推算纯度保留为外部来源读数；重算的免疫／基质分数与经过独立测量的纯度分别记录。'))
organization.append(('TLS12 编码读数','CCL2、CCL3、CCL4、CCL5、CCL8、CCL18、CCL19、CCL21、CXCL9、CXCL10、CXCL11、CXCL13；统一编码背景 ssGSEA，保留独立方法版本。'))
organization.append(('数据范围','BRCA、LUAD、COAD 开发患者：参考折2—4、查询折1；外层折0患者保持封存。需要参考拟合的参数只由参考患者确定。'))
summary=[];methods=[]
for cancer in ['BRCA','LUAD','COAD']:
    obs=pd.read_csv(REG/cancer/'OBSERVATIONS.tsv.gz',sep='\t',keep_default_na=False)
    for (cohort,kind,layer),g in obs.groupby(['cohort','kind','layer']):
        bound=g[g.bound_target_count.gt(0)];eligible=bound[bound.supervision_eligible]
        summary.append([cancer,cohort,kind,layer,len(g),len(bound),int(bound.patients_measured.gt(0).sum()),len(eligible)])
    method=pd.read_csv(REG/cancer/'SOURCE_READOUT_METHODS.tsv.gz',sep='\t',keep_default_na=False)
    method=method[method.measurement_layer.str.startswith('RNA_')].copy();method.insert(0,'cancer',cancer)
    methods.append(method[['cancer','cohort','source_target_id','canonical_id','display_category','measurement_layer','applied_method','method_variant']])
methods=pd.concat(methods,ignore_index=True)
pd.DataFrame(summary,columns=['cancer','cohort','kind','layer','observations','bound','measured','eligible']).to_csv(REVIEW/'READER_OBSERVATION_SUMMARY.tsv',sep='\t',index=False)
book=Workbook(write_only=True)
def sheet(name,rows):
    ws=book.create_sheet(name);ws.freeze_panes='A2';ws.sheet_view.showGridLines=False
    ws.column_dimensions['A'].width=28;ws.column_dimensions['B'].width=105
    for row in rows:ws.append(list(row))
sheet('阅读说明',intro);sheet('Y组织方式',organization);sheet('M0对照设置',m0)
sheet('Y观测汇总',[['癌种','队列','观测类型','测量层','定义观测项','绑定主干或State','其中有实测值','已核实计算依赖的绑定项'],*summary])
sheet('RNA读数方法',[['癌种','队列','来源定义ID','生物学身份ID','来源分类','测量层','评分方法','方法版本'],*methods.itertuples(index=False,name=None)])
sheet('外部RNA读数',[
 ['读数','来源与含义','在Y中的角色'],
 ['PanImmune TLS12 公开分数','GDC PanImmune / Wolf Chemokine12_score；保留原始数值和方法来源','外部来源参照'],
 ['TLS12 编码背景版本','GDC 相同12成员，固定编码基因背景，未归一化 ssGSEA','有独立方法版本的患者观测'],
 ['ESTIMATE 推算纯度','保留原始发布／原生推算值；不将重算 ESTIMATE 分数视为经校准纯度','外部来源参照'],
 ['ESTIMATE 免疫／基质分数','按统一编码背景重算，保留方法版本和组成','编码 RNA 患者观测'],
 ['DNA／病理来源纯度','按独立测量来源记录','各自定义的 State 观测；适用时作为独立协变量']])
updates=REVIEW/'dictionary_updates.xlsx';book.save(updates)
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main';RNS='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
def sheetmap(z):
    wb=ET.fromstring(z.read('xl/workbook.xml'));rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    paths={x.attrib['Id']:x.attrib['Target'].lstrip('/') for x in rels}
    return {x.attrib['name']:paths[x.attrib['{'+RNS+'}id']] for x in wb.find('{'+NS+'}sheets')}
dest=REVIEW/'Functional_pathway_State_dictionary_v6.xlsx'
with zipfile.ZipFile(old) as original,zipfile.ZipFile(updates) as change:
    omap=sheetmap(original);cmap=sheetmap(change);assert set(cmap)<=set(omap)
    replacements={omap[name]:change.read(path) for name,path in cmap.items()}
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
        for item in original.infolist():out.writestr(item,replacements.get(item.filename,original.read(item.filename)))
with zipfile.ZipFile(old) as original,zipfile.ZipFile(dest) as out:
    assert out.testzip() is None
    for item in original.infolist():
        if item.filename not in replacements:assert original.read(item.filename)==out.read(item.filename)
assert sha(old)==before;shutil.copyfile(dest,old)
save(REVIEW/'DICTIONARY_UPDATE.json',dict(target_host='149',dictionary=str(old),previous_sha256=before,sha256=sha(old),updated_sheets=list(cmap),other_workbook_parts_byte_identical=True,registry=str(REG)))
# Update methodological text; original published scores remain source-specific references.
p=REPO/'docs/MEASURED_Y_REPRESENTATION.rst';text=p.read_text();anchor='ESTIMATE 推算纯度与 PanImmune TLS12 公开读数保留为外部参照'
assert anchor in text;text=text[:text.index(anchor)]+'''TLS12 新版本采用 GDC PanImmune / Wolf 的12个趋化因子定义，使用固定
19907个蛋白编码符号背景，逐患者计算未归一化 ssGSEA（alpha=0.25）。
方法标识为 TLS12_GDC_SSGSEA_PC_RAW_V1。该版本保留完整背景缺失掩码，
并不声称数值复现公开 PanImmune 分数。公开原分数仍作为来源参照。

ESTIMATE 推算纯度保留为外部参照，默认监督观测不纳入其原始 RNA
依赖尚未明确的值。重算的 ESTIMATE 免疫／基质分数保持自己的定义，
不套用旧转换式并称为经过验证的纯度比例。其他已核实 State 不受影响。

固定交付的数据范围为 BRCA、LUAD、COAD 开发患者。固定的是通路身份、
观测及其方法、来源映射和缺失定义；需要估计的标准化参数仍必须在每次
建模的训练患者内拟合。通路测量集合不依赖任何 lncRNA 属性。
''';p.write_text(text,encoding='utf-8')
save(REVIEW/'FIXED_Y_DEFINITION.json',dict(target_host='149',definition_version='KEGG_CANCER_MEASURED_Y_20261112_R2',representation='PATHWAY_GROUPED_PATIENT_OBSERVATIONS',registry=str(REG),registry_receipt_sha256=sha(REG/'COMPLETE.json'),catalog=str(CAT),primary_pathways=477,state_identities=197,cancers=['BRCA','LUAD','COAD'],definition_independent_of_lncRNA=True,training_authorized=False,lncrna_association_authorized=False,reference_folds=[2,3,4],query_folds=[1],sealed_outer_fold=0,external_reference_policy=str(REVIEW/'EXTERNAL_RNA_POLICY.json'),condition_policy=str(REVIEW/'CONDITION_EVIDENCE_REVIEW.json'),validation=str(REVIEW/'MEASURED_Y_VALIDATION.json')))
for name in ['CANONICAL_TARGETS_CURRENT.json','CURRENT_CANONICAL_REFERENCE.json','CURRENT_Y_DELIVERY.json','CURRENT_READER_DELIVERY.json']:
    p=ROOT/name;x=json.loads(p.read_text());x['observation_registry']=str(REG)
    if name=='CANONICAL_TARGETS_CURRENT.json':x['reader_dictionary_sha256']=sha(old)
    if name in ['CURRENT_Y_DELIVERY.json','CURRENT_READER_DELIVERY.json']:x['dictionary_sha256']=sha(old)
    if name=='CURRENT_Y_DELIVERY.json':
        x['accepted_Y']=str(REVIEW/'FIXED_Y_DEFINITION.json');x['status']='FIXED_MEASURED_Y_DEFINITION'
        x['open_items']=['Reader report replacement and final delivery audit'];x['training_authorized']=False;x['lncrna_association_authorized']=False
    if name=='CURRENT_READER_DELIVERY.json':x['verified_files'][old.name]=sha(old);x['dictionary_update_receipt']=str(REVIEW/'DICTIONARY_UPDATE.json')
    save(p,x)
print('FIXED_Y_DICTIONARY_PUBLISHED',sha(old),flush=True)
