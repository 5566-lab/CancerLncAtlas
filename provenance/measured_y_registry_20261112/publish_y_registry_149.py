from pathlib import Path
import sys,socket,json,zipfile,hashlib,math,shutil,xml.etree.ElementTree as ET
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');WORK=ROOT/'kegg_cancer_redefinition_20261111_r1'
BASE=WORK/'measured_y_definition_r1';REG=BASE/'observation_registry_r1';DATA=BASE/'member_measurements_r1';REPO=ROOT/'source_integration/repo'
sys.path.insert(0,str(REPO))
sys.path.append('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps')
from cla_engine.common import sha
import pandas as pd
from openpyxl import load_workbook,Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill

def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
save(REG/'PUBLISH_PREFLIGHT.json',{'target_host':'149','operation':'publish typed measured Y registry and dictionary','training':False,'lncrna_association':False})
reg=json.loads((REG/'COMPLETE.json').read_text())
for cancer,h in reg['receipt_hashes'].items():
    assert sha(REG/cancer/'COMPLETE.json')==h
    r=json.loads((REG/cancer/'COMPLETE.json').read_text())
    for name,digest in r['files'].items():assert sha(REG/cancer/name)==digest
    assert (REG/cancer/'ACCESSOR_VALIDATION.json').is_file()
old=Path(json.loads((ROOT/'CANONICAL_TARGETS_CURRENT.json').read_text())['reader_dictionary'])
before=sha(old);base=DATA/'Functional_pathway_State_dictionary_member_r1.xlsx'
assert before=='ffbef0b1507fe41ee5af0e4e2c69586d87b168c1c57d417a60895fa901072ffd'
NS='http://schemas.openxmlformats.org/spreadsheetml/2006/main'
RNS='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
PNS='http://schemas.openxmlformats.org/package/2006/relationships'
CNS='http://schemas.openxmlformats.org/package/2006/content-types'
def sheet_map(z):
    w=ET.fromstring(z.read('xl/workbook.xml'))
    rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    paths={x.attrib['Id']:x.attrib['Target'].lstrip('/') for x in rels}
    return {x.attrib['name']:paths[x.attrib['{'+RNS+'}id']] for x in w.find('{'+NS+'}sheets')}
checks=[]
with zipfile.ZipFile(old) as z1,zipfile.ZipFile(base) as z2:
    assert z2.testzip() is None
    a=sheet_map(z1);b=sheet_map(z2)
    changed=[n for n in a if n!='阅读说明' and z1.getinfo(a[n]).CRC!=z2.getinfo(b[n]).CRC]
    checks.extend({'sheet':n,'exact_xml_preserved':True} for n in a if n!='阅读说明' and n not in changed)
    if changed:
        s1=load_workbook(old,read_only=True,data_only=False);s2=load_workbook(base,read_only=True,data_only=False)
        for name in changed:
            rows1=list(s1[name].values);rows2=list(s2[name].values)
            assert len(rows1)==len(rows2),name
            numerical_roundtrips=0
            for row1,row2 in zip(rows1,rows2):
                row1=list(row1);row2=list(row2)
                while row1 and row1[-1] is None:row1.pop()
                while row2 and row2[-1] is None:row2.pop()
                assert len(row1)==len(row2),name
                for x,y in zip(row1,row2):
                    if x==y:continue
                    assert isinstance(x,(int,float)) and isinstance(y,(int,float)) and math.isclose(x,y,rel_tol=1e-14,abs_tol=1e-14),(name,x,y)
                    numerical_roundtrips+=1
            checks.append({'sheet':name,'cell_values_preserved':True,'excel_float_roundtrips':numerical_roundtrips})
        s1.close();s2.close()
print('PRESERVED_SHEETS_VERIFIED',len(checks),flush=True)

summary=[]
for cancer in ['BRCA','LUAD','COAD']:
    obs=pd.read_csv(REG/cancer/'OBSERVATIONS.tsv.gz',sep='\t',keep_default_na=False)
    for (cohort,kind,layer),g in obs.groupby(['cohort','kind','layer']):
        bound=g[g.bound_target_count.gt(0)]
        summary.append({'癌种':cancer,'队列':cohort,'观测类型':kind,'测量层':layer,'定义观测项数':len(g),
            '绑定主干或State的观测项数':len(bound),'其中有实测值':int(bound.patients_measured.gt(0).sum()),
            '计算背景独立于lncRNA的绑定项':int(bound.supervision_eligible.sum())})
summary=pd.DataFrame(summary);summary.to_csv(REG/'READER_OBSERVATION_SUMMARY.tsv',sep='\t',index=False)
methods=pd.read_csv(REG/'BRCA/SOURCE_READOUT_METHODS.tsv.gz',sep='\t',keep_default_na=False)
methods=methods[methods.measurement_layer.str.startswith('RNA_')]
methods=methods[['cohort','source_target_id','canonical_id','display_category','measurement_layer','applied_method','method_variant']]
methods.columns=['队列','来源定义ID','生物学身份ID','来源分类','测量层','评分方法','方法版本']
book=Workbook(write_only=True)
def sheet(name,rows):
    ws=book.create_sheet(name);ws.freeze_panes='A2';ws.sheet_view.showGridLines=False
    for i,row in enumerate(rows):
        if i:ws.append(list(row));continue
        cells=[]
        for value in row:
            c=WriteOnlyCell(ws,value=value);c.font=Font(name='Arial',bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='17365D');cells.append(c)
        ws.append(cells)
def frame(name,df):sheet(name,[df.columns.tolist(),*df.where(pd.notna(df),'').itertuples(index=False,name=None)])
frame('Y观测汇总',summary);frame('RNA读数方法',methods)
sheet('Y组织方式',[
 ['组成','定义'],['研究目标','癌种 × lncRNA × 精确通路；患者分子测量为Y，lncRNA关系由后续模型评价。'],
 ['生物学身份','477个KEGG癌症机制身份；197个State身份。不同测量不增加通路身份。'],
 ['Y[c,i,k]','癌种c中患者i在通路k下的一组有来源定义的观测值及实测掩码。'],
 ['观测身份','抗体、质谱蛋白、位点和来源评分分别有独立观测ID；同一观测可归属多条通路，数值保留一份。'],
 ['通路读数','明确等价定义的来源评分作为该身份的读数；蛋白列表相似本身不构成读数等价。'],
 ['成员观测','保留编码RNA、蛋白、磷酸化位点、CNV及启动子甲基化的逐患者值和测量定义。'],
 ['调控程序','TF／激酶描述成员状态或调控信息；PROGENy保留信号通路读数定义。三者来源分类为Pathway。'],
 ['外部RNA参照','ESTIMATE推算纯度和PanImmune TLS12公开分数保留原值及来源，计算背景与编码基因重算读数分开登记。'],
 ['平台','TCPA与CPTAC各保留自己的观测、单位和患者；不把不同平台列拼成同一活性数值。'],
 ['条件维度','成员实测与特定变异、感染或暴露是否存在分别记录；未知条件不是阴性。'],
 ['缺失','缺测记为缺失；已测零值和常量保留。测量覆盖不依赖lncRNA关联是否可估计。'],
 ['M0输出','每个癌种以患者为行，预测唯一观测项a；复用观测的通路不重复拟合同一输出。'],
 ['权重','Pathway总权重1.0，State总权重0.5；任务和观测权重显式指定，显示分类不自动创建损失。']])
sheet('M0对照设置',[
 ['项目','设置'],['预测公式','Yhat[i,a] = intercept[a] + C[i]gamma[a] + X_lnc[i]beta_lnc[a] + X_mRNA[i]beta_mRNA[a]'],
 ['单位','每癌种独立；患者为行，唯一测量或来源读数a为输出。同一a通过归属表关联多个通路。'],
 ['三组','M0-L：全部合格lncRNA；M0-M：mRNA；M0-LM：二者联合。'],
 ['协变量','年龄、性别、分期、独立纯度及适用的队列／平台项不受Lasso惩罚；纯度目标不使用自身作协变量。'],
 ['准入','不以lncRNA相关系数或FDR筛选；表达质控和缺失处理仅由训练患者确定。'],
 ['标签','同Y、同患者分折、同实测掩码与评价单位；缺失观测不贡献该输出损失。'],
 ['平台','不同平台观测分别建模；队列项不能使抗体值与质谱值具有相同单位。'],
 ['比较','临床基线、M0-L、M0-M、M0-LM；比较CCHHGT时保持输入组学、Y及患者一致。'],
 ['图约束对照','普通Ridge与相同输入的图正则Ridge对照；Lasso另报告所选RNA及稳定性。'],
 ['目标与输入','预测非RNA测量时排除其所在组学及直接派生输入；mRNA构成的RNA评分在mRNA组属于分数重建。'],
 ['关系结果','从留出患者上的预测增量、系数及稳定性评价lncRNA；这些结果不反向生成Y。'],
 ['损失','Pathway=1.0、State=0.5；TF／激酶辅助任务由显式配置决定。']])
updates=REG/'dictionary_updates.xlsx';book.save(updates)
destination=REG/'Functional_pathway_State_dictionary_v6.xlsx'
with zipfile.ZipFile(base) as original,zipfile.ZipFile(updates) as changes:
    omap=sheet_map(original);umap=sheet_map(changes)
    workbook=ET.fromstring(original.read('xl/workbook.xml'));rels=ET.fromstring(original.read('xl/_rels/workbook.xml.rels'));types=ET.fromstring(original.read('[Content_Types].xml'))
    sheets=workbook.find('{'+NS+'}sheets');replacement={}
    next_sheet=max(int(s.attrib['sheetId']) for s in sheets)+1
    next_rel=max(int(x.attrib['Id'].replace('rId','')) for x in rels)+1
    for name,srcpath in umap.items():
        if name in omap:dest=omap[name]
        else:
            dest=f'xl/worksheets/sheet{next_sheet}.xml';rid=f'rId{next_rel}'
            ET.SubElement(sheets,'{'+NS+'}sheet',{'name':name,'sheetId':str(next_sheet),'state':'visible','{'+RNS+'}id':rid})
            ET.SubElement(rels,'{'+PNS+'}Relationship',{'Type':RNS+'/worksheet','Target':'/'+dest,'Id':rid})
            ET.SubElement(types,'{'+CNS+'}Override',{'PartName':'/'+dest,'ContentType':'application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml'})
            next_sheet+=1;next_rel+=1
        replacement[dest]=changes.read(srcpath)
    replacement['xl/workbook.xml']=ET.tostring(workbook,encoding='utf-8')
    replacement['xl/_rels/workbook.xml.rels']=ET.tostring(rels,encoding='utf-8')
    replacement['[Content_Types].xml']=ET.tostring(types,encoding='utf-8')
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as target:
        for item in original.infolist():
            if item.filename not in replacement:target.writestr(item,original.read(item.filename))
        for path,content in replacement.items():target.writestr(path,content)
with zipfile.ZipFile(destination) as z:assert z.testzip() is None
check=load_workbook(destination,read_only=True,data_only=False)
assert sum(1 for _ in check['KEGG癌症主干'].values)==478
assert sum(1 for _ in check['Y观测汇总'].values)==len(summary)+1
assert sum(1 for _ in check['RNA读数方法'].values)==len(methods)+1
check.close();assert sha(old)==before
shutil.copyfile(destination,old);assert sha(destination)==sha(old)
report={'target_host':'149','dictionary':str(old),'sha256':sha(old),'previous_sha256':before,
    'typed_registry':str(REG),'registry_receipt_sha256':sha(REG/'COMPLETE.json'),'preserved_sheets':checks,
    'new_summary_rows':len(summary),'source_category_rows':len(methods),'representation':'PATHWAY_GROUPED_PATIENT_OBSERVATIONS',
    'training':False,'lncrna_association':False,'final_acceptance_complete':False}
save(REG/'DICTIONARY_UPDATE.json',report)
for name in ['CANONICAL_TARGETS_CURRENT.json','CURRENT_CANONICAL_REFERENCE.json','CURRENT_Y_DELIVERY.json','CURRENT_READER_DELIVERY.json']:
    p=ROOT/name;x=json.loads(p.read_text());x['Y_representation']='PATHWAY_GROUPED_PATIENT_OBSERVATIONS'
    x['member_measurement_components']=str(DATA);x['observation_registry']=str(REG)
    if name=='CANONICAL_TARGETS_CURRENT.json':x['reader_dictionary_sha256']=report['sha256']
    if name in ['CURRENT_Y_DELIVERY.json','CURRENT_READER_DELIVERY.json']:x['dictionary_sha256']=report['sha256']
    if name=='CURRENT_Y_DELIVERY.json':
        x['accepted_Y']=None;x['training_authorized']=False;x['lncrna_association_authorized']=False
        x['status']='TYPED_Y_REGISTRY_MATERIALIZED_REVIEW_REMAINING'
        x['open_items']=['Published TLS12 and ESTIMATE purity RNA-background lineage','Patient condition evidence review for conditional mechanisms','Reader report and goal-wide completion audit']
    if name=='CURRENT_READER_DELIVERY.json':
        x['verified_files'][old.name]=report['sha256'];x['dictionary_update_receipt']=str(REG/'DICTIONARY_UPDATE.json')
    save(p,x)
print('DICTIONARY_PUBLISHED',report['sha256'],flush=True)
