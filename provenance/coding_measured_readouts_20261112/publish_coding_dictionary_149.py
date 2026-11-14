import json,socket,sys,shutil
from pathlib import Path
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
WORK=ROOT/'kegg_cancer_redefinition_20261111_r1'
BASE=WORK/'measured_y_definition_r1'
REPO=ROOT/'source_integration/repo'
sys.path[:0]=[str(REPO),'/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps']
from cla_engine.relation_evidence_bindings import file_sha
import pandas as pd
from openpyxl import Workbook,load_workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill

def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
save(BASE/'CODING_DICTIONARY_PREFLIGHT.json',{'target_host':'149','operation':'update_measured_method_dictionary','training':False,'lncrna_association':False})
check=json.loads((BASE/'CODING_MEASUREMENT_INTEGRATION.json').read_text())
assert len(check['cases'])==3
for r in check['cases']:assert r['primary_pathways']==477 and r['state_identities']==197

doc=REPO/'docs/M0_RELATION_BASELINE.rst'
text=doc.read_text()
before='预测目标所在组学不作为该任务输入；目标直接组成值及其派生输入一并排除。\nRNA 评分可由 lncRNA 预测；mRNA 直接构成该评分时，对应 mRNA／联合组\n属于评分重建，不作为独立蛋白层预测证据。'
after='RNA 来源目标由 lncRNA 预测；mRNA 直接构成该评分时，mRNA／联合组属于\n评分重建。对于蛋白、磷酸化等其他目标，排除该目标所在组学及直接派生输入。\nRNA 分数重建与独立蛋白层预测分别评价。'
assert text.count(before)==1
text=text.replace(before,after)
before='lncRNA 系数、选择稳定性和留出患者预测增量是后续关系证据。\n它们是建模结果，不是定义 Y 的原料，也不自动证明因果调控。'
after='每条 lncRNA 在每个精确通路的各来源读数上，分别保留系数、选择稳定性及\n可估计的条件预测增量。三组模型的整体性能差异不能直接分配给某一条 lncRNA。\n这些建模结果不反向定义 Y；跨来源的方向和差异分别保留，不用最大绝对相关系数归并。'
assert text.count(before)==1
doc.write_text(text.replace(before,after))

ptr=json.loads((ROOT/'CANONICAL_TARGETS_CURRENT.json').read_text())
path=Path(ptr['reader_dictionary']);oldsha=file_sha(path)
coverage=[];summary=[];methods=[];external=[]
for cancer in ('BRCA','LUAD','COAD'):
    folder=BASE/'measurements_coding_r2'/cancer/'validation'
    rec=json.loads((folder/'COMPLETE.json').read_text())
    for name,h in rec['files'].items():assert file_sha(folder/name)==h
    cov=pd.read_csv(folder/'TARGET_MEASUREMENT_COVERAGE.tsv',sep='\t')
    cov.insert(0,'cancer',cancer);coverage.append(cov)
    md=pd.read_csv(folder/'READOUTS.tsv',sep='\t',keep_default_na=False)
    md.insert(0,'cancer',cancer)
    methods.append(md[md.coding_expression_basis_verified.astype(str).str.lower().eq('true')])
    external.append(md[md.measurement_definition.eq('PRESERVED_EXTERNAL_RNA_SOURCE')])
    for (cohort,role),g in cov.groupby(['cohort','analysis_role']):
        summary.append({'cancer':cancer,'cohort':cohort,'target_type':'Pathway' if role=='primary_pathway' else 'State',
            'patients_in_development_partition':int(g.patients_in_scope.iloc[0]),'fixed_identities':len(g),
            'identities_with_identity_readout':int(g.patients_with_identity_measurement.gt(0).sum()),
            'identities_with_member_readout':int(g.patients_with_member_measurement.gt(0).sum()),
            'identities_with_any_measurement':int(g.patients_with_any_measurement.gt(0).sum())})
coverage=pd.concat(coverage,ignore_index=True)
methodcols=['canonical_id','display_category','analysis_role','original_target_id','target_id','source_target_id','measurement_layer','applied_method','method_variant']
methodtable=pd.concat(methods,ignore_index=True)[methodcols].drop_duplicates()
external=pd.concat(external,ignore_index=True)
externalcols=['cohort','source_target_id','measurement_layer']
ext=external[externalcols].drop_duplicates()

book=Workbook(write_only=True)
def sheet(name,rows):
    ws=book.create_sheet(name);ws.freeze_panes='A2';ws.sheet_view.showGridLines=False
    for i,row in enumerate(rows):
        if i:
            ws.append(list(row));continue
        cells=[]
        for value in row:
            c=WriteOnlyCell(ws,value=value);c.font=Font(name='Arial',bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='17365D');cells.append(c)
        ws.append(cells)
def frame(name,df):
    sheet(name,[df.columns.tolist(),*df.where(pd.notna(df),'').itertuples(index=False,name=None)])

src=load_workbook(path,read_only=True,data_only=False)
guide=list(src['阅读说明'].values)
guide=[r for r in guide if r[0] not in ('RNA评分背景','RNA评分方法版本')]
guide.extend([
    ('RNA评分背景','先按GENCODE蛋白编码基因ID选择测量，再在TPM尺度合并同名基因；评分不读取lncRNA表达矩阵。'),
    ('RNA评分方法版本','PC_GENE_ID_BACKGROUND_R1：统一蛋白编码背景；需要参考参数的方法在指定参考患者中估计并冻结。固定模型被限制到蛋白编码成员时，保留其原始成员定义和方法版本。'),
    ('测量与通路身份','通路身份和成员独立于患者测量覆盖。缺失记录为缺失；零值与常量仍属测量。外部发布的RNA读数与本项目重算读数分别记录来源。')])
sheet('阅读说明',guide)
skip={'阅读说明','主干实测覆盖汇总','患者实测覆盖','RNA测量依赖','RNA读数方法','外部RNA读数','M0对照设置'}
for ws in src:
    if ws.title not in skip:sheet(ws.title,ws.values)
src.close()
frame('主干实测覆盖汇总',pd.DataFrame(summary));frame('患者实测覆盖',coverage)
frame('RNA读数方法',methodtable);frame('外部RNA读数',ext)
sheet('RNA测量依赖',[
    ['评分环节','输入与参考'],
    ['表达输入','19,912个蛋白编码基因ID；同名基因在TPM尺度合并为19,907个符号。缺失测量不补零。'],
    ['通路／EMT／ECM／SenMayo','在固定蛋白编码背景中计算ssGSEA；范围参数仅由参考患者确定。'],
    ['APM','固定蛋白编码背景，Gaussian GSVA参考核由参考患者确定；同一参考用于查询患者。'],
    ['RNAss／EREG.EXPss','发布权重与蛋白编码基因ID匹配，按固定参考缩放；方法版本区别于原始全成员评分。'],
    ['TF读数','保留编码靶基因的原始方向权重；基因与程序标准化参数由参考患者确定。'],
    ['PROGENy','固定官方top100权重，仅使用蛋白编码成员；各程序的原始及实际使用成员数记录于参考文件。'],
    ['外部RNA读数','CIBERSORT相对组成、RNA推导纯度及发布的TLS12读数保留其来源数值和原始方法，列于外部RNA读数。'],
    ['患者范围','BRCA、LUAD、COAD；TCGA与CPTAC分别保留来源参考。参考患者为固定外层折2—4，查询患者为折1。']])
sheet('M0对照设置',[
    ['项目','设置'],['拟合单位','癌种内患者；输出固定通路／State的各测量读数。'],
    ['三组RNA输入','全部合格lncRNA；全部合格mRNA；二者联合。'],
    ['协变量','年龄、性别、分期、独立纯度及适用的队列／平台项；不受Lasso惩罚。'],
    ['特征准入','训练患者内确定测量与表达质控，不以rho/FDR预筛选。'],
    ['线性估计器','Lasso；比较图正则化Ridge时，另配相同输入的普通Ridge。'],
    ['RNA来源目标','lncRNA组做预测；直接使用评分组成mRNA的组别单列为分数重建。'],
    ['其他组学目标','排除目标所在组学和直接派生输入；比较时固定Y、患者和协变量。'],
    ['单条lncRNA证据','各来源读数分别保留系数、稳定性与条件预测增量；整体模型改善不直接归于单条lncRNA。'],
    ['任务权重','Pathway=1.0、State=0.5；TF／激酶／PROGENy分类不自动增加损失。']])
temp=BASE/'Functional_pathway_State_dictionary_coding_r2.xlsx';book.save(temp)
verify=load_workbook(temp,read_only=True,data_only=False)
expected={'KEGG癌症主干':478,'State':198,'来源蛋白组成':157039,'成员状态证据':5041,'患者实测覆盖':4045,'RNA读数方法':len(methodtable)+1}
counts={name:sum(1 for _ in verify[name].values) for name in expected}
assert counts==expected,(counts,expected)
verify.close();assert file_sha(path)==oldsha
shutil.copyfile(temp,path)
assert file_sha(temp)==file_sha(path)
receipt={'target_host':'149','dictionary':str(path),'sha256':file_sha(path),'previous_sha256':oldsha,'sheet_rows':counts,
         'measurement_root':str(BASE/'measurements_coding_r2'),'Y_complete':False,'model_training':False,'lncrna_association_computation':False}
save(BASE/'CODING_DICTIONARY_UPDATE.json',receipt)
for name in ('CANONICAL_TARGETS_CURRENT.json','CURRENT_Y_DELIVERY.json','CURRENT_CANONICAL_REFERENCE.json'):
    p=ROOT/name;r=json.loads(p.read_text())
    key='measurement_components' if name=='CURRENT_Y_DELIVERY.json' else 'measured_readout_components'
    r[key]=str(BASE/'measurements_coding_r2')
    if name=='CANONICAL_TARGETS_CURRENT.json':r['reader_dictionary_sha256']=receipt['sha256']
    if name=='CURRENT_Y_DELIVERY.json':
        r['dictionary_sha256']=receipt['sha256'];r['coding_readout_components']=str(BASE/'protein_coding_readouts')
        r['open_items']=['External RNA definitions: CIBERSORT, RNA-derived purity, published TLS12',
           'scalar-versus-multidimensional Y contract','complete member-level measurement integration and applicability','final Y acceptance and reader report alignment']
    save(p,r)
rp=ROOT/'CURRENT_READER_DELIVERY.json'
if rp.exists():
    r=json.loads(rp.read_text());r['dictionary_sha256']=receipt['sha256']
    r.setdefault('verified_files',{})[path.name]=receipt['sha256']
    r['dictionary_update_receipt']=str(BASE/'CODING_DICTIONARY_UPDATE.json')
    save(rp,r)
print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
