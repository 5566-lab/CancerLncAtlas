from pathlib import Path
import json,hashlib,socket,sys,shutil,tarfile
import pandas as pd
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'fixed_y_remake_20261114_r1';B=O/'Y_bundle';D=O/'deliverables';D.mkdir(exist_ok=True)
sys.path.append('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps')
import openpyxl
from openpyxl.styles import Font,PatternFill
from openpyxl.cell import WriteOnlyCell
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
sj(O/'PACKAGE_PREFLIGHT.json',dict(target_host='149',operation='Package fixed Y and scientific delivery'))
receipt=json.loads((B/'COMPLETE.json').read_text())
for name,h in receipt['files'].items():assert sha(B/name)==h,name
readme='''CancerLncAtlas：固定 Y 数据包

目录包含 476 条 Pathway 和 48 个 State，共 524 个顶层条目。
细胞组成 91 个身份、SBS 60 个特征作为两组的详情子项保留。

Y 的每一列是明确的分子观测或来源评分；一条通路拥有若干这样的列。
每个队列分别保存数值、缺失、方法与单位，不构造跨平台平均分。
本包仅包含与当前 524 个目标绑定的观测。未绑定的来源池读数不进入本包。

cohorts/TCGA_BRCA 等六个目录：
  Y_VALUES.parquet              患者 × 唯一观测，保留原值
  Y_OBSERVED.parquet            相同形状的实测掩码，零值可为真实观测
  OBSERVATIONS.tsv.gz           每列的分子/方法/单位/来源/监督资格
  TARGET_OBSERVATIONS.tsv.gz    通路或 State 与观测列的对应关系
  TARGET_APPLICABILITY.parquet  患者 × 524 目标，值为 1/0/缺失
  STATE_CHILD_OBSERVATIONS.tsv.gz  两个 State 组的原始子项与观测
  PATIENTS.tsv                  开发集患者和原参考/查询分组

TARGETS.tsv 为 524 个目标的名称和 ID。
STATE_GROUP_MEMBERSHIP.tsv 为两个 State 组及 151 个子项。
COHORT_SUMMARY.tsv 为患者数和覆盖结果。
TARGET_COVERAGE.tsv 为逐通路、逐 State、逐队列的实际覆盖。
LAYER_COVERAGE.tsv 区分成员实测和来源读数。

读取：把 reader.py 放入 Python 路径，使用
from reader import load_fixed_target
values, mask, applicable, metadata = load_fixed_target(
    'Y_bundle', 'TCGA_BRCA', 'CLA-P-922863C8FB5E9DF9')

for_supervision=True 时，掩码同时要求：实测存在、来源允许监督、目标适用条件为1。
机制条件未知不会变为阳性或阴性；原始观测依然保留。

本轮为 BRCA、LUAD、COAD，TCGA/CPTAC 共 1,910 名开发集患者。
全部476条Pathway都有至少一种观测；335条在当前患者条件下有非空的监督观测。
其余141条保留分子读数，其事件适用条件未知。
State实测覆盖为TCGA每癌种45/48、CPTAC每癌种33/48。
去分化、上皮/间质程序、癌相关ECM三个概念条目没有独立读数。

主任务仍为“癌种 × lncRNA × 精确通路”。
lncRNA不参与本包Y的构造；关系由后续模型比较评估。
Pathway和State任务权重保持1.0、0.5。
'''
(B/'使用说明.txt').write_text(readme,encoding='utf8');shutil.copy2(B/'使用说明.txt',D/'Y定义与使用说明.txt')
wb=openpyxl.Workbook(write_only=True)
frames={'Y条目':rd(B/'TARGETS.tsv'),'队列汇总':rd(B/'COHORT_SUMMARY.tsv'),'逐目标覆盖':rd(B/'TARGET_COVERAGE.tsv'),'组学与观测类型':rd(B/'LAYER_COVERAGE.tsv'),'通路分层覆盖':rd(B/'PATHWAY_LAYER_COVERAGE.tsv'),'State分组子项':rd(B/'STATE_GROUP_MEMBERSHIP.tsv'),'来源池与正式Y':rd(B/'POOL_ROUTING.tsv')}
for name,frame in frames.items():
    ws=wb.create_sheet(name);ws.freeze_panes='A2';cells=[]
    for header in frame:
        c=WriteOnlyCell(ws,value=header);c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='127F88');cells.append(c)
    ws.append(cells)
    for j,header in enumerate(frame,1):ws.column_dimensions[openpyxl.utils.get_column_letter(j)].width=30 if 'name' in header else 23
    for row in frame.itertuples(index=False,name=None):ws.append(list(row))
wb.save(D/'Y定义与实测覆盖_20261114.xlsx')
receipt['files']={str(p.relative_to(B)):sha(p) for p in B.rglob('*') if p.is_file() and p!=B/'COMPLETE.json'};sj(B/'COMPLETE.json',receipt)
archive=D/'CancerLncAtlas_fixed_Y_476Pathway_48State_20261114.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=5) as tar:tar.add(B,arcname='Y_bundle')
with tarfile.open(archive,'r:gz') as tar:
    files={m.name:m for m in tar.getmembers() if m.isfile()}
    for name,h in receipt['files'].items():assert hashlib.sha256(tar.extractfile(files['Y_bundle/'+name]).read()).hexdigest()==h,name
    assert hashlib.sha256(tar.extractfile('Y_bundle/COMPLETE.json').read()).hexdigest()==sha(B/'COMPLETE.json')
sj(O/'PACKAGE_VALIDATION.json',dict(target_host='149',bundle=str(B),bundle_receipt_sha256=sha(B/'COMPLETE.json'),archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,files_verified=len(receipt['files']),primary_pathways=476,state_top_level_count=48,total_targets=524,patients=1910))
print('Y_ARCHIVE_VERIFIED',archive.stat().st_size,flush=True)
