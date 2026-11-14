from pathlib import Path
import socket,sys,json,hashlib
import pandas as pd
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'state_groups_20261114_r1';N=R/'canonical_targets_r11_state_groups_20261114_r1';G=O/'observation_registry_r4';D=O/'deliverables';D.mkdir(exist_ok=True)
sys.path.append('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps')
import openpyxl
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
def sj(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
sj(O/'DICTIONARY_PREFLIGHT.json',dict(target_host='149',operation='Publish State collection dictionary v8'))
validation=json.loads((O/'VALIDATION.json').read_text());assert validation['state_after']==48
old=R/'protein_similarity_integration_20261111_r1/deliverables/Functional_pathway_State_dictionary_v7.xlsx';assert sha(old)=='e92c613ab28909eeb0a828d725165aa0fbeaad1b693c61f5352db236547055ad'
top=rd(N/'STATE_TOP_LEVEL_TARGETS.tsv');members=rd(N/'STATE_GROUP_MEMBERSHIP.tsv');master=rd(N/'TARGET_MASTER.tsv')
state=pd.DataFrame({'canonical_id':top.canonical_id,'State名称':top.display_name,'中文名称':top.display_name_zh,'条目类型':top.entity_type,'详情子项数':top.detail_component_count})
details=members.merge(rd(N/'STATE_DETAIL_TARGETS.tsv')[['canonical_id','source_database','authority_id']],left_on='component_id',right_on='canonical_id',validate='one_to_one').drop(columns='canonical_id')
details=details.rename(columns={'parent_name_zh':'上层名称','parent_name':'上层英文名','component_name':'子项名称','state_family':'类型'})
counts=pd.DataFrame([['细胞组成',91,1,91],['突变特征谱（SBS）',60,1,60],['其余 State',46,46,0],['State 合计',197,48,151],['Pathway',476,476,0],['主列表合计',673,524,151]],columns=['项目','调整前入口数','调整后入口数','转入详情的子项数'])
definitions=rd(N/'STATE_SCORE_DEFINITIONS.tsv');component_defs=definitions[definitions.canonical_id.isin(members.component_id)].copy();component_defs['parent_id']=component_defs.canonical_id.map(members.set_index('component_id').parent_id)
summary=[]
for ca in ['BRCA','LUAD','COAD']:
    rec=json.loads((G/ca/'COMPLETE.json').read_text());summary.append(dict(癌种=ca,主干通路数=476,State顶层条目数=48,总目标条目数=524,保留的State来源身份数=197,唯一观测数=rec['observations'],通路观测绑定数=rec['target_bindings'],TCGA患者=rec['patients']['TCGA'],CPTAC患者=rec['patients']['CPTAC']))
replacements={'State':state,'当前Y目标':rd(G/'BRCA/TARGETS.tsv'),'患者实测覆盖':pd.concat([rd(G/ca/'COVERAGE.tsv') for ca in ['BRCA','LUAD','COAD']],ignore_index=True),'Y观测汇总':pd.DataFrame(summary),'Y组织方式':pd.DataFrame([
    ['主列表','476 条 Pathway 与 48 个 State，共 524 个顶层条目'],
    ['细胞组成','一个入口；91 个来源身份在详情中按细胞类型和算法展开'],
    ['突变特征谱（SBS）','一个入口；60 个 SBS 特征在详情中展开'],
    ['其余 State','46 个条目保留原定义'],
    ['数值组织','每个上层条目包含原始多维观测；各子项分别保留数值、方法、单位和缺失'],
    ['通路成员','J=1 的 Hedgehog 来源机制仍组成一个共同下游模块'],
    ['来源机制','原始变异机制和适用条件作为详情信息保留'],
    ['模型输入','lncRNA 不参与 Y 定义；Y 使用原始患者分子观测']
],columns=['项目','定义'])}
book=openpyxl.load_workbook(old,read_only=True,data_only=True);new=openpyxl.Workbook(write_only=True);sheet_counts={}
def start(name,head):
    ws=new.create_sheet(name);ws.freeze_panes='A2';cells=[]
    for h in head:
        cell=WriteOnlyCell(ws,value=h);cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='127F88');cells.append(cell)
    ws.append(cells)
    for j,h in enumerate(head,1):ws.column_dimensions[openpyxl.utils.get_column_letter(j)].width=min(56,max(18,len(str(h))+4))
    return ws
def emit(name,frame):
    ws=start(name,list(frame))
    for row in frame.itertuples(index=False,name=None):ws.append(list(row))
    sheet_counts[name]=len(frame)
for source in book:
    if source.title in replacements:emit(source.title,replacements[source.title])
    else:
        rows=source.iter_rows(values_only=True);head=next(rows);ws=start(source.title,head);n=0
        for row in rows:
            if source.title=='阅读说明':
                row=[v.replace('197 个 State 身份','48 个 State 顶层条目').replace('197 个 State','48 个 State 顶层条目').replace('673 个目标','524 个顶层目标').replace('字典 v7','字典 v8') if isinstance(v,str) else v for v in row]
            ws.append(list(row));n+=1
        if source.title=='阅读说明':
            for text in ['State 主列表为 48 项：细胞组成、SBS 突变特征谱与其余 46 项。','91 个细胞组成身份及 60 个 SBS 特征列于 State分组子项，完整方法见 State子项评分来源。']:
                ws.append(['State 分组',text]+[None]*max(0,len(head)-2));n+=1
        sheet_counts[source.title]=n
    print('SHEET',source.title,sheet_counts[source.title],flush=True)
for name,frame in [('State分组子项',details),('State压缩汇总',counts),('State子项评分来源',component_defs)]:emit(name,frame)
book.close();dest=D/'Functional_pathway_State_dictionary_v8.xlsx';new.save(dest)
check=openpyxl.load_workbook(dest,read_only=True,data_only=True)
assert len(list(check['State'].values))-1==48
assert len(list(check['State分组子项'].values))-1==151
assert len(list(check['当前Y目标'].values))-1==524
assert len(list(check['KEGG癌症主干'].values))-1==476
check.close()
state.to_csv(D/'当前48个State条目.tsv',sep='\t',index=False);details.to_csv(D/'State分组与151个子项.tsv',sep='\t',index=False);counts.to_csv(D/'State压缩汇总.tsv',sep='\t',index=False)
sj(O/'DICTIONARY_VALIDATION.json',dict(target_host='149',path=str(dest),sha256=sha(dest),sheets=sheet_counts,primary_pathways=476,state_top_level_count=48,state_detail_components=151,total_targets=524))
print('V8_DICTIONARY_VERIFIED',flush=True)
