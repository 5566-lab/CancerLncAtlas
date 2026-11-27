from pathlib import Path
import socket,sys,json,hashlib,collections
import pandas as pd
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'exact_member_merge_20261114_r1';N=R/'canonical_targets_r10_exact_members_20261114_r1';G=O/'observation_registry_r3';B=W/'measured_y_definition_r1'
OUT=O/'deliverables';OUT.mkdir(exist_ok=True)
sys.path.append('/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps')
import openpyxl
from openpyxl.cell import WriteOnlyCell
from openpyxl.styles import Font,PatternFill
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
def sj(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
sj(O/'DICTIONARY_PREFLIGHT.json',dict(target_host='149',operation='publish merged scientific dictionary',source_version='v6',target_version='v7'))
definition=json.loads((O/'GROUP_DEFINITION.json').read_text());KEEP=definition['canonical_id'];IDS=[x['canonical_id'] for x in definition['source_mechanisms']];DROP=next(x for x in IDS if x!=KEEP);NAME=definition['display_name'];CN=definition['display_name_zh']
old=R/'protein_similarity_integration_20261111_r1/deliverables/Functional_pathway_State_dictionary_v6.xlsx';assert sha(old)=='78c90fe84a4b8789288a6c4ba02cdbaf7ac43b83ff922cd1bad7448d12176103'
book=openpyxl.load_workbook(old,read_only=True,data_only=True);new=openpyxl.Workbook(write_only=True)
coverage=pd.concat([rd(G/c/'COVERAGE.tsv') for c in ['BRCA','LUAD','COAD']],ignore_index=True)
mc=rd(B/'member_measurements_r1/READER_MEMBER_COVERAGE.tsv');mc=mc[mc.canonical_id.ne(DROP)].copy();mc['case']=mc.cohort+'_'+mc.cancer
ms=rd(B/'member_measurements_r1/READER_MEMBER_SUMMARY.tsv')
for j,r in ms.iterrows():
    rows=mc[(mc['case']==r['case'])&(mc.layer==r.layer)]
    ms.loc[j,'pathways_with_member_measurement']=int((rows.measured_member_features.astype(int)>0).sum())
    ms.loc[j,'pathways_with_ambiguous_group_measurement']=int((rows.patients_with_ambiguous_group_measurement.astype(int)>0).sum())
ms.to_csv(O/'READER_MEMBER_SUMMARY.tsv',sep='\t',index=False);mc.to_csv(O/'READER_MEMBER_COVERAGE.tsv',sep='\t',index=False)
summary=[]
for c in ['BRCA','LUAD','COAD']:
    r=json.loads((G/c/'COMPLETE.json').read_text())
    summary.append(dict(癌种=c,主干通路数=r['primary_pathways'],State身份数=r['state_identities'],总目标数=r['targets'],唯一观测数=r['observations'],通路观测绑定数=r['target_bindings'],TCGA患者=r['patients']['TCGA'],CPTAC患者=r['patients']['CPTAC']))
replacements={'主干组成关联':rd(N/'BACKBONE_ALIGNMENT.tsv'),'代表性主干关联':rd(N/'BACKBONE_ALIGNMENT_TOP3.tsv'),'来源蛋白组成':rd(N/'INTEGRATED_PROTEIN_COMPOSITION.tsv.gz'),'来源ID绑定':rd(N/'SOURCE_ID_BINDINGS.tsv'),'主干实测覆盖汇总':ms,'患者实测覆盖':coverage,'成员测量汇总':ms,'成员测量覆盖':mc,'Y观测汇总':pd.DataFrame(summary)}
replacements['Y组织方式']=pd.DataFrame([
    ['组织单位','476 个主干通路观测组与 197 个 State 身份'],
    ['完全相同成员','按原始蛋白集合 J=1 合并；相同观测 ID 只绑定一次'],
    ['Hedgehog 组',CN],
    ['来源定义','N00017 与 N00010 的名称、原始 ID、条件证据分别保留'],
    ['组的适用范围','共同成员观测不要求某一个特定上游变异'],
    ['变异机制','来源事件未知保持未知；不因合并而判定患者为某种变异驱动'],
    ['测量数值','按平台、组学、方法保留原始值和实测掩码'],
    ['来源评分','不同定义和方法的来源读数可展开查看'],
    ['模型输入','lncRNA 作为后续预测输入，不参与通路归组或 Y 定义']
],columns=['项目','定义'])
def start(name,headers):
    ws=new.create_sheet(name);ws.freeze_panes='A2'
    cells=[]
    for h in headers:
        cell=WriteOnlyCell(ws,value=h);cell.font=Font(bold=True,color='FFFFFF');cell.fill=PatternFill('solid',fgColor='127F88');cells.append(cell)
    ws.append(cells)
    for j,h in enumerate(headers,1):ws.column_dimensions[openpyxl.utils.get_column_letter(j)].width=min(54,max(16,len(str(h))+3))
    return ws
counts={}
for source in book:
    original_name=source.title;name='来源机制适用条件' if original_name=='通路适用条件' else original_name
    if original_name in replacements:
        frame=replacements[original_name];ws=start(name,list(frame));count=0
        for row in frame.itertuples(index=False,name=None):ws.append(list(row));count+=1
    else:
        itr=source.iter_rows(values_only=True);headers=list(next(itr));ws=start(name,headers);count=0
        idx={str(h):i for i,h in enumerate(headers)}
        for original_row in itr:
            row=list(original_row);cid=row[idx['canonical_id']] if 'canonical_id' in idx else (row[idx['生物学身份ID']] if '生物学身份ID' in idx else None)
            if original_name in ['KEGG癌症主干','Pathway与机制'] and cid==DROP:continue
            # Definitions and event conditions retain source identity, while their grouping link is updated.
            for col,j in idx.items():
                rewrite=('canonical_id' in col and not col.startswith(('previous','original','source_definition')) or col=='生物学身份ID')
                if original_name=='来源定义与身份':rewrite=col=='integrated_canonical_id'
                if original_name in ['通路适用条件','正式跨库关系','功能范围关系']:rewrite=False
                if rewrite and row[j]==DROP:row[j]=KEEP
            if original_name in ['KEGG癌症主干','Pathway与机制'] and cid==KEEP:
                for col in ['通路名称','display_name','名称']:
                    if col in idx:row[idx[col]]=NAME
                for col,value in {'NETWORK_ID':'N00017;N00010','机制类型':'Shared member module','机制定义':'Shared 29-protein downstream observation module of SMO/PTCH1 mechanisms','展开反应':'Source-specific reactions are retained under N00017 and N00010','适用条件':'共同成员观测；不要求特定上游变异','条件类型':'shared_member_observation','所需条件证据':'[]','entity_type':'pathway_observation_group','merged_definition_count':2}.items():
                    if col in idx:row[idx[col]]=value
            if original_name=='阅读说明':
                row=[v.replace('477','476') if isinstance(v,str) and not v.startswith(('http','/dsk2')) else v for v in row]
            ws.append(row);count+=1
    counts[name]=count;print('SHEET',name,count,flush=True)
for name,frame in [('J1相同成员合并',rd(N/'EXACT_MEMBER_GROUP_MEMBERS.tsv').query('merged == True')),('当前Y目标',rd(G/'BRCA/TARGETS.tsv')),('Hedgehog来源机制',rd(N/'GROUP_SOURCE_MECHANISMS.tsv'))]:
    ws=start(name,list(frame))
    for row in frame.itertuples(index=False,name=None):ws.append(list(row))
    counts[name]=len(frame)
book.close();dest=OUT/'Functional_pathway_State_dictionary_v7.xlsx';new.save(dest)
check=openpyxl.load_workbook(dest,read_only=True,data_only=True)
rows=list(check['KEGG癌症主干'].values);head=list(rows[0]);assert len(rows)-1==476
assert sum(r[head.index('生物学身份ID')]==KEEP for r in rows[1:])==1
assert not any(r[head.index('生物学身份ID')]==DROP for r in rows[1:])
assert len(list(check['当前Y目标'].values))-1==673
df=list(check['来源定义与身份'].values);h=list(df[0]);sel=[r for r in df[1:] if r[h.index('node_id')] in IDS]
assert len(sel)==2 and all(r[h.index('integrated_canonical_id')]==KEEP for r in sel)
check.close()
pd.DataFrame([dict(统一ID=KEEP,显示名称=CN,英文名称=NAME,来源机制='N00017;N00010',蛋白数=29,合并依据='原始蛋白列表 J=1.0',原始主干数=477,当前主干数=476)]).to_csv(OUT/'J1合并说明.tsv',sep='\t',index=False)
rd(N/'PRIMARY_PATHWAY_TARGETS.tsv')[['canonical_id','display_name','authority_id']].to_csv(OUT/'当前476条主干通路名称.tsv',sep='\t',index=False)
sj(O/'DICTIONARY_VALIDATION.json',dict(target_host='149',path=str(dest),sha256=sha(dest),sheets=counts,primary_pathways=476,merged_group_rows=1,source_definitions_retained=2,active_target_count=673))
print('DICTIONARY_VERIFIED',dest,sha(dest),flush=True)
