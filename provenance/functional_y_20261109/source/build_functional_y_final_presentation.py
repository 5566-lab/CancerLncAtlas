"""Produce the scientific construction report and standalone figures."""
from pathlib import Path
import sys,json,socket,hashlib
import numpy as np,pandas as pd
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R/'python_deps'),'/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps','/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps']
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
C=R/'canonical_targets_r6';Y=R/'functional_relation_y_catalog_r6';O=R/'reports';F=R/'reports/figures';O.mkdir(exist_ok=True);F.mkdir(exist_ok=True)
receipt=json.loads((C/'COMPLETE.json').read_text());master=pd.read_csv(C/'TARGET_MASTER.tsv',sep='\t');sources=[];demo=[]
for cancer in ['BRCA','LUAD','COAD']:
 p=Y/cancer/'sealed_test/COMPLETE.json';assert p.exists(),str(p);r=json.loads(p.read_text());demo.append(r);sources.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
df=pd.DataFrame(demo).set_index('cancer_id').loc[['BRCA','LUAD','COAD']]
plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
navy='#17364D';teal='#127E83';gold='#D9A441';grey='#64748B'
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=300,bbox_inches='tight')
 plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1.7,1]})
counts=[3656,3654,3465,3464,3456];labels=['来源定义','同一 Hallmark\n跨角色去重','激酶／构建体\n身份归并','WNT 跨库\n身份归并','State 生物学\n身份归并']
ax[0].plot(range(5),counts,'o-',color=teal,lw=2.5,ms=8);ax[0].set_xticks(range(5),labels,fontsize=10);ax[0].set_ylim(3390,3710);ax[0].set_ylabel('唯一身份数');ax[0].set_title('3,656 个来源定义 → 3,456 个身份定义',loc='left',fontweight='bold',color=navy,pad=15)
for i,n in enumerate(counts):ax[0].text(i,n+13,f'{n:,}',ha='center',fontsize=12,color=navy)
values=[receipt['pathway_roles']-receipt['shared_roles'],receipt['state_roles']-receipt['shared_roles'],receipt['shared_roles']];labs=['Pathway','State','共同角色'];ax[1].barh(labs,values,color=[teal,navy,gold]);ax[1].invert_yaxis();ax[1].set_xlim(0,max(values)*1.22);ax[1].set_xlabel('唯一身份数');ax[1].set_title('角色归属（共同角色只计一次）',loc='left',fontweight='bold',color=navy,pad=15)
for i,v in enumerate(values):ax[1].text(v+30,i,f'{v:,}',va='center',fontsize=12)
fig.tight_layout(w_pad=3);save(fig,'04_Functional_identity_consolidation')
fig,ax=plt.subplots(figsize=(12,5.7));ax.set_xlim(0,12);ax.set_ylim(0,5.7);ax.axis('off')
def box(x,y,w,h,title,detail,color=teal):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.10',fc='#F1F6F8',ec=color,lw=1.3));ax.text(x+w/2,y+h*.69,title,ha='center',va='center',fontsize=13,fontweight='bold',color=color);ax.text(x+w/2,y+h*.29,detail,ha='center',va='center',fontsize=10,color=navy)
def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,lw=1.5,color=grey))
box(.2,4.35,2.7,1,'RNA','固定背景 ssGSEA／State');box(3.15,4.35,2.7,1,'总蛋白','RPPA500／CPTAC 原生读数');box(6.1,4.35,2.7,1,'磷酸化','位点、方向与总蛋白校正');box(9.05,4.35,2.7,1,'DNA 表型','甲基化、CNV、突变 State')
for x in [1.55,4.5,7.45,10.4]:arrow((x,4.3),(x,3.65))
box(.3,2.6,11.4,1.0,'患者读数 S：各测量层保留自己的方法、参考人群和单位','共同 canonical ID 连接身份；读数保留来源与组成',navy)
arrow((6,2.55),(6,1.98));box(.3,.85,11.4,1.05,'关系 Y：癌种 × lncRNA × 精确 Pathway／State','支持等级 + 方向 + 测量层证据；同组学的多个算法只计一层',teal)
ax.text(6,.27,'评分定义与患者分折固定后，计算调整后的关联；缺测关系保留为未标注。',ha='center',fontsize=11,color=grey);save(fig,'05_Functional_Y_construction')
fig,axes=plt.subplots(1,3,figsize=(12,4.3));metrics=[('observable_relations','可估计关联'),('weak_positive','单层或宽松支持'),('strong_positive','至少两层严格支持')]
for ax,(col,title) in zip(axes,metrics):
 vals=df[col].to_numpy();ax.bar(df.index,vals,color=[teal,navy,gold]);ax.set_title(title,loc='left',fontweight='bold',color=navy);ax.set_ylim(0,max(vals.max()*1.22,1));ax.ticklabel_format(style='sci',axis='y',scilimits=(0,0));ax.set_ylabel('关系数')
 for i,v in enumerate(vals):ax.text(i,v+max(vals.max()*.025,.05),f'{int(v):,}',ha='center',fontsize=10)
fig.suptitle('BRCA、LUAD、COAD：独立标签患者分折中的关系 Y',x=.04,ha='left',fontsize=15,fontweight='bold',color=navy);fig.tight_layout(rect=[0,0,1,.92]);save(fig,'06_Three_cancer_relation_Y')
Q=R/'supervision_summary_r2';T=R/'reaction_scope_relations_r1'
assert json.loads((Q/'COMPLETE.json').read_text())['status']=='COMPLETE'
assert json.loads((R/'functional_y_validation_r2/COMPLETE.json').read_text())['status']=='PASS'
coverage=pd.read_csv(Q/'CURRENT_SCORE_COVERAGE.tsv.gz',sep='\t');patient_cov=pd.read_csv(Q/'PATIENT_SCORE_COVERAGE.tsv.gz',sep='\t')
def group(method,layer):
 if method=='Common-background ssGSEA raw ES':return 'RNA pathway'
 if method in ['Signature-specific State score','Source-specific State score']:return 'RNA State' if str(layer).startswith('RNA') else 'Non-RNA State'
 if 'PROTEIN' in layer:return 'Protein'
 if 'PHOSPH' in layer:return 'Phospho'
 if 'METHYL' in layer:return 'Methylation pathway'
 return 'Non-RNA State'
coverage['score_group']=[group(r.method,r.measurement_layer) for r in coverage.itertuples()];patient_cov['score_group']=[group(r.readout_group,r.measurement_layer) for r in patient_cov.itertuples()]
for frame in [coverage,patient_cov]:
 frame['cohort_display']=frame.cohort.map(lambda x:'TCGA/TCPA' if x in ['TCGA','TCPA_RPPA500'] else x)
 frame['row_label']=frame.cancer_id+'  '+frame.cohort_display
order=sorted(coverage.loc[coverage.cohort_display.eq('TCGA/TCPA'),'row_label'].unique())+sorted(coverage.loc[~coverage.cohort_display.eq('TCGA/TCPA'),'row_label'].unique())
groups=['RNA pathway','RNA State','Protein','Phospho','Methylation pathway','Non-RNA State'];labels=['RNA\n通路','RNA\nState','总蛋白','磷酸化','启动子\n甲基化通路','非 RNA\nState']
num=coverage.loc[coverage.patients_with_score.gt(0)].groupby(['row_label','score_group']).canonical_id.nunique().unstack(fill_value=0).reindex(index=order,columns=groups,fill_value=0)
people=patient_cov.loc[patient_cov.observed_canonical_targets.gt(0)].groupby(['row_label','score_group']).patient_id.nunique().unstack(fill_value=0).reindex(index=order,columns=groups,fill_value=0)
den=coverage.groupby('row_label').patients_in_scope.first().reindex(order);pct=people.div(den,axis=0)*100
heatrows=[]
for label in order:
 for g in groups:heatrows.append({'cohort_cancer':label,'score_group':g,'canonical_targets':int(num.loc[label,g]),'patients_with_score':int(people.loc[label,g]),'patients_in_scope':int(den[label]),'coverage_percent':float(pct.loc[label,g])})
pd.DataFrame(heatrows).to_csv(Q/'FIGURE_SCORE_COVERAGE.tsv',sep='\t',index=False)
def coverage_plot(rows,filename,title):
 fig,ax=plt.subplots(figsize=(11,max(4,len(rows)*.43+1.7)));v=num.loc[rows].to_numpy();ax.imshow(np.log10(v+1),cmap='YlGnBu',aspect='auto',vmin=0,vmax=np.log10(max(coverage.canonical_id.nunique(),1)+1));ax.set_xticks(range(6),labels,fontsize=11);ax.xaxis.tick_top();ax.set_yticks(range(len(rows)),[r.replace('  TCGA/TCPA','').replace('CPTAC:','') for r in rows],fontsize=10)
 for i,r in enumerate(rows):
  for j,g in enumerate(groups):
   n=int(num.loc[r,g]);t=f'{n:,}\n{pct.loc[r,g]:.1f}%' if n else '0';ax.text(j,i,t,ha='center',va='center',fontsize=9,color='white' if n>500 else navy)
 ax.set_title(title+'\n单元格：可评分身份数／至少一项评分的患者覆盖率',loc='left',fontsize=14,fontweight='bold',pad=47,color=navy);ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True);ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',bottom=False,left=False);fig.tight_layout();save(fig,filename)
coverage_plot(order,'07_Score_coverage_all_cohorts','TCGA/TCPA 与 CPTAC：通路及 State 评分覆盖')
coverage_plot(order[:17],'07a_TCGA_score_coverage','TCGA/TCPA 评分覆盖（1/2）');coverage_plot(order[17:33],'07b_TCGA_score_coverage','TCGA/TCPA 评分覆盖（2/2）');coverage_plot(order[33:],'07c_CPTAC_score_coverage','CPTAC 评分覆盖')
yall=pd.read_csv(Q/'Y_SUPERVISION_BY_CANCER.tsv',sep='\t').sort_values('cancer_id').set_index('cancer_id');yall['supported_relations']=yall.weak_positive+yall.strong_positive
metrics_y=['observable_relations','supported_relations','strong_positive'];ylabels=['可估计关联','至少一层支持','至少两层严格支持']
def yplot(rows,name,title):
 fig,ax=plt.subplots(figsize=(11,max(4,len(rows)*.35+1.7)));v=yall.loc[rows,metrics_y].to_numpy();ax.imshow(np.log10(v+1),cmap='PuBuGn',aspect='auto',vmin=0,vmax=np.log10(max(yall.observable_relations.max(),1)+1));ax.set_xticks(range(3),ylabels,fontsize=12);ax.xaxis.tick_top();ax.set_yticks(range(len(rows)),[f'{r} (n={int(yall.loc[r,"label_patients"])})' for r in rows],fontsize=10)
 for i,r in enumerate(rows):
  for j,k in enumerate(metrics_y):
   n=int(yall.loc[r,k]);ax.text(j,i,f'{n:,}',ha='center',va='center',fontsize=10,color='white' if n>200000 else navy)
 ax.set_title(title+'\n同一独立标签分折；单位：lncRNA × canonical target',loc='left',fontweight='bold',fontsize=14,pad=35,color=navy);ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True);ax.grid(which='minor',color='white',linewidth=.8);ax.tick_params(which='minor',bottom=False,left=False);fig.tight_layout();save(fig,name)
yplot(yall.index.tolist(),'08_Y_supervision_all_cancers','33 癌种的关系监督证据');yplot(yall.index[:17].tolist(),'08a_Y_supervision','关系监督证据（1/2）');yplot(yall.index[17:].tolist(),'08b_Y_supervision','关系监督证据（2/2）')
rr=json.loads((T/'COMPLETE.json').read_text());rt=rr['relation_types'];rv=[rt.get('shared_reaction_core',0),rt.get('left_is_annotated_ancestor',0)+rt.get('right_is_annotated_ancestor',0),rt.get('left_contains_right_reaction_scope',0)+rt.get('right_contains_left_reaction_scope',0)];fig,ax=plt.subplots(figsize=(10.5,4.5));ax.barh(['共同反应核心','已注释祖先／后代','反应范围包含'],rv,color=[teal,navy,gold]);ax.invert_yaxis();ax.set_xlim(0,max(rv)*1.2);ax.set_xlabel('通路对');ax.set_title(f"{rr['scored_pathways_with_reactions']:,} 条通路的 {rr['pairs']:,} 对反应范围关系",loc='left',fontweight='bold',fontsize=16,color=navy)
for i,v in enumerate(rv):ax.text(v+45,i,f'{v:,}',va='center',fontsize=13)
fig.tight_layout();save(fig,'09_Reaction_scope_relations')

prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5);FONT='Noto Sans CJK SC';slide_notes=[]
def color(h):return RGBColor.from_string(h.strip('#'))
def text(slide,x,y,w,h,content,size=20,bold=False,c=navy):
 shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=shape.text_frame;tf.word_wrap=True;tf.margin_left=tf.margin_right=Inches(.03)
 for i,line in enumerate(content.split('\n')):
  p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name=FONT;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=color(c);p.space_after=Pt(9)
 return shape

def slide(title,subtitle=''):
 s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=color('#FFFFFF');bar=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.13));bar.fill.solid();bar.fill.fore_color.rgb=color(teal);bar.line.fill.background();text(s,.55,.37,12.2,.6,title,29,True)
 if subtitle:text(s,.58,1.12,12.1,.6,subtitle,16,False,grey)
 text(s,.58,7.05,11.6,.2,'CancerLncAtlas  |  Functional pathway / State  |  2026-11-09',9,False,grey);text(s,12.1,7.0,.6,.3,str(len(prs.slides)),10,False,grey);return s

def picture(s,name,y=1.75,h=4.9):
 from PIL import Image
 p=F/(name+'.png')
 with Image.open(p) as im:iw,ih=im.size
 scale=min(12.0/iw,h/ih);w2,h2=iw*scale,ih*scale;s.shapes.add_picture(str(p),Inches((13.333-w2)/2),Inches(y+(h-h2)/2),width=Inches(w2),height=Inches(h2))

def table(s,cols,rows,y=1.85,widths=None,size=17):
 shape=s.shapes.add_table(len(rows)+1,len(cols),Inches(.65),Inches(y),Inches(12),Inches(.70*(len(rows)+1))).table
 if widths:
  for c,w in zip(shape.columns,widths):c.width=Inches(w)
 for i,row in enumerate([cols]+rows):
  for j,val in enumerate(row):
   cell=shape.cell(i,j);cell.text=str(val);cell.fill.solid();cell.fill.fore_color.rgb=color(navy if i==0 else ('#F0F5F7' if i%2 else '#FFFFFF'));cell.margin_left=Inches(.12);cell.margin_top=Inches(.07)
   for p in cell.text_frame.paragraphs:p.font.name=FONT;p.font.size=Pt(size);p.font.bold=(i==0);p.font.color.rgb=color('#FFFFFF' if i==0 else navy)
 return shape

def note(s,body):s.notes_slide.notes_text_frame.text=body
s=slide('Functional pathway / State 的 Y 构建','统一生物学身份，保留组学读数，汇总癌种内 lncRNA 关系')
text(s,.7,2.0,11.8,1.1,'癌种 × lncRNA × 精确通路／State',38,True,teal);text(s,.75,3.45,11.6,1.3,'患者层：S 描述某个测量层的通路或状态。\n关系层：Y 描述 lncRNA 与同一生物学目标的关联支持。',24);text(s,.75,5.7,11.5,.7,f"{receipt['identities_with_readouts']:,} 个身份定义  ·  {coverage.loc[coverage.patients_with_score.gt(0)].canonical_id.nunique():,} 个有实测评分",22,True)
s=slide('同一个通路身份可以容纳多个读数','数据库、测量层和算法记录在 readout 中');picture(s,'05_Functional_Y_construction',1.65,5.0)
s=slide('身份合并后的数量','Pathway 和 State 的共同身份在总数中只计一次');picture(s,'04_Functional_identity_consolidation',1.95,4.55);note(s,json.dumps(receipt,ensure_ascii=False))
s=slide('何时合并，何时保留子通路或分支')
table(s,['关系','处理','实例'],[['同一生物学范围','合并 canonical ID，保留多个读数','WikiPathways 与 Reactome 的 WNT'],['共同核心、范围有扩展','保留身份，连接共同核心与扩展','酪氨酸分解的 5 步核心与附加反应'],['受体／变异背景不同','保留具体分支身份','TRKA–PI3K 分支；SMO 与 PTCH1 变异'],['同一 State、同一计量对象','合并身份，记录各自算法和单位','ESTIMATE 与 ABSOLUTE 肿瘤纯度'],['不同分母或分子表型','连接组成／子群关系','免疫内相对比例；RNA 与 DNA 干性表型']],widths=[2.6,4.5,4.9],size=16)
s=slide('从反应结构判断共同核心与扩展','酪氨酸分解：同一组基因不意味着完全相同的反应范围');picture(s,'03_Tyrosine_shared_core',1.65,4.95);note(s,'https://www.kegg.jp/entry/N00708\nhttps://reactome.org/content/detail/R-HSA-8963684')
s=slide('State 的组成与分子表型关系','组成公式、分子表型和细胞子群使用不同的关系类型')
table(s,['目标','组成或读数','关系'],[['HRD_total','LOH + TAI + LST','三项计数相加；9,971 位患者逐行一致'],['ESTIMATEScore','ImmuneScore + StromalScore','两项相加；8,810 位患者最大差 0.01'],['DNA stemness','DMP / ENH / EREG 的 CpG 签名','联合签名拟合 OCLR'],['Oncogenic dedifferentiation','RNA 与甲基化干性表型','分子表型组成'],['CD8 T-cell abundance','xCell、MCP-counter；naive / Tcm / Tem','同一群体的不同读数与细胞子群']],widths=[2.8,4.1,5.1],size=16)
s=slide('蛋白层 S：按实测蛋白组计算','RPPA500 的抗体组映射保持原始检测范围')
text(s,.8,1.95,11.8,1.05,'S(protein) = 有效蛋白组 Z 值之和 / 有效蛋白组数',29,True,teal)
text(s,.85,3.15,11.6,2.2,'1. 训练参考患者拟合每个抗体的均值与标准差。\n2. 相同基因组映射的抗体合并为一个读数。\n3. 跨多个蛋白的抗体按一个检测组计数。\n4. 每位患者至少有 3 个有效组时生成该通路 S。',21)
text(s,.85,5.85,11.5,.55,'三癌种可评分蛋白通路：BRCA 945、LUAD 952、COAD 944。',21,True)
s=slide('磷酸化 S：精确位点、方向与总蛋白校正','匹配蛋白、残基、位置和位点周围序列')
text(s,.8,1.95,11.8,.8,'P(adj) = P − (α + β × Total protein)',30,True,teal);text(s,.8,2.9,11.8,.8,'S(phospho) = mean[ direction × Z(P(adj)) ]',29,True,teal)
text(s,.85,4.0,11.6,1.35,'回归和标准化参数由训练参考患者拟合。\n位点通过序列核对，再连接 PTMsigDB 的正／负方向签名。',22);text(s,.85,5.6,11.5,.8,'33 个位点具备序列与总蛋白匹配；21 个签名满足至少 3 位点。',21,True)
s=slide('RNA 与 State 的评分参考固定','统一注释与评分背景；按评分定义冻结参考参数')
text(s,.85,1.95,11.6,3.0,'RNA pathway：共同蛋白编码基因背景上的 ssGSEA 原始 ES。\nState：固定签名、加权程序、xCell、OCLR 及各自原生表型。\n需要中心化、范围缩放或回归的参数只由 source 患者拟合。\n全部 16,889 条 TCGA lncRNA 保留在特征轴上。',22)
text(s,.85,5.35,11.6,.95,'直接参与某个 RNA 评分的 lncRNA，从该读数的关联监督中排除。',21,True,teal)
s=slide('TCPA 与 CPTAC 通过身份和证据对齐','跨平台的原始 S、测量范围和方法保持可追溯')
table(s,['层级','共同字段','平台特有字段'],[['生物学身份','canonical ID、功能范围、组成关系','来源路径与来源名称'],['Protein readout','蛋白标识、通路成员','抗体组／质谱蛋白与评分方法'],['Phospho readout','位点与方向、激酶／通路身份','位点检测范围与原生评分参考'],['患者读数 S','患者 ID、癌种、目标 ID','单位、归一化参数、实测覆盖率'],['关系 Y','癌种、lncRNA、canonical ID','各层支持、效应方向与来源']],widths=[2.35,4.3,5.35],size=16)
text(s,.8,6.25,11.7,.6,'同一身份连接多个测量证据；RPPA 与质谱读数各自保留测量含义。',17,True,teal)
s=slide('关系 Y 的计算规则','每个癌种、每条 lncRNA、每个统一身份生成一个关系记录')
table(s,['字段','取值','计算定义'],[['membership','未标注／单层或宽松／多层严格支持','至少一个读数支持；至少两组学严格支持'],['strict evidence','|ρ| ≥ 0.20，BH q ≤ 0.05','协变量与局部 CNV 调整后的关联'],['additional evidence','|ρ| ≥ 0.15，BH q ≤ 0.10','扣除严格支持后的补充证据'],['direction','正／负／方向不一致','相对于对应读数的关联方向'],['evidence layers','RNA、蛋白、磷酸化、DNA 等','同一组学的多个算法计作一层']],widths=[2.4,4.1,5.5],size=16)
text(s,.8,6.25,11.7,.6,'未标注关系没有被定义为负例；Y 的支持等级表示关联证据。',17,True,teal)
s=slide('患者分折决定评分参考与监督标签','source 和 label 患者互不重叠')
table(s,['用途','source 患者分折','label 患者分折'],[['训练标签 A','3、4','2'],['训练标签 B','2、4','3'],['训练标签 C','2、3','4'],['验证','2、3、4','1'],['独立测试','2、3、4','0']],widths=[4,4,4],size=19)
text(s,.8,6.25,11.7,.6,'协变量按目标剔除自身或派生项；局部 CNV 用于 lncRNA 一侧的调整。',17,True,teal)
s=slide('三癌种独立标签分折中的关系数量','独立标签患者中的关联估计与多组学支持');picture(s,'06_Three_cancer_relation_Y',1.8,4.7);note(s,df.to_json(orient='index',force_ascii=False))
for filename,title in [('07a_TCGA_score_coverage','TCGA/TCPA 的评分覆盖（1/2）'),('07b_TCGA_score_coverage','TCGA/TCPA 的评分覆盖（2/2）'),('07c_CPTAC_score_coverage','CPTAC 的评分覆盖')]:
 s=slide(title,'统一身份下，分别统计可评分目标和有实测评分的患者');picture(s,filename,1.75,5.0)
for filename,title in [('08a_Y_supervision','33 癌种的可监督关系（1/2）'),('08b_Y_supervision','33 癌种的可监督关系（2/2）')]:
 s=slide(title,'每个癌种使用独立标签患者分折；同一关系只计一次');picture(s,filename,1.75,5.0)
s=slide('共同反应、范围扩展与层级关系','共同核心和新增反应均保留精确的反应 ID');picture(s,'09_Reaction_scope_relations',1.85,4.55);text(s,.85,6.45,11.6,.35,f"{rr['shared_core_sets']:,} 个共同反应核心；扩展集合可逐项查询。",17,True,teal)

s=slide('交付数据的组织方式')
table(s,['文件／表','每一行或元素代表','主要内容'],[['Target_master','一个统一的 Pathway／State 身份','固定 ID、名称、范围与角色'],['Source_IDs / Score_definitions','一个来源定义／评分读数','数据库、方法、单位、测量层'],['Components / Relations','一个组成或范围关系','基因、位点、反应、复合物、子 State'],['PATIENT_S.parquet','患者 × 读数','实测评分与参考参数'],['Y.npz / POSITIVE_RELATIONS','lncRNA × canonical ID／支持关系','支持等级、方向、测量层和分折']],widths=[3.55,3.7,4.75],size=16)
s=slide('模型接口与结果解释')
text(s,.85,1.95,11.6,1.0,'L = L(Pathway) + 0.5 × L(State) + 模型正则项',29,True,teal)
text(s,.85,3.2,11.5,2.5,'同一身份跨 Pathway / State 角色出现时只计一次。\n模型输入患者与关系标签患者遵循独立分折。\n不同组学的实测标签通过观测掩码参与对应监督。\n关系方向对应测量读数；因果作用由独立扰动证据检验。',22)
refs=['Reactome release 98: https://reactome.org/','WikiPathways WNT: https://sandbox.wikipathways.org/pathways/WP428.html','PTMsigDB / PTM-SEA source: pinned ssGSEA2.0 repository and project GMT receipt','Malta et al., Cell 2018: https://pmc.ncbi.nlm.nih.gov/articles/PMC5902191/','ESTIMATE: https://www.nature.com/articles/ncomms3612','MCP-counter: https://pmc.ncbi.nlm.nih.gov/articles/PMC5073889/']
for s in prs.slides:
 if not s.notes_slide.notes_text_frame.text.strip():note(s,'Scientific source records: '+str(C/'COMPLETE.json')+'\n'+str(Y/'Y_DEFINITION.json'))
s=slide('数据与方法来源')
text(s,.8,1.9,11.8,4.7,'\n'.join(refs),17);note(s,'Source file checksums and exact versions are stored in the corresponding data receipts.')
out=O/'Functional_Y_construction_report_v2.pptx';prs.save(out)
with __import__('zipfile').ZipFile(out) as z:assert z.testzip() is None
rec={'target_host':'149','file':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'slides':len(prs.slides),'canonical_registry':str(C),'canonical_receipt_sha256':hashlib.sha256((C/'COMPLETE.json').read_bytes()).hexdigest(),'demo_contexts':sources,'figures':['04_Functional_identity_consolidation','05_Functional_Y_construction','06_Three_cancer_relation_Y','07_Score_coverage_all_cohorts','08_Y_supervision_all_cancers','09_Reaction_scope_relations'],'all_cancer_summary_sha256':hashlib.sha256((Q/'COMPLETE.json').read_bytes()).hexdigest()};(R/'project_log/PRESENTATION_V2_RECEIPT.json').write_text(json.dumps(rec,indent=2,ensure_ascii=False));print('PRESENTATION_COMPLETE',json.dumps(rec),flush=True)
