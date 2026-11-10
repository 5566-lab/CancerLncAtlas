"""Problem-solution-result presentation from frozen Functional Y outputs. Host 149 only."""
from pathlib import Path
import sys,socket,json,hashlib,subprocess,re,os
assert socket.gethostname()=='149'
R=Path(__file__).resolve().parent
O=R/'reports/leukocyte_revision_20261110_r1'; F=O/'figures'; F.mkdir(exist_ok=True)
sys.path[:0]=[str(R/'python_deps'),'/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps']
import pandas as pd,numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE,MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR,PP_ALIGN
from PIL import Image,ImageOps,ImageDraw
Q=R/'supervision_summary_r2';C=R/'canonical_targets_r6'
y=pd.read_csv(Q/'Y_SUPERVISION_BY_CANCER.tsv',sep='\t').set_index('cancer_id')
y['supported']=y.strong_positive+y.weak_positive
cov=pd.read_csv(Q/'FIGURE_SCORE_COVERAGE.tsv',sep='\t')
strict=pd.read_csv(O/'STRICT_MULTILAYER_EXAMPLES.tsv',sep='\t')
assert len(strict)==495 and strict.display_name.eq('Tumor cell fraction').sum()==476
assert y.observable_relations.gt(0).sum()==22 and y.supported.gt(0).sum()==20
assert int(y.supported.sum())==13303346
NAVY='#18344A';TEAL='#087E87';BLUE='#527FC1';GOLD='#D8A13D';RED='#B65C48';GREY='#657583';LIGHT='#EEF4F6';PALE='#F6F8FA';WHITE='#FFFFFF'
plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':17,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False,'axes.edgecolor':'#C9D3D9','axes.labelcolor':NAVY,'text.color':NAVY,'xtick.color':GREY,'ytick.color':NAVY,'pdf.fonttype':42,'savefig.facecolor':'white'})
D=['BRCA','LUAD','COAD']
def save(fig,name):
 fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(F/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def get(c,platform,g,col):return cov.loc[cov.cohort_cancer.eq(c+'  '+('TCGA/TCPA' if platform=='TCPA' else 'CPTAC:'+c))&cov.score_group.eq(g),col].iloc[0]
fig,ax=plt.subplots(figsize=(11.8,4.5))
z=np.arange(3)
for offset,p,co in [(-.18,'TCPA',TEAL),(.18,'CPTAC',BLUE)]:
 vals=[get(c,p,'Protein','coverage_percent') for c in D]
 bars=ax.barh(z+offset,vals,height=.30,color=co,label=p)
 for i,(bar,v) in enumerate(zip(bars,vals)):
  n=get(D[i],p,'Protein','patients_with_score');den=get(D[i],p,'Protein','patients_in_scope')
  ax.text(v+1.5,bar.get_y()+.15,f'{v:.1f}%  ({int(n)}/{int(den)})',va='center',fontsize=17)
ax.set_yticks(z,D,fontsize=21);ax.invert_yaxis();ax.set_xlim(0,132);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('本队列患者覆盖率（%）',fontsize=18);ax.legend(loc='upper left',bbox_to_anchor=(0,1.17),frameon=False,ncol=2);ax.grid(axis='x',alpha=.13);ax.set_axisbelow(True);fig.tight_layout();save(fig,'10_Patient_coverage_demo')
fig,axs=plt.subplots(1,2,figsize=(13.2,4.7))
for ax,g,title in zip(axs,['Protein','Phospho'],['蛋白通路','磷酸化签名']):
 for offset,p,co in [(-.18,'TCPA',TEAL),(.18,'CPTAC',BLUE)]:
  vals=[get(c,p,g,'canonical_targets') for c in D];bs=ax.bar(z+offset,vals,.32,color=co,label=p)
  for b,v in zip(bs,vals):ax.text(b.get_x()+b.get_width()/2,v+max(vals)*.025,f'{int(v):,}',ha='center',fontsize=18)
 ax.set_xticks(z,D,fontsize=19);ax.set_ylim(0,1850 if g=='Protein' else 400);ax.set_title(title,loc='left',fontweight='bold',fontsize=23,pad=20);ax.set_ylabel('至少一位患者有评分的身份数');ax.grid(axis='y',alpha=.13);ax.set_axisbelow(True)
fig.legend(*axs[0].get_legend_handles_labels(),frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,-.085),fontsize=17);fig.tight_layout(w_pad=3);save(fig,'11_Target_breadth_demo')
fig,axs=plt.subplots(1,2,figsize=(13.2,4.7),gridspec_kw={'width_ratios':[1.05,1]})
for i,c in enumerate(D):
 a=int(y.loc[c,'canonical_targets_estimable']);b=int(y.loc[c,'canonical_targets_supported'])
 axs[0].barh(i,a,color='#DCE6EB',height=.52);axs[0].barh(i,b,color=TEAL,height=.52)
 axs[0].text(a+40,i,f'{b:,} / {a:,}',va='center',fontsize=18)
axs[0].set_yticks(range(3),D,fontsize=21);axs[0].invert_yaxis();axs[0].set_xlim(0,3550);axs[0].set_xticks([0,1000,2000]);axs[0].set_xlabel('有支持的目标 / 可估计目标');axs[0].set_title('覆盖了多少个生物学目标？',loc='left',fontweight='bold',fontsize=22,pad=20)
vals=[int(y.loc[c,'supported']) for c in D];axs[1].scatter(vals,z,s=160,color=BLUE,zorder=3)
for i,v in enumerate(vals):
 axs[1].hlines(i,10000,v,color='#C9D6E8',lw=4);axs[1].text(v*1.18,i,f'{v/10000:.2f} 万',va='center',fontsize=18)
axs[1].set_xscale('log');axs[1].set_xlim(10000,2e7);axs[1].set_xticks([1e4,1e5,1e6,1e7],['1万','10万','100万','1000万']);axs[1].set_yticks(z,D,fontsize=21);axs[1].invert_yaxis();axs[1].set_xlabel('支持关系数（对数刻度）');axs[1].set_title('形成了多少条关联证据？',loc='left',fontweight='bold',fontsize=22,pad=20);axs[1].grid(axis='x',alpha=.15)
fig.tight_layout(w_pad=3);save(fig,'12_Demo_Y_results')
fig,ax=plt.subplots(figsize=(6.5,5))
ax.pie([476,19],startangle=90,colors=[BLUE,GOLD],wedgeprops={'width':.30,'edgecolor':'white','linewidth':3})
ax.text(0,.10,'495',ha='center',va='center',fontsize=46,fontweight='bold');ax.text(0,-.23,'两层严格支持',ha='center',fontsize=18);ax.set_aspect('equal');fig.tight_layout();save(fig,'13_Strict_evidence_composition')
# Appendix: large readable panels, 11 cancers per slide; zero labels distinguish missing estimability.
for part in range(3):
 rows=y.iloc[part*11:(part+1)*11];fig,ax=plt.subplots(figsize=(12.8,5.8));pos=np.arange(len(rows))
 vals=rows.supported.to_numpy();ax.set_xscale('log');ax.set_xlim(.75,3e7)
 for i,(c,row) in enumerate(rows.iterrows()):
  v=int(row.supported)
  if v:
   ax.hlines(i,1,v,color='#C5DAE0',lw=4);ax.scatter(v,i,color=TEAL,s=70,zorder=3);ax.text(v*1.3,i,f'{v:,}',va='center',fontsize=16)
  else:ax.text(1.5,i,'可估计，未达到支持阈值' if row.observable_relations>0 else '该标签分折无可估计关系',va='center',fontsize=16,color=GREY)
 ax.set_yticks(pos,[f'{c}  (n={int(row.label_patients)})' for c,row in rows.iterrows()],fontsize=16);ax.invert_yaxis();ax.set_xticks([1,100,1e4,1e6,1e7],['1','100','1万','100万','1000万']);ax.set_xlabel('支持关系数（对数刻度）；n 为该癌种标签患者人数',fontsize=16);ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,f'14_All_cancer_support_{part+1}')
# Native editable presentation.
prs=Presentation();prs.slide_width=Inches(16);prs.slide_height=Inches(9)
FONT='Noto Sans CJK SC';sources=[];outline=[]
def rgb(h):return RGBColor.from_string(h.lstrip('#'))
def rect(s,x,y,w,h,fill=LIGHT,line=None,round=True):
 q=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if round else MSO_SHAPE.RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));q.fill.solid();q.fill.fore_color.rgb=rgb(fill)
 if line:q.line.color.rgb=rgb(line)
 else:q.line.fill.background()
 if round:q.adjustments[0]=.10
 return q
def tx(s,x,y,w,h,t,size=23,color=NAVY,bold=False,align=None):
 q=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));f=q.text_frame;f.word_wrap=True;f.margin_left=f.margin_right=0;f.margin_top=f.margin_bottom=0
 for i,l in enumerate(t.split('\n')):
  p=f.paragraphs[0] if i==0 else f.add_paragraph();p.text=l;p.font.name=FONT;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=rgb(color);p.space_after=Pt(8);p.line_spacing=1.15
  if align is not None:p.alignment=align
 return q
def arr(s,x,y,w=.48,color=GREY):
 q=s.shapes.add_shape(MSO_SHAPE.CHEVRON,Inches(x),Inches(y),Inches(w),Inches(.42));q.fill.solid();q.fill.fore_color.rgb=rgb(color);q.line.fill.background()
def slide(section,title,sub='',src=''):
 s=prs.slides.add_slide(prs.slide_layouts[6]);s.background.fill.solid();s.background.fill.fore_color.rgb=rgb(WHITE)
 rect(s,0,0,16,.10,TEAL,round=False);tx(s,.65,.30,14.7,.3,section,12,TEAL,True);tx(s,.65,.87,14.7,.66,title,32,bold=True)
 if sub:tx(s,.68,1.66,14.55,.52,sub,18,GREY)
 tx(s,.68,8.54,13.9,.24,'CancerLncAtlas  ·  Functional Y  ·  2026.11.10',10,GREY);tx(s,14.7,8.49,.6,.32,f'{len(prs.slides):02}',12,GREY,align=PP_ALIGN.RIGHT)
 note='数据口径：冻结版本 canonical_targets_r6 / functional_relation_y_catalog_r6；本次仅重做展示，不改标签、分折或阈值。\n'+src
 s.notes_slide.notes_text_frame.text=note;outline.append({'slide':len(prs.slides),'section':section,'title':title,'sources':src});return s
def band(s,t,y=7.65,color=TEAL):
 rect(s,.68,y,14.64,.60,LIGHT);tx(s,.9,y+.12,14.15,.4,t,19,color,True)
def card(s,x,y,w,h,label,body,col=TEAL):
 rect(s,x,y,w,h,PALE);rect(s,x,y,.06,h,col,round=False);tx(s,x+.24,y+.21,w-.48,.48,label,23,col,True);tx(s,x+.24,y+.94,w-.48,h-1.05,body,22)
def pic(s,name,x=.7,y0=2.35,w=14.6,h=4.95):
 p=F/(name+'.png')
 with Image.open(p) as im:iw,ih=im.size
 scale=min(w/iw,h/ih);nw,nh=iw*scale,ih*scale;s.shapes.add_picture(str(p),Inches(x+(w-nw)/2),Inches(y0+(h-nh)/2),width=Inches(nw),height=Inches(nh))
def metric(s,x,y,w,num,label,col=TEAL):
 tx(s,x,y,w,.95,num,47,col,True);tx(s,x,y+1.02,w,1.0,label,21)
# 1
s=slide('研究进展汇报','从多来源通路评分，到 lncRNA–功能关系 Y','遇到了什么问题？怎样整合？结果走到了哪一步？')
tx(s,.85,2.6,14.2,1.25,'关系 Y 已构建；\n跨平台统一活性分数尚未完成。',36,bold=True)
card(s,.85,4.8,4.5,2.15,'统一定义','3,456 个通路 / State 身份')
card(s,5.75,4.8,4.5,2.15,'形成关系标签','TCGA/TCPA\n20 癌种出现支持关系')
card(s,10.65,4.8,4.5,2.15,'保留平台信息','CPTAC：评分与身份已对齐',BLUE)
# 2
s=slide('背景','真正的问题：某癌种中，一条 lncRNA 与什么功能有关？','研究单位：癌种 × lncRNA × 精确通路 / State')
rect(s,.9,2.7,3.6,3.3,NAVY);tx(s,1.18,3.03,3.1,.5,'一条 lncRNA',28,WHITE,True);tx(s,1.18,4.0,3.0,1.2,'在某癌种内\n与功能状态相关吗？',23,WHITE)
arr(s,4.8,4.05,.55)
for yy,lab,body,col in [(2.65,'RNA 程序','转录响应',TEAL),(4.02,'蛋白 / 磷酸化','丰度与信号读数',BLUE),(5.39,'肿瘤 State','纯度、干性、免疫等表型',GOLD)]:
 rect(s,5.7,yy,8.95,1.07,PALE);tx(s,6,yy+.24,3.0,.5,lab,24,col,True);tx(s,9.0,yy+.25,5.35,.5,body,23)
band(s,'需要进一步检验：lncRNA 是否提供超出 mRNA 的信息，以及关联能否被扰动实验支持。')
# 3
s=slide('问题 → 解决办法 → 结果','三个问题，分别在三个层次处理','生物学定义、患者读数与关系证据各有明确职责')
for yy,a,b,c,col in [(2.6,'定义不一','按功能范围统一身份','3,456 个固定身份',TEAL),(4.14,'测量不一','同一身份保留各层 S','2,840 个身份有实测评分',BLUE),(5.68,'证据不一','调整关联后按测量层汇总','得到带支持等级与方向的 Y',GOLD)]:
 rect(s,.8,yy,3.65,1.15,PALE);tx(s,1.05,yy+.32,3.1,.5,a,25,col,True);arr(s,4.65,yy+.37)
 rect(s,5.33,yy,4.8,1.15,LIGHT);tx(s,5.56,yy+.32,4.35,.5,b,23,bold=True);arr(s,10.35,yy+.37)
 tx(s,11.07,yy+.18,4.1,.95,c,22,col,True)
# 4
s=slide('整合方法总览','实际采用：身份归并 + 分层读数 + 关联证据汇总','合并对象是同一生物学目标下的关系证据；患者读数保留自己的测量含义。',src='assemble_functional_relation_y_numerical.py; canonical_targets_r6/ID_REDIRECTS.tsv')
for x,w,n,t,b in [(.8,3.0,'01','统一身份','别名、功能范围\n父子与分支关系'),(4.23,3.0,'02','保留患者 S','各组学分别评分\n记录实测与缺失'),(7.66,3.0,'03','癌种内关联','临床与细胞组成\n局部 CNV 调整'),(11.09,4.0,'04','形成关系 Y','支持等级 + 方向\n+ 具体测量层证据')]:
 rect(s,x,2.85,w,3.55,LIGHT if n!='04' else NAVY);cc=NAVY if n!='04' else WHITE;tx(s,x+.22,3.1,w-.44,.5,n,24,TEAL if n!='04' else GOLD,True);tx(s,x+.22,3.85,w-.44,.6,t,26,cc,True);tx(s,x+.22,4.85,w-.44,1.25,b,20,cc)
 if n!='04':arr(s,x+w+.11,4.2,.36)
band(s,'TCPA 与 CPTAC 原始分数没有求平均；当前正式 Y 汇总 TCGA + 匹配 TCPA 的证据。')
# 5
s=slide('问题 1｜通路名字、组成与层级不同','如何解决：相同范围归一，相似范围保留关系','以生物学范围组织目标，再用基因、位点、反应结构说明其组成。')
card(s,.8,2.65,7.0,3.65,'同一响应程序 → 一个身份','缺氧：Hallmark / Buffa / PROGENY\n保留三种 RNA 读数\n同属 RNA，只计一个证据家族',TEAL)
card(s,8.2,2.65,7.0,3.65,'核心相同、范围不同 → 连接身份','广义 PI3K 与受体特异分支分别保留\n反应交集描述共同核心\n额外反应描述扩展范围',BLUE)
band(s,'已有 1,833 条通路关联到反应结构，形成 8,610 对共同核心、层级或范围关系。')
# 6
s=slide('问题 1｜解决结果','身份已经统一，实测评分覆盖其中 82.2%','身份数量与有实测评分的目标数量分开统计。',src='canonical_targets_r6/COMPLETE.json; supervision_summary_r2/CURRENT_SCORE_COVERAGE.tsv.gz')
metric(s,.85,2.55,4.3,'3,656','来源定义',GREY);arr(s,4.85,3.05,.55);metric(s,5.65,2.55,4.4,'3,456','归并后的生物学身份');metric(s,11.1,2.55,4.05,'2,840','至少有一个有限患者评分',BLUE)
rect(s,.9,5.1,14.2,.72,'#DDE4E9',round=False);rect(s,.9,5.1,14.2*2840/3456,.72,TEAL,round=False)
tx(s,1.15,5.26,9.5,.4,'有实测评分  2,840',22,WHITE,True);tx(s,12.0,6.04,3.1,.7,'其余 616 个\n尚无有限评分',19,GREY)
tx(s,.95,6.27,10.3,.8,'Pathway 角色 2,595；State 角色 864；\n其中 3 个身份同时具有两种角色。',22)
# 7
s=slide('问题 2｜TCPA 抗体与 CPTAC 质谱测量不同','如何解决：先统一“测什么”，各平台分别保留 S','TCGA：10,432 位 RNA 患者；CPTAC：10 个队列、1,043 位患者。')
card(s,.8,2.6,7.0,3.75,'TCPA / RPPA500','蛋白：实测抗体组的标准化均值\n磷酸化：总蛋白调整后的位点读数\n按可测蛋白组 / 位点计算',TEAL)
card(s,8.2,2.6,7.0,3.75,'CPTAC / 质谱','蛋白：项目计算的 ssGSEA\n磷酸化：带方向的 PTM-SEA\n保留每个评分的原生参考与单位',BLUE)
band(s,'当前完成身份与组成对齐；尚未建立通过验证的 TCPA ↔ CPTAC 统一数值标尺。')
# 8
s=slide('问题 2｜解决结果：患者覆盖','三癌种均已有蛋白 / 磷酸化评分，可用于后续证据检验','分母为各自队列患者；图示“至少一项评分”的覆盖率。',src='supervision_summary_r2/FIGURE_SCORE_COVERAGE.tsv; these rows have identical protein/phospho patient coverage')
pic(s,'10_Patient_coverage_demo',y0=2.25,h=5.05);band(s,'这三癌种的蛋白与磷酸化患者覆盖率相同；单个目标的有效样本数仍取决于实际缺失。')
# 9
s=slide('问题 2｜解决结果：检测范围','CPTAC 覆盖更广，TCPA 的磷酸化通路范围仍很有限','统计具有至少一个有限患者评分的目标；两平台目标集合有重叠。')
pic(s,'11_Target_breadth_demo',y0=2.3,h=4.95);band(s,'共同 ID 保留这些检测差异；扩大评分目录并不会自动增加跨平台可用的监督证据。')
# 10
s=slide('问题 3｜评分不一、证据可能相反','如何解决：逐层估计关联，再汇总支持等级','Y =（支持等级、方向、观测证据层、严格支持层、全部支持层）',src='functional_relation_y_catalog_r6/Y_DEFINITION.json; assemble_functional_relation_y_numerical.py')
for x,t,b,col in [(.8,'支持等级 0','未标注\n没有达到支持条件',GREY),(5.76,'支持等级 1','有支持\n尚不足两层严格支持',BLUE),(10.72,'支持等级 2','多层严格支持\n至少两个测量层达标',TEAL)]:card(s,x,2.6,4.48,2.35,t,b,col)
tx(s,.95,5.32,14.1,.7,'严格支持：|ρ| ≥ 0.20 且 BH q ≤ 0.05',26,TEAL,True)
tx(s,.95,6.15,14.1,.65,'补充支持：|ρ| ≥ 0.15 且 BH q ≤ 0.10，扣除严格支持部分',23)
band(s,'同层不同算法计一次；未测量与未达标分别记录；相反方向保留为“方向不一致”。')
# 11
s=slide('问题 3｜一条真实结果，展示整合过程','BRCA：同一条 lncRNA 的 mTORC1 关联得到 RNA 与蛋白支持','lncRNA：ENSG00000228401  ·  目标：MTORC1 SIGNALING',src='numerical_relation_evidence_r1/BRCA/sealed_test/{RNA,PROTEIN}/SUPPORTED_RELATIONS.parquet; REAL_EXAMPLE.json')
card(s,.8,2.6,5.15,2.65,'RNA 证据','调整后 ρ = −0.407\nq = 9.39 × 10⁻⁷；n = 206',TEAL)
card(s,.8,5.52,5.15,2.05,'蛋白证据','ρ = −0.436；q = 0.0305\nn = 163',BLUE)
arr(s,6.28,4.3,.65);rect(s,7.4,2.9,7.75,3.9,NAVY)
tx(s,7.75,3.25,7.0,.6,'同一个目标身份',24,WHITE,True);tx(s,7.75,4.15,7.0,.8,'Y = 2  ·  方向为负',32,WHITE,True);tx(s,7.75,5.25,7.0,1.1,'两种测量层严格支持\n原始分数和相关系数均没有相加',23,WHITE)
# 12
s=slide('问题 3｜控制关联估计中的重复与混杂','评分参考患者与关系标签患者分开','参考患者确定评分参考与协变量编码；标签患者内进行调整后的关联估计。',src='functional_relation_y_catalog_r6/BRCA/sealed_test/SPLIT.json; numerical_relation_evidence_r1; source and label disjoint verified')
card(s,.8,2.65,6.9,2.8,'参考患者（source）','确定评分参考与协变量编码\nBRCA：654 位参考患者',TEAL)
card(s,8.25,2.65,6.9,2.8,'标签患者（label）','按固定编码调整，估计 lncRNA–S 关联\nBRCA：218 位标签患者',BLUE)
tx(s,.95,5.93,14.1,1.2,'临床、纯度与细胞组成按目标调整；局部 CNV 调整 lncRNA 一侧。\n保留全部 16,889 条 lncRNA；直接构成该 RNA 评分的成员在对应关系中排除。',23)
band(s,'有效 n 按每个目标计算；预测独立于 mRNA 的增益，需要后续模型对照来回答。')
# 13
s=slide('问题 3｜解决结果：可用范围','33 个癌种中，22 个可估计，20 个形成支持关系','以下所有关系数量均来自同一独立标签分折（sealed_test），没有累加五个上下文。',src='supervision_summary_r2/Y_SUPERVISION_BY_CANCER.tsv')
for x,w,n,t,col in [(.8,4.35,'33','已组织患者与目标轴',GREY),(5.81,4.35,'22','有可估计关系',BLUE),(10.82,4.35,'20','有达到阈值的关系',TEAL)]:
 rect(s,x,2.8,w,3.2,PALE);metric(s,x+.35,3.12,w-.7,n,t,col)
arr(s,5.33,4,.35);arr(s,10.34,4,.35)
band(s,'共 1,330 万条支持关系；这是关联标签规模，不能据此宣称模型预测成功或发现了因果关系。')
# 14
s=slide('问题 3｜解决结果：三癌种 demo','三个癌种都形成了 Y，支持强度差别很大','BRCA / LUAD / COAD 的标签患者分别为 218 / 102 / 91 人。')
pic(s,'12_Demo_Y_results',y0=2.35,h=4.95);band(s,'至少两层严格支持的关系：BRCA 433 条；LUAD 0 条；COAD 0 条。')
# 15
s=slide('关键结果｜必须拆开看 Y 的证据来源','495 条多层严格支持，96.2% 对应肿瘤纯度 State','通路层面的多组学支持已经出现，但目前数量很少。',src='story_revision_20261110_r1/STRICT_MULTILAYER_EXAMPLES.tsv; derived from six cancers with membership==strong_positive')
pic(s,'13_Strict_evidence_composition',x=.7,y0=2.15,w=6.5,h=5.6)
card(s,7.6,2.65,7.5,1.9,'476 条：肿瘤纯度 State','RNA 与 CNV 来源读数支持',BLUE)
card(s,7.6,4.82,7.5,1.9,'19 条：Pathway','17 个通路身份；RNA 与蛋白支持',GOLD)
band(s,'该分折中尚无包含磷酸化层的多层严格支持关系；“Y 已生成”与“证据充分”需要分别判断。')
# 16
s=slide('覆盖缺口｜为什么有些癌种还没有可用 Y？','限制来自协变量完整性与标签分折的有效样本量','当前统计设计保持一致，因此数据条件不足的癌种保留为未标注。',src='adjustment_coverage_r1/ADJUSTMENT_SUPPORT_BY_CANCER.tsv; Y_SUPERVISION_BY_CANCER.tsv')
card(s,.8,2.6,7,3.75,'8 个癌种：所需协变量不足','DLBC、GBM、LAML、LGG\nPCPG、PRAD、SARC、THYM\n主要涉及分期、白细胞或巨噬细胞比例',RED)
card(s,8.2,2.6,7,3.75,'其余缺口：统计可估计性 / 支持','ACC、CHOL、UCS：该分折无可估计关系\nKICH、READ：可估计，但未达支持阈值\n每个关系的有效 n 仍需满足调整后条件',GREY)
band(s,'CPTAC 原 bundle 未接入甲基化白细胞比例；新找到的资源与修订方案见下一页。')
# Source-audit correction, new evidence rather than a new Y result.
s=slide('协变量核实｜问题来自未接入，不能断言没有数据','找到甲基化来源数据；主比较与补充调整分开开展','100 位 CCRCC 患者已有作者发布比例；7 癌种有 720 人的肿瘤探针文件关联。',src='cptac_leukocyte_audit_20261110_r1/COHORT_COVERAGE.tsv; doi:10.1021/acs.jproteome.5c00980')
card(s,.8,2.65,7.0,3.65,'主比较：两平台同一口径','临床 + 纯度 + 既有细胞组成\n两边均不强制甲基化白细胞比例\n按新版本重算关系 Y',TEAL)
card(s,8.2,2.65,7.0,3.65,'补充检验：同一批患者前后比较','有甲基化比例的患者\n分别拟合不加 / 加入该协变量\n保留原算法，比较关系是否稳定',BLUE)
band(s,'比例已下载并匹配；新方案的 Y 尚未重算。探针文件存在，不等于比例已算出。')
# 17
s=slide('最终判断','最后的 Y 是否成功？关系标签已完成，科学目标只完成一部分','把数据交付、跨平台融合和模型证明分别给出结论。')
for yy,lab,body,col in [(2.55,'已经完成','固定身份 + 分层 S + 可读取的 TCGA/TCPA 关系 Y',TEAL),(4.05,'尚未完成','TCPA/CPTAC 统一活性标尺；CPTAC 正式关系证据接入',GOLD),(5.55,'仍待检验','lncRNA 超出 mRNA 的预测增益，以及通路关系的因果支持',BLUE)]:
 rect(s,.8,yy,14.4,1.1,PALE);tx(s,1.04,yy+.3,2.3,.5,lab,24,col,True);tx(s,3.65,yy+.27,11.1,.64,body,23)
band(s,'当前可用于关系建模的 Y 是“带方向与测量层标记的关联支持”，不是统一绝对通路活性。')
# 18
s=slide('下一步｜按当前证据推进','先验证关系 Y 的价值，再扩展跨平台监督','所有比较使用相同癌种、分折与目标；以实测非 RNA 读数评估额外信息。')
card(s,.8,2.65,4.48,3.9,'01 解决 CPTAC 协变量','按共同协变量口径重算两平台 Y\n在实测子集检验甲基化调整',BLUE)
card(s,5.76,2.65,4.48,3.9,'02 检验 lncRNA 增益','lncRNA / mRNA / 二者联合\n在相同非 RNA 目标上对照',TEAL)
card(s,10.72,2.65,4.48,3.9,'03 检验网络与组学增益','逐层加入图结构和其他组学\n排除目标所在层的输入',GOLD)
band(s,'从线性模型到 CCHHGT 沿用同一评价口径；Pathway = 1.0、State = 0.5。')
# appendix
for i in range(3):
 s=slide('附录｜各癌种关系数量',f'33 癌种完整结果（{i+1}/3）','数字为达到支持阈值的关系数；每个癌种只统计一个独立标签分折。',src='supervision_summary_r2/Y_SUPERVISION_BY_CANCER.tsv')
 pic(s,f'14_All_cancer_support_{i+1}',y0=2.18,h=5.85)
s=slide('附录｜如何复核与读取','数据字典、患者评分和关系标签各回答一个问题','详细组成与全量覆盖保留在配套工作簿中，主文使用汇总图。')
card(s,.8,2.6,4.48,3.6,'数据字典','一个目标是什么？\n名称、别名、组成、分支\n固定 canonical ID',TEAL)
card(s,5.76,2.6,4.48,3.6,'患者评分 S','哪些患者测到了？\n读数、算法、参考与单位\n保留平台和缺失信息',BLUE)
card(s,10.72,2.6,4.48,3.6,'关系标签 Y','哪些 lncRNA 得到支持？\n等级、方向、测量层\n按癌种与分折读取',GOLD)
band(s,'165 个癌种 × 上下文数据包通过一致性校验；本报告不将文件校验等同于模型性能验证。')
# Citation/provenance in notes, no small source tables in main deck.
for s in prs.slides:
 for shape in s.shapes:
  assert shape.left>=0 and shape.top>=0
  assert shape.left+shape.width<=prs.slide_width+100 and shape.top+shape.height<=prs.slide_height+100
ppt=O/'Functional_Y_construction_report_v4.pptx';prs.save(ppt)
(O/'STORY_OUTLINE.json').write_text(json.dumps(outline,ensure_ascii=False,indent=2))
source_files=[Q/'Y_SUPERVISION_BY_CANCER.tsv',Q/'FIGURE_SCORE_COVERAGE.tsv',C/'COMPLETE.json',O/'STRICT_MULTILAYER_EXAMPLES.tsv',R/'assemble_functional_relation_y_numerical.py']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
rec={'target_host':'149','slides':len(prs.slides),'pptx':str(ppt),'sha256':sha(ppt),'source_files':{str(p):sha(p) for p in source_files},'key_findings':{'identities':3456,'identities_with_scores':2840,'estimable_cancers':22,'supported_cancers':20,'multilayer_strict_relations':495,'purity_state_strict_relations':476,'pathway_strict_relations':19,'pathway_identities_strict':17},'model_or_label_changed':False}
(O/'PRESENTATION_RECEIPT.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2))
print('PPT_WRITTEN',json.dumps(rec['key_findings']),len(prs.slides),flush=True)
render=O/'render';render.mkdir(exist_ok=True)
cmd=['libreoffice','-env:UserInstallation='+(render/'lo_profile').as_uri(),'--headless','--convert-to','pdf','--outdir',str(render),str(ppt)]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=300);(render/'libreoffice.log').write_text(r.stdout+r.stderr);assert r.returncode==0,(r.stdout,r.stderr)
pdf=render/(ppt.stem+'.pdf');assert pdf.exists()
info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1));assert pages==len(prs.slides)
subprocess.run(['pdftoppm','-png','-r','90',str(pdf),str(render/'slide')],check=True,timeout=300)
imgs=sorted(render.glob('slide-*.png'));assert len(imgs)==pages
for start in [0,12]:
 subset=imgs[start:start+12];tw,th=640,385;sheet=Image.new('RGB',(3*tw,((len(subset)+2)//3)*th),'#DCE4E9');draw=ImageDraw.Draw(sheet)
 for j,p in enumerate(subset):
  with Image.open(p) as im:im=ImageOps.contain(im.convert('RGB'),(tw-12,th-25));xx=(j%3)*tw+(tw-im.width)//2;yy=(j//3)*th+23;sheet.paste(im,(xx,yy));draw.text(((j%3)*tw+10,(j//3)*th+5),str(start+j+1),fill='black')
 sheet.save(render/f'contact_{start+1:02}.png')
rec.update(pdf=str(pdf),pdf_sha256=sha(pdf),preview_count=pages)
(O/'PRESENTATION_RECEIPT.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2));print('RENDER_COMPLETE',str(pdf),flush=True)
