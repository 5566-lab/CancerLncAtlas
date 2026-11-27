from pathlib import Path
import sys, socket, json, csv, gzip, hashlib
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');WORK=ROOT/'kegg_cancer_redefinition_20261111_r1'
BASE=WORK/'measured_y_definition_r1';REVIEW=BASE/'final_definition_review_r1';CAT=ROOT/'canonical_targets_r9_cancer_evidence_20261111_r6'
OUT=WORK/'reader_delivery_fixed_y_20261112';OUT.mkdir(exist_ok=True);FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
def savejson(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
savejson(OUT/'LAUNCH_PREFLIGHT.json',dict(target_host='149',operation='draw fixed Y and actual protein-list integration',training=False,lncrna_association=False))
assert (REVIEW/'MEASURED_Y_VALIDATION.json').exists()
for p in [ROOT/'python_deps',Path('/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps')]:sys.path.append(str(p))
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
font_file='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_file);font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
FN=font_manager.FontProperties(fname=font_file).get_name()
NAVY='#19384C';TEAL='#127F88';BLUE='#477EAB';GOLD='#D69B36';ORANGE='#CB6544';GRAY='#5E707D';PALE='#EDF4F6'
plt.rcParams.update({'font.family':FN,'font.size':15,'axes.unicode_minus':False,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def rows(p):
    op=gzip.open if str(p).endswith('.gz') else open
    with op(p,'rt',encoding='utf-8') as f:return list(csv.DictReader(f,delimiter='\t'))
data=json.loads((REVIEW/'CATALOG_REVIEW.json').read_text()); members=rows(BASE/'member_measurements_r1/READER_MEMBER_SUMMARY.tsv')
cluster=ROOT/'protein_similarity_integration_20261111_r1'
ids=['CLA-P-2F5E3CD19372C9E7','REF:WikiPathways:WP531','CLA-P-922863C8FB5E9DF9']
sets={i:set() for i in ids}
for r in rows(cluster/'PROTEIN_COMPOSITION.tsv.gz'):
    if r['node_id'] in sets:sets[r['node_id']].add(r['uniprot_id'])
assert [len(sets[i]) for i in ids]==[26,23,15]
J=np.array([[len(sets[a]&sets[b])/len(sets[a]|sets[b]) for b in ids] for a in ids]);core=set.intersection(*sets.values());union=set.union(*sets.values())
assert len(core)==14 and len(union)==26
roles=rows(CAT/'SOURCE_TASK_ROLES.tsv'); programs=[r for r in roles if r['source_target_id'].startswith(('PROGENY:','DOROTHEA:','KINASE-'))];assert len(programs)==860 and all(r['display_category']=='Pathway' for r in programs)
def canvas(h=5.3):
    fig,ax=plt.subplots(figsize=(12.1,h));ax.set(xlim=(0,12),ylim=(0,5.3));ax.axis('off');return fig,ax
def box(ax,x,y,w,h,title,body='',color=TEAL,size=17):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.04,rounding_size=.08',facecolor=PALE,edgecolor=color,lw=1.5))
    ax.text(x+w/2,y+h*(.70 if body else .5),title,ha='center',va='center',fontsize=size,color=color,fontweight='bold')
    if body:ax.text(x+w/2,y+h*.29,body,ha='center',va='center',fontsize=14,color=NAVY,linespacing=1.6)
def arrow(ax,x,y,x2,y2):ax.add_patch(FancyArrowPatch((x,y),(x2,y2),arrowstyle='-|>',mutation_scale=18,lw=1.5,color=GRAY))
def figsave(fig,name):
    for ext in ['png','pdf']:fig.savefig(FIG/(name+'.'+ext),dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig);return FIG/(name+'.png')
fig,ax=canvas();box(ax,.1,3.4,3.55,1.5,'平台不同','TCPA 抗体信号\nCPTAC 质谱蛋白／位点');box(ax,4.2,3.4,3.55,1.5,'定义不同','同名异义／异名同义\n总通路／子过程／特定变异');box(ax,8.3,3.4,3.55,1.5,'评分不同','成员、权重、背景不同\n同一生物学身份有多种读数')
for x in [1.9,6,10.1]:arrow(ax,x,3.3,6,2.5)
box(ax,2,0.6,8,1.7,'精确通路身份 + 患者多维观测','固定 Y，再评价 lncRNA 对该通路的预测贡献',size=20);f1=figsave(fig,'01_problem_solution')
fig,ax=plt.subplots(figsize=(12,4.7));vals=[199,23,255];colors=[TEAL,GOLD,BLUE];left=0
for n,c,label in zip(vals,colors,['官方癌症分类','恶性肿瘤疾病注释','癌症机制文献']):
    ax.barh([1],n,left=left,height=.46,color=c,label=f'{label}：{n}');ax.text(left+n/2,1,str(n),ha='center',va='center',color='white',fontsize=20,fontweight='bold');left+=n
ax.barh([0],181,color='#CFD8DD',height=.46);ax.text(181/2,0,'181',ha='center',va='center',fontsize=20,color=NAVY)
ax.set(yticks=[1,0],yticklabels=['纳入主干 477','排除 181'],xlim=(0,500),xlabel='KEGG NETWORK 定义数');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.23),frameon=False,fontsize=13);ax.spines[['left','right','top']].set_visible(False);f2=figsave(fig,'02_scope')
fig,ax=canvas();steps=[('统一分子身份','HGNC／UniProt\n保留原始蛋白列表'),('组成相似度与聚类','Jaccard；complete linkage\n同组两两 J ≥ 0.5'),('核对生物学范围','反应／功能／层级\n变异、感染及暴露条件'),('建立不同类型的关系','等价身份／父子过程\n相似组成／成员调控')]
for i,(a,b) in enumerate(steps):
    x=.12+i*3;box(ax,x,2.6,2.7,2.0,a,b,size=15)
    if i<3:arrow(ax,x+2.75,3.6,x+2.96,3.6)
box(ax,1.1,.35,9.8,1.35,'其他库完整成员列表继续保留','主干成员不被上下文蛋白静默扩充；相似来源可以连接，而非一律合并',size=18);f3=figsave(fig,'03_integration')
fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={'width_ratios':[1.1,1]});im=ax.imshow(J,vmin=0,vmax=1,cmap='YlGnBu');labels=['KEGG MMR\n26 proteins','WikiPathways WP531\n23 proteins','Reactome MMR\n15 proteins'];ax.set(xticks=range(3),yticks=range(3),xticklabels=['KEGG','WP531','Reactome'],yticklabels=labels)
for i in range(3):
    for j in range(3):ax.text(j,i,f'{J[i,j]:.2f}',ha='center',va='center',color='white' if J[i,j]>.7 else NAVY,fontsize=21)
ax.set_title('真实蛋白列表的 Jaccard 相似度',fontsize=17,pad=15);bx.axis('off');bx.text(.05,.88,'归入同一 Mismatch repair 身份',color=TEAL,fontsize=20,fontweight='bold');bx.text(.05,.59,'共同核心：14 个蛋白\n组成并集：26 个蛋白\n三份来源列表分别保留',fontsize=19,color=NAVY,linespacing=1.7);bx.text(.05,.18,'依据：共同修复过程 + 功能范围核对\n组成聚类提供候选，功能范围决定身份归并',fontsize=15,color=GRAY,linespacing=1.6);fig.tight_layout();f4=figsave(fig,'04_mmr_example')
fig,ax=canvas();box(ax,.3,2.5,5.1,2,'SMO 激活变异 → Hedgehog','KEGG N00017\n29 个蛋白；要求激活变异证据',TEAL,19);box(ax,6.6,2.5,5.1,2,'PTCH1 失活变异 → Hedgehog','KEGG N00010\n29 个蛋白；要求失活变异证据',ORANGE,19);arrow(ax,5.5,3.5,6.5,3.5);ax.text(6,4.65,'J = 1.00',ha='center',fontsize=20,color=NAVY,fontweight='bold');box(ax,1.3,.4,9.4,1.4,'同一组成簇，保留两个精确机制身份','相同蛋白列表不能替代上游事件与机制方向',NAVY,19);f5=figsave(fig,'05_same_list_different_mechanism')
fig,ax=canvas();box(ax,4.1,2.1,3.8,1.4,'一个精确通路 k','固定的生物学身份与成员',NAVY,21)
nodes=[(.2,3.8,'编码 RNA','成员表达'),(4.1,3.8,'蛋白丰度','抗体／质谱分别保留'),(8,3.8,'磷酸化','位点／表位及定义'),(.2,.3,'CNV／甲基化','基因值／启动子值'),(4.1,.3,'通路来源读数','明确等价定义'),(8,.3,'TF／激酶程序','分子身份连接成员')]
for x,y,t,b in nodes:
    box(ax,x,y,3.8,1.1,t,b,size=16)
    arrow(ax,x+1.9,y+(.0 if y>3 else 1.15),6,3.55 if y>3 else 2.05)
f6=figsave(fig,'06_Y_structure')
fig,axs=plt.subplots(1,2,figsize=(12,4.7),sharey=True)
for ax,layer,title in zip(axs,['PROTEIN','PHOSPHO'],['蛋白成员实测覆盖','磷酸化成员实测覆盖']):
    for j,(cohort,color,label) in enumerate([('TCGA',TEAL,'TCPA / TCGA'),('CPTAC',BLUE,'CPTAC')]):
        vals=[int(next(r for r in members if r['case']==cohort+'_'+c and r['layer']==layer)['pathways_with_member_measurement']) for c in ['BRCA','LUAD','COAD']]
        bars=ax.bar(np.arange(3)+(j-.5)*.34,vals,width=.32,color=color,label=label)
        ax.bar_label(bars,padding=3,fontsize=14)
    ax.set(xticks=np.arange(3),xticklabels=['BRCA','LUAD','COAD'],ylim=(0,530),title=title);ax.axhline(477,color=GRAY,ls='--',lw=1);ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2,fontsize=12)
axs[0].set_ylabel('至少有一项无歧义原始成员测量的主干通路数');fig.tight_layout();f7=figsave(fig,'07_measured_coverage')
fig,ax=canvas();box(ax,.2,2.5,5.5,2,'334 个 Reference 机制','无需额外变异／感染／暴露前提\n按实际成员与来源观测组织 Y',TEAL,22);box(ax,6.3,2.5,5.5,2,'143 个条件性机制','事件存在、事件不存在、未知分别记录\n成员测量值始终保留',ORANGE,22);box(ax,1.3,.4,9.4,1.4,'分子被测到 ≠ 特定机制条件已证实','基因级突变标记、连续 CNV 或烟草史不能替代每个具体机制的事件判定',NAVY,18);f8=figsave(fig,'08_condition')
fig,ax=canvas();ax.text(6,4.7,'Yhat[i,a] = intercept[a] + C[i]γ[a] + Xlnc[i]βlnc[a] + XmRNA[i]βmRNA[a]',ha='center',va='center',fontsize=19,color=NAVY)
for x,t,b in [(.2,'M0-L','临床／独立纯度 + lncRNA'),(4.2,'M0-M','临床／独立纯度 + mRNA'),(8.2,'M0-LM','临床／独立纯度 + 二者')]:box(ax,x,2.3,3.6,1.6,t,b,size=22)
box(ax,.8,.2,10.4,1.5,'同一患者、同一观测 Y、同一掩码和分折','RNA 系数用 Lasso；临床项不受惩罚；无相关性准入\n共享观测只拟合一次，随后映射到精确通路',NAVY,19);f9=figsave(fig,'09_m0')
fig,ax=canvas();items=[(.2,3.1,'477','癌症机制身份'),(4.2,3.1,'197','State 身份'),(8.2,3.1,'860','TF／激酶／PROGENy 来源定义')]
for x,y,t,b in items:box(ax,x,y,3.6,1.6,t,b,size=31)
box(ax,.7,.35,10.6,1.8,'最终 Y：按通路组织的患者多维观测','统一生物学身份，保留平台与方法差异\n固定成员、来源、适用条件及缺失；lncRNA 贡献由后续模型评价',NAVY,21);f10=figsave(fig,'10_result')

prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
def text(slide,x,y,w,h,content,size=20,color=NAVY,bold=False):
    shape=slide.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=shape.text_frame;tf.word_wrap=True
    for i,line in enumerate(content.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name=FN;p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color[1:])
    return shape
def slide(section,title,fig,caption,source):
    sl=prs.slides.add_slide(prs.slide_layouts[6]);bar=sl.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.12));bar.fill.solid();bar.fill.fore_color.rgb=RGBColor.from_string(TEAL[1:]);bar.line.fill.background()
    text(sl,.48,.24,12,.35,section,13,TEAL,True);text(sl,.48,.68,12.3,.68,title,29,NAVY,True)
    from PIL import Image
    im=Image.open(fig);iw,ih=im.size;scale=min(12.35/iw,4.9/ih);w,h=iw*scale,ih*scale
    sl.shapes.add_picture(str(fig),Inches((13.333-w)/2),Inches(1.42+(4.9-h)/2),width=Inches(w),height=Inches(h))
    text(sl,.52,6.36,12.2,.55,caption,17,NAVY);text(sl,.52,7.04,11.8,.22,source,9,GRAY);text(sl,12.42,7.02,.5,.25,str(len(prs.slides)),10,GRAY)
slide('背景与问题','研究单位：癌种 × lncRNA × 精确通路',f1,'需要统一的是生物学身份及测量定义；不同来源可以共同支持同一通路。','CancerLncAtlas | BRCA / LUAD / COAD')
slide('问题 1 → 筛选方法 → 结果','477 个癌症机制如何从 658 个定义中得到？',f2,'NETWORK 是精确机制与模块；477 不是 477 张 KEGG 大通路图。','KEGG_SOURCE_SCOPE_DECISIONS.tsv；原始 N 编号、癌症类别、疾病注释及文献证据逐条保留')
slide('问题 2 → 跨库对齐','按真实蛋白列表聚类，再核对功能范围',f3,'上下文组成连接：至少 3 个共同蛋白，且注释集合富集检验 BH q ≤ 0.05。','PROTEIN_COMPOSITION.tsv.gz / CLUSTER_MEMBERSHIP.tsv / BACKBONE_ALIGNMENT.tsv')
slide('实例：异名与不同成员 → 同一功能身份','MMR：三份来源定义如何归到一起？',f4,'共同核心、完整来源列表和来源读数同时保留；主干成员使用原始 KEGG 定义。','KEGG MMR / WikiPathways WP531 / Reactome R-HSA-5358508；统一 ID：CLA-P-922863C8FB5E9DF9')
slide('实例：相同成员 → 不同机制','蛋白列表完全相同，也可能是两个精确通路',f5,'组成相似用于关联；变异方向与机制条件决定是否属于同一个生物学身份。','KEGG NETWORK N00017 / N00010')
slide('问题 3 → Y 的定义','一个通路对应一组可追溯的患者观测',f6,'Y[c,i,k] = {(观测 a，数值，实测掩码，通路归属)}；同一测量跨通路只保存一份。','OBSERVATIONS / PATHWAY_OBSERVATIONS / MATRICES；平台特异单位及方法版本分别保留')
slide('结果：成员测量覆盖','三癌种中，哪些通路已有蛋白／磷酸化观测？',f7,'统计标准为“至少一项原始成员实测”；这不表示该通路全部成员均被测量。','READER_MEMBER_SUMMARY.tsv；仅开发患者折1—4；不使用 lncRNA 关联决定覆盖')
slide('问题 4 → 适用条件','成员状态与机制条件分开保存',f8,'当前条件性机制的精确事件判定为未知；未知不作为阴性，也不删除成员观测。','PATHWAY_CONDITIONS.tsv / CONDITION_APPLICABILITY.parquet / CONDITION_EVIDENCE_REVIEW.json')
slide('M0 对照设计','用患者输入预测固定观测，再解释通路关系',f9,'与 CCHHGT 比较时固定 Y 和输入组学；预测非 RNA 目标时排除目标层及直接派生输入。','M0_RELATION_BASELINE.rst；Pathway=1.0、State=0.5；辅助损失由任务配置指定')
slide('解决结果','固定多维 Y，为精确通路关系建模提供共同目标',f10,'TCPA 和 CPTAC 保留各自观测尺度；通路身份统一，实测信息与缺失可直接读取。','固定读数版本与成员映射；TLS12 使用 GDC 12 基因定义的编码背景版本')
ppt=OUT/'CancerLncAtlas_KEGG癌症主干与多维Y_20261112.pptx';prs.save(ppt)
savejson(OUT/'FIGURE_DATA.json',dict(target_host='149',catalog=data,MMR=dict(source_ids=ids,protein_counts=[len(sets[i]) for i in ids],jaccard=J.tolist(),common_core=len(core),union=len(union)),member_coverage=members,program_source_rows=860,representation='PATHWAY_GROUPED_PATIENT_OBSERVATIONS',lncrna_associations_used=False))
print('REPORT_BUILT',str(ppt),flush=True)
