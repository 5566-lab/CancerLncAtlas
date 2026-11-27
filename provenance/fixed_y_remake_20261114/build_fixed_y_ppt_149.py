from pathlib import Path
import sys,json,socket,textwrap
assert socket.gethostname().split('.')[0]=='149'
R=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1');W=R/'kegg_cancer_redefinition_20261111_r1';O=W/'fixed_y_remake_20261114_r1';D=O/'figure_data';F=O/'figures';F.mkdir(exist_ok=True);OUT=O/'deliverables';OUT.mkdir(exist_ok=True)
for p in [R/'python_deps',Path('/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps')]:sys.path.append(str(p))
import numpy as np
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
from scipy.cluster.hierarchy import linkage,dendrogram
from scipy.spatial.distance import squareform
for weight in ['Regular','Bold']:font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-'+weight+'.ttc')
FONT=font_manager.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc').get_name()
NAVY='19384C';TEAL='127F88';ORANGE='CB6544';BLUE='477EAB';GRAY='5E707D';PALE='EDF4F6'
plt.rcParams.update({'font.family':FONT,'font.size':15,'axes.unicode_minus':False,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def rd(p):return pd.read_csv(p,sep='\t',keep_default_na=False)
S=rd(D/'COHORT_SUMMARY.tsv');L=rd(D/'LAYER_COVERAGE.tsv');E=json.loads((D/'EXAMPLES.json').read_text());B=O/'Y_bundle';coverage=rd(B/'TARGET_COVERAGE.tsv')
def meta(k,c):return rd(D/(k+'_'+c+'_observations.tsv'))
def vals(k,c):return pd.read_parquet(D/(k+'_'+c+'_values.parquet'))
def savefig(fig,name):
    for ext in ['png','pdf']:fig.savefig(F/(name+'.'+ext),dpi=240,bbox_inches='tight',facecolor='white')
    plt.close(fig);return F/(name+'.png')
figures={};figure_records={}
# Exact list overlap, recomputed from the current catalog.
sources=E['MMR']['source_lists'];order=[next(k for k in sources if 'KEGG' in k),next(k for k in sources if 'WIKIPATHWAYS' in k),next(k for k in sources if 'REACTOME' in k)];sets=[set(sources[k]) for k in order]
j=np.array([[len(a&b)/len(a|b) for b in sets] for a in sets]);core=len(set.intersection(*sets));union=len(set.union(*sets))
fig=plt.figure(figsize=(6.4,6.2));grid=fig.add_gridspec(2,1,height_ratios=[1,3.5],hspace=.34);tree=fig.add_subplot(grid[0]);dendrogram(linkage(squareform(1-j,checks=False),method='average'),labels=['KEGG','WikiPathways','Reactome'],ax=tree,color_threshold=0,above_threshold_color='#127F88');tree.set_ylabel('1 − J',fontsize=12);tree.set_title('MMR 来源局部聚类：平均连接',fontsize=14);tree.tick_params(labelsize=11);ax=fig.add_subplot(grid[1]);im=ax.imshow(j,cmap='YlGnBu',vmin=0,vmax=1)
names=['KEGG\n26 蛋白','WikiPathways\n23 蛋白','Reactome\n15 蛋白'];ax.set_xticks(range(3),names);ax.set_yticks(range(3),names)
ax.set_xticklabels(['KEGG','Wiki-\nPathways','Reactome'],fontsize=12)
for a in range(3):
    for b in range(3):ax.text(b,a,f'{j[a,b]:.2f}',ha='center',va='center',fontsize=27,color='white' if j[a,b]>.7 else '#19384C')
ax.set_title('实际蛋白列表的 Jaccard 相似度',pad=12,fontsize=15);figures['mmr']=savefig(fig,'MMR_真实成员相似度');figure_records['MMR']=dict(source_ids=order,protein_counts=[len(s) for s in sets],jaccard=j.tolist(),core=core,union=union,example_clustering='average linkage of Jaccard distance for the three current MMR source definitions')
# Cohort counts.
fig,ax=plt.subplots(figsize=(11,4.5));x=np.arange(3)
for co,offset,color in [('TCGA',-.18,'#'+TEAL),('CPTAC',.18,'#'+ORANGE)]:
    numbers=[int(S.loc[S['case'].eq(co+'_'+ca),'patients'].iloc[0]) for ca in ['BRCA','LUAD','COAD']];bars=ax.bar(x+offset,numbers,.34,label=co,color=color)
    for bar,n in zip(bars,numbers):ax.text(bar.get_x()+bar.get_width()/2,n+14,str(n),ha='center',fontsize=19,fontweight='bold')
ax.set_xticks(x,['BRCA 乳腺癌','LUAD 肺腺癌','COAD 结肠癌']);ax.set_ylabel('开发集患者数');ax.set_ylim(0,1000);ax.legend(frameon=False);figures['patients']=savefig(fig,'三癌种真实患者数')
# Measured member dimensions are a different quantity from patient coverage.
fig,axes=plt.subplots(1,2,figsize=(12,4.4));memb=L[L.kind.eq('member_measurement')]
for ax,layer,title in zip(axes,['PROTEIN','PHOSPHO'],['总蛋白成员实测维度','磷酸化成员实测维度']):
    for co,offset,color in [('TCGA',-.18,'#'+TEAL),('CPTAC',.18,'#'+ORANGE)]:
        numbers=[]
        for ca in ['BRCA','LUAD','COAD']:
            row=memb[memb['case'].eq(co+'_'+ca)&memb.layer.eq(layer)];numbers.append(int(row.measured_dimensions.iloc[0]) if len(row) else 0)
        bars=ax.bar(x+offset,numbers,.34,label=co+' / '+('TCPA' if co=='TCGA' else 'MS'),color=color)
        for bar,n in zip(bars,numbers):ax.text(bar.get_x()+bar.get_width()/2,n+max(numbers)*.03,str(n),ha='center',fontsize=13)
    ax.set_xticks(x,['BRCA','LUAD','COAD']);ax.set_title(title);ax.set_ylim(0,max(ax.get_ylim())*1.18);ax.legend(frameon=False,fontsize=12)
figures['dimensions']=savefig(fig,'蛋白与磷酸化实测维度')
# Member-level patient coverage, not completeness of every pathway.
cases=S['case'].tolist();ls=['RNA','PROTEIN','PHOSPHO','CNV','METHYLATION'];a=np.zeros((6,5))
for i,case in enumerate(cases):
    for k,layer in enumerate(ls):
        rr=memb[memb['case'].eq(case)&memb.layer.eq(layer)]
        if len(rr):a[i,k]=100*rr.patients_measured.iloc[0]/rr.patients_total.iloc[0]
fig,ax=plt.subplots(figsize=(11.5,4.5));im=ax.imshow(a,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto');ax.set_yticks(range(6),cases);ax.set_xticks(range(5),['编码 RNA','总蛋白','磷酸化','基因 CNV','启动子甲基化'])
for i in range(6):
    for k in range(5):ax.text(k,i,f'{a[i,k]:.1f}%',ha='center',va='center',fontsize=17,color='white' if a[i,k]>60 else '#'+NAVY)
fig.colorbar(im,ax=ax,label='至少一项成员实测的患者比例（%）',shrink=.8);figures['coverage']=savefig(fig,'成员层患者覆盖');figure_records['member_patient_coverage']=a.tolist()
# Real pathway heatmaps: matched original measurements are never averaged across platforms.
fig,axes=plt.subplots(1,2,figsize=(12,5.3));cm=plt.colormaps['RdBu_r'].copy();cm.set_bad('#DFE4E7');heat_records={}
def short(row):
    d=json.loads(row.definition_json);d=d[0] if isinstance(d,list) else d
    if row.feature_id.startswith('RPPA500:'):return row.feature_id.split(':',1)[1]
    name=d.get('reference_symbol') or row.feature_id
    return name+(' '+d['site'] if d.get('site') else '')
for ax,co in zip(axes,['TCGA','CPTAC']):
    case=co+'_BRCA';m=meta('PI3K',case);v=vals('PI3K',case);m=m[m.kind.eq('member_measurement')&m.layer.isin(['PROTEIN','PHOSPHO'])].copy();m['label']=[short(r) for r in m.itertuples()]
    selected=pd.concat([m[m.layer.eq(layer)].sort_values(['patients_measured','label'],ascending=[False,True]).head(4) for layer in ['PROTEIN','PHOSPHO']]);vv=v[selected.observation_id];z=(vv-vv.mean())/vv.std(ddof=0).replace(0,np.nan);show=z.head(30)
    im=ax.imshow(show.to_numpy().T,cmap=cm,vmin=-2.5,vmax=2.5,aspect='auto');ax.set_yticks(range(len(selected)),selected.label,fontsize=11);ax.set_xticks([0,14,29],['1','15','30']);ax.set_xlabel('原始顺序前 30 名患者');ax.set_title('TCGA / TCPA BRCA' if co=='TCGA' else 'CPTAC BRCA');ax.axhline(3.5,color='white',lw=2)
    heat_records[case]=selected[['observation_id','feature_id','label','layer','patients_measured']].to_dict('records')
fig.subplots_adjust(wspace=.52);cax=fig.add_axes([.39,-.06,.25,.025]);fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[-2.5,0,2.5],label='分子内 z 值（仅展示）');figures['pi3k']=savefig(fig,'PI3K_真实蛋白与磷酸化观测');figure_records['PI3K_selection']=heat_records
# CD8 source definitions retain their actual measured distributions and methods.
fig,axes=plt.subplots(1,3,figsize=(12,4.5));m=meta('cell_composition','TCGA_BRCA');v=vals('cell_composition','TCGA_BRCA');selnames=['cibersort_relative::T.cells.CD8','xCell::CD8+ T-cells','MCPcounter::CD8 T cells'];titles=['LM22 CD8 相对比例','xCell CD8 来源读数','MCP-counter CD8 来源读数'];records=[]
for ax,source,title,color in zip(axes,selnames,titles,['#'+TEAL,'#'+BLUE,'#'+ORANGE]):
    r=m[m.source_target_id.eq(source)].iloc[0];arr=v[r.observation_id].dropna();ax.hist(arr,bins=24,color=color,alpha=.88);ax.set_title(title,fontsize=15);ax.set_xlabel('保留的原始数值');ax.set_ylabel('患者数');ax.text(.97,.95,f'n={len(arr)}',ha='right',va='top',transform=ax.transAxes,fontsize=14);records.append(dict(source=source,method=r['method'],units=r.units,n=len(arr),observation_id=r.observation_id))
fig.tight_layout();figures['cell']=savefig(fig,'细胞组成_CD8多来源读数');figure_records['cell_sources']=records
# SBS shows zeros and missing data differently.
m=meta('sbs_spectrum','TCGA_BRCA');v=vals('sbs_spectrum','TCGA_BRCA');wanted=['SBS1','SBS2','SBS3','SBS5','SBS13','SBS40'];selected=pd.concat([m[m.source_target_id.eq('SBS::'+s)] for s in wanted]);has=v.notna().any(axis=1);vv=v.loc[has,selected.observation_id].head(35)
fig,ax=plt.subplots(figsize=(11.8,3.9));arr=np.log1p(vv.to_numpy().T);im=ax.imshow(arr,cmap='YlGnBu',aspect='auto');ax.set_yticks(range(len(selected)),wanted);ax.set_xticks([0,16,34],['1','17','35']);ax.set_xlabel('有 SBS 观测的患者，按原始顺序取前 35 名');fig.colorbar(im,ax=ax,label='log(1 + 原始读数)，仅用于展示',shrink=.8);figures['sbs']=savefig(fig,'SBS_真实患者读数');figure_records['SBS']=dict(TCGA_BRCA_measured_patients=int(has.sum()),TCGA_BRCA_total=len(v),shown_signatures=wanted,shown_patients=len(vv),CPTAC_BRCA_measured_dimensions=len(meta('sbs_spectrum','CPTAC_BRCA')))
# Three distinct facts: a definition exists, observations exist, exact conditions permit supervision.
fig,axes=plt.subplots(1,2,figsize=(12,4.6));xx=np.arange(6)
axes[0].bar(xx,S.pathways_with_measurement,color='#B8DCDD',label='有实测观测');axes[0].bar(xx,S.pathways_with_supervision,color='#'+TEAL,label='条件允许使用')
axes[0].set_title('Pathway：成员观测与机制条件分开');axes[0].set_ylim(0,640);axes[0].set_yticks([0,100,200,300,400,500]);axes[0].axhline(476,color='#'+GRAY,ls='--',lw=1)
for i,r in enumerate(S.itertuples()):axes[0].text(i,482,str(r.pathways_with_measurement),ha='center',fontsize=12);axes[0].text(i,r.pathways_with_supervision/2,str(r.pathways_with_supervision),ha='center',color='white',fontsize=15)
axes[0].legend(frameon=False,fontsize=11,loc='upper left')
axes[1].bar(xx,S.states_with_measurement,color=['#'+TEAL if c=='TCGA' else '#'+ORANGE for c in S.cohort]);axes[1].axhline(48,color='#'+GRAY,ls='--',lw=1);axes[1].set_ylim(0,55);axes[1].set_title('State：当前有实测读数的条目')
for i,n in enumerate(S.states_with_measurement):axes[1].text(i,n+1,str(n),ha='center',fontsize=16)
for ax in axes:ax.set_xticks(xx,[s.replace('_','\n') for s in cases],fontsize=10);ax.set_ylabel('条目数（至少 1 名患者）')
fig.tight_layout();figures['readiness']=savefig(fig,'Y条目与实际观测覆盖')
prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
def tx(sl,x,y,w,h,text,size=22,color=NAVY,bold=False):
    sh=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=sh.text_frame;tf.word_wrap=True
    for i,line in enumerate(text.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line;p.font.name='Noto Sans CJK JP';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
    return sh
def slide(title,kicker='CancerLncAtlas',footer='数据：BRCA / LUAD / COAD · 当前固定 Y · 2026-11-14'):
    s=prs.slides.add_slide(prs.slide_layouts[6]);bar=s.shapes.add_shape(MSO_SHAPE.RECTANGLE,0,0,prs.slide_width,Inches(.1));bar.fill.solid();bar.fill.fore_color.rgb=RGBColor.from_string(TEAL);bar.line.fill.background();tx(s,.5,.25,12,.4,kicker,13,TEAL,True);tx(s,.5,.78,12.3,.9,title,30,bold=True);tx(s,.5,7.1,11.9,.25,footer,9,GRAY);tx(s,12.4,7.04,.4,.3,str(len(prs.slides)),11,GRAY);return s
def card(sl,x,y,w,h,title,body='',color=TEAL,ts=22,bs=19):
    sh=sl.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,Inches(x),Inches(y),Inches(w),Inches(h));sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(PALE);sh.line.color.rgb=RGBColor.from_string(color);sh.line.width=Pt(1.2);tx(sl,x+.18,y+.15,w-.36,.7,title,ts,color,True)
    if body:tx(sl,x+.18,y+.95,w-.36,h-1.05,body,bs)
def arrow(sl,x,y,w=.5):
    sh=sl.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,Inches(x),Inches(y),Inches(w),Inches(.32));sh.fill.solid();sh.fill.fore_color.rgb=RGBColor.from_string(GRAY);sh.line.fill.background()
def pic(sl,key,x=.75,y=1.9,w=11.8,h=4.8):
    path=figures[key];a,b=Image.open(path).size;scale=min(w/a,h/b);ww=a*scale;hh=b*scale;sl.shapes.add_picture(str(path),Inches(x+(w-ww)/2),Inches(y+(h-hh)/2),width=Inches(ww),height=Inches(hh))
sl=slide('重新固定 Y：通路身份统一，患者观测分层保留','研究目标：癌种 × lncRNA × 精确通路')
card(sl,.7,2,3.8,2.5,'476 条 Pathway','KEGG 癌症主干\n同成员条目已归并',TEAL,28,21);card(sl,4.8,2,3.8,2.5,'48 个 State','细胞组成与 SBS 收为两组\n其余 46 项保留',BLUE,28,21);card(sl,8.9,2,3.7,2.5,'1,910 名患者','三癌种 · 两个队列\n固定开发集',ORANGE,27,21);tx(sl,.8,5.1,11.8,1.2,'最终 Y 已形成 6 份患者观测矩阵。\n每个目标展开为具体分子与来源读数，并保留缺失和机制条件。',25,bold=True)
sl=slide('遇到的问题，以及这次如何解决','问题 → 方法 → 结果')
for x,title,body,color in [(.6,'跨库名称与范围不一致','蛋白组成生成候选\n功能范围决定身份归并\n→ 一个主名称，来源可展开',TEAL),(4.85,'平台测量对象不同','保留抗体、蛋白、位点身份\n按通路组织多维观测\n→ 同一通路下分层展示',BLUE),(9.1,'State 并列条目过多','91 个细胞组成 + 60 个 SBS\n分别转入两个详情页\n→ 197 个入口减为 48 个',ORANGE)]:card(sl,x,1.95,3.65,3.9,title,body,color,22,19)
tx(sl,.8,6.1,11.8,.6,'统一的是生物学条目与数据结构；原始测量数值和来源方法共同保留。',22)
sl=slide('来源定义如何进入当前主目录','通路归并方法')
for x,title,body in [(.6,'来源定义','KEGG / Reactome / WikiPathways\n完整原始成员列表'),(4.85,'蛋白组成与范围','计算 Jaccard 相似度\n核对过程、层级与机制'),(9.1,'统一主条目','相同过程：同一身份\n相关过程：上下文连接')]:card(sl,x,2,3.65,2.6,title,body,ts=23,bs=19)
arrow(sl,4.35,3.1,.4);arrow(sl,8.6,3.1,.4);tx(sl,.9,5.1,11.6,1.15,'本轮主干：477 条来源机制 → 476 个主条目。\n蛋白成员完全相同的 SMO / PTCH1 两条机制，归入共同下游模块。',25,bold=True)
sl=slide('MMR：三个数据库，共用一个“错配修复”入口','真实例子 1：跨数据库归并',footer='来源：KEGG MMR / WikiPathways DNA mismatch repair（WP531）/ Reactome MMR')
pic(sl,'mmr',.6,1.85,6.1,4.75);card(sl,7,2.2,5.6,3.8,'错配修复 · Mismatch repair',f'共同核心：{core} 个蛋白\n来源并集：{union} 个蛋白\n三份来源列表与读数分别保留',TEAL,24,22);tx(sl,7.1,6.15,5.3,.65,'成员相似提供证据；修复过程与功能范围共同确定身份。',17)
sl=slide('Hedgehog：完全相同成员合为一个主条目','真实例子 2：J = 1.00')
card(sl,.7,1.95,5.6,2.1,'SMO 激活变异','N00017 · 29 个蛋白',TEAL,27,23);card(sl,7,1.95,5.6,2.1,'PTCH1 失活变异','N00010 · 相同的 29 个蛋白',ORANGE,27,22)
card(sl,1.5,4.55,10.3,1.9,'Hedgehog 信号：SMO/PTCH1 共同下游模块','原始事件说明在详情中保留；患者观测按唯一分子与读数组织。',TEAL,26,19)
sl=slide('WNT：相似来源提供组成证据，不抹去层级','真实例子 3：同一信号系统中的不同范围')
ctx=E['WNT'].get('context',[])[:3]
for i,r in enumerate(ctx):
    name=str(r['source_name']).replace('REACTOME_','').replace('_',' ');name=textwrap.fill(name,width=36)
    card(sl,.6,1.9+i*1.5,6.4,1.32,name,'',TEAL,17,15);tx(sl,7.3,2.04+i*1.5,5.2,1,f'J = {float(r["jaccard"]):.2f}；共享 {int(r["shared_proteins"])} 个蛋白',22)
tx(sl,.8,6.45,11.8,.45,'保留原来源的成员、范围与反应关系，连接到当前 WNT 精确主干；组成关联不自动改写来源身份。',16)
sl=slide('最终 Y 是什么：每条通路对应一组患者观测','Y 的定义')
tx(sl,.9,1.85,11.6,.8,'Y（患者，通路）＝ { RNA、蛋白、磷酸化、CNV、甲基化、来源读数 }',25,TEAL,True)
for x,title,body in [(.8,'每一列是什么','明确的分子或来源评分\n保留 ID、方法与单位'),(4.85,'每个值是什么','该患者的原始观测值\n缺失单独保存在掩码中'),(8.9,'每个目标是什么','一组唯一观测的归属\n共享观测只存储一次')]:card(sl,x,3.1,3.65,2.5,title,body,ts=24,bs=21)
tx(sl,.9,6.05,11.5,.7,'524 是顶层生物学条目数；数值维度由每个队列的实际分子和读数构成。',23,bold=True)
sl=slide('PI3K：蛋白与磷酸化直接成为 Y 的组成维度','真实例子 4：患者 × 分子观测')
pic(sl,'pi3k',.6,1.8,12.1,4.75);tx(sl,.8,6.6,11.8,.35,'灰色＝缺失。各分子仅为显示转为 z 值；正式 Y 保留原值。上下两块分别为总蛋白和磷酸化。',14)
sl=slide('同属蛋白层，TCPA 与 CPTAC 的测量范围不同','平台差异如何进入 Y')
pic(sl,'dimensions',.7,1.85,11.9,4.7);tx(sl,.8,6.55,11.8,.45,'抗体靶标与质谱蛋白／位点各自保留身份；共同挂接通路，不直接拼成一个跨平台活动分数。',17)
sl=slide('本轮 Y 的样本范围：三个癌种，两个队列','实际数据结果')
pic(sl,'patients',.75,1.9,11.8,4.65);tx(sl,.8,6.5,11.8,.5,'合计 1,910 名开发集患者；保留原有参考／查询分折，外层测试集不在本轮展示范围。',18)
sl=slide('各组学的患者覆盖并不相同','实际数据结果：成员级测量')
pic(sl,'coverage',.6,1.85,12.1,4.65);tx(sl,.8,6.5,11.8,.6,'比例表示至少存在一项该层成员实测。CPTAC 的甲基化来源评分另行保留；本图统计成员级矩阵。',17)
sl=slide('State：197 个并列条目压缩为 48 个入口','State 组织方式')
card(sl,.8,2,5.65,2.3,'细胞组成：91 → 1','细胞类型与算法列在详情\n同一来源读数不重复存储',TEAL,29,22);card(sl,6.85,2,5.65,2.3,'SBS 特征：60 → 1','60 个特征作为子项保留\n每项拥有独立数值与缺失',ORANGE,29,22);tx(sl,1,5,11.3,.8,'2 个分组入口 + 46 个其他 State = 48 个 State',29,bold=True);tx(sl,1,6,11.3,.6,'合并展示与数据归属，保留每个子项的测量含义。',24)
sl=slide('细胞组成详情：三份 CD8 来源读数分别保留','真实例子 5：TCGA BRCA')
pic(sl,'cell',.65,1.95,12,4.65);tx(sl,.8,6.55,11.8,.4,'91 个来源身份共对应 96 个方法读数；图为当前保存值，具体计算方法与版本随列记录。',16)
sl=slide('突变特征谱详情：保留各个 SBS 的患者读数','真实例子 6：TCGA BRCA')
pic(sl,'sbs',.7,1.95,11.9,4.25);n=figure_records['SBS']['TCGA_BRCA_measured_patients'];tx(sl,.9,6.05,11.4,.8,f'BRCA 中 {n} / 872 名 TCGA 患者有 SBS 读数。\n当前 CPTAC BRCA 没有 SBS 实测标签，保留为缺失。',22)
sl=slide('Y 已固定；可用标签由实测和机制条件共同决定','实际覆盖结果')
pic(sl,'readiness',.7,1.85,11.9,4.75);tx(sl,.8,6.55,11.8,.45,'141 条事件相关主干缺少明确的患者事件条件；3 个 State 概念没有独立读数，均保留其缺失状态。',16)
sl=slide('M0 对照：相同 Y，比较不同 RNA 输入的信息','后续模型比较的固定设计')
for y,title,body,color in [(1.95,'lncRNA','全部合格 lncRNA',TEAL),(3.45,'mRNA','同批患者的编码 RNA',BLUE),(4.95,'lncRNA + mRNA','两类 RNA 联合',ORANGE)]:card(sl,.8,y,4.15,1.25,title,'',color,25);tx(sl,5.3,y+.25,2.3,.7,'线性 Lasso',25);arrow(sl,7.75,y+.4,.6)
card(sl,8.8,2.4,3.7,3.5,'同一套固定 Y','相同观测与缺失掩码\n相同患者分折\n临床／独立纯度作协变量',TEAL,25,20);tx(sl,.85,6.55,11.7,.4,'非 RNA 预测排除目标层输入；RNA 来源任务按重建解释。Pathway / State 权重为 1.0 / 0.5。',16)
sl=slide('最终交付：可追溯的多维 Y 与清晰的主目录','解决的结果')
card(sl,.8,1.95,3.65,2.8,'524 个主条目','476 条 Pathway\n48 个 State',TEAL,29,24);card(sl,4.85,1.95,3.65,2.8,'6 份患者矩阵','三癌种 × 两个队列\n数值、缺失与条件配套',BLUE,29,21);card(sl,8.9,1.95,3.65,2.8,'来源可追溯','原始成员与方法\n细胞／SBS 子项可展开',ORANGE,29,21)
tx(sl,.9,5.2,11.6,1.25,'通路身份负责组织信息，实测分子与来源读数构成 Y。\nlncRNA 作为后续模型输入，其通路关系通过同一 Y 上的模型比较评估。',25,bold=True)
for i,s in enumerate(prs.slides,1):s.notes_slide.notes_text_frame.text=f'重建报告第{i}页。数据依据 fixed_y_remake_20261114_r1/Y_bundle 与 figure_data。通路与State条目来自当前r11字典。'
out=OUT/'CancerLncAtlas_Y重建与通路State整合_20261114.pptx';prs.save(out)
(O/'FIGURE_PROVENANCE.json').write_text(json.dumps(figure_records,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
(O/'PPT_MANIFEST.json').write_text(json.dumps(dict(target_host='149',slides=len(prs.slides),path=str(out),figure_files=[str(p) for p in F.glob('*.png')]),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print('NEW_PPT_COMPLETE',len(prs.slides),flush=True)
