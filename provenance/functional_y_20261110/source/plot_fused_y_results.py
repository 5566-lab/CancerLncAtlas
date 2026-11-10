from pathlib import Path
import sys,socket,json,hashlib,argparse
R=Path(__file__).resolve().parent;F=R/'functional_relation_y_fused_r1';D=F/'dictionary_data';O=F/'reports';O.mkdir(exist_ok=True);assert socket.gethostname()=='149'; (O/'plot_fused_y_results.py.PREFLIGHT.json').write_text(json.dumps({"target_host":"149","operation":'plot_fused_y_results.py'}))
sys.path[:0]=[str(R/'python_deps'),'/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps']
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'pdf.fonttype':42})
NAVY='#203C54';TEAL='#1E8093';ORANGE='#CA723D';BLUE='#5598C5';PALE='#ECF3F5';GREY='#667684';GOLD='#BC9B45';files={}
def save(fig,name):
 for ext in ['png','pdf']:
  p=O/(name+'.'+ext);fig.savefig(p,dpi=300,bbox_inches='tight');files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
 plt.close(fig)
def canvas():
 fig,ax=plt.subplots(figsize=(12.8,5.7));ax.set_xlim(0,13);ax.set_ylim(0,6);ax.axis('off');return fig,ax
def box(ax,x,y,w,h,title,body='',color=TEAL,fs=15):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',facecolor=PALE,edgecolor=color,lw=1.4));ax.text(x+w/2,y+h*.68 if body else y+h/2,title,ha='center',va='center',fontsize=fs,color=color,fontweight='bold');
 if body:ax.text(x+w/2,y+h*.28,body,ha='center',va='center',fontsize=11.5,color=NAVY,linespacing=1.6)
def arr(ax,x1,y1,x2,y2,color=GREY):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=17,lw=1.5,color=color,connectionstyle='arc3,rad=0'))
def structures():
 fig,ax=canvas();box(ax,.2,3.6,3.2,1.5,'生物学身份','同义、功能范围、反应核心\n父子与分支关系');box(ax,4.4,3.6,3.5,1.5,'患者读数 S','RNA / 蛋白 / 磷酸化 / State\n保留平台、算法及测量定义');box(ax,9,3.6,3.6,1.5,'队列内关联','lncRNA 与各项读数的关联\n临床、纯度等协变量调整');arr(ax,3.45,4.35,4.35,4.35);arr(ax,7.95,4.35,8.95,4.35);arr(ax,10.8,3.5,10.8,2.6);box(ax,7.2,.75,5.4,1.8,'融合关系 Y','合并 p → BH q → 支持等级\n方向、来源、缺失与冲突另存',color=ORANGE);arr(ax,7.1,1.65,5.7,1.65);box(ax,.4,.75,5.2,1.8,'最终研究单位','癌种 × lncRNA × 精确通路 / State',fs=17);ax.text(6.5,.18,'一条通路一个统一身份；多来源读数共同提供关系证据。',ha='center',color=GREY);save(fig,'01_从患者读数到关系Y')
 fig,ax=canvas();box(ax,.2,4.55,3.3,1.05,'Reactome','R-HSA-195721');box(ax,.2,2.95,3.3,1.05,'WikiPathways','WP428');box(ax,.2,1.35,3.3,1.05,'PTMsigDB','PATH-WP_Wnt_signaling',color=ORANGE);box(ax,4.7,2.75,3.6,1.6,'WNT signaling','CLA-P-366F48557175A084',fs=17)
 for y in [5.05,3.45,1.85]:arr(ax,3.55,y,4.65,3.55)
 box(ax,9.5,4.7,3.2,.85,'TCF 依赖的 WNT 信号',fs=14);box(ax,9.5,3.25,3.2,.85,'β-catenin 降解',fs=14);box(ax,9.5,1.8,3.2,.85,'β-catenin 非依赖通路',fs=13)
 for y in [5.1,3.65,2.2]:arr(ax,8.35,3.55,9.45,y)
 ax.text(6.45,2.2,'Reactome / WP：等价定义\nPTMsigDB：位点读数归属',ha='center',color=GREY,fontsize=12);ax.text(11.1,1.1,'子通路保留独立 ID',ha='center',color=GREY,fontsize=12);ax.text(6.5,.35,'归并的是 WNT 总通路身份；层级不同的子通路仍可单独研究。',ha='center',color=NAVY);save(fig,'02_WNT来源归并与分支')
 fig,ax=canvas()
 for x,title,body in [(.3,'Hallmark','Hypoxia gene set'),(4.5,'Buffa','Hypoxia signature'),(8.7,'PROGENY','Hypoxia response model')]:box(ax,x,4.1,3.8,1.2,title,body);arr(ax,x+1.9,4,6.4,2.8)
 box(ax,3.3,1.2,6.2,1.5,'缺氧转录响应','Hypoxia transcriptional response\nCLA-P-3423B99760CD551B',fs=19);ax.text(6.4,.55,'保留三个 RNA 读数；融合时利用它们的相关结构校正重复证据。',ha='center',fontsize=13,color=GREY);save(fig,'03_缺氧多方法读数合并')
 fig,ax=canvas();core=[('TAT','酪氨酸 → HPP'),('HPD','HPP → 尿黑酸'),('HGD','尿黑酸 → 4-MAA'),('GSTZ1','4-MAA → 4-FAA'),('FAH','4-FAA → 延胡索酸\n+ 乙酰乙酸')]
 for i,(enzyme,body) in enumerate(core):
  box(ax,.2+i*2.55,3.5,2.3,1.35,enzyme,body,fs=17)
  if i<4:arr(ax,2.55+i*2.55,4.18,2.7+i*2.55,4.18)
 ax.text(6.5,5.45,'共同的 5 步酪氨酸分解核心',ha='center',fontsize=19,color=TEAL,fontweight='bold');box(ax,1.6,.8,4.5,1.45,'KEGG N00708','保留核心定义');box(ax,7,.8,4.5,1.45,'Reactome R-HSA-8963684','包含共同核心 + 2 项附加反应',color=ORANGE);ax.text(6.5,2.7,'范围扩展：反向转氨反应、额外 HPP 反应',ha='center',fontsize=14,color=ORANGE);ax.text(6.5,.2,'用共同核心与扩展关系连接两个身份；不把不同反应范围强行合并。',ha='center',fontsize=12,color=GREY);save(fig,'04_酪氨酸共同核心与扩展')
 fig,ax=canvas();box(ax,.7,3.5,3,1.4,'HRD–LOH','杂合性缺失相关读数');box(ax,5,3.5,3,1.4,'HRD–TAI','端粒等位基因失衡');box(ax,9.3,3.5,3,1.4,'HRD–LST','大尺度状态转变')
 for x in [2.2,6.5,10.8]:arr(ax,x,3.4,6.5,2.45)
 ax.text(4.25,4.2,'+',ha='center',fontsize=29,color=GREY);ax.text(8.65,4.2,'+',ha='center',fontsize=29,color=GREY);box(ax,3.8,.75,5.4,1.65,'HRD total','LOH + TAI + LST\nCLA-S-46263ADB8FE4F583',color=ORANGE,fs=20);ax.text(6.5,.15,'这里存在明确的评分组成公式；分项身份与组成关系同时保留。',ha='center',fontsize=12,color=GREY);save(fig,'05_State评分组成_HRD')
 fig,ax=canvas();box(ax,.3,3.65,3.45,1.5,'各读数关联','调整后 rho、p、有效样本数');box(ax,4.7,3.65,3.65,1.5,'依赖校正的合并检验',r'$T=-2\sum_j w_j\log(p_j)$');box(ax,9.3,3.65,3.4,1.5,'多重检验校正','同癌种 / 分折 / 主任务内 BH')
 arr(ax,3.8,4.4,4.65,4.4);arr(ax,8.4,4.4,9.25,4.4);arr(ax,11,3.55,11,2.55);box(ax,7.6,.7,5.1,1.75,'关系支持等级','强：q ≤ 0.05 且 max|rho| ≥ 0.20\n弱：q ≤ 0.10 且 max|rho| ≥ 0.15',color=ORANGE);box(ax,.3,.7,5.6,1.75,'依赖结构来自实测读数','保留读数间相关与缺失模式\n重复读数共用权重；方向冲突另存');arr(ax,3.1,2.5,6.5,3.55);ax.text(6.5,.15,'没有达到支持门槛的关系保留为未标注；缺测关系不赋零分。',ha='center',fontsize=12,color=GREY);save(fig,'06_关系Y融合公式')
 cov=pd.read_csv(R/'shared_relation_demo_r1/COVARIATE_COVERAGE.tsv',sep='\t');fig,ax=plt.subplots(figsize=(12.8,5.7));pos=np.arange(len(cov));ax.barh(pos,cov.RNA_patients,color='#CFDEE5',height=.66,label='队列内 RNA 患者');ax.barh(pos,cov.raw_all_required_observed,color=TEAL,height=.43,label='共同基础协变量完整');ax.set_yticks(pos,cov.dataset+' · '+cov.cancer_id);ax.invert_yaxis();ax.set_xlabel('患者人数');ax.legend(frameon=False,loc='lower right');ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
 for i,r in cov.iterrows():ax.text(r.RNA_patients+10,i,f'{r.raw_all_required_observed}/{r.RNA_patients}',va='center',fontsize=13)
 ax.set_xlim(0,max(cov.RNA_patients)*1.2);fig.tight_layout();save(fig,'07_共同协变量患者覆盖')
 fig,ax=plt.subplots(figsize=(12.8,4.8));n=np.array([588,16857,16743]);c=['BRCA','LUAD','COAD'];ax.barh(c,[16889]*3,color='#DDE6EB',height=.55,label='完整 lncRNA 轴');ax.barh(c,n,color=TEAL,height=.55,label='有实测局部 CNV 的 lncRNA');ax.invert_yaxis();ax.set_xlim(0,20500);ax.set_xlabel('lncRNA 数');ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.22),ncol=2);ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
 for y,v in enumerate(n):ax.text(v+250,y,f'{v:,} / 16,889',va='center',fontsize=14)
 fig.tight_layout();save(fig,'08_CPTAC局部CNV覆盖')
 layers=pd.read_csv(D/'LAYER_COVERAGE.tsv',sep='\t');sel=layers.loc[layers.assay_group.isin(['RNA','PROTEIN','PHOSPHO','STATE','METHYLATION'])];names=['RNA','STATE','PROTEIN','PHOSPHO','METHYLATION'];labels=['RNA 通路','State','蛋白通路','磷酸化签名','启动子甲基化'];fig,axs=plt.subplots(1,3,figsize=(13,5.7),sharey=True)
 for ax,c in zip(axs,['BRCA','LUAD','COAD']):
  for dy,ds,color in [(-.17,'TCGA',TEAL),(.17,'CPTAC',ORANGE)]:
   vv=sel.loc[sel.cancer_id.eq(c)&sel.dataset.eq(ds)].set_index('assay_group').reindex(names);y=np.arange(5)+dy;vals=vv.score_coverage_fraction.to_numpy()*100;ax.barh(y,np.nan_to_num(vals),height=.3,color=color,label='TCGA / TCPA' if ds=='TCGA' else 'CPTAC')
   for i,x in enumerate(vals):ax.text((0 if not np.isfinite(x) else x)+2,y[i],'—' if not np.isfinite(x) else f'{x:.0f}%',va='center',fontsize=10)
  ax.set_xlim(0,125);ax.set_xticks([0,50,100]);ax.set_title(c,fontsize=17);ax.set_xlabel('患者覆盖率（%）');ax.set_yticks(range(5),labels);ax.invert_yaxis();ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
 axs[1].legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,1.25),ncol=2);fig.tight_layout(w_pad=1.7);save(fig,'09_各层患者评分覆盖')
def results():
 summary=pd.read_csv(D/'Y_BY_CANCER.tsv',sep='\t').set_index('cancer_id').loc[['BRCA','LUAD','COAD']];counts=json.loads((D/'IDENTITY_COUNTS.json').read_text());fig,axs=plt.subplots(1,2,figsize=(12.8,5.4));roles=[2592,861,3];axs[0].barh(['Pathway 专属','State 专属','共享身份'],roles,color=[TEAL,ORANGE,GOLD]);axs[0].invert_yaxis();axs[0].set_xlim(0,3100)
 for y,v in enumerate(roles):axs[0].text(v+50,y,f'{v:,}',va='center',fontsize=16)
 axs[0].set_title('3,456 个去重后的生物学身份',fontsize=17);axs[0].set_xlabel('身份数');x=np.arange(3);axs[1].bar(x-.18,summary.observed_targets,.36,color=BLUE,label='可估计关联');axs[1].bar(x+.18,summary.supported_targets,.36,color=TEAL,label='有支持关系');axs[1].set_xticks(x,summary.index);axs[1].set_ylim(0,3456);axs[1].legend(frameon=False);axs[1].set_title('每个癌种实际获得关系的目标',fontsize=17);axs[1].set_ylabel('通路 / State 数')
 for i,(obs,sup) in enumerate(zip(summary.observed_targets,summary.supported_targets)):axs[1].text(i-.18,obs+55,str(obs),ha='center',fontsize=10);axs[1].text(i+.18,sup+55,str(sup),ha='center',fontsize=10)
 fig.tight_layout(w_pad=3);save(fig,'10_身份数量与实际监督范围')
 fig,axs=plt.subplots(1,3,figsize=(12.8,5.2))
 for ax,(c,row) in zip(axs,summary.iterrows()):
  vals=[row.strong_relations,row.weak_relations];bars=ax.bar(['强支持','弱支持'],np.array(vals)/10000,color=[TEAL,BLUE],width=.55);ax.set_title(c,fontsize=18);ax.set_ylabel('关系数（万）');ax.set_ylim(0,max(vals)/10000*1.28);ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
  for b,n in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(vals)/10000*.04,f'{int(n):,}',ha='center',fontsize=12)
  ax.text(.5,-.18,f'标签患者：TCGA {int(row.TCGA_label_patients)} / CPTAC {int(row.CPTAC_label_patients)}',transform=ax.transAxes,ha='center',fontsize=10,color=GREY)
 fig.tight_layout(w_pad=2);save(fig,'11_三癌种关系Y结果')
 fig,ax=plt.subplots(figsize=(12.8,5.6));start=np.zeros(3);parts=[('supported_with_RNA_only_individual_support','仅 RNA 读数单独达标',BLUE),('supported_with_nonRNA_individual_support','有非 RNA 读数单独达标',ORANGE),('supported_without_individual_readout_support','仅融合后达到支持门槛','#B7C7D1')]
 for col,label,color in parts:
  vals=summary[col].to_numpy()/summary.supported_relations.to_numpy()*100;ax.barh(summary.index,vals,left=start,color=color,height=.55,label=label)
  for i,v in enumerate(vals):
   if v>=4:ax.text(start[i]+v/2,i,f'{v:.1f}%',ha='center',va='center',fontsize=15,color='white' if color!= '#B7C7D1' else NAVY)
  start+=vals
 ax.set_xlim(0,124);ax.set_xticks([0,20,40,60,80,100]);ax.invert_yaxis()
 for i,v in enumerate(summary.supported_with_nonRNA_individual_support.to_numpy()/summary.supported_relations.to_numpy()*100):ax.text(101.5,i,f'非 RNA {v:.1f}%',va='center',fontsize=12,color=ORANGE)
 ax.set_xlabel('获得融合支持的关系中，各证据类型所占比例');ax.legend(frameon=False,loc='upper left',bbox_to_anchor=(0,1.28),ncol=2);fig.tight_layout();save(fig,'12_支持关系的证据来源')
 fig,ax=plt.subplots(figsize=(12.8,5));both=summary.joint_cohort_relations.to_numpy()/summary.observed_relations.to_numpy()*100;ax.barh(summary.index,[100]*3,color='#DDE6EB',height=.55,label='仅一个队列可估计');ax.barh(summary.index,both,color=TEAL,height=.55,label='两个队列均可估计');ax.invert_yaxis();ax.set_xlim(0,115);ax.set_xlabel('可估计关系中，两队列均有实测关联的比例（%）');ax.legend(frameon=False,loc='lower right')
 for i,v in enumerate(both):ax.text(v+2,i,f'{v:.1f}%',va='center',fontsize=16)
 fig.tight_layout();save(fig,'13_两队列关系覆盖')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--structures-only',action='store_true');args=ap.parse_args();structures()
 if not args.structures_only:results()
 p=O/'SCIENTIFIC_FIGURES.json';p.write_text(json.dumps({'target_host':'149','files':files},ensure_ascii=False,indent=2));print('FIGURES',len(files),flush=True)
