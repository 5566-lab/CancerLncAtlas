from pathlib import Path
import sys,socket,json,hashlib
R=Path(__file__).resolve().parent;sys.path[:0]=[str(R/'python_deps'),'/dsk2/user/dengsc/cla_m0_linear_20261104_r1/python_deps'];assert socket.gethostname()=='149'
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=R/'functional_relation_y_fused_r1/reports';O.mkdir(exist_ok=True)
(O/'LAUNCH_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','task':'Plot conditional-null validation'}))
plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,'pdf.fonttype':42})
p=R/'shared_relation_demo_r1/fusion_method_r1/conditional_missingness_validation/VALIDATION.json';d=json.loads(p.read_text());assert d['status']=='PASS';rows=d['records'];cohorts=[(s,c) for s in ['TCGA','CPTAC'] for c in ['BRCA','LUAD','COAD']]
fig,axs=plt.subplots(1,2,figsize=(12.8,6.3),sharey=True)
for ax,key,nom,xmax,title in zip(axs,['reject05','reject01'],[5,1],[6,1.25],['检验阈值 P ≤ 0.05','检验阈值 P ≤ 0.01']):
 for i,(s,c) in enumerate(cohorts):
  rr=[x for x in rows if x['dataset']==s and x['cancer']==c]
  for dy,x in zip(np.linspace(-.23,.23,len(rr)),rr):
   kin=x['target'].startswith('Phosphosite');ax.scatter(x[key]*100,5-i+dy,s=68,marker='D' if kin else 'o',color='#C66A38' if kin else '#247C96',edgecolors='white',linewidth=.6,zorder=3)
 ax.axvline(nom,color='#555555',ls='--',lw=1.5,label='名义假阳性率');ax.set_xlim(0,xmax);ax.set_ylim(-.55,5.55);ax.set_title(title,fontsize=15,pad=14);ax.set_xlabel('置换中被判为显著的比例（%）');ax.grid(axis='x',alpha=.16);ax.set_yticks(range(6));ax.set_yticklabels([f'{s} · {c}' for s,c in cohorts][::-1]);ax.tick_params(axis='y',length=0)
axs[1].scatter([],[],s=68,color='#247C96',label='通路示例');axs[1].scatter([],[],s=68,marker='D',color='#C66A38',label='实测磷酸化激酶程序');axs[1].legend(frameon=False,loc='lower right',fontsize=10.5)
fig.suptitle('融合检验的假阳性率与预设阈值相符',fontsize=19,y=.98);fig.text(.5,.91,f'{len(rows)} 个可检验组合 · {d["total_null_trials"]:,} 次置换 · BRCA / LUAD / COAD',ha='center',fontsize=12,color='#444444')
fig.text(.06,.025,'每个点代表一个队列内的通路或激酶程序；使用训练标签分折，保留实际缺失模式与协变量调整。',fontsize=10.5,color='#555555');fig.subplots_adjust(left=.15,right=.98,top=.81,bottom=.17,wspace=.18)
files={}
for ext in ['png','pdf']:
 out=O/f'融合检验_真实数据置换验证.{ext}';fig.savefig(out,dpi=300,bbox_inches='tight');files[out.name]=hashlib.sha256(out.read_bytes()).hexdigest()
plt.close(fig);(O/'FIGURE_RECEIPT.json').write_text(json.dumps({'target_host':'149','files':files,'null_validation_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2,ensure_ascii=False));print(json.dumps({'directory':str(O),'files':list(files)},ensure_ascii=False))
