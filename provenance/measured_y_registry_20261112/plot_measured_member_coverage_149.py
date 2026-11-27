import json,socket,sys
from pathlib import Path
assert socket.gethostname().split('.')[0]=='149'
ROOT=Path('/dsk2/user/dengsc/cla_functional_y_20261109_r1')
DATA=ROOT/'kegg_cancer_redefinition_20261111_r1/measured_y_definition_r1/member_measurements_r1'
sys.path[:0]=[str(ROOT/'source_integration/repo'),'/dsk2/user/dengsc/cla_m0_joint_20261105_r1/python_deps']
from cla_engine.common import sha
(DATA/'FIGURE_PREFLIGHT.json').write_text(json.dumps({'target_host':'149','operation':'plot measured molecular-member coverage','lncrna_association':False,'training':False}))
r=json.loads((DATA/'COMPLETE.json').read_text())
for name,digest in r['block_receipts'].items():assert sha(DATA/name)==digest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
font_file='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(font_file)
font_manager.fontManager.addfont('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
font_name=font_manager.FontProperties(fname=font_file).get_name()
plt.rcParams.update({'font.family':font_name,'font.size':12,'pdf.fonttype':42,'figure.facecolor':'white','axes.unicode_minus':False})
fig,axes=plt.subplots(1,2,figsize=(12,6.2))
palette={'TCGA':'#2B6F9C','CPTAC':'#D88A39'}
for ax,layer,title in zip(axes,['PROTEIN','PHOSPHO'],['蛋白丰度成员','磷酸化成员']):
    for j,cohort in enumerate(('TCGA','CPTAC')):
        values=[]
        for cancer in ('BRCA','LUAD','COAD'):
            row=next(x for x in r['blocks'] if x['case']==cohort+'_'+cancer and x['layer']==layer)
            values.append(row['pathways_with_member_measurement'])
        pos=np.arange(3)+(j-.5)*.32
        ax.barh(pos,values,height=.27,color=palette[cohort],label='TCPA 抗体' if cohort=='TCGA' else 'CPTAC 质谱')
        for y,value in zip(pos,values):ax.text(value+5,y,str(value),va='center',fontsize=13,color=palette[cohort],fontweight='bold')
    ax.set_yticks(np.arange(3),['BRCA','LUAD','COAD']);ax.invert_yaxis()
    ax.set_xlim(0,525);ax.set_xticks([0,100,200,300,400,477]);ax.set_xlabel('有明确成员实测值的主干通路数')
    ax.axvline(477,color='#687686',linestyle='--',linewidth=1)
    ax.set_title(title,fontsize=17,pad=12);ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False,bbox_to_anchor=(.5,.9))
fig.suptitle('477 条主干通路的蛋白和磷酸化成员覆盖',fontsize=21,fontweight='bold',y=.98)
fig.subplots_adjust(top=.78,bottom=.25,left=.08,right=.97,wspace=.27)
fig.text(.08,.12,'计数条件：至少一个原始 KEGG 成员有明确归属的实测值；仅部分抗体识别组重叠不计入。',fontsize=11,color='#425269')
fig.text(.08,.065,'范围：三癌种开发患者。成员覆盖不等于整条通路测量完整，也不等于通路激活。',fontsize=11,color='#425269')
for ext in ('png','pdf'):fig.savefig(DATA/('member_protein_phospho_coverage.'+ext),dpi=300,bbox_inches='tight')
plt.close(fig)
(DATA/'FIGURE_RECEIPT.json').write_text(json.dumps({'target_host':'149','source_receipt_sha256':sha(DATA/'COMPLETE.json'),'files':{ext:sha(DATA/('member_protein_phospho_coverage.'+ext)) for ext in ('png','pdf')}},indent=2)+'\n')
print('FIGURES_COMPLETE',flush=True)
