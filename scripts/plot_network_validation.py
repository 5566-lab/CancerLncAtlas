"""Export three separate RNA-arm figures and matched network control figures."""
import argparse,json,socket,sys,html
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from scripts.plot_m0_increment import classify,NAMES,LABELS,COLORS,ARMS,CANCERS
from scripts.prepare_graph_ridge import dump
from cla_engine.common import sha
LAYERS=list(NAMES)
SHORT={'PROTEIN_PATHWAY':'Protein','PHOSPHO_PTMSEA':'Phospho','METHYLATION_PATHWAY':'Methylation','RNA_PATHWAY':'RNA pathway','TF':'TF (RNA)','STATE_RNA':'State (RNA)','STATE_OTHER':'State (other)'}

def save(fig,out,name):
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(out/(name+'.'+ext),dpi=300,bbox_inches='tight')
    plt.close(fig)

def points(ax,frame,value,color,layers=LAYERS):
    rng=np.random.default_rng(11)
    for i,layer in enumerate(layers):
        vals=frame.loc[frame.display_layer.eq(layer),value].dropna().to_numpy()
        if len(vals):ax.scatter(i+rng.uniform(-.15,.15,len(vals)),vals,s=6,alpha=.18,color=color,rasterized=True);ax.plot([i-.25,i+.25],[np.median(vals)]*2,color='black',lw=2)
    ax.set_xticks(range(len(layers)),[SHORT[x] for x in layers],rotation=45,ha='right');ax.axhline(0,color='#777777',lw=.8,ls='--');ax.set_yscale('symlog',linthresh=.05);ax.spines[['top','right']].set_visible(False)

def plot(summary,training):
    assert socket.gethostname()=='149';out=Path(summary);training=Path(training)
    if not (out/'BOOTSTRAP_COMPLETE.json').exists():raise ValueError('BOOTSTRAP_NOT_COMPLETE')
    dump(out/'PLOT_PREFLIGHT.json',dict(target_host='149',summary=sha(out/'SUMMARY_COMPLETE.json'),bootstrap=sha(out/'BOOTSTRAP_COMPLETE.json'),code=sha(Path(__file__))))
    fonts={f.name for f in font_manager.fontManager.ttflist};plt.rcParams.update({'font.family':'Arial' if 'Arial' in fonts else 'DejaVu Sans','font.size':11,'axes.titlesize':13,'pdf.fonttype':42,'ps.fonttype':42})
    met=pd.read_parquet(out/'ALL_TARGET_METRICS.parquet');good=met[met.success&np.isfinite(met.r2)];main=good[good.variant.eq('module_ridge_union')&good.arm.isin(ARMS)]
    counts=main.groupby(['cancer_id','target_id','arm']).repeat.nunique();valid=counts[counts.eq(2)].index;avg=main.set_index(['cancer_id','target_id','arm']).loc[valid].reset_index().groupby(['cancer_id','target_id','arm','display_layer']).r2.mean().reset_index()
    limits=(min(-.1,float(avg.r2.min())*1.05),max(.1,float(avg.r2.max())*1.05))
    for i,arm in enumerate(ARMS,1):
        fig,axs=plt.subplots(1,3,figsize=(16,5.5),sharey=True)
        for ax,c in zip(axs,CANCERS):
            f=avg[avg.arm.eq(arm)&avg.cancer_id.eq(c)];points(ax,f,'r2',COLORS[arm]);ax.set_ylim(limits);ax.set_title(c+' | '+str(len(f))+' targets')
        axs[0].set_ylabel('Held-out R2, mean of two CV repeats');fig.suptitle(LABELS[arm]+' arm: network module ridge\nSame eligible patient omics and unpenalized clinical covariates',fontsize=14);save(fig,out,f'{i:02d}_{arm}_performance')
    three=pd.read_parquet(out/'THREE_ARM_COMPARISONS.parquet');three=three[three.variant.eq('module_ridge_union')];ct=three.groupby(['cancer_id','target_id']).repeat.nunique();keep=ct[ct.eq(2)].index;three=three.set_index(['cancer_id','target_id']).loc[keep].reset_index().groupby(['cancer_id','target_id','display_layer']).combined_minus_mrna.mean().reset_index()
    fig,axs=plt.subplots(1,3,figsize=(16,5.5),sharey=True)
    for ax,c in zip(axs,CANCERS):points(ax,three[three.cancer_id.eq(c)],'combined_minus_mrna',COLORS['combined']);ax.set_title(c)
    axs[0].set_ylabel('Paired Delta R2: combined minus mRNA');fig.suptitle('Increment from adding lncRNA to the mRNA arm');save(fig,out,'04_combined_vs_mrna')
    paired=pd.read_parquet(out/'PAIRED_MODEL_DELTAS.parquet');paired=paired.groupby(['cancer_id','target_id','arm','display_layer','method','baseline']).delta_r2.mean().reset_index()
    fig,axs=plt.subplots(2,3,figsize=(15,9),sharey='row');rng=np.random.default_rng(42)
    for row,layer in enumerate(['PROTEIN_PATHWAY','PHOSPHO_PTMSEA']):
        for ax,arm in zip(axs[row],ARMS):
            for x,method in enumerate(['module_ridge_ppi','module_ridge_rpi','module_ridge_union']):
                vals=paired.loc[paired.arm.eq(arm)&paired.display_layer.eq(layer)&paired.method.eq(method)&paired.baseline.eq('ridge_adaptive_omics'),'delta_r2'].to_numpy();ax.scatter(x+rng.uniform(-.16,.16,len(vals)),vals,s=6,alpha=.2,color=COLORS[arm],rasterized=True)
                if len(vals):ax.plot([x-.25,x+.25],[np.median(vals)]*2,color='black',lw=2)
            ax.set_xticks(range(3),['PPI','RPI','PPI + RPI']);ax.axhline(0,color='gray',ls='--');ax.set_yscale('symlog',linthresh=.02);ax.set_title(SHORT[layer]+' | '+LABELS[arm])
        axs[row,0].set_ylabel('Delta R2 versus adaptive omics ridge')
    fig.suptitle('Does biological network membership add prediction?');save(fig,out,'05_network_ablation')
    boot=pd.read_csv(out/'PATIENT_BOOTSTRAP.tsv',sep='\t');pooled=boot[boot.cancer_id.eq('EQUAL_CANCER_MEAN')];baselines=['ridge_equal','ridge_adaptive_omics','rna_laplacian_union','mean_five_null_models'];bnames=['Equal omics ridge','Adaptive omics ridge','RNA Laplacian','Five random networks']
    fig,axs=plt.subplots(2,3,figsize=(16,8),sharex=False)
    for row,layer in enumerate(['PROTEIN_PATHWAY','PHOSPHO_PTMSEA']):
        for ax,arm in zip(axs[row],ARMS):
            for j,b in enumerate(baselines):
                f=pooled[pooled.layer.eq(layer)&pooled.arm.eq(arm)&pooled.method.eq('module_ridge_union')&pooled.baseline.eq(b)]
                if len(f):v=f.iloc[0];ax.plot([v.ci_low,v.ci_high],[j,j],color=COLORS[arm],lw=2);ax.scatter(v.family_balanced_mean_delta_r2,j,color=COLORS[arm],s=35)
            ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(4),bnames);ax.invert_yaxis();ax.set_title(SHORT[layer]+' | '+LABELS[arm]);ax.set_xlabel('Mean paired Delta R2 (95% patient bootstrap)')
    fig.suptitle('Network module ridge versus four controls\nEqual cancer and score-family weighting; uncertainty conditional on fitted CV models',fontsize=13);save(fig,out,'06_primary_patient_bootstrap')
    real=['rna_only','omics_only','ridge_equal','ridge_adaptive_omics','rna_laplacian_ppi','rna_laplacian_rpi','rna_laplacian_union','module_ridge_ppi','module_ridge_rpi','module_ridge_union']
    fig,axs=plt.subplots(1,2,figsize=(13,8),sharey=True)
    for ax,layer in zip(axs,['PROTEIN_PATHWAY','PHOSPHO_PTMSEA']):
        cells=np.full((len(real),3),np.nan)
        for j,method in enumerate(real):
            for k,arm in enumerate(ARMS):
                f=good[good.variant.eq(method)&good.display_layer.eq(layer)&good.arm.eq('shared' if method=='omics_only' else arm)];cells[j,k]=f.groupby(['cancer_id','target_id']).r2.mean().median()
        im=ax.imshow(cells,aspect='auto',cmap='RdBu',vmin=-.2,vmax=.5)
        for j in range(len(real)):
            for k in range(3):ax.text(k,j,f'{cells[j,k]:.3f}',ha='center',va='center',fontsize=10)
        ax.set_yticks(range(len(real)),[s.replace('_',' ') for s in real]);ax.set_xticks(range(3),[LABELS[a] for a in ARMS],rotation=25,ha='right');ax.set_title(SHORT[layer])
    fig.suptitle('Median held-out R2 by model and RNA arm\nTwo-repeat mean for each target; full null-model tables accompany this figure');save(fig,out,'07_model_performance_table')
    cover=pd.read_csv(out/'NETWORK_FEATURE_COVERAGE.tsv',sep='\t');fig,axs=plt.subplots(1,3,figsize=(16,5),sharey=True);labs=[('RNA','lncrna','lncRNA'),('RNA','mrna','mRNA'),('CNV','shared','CNV'),('MUTATION','shared','Mutation'),('PROTEIN','shared','Protein'),('PHOSPHO','shared','Phospho')]
    for ax,c in zip(axs,CANCERS):
        for j,mode in enumerate(['ppi','rpi','union']):
            values=[cover.loc[cover.cancer_id.eq(c)&cover.block.eq(b)&cover.arm.eq(a)&cover['mode'].eq(mode),'fraction'].iloc[0]*100 for b,a,l in labs];ax.bar(np.arange(len(labs))+(j-1)*.24,values,width=.23,label=mode.upper())
        ax.set_xticks(range(len(labs)),[l for b,a,l in labs],rotation=45,ha='right');ax.set_ylim(0,100);ax.set_title(c)
    axs[0].set_ylabel('Quality-eligible features attached to graph (%)');axs[-1].legend();fig.suptitle('Measured feature coverage of the network prior');save(fig,out,'08_network_feature_coverage')
    hyp=pd.read_csv(out/'HYPERPARAMETERS.tsv',sep='\t');hyp=hyp[hyp.phase.ne('final')];fig,axs=plt.subplots(1,3,figsize=(15,6),sharey=True)
    methods=['ridge_adaptive_omics','module_ridge_ppi','module_ridge_rpi','module_ridge_union','rna_laplacian_ppi','rna_laplacian_rpi','rna_laplacian_union']
    for ax,arm in zip(axs,ARMS):
        for i,method in enumerate(methods):
            f=hyp[hyp.arm.eq(arm)&hyp.variant.eq(method)];v=f.loc[f.parameter.eq(0),'fits'].sum()/f.fits.sum()*100;ax.barh(i,v,color=COLORS[arm]);ax.text(v+.6,i,f'{v:.1f}%',va='center',fontsize=9)
        ax.set_yticks(range(len(methods)),[m.replace('_',' ') for m in methods]);ax.invert_yaxis();ax.set_xlim(0,110);ax.set_title(LABELS[arm]);ax.set_xlabel('Fits choosing zero adaptive/graph strength (%)')
    fig.suptitle('Training-only tuning can turn off the proposed prior');save(fig,out,'09_zero_strength_selection')
    cov=pd.read_csv(out/'COVERAGE_SUMMARY.tsv',sep='\t');cov=cov[cov.variant.eq('module_ridge_union')&cov.arm.eq('combined')].groupby(['cancer_id','display_layer'])[['attempted','successful']].mean().reset_index();fig,axs=plt.subplots(1,3,figsize=(16,5),sharey=True)
    for ax,c in zip(axs,CANCERS):
        f=cov[cov.cancer_id.eq(c)].set_index('display_layer').reindex(LAYERS);ax.bar(np.arange(len(LAYERS)),100*f.successful/f.attempted,color=COLORS['combined']);ax.set_xticks(range(len(LAYERS)),[SHORT[l] for l in LAYERS],rotation=45,ha='right');ax.set_ylim(0,100);ax.set_title(c)
    axs[0].set_ylabel('Targets completing all five folds + final fit (%)');fig.suptitle('Training completion by target layer; failed targets remain in the audit');save(fig,out,'10_training_coverage')
    tables=['THREE_ARM_SUMMARY.tsv','NETWORK_GAIN_SUMMARY.tsv','PATIENT_BOOTSTRAP.tsv','MODEL_SUMMARY.tsv','COVERAGE_SUMMARY.tsv','PATIENT_COVERAGE_SUMMARY.tsv','NETWORK_FEATURE_COVERAGE.tsv','NETWORK_PARTITIONS.tsv','PHASE_AUDIT.tsv','HYPERPARAMETERS.tsv']
    h="<!doctype html><meta charset='utf-8'><title>Network validation</title><style>body{font:16px Arial;max-width:1250px;margin:32px auto;padding:20px;color:#213547}p{line-height:1.6}img{width:100%;border:1px solid #ddd}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}a{color:#176294}</style><h1>PPI/RPI network validation: three RNA arms</h1>"
    h+='<p>CPTAC BRCA (121 patients), LUAD (110), COAD (106). Two repetitions of five-fold outer CV; repeat 0 preserves the original splits. Every outer fit and final fit uses two repetitions of three-fold inner tuning. Targets and upstream native score definitions are unchanged. All clinical/purity coefficients remain unpenalized.</p>'
    h+='<p>Protein outcomes exclude all protein predictors; phospho outcomes exclude all phospho predictors. Genomic-derived outcomes keep their documented exclusions. RNA-derived outcomes retain lncRNA; mRNA-containing models can reconstruct the score definition. Cohort preprocessing of the original scores is retained: this is a retrospective benchmark, not independent external validation or proof of pathway activation.</p>'
    h+='<p>Legacy model: a Laplacian smooths RNA coefficients only. Improved model: typed PPI/RPI partitions group measured RNA, CNV, mutation, protein and phosphosite features. Coefficients keep independent signs; group shrinkage weights are learned only within each training split. Zero adaptation recovers equal-modality ridge. Unmapped features remain in modality-specific background groups. Ridge coefficients are dense, not a selected gene set.</p>'
    h+='<p>Five null networks preserve unweighted node degrees and edge-weight distributions; weighted node strengths and community sizes are not fixed. Nulls are diagnostic controls, not a five-permutation significance test. Patient bootstrap intervals condition on the fitted CV models, pair both outer repeats, and average targets within families and cancers equally.</p>'
    h+='<h2>Three-arm summary</h2>'+pd.read_csv(out/'THREE_ARM_SUMMARY.tsv',sep='\t').query("variant == 'module_ridge_union'").to_html(index=False,float_format=lambda v:f'{v:.4f}')
    h+='<h2>Primary network controls</h2>'+pooled[pooled.arm.isin(ARMS)].to_html(index=False,float_format=lambda v:f'{v:.5f}')
    h+='<h2>Download tables</h2><ul>'+''.join('<li><a href="'+n+'">'+n+'</a></li>' for n in tables)+'</ul>'
    for f in sorted(out.glob('*.png')):h+='<h2>'+html.escape(f.stem.replace('_',' '))+'</h2><p><a href="'+f.with_suffix('.pdf').name+'">PDF</a></p><img src="'+f.name+'">'
    (out/'OPEN_REPORT.html').write_text(h,encoding='utf-8');dump(out/'PLOT_COMPLETE.json',dict(target_host='149',status='COMPLETE',figures=len(list(out.glob('*.png'))),files={p.name:sha(p) for p in out.iterdir() if p.suffix in ['.png','.pdf','.html']}));print('PLOT_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True);p.add_argument('--training',required=True);a=p.parse_args();plot(a.summary,a.training)
