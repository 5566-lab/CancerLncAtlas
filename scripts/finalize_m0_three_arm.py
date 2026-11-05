"""Verify paired three-arm results and export figures without patient data to Git."""
import argparse,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cla_engine.common import sha
from run_m0_linear import write_json
from cc_hhgt.m0_three_arm import ARMS,ARM_LABELS


def finalize(root,previous=None):
    if socket.gethostname()!='149':raise RuntimeError('Host 149 required')
    root=Path(root);frames=[];layers=[];receipts=[];coverage=[]
    for cancer in ['BRCA','LUAD','COAD']:
        d=root/cancer;rec=json.loads((d/'COMPLETE.json').read_text())
        if rec['status']!='COMPLETE':raise ValueError('Incomplete comparison '+cancer)
        for name,h in rec['files'].items():
            if sha(d/name)!=h:raise ValueError('Result drift: '+cancer+'/'+name)
        # Phase receipts also cover the fitted models and held-out predictions.
        for phase in [f'outer{k}' for k in range(5)]+['final']:
            for arm in ARMS:
                p=d/phase/arm;cr=json.loads((p/'COMPLETE.json').read_text())
                for name,h in cr['files'].items():
                    if sha(p/name)!=h:raise ValueError('Phase artifact drift')
        receipts.append(rec['summary'])
        f=pd.read_csv(d/'TARGET_COMPARISON.tsv',sep='\t');f['cancer_id']=cancer;frames.append(f)
        l=pd.read_csv(d/'LAYER_SUMMARY.tsv',sep='\t');l['cancer_id']=cancer;layers.append(l)
        t=pd.read_csv(d/'TARGETS.tsv',sep='\t')
        for layer,g in t.groupby('layer'):
            observed=g.loc[g.n_observed_y.gt(0),'n_observed_y_and_clinical_purity']
            coverage.append(dict(cancer_id=cancer,layer=layer,listed_targets=len(g),observed_targets=int(g.n_observed_y.gt(0).sum()),all_missing_targets=int(g.n_observed_y.eq(0).sum()),complete_fivefold_targets=int((f.layer.eq(layer)&f.complete_fivefold).sum()),min_observed_patients=int(observed.min()) if len(observed) else 0,max_observed_patients=int(observed.max()) if len(observed) else 0))
    val=pd.concat(frames,ignore_index=True);agg=pd.concat(layers,ignore_index=True)
    val.to_csv(root/'ALL_TARGET_COMPARISONS.tsv',sep='\t',index=False);agg.to_csv(root/'ALL_LAYER_SUMMARIES.tsv',sep='\t',index=False)
    pd.DataFrame(coverage).to_csv(root/'TARGET_COVERAGE.tsv',sep='\t',index=False)
    figs=root/'figures';figs.mkdir(exist_ok=True)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'lncrna':'#3C9277','mrna':'#2878B5','combined':'#D97724'}
    def save(fig,name):
        if f.empty:fig.text(.5,.5,'No targets succeeded in all three arms',ha='center',va='center',bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
        fig.savefig(figs/(name+'.png'),dpi=300,bbox_inches='tight');fig.savefig(figs/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    for cancer in ['BRCA','LUAD','COAD']:
        f=val.loc[val.cancer_id.eq(cancer)&val.complete_fivefold].copy();a=agg.loc[agg.cancer_id.eq(cancer)]
        names=a.layer.drop_duplicates().to_list();xx=np.arange(len(names));width=.25
        fig,ax=plt.subplots(figsize=(9,4.8))
        for i,arm in enumerate(ARMS):
            z=a.loc[a.arm.eq(arm)].set_index('layer').reindex(names)
            ax.bar(xx+(i-1)*width,z.median_oof_r2,width,label=ARM_LABELS[arm],color=colors[arm])
        ax.set_xticks(xx,[n.replace('_',' ') for n in names],rotation=15,ha='right');ax.set_ylabel('Median target OOF R2');ax.axhline(0,color='gray',lw=.8);ax.legend(frameon=False)
        ax.set_title(cancer+' | Common successful targets and held-out patients');save(fig,cancer+'_01_three_arm_r2')
        fig,ax=plt.subplots(figsize=(6.6,5.5))
        for name,g in f.groupby('layer'):ax.scatter(g.r2_mrna,g.r2_combined,s=9,alpha=.45,label=name.replace('_',' '),rasterized=True)
        ax.set_xscale('symlog',linthresh=.1);ax.set_yscale('symlog',linthresh=.1)
        lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],color='gray',lw=.8);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi)
        ax.set_xlabel('mRNA OOF R2 (symlog scale)');ax.set_ylabel('lncRNA + mRNA OOF R2 (symlog scale)');ax.legend(frameon=False,fontsize=7);ax.set_title(cancer+' | Added predictive value of lncRNA');save(fig,cancer+'_02_paired_r2')
        fig,ax=plt.subplots(figsize=(9,4.8))
        for i,arm in enumerate(ARMS):
            z=a.loc[a.arm.eq(arm)].set_index('layer').reindex(names)
            values=100*z.positive_r2_targets/z.n_targets
            ax.bar(xx+(i-1)*width,values,width,label=ARM_LABELS[arm],color=colors[arm])
        ax.set_xticks(xx,[n.replace('_',' ') for n in names],rotation=15,ha='right');ax.set_ylim(0,100);ax.set_ylabel('Targets with OOF R2 > 0 (%)');ax.legend(frameon=False);ax.set_title(cancer+' | R2 > 0 among common successful targets');save(fig,cancer+'_03_positive_r2')
        fig,ax=plt.subplots(figsize=(11,7))
        top=f.loc[f.delta_r2_combined_vs_mrna.notna()].assign(size=lambda z:z.delta_r2_combined_vs_mrna.abs()).nlargest(20,'size').sort_values('delta_r2_combined_vs_mrna')
        yy=np.arange(len(top));ax.hlines(yy,top.delta_r2_conditional_bootstrap_low,top.delta_r2_conditional_bootstrap_high,color='#B6B6B6',lw=1.3)
        ax.scatter(top.delta_r2_combined_vs_mrna,yy,c=np.where(top.delta_r2_combined_vs_mrna.ge(0),'#D97724','#2878B5'),s=22,zorder=3)
        labels=[t.split('|',1)[-1].replace('MSIGDB:','') for t in top.target_id];ax.set_yticks(yy,labels,fontsize=7);ax.axvline(0,color='gray',lw=.8)
        ax.set_xlabel('OOF delta R2: combined minus mRNA');ax.set_title(cancer+' | Largest differences (exploratory selection)')
        fig.text(.12,.005,'Intervals: paired patient bootstrap of fixed OOF errors; no model refitting or multiplicity correction.',fontsize=8)
        save(fig,cancer+'_04_largest_paired_differences')
    extra={}
    if (root/'BRCA'/'TARGET_TRAINING_STATUS.tsv').exists():
        from cc_hhgt.m0_no_screen import append_failure_reports
        extra=append_failure_reports(root,previous)
        if previous is not None:
            # The old paired-score table omits some attempted targets lacking
            # enough evaluation observations. Keep them in failure denominators.
            reference=pd.read_csv(root/'SCREENING_POLICY_COMPARISON.tsv',sep='\t')
            for cancer in ['BRCA','LUAD','COAD']:
                old=Path(previous)/cancer
                meta=pd.read_csv(old/'TARGETS.tsv',sep='\t')
                for arm in ARMS:
                    attempted=set()
                    for phase in [f'outer{k}' for k in range(5)]+['final']:
                        attempted.update(pd.read_parquet(old/phase/arm/'AUDIT.parquet',columns=['target_id']).target_id)
                    for layer,g in meta.groupby('layer'):
                        mask=reference.cancer_id.eq(cancer)&reference.arm.eq(arm)&reference.layer.eq(layer)
                        reference.loc[mask,'old_attempted_targets']=len(attempted&set(g.target_id))
            reference.to_csv(root/'SCREENING_POLICY_COMPARISON.tsv',sep='\t',index=False)
    summary=dict(target_host='149',status='COMPLETE',cases=receipts,total_target_comparisons=len(val),complete_fivefold_target_comparisons=int(val.complete_fivefold.sum()),figures_png=len(list(figs.glob('*.png'))),figures_pdf=len(list(figs.glob('*.pdf'))),primary_comparison='COMBINED_VS_MRNA_PAIRED_OOF',cohort_scope='CPTAC_ONLY_FOR_CURRENT_NATIVE_ASSAY_LABELS',interpretation='Retrospective prediction of native assay-derived scores; no causal or absolute activation claim. Native upstream normalization was not refit within outer folds. Bootstrap intervals condition on fitted OOF predictions and are exploratory.')
    summary.update(extra)
    write_json(root/'SUMMARY.json',summary)
    files={p.relative_to(root).as_posix():sha(p) for p in root.iterdir() if p.is_file() and p.suffix in ['.tsv','.json'] and p.name not in ['COMPLETE.json','STATUS.json','PROCESSES.json']}
    files.update({p.relative_to(root).as_posix():sha(p) for p in figs.iterdir() if p.is_file()})
    write_json(root/'COMPLETE.json',dict(status='COMPLETE',target_host='149',summary=summary,files=files));print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);p.add_argument('--previous');a=p.parse_args();finalize(a.results,a.previous)
