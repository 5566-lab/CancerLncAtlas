"""Audit completed three-cancer joint M0 results and export standalone figures."""
import argparse,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from cla_engine.common import sha
from run_m0_linear import write_json

def finalize(root):
    if socket.gethostname()!='149':raise RuntimeError('Host 149 required')
    root=Path(root);cancers=['BRCA','LUAD','COAD'];receipts=[];frames=[];transfers=[];audits=[];quality=[];features=[]
    for ca in cancers:
        d=root/ca;r=json.loads((d/'COMPLETE.json').read_text())
        if r['status']!='COMPLETE':raise ValueError('Incomplete case '+ca)
        for name,h in r['files'].items():
            if sha(d/name)!=h:raise ValueError('Result hash mismatch '+ca+'/'+name)
        receipts.append(r['summary']);f=pd.read_csv(d/'COHORT_VALIDATION.tsv',sep='\t');frames.append(f)
        tr=pd.read_csv(d/'CROSS_COHORT_VALIDATION.tsv',sep='\t');transfers.append(tr)
        a=pd.read_parquet(d/'ALL_AUDITS.parquet');a['cancer_id']=ca;audits.append(a)
        g=pd.read_parquet(d/'FINAL_GENE_SETS.parquet');g['cancer_id']=ca;features.append(g)
        metadata=pd.read_csv(d/'TARGETS.tsv',sep='\t');fin=a.loc[a.phase.eq('final')].merge(metadata,on='target_id',validate='one_to_one')
        for row in fin.itertuples():
            ff=f.loc[f.target_id.eq(row.target_id)];tt=tr.loc[tr.target_id.eq(row.target_id)]
            shared=row.scope=='SHARED_RECOMPUTED';both=set(ff.cohort)=={'TCGA','CPTAC'}
            quality.append(dict(cancer_id=ca,target_id=row.target_id,layer=row.layer,scope=row.scope,fit_status=row.status,n_selected=getattr(row,'n_selected',0),
                both_cohorts_evaluated=both,both_cohorts_positive_mean_r2=bool(both and (ff.mean_fold_r2>0).all()),
                both_cohorts_positive_increment=bool(both and (ff.mean_delta_r2_over_covariates>0).all()),
                both_transfer_directions_evaluated=bool(shared and set(tt.cohort)=={'TCGA','CPTAC'}),
                both_transfer_r2_positive=bool(shared and set(tt.cohort)=={'TCGA','CPTAC'} and (tt.r2>0).all())))
    val=pd.concat(frames,ignore_index=True);tr=pd.concat(transfers,ignore_index=True);aa=pd.concat(audits,ignore_index=True);gg=pd.concat(features,ignore_index=True);q=pd.DataFrame(quality)
    val.to_csv(root/'ALL_COHORT_VALIDATION.tsv',sep='\t',index=False);tr.to_csv(root/'ALL_CROSS_COHORT_VALIDATION.tsv',sep='\t',index=False);q.to_csv(root/'TARGET_VALIDATION_STATUS.tsv',sep='\t',index=False)
    gg.to_parquet(root/'ALL_FINAL_GENE_SETS.parquet',index=False,compression='zstd')
    summary=dict(target_host='149',status='COMPLETE',cases=receipts,total_final_models=sum(x['n_final_models'] for x in receipts),nonempty_signatures=sum(x['nonempty_signatures'] for x in receipts),
      shared_targets_both_cohorts_positive_r2_and_increment=int((q.both_cohorts_positive_mean_r2&q.both_cohorts_positive_increment&q.scope.eq('SHARED_RECOMPUTED')).sum()),
      note='Positive metrics are descriptive screening flags, not a causal or independently replicated signature claim. Missing clinical/purity values and unsupported clinical categories reduce evaluation coverage.')
    figdir=root/'figures';figdir.mkdir(exist_ok=True)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
    def save(fig,name):
        fig.savefig(figdir/(name+'.png'),bbox_inches='tight',dpi=300);fig.savefig(figdir/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    colors={'TCGA':'#2878B5','CPTAC':'#D97724'}
    for ca in cancers:
        v=val.loc[val.cancer_id.eq(ca)];t=tr.loc[tr.cancer_id.eq(ca)];s=q.loc[q.cancer_id.eq(ca)];genes=gg.loc[gg.cancer_id.eq(ca)]
        fig,ax=plt.subplots(figsize=(9,4.6));layers=list(v.layer.unique());pos=np.arange(len(layers));width=.34
        for i,cc in enumerate(['TCGA','CPTAC']):
            counts=[int(((s.layer==ly)&(s.n_selected.fillna(0)>0)&s.target_id.isin(v.loc[v.cohort.eq(cc),'target_id'])).sum()) for ly in layers]
            ax.bar(pos+(i-.5)*width,counts,width,label=cc,color=colors[cc])
        ax.set_xticks(pos,layers,rotation=25,ha='right');ax.set_ylabel('Nonempty signatures with cohort evaluation');ax.set_title(ca+' | Target coverage');ax.legend(frameon=False);save(fig,ca+'_01_target_coverage')
        vv=v.loc[v.scope.eq('SHARED_RECOMPUTED')].copy();p=vv.pivot(index='target_id',columns='cohort',values='mean_fold_r2').dropna()
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(p.TCGA,p.CPTAC,s=9,alpha=.35,color='#2878B5',rasterized=True);ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('TCGA mean held-out fold R2');ax.set_ylabel('CPTAC mean held-out fold R2');ax.set_title(ca+' | Shared targets');save(fig,ca+'_02_cohort_r2')
        p=vv.pivot(index='target_id',columns='cohort',values='mean_delta_r2_over_covariates').dropna()
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(p.TCGA,p.CPTAC,s=9,alpha=.35,color='#3C9277',rasterized=True);ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('TCGA delta R2 over clinical/purity baseline');ax.set_ylabel('CPTAC delta R2 over clinical/purity baseline');ax.set_title(ca+' | Added prediction from lncRNA');save(fig,ca+'_03_lncrna_increment')
        tt=t.loc[t.scope.eq('SHARED_RECOMPUTED')];p=tt.pivot(index='target_id',columns='cohort',values='r2').dropna()
        fig,ax=plt.subplots(figsize=(6,5));ax.scatter(p.TCGA,p.CPTAC,s=9,alpha=.35,color='#8B60A4',rasterized=True);ax.axhline(0,color='gray',lw=.8);ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Train CPTAC, test TCGA: R2');ax.set_ylabel('Train TCGA, test CPTAC: R2');ax.set_title(ca+' | Transfer without test recalibration');save(fig,ca+'_04_cross_cohort_transfer')
        score=vv.groupby('target_id').agg(worst_cohort_r2=('mean_fold_r2','min'),worst_increment=('mean_delta_r2_over_covariates','min'),cohorts=('cohort','nunique'));top=score.loc[score.cohorts.eq(2)].sort_values('worst_increment',ascending=False).head(20).sort_values('worst_increment')
        fig,ax=plt.subplots(figsize=(10,7));labels=[x.split('|',1)[-1].replace('MSIGDB:','') for x in top.index];ax.barh(np.arange(len(top)),top.worst_increment,color='#3C9277');ax.set_yticks(np.arange(len(top)),labels,fontsize=7);ax.axvline(0,color='gray',lw=.8);ax.set_xlabel('Smaller of the two cohort mean delta R2 values');ax.set_title(ca+' | Largest shared lncRNA prediction gains');save(fig,ca+'_05_shared_increment_top20')
        fig,axs=plt.subplots(1,2,figsize=(10,4));counts=genes.groupby('target_id').size();axs[0].hist(counts,bins=40,color='#2878B5');axs[0].set_xlabel('Selected lncRNAs per nonempty final signature');axs[0].set_ylabel('Targets');axs[1].hist(genes.selection_frequency.dropna(),bins=np.arange(-.1,1.2,.2),color='#3C9277');axs[1].set_xlabel('Outer-fold selection frequency');axs[1].set_ylabel('Final gene-target pairs');fig.suptitle(ca+' | Signature size and selection stability');save(fig,ca+'_06_signature_stability')
    pd.DataFrame(receipts).to_csv(root/'CASE_SUMMARY.tsv',sep='\t',index=False);write_json(root/'SUMMARY.json',summary)
    files={p.relative_to(root).as_posix():sha(p) for p in root.iterdir() if p.is_file() and p.suffix in ['.tsv','.json','.parquet'] and p.name not in ['COMPLETE.json','PROCESSES.json']}
    files.update({p.relative_to(root).as_posix():sha(p) for p in figdir.iterdir() if p.is_file()})
    write_json(root/'COMPLETE.json',dict(target_host='149',status='COMPLETE',summary=summary,files=files));print(json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',required=True);a=p.parse_args();finalize(a.results)
