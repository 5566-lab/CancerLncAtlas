"""Reuse M0 figures and add paired graph/multiomics ablations."""
import argparse,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from plot_m0_increment import plot,classify,ARMS,CANCERS,NAMES,COLORS,LABELS
from prepare_graph_ridge import dump
from cla_engine.common import sha
GROUPS={'01_assay_targets':['PROTEIN_PATHWAY','PHOSPHO_PTMSEA','METHYLATION_PATHWAY'],'02_rna_pathway':['RNA_PATHWAY'],'03_tf':['TF'],'04_state_rna':['STATE_RNA'],'05_state_other':['STATE_OTHER']}

def report(source,output,inputs,evidence=None,audit_file=None):
    assert socket.gethostname()=='149';source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    dump(out/'LAUNCH_PREFLIGHT.json',dict(target_host='149',stage='REPORT',source=str(source),code_sha256=sha(Path(__file__))))
    tables=[];metrics=[];statuses=[];audits=[]
    for c in CANCERS:
        d=source/c
        if not (d/'COMPLETE.json').exists():raise ValueError('TRAINING_INCOMPLETE: '+c)
        rec=json.loads((d/'COMPLETE.json').read_text())
        for name in ['TARGET_COMPARISON.tsv','ARM_TARGET_METRICS.tsv','TARGET_TRAINING_STATUS.tsv','AUDIT.parquet']:
            if sha(d/name)!=rec['files'][name]:raise ValueError('RESULT_HASH_MISMATCH')
        for name,coll in [('TARGET_COMPARISON.tsv',tables),('ARM_TARGET_METRICS.tsv',metrics),('TARGET_TRAINING_STATUS.tsv',statuses)]:
            f=pd.read_csv(d/name,sep='\t');f['cancer_id']=c;coll.append(f)
        a=pd.read_parquet(d/'AUDIT.parquet');a['cancer_id']=c;audits.append(a)
    tab=classify(pd.concat(tables,ignore_index=True));met=classify(pd.concat(metrics,ignore_index=True));status=classify(pd.concat(statuses,ignore_index=True));audit=pd.concat(audits,ignore_index=True)
    tab.to_csv(out/'ALL_VARIANT_COMPARISONS.tsv',sep='\t',index=False);met.to_csv(out/'ALL_VARIANT_METRICS.tsv',sep='\t',index=False);status.to_csv(out/'ALL_TRAINING_STATUS.tsv',sep='\t',index=False)
    primary=tab[tab.variant.eq('multiomics_graph')].copy();primary['complete_fivefold']=True
    pstatus=status[status.variant.eq('multiomics_graph')].copy();pm=met[met.variant.eq('multiomics_graph')]
    keys=['cancer_id','target_id','arm'];pstatus=pstatus.merge(pm[keys+['r2','rmse','mse','spearman','n']],on=keys,how='left',validate='one_to_one')
    allmeta=pstatus.drop_duplicates(['cancer_id','target_id'])[['cancer_id','target_id','layer','family','method','cohort','n_observed_y','role','excluded_blocks','allowed_blocks']]
    primary=allmeta.merge(primary.drop(columns=[c for c in allmeta if c not in ['cancer_id','target_id']]),on=['cancer_id','target_id'],how='left',validate='one_to_one');primary['complete_fivefold']=primary.complete_fivefold.fillna(False)
    primary=classify(primary)
    if evidence:
        evidence=Path(evidence)
        ov=pd.read_csv(evidence/'STATE_LNCRNA_OVERLAP_AUDIT.tsv',sep='\t')
        cols=['cancer_id','target_id','overlap_status','direct_lncrna_count','direct_lncrna_gene_ids']
        primary=primary.merge(ov[cols],on=['cancer_id','target_id'],how='left',validate='one_to_one')
        for name in ['STATE_LNCRNA_OVERLAP_AUDIT.tsv','RNA_SHARED_SYMBOL_AUDIT.tsv','FINAL_AUDIT.json','DOWNLOAD_COMPLETE.json','ORIGINAL_CODE_REUSE_AUDIT.json']:
            src=Path(audit_file) if name=='FINAL_AUDIT.json' and audit_file else evidence/name
            (out/name).write_bytes(src.read_bytes())
    primary.to_csv(out/'ALL_TARGET_COMPARISONS.tsv',sep='\t',index=False);pstatus.to_csv(out/'ALL_ARM_TARGET_METRICS.tsv',sep='\t',index=False)
    for name,layers in GROUPS.items():
        plot([str(out)],out/name,layers,model_label='Graph ridge + multiomics',coverage_caption='Training coverage | all five held-out folds and final fit required')
    summary=[]
    for layer,g in primary[primary.complete_fivefold].groupby('display_layer'):
        delta=g.delta_r2_combined_vs_mrna
        summary.append(dict(layer=layer,n_targets=len(g),improved=int(delta.gt(1e-9).sum()),improved_with_positive_combined_r2=int((delta.gt(1e-9)&g.r2_combined.gt(0)).sum()),median_delta_r2=float(delta.median()),**{'median_r2_'+a:float(g['r2_'+a].median()) for a in ARMS}))
    pd.DataFrame(summary).to_csv(out/'PRIMARY_SUMMARY.tsv',sep='\t',index=False)
    wide=met.pivot(index=['cancer_id','target_id','arm','display_layer'],columns='variant',values='r2').reset_index()
    wide=wide[wide.arm.isin(ARMS)].copy();wide['delta_multiomics']=wide.multiomics_ridge-wide.rna_only;wide['delta_graph']=wide.multiomics_graph-wide.multiomics_ridge
    shared=met[met.variant.eq('omics_only')][['cancer_id','target_id','r2']].rename(columns={'r2':'r2_omics_only'})
    wide=wide.merge(shared,on=['cancer_id','target_id'],how='left',validate='many_to_one');wide['delta_rna_over_omics_only']=wide.multiomics_graph-wide.r2_omics_only
    wide.to_csv(out/'PAIRED_ABLATIONS.tsv',sep='\t',index=False);layers=[x for x in NAMES if x in set(wide.display_layer)];ar=[]
    for effect,label,file in [('delta_multiomics','Added patient multiomics: same ridge model','06_multiomics_gain'),('delta_graph','Added graph constraint: same patient multiomics','07_graph_gain'),('delta_rna_over_omics_only','RNA contribution beyond shared patient multiomics','08_rna_increment_over_omics')]:
        fig,axs=plt.subplots(1,3,figsize=(16,6),sharey=True);rng=np.random.default_rng(14)
        for ax,arm in zip(axs,ARMS):
            for j,layer in enumerate(layers):
                f=wide[wide.arm.eq(arm)&wide.display_layer.eq(layer)];v=f[effect].dropna().to_numpy()
                if len(v):
                    ax.scatter(j+rng.uniform(-.15,.15,len(v)),v,s=8,alpha=.25,color=COLORS[arm],rasterized=True);ax.plot([j-.25,j+.25],[np.median(v)]*2,color='black',lw=2)
                ar.append(dict(effect=effect,arm=arm,layer=layer,n_targets=len(v),improved=int((v>1e-9).sum()),median_delta=float(np.median(v)) if len(v) else np.nan))
            ax.axhline(0,color='#888888',ls='--');ax.set_xticks(range(len(layers)),[{'STATE_RNA':'State (RNA)','STATE_OTHER':'State (other)'}.get(x,NAMES[x].replace(' pathways','')) for x in layers],rotation=55,ha='right');ax.set_yscale('symlog',linthresh=.05);ax.set_title(LABELS[arm])
        axs[0].set_ylabel('Paired held-out Delta R2');fig.suptitle(label);fig.tight_layout()
        for ext in ['png','pdf']:fig.savefig(out/(file+'.'+ext),dpi=300,bbox_inches='tight')
        plt.close(fig)
    pd.DataFrame(ar).to_csv(out/'ABLATION_SUMMARY.tsv',sep='\t',index=False)
    h=audit[audit.variant.eq('multiomics_graph')&audit.status.eq('FITTED')&audit.phase.ne('final')].groupby(['cancer_id','arm','eta']).size().rename('target_fold_fits').reset_index();h.to_csv(out/'GRAPH_STRENGTH_SELECTION.tsv',sep='\t',index=False)
    pd.read_csv(Path(inputs)/'COVERAGE.tsv',sep='\t').to_csv(out/'INPUT_COVERAGE.tsv',sep='\t',index=False)
    rows=[]
    for folder,layers in GROUPS.items():
        links=' | '.join(f'<a href="{folder}/{i:02d}_{slug}.png">{label}</a> (<a href="{folder}/{i:02d}_{slug}.pdf">PDF</a>)' for i,slug,label in [(1,'lncrna_performance','lncRNA'),(2,'mrna_performance','mRNA'),(3,'combined_performance','lncRNA + mRNA'),(4,'combined_vs_mrna_comparison','Comparison'),(5,'training_coverage','Coverage')]);rows.append('<tr><td>'+', '.join(NAMES[x] for x in layers)+'</td><td>'+links+'</td></tr>')
    text="<!doctype html><meta charset='utf-8'><title>Graph ridge multiomics comparison</title><style>body{font:16px Arial;max-width:1250px;margin:35px auto;padding:20px;color:#223344}td,th{padding:14px;border-bottom:1px solid #ddd;text-align:left}p{line-height:1.6}a{color:#176294}</style><h1>Graph-regularized ridge: three RNA arms</h1>"
    text+="<p>CPTAC BRCA, LUAD and COAD. All arms share target-eligible CNV, mutation, protein and phosphosite inputs. Original five outer folds, native Y and clinical/purity handling are reused. Hyperparameters and preprocessing use training data only. Upstream native scores retain their original cohort preprocessing. In revision 2, protein and phosphosite values are bounded by their observed TRAINING min/max before scaling; no coverage threshold is added. The original unbounded run is retained as a diagnostic comparison.</p>"
    text+="<p>Protein targets exclude protein inputs; phosphosite targets exclude phosphosite inputs. Genomic-derived State targets exclude corresponding genomic blocks. DNA purity excludes itself and genomic blocks. RNA-derived targets retain lncRNA predictors; direct constituent overlap must be considered. mRNA-containing models may reconstruct their score definitions.</p>"
    text+="<p>PPI/RPI constrain RNA coefficients. Unobserved arm nodes are topology-only bridges, never extra measured mRNA expression. The graph assumes coefficient smoothness, not activation/inhibition or causal regulation. Ridge weights are dense. Evaluation requires supported prediction across all five folds plus a final fit.</p><table><tr><th>Score family</th><th>Figures</th></tr>"+''.join(rows)+"</table>"
    text+="<p>Ablations: <a href='06_multiomics_gain.png'>Added multiomics</a> | <a href='07_graph_gain.png'>Added graph</a> | <a href='08_rna_increment_over_omics.png'>RNA beyond shared multiomics</a></p><p>Tables: <a href='PRIMARY_SUMMARY.tsv'>Primary summary</a> | <a href='ALL_TARGET_COMPARISONS.tsv'>Every primary target</a> | <a href='ALL_TRAINING_STATUS.tsv'>All training status</a> | <a href='PAIRED_ABLATIONS.tsv'>Paired ablations</a> | <a href='INPUT_COVERAGE.tsv'>Input coverage</a> | <a href='GRAPH_STRENGTH_SELECTION.tsv'>Graph strength</a></p>"
    text+="<p>Existing code: <a href='ORIGINAL_CODE_REUSE_AUDIT.json'>Original M0 module hashes and unchanged entry functions</a>. The Lasso implementation remains available; this run adds a graph-ridge estimator.</p><p>RNA overlap: <a href='STATE_LNCRNA_OVERLAP_AUDIT.tsv'>State constituent audit</a> | <a href='RNA_SHARED_SYMBOL_AUDIT.tsv'>Coding/lncRNA shared symbols</a>. Coding-symbol selection follows upstream same-symbol TPM summation and is not proof of zero lncRNA contribution. <a href='FINAL_AUDIT.json'>Prediction and leakage checks</a>.</p>"
    (out/'OPEN_REPORT.html').write_text(text,encoding='utf-8')
    dump(out/'REPORT_COMPLETE.json',dict(target_host='149',status='COMPLETE',source=str(source),summary=summary,files={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p!=out/'REPORT_COMPLETE.json'}))
    print('REPORT_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--inputs',required=True);p.add_argument('--evidence');p.add_argument('--audit-file');a=p.parse_args();report(a.source,a.output,a.inputs,a.evidence,a.audit_file)
