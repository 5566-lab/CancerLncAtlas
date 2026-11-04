"""Plot Lasso demo performance and final signatures; execute on host 149."""
import argparse,json,socket,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"scripts"))
from run_m0_linear import write_json
from plot_m0_linear import save,heatmap,COLORS,LABELS
from cla_engine.common import sha

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--results",required=True);ap.add_argument("--inputs",required=True);ap.add_argument("--output",required=True)
    a=ap.parse_args()
    if socket.gethostname()!="149":raise RuntimeError("Run on 149")
    root=Path(a.results);out=Path(a.output)
    if out.exists() and any(out.iterdir()):raise ValueError("Use a new figure directory")
    out.mkdir(parents=True)
    rec=json.loads((root/"COMPLETE.json").read_text());cases=rec["cases"]
    write_json(out/"LAUNCH_PREFLIGHT.json",dict(target_host="149",action="PLOT_LASSO_DEMO",results_complete_sha256=sha(root/"COMPLETE.json")))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    rows=[];signatures=[]
    for case in cases:
        d=root/case;m=pd.read_csv(d/"METRICS.tsv",sep="\t");fm=pd.read_csv(d/"FINAL_MODELS.tsv",sep="\t")
        genes=pd.read_parquet(d/"FINAL_GENE_SETS.parquet");joined=m.merge(fm,on="target_id",suffixes=("","_model"),validate="one_to_one")
        for layer,g in joined.groupby("layer"):
            v=g.loc[g.r2.notna()]
            rows.append(dict(case=case,layer=layer,n_evaluable=len(v),median_r2=v.r2.median(),
                fraction_r2_positive=v.r2.gt(0).mean(),median_selected_genes=g.n_selected.median(),
                nonempty_signatures=int(g.n_selected.gt(0).sum()),median_screened_genes=g.n_screened.median()))
        signatures.append(genes.assign(case=case))
        # Predetermined biology examples; selection does not depend on test performance.
        y=pd.read_parquet(Path(a.inputs)/case/"Y.parquet");pred=pd.read_parquet(d/"OOF_PREDICTIONS.parquet")
        targets=["STATE|signifinder::Hypoxia_Buffa","STATE|CYT::mean_log_GZMA_PRF1","STATE|stemness_rna::RNAss"]
        fig,axs=plt.subplots(1,3,figsize=(13,4.3))
        for ax,t in zip(axs,targets):
            ax.set_title(t.split("|",1)[1],fontsize=10)
            if t not in y or not np.isfinite(pred[t]).any():ax.text(.5,.5,"Unavailable",ha="center",transform=ax.transAxes);continue
            ok=np.isfinite(y[t])&np.isfinite(pred[t]);v=y.loc[ok,t];p=pred.loc[ok,t]
            ax.scatter(v,p,s=13,alpha=.6,color="#3274A1",edgecolors="none")
            lo=min(v.min(),p.min());hi=max(v.max(),p.max());ax.plot([lo,hi],[lo,hi],"--",c=".5",lw=1)
            stat=joined.set_index("target_id").loc[t]
            ax.text(.04,.96,f"OOF R2 = {stat.r2:.3f}\nFinal genes = {int(stat.n_selected) if np.isfinite(stat.n_selected) else 'NA'}",va="top",transform=ax.transAxes)
            ax.set(xlabel="Observed native score",ylabel="Held-out prediction")
        fig.suptitle("Correlation-screened Lasso | "+case)
        fig.text(.02,.005,"Five outer folds; screening and lambda selection exclude test patients. Final signatures are refitted on all observed patients.",fontsize=8)
        fig.tight_layout(rect=[0,.05,1,.93]);save(fig,out,case+"_examples")
        fig,axs=plt.subplots(1,2,figsize=(12,5))
        for layer in COLORS:
            vals=joined.loc[joined.layer.eq(layer),"n_selected"].dropna().sort_values().to_numpy()
            if len(vals):axs[0].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=LABELS[layer],c=COLORS[layer])
        axs[0].set(xlabel="Nonzero genes in final signature",ylabel="Fraction of fitted targets",title="Signature size")
        axs[0].set_xscale("symlog",linthresh=1);axs[0].legend(fontsize=8)
        ex=genes.loc[genes.target_id.eq(targets[0])].copy()
        ex=ex.loc[ex.standardized_coefficient.abs().sort_values(ascending=False).head(20).index].sort_values("standardized_coefficient")
        if len(ex):
            lab=ex.gene_symbol.fillna(ex.lncrna_id)
            colors=plt.get_cmap("viridis")(ex.selection_frequency.fillna(0))
            axs[1].barh(np.arange(len(ex)),ex.standardized_coefficient,color=colors)
            axs[1].set_yticks(np.arange(len(ex)),lab,fontsize=8);axs[1].axvline(0,color=".3",lw=.6)
            axs[1].set(xlabel="Coefficient per training SD of expression",title="Hypoxia: 20 largest final weights")
            sm=plt.cm.ScalarMappable(cmap="viridis",norm=plt.Normalize(0,1));fig.colorbar(sm,ax=axs[1],label="Fold selection frequency",shrink=.6)
        else:axs[1].text(.5,.5,"No selected Hypoxia genes",ha="center",transform=axs[1].transAxes)
        fig.suptitle(case+" | fixed final signatures")
        fig.text(.02,.005,"All nonzero genes are exported. The 20-gene display is not a selection limit; coefficients are predictive, not causal.",fontsize=8)
        fig.tight_layout(rect=[0,.05,1,.93]);save(fig,out,case+"_gene_sets")
    summary=pd.DataFrame(rows);summary.to_csv(out/"DEMO_SUMMARY.tsv",sep="\t",index=False)
    matrix=summary.pivot(index="case",columns="layer",values="median_r2").reindex(columns=[c for c in COLORS if c in set(summary.layer)])
    heatmap(matrix.rename(columns=LABELS),"Screened Lasso | median held-out R-squared","Median R-squared",out,"ALL_COHORTS_PERFORMANCE",vmin=-.3,vmax=1.)
    pd.concat(signatures,ignore_index=True).to_csv(out/"ALL_FINAL_GENE_SETS.tsv.gz",sep="\t",index=False)
    for case in cases:
        for file in ["FINAL_MODELS.tsv","METRICS.tsv","GENE_SELECTION_STABILITY.tsv.gz"]:
            source=root/case/file
            if source.exists():(out/(case+"_"+file)).write_bytes(source.read_bytes())
    text=("M0 correlation-screened Lasso demo\n"
          "Cancers: BRCA, LUAD, COAD; TCGA and CPTAC fitted separately.\n"
          "Both cohorts use the full shared GENCODE lncRNA axis and log2(TPM+1); upstream quantification batch effects are not removed.\n"
          "Screen: training-only |Spearman rho| >= 0.20 and BH FDR <= 0.05; observed pairs >= 10.\n"
          "Two-sided asymptotic Spearman p-values; BH family = every supplied lncRNA for this target and split, including untestable p=1.\n"
          "Lasso lambda selected on validation patients; outer test patients excluded from screening, imputation, scaling and tuning.\n"
          "Final gene sets are nonzero coefficients from full-data refits; validation metrics evaluate the nested training procedure.\n"
          "No passing genes / no predictive validation signal yields an intercept-only model. No arbitrary Top20 fallback.\n"
          "Existing frozen cohort-native Y scores are unchanged and are not cross-cohort calibrated.\n"
          "No MOGSA; HHGT configuration and Pathway=1.0, State=0.5 are unchanged.\n")
    (out/"RESULTS_README.txt").write_text(text)
    write_json(out/"FIGURES_COMPLETE.json",dict(status="COMPLETE",target_host="149",cases=cases,
       png=len(list(out.glob("*.png"))),pdf=len(list(out.glob("*.pdf"))),
       files={p.name:sha(p) for p in out.iterdir() if p.is_file()}))
if __name__=="__main__":main()
