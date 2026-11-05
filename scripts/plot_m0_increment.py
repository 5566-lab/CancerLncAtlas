"""Separate RNA-arm performance panels and matched incremental comparison."""
import argparse,json,socket,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
ARMS=["lncrna","mrna","combined"]
LABELS={"lncrna":"lncRNA","mrna":"mRNA","combined":"lncRNA + mRNA"}
COLORS={"lncrna":"#23836b","mrna":"#377eb8","combined":"#dd842b"}
CANCERS=["BRCA","LUAD","COAD"]
NAMES={"PROTEIN_PATHWAY":"Protein pathways","PHOSPHO_PTMSEA":"Phosphorylation","METHYLATION_PATHWAY":"Methylation pathways","RNA_PATHWAY":"RNA pathways","TF":"TF scores (RNA-derived)","STATE_RNA":"State (RNA-derived)","STATE_OTHER":"State (non-RNA / source endpoints)"}
def classify(f):
    f=f.copy();f["display_layer"]=f.layer
    if "family" in f:
        f.loc[f.layer.eq("STATE"),"display_layer"]="STATE_RNA"
        f.loc[f.layer.eq("STATE") & f.family.eq("transcription_factor"),"display_layer"]="TF"
        other=["purity","ploidy","mutation_burden","hrd_scar","msi","sbs_spectrum"]
        mask=f.layer.eq("STATE") & (f.family.isin(other) | f.target_id.str.contains("stemness_dna|leukocyte_fraction",regex=True))
        f.loc[mask,"display_layer"]="STATE_OTHER"
        f.loc[f.target_id.str.contains("ESTIMATE::TumorPurity",regex=False),"display_layer"]="STATE_RNA"
    return f

def plot(inputs,output):
    assert socket.gethostname()=="149"
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tab=classify(pd.concat([pd.read_csv(Path(p)/"ALL_TARGET_COMPARISONS.tsv",sep="\t") for p in inputs],ignore_index=True))
    if tab.duplicated(["cancer_id","target_id"]).any():raise ValueError("Overlapping targets in plot inputs")
    good=tab.loc[tab.complete_fivefold & np.isfinite(tab[["r2_"+a for a in ARMS]]).all(axis=1)].copy()
    layers=[k for k in NAMES if (tab.display_layer==k).any()]
    summary=[]
    for layer in layers:
        for cancer in ["ALL"]+CANCERS:
            z=good.loc[good.display_layer.eq(layer)]
            if cancer!="ALL":z=z.loc[z.cancer_id.eq(cancer)]
            d=z.delta_r2_combined_vs_mrna
            summary.append(dict(layer=layer,cancer_id=cancer,n_targets=len(z),improved=int((d>1e-9).sum()),worse=int((d< -1e-9).sum()),ties=int((d.abs()<=1e-9).sum()),median_delta_r2=d.median(),**{"median_r2_"+a:z["r2_"+a].median() for a in ARMS}))
    pd.DataFrame(summary).to_csv(output/"PLOT_SUMMARY.tsv",sep="\t",index=False)
    tab.to_csv(output/"PLOT_TARGETS.tsv",sep="\t",index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    def save(fig,name):
        for ext in ["png","pdf"]:fig.savefig(output/(name+"."+ext),dpi=300,bbox_inches="tight")
        plt.close(fig)
    rng=np.random.default_rng(20261105)
    # Every separate figure uses exactly the same target denominator and y scale.
    vals=good[["r2_"+a for a in ARMS]].to_numpy().ravel()
    lower=min(-.1,float(np.nanmin(vals))-.08) if len(vals) else -.1
    upper=max(.2,float(np.nanmax(vals))+.12) if len(vals) else 1
    for j,a in enumerate(ARMS,1):
        fig,axs=plt.subplots(1,len(layers),figsize=(max(8,4.2*len(layers)),4.8),squeeze=False,sharey=True)
        for ax,layer in zip(axs[0],layers):
            for i,c in enumerate(CANCERS):
                z=good.loc[good.display_layer.eq(layer)&good.cancer_id.eq(c),"r2_"+a].to_numpy()
                if len(z):
                    ax.scatter(i+rng.uniform(-.18,.18,len(z)),z,s=10,color=COLORS[a],alpha=.35,rasterized=True)
                    med=np.median(z);ax.plot([i-.25,i+.25],[med,med],color="#222222",lw=2)
                    ax.text(i,upper-.02*(upper-lower),f"n={len(z)}\nmedian={med:.3f}",ha="center",va="top",fontsize=9)
                else:ax.text(i,0,"No common\nsuccessful targets",ha="center",fontsize=8)
            ax.axhline(0,color="#999999",ls="--",lw=.8);ax.set_xticks(range(3),CANCERS);ax.set_xlim(-.6,2.6);ax.set_ylim(lower,upper);ax.set_title(NAMES[layer])
        axs[0,0].set_ylabel("Held-out pooled R? (five folds)")
        fig.suptitle(LABELS[a]+" | clinical and DNA purity covariates included",y=1.04,fontsize=15)
        fig.text(.5,-.02,"Each point: one cancer ? target. Same targets in all three figures; failed models excluded and reported separately.",ha="center",fontsize=10)
        fig.tight_layout();save(fig,f"0{j}_"+a+"_performance")
    fig,axs=plt.subplots(2,len(layers),figsize=(max(9,4.8*len(layers)),8.8),squeeze=False)
    for j,layer in enumerate(layers):
        z=good.loc[good.display_layer.eq(layer)];ax=axs[0,j]
        for i,c in enumerate(CANCERS):
            zz=z.loc[z.cancer_id.eq(c)];ax.scatter(zz.r2_mrna,zz.r2_combined,s=17,alpha=.6,color=["#4778ad","#b45b62","#779d48"][i],label=c,rasterized=True)
        lo=min(-.1,z[["r2_mrna","r2_combined"]].min().min()-.05) if len(z) else -.1
        hi=max(.2,z[["r2_mrna","r2_combined"]].max().max()+.05) if len(z) else 1
        ax.plot([lo,hi],[lo,hi],color="#777777",ls="--",lw=1);ax.set_xlim(lo,hi);ax.set_ylim(lo,hi);ax.set_xlabel("mRNA R?");ax.set_ylabel("lncRNA + mRNA R?");ax.set_title(NAMES[layer]);ax.legend(frameon=False,fontsize=8)
        ax=axs[1,j];d=z.delta_r2_combined_vs_mrna.to_numpy()
        for i,c in enumerate(CANCERS):
            dd=z.loc[z.cancer_id.eq(c),"delta_r2_combined_vs_mrna"].to_numpy()
            if len(dd):
                ax.scatter(i+rng.uniform(-.18,.18,len(dd)),dd,s=12,alpha=.5,color=np.where(dd>1e-9,"#23836b",np.where(dd< -1e-9,"#b75e59","#999999")),rasterized=True)
                ax.plot([i-.25,i+.25],[np.median(dd)]*2,color="#222222",lw=2)
        ax.axhline(0,color="#666666",ls="--",lw=1);ax.set_xticks(range(3),CANCERS);ax.set_ylabel("?R? = combined ? mRNA")
        n=len(d);pos=int((d>1e-9).sum());pct=100*pos/n if n else 0
        ax.set_title(f"Improved: {pos}/{n} ({pct:.1f}%) | median ?={np.median(d):+.3f}" if n else "No common successful targets",fontsize=11)
    fig.suptitle("Does adding lncRNA improve prediction?",fontsize=17,y=1.02)
    fig.text(.5,-.015,"Above diagonal / ?R? > 0: improvement. Exploratory OOF results; target families are correlated; no causal claim.",ha="center",fontsize=10)
    fig.tight_layout();save(fig,"04_combined_vs_mrna_comparison")
    # Explicit all-attempted training coverage, independent of common-success performance.
    statuses=[]
    for p in inputs:
        f=Path(p)/"ALL_ARM_TARGET_METRICS.tsv"
        if f.exists():statuses.append(pd.read_csv(f,sep="\t"))
    if statuses:
        st=classify(pd.concat(statuses,ignore_index=True));rows=[]
        fig,axs=plt.subplots(1,len(layers),figsize=(max(8,4.2*len(layers)),4.6),squeeze=False,sharey=True)
        for ax,layer in zip(axs[0],layers):
            for j,a in enumerate(ARMS):
                vv=[]
                for c in CANCERS:
                    f=st.loc[st.display_layer.eq(layer)&st.arm.eq(a)&st.cancer_id.eq(c)];n=int(f.attempted.sum());s=int(f.training_success.sum());vv.append(100*s/n if n else np.nan)
                    rows.append(dict(layer=layer,cancer_id=c,arm=a,listed=len(f),attempted=n,success=s,failed=n-s,unavailable=len(f)-n))
                ax.bar(np.arange(3)+(j-1)*.24,vv,width=.24,label=LABELS[a],color=COLORS[a])
            ax.set_xticks(range(3),CANCERS);ax.set_ylim(0,105);ax.set_title(NAMES[layer]);ax.legend(frameon=False,fontsize=8)
        axs[0,0].set_ylabel("Successful targets / all attempted (%)")
        fig.suptitle("Training coverage | zero RNA in any fold or final fit = failure",y=1.04)
        fig.tight_layout();save(fig,"05_training_coverage")
        pd.DataFrame(rows).to_csv(output/"PLOT_TRAINING_COVERAGE.tsv",sep="\t",index=False)
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.suffix in ["tsv",".tsv",".png",".pdf"]}
    (output/"COMPLETE.json").write_text(json.dumps(dict(target_host="149",inputs=inputs,common_successful_targets=len(good),files=files),indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--inputs",nargs="+",required=True);p.add_argument("--output",required=True);a=p.parse_args();plot(a.inputs,a.output)
