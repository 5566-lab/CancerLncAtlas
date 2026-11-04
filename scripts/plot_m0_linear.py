#!/usr/bin/env python3
"""Export M0 out-of-fold performance figures and compact review tables on 149."""
import argparse,json,socket,sys,textwrap
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cla_engine.common import sha

COLORS={"RNA_PATHWAY":"#3274A1","PROTEIN_PATHWAY":"#E1812C","METHYLATION_PATHWAY":"#3A923A",
        "PHOSPHO_PTMSEA":"#C03D3E","STATE":"#9372B2"}
LABELS={"RNA_PATHWAY":"RNA pathway / TF","PROTEIN_PATHWAY":"Protein pathway",
        "METHYLATION_PATHWAY":"Promoter methylation","PHOSPHO_PTMSEA":"Phosphosite PTM-SEA","STATE":"State"}
def save(fig,out,name):
    fig.savefig(out/(name+".png"),dpi=300,bbox_inches="tight",facecolor="white")
    fig.savefig(out/(name+".pdf"),bbox_inches="tight",facecolor="white")
    plt.close(fig)

def heatmap(data,title,label,out,name,*,vmin=-.3,vmax=1.0,percent=False):
    fig,ax=plt.subplots(figsize=(max(8,len(data.columns)*1.1),max(5,len(data)*.32+2)))
    z=data.to_numpy(float)
    cmap=plt.get_cmap("YlGnBu").copy() if percent else plt.get_cmap("RdBu").copy()
    cmap.set_bad("#e6e6e6")
    norm=None if percent else TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=vmax)
    im=ax.imshow(z,aspect="auto",cmap=cmap,norm=norm,vmin=vmin if percent else None,vmax=vmax if percent else None)
    ax.set_xticks(range(len(data.columns)),[str(c).replace("_"," ") for c in data.columns],rotation=35,ha="right",fontsize=9)
    ax.set_yticks(range(len(data)),data.index,fontsize=9)
    for i in range(len(data)):
        for j in range(len(data.columns)):
            v=z[i,j]
            ax.text(j,i,("NA" if not np.isfinite(v) else f"{v:.0%}" if percent else f"{v:.2f}"),
                    ha="center",va="center",fontsize=8,color="white" if np.isfinite(v) and (v>.65 if percent else v>.65 or v<-.2) else "black")
    ax.set_title(title,pad=14,fontsize=14)
    fig.colorbar(im,ax=ax,shrink=.7,label=label,extend="neither" if percent else "both")
    fig.tight_layout(rect=[0,.12,1,1])
    fig.text(.02,.005,"Patient-disjoint outer folds; existing cohort-native scores. Grey = unavailable. Cohorts are not pooled.",fontsize=9)
    save(fig,out,name)

def case_figures(case,out):
    m=pd.read_csv(case/"METRICS.tsv",sep="\t")
    info=json.loads((case/"INPUTS.json").read_text())["info"]
    label=info["dataset"]+" / "+info["cancer_id"]
    layers=[k for k in COLORS if k in set(m.layer)]
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for layer in layers:
        v=np.sort(m.loc[m.layer.eq(layer),"r2"].dropna().to_numpy())
        if len(v):axes[0].plot(v,np.arange(1,len(v)+1)/len(v),label=LABELS[layer],color=COLORS[layer],lw=1.8)
    axes[0].axvline(0,color=".3",ls="--",lw=1)
    axes[0].set(xlim=(-1,1),ylim=(0,1.02),xlabel="Out-of-fold R-squared",ylabel="Fraction of evaluated targets")
    low=int((m.r2 < -1).sum())
    axes[0].set_title(f"Prediction performance ({low} targets below -1)")
    axes[0].legend(fontsize=8,loc="upper left")
    for i,layer in enumerate(layers):
        row=m.loc[m.layer.eq(layer)]
        evaluated=row.r2.notna()
        axes[1].barh(i,int(evaluated.sum()),color=COLORS[layer],alpha=.85)
        axes[1].text(int(evaluated.sum())+5,i,f"{evaluated.sum()} / {len(row)}",va="center",fontsize=9)
        fr=row.loc[evaluated,"skill_vs_train_mean"].gt(0).mean()
        axes[2].barh(i,0 if not np.isfinite(fr) else fr,color=COLORS[layer],alpha=.85)
        axes[2].text((0 if not np.isfinite(fr) else fr)+.01,i,"NA" if not np.isfinite(fr) else f"{fr:.1%}",va="center",fontsize=9)
    for ax in axes[1:]:
        ax.set_yticks(range(len(layers)),[LABELS[l] for l in layers],fontsize=9)
        ax.invert_yaxis()
    axes[1].set_title("Evaluated / available target axis");axes[1].set_xlim(0,max(m.layer.value_counts())*1.35)
    axes[2].set_title("Better than training-mean baseline");axes[2].set_xlim(0,1.17)
    fig.suptitle(f"M0 | {label} | {info['n_patients']} patients | {info['n_features']:,} lncRNAs",fontsize=15)
    fig.text(.02,.005,"All results use held-out patients. Native score prediction is not evidence of causal pathway activation.",fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.92]);save(fig,out,case.name+"_performance")

    # Fixed examples chosen by biology/identity, never by observed test performance.
    desired=["HALLMARK_HYPOXIA","HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",
             "HALLMARK_OXIDATIVE_PHOSPHORYLATION","CYT::mean_log_GZMA_PRF1",
             "stemness_rna::RNAss","signifinder::Hypoxia_Buffa"]
    ys=pd.read_parquet(case/"OBSERVED_SCORES.parquet")
    yp=pd.read_parquet(case/"OOF_PREDICTIONS.parquet")
    folds=pd.read_csv(case/"PATIENT_FOLDS.tsv",sep="\t").set_index("patient_id").outer_fold.reindex(ys.index)
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,wanted in zip(axes.flat,desired):
        ids=[t for t in ys if (t.startswith("RNA_PATHWAY|") and t.endswith(":"+wanted)) or t=="STATE|"+wanted]
        if not ids:
            ax.text(.5,.5,"Not available",ha="center",va="center",transform=ax.transAxes);ax.set_title(wanted,fontsize=9);continue
        tid=ids[0];observed=ys[tid].to_numpy();pred=yp[tid].to_numpy()
        keep=np.isfinite(observed)&np.isfinite(pred)
        stat=m.set_index("target_id").loc[tid]
        ax.scatter(observed[keep],pred[keep],c=folds.to_numpy()[keep],cmap="viridis",s=13,alpha=.65,edgecolors="none",vmin=0,vmax=4)
        if keep.any():
            lo=min(observed[keep].min(),pred[keep].min());hi=max(observed[keep].max(),pred[keep].max())
            ax.plot([lo,hi],[lo,hi],color=".4",ls="--",lw=1)
        ax.set_title("\n".join(textwrap.wrap(wanted.replace("HALLMARK_",""),38)),fontsize=10)
        ax.set_xlabel("Observed native score",fontsize=9);ax.set_ylabel("Held-out prediction",fontsize=9)
        ax.text(.03,.97,f"n={int(keep.sum())}  R2={stat.r2:.2f}\nrho={stat.spearman:.2f}",
                transform=ax.transAxes,va="top",fontsize=9,bbox=dict(facecolor="white",alpha=.7,edgecolor="none"))
    fig.suptitle("M0 | "+label+" | Prespecified examples (colour = outer fold)",fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95]);save(fig,out,case.name+"_examples")

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--results",required=True);ap.add_argument("--output",required=True)
    a=ap.parse_args()
    if socket.gethostname()!="149":raise RuntimeError("Plotting must execute on 149")
    r=Path(a.results);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if not (r/"COMPLETE.json").exists():raise ValueError("M0 training is incomplete")
    (out/"LAUNCH_PREFLIGHT.json").write_text(json.dumps(dict(target_host="149",operation="M0_FIGURES",source_complete_sha256=sha(r/"COMPLETE.json"))))
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
    m=pd.read_parquet(r/"ALL_METRICS.parquet");summary=pd.read_csv(r/"RUN_SUMMARY.tsv",sep="\t")
    group=m.groupby(["dataset","cancer_id","layer"],sort=True)
    coverage=group.agg(target_axis=("target_id","size"),evaluated_targets=("r2","count"),median_r2=("r2","median"),
                       median_spearman=("spearman","median"),median_observed_fraction=("observed_fraction","median"),
                       heldout_values=("n_oof","sum")).reset_index()
    skill=group.skill_vs_train_mean.apply(lambda s:s.dropna().gt(0).mean()).rename("fraction_better_than_training_mean").reset_index()
    coverage=coverage.merge(skill,on=["dataset","cancer_id","layer"])
    coverage.to_csv(out/"M0_PERFORMANCE_SUMMARY.tsv",sep="\t",index=False)
    summary.to_csv(out/"M0_COHORT_COUNTS.tsv",sep="\t",index=False)
    methodmap=m[["dataset","cancer_id","target_id","native_target_id","layer","method"]].copy()
    methodmap=methodmap.rename(columns={"method":"catalog_method"})
    methodmap["native_method"]=methodmap.catalog_method
    if "CPTAC" in set(m.dataset):
        cp=next(r.glob("CPTAC_*/INPUTS.json"))
        src=json.loads(cp.read_text())["sources"]
        statepath=next(Path(v["path"]) for v in src if Path(v["path"]).name=="STATE_PATIENT.parquet")
        alignment=pd.read_csv(statepath.parent/"STATE_ALIGNMENT.tsv",sep="\t").set_index("column")
        take=methodmap.dataset.eq("CPTAC") & methodmap.layer.eq("STATE")
        methodmap.loc[take,"native_method"]=methodmap.loc[take,"native_target_id"].map(alignment.cptac_method)
    methodmap.to_csv(out/"M0_TARGET_METHODS.tsv.gz",sep="\t",index=False,compression="gzip")
    m=m.rename(columns={"method":"catalog_method"}).merge(
        methodmap[["dataset","cancer_id","target_id","native_method"]],
        on=["dataset","cancer_id","target_id"],validate="one_to_one")
    m.to_csv(out/"M0_ALL_TARGET_METRICS.tsv.gz",sep="\t",index=False,compression="gzip")
    for dataset in sorted(m.dataset.unique()):
        sub=coverage.loc[coverage.dataset.eq(dataset)]
        v=sub.pivot(index="cancer_id",columns="layer",values="median_r2").rename(columns=LABELS)
        heatmap(v,"M0 | "+dataset+" | Median target R-squared","Median held-out R-squared",out,dataset+"_overview_R2")
        v=sub.pivot(index="cancer_id",columns="layer",values="fraction_better_than_training_mean").rename(columns=LABELS)
        heatmap(v,"M0 | "+dataset+" | Targets beating the training-mean baseline","Fraction of evaluable targets",
                out,dataset+"_overview_baseline",vmin=0,vmax=1,percent=True)
        state=m.loc[m.dataset.eq(dataset)&m.layer.eq("STATE")]
        v=state.pivot_table(index="cancer_id",columns="family",values="r2",aggfunc="median")
        heatmap(v,"M0 | "+dataset+" | State families","Median held-out R-squared",out,dataset+"_state_families")
    for case in sorted(r.glob("*_*")):
        if case.is_dir() and (case/"METRICS.tsv").exists():
            case_figures(case,out);print("PLOTTED",case.name,flush=True)
    # Shared feature identity is an alignment diagnostic, not a batch correction.
    overlap=[]
    for t in sorted(r.glob("CPTAC_*")):
        ci=json.loads((t/"INPUTS.json").read_text())["info"];tc=r/("TCGA_"+ci["cancer_id"])
        if not tc.exists():continue
        f1=set(pd.read_csv(tc/"FEATURES.tsv",sep="\t").lncrna_id);f2=set(pd.read_csv(t/"FEATURES.tsv",sep="\t").lncrna_id)
        overlap.append(dict(cancer_id=ci["cancer_id"],tcga_lncrna=len(f1),cptac_lncrna=len(f2),common_lncrna=len(f1&f2),
                           tcga_unit="native_edgeR_TMM_logCPM",cptac_unit="log2(TPM+1)",pooling="NOT_VALIDATED"))
    pd.DataFrame(overlap).to_csv(out/"TCGA_CPTAC_FEATURE_ALIGNMENT.tsv",sep="\t",index=False)
    files={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!="FIGURES_COMPLETE.json"}
    (out/"FIGURES_COMPLETE.json").write_text(json.dumps(dict(status="COMPLETE",target_host="149",files=files,
              png_count=len(list(out.glob("*.png"))),pdf_count=len(list(out.glob("*.pdf")))),indent=2))
    print("M0_FIGURES_COMPLETE",flush=True)
if __name__=="__main__":main()
