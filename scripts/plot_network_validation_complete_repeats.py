"""Re-render figures05and07 using targets complete in both outer-CV repeats.

This is a plotting correction. Native scores, fitted models, per-repeat metrics,
and failure tables are unchanged. Outputs are written to a new review directory.
"""
from __future__ import annotations
import argparse,json,socket,sys
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cla_engine.common import sha
from scripts.plot_network_validation import save,ARMS,LABELS,COLORS,SHORT
LAYERS=("PROTEIN_PATHWAY","PHOSPHO_PTMSEA")
METHODS=("rna_only","omics_only","ridge_equal","ridge_adaptive_omics",
 "rna_laplacian_ppi","rna_laplacian_rpi","rna_laplacian_union",
 "module_ridge_ppi","module_ridge_rpi","module_ridge_union")
NETWORKS=("module_ridge_ppi","module_ridge_rpi","module_ridge_union")
def dump(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n")
def both_repeat_means(frame,keys,value):
    """Each returned row has exactly one finite observation in repeats0and1."""
    if frame.duplicated(keys+["repeat"]).any():
        raise ValueError("DUPLICATE_REPEAT")
    if not set(frame["repeat"]).issubset({0,1}):
        raise ValueError("UNEXPECTED_REPEAT")
    if not np.isfinite(frame[value].to_numpy(float)).all():
        raise ValueError("NONFINITE_PLOT_VALUE")
    counts=frame.groupby(keys,dropna=False,observed=True)["repeat"].transform("nunique")
    kept=frame.loc[counts.eq(2)]
    return kept.groupby(keys,dropna=False,observed=True)[value].mean().reset_index()
def render(summary,output):
    if socket.gethostname()!="149":
        raise RuntimeError("HOST_149_REQUIRED")
    src,out=Path(summary),Path(output)
    out.mkdir(parents=True,exist_ok=False)
    dump(out/"PREFLIGHT.json",dict(target_host="149",stage="COMPLETE_REPEAT_PLOT_CORRECTION",
         summary=str(src),producer_sha256=sha(Path(__file__)),
         style_helper_sha256=sha(ROOT/"scripts/plot_network_validation.py")))
    receipt=json.loads((src/"SUMMARY_COMPLETE.json").read_text())
    names=("ALL_TARGET_METRICS.parquet","PAIRED_MODEL_DELTAS.parquet")
    for name in names:
        if sha(src/name)!=receipt["files"][name]:
            raise ValueError("SOURCE_CHECKSUM_DRIFT: "+name)
    metrics=pd.read_parquet(src/names[0])
    good=metrics[metrics.success&np.isfinite(metrics.r2)&metrics.display_layer.isin(LAYERS)&metrics.variant.isin(METHODS)]
    metric_keys=["cancer_id","target_id","variant","arm","display_layer"]
    averaged=both_repeat_means(good,metric_keys,"r2")
    paired=pd.read_parquet(src/names[1])
    paired=paired[paired.display_layer.isin(LAYERS)&paired.method.isin(NETWORKS)&paired.baseline.eq("ridge_adaptive_omics")]
    delta_keys=["cancer_id","target_id","arm","display_layer","method","baseline"]
    deltas=both_repeat_means(paired,delta_keys,"delta_r2")
    averaged.to_parquet(out/"FIGURE07_PLOTTED_VALUES.parquet",index=False)
    deltas.to_parquet(out/"FIGURE05_PLOTTED_VALUES.parquet",index=False)
    selection=[]
    for name,frame,keys in (("figure05",paired,delta_keys),("figure07",good,metric_keys)):
        n=frame.groupby(keys,dropna=False,observed=True)["repeat"].nunique()
        selection.append(dict(figure=name,available_model_targets=len(n),
                              both_repeat_model_targets=int(n.eq(2).sum()),
                              single_repeat_model_targets_excluded=int(n.eq(1).sum())))
    pd.DataFrame(selection).to_csv(out/"REPEAT_SELECTION.tsv",sep="\t",index=False)
    fonts={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams.update({"font.family":"Arial" if "Arial" in fonts else "DejaVu Sans",
                         "font.size":11,"axes.titlesize":13,"pdf.fonttype":42,"ps.fonttype":42})
    fig,axs=plt.subplots(2,3,figsize=(15,9),sharey="row")
    rng=np.random.default_rng(42)
    for row,layer in enumerate(LAYERS):
        for ax,arm in zip(axs[row],ARMS):
            for x,method in enumerate(NETWORKS):
                vals=deltas.loc[deltas.arm.eq(arm)&deltas.display_layer.eq(layer)&deltas.method.eq(method),"delta_r2"].to_numpy()
                ax.scatter(x+rng.uniform(-.16,.16,len(vals)),vals,s=6,alpha=.2,color=COLORS[arm],rasterized=True)
                if len(vals):
                    ax.plot([x-.25,x+.25],[np.median(vals)]*2,color="black",lw=2)
            ax.set_xticks(range(3),["PPI","RPI","PPI + RPI"])
            ax.axhline(0,color="gray",ls="--")
            ax.set_yscale("symlog",linthresh=.02)
            ax.set_title(SHORT[layer]+" | "+LABELS[arm])
        axs[row,0].set_ylabel("Delta R2 versus adaptive omics ridge")
    fig.suptitle("Does biological network membership add prediction?\nEach target completed both outer-CV repeats; paired Delta R2 averaged across repeats",fontsize=13)
    save(fig,out,"05_network_ablation")
    fig,axs=plt.subplots(1,2,figsize=(13,8),sharey=True)
    cells_out=[]
    for ax,layer in zip(axs,LAYERS):
        cells=np.full((len(METHODS),len(ARMS)),np.nan)
        for j,method in enumerate(METHODS):
            for k,arm in enumerate(ARMS):
                source_arm="shared" if method=="omics_only" else arm
                vals=averaged.loc[averaged.variant.eq(method)&averaged.display_layer.eq(layer)&averaged.arm.eq(source_arm),"r2"]
                cells[j,k]=vals.median() if len(vals) else np.nan
                cells_out.append(dict(display_layer=layer,variant=method,arm=arm,source_arm=source_arm,
                                      cancer_targets=len(vals),median_r2=cells[j,k]))
        ax.imshow(cells,aspect="auto",cmap="RdBu",vmin=-.2,vmax=.5)
        for j in range(len(METHODS)):
            for k in range(len(ARMS)):
                ax.text(k,j,f"{cells[j,k]:.3f}" if np.isfinite(cells[j,k]) else "NA",ha="center",va="center",fontsize=10)
        ax.set_yticks(range(len(METHODS)),[s.replace("_"," ") for s in METHODS])
        ax.set_xticks(range(len(ARMS)),[LABELS[a] for a in ARMS],rotation=25,ha="right")
        ax.set_title(SHORT[layer])
    fig.suptitle("Median held-out R2 by model and RNA arm\nEach target completed both CV repeats; median of target-level two-repeat means",fontsize=13)
    save(fig,out,"07_model_performance_table")
    pd.DataFrame(cells_out).to_csv(out/"FIGURE07_HEATMAP.tsv",sep="\t",index=False)
    dump(out/"COMPLETE.json",dict(target_host="149",status="GENERATED_REVIEW_PENDING",
         scope="Original figures05and07 only; all raw per-repeat metrics and failure tables retained unchanged",
         source_summary_sha256=sha(src/"SUMMARY_COMPLETE.json"),source_files={n:sha(src/n) for n in names},
         selections=selection,files={f.name:sha(f) for f in out.iterdir() if f.is_file()}))
    return out
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--summary",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    render(args.summary,args.output)
