"""Export all score-family comparisons, readable figure groups and State heatmaps."""
import argparse,html,json,socket,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0,str(Path(__file__).parent))
from plot_m0_increment import plot,classify,ARMS,LABELS,NAMES,CANCERS
GROUPS={"01_existing_assays":["PROTEIN_PATHWAY","PHOSPHO_PTMSEA","METHYLATION_PATHWAY"],"02_rna_pathway":["RNA_PATHWAY"],"03_tf":["TF"],"04_state_rna":["STATE_RNA"],"05_state_other":["STATE_OTHER"]}
def pairwise_sensitivity(inputs,output):
    # The primary figures use three-arm common successes. This sensitivity
    # report does not require the lncRNA-only model to succeed when comparing
    # mRNA with the combined model; outcomes and held-out patients still match.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from cc_hhgt.m0_three_arm import regression_metrics
    rows=[]
    for source in inputs:
        for cancer in CANCERS:
            d=Path(source)/cancer;status=pd.read_csv(d/"TARGET_TRAINING_STATUS.tsv",sep="\t")
            common=set.intersection(*[set(status.loc[status.arm.eq(a)&status.training_success,"target_id"]) for a in ["mrna","combined"]])
            frames={a:pd.concat([pd.read_parquet(d/f"outer{k}"/a/"PREDICTIONS.parquet") for k in range(5)],ignore_index=True) for a in ["mrna","combined"]}
            keys=["target_id","patient_id","phase","cohort"]
            aa=frames["mrna"].loc[frames["mrna"].target_id.isin(common)].set_index(keys).sort_index()
            bb=frames["combined"].loc[frames["combined"].target_id.isin(common)].set_index(keys).sort_index()
            if not aa.index.equals(bb.index):raise ValueError("Pairwise held-out patients differ")
            if not np.allclose(aa.observed,bb.observed,rtol=1e-10,atol=1e-10):raise ValueError("Pairwise outcomes differ")
            ff=aa[["observed","predicted"]].rename(columns={"predicted":"mrna"});ff["combined"]=bb.predicted;ff=ff.reset_index()
            meta=pd.read_csv(d/"TARGETS.tsv",sep="\t").set_index("target_id")
            for tid,g in ff.groupby("target_id"):
                a=regression_metrics(g.observed,g.mrna);b=regression_metrics(g.observed,g.combined);m=meta.loc[tid]
                rows.append(dict(target_id=tid,cancer_id=cancer,layer=m.layer,family=m.family,n_patients=len(g),r2_mrna=a["r2"],r2_combined=b["r2"],delta_r2=b["r2"]-a["r2"]))
    pair=classify(pd.DataFrame(rows));pair["metric_defined"]=np.isfinite(pair[["r2_mrna","r2_combined"]]).all(axis=1);pair.to_csv(output/"PAIRED_MRNA_COMBINED.tsv",sep="\t",index=False)
    ss=[]
    for layer,g in pair.loc[pair.metric_defined].groupby("display_layer"):
        d=g.delta_r2;ss.append(dict(display_layer=layer,n_targets=len(g),improved=int(d.gt(1e-9).sum()),improved_with_positive_combined_r2=int((d.gt(1e-9)&g.r2_combined.gt(0)).sum()),median_delta=d.median()))
    pd.DataFrame(ss).to_csv(output/"PAIRED_MRNA_COMBINED_SUMMARY.tsv",sep="\t",index=False)

def report(inputs,output):
    assert socket.gethostname()=="149"
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    tab=classify(pd.concat([pd.read_csv(Path(p)/"ALL_TARGET_COMPARISONS.tsv",sep="\t") for p in inputs],ignore_index=True))
    if tab.duplicated(["target_id","cancer_id"]).any():raise ValueError("Duplicate cancer target")
    status=classify(pd.concat([pd.read_csv(Path(p)/"ALL_ARM_TARGET_METRICS.tsv",sep="\t") for p in inputs],ignore_index=True))
    tab.to_csv(output/"ALL_TARGET_COMPARISONS.tsv",sep="\t",index=False)
    status.to_csv(output/"ALL_ARM_TARGET_METRICS.tsv",sep="\t",index=False)
    summary=[]
    for keys,f in tab.groupby(["display_layer","family","cancer_id"],dropna=False):
        z=f.loc[f.complete_fivefold & np.isfinite(f.delta_r2_combined_vs_mrna)]
        summary.append(dict(display_layer=keys[0],family=keys[1],cancer_id=keys[2],listed=len(f),observed=int(f.n_observed_y.gt(0).sum()),common_success=len(z),improved=int(z.delta_r2_combined_vs_mrna.gt(1e-9).sum()),improved_with_positive_combined_r2=int((z.delta_r2_combined_vs_mrna.gt(1e-9)&z.r2_combined.gt(0)).sum()),worse=int(z.delta_r2_combined_vs_mrna.lt(-1e-9).sum()),median_delta=z.delta_r2_combined_vs_mrna.median(),**{"median_r2_"+a:z["r2_"+a].median() for a in ARMS}))
    pd.DataFrame(summary).to_csv(output/"SCORE_FAMILY_SUMMARY.tsv",sep="\t",index=False)
    for name,layers in GROUPS.items():
        plot(inputs,output/name,layers)
    pairwise_sensitivity(inputs,output)
    # Functional State detail excludes TF and cell-composition panels, which
    # have their own tables. No unobserved/failed target is assigned a zero.
    st=tab.loc[tab.layer.eq("STATE") & ~tab.family.isin(["transcription_factor","cell_composition"])].copy()
    st.to_csv(output/"STATE_FUNCTIONAL_TARGETS.tsv",sep="\t",index=False)
    tab.loc[tab.layer.eq("STATE")].to_csv(output/"ALL_STATE_TARGETS.tsv",sep="\t",index=False)
    valid=st.loc[st.complete_fivefold].copy();ids=sorted(valid.target_id.unique())
    detail=output/"06_state_detail";detail.mkdir(exist_ok=True)
    def readable(t):return t.split("|",1)[-1].replace("signifinder::","")
    if ids:
        for n,key,title,lim in [(1,"r2_lncrna","lncRNA",(-.5,.8)),(2,"r2_mrna","mRNA",(-.5,.8)),(3,"r2_combined","lncRNA + mRNA",(-.5,.8)),(4,"delta_r2_combined_vs_mrna","Combined - mRNA",(-.2,.2))]:
            a=valid.pivot(index="target_id",columns="cancer_id",values=key).reindex(index=ids,columns=CANCERS)
            fig,ax=plt.subplots(figsize=(9,max(4,len(ids)*.3)))
            cm=plt.get_cmap("RdBu").copy();cm.set_bad("#e5e5e5")
            im=ax.imshow(a.to_numpy(float),aspect="auto",cmap=cm,norm=TwoSlopeNorm(vmin=lim[0],vcenter=0,vmax=lim[1]))
            ax.set_xticks(range(3),CANCERS);ax.set_yticks(range(len(ids)),[readable(t) for t in ids],fontsize=8)
            for i in range(len(ids)):
                for j in range(3):
                    x=a.iloc[i,j]
                    rgb=cm(im.norm(x))[:3] if np.isfinite(x) else (.9,.9,.9)
                    luminance=.2126*rgb[0]+.7152*rgb[1]+.0722*rgb[2]
                    ax.text(j,i,f"{x:.2f}" if np.isfinite(x) else "NA",ha="center",va="center",fontsize=8,color="black" if luminance>.55 else "white")
            ax.set_title("Functional State | "+title+"\nSame target rows across three arm figures",fontsize=12)
            fig.colorbar(im,ax=ax,extend="both",label="Delta R2" if n==4 else "Held-out R2",fraction=.035,pad=.04)
            fig.text(.5,.005,"Gray = no common successful comparison. Color saturates at limits; numbers retain actual values.",ha="center",fontsize=8)
            fig.tight_layout(rect=(0,.02,1,1))
            for ext in ["png","pdf"]:fig.savefig(detail/(f"0{n}_"+key+"."+ext),dpi=300,bbox_inches="tight")
            plt.close(fig)
    # A single local entry point to all figure sets; no patient data embedded.
    rows=[]
    for name,layers in GROUPS.items():
        label=", ".join(NAMES[x] for x in layers)
        links=" | ".join(f'<a href="{name}/{i:02d}_{slug}.png">{html.escape(text)}</a> (<a href="{name}/{i:02d}_{slug}.pdf">PDF</a>)' for i,slug,text in [(1,"lncrna_performance","lncRNA"),(2,"mrna_performance","mRNA"),(3,"combined_performance","lncRNA + mRNA"),(4,"combined_vs_mrna_comparison","Comparison"),(5,"training_coverage","Training coverage")])
        rows.append(f"<tr><td>{html.escape(label)}</td><td>{links}</td></tr>")
    content="""<!doctype html><meta charset="utf-8"><title>lncRNA incremental prediction</title><style>body{font:16px Arial,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#24313e}table{border-collapse:collapse;width:100%}td,th{padding:14px;border-bottom:1px solid #ddd;text-align:left}a{color:#176294}p{line-height:1.6}</style><h1>lncRNA / mRNA / combined prediction</h1><p>CPTAC BRCA, LUAD and COAD. Five held-out folds with unpenalized clinical and DNA-purity covariates; purity is excluded when predicting itself. No correlation/FDR admission. Zero RNA selected in any fold or final model means training failure.</p><p>Each performance point represents one cancer x target, using the same successful targets in all three arms. Comparison: combined minus mRNA R2. Failed and unavailable targets remain in the downloadable tables. RNA-derived scores measure reconstruction, not independent protein activity or causal regulation.</p><table><tr><th>Score group</th><th>Figures</th></tr>"""+"".join(rows)+"</table><p><a href='ALL_TARGET_COMPARISONS.tsv'>Every target, including failures</a> | <a href='ALL_ARM_TARGET_METRICS.tsv'>Every target-arm status</a> | <a href='SCORE_FAMILY_SUMMARY.tsv'>Score-family summary</a> | <a href='ALL_STATE_TARGETS.tsv'>All State scores</a> | <a href='PAIRED_MRNA_COMBINED_SUMMARY.tsv'>Two-arm sensitivity (lncRNA-only success not required)</a></p><p>Functional State detail: <a href='06_state_detail/01_r2_lncrna.png'>lncRNA</a> | <a href='06_state_detail/02_r2_mrna.png'>mRNA</a> | <a href='06_state_detail/03_r2_combined.png'>Combined</a> | <a href='06_state_detail/04_delta_r2_combined_vs_mrna.png'>Difference</a></p>"
    (output/"OPEN_REPORT.html").write_text(content,encoding="utf-8")
    files={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob("*") if p.is_file() and p.name!="REPORT_COMPLETE.json"}
    (output/"REPORT_COMPLETE.json").write_text(json.dumps(dict(target_host="149",status="COMPLETE",inputs=inputs,files=files),indent=2))
    print(json.dumps(dict(status="COMPLETE",files=len(files),figures_png=len(list(output.rglob("*.png")))),indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--inputs",nargs="+",required=True);p.add_argument("--output",required=True);a=p.parse_args();report(a.inputs,a.output)
