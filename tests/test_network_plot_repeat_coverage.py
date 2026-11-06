"""A singleton repeat must not enter a two-repeat figure; negative results remain."""
import json
import numpy as np,pandas as pd,pytest
from cla_engine.common import sha
from scripts.plot_network_validation_complete_repeats import render,both_repeat_means,LAYERS,METHODS,NETWORKS,ARMS
def repeat_fixture(tmp_path):
    src=tmp_path/"SIMULATED_SUMMARY";out=tmp_path/"SIMULATED_REVIEWED_PLOTS";src.mkdir()
    metrics=[];deltas=[]
    for cancer in ("BRCA","LUAD","COAD"):
        for layer in LAYERS:
            for method in METHODS:
                for arm in (("shared",) if method=="omics_only" else ARMS):
                    for target,values,success in (("negative",[-.4,-.2],True),("positive",[.2,.4],True),
                                                  ("singleton",[.99],True),("failed",[.8,.9],False)):
                        for repeat,r2 in enumerate(values):
                            metrics.append(dict(cancer_id=cancer,target_id=layer+"|"+target,variant=method,arm=arm,display_layer=layer,repeat=repeat,r2=r2,success=success))
            for method in NETWORKS:
                for arm in ARMS:
                    for target,values in (("negative",[-.5,-.3]),("positive",[.1,.3]),("singleton",[.99])):
                        for repeat,delta in enumerate(values):
                            deltas.append(dict(cancer_id=cancer,target_id=layer+"|"+target,method=method,baseline="ridge_adaptive_omics",arm=arm,display_layer=layer,repeat=repeat,delta_r2=delta))
    m=pd.DataFrame(metrics);d=pd.DataFrame(deltas)
    m.to_parquet(src/"ALL_TARGET_METRICS.parquet",index=False);d.to_parquet(src/"PAIRED_MODEL_DELTAS.parquet",index=False)
    names=("ALL_TARGET_METRICS.parquet","PAIRED_MODEL_DELTAS.parquet")
    original={n:sha(src/n) for n in names}
    (src/"SUMMARY_COMPLETE.json").write_text(json.dumps({"files":original}))
    return src,out,m,d,original

def test_complete_repeat_plot_exports_and_integrity(tmp_path):
    src,out,m,d,original=repeat_fixture(tmp_path)
    render(src,out)
    assert {n:sha(src/n) for n in original}==original
    a=pd.read_parquet(out/"FIGURE07_PLOTTED_VALUES.parquet")
    b=pd.read_parquet(out/"FIGURE05_PLOTTED_VALUES.parquet")
    assert not a.target_id.str.endswith(("singleton","failed")).any()
    assert not b.target_id.str.endswith("singleton").any()
    np.testing.assert_allclose(a.loc[a.target_id.str.endswith("negative"),"r2"],-.3)
    np.testing.assert_allclose(b.loc[b.target_id.str.endswith("negative"),"delta_r2"],-.4)
    heat=pd.read_csv(out/"FIGURE07_HEATMAP.tsv",sep="\t")
    assert heat.cancer_targets.eq(6).all()
    np.testing.assert_allclose(heat.median_r2,0,atol=1e-14)
    sel=pd.read_csv(out/"REPEAT_SELECTION.tsv",sep="\t").set_index("figure")
    assert sel.loc["figure05","single_repeat_model_targets_excluded"]==54
    assert sel.loc["figure07","single_repeat_model_targets_excluded"]==168
    assert len(list(out.glob("*.png")))==2 and len(list(out.glob("*.pdf")))==2
    receipt=json.loads((out/"COMPLETE.json").read_text())
    assert receipt["status"]=="GENERATED_REVIEW_PENDING"
    assert all(sha(out/n)==h for n,h in receipt["files"].items())
    keys=["cancer_id","target_id","variant","arm","display_layer"]
    good=m[m.success].copy()
    with pytest.raises(ValueError,match="DUPLICATE_REPEAT"):
        both_repeat_means(pd.concat([good,good.iloc[[0]]]),keys,"r2")
    bad=good.copy();bad.loc[0,"repeat"]=2
    with pytest.raises(ValueError,match="UNEXPECTED_REPEAT"):
        both_repeat_means(bad,keys,"r2")
    bad=good.copy();bad.loc[0,"r2"]=np.nan
    with pytest.raises(ValueError,match="NONFINITE_PLOT_VALUE"):
        both_repeat_means(bad,keys,"r2")
    (src/"PAIRED_MODEL_DELTAS.parquet").write_bytes(b"corrupt source")
    with pytest.raises(ValueError,match="SOURCE_CHECKSUM_DRIFT"):
        render(src,tmp_path/"CORRUPT_SOURCE_REJECTED")

def test_default_plot_entry_requires_both_repeats(tmp_path,monkeypatch):
    """Inspect actual figure data at the normal entry point, not just its helper."""
    from scripts import plot_network_validation as canonical
    src,out,m,d,original=repeat_fixture(tmp_path)
    (src/"BOOTSTRAP_COMPLETE.json").write_text(json.dumps({"synthetic_fixture":True}))
    pd.DataFrame([dict(cancer_id=c,target_id=layer+"|negative",variant="module_ridge_union",
                      display_layer=layer,repeat=r,combined_minus_mrna=0.)
                  for c in canonical.CANCERS for layer in LAYERS for r in (0,1)]).to_parquet(src/"THREE_ARM_COMPARISONS.parquet",index=False)
    pd.DataFrame(columns=["cancer_id","layer","arm","method","baseline","ci_low","ci_high",
                          "family_balanced_mean_delta_r2"]).to_csv(src/"PATIENT_BOOTSTRAP.tsv",sep="\t",index=False)
    labs=[("RNA","lncrna"),("RNA","mrna"),("CNV","shared"),("MUTATION","shared"),("PROTEIN","shared"),("PHOSPHO","shared")]
    pd.DataFrame([dict(cancer_id=c,block=b,arm=a,mode=mode,fraction=.5)
                  for c in canonical.CANCERS for mode in ("ppi","rpi","union") for b,a in labs]).to_csv(src/"NETWORK_FEATURE_COVERAGE.tsv",sep="\t",index=False)
    methods=["ridge_adaptive_omics","module_ridge_ppi","module_ridge_rpi","module_ridge_union",
             "rna_laplacian_ppi","rna_laplacian_rpi","rna_laplacian_union"]
    pd.DataFrame([dict(phase="repeat0_outer0",variant=v,arm=a,parameter=0.,fits=1)
                  for v in methods for a in ARMS]).to_csv(src/"HYPERPARAMETERS.tsv",sep="\t",index=False)
    pd.DataFrame([dict(cancer_id=c,display_layer=l,variant="module_ridge_union",arm="combined",attempted=4,successful=2)
                  for c in canonical.CANCERS for l in canonical.LAYERS]).to_csv(src/"COVERAGE_SUMMARY.tsv",sep="\t",index=False)
    pd.DataFrame([dict(variant="module_ridge_union",synthetic=True)]).to_csv(src/"THREE_ARM_SUMMARY.tsv",sep="\t",index=False)
    observed={}
    real_save=canonical.save
    def inspect_and_save(fig,folder,name):
        if name=="05_network_ablation":
            assert len(fig.axes)==6
            for ax in fig.axes:
                assert len(ax.collections)==3
                for points in ax.collections:
                    values=np.asarray(points.get_offsets()[:,1],float)
                    np.testing.assert_allclose(np.sort(values),[-.4,-.4,-.4,.2,.2,.2],atol=1e-14)
            observed[name]=True
        elif name=="07_model_performance_table":
            assert len(fig.axes)==2
            for ax in fig.axes:
                assert len(ax.images)==1
                cells=np.asarray(ax.images[0].get_array(),float)
                assert cells.shape==(10,3)
                np.testing.assert_allclose(cells,0.,atol=1e-14)
            observed[name]=True
        if name in observed:real_save(fig,folder,name)
        else:canonical.plt.close(fig)
    monkeypatch.setattr(canonical,"save",inspect_and_save)
    canonical.plot(src,tmp_path/"SIMULATED_TRAINING")
    assert set(observed)=={"05_network_ablation","07_model_performance_table"}
    assert len(list(src.glob("*.png")))==len(list(src.glob("*.pdf")))==2
    assert {n:sha(src/n) for n in original}==original
    html=(src/"OPEN_REPORT.html").read_text()
    assert "Every point in the two-repeat" in html
    assert "Module model:" in html and "Improved model:" not in html
