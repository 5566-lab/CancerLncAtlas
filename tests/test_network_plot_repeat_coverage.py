"""A singleton repeat must not enter a two-repeat figure; negative results remain."""
import json
import numpy as np,pandas as pd,pytest
from cla_engine.common import sha
from scripts.plot_network_validation_complete_repeats import render,both_repeat_means,LAYERS,METHODS,NETWORKS,ARMS
def test_complete_repeat_plot_exports_and_integrity(tmp_path):
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
    render(src,out)
    assert {n:sha(src/n) for n in names}==original
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
