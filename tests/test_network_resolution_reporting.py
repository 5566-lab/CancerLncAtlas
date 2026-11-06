"""Synthetic paired reports: independent errors, failure retention and identity rejection."""
import importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from cla_engine.common import sha
from scripts import report_network_resolution_validation as fine
from scripts.summarize_network_validation import summarize as old_summary

def fixture_module():
    spec=importlib.util.spec_from_file_location("existing_report_fixture",Path(__file__).with_name("test_network_reporting.py"))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_fine_report_independent_statistics_and_exports(tmp_path):
    fixture=fixture_module();fixture.test_reporting_statistics_and_figures(tmp_path)
    old=tmp_path/"SIMULATED_FITS";inputs=tmp_path/"SIMULATED_INPUTS"
    oldout=tmp_path/"SIMULATED_REPORT";network=tmp_path/"SIMULATED_NETWORKS"
    new=tmp_path/"SIMULATED_FINE_FITS";out=tmp_path/"SIMULATED_FINE_REPORT"
    unavailable="STATE|synthetic_unavailable"
    phases=[f"repeat{r}_outer{f}" for r in range(2) for f in range(5)]+["final"]
    for ci,cancer in enumerate(fine.CANCERS):
        y=pd.read_parquet(inputs/cancer/"Y.parquet");tids=list(y.columns)
        y[unavailable]=np.nan;y.to_parquet(inputs/cancer/"Y.parquet")
        for phase in phases:
            original=old/cancer/phase;folder=new/cancer/phase;folder.mkdir(parents=True)
            meta=pd.read_csv(original/"TARGETS.tsv",sep="\t")
            meta=pd.concat([meta,pd.DataFrame([dict(target_id=unavailable,layer="STATE",family="EMT",method="synthetic",cohort="SIMULATED")])],ignore_index=True)
            meta.to_csv(original/"TARGETS.tsv",sep="\t",index=False);meta.to_csv(folder/"TARGETS.tsv",sep="\t",index=False)
            a=pd.read_parquet(original/"AUDIT.parquet")
            missing=a[a.target_id.eq(tids[0])].copy();missing["target_id"]=unavailable;missing["status"]="NO_OBSERVED_Y"
            a=pd.concat([a,missing],ignore_index=True);a.to_parquet(original/"AUDIT.parquet")
            launch=json.loads((original/"LAUNCH_PREFLIGHT.json").read_text());launch["input_receipt"]="SYNTHETIC_IDENTICAL_INPUT_"+cancer
            (original/"LAUNCH_PREFLIGHT.json").write_text(json.dumps(launch))
            final=phase=="final";repeat=-1 if final else int(phase[6]);fold=-1 if final else int(phase[-1])
            test=[] if final else [i for i in range(30) if (i+repeat)%5==fold]
            train=[i for i in range(30) if i not in test]
            split=dict(repeat=repeat,fold=fold,train=train,test=test,inner=[])
            for seed in range(2):
                for k in range(3):
                    val=[i for j,i in enumerate(train) if (j+seed)%3==k]
                    split["inner"].append(dict(seed=seed,train=[i for i in train if i not in val],validation=val))
            for root in (original,folder):(root/"SPLIT.json").write_text(json.dumps(split))
            fixture.seal(original)
            launch["code"]={**launch["code"],"fine_synthetic_fixture":"true"}
            (folder/"LAUNCH_PREFLIGHT.json").write_text(json.dumps(launch))
            baseline=a[a.variant.eq("ridge_equal")];rows=[];pred=[]
            for mi,variant in enumerate(fine.FINE):
                block=baseline.copy();block["variant"]=variant;block["parameter"]=.5
                if final:block.loc[block.target_id.eq(tids[-1]),"status"]="SYNTHETIC_FINAL_FAILURE"
                rows.append(block)
                for ai,arm in enumerate(fine.ARMS):
                    for ti,tid in enumerate(tids):
                        for i in test:
                            amplitude=.14+.015*mi+.02*ai
                            if ti==3 and mi==0:amplitude=5.
                            sign=-1 if mi%2 else 1
                            error=sign*amplitude*np.cos(i+ci+repeat*.1)
                            pred.append(dict(patient_id=y.index[i],target_id=tid,variant=variant,arm=arm,
                                phase=phase,repeat=repeat,fold=fold,observed=y.loc[y.index[i],tid],
                                predicted=y.loc[y.index[i],tid]+error))
            pd.concat(rows,ignore_index=True).to_parquet(folder/"AUDIT.parquet",index=False)
            pd.DataFrame(pred).to_parquet(folder/"PREDICTIONS.parquet",index=False);fixture.seal(folder)
    old_summary(old,inputs,network,oldout)
    fine.report(old,new,inputs,oldout,out,replicates=50)
    assert len(list(out.glob("*.png")))==len(list(out.glob("*.pdf")))==6
    assert (out/"OPEN_REPORT.html").exists()
    assert len(pd.read_csv(out/"FINE_PHASE_AUDIT.tsv",sep="\t"))==33
    assert len(pd.read_csv(out/"PAIRED_PHASE_IDENTITIES.tsv",sep="\t"))==30
    metrics=pd.read_parquet(out/"FINE_TARGET_METRICS.parquet")
    m=metrics[(metrics.cancer_id=="BRCA")&(metrics.repeat==0)&(metrics.target_id==tids[0])&(metrics.variant==fine.MAIN)&(metrics.arm=="lncrna")].iloc[0]
    y=pd.read_parquet(inputs/"BRCA/Y.parquet")
    expected=1-np.sum((.14*np.cos(np.arange(30)))**2)/np.sum((y[tids[0]]-y[tids[0]].mean())**2)
    np.testing.assert_allclose(m.r2,expected,atol=1e-12)
    assert (metrics[metrics.target_id.eq(tids[3])&metrics.variant.eq(fine.MAIN)].r2<0).all()
    assert not metrics[metrics.target_id.eq(tids[-1])].success.any()
    status=pd.read_parquet(out/"FINE_TRAINING_STATUS.parquet")
    assert len(status[status.target_id.eq(unavailable)])==3*2*18
    assert not status[status.target_id.eq(unavailable)].success.any()
    pair=pd.read_parquet(out/"FINE_PAIRED_MODEL_DELTAS.parquet")
    row=pair[(pair.cancer_id=="BRCA")&(pair.repeat==0)&(pair.target_id==tids[0])&(pair.arm=="lncrna")&(pair.baseline=="ridge_equal")].iloc[0]
    reference=(.27**2-.14**2)*np.sum(np.cos(np.arange(30))**2)/np.sum((y[tids[0]]-y[tids[0]].mean())**2)
    np.testing.assert_allclose(row.delta_r2,reference,atol=1e-12)
    boot=pd.read_csv(out/"FINE_PATIENT_BOOTSTRAP.tsv",sep="\t")
    assert np.isfinite(boot[["family_balanced_mean_delta_r2","ci_low","ci_high"]]).all().all()
    assert (boot.ci_low<=boot.ci_high).all()
    b=boot[(boot.cancer_id=="BRCA")&(boot.layer=="PROTEIN_PATHWAY")&(boot.arm=="lncrna")&(boot.baseline=="mean_five_resolution4_nulls")].iloc[0]
    expected_null=np.mean([(.14+.015*i)**2 for i in range(1,6)])
    cosine=np.mean([np.cos(np.arange(30)+repeat*.1)**2 for repeat in range(2)],axis=0)
    expected_delta=float(np.mean((expected_null-.14**2)*cosine)/np.var(y[tids[0]]))
    np.testing.assert_allclose(b.family_balanced_mean_delta_r2,expected_delta,atol=1e-12)
    averaged_amplitude=np.mean([(-1 if i%2 else 1)*(.14+.015*i) for i in range(1,6)])
    wrong_ensemble=float(np.mean((averaged_amplitude**2-.14**2)*cosine)/np.var(y[tids[0]]))
    assert abs(expected_delta-wrong_ensemble)>1e-3
    # Every report output is bound in the final generated-but-not-reviewed receipt.
    receipt=json.loads((out/"REPORT_READY_FOR_REVIEW.json").read_text())
    assert receipt["status"]=="GENERATED_REVIEW_PENDING"
    assert all(sha(out/name)==digest for name,digest in receipt["files"].items())

def test_fine_pairing_rejects_wrong_labels_missing_patients_and_duplicates(tmp_path):
    original=pd.DataFrame([dict(patient_id=f"p{i}",target_id="t",variant="ridge_equal",arm="lncrna",observed=float(i)) for i in range(3)])
    path=tmp_path/"original.parquet";original.to_parquet(path)
    candidate=original.assign(variant=fine.MAIN)
    assert fine.verify_pairing(candidate,path)["patient_sets_exact"]
    wrong=candidate.copy();wrong.loc[0,"observed"]=99.
    with pytest.raises(ValueError,match="DIFFERENT_OBSERVED_TARGET"):fine.verify_pairing(wrong,path)
    with pytest.raises(ValueError,match="DIFFERENT_PATIENT_SUPPORT"):fine.verify_pairing(candidate.iloc[:2],path)
    with pytest.raises(ValueError,match="DUPLICATE_PAIRED_PREDICTION"):fine.verify_pairing(pd.concat([candidate,candidate.iloc[:1]]),path)
    wrong=candidate.copy();wrong.loc[0,"patient_id"]="unknown"
    with pytest.raises(ValueError,match="UNPAIRED_PATIENT"):fine.verify_pairing(wrong,path)
