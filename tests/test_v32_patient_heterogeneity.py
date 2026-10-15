from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.patient_heterogeneity import (HeterogeneityConfig, _design, add_heterogeneity_fdr,
    analyze_patient_heterogeneity, mutation_subgroups)


def fixture(seed=19):
    rng=np.random.default_rng(seed);n=120
    x=rng.normal(size=n);group=np.arange(n)%2;cnv=rng.normal(size=n)
    y=(1-2.5*group)*x+.4*group+.25*cnv+rng.normal(size=n)*(.08+.1*abs(x))
    return pd.DataFrame({"cancer_id":"BRCA","patient_id":[f"TCGA-AA-{i:04}" for i in range(n)],
        "lncrna_id":"L1","pathway_id":"P1","group_id":"MUTATION_L1","patient_fold_id":np.arange(n)%5,
        "lncrna_expression":x,"activity_score":y,"group_value":group.astype(float),"group_available":True,
        "group_unavailable_reason":None,"local_cnv_log2":cnv,"local_cnv_log2__available":True})


def config():return HeterogeneityConfig(covariates=("local_cnv_log2",),min_group_patients=8)


def test_detects_opposite_group_associations_and_retains_individual_oof_rows():
    patients,summary,metrics,audit=analyze_patient_heterogeneity(fixture(),config=config())
    assert patients.prediction_available.all()
    assert summary.group0_slope.iloc[0]==pytest.approx(1,abs=.1)
    assert summary.group1_slope.iloc[0]==pytest.approx(-1.5,abs=.1)
    assert summary.slope_difference_ci_high.iloc[0]<0
    assert summary.interaction_p_value.iloc[0]<1e-12
    assert metrics.delta_mse.gt(0).all()
    assert patients.loc[patients.group_value.eq(0),"conditional_slope_oof"].gt(0).all()
    assert patients.loc[patients.group_value.eq(1),"conditional_slope_oof"].lt(0).all()
    assert not summary.causal_claimed.any() and not summary.subject_specific_effect_claimed.any()
    assert not audit["heldout_outcomes_used_for_own_fit"]


def test_hc3_uncertainty_matches_independent_statsmodels_reference():
    import statsmodels.api as sm
    data=fixture();settings=config()
    _,summary,_,_=analyze_patient_heterogeneity(data,config=settings)
    x,_,transform=_design(data,data,settings.covariates)
    reference=sm.OLS(data.activity_score,x).fit(cov_type="HC3")
    coefficient=reference.params.iloc[3]/transform["expression_sd"]
    se=reference.bse.iloc[3]/transform["expression_sd"]
    assert summary.slope_difference.iloc[0]==pytest.approx(coefficient,rel=1e-10)
    assert summary.slope_difference_se.iloc[0]==pytest.approx(se,rel=1e-10)


def test_heldout_outcomes_cannot_change_own_predictions_or_fit():
    data=fixture();mask=data.patient_fold_id.eq(2)
    original,_,_,audit=analyze_patient_heterogeneity(data,config=config())
    changed=data.copy();changed.loc[mask,"activity_score"]+=1000
    rerun,_,_,other=analyze_patient_heterogeneity(changed,config=config())
    columns=["predicted_activity_oof","nuisance_prediction_oof","association_component_oof","conditional_slope_oof"]
    pd.testing.assert_frame_equal(original.loc[mask,columns],rerun.loc[mask,columns])
    assert audit["folds"][2]==other["folds"][2]


def test_unknown_subgroups_and_missing_outcomes_are_not_silently_zero_filled():
    data=fixture()
    data.loc[0,["group_value","group_available","group_unavailable_reason"]]=[np.nan,False,"NO_LOCUS_COVERAGE"]
    data.loc[1,"activity_score"]=np.nan
    patient,summary,metrics,_=analyze_patient_heterogeneity(data,config=config())
    assert not patient.loc[0,"prediction_available"]
    assert pd.isna(patient.loc[0,"conditional_slope_oof"])
    assert patient.loc[0,"prediction_unavailable_reason"]=="NO_LOCUS_COVERAGE"
    assert patient.loc[1,"prediction_available"]
    assert pd.isna(patient.loc[1,"prediction_residual"])
    assert metrics.evaluated_patients.sum()==118
    assert summary.group_unknown_patients.iloc[0]==1


def test_wes_no_report_is_unknown_without_locus_coverage():
    calls=pd.DataFrame({"cancer_id":"BRCA","patient_id":["P1","P2","P3","P4"],"lncrna_id":"L1",
        "reported_variant_count":[1,0,0,np.nan],"mutation_assay_available":[True,True,True,False],
        "locus_callable":[False,False,True,False]})
    result=mutation_subgroups(calls,group_id="MUTATION_L1")
    assert result.group_value.iloc[0]==1
    assert pd.isna(result.group_value.iloc[1])
    assert result.group_value.iloc[2]==0
    assert pd.isna(result.group_value.iloc[3])
    assert result.group_available.tolist()==[True,False,True,False]
    with pytest.raises(ValueError,match="contradiction"):
        mutation_subgroups(calls.assign(locus_callable=True),group_id="MUTATION_L1")


def test_sparse_group_is_typed_unavailable_with_no_fake_effect():
    data=fixture();data["group_value"]=0.;data.loc[0,"group_value"]=1.
    patient,summary,_,_=analyze_patient_heterogeneity(data,config=config())
    assert not patient.prediction_available.any()
    assert not summary.inference_available.any()
    assert summary.slope_difference.isna().all()
    assert summary.interaction_p_value.isna().all()


def test_fdr_uses_full_declared_family_including_unavailable_tests():
    summary=pd.DataFrame({"cancer_id":"BRCA","group_id":"G","interaction_p_value":[.01,.04,np.nan],
                          "inference_available":[True,True,False]})
    result=add_heterogeneity_fdr(summary)
    np.testing.assert_allclose(result.interaction_fdr.iloc[:2],[.03,.06])
    assert pd.isna(result.interaction_fdr.iloc[2])


@pytest.mark.parametrize("fault",["duplicate","boolean","fold","covariate"])
def test_bad_patient_authority_fails_closed(fault):
    data=fixture()
    if fault=="duplicate":data=pd.concat([data,data.iloc[[0]]])
    elif fault=="boolean":data["group_available"]="False"
    elif fault=="fold":
        data["patient_fold_id"]=data.patient_fold_id.astype(float)
        data.loc[0,"patient_fold_id"]=1.5
    else:data.loc[0,"local_cnv_log2__available"]=False
    with pytest.raises(ValueError):analyze_patient_heterogeneity(data,config=config())


def test_cli_streams_patient_results_and_all_candidate_inference(tmp_path):
    from scripts.analyze_v32_patient_heterogeneity import main
    data=fixture();keys=["cancer_id","patient_id"]
    tables={"expression":data[keys+["lncrna_id","lncrna_expression"]],
        "activity":data[keys+["pathway_id","activity_score"]],
        "groups":data[keys+["lncrna_id","group_id","group_value","group_available","group_unavailable_reason"]],
        "covariates":data[keys+["lncrna_id","local_cnv_log2","local_cnv_log2__available"]],
        "folds":data[keys+["patient_fold_id"]],
        "candidates":pd.DataFrame({"cancer_id":["BRCA","BRCA"],"lncrna_id":["L1","L2"],"pathway_id":["P1","P1"]})}
    args=[]
    for label,table in tables.items():
        path=tmp_path/f"{label}.parquet";table.to_parquet(path,index=False);args += ["--"+label,str(path)]
    args += ["--group-id","MUTATION_L1","--nuisance-columns","local_cnv_log2","--candidate-selection","prespecified_independent",
             "--output",str(tmp_path/"result")]
    assert main(args)==0
    summary=pd.read_parquet(tmp_path/"result/INTERACTION_SUMMARY.parquet")
    assert len(summary)==2 and summary.inference_available.tolist()==[True,False]
    patients=pd.read_parquet(tmp_path/"result/patient_oof")
    assert len(patients)==240
    assert not patients.loc[patients.lncrna_id.eq("L2"),"prediction_available"].any()
    audit=json.loads((tmp_path/"result/AUDIT.json").read_text())
    assert audit["patient_prediction_rows"]==240 and not audit["training_started"]
