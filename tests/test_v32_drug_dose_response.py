"""Observed dose/response semantics in the existing analysis entry, without fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import drug_training as base
from cc_hhgt.v32.drug_dose_response import normalize_point_response, condition_inputs, iter_dose_response_scan, dose_key
from cc_hhgt.v32.drug_response_analysis import DrugResponseScanConfig
from cc_hhgt.v32.drug_staging import artifact_sha256


@pytest.fixture(autouse=True)
def forbid_fitting(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError("Dose analysis must not optimize")
    for name in ("Adam","AdamW","SGD"): monkeypatch.setattr(torch.optim,name,forbidden)
    monkeypatch.setattr(base,"_fit_head",forbidden)


def fixture():
    models=[f"M{i}" for i in range(40)]
    mapping=pd.DataFrame([{"dataset_id":d,"cell_line_id":f"{d}_{m}","canonical_model_id":m,"cancer_id":"BRCA"}
        for d in ("GDSC","PRISM") for m in models])
    expression=pd.DataFrame({"canonical_model_id":models,"lncrna_id":"L1","expression_value":np.arange(40)%20})
    subtypes=pd.DataFrame({"canonical_model_id":models,"cancer_id":"BRCA","subtype_system":"PAM50",
        "subtype_id":["Basal"]*20+["LumA"]*20,"subtype_available":True,"source":"synthetic"})
    rows=[]
    for d in ("GDSC","PRISM"):
        for i,m in enumerate(models):
            x=(i%20)/20
            for high in (False,True):
                sensitivity=(.8 if i<20 else -.8)*x if high else .1*x
                rows.append({"dataset_id":d,"cell_line_id":f"{d}_{m}","drug_id":"ASSAY:unmapped",
                    "assay_id":f"{m}-{high}","dose_value":(100. if d=="GDSC" else .1) if high else 0.,
                    "dose_unit":"nM" if d=="GDSC" else "uM","exposure_hours":72.,"response_value":1-sensitivity,
                    "higher_is_sensitive":False,"response_metric":"POINT_VIABILITY","response_unit":"fraction","source_id":"synthetic"})
    return pd.DataFrame(rows),dict(mapping=mapping,expression=expression,subtypes=subtypes,
        exact=pd.DataFrame({"cancer_id":["BRCA","BRCA","KICH"],"lncrna_id":["L1","Lmissing","L1"]}),
        targets=pd.DataFrame({"drug_id":["DC:1"],"gene_id":["G1"]}))


def scans(raw,tables):
    points,audit=normalize_point_response(raw,tables["mapping"])
    rows=list(iter_dose_response_scan(points,**tables,config=DrugResponseScanConfig(
        min_total_cell_lines=4,min_cell_lines_per_dataset=4,candidate_chunk_rows=1)))
    return points,audit,rows


def results(rows,kind):
    return pd.concat([frame for group,_,frame in rows if group==kind and not frame.empty],ignore_index=True)


def test_exact_dose_and_exposure_keep_signed_subtypes_and_paired_model_changes():
    raw,tables=fixture(); points,_,rows=scans(raw,tables)
    assert set(points.dose_molar_key)=={"0","0.0000001"}
    assert dose_key(.1,"uM")==dose_key(100.,"nM") and dose_key(0,"nM")=="0"
    assert dose_key(None,"nM") is None
    point=results(rows,"dose_associations")
    high=point.loc[point.dose_molar.gt(0)&point.lncrna_id.eq("L1")&point.subtype_system.eq("PAM50")].set_index("subtype_id")
    assert high.loc["Basal","signed_expression_sensitivity_rho"]==pytest.approx(1.,abs=1e-6)
    assert high.loc["LumA","signed_expression_sensitivity_rho"]==pytest.approx(-1.,abs=1e-6)
    assert high.unique_cell_models.eq(20).all() and high.contributing_datasets.eq(2).all()
    assert not high.target_annotation_available.any()
    change=results(rows,"dose_contrasts")
    local=change.loc[change.lncrna_id.eq("L1")&change.subtype_system.eq("PAM50")].set_index("subtype_id")
    assert local.loc["Basal","signed_expression_sensitivity_rho"]==pytest.approx(1.,abs=1e-6)
    assert local.loc["LumA","signed_expression_sensitivity_rho"]==pytest.approx(-1.,abs=1e-6)
    assert local.low_dose_molar.eq(0).all() and local.high_dose_molar.eq(1e-7).all()
    assert local.unique_cell_models.eq(20).all()
    assert local.observed_direction.str.contains("SENSITIVITY_INCREASE").all()
    assert not change.fitted_dose_response_curve.any() and not change.dose_interaction_causal_claimed.any()
    assert change.loc[change.lncrna_id.eq("Lmissing"),"signed_expression_sensitivity_rho"].isna().all()


def test_exposure_and_endpoint_units_are_not_pooled_or_paired():
    raw,tables=fixture()
    extra=raw.assign(exposure_hours=24.,assay_id=raw.assay_id+"-24",response_value=-raw.response_value)
    normalized,_=normalize_point_response(pd.concat([raw,extra],ignore_index=True),tables["mapping"])
    contexts=list(condition_inputs(normalized,tables["mapping"]))
    assert len([x for x in contexts if x[0]=="dose_associations"])==4
    assert {x[1]["exposure_hours"] for x in contexts if x[0]=="dose_contrasts"}=={24.,72.}
    assert {x[1]["exposure_hours_key"] for x in contexts}=={"24","72"}
    raw.loc[raw.dose_value.gt(0),"response_unit"]="percent"
    points,_=normalize_point_response(raw,tables["mapping"])
    assert all(kind!="dose_contrasts" for kind,*_ in condition_inputs(points,tables["mapping"]))


@pytest.mark.parametrize("split",["dataset","models"])
def test_contrasts_require_the_same_models_within_the_same_dataset(split):
    raw,tables=fixture()
    if split=="dataset":
        keep=(raw.dataset_id.eq("GDSC")&raw.dose_value.eq(0))|(raw.dataset_id.eq("PRISM")&raw.dose_value.gt(0))
    else:
        first=raw.cell_line_id.str.extract(r"M(\d+)")[0].astype(int).lt(20)
        keep=(first&raw.dose_value.eq(0))|(~first&raw.dose_value.gt(0))
    points,_,rows=scans(raw.loc[keep].copy(),tables)
    contrast=next(response for kind,_,response,_ in condition_inputs(points,tables["mapping"]) if kind=="dose_contrasts")
    assert contrast.empty
    frame=results(rows,"dose_contrasts")
    assert not frame.association_available.any() and frame.signed_expression_sensitivity_rho.isna().all()
    assert frame.unique_cell_models.eq(0).all()


def test_duplicates_do_not_inflate_model_support_or_reweight_one_assay():
    raw,tables=fixture(); repeated=pd.concat([raw,*[raw.iloc[:10]]*7],ignore_index=True)
    one,_=normalize_point_response(raw,tables["mapping"])
    two,disposition=normalize_point_response(repeated,tables["mapping"])
    assert len(two)==len(one) and len(disposition)==len(repeated)
    originals=list(condition_inputs(one,tables["mapping"])); repeats=list(condition_inputs(two,tables["mapping"]))
    for a,b in zip(originals,repeats): pd.testing.assert_frame_equal(a[2],b[2])


def test_observed_zero_dose_change_is_not_missing_or_a_zero_correlation():
    raw,tables=fixture()
    for dataset in ("GDSC","PRISM"):
        selected=raw.dataset_id.eq(dataset)
        low=raw.loc[selected&raw.dose_value.eq(0)].set_index("cell_line_id").response_value
        high=selected&raw.dose_value.gt(0)
        raw.loc[high,"response_value"]=raw.loc[high,"cell_line_id"].map(low)
    _,_,rows=scans(raw,tables)
    frame=results(rows,"dose_contrasts")
    assert frame.measured_response_available.all() and frame.mean_observed_oriented_response.eq(0).all()
    assert not frame.within_dataset_response_variation.any()
    assert not frame.association_available.any() and frame.signed_expression_sensitivity_rho.isna().all()
    assert frame.loc[frame.subtype_id.eq("ALL"),"measured_unique_cell_models"].eq(40).all()


@pytest.mark.parametrize("fault",["summary_metric","negative_dose","unknown_unit","conflicting_assay","orientation"])
def test_summary_metrics_invalid_units_and_conflicting_native_assays_are_rejected(fault):
    raw,tables=fixture()
    if fault=="summary_metric": raw["response_metric"]="AUC"
    if fault=="negative_dose": raw.loc[0,"dose_value"]=-1
    if fault=="unknown_unit": raw.loc[0,"dose_unit"]="mg/kg"
    if fault=="conflicting_assay": raw=pd.concat([raw,raw.iloc[[0]].assign(response_value=.123)],ignore_index=True)
    if fault=="orientation": raw.loc[0,"higher_is_sensitive"]=True
    with pytest.raises(ValueError): normalize_point_response(raw,tables["mapping"])


def test_missing_metadata_and_all_missing_measurements_remain_explicit():
    raw,tables=fixture()
    raw.loc[0,["dose_value","dose_unit"]]=[None,None]
    raw.loc[1,"exposure_hours"]=None
    raw.loc[raw.dose_value.gt(0),"response_value"]=np.nan
    points,disposition,rows=scans(raw,tables)
    assert disposition.measurement_unavailable_reason.iloc[0]=="MISSING_POINT_DOSE"
    assert disposition.measurement_unavailable_reason.iloc[1]=="MISSING_EXPOSURE_DURATION"
    high=results(rows,"dose_associations").loc[lambda t:t.dose_molar.gt(0)]
    assert not high.empty and high.signed_expression_sensitivity_rho.isna().all()
    assert not high.association_available.any()
    zero=raw.iloc[[2]].assign(dose_value=0,dose_unit="nM",exposure_hours=0)
    normalized,_=normalize_point_response(zero,tables["mapping"])
    assert normalized.measurement_available.all() and normalized.exposure_hours_key.eq("0").all()


def test_existing_cli_consumes_actual_native_point_sources_and_binds_staging(tmp_path):
    from test_v32_drug_identity import native_staging_fixture
    from scripts.analyze_v32_native_drug_response import main
    staging,_=native_staging_fixture(tmp_path)
    points=[]
    for i in range(5):
        for dose in (0,100):
            points.append({"model":f"SIDM{i}","drug":"17","well":f"w{i}-{dose}","dose":dose,
                "viability":1-(i/20 if dose==0 else i/10)})
    points.append({"model":"SIDM0","drug":"unknown-id","well":"bad","dose":1,"viability":.1})
    source=tmp_path/"native_points.csv"; pd.DataFrame(points).to_csv(source,index=False)
    record=lambda p:{"path":str(p),"sha256":artifact_sha256(p)}
    manifest={"format":"NATIVE_DRUG_POINT_RESPONSE_V1","staging_manifest":record(staging/"STAGING_MANIFEST.json"),
        "measurements":[{"source":record(source),"condition_kind":"POINT_CONCENTRATION","measurement_level":"OBSERVED_POINT_RESPONSE",
            "columns":{"cell_line_id":"model","source_drug_id":"drug","assay_id":"well","dose_value":"dose","response_value":"viability"},
            "constants":{"dataset_id":"GDSC1_AUC","dose_unit":"nM","exposure_hours":72,"higher_is_sensitive":False,
                         "response_metric":"POINT_VIABILITY","response_unit":"fraction"}}]}
    path=tmp_path/"point_manifest.json"; path.write_text(json.dumps(manifest))
    inputs={"exact":tmp_path/"exact.parquet","response":staging/"raw_cell_line_drug_response",
        "expression":staging/"raw_cell_line_lncrna_expression.parquet","mapping":staging/"cell_line_map.parquet",
        "targets":staging/"drug_gene_target.static.parquet","subtypes":staging/"MODEL_SUBTYPES.parquet"}
    args=[value for name,p in inputs.items() for value in ("--"+name,str(p))]
    output=tmp_path/"scan"
    assert main([*args,"--output",str(output),"--dose-manifest",str(path),"--min-models","4","--min-models-per-dataset","4"])==0
    audit=json.loads((output/"AUDIT.json").read_text()); dose=audit["dose_response_analysis"]
    assert dose["available_point_rows"]==10 and dose["native_rows_without_staged_identity"]==1
    assert not dose["training_started"] and not dose["curve_fitting_performed"]
    delta=pd.read_parquet(output/"dose_contrasts")
    actual=delta.loc[delta.cancer_id.eq("BRCA")&delta.subtype_id.eq("ALL")].iloc[0]
    # The existing single-dataset Fisher path clips a perfect rho at 0.999999.
    assert actual.unique_cell_models==5 and actual.signed_expression_sensitivity_rho==np.float32(.999999)
    assert dose["sources"]["measurement_0"]["sha256"]==artifact_sha256(source)
    assert audit["outputs"]["PRIVATE_DOSE_MEASUREMENT_DISPOSITION.parquet"]["public_download_eligible"] is False
    public=pd.read_parquet(output/"dose_associations")
    assert not {"cell_line_id","canonical_model_id","response_value","source_id"}&set(public)
    # Renaming a fitted source column through a point declaration is rejected.
    native=pd.read_csv(source); native["AUC"]=.5; native.to_csv(source,index=False)
    bad=json.loads(json.dumps(manifest)); bad["measurements"][0]["source"]=record(source)
    bad["measurements"][0]["columns"]["response_value"]="AUC"
    bad_path=tmp_path/"summary_as_points.json"; bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match="summary metric"):
        main([*args,"--output",str(tmp_path/"summary_rejected"),"--dose-manifest",str(bad_path)])
    # A changed native source cannot be silently reused through the same manifest.
    source.write_text(source.read_text()+"\n")
    with pytest.raises(ValueError,match="SHA256"):
        main([*args,"--output",str(tmp_path/"changed"),"--dose-manifest",str(path)])
    assert not (tmp_path/"changed/SUCCESS.json").exists()
