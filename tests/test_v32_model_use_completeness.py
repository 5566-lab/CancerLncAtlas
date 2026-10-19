"""Registered-source coverage and strict metrics on an actual GDC input slice.

This is NOT a complete core/fusion release or independent biological validation.
Archived synthetic API pages and STAR files enter real native assembly/replay;
declared fixture scores exercise the existing isolation and metric consumers.
"""
import copy
import socket

import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import gdc_star_counts as star
from cc_hhgt.v32.current_model_use import Inventory,inventory_tables,read,record
from cc_hhgt.v32.model_use_completeness import assess,is_complete,ENTITY,STUDY
from cc_hhgt.v32.external_validation_sources import read_sources,isolate_sources
from cc_hhgt.v32.external_validation_metrics import evaluate_external
from cc_hhgt.v32.patient_source_identity import ExternalPatientIsolation
from test_v32_gdc_star_counts import inputs
from test_v32_patient_source_identity import crosswalk


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname()=="149"
    def forbidden(*a,**kw):raise AssertionError("Source coverage checks must not fit a model")
    for name in ("Adam","AdamW","SGD"):monkeypatch.setattr(torch.optim,name,forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression,"fit",forbidden)


@pytest.fixture(scope="module")
def covered_slice(tmp_path_factory):
    assert socket.gethostname()=="149"
    root=tmp_path_factory.mktemp("source_coverage")
    query,cache,_=inputs(root/"raw",n=2)
    counts=star.materialize(query,"BRCA",cache,root/"counts")
    native=read(str(counts)+".sources.json")
    samples=pd.read_csv(native["samples"]["path"],sep="\t")
    external=root/"external.parquet"
    lncs=["LNC:NEW_A","LNC:NEW_B","LNC:SEEN","LNC:NEW_C","LNC:NEW_D"]
    raw=pd.DataFrame({"record_id":["assay_"+str(i) for i in range(5)],"lncrna":lncs,"cancer":"BRCA","pathway":"P1","label":[1,0,1,0,0],
        "is_experimental":True,"is_predicted":False,"assay":"synthetic measured outcome",
        "source_dataset":["GSE999999","GSE999999","GSE999999","TCGA-BRCA","GSE888888"],
        "patient_id":["external_"+str(i) for i in range(5)]})
    raw.to_parquet(external,index=False)
    rows=[]
    for row in samples.itertuples():
        for column in ("patient_id","sample_id"):
            rows.append((native["samples"]["sha256"],column,getattr(row,column),row.patient_id))
    for i in range(5):
        canonical=samples.iloc[0].patient_id if i==4 else "distinct_person_"+str(i)
        rows.append((record(external)["sha256"],"patient_id","external_"+str(i),canonical))
    patient_map=crosswalk(root/"patients.parquet",rows)
    inv=Inventory([],root,patient_map)
    star.inspect(inv,{"inputs":{"counts":record(counts)},"counts_assembly":record(str(counts)+".sources.json")},
                 root,"input_slice","expression",0)
    inv.entities(pd.DataFrame({"lncrna_id":["LNC:SEEN"]}),model="input_slice",fold=0,
                 phase="FEATURE_CONSTRUCTION",view="GRAPH_NODE_INPUT",digest=record(counts)["sha256"])
    tables=inventory_tables(inv,["input_slice"])
    checker=ExternalPatientIsolation(patient_map,root,{"patient_identity_namespace":patient_map["namespace"]},tables)
    spec={"kind":"exact_assay","source_id":"measured_fixture","table":record(external),
          "measured_negative_definition":"Synthetic measured zero outcome; no missing outcome converted to zero"}
    maps={"lncrna":{v:v for v in lncs},"cancer":{"BRCA":"BRCA"},"pathway":{"P1":"P1"}}
    observations=read_sources([spec],maps,root,patients=checker)
    scores=pd.DataFrame({"cancer_id":"BRCA","lncrna_id":lncs,"pathway_id":"P1","score":[.9,.1,.8,.4,.3]})
    return inv,tables,observations,scores


def isolate(tables,observations):
    coverage=assess(["input_slice"],tables)
    return isolate_sources(observations,tables["usage"],dependency_complete=is_complete(coverage),
                           dependency_hashes=tables["dependencies"].sha256)


def test_replayed_gdc_boundary_has_positive_coverage_without_claiming_experimental_veracity(covered_slice):
    _,tables,observations,scores=covered_slice
    coverage=assess(["input_slice"],tables)
    assert is_complete(coverage)
    row=coverage.iloc[0]
    assert row.entity_exposure_rows==1 and row.study_source_rows==1
    assert row.blocking_gap_count==0 and row.upstream_veracity_limit_count==1
    assert set(tables["usage"].usage_record_kind)=={ENTITY,STUDY}
    assert observations.patient_overlap.tolist()==[False,False,False,False,True]
    isolated,used=isolate(tables,observations)
    assert isolated.provenance_held_out.tolist()==[True,True,True,False,False]
    assert isolated.pair_held_out.tolist()==[True,True,False,False,False]
    assert isolated.lncrna_held_out.tolist()==[True,True,False,False,False]
    assert isolated.iloc[3].study_overlap and isolated.iloc[4].patient_overlap
    metrics,_,counts=evaluate_external(isolated,used,{"input_slice":scores},primary_model="input_slice",
                                      cutoffs=[1],lncrna_aggregation="MAX",bootstrap_replicates=0,seed=0)
    selected=metrics.loc[metrics.source_group.eq("ALL_SOURCES") & metrics.isolation_scope.eq("lncrna_held_out")].set_index("metric")
    assert selected.loc["auroc","available"] and selected.loc["auroc","value"]==1.
    assert selected.loc["brier","value"]==pytest.approx(.01)
    row=counts.loc[counts.source_group.eq("ALL_SOURCES") & counts.isolation_scope.eq("lncrna_held_out")].iloc[0]
    assert row.matched_targets==2 and row.eligible_negative_targets==1


@pytest.mark.parametrize("fault",["unknown_ancestry","unknown_gap","missing_file","missing_study","no_study_sources",
                                  "losing_comparator","zero_weight_expert","missing_model"])
def test_any_uncovered_dependency_context_keeps_strict_evaluation_empty(covered_slice,fault):
    inv,_,observations,_=covered_slice
    inv=copy.deepcopy(inv)
    if fault in {"unknown_ancestry","unknown_gap"}:
        inv.gap("input_slice","input","RAW_UPSTREAM_TRANSFORMATIONS_NOT_RECONSTRUCTED" if fault=="unknown_ancestry" else "FUTURE_UNKNOWN_GAP")
    elif fault=="missing_file":inv.dependencies[0]["status"]="UNRESOLVED_LOCATION"
    elif fault in {"missing_study","no_study_sources"}:
        if fault=="missing_study":inv.usage[0]["study_tokens_json"]="[]"
        else:inv.usage=[v for v in inv.usage if v.get("usage_record_kind")==ENTITY]
    elif fault in {"losing_comparator","zero_weight_expert"}:
        with inv.context("G012_COMPARISON:G0" if fault=="losing_comparator" else "FUSION_EXPERT:ZERO_WEIGHT"):
            inv.gap("input_slice","input","HASH_BOUND_AUTHORITY_NOT_LOCATED")
    tables=inventory_tables(inv,["input_slice","missing_model"] if fault=="missing_model" else ["input_slice"])
    assert not is_complete(tables["completeness"])
    isolated,_=isolate_sources(observations,tables["usage"],dependency_complete=is_complete(tables["completeness"]))
    assert not isolated.provenance_held_out.any() and isolated.descriptive_eligible.all()
    assert isolated.isolation_reasons_json.str.contains("NATIVE_DEPENDENCIES_INCOMPLETE").all()


@pytest.mark.parametrize("fault",["file_overlap","external_identity","native_mapping"])
def test_positive_coverage_does_not_override_source_or_patient_exclusions(covered_slice,fault):
    _,tables,observations,_=covered_slice
    external=observations.iloc[:2].copy()
    if fault=="file_overlap":external["source_file_sha256"]=tables["dependencies"].iloc[0].sha256
    elif fault=="external_identity":external["external_patient_identity_complete"]=False
    else:external["registered_patient_mapping_complete"]=False
    isolated,_=isolate(tables,external)
    assert isolated.model_usage_provenance_complete.all() and not isolated.provenance_held_out.any()


def test_generic_table_cannot_gain_complete_ancestry_from_file_hash_and_study_tokens(covered_slice,tmp_path):
    from cc_hhgt.v32.current_auxiliary_model_use import input_source
    inv=Inventory([],tmp_path)
    source=tmp_path/"source.parquet"
    pd.DataFrame({"source_dataset":["TCGA-BRCA"]}).to_parquet(source,index=False)
    input_source(inv,record(source),tmp_path,"input_slice","unknown_transform")
    tables=inventory_tables(inv,["input_slice"])
    assert not is_complete(tables["completeness"])
    assert tables["completeness"].iloc[0].unverified_study_source_rows==0
    assert tables["completeness"].iloc[0].blocking_gap_count==1


@pytest.mark.parametrize("fault",["spoof_kind","missing_kind","unknown_kind"])
def test_only_typed_native_entity_rows_are_exempt_from_study_metadata(covered_slice,fault):
    _,original,_,_=covered_slice
    tables={k:v.copy(deep=True) for k,v in original.items()}
    usage=tables["usage"]
    if fault=="spoof_kind":usage.loc[usage.usage_record_kind.eq(STUDY),"usage_record_kind"]=ENTITY
    elif fault=="missing_kind":usage["usage_record_kind"]=STUDY
    else:usage["usage_record_kind"]="UNKNOWN"
    with pytest.raises(ValueError,match="record|exposure"):assess(["input_slice"],tables)
