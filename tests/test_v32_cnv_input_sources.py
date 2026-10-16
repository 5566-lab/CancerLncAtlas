"""Actual CNV readers -> native adapter -> fusion -> isolation, without fits."""
import json
from pathlib import Path
import socket

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32.cnv_input_sources import SourceCapture, ARRAYS, validate_consumed_sources
from cc_hhgt.v32.cnv_only_training import CompactCNV, write_cnv_event_oof_manifest, _canonical_patient
from cc_hhgt.v32.cnv_only_training_v2 import DirectionalCompactCNV, _formal_labels, EXPECTED_ASSOCIATION_RUN_ID
from cc_hhgt.v32.current_model_use import read, write, record, derive_manifest, materialize_current_model_use
from cc_hhgt.v32.native_auxiliary_fusion import Sources, native_sources
from cc_hhgt.v32.current_core_export import sha256
from test_v32_native_auxiliary_fusion import fixture, repin_native
from test_v32_auxiliary_model_use import setup


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    assert socket.gethostname() == "149"
    def forbidden(*a, **kw):raise AssertionError("CNV source checks must not fit or predict")
    for name in ("Adam", "AdamW", "SGD"):monkeypatch.setattr(torch.optim, name, forbidden)
    import cc_hhgt.v32.cnv_only_training as event
    import cc_hhgt.v32.cnv_only_training_v2 as directional
    for module in (event, directional):
        monkeypatch.setattr(module, "_fit_head", forbidden)
        monkeypatch.setattr(module, "_predict_head", forbidden)


def attach_consumed(path):
    manifest=read(path);kind=manifest["producer_kind"];root=path.parent
    native_path=Path(manifest["native_receipt"]["path"]);native=read(native_path)
    capture=SourceCapture()
    candidates=capture.read(manifest["candidate_authority"]["path"],pd.read_parquet,role="candidates",
                            columns=["cancer_id","lncrna_id","pathway_id"])
    folds=capture.read(manifest["patient_folds"]["path"],pd.read_parquet,role="folds")
    capture.json(manifest["core_manifest"]["path"],role="core")
    raw=root/"segments.tsv"
    pd.DataFrame({"pmid":["33445566"],"dataset_accession":["GSE_SYNTHETIC_CNV"],
                  "patient_id":["PRIVATE_CNV_DONOR"],"value":[.2]}).to_csv(raw,sep="\t",index=False)
    aggregate={"status":"SUCCESS","cancers":{}}
    for cancer,local in candidates.groupby("cancer_id"):
        partition=root/f"compact_{cancer}";partition.mkdir()
        ids={"patient_ids":folds.loc[folds.cancer_id.eq(cancer),"patient_id"].map(_canonical_patient).tolist(),
             "lncrna_ids":local.lncrna_id.unique().tolist(),"pathway_ids":local.pathway_id.unique().tolist()}
        for name,value in ids.items():write(partition/(name+".json"),value)
        for name in ARRAYS[kind]:
            width=len(ids["lncrna_ids" if name.startswith("lncrna_") else "pathway_ids"])
            np.save(partition/name,np.ones((len(ids["patient_ids"]),width),np.float32))
        write(partition/"SOURCE_FILES.json",{"PRIVATE_CNV_DONOR_KEY":[record(raw)]})
        success={"status":"SUCCESS","signed_values_preserved":True,"amplification_deletion_separate":True,
                 "absolute_burden_used_as_continuous":False,
                 "files":{p.name:sha256(p) for p in partition.iterdir()}}
        write(partition/"SUCCESS.json",success)
        aggregate["cancers"][cancer]={"path":str(partition),"success_sha256":sha256(partition/"SUCCESS.json")}
        cls=DirectionalCompactCNV if kind=="cnv" else CompactCNV
        cls(partition,capture=capture,cancer=cancer)
    success_path=root/"COMPACT_SUCCESS.json";write(success_path,aggregate)
    capture.json(success_path,role="materialization_success")
    if kind=="cnv":
        association=root/"association";association.mkdir()
        write(association/"SUCCESS.json",{"status":"SUCCESS"})
        capture.json(association/"SUCCESS.json",role="association_success")
        for cancer,local in candidates.groupby("cancer_id"):
            for fold in range(5):
                target=association/"pair_level"/f"cancer={cancer}"/f"fold={fold}"/"part.parquet"
                target.parent.mkdir(parents=True)
                local[["lncrna_id","pathway_id"]].assign(label_A0_full="unlabeled",label_A0_validation="unavailable",
                    run_id=EXPECTED_ASSOCIATION_RUN_ID).to_parquet(target,index=False)
                for split in ("train","validation"):
                    _formal_labels(association,cancer,fold,local,split,capture=capture)
    consumed=capture.write(root/"CONSUMED_SOURCES.json",kind=kind,run_id=manifest["native_run_id"])
    if kind=="cnv_event":
        native_path.unlink()  # Temporary synthetic fixture; exercise actual producer writer.
        write_cnv_event_oof_manifest(native_path.parent,training_run_id=manifest["native_run_id"],
            checkpoints=native["checkpoints"],predictions=native["predictions"],
            core_manifest_sha=manifest["core_manifest"]["sha256"],core_composite="a"*64,
            core_hashes_before=native["core_parameters_before_sha256"],core_hashes_after=native["core_parameters_after_sha256"],
            consumed_sources=consumed)
    else:
        native["consumed_sources"]=consumed;write(native_path,native)
    repin_native(path)


@pytest.mark.parametrize("kind",["cnv","cnv_event"])
def test_consumed_cnv_sources_reach_external_study_isolation(tmp_path,kind):
    from cc_hhgt.v32.external_validation_current import materialize_current_external,CurrentExternalValidationQuery
    path,_=setup(tmp_path/"native",kind,prepare_native=attach_consumed)
    _,_,tables,_=derive_manifest(path)
    assert "CNV_CONSUMED_PARTITION_ANCESTRY_NOT_RECONSTRUCTED" not in set(tables["gaps"].reason)
    assert tables["usage"].study_tokens_json.str.contains("33445566").any()
    assert not tables["usage"].astype(str).apply(lambda c:c.str.contains("PRIVATE_CNV",regex=False)).any().any()
    assert not tables["dependencies"].role.str.contains("PRIVATE_CNV",regex=False).any()
    assert tables["dependencies"].path.str.endswith("lncrna_callable.npy").any()
    inventory=Path(materialize_current_model_use(path,tmp_path/"inventory")["binding_path"])
    crosswalk=tmp_path/"crosswalk.parquet"
    pd.DataFrame([("lncrna","new","LNC:NEW"),("cancer","breast","BRCA"),("pathway","p","PATHWAY:P1")],
                 columns=["namespace","raw_id","canonical_id"]).to_parquet(crosswalk,index=False)
    external=tmp_path/"external.parquet"
    pd.DataFrame({"record_id":["synthetic"],"lncrna":["new"],"cancer":["breast"],"pathway":["p"],"label":[1],
        "pmid":["33445566"],"is_experimental":[True],"is_predicted":[False],"assay":["Synthetic assay"]}).to_parquet(external,index=False)
    source=tmp_path/"external.json"
    write(source,{"format":"CURRENT_EXTERNAL_VALIDATION_INPUT_V2","models":read(path)["models"],
        "model_use_inventory":record(inventory),"identity_crosswalk":record(crosswalk),
        "external_sources":[{"kind":"exact_assay","source_id":"cnv_only_study","table":record(external)}],
        "evaluation":{"primary_model":"fusion","cutoffs":[1],"lncrna_aggregation":"MAX","bootstrap_replicates":0,"seed":0}})
    result=materialize_current_external(source,tmp_path/"external")
    query=CurrentExternalValidationQuery(result["binding_path"],expected_sha=result["binding_sha256"])
    rows=query.query(table="isolation")["rows"]
    assert rows and all(r["study_overlap"] and not r["lncrna_overlap"] and not r["provenance_held_out"] for r in rows)
    assert all("path" not in r for r in query.query(table="model_dependencies")["rows"])


@pytest.mark.parametrize("fault",["missing_array","missing_label","wrong_split","wrong_shape","source_change","matrix_change"])
def test_drift_and_repinned_incomplete_receipts_rejected(tmp_path,fault):
    path,_=fixture(tmp_path,"cnv");attach_consumed(path)
    manifest=read(path);native_path=Path(manifest["native_receipt"]["path"]);native=read(native_path)
    rp=Path(native["consumed_sources"]["path"]);receipt=read(rp)
    if fault=="source_change":
        pd.DataFrame({"pmid":["99999999"]}).to_csv(tmp_path/"segments.tsv",sep="\t",index=False)
        from cc_hhgt.v32.current_model_use import Inventory
        from cc_hhgt.v32.current_cnv_model_use import inspect_consumed_sources
        context=native_sources(path,expected_core_run_id="current-G2")
        with pytest.raises(ValueError,match="SHA256 mismatch"):
            inspect_consumed_sources(Inventory([],tmp_path),context,"fusion","cnv")
        return
    if fault in {"missing_array","matrix_change","wrong_shape"}:
        selected=next(r for r in receipt["records"] if Path(r["path"]).name=="lncrna_signed_mean.npy")
        if fault=="missing_array":receipt["records"].remove(selected)
        else:
            np.save(selected["path"],np.zeros((1,1),np.float32))
            if fault=="wrong_shape":selected["sha256"]=sha256(selected["path"])
    else:
        selected=next(r for r in receipt["records"] if r["role"]=="formal_labels")
        if fault=="missing_label":receipt["records"].remove(selected)
        else:selected["split"]="oof"
    write(rp,receipt);native["consumed_sources"]=record(rp);write(native_path,native);repin_native(path)
    with pytest.raises(ValueError,match="incomplete|columns differ|shape differs|SHA256 mismatch"):
        native_sources(path,expected_core_run_id="current-G2")


def test_capture_detects_changes_during_read_and_before_sealing(tmp_path):
    path=tmp_path/"input.json";write(path,{"value":1});capture=SourceCapture()
    def changing(p):write(p,{"value":2});return {}
    with pytest.raises(ValueError,match="while reading"):capture.read(path,changing,role="core")
    capture.json(path,role="core");write(path,{"value":3})
    with pytest.raises(ValueError,match="before receipt"):
        capture.write(tmp_path/"sources.json",kind="cnv",run_id="synthetic")


@pytest.mark.parametrize("kind",["cnv","cnv_event"])
def test_native_reader_rejects_materialized_matrix_drift_before_fit(tmp_path,kind):
    path,_=fixture(tmp_path,kind);attach_consumed(path)
    partition=tmp_path/"compact_BRCA"
    np.save(partition/"lncrna_callable.npy",np.zeros((5,80),np.float32))
    cls=DirectionalCompactCNV if kind=="cnv" else CompactCNV
    with pytest.raises(ValueError,match="materializer and consumed matrix hashes differ"):
        cls(partition,cancer="BRCA")
