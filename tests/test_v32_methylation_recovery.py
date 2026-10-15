from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32 import methylation_recovery as recovery


def fixture_inputs(tmp_path):
    batches = tmp_path / "failed_run" / "sample_feature_batches"
    batches.mkdir(parents=True)
    rows = []
    for file_id, sample, beta in [("F1", "S1", .2), ("F2", "S1", .6), ("F3", "S2", .9)]:
        rows.append(dict(file_id=file_id,cancer_id="BRCA",patient_id="P1",sample_id=sample,
                         lncrna_id="L1",promoter_methylation_beta=beta,distal_methylation_beta=np.nan,
                         promoter_probe_count=2.,distal_probe_count=np.nan))
    rows.append(dict(file_id="F4",cancer_id="BRCA",patient_id="P2",sample_id="S3",lncrna_id="L2",
                     promoter_methylation_beta=np.nan,distal_methylation_beta=0.,
                     promoter_probe_count=np.nan,distal_probe_count=1.))
    frame = pd.DataFrame(rows)
    batch = batches / "BATCH_00000.parquet"
    frame.to_parquet(batch,index=False)
    selected = tmp_path / "selected.tsv"
    frame[["file_id","cancer_id","patient_id","sample_id"]].to_csv(selected,sep="\t",index=False)
    probes = tmp_path / "probes.parquet"
    pd.DataFrame({"probe_id":["cg1","cg2","cg3"],"lncrna_id":["L1","L1","L2"],
                  "region_type":["promoter","promoter","distal"]}).to_parquet(probes,index=False)
    folds = tmp_path / "folds.tsv"
    pd.DataFrame({"cancer_id":["BRCA","BRCA"],"patient_id":["P1","P2"],
                  "patient_fold_id":[0,1]}).to_csv(folds,sep="\t",index=False)
    return batches,selected,probes,folds


def run_recovery(tmp_path,inputs):
    return recovery.recover_methylation_batches(*inputs,tmp_path / "recovered",
                                               download_root=tmp_path / "RAW_DOES_NOT_EXIST",threads=1)


def test_recovers_without_raw_and_preserves_patient_weight_and_missingness(tmp_path):
    inputs=fixture_inputs(tmp_path)
    audit=run_recovery(tmp_path,inputs)
    frame=pd.read_parquet(audit["outputs"]["patient_features"]["path"])
    p1=frame.loc[frame.patient_id.eq("P1")].iloc[0]
    p2=frame.loc[frame.patient_id.eq("P2")].iloc[0]
    assert p1.promoter_methylation_beta == pytest.approx(.65)
    assert p1.promoter_sample_count == 2
    assert p1.promoter_probe_count == 2  # replicate files must not inflate count
    assert pd.isna(p1.distal_methylation_beta)
    assert not p1.distal_methylation_beta__available
    assert p2.distal_methylation_beta == 0
    assert p2.distal_methylation_beta__available
    assert audit["download_files_md5_verified"] == audit["raw_files_read_this_run"] == 0
    assert audit["recovered_files"] == 4
    assert audit["raw_download_delete_ready"] is False
    assert not recovery.raw_deletion_eligible(audit)
    assert (tmp_path / "recovered/SUCCESS.json").is_file()
    assert (inputs[0] / "BATCH_00000.parquet").is_file()


@pytest.mark.parametrize("change",["patient","duplicate","beta","count","unknown_lnc","incomplete"])
def test_corrupt_or_incomplete_batches_cannot_produce_success(tmp_path,change):
    inputs=fixture_inputs(tmp_path)
    path=inputs[0]/"BATCH_00000.parquet";frame=pd.read_parquet(path)
    if change=="patient":frame.loc[0,"patient_id"]="WRONG"
    elif change=="duplicate":frame=pd.concat([frame,frame.iloc[[0]]])
    elif change=="beta":frame.loc[0,"promoter_methylation_beta"]=np.inf
    elif change=="count":frame.loc[0,"promoter_probe_count"]=3
    elif change=="unknown_lnc":frame.loc[0,"lncrna_id"]="UNKNOWN"
    elif change=="incomplete":frame=frame.iloc[:-1]
    frame.to_parquet(path,index=False)
    with pytest.raises(ValueError):run_recovery(tmp_path,inputs)
    assert not (tmp_path/"recovered/SUCCESS.json").exists()


def test_zero_probe_file_requires_processing_evidence(tmp_path):
    inputs=fixture_inputs(tmp_path)
    selected=pd.read_csv(inputs[1],sep="\t")
    selected.loc[len(selected)]=["EMPTY","BRCA","P3","S4"]
    selected.to_csv(inputs[1],sep="\t",index=False)
    pd.DataFrame([{"file_id":"EMPTY","output_lncrnas":0,"promoter_available":0,"distal_available":0}]).to_parquet(
        inputs[0].parent/"SAMPLE_PROCESSING_RECORDS.parquet",index=False)
    audit=run_recovery(tmp_path,inputs)
    assert audit["recovered_files"]==5
    assert audit["patients"]==2


def test_conflicting_patient_folds_rejected(tmp_path):
    inputs=fixture_inputs(tmp_path)
    folds=pd.read_csv(inputs[3],sep="\t")
    folds.loc[len(folds)]=["BRCA","P1",2]
    folds.to_csv(inputs[3],sep="\t",index=False)
    with pytest.raises(ValueError,match="conflicting"):
        run_recovery(tmp_path,inputs)


def test_recovery_refuses_overwrite_and_non149_host(tmp_path,monkeypatch):
    inputs=fixture_inputs(tmp_path)
    (tmp_path/"recovered").mkdir()
    with pytest.raises(FileExistsError):run_recovery(tmp_path,inputs)
    monkeypatch.setattr(recovery.socket,"gethostname",lambda:"other-host")
    with pytest.raises(RuntimeError,match="149"):run_recovery(tmp_path,inputs)


def test_raw_cleanup_eligibility_cannot_be_inferred_from_feature_success():
    base={"status":"PASS_PATIENT_LNCRNA_METHYLATION_FEATURES","raw_download_delete_ready":True,
          "download_files_expected":2,"download_files_md5_verified":1,"download_bytes_md5_verified":100}
    assert not recovery.raw_deletion_eligible(base)
    assert recovery.raw_deletion_eligible({**base,"download_files_md5_verified":2})


def test_recovery_cannot_replace_the_original_aggregation_map(tmp_path):
    inputs=fixture_inputs(tmp_path)
    (inputs[0].parent/"BATCH_INPUT_AUTHORITY.json").write_text(json.dumps({
        "probe_map_sha256":"different", "selected_files_sha256":recovery._sha256(inputs[1]),
        "patient_folds_sha256":recovery._sha256(inputs[3])}))
    with pytest.raises(ValueError,match="original batch aggregation authority"):
        run_recovery(tmp_path,inputs)
    assert not (tmp_path/"recovered").exists()


def test_fresh_cli_reaches_final_audit_and_matches_recovery(tmp_path):
    raw=tmp_path/"raw";raw.mkdir()
    records=[]
    for i,beta in enumerate([.2,.6]):
        file_id=f"F{i}";folder=raw/file_id;folder.mkdir()
        path=folder/f"sample{i}.txt";path.write_text(f"cg1\t{beta}\ncg2\tNA\n")
        records.append(dict(file_id=file_id,file_name=path.name,md5sum=hashlib.md5(path.read_bytes()).hexdigest(),
                            file_size=path.stat().st_size,patient_id="P1",cancer_id="BRCA",sample_id=f"S{i}"))
    selected=tmp_path/"selected.tsv";pd.DataFrame(records).to_csv(selected,sep="\t",index=False)
    probes=tmp_path/"probes.parquet"
    pd.DataFrame({"probe_id":["cg1","cg2"],"lncrna_id":["L1","L1"],
                  "region_type":["promoter","distal"]}).to_parquet(probes,index=False)
    folds=tmp_path/"folds.tsv"
    pd.DataFrame({"cancer_id":["BRCA"],"patient_id":["P1"],"patient_fold_id":[0]}).to_csv(folds,sep="\t",index=False)
    repo=Path(__file__).resolve().parents[1]
    command=[sys.executable,str(repo/"scripts/materialize_v32_methylation_lncrna.py"),
             "--selected-files",str(selected),"--download-root",str(raw),"--probe-map",str(probes),
             "--patient-folds",str(folds),"--output",str(tmp_path/"fresh"),"--workers","1","--batch-size","1"]
    result=subprocess.run(command,cwd=repo,env={**os.environ,"PYTHONPATH":str(repo)},capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr
    audit=json.loads((tmp_path/"fresh/AUDIT.json").read_text())
    assert audit["rows"]==1 and audit["download_files_md5_verified"]==2
    recovered=recovery.recover_methylation_batches(tmp_path/"fresh/sample_feature_batches",selected,probes,folds,
                                                  tmp_path/"recovered",download_root=raw,threads=1)
    pd.testing.assert_frame_equal(pd.read_parquet(audit["outputs"]["patient_features"]["path"]),
                                  pd.read_parquet(recovered["outputs"]["patient_features"]["path"]))
