"""Actual two-cancer CLI preprocessing and native consumers, without fitting."""
import json
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32 import native_expression_workflow as workflow
from cc_hhgt.v32 import gdc_star_counts as star
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_gdc_star_counts import inputs, uid, GENES, LNCS, ROOT, no_fit


def fixture_input(root, *, bad_counts=False, two_cancers=False):
    _, original_cache, hits = inputs(root/"raw", n=20, fault="fractional" if bad_counts else None)
    cancers = ["BRCA"]
    if two_cancers:
        cancers.append("LUAD")
        converted = []
        for hit in hits:
            value = json.dumps(hit).replace("TCGA-ZZ-", "TCGA-YY-").replace("TCGA-BRCA", "TCGA-LUAD")
            value = re.sub(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", lambda m: uid("LUAD/"+m[0]), value)
            item = json.loads(value)
            target = original_cache/item["file_id"]/item["file_name"]; target.parent.mkdir()
            shutil.copyfile(original_cache/hit["file_id"]/hit["file_name"], target)
            converted.append(item)
        hits += converted
    def request(payload):
        chosen = hits[payload["from"]:payload["from"]+500]
        return {"data":{"hits":chosen,"pagination":{"from":payload["from"],"count":len(chosen),"total":len(hits)}}}
    query = star.query(cancers, root/"query", request=request)
    patients = root/"patients.tsv"
    pd.DataFrame([{"cancer_id":cancer, "patient_id":f"TCGA-{prefix}-{i:04}", "patient_fold_id":i%5}
                  for cancer, prefix in zip(cancers, ("ZZ", "YY")) for i in range(20)]).to_csv(patients, sep="\t", index=False)
    annotation = root/"annotation.tsv"
    pd.DataFrame({"ensembl_gene_id":[g.split(".")[0] for g in GENES],
        "gene_symbol":["Z", "LOW", "LEAK", *[f"G{i}" for i in range(1,81)]],
        "gene_type":["lncRNA"]*3+["protein_coding"]*80}).to_csv(annotation, sep="\t", index=False)
    gmt = root/"hallmark.gmt"; gmt.write_text("P1\tsynthetic\t"+"\t".join(f"G{i}" for i in range(1,21))+"\n")
    gmt_manifest = root/"gmt.json"
    write(gmt_manifest,{"format":"SSGSEA_GMT_INPUT_V1","collections":[{"collection":"HALLMARK","file":record(gmt)}]})
    manifest = root/"workflow.json"
    write(manifest,{"format":workflow.INPUT,"cancers":cancers,"gdc_query":record(query),"download_root":str(original_cache),
        "patient_map":record(patients),"annotation":record(annotation),"gmt_manifest":record(gmt_manifest),"budget_per_cancer":3,"seed":20260726})
    return manifest


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    assert socket.gethostname() == "149"
    root = tmp_path_factory.mktemp("native_expression_workflow")
    manifest = fixture_input(root/"inputs", two_cancers=True)
    command = [sys.executable,"-m","scripts.run_native_expression_workflow","--manifest",str(manifest),"--output",str(root/"run")]
    planned = subprocess.run([*command,"--plan"],cwd=ROOT,capture_output=True,text=True,timeout=25)
    assert planned.returncode == 0, planned.stderr
    plan = json.loads(planned.stdout)
    assert not (root/"run").exists() and len(plan["commands"]) == 9
    assert not plan["network_acquisition_enabled"] and not plan["training_started"]
    actual = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=150)
    assert actual.returncode == 0, actual.stderr
    return root/"run", manifest


def test_actual_two_cancer_cli_reaches_same_normalization_consumers(completed):
    root, manifest = completed
    path, binding = workflow.validate(record(root/"NATIVE_EXPRESSION_BINDING.json"), root)
    assert path.is_file() and binding["association_analysis_executed"] is False
    assert len(binding["outputs"]["partitions"]) == 2*5*3
    assert read(root/"SOURCE_CONFIG_FRAGMENT.json")["candidate_definition"] == binding["outputs"]["candidate_definition"]
    from cc_hhgt.v32 import fold_local_scope as scope
    from cc_hhgt.v32.current_model_use import Inventory
    inventory = Inventory([], root)
    for cancer in ("BRCA","LUAD"):
        for fold in range(5):
            table = pd.read_parquet(root/f"expression_tables/lncrna/fold={fold}/cancer_id={cancer}/part-0.parquet")
            assert set(table.lncrna_id) == set(LNCS) and len(table) == 3*16
            assert set(table.split_role) == {"train","validation"}
            assert not table.loc[table.lncrna_id.eq(LNCS[1]),"analysis_eligible"].any()
            values = pd.read_parquet(root/f"activity_tables/fold={fold}/cancer_id={cancer}/part-0.parquet")
            assert len(values) == 16 and values.analysis_eligible.all() and np.isfinite(values.activity_score).all()
    output = root/"scope/fold=0/scope.parquet"
    scope.inspect_scope(inventory,record(output.parent/"SCOPE_SOURCES.json"),root,output,"core","scope",0)
    expected_raw = set(p["path"] for p in read(root/"LAUNCH_PREFLIGHT.json")["sources"] if p["path"].endswith(".tsv") and "/cache/" in p["path"])
    assert len(expected_raw) == 40 and expected_raw <= {r["path"] for r in inventory.dependencies}
    assert any("GDC_PROJECT:TCGA-LUAD" in r["study_tokens_json"] for r in inventory.usage)


@pytest.mark.parametrize("fault",["patient","cache","cancers","budget","gmt_relative","unknown_stage"])
def test_invalid_inputs_fail_before_any_output(tmp_path,fault):
    manifest = fixture_input(tmp_path/"inputs"); spec = read(manifest)
    if fault == "patient":
        path = Path(spec["patient_map"]["path"]); frame = pd.read_csv(path,sep="\t"); frame.loc[0,"patient_id"]="TCGA-ZZ-9999"; frame.to_csv(path,sep="\t",index=False); spec["patient_map"]=record(path)
    elif fault == "cache": next(Path(spec["download_root"]).rglob("*.tsv")).write_text("changed")
    elif fault == "cancers": spec["cancers"]=["LUAD"]
    elif fault == "budget": spec["budget_per_cancer"]=300000
    elif fault == "gmt_relative":
        path = Path(spec["gmt_manifest"]["path"]); gmt = read(path); gmt["collections"][0]["file"]["path"]="hallmark.gmt"; write(path,gmt); spec["gmt_manifest"]=record(path)
    else: spec["commands"]=["train"]
    write(manifest,spec)
    with pytest.raises(ValueError): workflow.run(manifest,tmp_path/"bad")
    assert not (tmp_path/"bad").exists()


def test_real_count_parser_failure_records_stage_without_success(tmp_path):
    manifest = fixture_input(tmp_path/"inputs",bad_counts=True)
    with pytest.raises(ValueError,match="counts:BRCA"): workflow.run(manifest,tmp_path/"bad")
    status = read(tmp_path/"bad/STAGE_STATUS.json")
    assert status["status"] == "FAILED" and status["failed_stage"] == "counts:BRCA"
    assert status["stages"][0]["exit_code"] != 0 and status["stages"][0]["status"] == "FAILED"
    assert not (tmp_path/"bad/SUCCESS.json").exists() and not (tmp_path/"bad/normalization").exists()


def test_completed_receipt_rejects_changed_original_raw_source(completed):
    root,_ = completed
    item = next(i for i in read(root/"LAUNCH_PREFLIGHT.json")["sources"] if "/cache/" in i["path"])
    path = Path(item["path"]); original = path.read_bytes()
    try:
        path.write_bytes(original+b"changed")
        with pytest.raises(ValueError): workflow.validate(record(root/"NATIVE_EXPRESSION_BINDING.json"),root)
    finally: path.write_bytes(original)


def test_wrong_host_is_rejected_before_reading_inputs(tmp_path,monkeypatch):
    monkeypatch.setattr(socket,"gethostname",lambda:"windows")
    with pytest.raises(RuntimeError,match="149"): workflow.preflight(tmp_path/"missing",tmp_path/"bad")
