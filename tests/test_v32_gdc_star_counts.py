"""Archived synthetic GDC responses -> real RDS -> real TMM/native consumers."""
from pathlib import Path
import copy
import hashlib
import io
import socket
import subprocess
import uuid

import numpy as np
import pandas as pd
import pytest
import torch

from cc_hhgt.v32 import gdc_star_counts as star
from cc_hhgt.v32.formal_expression_sources import read, write, record, binding, inspect
from cc_hhgt.v32.fold_local_expression import convert
from cc_hhgt.v32.current_model_use import Inventory
from test_v32_fold_local_expression import run_r, forbidden

ROOT = Path(__file__).resolve().parents[1]
GENES = [f"ENSG{100000 + i:011}.1" for i in range(83)]
LNCS = ["LNC:" + v.split(".")[0] for v in GENES[:3]]


@pytest.fixture(autouse=True)
def no_fit(monkeypatch):
    assert socket.gethostname() == "149"
    for name in ("Adam", "AdamW", "SGD"): monkeypatch.setattr(torch.optim, name, forbidden)
    from sklearn.linear_model import LogisticRegression
    monkeypatch.setattr(LogisticRegression, "fit", forbidden)


def uid(value): return str(uuid.uuid5(uuid.NAMESPACE_URL, "synthetic-gdc-counts/" + value))


def inputs(root, n=120, fault=None):
    root.mkdir(parents=True)
    hits = []
    for i in range(n):
        patient = f"TCGA-ZZ-{i if i < 120 else 0:04}"
        sample = patient + ("-01A" if i < 120 else "-11A" if i == 120 else "-02A")
        aliquot = sample + "-01R-0000-01"
        file_id = uid("file" + str(i)); case_id = uid(patient); aliquot_id = uid(aliquot)
        name = "synthetic.rna_seq.augmented_star_gene_counts.tsv"
        raw = root / "cache" / file_id / name; raw.parent.mkdir(parents=True)
        rows = ["# gene-model: synthetic GENCODE shape", "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\ttpm_unstranded"]
        rows += [f"{s}\tNA\tNA\t11\t22\t33\tNA" for s in ("N_unmapped", "N_multimapping", "N_noFeature", "N_ambiguous")]
        for g, gene in enumerate(GENES):
            count = ((g * 17 + i * 13) % 100 + 50) * 1000
            if g == 0: count = round(80000 + 20000 * (np.sin(i * .63) + np.cos(i * .27)))
            if g == 1: count = 0
            if g == 2: count = 250000 if i % 5 in (0, 1) else 0
            symbol = ["Z", "LOW", "LEAK"][g] if g < 3 else "G" + str(g - 2)
            kind = "lncRNA" if g < 3 else "protein_coding"
            if fault == "annotation" and i == 1 and g == 0: symbol = "WRONG"
            value = "1.25" if fault == "fractional" and i == 0 and g == 0 else str(count)
            if fault == "negative" and i == 0 and g == 0: value = "-1"
            rows.append(f"{gene}\t{symbol}\t{kind}\t{value}\t{count + 7}\t{count + 19}\t0.5")
        if fault == "missing_summary" and i == 0: rows.pop(2)
        if fault == "duplicate_gene" and i == 0: rows.append(rows[-1])
        raw.write_text("\n".join(rows) + "\n", encoding="utf-8")
        hits.append({"file_id": file_id, "file_name": name, "file_size": raw.stat().st_size,
            "md5sum": hashlib.md5(raw.read_bytes()).hexdigest(), "access": "open", "state": "released",
            "data_category": "Transcriptome Profiling", "data_type": "Gene Expression Quantification",
            "experimental_strategy": "RNA-Seq", "data_format": "TSV", "analysis": {"workflow_type": "STAR - Counts"},
            "associated_entities": [{"case_id": case_id, "entity_type": "aliquot", "entity_id": aliquot_id, "entity_submitter_id": aliquot}],
            "cases": [{"case_id": case_id, "submitter_id": patient, "project": {"project_id": "TCGA-BRCA"},
                "samples": [{"sample_id": uid(sample), "submitter_id": sample, "portions": [{"analytes": [{"aliquots": [
                    {"aliquot_id": aliquot_id, "submitter_id": aliquot}]}]}]}]}]})
    def request(payload):
        offset = payload["from"]; chosen = hits[offset:offset + 500]
        return {"data": {"hits": chosen, "pagination": {"from": offset, "count": len(chosen), "total": len(hits)}}}
    manifest = star.query(["BRCA"], root / "metadata", request=request)
    return manifest, root / "cache", hits


@pytest.fixture(scope="module")
def assembled(tmp_path_factory):
    assert socket.gethostname() == "149"
    root = tmp_path_factory.mktemp("gdc_star")
    manifest, cache, _ = inputs(root / "sources", n=122)
    counts = star.materialize(manifest, "BRCA", cache, root / "assembly")
    patients = root / "patients.tsv"
    pd.DataFrame({"cancer_id": "BRCA", "patient_id": [f"TCGA-ZZ-{i:04}" for i in range(120)],
                  "patient_fold_id": np.arange(120) % 5}).to_csv(patients, sep="\t", index=False)
    run_r(ROOT / "scripts/materialize_fold_local_expression.R", counts, patients, "BRCA", root / "normal")
    convert([root / "normal"], root / "converted", runner=ROOT / "scripts/convert_fold_local_expression.py")
    return root, counts, manifest, cache


def test_actual_three_assay_rds_and_five_fold_normalization(assembled, tmp_path):
    root, counts, _, _ = assembled
    script = tmp_path / "oracle.R"
    script.write_text('''library(SummarizedExperiment)
x<-readRDS(commandArgs(trailingOnly=TRUE)[[1]])
stopifnot(nrow(x)==83L,ncol(x)==122L,identical(assayNames(x),c('unstranded','stranded_first','stranded_second')))
stopifnot(all(assay(x,'stranded_first')-assay(x,'unstranded')==7),all(assay(x,'stranded_second')-assay(x,'unstranded')==19))
stopifnot(assay(x,'unstranded')[1,1]==100000,all(assay(x,'unstranded')[2,]==0))
stopifnot(!any(startsWith(rownames(x),'N_')),length(metadata(x)$star_summary_rows)==122L)
''')
    run_r(script, counts)
    for fold in range(5):
        info = read(root / f"normal/fold={fold}/NORMALIZATION_SOURCES.json")
        assert info["counts_assembly"] == record(str(counts) + ".sources.json")
        table = pd.read_parquet(root / f"converted/lncrna/fold={fold}/cancer_id=BRCA/part-0.parquet")
        assert len(table) == 3 * 96 and set(table.lncrna_id) == set(LNCS)
        assert not table.split_role.eq("test").any()
        assert not table.loc[table.lncrna_id.eq(LNCS[1]), "analysis_eligible"].any()


def test_actual_native_association_and_preparation_inventory_reach_raw_files(assembled, tmp_path, monkeypatch):
    from test_v32_local_cnv_source_ancestry import native_inputs, prepared_capture
    from cc_hhgt.v32.preparation_sources import attach_sources
    root, counts, manifest, cache = assembled
    args, vargs, _, _ = native_inputs(tmp_path / "native", monkeypatch,
        fold_expression_root=root / "converted", patient_map_source=root / "patients.tsv", candidate_ids=LNCS)
    receipt, frames = prepared_capture(tmp_path / "prepared", args, vargs, 0)
    assert all(len(f) == 3 for f in frames.values())
    assert frames["train"].loc[frames["train"].lncrna_id.eq(LNCS[0]), "association_available"].all()
    payload = {"patient_fold": 0, "input_scope": {"association_mode": "local-cnv-adjusted"}}
    attach_sources(payload, receipt)
    inventory = Inventory([], tmp_path)
    inventory.preparation(payload, tmp_path, "core", 0, False)
    dependencies = {r["path"] for r in inventory.dependencies}
    assert {str(p) for p in cache.rglob("*.tsv")} <= dependencies
    assert str(manifest) in dependencies
    assert any("TCGA-BRCA" in r["study_tokens_json"] for r in inventory.usage)
    assert not any(r["reason"] == "EXPRESSION_GDC_ACQUISITION_BEFORE_COUNT_RDS_UNVERIFIED" for r in inventory.gaps)
    assert any(r["reason"] == "GDC_UPSTREAM_ALIGNMENT_AND_BIOSPECIMEN_VERACITY_UNVERIFIED" for r in inventory.gaps)
    assert not any("TCGA-ZZ-" in str(r) for r in inventory.usage)


@pytest.mark.parametrize("fault", ["workflow", "case", "aliquot", "sample", "duplicate_case", "filename", "checksum"])
def test_metadata_mapping_rejects_ambiguity_and_wrong_inputs(tmp_path, fault):
    _, _, hits = inputs(tmp_path / "source", n=1)
    hit = copy.deepcopy(hits[0])
    if fault == "workflow": hit["analysis"]["workflow_type"] = "HTSeq - Counts"
    elif fault == "case": hit["associated_entities"][0]["case_id"] = uid("other")
    elif fault == "aliquot": hit["associated_entities"][0]["entity_id"] = uid("other")
    elif fault == "sample": hit["cases"][0]["samples"][0]["submitter_id"] = "TCGA-ZZ-9999-01A"
    elif fault == "duplicate_case": hit["cases"].append(copy.deepcopy(hit["cases"][0]))
    elif fault == "filename": hit["file_name"] = "../escape.tsv"
    else: hit["md5sum"] = "bad"
    with pytest.raises(ValueError): star.sample_record(hit, ["BRCA"])


@pytest.mark.parametrize("fault", ["fractional", "negative", "annotation", "missing_summary", "duplicate_gene"])
def test_actual_r_parser_rejects_bad_raw_counts(tmp_path, fault):
    manifest, cache, _ = inputs(tmp_path / "source", n=2, fault=fault)
    with pytest.raises(ValueError, match="STAR count assembly/replay failed"):
        star.materialize(manifest, "BRCA", cache, tmp_path / "bad")
    assert not (tmp_path / "bad/SUCCESS.json").exists()


def test_download_bytes_checked_and_missing_cache_never_downloaded_implicitly(tmp_path):
    manifest, cache, _ = inputs(tmp_path / "source", n=1)
    row = read(manifest)["files"][0]; raw = cache / row["file_id"] / row["file_name"]
    payload = raw.read_bytes(); calls = []
    def opener(url, timeout):
        calls.append(url); return io.BytesIO(payload)
    with pytest.raises(FileNotFoundError): star.acquire(row, tmp_path / "new", opener=opener)
    assert not calls
    result = star.acquire(row, tmp_path / "new", download=True, opener=opener)
    assert Path(result["path"]).read_bytes() == payload and calls == [star.API + "/data/" + row["file_id"]]
    with pytest.raises(ValueError, match="exceeds|MD5"):
        star.acquire(row, tmp_path / "broken", download=True, opener=lambda *a, **k: io.BytesIO(payload + b"X"))
    assert not list((tmp_path / "broken").rglob("*.partial"))


def test_modified_cache_and_repinned_rds_values_rejected(assembled, tmp_path):
    _, counts, _, _ = assembled
    changed = tmp_path / "changed.RDS"
    script = tmp_path / "change.R"
    script.write_text("library(SummarizedExperiment);a<-commandArgs(TRUE);x<-readRDS(a[[1]]);assay(x,'unstranded')[1,1]<-0;saveRDS(x,a[[2]])")
    run_r(script, counts, changed)
    source = read(str(counts) + ".sources.json"); source["output"] = record(changed)
    receipt = tmp_path / "source.json"; write(receipt, source)
    write(tmp_path / "SUCCESS.json", {"status": "PASS", "target_host": "149", "training_started": False, "source": record(receipt)})
    with pytest.raises(ValueError, match="values differ"):
        star.validate(record(receipt), tmp_path, changed, replay=True)
    manifest, cache, _ = inputs(tmp_path / "bad_source", n=1)
    row = read(manifest)["files"][0]
    (cache / row["file_id"] / row["file_name"]).write_text("changed")
    with pytest.raises(ValueError, match="size/MD5"):
        star.materialize(manifest, "BRCA", cache, tmp_path / "bad")


def test_unfinished_count_assembly_is_not_consumable(assembled, tmp_path):
    _, counts, _, _ = assembled
    receipt = tmp_path / "source.json"; write(receipt, read(str(counts) + ".sources.json"))
    with pytest.raises(ValueError, match="completed SUCCESS"):
        star.validate(record(receipt), tmp_path, counts)


def test_incomplete_pagination_duplicate_file_and_removed_manifest_rows_rejected(tmp_path):
    manifest, _, _ = inputs(tmp_path / "source", n=2)
    value = read(manifest); value["files"].pop(); write(manifest, value)
    with pytest.raises(ValueError, match="mapping differs"):
        star.materialize(manifest, "BRCA", tmp_path / "source/cache", tmp_path / "bad")
    page = Path(value["pages"][0]["path"]); document = read(page)
    document["response"]["data"]["pagination"]["total"] = 3; write(page, document)
    value["pages"] = [record(page)]
    with pytest.raises(ValueError, match="coverage"): star.replay_pages(value, manifest.parent)
    document["response"]["data"]["pagination"]["total"] = 2
    document["response"]["data"]["hits"][1] = document["response"]["data"]["hits"][0]
    write(page, document); value["pages"] = [record(page)]
    with pytest.raises(ValueError, match="coverage"): star.replay_pages(value, manifest.parent)


def test_wrong_host_prevents_query_before_network_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "gethostname", lambda: "not149")
    with pytest.raises(RuntimeError, match="149"): star.query(["BRCA"], tmp_path / "bad", request=forbidden)
    assert not (tmp_path / "bad").exists()
