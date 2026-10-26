"""Synthetic GTF -> actual annotation/candidates/TMM/ssGSEA; no learned fit."""
import gzip
from pathlib import Path
import shutil
import socket
import subprocess
import sys

import pandas as pd
import pytest

from cc_hhgt.v32 import native_annotation_workflow as annotation
from cc_hhgt.v32 import native_expression_workflow as expression
from cc_hhgt.v32 import native_workflow_chain as chain
from cc_hhgt.v32.formal_expression_sources import read, write, record
from test_v32_native_expression_workflow import fixture_input
from test_v32_gdc_star_counts import GENES, LNCS, ROOT, no_fit, forbidden


def gtf_text(*, extra=False):
    lines = ["##description: evidence-based annotation of the human genome (GRCh38.p13), version 36 (Ensembl 102)",
             "##provider: GENCODE", "##format: gtf"]
    for i, gene in enumerate(GENES):
        symbol = ["Z", "LOW", "LEAK"][i] if i < 3 else "G" + str(i - 2)
        kind = "lncRNA" if i < 3 else "protein_coding"
        lines.append(f'chr1\tHAVANA\tgene\t{i*100+1}\t{i*100+90}\t.\t+\t.\tgene_id "{gene}"; gene_type "{kind}"; gene_name "{symbol}"; level 2; tag "basic"; tag "extra";')
    if extra:
        lines += [f'chrY\tHAVANA\tgene\t1\t40\t.\t-\t.\tgene_id "{GENES[0]}_PAR_Y"; gene_type "lncRNA"; gene_name "Z";',
                  'KI270728.1\tHAVANA\tgene\t2\t50\t.\t+\t.\tgene_id "ENSG99999999999.9"; gene_type "pseudogene"; gene_name "Z";',
                  'chr1\tHAVANA\ttranscript\t1\t20\t.\t+\t.\ttranscript_id "transcript-only";']
    return "\n".join(lines) + "\n"


def inputs(root, *, text=None, compressed=False):
    root.mkdir(parents=True, exist_ok=True)
    raw = root / ("source.gtf.gz" if compressed else "source.gtf")
    payload = (text if text is not None else gtf_text()).encode()
    raw.write_bytes(gzip.compress(payload, mtime=0) if compressed else payload)
    manifest = root / "annotation.json"
    write(manifest, {"format": annotation.INPUT, "gtf": record(raw), "gencode_release": 36, "genome_build": "GRCh38"})
    return manifest


@pytest.mark.parametrize("compressed", [False, True])
def test_original_gene_rows_coordinates_and_par_y_are_preserved(tmp_path, compressed):
    manifest = inputs(tmp_path / "input", text=gtf_text(extra=True), compressed=compressed)
    binding = annotation.run(manifest, tmp_path / "output")
    _, value = annotation.validate(record(binding), tmp_path)
    table = annotation.annotation_table(value["annotation"], binding.parent)
    assert len(table) == 85 and table.ensembl_gene_id.is_unique
    assert set(table.gene_type) == {"lncRNA", "protein_coding", "pseudogene"}
    assert table.iloc[0][["start", "end", "gtf_line"]].tolist() == ["1", "90", "4"]
    par = table.loc[table.ensembl_gene_id.str.endswith("_PAR_Y")].iloc[0]
    assert par.versioned_gene_id == GENES[0] + "_PAR_Y" and par.strand == "-"
    assert table.gene_symbol.eq("Z").sum() == 3
    assert value["metadata"]["header_genome_build"] == "GRCh38.p13"
    from cc_hhgt.v32.fold_local_scope import annotation_ids
    assert annotation_ids(value["annotation"], binding.parent) == LNCS
    assert not value["reference_authenticity_verified"] and not value["source_ancestry_complete"]
    assert set(table.coordinate_system) == {"1_BASED_CLOSED"}


@pytest.mark.parametrize("fault", ["release", "build", "missing_header", "two_headers", "duplicate", "duplicate_attribute",
                                  "missing_type", "invalid_id", "lifted_id", "zero_start", "reverse", "strand", "malformed", "empty"])
def test_invalid_raw_reference_rejected_before_output(tmp_path, fault):
    source = gtf_text()
    if fault == "release": source = source.replace("version 36", "version 37")
    elif fault == "build": source = source.replace("GRCh38.p13", "GRCh37")
    elif fault == "missing_header": source = "\n".join(source.splitlines()[1:])
    elif fault == "two_headers": source += source.splitlines()[0] + "\n"
    elif fault == "duplicate": source += source.splitlines()[3].replace(GENES[0], GENES[0].replace(".1", ".2")) + "\n"
    elif fault == "duplicate_attribute": source = source.replace('gene_type "lncRNA";', 'gene_type "lncRNA"; gene_type "lncRNA";', 1)
    elif fault == "missing_type": source = source.replace('gene_type "lncRNA";', '', 1)
    elif fault == "invalid_id": source = source.replace(GENES[0], "BAD.1", 1)
    elif fault == "lifted_id": source = source.replace(GENES[0], GENES[0] + "_2", 1)
    elif fault == "zero_start": source = source.replace("\t1\t90\t", "\t0\t90\t", 1)
    elif fault == "reverse": source = source.replace("\t1\t90\t", "\t91\t90\t", 1)
    elif fault == "strand": source = source.replace("\t+\t", "\t.\t", 1)
    elif fault == "malformed": source = source.replace('level 2;', 'malformed;', 1)
    else: source = "\n".join(source.splitlines()[:3]) + "\n"
    manifest = inputs(tmp_path / "input", text=source)
    with pytest.raises(ValueError): annotation.run(manifest, tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("fault", ["gene_symbol", "start", "gene_type", "gtf_line", "omission", "metadata"])
def test_rehashed_table_or_metadata_cannot_replace_raw_replay(tmp_path, fault):
    manifest = inputs(tmp_path / "input"); binding = annotation.run(manifest, tmp_path / "output")
    value = read(binding); output = Path(value["annotation"]["path"])
    if fault == "metadata": value["metadata"]["gene_count"] -= 1
    else:
        table = pd.read_csv(output, sep="\t", dtype=str)
        if fault == "omission": table = table.iloc[1:]
        else: table.loc[0, fault] = "changed"
        table.to_csv(output, sep="\t", index=False); value["annotation"] = record(output)
    write(binding, value)
    with pytest.raises(ValueError, match="GTF replay"): annotation.validate(record(binding), tmp_path)


def test_bound_raw_bytes_rechecked_after_cached_validation(tmp_path):
    manifest = inputs(tmp_path / "input"); binding = annotation.run(manifest, tmp_path / "output")
    raw = Path(read(manifest)["gtf"]["path"]); raw.write_text(gtf_text().replace("version 36", "version 37"))
    with pytest.raises(ValueError): annotation.validate(record(binding), tmp_path)


def test_missing_or_different_sidecar_is_rejected_at_actual_consumers(tmp_path):
    manifest = inputs(tmp_path / "input"); binding = annotation.run(manifest, tmp_path / "output")
    item = read(binding)["annotation"]; copied = tmp_path / "copied.tsv"; shutil.copyfile(item["path"], copied)
    from cc_hhgt.v32.fold_local_scope import annotation_ids
    with pytest.raises((FileNotFoundError, ValueError)): annotation_ids(record(copied), tmp_path)
    wrong = {**item, "upstream_annotation": record(manifest)}
    with pytest.raises(ValueError, match="pointer differs"): annotation.annotation_table(wrong, tmp_path)


@pytest.mark.parametrize("fault", ["version", "symbol", "biotype", "missing"])
def test_star_gene_metadata_mismatch_blocks_expression_before_output(tmp_path, fault):
    manifest = fixture_input(tmp_path / "expression"); spec = read(manifest)
    raw = gtf_text()
    if fault == "version": raw = raw.replace(GENES[0], GENES[0].replace(".1", ".2"), 1)
    elif fault == "symbol": raw = raw.replace('gene_name "Z"', 'gene_name "WRONG"', 1)
    elif fault == "biotype": raw = raw.replace('gene_type "lncRNA"', 'gene_type "pseudogene"', 1)
    else: raw = "\n".join(raw.splitlines()[:3] + raw.splitlines()[4:]) + "\n"
    source = inputs(tmp_path / "gtf", text=raw)
    binding = annotation.run(source, tmp_path / "annotation")
    spec["annotation"] = read(binding)["annotation"]; write(manifest, spec)
    with pytest.raises(ValueError, match="STAR gene"): expression.run(manifest, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_host_and_immutable_output_guards(tmp_path, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(socket, "gethostname", lambda: "Windows")
        with pytest.raises(RuntimeError, match="149"): annotation.plan(tmp_path / "missing", tmp_path / "new")
    manifest = inputs(tmp_path / "input")
    with pytest.raises(FileExistsError): annotation.run(manifest, tmp_path / "input")
    assert not (tmp_path / "input" / annotation.OUTPUT).exists()


def test_native_cli(tmp_path):
    manifest = inputs(tmp_path / "input")
    result = subprocess.run([sys.executable, "-m", "scripts.run_native_annotation_workflow", "--manifest", str(manifest),
                             "--output", str(tmp_path / "output")], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    annotation.validate(record(tmp_path / "output" / annotation.BINDING), tmp_path)


def test_actual_annotation_to_expression_chain_reaches_model_input_inspection(tmp_path, monkeypatch):
    manifest = fixture_input(tmp_path / "expression"); spec = read(manifest)
    raw = inputs(tmp_path / "gtf", compressed=True)
    spec["annotation"] = {"stage_ref": "annotation", "pointer": "/annotation"}
    request = tmp_path / "chain.json"
    write(request, {"format": chain.INPUT, "chain_id": "synthetic-annotation-expression", "stages": [
        {"id": "annotation", "operation": "annotation.run", "inputs": read(raw)},
        {"id": "expression", "operation": "expression.run", "inputs": spec}]})
    output = tmp_path / "chain"; chain.initialize(request, output); chain.advance(output)
    state = read(output / "CHAIN.json")
    assert state["status"] == "DECLARED_STAGES_COMPLETE", state
    result = Path(state["results"]["expression"]["path"]); value = read(result); run = result.parent
    plan = read(value["preflight"]["path"])
    checks = plan["annotation_reference"]["star_gene_metadata_checks"]
    assert checks["BRCA"]["counted_genes_checked"] == 83
    assert checks["BRCA"]["uncounted_annotation_genes"] == 0
    assert not checks["BRCA"]["star_release_independently_verified"]
    from cc_hhgt.v32.current_model_use import Inventory
    from cc_hhgt.v32 import fold_local_scope, fold_local_activity
    inventory = Inventory([], run)
    candidate = run / "scope/candidates.parquet"
    fold_local_scope.inspect_candidates(inventory, record(candidate.parent / "CANDIDATE_SOURCES.json"), run, candidate, "core", "candidates", 0)
    activity = run / "activity_tables/fold=0/cancer_id=BRCA/part-0.parquet"
    fold_local_activity.inspect(inventory, record(activity.parent / "ACTIVITY_SOURCES.json"), run, activity, "core", "activity", 0)
    gtf = read(raw)["gtf"]
    for role in ("candidates.annotation", "activity.annotation"):
        assert any(r["path"] == gtf["path"] and r["role"].startswith(role) for r in inventory.dependencies)
        reasons = {g["reason"] for g in inventory.gaps if g["role"] == role}
        assert reasons == {"GENCODE_REFERENCE_ACQUISITION_AND_AUTHENTICITY_UNVERIFIED"}
    assert sum(r["format"] == annotation.FORMAT for r in inventory.transform_replays) == 2
    assert set(pd.read_parquet(candidate).lncrna_id) == set(LNCS)
    assert len(pd.read_parquet(activity)) == 16
    before = {str(p): p.stat().st_mtime_ns for p in (output / "stages").rglob("*") if p.is_file()}
    monkeypatch.setattr(chain, "dispatch", forbidden); chain.advance(output)
    assert before == {str(p): p.stat().st_mtime_ns for p in (output / "stages").rglob("*") if p.is_file()}


def test_legacy_table_keeps_unknown_ancestry(tmp_path):
    path = tmp_path / "legacy.tsv"
    pd.DataFrame({"ensembl_gene_id": ["ENSG00000000001"], "gene_type": ["lncRNA"]}).to_csv(path, sep="\t", index=False)
    from cc_hhgt.v32.current_model_use import Inventory
    from cc_hhgt.v32.current_auxiliary_model_use import input_source
    inventory = Inventory([], tmp_path); input_source(inventory, record(path), tmp_path, "core", "annotation", 0)
    assert any(r["reason"] == "RAW_UPSTREAM_TRANSFORMATIONS_NOT_RECONSTRUCTED" for r in inventory.gaps)
    assert not inventory.transform_replays
