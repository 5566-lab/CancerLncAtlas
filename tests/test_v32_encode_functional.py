from dataclasses import replace
import hashlib
import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

from cc_hhgt.v32.encode_functional import (FunctionalContrast, bh, validate_contrast_selection, verified_encode_file, read_contrast_table,
    parse_functional_table, gene_level_tests, pathway_support, contextual_functional_evidence)


def contrast(**kw):
    return replace(FunctionalContrast("KD-A", "ENCFF000AAA", "ENCSR000AAA", "GENE:RBP1", "K562", "deseq2", "KD_MINUS_CONTROL"), **kw)


def fixture():
    # Twenty tested genes, with every changed gene in P1. Background is 20,
    # not the four significant genes or the entire annotated genome.
    source = [f"ENSG{i:011d}.2" for i in range(20)]
    frame = pd.DataFrame({"gene_id": source, "log2FoldChange": [2., 3., -2., -3.]+[.1]*16,
                          "pvalue": [1e-6]*4+[.8]*16, "padj": [1e-5]*4+[.9]*16})
    crosswalk = pd.DataFrame({"source_gene_id": source, "gene_id": [f"G{i}" for i in range(20)]})
    membership = pd.DataFrame({"pathway_id": ["P1"]*4+["P2"]*3+["Pmissing"]*3,
                               "gene_id": ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G20", "G21", "G22"]})
    return frame, crosswalk, membership


def test_de_contrast_direction_and_tested_background_are_preserved():
    frame, crosswalk, members = fixture()
    records = parse_functional_table(frame, contrast(), crosswalk)
    inverse = parse_functional_table(frame, contrast(effect_orientation="CONTROL_MINUS_KD"), crosswalk)
    np.testing.assert_array_equal(records.effect, -inverse.effect)
    genes = gene_level_tests(records)
    result = pathway_support(genes, members)
    row = result.loc[result.pathway_id.eq("P1") & result.direction.eq("altered")].iloc[0]
    assert row.background_tested_genes == 20 and row.overlap_genes == 4
    assert row.pvalue == pytest.approx(hypergeom.sf(3, 20, 4, 4))
    assert row.qvalue == pytest.approx(row.pvalue*3)  # Pmissing stays in the family.
    assert result.loc[result.pathway_id.eq("Pmissing"), "pvalue"].isna().all()
    assert not records.physical_binding_claimed.any() and not records.lncrna_perturbation_claimed.any()


def test_missing_tests_and_missing_version_mapping_are_not_no_effect():
    frame, crosswalk, members = fixture()
    frame.loc[0, "padj"] = np.nan; frame.loc[1, "gene_id"] = "ENSG00000000001.99"
    records = parse_functional_table(frame, contrast(), crosswalk)
    assert not records.available.iloc[:2].any()
    assert records.unavailable_reason.iloc[0] == "MISSING_OR_NONFINITE_TEST"
    assert records.unavailable_reason.iloc[1] == "GENE_CROSSWALK_UNAVAILABLE"
    result = pathway_support(gene_level_tests(records), members, min_tested_genes=2)
    assert result.background_tested_genes.eq(18).all()
    assert len(records) == 20


def test_completely_unmapped_contrast_retains_unavailable_pathway_universe():
    frame, crosswalk, members = fixture()
    records = parse_functional_table(frame, contrast(), crosswalk.iloc[:0])
    genes = gene_level_tests(records)
    result = pathway_support(genes, members, contexts=records)
    assert len(result) == 9 and not result.available.any()
    assert result.background_tested_genes.eq(0).all() and result.qvalue.isna().all()


def test_rmats_counts_genes_once_and_preserves_opposite_event_directions():
    raw = pd.DataFrame({"GeneID": ['"a"', '"a"', '"b"', '"c"'], "ID": ["SE1", "SE2", "SE3", "SE4"],
        "IncLevelDifference": [.4, -.3, .01, np.nan], "PValue": [1e-6, 2e-6, .8, np.nan], "FDR": [1e-5, 2e-5, .9, np.nan]})
    crosswalk = pd.DataFrame({"source_gene_id": ["a", "b", "c"], "gene_id": ["A", "B", "C"]})
    records = parse_functional_table(raw, contrast(schema="rmats"), crosswalk)
    genes = gene_level_tests(records).set_index("gene_id")
    assert len(genes) == 3 and genes.loc["A", "up"] and genes.loc["A", "down"]
    assert genes.loc["A", "pvalue"] == pytest.approx(2e-6)
    assert genes.loc["A", "qvalue"] == pytest.approx(6e-6)
    assert not genes.loc["C", "available"] and np.isnan(genes.loc["C", "qvalue"])
    unmapped = parse_functional_table(raw, contrast(schema="rmats"), crosswalk.iloc[:2])
    with_unmapped = gene_level_tests(unmapped).set_index("gene_id")
    assert with_unmapped.loc["A", "qvalue"] == pytest.approx(6e-6)
    with pytest.raises(ValueError, match="Duplicate"):
        parse_functional_table(pd.concat([raw, raw.iloc[:1]], ignore_index=True), contrast(schema="rmats"), crosswalk)


def test_native_cuffdiff_status_and_orientation():
    raw = pd.DataFrame({"gene_id": ["a", "b"], "log2(fold_change)": [2., 4.], "p_value": [.001, .001],
                        "q_value": [.01, .01], "status": ["OK", "NOTEST"]})
    crosswalk = pd.DataFrame({"source_gene_id": ["a", "b"], "gene_id": ["A", "B"]})
    result = parse_functional_table(raw, contrast(schema="cuffdiff", effect_orientation="CONTROL_MINUS_KD"), crosswalk)
    assert result.effect.iloc[0] == -2 and not result.available.iloc[1]
    with pytest.raises(ValueError, match="orientation"):
        parse_functional_table(raw, contrast(schema="cuffdiff", effect_orientation="UNKNOWN"), crosswalk)


def test_archive_member_selection_does_not_extract_or_double_count(tmp_path):
    path = tmp_path/"splicing.tar.gz"
    raw = b"GeneID\tID\tIncLevelDifference\tPValue\tFDR\na\tSE1\t0.5\t0.01\t0.02\n"
    with tarfile.open(path, "w:gz") as archive:
        for name in ["results/SE.MATS.JC.txt", "results/SE.MATS.JCEC.txt", "../escape.txt"]:
            info = tarfile.TarInfo(name); info.size = len(raw); archive.addfile(info, io.BytesIO(raw))
    assert len(read_contrast_table(path, contrast(schema="rmats", member="results/SE.MATS.JC.txt"))) == 1
    assert not (tmp_path/"results").exists()
    with pytest.raises(ValueError, match="explicit member"): read_contrast_table(path, contrast(schema="rmats"))
    with pytest.raises(ValueError, match="Unsafe"): read_contrast_table(path, contrast(member="../escape.txt"))
    jc = contrast(schema="rmats", member="results/SE.MATS.JC.txt", event_type="SE", quantification="JC")
    jcec = replace(jc, contrast_id="KD-JCEC", member="results/SE.MATS.JCEC.txt", quantification="JCEC")
    with pytest.raises(ValueError, match="one processing"): validate_contrast_selection([jc, jcec])
    validate_contrast_selection([jc, replace(jcec, event_type="RI", member="results/RI.MATS.JCEC.txt")])
    with pytest.raises(ValueError, match="one processing"):
        validate_contrast_selection([contrast(), contrast(contrast_id="same-experiment-reprocessed")])


def test_cached_file_is_verified_and_missing_does_not_trigger_download(tmp_path, monkeypatch):
    import subprocess
    def no_network(*a, **k): raise AssertionError("No download was requested")
    monkeypatch.setattr(subprocess, "run", no_network)
    payload = b"gene_id\tpvalue\na\t0.01\n"
    record = {"file_accession": "ENCFF000AAA", "href": "/files/ENCFF000AAA/@@download/ENCFF000AAA.tsv",
        "file_size": len(payload), "md5sum": hashlib.md5(payload).hexdigest(), "status": "released"}
    with pytest.raises(FileNotFoundError): verified_encode_file(record, tmp_path)
    target = tmp_path/"ENCFF000AAA.tsv"; target.write_bytes(payload)
    assert verified_encode_file(record, tmp_path) == target
    target.write_bytes(payload.replace(b"0.01", b"0.99"))
    with pytest.raises(ValueError, match="checksum"): verified_encode_file(record, tmp_path)


def test_functional_bridge_requires_physical_assay_same_biosample_and_exact_pathway():
    frame, crosswalk, members = fixture()
    pathways = pathway_support(gene_level_tests(parse_functional_table(frame, contrast(), crosswalk)), members)
    binding = pd.DataFrame({"lncrna_id": ["L1", "L2", "L3", "L4"], "rbp_id": "GENE:RBP1",
        "biosample": ["K562", "K562", "HepG2", "K562"], "graph_assay_class": ["eclip", "predicted", "eclip", "eclip"],
        "binding_source_id": ["E1", "P2", "E3", "E4"]})
    exact = pd.DataFrame({"cancer_id": "LAML", "lncrna_id": ["L1", "L2", "L3", "L4"], "pathway_id": ["P1", "P1", "P1", "P2"]})
    result = contextual_functional_evidence(pathways, binding, exact)
    assert set(result.lncrna_id) == {"L1"}
    assert result.binding_source_id.eq("E1").all() and not result.lncrna_perturbation_claimed.any()
    assert not result.independent_external_validation.any()


def test_complete_command_consumes_cached_native_data_without_training(tmp_path):
    from scripts.analyze_v32_encode_functional import main
    from cc_hhgt.v32.input_lineage import artifact_sha256
    frame, crosswalk, members = fixture()
    cache = tmp_path/"cache"; cache.mkdir(); raw = cache/"ENCFF000AAA.tsv"; frame.to_csv(raw, sep="\t", index=False)
    tables = {
        "inventory": pd.DataFrame([{"file_accession": "ENCFF000AAA", "experiment_accession": "ENCSR000AAA", "rbp": "RBP1",
            "biosample": "K562", "status": "released", "href": "/files/ENCFF000AAA/@@download/ENCFF000AAA.tsv",
            "md5sum": hashlib.md5(raw.read_bytes()).hexdigest(), "file_size": raw.stat().st_size,
            "output_type": "differential expression quantifications"}]),
        "gene_crosswalk": crosswalk, "rbp_crosswalk": pd.DataFrame({"source_rbp_label": ["RBP1"], "rbp_id": ["GENE:RBP1"]}),
        "pathway_membership": members,
        "physical_binding": pd.DataFrame({"lncrna_id": ["L1"], "rbp_id": ["GENE:RBP1"], "biosample": ["K562"],
            "graph_assay_class": ["eclip"], "binding_source_id": ["ENCFF_BINDING"]}),
        "exact_candidates": pd.DataFrame({"cancer_id": ["LAML"], "lncrna_id": ["L1"], "pathway_id": ["P1"]})}
    artifacts = {}
    for role, table in tables.items():
        path = tmp_path/(role+".parquet"); table.to_parquet(path, index=False)
        artifacts[role] = {"path": str(path), "sha256": artifact_sha256(path)}
    manifest = {"format": "ENCODE_FUNCTIONAL_ANALYSIS_V1", "reference_annotation_id": "synthetic-v2", "artifacts": artifacts,
        "contrasts": [{**contrast().__dict__, "orientation_source": "synthetic controlled contrast"}]}
    path = tmp_path/"manifest.json"; path.write_text(json.dumps(manifest)); output = tmp_path/"output"
    assert main(["--manifest", str(path), "--cache", str(cache), "--output", str(output)]) == 0
    audit = json.loads((output/"AUDIT.json").read_text())
    assert not audit["training_started"] and not audit["primary_graph_changed"]
    assert audit["outputs"]["FUNCTIONAL_EVIDENCE.parquet"]["rows"] > 0
    assert len(pd.read_parquet(output/"RBP_PATHWAY_SUPPORT.parquet")) == 9
