import hashlib
import itertools
import json

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.rbns_sequence import (parse_rbns_profile, iter_transcript_scores, aggregate_gene_scores, predicted_binding_inputs)


def native_profile():
    kmers = ["".join(x) for x in itertools.product("ACGT", repeat=4)]
    return pd.DataFrame({"[RBP1]": kmers, "5 nM": [4. if x == "AACG" else 1. for x in kmers], "20 nM": 1.})


def profile(frame=None):
    return parse_rbns_profile(native_profile() if frame is None else frame, profile_id="RBP1-4mer-5nM", rbp_id="GENE:RBP1",
        file_accession="ENCFF000AAA", experiment_accession="ENCSR000AAA", concentration="5 nM", source_rbp_label="RBP1")


def crosswalk():
    return pd.DataFrame({"transcript_id": ["ENST0001.2", "ENST0002.2", "ENST0003.2", "ENST0004.2"],
        "source_gene_id": ["ENSG0001.4", "ENSG0001.4", "ENSG0002.1", "ENSG0003.1"],
        "lncrna_id": ["LNC:ENSG0001", "LNC:ENSG0001", "LNC:ENSG0002", "LNC:ENSG0003"], "annotation_id": "test-v2"})


def test_sequence_windows_use_rna_orientation_and_composition_comparator(tmp_path):
    fasta = tmp_path/"rna.fa"
    fasta.write_text(">ENST0001.2|ENSG0001.4|\nAACGAACGAACG\n>ENST0002.2|ENSG0001.4|\nGCAAGCAAGCAA\n")
    result = pd.DataFrame(iter_transcript_scores(fasta, crosswalk(), [profile()], annotation_id="test-v2", min_valid_windows=1))
    assert result.mean_log2_enrichment.iloc[0] == pytest.approx(2/3)
    assert result.mean_log2_enrichment.iloc[1] == 0
    assert result.composition_expected_log2_enrichment.iloc[0] == pytest.approx(2*(.5**2)*(.25**2))
    assert result.composition_expected_log2_enrichment.iloc[0] == result.composition_expected_log2_enrichment.iloc[1]
    assert result.valid_windows.eq(9).all() and not result.lncrna_binding_observed.any()


def test_incomplete_kmer_profile_and_undeclared_concentration_are_rejected():
    with pytest.raises(ValueError, match="Incomplete"): profile(native_profile().iloc[:-1])
    bad = native_profile(); bad.loc[0, "5 nM"] = np.nan
    with pytest.raises(ValueError, match="Invalid native"): profile(bad)
    with pytest.raises(ValueError, match="concentration"): profile(native_profile().drop(columns=["5 nM"]))
    # A measured zero is accepted under the documented log floor, never treated as missing.
    zero = native_profile(); zero.loc[0, "5 nM"] = 0
    assert np.isfinite(profile(zero).log2_enrichment).all()


def test_sequence_version_and_ambiguity_cannot_be_silently_ignored(tmp_path):
    path = tmp_path/"rna.fa"; path.write_text(">ENST0001.2|ENSG0001.99|\nAACGAACG\n")
    with pytest.raises(ValueError, match="gene version"):
        list(iter_transcript_scores(path, crosswalk(), [profile()], annotation_id="test-v2", min_valid_windows=1))
    path.write_text(">ENST0001.2|ENSG0001.4|\nAACGNNNNNNNN\n")
    result = list(iter_transcript_scores(path, crosswalk(), [profile()], annotation_id="test-v2", min_valid_windows=1))[0]
    assert not result["available"] and np.isnan(result["mean_log2_enrichment"])
    assert result["valid_windows"] == 1 and result["possible_windows"] == 9
    with pytest.raises(ValueError, match="annotation version"):
        list(iter_transcript_scores(path, crosswalk(), [profile()], annotation_id="other-v3"))


def mapping_and_universe():
    protein = pd.DataFrame({"gene_id": ["GENE:RBP1"], "protein_id": ["PROTEIN:P1"], "uniprot_accession": ["P1"],
        "mapping_multiplicity": [1], "mapping_weight": [1.]})
    universe = pd.DataFrame({"lncrna_id": ["LNC:ENSG0001", "LNC:ENSG0002", "LNC:ENSG0003", "LNC:UNANNOTATED"]})
    return protein, universe


def test_gene_coverage_and_actual_typed_graph_materializer_preserve_predicted_status():
    protein, universe = mapping_and_universe()
    summary = pd.DataFrame({"lncrna_id": ["LNC:ENSG0001", "LNC:ENSG0002"], "profile_id": "RBP1-4mer-5nM",
        "observed_transcripts": [2, 1], "scorable_transcripts": [2, 0], "mean_log2_enrichment": [1., np.nan],
        "mean_composition_excess": [.8, np.nan]})
    result = aggregate_gene_scores(summary, crosswalk(), universe, [profile()])
    assert len(result) == 4 and result.available.sum() == 1
    assert result.loc[~result.available, "prediction_score"].isna().all()
    relation, binding, audit = predicted_binding_inputs(result, protein, universe)
    assert len(binding) == 1 and binding.relation_type.eq("binds_protein_predicted").all()
    assert binding.weight.iloc[0] == pytest.approx(.4)
    assert relation.is_predicted.all() and not relation.is_experimental.any()
    assert not audit["main_training_graph_modified"]
    # Missing one of two isoforms makes a gene unavailable under the complete-coverage default.
    summary.loc[0, "scorable_transcripts"] = 1
    incomplete = aggregate_gene_scores(summary, crosswalk(), universe, [profile()])
    assert not incomplete.available.any()
    assert predicted_binding_inputs(incomplete, protein, universe)[1].empty


def test_full_command_streams_native_scores_and_writes_predicted_binding(tmp_path):
    from cc_hhgt.v32.input_lineage import artifact_sha256
    from scripts.analyze_v32_rbns_sequences import main
    cache = tmp_path/"cache"; cache.mkdir(); native = cache/"ENCFF000AAA.tsv"
    native_profile().to_csv(native, sep="\t", index=False)
    protein, universe = mapping_and_universe()
    tables = {"inventory": pd.DataFrame([{"file_accession": "ENCFF000AAA", "experiment_accession": "ENCSR000AAA", "rbp": "RBP1",
        "status": "released", "output_type": "enrichment", "href": "/files/ENCFF000AAA/@@download/ENCFF000AAA.tsv",
        "file_size": native.stat().st_size, "md5sum": hashlib.md5(native.read_bytes()).hexdigest()}]),
        "rbp_crosswalk": pd.DataFrame({"source_rbp_label": ["RBP1"], "rbp_id": ["GENE:RBP1"]}),
        "transcript_crosswalk": crosswalk(), "lncrnas": universe, "protein_gene_map": protein}
    artifacts = {}
    for role, table in tables.items():
        path = tmp_path/(role+".parquet"); table.to_parquet(path, index=False)
        artifacts[role] = {"path": str(path), "sha256": artifact_sha256(path)}
    path = tmp_path/"rna.fa"
    path.write_text(">ENST0001.2|ENSG0001.4|\n"+"AACG"*20+"\n>ENST0002.2|ENSG0001.4|\n"+"AACG"*20+
                    "\n>ENST0003.2|ENSG0002.1|\nNNNNNNNNNN\n")
    artifacts["transcript_fasta"] = {"path": str(path), "sha256": artifact_sha256(path)}
    manifest = {"format": "RBNS_SEQUENCE_ANALYSIS_V1", "annotation_id": "test-v2", "artifacts": artifacts,
        "profiles": [{"profile_id": "RBP1-4mer-5nM", "file_accession": "ENCFF000AAA", "rbp_id": "GENE:RBP1", "concentration": "5 nM"}]}
    path = tmp_path/"manifest.json"; path.write_text(json.dumps(manifest)); output = tmp_path/"output"
    assert main(["--manifest", str(path), "--cache", str(cache), "--output", str(output), "--chunk-rows", "1"]) == 0
    audit = json.loads((output/"AUDIT.json").read_text())
    assert audit["available_gene_profiles"] == 1 and audit["unavailable_gene_profiles"] == 3
    assert len(list(output.glob("transcript_scores_*.parquet"))) == 3
    assert not audit["training_started"] and not audit["lncrna_binding_measured"]
    assert len(pd.read_parquet(output/"RBNS_TYPED_BINDING.parquet")) == 1
