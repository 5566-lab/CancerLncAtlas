from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.distal_topology import (coordinate_distal_candidates, topology_lineage,
    feature_topology_lineage, require_shared_predictive_topology, sha256)
from scripts.build_v32_independent_distal_topology import main as build


def inputs(tmp_path):
    promoters = pd.DataFrame({"#chrom": ["chr1", "chr1", "chr2"], "start": [100,300,100],
                              "end": [110,310,110], "gene_id": ["L1.1","L1.1","L2.1"]})
    peaks = pd.DataFrame({"seqnames": ["chr1"]*4+["chr2"], "start": [105,201,305,10001,201],
                         "end": [109,210,307,10010,210], "name": ["promoter","distal1","other_tss","far","distal2"]})
    genes = pd.Series(["LNC:L1", "LNC:L2", "LNC:L3"], name="gene_id")
    promoters.to_csv(tmp_path/"promoters.tsv", sep="\t", index=False)
    peaks.to_csv(tmp_path/"peaks.tsv", sep="\t", index=False)
    genes.to_frame().to_csv(tmp_path/"genes.tsv", sep="\t", index=False)
    return promoters, peaks, genes


def build_fixture(tmp_path):
    inputs(tmp_path)
    assert build(["--promoters", str(tmp_path/"promoters.tsv"), "--peak-set", str(tmp_path/"peaks.tsv"),
                  "--required-genes", str(tmp_path/"genes.tsv"), "--reference-build", "GRCh38",
                  "--max-distance-bp", "200", "--output", str(tmp_path/"topology")]) == 0
    return tmp_path/"topology/INDEPENDENT_DISTAL_LINKS.tsv.gz"


def test_coordinates_exclude_all_transcript_promoters_and_preserve_uncovered_genes(tmp_path):
    path = build_fixture(tmp_path)
    result = pd.read_csv(path, sep="\t")
    assert list(zip(result.lncrna_id, result.peak_name)) == [("LNC:L1","distal1"),("LNC:L2","distal2")]
    assert result.peak_start0.tolist() == [200,200]
    assert result.distance_to_promoter_bp.tolist() == [90,90]
    assert pd.read_csv(path.parent/"GENE_COVERAGE.tsv",sep="\t").linked_peaks.tolist() == [1,1,0]
    assert topology_lineage(path)["predictive_evaluation_eligible"] is True


def test_topology_does_not_consume_expression_or_association_columns(tmp_path):
    p,q,g = inputs(tmp_path)
    first = coordinate_distal_candidates(p,q,g,max_distance_bp=200)
    second = coordinate_distal_candidates(p.assign(expression=[1,9,1000]),
        q.assign(correlation=[1,1,-1,0,0], fdr=0), g, max_distance_bp=200)
    pd.testing.assert_frame_equal(first,second)
    with pytest.raises(ValueError,match="finite integers"):
        coordinate_distal_candidates(p,q.assign(start=1.5),g)


def test_legacy_changed_and_mixed_topologies_cannot_be_formal_inputs(tmp_path):
    path = build_fixture(tmp_path)
    lineage = topology_lineage(path)
    legacy = tmp_path/"legacy.tsv.gz"
    pd.DataFrame({"lncrna_id":["LNC:L1"],"peak_name":["distal1"]}).to_csv(legacy,sep="\t",index=False)
    assert not topology_lineage(legacy)["predictive_evaluation_eligible"]
    with pytest.raises(ValueError, match="outcome-independent"):
        require_shared_predictive_topology(lineage,topology_lineage(legacy))
    with pytest.raises(ValueError, match="same distal topology"):
        require_shared_predictive_topology(lineage,{**lineage,"links_sha256":"other"})
    path.write_bytes(path.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="does not match"):
        topology_lineage(path)


def test_real_mutation_materializer_consumes_new_topology_and_preserves_wes_semantics(tmp_path,monkeypatch):
    import scripts.materialize_v32_distal_mutation_lncrna as script
    links = build_fixture(tmp_path)
    patients = [f"TCGA-AA-{i:04}" for i in range(5)]
    folds = pd.DataFrame({"cancer_id":"BRCA","patient_id":patients,"patient_fold_id":range(5)})
    folds.to_csv(tmp_path/"folds.tsv",sep="\t",index=False)
    pd.DataFrame({"cancer_id":["BRCA"],"patient_id":[patients[0]],"variant_id":["v1"],
                  "source_reference_build":["GRCh37"]}).to_parquet(tmp_path/"events.parquet",index=False)
    pd.DataFrame({"variant_id":["v1"],"chromosome_grch38":["1"],"start0_grch38":[205],
                  "end0_grch38":[206],"strand_liftover":["+"],"liftover_available":[True],
                  "failure_reason":[None]}).to_parquet(tmp_path/"lifted.parquet",index=False)
    folds.assign(mutation_assay_available=[True,True,False,False,False],locus_coverage_available=False).to_parquet(
        tmp_path/"crosswalk.parquet",index=False)
    for label in ["mc3","chain"]:
        (tmp_path/label).write_text("fixture source")
    monkeypatch.setattr(script,"EXPECTED_MC3_SHA256",sha256(tmp_path/"mc3"))
    monkeypatch.setattr(script,"EXPECTED_CHAIN_SHA256",sha256(tmp_path/"chain"))
    monkeypatch.setattr(sys,"argv",["materialize","--mc3",str(tmp_path/"mc3"),
        "--variant-events",str(tmp_path/"events.parquet"),"--lifted-coordinates",str(tmp_path/"lifted.parquet"),
        "--sample-crosswalk",str(tmp_path/"crosswalk.parquet"),"--chain",str(tmp_path/"chain"),
        "--peak-set",str(tmp_path/"peaks.tsv"),"--topology-links",str(links),
        "--patient-folds",str(tmp_path/"folds.tsv"),"--output",str(tmp_path/"mutation")])
    assert script.main()==0
    path=tmp_path/"mutation/patient_lncrna_distal_mutation.parquet"
    result=pd.read_parquet(path)
    assert result.loc[result.patient_id.eq(patients[0]) & result.lncrna_id.eq("LNC:L1"),"distal_variant_count"].iloc[0]==1
    assert result.loc[result.patient_id.eq(patients[1]),"distal_variant_count"].eq(0).all()
    assert result.loc[result.patient_id.eq(patients[2]),"distal_variant_count"].isna().all()
    assert not result.distal_mutation_locus_callable.any()
    lineage=feature_topology_lineage(path,kind="distal_mutation")
    assert lineage["links_sha256"]==sha256(links)
    assert lineage["predictive_evaluation_eligible"] is True
    result.loc[0,"distal_variant_count"]=99
    result.to_parquet(path,index=False)
    with pytest.raises(ValueError,match="differs"):
        feature_topology_lineage(path,kind="distal_mutation")
