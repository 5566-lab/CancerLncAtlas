from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd

from cc_hhgt.v32.genomic_training import (
    ASSAY_CALLABLE_SENTINEL,
    build_exact_pathway_calls,
    candidate_statistics,
    exact_pathway_callable_entities,
    normalise_cnv_calls,
    normalise_cnv_entity_coverage,
)
from cc_hhgt.v32.pancancer_cnv import (
    GisticPreparationConfig,
    prepare_pancancer_gistic_bundle,
)


def test_sparse_gistic_bundle_preserves_static_entity_callability(tmp_path: Path) -> None:
    folds = pd.DataFrame(
        [
            {"cancer_id": "BRCA", "sample_id": "TCGA-AA-0001-01A", "patient_id": "TCGA-AA-0001", "patient_fold_id": 0},
            {"cancer_id": "BRCA", "sample_id": "TCGA-AA-0003-01A", "patient_id": "TCGA-AA-0003", "patient_fold_id": 1},
            {"cancer_id": "COAD", "sample_id": "TCGA-BB-0002-01A", "patient_id": "TCGA-BB-0002", "patient_fold_id": 1},
        ]
    )
    folds_path = tmp_path / "folds.tsv"
    folds.to_csv(folds_path, sep="\t", index=False)
    intervals = pd.DataFrame(
        [
            {"entity_type": "gene", "entity_id": "ENSG1", "chromosome": "1", "start": 1, "end": 2},
            {"entity_type": "lncrna", "entity_id": "LNC:ENSL1", "chromosome": "1", "start": 3, "end": 4},
        ]
    )
    intervals_path = tmp_path / "intervals.parquet"
    intervals.to_parquet(intervals_path, index=False)
    membership = pd.DataFrame([{"pathway_id": "P1", "gene_id": "ENSG1"}])
    membership_path = tmp_path / "membership.parquet"
    membership.to_parquet(membership_path, index=False)
    candidates = pd.DataFrame(
        [
            {"cancer_id": cancer, "lncrna_id": "LNC:ENSL1", "pathway_id": "P1"}
            for cancer in ("BRCA", "COAD")
        ]
    )
    candidates_path = tmp_path / "candidates.parquet"
    candidates.to_parquet(candidates_path, index=False)
    gtf_path = tmp_path / "gencode.gtf.gz"
    with gzip.open(gtf_path, "wt") as handle:
        handle.write('1\ttest\tgene\t1\t2\t.\t+\t.\tgene_id "ENSG1.1"; gene_name "GENE1"; gene_type "protein_coding";\n')
        handle.write('1\ttest\tgene\t3\t4\t.\t+\t.\tgene_id "ENSL1.1"; gene_name "LNC1"; gene_type "lncRNA";\n')
    gistic_path = tmp_path / "gistic.txt.gz"
    with gzip.open(gistic_path, "wt") as handle:
        handle.write("Gene Symbol\tLocus ID\tCytoband\tTCGA-AA-0001-01A-X\tTCGA-AA-0003-01A-X\tTCGA-BB-0002-01A-X\n")
        handle.write("GENE1\t1\t1p\t1\t0\t0\n")
        handle.write("LNC1\t2\t1p\t1\t0\t-2\n")

    output = tmp_path / "bundle"
    result = prepare_pancancer_gistic_bundle(
        gistic_path=gistic_path,
        gencode_gtf_path=gtf_path,
        patient_folds_path=folds_path,
        entity_intervals_path=intervals_path,
        pathway_membership_path=membership_path,
        candidates_path=candidates_path,
        output_root=output,
        run_id="v32-routed-cnv-candidate-test",
        config=GisticPreparationConfig(min_patients_per_cancer=1, parquet_row_group_size=2),
    )
    assert result["status"] == "SUCCESS"
    audit = json.loads((output / "AUDIT.json").read_text())
    assert audit["sample_selection"]["covered_cancers"] == ["BRCA", "COAD"]
    assert audit["covered_gene_entities"] == 1
    assert audit["covered_lncrna_entities"] == 1
    gene_raw = pd.read_parquet(output / "cnv_gene_sparse_calls.parquet")
    lnc_raw = pd.read_parquet(output / "cnv_lncrna_sparse_calls.parquet")
    assert set(gene_raw.loc[gene_raw.entity_id.eq(ASSAY_CALLABLE_SENTINEL), "patient_id"]) == {
        "TCGA-AA-0001", "TCGA-AA-0003", "TCGA-BB-0002"
    }
    assert not ((gene_raw.entity_id.eq("ENSG1")) & gene_raw.cnv_value.eq(0)).any()

    coverage = normalise_cnv_entity_coverage(
        pd.read_parquet(output / "cnv_entity_coverage.parquet")
    )
    path_coverage = exact_pathway_callable_entities(membership, coverage["gene"])
    gene_calls = normalise_cnv_calls(gene_raw, entity_kind="gene", event_threshold=0.3)
    lnc_calls = normalise_cnv_calls(lnc_raw, entity_kind="lncrna", event_threshold=0.3)
    pathway_calls = build_exact_pathway_calls(gene_calls, membership)
    stats = candidate_statistics(
        candidates.iloc[[0]],
        lnc_calls.loc[lnc_calls.cancer_id.eq("BRCA")],
        pathway_calls.loc[pathway_calls.cancer_id.eq("BRCA")],
        ["TCGA-AA-0001", "TCGA-AA-0003"],
        modality="cnv",
        min_pair_callable=2,
        lncrna_callable_entities=coverage["lncrna"]["BRCA"],
        pathway_callable_entities=path_coverage["BRCA"],
    )
    assert stats.available.tolist() == [True]
