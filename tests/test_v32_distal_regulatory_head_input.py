from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.materialize_v32_distal_regulatory_head_input as materialize


def test_materializer_preserves_typed_missingness_and_patient_keys(
    tmp_path: Path, monkeypatch
) -> None:
    expression_root = tmp_path / "expression"
    expression_dir = expression_root / "cancer_id=BRCA"
    expression_dir.mkdir(parents=True)
    patients = [f"TCGA-AA-{index:04d}" for index in range(10)]
    lncrnas = ["LNC:L1", "LNC:L2"]
    expression = pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "patient_id": patient,
                "lncrna_id": lnc,
                "logcpm": float(index + (lnc == "L2")),
            }
            for index, patient in enumerate(patients)
            for lnc in lncrnas
        ]
    )
    expression.to_parquet(expression_dir / "part-0.parquet", index=False)

    mutation = pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "patient_id": patient,
                "patient_fold_id": index % 5,
                "lncrna_id": lnc,
                "distal_mutation_burden": float(index % 2),
                "distal_mutation_burden__available": True,
            }
            for index, patient in enumerate(patients)
            for lnc in lncrnas
        ]
    )
    mutation_path = tmp_path / "mutation.parquet"
    mutation.to_parquet(mutation_path, index=False)

    atac_root = tmp_path / "atac"
    atac_root.mkdir()
    # The real DataS7-derived ATAC files use bare ENSG IDs whereas formal
    # expression/mutation keys use the LNC: namespace.
    atac = pd.DataFrame({"gene_id": ["L1"], **{patient: [0.2] for patient in patients[:5]}})
    with gzip.open(atac_root / "BRCA.distal_accessibility.tsv.gz", "wt") as handle:
        atac.to_csv(handle, sep="\t", index=False)

    methylation = pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "patient_id": patient,
                "lncrna_id": "LNC:L1",
                "promoter_methylation_beta": 0.4,
                "promoter_methylation_beta__available": True,
                "distal_methylation_beta": np.nan,
                "distal_methylation_beta__available": False,
            }
            for patient in patients
        ]
    )
    methylation_path = tmp_path / "methylation.parquet"
    methylation.to_parquet(methylation_path, index=False)

    cnv = pd.DataFrame(
        [
            {
                "cancer_id": "BRCA",
                "patient_id": patient,
                "lncrna_id": lnc,
                "cnv_value": 0.1,
                "cnv_callable": True,
            }
            for patient in patients
            for lnc in lncrnas
        ]
    )
    cnv_path = tmp_path / "cnv.parquet"
    cnv.to_parquet(cnv_path, index=False)
    purity = pd.DataFrame(
        {
            "cancer_id": ["BRCA"] * len(patients),
            "patient_id": patients,
            "purity": [0.7] * len(patients),
        }
    )
    purity_path = tmp_path / "purity.parquet"
    purity.to_parquet(purity_path, index=False)
    output = tmp_path / "assembled"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize",
            "--formal-expression-root", str(expression_root),
            "--distal-mutation", str(mutation_path),
            "--distal-atac-root", str(atac_root),
            "--methylation", str(methylation_path),
            "--local-cnv", str(cnv_path),
            "--purity-covariates", str(purity_path),
            "--output", str(output),
            "--cancers", "BRCA",
        ],
    )
    assert materialize.main() == 0
    assembled = pd.read_parquet(output / "cancer_id=BRCA" / "part-0.parquet")
    assert len(assembled) == 20
    assert not assembled.duplicated(["cancer_id", "patient_id", "lncrna_id"]).any()
    assert assembled.distal_methylation_beta.isna().all()
    assert not assembled.distal_methylation_beta__available.any()
    assert assembled.loc[assembled.lncrna_id.eq("LNC:L2"), "promoter_methylation_beta"].isna().all()
    assert assembled.loc[
        assembled.lncrna_id.eq("LNC:L1"), "atac_distal_accessibility__available"
    ].sum() == 5
    audit = json.loads((output / "AUDIT.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["atac_lncrna_namespace_normalized_to_formal_ids"] is True
