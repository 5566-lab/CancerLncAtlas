from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.materialize_v32_distal_regulatory_head_input as materialize


@pytest.mark.parametrize("diagnostic_only", [False, True])
def test_materializer_preserves_typed_missingness_and_patient_keys(
    tmp_path: Path, monkeypatch, diagnostic_only
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
    if not diagnostic_only:
        import hashlib
        from cc_hhgt.v32.distal_topology import POLICY, sha256
        lineage = {"predictive_evaluation_eligible": True, "policy": POLICY,
                   "coordinate_build": "GRCh38", "links_sha256": "a"*64}
        (tmp_path/"AUDIT.json").write_text(json.dumps({"topology_lineage": lineage,
            "outputs": {name: {"path":str(path), "sha256":sha256(path)} for name,path in
                        [("mutation",mutation_path),("methylation",methylation_path)]}}))
        atac_path=atac_root/"BRCA.distal_accessibility.tsv.gz"
        (atac_root/"DISTAL_ATAC_AUDIT.json").write_text(json.dumps({"topology_lineage":lineage,
            "output_md5":{str(atac_path):hashlib.md5(atac_path.read_bytes()).hexdigest()}}))
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
        ] + (["--allow-cohort-selected-topology"] if diagnostic_only else []),
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
    assert audit["topology_lineage"]["predictive_evaluation_eligible"] is (not diagnostic_only)


def test_atac_aliquot_means_are_order_independent_and_id_collisions_rejected(tmp_path):
    path=tmp_path/"BRCA.distal_accessibility.tsv.gz"
    pd.DataFrame({"gene_id":["L1"],"TCGA-AA-0001-A":[.2],"TCGA-AA-0001-B":[.8]}).to_csv(path,sep="\t",index=False)
    assert materialize._atac_long(tmp_path,"BRCA",{"LNC:L1"}).atac_distal_accessibility.iloc[0]==pytest.approx(.5)
    pd.DataFrame({"gene_id":["L1.1","L1.2"],"TCGA-AA-0001":[.2,.8]}).to_csv(path,sep="\t",index=False)
    with pytest.raises(ValueError,match="collide"):
        materialize._atac_long(tmp_path,"BRCA",{"LNC:L1"})


def test_feature_availability_strings_cannot_be_silently_cast_to_true():
    frame=pd.DataFrame({"test":[.2],"test__available":["False"]})
    with pytest.raises(ValueError,match="boolean"):
        materialize._typed_feature(frame,"test")
