import json

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.drug_response_analysis import iter_response_scan, DrugResponseScanConfig


def fixture():
    models = [f"M{i}" for i in range(100)]
    mapping = pd.DataFrame([dict(dataset_id=d, cell_line_id=f"{d}_{m}", canonical_model_id=m, cancer_id="BRCA")
                            for d in ["GDSC", "PRISM"] for m in models])
    expression = pd.DataFrame({"canonical_model_id": models, "lncrna_id": "L1", "expression_value": np.arange(100)%50})
    subtypes = pd.DataFrame({"canonical_model_id": models, "cancer_id": "BRCA", "subtype_system": "PAM50",
        "subtype_id": ["Basal"]*50+["LumA"]*50, "subtype_available": True, "source": "curated_fixture"})
    response = []
    for dataset in ["GDSC", "PRISM"]:
        for i, model in enumerate(models):
            sensitivity = (i%50)*(1 if i < 50 else -1)
            for drug in ["ASSAY:unmapped", "DC:1"]:
                if drug == "DC:1" and i >= 50: continue
                response.append(dict(dataset_id=dataset, cell_line_id=f"{dataset}_{model}", drug_id=drug,
                                     response_value=-sensitivity, higher_is_sensitive=False))
    exact = pd.DataFrame({"cancer_id": ["BRCA", "BRCA", "KICH"], "lncrna_id": ["L1", "Lmissing", "L1"]})
    targets = pd.DataFrame({"drug_id": ["DC:1"], "gene_id": ["G1"]})
    return dict(exact=exact, response=pd.DataFrame(response), expression=expression, mapping=mapping, subtypes=subtypes, targets=targets)


def test_all_assayed_drugs_keep_signed_subtype_effect_and_unique_models():
    outputs = list(iter_response_scan(**fixture(), config=DrugResponseScanConfig(candidate_chunk_rows=3)))
    frame = pd.concat([part for _, part in outputs if not part.empty], ignore_index=True)
    local = frame.loc[frame.lncrna_id.str.endswith("L1") & frame.drug_id.eq("ASSAY:unmapped") & frame.subtype_system.eq("PAM50")].set_index("subtype_id")
    assert local.loc["Basal", "signed_expression_sensitivity_rho"] == pytest.approx(1., abs=1e-6)
    assert local.loc["LumA", "signed_expression_sensitivity_rho"] == pytest.approx(-1., abs=1e-6)
    assert local.unique_cell_models.tolist() == [50, 50]  # two assays do not double the patients
    assert not local.target_annotation_available.any()
    finite_fold_counts = local[[f"cell_fold_{fold}_rho" for fold in range(5)]].notna().sum(axis=1)
    np.testing.assert_array_equal(local.folds_with_same_direction, finite_fold_counts)
    assert (finite_fold_counts > 0).all() and local.folds_with_opposite_direction.eq(0).all()
    missing = frame.loc[frame.subtype_id.eq("LumA") & frame.drug_id.eq("DC:1")]
    assert not missing.association_available.any() and missing.signed_expression_sensitivity_rho.isna().all()
    assert missing.observed_direction.isna().all()
    kich = next(receipt for receipt, _ in outputs if receipt["cancer_id"] == "KICH")
    assert not kich["context_available"] and kich["candidate_rows"] == 0
    assert len(frame) == 12


def test_subtype_annotations_cannot_relabel_cancer_or_multiply_model_identity():
    tables = fixture(); tables["subtypes"].loc[0, "cancer_id"] = "LUAD"
    with pytest.raises(ValueError, match="Subtype cancer"):
        list(iter_response_scan(**tables))
    tables = fixture(); duplicate = tables["subtypes"].iloc[[0]].copy(); duplicate["subtype_id"] = "LumA"
    tables["subtypes"] = pd.concat([tables["subtypes"], duplicate], ignore_index=True)
    with pytest.raises(ValueError, match="conflicting subtypes"):
        list(iter_response_scan(**tables))


def test_command_materializes_results_and_hash_bound_coverage_without_training(tmp_path):
    from scripts.analyze_v32_native_drug_response import main
    args = []
    for name, frame in fixture().items():
        path = tmp_path / (name+".parquet"); frame.to_parquet(path, index=False)
        args += ["--"+name, str(path)]
    output = tmp_path / "analysis"
    assert main([*args, "--output", str(output), "--chunk-rows", "3"]) == 0
    audit = json.loads((output/"AUDIT.json").read_text())
    assert audit["rows"] == 12 and audit["contexts"] == 4
    assert not audit["training_started"] and not audit["requires_target_annotation"]
    coverage = pd.read_parquet(output/"CANCER_SUBTYPE_COVERAGE.parquet")
    assert len(coverage) == 4 and coverage.targetless_assayed_drugs.sum() == 3
    assert len(pd.read_parquet(output/"associations")) == 12
