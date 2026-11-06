import pytest
from scripts.audit_tcpa_cptac_common_panel import tcga_patient

def test_patient_barcode_normalization_is_exact_and_rejects_aliquot_truncation():
    assert tcga_patient("TCGA.AA.0001") == "TCGA-AA-0001"
    assert tcga_patient("TCGA-AA-0001") == "TCGA-AA-0001"
    for invalid in ["01BR001", "TCGA-AA-0001-01A", "CPTAC:BRCA:01BR001", "TCGA-AA-001"]:
        with pytest.raises(ValueError):
            tcga_patient(invalid)
