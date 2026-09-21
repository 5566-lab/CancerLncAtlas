from __future__ import annotations

import pandas as pd

from cc_hhgt.candidates import build_pancancer_lnc_eligibility
from cc_hhgt.common import load_config
from cc_hhgt.v30_samples import build_canonical_sample_outputs


def test_pancancer_lnc_filter_uses_locked_three_cancer_rule() -> None:
    rows = []
    for cancer in [f"C{i:02d}" for i in range(1, 8)]:
        rows.append(
            {
                "cancer_id": cancer,
                "lncrna_id": "LNC:KEEP",
                "bulk_detection_rate": 0.20,
            }
        )
    for cancer in ["C01", "C02", "C03", "C04"]:
        rows.append(
            {
                "cancer_id": cancer,
                "lncrna_id": "LNC:DROP",
                "bulk_detection_rate": 0.90,
            }
        )
    for cancer in ["C01", "C02"]:
        rows.append(
            {
                "cancer_id": cancer,
                "lncrna_id": "LNC:BELOW_3C",
                "bulk_detection_rate": 0.90,
            }
        )
    settings = {
        "bulk_min_detection_rate": 0.10,
        "pancancer_lnc_filter": {
            "enabled": True,
            "within_cancer_min_sample_detection_rate": 0.10,
            "minimum_detected_cancers": 3,
        },
    }
    result = build_pancancer_lnc_eligibility(
        pd.DataFrame(rows), [f"C{i:02d}" for i in range(1, 8)], settings
    ).set_index("lncrna_id")
    assert bool(result.loc["LNC:KEEP", "eligible"])
    assert int(result.loc["LNC:KEEP", "detected_cancers"]) == 7
    assert bool(result.loc["LNC:DROP", "eligible"])
    assert int(result.loc["LNC:DROP", "detected_cancers"]) == 4
    assert not bool(result.loc["LNC:BELOW_3C", "eligible"])
    assert int(result.loc["LNC:BELOW_3C", "detected_cancers"]) == 2


def test_formal_config_is_33_cancers_and_297_tasks(tmp_path) -> None:
    config = load_config("config/model_v3_1_formal_33c.yaml")
    assert config["formal_contract"]["expected_folds"] == 33
    assert config["formal_contract"]["expected_tasks"] == 297
    assert config["analysis_cancers"]["reference_only"] == []
    assert config["cancer_scope"]["reference_only"] == []
    assert config["formal_contract"]["required_formal_cancers"] == ["HNSC", "LGG"]
    assert config["candidate_universe"]["pancancer_lnc_filter"]["minimum_detected_cancers"] == 3
    assert config["candidate_universe"]["pancancer_lnc_filter"]["detection_semantics"] == "logcpm_gt_0"
    assert config["candidate_universe"]["pancancer_lnc_filter"]["expected_eligible_lncrna_ids_full_gdc"] == 4712
    assert config["candidate_universe"]["pancancer_lnc_filter"]["decision_id"] == "PANCANCER_LNCRNA_010_3C_LOCK_20260824"
    assert config["candidate_universe"]["pancancer_lnc_filter"]["decision_status"] == "locked"
    assert config["task_contract"]["pathway"]["role"] == "primary_website_discovery_task"


def test_hnsc_lgg_are_formal_in_canonical_sample_builder() -> None:
    samples = []
    scores = []
    for cancer in ("HNSC", "LGG"):
        for index in range(2):
            patient = f"TCGA-{cancer[:2]}-{index:04d}"
            sample = f"{patient}-01A"
            samples.append(
                {
                    "cancer_id": cancer,
                    "patient_id": patient,
                    "sample_id": sample,
                    "patient_fold_id": index,
                    "split": "train",
                }
            )
            scores.append(
                {
                    "sample_id": sample,
                    "patient_id": patient,
                    "cancer_type": cancer,
                    "sample_type_code": 1,
                    "is_tumor": True,
                    "RNAss": 0.2 + index,
                    "DNAss": 0.3 + index,
                }
            )
    _, _, _, eligibility, audit = build_canonical_sample_outputs(
        pd.DataFrame(samples),
        pd.DataFrame(scores),
        target_states=["RNAss", "DNAss"],
        reference_only=[],
        minimum_observed=2,
    )
    assert set(eligibility.cancer_role) == {"FORMAL"}
    assert eligibility.eligibility.eq("ELIGIBLE").all()
    assert audit["reference_only"] == []
