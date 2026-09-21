from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.label_contract import (
    LabelContractError,
    attach_explicit_pathway_labels,
    candidate_manifest_sha256,
    load_label_contract,
    split_manifest_sha256,
    stable_stratified_split,
    validate_candidate_manifest,
    validate_contract_hashes,
    validate_model_target,
    validate_positive_counts,
    validate_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "label_contracts" / "PATHWAY_PROXY_V1.yaml"


def contract():
    return load_label_contract(CONTRACT_PATH)


def manifest(n: int = 100_000, cancer: str = "HNSC") -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "cancer_id": cancer,
            "lncrna_id": [f"L{i // 163}" for i in range(n)],
            "pathway_family_id": [f"P{i % 163}" for i in range(n)],
            "candidate_id": [f"CAND:{cancer}:{i}" for i in range(n)],
            "label_class": np.where(np.arange(n) < 3302, "weak_positive", "unlabeled"),
            "candidate_sampling_seed": 20260726,
            "candidate_sampling_policy": "identity-only deterministic",
            "label_contract_version": "PATHWAY_PROXY_V1",
            "label_contract_sha256": contract()["_contract_sha256"],
        }
    )
    frame = attach_explicit_pathway_labels(frame)
    frame["split"] = stable_stratified_split(frame.candidate_id, frame.association_proxy_label)
    frame["split_seed"] = 20260726
    candidate_hash = candidate_manifest_sha256(frame)
    split_hash = split_manifest_sha256(frame)
    frame["candidate_manifest_sha256"] = candidate_hash
    frame["split_manifest_sha256"] = split_hash
    return frame


def test_contract_sidecar_and_explicit_mapping_pass():
    payload = contract()
    assert payload["target"]["column"] == "association_proxy_label"
    frame = manifest()
    validate_candidate_manifest(frame, payload)
    validate_split_manifest(frame, payload)


def test_row_level_mapping_drift_fails_closed():
    frame = manifest()
    frame.loc[0, "association_proxy_label"] = 0
    with pytest.raises(LabelContractError, match="mapping"):
        validate_candidate_manifest(frame, contract())


def test_ambiguous_model_target_and_hash_drift_fail_closed():
    frame = manifest()
    good = contract()["_contract_sha256"]
    roles = {
        "training_target_contract_sha256": good,
        "evaluation_target_contract_sha256": good,
        "baseline_target_contract_sha256": good,
    }
    labels = validate_model_target(frame, "association_proxy_label", contract(), roles)
    assert int(labels.sum()) == 3302
    with pytest.raises(LabelContractError, match="Ambiguous"):
        validate_model_target(frame.assign(label=labels), "label", contract(), roles)
    roles["baseline_target_contract_sha256"] = "0" * 64
    with pytest.raises(LabelContractError, match="not identical"):
        validate_contract_hashes(roles)


def test_positive_count_gate():
    frames = [manifest(cancer=cancer) for cancer in ("HNSC", "LGG", "UCEC")]
    result = validate_positive_counts(pd.concat(frames, ignore_index=True))
    assert result.gate.eq("PASS").all()
    sparse = pd.concat(frames, ignore_index=True)
    sparse.loc[sparse.cancer_id.eq("UCEC"), "association_proxy_label"] = 0
    with pytest.raises(LabelContractError, match="Positive-count gate failed"):
        validate_positive_counts(sparse)
