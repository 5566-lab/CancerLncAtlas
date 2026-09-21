import json
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.v32.cnv_only_training_v2 import (
    CNV_DOMAIN_FEATURES,
    DirectionalCompactCNV,
)


def _write_store(root: Path) -> None:
    root.mkdir()
    (root / "patient_ids.json").write_text(json.dumps(["P1", "P2", "P3"]))
    (root / "lncrna_ids.json").write_text(json.dumps(["L1"]))
    (root / "pathway_ids.json").write_text(json.dumps(["PW1"]))
    arrays = {
        "lncrna_signed_mean.npy": np.array([[0.5], [-0.6], [np.nan]], np.float32),
        "lncrna_callable.npy": np.array([[1], [1], [0]], np.uint8),
        "lncrna_amplification.npy": np.array([[1], [0], [0]], np.uint8),
        "lncrna_deletion.npy": np.array([[0], [1], [0]], np.uint8),
        "pathway_signed_mean.npy": np.array([[0.3], [-0.2], [0.1]], np.float32),
        "pathway_callable.npy": np.array([[1], [1], [1]], np.uint8),
        "pathway_amplification.npy": np.array([[1], [0], [0]], np.uint8),
        "pathway_deletion.npy": np.array([[0], [1], [0]], np.uint8),
    }
    for name, value in arrays.items():
        np.save(root / name, value, allow_pickle=False)
    (root / "SUCCESS.json").write_text(json.dumps({
        "status": "SUCCESS", "signed_values_preserved": True,
        "amplification_deletion_separate": True,
        "absolute_burden_used_as_continuous": False,
    }))


def test_directional_features_keep_amp_and_deletion_separate(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _write_store(root)
    store = DirectionalCompactCNV(root)
    pairs = pd.DataFrame({"lncrna_id": ["L1"], "pathway_id": ["PW1"]})
    stats = store.candidate_statistics(pairs, ["P1", "P2", "P3"], 2)
    assert stats.cnv_available.tolist() == [True]
    assert stats.lncrna_local_available.tolist() == [True]
    assert stats.pathway_available.tolist() == [True]
    values = dict(zip(CNV_DOMAIN_FEATURES, stats.domain[0], strict=True))
    assert values["lncrna_signed_mean"] < 0
    assert values["lncrna_amplification_fraction"] == 0.5
    assert values["lncrna_deletion_fraction"] == 0.5
    assert values["pathway_amplification_fraction"] == 0.5
    assert values["pathway_deletion_fraction"] == 0.5


def test_unavailable_is_not_imputed_to_available(tmp_path: Path) -> None:
    root = tmp_path / "store"
    _write_store(root)
    store = DirectionalCompactCNV(root)
    pairs = pd.DataFrame({"lncrna_id": ["L1", "missing"], "pathway_id": ["PW1", "PW1"]})
    stats = store.candidate_statistics(pairs, ["P3"], 1)
    assert stats.cnv_available.tolist() == [False, False]
    assert stats.reasons[0] == "CNV_INSUFFICIENT_PAIR_CALLABILITY"
    assert stats.reasons[1] == "CNV_LNCRNA_UNAVAILABLE"
