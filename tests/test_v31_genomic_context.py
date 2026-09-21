from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "87_build_v31_genomic_context.py"
SPEC = importlib.util.spec_from_file_location("build_v31_genomic_context", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_lnc_mutation_uses_canonical_patients_and_never_infers_wildtype() -> None:
    canonical = pd.DataFrame(
        {"cancer_id": ["A", "A"], "patient_id": ["P1", "P2"]}
    )
    events = pd.DataFrame(
        {
            "cancer_id": ["A", "A"],
            "patient_id": ["P1", "OUTSIDE"],
            "lncrna_id": ["L1", "L1"],
            "exonic_variant_count": [1, 9],
            "splice_variant_count": [0, 0],
            "promoter_variant_count": [0, 0],
            "recurrent_variant_count": [0, 0],
            "max_vaf": [0.4, 0.9],
            "length_normalized_mutation_burden": [2.0, 9.0],
            "tmb_adjusted_mutation_burden": [1.0, 9.0],
            "locus_coverage_available": [False, False],
            "absence_is_wildtype": [False, False],
        }
    )
    result = MODULE.aggregate_lnc_mutation(events, canonical).iloc[0]
    assert result.genomic_lnc_observed_event_patients == 1
    assert np.isclose(result.genomic_lnc_observed_event_fraction, 0.5)
    assert result.genomic_lnc_absence_wildtype_rows == 0


def test_missing_genomic_key_is_explicitly_unavailable() -> None:
    keys = pd.DataFrame({"cancer_id": ["A", "A"], "lncrna_id": ["L1", "L2"]})
    aggregate = pd.DataFrame(
        {"cancer_id": ["A"], "lncrna_id": ["L1"], "feature": [1.0]}
    )
    result = MODULE._mask_merged_features(
        keys, aggregate, ["cancer_id", "lncrna_id"]
    )
    assert bool(result.loc[result.lncrna_id.eq("L1"), "feature__available"].iloc[0])
    assert not bool(result.loc[result.lncrna_id.eq("L2"), "feature__available"].iloc[0])
