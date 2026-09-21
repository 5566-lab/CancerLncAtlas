"""Phase 2 tests: assay identity must reach the Evidence EventSet features.

The Phase 0 audit proved that ``experiment_raw`` never reached
``EVENT_FEATURE_FIELDS``, so eCLIP and RIP were hashed to the *same* bucket.
These tests pin the fix and, equally importantly, pin the legacy-equivalence
guarantee that makes ablation mode A meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cc_hhgt.v32.evidence_training import (
    EVENT_FEATURE_FIELDS,
    LEGACY_EVENT_FEATURE_FIELDS,
    _event_feature_matrix,
    build_exact_event_bags,
    event_feature_fields,
)


def _members() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pathway_id": ["R-HSA-1"],
            "gene_id": ["ENSG000002"],
            "mapping_status": ["mapped"],
        }
    )


def _binding_rows(records: list[tuple[str, str, bool]]) -> pd.DataFrame:
    """Build interaction rows: (source_record_id, experiment_raw, is_predicted)."""

    rows = []
    for record_id, experiment_raw, is_predicted in records:
        rows.append(
            {
                "interaction_id": record_id,
                "source_record_id": record_id,
                "lncrna_id": "ENSG000001",
                "partner_id": "ENSG000002",
                "cancer_id": "BRCA",
                "source_database": "NPInter",
                "source_dataset": "NPInter-5",
                "pmid": "12345678",
                "relation_type": "binding_or_interaction",
                "experiment_raw": experiment_raw,
                "experiment_family": "physical_binding",
                "direction": "positive",
                "experimental": not is_predicted,
                "is_predicted": is_predicted,
                "physical": True,
            }
        )
    return pd.DataFrame(rows)


def _events_for(records: list[tuple[str, str, bool]]) -> pd.DataFrame:
    interaction = _binding_rows(records)
    empty_evidence = pd.DataFrame(
        columns=[
            "evidence_event_id", "lncrna_id", "partner_id", "pathway_id",
            "cancer_id", "source_database", "source_dataset", "source_record_id",
            "pmid", "experiment_family", "relation_type", "direction",
        ]
    )
    result = build_exact_event_bags(empty_evidence, interaction, _members())
    return result.events


# ---------------------------------------------------------------------------
# The event frame must carry the taxonomy
# ---------------------------------------------------------------------------


def test_event_frame_exposes_the_taxonomy_columns() -> None:
    events = _events_for([("REC-1", "eCLIP", False)])
    for column in ("experiment_raw", "experiment_family", "assay_subtype", "graph_assay_class"):
        assert column in events.columns, column


def test_experiment_raw_is_preserved_verbatim_on_the_event() -> None:
    events = _events_for([("REC-1", "PAR-CLIP (HepG2)", False)])
    assert events.iloc[0].experiment_raw == "PAR-CLIP (HepG2)"
    assert events.iloc[0].assay_subtype == "par_clip"
    assert events.iloc[0].graph_assay_class == "other_clip"


def test_binding_assays_receive_distinct_subtypes_in_the_event_frame() -> None:
    events = _events_for(
        [
            ("REC-1", "eCLIP", False),
            ("REC-2", "RIP-seq", False),
            ("REC-3", "ChIRP", False),
        ]
    )
    subtypes = dict(zip(events.experiment_raw, events.assay_subtype))
    assert subtypes["eCLIP"] == "eclip"
    assert subtypes["RIP-seq"] == "rip"
    assert subtypes["ChIRP"] == "chirp"
    assert len(set(subtypes.values())) == 3


def test_same_family_rows_keep_distinct_graph_classes() -> None:
    events = _events_for([("REC-1", "eCLIP", False), ("REC-2", "RIP-seq", False)])
    assert set(events.experiment_family) == {"physical_binding"}
    assert set(events.graph_assay_class) == {"eclip", "rip"}


# ---------------------------------------------------------------------------
# The core Phase 2 fix: features must differ
# ---------------------------------------------------------------------------


def test_eclip_and_rip_produce_different_eventset_features() -> None:
    """The whole point of Phase 2: the head must be able to tell them apart."""

    events = _events_for([("REC-1", "eCLIP", False), ("REC-2", "RIP-seq", False)])
    matrix = _event_feature_matrix(events, 128, preserve_assay_type=True)
    assert matrix.shape[0] == 2
    assert not np.array_equal(matrix[0], matrix[1]), (
        "eCLIP and RIP produced identical EventSet features"
    )


def test_legacy_mode_collapses_them_exactly_as_before() -> None:
    """Mode A must still be provably the old model."""

    events = _events_for([("REC-1", "eCLIP", False), ("REC-2", "RIP-seq", False)])
    # Under legacy semantics experiment_type resolves to the coarse family, so
    # every hashed field is identical and the two rows must be byte-identical.
    legacy = _event_feature_matrix(events, 128, preserve_assay_type=False)
    assert set(events.experiment_type) == {"physical_binding"}
    assert np.array_equal(legacy[0], legacy[1]), (
        "legacy mode must still collapse eCLIP and RIP to the same feature row"
    )


def test_preserve_assay_type_actually_changes_the_matrix() -> None:
    events = _events_for([("REC-1", "eCLIP", False)])
    legacy = _event_feature_matrix(events, 128, preserve_assay_type=False)
    typed = _event_feature_matrix(events, 128, preserve_assay_type=True)
    assert not np.array_equal(legacy, typed)


def test_legacy_matrix_reproduces_the_historical_field_list_bit_for_bit() -> None:
    """Recompute mode A by hand from the original field list and compare."""

    import hashlib

    events = _events_for([("REC-1", "RNA pull-down", False)])
    produced = _event_feature_matrix(events, 128, preserve_assay_type=False)

    manual = np.zeros((1, 128), dtype=np.float32)
    row = events.to_dict("records")[0]
    for field in LEGACY_EVENT_FEATURE_FIELDS:
        token = f"{field}={str(row.get(field, 'unknown')).lower()}"
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % (128 - 8)
        sign = 1.0 if digest[4] & 1 else -1.0
        manual[0, bucket] += sign
    manual[0, 120] = float(bool(row.get("is_experimental", False)))
    manual[0, 121] = float(bool(row.get("is_computational", False)))
    manual[0, 122] = float(bool(row.get("is_physical", False)))
    manual[0, 123] = 1.0
    manual[0, 124] = 1.0 / 5.0
    manual[0, 125] = float(row.get("route_type") == "DIRECT_EXACT_ASSERTION")
    manual[0, 126] = float(row.get("route_type") == "PARTNER_EXACT_MEMBER")
    manual[0, 127] = 1.0

    assert np.array_equal(produced, manual)


def test_field_lists_are_distinct_and_closed() -> None:
    assert event_feature_fields(preserve_assay_type=False) == LEGACY_EVENT_FEATURE_FIELDS
    assert event_feature_fields(preserve_assay_type=True) == EVENT_FEATURE_FIELDS
    assert "experiment_type" in LEGACY_EVENT_FEATURE_FIELDS
    assert "experiment_type" not in EVENT_FEATURE_FIELDS
    assert "assay_subtype" in EVENT_FEATURE_FIELDS
    assert "experiment_family" in EVENT_FEATURE_FIELDS
    # raw text must never become a hashed feature (unbounded bucket cardinality)
    assert "experiment_raw" not in EVENT_FEATURE_FIELDS
    assert "experiment_raw" not in LEGACY_EVENT_FEATURE_FIELDS


# ---------------------------------------------------------------------------
# Fail-closed contracts still hold end to end
# ---------------------------------------------------------------------------


def test_predicted_rows_are_never_marked_experimental_in_events() -> None:
    events = _events_for([("REC-1", "eCLIP", True)])
    assert bool(events.iloc[0].is_experimental) is False
    assert events.iloc[0].graph_assay_class == "predicted"


def test_unknown_assay_does_not_raise_and_stays_unknown() -> None:
    """With no raw text AND no coarse family, the result must be unknown."""

    interaction = _binding_rows([("REC-1", "", False)]).drop(columns=["experiment_family"])
    empty_evidence = pd.DataFrame(
        columns=[
            "evidence_event_id", "lncrna_id", "partner_id", "pathway_id",
            "cancer_id", "source_database", "source_dataset", "source_record_id",
            "pmid", "experiment_family", "relation_type", "direction",
        ]
    )
    events = build_exact_event_bags(empty_evidence, interaction, _members()).events
    assert events.iloc[0].assay_subtype == "unspecified"
    assert events.iloc[0].graph_assay_class == "unknown"


def test_coarse_family_fallback_recovers_physical_semantics() -> None:
    """Empty raw text but a coarse family present must not fall to 'unknown'.

    The real ``evidence_event`` training input carries ``experiment_family`` and
    not ``experiment_raw`` (see the Phase 0 audit), so this fallback is the path
    that actually executes for that table.
    """

    events = _events_for([("REC-1", "", False)])
    assert events.iloc[0].experiment_raw == ""
    assert events.iloc[0].experiment_family == "physical_binding"
    assert events.iloc[0].graph_assay_class == "other_physical"


def test_unrecognised_assay_never_becomes_a_binding_class() -> None:
    events = _events_for([("REC-1", "some uncharacterised procedure", False)])
    assert events.iloc[0].assay_subtype == "other_experimental"
    assert events.iloc[0].graph_assay_class == "experimental_unspecified"


def test_feature_dimension_guard_still_applies() -> None:
    from cc_hhgt.v32.evidence_training import EvidenceTrainingContractError

    events = _events_for([("REC-1", "eCLIP", False)])
    with pytest.raises(EvidenceTrainingContractError):
        _event_feature_matrix(events, 16, preserve_assay_type=True)
