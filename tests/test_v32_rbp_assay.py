"""Phase 1 tests: deterministic RBP/lncRNA-protein assay taxonomy.

These tests encode the non-negotiable contracts of ``cc_hhgt.v32.rbp_assay``:

* the taxonomy is deterministic and closed;
* fine-grained assay identity survives the coarse family collapse;
* the original ``experiment_raw`` string is never altered;
* unclassifiable input fails closed and never masquerades as a binding claim;
* computational predictions can never be relabelled as experimental evidence.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cc_hhgt.v32 import rbp_assay
from cc_hhgt.v32.rbp_assay import (
    ASSAY_SUBTYPES,
    GRAPH_ASSAY_CLASSES,
    GRAPH_ASSAY_RELATION_TYPES,
    classify_assay,
    classify_frame,
    graph_assay_relation_type,
    legacy_experiment_family,
    taxonomy_table,
)


# ---------------------------------------------------------------------------
# Determinism and closure
# ---------------------------------------------------------------------------


def test_taxonomy_is_deterministic_across_repeated_calls() -> None:
    corpus = [
        "eCLIP",
        "RIP-seq",
        "PAR-CLIP",
        "ChIRP",
        "luciferase reporter",
        "siRNA knockdown",
        "",
        "some unknown wet-lab procedure",
    ]
    for value in corpus:
        first = classify_assay(value)
        for _ in range(50):
            assert classify_assay(value) == first


def test_all_emitted_values_are_inside_closed_vocabularies() -> None:
    corpus = [
        "eCLIP", "iCLIP", "PAR-CLIP", "HITS-CLIP", "CLIP", "RIP", "ChIRP", "ChART",
        "RAP-MS", "RNA pull-down", "EMSA", "co-IP", "immunoprecipitation",
        "siRNA knockdown", "CRISPR knockout", "luciferase reporter",
        "qPCR", "RNA-seq", "microarray", "western blot",
        "predicted by sequence", "in silico inference",
        "", "???", "totally unrecognised method",
    ]
    for value in corpus:
        result = classify_assay(value)
        assert result.assay_subtype in ASSAY_SUBTYPES, value
        assert result.graph_assay_class in GRAPH_ASSAY_CLASSES, value


def test_taxonomy_table_is_complete_and_versioned() -> None:
    table = taxonomy_table()
    assert set(table.taxonomy_version) == {rbp_assay.TAXONOMY_VERSION}
    # every graph assay class must be reachable from at least one row
    assert set(table.graph_assay_class) == set(GRAPH_ASSAY_CLASSES)


# ---------------------------------------------------------------------------
# Fine-grained identity survives the coarse collapse  (plan Phase 1 core)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "left, right",
    [
        ("eCLIP", "RIP-seq"),
        ("eCLIP", "ChIRP"),
        ("eCLIP", "PAR-CLIP"),
        ("eCLIP", "RNA pull-down"),
        ("RIP", "ChIRP"),
        ("PAR-CLIP", "HITS-CLIP"),
        ("EMSA", "co-IP"),
    ],
)
def test_distinct_assays_get_distinct_subtypes(left: str, right: str) -> None:
    a = classify_assay(left)
    b = classify_assay(right)
    assert a.assay_subtype != b.assay_subtype
    assert a.graph_assay_class != b.graph_assay_class or a.assay_subtype != b.assay_subtype


def test_eclip_and_rip_differ_in_both_subtype_and_graph_class() -> None:
    eclip = classify_assay("eCLIP")
    rip = classify_assay("RIP-seq")
    assert eclip.assay_subtype == "eclip"
    assert rip.assay_subtype == "rip"
    assert eclip.graph_assay_class == "eclip"
    assert rip.graph_assay_class == "rip"
    assert eclip.graph_assay_class != rip.graph_assay_class


def test_same_physical_binding_family_does_not_erase_subtype() -> None:
    """The historical collapse point: family is identical, subtype must not be."""

    pairs = [
        ("eCLIP", "RIP-seq"),
        ("eCLIP", "ChIRP"),
        ("PAR-CLIP", "EMSA"),
        ("iCLIP", "RAP-MS"),
    ]
    for left, right in pairs:
        a = classify_assay(left)
        b = classify_assay(right)
        assert a.experiment_family == b.experiment_family == "physical_binding"
        assert a.assay_subtype != b.assay_subtype
        assert a.graph_assay_class != b.graph_assay_class


def test_clip_variants_are_distinguished_from_generic_clip() -> None:
    assert classify_assay("eCLIP").assay_subtype == "eclip"
    assert classify_assay("PAR-CLIP").assay_subtype == "par_clip"
    assert classify_assay("iCLIP").assay_subtype == "iclip"
    assert classify_assay("HITS-CLIP").assay_subtype == "hits_clip"
    assert classify_assay("CLIP").assay_subtype == "clip_unspecified"
    # every variant collapses to other_clip except eCLIP, which keeps its own class
    assert classify_assay("PAR-CLIP").graph_assay_class == "other_clip"
    assert classify_assay("eCLIP").graph_assay_class == "eclip"


def test_rna_capture_family_is_grouped_but_subtyped() -> None:
    for text in ("ChIRP", "ChART", "RAP-MS", "RNA pull-down"):
        result = classify_assay(text)
        assert result.graph_assay_class == "rna_capture"
    # subtypes stay distinct inside the group
    assert classify_assay("ChIRP").assay_subtype == "chirp"
    assert classify_assay("ChART").assay_subtype == "chart"
    assert classify_assay("RAP-MS").assay_subtype == "rap"
    assert classify_assay("RNA pull-down").assay_subtype == "rna_pulldown"


# ---------------------------------------------------------------------------
# Provenance: experiment_raw is preserved verbatim
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "eCLIP",
        "  eCLIP  ",
        "PAR-CLIP (HepG2)",
        "RIP-Seq / RIP-qPCR",
        "RNA:protein pull-down",
        "LUCIFERASE reporter assay",
        "未知方法",
    ],
)
def test_experiment_raw_is_preserved_verbatim(raw: str) -> None:
    result = classify_assay(raw)
    # only surrounding whitespace may be trimmed; interior content is untouched
    assert result.experiment_raw == raw.strip()
    assert raw.strip() in result.experiment_raw or result.experiment_raw == raw.strip()


def test_experiment_raw_survives_frame_classification() -> None:
    frame = pd.DataFrame(
        {
            "experiment_raw": ["eCLIP", "RIP-seq", ""],
            "is_predicted": [False, False, False],
            "is_experimental": [True, True, True],
        }
    )
    out = classify_frame(frame)
    assert list(out.experiment_raw) == ["eCLIP", "RIP-seq", ""]
    assert list(out.assay_subtype) == ["eclip", "rip", "unspecified"]


# ---------------------------------------------------------------------------
# Fail-closed behaviour
# ---------------------------------------------------------------------------


def test_empty_input_is_unknown_and_not_experimental() -> None:
    for value in ("", "   ", None):
        result = classify_assay(value)
        assert result.assay_subtype == "unspecified"
        assert result.graph_assay_class == "unknown"
        assert result.is_experimental is False
        assert result.is_predicted is False


def test_unrecognised_text_never_claims_a_binding_assay() -> None:
    result = classify_assay("some entirely uncharacterised procedure")
    assert result.assay_subtype == "other_experimental"
    assert result.graph_assay_class == "experimental_unspecified"
    # it must not be promoted into any physical-binding class
    assert result.graph_assay_class not in {"eclip", "other_clip", "rip", "rna_capture", "other_physical"}


def test_unknown_does_not_raise() -> None:
    for value in ("", "   ", None, float("nan"), "???"):
        classify_assay(value)  # must not raise


# ---------------------------------------------------------------------------
# Computational predictions can never become experimental evidence
# ---------------------------------------------------------------------------


def test_explicit_predicted_flag_forces_predicted_class() -> None:
    result = classify_assay("eCLIP", is_predicted=True)
    assert result.assay_subtype == "computational_prediction"
    assert result.graph_assay_class == "predicted"
    assert result.is_predicted is True
    assert result.is_experimental is False


def test_explicit_non_experimental_flag_forces_predicted_class() -> None:
    result = classify_assay("RIP-seq", is_experimental=False)
    assert result.graph_assay_class == "predicted"
    assert result.is_experimental is False


def test_predicted_text_is_classified_as_prediction() -> None:
    for text in ("predicted interaction", "computational inference", "in silico") :
        assert classify_assay(text).graph_assay_class == "predicted"


def test_predicted_flag_beats_experimental_text() -> None:
    """A prediction record must not be laundered into a binding observation."""

    result = classify_assay("CLIP-seq supporting evidence", is_predicted=True)
    assert result.graph_assay_class == "predicted"
    assert result.is_experimental is False


def test_predicted_rows_in_frame_stay_predicted() -> None:
    frame = pd.DataFrame(
        {
            "experiment_raw": ["CLIP", "CLIP"],
            "is_predicted": [True, False],
            "is_experimental": [False, True],
        }
    )
    out = classify_frame(frame)
    assert list(out.graph_assay_class) == ["predicted", "other_clip"]


# ---------------------------------------------------------------------------
# Historical false positives must be gone
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "transcript abundance measured",
        "manuscript curation",
        "description of the method",
        "graph based inference",
        "rapid amplification",
        "trip assay",
    ],
)
def test_historical_substring_false_positives_are_not_physical(text: str) -> None:
    """The legacy regex matched rip inside 'transcript' and rap inside 'graph'."""

    assert legacy_experiment_family(text) == "physical_binding"  # legacy behaviour
    result = classify_assay(text)
    assert result.graph_assay_class != "eclip"
    assert result.assay_subtype not in {"eclip", "rip", "rap"}


def test_legacy_helper_reproduces_the_original_implementation() -> None:
    """Mode A (legacy_generic_binding) must reproduce old semantics exactly."""

    from cc_hhgt.v32.evidence_interaction_rematerialization import experiment_family

    corpus = [
        "eCLIP", "iCLIP", "PAR-CLIP", "HITS-CLIP", "CLIP", "ChIRP", "ChART",
        "RAP-MS", "RIP", "RIP-seq", "RNA pull-down", "pulldown", "EMSA",
        "co-IP", "immunoprecipitation", "IP",
        "siRNA knockdown", "shRNA", "CRISPR", "overexpression", "transfection",
        "luciferase reporter", "reporter assay",
        "qPCR", "RT-PCR", "RNA-seq", "microarray", "western blot",
        "", "   ", "transcript abundance", "description", "graph inference",
        "totally unknown method",
    ]
    for value in corpus:
        assert legacy_experiment_family(value) == experiment_family(value), value


# ---------------------------------------------------------------------------
# Relation typing
# ---------------------------------------------------------------------------


def test_relation_types_are_closed_and_prefixed() -> None:
    for cls in GRAPH_ASSAY_CLASSES:
        rel = graph_assay_relation_type(cls)
        assert rel == f"binds_protein_{cls}"
        assert rel in set(GRAPH_ASSAY_RELATION_TYPES.values())


def test_unknown_relation_type_fails_closed() -> None:
    assert graph_assay_relation_type("not_a_real_class") == "binds_protein_unknown"
    assert graph_assay_relation_type(None) == "binds_protein_unknown"


def test_relation_types_are_unique_per_class() -> None:
    values = list(GRAPH_ASSAY_RELATION_TYPES.values())
    assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# Frame integration
# ---------------------------------------------------------------------------


def test_frame_classification_preserves_index() -> None:
    frame = pd.DataFrame(
        {"experiment_raw": ["eCLIP", "RIP"]}, index=[10, 20]
    )
    out = classify_frame(frame)
    assert list(out.index) == [10, 20]


def test_frame_falls_back_when_raw_column_absent() -> None:
    frame = pd.DataFrame({"experiment_family": ["physical_binding", "functional_perturbation"]})
    out = classify_frame(frame)
    # fallback text has no fine-grained assay, so it must not invent one
    assert list(out.graph_assay_class) == ["other_physical", "experimental_unspecified"]


def test_frame_with_empty_frame_is_safe() -> None:
    out = classify_frame(pd.DataFrame({"experiment_raw": []}))
    assert len(out) == 0
    assert "assay_subtype" in out.columns


# ---------------------------------------------------------------------------
# Regressions found on REAL NPInter5 data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Conserved miRNAs target sites predicted by TargetScan and miRanda "
        "overlap with the AGO CLIP dataset",
        "predicted interaction",
        "in silico inference",
    ],
)
def test_prediction_language_outranks_assay_keywords(text: str) -> None:
    """A row that says it is predicted must not become an observed assay.

    The TargetScan/miRanda string occurs 116,499 times in NPInter5 and merely
    *references* the AGO CLIP dataset.  The historical regex recorded every one
    of them as ``physical_binding``.
    """

    result = classify_assay(text)
    assert result.graph_assay_class == "predicted"
    assert result.is_experimental is False


@pytest.mark.parametrize(
    "text, expected",
    [
        ("easyCLIP", "other_clip"),
        ("fPARCLIP", "other_clip"),
        ("fCLIP", "other_clip"),
        ("GoldCLIP", "other_clip"),
        ("pCLIP", "other_clip"),
        ("seCLIP-Seq", "other_clip"),
        ("PARCLIP", "other_clip"),
        ("MY-CLIP", "other_clip"),
        ("clip seq", "other_clip"),
        ("BrdU-CLIP", "other_clip"),
    ],
)
def test_prefixed_clip_variants_stay_in_a_clip_class(text: str, expected: str) -> None:
    """Real suffix/prefix CLIP variants must not be demoted out of the CLIP family.

    These cover 18,378 NPInter5 rows that a strict left boundary would wrongly
    push to ``experimental_unspecified``, which would be a regression against
    the historical behaviour.
    """

    assert classify_assay(text).graph_assay_class == expected


def test_relaxed_clip_prefix_still_rejects_prose() -> None:
    """The prefix allowance must not resurrect the historical prose matches."""

    for text in ("transcript abundance", "description of the method", "manuscript"):
        assert classify_assay(text).graph_assay_class != "other_clip"

@pytest.mark.parametrize(
    "text",
    [
        "catRAPID",
        "catRAPID//Scan pipeline widely used MATCH algorithm",
        "Scan pipeline widely used MATCH algorithm",
        "RPISeq prediction",
    ],
)
def test_sequence_based_interaction_predictors_are_not_experimental(text: str) -> None:
    """catRAPID and MATCH are sequence-based predictors, not assays.

    Real data: 197,094 lncRNA-RBP rows carry "Scan pipeline widely used MATCH
    algorithm" and 30,276 carry "catRAPID".  If either were treated as an
    experimental binding observation, roughly 65% of the lncRNA-RBP evidence
    would enter the main graph as if it had been measured.
    """

    result = classify_assay(text)
    assert result.graph_assay_class == "predicted"
    assert result.is_experimental is False
    assert result.is_predicted is True