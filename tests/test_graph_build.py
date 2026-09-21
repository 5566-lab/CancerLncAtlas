import pandas as pd

from cc_hhgt.graph_build import _edge_frame
from cc_hhgt.interaction_context import apply_strict_cancer_context

def test_scalar_source_database_is_broadcast():
    frame=_edge_frame(['a','b'],['c','d'],'gene','gene','rel',source_database='STRING')
    assert frame.source_database.tolist()==['STRING','STRING']


def test_edge_frame_marks_context_per_row_not_per_column_argument():
    frame = _edge_frame(
        ["a", "b"], ["c", "d"], "lncRNA", "gene", "rel",
        cancer_id=pd.Series([pd.NA, "BRCA"], dtype="string"),
    )
    assert frame.is_context_specific.tolist() == [False, True]


def test_strict_interaction_context_never_globalizes_contextual_rows():
    dim = pd.DataFrame({"cancer_id": ["BRCA", "CESC"], "english_name": ["Breast cancer", "Cervical cancer"]})
    raw = pd.DataFrame(
        {
            "disease_raw": ["", "", ""],
            "tissue": ["", "HeLa", "unknown primary culture"],
            "cell_line": ["", "HeLa", "unknown primary culture"],
        }
    )
    retained, audit = apply_strict_cancer_context(raw, dim)
    assert len(retained) == 2
    assert retained.cancer_id.isna().sum() == 1
    assert retained.cancer_id.dropna().tolist() == ["CESC"]
    assert audit["context_specific_lost_to_global"] == 0
    assert audit["excluded_unmapped_or_ambiguous_context_rows"] == 1
