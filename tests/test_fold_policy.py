import pandas as pd

from cc_hhgt.gnn import localize_cancer_state_edges

def test_fold_context_exclusion_policy():
    edges=pd.DataFrame({
        'cancer_id':['LUAD','BRCA',None],
        'is_context_specific':[True,True,False],
    })
    excluded={'LUAD'}
    context=edges.is_context_specific.fillna(False).astype(bool)
    keep=~(context & edges.cancer_id.astype(str).isin(excluded))
    assert keep.tolist()==[False,True,True]


def test_cancer_state_relation_is_normalized_after_fold_filtering():
    edges = pd.DataFrame(
        {
            "source_canonical_id": ["A", "B", "C"],
            "target_canonical_id": ["S", "S", "S"],
            "relation_type": ["state_profile_requires_fold_localization"] * 3,
            "weight": [1.0] * 3,
            "raw_effect": [0.0, 1.0, 100.0],
            "requires_fold_localization": [True] * 3,
        }
    )
    # C is the held-out cancer and must be removed before normalization.
    localized = localize_cancer_state_edges(edges.loc[edges.source_canonical_id.ne("C")])
    assert localized.relation_type.tolist() == ["state_depleted", "state_enriched"]
    assert localized.fold_local_mean.nunique() == 1
    assert localized.fold_local_mean.iloc[0] == 0.5
