from __future__ import annotations

import pandas as pd

from cc_hhgt_v26.evidence_transformer import (
    UNKNOWN,
    EvidenceStore,
    VocabularyBundle,
    clean_evidence_frame,
)


def test_fold_local_vocab_maps_unseen_clean_context_to_unk_and_removes_constants() -> None:
    train = clean_evidence_frame(
        pd.DataFrame(
            {
                "tissue": ["Breast / Tumor", "Colon Tumor"],
                "cell_line": ["MCF-7", "HCT 116"],
                "species": ["human", "human"],
                "cancer_id": ["BRCA", "COAD"],
                "lncrna_id": ["L1", "L2"],
                "pathway_family_id": ["P1", "P2"],
            }
        )
    )
    test = clean_evidence_frame(
        pd.DataFrame(
            {
                "tissue": ["Kidney_tumor"],
                "cell_line": ["786/O"],
                "species": ["human"],
                "cancer_id": ["KIRP"],
                "lncrna_id": ["L3"],
                "pathway_family_id": ["P3"],
            }
        )
    )
    vocab = VocabularyBundle.fit(
        train, drop_constant_fields=True, merge_redundant_fields=True
    )
    assert "species" not in (vocab.fields or [])
    combined = pd.concat([train, test], ignore_index=True)
    store = EvidenceStore(combined, vocab)
    for field in ("tissue", "cell_line"):
        position = (vocab.fields or []).index(field)
        assert store.cat[-1, position] == vocab.mapping(field)[UNKNOWN]
