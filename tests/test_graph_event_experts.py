from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from cc_hhgt_v26.evidence_transformer import (
    CAT_FIELDS,
    NUM_FIELDS,
    CandidateDataset,
    EvidenceStore,
    EvidenceTransformer,
    VocabularyBundle,
    collate_candidates,
)
from cc_hhgt_v26.graph_stacking import train_graph_stacker


def test_graph_stacker_masks_missing_expert():
    rng = np.random.default_rng(1)
    rows = []
    cancers = [f"C{i}" for i in range(10)]
    for i in range(800):
        label = int(rng.random() < 0.25)
        signal = 0.65 if label else 0.20
        rows.append({
            "cancer_id": cancers[i % len(cancers)],
            "lncrna_id": f"L{i}",
            "pathway_family_id": f"P{i % 19}",
            "proxy_label": label,
            "rgcn_probability": np.clip(signal + rng.normal(0, 0.12), 0.01, 0.99),
            "hgt_probability": np.clip(signal + rng.normal(0, 0.18), 0.01, 0.99),
            "cc_hhgt_probability": np.clip(signal + rng.normal(0, 0.15), 0.01, 0.99),
            "rgcn_probability_sd": 0.03,
            "hgt_probability_sd": 0.04,
            "cc_hhgt_probability_sd": 0.03,
            "rgcn_n_seeds": 3,
            "hgt_n_seeds": 3,
            "cc_hhgt_n_seeds": 3,
            "rgcn_seed_complete": 1,
            "hgt_seed_complete": 1,
            "cc_hhgt_seed_complete": 1,
            "rgcn_available": 1,
            "hgt_available": 1,
            "cc_hhgt_available": 1,
        })
    frame = pd.DataFrame(rows)
    frame.loc[:49, "hgt_probability"] = np.nan
    result = train_graph_stacker(frame, epochs=4, patience=2, n_folds=5)
    missing = result.full["hgt_probability"].isna()
    assert (result.full.loc[missing, "graph_weight_hgt"].abs() < 1e-6).all()
    assert result.full["graph_ensemble_probability"].notna().all()


def test_event_transformer_separates_direct_and_indirect():
    events = pd.DataFrame([
        {
            "event_id": "E1", "cancer_id": "PAN_CANCER", "lncrna_id": "L1", "pathway_family_id": "P1",
            "route_type": "indirect_interaction_route", "source_database": "NPInter", "pmid": "1",
            "experiment_family": "physical_binding", "relation_type": "binding", "direction": "unknown",
            "tissue": "lung", "cell_line": "A549", "species": "human", "manual_review_status": "reviewed",
            "evidence_level": "indirect", "partner_type": "protein", "event_weight": 1.0, "fdr_score": 0.7,
            "n_independent_pmids": 1, "n_independent_events": 1, "is_experimental": 1, "is_predicted": 0,
            "direct_target_evidence": 0,
        },
        {
            "event_id": "E2", "cancer_id": "LUAD", "lncrna_id": "L1", "pathway_family_id": "P1",
            "route_type": "direct_literature", "source_database": "Manual", "pmid": "2",
            "experiment_family": "knockdown", "relation_type": "direct_pathway_assertion", "direction": "positive",
            "tissue": "lung", "cell_line": "A549", "species": "human", "manual_review_status": "reviewed",
            "evidence_level": "direct", "partner_type": "pathway", "event_weight": 1.0, "fdr_score": 0.0,
            "n_independent_pmids": 1, "n_independent_events": 1, "is_experimental": 1, "is_predicted": 0,
            "direct_target_evidence": 1,
        },
    ])
    vocab = VocabularyBundle.fit(events)
    store = EvidenceStore(events, vocab)
    candidates = pd.DataFrame([{"cancer_id": "LUAD", "lncrna_id": "L1", "pathway_family_id": "P1", "label": 1.0}])
    dataset = CandidateDataset(candidates, store, max_events=4)
    batch = collate_candidates([dataset[0]], store, 4)
    model = EvidenceTransformer([len(vocab.values[field]) for field in CAT_FIELDS], len(NUM_FIELDS), d_model=32)
    output = model(batch)
    assert bool(output["indirect_available"][0])
    assert bool(output["direct_available"][0])
    assert batch["mask_indirect"].sum().item() == 1
    assert batch["mask_direct"].sum().item() == 1
    assert torch.isfinite(batch["direct_quality_target"]).all()
    assert 0 < float(batch["direct_quality_target"][0]) <= 1


def test_event_store_pan_cancer_and_local_merge():
    rows = []
    for index, cancer in enumerate(["PAN_CANCER", "LUAD"]):
        row = {field: "x" for field in CAT_FIELDS}
        row.update({
            "event_id": f"E{index}", "cancer_id": cancer, "lncrna_id": "L1", "pathway_family_id": "P1",
            "pmid": str(index), "event_weight": 1.0, "fdr_score": 0.0,
            "n_independent_pmids": 1, "n_independent_events": 1, "is_experimental": 1, "is_predicted": 0,
            "direct_target_evidence": 0,
        })
        rows.append(row)
    events = pd.DataFrame(rows)
    vocab = VocabularyBundle.fit(events)
    store = EvidenceStore(events, vocab)
    assert len(store.indices_for("LUAD", "L1", "P1")) == 2
    assert len(store.indices_for("BRCA", "L1", "P1")) == 1
