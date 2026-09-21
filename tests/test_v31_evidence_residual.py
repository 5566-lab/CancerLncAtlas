from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from cc_hhgt.v31_evidence_residual import (
    COUNT_FEATURES,
    GLOBAL_FEATURES,
    EvidenceCandidateDataset,
    GlobalFeatureTransform,
    build_candidate_event_records,
    count_crossfit_offsets,
    encode_evidence_events,
    fit_evidence_model,
    prepare_evidence_candidates,
)
from cc_hhgt_v26.evidence_transformer import CAT_FIELDS, NUM_FIELDS, VocabularyBundle


def test_evidence_residual_runner_imports_candidate_keys() -> None:
    path = Path(__file__).parents[1] / "scripts" / "91_run_v31_evidence_residual.py"
    spec = importlib.util.spec_from_file_location("v31_evidence_residual_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert tuple(module.KEYS) == ("cancer_id", "lncrna_id", "pathway_family_id")
    assert module.MIN_VALIDATION_DELTA_AUPRC == 0.01
    source = path.read_text(encoding="utf-8")
    assert "validation_delta >= MIN_VALIDATION_DELTA_AUPRC" in source
    assert "off_invariant_evaluated_when_module_admitted" in source


def _events() -> pd.DataFrame:
    rows = []
    for index in range(12):
        row = {
            "event_id": f"E{index}",
            "cancer_id": "PAN_CANCER",
            "lncrna_id": f"L{index % 6}",
            "pathway_family_id": "P1",
            "pmid": f"PMID{index // 2}",
            "source_database": "source_a" if index % 2 else "source_b",
            "experiment_family": "binding" if index % 2 else "perturbation",
            "partner_id": f"G{index % 3}",
            "relation_type": "association",
            "direction": "positive" if index % 2 else "negative",
            "tissue": "Breast / Tumor" if index % 2 else "Colon Tumor",
            "cell_line": "MCF-7" if index % 2 else "HCT 116",
            "source_file": "source.tsv",
            "event_weight": 0.1 + index / 20,
            "is_experimental": 1,
            "is_predicted": 0,
            "route_type": "indirect",
            "species": "human",
            "manual_review_status": "reviewed",
            "evidence_level": "indirect",
            "partner_type": "gene",
            "fdr_score": 0.0,
            "n_independent_pmids": 1,
            "n_independent_events": 1,
            "direct_target_evidence": 0,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _historical_candidates() -> pd.DataFrame:
    rows = []
    fractions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    for cancer in ("BRCA", "COAD", "KIRP"):
        for index, fraction in enumerate(fractions):
            rows.append(
                {
                    "cancer_id": cancer,
                    "lncrna_id": f"L{index}",
                    "pathway_family_id": "P1",
                    "label": fraction,
                }
            )
    return pd.DataFrame(rows)


def test_evidence_label_and_parent_event_contracts() -> None:
    candidates, audit = prepare_evidence_candidates(
        _historical_candidates(), cancers=["BRCA", "COAD", "KIRP"]
    )
    assert audit["replicate_k_values"] == [0, 1, 2, 3, 4, 5]
    assert candidates.groupby("lncrna_id").evidence_fold.nunique().eq(1).all()
    records, events, event_audit = build_candidate_event_records(
        candidates, _events(), max_events=4
    )
    assert len(records) == len(candidates)
    assert event_audit["outcome_used_for_event_selection"] is False
    assert records.deduplicated_event_count.le(records.raw_event_count).all()
    assert all(column in records for column in GLOBAL_FEATURES)
    assert len(events) <= len(_events())


def test_count_offsets_and_zero_initialized_residual() -> None:
    candidates, _ = prepare_evidence_candidates(
        _historical_candidates(), cancers=["BRCA", "COAD", "KIRP"]
    )
    records, events, _ = build_candidate_event_records(candidates, _events(), max_events=4)
    # The strict lnc grouping leaves three cancer rows per inner fold group.
    offsets = count_crossfit_offsets(records, c_value=1.0, seed=11)
    assert np.isfinite(offsets).all()
    train = records.iloc[:12].reset_index(drop=True)
    validation = records.iloc[12:].reset_index(drop=True)
    vocab_indices = sorted(
        {int(index) for values in train.selected_event_indices for index in values}
    )
    vocab = VocabularyBundle.fit(
        events.iloc[vocab_indices],
        drop_constant_fields=True,
        merge_redundant_fields=True,
    )
    encoded = encode_evidence_events(events, vocab)
    transform = GlobalFeatureTransform.fit(train)
    train_dataset = EvidenceCandidateDataset(
        train,
        transform.transform(train),
        offsets[: len(train)],
    )
    validation_dataset = EvidenceCandidateDataset(
        validation,
        transform.transform(validation),
        offsets[len(train) :],
    )
    fit = fit_evidence_model(
        train_dataset,
        validation_dataset,
        encoded,
        vocab,
        architecture="DEEPSETS_MEAN",
        mode="residual",
        residual_shrinkage_lambda=0.1,
        seed=7,
        epochs=2,
        patience=1,
        batch_size=6,
        max_events=4,
        device="cpu",
    )
    assert fit.initialization_max_abs_error <= 1e-7
    assert fit.history
