import pandas as pd

from cc_hhgt.training_data import sample_training_pairs


def _config(ratio=2.0):
    return {
        "training": {
            "unlabeled_to_positive_ratio": ratio,
            "max_training_pairs_per_fold": 10000,
        }
    }


def test_weak_positive_is_positive_with_lower_confidence_not_label_zero():
    frame = pd.DataFrame(
        {
            "candidate_id": ["strong", "weak", "unlabeled"],
            "label_class": ["strong_positive", "weak_positive", "unlabeled"],
            "direction": ["positive", "negative", "unknown"],
        }
    )
    sampled = sample_training_pairs(_config(), frame, seed=7).set_index("candidate_id")
    assert sampled.loc["strong", "proxy_label"] == 1
    assert sampled.loc["weak", "proxy_label"] == 1
    assert sampled.loc["unlabeled", "proxy_label"] == 0
    assert pd.isna(sampled.loc["unlabeled", "direction_label"])


def test_unlabeled_ratio_uses_sampled_positive_count():
    strong = pd.DataFrame(
        {
            "candidate_id": [f"s{i}" for i in range(2)],
            "label_class": "strong_positive",
            "direction": "positive",
        }
    )
    weak = pd.DataFrame(
        {
            "candidate_id": [f"w{i}" for i in range(5000)],
            "label_class": "weak_positive",
            "direction": "positive",
        }
    )
    unlabeled = pd.DataFrame(
        {
            "candidate_id": [f"u{i}" for i in range(5000)],
            "label_class": "unlabeled",
            "direction": "unknown",
        }
    )
    sampled = sample_training_pairs(
        _config(ratio=2.0),
        pd.concat([strong, weak, unlabeled], ignore_index=True),
        seed=11,
    )
    positive_count = int(sampled.proxy_label.sum())
    unlabeled_count = int((sampled.proxy_label == 0).sum())
    assert positive_count == 1002
    assert unlabeled_count == 2 * positive_count


def test_pathway_sampler_retains_unlabeled_when_positives_exceed_cap():
    frame = pd.DataFrame(
        {
            "candidate_id": [f"p{i}" for i in range(2000)] + [f"u{i}" for i in range(5000)],
            "label_class": ["strong_positive"] * 2000 + ["unlabeled"] * 5000,
            "direction": ["positive"] * 7000,
        }
    )
    cfg = _config(ratio=4.0)
    cfg["training"]["max_training_pairs_per_fold"] = 1000
    sampled = sample_training_pairs(cfg, frame, seed=19)
    assert len(sampled) == 1000
    assert int(sampled.proxy_label.sum()) == 200
    assert int(sampled.proxy_label.eq(0).sum()) == 800
    assert sampled.loc[sampled.proxy_label.eq(0), "direction_label"].isna().all()
