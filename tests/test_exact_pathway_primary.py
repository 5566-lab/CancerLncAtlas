from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cc_hhgt.common import load_config
from cc_hhgt.exact_pathway_candidates import build_exact_candidate_universe
from cc_hhgt.exact_pathway_evidence import merge_exact_evidence
from cc_hhgt.gnn import candidate_tensors
from cc_hhgt.pathway_target import (
    pathway_target_column,
    pathway_target_level,
    pathway_target_node_type,
    require_exact_pathway_contract,
)


def test_formal_config_registers_exact_pathway_and_all_cancers() -> None:
    cfg = load_config("config/model_v3_1_exact_pathway_33c.yaml")
    require_exact_pathway_contract(cfg)
    assert pathway_target_level(cfg) == "exact_pathway"
    assert pathway_target_column(cfg) == "pathway_id"
    assert pathway_target_node_type(cfg) == "pathway"
    assert cfg["analysis_cancers"]["reference_only"] == []
    assert cfg["cancer_scope"]["reference_only"] == []
    assert set(cfg["formal_contract"]["required_formal_cancers"]) == {"HNSC", "LGG"}
    assert cfg["exact_pathway_evidence"]["label_truth_layer"] == (
        "patient_statistical_association_only"
    )
    assert not cfg["exact_pathway_evidence"]["annotation_sources_may_define_label"]
    assert cfg["state_training"]["mask_pair_evidence_for_all_pathway_models"] is True
    assert cfg["state_training"]["strict_pair_evidence_mask_probability"] == 1.0
    assert cfg["state_training"]["auxiliary_loss_weight"] == 0.50
    assert cfg["training"]["max_edges_per_relation"] is None
    assert cfg["runtime_graph_sampling"]["enabled"] is True
    assert cfg["runtime_graph_sampling"]["coverage_cycle"] == "all_runtime_chunks"
    assert not {
        "interaction_support",
        "perturbation_support",
        "drug_support",
    }.intersection(cfg["training"]["feature_columns"])


def test_v29_reference_only_wording_is_explicitly_historical_and_superseded() -> None:
    contract = (Path(__file__).resolve().parents[1] / "V2_9_DATA_CONTRACT.md").read_text(
        encoding="utf-8"
    )
    assert "历史说明（仅适用于已废止的 V2.9 31-cancer 协议）" in contract
    assert "HNSC、LGG 与其余 31 个癌种一样" in contract
    assert "不得再按 reference-only 处理" in contract


def test_exact_evidence_keeps_pathways_separate_and_never_needs_family() -> None:
    cfg = {
        "analysis_version": "test",
        "pair_evidence": {
            "weights": {
                "bulk": 0.5,
                "sc_ssgsea": 0.5,
                "ucell": 0.0,
                "replication": 0.0,
                "interaction": 0.0,
                "perturbation": 0.0,
                "drug": 0.0,
            },
            "strong_positive_min_score": 0.1,
            "strong_positive_min_sources": 2,
        },
        "training": {"weak_positive_weight": 0.35, "unlabeled_weight": 0.12},
        "exact_pathway_evidence": {
            "family_level_evidence_may_define_label": False,
            "annotation_sources_may_define_label": False,
            "bulk_strong_max_fdr": 0.05,
            "bulk_strong_min_abs_effect": 0.20,
        },
    }
    bulk = pd.DataFrame(
        {
            "cancer_id": ["BRCA", "BRCA"],
            "lncrna_id": ["L1", "L1"],
            "pathway_id": ["P1", "P2"],
            "bulk_support": [0.8, 0.2],
            "bulk_effect": [0.5, -0.2],
            "bulk_fdr": [0.01, 0.10],
            "bulk_direction": ["positive", "negative"],
            "bulk_available": [True, True],
        }
    )
    sc = pd.DataFrame(
        {
            "cancer_id": ["BRCA"],
            "lncrna_id": ["L1"],
            "pathway_id": ["P1"],
            "sc_ssgsea_support": [0.7],
            "sc_effect": [0.4],
            "sc_direction": ["positive"],
            "sc_available": [True],
        }
    )
    merged = merge_exact_evidence(cfg, [bulk, sc]).sort_values("pathway_id")
    assert merged.pathway_id.tolist() == ["P1", "P2"]
    assert merged.label_class.tolist() == ["strong_positive", "weak_positive"]
    assert "pathway_family_id" not in merged
    assert merged.pair_id.nunique() == 2


def test_annotation_only_evidence_cannot_create_association_label() -> None:
    cfg = {
        "analysis_version": "test",
        "pair_evidence": {
            "weights": {
                "bulk": 0.2,
                "sc_ssgsea": 0.2,
                "ucell": 0.0,
                "replication": 0.0,
                "interaction": 0.3,
                "perturbation": 0.2,
                "drug": 0.1,
            }
        },
        "training": {"weak_positive_weight": 0.35, "unlabeled_weight": 0.12},
        "exact_pathway_evidence": {
            "family_level_evidence_may_define_label": False,
            "annotation_sources_may_define_label": False,
            "bulk_strong_max_fdr": 0.05,
            "bulk_strong_min_abs_effect": 0.20,
        },
    }
    annotation = pd.DataFrame(
        {
            "cancer_id": ["BRCA"],
            "lncrna_id": ["L1"],
            "pathway_id": ["P1"],
            "interaction_support": [1.0],
            "perturbation_support": [1.0],
            "drug_support": [1.0],
            "interaction_available": [True],
            "drug_available": [True],
        }
    )
    merged = merge_exact_evidence(cfg, [annotation])
    assert merged.loc[0, "observed_evidence_score"] > 0
    assert merged.loc[0, "n_positive_evidence_sources"] == 3
    assert merged.loc[0, "n_positive_association_sources"] == 0
    assert merged.loc[0, "label_class"] == "unlabeled"
    assert merged.loc[0, "association_proxy_label"] == 0


def test_candidate_tensors_use_exact_pathway_node_when_both_columns_exist() -> None:
    torch = pytest.importorskip("torch")
    frame = pd.DataFrame(
        {
            "candidate_id": ["C1"],
            "cancer_id": ["BRCA"],
            "lncrna_id": ["L1"],
            "pathway_id": ["P_EXACT"],
            "pathway_family_id": ["PF_AUX"],
            "proxy_label": [1.0],
            "label_class": ["weak_positive"],
            "direction_label": [1.0],
            "bulk_support": [0.3],
        }
    )
    bundle = SimpleNamespace(
        node_maps={
            "lncRNA": {"L1": 0},
            "pathway": {"P_EXACT": 3},
            "pathway_family": {"PF_AUX": 7},
            "cancer": {"BRCA": 0},
        },
        homogeneous={
            "global_map": {
                ("lncRNA", "L1"): 1,
                ("pathway", "P_EXACT"): 2,
                ("pathway_family", "PF_AUX"): 9,
                ("cancer", "BRCA"): 4,
            }
        },
    )
    _, batch = candidate_tensors(frame, bundle, ["bulk_support"], "cpu", "rgcn")
    assert torch.equal(batch["p"], torch.tensor([2]))


def test_exact_candidate_builder_filters_lncrna_before_pathway_join(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path
    results = root / "results"
    standardized = root / "standardized"
    tables = results / "tables"
    association = root / "bulk_association"
    association.mkdir(parents=True)
    tables.mkdir(parents=True)
    standardized.mkdir(parents=True)

    pd.DataFrame({"cancer_id": ["HNSC", "LGG"]}).to_parquet(
        standardized / "dim_cancer.parquet", index=False
    )
    pd.DataFrame(
        {
            "cancer_id": ["HNSC", "HNSC", "LGG", "LGG"],
            "lncrna_id": ["L_KEEP", "L_DROP", "L_KEEP", "L_DROP"],
            "pathway_id": ["P1", "P1", "P2", "P2"],
        }
    ).to_parquet(association / "part.parquet", index=False)
    pd.DataFrame(
        {
            "pathway_id": ["P1", "P2"],
            "pathway_family_id": ["PF1", "PF1"],
            "membership_weight": [1.0, 1.0],
        }
    ).to_parquet(tables / "pathway_family_member.parquet", index=False)
    evidence = pd.DataFrame(
        {
            "cancer_id": ["HNSC", "LGG"],
            "lncrna_id": ["L_KEEP", "L_KEEP"],
            "pathway_id": ["P1", "P2"],
            "pathway_family_id": ["PF1", "PF1"],
            "pair_id": ["PAIR:1", "PAIR:2"],
            "label_class": ["weak_positive", "weak_positive"],
            # Production evidence stores these as booleans.  The candidate SQL
            # must not mix BOOLEAN values with numeric COALESCE defaults.
            "association_proxy_label": [True, True],
            "strong_evidence_label": [False, False],
            "sample_weight": [0.35, 0.35],
            "direction": ["positive", "negative"],
            "observed_evidence_score": [0.4, 0.4],
            "bulk_support": [0.4, 0.4],
            "bulk_available": [True, True],
        }
    )
    evidence.to_parquet(tables / "pair_evidence.parquet", index=False)

    bulk_detection = pd.DataFrame(
        {
            "cancer_id": ["HNSC", "LGG", "HNSC", "LGG"],
            "lncrna_id": ["L_KEEP", "L_KEEP", "L_DROP", "L_DROP"],
            "bulk_detection_rate": [0.5, 0.5, 0.5, 0.0],
        }
    )
    sc_detection = pd.DataFrame(
        columns=["cancer_id", "lncrna_id", "sc_detection_rate"]
    )
    import cc_hhgt.candidates as candidate_module

    monkeypatch.setattr(candidate_module, "build_bulk_detection", lambda cfg: bulk_detection)
    monkeypatch.setattr(candidate_module, "build_sc_detection", lambda cfg: sc_detection)
    cfg = {
        "_root": root,
        "_results": results,
        "_standardized": standardized,
        "_sample_universe_sha256": "sample-hash",
        "_formal_lineage": {"lineage_sha256": "lineage-hash"},
        "analysis_version": "test",
        "inputs": {"bulk_lnc_pathway": "bulk_association"},
        "analysis_cancers": {"reference_only": [], "exclude_from_training": []},
        "candidate_universe": {
            "bulk_min_detection_rate": 0.1,
            "sc_min_detection_rate": 0.03,
            "pancancer_lnc_filter": {
                "enabled": True,
                "minimum_detected_cancers": 2,
                "within_cancer_min_sample_detection_rate": 0.1,
            },
        },
        "training": {
            "feature_columns": [
                "bulk_support",
                "bulk_available",
                "bulk_detection_rate",
                "sc_detection_rate",
                "state_support",
            ],
            "unlabeled_weight": 0.12,
        },
    }
    manifest = build_exact_candidate_universe(cfg)
    assert set(manifest.cancer_id) == {"HNSC", "LGG"}
    assert manifest.n_candidates.tolist() == [1, 1]
    assert manifest.path.tolist() == [
        "tables/candidate_universe/cancer_id=HNSC/part-0.parquet",
        "tables/candidate_universe/cancer_id=LGG/part-0.parquet",
    ]
    assert not manifest.path.str.contains(".assets.tmp", regex=False).any()
    for cancer, pathway in [("HNSC", "P1"), ("LGG", "P2")]:
        part = pd.read_parquet(
            tables / "candidate_universe" / f"cancer_id={cancer}" / "part-0.parquet"
        )
        assert part.lncrna_id.tolist() == ["L_KEEP"]
        assert part.pathway_id.tolist() == [pathway]
        assert part.pathway_family_id.tolist() == ["PF1"]
        assert part.association_proxy_label.tolist() == [1]
        assert part.bulk_available.tolist() == [1.0]
        assert set(part.label_semantics) == {
            "EXACT_PATHWAY_PATIENT_ASSOCIATION_V2:cancer_x_lncrna_x_pathway_id"
        }
