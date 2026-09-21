from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from website.backend.v31_exact_store import ExactPathwayReleaseStore


def _release(root: Path) -> Path:
    audits = []
    all_selected = []
    cancers = [*[f"C{index:02d}" for index in range(31)], "HNSC", "LGG"]
    for cancer in cancers:
        fold = f"LOCO_{cancer}"
        frame = pd.DataFrame(
            {
                "candidate_id": [f"{cancer}:0", f"{cancer}:1"],
                "cancer_id": [cancer, cancer],
                "lncrna_id": ["LNC:1", "LNC:2"],
                "gene_symbol": ["L1", "L2"],
                "pathway_id": ["PW:EMT", "PW:DDR"],
                "pathway_name": ["EMT", "DNA repair"],
                "pathway_family_id": ["PF:EMT", "PF:DDR"],
                "final_direction": ["positive", "negative"],
                "calibrated_probability": [0.95, 0.85],
                "seed_probability_std": [0.02, 0.03],
                "observed_evidence_score": [0.8, 0.5],
                "relationship_class": ["observed_core", "model_supported"],
                "member_weight": [0.9, 0.7],
            }
        )
        for directory in ("web_exact_pathway_score", "web_exact_pathway_selected"):
            path = root / directory / f"cancer_id={cancer}" / "part-0.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
        all_selected.append(frame)
        audits.append(
            {
                "fold_id": fold,
                "cancer_id": cancer,
                "rows": 2,
                "selected_rows": 2,
                "score_sha256": "a" * 64,
                "selected_sha256": "b" * 64,
                "inference_success_sha256": "c" * 64,
            }
        )
    master = pd.DataFrame(
        {
            "geneset_id": ["GS1", "GS2"],
            "geneset_name": ["C00__PW:EMT__POSITIVE", "C00__PW:DDR__NEGATIVE"],
            "cancer_id": ["C00", "C00"],
            "pathway_id": ["PW:EMT", "PW:DDR"],
            "pathway_name": ["EMT", "DNA repair"],
            "pathway_family_id": ["PF:EMT", "PF:DDR"],
            "direction": ["positive", "negative"],
            "member_count": [1, 1],
        }
    )
    master.to_parquet(root / "geneset_master_exact_pathway.parquet", index=False)
    member = pd.concat(all_selected[:1], ignore_index=True)
    member["geneset_id"] = ["GS1", "GS2"]
    member["rank"] = 1
    member.to_parquet(root / "geneset_member_exact_pathway.parquet", index=False)
    pd.DataFrame(
        {"cancer_id": ["C00"], "pathway_id": ["PW:EMT"], "selected_lncRNAs": [1]}
    ).to_parquet(root / "web_cancer_exact_pathway_summary.parquet", index=False)
    (root / "SUCCESS.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "pathway_target_level": "exact_pathway",
                "pathway_family_role": "auxiliary_hierarchy_only",
                "cancers": 33,
                "reference_only_cancers": [],
                "heldout_label_columns_in_public_scores": False,
                "score_rows": 66,
                "selected_rows": 66,
                "genesets": 2,
                "geneset_members": 2,
                "pancancer_eligible_lncRNAs": 2751,
                "pancancer_lncrna_filter": {
                    "enabled": True,
                    "within_cancer_min_sample_detection_rate": 0.10,
                    "minimum_detected_cancers": 3,
                    "evidence_cannot_bypass_filter": True,
                },
                "selected_model": "hgt",
                "fold_audits": audits,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_v31_website_defaults_to_fail_closed_without_final_release() -> None:
    app_source = (
        Path(__file__).resolve().parents[1] / "website" / "backend" / "app.py"
    ).read_text(encoding="utf-8")
    assert (
        '_environment_flag("CANCERLNCATLAS_REQUIRE_V31_EXACT_PATHWAY", default=True)'
        in app_source
    )
    assert "CANCERLNCATLAS_REQUIRE_V31_EXACT_PATHWAY=0" in app_source


def test_exact_store_exposes_33_formal_cancers_and_exact_pathway_stats(
    tmp_path: Path,
) -> None:
    store = ExactPathwayReleaseStore(_release(tmp_path))
    stats = store.stats()
    assert stats["cancers"] == 33
    assert stats["reference_only_cancers"] == 0
    assert stats["lncrnas"] == 2751
    assert stats["pathway_target_level"] == "exact_pathway"
    overview = store.cancer_overview()
    assert len(overview) == 33
    assert not overview.reference_only.any()


def test_exact_store_returns_exact_cancer_and_lnc_profiles(tmp_path: Path) -> None:
    store = ExactPathwayReleaseStore(_release(tmp_path))
    cancer = store.cancer_profile("C00", 10)
    assert cancer["lncrna_ranking"][0]["pathway_id"] == "PW:EMT"
    assert cancer["pathway_ranking"][0]["pathway_id"] == "PW:EMT"
    lnc = store.lnc_profile("LNC:1", "C00")
    assert lnc["top_relationships"][0]["pathway_id"] == "PW:EMT"


def test_exact_store_genesets_filter_on_exact_pathway(tmp_path: Path) -> None:
    store = ExactPathwayReleaseStore(_release(tmp_path))
    result = store.genesets(
        search=None, pathway="PW:EMT", cancer="C00", limit=10
    )
    assert result["total_genesets"] == 1
    assert result["genesets"][0]["pathway_id"] == "PW:EMT"
    assert result["members"][0]["lncrna_id"] == "LNC:1"


def test_exact_store_geneset_detail_resolves_exact_and_legacy_aliases(
    tmp_path: Path,
) -> None:
    store = ExactPathwayReleaseStore(_release(tmp_path))
    exact = store.geneset_detail("GS1")
    assert exact["resolution"] == "geneset_id"
    assert exact["overview"]["pathway_id"] == "PW:EMT"
    assert exact["members"][0]["lncrna_id"] == "LNC:1"
    assert exact["legacy_family_broadcast"] is False

    family = store.geneset_detail("PF:EMT")
    assert family["resolution"] == "pathway_family_id"
    assert [row["geneset_id"] for row in family["genesets"]] == ["GS1"]
    assert family["legacy_family_broadcast"] is False

    pathway = store.geneset_detail("PW:DDR")
    assert pathway["resolution"] == "pathway_id"
    assert pathway["genesets"][0]["geneset_id"] == "GS2"

    with pytest.raises(KeyError):
        store.geneset_detail("GS:UNKNOWN")


def test_exact_store_rejects_labels_in_public_partition(tmp_path: Path) -> None:
    root = _release(tmp_path)
    path = root / "web_exact_pathway_selected" / "cancer_id=C00" / "part-0.parquet"
    frame = pd.read_parquet(path)
    frame["proxy_label"] = [1, 0]
    frame.to_parquet(path, index=False)
    store = ExactPathwayReleaseStore(root)
    with pytest.raises(RuntimeError, match="contains labels"):
        store.cancer_rows("C00")


def test_exact_store_rejects_any_reference_only_cancer(tmp_path: Path) -> None:
    root = _release(tmp_path)
    success_path = root / "SUCCESS.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    success["reference_only_cancers"] = ["HNSC"]
    success_path.write_text(json.dumps(success), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Invalid V3.1 exact-pathway"):
        ExactPathwayReleaseStore(root)


def test_exact_store_rejects_missing_hnsc_or_lgg_fold(tmp_path: Path) -> None:
    root = _release(tmp_path)
    success_path = root / "SUCCESS.json"
    success = json.loads(success_path.read_text(encoding="utf-8"))
    hnsc = next(row for row in success["fold_audits"] if row["cancer_id"] == "HNSC")
    hnsc["cancer_id"] = "C31"
    success_path.write_text(json.dumps(success), encoding="utf-8")
    with pytest.raises(RuntimeError, match="33 unique cancer fold audits"):
        ExactPathwayReleaseStore(root)
