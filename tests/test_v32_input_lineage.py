from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32.input_lineage import (
    InputLineageArtifact,
    InputLineageError,
    artifact_sha256,
    audit_input_artifact,
    audit_input_lineage,
    validate_input_lineage,
)


def _table(path: Path, **columns: object) -> Path:
    pd.DataFrame([columns]).to_csv(path, index=False)
    return path


def _spec(path: Path, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "path": path,
        "generation": "V3.2",
        "source_role": "standardized_input",
        "outcome_derived": False,
        "fold_fitted": False,
        "use_role": "core_input",
    }
    value.update(overrides)
    return value


def _reason_codes(record: dict[str, object]) -> str:
    return "\n".join(record["reasons"])  # type: ignore[arg-type]


def test_read_only_audit_allows_raw_and_static_and_records_lineage(tmp_path: Path) -> None:
    raw = _table(tmp_path / "mc3.maf", sample_id="S1", gene_id="G1")
    annotation = _table(
        tmp_path / "pathway_gene.tsv",
        pathway_id="P1",
        pathway_family_id="F1",
        gene_id="G1",
        membership_weight=1.0,
    )
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (raw, annotation)}

    audit = audit_input_lineage(
        [
            InputLineageArtifact(
                path=raw,
                generation="MC3-v0.2.8",
                source_role="raw_data",
                outcome_derived=False,
                fold_fitted=False,
            ),
            _spec(
                annotation,
                generation="MSigDB-2025.1",
                source_role="static_annotation",
            ),
        ]
    )

    assert audit["status"] == "PASS"
    assert audit["audit_mode"] == "READ_ONLY"
    assert len(audit["lineage_sha256"]) == 64
    for record in audit["artifacts"]:
        assert len(record["sha256"]) == 64
        assert {
            "generation",
            "source_role",
            "outcome_derived",
            "fold_fitted",
        }.issubset(record)
    after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (raw, annotation)}
    assert after == before


def test_old_standardized_input_is_allowed_only_when_clean_and_unfitted(tmp_path: Path) -> None:
    clean = _table(
        tmp_path / "interaction_relation.csv",
        lncrna_id="L1",
        protein_id="UNIPROT:P1",
        source_database="NPInter",
    )
    record = audit_input_artifact(
        _spec(
            clean,
            generation="V2.9",
            outcome_derived=False,
            fold_fitted=False,
        )
    )
    assert record["status"] == "PASS"
    assert record["columns"] == ["lncrna_id", "protein_id", "source_database"]


@pytest.mark.parametrize(
    ("overrides", "columns", "reason"),
    [
        ({"outcome_derived": True}, {"lncrna_id": "L1"}, "OLD_STANDARDIZED_OUTCOME_DERIVED"),
        ({"fold_fitted": True}, {"lncrna_id": "L1"}, "OLD_STANDARDIZED_FOLD_FITTED"),
        ({}, {"lncrna_id": "L1", "legacy_score": 0.8}, "OLD_STANDARDIZED_FORBIDDEN_COLUMNS"),
        ({}, {"lncrna_id": "L1", "proxy_label": 1}, "OLD_STANDARDIZED_FORBIDDEN_COLUMNS"),
    ],
)
def test_old_standardized_input_fails_closed(
    tmp_path: Path,
    overrides: dict[str, object],
    columns: dict[str, object],
    reason: str,
) -> None:
    path = _table(tmp_path / "old_standardized.csv", **columns)
    record = audit_input_artifact(
        _spec(path, generation="V2.9", **overrides)
    )
    assert record["status"] == "FAIL"
    assert reason in _reason_codes(record)


@pytest.mark.parametrize(
    "source_role",
    [
        "historical_checkpoint",
        "historical_embedding",
        "oof_prediction",
        "historical_probability",
        "historical_ranking",
        "historical_web_table",
        "historical_release_table",
    ],
)
def test_historical_result_roles_are_rejected(tmp_path: Path, source_role: str) -> None:
    path = _table(tmp_path / "innocent_name.csv", entity_id="X1", value=0.5)
    record = audit_input_artifact(
        _spec(path, generation="V2.9", source_role=source_role)
    )
    assert record["status"] == "FAIL"
    assert "FORBIDDEN_RESULT_ROLE" in _reason_codes(record)


def test_result_content_or_path_cannot_be_disguised_as_standardized_input(tmp_path: Path) -> None:
    disguised_content = _table(
        tmp_path / "features.csv",
        cancer_id="BRCA",
        association_membership_probability=0.9,
    )
    content_record = audit_input_artifact(_spec(disguised_content))
    assert content_record["status"] == "FAIL"
    assert "FORBIDDEN_RESULT_COLUMNS" in _reason_codes(content_record)

    disguised_path = _table(tmp_path / "public_release_table.csv", entity_id="E1")
    path_record = audit_input_artifact(
        _spec(disguised_path, generation="V2.9")
    )
    assert path_record["status"] == "FAIL"
    assert "FORBIDDEN_RESULT_PATH" in _reason_codes(path_record)

    disguised_embedding = _table(
        tmp_path / "node_features.csv",
        node_id="L1",
        emb_0=0.1,
        emb_1=0.2,
    )
    embedding_record = audit_input_artifact(
        _spec(disguised_embedding, generation="V2.9")
    )
    assert embedding_record["status"] == "FAIL"
    assert "FORBIDDEN_RESULT_COLUMNS" in _reason_codes(embedding_record)


def test_v32_core_checkpoint_is_aux_parent_only(tmp_path: Path) -> None:
    checkpoint = tmp_path / "v32_core.ckpt"
    checkpoint.write_bytes(b"newly-trained-v32-core")
    safe = _spec(
        checkpoint,
        generation="CancerLncAtlas_V3.2_full",
        source_role="v32_core_checkpoint",
        use_role="aux_parent",
        outcome_derived=True,
        fold_fitted=True,
    )
    assert audit_input_artifact(safe)["status"] == "PASS"

    core_input = audit_input_artifact({**safe, "use_role": "core_input"})
    assert core_input["status"] == "FAIL"
    assert "V32_CORE_CHECKPOINT_NOT_AUX_PARENT" in _reason_codes(core_input)

    historical = audit_input_artifact({**safe, "generation": "V3.1"})
    assert historical["status"] == "FAIL"
    assert "HISTORICAL_CHECKPOINT_FORBIDDEN" in _reason_codes(historical)


@pytest.mark.parametrize("forbidden_column", ["label", "sample_weight", "score"])
def test_old_pair_evidence_label_weight_and_score_are_explicitly_forbidden(
    tmp_path: Path, forbidden_column: str
) -> None:
    path = _table(
        tmp_path / f"pair_evidence_{forbidden_column}.csv",
        cancer_id="BRCA",
        lncrna_id="L1",
        pathway_id="P1",
        **{forbidden_column: 1.0},
    )
    record = audit_input_artifact(_spec(path, generation="V2.9"))
    assert record["status"] == "FAIL"
    assert "PAIR_EVIDENCE_FORBIDDEN_COLUMNS" in _reason_codes(record)


def test_pair_evidence_semantic_score_suffix_is_forbidden_even_if_role_says_raw(
    tmp_path: Path,
) -> None:
    path = _table(
        tmp_path / "pair_evidence_events.csv",
        cancer_id="BRCA",
        lncrna_id="L1",
        pathway_id="P1",
        observed_evidence_score=0.7,
    )
    record = audit_input_artifact(
        _spec(path, generation="V2.9", source_role="raw_data")
    )
    assert record["status"] == "FAIL"
    assert "PAIR_EVIDENCE_FORBIDDEN_COLUMNS" in _reason_codes(record)


def test_clean_old_pair_evidence_events_remain_restandardizable(tmp_path: Path) -> None:
    path = _table(
        tmp_path / "pair_evidence_events.csv",
        cancer_id="BRCA",
        lncrna_id="L1",
        pathway_id="P1",
        source_database="NPInter",
        support_type="physical",
    )
    record = audit_input_artifact(_spec(path, generation="V2.9"))
    assert record["status"] == "PASS"


def test_family_annotation_is_allowed_but_family_to_exact_broadcast_is_not(tmp_path: Path) -> None:
    hierarchy = _table(
        tmp_path / "pathway_hierarchy.csv",
        pathway_family_id="F1",
        pathway_id="P1",
        membership_weight=1.0,
    )
    safe = audit_input_artifact(
        _spec(
            hierarchy,
            generation="MSigDB-2025.1",
            source_role="static_annotation",
        )
    )
    assert safe["status"] == "PASS"

    declared_broadcast = audit_input_artifact(
        _spec(
            hierarchy,
            generation="V2.9",
            source_target_level="pathway_family",
            target_level="exact_pathway",
        )
    )
    assert declared_broadcast["status"] == "FAIL"
    assert "FAMILY_TO_EXACT_BROADCAST_FORBIDDEN" in _reason_codes(declared_broadcast)

    policy_broadcast = audit_input_artifact(
        _spec(
            hierarchy,
            generation="V2.9",
            mapping_policy="many_to_many_exact_pathway_jaccard_from_family",
        )
    )
    assert policy_broadcast["status"] == "FAIL"
    assert "FAMILY_TO_EXACT_BROADCAST_FORBIDDEN" in _reason_codes(policy_broadcast)

    scored = _table(
        tmp_path / "family_exact_values.csv",
        pathway_family_id="F1",
        pathway_id="P1",
        score=0.9,
    )
    content_broadcast = audit_input_artifact(_spec(scored, generation="V2.9"))
    assert content_broadcast["status"] == "FAIL"
    assert "FAMILY_TO_EXACT_BROADCAST_FORBIDDEN" in _reason_codes(content_broadcast)


def test_sha256_is_computed_and_declared_hash_drift_is_rejected(tmp_path: Path) -> None:
    path = _table(tmp_path / "raw.csv", sample_id="S1", value=2)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifact_sha256(path) == expected
    assert audit_input_artifact(
        _spec(path, source_role="raw_data", generation="source-v1", expected_sha256=expected)
    )["status"] == "PASS"

    bad = audit_input_artifact(
        _spec(path, source_role="raw_data", generation="source-v1", expected_sha256="0" * 64)
    )
    assert bad["status"] == "FAIL"
    assert "SHA256_MISMATCH" in _reason_codes(bad)


def test_validate_raises_on_any_failed_artifact(tmp_path: Path) -> None:
    path = _table(tmp_path / "oof_predictions.csv", entity_id="E1", probability=0.9)
    with pytest.raises(InputLineageError, match="V3.2 input lineage rejected"):
        validate_input_lineage([_spec(path, generation="V2.9")])
