from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
pytest.importorskip("duckdb")

from cc_hhgt.v32.evidence_streaming_training import (
    CONVERSION_FORMAT,
    CONVERSION_PASS,
    EMPTY_EVENT_COLUMNS,
    PATIENT_RECEIPT_FORMAT,
    STAGING_STATUS,
    StreamingEvidenceStageError,
    iter_fold_event_bag_batches,
    iter_fold_trainer_batches,
    materialize_streaming_evidence_stage,
    validate_streaming_stage,
)
from cc_hhgt.v32.evidence_training import (
    build_exact_event_bags,
    materialize_candidate_events,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False)


def _fixture(tmp_path: Path, *, shuffled: bool = False) -> dict[str, object]:
    root = tmp_path / ("inputs_shuffled" if shuffled else "inputs")
    root.mkdir(parents=True)
    pathways = [f"PATH-{index:02d}" for index in range(12)]
    lncs = [f"LNC-{index:02d}" for index in range(12)]
    genes = [f"GENE-{index:02d}" for index in range(12)]
    members = pd.DataFrame(
        {
            "pathway_id": pathways + ["PATH-MULTI-A", "PATH-MULTI-B"],
            "gene_id": genes + ["GENE-MULTI", "GENE-MULTI"],
            "member_type": ["gene"] * 14,
            "mapping_status": ["mapped"] * 14,
        }
    )
    candidates = pd.DataFrame(
        {
            "cancer_id": ["BRCA"] * 12 + ["LUAD", "BRCA", "BRCA"],
            "lncrna_id": lncs + [lncs[0], "LNC-MULTI", "LNC-MULTI"],
            "pathway_id": pathways + [pathways[0], "PATH-MULTI-A", "PATH-MULTI-B"],
        }
    )
    rows: list[dict[str, object]] = []
    for index, (lnc, gene) in enumerate(zip(lncs, genes)):
        rows.append(
            {
                "interaction_id": f"INT-{index}",
                "source_row_id": f"SRC-{index}",
                "source_row_index": str(index + 1),
                "source_row_sha256": hashlib.sha256(f"raw-{index}".encode()).hexdigest(),
                "source_sha256": "a" * 64,
                "source_record_id": f"REC-{index}",
                "source_database": "DB-SHARED" if index < 2 else f"DB-{index % 3}",
                "source_dataset": "DS-SHARED" if index < 2 else "DS-1",
                "cancer_id": "PAN_CANCER",
                "lncrna_id": lnc,
                "partner_id": gene,
                "lncrna_mapping_route": "unique_symbol",
                "lncrna_mapping_candidate_count": "1",
                "partner_mapping_route": "unique_symbol",
                "partner_mapping_candidate_count": "1",
                "mapping_status": "MAPPED_BOTH",
                "relation_type": "physical_binding" if index == 0 else "regulatory",
                "direction": "positive" if index % 2 == 0 else "negative",
                "experiment_family": "RNA pull-down" if index == 0 else "knockdown",
                "is_experimental": "true",
                "is_predicted": "false",
                "pmid": "PMID-SHARED" if index < 2 else f"PMID-{index}",
                "species": "human",
                "cell_line": None if index == 3 else "CELL",
                "tissue": "breast",
            }
        )
    # Duplicate the exact canonical event: one model event, two lineage rows.
    rows.append({**rows[0], "interaction_id": "INT-0-DUP", "source_row_id": "SRC-0-DUP"})
    # One partner maps to two exact pathways, never to a family broadcast.
    rows.append(
        {
            **rows[5],
            "interaction_id": "INT-MULTI",
            "source_row_id": "SRC-MULTI",
            "source_row_sha256": hashlib.sha256(b"raw-multi").hexdigest(),
            "source_record_id": "REC-MULTI",
            "lncrna_id": "LNC-MULTI",
            "partner_id": "GENE-MULTI",
            "pmid": "PMID-MULTI|PMID-SECOND",
        }
    )
    # Direct exact assertion wins over a member route to the same exact pathway.
    rows.append(
        {
            **rows[6],
            "interaction_id": "INT-DIRECT",
            "source_row_id": "SRC-DIRECT",
            "source_row_sha256": hashlib.sha256(b"raw-direct").hexdigest(),
            "source_record_id": "REC-DIRECT",
            "pathway_id": pathways[6],
        }
    )
    # Mapped-but-ambiguous remains explicit in lineage; legacy still routes the ID.
    rows[7]["lncrna_mapping_route"] = "ambiguous_alias"
    rows[7]["lncrna_mapping_candidate_count"] = "2"
    rows[7]["mapping_status"] = "AMBIGUOUS_LNC_PARTNER_MAPPED"
    # Fail-closed raw rows.
    rows.extend(
        [
            {
                **rows[8],
                "interaction_id": "INT-UNMAPPED",
                "source_row_id": "SRC-UNMAPPED",
                "source_row_sha256": hashlib.sha256(b"raw-unmapped").hexdigest(),
                "source_record_id": "REC-UNMAPPED",
                "lncrna_id": "",
                "lncrna_mapping_route": "unresolved_alias",
                "lncrna_mapping_candidate_count": "0",
                "mapping_status": "LNC_UNMAPPED_PARTNER_MAPPED",
            },
            {
                **rows[9],
                "interaction_id": "INT-FAMILY",
                "source_row_id": "SRC-FAMILY",
                "source_row_sha256": hashlib.sha256(b"raw-family").hexdigest(),
                "source_record_id": "REC-FAMILY",
                "partner_id": "",
                "pathway_family_id": "FAMILY:IMMUNE",
            },
            {
                **rows[10],
                "interaction_id": "INT-OUTSIDE",
                "source_row_id": "SRC-OUTSIDE",
                "source_row_sha256": hashlib.sha256(b"raw-outside").hexdigest(),
                "source_record_id": "REC-OUTSIDE",
                "pathway_id": "PATH-NOT-A-CANDIDATE",
                "partner_id": "",
            },
        ]
    )
    relation = pd.DataFrame(rows)
    if shuffled:
        relation = relation.sample(frac=1.0, random_state=71).reset_index(drop=True)

    relation_path = root / "interaction_relation_recovered.parquet"
    event_path = root / "evidence_event_empty_by_contract.parquet"
    member_path = root / "exact_members.parquet"
    candidate_path = root / "FORMAL_CANDIDATE_UNIVERSE.parquet"
    _write_parquet(relation, relation_path)
    empty_schema = pa.schema([pa.field(column, pa.string()) for column in EMPTY_EVENT_COLUMNS])
    pq.write_table(pa.Table.from_pylist([], schema=empty_schema), event_path)
    _write_parquet(members, member_path)
    _write_parquet(candidates, candidate_path)

    patient = pd.DataFrame(
        {
            "cancer_id": ["BRCA"] * 5 + ["LUAD"] * 5,
            "patient_id": [f"PAT-{index}" for index in range(10)],
            "patient_fold_id": list(range(5)) * 2,
            "fold_seed": [20260726] * 10,
        }
    )
    patient_path = root / "PATIENT_FOLD_AUTHORITY.tsv"
    patient.to_csv(patient_path, sep="\t", index=False)
    patient_receipt = {
        "format": PATIENT_RECEIPT_FORMAT,
        "n_folds": 5,
        "artifacts": {
            "patient_fold_authority": {
                "filename": patient_path.name, "sha256": _sha(patient_path)
            }
        },
        "gates": {
            "all_five_folds_per_cancer": True,
            "patient_cross_cancer_count_zero": True,
            "patient_cross_fold_count_zero": True,
            "output_reuse_forbidden": True,
        },
        "observed": {"cancers": 2, "patients": 10},
    }
    patient_receipt_path = root / "PATIENT_FOLD_AUTHORITY_RECEIPT.json"
    patient_receipt_path.write_text(json.dumps(patient_receipt), encoding="utf-8")
    conversion = {
        "format": CONVERSION_FORMAT,
        "status": CONVERSION_PASS,
        "contract": {"independent_validation_required_and_bound": True},
        "counts": {"rows": len(relation)},
        "outputs": {
            "interaction_relation": {
                "path": str(relation_path.resolve()), "sha256": _sha(relation_path)
            },
            "evidence_event": {
                "path": str(event_path.resolve()), "sha256": _sha(event_path)
            },
        },
    }
    conversion_path = root / "CONVERSION_MANIFEST.json"
    conversion_path.write_text(json.dumps(conversion), encoding="utf-8")
    paths = {
        "interaction_relation": relation_path,
        "empty_evidence_event": event_path,
        "exact_pathway_members": member_path,
        "formal_candidates": candidate_path,
        "conversion_manifest": conversion_path,
        "patient_fold_authority": patient_path,
        "patient_fold_receipt": patient_receipt_path,
    }
    return {
        "root": root,
        "paths": paths,
        "expected_hashes": {role: _sha(path) for role, path in paths.items()},
        "relation": relation,
        "members": members,
        "candidates": candidates,
    }


def _run(fixture: dict[str, object], output: Path) -> dict[str, object]:
    paths = fixture["paths"]
    assert isinstance(paths, dict)
    return materialize_streaming_evidence_stage(
        interaction_relation_path=paths["interaction_relation"],
        empty_evidence_event_path=paths["empty_evidence_event"],
        exact_pathway_members_path=paths["exact_pathway_members"],
        formal_candidates_path=paths["formal_candidates"],
        conversion_manifest_path=paths["conversion_manifest"],
        patient_fold_authority_path=paths["patient_fold_authority"],
        patient_fold_receipt_path=paths["patient_fold_receipt"],
        output_root=output,
        expected_hashes=fixture["expected_hashes"],
        strict_formal=False,
        batch_size=3,
        row_group_size=4,
        memory_limit="512MB",
        threads=1,
    )


def test_streamed_exact_tables_are_bidirectionally_equivalent_to_legacy(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "stage"
    manifest = _run(fixture, output)
    relation = fixture["relation"]
    members = fixture["members"]
    candidates = fixture["candidates"]
    assert isinstance(relation, pd.DataFrame)
    assert isinstance(members, pd.DataFrame)
    assert isinstance(candidates, pd.DataFrame)
    empty = pd.DataFrame(columns=EMPTY_EVENT_COLUMNS)
    legacy = build_exact_event_bags(
        empty, relation, members, candidate_universe=candidates,
        evidence_input_name=str(fixture["paths"]["empty_evidence_event"]),
        interaction_input_name=str(fixture["paths"]["interaction_relation"]),
    )
    legacy_candidate = materialize_candidate_events(legacy.events, candidates)
    streamed = pd.read_parquet(output / "source_exact_events.parquet")
    streamed_candidate = pd.read_parquet(output / "candidate_exact_events.parquet")
    event_columns = list(legacy.events.columns)
    pdt.assert_frame_equal(
        streamed[event_columns].sort_values(["event_id"]).reset_index(drop=True),
        legacy.events[event_columns].sort_values(["event_id"]).reset_index(drop=True),
        check_dtype=False,
    )
    candidate_columns = list(legacy_candidate.columns)
    pdt.assert_frame_equal(
        streamed_candidate[candidate_columns].sort_values(["event_id"]).reset_index(drop=True),
        legacy_candidate[candidate_columns].sort_values(["event_id"]).reset_index(drop=True),
        check_dtype=False,
    )
    physical = pd.read_parquet(output / "physical_interaction_facts.parquet")
    pdt.assert_frame_equal(
        physical[list(legacy.physical_facts.columns)].sort_values("physical_fact_id").reset_index(drop=True),
        legacy.physical_facts.sort_values("physical_fact_id").reset_index(drop=True),
        check_dtype=False,
    )
    rejected = pd.read_parquet(output / "rejected_raw_events.parquet")
    assert set(rejected.rejection_reason) == {
        "LNC_ID_UNMAPPED",
        "FAMILY_ONLY_NOT_BROADCAST_TO_EXACT",
        "OUTSIDE_EXACT_CANDIDATE_UNIVERSE",
    }
    assert manifest["exact_event_contract"]["family_broadcast_used"] is False
    assert manifest["formal_training_started"] is False
    assert manifest["real_6160707_server_preflight_executed"] is False


def test_mapping_states_folds_provenance_exclusion_and_bounded_reader(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "stage"
    manifest = _run(fixture, output)
    lineage = pd.read_parquet(output / "event_lineage.parquet")
    assert "AMBIGUOUS" in set(lineage.lncrna_mapping_state)
    ambiguous_lnc = lineage.loc[lineage.lncrna_mapping_state.eq("AMBIGUOUS")]
    assert ambiguous_lnc.partner_mapping_state.eq("MAPPED_UNIQUE").all()
    rejected = pd.read_parquet(output / "rejected_raw_events.parquet")
    assert "UNMAPPED" in set(rejected.lncrna_mapping_state)
    folded = pd.read_parquet(output / "candidate_exact_events_pair_folded.parquet")
    assert folded.groupby(["lncrna_id", "pathway_id"]).leakage_fold.nunique().max() == 1
    assert set(folded.leakage_fold) == set(range(5))
    for fold in range(5):
        audit = manifest["evaluation_provenance_exclusion_audit"]["folds"][str(fold)]
        assert audit["residual_train_evaluation_provenance_overlap_count"] == 0
        for split, expected in (
            ("evaluation", audit["evaluation_event_rows"]),
            ("train", audit["train_event_rows_after"]),
        ):
            batches = list(
                iter_fold_event_bag_batches(
                    output, fold=fold, split=split, arrow_batch_rows=2,
                    max_bags_per_batch=2, max_events_per_batch=3,
                )
            )
            assert sum(batch.event_rows for batch in batches) == expected
            for batch in batches:
                assert len(batch.bags) <= 2
                for bag in batch.bags:
                    assert all(
                        tuple(str(row[column]) for column in ("cancer_id", "lncrna_id", "pathway_id"))
                        == bag.key
                        for row in bag.events
                    )
    availability = pd.read_parquet(output / "candidate_event_availability.parquet")
    assert not availability.prediction_available.any()
    assert not availability.formal_training_started.any()

    class CoreFixture:
        @staticmethod
        def vector_for(*_key: str) -> np.ndarray:
            return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    trainer_batches = list(
        iter_fold_trainer_batches(
            output, fold=0, split="evaluation", core=CoreFixture(),
            max_events_per_bag=2, max_bags_per_batch=2,
        )
    )
    assert trainer_batches
    assert all(len(batch.examples) <= 2 for batch in trainer_batches)
    assert all(not batch.missing_core for batch in trainer_batches)
    assert all(example.event_count <= 2 for batch in trainer_batches for example in batch.examples)


def test_no_overwrite_hash_gate_and_test_firewall(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "stage"
    _run(fixture, output)
    with pytest.raises(FileExistsError):
        _run(fixture, output)
    validate_streaming_stage(output)
    folded = output / "candidate_exact_events_pair_folded.parquet"
    folded.write_bytes(folded.read_bytes() + b"tamper")
    with pytest.raises(StreamingEvidenceStageError, match="hash drift"):
        validate_streaming_stage(output)

    attack = _fixture(tmp_path / "attack")
    candidate_path = attack["paths"]["formal_candidates"]
    candidates = pd.read_parquet(candidate_path)
    candidates["sealed_test_label"] = 1
    _write_parquet(candidates, candidate_path)
    attack["expected_hashes"]["formal_candidates"] = _sha(candidate_path)
    with pytest.raises(StreamingEvidenceStageError, match="test firewall"):
        _run(attack, tmp_path / "attack_stage")


def test_input_order_does_not_change_event_or_physical_semantics(tmp_path: Path) -> None:
    first = _fixture(tmp_path / "first", shuffled=False)
    second = _fixture(tmp_path / "second", shuffled=True)
    out_first, out_second = tmp_path / "out_first", tmp_path / "out_second"
    _run(first, out_first)
    _run(second, out_second)
    for filename, keys in (
        ("source_exact_events.parquet", ["event_id"]),
        ("candidate_exact_events.parquet", ["event_id"]),
        ("physical_interaction_facts.parquet", ["physical_fact_id"]),
    ):
        left = pd.read_parquet(out_first / filename).sort_values(keys).reset_index(drop=True)
        right = pd.read_parquet(out_second / filename).sort_values(keys).reset_index(drop=True)
        # Raw lineage hashes/row indices are intentionally positional.  Exact
        # event and fact semantics are not.
        pdt.assert_frame_equal(left, right, check_dtype=False)
