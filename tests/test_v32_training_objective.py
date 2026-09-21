from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

from cc_hhgt.v32.training import (
    PREPARED_FORMAT,
    _aggregate_chunk_outputs,
    _candidate_chunk_schedule,
    _evaluate,
    _frozen_chunk_permutation,
    _global_loss_from_outputs,
    _runtime_chunk_contract,
    _validate_prepared,
)
from cc_hhgt.v32.patient_fold_authority import (
    FROZEN_V32_RECEIPT_SHA256,
    FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
    FORMAL_PREPARED_BINDING_FORMAT,
    validate_frozen_v32_patient_fold_binding,
)
from cc_hhgt.v32.formal_graph_authority import (
    FOLD_ARTIFACT_IDS,
    GRAPH_PAYLOAD_BINDING_FORMAT,
    GRAPH_PAYLOAD_BINDING_STATUS,
    STATIC_ARTIFACT_IDS,
)
from cc_hhgt.v32.contracts import dataframe_sha256
from cc_hhgt.v32.safe_graph import EDGE_KEYS


torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]


def _patient_binding():
    authority_root = ROOT / "artifacts/v32_patient_fold_authority_20260829_r1"
    audit = validate_frozen_v32_patient_fold_binding(
        authority_root / "SAMPLE_PATIENT_FOLD_MAP.tsv",
        authority_root / "PATIENT_FOLD_AUTHORITY_RECEIPT.json",
    )
    return {
        "format": FORMAL_PREPARED_BINDING_FORMAT,
        "status": "PASS_FROZEN_V32_PATIENT_FIRST_AUTHORITY",
        "authority": audit,
        "sample_patient_fold_map": {"sha256": audit["manifest_sha256"]},
        "authority_receipt": {"sha256": audit["receipt_sha256"]},
        "sample_id_patient_fallback_used": False,
        "legacy_patient_fold_manifest_used": False,
    }


def _graph_bundle():
    nodes = pd.DataFrame({"node_type": ["lncRNA", "pathway", "cancer"], "canonical_id": ["L1", "P1", "BRCA"]})
    edges = pd.DataFrame({
        "source_type": ["lncRNA"], "source_id": ["L1"],
        "relation_type": ["expressed_in"], "target_type": ["cancer"],
        "target_id": ["BRCA"],
    })
    return SimpleNamespace(nodes=nodes, edges=edges)


def _graph_binding(bundle):
    node_sha = dataframe_sha256(bundle.nodes, ["node_type", "canonical_id"])
    edge_sha = dataframe_sha256(bundle.edges, list(EDGE_KEYS))
    return {
        "format": GRAPH_PAYLOAD_BINDING_FORMAT,
        "status": GRAPH_PAYLOAD_BINDING_STATUS,
        "outer_fold": 0,
        "variant": "G2",
        "receipt": {"sha256": "a" * 64},
        "patient_fold_authority": {
            "manifest_sha256": FROZEN_V32_SAMPLE_PATIENT_MAP_SHA256,
            "receipt_sha256": FROZEN_V32_RECEIPT_SHA256,
        },
        "static_artifacts": {
            key: {"path": f"/authority/{key}.parquet", "sha256": "d" * 64}
            for key in STATIC_ARTIFACT_IDS
        },
        "fold_artifacts": {
            key: {"path": f"/authority/fold0/{key}.parquet", "sha256": "e" * 64}
            for key in FOLD_ARTIFACT_IDS
        },
        "graph": {
            "node_sha256": node_sha,
            "edge_sha256": edge_sha,
            "master_edge_sha256": edge_sha,
            "edge_count": len(bundle.edges),
            "same_relation_schema_all_variants": True,
        },
        "gates": {
            "frozen_patient_first_authority": True,
            "outer_train_expression_only": True,
            "outer_train_coexpression_only": True,
            "static_evidence_outcome_free": True,
            "historical_graph_rows_used": False,
            "legacy_graph_root_fallback_used": False,
            "toy_or_synthetic_fallback_used": False,
            "same_node_and_relation_schema_all_variants": True,
            "all_required_fold_declarations_present": True,
        },
    }


def _rows():
    logits = torch.tensor([4.0, -3.0, 0.5, 2.0, -1.0, 1.5], dtype=torch.float64)
    direction = torch.tensor([-2.0, 1.0, 0.2, -0.4, 1.7, -1.2], dtype=torch.float64)
    residual = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=torch.float64)
    proxy = torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float64)
    weak = torch.tensor([False, False, False, True, False, False])
    direction_label = torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.float64)
    direction_available = torch.tensor([True, True, False, True, False, True])
    return logits, direction, residual, proxy, weak, direction_label, direction_available


def _parts(boundaries):
    rows = _rows()
    outputs = []
    batches = []
    for start, stop in boundaries:
        outputs.append(
            {
                "final_logit": rows[0][start:stop],
                "direction_logit": rows[1][start:stop],
                "raw_graph_residual": rows[2][start:stop],
            }
        )
        batches.append(
            {
                "proxy_label": rows[3][start:stop],
                "weak_positive": rows[4][start:stop],
                "direction_label": rows[5][start:stop],
                "direction_available": rows[6][start:stop],
            }
        )
    return outputs, batches


def test_global_objective_is_invariant_to_storage_batch_boundaries() -> None:
    one_output, one_batch = _parts([(0, 6)])
    split_output, split_batch = _parts([(0, 1), (1, 4), (4, 6)])
    expected = _global_loss_from_outputs(
        one_output,
        one_batch,
        torch=torch,
        direction_loss_weight=0.25,
        shrinkage=1e-4,
    )
    observed = _global_loss_from_outputs(
        split_output,
        split_batch,
        torch=torch,
        direction_loss_weight=0.25,
        shrinkage=1e-4,
    )
    assert float(observed) == pytest.approx(float(expected), rel=0, abs=1e-14)


def test_independently_clamped_batch_average_is_not_the_global_nnpu_objective() -> None:
    outputs, batches = _parts([(0, 1), (1, 4), (4, 6)])
    global_loss = _global_loss_from_outputs(
        outputs,
        batches,
        torch=torch,
        direction_loss_weight=0.0,
        shrinkage=0.0,
    )
    old_batch_average = torch.stack(
        [
            _global_loss_from_outputs(
                [output],
                [batch],
                torch=torch,
                direction_loss_weight=0.0,
                shrinkage=0.0,
            )
            for output, batch in zip(outputs, batches)
        ]
    ).mean()
    assert abs(float(global_loss - old_batch_average)) > 1e-3


class _IdentityEncoder:
    def encode(self, graph):
        return None


class _IdentityModel:
    def __init__(self):
        self.encoder = _IdentityEncoder()

    def eval(self):
        return self

    def __call__(
        self,
        graph,
        candidate_batch,
        base_logit,
        conservation_context,
        graph_available,
        *,
        admitted,
        encoded,
    ):
        return {
            "final_logit": base_logit,
            "direction_logit": conservation_context[:, 0],
            "raw_graph_residual": conservation_context[:, 1],
        }


def _evaluation_batches(boundaries):
    rows = _rows()
    result = []
    for start, stop in boundaries:
        count = stop - start
        result.append(
            {
                "candidate_batch": {
                    "l": torch.arange(count),
                    "p": torch.arange(count),
                    "c": torch.zeros(count, dtype=torch.long),
                },
                "base_logit": rows[0][start:stop].to(torch.float32),
                "conservation_context": torch.column_stack(
                    [rows[1][start:stop], rows[2][start:stop]]
                ).to(torch.float32),
                "proxy_label": rows[3][start:stop].to(torch.float32),
                "weak_positive": rows[4][start:stop],
                "direction_label": rows[5][start:stop].to(torch.float32),
                "direction_available": rows[6][start:stop],
                "graph_available": torch.ones(count, dtype=torch.bool),
            }
        )
    return result


def test_validation_loss_is_invariant_to_storage_partition() -> None:
    kwargs = {
        "torch": torch,
        "device": "cpu",
        "direction_loss_weight": 0.25,
        "shrinkage": 1e-4,
    }
    one = _evaluate(_IdentityModel(), None, _evaluation_batches([(0, 6)]), **kwargs)
    split = _evaluate(
        _IdentityModel(), None, _evaluation_batches([(0, 2), (2, 3), (3, 6)]), **kwargs
    )
    assert split == pytest.approx(one, rel=0, abs=1e-14)


def test_candidate_chunk_schedule_is_exact_cartesian_and_deterministic() -> None:
    chunks = (0, 1, 2)
    permutation = _frozen_chunk_permutation(chunks, seed=20260726, fold=2)
    schedule = _candidate_chunk_schedule(
        batch_count=4, chunks=chunks, permutation=permutation
    )
    assert len(schedule) == 3
    assert all(len(candidate_pass) == 4 for candidate_pass in schedule)
    flattened = [pair for candidate_pass in schedule for pair in candidate_pass]
    assert len(flattened) == 12
    assert set(flattened) == {
        (batch, chunk) for batch in range(4) for chunk in chunks
    }
    assert schedule == _candidate_chunk_schedule(
        batch_count=4,
        chunks=chunks,
        permutation=_frozen_chunk_permutation(chunks, seed=20260726, fold=2),
    )


def test_runtime_chunk_contract_requires_complete_backbone_and_uses_variable_mass() -> None:
    rows = [
        {
            "runtime_chunk": -1,
            "runtime_residency": "resident_backbone",
            "canonical_edge_id": edge,
        }
        for edge in ("s1", "s2")
    ]
    rows.extend(
        [
            {"runtime_chunk": 0, "runtime_residency": "rotating_variable", "canonical_edge_id": "v1"},
            {"runtime_chunk": 0, "runtime_residency": "rotating_variable", "canonical_edge_id": "v2"},
            {"runtime_chunk": 1, "runtime_residency": "rotating_variable", "canonical_edge_id": "v3"},
        ]
    )
    chunks, weights = _runtime_chunk_contract(
        SimpleNamespace(runtime_schedule=pd.DataFrame(rows))
    )
    assert chunks == (0, 1)
    assert weights == pytest.approx({0: 2 / 3, 1: 1 / 3})
    broken = pd.DataFrame(rows)
    broken.loc[broken.canonical_edge_id.eq("s2"), "runtime_chunk"] = 0
    with pytest.raises(RuntimeError, match="resident backbone|Resident rows"):
        _runtime_chunk_contract(SimpleNamespace(runtime_schedule=broken))


def test_chunk_logit_aggregation_uses_canonical_order_and_float64_weights() -> None:
    def output(value):
        tensor = torch.tensor([value, -value], dtype=torch.float32)
        return {
            "final_logit": tensor,
            "direction_logit": tensor + 1,
            "raw_graph_residual": tensor - 1,
        }

    by_chunk = {2: [output(3.0)], 0: [output(1.0)], 1: [output(2.0)]}
    weights = {0: 0.2, 1: 0.3, 2: 0.5}
    observed = _aggregate_chunk_outputs(
        by_chunk, chunks=(0, 1, 2), weights=weights, torch=torch
    )[0]
    assert observed["final_logit"].dtype == torch.float64
    assert observed["final_logit"].tolist() == pytest.approx([2.3, -2.3])
    reversed_mapping = {key: by_chunk[key] for key in (1, 2, 0)}
    second = _aggregate_chunk_outputs(
        reversed_mapping, chunks=(0, 1, 2), weights=weights, torch=torch
    )[0]
    assert torch.equal(observed["final_logit"], second["final_logit"])


def _prepared_payload():
    bundle = _graph_bundle()
    batch = {
        "candidate_batch": {"l": torch.tensor([0]), "p": torch.tensor([0]), "c": torch.tensor([0])},
        "base_logit": torch.tensor([0.0]),
        "conservation_context": torch.zeros((1, 2)),
        "proxy_label": torch.tensor([1.0]),
        "weak_positive": torch.tensor([False]),
        "direction_label": torch.tensor([1.0]),
        "direction_available": torch.tensor([True]),
    }
    return {
        "prepared_format": PREPARED_FORMAT,
        "patient_fold": 0,
        "bundle": bundle,
        "feature_dim": 2,
        "legacy_model_config": {},
        "train_batches": [batch],
        "validation_batches": [batch],
        "patient_fold_authority": _patient_binding(),
        "formal_graph_variant": "G2",
        "formal_graph_authority": _graph_binding(bundle),
        "label_contract": {
            "train_direction_label_source": "train_discovery_effect",
            "validation_direction_label_source": "validation_replication_effect",
            "heldout_direction_is_model_input": False,
            "test_labels_in_training_payload": False,
            "test_logits_in_training_payload": False,
            "test_metrics_computed_before_winner_lock": False,
        },
    }


def test_prepared_contract_hard_fails_test_before_lock_and_direction_reuse() -> None:
    payload = _prepared_payload()
    _validate_prepared(payload, fold=0, artifact_hashes={})
    payload["test_batches"] = payload["validation_batches"]
    with pytest.raises(RuntimeError, match="sealed-test firewall"):
        _validate_prepared(payload, fold=0, artifact_hashes={})
    payload = _prepared_payload()
    payload["label_contract"]["validation_direction_label_source"] = "train_discovery_effect"
    with pytest.raises(RuntimeError, match="governance contract is unsafe"):
        _validate_prepared(payload, fold=0, artifact_hashes={})
    payload = _prepared_payload()
    payload["validation_batches"][0]["candidate_batch"]["replication_effect"] = torch.tensor([0.5])
    with pytest.raises(RuntimeError, match="held-out labels as model inputs"):
        _validate_prepared(payload, fold=0, artifact_hashes={})
