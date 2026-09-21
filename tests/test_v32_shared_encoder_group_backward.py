from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from cc_hhgt.v32 import training  # noqa: E402
from cc_hhgt.v32.training import (  # noqa: E402
    _global_loss_from_outputs,
    shared_encoder_conventional_group_backward,
)


class _CountingEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.35))
        self.forward_calls = 0

    def encode(self, graph):
        self.forward_calls += 1
        return graph.reshape(()) * self.scale


class _CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _CountingEncoder()
        self.membership_scale = torch.nn.Parameter(torch.tensor(0.45))
        self.direction_scale = torch.nn.Parameter(torch.tensor(-0.20))
        self.decoder_forward_calls = 0

    def forward(
        self,
        graph,
        candidate_batch,
        base_logit,
        conservation_context,
        graph_available,
        *,
        admitted,
        encoded=None,
    ):
        assert admitted is True
        self.decoder_forward_calls += 1
        if encoded is None:
            encoded = self.encoder.encode(graph)
        feature = candidate_batch["feature"].reshape(-1).to(dtype=torch.float32)
        context = conservation_context[:, 0].to(dtype=torch.float32)
        residual = self.membership_scale * feature + encoded + 0.05 * context
        return {
            "final_logit": base_logit.reshape(-1) + residual,
            "direction_logit": self.direction_scale * feature + 0.25 * encoded,
            "raw_graph_residual": residual,
        }


def _batch(rows: int, offset: int) -> dict[str, object]:
    feature = torch.linspace(-0.7, 0.9, rows) + 0.1 * offset
    proxy = torch.tensor(
        [1.0 if (offset + index) % 3 == 0 else 0.0 for index in range(rows)]
    )
    return {
        "candidate_batch": {"feature": feature},
        "base_logit": torch.linspace(-0.2, 0.3, rows),
        "conservation_context": torch.stack(
            (feature, feature.square(), -feature, torch.ones_like(feature)), dim=1
        ),
        "graph_available": torch.ones(rows, dtype=torch.bool),
        "proxy_label": proxy,
        "weak_positive": proxy.to(dtype=torch.bool) & (feature < 0),
        "direction_label": (feature > 0).to(dtype=torch.float32),
        "direction_available": torch.tensor(
            [index % 2 == 0 for index in range(rows)], dtype=torch.bool
        ),
    }


def _precision() -> dict[str, object]:
    return {"active": "fp32", "autocast_enabled": False}


def _graph_counter():
    calls: list[int] = []

    def graph_for_chunk(chunk: int):
        calls.append(int(chunk))
        return torch.tensor(float(chunk + 2))

    return calls, graph_for_chunk


def test_shared_group_matches_conventional_one_encode_loss_and_gradients(
    monkeypatch,
) -> None:
    candidate_batches = [_batch(rows, index) for index, rows in enumerate((3, 4, 2, 5))]
    work_items = [(batch, 7) for batch in candidate_batches]
    observed_model = _CountingModel()
    reference_model = _CountingModel()
    reference_model.load_state_dict(observed_model.state_dict())

    original_global_loss = training._global_loss_from_outputs
    global_loss_call_count = 0

    def counted_global_loss(*args, **kwargs):
        nonlocal global_loss_call_count
        global_loss_call_count += 1
        return original_global_loss(*args, **kwargs)

    monkeypatch.setattr(training, "_global_loss_from_outputs", counted_global_loss)

    graph_calls, graph_for_chunk = _graph_counter()
    plan, telemetry = shared_encoder_conventional_group_backward(
        observed_model,
        work_items,
        graph_for_chunk,
        torch=torch,
        device="cpu",
        precision=_precision(),
        direction_loss_weight=0.25,
        shrinkage=1e-4,
    )
    assert global_loss_call_count == 1
    monkeypatch.setattr(training, "_global_loss_from_outputs", original_global_loss)

    reference_graph = torch.tensor(9.0)
    reference_encoded = reference_model.encoder.encode(reference_graph)
    reference_outputs = [
        reference_model(
            reference_graph,
            batch["candidate_batch"],
            batch["base_logit"],
            batch["conservation_context"],
            batch["graph_available"],
            admitted=True,
            encoded=reference_encoded,
        )
        for batch in candidate_batches
    ]
    reference_loss = _global_loss_from_outputs(
        reference_outputs,
        candidate_batches,
        torch=torch,
        direction_loss_weight=0.25,
        shrinkage=1e-4,
        reduction_dtype=torch.float32,
    )
    reference_loss.backward()

    assert plan.objective_value == pytest.approx(float(reference_loss.detach()), rel=1e-6)
    assert graph_calls == [7]
    assert telemetry.unique_chunks == 1
    assert telemetry.group_rows == 14
    assert telemetry.encoder_forward_calls == 1
    assert telemetry.decoder_forward_calls == 4
    assert telemetry.global_loss_calls == 1
    assert telemetry.backward_calls == 1
    assert observed_model.encoder.forward_calls == 1
    assert observed_model.decoder_forward_calls == 4
    for (observed_name, observed), (reference_name, reference) in zip(
        observed_model.named_parameters(), reference_model.named_parameters(), strict=True
    ):
        assert observed_name == reference_name
        assert observed.grad is not None and reference.grad is not None
        torch.testing.assert_close(observed.grad, reference.grad, rtol=1e-6, atol=1e-7)


def test_shared_group_supports_canonical_partial_final_group() -> None:
    model = _CountingModel()
    rows = (4, 4, 3)
    work_items = [(_batch(count, index), 2) for index, count in enumerate(rows)]
    graph_calls, graph_for_chunk = _graph_counter()
    _, telemetry = shared_encoder_conventional_group_backward(
        model,
        work_items,
        graph_for_chunk,
        torch=torch,
        device="cpu",
        precision=_precision(),
        direction_loss_weight=0.25,
        shrinkage=1e-4,
    )
    assert graph_calls == [2]
    assert telemetry.group_rows == sum(rows)
    assert telemetry.encoder_forward_calls == 1
    assert telemetry.decoder_forward_calls == 3
    assert telemetry.global_loss_calls == 1
    assert telemetry.backward_calls == 1


def test_shared_group_rejects_empty_mixed_and_unknown_chunk_work() -> None:
    model = _CountingModel()
    graph_calls, graph_for_chunk = _graph_counter()
    with pytest.raises(RuntimeError, match="empty"):
        shared_encoder_conventional_group_backward(
            model,
            [],
            graph_for_chunk,
            torch=torch,
            device="cpu",
            precision=_precision(),
            direction_loss_weight=0.25,
            shrinkage=1e-4,
        )
    with pytest.raises(RuntimeError, match="REQUIRES_UNIQUE_CHUNK"):
        shared_encoder_conventional_group_backward(
            model,
            [(_batch(2, 0), 0), (_batch(2, 1), 1)],
            graph_for_chunk,
            torch=torch,
            device="cpu",
            precision=_precision(),
            direction_loss_weight=0.25,
            shrinkage=1e-4,
        )
    assert graph_calls == []

    def reject_unknown(chunk: int):
        raise RuntimeError(f"UNKNOWN_RUNTIME_CHUNK={chunk}")

    with pytest.raises(RuntimeError, match="UNKNOWN_RUNTIME_CHUNK"):
        shared_encoder_conventional_group_backward(
            model,
            [(_batch(2, 0), 999)],
            reject_unknown,
            torch=torch,
            device="cpu",
            precision=_precision(),
            direction_loss_weight=0.25,
            shrinkage=1e-4,
        )
