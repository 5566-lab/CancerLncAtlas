from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from cc_hhgt.v32.orchestration import (  # noqa: E402
    ExecutionPolicy,
    build_task_manifest,
    write_task_manifest,
)
from cc_hhgt.v32.training import (  # noqa: E402
    _capture_torch_rng_state,
    _global_loss_from_outputs,
    _loss_plan_from_probe_outputs,
    _streaming_fragment_loss,
    _torch_rng_states_equal,
    load_prepared_artifact_from_authorized_handle,
    optimizer_step_with_guards,
    resolve_mixed_precision_contract,
    resolve_runtime_training_plan,
    run_authorized_task,
    split_work_items_for_oom_fallback,
    streaming_group_backward,
    validate_prepared_artifact_against_input_manifest,
)
from cc_hhgt.v32.training_guard import TrainingAuthorizationError  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def test_formal_runtime_seals_best_checkpoint_and_emits_validation_heartbeat() -> None:
    source = (ROOT / "cc_hhgt/v32/training.py").read_text(encoding="utf-8")
    assert '"status": "VALIDATION_HEARTBEAT"' in source
    assert 'best_model_state_sha256 = _file_sha256(best_path)' in source
    assert '"best_model_state_sha256": best_model_state_sha256' in source
    assert '"best_model_state_size_bytes": best_model_state_size_bytes' in source


def test_formal_training_entry_rejects_untrusted_mapping_context() -> None:
    with pytest.raises(TrainingAuthorizationError, match="guarded"):
        run_authorized_task({})


def _authorized_policy() -> ExecutionPolicy:
    return ExecutionPolicy(
        execution_mode="TRAINING",
        training_authorized=True,
        paid_enabled=True,
        max_paid_hours=96,
        max_cost_cny=210,
    )


def test_real_paid_manifest_empty_blocked_reason_survives_bash_read(
    tmp_path: Path,
) -> None:
    run_id = "v32-test-paid"
    manifest = write_task_manifest(
        tmp_path / "TASK_MANIFEST.tsv",
        build_task_manifest(
            run_id=run_id,
            seed=20260726,
            policy=_authorized_policy(),
            owner="paid_gpu",
            hardware_class="PAID_PREEMPTIBLE_GPU",
            paid_task=True,
        ),
    )
    normalized = tmp_path / "TASK_MANIFEST.usv"
    command = [
        shutil.which("python") or "python",
        str(ROOT / "scripts/normalize_v32_task_manifest.py"),
        "--manifest",
        str(manifest),
        "--output",
        str(normalized),
        "--run-id",
        run_id,
        "--owner",
        "paid_gpu",
        "--hardware-class",
        "PAID_PREEMPTIBLE_GPU",
        "--paid-task",
        "true",
        "--expected-folds",
        "0,1,2,3,4",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    rows = normalized.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 5
    assert all(len(row.split("\x1f")) == 11 for row in rows)
    assert all(row.split("\x1f")[9] == "" for row in rows)
    assert all(row.split("\x1f")[10] == "true" for row in rows)

    bash = shutil.which("bash")
    if bash:
        script = r"""
while IFS=$'\x1f' read -r task_id run_id task_type model fold seed owner hardware status blocked paid; do
  test -z "$blocked" || exit 31
  test "$paid" = true || exit 32
done < "$1"
"""
        bash_result = subprocess.run(
            [bash, "-c", script, "manifest-check", normalized.as_posix()],
            capture_output=True,
            text=True,
            check=False,
        )
        assert bash_result.returncode == 0, bash_result.stderr


def test_manifest_preflight_rejects_nonempty_blocked_reason(tmp_path: Path) -> None:
    run_id = "v32-test-paid"
    manifest = write_task_manifest(
        tmp_path / "TASK_MANIFEST.tsv",
        build_task_manifest(
            run_id=run_id,
            seed=1,
            policy=_authorized_policy(),
            owner="paid_gpu",
            hardware_class="PAID_PREEMPTIBLE_GPU",
            paid_task=True,
        ),
    )
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace("\t\ttrue\n", "\tSHOULD_BLOCK\ttrue\n", 1), encoding="utf-8")
    completed = subprocess.run(
        [
            shutil.which("python") or "python",
            str(ROOT / "scripts/normalize_v32_task_manifest.py"),
            "--manifest",
            str(manifest),
            "--output",
            str(tmp_path / "out.usv"),
            "--run-id",
            run_id,
            "--owner",
            "paid_gpu",
            "--hardware-class",
            "PAID_PREEMPTIBLE_GPU",
            "--paid-task",
            "true",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "blocked_reason" in completed.stderr


def test_bf16_contract_is_real_on_cuda_and_cpu_safe() -> None:
    cpu = resolve_mixed_precision_contract(
        torch, {"mixed_precision": "bf16"}, device="cpu"
    )
    assert cpu == {
        "requested": "bf16",
        "active": "fp32",
        "autocast_enabled": False,
        "device_type": "cpu",
        "cpu_test_fallback": True,
    }
    assert resolve_mixed_precision_contract(
        torch, {"mixed_precision": False}, device="cpu"
    )["active"] == "fp32"
    with pytest.raises(RuntimeError, match="Unsupported"):
        resolve_mixed_precision_contract(
            torch, {"mixed_precision": "fp16"}, device="cpu"
        )


def test_runtime_oom_plan_preserves_optimizer_group_and_rejects_edge_fake() -> None:
    plan = resolve_runtime_training_plan(
        {
            "candidate_microbatch_size": 8,
            "gradient_accumulation": 4,
            "oom_fallback_microbatch_size": 4,
            "oom_fallback_gradient_accumulation": 8,
        }
    )
    assert plan.candidate_microbatch_size * plan.gradient_accumulation == (
        plan.fallback_microbatch_size * plan.fallback_gradient_accumulation
    )
    with pytest.raises(RuntimeError, match="sealed prepared graphs"):
        resolve_runtime_training_plan(
            {
                "candidate_microbatch_size": 8,
                "gradient_accumulation": 4,
                "oom_fallback_microbatch_size": 4,
                "oom_fallback_gradient_accumulation": 8,
                "oom_fallback_edge_chunk_size": 50_000,
            }
        )


@pytest.mark.parametrize(
    ("logits", "labels", "weak", "with_direction"),
    [
        ([-2.0, -0.5, 0.2, 1.0], [1.0, 0.0, 1.0, 0.0], [False, False, True, False], True),
        ([20.0, -20.0, 18.0, -18.0], [1.0, 0.0, 1.0, 0.0], [False] * 4, True),
        ([-1.0, 0.0, 1.0, 2.0], [0.0] * 4, [False] * 4, False),
        ([-1.0, 0.0, 1.0, 2.0], [1.0] * 4, [False, True, False, True], False),
    ],
)
def test_streaming_loss_fragments_match_monolithic_gradient_edge_cases(
    logits, labels, weak, with_direction
) -> None:
    direct_logits = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
    direct_direction = torch.linspace(-0.3, 0.6, 4, dtype=torch.float64, requires_grad=True)
    direct_residual = torch.linspace(-0.2, 0.4, 4, dtype=torch.float64, requires_grad=True)
    replay_logits = direct_logits.detach().clone().requires_grad_(True)
    replay_direction = direct_direction.detach().clone().requires_grad_(True)
    replay_residual = direct_residual.detach().clone().requires_grad_(True)
    slices = (slice(0, 2), slice(2, 4))

    def make(outputs_logits, outputs_direction, outputs_residual):
        outputs = []
        batches = []
        for section in slices:
            outputs.append(
                {
                    "final_logit": outputs_logits[section],
                    "direction_logit": outputs_direction[section],
                    "raw_graph_residual": outputs_residual[section],
                }
            )
            batch = {
                "proxy_label": torch.tensor(labels, dtype=torch.float64)[section],
                "weak_positive": torch.tensor(weak, dtype=torch.bool)[section],
            }
            if with_direction:
                batch.update(
                    {
                        "direction_label": torch.tensor(
                            [1.0, 0.0, 0.0, 1.0], dtype=torch.float64
                        )[section],
                        "direction_available": torch.tensor(
                            [True, False, True, True]
                        )[section],
                    }
                )
            batches.append(batch)
        return outputs, batches

    direct_outputs, direct_batches = make(
        direct_logits, direct_direction, direct_residual
    )
    direct = _global_loss_from_outputs(
        direct_outputs,
        direct_batches,
        torch=torch,
        direction_loss_weight=0.25,
        shrinkage=1e-4,
        reduction_dtype=torch.float32,
    )
    direct.backward()

    replay_outputs, replay_batches = make(
        replay_logits, replay_direction, replay_residual
    )
    plan = _loss_plan_from_probe_outputs(
        [{key: value.detach() for key, value in output.items()} for output in replay_outputs],
        replay_batches,
        torch=torch,
        direction_loss_weight=0.25,
        shrinkage=1e-4,
    )
    replay_total = 0.0
    for output, batch in zip(replay_outputs, replay_batches, strict=True):
        fragment = _streaming_fragment_loss(
            output,
            batch,
            plan,
            torch=torch,
            direction_loss_weight=0.25,
            shrinkage=1e-4,
        )
        replay_total += float(fragment.detach())
        fragment.backward()
    assert replay_total == pytest.approx(float(direct.detach()), rel=2e-6, abs=2e-7)
    assert torch.allclose(replay_logits.grad, direct_logits.grad, rtol=2e-5, atol=2e-6)
    assert torch.allclose(replay_residual.grad, direct_residual.grad, rtol=2e-5, atol=2e-6)
    if with_direction:
        assert torch.allclose(
            replay_direction.grad, direct_direction.grad, rtol=2e-5, atol=2e-6
        )
    else:
        assert replay_direction.grad is None and direct_direction.grad is None


def test_prepared_pt_path_and_content_are_revalidated_before_load(
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "PATIENT_FOLD_2.pt"
    prepared.write_bytes(b"authorized fold bytes")
    digest = hashlib.sha256(prepared.read_bytes()).hexdigest()
    manifest = tmp_path / "INPUT_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "fold_inputs": [
                    {"fold": 2, "path": str(prepared.resolve()), "sha256": digest}
                ]
            }
        ),
        encoding="utf-8",
    )
    assert (
        validate_prepared_artifact_against_input_manifest(
            manifest, fold=2, prepared_path=prepared
        )
        == digest
    )
    other = tmp_path / "other.pt"
    other.write_bytes(prepared.read_bytes())
    with pytest.raises(RuntimeError, match="path differs"):
        validate_prepared_artifact_against_input_manifest(
            manifest, fold=2, prepared_path=other
        )
    prepared.write_bytes(b"tampered fold bytes")
    with pytest.raises(RuntimeError, match="SHA256 differs"):
        validate_prepared_artifact_against_input_manifest(
            manifest, fold=2, prepared_path=prepared
        )


def test_prepared_pt_is_hashed_and_loaded_through_the_same_handle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("V32_PREPARED_ARTIFACT_MMAP", raising=False)
    prepared = tmp_path / "PATIENT_FOLD_1.pt"
    expected_payload = {"patient_fold": 1, "tensor": torch.arange(4)}
    torch.save(expected_payload, prepared)
    digest = hashlib.sha256(prepared.read_bytes()).hexdigest()
    manifest = tmp_path / "INPUT_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "fold_inputs": [
                    {"fold": 1, "path": str(prepared.resolve()), "sha256": digest}
                ]
            }
        ),
        encoding="utf-8",
    )

    class RecordingTorch:
        def __init__(self) -> None:
            self.source = None

        def load(self, source, **kwargs):
            self.source = source
            assert hasattr(source, "read") and hasattr(source, "fileno")
            return torch.load(source, **kwargs)

    recording = RecordingTorch()
    observed_payload, observed_digest = load_prepared_artifact_from_authorized_handle(
        recording,
        manifest,
        fold=1,
        prepared_path=prepared,
    )
    assert recording.source is not None
    assert recording.source.closed
    assert observed_digest == digest
    assert observed_payload["patient_fold"] == 1
    assert torch.equal(observed_payload["tensor"], expected_payload["tensor"])


@pytest.mark.skipif(
    os.name != "posix" or not Path("/proc/self/fd").is_dir(),
    reason="authorized mmap loading requires Linux /proc/self/fd",
)
def test_prepared_pt_mmap_uses_the_authorized_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = tmp_path / "PATIENT_FOLD_4.pt"
    expected_payload = {"patient_fold": 4, "tensor": torch.arange(16)}
    torch.save(expected_payload, prepared)
    digest = hashlib.sha256(prepared.read_bytes()).hexdigest()
    manifest = tmp_path / "INPUT_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "fold_inputs": [
                    {"fold": 4, "path": str(prepared.resolve()), "sha256": digest}
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("V32_PREPARED_ARTIFACT_MMAP", "1")

    class RecordingTorch:
        def __init__(self) -> None:
            self.source: str | None = None
            self.kwargs: dict[str, object] = {}

        def load(self, source, **kwargs):
            self.source = str(source)
            self.kwargs = dict(kwargs)
            assert self.source.startswith("/proc/self/fd/")
            assert Path(self.source).is_file()
            assert os.stat(self.source).st_ino == prepared.stat().st_ino
            return torch.load(source, **kwargs)

    recording = RecordingTorch()
    observed_payload, observed_digest = load_prepared_artifact_from_authorized_handle(
        recording,
        manifest,
        fold=4,
        prepared_path=prepared,
    )
    assert recording.source is not None
    assert recording.kwargs == {
        "map_location": "cpu",
        "weights_only": False,
        "mmap": True,
    }
    assert observed_digest == digest
    assert observed_payload["patient_fold"] == 4
    assert torch.equal(observed_payload["tensor"], expected_payload["tensor"])


def test_prepared_pt_in_place_drift_during_load_is_rejected(tmp_path: Path) -> None:
    prepared = tmp_path / "PATIENT_FOLD_3.pt"
    torch.save({"patient_fold": 3}, prepared)
    digest = hashlib.sha256(prepared.read_bytes()).hexdigest()
    manifest = tmp_path / "INPUT_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "fold_inputs": [
                    {"fold": 3, "path": str(prepared.resolve()), "sha256": digest}
                ]
            }
        ),
        encoding="utf-8",
    )

    class MutatingTorch:
        @staticmethod
        def load(source, **kwargs):
            payload = torch.load(source, **kwargs)
            stat = prepared.stat()
            changed = stat.st_mtime_ns + 2_000_000_000
            prepared.touch()
            import os

            os.utime(prepared, ns=(stat.st_atime_ns, changed))
            return payload

    with pytest.raises(RuntimeError, match="changed while loading"):
        load_prepared_artifact_from_authorized_handle(
            MutatingTorch,
            manifest,
            fold=3,
            prepared_path=prepared,
        )


class _ToyEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(5, 6)
        self.dropout = torch.nn.Dropout(0.25)
        self.live_grad_forwards = 0
        self.peak_live_grad_forwards = 0

    def encode(self, graph):
        value = self.dropout(torch.tanh(self.projection(graph)))
        if torch.is_grad_enabled():
            self.live_grad_forwards += 1
            self.peak_live_grad_forwards = max(
                self.peak_live_grad_forwards, self.live_grad_forwards
            )

            def release(gradient):
                self.live_grad_forwards -= 1
                return gradient

            value.register_hook(release)
        return {"lncRNA": value, "pathway": value, "cancer": value}


class _ToyResidual(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ToyEncoder()
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(18, 8),
            torch.nn.GELU(),
            torch.nn.Dropout(0.20),
        )
        self.residual = torch.nn.Linear(8, 1)
        self.direction = torch.nn.Linear(18, 1)
        self.register_buffer("forward_count", torch.zeros((), dtype=torch.int64))

    def forward(
        self,
        graph,
        candidate_batch,
        base_logit,
        conservation_context,
        graph_available=None,
        *,
        admitted=True,
        encoded=None,
    ):
        del conservation_context, graph_available, admitted
        if self.training:
            self.forward_count.add_(1)
        encoded = self.encoder.encode(graph) if encoded is None else encoded
        pieces = [
            encoded["lncRNA"][candidate_batch["l"]],
            encoded["pathway"][candidate_batch["p"]],
            encoded["cancer"][candidate_batch["c"]],
        ]
        features = torch.cat(pieces, dim=1)
        raw = self.residual(self.decoder(features)).squeeze(-1)
        return {
            "final_logit": base_logit + torch.tanh(raw),
            "direction_logit": self.direction(features).squeeze(-1),
            "raw_graph_residual": raw,
        }


def _toy_batch(offset: int) -> dict[str, object]:
    index = torch.tensor([(offset + i) % 7 for i in range(4)], dtype=torch.long)
    return {
        "candidate_batch": {"l": index, "p": (index + 1) % 7, "c": (index + 2) % 7},
        "base_logit": torch.linspace(-0.4, 0.5, 4),
        "conservation_context": torch.zeros(4, 3),
        "graph_available": torch.ones(4, dtype=torch.bool),
        "proxy_label": torch.tensor([1.0, 0.0, 1.0, 0.0]),
        "weak_positive": torch.tensor([False, False, True, False]),
        "direction_label": torch.tensor([1.0, 0.0, 0.0, 1.0]),
        "direction_available": torch.tensor([True, False, True, True]),
    }


def _forward_group(model, graph_for_chunk, work_items):
    outputs = []
    batches = []
    for batch, chunk in work_items:
        graph = graph_for_chunk(chunk)
        outputs.append(
            model(
                graph,
                batch["candidate_batch"],
                batch["base_logit"],
                batch["conservation_context"],
                batch["graph_available"],
                admitted=True,
            )
        )
        batches.append(batch)
    return outputs, batches


def test_streaming_group_matches_single_global_loss_gradients_rng_and_buffers() -> None:
    torch.manual_seed(17)
    graph = torch.randn(7, 5)
    baseline = _ToyResidual().train()
    streaming = copy.deepcopy(baseline).train()
    work_items = [(_toy_batch(offset), 0) for offset in (0, 1, 2, 3)]

    def graph_for_chunk(_chunk):
        # Locks RNG capture to the boundary before graph materialization.
        torch.rand(())
        return graph

    torch.manual_seed(991)
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=1e-3)
    baseline_optimizer.zero_grad(set_to_none=True)
    outputs, batches = _forward_group(baseline, graph_for_chunk, work_items)
    direct_loss = _global_loss_from_outputs(
        outputs,
        batches,
        torch=torch,
        direction_loss_weight=0.25,
        shrinkage=1e-4,
        reduction_dtype=torch.float32,
    )
    direct_loss.backward()
    baseline_rng = _capture_torch_rng_state(torch)
    baseline_buffers = {name: value.clone() for name, value in baseline.named_buffers()}
    baseline_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in baseline.named_parameters()
    }
    baseline_optimizer.step()
    assert baseline.encoder.peak_live_grad_forwards == len(work_items)
    assert baseline.encoder.live_grad_forwards == 0

    torch.manual_seed(991)
    streaming_optimizer = torch.optim.AdamW(streaming.parameters(), lr=1e-3)
    streaming_optimizer.zero_grad(set_to_none=True)
    plan = streaming_group_backward(
        streaming,
        work_items,
        graph_for_chunk,
        torch=torch,
        device="cpu",
        precision=resolve_mixed_precision_contract(
            torch, {"mixed_precision": "bf16"}, device="cpu"
        ),
        direction_loss_weight=0.25,
        shrinkage=1e-4,
    )
    assert plan.objective_value == pytest.approx(
        float(direct_loss.detach()), rel=1e-6, abs=1e-7
    )
    for name, parameter in streaming.named_parameters():
        assert torch.allclose(
            parameter.grad,
            baseline_gradients[name],
            rtol=2e-5,
            atol=2e-6,
        ), name
    assert _torch_rng_states_equal(torch, _capture_torch_rng_state(torch), baseline_rng)
    assert {
        name: value for name, value in streaming.named_buffers()
    }.keys() == baseline_buffers.keys()
    for name, value in streaming.named_buffers():
        assert torch.equal(value, baseline_buffers[name]), name
    assert streaming.encoder.peak_live_grad_forwards == 1
    assert streaming.encoder.live_grad_forwards == 0

    guard = optimizer_step_with_guards(
        streaming,
        streaming_optimizer,
        torch=torch,
        objective_value=plan.objective_value,
    )
    assert guard["grad_finite"] is True
    assert guard["parameters_finite"] is True
    assert guard["parameter_delta_positive"] is True
    assert guard["update_probe_max_abs_delta"] > 0
    for (name, direct), (_, replayed) in zip(
        baseline.named_parameters(), streaming.named_parameters(), strict=True
    ):
        assert torch.allclose(direct, replayed, rtol=2e-5, atol=2e-6), name


def test_optimizer_guard_rejects_nonfinite_unprobed_parameter() -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1, bias=False),
        torch.nn.Linear(1, 1, bias=False),
    )
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)

    class CorruptingOptimizer:
        @staticmethod
        def step() -> None:
            parameters = list(model.parameters())
            with torch.no_grad():
                parameters[0].fill_(float("inf"))
                parameters[1].add_(0.1)

    with pytest.raises(RuntimeError, match="NONFINITE_PARAMETER_AFTER_STEP"):
        optimizer_step_with_guards(
            model,
            CorruptingOptimizer(),
            torch=torch,
            objective_value=1.0,
        )


def test_fallback_split_preserves_one_optimizer_group() -> None:
    work_items = [(_toy_batch(offset), 0) for offset in (0, 1)]
    fragments = split_work_items_for_oom_fallback(
        work_items, microbatch_size=2, maximum_fragments=4
    )
    assert len(fragments) == 4
    assert sum(len(batch["base_logit"]) for batch, _ in fragments) == 8
    with pytest.raises(RuntimeError, match="optimizer-step semantics"):
        split_work_items_for_oom_fallback(
            work_items, microbatch_size=1, maximum_fragments=4
        )
