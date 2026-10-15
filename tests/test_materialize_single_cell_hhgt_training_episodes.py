from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "materialize_single_cell_hhgt_training_episodes_149.py"
)
SPEC = importlib.util.spec_from_file_location("hhgt_episode_materializer", SCRIPT)
MATERIALIZER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MATERIALIZER)


def test_atomic_torch_save_retries_wrapped_sshfs_permission_error(
    tmp_path, monkeypatch
) -> None:
    destination = tmp_path / "episode.pt"
    real_save = torch.save
    calls = 0

    def flaky_save(value, path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(
                "open file failed with strerror: Operation not permitted"
            )
        real_save(value, path)

    monkeypatch.setattr(MATERIALIZER.torch, "save", flaky_save)
    monkeypatch.setattr(MATERIALIZER.time, "sleep", lambda _seconds: None)

    MATERIALIZER.atomic_torch_save(destination, {"value": torch.tensor([1.0])})

    assert calls == 2
    assert torch.load(destination, weights_only=True)["value"].item() == 1.0


def test_atomic_torch_save_does_not_retry_unrelated_runtime_error(
    tmp_path, monkeypatch
) -> None:
    calls = 0

    def invalid_save(_value, _path):
        nonlocal calls
        calls += 1
        raise RuntimeError("serialization contract failed")

    monkeypatch.setattr(MATERIALIZER.torch, "save", invalid_save)

    try:
        MATERIALIZER.atomic_torch_save(tmp_path / "episode.pt", {"value": 1})
    except RuntimeError as error:
        assert str(error) == "serialization contract failed"
    else:
        raise AssertionError("unrelated RuntimeError was swallowed")
    assert calls == 1


def test_cached_episode_metadata_check_does_not_rehash(tmp_path, monkeypatch) -> None:
    payload = tmp_path / "episode.pt"
    payload.write_bytes(b"receipted-payload")
    record = {
        "path": str(payload.resolve()),
        "bytes": payload.stat().st_size,
        "sha256": "already-recorded",
    }
    monkeypatch.setattr(
        MATERIALIZER,
        "sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("unexpected rehash")),
    )

    MATERIALIZER.verify_record_metadata(record, payload, "CACHED_EPISODE")


def test_finalize_does_not_rehash_large_fold_graph_assets() -> None:
    source = inspect.getsource(MATERIALIZER.finalize)
    assert (
        '_read_success(\n                graph_fold, "FOLD_GRAPH_ASSETS_READY", verify_files=False\n            )'
        in source
    )


def test_finalize_loads_each_receipted_episode_once() -> None:
    source = inspect.getsource(MATERIALIZER.finalize)
    assert 'verify_record_metadata(record, path, "FINAL_EPISODE")' in source
    assert 'verify_record(record, path, "FINAL_EPISODE")' not in source
    assert source.count("torch.load(path, map_location=\"cpu\", weights_only=True)") == 1
    assert "fold_roles[fold].add(role)" in source
