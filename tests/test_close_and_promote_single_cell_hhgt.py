import hashlib
import json
from pathlib import Path

import pytest

from scripts.close_and_promote_single_cell_hhgt_149 import (
    ClosureError,
    EVENT_ID,
    publish,
)


def _write_json(path: Path, value: dict) -> bytes:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(payload)
    return payload


def test_publish_replaces_pointer_and_appends_one_jsonl_event(tmp_path):
    pointer = tmp_path / "CURRENT.json"
    old = _write_json(pointer, {"status": "OLD"})
    log = tmp_path / "VERSION_CHANGE_LOG.jsonl"
    log.write_text('{"event_id":"OLDER"}\n', encoding="utf-8")
    new_pointer = {"status": "PASS_ACTIVATED_SC_HHGT"}
    event = {"event_id": EVENT_ID, "status": "PASS_ACTIVATED"}

    pointer_sha, log_sha = publish(
        pointer, log, hashlib.sha256(old).hexdigest(), new_pointer, event,
    )

    assert json.loads(pointer.read_text()) == new_pointer
    assert [json.loads(line)["event_id"] for line in log.read_text().splitlines()] == [
        "OLDER", EVENT_ID,
    ]
    assert pointer_sha == hashlib.sha256(pointer.read_bytes()).hexdigest()
    assert log_sha == hashlib.sha256(log.read_bytes()).hexdigest()


def test_publish_rejects_stale_pointer_without_writing(tmp_path):
    pointer = tmp_path / "CURRENT.json"
    original = _write_json(pointer, {"status": "OLD"})
    log = tmp_path / "VERSION_CHANGE_LOG.jsonl"
    log.write_text('{"event_id":"OLDER"}\n', encoding="utf-8")
    original_log = log.read_bytes()

    with pytest.raises(ClosureError, match="FORMAL_POINTER_CHANGED"):
        publish(pointer, log, "0" * 64, {"status": "NEW"}, {"event_id": EVENT_ID})

    assert pointer.read_bytes() == original
    assert log.read_bytes() == original_log


def test_publish_rejects_duplicate_event_without_writing(tmp_path):
    pointer = tmp_path / "CURRENT.json"
    original = _write_json(pointer, {"status": "OLD"})
    log = tmp_path / "VERSION_CHANGE_LOG.jsonl"
    log.write_text(json.dumps({"event_id": EVENT_ID}) + "\n", encoding="utf-8")
    original_log = log.read_bytes()

    with pytest.raises(ClosureError, match="ACTIVATION_EVENT_ALREADY_EXISTS"):
        publish(
            pointer, log, hashlib.sha256(original).hexdigest(),
            {"status": "NEW"}, {"event_id": EVENT_ID},
        )

    assert pointer.read_bytes() == original
    assert log.read_bytes() == original_log
