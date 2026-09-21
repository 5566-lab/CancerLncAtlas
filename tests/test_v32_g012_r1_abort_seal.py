from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts import seal_v32_g012_r1_aborted_no_gpu as seal


NOW = datetime(2026, 9, 1, 6, 5, tzinfo=timezone.utc)


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    stopped: bool = True,
    checked_at: datetime | None = None,
) -> tuple[Path, Path, Path]:
    root = tmp_path / "DSC" / "CancerLncAtlas"
    paths = seal.resolve_paths(root)
    paths.bootstrap_root.mkdir(parents=True)
    # A sentinel proves the sealer performs no result-tree operation.
    paths.result_root.mkdir(parents=True)
    formal_sentinel = paths.result_root / "DO_NOT_TOUCH_FORMAL_RESULT.txt"
    formal_sentinel.write_text("immutable-r1-formal-result\n", encoding="utf-8")

    patch = {
        "status": "PATCH_READY",
        "gpu_visible": False,
        "baseline_retained": True,
        "code_root": "./data/CancerLncAtlas/runtime/tools/"
        "v32_g012_paid_gpu_20260831_r1/code",
        "code_tree_sha256": seal.R1_CODE_TREE_SHA256,
        "overlay_sha256": seal.R1_OVERLAY_SHA256,
    }
    paths.live_patch_ready.write_text(
        json.dumps(patch, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    patch_sha = hashlib.sha256(paths.live_patch_ready.read_bytes()).hexdigest()
    monkeypatch.setattr(seal, "R1_PATCH_READY_SHA256", patch_sha)

    variants = {
        variant: {
            "folds": list(range(5)),
            "tasks": 5,
            "fold_inputs": [{"fold": fold} for fold in range(5)],
        }
        for variant in ("G0", "G1", "G2")
    }
    static = {
        "status": "STATIC_AUTH_READY",
        "gpu_visible": False,
        "formal_seed": 20260726,
        "patch_ready_sha256": patch_sha,
        "code_tree_sha256": seal.R1_CODE_TREE_SHA256,
        "variants": variants,
    }
    paths.live_static_auth_ready.write_text(
        json.dumps(static, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    static_sha = hashlib.sha256(paths.live_static_auth_ready.read_bytes()).hexdigest()
    monkeypatch.setattr(seal, "R1_STATIC_AUTH_R6_SHA256", static_sha)

    stopped_receipt = root / "runtime" / "bootstrap" / "STOPPED_RECEIPT.json"
    timestamp = checked_at or (NOW - timedelta(minutes=1))
    stopped_receipt.write_text(
        json.dumps(
            {
                "format": seal.STOPPED_RECEIPT_FORMAT,
                "status": "INSTANCE_STOPPED_STATE_CONFIRMED",
                "instance_id": seal.INSTANCE_ID,
                "observed_state": "Stopped" if stopped else "Running",
                "provider_domain_state": "DOMAIN_SHUT_OFF",
                "gpu_billing_active": not stopped,
                "checked_at": timestamp.isoformat(),
                "source": "DIRECT_COMPSHARE_INSTANCE_SHOW_QUERY",
                "raw_provider_response_embedded": False,
                "credentials_embedded": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return root, stopped_receipt, formal_sentinel


def test_default_plan_is_read_only_and_does_not_inspect_formal_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, formal_sentinel = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    before = formal_sentinel.read_bytes()
    plan = seal.build_dry_run_plan(
        project_root=root, stopped_receipt_path=stopped, now=NOW
    )
    assert plan["status"] == "DRY_RUN_R1_ABORT_SEAL_READY"
    assert plan["would_modify_filesystem"] is False
    assert plan["apply_required_for_mutation"] is True
    assert plan["formal_result_root_examined"] is False
    assert plan["formal_result_root_modified"] is False
    assert plan["formal_result_artifacts_reused"] is False
    assert paths.live_patch_ready.is_file()
    assert paths.live_static_auth_ready.is_file()
    assert not paths.archive_root.exists()
    assert not paths.aborted_receipt.exists()
    assert formal_sentinel.read_bytes() == before


def test_apply_requires_independent_stopped_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, running_receipt, _ = _fixture(tmp_path, monkeypatch, stopped=False)
    with pytest.raises(seal.AbortSealError, match="STOPPED_RECEIPT_CONTRACT_DRIFT"):
        seal.apply_seal(
            project_root=root, stopped_receipt_path=running_receipt, now=NOW
        )
    paths = seal.resolve_paths(root)
    assert paths.live_patch_ready.is_file()
    assert paths.live_static_auth_ready.is_file()
    assert not paths.aborted_receipt.exists()


def test_stale_stopped_receipt_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, receipt, _ = _fixture(
        tmp_path, monkeypatch, checked_at=NOW - timedelta(hours=1)
    )
    with pytest.raises(seal.AbortSealError, match="STOPPED_RECEIPT_STALE_SECONDS"):
        seal.apply_seal(project_root=root, stopped_receipt_path=receipt, now=NOW)


def test_apply_archives_only_live_auth_and_writes_fail_closed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, formal_sentinel = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    formal_before = formal_sentinel.read_bytes()
    patch_before = paths.live_patch_ready.read_bytes()
    static_before = paths.live_static_auth_ready.read_bytes()
    receipt = seal.apply_seal(
        project_root=root, stopped_receipt_path=stopped, now=NOW
    )
    assert receipt["status"] == seal.ABORTED_STATUS
    assert receipt["resume_authorized"] is False
    assert receipt["instance_id"] == seal.INSTANCE_ID
    assert receipt["live_authorizations_removed"] is True
    assert receipt["formal_result_root_examined"] is False
    assert receipt["formal_result_root_modified"] is False
    assert receipt["formal_result_artifacts_reused"] is False
    assert receipt["deletion_performed"] is False
    assert not paths.live_patch_ready.exists()
    assert not paths.live_static_auth_ready.exists()
    assert paths.archived_patch_ready.read_bytes() == patch_before
    assert paths.archived_static_auth_ready.read_bytes() == static_before
    assert json.loads(paths.aborted_receipt.read_text(encoding="utf-8")) == receipt
    assert formal_sentinel.read_bytes() == formal_before


def test_existing_gpu_training_complete_refuses_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    paths.gpu_training_complete.write_text(
        '{"status":"GPU_TRAINING_COMPLETE"}\n', encoding="utf-8"
    )
    with pytest.raises(seal.AbortSealError, match="GPU_TRAINING_COMPLETE_PRESENT"):
        seal.apply_seal(project_root=root, stopped_receipt_path=stopped, now=NOW)
    assert paths.live_patch_ready.is_file()
    assert paths.live_static_auth_ready.is_file()


def test_apply_is_idempotent_only_after_complete_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    first = seal.apply_seal(
        project_root=root, stopped_receipt_path=stopped, now=NOW
    )
    second = seal.apply_seal(
        project_root=root, stopped_receipt_path=stopped, now=NOW
    )
    assert second == first


def test_existing_seal_rejects_broken_live_authorization_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    seal.apply_seal(project_root=root, stopped_receipt_path=stopped, now=NOW)
    try:
        paths.live_patch_ready.symlink_to(paths.bootstrap_root / "missing-target.json")
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(seal.AbortSealError, match="LIVE_AUTH_PRESENT"):
        seal.apply_seal(project_root=root, stopped_receipt_path=stopped, now=NOW)


def test_half_transaction_is_rolled_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    real_replace = seal.os.replace
    failed = False

    def fail_second_move(source: object, target: object) -> None:
        nonlocal failed
        if Path(source) == paths.live_static_auth_ready and not failed:
            failed = True
            raise OSError("injected second-move failure")
        real_replace(source, target)

    monkeypatch.setattr(seal.os, "replace", fail_second_move)
    with pytest.raises(seal.AbortSealError, match="TRANSACTION_FAILED"):
        seal.apply_seal(project_root=root, stopped_receipt_path=stopped, now=NOW)
    assert paths.live_patch_ready.is_file()
    assert paths.live_static_auth_ready.is_file()
    assert not paths.partial_receipt.exists()
    assert not paths.archive_root.exists()
    assert not paths.aborted_receipt.exists()


def test_apply_recovers_journal_written_before_archive_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    pending = {
        "format": seal.ABORTED_RECEIPT_FORMAT,
        "status": "ABORT_SEAL_APPLY_IN_PROGRESS_FAIL_CLOSED",
        "resume_authorized": False,
        "instance_id": seal.INSTANCE_ID,
        "stopped_receipt_sha256": "0" * 64,
        "r1_patch_ready_sha256": seal.R1_PATCH_READY_SHA256,
        "r1_static_auth_r6_sha256": seal.R1_STATIC_AUTH_R6_SHA256,
    }
    paths.partial_receipt.write_text(
        json.dumps(pending, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    receipt = seal.apply_seal(
        project_root=root, stopped_receipt_path=stopped, now=NOW
    )
    assert receipt["status"] == seal.ABORTED_STATUS
    assert paths.aborted_receipt.is_file()
    assert not paths.partial_receipt.exists()


@pytest.mark.parametrize("interrupted_journal", [b"", b'{"format":'])
def test_apply_recovers_interrupted_private_journal_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupted_journal: bytes,
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    paths.partial_receipt.write_bytes(interrupted_journal)

    receipt = seal.apply_seal(
        project_root=root, stopped_receipt_path=stopped, now=NOW
    )

    assert receipt["status"] == seal.ABORTED_STATUS
    assert paths.aborted_receipt.is_file()
    assert not paths.partial_receipt.exists()


def test_apply_recovers_crash_between_exclusive_link_and_temp_unlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    real_unlink = seal.os.unlink
    injected = False

    def fail_once(path: object, *args: object, **kwargs: object) -> None:
        nonlocal injected
        if Path(path) == paths.temporary_final_receipt and not injected:
            injected = True
            raise OSError("injected crash-window unlink failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(seal.os, "unlink", fail_once)
    with pytest.raises(seal.AbortSealError, match="COMMIT_CLEANUP_FAILED"):
        seal.apply_seal(project_root=root, stopped_receipt_path=stopped, now=NOW)
    assert paths.aborted_receipt.is_file()
    assert paths.temporary_final_receipt.is_file()
    assert os.lstat(paths.aborted_receipt).st_ino == os.lstat(
        paths.temporary_final_receipt
    ).st_ino
    assert os.lstat(paths.aborted_receipt).st_nlink == 2

    recovered = seal.apply_seal(
        project_root=root, stopped_receipt_path=stopped, now=NOW
    )
    assert recovered["status"] == seal.ABORTED_STATUS
    assert not paths.temporary_final_receipt.exists()
    assert not paths.partial_receipt.exists()
    assert os.lstat(paths.aborted_receipt).st_nlink == 1


def test_apply_removes_legacy_empty_archive_orphan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    paths.archive_root.mkdir()
    receipt = seal.apply_seal(
        project_root=root, stopped_receipt_path=stopped, now=NOW
    )
    assert receipt["status"] == seal.ABORTED_STATUS


def test_aborted_receipt_commit_never_overwrites_a_racing_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    real_link = seal.os.link
    injected = False

    def inject_target_before_commit(
        source: object, target: object, *, follow_symlinks: bool = True
    ) -> None:
        nonlocal injected
        if Path(target) == paths.aborted_receipt and not injected:
            injected = True
            paths.aborted_receipt.write_bytes(b"racing-target-must-survive\n")
        real_link(source, target, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(seal.os, "link", inject_target_before_commit)
    with pytest.raises(seal.AbortSealError, match="EXCLUSIVE_COMMIT_FAILED"):
        seal.apply_seal(project_root=root, stopped_receipt_path=stopped, now=NOW)
    assert paths.aborted_receipt.read_bytes() == b"racing-target-must-survive\n"
    assert paths.live_patch_ready.is_file()
    assert paths.live_static_auth_ready.is_file()


def test_live_authorization_hardlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    original = paths.bootstrap_root / "PATCH_READY.original.json"
    os.replace(paths.live_patch_ready, original)
    try:
        os.link(original, paths.live_patch_ready)
    except OSError:
        pytest.skip("hardlink creation unavailable")
    with pytest.raises(seal.AbortSealError, match="HARDLINK_FORBIDDEN"):
        seal.apply_seal(project_root=root, stopped_receipt_path=stopped, now=NOW)


def test_stopped_receipt_symlink_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    link = stopped.with_name("STOPPED_RECEIPT.link.json")
    try:
        link.symlink_to(stopped)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(seal.AbortSealError, match="SYMLINK_FORBIDDEN"):
        seal.apply_seal(project_root=root, stopped_receipt_path=link, now=NOW)


def test_bootstrap_symlinked_parent_component_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, stopped, _ = _fixture(tmp_path, monkeypatch)
    paths = seal.resolve_paths(root)
    real_bootstrap = paths.bootstrap_root.with_name(paths.bootstrap_root.name + ".real")
    os.replace(paths.bootstrap_root, real_bootstrap)
    try:
        paths.bootstrap_root.symlink_to(real_bootstrap, target_is_directory=True)
    except OSError:
        os.replace(real_bootstrap, paths.bootstrap_root)
        pytest.skip("directory symlink creation unavailable")
    with pytest.raises(seal.AbortSealError, match="SYMLINK_COMPONENT"):
        seal.apply_seal(project_root=root, stopped_receipt_path=stopped, now=NOW)


def test_production_cli_has_no_root_or_instance_override() -> None:
    source = Path(seal.__file__).read_text(encoding="utf-8")
    assert 'parser.add_argument("--project-root"' not in source
    assert 'parser.add_argument("--expected-instance-id"' not in source
    assert 'INSTANCE_ID = "uhost-1up504geeqzj"' in source
