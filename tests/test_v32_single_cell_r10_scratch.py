from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from cc_hhgt.v32 import single_cell_association_scratch as scratch


ROOT = Path(__file__).resolve().parents[1]
R10_RUNNER = ROOT / "scripts" / "run_v32_single_cell_r10_streaming.py"
R9_RUNNER = ROOT / "scripts" / "run_v32_single_cell_r7_streaming.py"


def load_r10():
    spec = importlib.util.spec_from_file_location("test_r10_runner", R10_RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fake_ext4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    mount = tmp_path.resolve()
    monkeypatch.setattr(
        scratch, "_filesystem_type_and_mount", lambda _path: ("ext4", mount)
    )
    monkeypatch.setattr(scratch, "_disk_free_bytes", lambda _path: 256 * 1024**3)
    owner = os.lstat(tmp_path)
    monkeypatch.setattr(
        scratch, "_effective_ids", lambda: (int(owner.st_uid), int(owner.st_gid))
    )
    monkeypatch.setattr(scratch, "_observed_permission_mode", lambda _stat: 0o700)
    return mount


def test_exact_private_ext4_root_is_created_pinned_revalidated_and_removed(
    tmp_path: Path, fake_ext4: Path
) -> None:
    parent = tmp_path / "local_ext4_parent"
    parent.mkdir()
    root = parent / "association_ACC_run_1"
    contract = scratch.create_association_scratch_contract(
        root,
        estimate_bytes=256 * 1024**2,
        forbidden_output_roots=(tmp_path / "durable_output",),
    )
    observed = os.lstat(root)
    assert contract["mode"] == 0o700
    assert contract["st_dev"] == observed.st_dev
    assert contract["st_ino"] == observed.st_ino
    assert contract["uid"] == observed.st_uid
    assert contract["gid"] == observed.st_gid
    assert contract["filesystem_type"] == "ext4"
    assert contract["required_free_bytes"] == 8 * 1024**3
    assert scratch.validate_association_scratch_contract(
        contract, phase="POST_EXEC"
    ) == root.resolve()
    assert scratch.association_work_path(contract, phase="POST_EXEC") == (
        root.resolve() / "duckdb_work"
    )
    scratch.remove_empty_association_scratch_root(contract, phase="SUCCESS_CLEANUP")
    assert not root.exists()


def test_space_gate_is_maximum_of_8_gib_and_16_times_estimate() -> None:
    assert scratch.required_free_bytes(1) == 8 * 1024**3
    assert scratch.required_free_bytes(1024**3) == 16 * 1024**3


def test_preexisting_root_and_output_overlap_fail_closed(
    tmp_path: Path, fake_ext4: Path
) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(
        scratch.AssociationScratchContractError,
        match="ASSOCIATION_SCRATCH_REUSE_FORBIDDEN",
    ):
        scratch.preflight_association_scratch_root(existing, estimate_bytes=1)

    output = tmp_path / "durable"
    output.mkdir()
    with pytest.raises(
        scratch.AssociationScratchContractError,
        match="ASSOCIATION_SCRATCH_OUTPUT_OVERLAP",
    ):
        scratch.preflight_association_scratch_root(
            output / "scratch",
            estimate_bytes=1,
            forbidden_output_roots=(output,),
        )


def test_shared_server_tmp_is_never_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shared_tmp = tmp_path / "server_tmp"
    shared_tmp.mkdir()
    monkeypatch.setattr(scratch, "FORBIDDEN_TEMP_ROOTS", (shared_tmp,))
    with pytest.raises(
        scratch.AssociationScratchContractError,
        match="ASSOCIATION_SCRATCH_SHARED_TMP_FORBIDDEN",
    ) as captured:
        scratch.preflight_association_scratch_root(
            shared_tmp / "CancerLncAtlas-r10", estimate_bytes=1
        )
    assert captured.value.as_payload()["server_tmp_fallback_used"] is False
    assert tuple(
        str(value).replace("\\", "/")
        for value in (Path("/tmp"), Path("/var/tmp"), Path("/dev/shm"))
    ) == ("/tmp", "/var/tmp", "/dev/shm")


def test_symlink_component_and_inode_replacement_are_rejected(
    tmp_path: Path, fake_ext4: Path
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        os.symlink(real, link, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(
        scratch.AssociationScratchContractError,
        match="ASSOCIATION_SCRATCH_SYMLINK_FORBIDDEN",
    ):
        scratch.preflight_association_scratch_root(
            link / "association", estimate_bytes=1
        )

    root = real / "association"
    contract = scratch.create_association_scratch_contract(root, estimate_bytes=1)
    os.rmdir(root)
    root.mkdir(mode=0o700)
    os.chmod(root, 0o700)
    with pytest.raises(
        scratch.AssociationScratchContractError,
        match="ASSOCIATION_SCRATCH_IDENTITY_DRIFT",
    ):
        scratch.validate_association_scratch_contract(contract, phase="POST_EXEC")


def test_insufficient_space_has_a_typed_reason(
    tmp_path: Path, fake_ext4: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(scratch, "_disk_free_bytes", lambda _path: 7 * 1024**3)
    with pytest.raises(
        scratch.AssociationScratchContractError,
        match="ASSOCIATION_SCRATCH_SPACE_INSUFFICIENT",
    ) as captured:
        scratch.preflight_association_scratch_root(
            tmp_path / "too_small", estimate_bytes=1
        )
    payload = captured.value.as_payload()
    assert payload["status"] == "TYPED_FAILURE"
    assert payload["reason_code"] == "ASSOCIATION_SCRATCH_SPACE_INSUFFICIENT"


def test_formal_run_requires_explicit_scratch_and_does_not_extend_r9_outputs() -> None:
    r10 = load_r10()
    r9 = r10._load_r9()
    args = r10.build_parser(r9).parse_args(
        [
            "--mode",
            "run",
            "--r7-root",
            "authority",
            "--server-preflight-json",
            "preflight.json",
            "--expected-server-preflight-sha256",
            "a" * 64,
            "--expected-run-status-sha256",
            "b" * 64,
            "--cancer-id",
            "ACC",
            "--output-parent",
            "output",
        ]
    )
    with pytest.raises(
        scratch.AssociationScratchContractError,
        match="ASSOCIATION_SCRATCH_ARGUMENT_REQUIRED",
    ):
        r10._validate_public_args(args, r9)
    assert all("/tmp" not in value for value in r9.AUTHORIZED_OUTPUT_PREFIXES)


def _write_raw_fixture(path: Path) -> None:
    frame = pd.DataFrame(
        {
            "dataset_id": ["fixture"] * 6,
            "cancer_id": ["ACC"] * 6,
            "compartment_order": [0, 0, 1, 1, 2, 2],
            "compartment": ["malignant", "malignant", "immune", "immune", "stromal", "stromal"],
            "lncrna_id": ["L1", "L2", "L1", "L2", "L1", "L2"],
            "lncrna_symbol": ["LS1", "LS2", "LS1", "LS2", "LS1", "LS2"],
            "pathway_id": ["P1", "P2", "P1", "P2", "P1", "P2"],
            "n_donors": [8] * 6,
            "spearman_rho": [0.7, -0.8, 0.6, -0.7, 0.9, -0.6],
            "nominal_p": [0.02, 0.01, 0.04, 0.03, 0.005, 0.045],
            "total_tests": [10, 10, 12, 12, 8, 8],
        }
    )
    frame.to_parquet(path, index=False)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r9_r10_small_fixture_bh_outputs_are_byte_equivalent(
    tmp_path: Path, fake_ext4: Path
) -> None:
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    r10 = load_r10()
    r9 = r10._load_r9()

    raw_r9 = tmp_path / "raw_r9.parquet"
    raw_r10 = tmp_path / "raw_r10.parquet"
    _write_raw_fixture(raw_r9)
    raw_r10.write_bytes(raw_r9.read_bytes())
    output_r9 = tmp_path / "evidence_r9.parquet"
    output_r10 = tmp_path / "evidence_r10.parquet"
    rows_r9 = r9._finalize_association_evidence(
        raw_path=raw_r9,
        output_path=output_r9,
        scratch=tmp_path / "r9_duckdb_work",
        raw_rows=6,
    )

    r10_root = tmp_path / "r10_ext4_exact"
    contract = scratch.create_association_scratch_contract(
        r10_root, estimate_bytes=1
    )
    rows_r10 = r9._finalize_association_evidence_in_process(
        raw_path=raw_r10,
        output_path=output_r10,
        scratch=scratch.association_work_path(contract, phase="POST_EXEC"),
        raw_rows=6,
    )
    scratch.validate_association_scratch_contract(
        contract, phase="POST_BH", require_empty=True
    )
    scratch.remove_empty_association_scratch_root(contract, phase="SUCCESS_CLEANUP")
    assert rows_r9 == rows_r10 == 6
    pd.testing.assert_frame_equal(
        pd.read_parquet(output_r9), pd.read_parquet(output_r10)
    )
    assert _file_sha(output_r9) == _file_sha(output_r10)


def test_interruption_before_bh_completion_never_publishes_final(
    tmp_path: Path, fake_ext4: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r10 = load_r10()
    r9 = r10._load_r9()
    monkeypatch.setattr(r9, "_authorized_output", lambda _path: None)
    monkeypatch.setattr(r10, "_require_dell2_durable_output", lambda _path: None)
    contract_sha = "a" * 64
    producer_pid = os.getpid()
    publish = tmp_path / f".cancer_id=ACC.{contract_sha[:16]}.publish.{producer_pid}"
    final = tmp_path / "cancer_id=ACC"
    publish.mkdir()
    for relative in r9.POST_BH_RELATIVE_PATHS:
        if relative != "association_evidence.parquet":
            (publish / relative).write_bytes(b"fixture")
    raw = publish / ".association_evidence_raw.parquet"
    raw.write_bytes(b"fixture")
    scratch_contract = scratch.create_association_scratch_contract(
        tmp_path / "interrupt_ext4_exact", estimate_bytes=1
    )
    handoff = {
        "format": r10.R10_POST_BH_HANDOFF_FORMAT,
        "association_engine": r9.ASSOCIATION_ENGINE,
        "cancer_id": "ACC",
        "producer_pid": producer_pid,
        "publish": str(publish),
        "final": str(final),
        "raw_path": str(raw),
        "evidence_path": str(publish / "association_evidence.parquet"),
        "scratch": str(Path(scratch_contract["root"]) / "duckdb_work"),
        "raw_rows": 1,
        "relative_paths": list(r9.POST_BH_RELATIVE_PATHS),
        "lineage_base": {"cancer_id": "ACC", "contract_sha256": contract_sha},
        "success_base": {"cancer_id": "ACC"},
        "association_scratch_contract": scratch_contract,
        "association_scratch_contract_sha256": scratch.canonical_contract_sha256(
            scratch_contract
        ),
        "_verified_handoff_sha256": "b" * 64,
    }

    def interrupt(**_kwargs):
        raise KeyboardInterrupt("fixture interruption")

    monkeypatch.setattr(r9, "_finalize_association_evidence_in_process", interrupt)
    with pytest.raises(KeyboardInterrupt, match="fixture interruption"):
        r10._post_bh_exec_publish_r10(handoff, r9)
    assert not final.exists()
    assert not (publish / "SUCCESS.json").exists()
    assert Path(scratch_contract["root"]).is_dir()
    scratch.remove_empty_association_scratch_root(
        scratch_contract, phase="NEGATIVE_GATE_CLEANUP"
    )


def test_exec_handoff_is_rewritten_and_scratch_identity_crosses_exec(
    tmp_path: Path, fake_ext4: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r10 = load_r10()
    r9 = r10._load_r9()
    scratch_contract = scratch.create_association_scratch_contract(
        tmp_path / "exec_ext4_exact", estimate_bytes=1
    )
    handoff_path = tmp_path / "POST_BH_HANDOFF.123.json"
    original = {
        "format": r9.POST_BH_HANDOFF_FORMAT,
        "association_engine": r9.ASSOCIATION_ENGINE,
        "cancer_id": "ACC",
        "producer_pid": os.getpid(),
        "scratch": str(tmp_path / "old_scratch"),
        "lineage_base": {"cancer_id": "ACC", "contract_sha256": "a" * 64},
        "success_base": {"cancer_id": "ACC"},
    }
    handoff_path.write_text(json.dumps(original), encoding="utf-8")
    original_sha = _file_sha(handoff_path)
    captured = {}

    class ExecBoundary(RuntimeError):
        pass

    def fake_exec(executable, command, environment):
        captured.update(
            executable=executable, command=command, environment=environment
        )
        raise ExecBoundary("expected fixture exec boundary")

    monkeypatch.setattr(r10, "_REAL_EXECVE", fake_exec)
    with pytest.raises(ExecBoundary, match="expected fixture exec boundary"):
        r10._rewrite_handoff_and_exec(
            r9=r9,
            scratch_contract=scratch_contract,
            executable="python",
            command=[
                "python",
                str(R9_RUNNER),
                "--internal-post-bh-r9",
                "--handoff-json",
                str(handoff_path),
                "--expected-handoff-sha256",
                original_sha,
            ],
            environment={"FIXTURE": "1"},
        )
    promoted = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert captured["command"][2] == "--internal-post-bh-r10"
    assert captured["command"][6] == _file_sha(handoff_path)
    assert promoted["format"] == r10.R10_POST_BH_HANDOFF_FORMAT
    assert promoted["scratch"] == str(
        Path(scratch_contract["root"]) / "duckdb_work"
    )
    assert promoted["association_scratch_contract_sha256"] == (
        scratch.canonical_contract_sha256(scratch_contract)
    )
    assert promoted["lineage_base"]["association_scratch_st_ino"] == (
        scratch_contract["st_ino"]
    )
    scratch.remove_empty_association_scratch_root(
        scratch_contract, phase="EXEC_FIXTURE_CLEANUP"
    )


def test_frozen_r9_runner_bytes_were_not_modified() -> None:
    assert _file_sha(R9_RUNNER) == (
        "903658231bfe31387d12cab56b9de45ec88258e386f9be0e3ba442b61495932c"
    )
