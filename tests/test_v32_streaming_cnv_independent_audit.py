from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cc_hhgt.v32.streaming_cnv_independent_audit import (
    FOLD_FORMAT,
    PARTITION_FORMAT,
    STORE_FORMAT,
    SUPERVISOR_FORMAT,
    StreamingCNVIndependentAuditError,
    audit_streaming_cnv_store,
    canonical_sha256,
    sha256_file,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _partition(root: Path, cancer: str, context_sha: str) -> dict[str, object]:
    target = root / f"cancer={cancer}"
    target.mkdir()
    patient_ids = [f"{cancer}-P0", f"{cancer}-P1"]
    lncrna_ids = ["LNC:1", "LNC:2"]
    pathway_ids = ["PATH:1"]
    _json(target / "patient_ids.json", patient_ids)
    _json(target / "lncrna_ids.json", lncrna_ids)
    _json(target / "pathway_ids.json", pathway_ids)
    for prefix, columns in (("lncrna", 2), ("pathway", 1)):
        shape = (2, columns)
        np.save(target / f"{prefix}_event.npy", np.zeros(shape, dtype=np.uint8), allow_pickle=False)
        np.save(target / f"{prefix}_callable.npy", np.ones(shape, dtype=np.uint8), allow_pickle=False)
        np.save(target / f"{prefix}_burden.npy", np.full(shape, 0.1, dtype=np.float32), allow_pickle=False)
    folds = []
    for fold in range(5):
        fold_root = target / f"fold={fold}"
        fold_root.mkdir()
        indices = np.asarray([fold] if fold < 2 else [], dtype=np.int32)
        index_path = fold_root / "test_patient_rows.npy"
        np.save(index_path, indices, allow_pickle=False)
        marker = {
            "format": FOLD_FORMAT,
            "status": "SUCCESS",
            "cancer_id": cancer,
            "patient_fold": fold,
            "context_sha256": context_sha,
            "test_patient_rows_sha256": sha256_file(index_path),
            "test_patients": len(indices),
        }
        _json(fold_root / "SUCCESS.json", marker)
        folds.append(marker)
    files = {
        path.relative_to(target).as_posix(): sha256_file(path)
        for path in sorted(target.rglob("*"))
        if path.is_file() and path.name != "SUCCESS.json"
    }
    payload = {
        "format": PARTITION_FORMAT,
        "status": "SUCCESS",
        "cancer_id": cancer,
        "context_sha256": context_sha,
        "patients": 2,
        "processed_segment_patients": 2,
        "typed_unavailable_patients": 0,
        "lncrnas": 2,
        "pathways": 1,
        "membership_genes": 1,
        "matrix_disk_bytes_estimate": 36,
        "memory_gate_bytes": 536_870_912,
        "estimated_peak_transient_bytes": 100,
        "observed_peak_process_bytes": 200,
        "observed_process_memory_gate_enforced": True,
        "max_segment_rows_observed": 1,
        "max_segment_rows_gate": 2_000_000,
        "max_materialized_long_rows": 0,
        "folds": folds,
        "files": files,
    }
    _json(target / "SUCCESS.json", payload)
    return {
        "path": str(target.resolve()),
        "success_sha256": sha256_file(target / "SUCCESS.json"),
        "files_composite_sha256": canonical_sha256(files),
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "store"
    root.mkdir()
    candidate = tmp_path / "FORMAL_CANDIDATE_UNIVERSE.parquet"
    candidate.write_bytes(b"candidate-authority")
    gate = tmp_path / "DOWNLOAD_COMPLETE.json"
    gate.write_text("{}\n", encoding="utf-8")
    streaming_sha = "e" * 64
    supervisor_sha = "f" * 64
    context_without_self = {
        "format": STORE_FORMAT,
        "bindings": {
            "candidates": {"path": str(candidate), "sha256": sha256_file(candidate)}
        },
        "candidate_loading": "CANCER_FILTERED_NO_GLOBAL_OBJECT_TABLE",
        "download_gate": str(gate),
        "download_gate_sha256": sha256_file(gate),
        "implementation_sha256": {"segment_cnv_streaming.py": streaming_sha},
        "max_memory_bytes": 536_870_912,
        "observed_process_memory_gate_enforced": True,
    }
    context_sha = canonical_sha256(context_without_self)
    _json(root / "RUN_CONTEXT.json", {**context_without_self, "context_sha256": context_sha})
    cancers = ("ACC", "BLCA")
    records = {cancer: _partition(root, cancer, context_sha) for cancer in cancers}
    aggregate = {
        "format": STORE_FORMAT,
        "status": "SUCCESS",
        "context_sha256": context_sha,
        "run_context_sha256": sha256_file(root / "RUN_CONTEXT.json"),
        "cancers": records,
        "patient_folds": 5,
        "required_cancers": list(cancers),
        "typed_unavailable_is_never_zero": True,
    }
    _json(root / "SUCCESS.json", aggregate)
    supervisor = tmp_path / "supervisor"
    supervisor.mkdir()
    (supervisor / "CHILD.log").write_bytes(b"")
    (supervisor / "MEMORY_SAMPLES.tsv").write_text(
        "elapsed_seconds\trss_bytes\thwm_bytes\n0.0\t100\t200\n1.0\t0\t200\n",
        encoding="utf-8",
    )
    supervisor_marker = {
        "format": SUPERVISOR_FORMAT,
        "status": "SUCCESS",
        "supervisor_sha256": supervisor_sha,
        "max_memory_bytes": 536_870_912,
        "poll_seconds": 1.0,
        "command": ["python", "runner.py", "--output-root", str(root.resolve())],
        "child_pid": 123,
        "child_return_code": 0,
        "reason": "CHILD_EXITED",
        "termination": None,
        "max_rss_bytes": 100,
        "max_hwm_bytes": 200,
        "memory_samples": 2,
        "child_log_sha256": sha256_file(supervisor / "CHILD.log"),
        "memory_samples_sha256": sha256_file(supervisor / "MEMORY_SAMPLES.tsv"),
    }
    _json(supervisor / "SUCCESS.json", supervisor_marker)
    return {
        "root": root,
        "supervisor": supervisor,
        "cancers": cancers,
        "context_sha": context_sha,
        "candidate_sha": sha256_file(candidate),
        "streaming_sha": streaming_sha,
        "supervisor_sha": supervisor_sha,
    }


def _audit(paths: dict[str, object]) -> dict[str, object]:
    root = paths["root"]
    supervisor = paths["supervisor"]
    assert isinstance(root, Path) and isinstance(supervisor, Path)
    return audit_streaming_cnv_store(
        aggregate_success_path=root / "SUCCESS.json",
        supervisor_success_path=supervisor / "SUCCESS.json",
        expected_store_root=root,
        expected_context_sha256=str(paths["context_sha"]),
        expected_candidate_sha256=str(paths["candidate_sha"]),
        expected_streaming_code_sha256=str(paths["streaming_sha"]),
        expected_supervisor_sha256=str(paths["supervisor_sha"]),
        expected_total_patients=4,
        expected_total_processed=4,
        expected_total_typed_unavailable=0,
        expected_cancers=paths["cancers"],
    )


def test_independent_audit_rehashes_arrays_folds_context_and_supervisor(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    report = _audit(paths)
    assert report["status"] == "PASS"
    assert report["cancer_count"] == 2
    assert report["patient_totals"] == {
        "patients": 4,
        "processed_segment_patients": 4,
        "typed_unavailable_patients": 0,
    }
    assert report["all_declared_files_rehashed"] is True
    assert report["all_fold_success_receipts_semantically_verified"] is True
    assert report["partition_inventory_exact_including_fold_receipts"] is True
    assert report["all_compact_arrays_semantically_scanned"] is True
    assert report["port_8260_assessed_by_this_audit"] is False


def test_independent_audit_rejects_aggregate_partition_path_swap(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    aggregate = json.loads((root / "SUCCESS.json").read_text(encoding="utf-8"))
    aggregate["cancers"]["BLCA"]["path"] = aggregate["cancers"]["ACC"]["path"]
    _json(root / "SUCCESS.json", aggregate)
    with pytest.raises(StreamingCNVIndependentAuditError, match="partition path drift"):
        _audit(paths)


def test_independent_audit_rejects_nonnull_burden_for_uncallable_value(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    partition = root / "cancer=ACC"
    callable_path = partition / "lncrna_callable.npy"
    callable_values = np.load(callable_path)
    callable_values[0, 0] = 0
    np.save(callable_path, callable_values, allow_pickle=False)
    payload_path = partition / "SUCCESS.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["files"]["lncrna_callable.npy"] = sha256_file(callable_path)
    _json(payload_path, payload)
    aggregate_path = root / "SUCCESS.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["cancers"]["ACC"]["success_sha256"] = sha256_file(payload_path)
    aggregate["cancers"]["ACC"]["files_composite_sha256"] = canonical_sha256(payload["files"])
    _json(aggregate_path, aggregate)
    with pytest.raises(StreamingCNVIndependentAuditError, match="typed-null semantics"):
        _audit(paths)


def test_independent_audit_rejects_missing_fold_success_receipt(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    (root / "cancer=ACC" / "fold=0" / "SUCCESS.json").unlink()
    with pytest.raises(StreamingCNVIndependentAuditError, match="file inventory drift"):
        _audit(paths)


def test_independent_audit_rejects_unexpected_partition_file(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    (root / "cancer=ACC" / "undeclared.tmp").write_bytes(b"unexpected")
    with pytest.raises(StreamingCNVIndependentAuditError, match="file inventory drift"):
        _audit(paths)


def test_independent_audit_rejects_fold_success_in_declared_file_map(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    partition = root / "cancer=ACC"
    payload_path = partition / "SUCCESS.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    relative = "fold=0/SUCCESS.json"
    payload["files"][relative] = sha256_file(partition / relative)
    _json(payload_path, payload)
    aggregate_path = root / "SUCCESS.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    aggregate["cancers"]["ACC"]["success_sha256"] = sha256_file(payload_path)
    aggregate["cancers"]["ACC"]["files_composite_sha256"] = canonical_sha256(payload["files"])
    _json(aggregate_path, aggregate)
    with pytest.raises(StreamingCNVIndependentAuditError, match="file contract drift"):
        _audit(paths)


def test_independent_audit_rejects_fold_success_semantic_drift(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    root = paths["root"]
    assert isinstance(root, Path)
    marker_path = root / "cancer=ACC" / "fold=0" / "SUCCESS.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["test_patients"] = 999
    _json(marker_path, marker)
    with pytest.raises(StreamingCNVIndependentAuditError, match="Fold SUCCESS drift"):
        _audit(paths)
