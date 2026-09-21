from __future__ import annotations

import csv
import errno
import hashlib
import json
import os
import time
from pathlib import Path

import pytest

from cc_hhgt.v32 import gdc_segment_cnv as cnv


def _hit(file_id: str, case_id: str, sample_type: str, cancer: str = "BRCA") -> dict:
    return {
        "file_id": file_id,
        "file_name": f"{file_id}.nocnv_grch38.seg.v2.txt",
        "md5sum": "a" * 32,
        "file_size": 123,
        "data_type": "Masked Copy Number Segment",
        "analysis": {"workflow_type": "DNAcopy"},
        "cases": [
            {
                "case_id": case_id,
                "submitter_id": f"TCGA-AA-{case_id}",
                "project": {"project_id": f"TCGA-{cancer}"},
                "samples": [
                    {
                        "sample_id": f"sample-{file_id}",
                        "submitter_id": f"TCGA-AA-{case_id}-01A",
                        "sample_type": sample_type,
                    }
                ],
            }
        ],
    }


def test_gdc_filter_is_exact_masked_open_dnacopy() -> None:
    payload = cnv.gdc_filter(["BRCA", "COAD"])
    contents = [item["content"] for item in payload["content"]]
    assert {"field": "files.data_type", "value": ["Masked Copy Number Segment"]} in contents
    assert {"field": "files.analysis.workflow_type", "value": ["DNAcopy"]} in contents
    assert {"field": "files.access", "value": ["open"]} in contents
    assert contents[0]["value"] == ["TCGA-BRCA", "TCGA-COAD"]


def test_flatten_excludes_normal_and_selects_one_file_per_case() -> None:
    hits = [
        _hit("f1", "0001", "Primary Tumor"),
        _hit("f2", "0001", "Primary Tumor"),
        _hit("normal", "0002", "Blood Derived Normal"),
    ]
    rows, audit = cnv.flatten_tumour_manifest(hits)
    assert len(rows) == 2
    assert sum(bool(row["selected_for_patient"]) for row in rows) == 1
    assert audit["excluded_files_without_tumour_sample"] == 1


def test_write_manifest_binds_table_hash(tmp_path: Path) -> None:
    rows, _ = cnv.flatten_tumour_manifest([_hit("f1", "0001", "Primary Tumor")])
    payload = cnv.write_manifest(rows, tmp_path / "candidate_manifest", {"queried_cancers": ["BRCA"]})
    table = tmp_path / "candidate_manifest" / payload["table"]["path"]
    assert payload["candidate_only"] is True
    assert payload["selected_patient_files"] == 1
    assert payload["table"]["sha256"] == cnv.sha256_file(table)


def test_existing_files_are_reconciled_by_filename_and_md5_not_cancer(tmp_path: Path) -> None:
    content = b"segment"
    hit = _hit("f1", "0001", "Primary Tumor")
    hit["md5sum"] = hashlib.md5(content).hexdigest()  # noqa: S324
    rows, _ = cnv.flatten_tumour_manifest([hit, _hit("f2", "0002", "Primary Tumor")])
    root = tmp_path / "existing" / "BRCA"
    root.mkdir(parents=True)
    cached = root / hit["file_name"]
    cached.write_bytes(content)
    audit = cnv.reconcile_existing_files(rows, [tmp_path / "existing"])
    assert audit["existing_files_reusable"] == 1
    assert {row["file_id"]: row["existing_reusable"] for row in rows} == {"f1": True, "f2": False}


def test_existing_inventory_is_reconciled_by_uuid_and_md5() -> None:
    content = b"segment"
    hit = _hit("11111111-2222-3333-4444-555555555555", "0001", "Primary Tumor")
    hit["md5sum"] = hashlib.md5(content).hexdigest()  # noqa: S324
    rows, _ = cnv.flatten_tumour_manifest([hit, _hit("f2", "0002", "Primary Tumor")])
    audit = cnv.reconcile_existing_inventory(
        rows,
        [
            (
                hit["md5sum"],
                "/old/subset/BRCA/11111111-2222-3333-4444-555555555555/"
                + hit["file_name"],
            )
        ],
    )
    assert audit["existing_files_reusable"] == 1
    assert rows[0]["existing_reusable"] is True
    assert rows[0]["existing_path"].startswith("/old/subset/BRCA/")
    assert rows[1]["existing_reusable"] is False


class _Response:
    status = 200

    def __init__(self, content: bytes):
        self.content = content
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, size: int = -1) -> bytes:
        if self.offset >= len(self.content):
            return b""
        if size < 0:
            size = len(self.content) - self.offset
        chunk = self.content[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk


def test_downloader_md5_verifies_and_enforces_candidate_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    content = b"Chromosome\tStart\tEnd\tSegment_Mean\n1\t1\t2\t0.5\n"
    digest = hashlib.md5(content).hexdigest()  # noqa: S324
    manifest = tmp_path / "manifest.tsv"
    columns = [
        "cancer_id", "file_id", "file_name", "md5sum", "case_submitter_id",
        "sample_submitter_id", "selected_for_patient",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "cancer_id": "BRCA", "file_id": "uuid", "file_name": "x.seg.v2.txt",
                "md5sum": digest, "case_submitter_id": "TCGA-AA-0001",
                "sample_submitter_id": "TCGA-AA-0001-01A", "selected_for_patient": "True",
            }
        )
    monkeypatch.setattr(cnv.urllib.request, "urlopen", lambda *_args, **_kwargs: _Response(content))
    output = tmp_path / "candidate_segments"
    result = cnv.download_manifest(manifest, output, workers=1)
    assert result[0]["status"] == "DOWNLOADED_VERIFIED"
    assert cnv.md5_file(Path(result[0]["path"])) == digest
    with pytest.raises(cnv.SegmentCNVError, match="candidate-only"):
        cnv.download_manifest(manifest, tmp_path / "formal_segments", workers=1)


def test_downloader_quarantines_bad_partial_and_recovers_stale_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = b"good-segment"
    digest = hashlib.md5(good).hexdigest()  # noqa: S324
    manifest = tmp_path / "manifest.tsv"
    columns = [
        "cancer_id", "file_id", "file_name", "md5sum", "case_submitter_id",
        "sample_submitter_id", "selected_for_patient",
    ]
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "cancer_id": "BRCA", "file_id": "uuid", "file_name": "x.seg.v2.txt",
                "md5sum": digest, "case_submitter_id": "TCGA-AA-0001",
                "sample_submitter_id": "TCGA-AA-0001-01A", "selected_for_patient": "True",
            }
        )
    output = tmp_path / "candidate_segments"
    target_dir = output / "BRCA"
    target_dir.mkdir(parents=True)
    lock = target_dir / "TCGA-AA-0001-01A__x.seg.v2.txt.lock"
    lock.write_text("stale", encoding="utf-8")
    old = time.time() - 100
    os.utime(lock, (old, old))
    responses = iter([b"bad", good])
    monkeypatch.setattr(
        cnv.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _Response(next(responses)),
    )
    result = cnv.download_manifest(
        manifest, output, workers=1, retries=1, lock_ttl_seconds=1
    )
    assert result[0]["status"] == "DOWNLOADED_VERIFIED"
    assert not lock.exists()
    assert list(target_dir.glob("*.partial.corrupt.attempt1"))


def test_postdownload_audit_requires_exact_size_md5_and_writes_retry_rows(tmp_path: Path) -> None:
    file_id = "11111111-2222-3333-4444-555555555555"
    content = b"Chromosome\tStart\tEnd\tSegment_Mean\n1\t1\t2\t0.5\n"
    hit = _hit(file_id, "0001", "Primary Tumor")
    hit["file_size"] = len(content)
    hit["md5sum"] = hashlib.md5(content).hexdigest()  # noqa: S324
    rows, audit = cnv.flatten_tumour_manifest([hit])
    authority = cnv.write_manifest(rows, tmp_path / "manifest", {**audit, "queried_cancers": ["BRCA"]})
    table = tmp_path / "manifest" / authority["table"]["path"]
    root = tmp_path / "candidate_full33_segments"
    target = cnv.selected_gdc_download_candidates(rows[0], root)[0]
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    ready = cnv.audit_selected_segment_download(table, tmp_path / "manifest" / "MANIFEST.json", root)
    assert ready["status"] == "READY"
    assert ready["verified_files"] == 1
    target.write_bytes(content + b"drift")
    blocked = cnv.audit_selected_segment_download(table, tmp_path / "manifest" / "MANIFEST.json", root)
    assert blocked["status"] == "BLOCKED"
    assert blocked["retry_rows"][0]["preflight_retry_reason"] == "SIZE_MISMATCH"


def test_postdownload_audit_accepts_standard_gdc_layout_and_rejects_layout_conflict(tmp_path: Path) -> None:
    file_id = "11111111-2222-3333-4444-555555555555"
    content = b"segment"
    hit = _hit(file_id, "0001", "Primary Tumor")
    hit["file_size"] = len(content)
    hit["md5sum"] = hashlib.md5(content).hexdigest()  # noqa: S324
    rows, manifest_audit = cnv.flatten_tumour_manifest([hit])
    authority = cnv.write_manifest(
        rows, tmp_path / "manifest", {**manifest_audit, "queried_cancers": ["BRCA"]}
    )
    table = tmp_path / "manifest" / authority["table"]["path"]
    root = tmp_path / "candidate_uuid_downloads"
    nested, standard = cnv.selected_gdc_download_candidates(rows[0], root)
    standard.parent.mkdir(parents=True)
    standard.write_bytes(content)
    ready = cnv.audit_selected_segment_download(table, tmp_path / "manifest" / "MANIFEST.json", root)
    assert ready["status"] == "READY"
    assert ready["records"][0]["source_layout"] == "UUID_FILENAME"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(content)
    conflict = cnv.audit_selected_segment_download(table, tmp_path / "manifest" / "MANIFEST.json", root)
    assert conflict["status"] == "BLOCKED"
    assert conflict["records"][0]["status"] == "LAYOUT_CONFLICT"


def test_failed_staging_never_publishes_final_and_retains_typed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_id = "11111111-2222-3333-4444-555555555555"
    content = b"segment"
    hit = _hit(file_id, "0001", "Primary Tumor")
    hit["file_size"] = len(content)
    hit["md5sum"] = hashlib.md5(content).hexdigest()  # noqa: S324
    rows, manifest_audit = cnv.flatten_tumour_manifest([hit])
    authority = cnv.write_manifest(
        rows, tmp_path / "manifest", {**manifest_audit, "queried_cancers": ["BRCA"]}
    )
    source_root = tmp_path / "candidate_uuid_downloads"
    source = cnv.selected_gdc_download_candidates(rows[0], source_root)[0]
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    audit = cnv.audit_selected_segment_download(
        tmp_path / "manifest" / authority["table"]["path"],
        tmp_path / "manifest" / "MANIFEST.json",
        source_root,
    )
    staging = tmp_path / "candidate_training_staging"
    monkeypatch.setattr(cnv, "md5_file", lambda _path: "0" * 32)
    with pytest.raises(cnv.SegmentCNVError, match="MD5 drift"):
        cnv.materialize_verified_segment_staging(audit, staging)
    assert not staging.exists()
    evidence = list(tmp_path.glob(".candidate_training_staging.building.*/STAGING_INCOMPLETE.json"))
    assert len(evidence) == 1
    assert json.loads(evidence[0].read_text(encoding="utf-8"))["status"] == "FAILED"


def test_marker_last_publish_is_ready_only_after_all_payload_moves(tmp_path: Path) -> None:
    source = tmp_path / ".partition.building"
    target = tmp_path / "partition"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"payload")
    (source / "SUCCESS.json").write_text('{"status":"SUCCESS"}\n', encoding="utf-8")

    method = cnv._marker_last_directory_publish(source, target)

    assert method == "MKDIR_MARKER_LAST_NOREPLACE"
    assert not source.exists()
    assert (target / "payload.bin").read_bytes() == b"payload"
    assert (target / "SUCCESS.json").is_file()
    assert not (target / "PUBLISH_INCOMPLETE.json").exists()


def test_marker_last_publish_never_replaces_existing_target(tmp_path: Path) -> None:
    source = tmp_path / ".partition.building"
    target = tmp_path / "partition"
    source.mkdir()
    (source / "SUCCESS.json").write_text('{"status":"SUCCESS"}\n', encoding="utf-8")
    target.mkdir()
    (target / "sentinel.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(cnv.SegmentCNVError, match="collision"):
        cnv._marker_last_directory_publish(source, target)

    assert (target / "sentinel.txt").read_text(encoding="utf-8") == "keep"
    assert (source / "SUCCESS.json").is_file()


def test_marker_last_publish_failure_cannot_expose_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / ".partition.building"
    target = tmp_path / "partition"
    source.mkdir()
    (source / "a.bin").write_bytes(b"a")
    (source / "b.bin").write_bytes(b"b")
    (source / "SUCCESS.json").write_text('{"status":"SUCCESS"}\n', encoding="utf-8")
    real_rename = cnv.os.rename

    def fail_second_payload(src: str | Path, dst: str | Path) -> None:
        if Path(src).name == "b.bin":
            raise OSError(errno.EIO, "injected publish failure")
        real_rename(src, dst)

    monkeypatch.setattr(cnv.os, "rename", fail_second_payload)
    with pytest.raises(OSError, match="injected publish failure"):
        cnv._marker_last_directory_publish(source, target)

    assert not (target / "SUCCESS.json").exists()
    failure = json.loads((target / "PUBLISH_INCOMPLETE.json").read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED"
    assert failure["moved_entries"] == 1


def test_ready_gate_is_reverified_and_binding_drift_is_rejected(tmp_path: Path) -> None:
    file_id = "11111111-2222-3333-4444-555555555555"
    content = b"segment"
    hit = _hit(file_id, "0001", "Primary Tumor")
    hit["file_size"] = len(content)
    hit["md5sum"] = hashlib.md5(content).hexdigest()  # noqa: S324
    rows, audit = cnv.flatten_tumour_manifest([hit])
    authority = cnv.write_manifest(rows, tmp_path / "manifest", {**audit, "queried_cancers": ["BRCA"]})
    table = tmp_path / "manifest" / authority["table"]["path"]
    root = tmp_path / "candidate_full33_segments"
    target = cnv.selected_gdc_download_candidates(rows[0], root)[0]
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    audited = cnv.audit_selected_segment_download(table, tmp_path / "manifest" / "MANIFEST.json", root)
    staging_root = tmp_path / "candidate_training_staging"
    staging = cnv.materialize_verified_segment_staging(audited, staging_root)
    staged_target = cnv.selected_segment_target(rows[0], staging_root)
    assert staged_target.read_bytes() == content
    assert (staging_root / "STAGING_COMPLETE.json").is_file()
    binding = tmp_path / "candidates.parquet"
    binding.write_bytes(b"bound")
    callability = tmp_path / "PATIENT_CNV_SEGMENT_CALLABILITY.tsv"
    callability.write_text(
        "cancer_id\tpatient_id\tpatient_fold_id\tsegment_manifest_available\t"
        "selected_file_id\tcnv_callability_state\tcnv_unavailable_reason\n"
        f"BRCA\tTCGA-AA-0001\t0\tTrue\t{file_id}\tSELECTED_SEGMENT_BOUND\t\n",
        encoding="utf-8",
    )
    marker_payload = {
        **{key: value for key, value in audited.items() if key not in {"columns", "retry_rows", "records"}},
        "format": "CC_HHGT_V3_2_FULL33_SEGMENT_DOWNLOAD_GATE_V1",
        "required_cancers": list(cnv.TCGA_CANCERS),
        "patient_folds": 5,
        "implementation_sha256": cnv.full33_gate_implementation_hashes(),
        "source_download_root": str(root.resolve()),
        "source_inventory_sha256": audited["inventory_sha256"],
        "staging_root": str(staging_root.resolve()),
        "staging_inventory_sha256": staging["inventory_sha256"],
        "patient_segment_callability": {
            "path": str(callability.resolve()), "sha256": cnv.sha256_file(callability),
            "fold_patients": 1, "typed_unavailable": 0,
        },
        "bindings": {
            "candidates": {"path": str(binding.resolve()), "sha256": cnv.sha256_file(binding)}
        },
    }
    marker = tmp_path / "DOWNLOAD_COMPLETE.json"
    marker.write_text(json.dumps(marker_payload), encoding="utf-8")
    cnv.validate_segment_download_gate(marker, staging_root=staging_root, bindings={"candidates": binding})
    binding.write_bytes(b"drift")
    with pytest.raises(cnv.SegmentCNVError, match="binding drift"):
        cnv.validate_segment_download_gate(marker, staging_root=staging_root, bindings={"candidates": binding})


def test_downloader_never_overwrites_conflicting_target_or_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = b"expected"
    manifest = tmp_path / "manifest.tsv"
    row = {
        "cancer_id": "BRCA", "file_id": "uuid", "file_name": "x.seg.v2.txt",
        "md5sum": hashlib.md5(good).hexdigest(), "file_size": str(len(good)),  # noqa: S324
        "case_submitter_id": "TCGA-AA-0001", "sample_submitter_id": "TCGA-AA-0001-01A",
        "selected_for_patient": "True",
    }
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    output = tmp_path / "candidate_segments"
    target = cnv.selected_segment_target(row, output)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"preserve-me")
    monkeypatch.setattr(cnv.urllib.request, "urlopen", lambda *_a, **_k: pytest.fail("network used"))
    conflict = cnv.download_manifest(manifest, output, workers=1)
    assert conflict[0]["status"] == "TARGET_CONFLICT"
    assert target.read_bytes() == b"preserve-me"
    target.unlink()
    monkeypatch.setattr(cnv.urllib.request, "urlopen", lambda *_a, **_k: _Response(b"bad"))
    assert cnv.download_manifest(manifest, output, workers=1, retries=0)[0]["status"] == "FAILED"
    assert cnv.download_manifest(manifest, output, workers=1, retries=0)[0]["status"] == "FAILED"
    quarantines = sorted(target.parent.glob(f"{target.name}.partial.corrupt.attempt1*"))
    assert len(quarantines) == 2
