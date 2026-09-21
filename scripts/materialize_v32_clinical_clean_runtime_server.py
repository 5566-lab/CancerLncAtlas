#!/usr/bin/env python3
"""Create a clean, hard-linked Clinical runtime view without transfer residues."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
from pathlib import Path

import pyarrow.parquet as pq


EXPRESSION_SHA256 = "9b7907242f99be0964311c66573dd799805cd3e857ba1d804e5556611a575b99"
CURVES_SHA256 = "242a838ac25c54222142683d85bda4a29f25308ba8bdac5e22813a3bdd850d06"
EXPRESSION_ROWS = 23_955_621
CURVE_ROWS = 4_265_340
_CANCER_EXPRESSION = re.compile(r"^cancer_id=[A-Z0-9_-]{2,16}/part-0\.parquet$")
_CANCER_QC = re.compile(r"^cancer_id=[A-Z0-9_-]{2,16}/QC\.json$")
_CURVE_PART = re.compile(r"^part-[A-Z0-9_-]{2,16}\.parquet$")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for item in sorted(
        files, key=lambda value: value.relative_to(root).as_posix().casefold()
    ):
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def selected_expression_files(root: Path) -> list[Path]:
    selected: list[Path] = []
    for item in root.rglob("*"):
        if not item.is_file() or item.is_symlink():
            continue
        rel = item.relative_to(root).as_posix()
        if rel == "SUCCESS.json" or _CANCER_EXPRESSION.fullmatch(rel) or _CANCER_QC.fullmatch(rel):
            selected.append(item)
    return sorted(selected)


def selected_curve_files(root: Path) -> list[Path]:
    return sorted(
        item
        for item in root.iterdir()
        if item.is_file()
        and not item.is_symlink()
        and _CURVE_PART.fullmatch(item.name)
    )


def materialize(source_root: Path, target_root: Path, files: list[Path]) -> None:
    target_root.mkdir(parents=True, exist_ok=False)
    for source in files:
        relative = source.relative_to(source_root)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(source, target)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression-source", required=True)
    parser.add_argument("--expression-qc-archive", required=True)
    parser.add_argument("--curves-source", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--audit-existing", action="store_true")
    args = parser.parse_args()

    expression_source = Path(args.expression_source).resolve(strict=True)
    qc_archive = Path(args.expression_qc_archive).resolve(strict=True)
    curves_source = Path(args.curves_source).resolve(strict=True)
    output_root = Path(args.output_root).resolve()
    allowed_sources = Path("./data/CancerLncAtlas/runtime/web_candidates").resolve()
    allowed_output = Path("./data/CancerLncAtlas/runtime/authorized_payloads").resolve()
    allowed_metadata = Path("./data/CancerLncAtlas/runtime/authorized_bindings").resolve()
    for label, source in (("expression", expression_source), ("curves", curves_source)):
        if source.is_symlink() or not source.is_dir() or allowed_sources not in source.parents:
            raise RuntimeError(f"Unsafe {label} source: {source}")
    if output_root == allowed_output or allowed_output not in output_root.parents:
        raise RuntimeError(f"Output is outside authorized_payloads: {output_root}")
    if (
        not qc_archive.is_file()
        or qc_archive.is_symlink()
        or allowed_metadata not in qc_archive.parents
    ):
        raise RuntimeError(f"Unsafe expression QC archive: {qc_archive}")
    if output_root.exists() and not args.audit_existing:
        raise RuntimeError(f"Refusing to overwrite clean runtime root: {output_root}")
    if args.audit_existing and not output_root.is_dir():
        raise RuntimeError(f"Existing clean runtime root is missing: {output_root}")

    expression_files = selected_expression_files(expression_source)
    curve_files = selected_curve_files(curves_source)
    expression_rel = [item.relative_to(expression_source).as_posix() for item in expression_files]
    curve_rel = [item.relative_to(curves_source).as_posix() for item in curve_files]
    if len(expression_files) != 34 or expression_rel.count("SUCCESS.json") != 1:
        raise RuntimeError(f"Expected 33 expression parts plus SUCCESS, got {len(expression_files)}")
    if len(curve_files) != 33:
        raise RuntimeError(f"Expected 33 Clinical curve parts, got {len(curve_files)}")
    qc_payloads: dict[str, bytes] = {}
    with tarfile.open(qc_archive, "r:*") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not _CANCER_QC.fullmatch(member.name):
                raise RuntimeError(f"Unsafe or unexpected QC archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Unreadable QC archive member: {member.name}")
            qc_payloads[member.name] = stream.read()
    if len(qc_payloads) != 33:
        raise RuntimeError(f"Expected 33 QC metadata leaves, got {len(qc_payloads)}")
    expression_entries = {
        item.relative_to(expression_source).as_posix(): file_sha256(item)
        for item in expression_files
    }
    expression_entries.update(
        {name: hashlib.sha256(payload).hexdigest() for name, payload in qc_payloads.items()}
    )
    expression_digest = hashlib.sha256()
    for name in sorted(expression_entries, key=str.casefold):
        expression_digest.update(name.encode("utf-8"))
        expression_digest.update(b"\0")
        expression_digest.update(expression_entries[name].encode("ascii"))
        expression_digest.update(b"\n")
    expression_hash = expression_digest.hexdigest()
    curves_hash = tree_sha256(curves_source, curve_files)
    if expression_hash != EXPRESSION_SHA256:
        raise RuntimeError(f"Clean expression tree SHA mismatch: {expression_hash}")
    if curves_hash != CURVES_SHA256:
        raise RuntimeError(f"Clean curves tree SHA mismatch: {curves_hash}")
    expression_rows = sum(
        int(pq.ParquetFile(path).metadata.num_rows)
        for path in expression_files
        if path.suffix == ".parquet"
    )
    curve_rows = sum(int(pq.ParquetFile(path).metadata.num_rows) for path in curve_files)
    if expression_rows != EXPRESSION_ROWS or curve_rows != CURVE_ROWS:
        raise RuntimeError(
            f"Clinical row mismatch: expression={expression_rows}, curves={curve_rows}"
        )

    expression_target = output_root / "formal_lncRNA_expression"
    curves_target = output_root / "clinical_km_curves"
    if not args.audit_existing:
        output_root.mkdir(parents=True, exist_ok=False)
        # The root now exists, so materialize each child as a new directory.
        materialize(expression_source, expression_target, expression_files)
        materialize(curves_source, curves_target, curve_files)
        for relative, payload in sorted(qc_payloads.items()):
            target = expression_target / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)
    target_expression_files = selected_expression_files(expression_target)
    target_curve_files = selected_curve_files(curves_target)
    if tree_sha256(expression_target, target_expression_files) != EXPRESSION_SHA256:
        raise RuntimeError("Materialized expression tree SHA mismatch")
    if tree_sha256(curves_target, target_curve_files) != CURVES_SHA256:
        raise RuntimeError("Materialized curves tree SHA mismatch")
    if any(".partial." in item.name for item in output_root.rglob("*")):
        raise RuntimeError("Transfer residue entered clean Clinical runtime view")
    expression_identity = [
        os.path.samefile(source, expression_target / source.relative_to(expression_source))
        for source in expression_files
    ]
    curve_identity = [
        os.path.samefile(source, curves_target / source.relative_to(curves_source))
        for source in curve_files
    ]
    source_target_sha_mismatches = 0
    for source in expression_files:
        target = expression_target / source.relative_to(expression_source)
        source_target_sha_mismatches += int(file_sha256(source) != file_sha256(target))
    for source in curve_files:
        target = curves_target / source.relative_to(curves_source)
        source_target_sha_mismatches += int(file_sha256(source) != file_sha256(target))
    if source_target_sha_mismatches:
        raise RuntimeError("Clinical source/target file SHA mismatch")
    identity_preserved = all(expression_identity) and all(curve_identity)
    target_payload_bytes = sum(item.stat().st_size for item in target_expression_files)
    target_payload_bytes += sum(item.stat().st_size for item in target_curve_files)

    receipt = {
        "format": "CANCERLNCATLAS_V32_CLINICAL_CLEAN_RUNTIME_PAYLOAD_V1",
        "status": "PASS",
        "cause": "TRANSFER_PARTIAL_RESIDUES_EXCLUDED_AND_OMITTED_QC_METADATA_RESTORED",
        "source_residue_files": {
            "expression": sum(1 for item in expression_source.rglob("*") if item.is_file() and ".partial." in item.name),
            "curves": sum(1 for item in curves_source.rglob("*") if item.is_file() and ".partial." in item.name),
        },
        "expression": {
            "path": str(expression_target),
            "files": len(expression_files) + len(qc_payloads),
            "rows": expression_rows,
            "sha256_tree": EXPRESSION_SHA256,
        },
        "curves": {
            "path": str(curves_target),
            "files": len(curve_files),
            "rows": curve_rows,
            "sha256_tree": CURVES_SHA256,
        },
        "qc_metadata_archive": {
            "path": str(qc_archive),
            "sha256": file_sha256(qc_archive),
            "files": len(qc_payloads),
            "bytes": sum(map(len, qc_payloads.values())),
        },
        "materialization_mode": (
            "HARDLINK_IDENTITY_VERIFIED"
            if identity_preserved
            else "SSHFS_LINK_IDENTITY_NOT_PRESERVED_TREATED_AS_PHYSICAL_COPY"
        ),
        "hardlink_identity_verified_files": (
            len(expression_files) + len(curve_files) if identity_preserved else 0
        ),
        "source_target_sha_mismatches": source_target_sha_mismatches,
        "physical_copy_bytes": (
            sum(map(len, qc_payloads.values()))
            if identity_preserved
            else target_payload_bytes
        ),
        "audit_existing": args.audit_existing,
        "production_deployed": False,
        "release_ready": False,
        "main_score_changed": False,
        "production_port_8260_touched": False,
    }
    receipt_path = output_root / "CLEAN_RUNTIME_PAYLOAD_AUDIT.json"
    encoded = (json.dumps(receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with receipt_path.open("xb") as stream:
        stream.write(encoded)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
