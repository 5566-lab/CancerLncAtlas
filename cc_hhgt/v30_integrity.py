from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


CHUNK_SIZE = 8 * 1024 * 1024


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def merkle_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    leaves: list[bytes] = []
    for record in sorted(records, key=lambda item: str(item["relative_path"]).replace("\\", "/")):
        normalized = {
            "relative_path": str(record["relative_path"]).replace("\\", "/"),
            "size_bytes": int(record["size_bytes"]),
            "sha256": str(record["sha256"]).lower(),
        }
        leaves.append(hashlib.sha256(canonical_json_bytes(normalized)).digest())
    if not leaves:
        return hashlib.sha256(b"").hexdigest()
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [hashlib.sha256(leaves[index] + leaves[index + 1]).digest() for index in range(0, len(leaves), 2)]
    return leaves[0].hex()


def build_file_manifest(root: str | Path, files: Iterable[str | Path] | None = None) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    paths = sorted((Path(path) for path in files), key=lambda path: path.as_posix()) if files is not None else sorted(
        (path for path in root_path.rglob("*") if path.is_file()), key=lambda path: path.as_posix()
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root_path).as_posix()
        except ValueError as exc:
            raise ValueError(f"Manifest file is outside root: {resolved} not under {root_path}") from exc
        stat = resolved.stat()
        records.append(
            {
                "relative_path": relative,
                "size_bytes": int(stat.st_size),
                "sha256": file_sha256(resolved),
            }
        )
    return records


def verify_file_manifest(root: str | Path, records: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    root_path = Path(root).resolve()
    mismatches: list[dict[str, str]] = []
    for record in records:
        relative = str(record["relative_path"]).replace("/", os.sep)
        path = (root_path / relative).resolve()
        try:
            path.relative_to(root_path)
        except ValueError:
            mismatches.append({"relative_path": relative, "expected": str(record["sha256"]), "observed": "OUTSIDE_ROOT"})
            continue
        observed = file_sha256(path) if path.is_file() else "MISSING"
        if observed.lower() != str(record["sha256"]).lower():
            mismatches.append({"relative_path": relative, "expected": str(record["sha256"]), "observed": observed})
    return mismatches


def atomic_write_bytes(path: str | Path, content: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def atomic_write_json(path: str | Path, payload: Any) -> None:
    atomic_write_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n")


def stable_partition(value: str, modulo: int, seed: int) -> int:
    if modulo <= 0:
        raise ValueError("modulo must be positive")
    digest = hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False) % modulo


def cache_key_sha256(
    *, code_sha256: str, config_sha256: str, input_sha256: str, fold_sha256: str, schema_sha256: str
) -> str:
    return canonical_json_sha256(
        {
            "code_sha256": code_sha256,
            "config_sha256": config_sha256,
            "input_sha256": input_sha256,
            "fold_sha256": fold_sha256,
            "schema_sha256": schema_sha256,
        }
    )

