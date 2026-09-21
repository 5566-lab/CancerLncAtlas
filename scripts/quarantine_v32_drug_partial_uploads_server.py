#!/usr/bin/env python3
"""Quarantine stale Drug upload partials after exact authorization checks.

The operation is deliberately narrow: only files named
``<authorized-file>.partial.<authorized-sha256>`` may be moved.  Authorized
payloads are never modified, and every moved file remains recoverable beneath
the requested quarantine root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


FORMAT = "CANCERLNCATLAS_V32_DRUG_STALE_UPLOAD_PARTIAL_QUARANTINE_V1"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_AUTHORIZED_BINDING_V1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PARTIAL = re.compile(r"^(?P<canonical>.+)\.partial\.(?P<sha>[0-9a-f]{64})$")
EXPECTED_AUTHORIZED_FILES = 699
EXPECTED_PARTIAL_FILES = 699
EXPECTED_AVAILABLE_PARTIAL_FILES = 694
EXPECTED_FACTOR_PARTIAL_FILES = 5


class PartialQuarantineError(RuntimeError):
    """The stale-upload quarantine could not be proven safe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PartialQuarantineError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pinned(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    expected = str(expected_sha256).lower()
    _require(SHA256.fullmatch(expected) is not None, f"{label} SHA256 is invalid")
    _require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked")
    _require(_sha256_file(path) == expected, f"{label} SHA256 mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _files_below(root: Path) -> Iterable[Path]:
    """Yield regular files without following directory symlinks."""
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                _require(not entry.is_symlink(), f"Symlink is forbidden in artifact tree: {path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)
                elif entry.is_file(follow_symlinks=False):
                    yield path
                else:
                    raise PartialQuarantineError(f"Unsupported filesystem entry: {path}")


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    _require(not path.exists(), f"Refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def _inventory_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    lines = [
        "\t".join(
            (
                str(item["relative_path"]),
                str(item["sha256"]),
                str(int(item["bytes"])),
                str(item["classification"]),
            )
        )
        for item in sorted(records, key=lambda value: str(value["relative_path"]))
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def quarantine(
    *,
    artifact_root: Path,
    binding_path: Path,
    binding_sha256: str,
    core_receipt_path: Path,
    core_receipt_sha256: str,
    quarantine_root: Path,
    apply: bool,
    approved_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    requested_root = artifact_root.absolute()
    root = artifact_root.resolve(strict=True)
    _require(
        requested_root == root and root.is_dir() and not root.is_symlink(),
        "Artifact root must be an existing canonical non-symlink directory",
    )
    quarantine_root = quarantine_root.absolute()
    _require(not quarantine_root.exists(), "Quarantine root already exists")
    _require(root not in quarantine_root.parents, "Quarantine root cannot be inside artifact root")

    binding = _load_pinned(binding_path.resolve(strict=True), binding_sha256, "binding")
    core = _load_pinned(
        core_receipt_path.resolve(strict=True), core_receipt_sha256, "core receipt"
    )
    _require(binding.get("format") == BINDING_FORMAT, "Binding format drifted")
    _require(binding.get("analysis_version") == ANALYSIS_VERSION, "Binding analysis drifted")
    authority = core.get("artifact_root_authority", {})
    _require(core.get("status") == "PASS", "Core receipt is not PASS")
    _require(authority.get("path") == str(root), "Core receipt audited a different root")
    _require(
        authority.get("payload_files_hash_and_bytes_verified") == EXPECTED_AUTHORIZED_FILES,
        "Core receipt did not hash-and-byte verify all authorized payloads",
    )

    payloads = binding.get("components", {}).get("drug", {}).get("payloads")
    _require(
        isinstance(payloads, list) and len(payloads) == EXPECTED_AUTHORIZED_FILES,
        "Authorized Drug payload denominator drifted",
    )
    authorized: dict[Path, dict[str, Any]] = {}
    for item in payloads:
        _require(isinstance(item, dict), "Authorized payload is malformed")
        path = Path(str(item.get("path", ""))).absolute()
        _require(root in path.parents, f"Authorized payload escaped root: {path}")
        expected_sha = str(item.get("sha256", "")).lower()
        expected_bytes = int(item.get("bytes", -1))
        _require(SHA256.fullmatch(expected_sha) is not None, "Authorized SHA256 is invalid")
        _require(path.is_file() and not path.is_symlink(), f"Authorized payload missing: {path}")
        _require(path.stat().st_size == expected_bytes, f"Authorized bytes drifted: {path}")
        _require(path not in authorized, f"Authorized path duplicated: {path}")
        authorized[path] = {"sha256": expected_sha, "bytes": expected_bytes}

    physical = set(_files_below(root))
    authorized_paths = set(authorized)
    _require(authorized_paths <= physical, "One or more authorized files are absent")
    unexpected = sorted(physical - authorized_paths, key=lambda value: value.as_posix())
    _require(
        len(unexpected) == EXPECTED_PARTIAL_FILES,
        f"Unexpected-file denominator is {len(unexpected)}, not {EXPECTED_PARTIAL_FILES}",
    )

    authorized_inventory = [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": item["sha256"],
            "bytes": item["bytes"],
            "classification": "AUTHORIZED_CANONICAL_PAYLOAD",
        }
        for path, item in authorized.items()
    ]
    planned: list[dict[str, Any]] = []
    partial_inventory: list[dict[str, Any]] = []
    for source in unexpected:
        match = PARTIAL.fullmatch(source.name)
        _require(match is not None, f"Unexpected file is not a typed upload partial: {source}")
        canonical = source.with_name(match.group("canonical"))
        authorized_item = authorized.get(canonical)
        _require(authorized_item is not None, f"Partial lacks authorized canonical peer: {source}")
        _require(
            match.group("sha") == authorized_item["sha256"],
            f"Partial filename SHA differs from authorized peer: {source}",
        )
        size = source.stat().st_size
        _require(size == authorized_item["bytes"], f"Partial bytes differ: {source}")
        observed_sha = _sha256_file(source)
        _require(
            observed_sha == authorized_item["sha256"],
            f"Partial content SHA differs from authorized canonical peer: {source}",
        )
        relative = source.relative_to(root)
        destination = quarantine_root / relative
        planned.append(
            {
                "source": str(source),
                "destination": str(destination),
                "canonical_peer": str(canonical),
                "declared_sha256_in_partial_name": match.group("sha"),
                "observed_sha256": observed_sha,
                "bytes": size,
            }
        )
        partial_inventory.append(
            {
                "relative_path": relative.as_posix(),
                "sha256": observed_sha,
                "bytes": size,
                "classification": "UNAUTHORIZED_STALE_UPLOAD_PARTIAL",
            }
        )

    _require(
        len({item["canonical_peer"] for item in planned}) == EXPECTED_PARTIAL_FILES,
        "More than one upload partial maps to the same canonical payload",
    )
    _require(
        {Path(item["canonical_peer"]) for item in planned} == authorized_paths,
        "Upload partials are not in exact one-to-one correspondence with all authorized payloads",
    )
    available_partial_files = sum(
        item["relative_path"].startswith("drug_response_association/")
        for item in partial_inventory
    )
    factor_partial_files = sum(
        item["relative_path"].startswith("drug_sparse_query_factors/")
        for item in partial_inventory
    )
    _require(
        available_partial_files == EXPECTED_AVAILABLE_PARTIAL_FILES
        and factor_partial_files == EXPECTED_FACTOR_PARTIAL_FILES,
        "Upload partial subtree denominators drifted",
    )
    pre_inventory = sorted(
        authorized_inventory + partial_inventory,
        key=lambda item: item["relative_path"],
    )
    pre_inventory_sha256 = _inventory_sha256(pre_inventory)
    authorized_inventory = sorted(
        authorized_inventory, key=lambda item: item["relative_path"]
    )
    authorized_inventory_sha256 = _inventory_sha256(authorized_inventory)
    partial_inventory = sorted(partial_inventory, key=lambda item: item["relative_path"])
    partial_inventory_sha256 = _inventory_sha256(partial_inventory)
    if apply:
        _require(isinstance(approved_plan, Mapping), "Apply requires a pinned plan")
        _require(approved_plan.get("status") == "PLAN_PASS", "Approved plan is not PLAN_PASS")
        _require(approved_plan.get("artifact_root") == str(root), "Plan artifact root drifted")
        _require(
            approved_plan.get("post_operation", {}).get("quarantine_root")
            == str(quarantine_root),
            "Plan quarantine root drifted",
        )
        _require(
            approved_plan.get("pre_operation", {}).get("physical_inventory_sha256")
            == pre_inventory_sha256,
            "Physical tree drifted after the approved plan",
        )
    moved = 0
    if apply:
        quarantine_root.mkdir(parents=True, exist_ok=False)
        for item in planned:
            source = Path(item["source"])
            destination = Path(item["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            _require(not destination.exists(), f"Quarantine destination exists: {destination}")
            os.replace(source, destination)
            moved += 1
        remaining = set(_files_below(root))
        _require(remaining == authorized_paths, "Artifact root is not exact after quarantine")
        quarantined = set(_files_below(quarantine_root))
        _require(len(quarantined) == EXPECTED_PARTIAL_FILES, "Quarantine file count drifted")

    return {
        "format": FORMAT,
        "status": "PASS" if apply else "PLAN_PASS",
        "analysis_version": ANALYSIS_VERSION,
        "operation": "ATOMIC_MOVE_TO_RECOVERABLE_QUARANTINE" if apply else "READ_ONLY_PLAN",
        "artifact_root": str(root),
        "authorized_binding": {
            "path": str(binding_path.resolve()),
            "sha256": str(binding_sha256).lower(),
        },
        "core_receipt": {
            "path": str(core_receipt_path.resolve()),
            "sha256": str(core_receipt_sha256).lower(),
        },
        "pre_operation": {
            "authorized_files": len(authorized_paths),
            "unexpected_files": len(unexpected),
            "unexpected_files_other_than_typed_upload_partials": 0,
            "available_prediction_partial_files": available_partial_files,
            "factor_partial_files": factor_partial_files,
            "physical_files": len(physical),
            "partial_bytes": sum(item["bytes"] for item in planned),
            "all_unexpected_files_typed_as_partial_of_exact_authorized_peer": True,
            "partial_filename_sha_matches_authorized_peer": True,
            "partial_bytes_match_authorized_peer": True,
            "partial_content_rehashed": True,
            "partial_content_sha_matches_authorized_canonical_sha": True,
            "physical_inventory_sha256": pre_inventory_sha256,
            "physical_inventory": pre_inventory,
        },
        "post_operation": {
            "moved_files": moved,
            "artifact_files": EXPECTED_AUTHORIZED_FILES if apply else None,
            "unexpected_files": 0 if apply else None,
            "quarantine_files": EXPECTED_PARTIAL_FILES if apply else None,
            "quarantine_root": str(quarantine_root),
            "recoverable": True,
            "artifact_inventory_sha256": (
                authorized_inventory_sha256 if apply else None
            ),
            "artifact_inventory": authorized_inventory if apply else None,
            "quarantine_inventory_sha256": (
                partial_inventory_sha256 if apply else None
            ),
            "quarantine_inventory": partial_inventory if apply else None,
        },
        "root_cause": (
            "UPLOAD_FINALIZATION_LEFT_ONE_DOT_PARTIAL_DOT_SHA256_FILE_BESIDE_"
            "EACH_OF_ALL_699_AUTHORIZED_DRUG_PAYLOADS_INCLUDING_694_AVAILABLE_"
            "PREDICTION_PARTS_AND_5_FACTOR_FILES"
        ),
        "security_gates": {
            "authorized_payload_modified": False,
            "unexpected_non_partial_file_accepted": False,
            "partial_without_exact_authorized_peer_accepted": False,
            "symlink_accepted": False,
            "old_audit_overwritten": False,
        },
        "scientific_table_rows_read": 0,
        "primary_score_changed": False,
        "production_port_8260_touched": False,
        "production_deployed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--core-receipt", type=Path, required=True)
    parser.add_argument("--core-receipt-sha256", required=True)
    parser.add_argument("--quarantine-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--approved-plan", type=Path)
    parser.add_argument("--approved-plan-sha256")
    args = parser.parse_args()
    output = args.output.absolute()
    _require(not output.exists(), f"Refusing to overwrite receipt: {output}")
    approved_plan = None
    if args.apply:
        _require(
            args.approved_plan is not None and args.approved_plan_sha256 is not None,
            "--apply requires --approved-plan and --approved-plan-sha256",
        )
        approved_plan = _load_pinned(
            args.approved_plan.resolve(strict=True),
            args.approved_plan_sha256,
            "approved plan",
        )
    else:
        _require(
            args.approved_plan is None and args.approved_plan_sha256 is None,
            "Read-only plan cannot accept an approved plan",
        )
    payload = quarantine(
        artifact_root=args.artifact_root,
        binding_path=args.binding,
        binding_sha256=args.binding_sha256,
        core_receipt_path=args.core_receipt,
        core_receipt_sha256=args.core_receipt_sha256,
        quarantine_root=args.quarantine_root,
        apply=args.apply,
        approved_plan=approved_plan,
    )
    _write_exclusive(output, payload)
    print(payload["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
