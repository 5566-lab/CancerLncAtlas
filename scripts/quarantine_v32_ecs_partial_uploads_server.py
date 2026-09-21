#!/usr/bin/env python3
"""Fail-closed quarantine for Evidence/Clinical/State upload partials."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping


FORMAT = "CANCERLNCATLAS_V32_ECS_UPLOAD_PARTIAL_QUARANTINE_V1"
BINDING_FORMAT = "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_AUTHORIZED_BINDING_V1"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
EXPECTED = {"evidence": 21, "clinical": 34, "state_gene_set": 4}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class QuarantineError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QuarantineError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    expected_sha256 = expected_sha256.lower()
    require(SHA256.fullmatch(expected_sha256) is not None, f"Invalid {label} SHA256")
    path = path.resolve(strict=True)
    require(path.is_file() and not path.is_symlink(), f"Missing or symlinked {label}")
    require(sha256_file(path) == expected_sha256, f"{label} SHA256 mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path = path.absolute()
    require(not path.exists(), f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def inventory_sha(records: list[dict[str, Any]]) -> str:
    lines = []
    for row in sorted(records, key=lambda item: (item["component"], item["canonical_path"])):
        lines.append("\t".join((row["component"], row["canonical_path"], row["partial_path"], row["sha256"], str(row["bytes"]))))
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def inspect(binding: Mapping[str, Any], candidate_root: Path, quarantine_root: Path) -> list[dict[str, Any]]:
    components = binding.get("components")
    require(isinstance(components, dict), "Binding components are missing")
    records: list[dict[str, Any]] = []
    seen_canonical: set[Path] = set()
    seen_partial: set[Path] = set()
    for component, expected_count in EXPECTED.items():
        declaration = components.get(component)
        require(isinstance(declaration, dict), f"Missing component: {component}")
        require(declaration.get("release_ready") is False, f"{component} release_ready drifted")
        require(declaration.get("production_deployed") is False, f"{component} production_deployed drifted")
        payloads = declaration.get("payloads")
        require(isinstance(payloads, list) and len(payloads) == expected_count, f"{component} payload count drifted")
        require(declaration.get("payload_file_count") == expected_count, f"{component} declared payload count drifted")
        for payload in payloads:
            require(isinstance(payload, dict), f"Malformed {component} payload")
            canonical = Path(str(payload.get("path", ""))).absolute()
            require(candidate_root in canonical.parents, f"Canonical escaped candidate root: {canonical}")
            require(canonical not in seen_canonical, f"Duplicate canonical declaration: {canonical}")
            seen_canonical.add(canonical)
            expected_sha = str(payload.get("sha256", "")).lower()
            expected_bytes = int(payload.get("bytes", -1))
            require(SHA256.fullmatch(expected_sha) is not None, f"Invalid canonical SHA: {canonical}")
            require(canonical.is_file() and not canonical.is_symlink(), f"Missing or symlinked canonical: {canonical}")
            require(canonical.stat().st_size == expected_bytes, f"Canonical byte drift: {canonical}")
            require(sha256_file(canonical) == expected_sha, f"Canonical content SHA drift: {canonical}")
            matches = sorted(canonical.parent.glob(canonical.name + ".partial.*"))
            require(len(matches) == 1, f"Expected exactly one partial for {canonical}; found {len(matches)}")
            partial = matches[0]
            require(partial.is_file() and not partial.is_symlink(), f"Missing or symlinked partial: {partial}")
            require(partial not in seen_partial, f"Partial maps more than once: {partial}")
            seen_partial.add(partial)
            suffix = partial.name.removeprefix(canonical.name + ".partial.")
            require(suffix == expected_sha, f"Partial filename SHA drift: {partial}")
            require(partial.stat().st_size == expected_bytes, f"Partial byte drift: {partial}")
            observed = sha256_file(partial)
            require(observed == expected_sha, f"Partial content SHA drift: {partial}")
            relative = partial.relative_to(candidate_root)
            destination = quarantine_root / relative
            require(not destination.exists(), f"Quarantine destination exists: {destination}")
            records.append({
                "component": component,
                "canonical_path": str(canonical),
                "partial_path": str(partial),
                "quarantine_path": str(destination),
                "sha256": observed,
                "bytes": expected_bytes,
                "canonical_rehashed_equal": True,
                "partial_rehashed_equal": True,
            })
    require(len(records) == sum(EXPECTED.values()), "Total one-to-one denominator drifted")
    require(len(seen_canonical) == len(records) == len(seen_partial), "Canonical/partial uniqueness failed")
    return records


def build_payload(*, binding_path: Path, binding_sha256: str, quarantine_root: Path, apply: bool, approved_plan: Mapping[str, Any] | None) -> dict[str, Any]:
    binding = load_pinned(binding_path, binding_sha256, "authorized binding")
    require(binding.get("format") == BINDING_FORMAT, "Binding format drifted")
    require(binding.get("analysis_version") == ANALYSIS_VERSION, "Analysis version drifted")
    require(binding.get("release_ready") is False, "Binding release_ready drifted")
    require(binding.get("production_deployed") is False, "Binding production_deployed drifted")
    guards = binding.get("scope_guards", {})
    require(guards.get("production_port_8260_touched") is False, "Binding port guard drifted")
    candidate_root = Path(str(binding.get("candidate_payload_root", ""))).resolve(strict=True)
    require(candidate_root.is_dir() and not candidate_root.is_symlink(), "Candidate root invalid")
    quarantine_root = quarantine_root.absolute()
    require(not quarantine_root.exists(), "Quarantine root already exists")
    require(candidate_root not in quarantine_root.parents, "Quarantine root cannot be within candidate root")
    records = inspect(binding, candidate_root, quarantine_root)
    inv_sha = inventory_sha(records)
    if apply:
        require(isinstance(approved_plan, Mapping), "Apply requires pinned plan")
        require(approved_plan.get("status") == "PLAN_PASS", "Approved plan is not PLAN_PASS")
        require(approved_plan.get("inventory_sha256") == inv_sha, "Inventory drifted after plan")
        require(approved_plan.get("quarantine_root") == str(quarantine_root), "Quarantine root drifted after plan")
        quarantine_root.mkdir(parents=True, exist_ok=False)
        for row in records:
            source = Path(row["partial_path"])
            destination = Path(row["quarantine_path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            require(not destination.exists(), f"Destination appeared during apply: {destination}")
            os.replace(source, destination)
        for row in records:
            canonical = Path(row["canonical_path"])
            source = Path(row["partial_path"])
            destination = Path(row["quarantine_path"])
            require(not source.exists(), f"Partial remained after move: {source}")
            require(canonical.is_file(), f"Canonical disappeared: {canonical}")
            require(destination.is_file() and not destination.is_symlink(), f"Quarantined file missing: {destination}")
            require(destination.stat().st_size == row["bytes"], f"Quarantined bytes drifted: {destination}")
            require(sha256_file(destination) == row["sha256"], f"Quarantined SHA drifted: {destination}")
    return {
        "format": FORMAT,
        "status": "PASS" if apply else "PLAN_PASS",
        "analysis_version": ANALYSIS_VERSION,
        "operation": "PER_FILE_ATOMIC_MOVE_TO_RECOVERABLE_QUARANTINE" if apply else "READ_ONLY_FAIL_CLOSED_PLAN",
        "authorized_binding": {"path": str(binding_path.resolve()), "sha256": binding_sha256.lower()},
        "candidate_root": str(candidate_root),
        "quarantine_root": str(quarantine_root),
        "expected_counts": EXPECTED,
        "validated_files": len(records),
        "validated_bytes": sum(row["bytes"] for row in records),
        "inventory_sha256": inv_sha,
        "one_to_one_unique_canonical": True,
        "canonical_content_rehashed_equal": True,
        "partial_filename_sha_equal": True,
        "partial_content_rehashed_equal": True,
        "moved_files": len(records) if apply else 0,
        "recoverable": True,
        "records": records,
        "scientific_rows_read": 0,
        "main_score_changed": False,
        "production_port_8260_touched": False,
        "production_deployed": False,
        "release_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--quarantine-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--approved-plan", type=Path)
    parser.add_argument("--approved-plan-sha256")
    args = parser.parse_args()
    require(not args.output.absolute().exists(), f"Refusing to overwrite output: {args.output}")
    plan = None
    if args.apply:
        require(args.approved_plan is not None and args.approved_plan_sha256 is not None, "Apply requires plan and SHA")
        plan = load_pinned(args.approved_plan, args.approved_plan_sha256, "approved plan")
    else:
        require(args.approved_plan is None and args.approved_plan_sha256 is None, "Plan mode cannot accept approved plan")
    payload = build_payload(
        binding_path=args.binding,
        binding_sha256=args.binding_sha256,
        quarantine_root=args.quarantine_root,
        apply=args.apply,
        approved_plan=plan,
    )
    write_exclusive(args.output, payload)
    print(payload["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
