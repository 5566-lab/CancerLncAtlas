#!/usr/bin/env python
"""Independent audit for the V3.2 versioned staging download catalog.

This file intentionally does not import ``cc_hhgt.v32.download_catalog``.
All source hashes and partition-tree hashes are recomputed independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cc_hhgt.v32.capability_parity import load_parity_config


CATALOG_FORMAT = "CANCERLNCATLAS_V32_STAGING_DOWNLOAD_CATALOG_V1"
BINDING_FORMAT = "CANCERLNCATLAS_V32_STAGING_DOWNLOAD_CATALOG_BINDING_V1"
AUDIT_FORMAT = "CANCERLNCATLAS_V32_STAGING_DOWNLOAD_CATALOG_INDEPENDENT_AUDIT_V1"
AUDIT_BINDING_FORMAT = (
    "CANCERLNCATLAS_V32_STAGING_DOWNLOAD_CATALOG_INDEPENDENT_AUDIT_BINDING_V1"
)
MODEL_VERSION = "V3.2"
CATALOG_VERSION = "v3.2-20260827-r7"
ALLOWED_STATUSES = {
    "READY_FILE",
    "READY_PARTS",
    "PARTIAL_READY_PARTS",
    "DYNAMIC_QUERY_EXPORT",
    "PENDING_FORMAL_SUCCESS",
    "GAP_TYPED_UNAVAILABLE",
    "DATA_PRESENT_DOWNLOAD_NOT_IMPLEMENTED",
    "SERVER_HASH_PINNED_DOWNLOAD",
}
EXPECTED_STATUS_COUNTS = {
    "DYNAMIC_QUERY_EXPORT": 4,
    "READY_FILE": 39,
    "READY_PARTS": 8,
    "SERVER_HASH_PINNED_DOWNLOAD": 3,
}
PINNED_AUTHORITIES = {
    "historical_remediation": "b128681bff1856ab09df21c387013b431e0da956bce6f32243b1361661e0c4d8",
    "historical_remediation_audit": "b652fcb0cdecddfd698983322c6b485025a4a656f0b86f7856b52ef20840e3ae",
    "unified_network": "ee5b954f1aa4b162a363efe2fa16296ee6bbfdd6cdb374cc5d4e0243837e7907",
    "unified_network_audit": "290c228b261f8116a8873b77912e2c49ce83a0c44484b6ddad6109108eea185d",
    "gene_set_release": "34d8d59be3a995e86c07e3a0eaee9db63448f90a0ef983acddf2392c9947c725",
    "gene_set_audit": "5ead03cb90d02e9e78417996fa490150c4b7a5005234ef22310818df18e06da7",
    "single_cell_gap_audit": "26e94b0e0a9aa7c217d50494c2b92ac477f4c01fdcb2a2147d61dc89eab76ccf",
    "single_cell_gap_audit_independent": "1e909bc81a2180f1ab301233bf57d77911305120627f6443f7acafce7cbc1784",
    "single_cell_formal_context": "798df5cd58278e93daba3d6b7c20e47a394ce65a6c0ee3df06693c4df3c8cc4c",
    "single_cell_formal_context_independent": "612bfeb3a74384c706d6af98dc871535f462664580ed3dd478be90555fe1410c",
    "evidence_direction": "2577e382bd72e27850b10fa30d7b8af253b338adf7995031c901146618871da1",
    "evidence_direction_audit": "b9dca24cbf30fe0e48bf9791d6f1ca1d12ad7ef767ed844221962a1fbe4e3880",
    "single_cell_ucell_17c_server": "4925433dde88bbb8c93ee07fcc668ae6cc043a135d7b1c1c3a8ace342e265c75",
    "drug_r6_sparse": "2af597e2a8c50430a58664bf55e486bddacfc0b9894ba0029b4fe8c7fb5111ed",
    "drug_mechanism_server": "5ea2c59ef01f43c05fd9e750d08b1c0b193c10b1054dca1c0960efe0d962bf5e",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_SOURCE = re.compile(
    r"(?:^|/)(?:v?2[._-][0-9]+|v?3[._-][01])(?:/|$)|"
    r"(?:^|/)(?:legacy|old)(?:/|$)",
    re.IGNORECASE,
)


class IndependentAuditError(RuntimeError):
    pass


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    if source.is_symlink() or not source.is_file():
        raise IndependentAuditError(f"{label} is missing or symlinked: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndependentAuditError(f"{label} is invalid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise IndependentAuditError(f"{label} must be a JSON object")
    return source, value


def add_check(
    checks: list[dict[str, Any]], name: str, observed: Any, expected: Any
) -> None:
    checks.append(
        {
            "name": name,
            "observed": observed,
            "expected": expected,
            "status": "PASS" if observed == expected else "FAIL",
        }
    )


def safe_relative(value: str) -> bool:
    candidate = PurePosixPath(value.replace("\\", "/"))
    return bool(
        not candidate.is_absolute()
        and candidate.parts
        and all(part not in {"", ".", ".."} for part in candidate.parts)
        and ":" not in candidate.parts[0]
    )


def resolve_source(record: Mapping[str, Any], repo_root: Path) -> Path:
    path = Path(str(record.get("source_path", "")))
    if path.is_symlink():
        raise IndependentAuditError(f"Download source may not be symlinked: {path}")
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise IndependentAuditError(f"Download source escapes repository: {resolved}") from exc
    if relative != record.get("source_repo_relative_path"):
        raise IndependentAuditError(f"Source-relative path drift: {resolved}")
    if _FORBIDDEN_SOURCE.search(relative):
        raise IndependentAuditError(f"Legacy derived source exposed: {relative}")
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise IndependentAuditError(f"Download source is missing or empty: {resolved}")
    if not safe_relative(str(record.get("relative_name", ""))):
        raise IndependentAuditError(
            f"Unsafe public relative name: {record.get('relative_name')!r}"
        )
    expected = str(record.get("sha256", ""))
    if not _SHA256.fullmatch(expected) or sha256(resolved) != expected:
        raise IndependentAuditError(f"Download source SHA256 drift: {resolved}")
    if resolved.stat().st_size != record.get("bytes"):
        raise IndependentAuditError(f"Download source byte-count drift: {resolved}")
    if resolved.suffix.lower() in {".pt", ".pth", ".ckpt", ".npz"}:
        raise IndependentAuditError(f"Private/model checkpoint exposed: {resolved}")
    return resolved


def tree_sha(parts: list[Mapping[str, Any]]) -> str:
    rows = [
        f"{part['relative_name']}\t{part['sha256']}\t{part['bytes']}"
        for part in sorted(parts, key=lambda item: str(item["relative_name"]))
    ]
    return hashlib.sha256(("\n".join(rows) + "\n").encode("utf-8")).hexdigest()


def expected_download_map(contract: Mapping[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for capability_id, capability in contract["capabilities"].items():
        for download_id in capability["gates"]["download"]["ids"]:
            result.setdefault(str(download_id), []).append(str(capability_id))
    return {key: sorted(value) for key, value in result.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--expected-binding-sha256", required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    repository = args.repo_root.resolve()
    binding_path, binding = read_json(args.binding, "download catalog binding")
    expected_binding_sha = str(args.expected_binding_sha256).lower()
    if not _SHA256.fullmatch(expected_binding_sha) or sha256(binding_path) != expected_binding_sha:
        raise IndependentAuditError("Download catalog binding SHA256 drift")
    output = args.output_root.resolve()
    try:
        output.relative_to(repository)
    except ValueError as exc:
        raise IndependentAuditError("Audit output must remain inside repository") from exc
    if output.exists() and any(output.iterdir()):
        raise IndependentAuditError(f"Refusing non-empty audit output reuse: {output}")
    output.mkdir(parents=True, exist_ok=True)

    catalog_declaration = binding.get("catalog")
    if not isinstance(catalog_declaration, Mapping):
        raise IndependentAuditError("Binding lacks a catalog declaration")
    catalog_path, catalog = read_json(catalog_declaration.get("path", ""), "download catalog")
    if (
        sha256(catalog_path) != catalog_declaration.get("sha256")
        or catalog_path.stat().st_size != catalog_declaration.get("bytes")
    ):
        raise IndependentAuditError("Catalog declaration drift")
    try:
        catalog_path.relative_to(repository)
    except ValueError as exc:
        raise IndependentAuditError("Catalog escapes repository") from exc

    checks: list[dict[str, Any]] = []
    add_check(checks, "binding_format", binding.get("format"), BINDING_FORMAT)
    add_check(checks, "binding_model_version", binding.get("model_version"), MODEL_VERSION)
    add_check(
        checks, "binding_catalog_version", binding.get("catalog_version"), CATALOG_VERSION
    )
    add_check(checks, "binding_download_count", binding.get("download_count"), 54)
    add_check(checks, "binding_production_false", binding.get("production_deployed"), False)
    add_check(checks, "binding_release_ready_false", binding.get("release_ready"), False)
    add_check(checks, "catalog_format", catalog.get("format"), CATALOG_FORMAT)
    add_check(checks, "catalog_model_version", catalog.get("model_version"), MODEL_VERSION)
    add_check(checks, "catalog_version", catalog.get("catalog_version"), CATALOG_VERSION)
    add_check(checks, "catalog_environment", catalog.get("environment"), "staging")
    add_check(checks, "catalog_download_count", catalog.get("download_count"), 54)
    add_check(checks, "catalog_production_false", catalog.get("production_deployed"), False)
    add_check(checks, "catalog_release_ready_false", catalog.get("release_ready"), False)
    add_check(checks, "catalog_family_broadcast_false", catalog.get("family_to_exact_broadcast"), False)
    add_check(
        checks,
        "catalog_no_historical_derived_downloads",
        catalog.get("historical_prediction_ranking_or_checkpoint_downloaded"),
        False,
    )

    contract_path = repository / "config/v32_historical_capability_parity.yaml"
    contract = load_parity_config(contract_path)
    expected = expected_download_map(contract)
    add_check(checks, "parity_unique_download_ids", len(expected), 54)
    downloads = catalog.get("downloads")
    if not isinstance(downloads, list):
        raise IndependentAuditError("Catalog downloads must be a list")
    observed = {
        str(value.get("download_id")): value
        for value in downloads
        if isinstance(value, Mapping)
    }
    add_check(checks, "catalog_unique_download_rows", len(observed), len(downloads))
    add_check(checks, "catalog_id_closure", sorted(observed), sorted(expected))

    authorities = catalog.get("authorities")
    if not isinstance(authorities, Mapping):
        raise IndependentAuditError("Catalog authorities must be a mapping")
    for authority_id, record in sorted(authorities.items()):
        if not isinstance(record, Mapping):
            raise IndependentAuditError(f"Invalid authority record: {authority_id}")
        source = Path(str(record.get("path", ""))).resolve()
        try:
            relative = source.relative_to(repository).as_posix()
        except ValueError as exc:
            raise IndependentAuditError(f"Authority escapes repository: {source}") from exc
        valid = (
            not source.is_symlink()
            and source.is_file()
            and source.stat().st_size == record.get("bytes")
            and relative == record.get("repo_relative_path")
            and _SHA256.fullmatch(str(record.get("sha256", ""))) is not None
            and sha256(source) == record.get("sha256")
        )
        add_check(checks, f"authority_{authority_id}_hash_and_path", valid, True)
    for authority_id, digest in PINNED_AUTHORITIES.items():
        record = authorities.get(authority_id)
        add_check(
            checks,
            f"pinned_authority_{authority_id}",
            record.get("sha256") if isinstance(record, Mapping) else None,
            digest,
        )
    diagnostic_record = authorities.get("single_cell_diagnostic_server")
    diagnostic_audit_record = authorities.get(
        "single_cell_diagnostic_server_independent"
    )
    diagnostic = {}
    diagnostic_audit = {}
    if isinstance(diagnostic_record, Mapping):
        _, diagnostic = read_json(
            diagnostic_record.get("path", ""),
            "single-cell diagnostic deployment authority",
        )
    if isinstance(diagnostic_audit_record, Mapping):
        _, diagnostic_audit = read_json(
            diagnostic_audit_record.get("path", ""),
            "single-cell diagnostic independent-audit authority",
        )
    diagnostic_smoke = diagnostic.get("real_query_smoke", {})
    diagnostic_semantics = diagnostic.get("semantics", {})
    add_check(
        checks,
        "single_cell_diagnostic_deployment_contract",
        (
            diagnostic.get("format"),
            diagnostic.get("status"),
            diagnostic.get("cancer_count"),
            diagnostic.get("figure_count"),
            int(diagnostic.get("numeric_pseudotime_rows", 0)) > 0,
            diagnostic.get("production_deployed"),
        ),
        (
            "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_SERVER_DEPLOYMENT_BINDING_V1",
            "SERVER_MOUNT_READY_HASH_PINNED",
            17,
            153,
            True,
            False,
        ),
    )
    add_check(
        checks,
        "single_cell_diagnostic_real_smoke",
        tuple(
            diagnostic_smoke.get(field)
            for field in (
                "status",
                "real_numeric_pathway_query_exercised",
                "real_numeric_cell_query_exercised",
                "real_figure_manifest_exercised",
                "real_figure_file_rehash_exercised",
                "real_download_manifest_exercised",
                "request_time_payload_rehash_exercised",
                "path_traversal_rejected",
            )
        ),
        ("PASS", True, True, True, True, True, True, True),
    )
    add_check(
        checks,
        "single_cell_diagnostic_zero_weight_semantics",
        (
            diagnostic_semantics.get("root_provenance"),
            diagnostic_semantics.get("root_is_explicit"),
            diagnostic_semantics.get("diagnostic_only"),
            diagnostic_semantics.get("model_fusion_permitted"),
            diagnostic_semantics.get("primary_score_weight"),
            diagnostic_semantics.get("secondary_score_weight"),
            diagnostic_semantics.get("historical_outputs_used"),
            diagnostic_semantics.get("changes_primary_ranking"),
        ),
        (
            "INFERRED_CYTOTRACE2_UCELL_CONSENSUS",
            False,
            True,
            False,
            0,
            0,
            False,
            False,
        ),
    )
    add_check(
        checks,
        "single_cell_diagnostic_independent_audit",
        (
            diagnostic_audit.get("format"),
            diagnostic_audit.get("status"),
            diagnostic_audit.get("deployment_binding_sha256"),
            diagnostic_audit.get("failed_checks"),
            diagnostic_audit.get("accepted_for_staging_integration"),
            diagnostic_audit.get("production_deployed"),
        ),
        (
            "CC_HHGT_V3_2_SINGLE_CELL_DIAGNOSTIC_SERVER_INDEPENDENT_AUDIT_BINDING_V1",
            "PASS_HASH_BOUND",
            diagnostic_record.get("sha256")
            if isinstance(diagnostic_record, Mapping)
            else None,
            0,
            True,
            False,
        ),
    )

    status_counts: dict[str, int] = {}
    total_file_records = 0
    total_file_bytes = 0
    seen_public_names: dict[str, str] = {}
    for download_id in sorted(expected):
        entry = observed.get(download_id)
        if not isinstance(entry, Mapping):
            continue
        add_check(
            checks,
            f"{download_id}_capability_closure",
            sorted(entry.get("capability_ids", [])),
            expected[download_id],
        )
        add_check(checks, f"{download_id}_model_version", entry.get("model_version"), MODEL_VERSION)
        add_check(
            checks,
            f"{download_id}_availability_encoding",
            entry.get("availability_encoding"),
            "null_with_reason",
        )
        add_check(
            checks,
            f"{download_id}_unavailable_fill_null",
            entry.get("unavailable_fill_value"),
            None,
        )
        add_check(
            checks,
            f"{download_id}_family_broadcast_false",
            entry.get("family_to_exact_broadcast"),
            False,
        )
        add_check(
            checks,
            f"{download_id}_production_false",
            entry.get("production_deployed"),
            False,
        )
        authority_ids = entry.get("authority_ids")
        authority_ids_valid = isinstance(authority_ids, list) and all(
            authority_id in authorities for authority_id in authority_ids
        )
        add_check(checks, f"{download_id}_authority_closure", authority_ids_valid, True)
        status = str(entry.get("status", ""))
        add_check(checks, f"{download_id}_allowed_status", status in ALLOWED_STATUSES, True)
        status_counts[status] = status_counts.get(status, 0) + 1

        if status == "READY_FILE":
            add_check(checks, f"{download_id}_data_present", entry.get("data_present"), True)
            add_check(
                checks,
                f"{download_id}_download_implemented",
                entry.get("download_implemented"),
                True,
            )
            add_check(
                checks, f"{download_id}_contract_complete", entry.get("contract_complete"), True
            )
            record = entry.get("file")
            if not isinstance(record, Mapping):
                raise IndependentAuditError(f"READY_FILE lacks a file: {download_id}")
            resolve_source(record, repository)
            total_file_records += 1
            total_file_bytes += int(record["bytes"])
            public_name = str(record["relative_name"])
            if public_name in seen_public_names and record["source_path"] != seen_public_names[public_name]:
                raise IndependentAuditError(f"Public name collision: {public_name}")
            seen_public_names[public_name] = str(record["source_path"])
        elif status in {"READY_PARTS", "PARTIAL_READY_PARTS"}:
            parts = entry.get("parts")
            if not isinstance(parts, list) or not parts:
                raise IndependentAuditError(f"Partition entry lacks parts: {download_id}")
            relative_names = [str(part.get("relative_name", "")) for part in parts]
            add_check(
                checks,
                f"{download_id}_part_names_unique",
                len(relative_names),
                len(set(relative_names)),
            )
            for record in parts:
                if not isinstance(record, Mapping):
                    raise IndependentAuditError(f"Invalid part record: {download_id}")
                resolve_source(record, repository)
                total_file_records += 1
                total_file_bytes += int(record["bytes"])
                public_name = f"{download_id}/{record['relative_name']}"
                if public_name in seen_public_names:
                    raise IndependentAuditError(f"Public part collision: {public_name}")
                seen_public_names[public_name] = str(record["source_path"])
            add_check(checks, f"{download_id}_part_count", entry.get("part_count"), len(parts))
            add_check(
                checks,
                f"{download_id}_part_total_bytes",
                entry.get("total_bytes"),
                sum(int(part["bytes"]) for part in parts),
            )
            add_check(
                checks,
                f"{download_id}_tree_sha256",
                entry.get("sha256_tree"),
                tree_sha(parts),
            )
            add_check(checks, f"{download_id}_data_present", entry.get("data_present"), True)
            add_check(
                checks,
                f"{download_id}_download_implemented",
                entry.get("download_implemented"),
                True,
            )
            if status == "PARTIAL_READY_PARTS":
                add_check(
                    checks,
                    f"{download_id}_contract_incomplete",
                    entry.get("contract_complete"),
                    False,
                )
                add_check(
                    checks,
                    f"{download_id}_missing_components_declared",
                    bool(entry.get("missing_components")) and bool(entry.get("known_gap")),
                    True,
                )
            else:
                add_check(
                    checks,
                    f"{download_id}_contract_complete",
                    entry.get("contract_complete"),
                    True,
                )
        elif status == "DYNAMIC_QUERY_EXPORT":
            add_check(checks, f"{download_id}_no_placeholder_file", "file" in entry or "parts" in entry, False)
            add_check(checks, f"{download_id}_data_not_precomputed", entry.get("data_present"), False)
            add_check(
                checks,
                f"{download_id}_dynamic_export_implemented",
                entry.get("download_implemented"),
                True,
            )
            add_check(
                checks,
                f"{download_id}_dynamic_route_v32",
                str(entry.get("export_endpoint", "")).startswith(
                    "POST /v3.2-staging/"
                ),
                True,
            )
        elif status == "SERVER_HASH_PINNED_DOWNLOAD":
            add_check(
                checks,
                f"{download_id}_server_data_present",
                entry.get("data_present"),
                True,
            )
            add_check(
                checks,
                f"{download_id}_server_download_implemented",
                entry.get("download_implemented"),
                True,
            )
            add_check(
                checks,
                f"{download_id}_server_contract_complete",
                entry.get("contract_complete"),
                True,
            )
            add_check(
                checks,
                f"{download_id}_server_manifest_route",
                str(entry.get("manifest_endpoint", "")).startswith(
                    "GET /v3.2-staging/"
                ),
                True,
            )
            add_check(
                checks,
                f"{download_id}_server_download_route",
                str(entry.get("download_endpoint", "")).startswith(
                    "GET /v3.2-staging/"
                ),
                True,
            )
            add_check(
                checks,
                f"{download_id}_server_request_rehash",
                entry.get("request_time_payload_rehash"),
                True,
            )
            add_check(
                checks,
                f"{download_id}_server_no_local_payload_claim",
                "file" in entry or "parts" in entry,
                False,
            )
        elif status in {"PENDING_FORMAL_SUCCESS", "GAP_TYPED_UNAVAILABLE"}:
            add_check(checks, f"{download_id}_not_claimed_implemented", entry.get("download_implemented"), False)
            add_check(checks, f"{download_id}_contract_incomplete", entry.get("contract_complete"), False)
            add_check(checks, f"{download_id}_reason_declared", bool(entry.get("unavailable_reason")), True)
            add_check(checks, f"{download_id}_no_ready_payload", "file" in entry or "parts" in entry, False)
        elif status == "DATA_PRESENT_DOWNLOAD_NOT_IMPLEMENTED":
            add_check(checks, f"{download_id}_data_present_only", entry.get("data_present"), True)
            add_check(checks, f"{download_id}_not_implemented", entry.get("download_implemented"), False)

    add_check(checks, "status_counts_recomputed", dict(sorted(status_counts.items())), EXPECTED_STATUS_COUNTS)
    add_check(checks, "status_counts_catalog", catalog.get("status_counts"), EXPECTED_STATUS_COUNTS)
    add_check(checks, "status_counts_binding", binding.get("status_counts"), EXPECTED_STATUS_COUNTS)
    add_check(checks, "drug_predictions_ready", observed["drug_response_predictions"]["status"], "READY_PARTS")
    add_check(checks, "drug_evidence_ready", observed["drug_evidence"]["status"], "READY_PARTS")
    add_check(checks, "drug_mechanisms_server_ready", observed["drug_mechanisms"]["status"], "SERVER_HASH_PINNED_DOWNLOAD")
    add_check(checks, "pseudotime_server_ready", observed["single_cell_pseudotime"]["status"], "SERVER_HASH_PINNED_DOWNLOAD")
    add_check(checks, "ucell_formal17_server_ready", observed["single_cell_ucell"]["status"], "SERVER_HASH_PINNED_DOWNLOAD")
    evidence_probability = observed["evidence_probabilities"]
    evidence_part_names = {
        str(value.get("relative_name")): value
        for value in evidence_probability.get("parts", [])
        if isinstance(value, Mapping)
    }
    add_check(
        checks,
        "evidence_probabilities_complete",
        (
            evidence_probability.get("status"),
            evidence_probability.get("contract_complete"),
            evidence_probability.get("missing_components"),
        ),
        ("READY_PARTS", True, None),
    )
    add_check(
        checks,
        "evidence_direction_probability_part",
        evidence_part_names.get(
            "evidence_probabilities/evidence_direction_probabilities.parquet", {}
        ).get("sha256"),
        "a2b41290275cb57c339abe76df248a3f3f2f69ef08e8a33f16b83ae8255f40a9",
    )
    add_check(
        checks,
        "evidence_direction_authorities",
        {
            "evidence_direction",
            "evidence_direction_audit",
        }.issubset(set(evidence_probability.get("authority_ids", []))),
        True,
    )
    known_gaps = catalog.get("known_capability_gaps", [])
    known_gap_ids = {
        str(value.get("artifact_id"))
        for value in known_gaps
        if isinstance(value, Mapping)
    }
    add_check(checks, "single_cell_figure_gap_closed", "v32_sc_figure_manifest" in known_gap_ids, False)
    add_check(
        checks,
        "evidence_direction_no_longer_gap",
        "v32_evidence_direction_probability" in known_gap_ids,
        False,
    )
    capability_audit_metadata = catalog.get("capability_audit_metadata")
    single_cell_metadata = (
        capability_audit_metadata.get("single_cell")
        if isinstance(capability_audit_metadata, Mapping)
        else None
    )
    if not isinstance(single_cell_metadata, Mapping):
        single_cell_metadata = {}
    expected_sc_authorities = [
        "single_cell_gap_audit",
        "single_cell_gap_audit_independent",
    ]
    add_check(
        checks,
        "single_cell_audit_metadata_kind",
        single_cell_metadata.get("kind"),
        "FORMAL_AUDIT_METADATA_REFERENCE_NO_NEW_DOWNLOAD_CONTRACT_ID",
    )
    add_check(
        checks,
        "single_cell_audit_no_new_contract_id",
        single_cell_metadata.get("download_contract_id_created"),
        False,
    )
    add_check(
        checks,
        "single_cell_audit_authorities",
        single_cell_metadata.get("authority_ids"),
        expected_sc_authorities,
    )
    add_check(
        checks,
        "single_cell_audit_checks",
        (
            single_cell_metadata.get("independent_audit_checks"),
            single_cell_metadata.get("independent_audit_failed_checks"),
        ),
        (68, 0),
    )
    add_check(
        checks,
        "single_cell_audit_lncrna_counts",
        {
            key: single_cell_metadata.get("lncrna_detection_counts", {}).get(key)
            for key in ("cancers", "cells", "detected_union", "detected_intersection")
        },
        {
            "cancers": 33,
            "cells": 1_919_578,
            "detected_union": 15_879,
            "detected_intersection": 4,
        },
    )
    add_check(
        checks,
        "single_cell_diagnostic_counts",
        (
            int(single_cell_metadata.get("pseudotime_numeric_rows", 0)) > 0,
            single_cell_metadata.get("ucell_formal_cancers_covered"),
            single_cell_metadata.get("ucell_formal_cancers_total"),
            single_cell_metadata.get("current_v32_figure_files"),
        ),
        (True, 17, 17, 153),
    )
    sc_gap_rows = [
        value
        for value in known_gaps
        if isinstance(value, Mapping) and value.get("capability_id") == "single_cell"
    ]
    add_check(
        checks,
        "single_cell_gaps_reference_formal_audit",
        len(sc_gap_rows) == 0,
        True,
    )
    add_check(
        checks,
        "single_cell_download_rows_reference_formal_audit",
        set(expected_sc_authorities).issubset(
            set(observed["single_cell_pseudotime"].get("authority_ids", []))
        )
        and set(expected_sc_authorities).issubset(
            set(observed["single_cell_ucell"].get("authority_ids", []))
        ),
        True,
    )
    expected_context_authorities = [
        "single_cell_formal_context",
        "single_cell_formal_context_independent",
    ]
    add_check(
        checks,
        "single_cell_formal_context_authorities",
        single_cell_metadata.get("formal_context_authority_ids"),
        expected_context_authorities,
    )
    add_check(
        checks,
        "single_cell_formal_context_binding_sha",
        single_cell_metadata.get("formal_context_binding_sha256"),
        PINNED_AUTHORITIES["single_cell_formal_context"],
    )
    add_check(
        checks,
        "single_cell_formal_context_audit_binding_sha",
        single_cell_metadata.get(
            "formal_context_independent_audit_binding_sha256"
        ),
        PINNED_AUTHORITIES["single_cell_formal_context_independent"],
    )
    add_check(
        checks,
        "single_cell_formal_context_audit_report_sha",
        single_cell_metadata.get("formal_context_independent_audit_report_sha256"),
        "d9067bff365d355c06ccaa7c28112234f049848a7616296333f54b46d113f5e6",
    )
    add_check(
        checks,
        "single_cell_formal_context_audit_counts",
        (
            single_cell_metadata.get("formal_context_independent_audit_checks"),
            single_cell_metadata.get("formal_context_independent_audit_failures"),
        ),
        (25, 0),
    )
    add_check(
        checks,
        "single_cell_formal_context_coverage",
        (
            single_cell_metadata.get("raw_h5_cancers_audited"),
            single_cell_metadata.get("formal_context_cancers_available"),
            single_cell_metadata.get("formal_context_cancers_total"),
        ),
        (33, 17, 33),
    )
    add_check(
        checks,
        "single_cell_formal_context_ucell_scope",
        (
            single_cell_metadata.get("cell_level_ucell_formal_cancers_covered"),
            single_cell_metadata.get("cell_level_ucell_formal_cancers_total"),
            single_cell_metadata.get("cell_level_ucell_remaining_cancers"),
        ),
        (17, 17, 0),
    )
    add_check(
        checks,
        "single_cell_association_context_authorities",
        set(expected_context_authorities).issubset(
            set(observed["single_cell_associations"].get("authority_ids", []))
        ),
        True,
    )
    add_check(
        checks,
        "single_cell_activity_context_authorities",
        set(expected_context_authorities).issubset(
            set(observed["single_cell_activity"].get("authority_ids", []))
        ),
        True,
    )
    association_parts = {
        str(value.get("relative_name")): value
        for value in observed["single_cell_associations"].get("parts", [])
        if isinstance(value, Mapping)
    }
    add_check(
        checks,
        "single_cell_association_part_hashes",
        {
            name: association_parts.get(name, {}).get("sha256")
            for name in (
                "single_cell_associations/lncrna_celltype.parquet",
                "single_cell_associations/lncrna_exact_pathway.parquet",
                "single_cell_associations/celltype_typed_predictions.parquet",
            )
        },
        {
            "single_cell_associations/lncrna_celltype.parquet": "28d131e3b67432d7cb8fadffa2dc35ba7d899315c521a0e83c89c190ebc9317a",
            "single_cell_associations/lncrna_exact_pathway.parquet": "83e25646ef604b9d1a76f5872c73421ed0041c2fe097549e7c8838848ede7112",
            "single_cell_associations/celltype_typed_predictions.parquet": "228ae23cf8b5d320a66ab2893d66cd017671bcd804d7a17b109da7c548bfd65c",
        },
    )
    add_check(
        checks,
        "single_cell_formal_context_artifact_hashes",
        {
            key: single_cell_metadata.get("formal_context_artifacts", {})
            .get(key, {})
            .get("sha256")
            for key in (
                "lncrna_celltype",
                "pathway_activity",
                "typed_predictions",
                "exact_association",
            )
        },
        {
            "lncrna_celltype": "28d131e3b67432d7cb8fadffa2dc35ba7d899315c521a0e83c89c190ebc9317a",
            "pathway_activity": "fcd1a2252c181841b8b51de016d65fac2599b697d051ea4eac4649e0a4ed10f1",
            "typed_predictions": "228ae23cf8b5d320a66ab2893d66cd017671bcd804d7a17b109da7c548bfd65c",
            "exact_association": "83e25646ef604b9d1a76f5872c73421ed0041c2fe097549e7c8838848ede7112",
        },
    )
    add_check(
        checks,
        "single_cell_formal_context_score_policy",
        (
            single_cell_metadata.get("typed_predictions_downloaded"),
            single_cell_metadata.get("single_cell_fusion_weight_zero"),
            single_cell_metadata.get("exact_primary_modified"),
        ),
        (True, True, False),
    )
    add_check(
        checks,
        "single_cell_formal_context_download_ids",
        single_cell_metadata.get("formal_context_download_ids"),
        ["single_cell_associations", "single_cell_activity"],
    )
    add_check(checks, "file_records_positive", total_file_records > 0, True)
    add_check(checks, "download_bytes_positive", total_file_bytes > 0, True)

    dataset_declaration = binding.get("dataset_catalog")
    if not isinstance(dataset_declaration, Mapping):
        raise IndependentAuditError("Binding lacks dataset catalog")
    dataset_path, dataset_catalog = read_json(
        dataset_declaration.get("path", ""), "dataset catalog"
    )
    add_check(checks, "dataset_catalog_hash", sha256(dataset_path), dataset_declaration.get("sha256"))
    add_check(checks, "dataset_catalog_bytes", dataset_path.stat().st_size, dataset_declaration.get("bytes"))
    add_check(checks, "dataset_catalog_count", dataset_catalog.get("download_count"), 54)
    add_check(checks, "dataset_catalog_production_false", dataset_catalog.get("production_deployed"), False)
    add_check(checks, "dataset_catalog_release_ready_false", dataset_catalog.get("release_ready"), False)
    dataset_ids = sorted(
        str(value.get("download_id"))
        for value in dataset_catalog.get("datasets", [])
        if isinstance(value, Mapping)
    )
    add_check(checks, "dataset_catalog_id_closure", dataset_ids, sorted(expected))

    fail_count = sum(check["status"] == "FAIL" for check in checks)
    pass_count = len(checks) - fail_count
    report = {
        "format": AUDIT_FORMAT,
        "status": "PASS" if fail_count == 0 else "FAIL",
        "model_version": MODEL_VERSION,
        "binding": {"path": str(binding_path), "sha256": expected_binding_sha},
        "catalog": {"path": str(catalog_path), "sha256": sha256(catalog_path)},
        "independent_of_materializer_implementation": True,
        "materializer_imported": False,
        "all_file_hashes_recomputed": True,
        "all_partition_tree_hashes_recomputed": True,
        "checked_download_ids": sorted(expected),
        "download_count": len(expected),
        "file_records_rehashed": total_file_records,
        "bytes_rehashed": total_file_bytes,
        "checks": checks,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "accepted_for_staging_api_integration": fail_count == 0,
        "production_deployed": False,
        "release_ready": False,
    }
    report_path = output / "DOWNLOAD_CATALOG_INDEPENDENT_AUDIT_REPORT.json"
    atomic_json(report_path, report)
    audit_binding = {
        "format": AUDIT_BINDING_FORMAT,
        "status": report["status"],
        "model_version": MODEL_VERSION,
        "release_binding": report["binding"],
        "catalog": report["catalog"],
        "report": {
            "path": str(report_path),
            "sha256": sha256(report_path),
            "bytes": report_path.stat().st_size,
        },
        "download_count": len(expected),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "accepted_for_staging_api_integration": fail_count == 0,
        "independent_of_materializer_implementation": True,
        "materializer_imported": False,
        "production_deployed": False,
        "release_ready": False,
    }
    audit_binding_path = output / "DOWNLOAD_CATALOG_INDEPENDENT_AUDIT_BINDING.json"
    atomic_json(audit_binding_path, audit_binding)
    atomic_json(
        output / "SUCCESS.json",
        {
            "status": audit_binding["status"],
            "binding": audit_binding_path.name,
            "binding_sha256": sha256(audit_binding_path),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "production_deployed": False,
            "release_ready": False,
        },
    )
    print(json.dumps(audit_binding, ensure_ascii=False, indent=2, sort_keys=True))
    if fail_count:
        raise IndependentAuditError(
            f"Independent download catalog audit failed {fail_count}/{len(checks)} checks"
        )


if __name__ == "__main__":
    main()
