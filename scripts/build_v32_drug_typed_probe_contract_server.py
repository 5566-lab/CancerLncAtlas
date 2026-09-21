#!/usr/bin/env python3
"""Build the SHA-pinnable formal Drug typed-probe contract, fail closed."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from cc_hhgt.v32.drug_sparse_query import load_drug_sparse_query_bundle
from smoke_v32_authorized_drug_partitioned_server import (
    CORE_REACHABILITY_STATUS,
    MODEL_RUN_AUDITED_UNAVAILABLE,
    MODEL_RUN_REASON_NOT_APPLICABLE,
    PROBE_FORMAT,
    SUCCESS_PROBE_OUTCOMES,
    dedicated_duckdb_runtime_environment,
    load_core_unreachable_receipt,
    validate_exact_probes,
)


ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"
BINDING_FORMAT = "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_AUTHORIZED_BINDING_V1"
AGGREGATE_AUDIT_FORMAT = (
    "CANCERLNCATLAS_V32_DRUG_AVAILABLE_PREDICTIONS_ARTIFACT_AUDIT_V1"
)
QUARANTINE_FORMAT = (
    "CANCERLNCATLAS_V32_DRUG_STALE_UPLOAD_PARTIAL_QUARANTINE_V1"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FORMAL_PROBES: dict[str, dict[str, Any]] = {
    "AVAILABLE": {
        "cancer_id": "BLCA",
        "lncrna_id": "LNC:ENSG00000099869",
        "drug_id": "DRUG:02ebaf6144107cac",
        "expected_availability": True,
        "expected_failure_reason": None,
    },
    "OUTSIDE_CONCEPTUAL_UNIVERSE": {
        "cancer_id": "BLCA",
        "lncrna_id": "LNC:ENSG00000099869",
        "drug_id": "STRICT_AUDIT_UNKNOWN_DRUG",
        "expected_availability": False,
        "expected_failure_reason": "OUTSIDE_CONCEPTUAL_UNIVERSE",
    },
    "NO_HELD_OUT_NATIVE_ASSAY": {
        "cancer_id": "ACC",
        "lncrna_id": "LNC:ENSG00000082929",
        "drug_id": "DRUG:04bb36f9bebe5caf",
        "expected_availability": False,
        "expected_failure_reason": "NO_HELD_OUT_NATIVE_ASSAY",
    },
    "NO_MATCHED_LNCRNA_EXPRESSION": {
        "cancer_id": "BLCA",
        "lncrna_id": "LNC:ENSG00000179066",
        "drug_id": "DRUG:060ec0763845e3fe",
        "expected_availability": False,
        "expected_failure_reason": "NO_MATCHED_LNCRNA_EXPRESSION",
    },
    "MODEL_OUTPUT_MISSING_FAIL_CLOSED": {
        "cancer_id": "BLCA",
        "lncrna_id": "LNC:ENSG00000223534",
        "drug_id": "DRUG:02ebaf6144107cac",
        "expected_availability": False,
        "expected_failure_reason": "MODEL_OUTPUT_MISSING_FAIL_CLOSED",
    },
}


class ProbeBuildError(RuntimeError):
    """The authorized metadata or an exact formal probe failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProbeBuildError(message)


def _load_pinned(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    expected = str(expected_sha256).lower()
    _require(SHA256.fullmatch(expected) is not None, f"{role} SHA256 is invalid")
    _require(path.is_file() and not path.is_symlink(), f"{role} is missing or symlinked")
    _require(sha256_file(path) == expected, f"{role} SHA256 mismatch")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeBuildError(f"{role} is invalid JSON") from exc
    _require(isinstance(payload, dict), f"{role} must be a JSON object")
    return payload


def build_contract(
    manifest_path: Path,
    manifest_sha256: str,
    artifact_root: Path,
    binding_path: Path,
    binding_sha256: str,
    core_receipt_path: Path,
    core_receipt_sha256: str,
    aggregate_audit_path: Path,
    aggregate_audit_sha256: str,
    quarantine_receipt_path: Path,
    quarantine_receipt_sha256: str,
) -> dict[str, Any]:
    binding_path = binding_path.resolve(strict=True)
    manifest_path = manifest_path.resolve(strict=True)
    artifact_root = artifact_root.resolve(strict=True)
    binding = _load_pinned(binding_path, binding_sha256, "authorized binding")
    aggregate_audit = _load_pinned(
        aggregate_audit_path.resolve(strict=True),
        aggregate_audit_sha256,
        "available-predictions aggregate audit",
    )
    quarantine_receipt = _load_pinned(
        quarantine_receipt_path.resolve(strict=True),
        quarantine_receipt_sha256,
        "stale-upload quarantine receipt",
    )
    _require(binding.get("format") == BINDING_FORMAT, "Binding format drifted")
    _require(binding.get("analysis_version") == ANALYSIS_VERSION, "Binding analysis drifted")
    _require(binding.get("production_deployed") is False, "Binding claims deployment")
    _require(binding.get("release_ready") is False, "Binding claims release readiness")
    _require(
        aggregate_audit.get("format") == AGGREGATE_AUDIT_FORMAT
        and aggregate_audit.get("status") == "PASS"
        and aggregate_audit.get("artifact_root") == str(artifact_root)
        and aggregate_audit.get("observed", {}).get("loader_tree_sha256")
        == aggregate_audit.get("declared", {}).get("sha256"),
        "Available-predictions aggregate audit is not closed for this root",
    )
    _require(
        quarantine_receipt.get("format") == QUARANTINE_FORMAT
        and quarantine_receipt.get("status") == "PASS"
        and quarantine_receipt.get("artifact_root") == str(artifact_root)
        and quarantine_receipt.get("post_operation", {}).get("unexpected_files") == 0,
        "Stale-upload quarantine is not closed for this root",
    )
    snapshots = binding.get("components", {}).get("drug", {}).get(
        "binding_snapshots", []
    )
    matching = [
        item
        for item in snapshots
        if Path(str(item.get("published_snapshot_path", ""))).name
        == "DRUG_SPARSE_QUERY_MANIFEST.json"
        and item.get("sha256") == str(manifest_sha256).lower()
    ]
    _require(len(matching) == 1, "Drug manifest SHA is not authorized by binding")
    _require(
        set(FORMAL_PROBES) == set(SUCCESS_PROBE_OUTCOMES),
        "Formal reachable probe set drifted from runtime contract",
    )
    contract = {
        "format": PROBE_FORMAT,
        "analysis_version": ANALYSIS_VERSION,
        "manifest_sha256": str(manifest_sha256).lower(),
        "artifact_root": str(artifact_root),
        "authorized_binding_sha256": str(binding_sha256).lower(),
        "artifact_hygiene": {
            "available_predictions_aggregate_audit": {
                "path": str(aggregate_audit_path.resolve()),
                "sha256": str(aggregate_audit_sha256).lower(),
            },
            "stale_upload_quarantine_receipt": {
                "path": str(quarantine_receipt_path.resolve()),
                "sha256": str(quarantine_receipt_sha256).lower(),
            },
            "unexpected_files_after_quarantine": 0,
        },
        "model_run_status": "SUCCESS",
        "semantic_sampling": False,
        "probe_selection": (
            "FIXED_KEYS_FROM_FULL_LOCAL_FORMAL_MIRROR_THEN_EXACTLY_"
            "REVALIDATED_AGAINST_PINNED_SERVER_MANIFEST"
        ),
        "non_applicable_typed_absence": {
            MODEL_RUN_AUDITED_UNAVAILABLE: MODEL_RUN_REASON_NOT_APPLICABLE
        },
        "exhaustive_unreachable_typed_absence": {
            "CURRENT_V32_CORE_UNAVAILABLE": {
                "status": CORE_REACHABILITY_STATUS,
                "receipt": {
                    "path": str(core_receipt_path.resolve(strict=True)),
                    "sha256": str(core_receipt_sha256).lower(),
                },
            }
        },
        "probes": FORMAL_PROBES,
    }
    core_receipt = load_core_unreachable_receipt(contract)
    _require(
        core_receipt.get("artifact_root_authority", {}).get("path")
        == str(artifact_root),
        "Drug artifact root is not the exhaustively audited root",
    )
    bundle = load_drug_sparse_query_bundle(
        manifest_path,
        expected_manifest_sha256=str(manifest_sha256).lower(),
        artifact_root=artifact_root,
    )
    validate_exact_probes(bundle, contract)
    return contract


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ProbeBuildError(f"Refusing to overwrite probe contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--core-receipt", required=True)
    parser.add_argument("--core-receipt-sha256", required=True)
    parser.add_argument("--aggregate-audit", required=True)
    parser.add_argument("--aggregate-audit-sha256", required=True)
    parser.add_argument("--quarantine-receipt", required=True)
    parser.add_argument("--quarantine-receipt-sha256", required=True)
    parser.add_argument("--temp-directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise ProbeBuildError(f"Refusing to overwrite probe contract: {output}")
    with dedicated_duckdb_runtime_environment(
        Path(args.temp_directory),
        memory_limit="512MB",
        threads="1",
        max_temp_size="4GB",
    ):
        payload = build_contract(
            Path(args.manifest),
            args.manifest_sha256,
            Path(args.artifact_root),
            Path(args.binding),
            args.binding_sha256,
            Path(args.core_receipt),
            args.core_receipt_sha256,
            Path(args.aggregate_audit),
            args.aggregate_audit_sha256,
            Path(args.quarantine_receipt),
            args.quarantine_receipt_sha256,
        )
    _write(output, payload)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
