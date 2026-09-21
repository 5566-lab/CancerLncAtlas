#!/usr/bin/env python3
"""Full strict validation plus exact typed-result probes for one Drug bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from cc_hhgt.v32.drug_sparse_query import (
    CURRENT_V32_CORE_UNAVAILABLE,
    MODEL_OUTPUT_MISSING_FAIL_CLOSED,
    MODEL_RUN_AUDITED_UNAVAILABLE,
    NO_HELD_OUT_NATIVE_ASSAY,
    NO_MATCHED_LNCRNA_EXPRESSION,
    OUTSIDE_CONCEPTUAL_UNIVERSE,
    STRICT_VALIDATION_STRATEGY,
    DUCKDB_MAX_TEMP_ENV,
    DUCKDB_MEMORY_ENV,
    DUCKDB_TEMP_ENV,
    DUCKDB_THREADS_ENV,
    _available_partition_groups,
    load_drug_sparse_query_bundle,
)


PROBE_FORMAT = "CANCERLNCATLAS_V32_DRUG_TYPED_PROBE_CONTRACT_V1"
SMOKE_FORMAT = "CANCERLNCATLAS_V32_AUTHORIZED_DRUG_RUNTIME_SMOKE_V2"
CORE_REACHABILITY_FORMAT = (
    "CANCERLNCATLAS_V32_DRUG_CORE_UNAVAILABLE_REACHABILITY_AUDIT_V1"
)
CORE_REACHABILITY_STATUS = "UNREACHABLE_IN_FORMAL_FACTORS"
SUCCESS_PROBE_OUTCOMES: dict[str, tuple[bool, str | None]] = {
    "AVAILABLE": (True, None),
    OUTSIDE_CONCEPTUAL_UNIVERSE: (False, OUTSIDE_CONCEPTUAL_UNIVERSE),
    NO_HELD_OUT_NATIVE_ASSAY: (False, NO_HELD_OUT_NATIVE_ASSAY),
    NO_MATCHED_LNCRNA_EXPRESSION: (False, NO_MATCHED_LNCRNA_EXPRESSION),
    MODEL_OUTPUT_MISSING_FAIL_CLOSED: (False, MODEL_OUTPUT_MISSING_FAIL_CLOSED),
}
MODEL_RUN_REASON_NOT_APPLICABLE = (
    "NOT_APPLICABLE_TO_A_SUCCESS_BUNDLE;REQUIRES_A_SEPARATE_"
    "AUDITED_UNAVAILABLE_MODEL_RUN"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DrugSmokeError(RuntimeError):
    """The full bundle or an exact typed-result probe failed closed."""


@contextmanager
def dedicated_duckdb_runtime_environment(
    spill_parent: Path,
    *,
    memory_limit: str,
    threads: str,
    max_temp_size: str,
) -> Iterator[Path]:
    """Give every validator connection a private, self-cleaning spill tree.

    The caller supplies only a parent directory. A unique child is created
    for this process, exported as the module spill base, and removed on both
    success and exception. Existing paths are never recursively removed.
    """

    parent = spill_parent.resolve()
    _require(not parent.is_symlink(), "DuckDB spill parent cannot be a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    _require(parent.is_dir(), "DuckDB spill parent is not a directory")
    names = (
        DUCKDB_MEMORY_ENV,
        DUCKDB_THREADS_ENV,
        DUCKDB_MAX_TEMP_ENV,
        DUCKDB_TEMP_ENV,
    )
    previous = {name: os.environ.get(name) for name in names}
    with tempfile.TemporaryDirectory(
        prefix="v32_drug_partitioned_", dir=str(parent)
    ) as run_directory:
        run_path = Path(run_directory).resolve()
        os.environ[DUCKDB_MEMORY_ENV] = str(memory_limit)
        os.environ[DUCKDB_THREADS_ENV] = str(threads)
        os.environ[DUCKDB_MAX_TEMP_ENV] = str(max_temp_size)
        os.environ[DUCKDB_TEMP_ENV] = str(run_path)
        try:
            yield run_path
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DrugSmokeError(message)


def load_probe_contract(
    path: Path,
    expected_sha256: str,
    manifest_sha256: str,
    artifact_root: Path,
) -> dict[str, Any]:
    expected = str(expected_sha256).lower()
    _require(
        _SHA256.fullmatch(expected) is not None,
        "Expected probe-contract SHA256 is invalid",
    )
    _require(
        not path.is_symlink() and path.is_file(),
        "Drug probe contract is missing or a symlink",
    )
    observed = sha256_file(path)
    _require(
        observed == expected,
        f"Drug probe-contract SHA256 mismatch: {observed} != {expected}",
    )
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DrugSmokeError("Drug probe contract is invalid JSON") from exc
    _require(isinstance(contract, dict), "Drug probe contract must be a JSON object")
    _require(contract.get("format") == PROBE_FORMAT, "Drug probe-contract format drifted")
    _require(
        contract.get("manifest_sha256") == manifest_sha256,
        "Drug probe contract targets another manifest",
    )
    _require(
        contract.get("artifact_root") == str(artifact_root.resolve(strict=True)),
        "Drug probe contract targets another artifact root",
    )
    _require(
        _SHA256.fullmatch(str(contract.get("authorized_binding_sha256", "")))
        is not None,
        "Drug probe contract has an invalid authorized-binding SHA256",
    )
    _require(
        contract.get("model_run_status") == "SUCCESS",
        "Drug probe contract is not for a SUCCESS bundle",
    )
    _require(
        contract.get("semantic_sampling") is False,
        "Drug probe contract claims sampling",
    )
    _require(
        contract.get("non_applicable_typed_absence")
        == {MODEL_RUN_AUDITED_UNAVAILABLE: MODEL_RUN_REASON_NOT_APPLICABLE},
        "Drug probe contract misstates the model-run-level unavailable reason",
    )
    probes = contract.get("probes")
    _require(isinstance(probes, dict), "Drug probe contract lacks probes")
    _require(
        set(probes) == set(SUCCESS_PROBE_OUTCOMES),
        "Drug probe outcome set is incomplete or has extras",
    )
    seen: set[tuple[str, str, str]] = set()
    for outcome, (availability, reason) in SUCCESS_PROBE_OUTCOMES.items():
        probe = probes[outcome]
        _require(isinstance(probe, dict), f"Drug probe is malformed: {outcome}")
        key = tuple(
            str(probe.get(name, "")).strip()
            for name in ("cancer_id", "lncrna_id", "drug_id")
        )
        _require(all(key), f"Drug probe has an empty key: {outcome}")
        _require(key not in seen, f"Drug probe key is duplicated: {outcome}")
        seen.add(key)
        _require(
            probe.get("expected_availability") is availability,
            f"Drug probe availability drifted: {outcome}",
        )
        _require(
            probe.get("expected_failure_reason") == reason,
            f"Drug probe reason drifted: {outcome}",
        )
    load_core_unreachable_receipt(contract)
    return contract


def load_core_unreachable_receipt(
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    unreachable = contract.get("exhaustive_unreachable_typed_absence")
    _require(
        isinstance(unreachable, dict)
        and set(unreachable) == {CURRENT_V32_CORE_UNAVAILABLE},
        "Drug core-unavailable reachability contract is missing or has extras",
    )
    declaration = unreachable[CURRENT_V32_CORE_UNAVAILABLE]
    _require(isinstance(declaration, dict), "Drug core-unavailable declaration is malformed")
    _require(
        declaration.get("status") == CORE_REACHABILITY_STATUS,
        "Drug core-unavailable status drifted",
    )
    receipt_ref = declaration.get("receipt")
    _require(isinstance(receipt_ref, dict), "Drug core-unavailable receipt reference is missing")
    receipt_path = Path(str(receipt_ref.get("path", ""))).resolve()
    receipt_sha = str(receipt_ref.get("sha256", "")).lower()
    _require(_SHA256.fullmatch(receipt_sha) is not None, "Drug core receipt SHA256 is invalid")
    _require(
        receipt_path.is_file() and not receipt_path.is_symlink(),
        "Drug core receipt is missing or symlinked",
    )
    _require(
        sha256_file(receipt_path) == receipt_sha,
        "Drug core receipt SHA256 mismatch",
    )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DrugSmokeError("Drug core receipt is invalid JSON") from exc
    _require(isinstance(receipt, dict), "Drug core receipt must be a JSON object")
    exhaustive = receipt.get("exhaustive_audit", {})
    resource = receipt.get("resource_gate", {})
    _require(
        receipt.get("format") == CORE_REACHABILITY_FORMAT
        and receipt.get("status") == "PASS"
        and receipt.get("failure_reason") == CURRENT_V32_CORE_UNAVAILABLE
        and receipt.get("reachability_status") == CORE_REACHABILITY_STATUS
        and receipt.get("formal_manifest", {}).get("sha256")
        == contract.get("manifest_sha256")
        and receipt.get("authorized_binding", {}).get("sha256")
        == contract.get("authorized_binding_sha256")
        and exhaustive.get("semantic_sampling") is False
        and exhaustive.get("duckdb_used") is False
        and exhaustive.get("exact_rows_examined") == 3_300_000
        and exhaustive.get(
            "conceptual_nonempty_assay_intersect_expression_combinations_examined"
        )
        == 44_344_518
        and exhaustive.get("all_eligible_folds_missing_at_least_one_core_type")
        == 0
        and exhaustive.get("reachable_current_v32_core_unavailable_keys") == 0
        and resource.get("passed") is True
        and 0 < int(resource.get("kernel_peak_rss_kB") or 0)
        <= int(resource.get("max_peak_rss_kB") or 0)
        <= 2 * 1024 * 1024
        and receipt.get("api_defensive_enum_retained") is True
        and receipt.get("changes_primary_ranking") is False
        and receipt.get("production_deployed") is False
        and receipt.get("release_ready") is False,
        "Drug core-unavailable exhaustive receipt contract failed",
    )
    return receipt


def validate_exact_probes(
    bundle: Any, contract: Mapping[str, Any]
) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for outcome, (
        expected_availability,
        expected_reason,
    ) in SUCCESS_PROBE_OUTCOMES.items():
        probe = contract["probes"][outcome]
        result = bundle.resolve(
            probe["cancer_id"], probe["lncrna_id"], probe["drug_id"]
        )
        _require(
            result.get("availability") is expected_availability,
            f"Drug exact probe availability failed: {outcome}",
        )
        _require(
            result.get("failure_reason") == expected_reason,
            f"Drug exact probe reason failed: {outcome}",
        )
        probability = result.get("drug_response_association_probability")
        if expected_availability:
            _require(
                isinstance(probability, (int, float))
                and not isinstance(probability, bool)
                and math.isfinite(float(probability))
                and 0 <= float(probability) <= 1,
                "Available Drug exact probe has an invalid probability",
            )
        else:
            _require(
                probability is None,
                f"Absent Drug exact probe has a probability: {outcome}",
            )
        _require(
            result.get("scientific_status") == "diagnostic_only",
            "Drug status was promoted",
        )
        _require(
            result.get("tcga_patient_response_claimed") is False,
            "Drug result claims TCGA response",
        )
        observed[outcome] = result
    return observed


def run_smoke(
    manifest_path: Path,
    expected_manifest_sha256: str,
    artifact_root: Path,
    probe_path: Path,
    expected_probe_sha256: str,
) -> dict[str, Any]:
    manifest_sha = str(expected_manifest_sha256).lower()
    _require(
        _SHA256.fullmatch(manifest_sha) is not None,
        "Expected Drug manifest SHA256 is invalid",
    )
    contract = load_probe_contract(
        probe_path,
        expected_probe_sha256,
        manifest_sha,
        artifact_root,
    )
    bundle = load_drug_sparse_query_bundle(
        manifest_path,
        expected_manifest_sha256=manifest_sha,
        artifact_root=artifact_root,
    )
    _require(
        bundle.manifest.get("model_run_status") == "SUCCESS",
        "Drug bundle is not SUCCESS",
    )
    groups = _available_partition_groups(
        bundle.paths["available_predictions"]
    )
    physical_parts = {item for files in groups.values() for item in files}
    _require(
        len(physical_parts) == 694,
        "Formal Drug runtime requires all 694 physical partitions, "
        f"got {len(physical_parts)}",
    )
    probes = validate_exact_probes(bundle, contract)
    return {
        "format": SMOKE_FORMAT,
        "status": "PASS",
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "artifact_root": str(artifact_root.resolve(strict=True)),
        "probe_contract": {
            "path": str(probe_path),
            "sha256": sha256_file(probe_path),
            "format": contract["format"],
        },
        "exact_probes": probes,
        "typed_absence_coverage": {
            "reachable_success_bundle_reasons_exactly_probed": sorted(
                reason
                for reason in SUCCESS_PROBE_OUTCOMES
                if reason != "AVAILABLE"
            ),
            "exhaustively_proven_unreachable": {
                CURRENT_V32_CORE_UNAVAILABLE: {
                    "status": CORE_REACHABILITY_STATUS,
                    "receipt": contract["exhaustive_unreachable_typed_absence"][
                        CURRENT_V32_CORE_UNAVAILABLE
                    ]["receipt"],
                    "exact_rows_examined": 3_300_000,
                    "conceptual_nonempty_assay_intersect_expression_combinations_examined": 44_344_518,
                    "reachable_keys": 0,
                }
            },
            "model_run_level_reason": {
                "reason": MODEL_RUN_AUDITED_UNAVAILABLE,
                "status": MODEL_RUN_REASON_NOT_APPLICABLE,
            },
            "all_reachable_success_bundle_reasons_exactly_probed": True,
            "all_success_bundle_reasons_accounted_for": True,
        },
        "strict_validation": {
            "strategy": STRICT_VALIDATION_STRATEGY,
            "semantic_sampling": False,
            "physical_partitions_validated": len(physical_parts),
            "logical_cancer_partitions_validated": len(groups),
            "cancers_validated": len(groups),
            "available_rows_validated": bundle.manifest["available_rows"],
            "conceptual_candidate_rows_validated": bundle.manifest[
                "conceptual_candidate_rows"
            ],
            "five_fold_support_validated": True,
            "cross_file_key_uniqueness_validated_per_cancer": True,
            "all_artifact_hashes_and_rows_validated": True,
        },
        "scientific_status": "diagnostic_only",
        "evidence_scope": "CELL_LINE_ASSOCIATION_NOT_TCGA_PATIENT_RESPONSE",
        "changes_primary_ranking": False,
        "production_port_8260_touched": False,
        "production_deployed": False,
        "release_ready": False,
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as stream:
        stream.write(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--probes-sha256", required=True)
    parser.add_argument("--temp-directory", required=True)
    parser.add_argument("--duckdb-memory-limit", default="512MB")
    parser.add_argument("--duckdb-threads", default="1")
    parser.add_argument("--duckdb-max-temp-size", default="4GB")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite smoke receipt: {output}")
    run_directory: Path | None = None
    spill_parent = Path(args.temp_directory).resolve()
    resource_controls = {
        "duckdb_memory_limit": str(args.duckdb_memory_limit),
        "duckdb_threads": str(args.duckdb_threads),
        "duckdb_max_temp_directory_size": str(args.duckdb_max_temp_size),
        "spill_parent": str(spill_parent),
    }
    try:
        with dedicated_duckdb_runtime_environment(
            spill_parent,
            memory_limit=args.duckdb_memory_limit,
            threads=args.duckdb_threads,
            max_temp_size=args.duckdb_max_temp_size,
        ) as run_directory:
            payload = run_smoke(
                Path(args.manifest).resolve(),
                args.sha256,
                Path(args.artifact_root).resolve(),
                Path(args.probes).resolve(),
                args.probes_sha256,
            )
    except Exception as exc:
        cleanup_verified = run_directory is None or not run_directory.exists()
        payload = {
            "format": SMOKE_FORMAT,
            "status": "FAIL",
            "manifest": {
                "path": str(Path(args.manifest).resolve()),
                "expected_sha256": str(args.sha256).lower(),
            },
            "artifact_root": str(Path(args.artifact_root).resolve()),
            "probe_contract": {
                "path": str(Path(args.probes).resolve()),
                "expected_sha256": str(args.probes_sha256).lower(),
            },
            "failure_type": type(exc).__name__,
            "failure": str(exc),
            "runtime_resource_controls": {
                **resource_controls,
                "run_directory": (
                    None if run_directory is None else str(run_directory)
                ),
                "spill_cleanup_verified": cleanup_verified,
            },
            "scientific_status": "diagnostic_only",
            "changes_primary_ranking": False,
            "production_port_8260_touched": False,
            "production_deployed": False,
            "release_ready": False,
        }
        _write(output, payload)
        print("FAIL")
        return 1
    cleanup_verified = run_directory is not None and not run_directory.exists()
    _require(cleanup_verified, "Dedicated DuckDB spill tree was not removed")
    payload["runtime_resource_controls"] = {
        **resource_controls,
        "run_directory": str(run_directory),
        "spill_cleanup_verified": True,
    }
    _write(output, payload)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
