#!/usr/bin/env python3
"""Build an isolated, server-149-only V3.2 r8 website staging candidate.

The script copies the r6 runtime tree and only the seven formally mounted
sidecars selected for r8.  Files on ${PRIVATE_WORK_ROOT} are hard-linked when possible;
cross-filesystem CNV files are copied.  Every path-bearing binding is rebound
to the new candidate root and all binding hashes are recomputed.  No source
directory is modified and no GPU/training process is started.

This is deliberately a server-side script.  It is uploaded to 149 and run
there by the coordination agent; the local checkout is only the auditable
source copy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import shutil
import stat
import sys
from typing import Any, Callable, Iterable


HOST = "149"
R6_ROOT = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260905_r6")
R7_ROOT = Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r7")
R8_ROOT = Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r8")
R6_CODE = R6_ROOT / "code"
R7_MANIFEST = R7_ROOT / "code/config/v32_server_unified_staging_bindings_20260905_r7.json"
R7_CATALOG = R7_ROOT / "catalog_candidate_20260905_r7/v32-capability-catalog.json"
# R7's scope was stored under ``scope/`` (the filename retains the r6
# revision); it is copied only as immutable bootstrap/history.
R7_SCOPE = R7_ROOT / "scope/capability_scope_20260905_r6.json"


class BuildError(RuntimeError):
    pass


class CopyLedger:
    def __init__(self) -> None:
        self.files = 0
        self.bytes = 0
        self.hardlinks = 0
        self.copies = 0
        self.skipped_pycache = 0
        self.records: list[dict[str, Any]] = []

    def add(self, src: Path, dst: Path, mode: str) -> None:
        size = src.stat().st_size
        self.files += 1
        self.bytes += size
        if mode == "hardlink":
            self.hardlinks += 1
        else:
            self.copies += 1
        self.records.append(
            {"source": str(src), "destination": str(dst), "bytes": size, "mode": mode}
        )


LEDGER = CopyLedger()
# A previous bounded invocation may have created the new r8 root and stopped
# while copying the 145 MB code tree.  Resume is allowed only when the marker
# inside that exact root proves it is our own partial build; existing files
# are hash-checked and never overwritten.
RESUME_MODE = False
REBOUND_SIDEcar_NAMES = {
    "SINGLE_CELL_FORMAL23_SUCCESS.json",
    "DIRECTIONAL_CNV_WEBSITE_BINDING_PROVENANCE_20260904_r3.json",
    "INDEPENDENT_AUDIT_BINDING_PROVENANCE_20260905_r6.json",
}


def fail(message: str) -> "NoReturn":
    raise BuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clinical_artifact_sha256(path: Path) -> str:
    """Match clinical_km_release.artifact_sha256 exactly."""
    path = path.resolve()
    if path.is_symlink():
        fail(f"symlink artifact is forbidden: {path}")
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        fail(f"missing artifact: {path}")
    files = [item for item in path.rglob("*") if item.is_file()]
    if not files:
        fail(f"empty artifact directory: {path}")
    files.sort(key=lambda item: item.relative_to(path).as_posix().casefold())
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink():
            fail(f"symlink inside artifact directory: {item}")
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def registry_artifact_sha256(path: Path) -> str:
    """Match release_registry.artifact_sha256 for a file/tree."""
    path = path.resolve()
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            fail(f"empty registry artifact: {path}")
        digest = hashlib.sha256()
        for item in files:
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(bytes.fromhex(registry_artifact_sha256(item)))
            digest.update(b"\0")
        return digest.hexdigest()
    fail(f"missing registry artifact: {path}")


def load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        fail(f"missing or unsafe JSON: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid JSON {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.building")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def ensure_regular_source(path: Path, label: str) -> None:
    if path.is_symlink() or not path.exists():
        fail(f"{label} missing or symlink: {path}")
    if not path.is_file() and not path.is_dir():
        fail(f"{label} is not a regular file/directory: {path}")


def copy_file(src: Path, dst: Path, *, hardlink_preferred: bool = True) -> str:
    ensure_regular_source(src, "source")
    if not src.is_file():
        fail(f"copy_file source is not a file: {src}")
    if dst.exists() or dst.is_symlink():
        if RESUME_MODE and dst.is_file() and not dst.is_symlink():
            # These three files are deliberately rewritten by the r8
            # localization stages.  During a resume, the r6 tree traversal
            # must not treat their expected hash change as corruption.
            if dst.name in REBOUND_SIDEcar_NAMES and "artifacts" in dst.parts:
                LEDGER.records.append(
                    {"source": str(src), "destination": str(dst), "bytes": src.stat().st_size, "mode": "existing_rebound"}
                )
                return "existing_rebound"
            if sha256_file(src) != sha256_file(dst):
                fail(f"resume destination hash mismatch (refusing overwrite): {dst}")
            LEDGER.records.append(
                {"source": str(src), "destination": str(dst), "bytes": src.stat().st_size, "mode": "existing"}
            )
            return "existing"
        fail(f"destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    mode = "copy"
    if hardlink_preferred:
        try:
            os.link(src, dst)
            mode = "hardlink"
        except OSError:
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)
    # Preserve the source mode for copied files; hardlinks already share it.
    if mode == "copy":
        shutil.copymode(src, dst)
    LEDGER.add(src, dst, mode)
    return mode


def copy_tree(src: Path, dst: Path, *, skip_pycache: bool = False) -> None:
    ensure_regular_source(src, "source tree")
    if not src.is_dir():
        fail(f"source tree is not a directory: {src}")
    if dst.exists() or dst.is_symlink():
        if not (RESUME_MODE and dst.is_dir() and not dst.is_symlink()):
            fail(f"destination tree already exists: {dst}")
    else:
        dst.mkdir(parents=True)
    for current, dirs, files in os.walk(src, topdown=True, followlinks=False):
        current_path = Path(current)
        kept_dirs: list[str] = []
        for name in sorted(dirs):
            source_dir = current_path / name
            if source_dir.is_symlink():
                fail(f"source tree contains symlink: {source_dir}")
            if skip_pycache and name == "__pycache__":
                LEDGER.skipped_pycache += 1
                continue
            kept_dirs.append(name)
            (dst / source_dir.relative_to(src)).mkdir(parents=True, exist_ok=True)
        dirs[:] = kept_dirs
        for name in sorted(files):
            source_file = current_path / name
            if source_file.is_symlink():
                fail(f"source tree contains symlink: {source_file}")
            target = dst / source_file.relative_to(src)
            copy_file(source_file, target)
            if LEDGER.files % 25 == 0:
                print(f"copied/verified {LEDGER.files} files", flush=True)


def copy_one_tree_or_file(src: Path, dst: Path) -> None:
    if src.is_dir():
        copy_tree(src, dst)
    else:
        copy_file(src, dst)


def replace_paths(value: Any, mappings: dict[str, str]) -> Any:
    """Replace exact path strings/prefixes, longest source prefix first."""
    if isinstance(value, dict):
        return {key: replace_paths(item, mappings) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_paths(item, mappings) for item in value]
    if not isinstance(value, str):
        return value
    for source, target in sorted(mappings.items(), key=lambda pair: len(pair[0]), reverse=True):
        if value == source:
            return target
        if value.startswith(source.rstrip("/") + "/"):
            return target.rstrip("/") + value[len(source.rstrip("/")):]
    return value


def canonical_without(payload: dict[str, Any], field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def source_dest_mapping(src: Path, dst: Path, mappings: dict[str, str]) -> None:
    mappings[str(src.resolve())] = str(dst.resolve())


def copy_json_rebound(
    src: Path,
    dst: Path,
    mappings: dict[str, str],
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    payload = replace_paths(load_json(src), mappings)
    if mutate is not None:
        mutate(payload)
    write_json(dst, payload)
    return payload


def verify_hash(path: Path, expected: str, label: str, *, clinical: bool = False) -> None:
    observed = clinical_artifact_sha256(path) if clinical else sha256_file(path)
    if observed.lower() != str(expected).lower():
        fail(f"{label} hash mismatch: expected={expected} observed={observed}")


def copy_formal23() -> dict[str, Any]:
    root = R8_ROOT / "code/artifacts/r8_payloads/single_cell"
    sidecar_dir = R8_ROOT / "code/artifacts/v32_server_bindings"
    source_success_sidecar = R7_ROOT / "code/artifacts/v32_server_bindings/SINGLE_CELL_FORMAL23_SUCCESS.json"
    source_binding = Path("./data/CancerLncAtlas/results/single_cell_r11_formal23_20260903_r1/COHORT_BINDING.json")
    source_audit = Path("./data/CancerLncAtlas/runtime/audits/single_cell_r11_formal23_20260903_r1/AUDIT.json")
    source_run_status = Path("./data/CancerLncAtlas/runtime/single_cell_cell_level_r11_rescued_20260903_r1/RUN_STATUS.json")
    source_success = Path("./data/CancerLncAtlas/runtime/single_cell_cell_level_r11_rescued_20260903_r1/SUCCESS.json")
    source_summary: Path | None = None

    audit_source_payload = load_json(source_audit)
    if isinstance(audit_source_payload.get("summary_tsv_path"), str):
        candidate = Path(audit_source_payload["summary_tsv_path"])
        if candidate.is_file() and not candidate.is_symlink():
            source_summary = candidate

    mappings: dict[str, str] = {}
    source_dest_mapping(source_binding, root / "COHORT_BINDING.json", mappings)
    source_dest_mapping(source_audit, root / "AUDIT.json", mappings)
    source_dest_mapping(source_run_status, root / "RUN_STATUS.json", mappings)
    source_dest_mapping(source_success, root / "R11_SUCCESS.json", mappings)
    if source_summary is not None:
        source_dest_mapping(source_summary, root / "SUMMARY.tsv", mappings)

    # Materialize each of the exactly 23 audited output roots.  The audit is
    # the universe authority; no cancer list is hand-written here.
    per_cancer = audit_source_payload.get("per_cancer")
    if not isinstance(per_cancer, list) or len(per_cancer) != 23:
        fail("formal23 audit does not contain 23 per-cancer records")
    for record in per_cancer:
        if not isinstance(record, dict):
            fail("formal23 audit has a malformed cancer record")
        cancer = str(record.get("cancer_id", ""))
        source_root = Path(str(record.get("output_root", ""))).resolve()
        if not cancer or not source_root.is_dir() or source_root.is_symlink():
            fail(f"formal23 output root unavailable: {cancer} {source_root}")
        target_root = root / f"cancer_id={cancer}"
        source_dest_mapping(source_root, target_root, mappings)
    # Copy after all mappings are known so every nested path can be rebound.
    for record in per_cancer:
        cancer = str(record["cancer_id"])
        source_root = Path(str(record["output_root"])).resolve()
        copy_tree(source_root, root / f"cancer_id={cancer}")
    copy_json_rebound(source_binding, root / "COHORT_BINDING.json", mappings)
    copy_json_rebound(source_audit, root / "AUDIT.json", mappings)
    if source_run_status.is_file():
        copy_json_rebound(source_run_status, root / "RUN_STATUS.json", mappings)
    if source_success.is_file():
        copy_json_rebound(source_success, root / "R11_SUCCESS.json", mappings)
    if source_summary is not None:
        copy_file(source_summary, root / "SUMMARY.tsv")

    binding_path = root / "COHORT_BINDING.json"
    audit_path = root / "AUDIT.json"
    # Rebind the audit's explicit cohort-success edge and recompute its
    # self-excluding contract hash after all path changes.
    audit = load_json(audit_path)
    audit["cohort_success_path"] = str(binding_path)
    audit["cohort_success_sha256"] = sha256_file(binding_path)
    audit["audit_contract_sha256"] = canonical_without(audit, "audit_contract_sha256")
    write_json(audit_path, audit)

    # The r11 success marker is copied into the local payload for provenance;
    # its content is not used by the formal23 query but is kept path-safe.
    local_r11_success = root / "R11_SUCCESS.json"
    local_r11_success_sha = sha256_file(local_r11_success) if local_r11_success.exists() else None

    local_sidecar = sidecar_dir / "SINGLE_CELL_FORMAL23_SUCCESS.json"
    sidecar = copy_json_rebound(source_success_sidecar, local_sidecar, mappings)
    sidecar["formal_binding_path"] = str(binding_path)
    sidecar["formal_binding_sha256"] = sha256_file(binding_path)
    sidecar["independent_audit_path"] = str(audit_path)
    sidecar["independent_audit_sha256"] = sha256_file(audit_path)
    sidecar["run_status_path"] = str(root / "RUN_STATUS.json")
    if (root / "RUN_STATUS.json").is_file():
        sidecar["run_status_sha256"] = sha256_file(root / "RUN_STATUS.json")
    sidecar["website_bound"] = False
    write_json(local_sidecar, sidecar)
    # Re-check the generated outer chain now, before API startup.
    if sidecar.get("formal_binding_sha256") != sha256_file(binding_path):
        fail("formal23 sidecar binding hash was not updated")
    if sidecar.get("independent_audit_sha256") != sha256_file(audit_path):
        fail("formal23 sidecar audit hash was not updated")
    return {
        "success_path": str(local_sidecar),
        "success_sha256": sha256_file(local_sidecar),
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "audit_path": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "r11_success_path": str(local_r11_success),
        "r11_success_sha256": local_r11_success_sha,
        "cancer_count": len(per_cancer),
        "summary_localized": source_summary is not None,
    }


def copy_cnv() -> dict[str, Any]:
    root = R8_ROOT / "code/artifacts/r8_payloads/cnv"
    sidecar_dir = R8_ROOT / "code/artifacts/v32_server_bindings"
    source_binding = R7_ROOT / "code/artifacts/v32_server_bindings/DIRECTIONAL_CNV_WEBSITE_BINDING_PROVENANCE_20260904_r3.json"
    source_audit_binding = R7_ROOT / "code/artifacts/v32_server_bindings/INDEPENDENT_AUDIT_BINDING_PROVENANCE_20260905_r6.json"
    source_prediction = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260904_r5/cnv_artifacts/directional_cnv_typed_predictions.parquet")
    source_coverage = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260904_r5/cnv_artifacts/directional_cnv_coverage_33c.parquet")
    source_lineage = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260904_r5/cnv_artifacts/LINEAGE.json")
    source_transform = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260904_r5/cnv_artifacts/TRANSFORMATION_AUDIT.json")
    source_report = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260905_r6/audit/directional_cnv_independent_20260904_r4/AUDIT.json")
    if not source_report.is_file():
        source_report = Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260904_r5/audit/directional_cnv_independent_20260904_r4/AUDIT.json")
    # The r7 audit sidecar is authoritative for the exact report path.  Use it
    # if the convenience path above differs.
    audit_source_payload = load_json(source_audit_binding)
    report_decl = audit_source_payload.get("report", {})
    if isinstance(report_decl, dict) and report_decl.get("path"):
        declared_report = Path(str(report_decl["path"]))
        if declared_report.is_file():
            source_report = declared_report
    sources = [source_prediction, source_coverage, source_lineage, source_transform, source_report]
    for source in sources:
        ensure_regular_source(source, "CNV source")
    destinations = {
        source_prediction: root / source_prediction.name,
        source_coverage: root / source_coverage.name,
        source_lineage: root / source_lineage.name,
        source_transform: root / source_transform.name,
        source_report: root / "INDEPENDENT_AUDIT.json",
    }
    mappings: dict[str, str] = {}
    for source, target in destinations.items():
        source_dest_mapping(source, target, mappings)
    local_binding = sidecar_dir / source_binding.name
    local_audit_binding = sidecar_dir / source_audit_binding.name
    source_dest_mapping(source_binding, local_binding, mappings)
    source_dest_mapping(source_audit_binding, local_audit_binding, mappings)
    source_dest_mapping(
        Path("${PRIVATE_WORK_ROOT}/CancerLncAtlas_v32_staging_20260904_r3/code/cc_hhgt/v32/directional_cnv_query.py"),
        R8_ROOT / "code/cc_hhgt/v32/directional_cnv_query.py",
        mappings,
    )
    for source, target in destinations.items():
        copy_file(source, target, hardlink_preferred=False)
    release = copy_json_rebound(source_binding, local_binding, mappings)
    release.setdefault("query_code", {})["path"] = str(R8_ROOT / "code/cc_hhgt/v32/directional_cnv_query.py")
    release["query_code"]["sha256"] = sha256_file(R8_ROOT / "code/cc_hhgt/v32/directional_cnv_query.py")
    write_json(local_binding, release)
    release_sha = sha256_file(local_binding)
    report = copy_json_rebound(source_report, root / "INDEPENDENT_AUDIT.json", mappings)
    # The independent report is cryptographically tied to the release
    # binding.  Rebinding the release path necessarily changes its digest;
    # carry that new digest into the report before hashing the audit binding.
    if "release_binding_sha256" in report:
        report["release_binding_sha256"] = release_sha
    if isinstance(report.get("release_binding"), dict):
        report["release_binding"]["path"] = str(local_binding)
        report["release_binding"]["sha256"] = release_sha
    write_json(root / "INDEPENDENT_AUDIT.json", report)
    report_sha = sha256_file(root / "INDEPENDENT_AUDIT.json")
    audit_binding = copy_json_rebound(source_audit_binding, local_audit_binding, mappings)
    audit_binding["release_binding"] = {"path": str(local_binding), "sha256": release_sha}
    audit_binding["report"] = {"path": str(root / "INDEPENDENT_AUDIT.json"), "sha256": report_sha}
    audit_binding.setdefault("query_code", {})["path"] = str(R8_ROOT / "code/cc_hhgt/v32/directional_cnv_query.py")
    audit_binding["query_code"]["sha256"] = sha256_file(R8_ROOT / "code/cc_hhgt/v32/directional_cnv_query.py")
    write_json(local_audit_binding, audit_binding)
    audit_sha = sha256_file(local_audit_binding)
    return {
        "binding_path": str(local_binding),
        "binding_sha256": release_sha,
        "audit_binding_path": str(local_audit_binding),
        "audit_binding_sha256": audit_sha,
        "prediction_path": str(destinations[source_prediction]),
        "prediction_sha256": sha256_file(destinations[source_prediction]),
        "coverage_path": str(destinations[source_coverage]),
        "coverage_sha256": sha256_file(destinations[source_coverage]),
        "report_path": str(root / "INDEPENDENT_AUDIT.json"),
        "report_sha256": report_sha,
    }


def copy_clinical() -> dict[str, Any]:
    root = R8_ROOT / "code/artifacts/r8_payloads/clinical"
    sidecar_dir = R8_ROOT / "code/artifacts/r7_server_native/clinical"
    source_binding = R7_ROOT / "code/artifacts/r7_server_native/clinical/CLINICAL_KM_BINDING.json"
    source_binding_dir = Path("./data/CancerLncAtlas/runtime/authorized_bindings/v32_independent_heads_20260829_r2/runtime_bindings_r3/clinical")
    source_success = source_binding_dir / "SUCCESS.json"
    source_module = source_binding_dir / "MODULE_LINEAGE.json"
    source_inputs = source_binding_dir / "SOURCE_INPUTS.json"
    source_curves = Path("./data/CancerLncAtlas/runtime/authorized_payloads/v32_clinical_query_20260829_r3/clinical_km_curves")
    source_expression = Path("./data/CancerLncAtlas/runtime/authorized_payloads/v32_clinical_query_20260829_r3/formal_lncRNA_expression")
    source_stats = Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3/v32/artifacts/v32_clinical_km_fresh_20260826_r1/clinical_km_statistics.parquet")
    source_candidate = Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3/v32/artifacts/formal_prepared/FORMAL_CANDIDATE_UNIVERSE.parQUET")
    # Correct the case typo defensively; Linux paths are case-sensitive.
    if not source_candidate.is_file():
        source_candidate = source_candidate.with_name("FORMAL_CANDIDATE_UNIVERSE.parquet")
    source_workbook = Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3/v32/inputs/v32_full_multitask/clinical/TCGA-CDR-SupplementalTableS1.xlsx")
    sources: list[Path] = [
        source_success,
        source_module,
        source_inputs,
        source_curves,
        source_expression,
        source_stats,
        source_candidate,
        source_workbook,
    ]
    for source in sources:
        ensure_regular_source(source, "clinical source")
    targets = {
        source_success: root / "SUCCESS.source.json",
        source_module: root / "MODULE_LINEAGE.json",
        source_inputs: root / "SOURCE_INPUTS.json",
        source_curves: root / "clinical_km_curves",
        source_expression: root / "formal_lncRNA_expression",
        source_stats: root / "clinical_km_statistics.parquet",
        source_candidate: root / "FORMAL_CANDIDATE_UNIVERSE.parquet",
        source_workbook: root / "TCGA-CDR-SupplementalTableS1.xlsx",
    }
    mappings: dict[str, str] = {}
    for source, target in targets.items():
        source_dest_mapping(source, target, mappings)
    local_binding = sidecar_dir / "CLINICAL_KM_BINDING.json"
    source_dest_mapping(source_binding, local_binding, mappings)
    source_dest_mapping(source_success, sidecar_dir / "SUCCESS.json", mappings)
    for source, target in targets.items():
        copy_one_tree_or_file(source, target)
    binding = copy_json_rebound(source_binding, local_binding, mappings)
    write_json(local_binding, binding)
    binding_sha = sha256_file(local_binding)
    success = copy_json_rebound(source_success, sidecar_dir / "SUCCESS.json", mappings)
    success["binding"] = local_binding.name
    success["binding_sha256"] = binding_sha
    write_json(sidecar_dir / "SUCCESS.json", success)
    # Check the declared Merkle hashes now; this catches accidental omitted
    # hidden files or path-layout changes before API startup.
    artifacts = binding.get("artifacts", {})
    for role in ("curves", "module_lineage", "source_inputs", "statistics"):
        declaration = artifacts.get(role)
        if not isinstance(declaration, dict):
            fail(f"clinical binding lacks artifact {role}")
        path = Path(str(declaration.get("path", "")))
        verify_hash(path, str(declaration.get("sha256", "")), f"clinical {role}", clinical=True)
    authorities = binding.get("authorities", {})
    for role in ("current_exact_candidates", "current_patient_logcpm", "tcga_cdr_workbook"):
        declaration = authorities.get(role)
        if not isinstance(declaration, dict):
            fail(f"clinical binding lacks authority {role}")
        verify_hash(Path(str(declaration["path"])), str(declaration["sha256"]), f"clinical authority {role}", clinical=True)
    return {
        "binding_path": str(local_binding),
        "binding_sha256": binding_sha,
        "success_path": str(sidecar_dir / "SUCCESS.json"),
        "success_sha256": sha256_file(sidecar_dir / "SUCCESS.json"),
        "curves_path": str(targets[source_curves]),
        "curves_sha256": clinical_artifact_sha256(targets[source_curves]),
        "expression_path": str(targets[source_expression]),
        "expression_sha256": clinical_artifact_sha256(targets[source_expression]),
    }


def copy_state() -> dict[str, Any]:
    root = R8_ROOT / "code/artifacts/r8_payloads/state_gene_set"
    sidecar_dir = R8_ROOT / "code/artifacts/r7_server_native/state_gene_set"
    source_binding = R7_ROOT / "code/artifacts/r7_server_native/state_gene_set/STATE_GENE_SET_BINDING.json"
    source_dir = Path("./data/CancerLncAtlas/runtime/authorized_bindings/v32_independent_heads_20260829_r2/runtime_bindings_r3/state_gene_set")
    source_success = source_dir / "SUCCESS.json"
    source_checkpoint = source_dir / "SOURCE_CHECKPOINT_MANIFEST.json"
    source_state_lineage = source_dir / "SOURCE_STATE_MODULE_LINEAGE.json"
    source_module_lineage = source_dir / "MODULE_LINEAGE.json"
    output_dir = Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3/v32/artifacts/v32_state_gene_sets_20260826_r2_code_bound")
    source_catalog = output_dir / "state_gene_set_catalog.parquet"
    source_gmt = output_dir / "state_gene_sets.gmt"
    source_members = output_dir / "state_gene_set_members.parquet"
    source_report = output_dir / "STATE_GENE_SET_REPORT.md"
    source_report_manifest = source_dir / "STATE_REPORT_MANIFEST.json"
    source_materializer = Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3/v32/cc_hhgt/v32/state_gene_set_release.py")
    source_runner = Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3/v32/scripts/materialize_v32_state_gene_sets.py")
    source_predictions = R6_CODE / "artifacts/v32_full_multitask/state_release_complete/state_typed_predictions.parquet"
    source_files = [
        source_success,
        source_checkpoint,
        source_state_lineage,
        source_module_lineage,
        source_report_manifest,
        source_catalog,
        source_gmt,
        source_members,
        source_report,
        source_materializer,
        source_runner,
        source_predictions,
    ]
    for source in source_files:
        ensure_regular_source(source, "state source")
    targets = {
        source_success: root / "SUCCESS.source.json",
        source_checkpoint: root / "SOURCE_CHECKPOINT_MANIFEST.json",
        source_state_lineage: root / "SOURCE_STATE_MODULE_LINEAGE.json",
        source_module_lineage: root / "MODULE_LINEAGE.json",
        source_report_manifest: root / "STATE_REPORT_MANIFEST.json",
        source_catalog: root / "state_gene_set_catalog.parquet",
        source_gmt: root / "state_gene_sets.gmt",
        source_members: root / "state_gene_set_members.parquet",
        source_report: root / "STATE_GENE_SET_REPORT.md",
        source_materializer: root / "state_gene_set_release.py",
        source_runner: root / "materialize_v32_state_gene_sets.py",
        source_predictions: root / "state_typed_predictions.parquet",
    }
    mappings: dict[str, str] = {}
    for source, target in targets.items():
        source_dest_mapping(source, target, mappings)
    local_binding = sidecar_dir / "STATE_GENE_SET_BINDING.json"
    source_dest_mapping(source_binding, local_binding, mappings)
    source_dest_mapping(source_success, sidecar_dir / "SUCCESS.json", mappings)
    for source, target in targets.items():
        copy_one_tree_or_file(source, target)
    binding = copy_json_rebound(source_binding, local_binding, mappings)
    write_json(local_binding, binding)
    binding_sha = sha256_file(local_binding)
    success = copy_json_rebound(source_success, sidecar_dir / "SUCCESS.json", mappings)
    success["binding"] = local_binding.name
    success["binding_sha256"] = binding_sha
    write_json(sidecar_dir / "SUCCESS.json", success)
    # Source declarations are all files and are checked by StateGeneSetQuery.
    for role, declaration in binding.get("source_artifacts", {}).items():
        if not isinstance(declaration, dict):
            fail(f"state source declaration malformed: {role}")
        verify_hash(Path(str(declaration.get("path", ""))), str(declaration.get("sha256", "")), f"state source {role}")
    return {
        "binding_path": str(local_binding),
        "binding_sha256": binding_sha,
        "success_path": str(sidecar_dir / "SUCCESS.json"),
        "success_sha256": sha256_file(sidecar_dir / "SUCCESS.json"),
        "prediction_path": str(targets[source_predictions]),
        "prediction_sha256": sha256_file(targets[source_predictions]),
    }


def copy_evidence_direction() -> dict[str, Any]:
    root = R8_ROOT / "code/artifacts/r8_payloads/evidence_direction"
    sidecar_dir = R8_ROOT / "code/artifacts/r7_server_native/evidence_direction"
    source_binding = R7_ROOT / "code/artifacts/r7_server_native/evidence_direction/EVIDENCE_DIRECTION_PROBABILITY_BINDING.json"
    source_audit_binding = R7_ROOT / "code/artifacts/r7_server_native/evidence_direction/INDEPENDENT_AUDIT_BINDING.json"
    source_artifact = Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260829_r3/v32/artifacts/v32_evidence_direction_probabilities_20260826_r1_fixed_seed_reinference/evidence_direction_probabilities.parquet")
    source_report = Path("./data/CancerLncAtlas/runtime/authorized_bindings/v32_independent_heads_20260829_r2/runtime_bindings_r3/evidence_direction/EVIDENCE_DIRECTION_INDEPENDENT_AUDIT.json")
    ensure_regular_source(source_artifact, "Evidence direction artifact")
    ensure_regular_source(source_report, "Evidence direction audit report")
    target_artifact = root / "evidence_direction_probabilities.parquet"
    target_report = root / "EVIDENCE_DIRECTION_INDEPENDENT_AUDIT.json"
    mappings: dict[str, str] = {}
    source_dest_mapping(source_artifact, target_artifact, mappings)
    source_dest_mapping(source_report, target_report, mappings)
    local_binding = sidecar_dir / "EVIDENCE_DIRECTION_PROBABILITY_BINDING.json"
    local_audit = sidecar_dir / "INDEPENDENT_AUDIT_BINDING.json"
    source_dest_mapping(source_binding, local_binding, mappings)
    source_dest_mapping(source_audit_binding, local_audit, mappings)
    copy_file(source_artifact, target_artifact)
    copy_file(source_report, target_report)
    binding = copy_json_rebound(source_binding, local_binding, mappings)
    binding.setdefault("artifact", {})["path"] = str(target_artifact)
    binding.setdefault("artifact", {})["sha256"] = sha256_file(target_artifact)
    binding.setdefault("authority", {})["core_manifest_path"] = binding.get("authority", {}).get("core_manifest_path")
    write_json(local_binding, binding)
    binding_sha = sha256_file(local_binding)
    audit = copy_json_rebound(source_audit_binding, local_audit, mappings)
    audit["release_binding"] = {"path": str(local_binding), "sha256": binding_sha}
    audit["artifact"] = {"path": str(target_artifact), "sha256": sha256_file(target_artifact)}
    audit["report"] = {"path": str(target_report), "sha256": sha256_file(target_report)}
    write_json(local_audit, audit)
    audit_sha = sha256_file(local_audit)
    report = load_json(target_report)
    if report.get("artifact", {}).get("sha256") != sha256_file(target_artifact):
        # The audit report is a copied authority; only path relocation is
        # allowed.  Its artifact digest is a scientific hash and must match.
        fail("Evidence direction audit report artifact hash drift")
    return {
        "binding_path": str(local_binding),
        "binding_sha256": binding_sha,
        "audit_binding_path": str(local_audit),
        "audit_binding_sha256": audit_sha,
        "artifact_path": str(target_artifact),
        "artifact_sha256": sha256_file(target_artifact),
        "report_path": str(target_report),
        "report_sha256": sha256_file(target_report),
    }


def write_r8_manifest(parts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = load_json(R7_MANIFEST)
    manifest = json.loads(json.dumps(source))
    manifest["staging_revision"] = "SELF_CONTAINED_RUNTIME_CLOSURE_20260905_R8"
    manifest["supersedes_manifest"] = {
        "path": "history/r7_manifest.json",
        "sha256": sha256_file(R7_MANIFEST),
    }
    mounted = {
        "single_cell_formal23": parts["single_cell"]["success_path"],
        "directional_cnv": parts["cnv"]["binding_path"],
        "directional_cnv_independent_audit": parts["cnv"]["audit_binding_path"],
        "clinical_kaplan_meier": parts["clinical"]["binding_path"],
        "state_gene_sets": parts["state"]["binding_path"],
        "evidence_direction_probabilities": parts["evidence"]["binding_path"],
        "evidence_direction_probabilities_independent_audit": parts["evidence"]["audit_binding_path"],
    }
    for capability_id, absolute_path in mounted.items():
        entry = manifest["bindings"][capability_id]
        path = Path(absolute_path).resolve()
        try:
            relative = path.relative_to(R8_ROOT / "code")
        except ValueError:
            fail(f"mounted binding escapes r8 code root: {path}")
        entry["path"] = relative.as_posix()
        entry["sha256"] = sha256_file(path)
        entry["status"] = "MOUNTED_HASH_PINNED"
        entry["source"] = "r8 self-contained localized payload"
    registry = R8_ROOT / "code/artifacts/v32_staging/core_registry_refresh_20260903_aux_r4/RELEASE_REGISTRY.json"
    if not registry.is_file():
        fail(f"r6 core registry did not copy: {registry}")
    if registry_artifact_sha256(registry) != manifest["registry"]["sha256"]:
        fail("copied core registry hash differs from r6 authority")
    manifest["registry"]["path"] = "artifacts/v32_staging/core_registry_refresh_20260903_aux_r4/RELEASE_REGISTRY.json"
    manifest["registry"]["sha256"] = registry_artifact_sha256(registry)
    manifest["registry"]["role"] = "V3.2 core release registry; copied byte-identically from r6"
    manifest["code_pins"] = {
        "staging_api": {
            "path": "website/backend/v32_staging_api.py",
            "role": "isolated_staging_runtime",
            "sha256": sha256_file(R8_ROOT / "code/website/backend/v32_staging_api.py"),
        },
        "formal23_query": {
            "path": "cc_hhgt/v32/single_cell_formal23_query.py",
            "role": "isolated_staging_query_runtime",
            "sha256": sha256_file(R8_ROOT / "code/cc_hhgt/v32/single_cell_formal23_query.py"),
        },
        "directional_cnv_query": {
            "path": "cc_hhgt/v32/directional_cnv_query.py",
            "role": "isolated_staging_query_runtime",
            "sha256": sha256_file(R8_ROOT / "code/cc_hhgt/v32/directional_cnv_query.py"),
        },
    }
    merge = manifest.setdefault("merge_assessment", {})
    merge.update(
        {
            "schema": "cancerlncatlas.v32.r8.self_contained_merge_assessment.v1",
            "new_server_overlay_entry_count": 7,
            "verified_mounted_entry_count": 7,
            "pending_entry_count": sum(
                1 for item in manifest["bindings"].values() if item.get("status") == "PENDING_FORMAL_SUCCESS"
            ),
            "large_payload_hashes_skipped": False,
            "r6_preserved": True,
            "exact_gaps_are_explicit": True,
            "production_deployed": False,
            "release_ready": False,
            "decision": "R8_SELF_CONTAINED_RUNTIME_CLOSURE_WITH_EXPLICIT_GAPS",
        }
    )
    path = R8_ROOT / "code/config/v32_server_unified_staging_bindings_20260905_r8.json"
    write_json(path, manifest)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "mounted_count": len(mounted),
        "pending_count": merge["pending_entry_count"],
    }


def write_runner() -> Path:
    """Write the bounded r8 launcher used for the actual loopback smoke."""
    path = R8_ROOT / "code/scripts/run_v32_website_staging_candidate_20260905_r8.py"
    script = r'''#!/usr/bin/env python3
"""Bounded r8 staging launcher; server 149, loopback only, no GPU/training."""
from __future__ import annotations
import hashlib, json, os, shutil, signal, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HOST = "149"
ROOT = Path("./data/CancerLncAtlas/runtime/web_candidates/v32_20260905_r8")
CODE = ROOT / "code"
MANIFEST = CODE / "config/v32_server_unified_staging_bindings_20260905_r8.json"
CATALOG_DIR = ROOT / "catalog_candidate_20260905_r8"
CATALOG = CATALOG_DIR / "v32-capability-catalog.json"
FRONTEND_CATALOG = CODE / "website/frontend/v32-capability-catalog.json"
SCOPE = ROOT / "capability_scope_20260905_r8.json"
SMOKE = ROOT / "receipts/R8_ROUTE_SMOKE_20260905.json"
RECEIPT = ROOT / "receipts/R8_SELF_CONTAINED_STAGING_RECEIPT_20260905.json"
PORT = int(os.environ.get("V32_R8_PORT", "8298"))
PYTHON = Path(sys.executable).resolve()
BASE = f"http://127.0.0.1:{PORT}"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()

def req(method: str, path: str, payload=None):
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode(); headers["Content-Type"] = "application/json"
    try:
        with urlopen(Request(BASE + path, data=data, headers=headers, method=method), timeout=30) as r:
            raw = r.read()
            try: body = json.loads(raw.decode())
            except Exception: body = raw.decode(errors="replace")
            return int(r.status), body, None
    except HTTPError as e:
        raw = e.read()
        try: body = json.loads(raw.decode())
        except Exception: body = raw.decode(errors="replace")
        return int(e.code), body, None
    except (OSError, URLError, TimeoutError) as e:
        return 0, None, f"{type(e).__name__}: {e}"

def main() -> int:
    started = datetime.now(timezone.utc).isoformat()
    ROOT.joinpath("receipts").mkdir(parents=True, exist_ok=True)
    result = {"format": "CANCERLNCATLAS_V32_R8_ROUTE_SMOKE_V1", "host": HOST,
              "environment": "staging", "production_deployed": False,
              "release_ready": False, "gpu_started": False, "training_started": False,
              "port": PORT, "base_url": BASE, "started_at_utc": started, "probes": []}
    proc = None
    status = "FAIL"
    error = None
    try:
        if subprocess.check_output(["hostname", "-s"], text=True).strip() != HOST:
            raise RuntimeError("BLOCKED_WRONG_HOST")
        if PORT in {8260, 8261, 8262}:
            raise RuntimeError("BLOCKED_PRODUCTION_PORT")
        if not MANIFEST.is_file() or not CATALOG.is_file():
            raise RuntimeError("BLOCKED_MISSING_MANIFEST_OR_BOOTSTRAP_CATALOG")
        # Static manifest validation happens before opening a socket.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(CODE)
        check = subprocess.run([str(PYTHON), "-c", (
            "from cc_hhgt.v32.unified_staging_bindings import load_unified_staging_bindings; "
            f"load_unified_staging_bindings({str(MANIFEST)!r}, repo_root={str(CODE)!r})"
        )], env=env, capture_output=True, text=True, timeout=300)
        if check.returncode:
            raise RuntimeError("MANIFEST_VALIDATION_FAILED: " + check.stderr[-4000:])
        server = CODE / "scripts/serve_v32_unified_staging_candidate.py"
        log = ROOT / "receipts/R8_SERVER.log"
        with log.open("w", encoding="utf-8") as stream:
            proc = subprocess.Popen([str(PYTHON), str(server), "--repo-root", str(CODE),
                                     "--manifest", str(MANIFEST), "--catalog", str(CATALOG),
                                     "--host", "127.0.0.1", "--port", str(PORT)],
                                    cwd=str(CODE), env=env, stdout=stream, stderr=subprocess.STDOUT)
        ready = False
        for _ in range(600):
            code, body, err = req("GET", "/v3.2-staging/health")
            if code == 200:
                ready = True; break
            if proc.poll() is not None:
                raise RuntimeError("SERVER_EXITED: " + log.read_text(errors="replace")[-4000:])
            time.sleep(1)
        if not ready: raise RuntimeError("SERVER_NOT_READY_AFTER_600S")
        probe = CODE / "scripts/probe_v32_website_staging_candidate_20260904.py"
        first = subprocess.run([str(PYTHON), str(probe), "--base-url", BASE,
                                "--repo-root", str(CODE), "--manifest", str(MANIFEST),
                                "--catalog", str(CATALOG), "--output", str(SCOPE),
                                "--allow-stale-catalog"], env=env, capture_output=True, text=True, timeout=900)
        if first.returncode: raise RuntimeError("INITIAL_ROUTE_PROBE_FAILED: " + first.stderr[-4000:])
        scope_sha = sha(SCOPE)
        builder = CODE / "scripts/build_v32_staging_catalog_from_manifest_20260904.py"
        r11 = CODE / "artifacts/v32_single_cell_r11_rescued_contract_20260903_r1_server_manifest"
        build = subprocess.run([str(PYTHON), str(builder), "--repo-root", str(CODE),
                                "--manifest", str(MANIFEST), "--scope", str(SCOPE),
                                "--scope-sha256", scope_sha, "--output", str(CATALOG),
                                "--r11-handoff", str(r11 / "TRAINING_HANDOFF.json"),
                                "--r11-handoff-sha256", sha(r11 / "TRAINING_HANDOFF.json"),
                                "--r11-run-status", str(r11 / "RUN_STATUS.json"),
                                "--r11-run-status-sha256", sha(r11 / "RUN_STATUS.json"),
                                "--r11-success", str(r11 / "SUCCESS.json"),
                                "--r11-success-sha256", sha(r11 / "SUCCESS.json")],
                               env=env, capture_output=True, text=True, timeout=1200)
        if build.returncode: raise RuntimeError("CATALOG_BUILD_FAILED: " + build.stderr[-4000:])
        # Keep the frontend copy byte-identical to the served catalog.
        temporary = FRONTEND_CATALOG.with_name(".v32-capability-catalog.json.r8.tmp")
        shutil.copy2(CATALOG, temporary); os.replace(temporary, FRONTEND_CATALOG)
        strict = subprocess.run([str(PYTHON), str(probe), "--base-url", BASE,
                                 "--repo-root", str(CODE), "--manifest", str(MANIFEST),
                                 "--catalog", str(CATALOG), "--scope", str(SCOPE),
                                 "--scope-sha256", scope_sha, "--output", str(SMOKE)],
                                env=env, capture_output=True, text=True, timeout=900)
        if strict.returncode: raise RuntimeError("STRICT_ROUTE_PROBE_FAILED: " + strict.stderr[-4000:])
        # Extra mounted-head and download metadata smoke.  No payload is
        # streamed, so a 60 MB CNV file cannot accidentally be downloaded.
        extra = [
            ("state", "GET", "/v3.2-staging/state?limit=1", None, {200}),
            ("state_gene_sets", "GET", "/v3.2-staging/state-gene-sets?limit=1", None, {200}),
            ("clinical", "GET", "/v3.2-staging/clinical?clinical_endpoint=OS&limit=1", None, {200}),
            ("evidence_direction_capability", "GET", "/v3.2-staging/evidence/direction/capability", None, {200}),
            ("evidence_direction_query", "GET", "/v3.2-staging/evidence/direction/probabilities?cancer_id=BRCA&limit=1", None, {200}),
            ("cnv_download_metadata", "GET", "/v3.2-staging/downloads/cnv_context", None, {200}),
            ("single_cell_download_metadata", "GET", "/v3.2-staging/downloads/single_cell_associations", None, {200}),
        ]
        for ident, method, route, body, expected in extra:
            code, response, err = req(method, route, body)
            item = {"probe_id": ident, "method": method, "path": route,
                    "http_status": code, "semantic_ok": code in expected, "error": err}
            result["probes"].append(item)
            if code not in expected:
                raise RuntimeError(f"EXTRA_SMOKE_FAILED {ident}: {code} {err or response}")
        result["strict_probe_path"] = str(SMOKE)
        result["strict_probe_sha256"] = sha(SMOKE)
        result["scope_path"] = str(SCOPE)
        result["scope_sha256"] = scope_sha
        result["catalog_path"] = str(CATALOG)
        result["catalog_sha256"] = sha(CATALOG)
        result["manifest_path"] = str(MANIFEST)
        result["manifest_sha256"] = sha(MANIFEST)
        result["frontend_catalog_byte_equal"] = sha(FRONTEND_CATALOG) == sha(CATALOG)
        if not result["frontend_catalog_byte_equal"]:
            raise RuntimeError("FRONTEND_CATALOG_HASH_MISMATCH")
        result["strict_probe_status"] = json.loads(SMOKE.read_text()).get("status")
        status = "PASS_STAGING_WITH_EXPLICIT_GAPS"
    except Exception as exc:
        error = str(exc)
        result["error"] = error
    finally:
        if proc is not None:
            try:
                proc.terminate(); proc.wait(timeout=20)
            except Exception:
                try: proc.kill(); proc.wait(timeout=10)
                except Exception: pass
        result["server_stopped"] = proc is None or proc.poll() is not None
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["status"] = status
        RECEIPT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (RECEIPT.with_suffix(RECEIPT.suffix + ".sha256")).write_text(sha(RECEIPT) + "  " + RECEIPT.name + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "receipt": str(RECEIPT), "error": error}, ensure_ascii=False))
    return 0 if status == "PASS_STAGING_WITH_EXPLICIT_GAPS" else 1

if __name__ == "__main__": raise SystemExit(main())
'''
    path.write_text(script, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def main() -> int:
    # ``platform.node()`` may perform a reverse-DNS lookup on 149 and can
    # block indefinitely when the storage/network resolver is busy.  The
    # execution guard only needs the kernel hostname, which is immediate.
    observed = os.uname().nodename.split(".", 1)[0] if hasattr(os, "uname") else socket.gethostname().split(".", 1)[0]
    if observed != HOST:
        fail(f"BLOCKED_WRONG_HOST expected={HOST} observed={observed}")
    global RESUME_MODE
    if R8_ROOT.exists():
        marker = R8_ROOT / "receipts/R8_COPY_PLAN_20260905.json"
        if not marker.is_file():
            fail(f"R8_ALREADY_EXISTS_NO_OVERWRITE: {R8_ROOT}")
        try:
            marker_payload = load_json(marker)
        except BuildError:
            fail(f"R8_ALREADY_EXISTS_UNVERIFIABLE: {R8_ROOT}")
        if marker_payload.get("destination") != str(R8_ROOT) or marker_payload.get("source_r6") != str(R6_ROOT):
            fail(f"R8_ALREADY_EXISTS_FOREIGN_ROOT_NO_OVERWRITE: {R8_ROOT}")
        RESUME_MODE = True
        print("resuming verified partial r8 root", flush=True)
    for path in (R6_CODE, R7_MANIFEST, R7_CATALOG, R7_SCOPE):
        ensure_regular_source(path, "required source")
    if not R8_ROOT.exists():
        R8_ROOT.mkdir(parents=True)
        (R8_ROOT / "receipts").mkdir()
    plan = {
        "format": "CANCERLNCATLAS_V32_R8_COPY_PLAN_V1",
        "host": HOST,
        "source_r6": str(R6_ROOT),
        "source_r7": str(R7_ROOT),
        "destination": str(R8_ROOT),
        "gpu_started": False,
        "training_started": False,
        "production_touched": False,
        "sealed_test_read": False,
        "sidecars": [
            "single_cell_formal23", "directional_cnv", "clinical_kaplan_meier",
            "state_gene_sets", "evidence_direction_probabilities",
        ],
        "omitted": ["drug", "pending_binding_payloads", "sealed_test", "GPU", "__pycache__"],
    }
    write_json(R8_ROOT / "receipts/R8_COPY_PLAN_20260905.json", plan)
    try:
        print("copying r6 code tree", flush=True)
        copy_tree(R6_CODE, R8_ROOT / "code", skip_pycache=True)
        # Preserve the r7 metadata as immutable local history/bootstrap only.
        copy_file(R7_MANIFEST, R8_ROOT / "history/r7_manifest.json")
        copy_file(R7_CATALOG, R8_ROOT / "history/r7_catalog.json")
        copy_file(R7_SCOPE, R8_ROOT / "history/r7_scope.json")
        CATALOG = R8_ROOT / "catalog_candidate_20260905_r8/v32-capability-catalog.json"
        copy_file(R7_CATALOG, CATALOG)
        # Localize each independent formal overlay.
        parts = {
            "single_cell": copy_formal23(),
            "cnv": copy_cnv(),
            "clinical": copy_clinical(),
            "state": copy_state(),
            "evidence": copy_evidence_direction(),
        }
        manifest = write_r8_manifest(parts)
        runner = write_runner()
        # Record copy/hash details for the runner and later independent audit.
        plan.update(
            {
                "status": "PASS_COPY_AND_REBIND",
                "manifest": manifest,
                "runner_path": str(runner),
                "runner_sha256": sha256_file(runner),
                "copy_ledger": {
                    "files": LEDGER.files,
                    "bytes": LEDGER.bytes,
                    "hardlinks": LEDGER.hardlinks,
                    "copies": LEDGER.copies,
                    "skipped_pycache_directories": LEDGER.skipped_pycache,
                },
                "parts": parts,
            }
        )
        write_json(R8_ROOT / "receipts/R8_COPY_PLAN_20260905.json", plan)
        # Keep a compact ledger (full per-file list is useful for audit but is
        # not needed by the web runtime).
        write_json(
            R8_ROOT / "receipts/R8_COPY_LEDGER_20260905.json",
            {
                "format": "CANCERLNCATLAS_V32_R8_COPY_LEDGER_V1",
                "host": HOST,
                "files": LEDGER.files,
                "bytes": LEDGER.bytes,
                "hardlinks": LEDGER.hardlinks,
                "copies": LEDGER.copies,
                "skipped_pycache_directories": LEDGER.skipped_pycache,
                "records": LEDGER.records,
            },
        )
        print(json.dumps({"status": "PASS_COPY_AND_REBIND", "root": str(R8_ROOT), "manifest": manifest, "parts": parts}, ensure_ascii=False))
        return 0
    except Exception as exc:
        failure = {
            "format": "CANCERLNCATLAS_V32_R8_SELF_CONTAINED_BUILD_RECEIPT_V1",
            "status": "FAIL_BUILD",
            "host": HOST,
            "destination": str(R8_ROOT),
            "error": f"{type(exc).__name__}: {exc}",
            "gpu_started": False,
            "training_started": False,
            "production_touched": False,
            "release_ready": False,
            "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }
        write_json(R8_ROOT / "receipts/R8_SELF_CONTAINED_BUILD_FAIL_20260905.json", failure)
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
