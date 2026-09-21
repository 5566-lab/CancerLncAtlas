"""Materialize server-native metadata closures for V3.2 auxiliary queries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


class RuntimeBindingError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoded(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise RuntimeBindingError(f"Refusing to overwrite runtime binding: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _copy(source: Path, target: Path) -> str:
    if source.is_symlink() or not source.is_file():
        raise RuntimeBindingError(f"Metadata source is missing/unsafe: {source}")
    _write(target, source.read_bytes())
    return _sha(target)


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeBindingError(f"JSON source is missing/unsafe: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _success(binding: dict[str, Any], name: str, binding_sha: str, role: str) -> dict[str, Any]:
    common = {
        "status": binding["status"],
        "binding": name,
        "binding_sha256": binding_sha,
        "production_deployed": False,
        "release_ready": False,
    }
    if role == "evidence":
        return common
    if role == "clinical":
        return {
            **common,
            "analysis_version": binding["analysis_version"],
            "computation_run_id": binding["computation_run_id"],
            "fresh_statistical_calculation": True,
            "training_not_applicable": True,
            "changes_primary_ranking": False,
        }
    if role == "state":
        return {
            **common,
            "analysis_version": binding["analysis_version"],
            "computation_run_id": binding["computation_run_id"],
            "all_seven_states_present": True,
            "historical_state_outputs_used": False,
        }
    raise RuntimeBindingError(f"Unknown success role: {role}")


def build_runtime_bindings(
    *,
    portable_overlay_root: str | Path,
    output_root: str | Path,
    server_root: str,
    clinical_expression_server_root: str | None = None,
    clinical_curves_server_root: str | None = None,
) -> dict[str, Any]:
    overlay = Path(portable_overlay_root).resolve()
    output = Path(output_root).resolve()
    if not server_root.startswith("./data/CancerLncAtlas/"):
        raise RuntimeBindingError("Runtime binding server root is outside authority")
    clinical_clean_roots = {
        "expression": clinical_expression_server_root,
        "curves": clinical_curves_server_root,
    }
    for role, clean_root in clinical_clean_roots.items():
        if clean_root is not None and not clean_root.startswith(
            "./data/CancerLncAtlas/runtime/authorized_payloads/"
        ):
            raise RuntimeBindingError(
                f"Clinical clean {role} root is outside authorized_payloads"
            )
    if (clinical_expression_server_root is None) != (clinical_curves_server_root is None):
        raise RuntimeBindingError(
            "Clinical clean expression and curves roots must be supplied together"
        )
    if output.exists() and (output.is_symlink() or any(output.iterdir())):
        raise RuntimeBindingError("Runtime binding output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)

    records: dict[str, Any] = {}

    # Evidence: the query loader needs a sibling SUCCESS and the split audit.
    evidence_dir = output / "evidence"
    evidence_server = f"{server_root}/evidence"
    split_source = overlay / "artifacts/evidence_pairblocked_fresh_20260826_r2_pinned_local/PAIR_BLOCKED_SPLIT_AUDIT.json"
    split_sha = _copy(split_source, evidence_dir / "PAIR_BLOCKED_SPLIT_AUDIT.json")
    evidence = _load(
        overlay / "artifacts/v32_evidence_output_binding_20260826_r2_local/EVIDENCE_OUTPUT_BINDING.json"
    )
    evidence["artifacts"]["split_integrity_audit"]["path"] = (
        f"{evidence_server}/PAIR_BLOCKED_SPLIT_AUDIT.json"
    )
    evidence["artifacts"]["split_integrity_audit"]["sha256"] = split_sha
    evidence_bytes = _encoded(evidence)
    evidence_path = evidence_dir / "EVIDENCE_OUTPUT_BINDING.json"
    _write(evidence_path, evidence_bytes)
    evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
    _write(
        evidence_dir / "SUCCESS.json",
        _encoded(_success(evidence, evidence_path.name, evidence_sha, "evidence")),
    )
    records["evidence"] = {
        "binding_path": f"{evidence_server}/{evidence_path.name}",
        "binding_sha256": evidence_sha,
        "scientific_status": "partial_not_publishable",
        "production_deployed": False,
        "release_ready": False,
    }

    # Clinical: copy the two JSON metadata leaves and bind their new locations.
    # A portable transfer can leave ``*.partial.<sha>`` siblings beside final
    # files.  Those siblings do not change any declared payload, but they do
    # change directory-tree hashes.  When clean runtime roots are supplied,
    # rebind the two directory authorities to residue-free hard-link views and
    # update SOURCE_INPUTS without changing any formal source hash.
    clinical_dir = output / "clinical"
    clinical_server = f"{server_root}/clinical"
    clinical_source_root = overlay / "artifacts/v32_clinical_km_fresh_20260826_r1"
    clinical = _load(clinical_source_root / "CLINICAL_KM_BINDING.json")
    module_name = "MODULE_LINEAGE.json"
    module_digest = _copy(clinical_source_root / module_name, clinical_dir / module_name)
    clinical["artifacts"]["module_lineage"]["path"] = f"{clinical_server}/{module_name}"
    clinical["artifacts"]["module_lineage"]["sha256"] = module_digest
    source_name = "SOURCE_INPUTS.json"
    if clinical_expression_server_root is None:
        source_digest = _copy(clinical_source_root / source_name, clinical_dir / source_name)
    else:
        source_inputs = _load(clinical_source_root / source_name)
        source_inputs["inputs"]["current_patient_logcpm"]["path"] = (
            clinical_expression_server_root
        )
        source_bytes = _encoded(source_inputs)
        _write(clinical_dir / source_name, source_bytes)
        source_digest = hashlib.sha256(source_bytes).hexdigest()
        clinical["authorities"]["current_patient_logcpm"]["path"] = (
            clinical_expression_server_root
        )
        clinical["artifacts"]["curves"]["path"] = clinical_curves_server_root
    clinical["artifacts"]["source_inputs"]["path"] = f"{clinical_server}/{source_name}"
    clinical["artifacts"]["source_inputs"]["sha256"] = source_digest
    clinical_bytes = _encoded(clinical)
    clinical_path = clinical_dir / "CLINICAL_KM_BINDING.json"
    _write(clinical_path, clinical_bytes)
    clinical_sha = hashlib.sha256(clinical_bytes).hexdigest()
    _write(
        clinical_dir / "SUCCESS.json",
        _encoded(_success(clinical, clinical_path.name, clinical_sha, "clinical")),
    )
    records["clinical"] = {
        "binding_path": f"{clinical_server}/{clinical_path.name}",
        "binding_sha256": clinical_sha,
        "scientific_status": "secondary_fresh_lncrna_survival_statistics",
        "production_deployed": False,
        "release_ready": False,
        "clean_directory_runtime_view_bound": clinical_expression_server_root is not None,
    }

    # State: close both result metadata and the two missing source JSON leaves.
    state_dir = output / "state_gene_set"
    state_server = f"{server_root}/state_gene_set"
    state_source_root = overlay / "artifacts/v32_state_gene_sets_20260826_r2_code_bound"
    state = _load(state_source_root / "STATE_GENE_SET_BINDING.json")
    for role, name in (("module_lineage", "MODULE_LINEAGE.json"), ("report_manifest", "STATE_REPORT_MANIFEST.json")):
        digest = _copy(state_source_root / name, state_dir / name)
        state["artifacts"][role]["path"] = f"{state_server}/{name}"
        state["artifacts"][role]["sha256"] = digest
    # These two source-authority hashes are compile-time formal constants in
    # ``state_gene_set_query``.  Use the byte-identical original authority
    # JSON, not the portable rewrite (whose internal path rewrite necessarily
    # changes its hash), because the query loader only needs a relocated copy
    # of the frozen authority bytes.
    state_release = overlay.parent / "v32_full_multitask/state_release_complete"
    for role, source_name, target_name in (
        ("state_checkpoint_manifest", "CHECKPOINT_MANIFEST.json", "SOURCE_CHECKPOINT_MANIFEST.json"),
        ("state_module_lineage", "MODULE_LINEAGE.json", "SOURCE_STATE_MODULE_LINEAGE.json"),
    ):
        digest = _copy(state_release / source_name, state_dir / target_name)
        state["source_artifacts"][role]["path"] = f"{state_server}/{target_name}"
        state["source_artifacts"][role]["sha256"] = digest
    state_bytes = _encoded(state)
    state_path = state_dir / "STATE_GENE_SET_BINDING.json"
    _write(state_path, state_bytes)
    state_sha = hashlib.sha256(state_bytes).hexdigest()
    _write(
        state_dir / "SUCCESS.json",
        _encoded(_success(state, state_path.name, state_sha, "state")),
    )
    records["state_gene_set"] = {
        "binding_path": f"{state_server}/{state_path.name}",
        "binding_sha256": state_sha,
        "scientific_status": "secondary_state_gene_set_and_report",
        "production_deployed": False,
        "release_ready": False,
    }

    # Direction probabilities already have a complete independently audited
    # artifact.  Rebind only the audit's references to the copied metadata.
    direction_dir = output / "evidence_direction"
    direction_server = f"{server_root}/evidence_direction"
    release_source = overlay / "artifacts/v32_evidence_direction_probabilities_20260826_r1_fixed_seed_reinference/EVIDENCE_DIRECTION_PROBABILITY_BINDING.json"
    report_source = overlay / "artifacts/v32_evidence_direction_probabilities_20260826_r3_independent_audit/EVIDENCE_DIRECTION_INDEPENDENT_AUDIT.json"
    audit_source = overlay / "artifacts/v32_evidence_direction_probabilities_20260826_r3_independent_audit/INDEPENDENT_AUDIT_BINDING.json"
    release_sha = _copy(release_source, direction_dir / release_source.name)
    report_sha = _copy(report_source, direction_dir / report_source.name)
    audit = _load(audit_source)
    audit["release_binding"]["path"] = f"{direction_server}/{release_source.name}"
    audit["release_binding"]["sha256"] = release_sha
    audit["report"]["path"] = f"{direction_server}/{report_source.name}"
    audit["report"]["sha256"] = report_sha
    audit_bytes = _encoded(audit)
    audit_path = direction_dir / audit_source.name
    _write(audit_path, audit_bytes)
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    records["evidence_direction"] = {
        "binding_path": f"{direction_server}/{release_source.name}",
        "binding_sha256": release_sha,
        "audit_binding_path": f"{direction_server}/{audit_source.name}",
        "audit_binding_sha256": audit_sha,
        "scientific_status": "auxiliary_direction_probability",
        "production_deployed": False,
        "release_ready": False,
    }

    receipt = {
        "format": "CANCERLNCATLAS_V32_INDEPENDENT_HEAD_RUNTIME_BINDINGS_V1",
        "status": "SERVER_UPLOAD_PENDING_RUNTIME_VALIDATION",
        "server_root": server_root,
        "bindings": records,
        "gene_set_ranked_subtype": {
            "status": "DIRECT_PAYLOAD_VERIFIED_TRANSITIVE_METADATA_PENDING",
            "production_deployed": False,
            "release_ready": False,
        },
        "main_score_changed": False,
        "production_deployed": False,
        "release_ready": False,
    }
    _write(output / "RUNTIME_BINDINGS.json", _encoded(receipt))
    return receipt


__all__ = ["RuntimeBindingError", "build_runtime_bindings"]
