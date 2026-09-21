"""Fail-closed registry for the non-production V3.2 website staging API.

The registry is the only bridge between model artifacts and the staging web
service.  A filename is never treated as provenance: every mounted artifact is
hashed, assigned to one of the eight V3.2 modules, and checked against a fresh
training lineage (or an explicit audited-unavailable record).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .full_model_contract import (
    MODULE_CONTRACTS,
    FullModelContractError,
    validate_module_lineage,
)


REGISTRY_SCHEMA_VERSION = "CancerLncAtlas_V3.2_staging_release_registry_v1"
V32_PREFIX = "CancerLncAtlas_V3.2"
MODULE_STATUSES = frozenset({"SUCCESS_NEWLY_TRAINED", "AUDITED_UNAVAILABLE"})
REQUIRED_MODULES = tuple(MODULE_CONTRACTS)
MIXED_QUERY_ROLES = (
    "mixed_query_identifier_map",
    "mixed_query_exact_pathway_membership",
    "mixed_query_v32_lnc_exact_association",
)
ALLOWED_ARTIFACT_KINDS = frozenset(
    {
        "static_annotation",
        "v32_public_prediction",
        "v32_public_statistical_result",
        "v32_public_metadata",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_OLD_GENERATION_RE = re.compile(
    r"(?:^|[\\/_\-.])(?:v2(?:[._-][0-9]+)?|v3[._-]?[01])(?:$|[\\/_\-.])",
    re.IGNORECASE,
)
_OLD_RESULT_RE = re.compile(
    r"checkpoint|prediction|probabilit|ranking|ranked|release|web[_-]?table|oof|embedding",
    re.IGNORECASE,
)


class ReleaseRegistryError(RuntimeError):
    """Raised when an artifact cannot be mounted as a current V3.2 result."""


@dataclass(frozen=True)
class ValidatedReleaseRegistry:
    """Validated registry plus resolved, hash-checked website artifact paths."""

    manifest: Mapping[str, Any]
    registry_path: Path
    registry_sha256: str
    artifacts: Mapping[str, Path]
    artifact_hashes: Mapping[str, str]

    @property
    def release_id(self) -> str:
        return str(self.manifest["release_id"])

    @property
    def analysis_version(self) -> str:
        return str(self.manifest["analysis_version"])

    def artifact(self, role: str) -> Path:
        try:
            return self.artifacts[role]
        except KeyError as exc:
            raise ReleaseRegistryError(f"Website artifact role is not registered: {role}") from exc


def artifact_sha256(path: str | Path) -> str:
    """Hash one file or a directory using a deterministic relative-path Merkle hash."""

    target = Path(path).resolve()
    if target.is_file():
        digest = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if target.is_dir():
        digest = hashlib.sha256()
        files = sorted(item for item in target.rglob("*") if item.is_file())
        if not files:
            raise ReleaseRegistryError(f"Registered artifact directory is empty: {target}")
        for item in files:
            relative = item.relative_to(target).as_posix().encode("utf-8")
            digest.update(relative)
            digest.update(b"\0")
            digest.update(bytes.fromhex(artifact_sha256(item)))
            digest.update(b"\0")
        return digest.hexdigest()
    raise ReleaseRegistryError(f"Registered artifact does not exist: {target}")


def _require_fields(value: Mapping[str, Any], fields: tuple[str, ...], context: str) -> None:
    missing = [field for field in fields if field not in value or value[field] in (None, "", [])]
    if missing:
        raise ReleaseRegistryError(f"{context} lacks required fields: {missing}")


def _require_v32(value: Any, context: str) -> str:
    text = str(value)
    if not text.startswith(V32_PREFIX):
        raise ReleaseRegistryError(f"{context} is not current V3.2: {text!r}")
    return text


def _require_sha256(value: Any, context: str) -> str:
    text = str(value).lower()
    if SHA256_RE.fullmatch(text) is None:
        raise ReleaseRegistryError(f"{context} is not a SHA256 digest")
    return text


def _resolve_path(raw: Any, base_dir: Path, context: str) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if not path.exists():
        raise ReleaseRegistryError(f"{context} does not exist: {path}")
    return path


def _verify_hashed_path(
    raw_path: Any,
    raw_hash: Any,
    *,
    base_dir: Path,
    context: str,
) -> tuple[Path, str]:
    path = _resolve_path(raw_path, base_dir, context)
    expected = _require_sha256(raw_hash, f"{context}.sha256")
    observed = artifact_sha256(path)
    if observed != expected:
        raise ReleaseRegistryError(
            f"{context} hash mismatch: expected={expected}, observed={observed}, path={path}"
        )
    return path, observed


def _read_json(path: Path, context: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseRegistryError(f"Cannot read {context}: {path}") from exc
    if not isinstance(value, Mapping):
        raise ReleaseRegistryError(f"{context} must contain a JSON object: {path}")
    return value


def _reject_old_result_path(path: Path, kind: str, context: str) -> None:
    text = path.as_posix()
    old_generation = _OLD_GENERATION_RE.search(text) is not None
    result_like = _OLD_RESULT_RE.search(text) is not None
    if old_generation and (
        result_like or kind in {"v32_public_prediction", "v32_public_statistical_result"}
    ):
        raise ReleaseRegistryError(
            f"{context} points at an old checkpoint/prediction/ranking path: {path}"
        )
    lowered = text.casefold()
    if kind in {"v32_public_prediction", "v32_public_statistical_result"} and any(
        token in lowered for token in ("/historical/", "\\historical\\", "/legacy/", "\\legacy\\")
    ):
        raise ReleaseRegistryError(f"{context} points at a historical result path: {path}")


def _validate_success_module(
    module_id: str,
    entry: Mapping[str, Any],
    *,
    base_dir: Path,
) -> None:
    _require_fields(
        entry,
        (
            "analysis_version",
            "training_run_id",
            "lineage_path",
            "lineage_sha256",
            "new_training_attestation",
            "old_checkpoint_loaded",
            "old_predictions_used_as_features",
            "old_rankings_used_as_outputs",
        ),
        f"module {module_id}",
    )
    _require_v32(entry["analysis_version"], f"module {module_id}.analysis_version")
    if entry["new_training_attestation"] is not True:
        raise ReleaseRegistryError(f"module {module_id} lacks a fresh-training attestation")
    for flag in (
        "old_checkpoint_loaded",
        "old_predictions_used_as_features",
        "old_rankings_used_as_outputs",
    ):
        if entry[flag] is not False:
            raise ReleaseRegistryError(f"module {module_id} did not exclude historical result reuse: {flag}")
    lineage_path, _ = _verify_hashed_path(
        entry["lineage_path"],
        entry["lineage_sha256"],
        base_dir=base_dir,
        context=f"module {module_id} lineage",
    )
    _reject_old_result_path(lineage_path, "v32_public_metadata", f"module {module_id} lineage")
    lineage = _read_json(lineage_path, f"module {module_id} lineage")
    try:
        validate_module_lineage(module_id, lineage)
    except FullModelContractError as exc:
        raise ReleaseRegistryError(f"module {module_id} failed the V3.2 lineage contract: {exc}") from exc
    if str(lineage.get("training_status", "")).upper() != "SUCCESS":
        raise ReleaseRegistryError(
            f"module {module_id} is marked SUCCESS_NEWLY_TRAINED but its lineage is unavailable"
        )
    if str(lineage.get("statistical_training_status", "")).upper().startswith("UNAVAILABLE_"):
        raise ReleaseRegistryError(
            f"module {module_id} is marked SUCCESS_NEWLY_TRAINED but statistical training is unavailable"
        )
    if str(lineage.get("analysis_version")) != str(entry["analysis_version"]):
        raise ReleaseRegistryError(f"module {module_id} registry/lineage analysis version mismatch")
    if str(lineage.get("training_run_id")) != str(entry["training_run_id"]):
        raise ReleaseRegistryError(f"module {module_id} registry/lineage training run mismatch")


def _validate_unavailable_module(
    module_id: str,
    entry: Mapping[str, Any],
    *,
    base_dir: Path,
) -> None:
    _require_fields(
        entry,
        ("analysis_version", "reason_code", "lineage_path", "lineage_sha256"),
        f"module {module_id}",
    )
    _require_v32(entry["analysis_version"], f"module {module_id}.analysis_version")
    lineage_path, _ = _verify_hashed_path(
        entry["lineage_path"],
        entry["lineage_sha256"],
        base_dir=base_dir,
        context=f"module {module_id} unavailable lineage",
    )
    lineage = _read_json(lineage_path, f"module {module_id} unavailable lineage")
    try:
        validate_module_lineage(module_id, lineage)
    except FullModelContractError as exc:
        raise ReleaseRegistryError(
            f"module {module_id} failed the V3.2 unavailable-lineage contract: {exc}"
        ) from exc
    unavailable = (
        str(lineage.get("training_status", "")).upper() == "AUDITED_UNAVAILABLE"
        or str(lineage.get("statistical_training_status", "")).upper().startswith("UNAVAILABLE_")
    )
    if not unavailable:
        raise ReleaseRegistryError(
            f"module {module_id} is marked AUDITED_UNAVAILABLE but its lineage reports trained results"
        )
    if str(lineage.get("analysis_version")) != str(entry["analysis_version"]):
        raise ReleaseRegistryError(f"module {module_id} unavailable registry/lineage version mismatch")
    if any(
        lineage.get(flag) is not False
        for flag in (
            "old_checkpoint_loaded",
            "old_predictions_used_as_features",
            "old_rankings_used_as_outputs",
        )
    ):
        raise ReleaseRegistryError(f"module {module_id} unavailable lineage did not exclude old results")


def validate_release_registry(
    manifest: Mapping[str, Any],
    *,
    base_dir: str | Path,
    registry_path: str | Path | None = None,
    require_staging: bool = True,
) -> ValidatedReleaseRegistry:
    """Validate all eight modules and every artifact exposed to the website."""

    if not isinstance(manifest, Mapping):
        raise ReleaseRegistryError("Release registry must be a JSON object")
    _require_fields(
        manifest,
        (
            "schema_version",
            "release_id",
            "analysis_version",
            "environment",
            "production_deployed",
            "release_ready",
            "all_non_null_predictions_newly_trained_v32",
            "all_published_results_generated_in_v32",
            "modules",
            "website_artifacts",
            "capabilities",
        ),
        "release registry",
    )
    if manifest["schema_version"] != REGISTRY_SCHEMA_VERSION:
        raise ReleaseRegistryError("Release registry schema version mismatch")
    _require_v32(manifest["analysis_version"], "release registry analysis_version")
    if require_staging:
        if str(manifest["environment"]).lower() != "staging":
            raise ReleaseRegistryError("The staging loader accepts only environment=staging")
        if manifest["production_deployed"] is not False:
            raise ReleaseRegistryError("The staging registry may not claim production deployment")
    if manifest["all_non_null_predictions_newly_trained_v32"] is not True:
        raise ReleaseRegistryError(
            "The registry must attest that every non-null prediction is newly trained V3.2"
        )
    if manifest["all_published_results_generated_in_v32"] is not True:
        raise ReleaseRegistryError(
            "The registry must attest that every published result was generated in V3.2"
        )
    if not isinstance(manifest["release_ready"], bool):
        raise ReleaseRegistryError("release_ready must be boolean")
    if "supersedes" in manifest:
        supersedes = manifest["supersedes"]
        if (
            not isinstance(supersedes, list)
            or not supersedes
            or any(not isinstance(item, str) or not item.strip() for item in supersedes)
            or not str(manifest.get("supersession_reason", "")).strip()
        ):
            raise ReleaseRegistryError(
                "supersedes requires non-empty artifact identifiers and a supersession_reason"
            )

    base = Path(base_dir).resolve()
    modules = manifest["modules"]
    if not isinstance(modules, Mapping):
        raise ReleaseRegistryError("release registry modules must be a mapping")
    missing = sorted(set(REQUIRED_MODULES) - set(modules))
    extra = sorted(set(modules) - set(REQUIRED_MODULES))
    if missing or extra:
        raise ReleaseRegistryError(f"release registry module mismatch: missing={missing}, extra={extra}")
    for module_id in REQUIRED_MODULES:
        entry = modules[module_id]
        if not isinstance(entry, Mapping):
            raise ReleaseRegistryError(f"module {module_id} entry must be a mapping")
        status = str(entry.get("status", ""))
        if status not in MODULE_STATUSES:
            raise ReleaseRegistryError(f"module {module_id} has invalid status: {status!r}")
        if str(entry.get("analysis_version")) != str(manifest["analysis_version"]):
            raise ReleaseRegistryError(
                f"module {module_id} is not bound to the registry analysis version"
            )
        if status == "SUCCESS_NEWLY_TRAINED":
            _validate_success_module(module_id, entry, base_dir=base)
        else:
            _validate_unavailable_module(module_id, entry, base_dir=base)
    unavailable_modules = [
        module_id
        for module_id, entry in modules.items()
        if entry["status"] == "AUDITED_UNAVAILABLE"
    ]
    if unavailable_modules and manifest["release_ready"] is not False:
        raise ReleaseRegistryError(
            "A registry containing audited-unavailable modules must have release_ready=false: "
            + repr(sorted(unavailable_modules))
        )

    website_artifacts = manifest["website_artifacts"]
    if not isinstance(website_artifacts, list):
        raise ReleaseRegistryError("website_artifacts must be a list")
    resolved: dict[str, Path] = {}
    observed_hashes: dict[str, str] = {}
    for index, artifact in enumerate(website_artifacts):
        if not isinstance(artifact, Mapping):
            raise ReleaseRegistryError(f"website artifact {index} must be a mapping")
        _require_fields(
            artifact,
            ("role", "path", "sha256", "artifact_kind", "generation", "source_module"),
            f"website artifact {index}",
        )
        role = str(artifact["role"])
        if role in resolved:
            raise ReleaseRegistryError(f"duplicate website artifact role: {role}")
        kind = str(artifact["artifact_kind"])
        if kind not in ALLOWED_ARTIFACT_KINDS:
            raise ReleaseRegistryError(f"website artifact {role} has invalid kind: {kind}")
        _require_v32(artifact["generation"], f"website artifact {role}.generation")
        if str(artifact["generation"]) != str(manifest["analysis_version"]):
            raise ReleaseRegistryError(f"website artifact {role} generation differs from the registry")
        source_module = str(artifact["source_module"])
        if source_module not in modules:
            raise ReleaseRegistryError(f"website artifact {role} has unknown source module")
        if kind in {"v32_public_prediction", "v32_public_statistical_result"} and modules[
            source_module
        ]["status"] != "SUCCESS_NEWLY_TRAINED":
            raise ReleaseRegistryError(
                f"website prediction {role} is attached to a module that was not newly trained"
            )
        path, observed = _verify_hashed_path(
            artifact["path"],
            artifact["sha256"],
            base_dir=base,
            context=f"website artifact {role}",
        )
        _reject_old_result_path(path, kind, f"website artifact {role}")
        resolved[role] = path
        observed_hashes[role] = observed

    capabilities = manifest["capabilities"]
    if not isinstance(capabilities, Mapping):
        raise ReleaseRegistryError("capabilities must be a mapping")
    mixed = capabilities.get("mixed_exact_pathway_query")
    if not isinstance(mixed, Mapping) or "enabled" not in mixed:
        raise ReleaseRegistryError("mixed_exact_pathway_query capability is not declared")
    if mixed["enabled"] is True:
        if modules["exact_pathway"]["status"] != "SUCCESS_NEWLY_TRAINED":
            raise ReleaseRegistryError("mixed query requires a newly trained V3.2 exact-pathway module")
        missing_roles = sorted(set(MIXED_QUERY_ROLES) - set(resolved))
        if missing_roles:
            raise ReleaseRegistryError(f"mixed query lacks website artifacts: {missing_roles}")
        association = next(
            item for item in website_artifacts if item["role"] == "mixed_query_v32_lnc_exact_association"
        )
        if association["artifact_kind"] != "v32_public_prediction":
            raise ReleaseRegistryError("lnc exact-pathway association must be a V3.2 public prediction")
        if association["source_module"] != "exact_pathway":
            raise ReleaseRegistryError("lnc exact-pathway association must come from exact_pathway")
        asset_manifest_role = "mixed_query_asset_manifest"
        if asset_manifest_role not in resolved:
            raise ReleaseRegistryError(
                "An enabled mixed query requires a hash-registered asset manifest"
            )
        if asset_manifest_role in resolved:
            asset_manifest = _read_json(
                resolved[asset_manifest_role], "mixed exact-pathway asset manifest"
            )
            exact_module = modules["exact_pathway"]
            exact_lineage_path = _resolve_path(
                exact_module["lineage_path"], base, "exact-pathway lineage"
            )
            exact_lineage = _read_json(exact_lineage_path, "exact-pathway lineage")
            if (
                asset_manifest.get("status") != "SUCCESS"
                or asset_manifest.get("analysis_version") != manifest["analysis_version"]
                or asset_manifest.get("training_run_id") != exact_module.get("training_run_id")
                or asset_manifest.get("pathway_target_level") != "exact_pathway"
                or asset_manifest.get("historical_predictions_used") is not False
                or asset_manifest.get("historical_rankings_used") is not False
                or asset_manifest.get("historical_checkpoints_used") is not False
                or asset_manifest.get("static_annotation_only_no_predictions_or_rankings")
                is not True
                or asset_manifest.get("pathway_family_broadcast") is not False
                or asset_manifest.get("membership_restricted_to_current_v32_exact_pathways")
                is not True
            ):
                raise ReleaseRegistryError(
                    "The mixed exact-pathway asset manifest failed its V3.2 provenance contract"
                )
            key_coverage = asset_manifest.get("key_coverage")
            if (
                not isinstance(key_coverage, Mapping)
                or key_coverage.get("format")
                != "CANCERLNCATLAS_V32_MIXED_QUERY_KEY_COVERAGE_V1"
                or key_coverage.get("scope") not in {
                    "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_CARTESIAN",
                    "ELIGIBLE_CANCER_LNCRNA_X_EXACT_PATHWAY_OBSERVED_ROWS",
                }
                or key_coverage.get("complete_key_coverage") is not True
                or key_coverage.get("missing_key_count") != 0
                or key_coverage.get("duplicate_key_count") != 0
                or key_coverage.get("unscored_key_encoding")
                != "NULL_WITH_TYPED_NOT_EVALUATED_REASON"
                or key_coverage.get("eligible_pair_authority_sha256")
                != asset_manifest.get("source_exact_association_sha256")
            ):
                raise ReleaseRegistryError(
                    "The mixed asset manifest lacks a complete typed key-coverage contract"
                )
            if asset_manifest.get("source_exact_lineage_sha256") != exact_module[
                "lineage_sha256"
            ]:
                raise ReleaseRegistryError(
                    "The mixed asset manifest is not bound to the registered exact-pathway lineage"
                )
            if asset_manifest.get("source_exact_association_sha256") != exact_lineage.get(
                "prediction_sha256"
            ):
                raise ReleaseRegistryError(
                    "The mixed asset manifest is not bound to the registered exact-pathway ensemble"
                )
            declared_assets = asset_manifest.get("artifacts")
            if not isinstance(declared_assets, Mapping):
                raise ReleaseRegistryError("The mixed asset manifest lacks its artifact hash mapping")
            for role in (*MIXED_QUERY_ROLES, "mixed_query_pathway_metadata"):
                if role not in resolved:
                    continue
                declaration = declared_assets.get(role)
                if not isinstance(declaration, Mapping):
                    raise ReleaseRegistryError(
                        f"The mixed asset manifest does not declare registered role {role}"
                    )
                if declaration.get("sha256") != observed_hashes[role]:
                    raise ReleaseRegistryError(
                        f"The mixed asset manifest hash differs for registered role {role}"
                    )
    elif mixed["enabled"] is not False:
        raise ReleaseRegistryError("mixed_exact_pathway_query.enabled must be boolean")

    clinical = capabilities.get("clinical")
    if not isinstance(clinical, Mapping):
        raise ReleaseRegistryError("clinical capability layers are not declared")
    patient_risk = clinical.get("patient_risk")
    if not isinstance(patient_risk, Mapping):
        raise ReleaseRegistryError("clinical.patient_risk capability is not declared")
    patient_role = str(patient_risk.get("artifact_role", ""))
    if patient_risk.get("status") == "STAGING_ARTIFACT_REGISTERED" and patient_role not in resolved:
        raise ReleaseRegistryError("Registered clinical patient-risk artifact is missing")
    entity = clinical.get("entity_association")
    if not isinstance(entity, Mapping) or "enabled" not in entity:
        raise ReleaseRegistryError("clinical.entity_association capability is not declared")
    if entity["enabled"] is True:
        if entity.get("status") != "STAGING_ARTIFACT_REGISTERED":
            raise ReleaseRegistryError("Enabled clinical entity association lacks registered status")
        entity_role = str(entity.get("artifact_role", ""))
        validation_role = str(entity.get("validation_role", ""))
        lineage_role = str(entity.get("lineage_role", ""))
        missing_clinical_roles = sorted(
            {entity_role, validation_role, lineage_role} - set(resolved)
        )
        if missing_clinical_roles:
            raise ReleaseRegistryError(
                f"Clinical entity association lacks registered artifacts: {missing_clinical_roles}"
            )
        validation = _read_json(resolved[validation_role], "clinical entity validation")
        lineage = _read_json(resolved[lineage_role], "clinical entity lineage")
        if (
            validation.get("status") != "PASS"
            or validation.get("all_non_null_results_newly_computed_v32") is not True
            or validation.get("historical_results_used") is not False
            or validation.get("changes_primary_ranking") is not False
        ):
            raise ReleaseRegistryError("Clinical entity association failed its dedicated public contract")
        if not str(lineage.get("analysis_version", "")).startswith(V32_PREFIX):
            raise ReleaseRegistryError("Clinical entity association lineage is not V3.2")
        if (
            lineage.get("training_status") != "SUCCESS"
            or lineage.get("old_checkpoint_loaded") is not False
            or lineage.get("old_predictions_used_as_features") is not False
            or lineage.get("old_rankings_used_as_outputs") is not False
        ):
            raise ReleaseRegistryError("Clinical entity association lineage permits historical results")
        if validation.get("summary_sha256") != observed_hashes[entity_role]:
            raise ReleaseRegistryError("Clinical entity public artifact hash differs from validation")
        if validation.get("lineage_sha256") != observed_hashes[lineage_role]:
            raise ReleaseRegistryError("Clinical entity lineage hash differs from validation")
    elif entity["enabled"] is not False:
        raise ReleaseRegistryError("clinical.entity_association.enabled must be boolean")

    resolved_registry_path = (
        Path(registry_path).resolve() if registry_path is not None else base / "<in-memory-registry>"
    )
    registry_hash = artifact_sha256(resolved_registry_path) if resolved_registry_path.is_file() else (
        hashlib.sha256(
            json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )
    return ValidatedReleaseRegistry(
        manifest=dict(manifest),
        registry_path=resolved_registry_path,
        registry_sha256=registry_hash,
        artifacts=resolved,
        artifact_hashes=observed_hashes,
    )


def load_release_registry(path: str | Path, *, require_staging: bool = True) -> ValidatedReleaseRegistry:
    """Read and validate a registry, resolving relative paths beside its JSON file."""

    registry_path = Path(path).resolve()
    manifest = _read_json(registry_path, "release registry")
    return validate_release_registry(
        manifest,
        base_dir=registry_path.parent,
        registry_path=registry_path,
        require_staging=require_staging,
    )


__all__ = [
    "ALLOWED_ARTIFACT_KINDS",
    "MIXED_QUERY_ROLES",
    "MODULE_STATUSES",
    "REGISTRY_SCHEMA_VERSION",
    "REQUIRED_MODULES",
    "ReleaseRegistryError",
    "ValidatedReleaseRegistry",
    "artifact_sha256",
    "load_release_registry",
    "validate_release_registry",
]
