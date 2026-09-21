"""Fail-closed V3.2 historical capability parity contract.

The parity config describes what must exist.  A release evidence manifest
attests what was newly trained and published.  Passing this validator means
that every required historical capability has a V3.2 lineage, non-empty
artifacts, and every applicable API/UI/download surface.  It deliberately
does not infer availability from a column name or from an older release.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CONFIG_SCHEMA_VERSION = "CANCERLNCATLAS_V32_HISTORICAL_CAPABILITY_PARITY_V1"
EVIDENCE_SCHEMA_VERSION = "CANCERLNCATLAS_V32_CAPABILITY_EVIDENCE_V1"
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "v32_historical_capability_parity.yaml"
)

GATE_KINDS = ("artifact", "api", "ui", "download")
REUSABLE_INPUT_ROLES = frozenset(
    {
        "raw_source_data",
        "provenance_audited_standardized_source",
        "identifier_crosswalk",
        "pathway_membership",
        "task_definition",
        "split_definition",
    }
)
FORBIDDEN_DERIVED_INPUT_ROLES = frozenset(
    {
        "checkpoint",
        "prediction",
        "probability",
        "ranking",
        "embedding",
        "processed_training_example",
        "fitted_fold_feature",
        "web_table",
        "cache",
    }
)

EXPECTED_STATE_PROGRAMS = {
    "state_rnass": ("stemness_rna::RNAss", True),
    "state_dnass": ("stemness_dna::DNAss", True),
    "state_extend": ("EXTEND::published_score", True),
    "state_ereg_expss": ("stemness_rna::EREG.EXPss", True),
    "state_dmpss": ("stemness_dna::DMPss", False),
    "state_enhss": ("stemness_dna::ENHss", False),
    "state_ereg_methss": ("stemness_dna::EREG-METHss", False),
}
EXPECTED_CLINICAL_ENDPOINTS = frozenset({"OS", "DSS", "PFI", "PFS", "DFI", "DFS"})
EXPECTED_CAPABILITY_IDS = frozenset(
    {
        "exact_pathway",
        "gene_set",
        "ranked_subtype",
        "network",
        *EXPECTED_STATE_PROGRAMS,
        "clinical",
        "expression_landscape",
        "survival_kaplan_meier",
        "bulk_coexpression",
        "external_validation",
        "continuous_pathway_activity",
        "single_cell",
        "mutation",
        "cnv",
        "drug",
        "physical_interaction",
        "experiment_perturbation",
        "evidence_transformer",
        "mixed_lncrna_protein_pathway_query",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_LEGACY_VERSION_RE = re.compile(
    r"(?:^|[^a-z0-9])v(?:2(?:[._-]?\d+)*|3[._-]?[01])(?:[^0-9]|$)",
    flags=re.IGNORECASE,
)
_LEGACY_WORD_RE = re.compile(
    r"(?:^|[/\\._-])(?:legacy|historical|old)(?:[/\\._-]|$)",
    flags=re.IGNORECASE,
)
_DERIVED_KEY_RE = re.compile(
    r"checkpoint|prediction|probabilit|ranking|embedding|processed.*example|"
    r"fitted.*feature|web.*table|cache",
    flags=re.IGNORECASE,
)
_LEGACY_DERIVED_FLAG_RE = re.compile(
    r"(?:old|legacy|historical|reused?).*"
    r"(?:checkpoint|prediction|probabilit|ranking|embedding|web.*table|cache)|"
    r"(?:checkpoint|prediction|probabilit|ranking|embedding|web.*table|cache).*"
    r"(?:old|legacy|historical|reused?)",
    flags=re.IGNORECASE,
)
_UNAVAILABLE_STATES = frozenset(
    {"unavailable", "not_available", "missing", "not_measured", "not_applicable"}
)
_UNAVAILABLE_VALUE_KEYS = frozenset(
    {
        "value",
        "score",
        "probability",
        "fill_value",
        "sentinel",
        "unavailable_value",
        "unavailable_fill_value",
    }
)


class CapabilityParityError(RuntimeError):
    """Raised when a V3.2 release fails historical capability parity."""


@dataclass(frozen=True)
class ParityValidationReport:
    schema_version: str
    release_id: str
    model_version: str
    status: str
    capability_count: int
    checked_capabilities: tuple[str, ...]
    diagnostic_only_capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CapabilityParityError(f"{label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise CapabilityParityError(f"{label} must be an array")
    return value


def _nonempty_strings(value: Any, label: str) -> list[str]:
    values = _sequence(value, label)
    normalized = [str(item).strip() for item in values]
    if not normalized or any(not item for item in normalized):
        raise CapabilityParityError(f"{label} must contain non-empty identifiers")
    if len(normalized) != len(set(normalized)):
        raise CapabilityParityError(f"{label} contains duplicate identifiers")
    return normalized


def _read_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise CapabilityParityError(f"Parity file is missing: {source}")
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CapabilityParityError(f"Invalid JSON parity file: {source}") from exc
    else:
        try:
            import yaml
        except ModuleNotFoundError as exc:  # pragma: no cover - declared dependency
            raise CapabilityParityError("PyYAML is required for the parity contract") from exc
        try:
            payload = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise CapabilityParityError(f"Invalid YAML parity file: {source}") from exc
    if not isinstance(payload, Mapping):
        raise CapabilityParityError(f"Expected a mapping in parity file: {source}")
    return dict(payload)


def load_parity_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the canonical parity checklist."""

    config = _read_mapping(path)
    validate_parity_config(config)
    return config


def _coerce_mapping(value: Mapping[str, Any] | str | Path, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    try:
        return _read_mapping(value)
    except TypeError as exc:
        raise CapabilityParityError(f"{label} must be a mapping or file path") from exc


def validate_parity_config(config: Mapping[str, Any]) -> None:
    """Validate the checklist itself before it is used as a release gate."""

    root = _mapping(config, "parity config")
    if root.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise CapabilityParityError("Parity config schema_version mismatch")
    if str(root.get("release_model_version", "")).upper() != "V3.2":
        raise CapabilityParityError("Parity config must require V3.2")
    if root.get("release_target_level") != "exact_pathway":
        raise CapabilityParityError("Parity config must require exact_pathway")

    lineage = _mapping(root.get("lineage_policy"), "lineage_policy")
    required_true = (
        "fail_closed",
        "require_new_training",
        "require_from_scratch_or_new_v32_head",
        "require_exact_pathway_source_for_exact_outputs",
    )
    for key in required_true:
        if lineage.get(key) is not True:
            raise CapabilityParityError(f"lineage_policy.{key} must be true")
    if lineage.get("family_to_exact_broadcast") != "forbidden":
        raise CapabilityParityError("family-to-exact broadcast must be forbidden")
    if lineage.get("unavailable_encoding") != "null_with_reason":
        raise CapabilityParityError("Unavailable values must use null_with_reason")
    sentinels = set(_sequence(lineage.get("forbidden_unavailable_sentinels"), "forbidden sentinels"))
    if not {0.0, 0.5}.issubset(sentinels):
        raise CapabilityParityError("Unavailable sentinel policy must forbid both 0 and 0.5")
    reusable = set(_nonempty_strings(lineage.get("reusable_input_roles"), "reusable input roles"))
    forbidden = set(
        _nonempty_strings(lineage.get("forbidden_derived_input_roles"), "forbidden input roles")
    )
    if not REUSABLE_INPUT_ROLES.issubset(reusable):
        raise CapabilityParityError("Parity config omits reusable source-input roles")
    if not FORBIDDEN_DERIVED_INPUT_ROLES.issubset(forbidden):
        raise CapabilityParityError("Parity config does not forbid all historical derived roles")

    performance = _mapping(root.get("performance_policy"), "performance_policy")
    if performance.get("no_increment_status") != "DIAGNOSTIC_ONLY":
        raise CapabilityParityError("NO_INCREMENT must map to DIAGNOSTIC_ONLY")
    if performance.get("capability_removal_allowed") is not False:
        raise CapabilityParityError("Performance policy must forbid capability removal")
    outcomes = set(_nonempty_strings(performance.get("allowed_outcomes"), "allowed outcomes"))
    if "NO_INCREMENT" not in outcomes:
        raise CapabilityParityError("Performance policy must retain NO_INCREMENT")

    state_rows = _sequence(root.get("required_state_programs"), "required_state_programs")
    observed_states: dict[str, tuple[str, bool]] = {}
    for index, row_value in enumerate(state_rows):
        row = _mapping(row_value, f"required_state_programs[{index}]")
        capability_id = str(row.get("capability_id", "")).strip()
        state_id = str(row.get("state_id", "")).strip()
        if not capability_id or not state_id:
            raise CapabilityParityError("Every state program needs state_id and capability_id")
        if capability_id in observed_states:
            raise CapabilityParityError(f"Duplicate state capability: {capability_id}")
        observed_states[capability_id] = (state_id, row.get("mainline_required") is True)
    if observed_states != EXPECTED_STATE_PROGRAMS:
        raise CapabilityParityError(
            f"Historical seven-State contract mismatch: expected {EXPECTED_STATE_PROGRAMS}, "
            f"observed {observed_states}"
        )

    endpoints = set(
        _nonempty_strings(root.get("required_clinical_endpoints"), "required_clinical_endpoints")
    )
    if not EXPECTED_CLINICAL_ENDPOINTS.issubset(endpoints):
        missing = sorted(EXPECTED_CLINICAL_ENDPOINTS - endpoints)
        raise CapabilityParityError(f"Clinical endpoint contract is incomplete: {missing}")

    capabilities = _mapping(root.get("capabilities"), "capabilities")
    missing = sorted(EXPECTED_CAPABILITY_IDS - set(capabilities))
    if missing:
        raise CapabilityParityError(f"Parity config is missing required capabilities: {missing}")
    for capability_id in sorted(EXPECTED_CAPABILITY_IDS):
        capability = _mapping(capabilities[capability_id], f"capabilities.{capability_id}")
        if capability.get("required") is not True:
            raise CapabilityParityError(f"Capability {capability_id} must be required")
        target_level = str(capability.get("target_level", "")).strip()
        if not target_level:
            raise CapabilityParityError(f"Capability {capability_id} lacks target_level")
        training = _mapping(
            capability.get("new_training"), f"capabilities.{capability_id}.new_training"
        )
        if training.get("required") is not True:
            raise CapabilityParityError(f"Capability {capability_id} must require new_training")
        initializations = set(
            _nonempty_strings(
                training.get("allowed_initializations"),
                f"capabilities.{capability_id}.new_training.allowed_initializations",
            )
        )
        global_initializations = set(
            _nonempty_strings(
                lineage.get("allowed_initializations"),
                "lineage_policy.allowed_initializations",
            )
        )
        if not initializations.issubset(global_initializations):
            raise CapabilityParityError(
                f"Capability {capability_id} permits an initialization outside V3.2 policy"
            )
        gates = _mapping(capability.get("gates"), f"capabilities.{capability_id}.gates")
        for gate_kind in GATE_KINDS:
            gate = _mapping(gates.get(gate_kind), f"{capability_id}.{gate_kind} gate")
            if not isinstance(gate.get("required"), bool):
                raise CapabilityParityError(
                    f"Capability {capability_id} must declare whether {gate_kind} is required"
                )
            if gate.get("required") is True:
                _nonempty_strings(gate.get("ids"), f"{capability_id}.{gate_kind}.ids")
        artifact_gate = _mapping(gates["artifact"], f"{capability_id}.artifact gate")
        if artifact_gate.get("required") is not True:
            raise CapabilityParityError(f"Capability {capability_id} needs an artifact gate")
        if artifact_gate.get("require_nonempty") is not True:
            raise CapabilityParityError(f"Capability {capability_id} artifacts must be non-empty")

    for capability_id, (state_id, mainline_required) in EXPECTED_STATE_PROGRAMS.items():
        capability = _mapping(capabilities[capability_id], capability_id)
        if capability.get("state_id") != state_id:
            raise CapabilityParityError(f"State identity mismatch for {capability_id}")
        if (capability.get("mainline_required") is True) != mainline_required:
            raise CapabilityParityError(f"State mainline requirement mismatch for {capability_id}")
    clinical = _mapping(capabilities["clinical"], "capabilities.clinical")
    if not EXPECTED_CLINICAL_ENDPOINTS.issubset(
        set(_nonempty_strings(clinical.get("endpoints"), "clinical.endpoints"))
    ):
        raise CapabilityParityError("Clinical capability does not expose every historical endpoint")
    survival = _mapping(
        capabilities["survival_kaplan_meier"],
        "capabilities.survival_kaplan_meier",
    )
    if not EXPECTED_CLINICAL_ENDPOINTS.issubset(
        set(_nonempty_strings(survival.get("endpoints"), "survival_kaplan_meier.endpoints"))
    ):
        raise CapabilityParityError(
            "Kaplan-Meier capability does not expose every historical endpoint"
        )
    continuous = _mapping(
        capabilities["continuous_pathway_activity"],
        "capabilities.continuous_pathway_activity",
    )
    if continuous.get("independent_from_exact_primary") is not True:
        raise CapabilityParityError(
            "Continuous pathway activity must remain independent from the exact-pathway primary"
        )
    evidence = _mapping(capabilities["evidence_transformer"], "evidence_transformer")
    if evidence.get("confidence_separate_from_primary") is not True:
        raise CapabilityParityError("Evidence confidence must remain separate from primary ranking")


def _is_v32(value: Any) -> bool:
    normalized = str(value or "").strip().upper().replace("_", ".").replace("-", ".")
    return normalized == "V3.2" or normalized.startswith("V3.2.")


def _is_positive_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _is_forbidden_sentinel(value: Any, sentinels: set[float]) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return any(math.isclose(numeric, sentinel, rel_tol=0.0, abs_tol=1e-12) for sentinel in sentinels)


def _walk(value: Any, path: str = "$" ):
    yield path, value
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")


def _reject_legacy_derived_assets(manifest: Mapping[str, Any]) -> None:
    for path, value in _walk(manifest):
        key = path.rsplit(".", 1)[-1]
        if _LEGACY_DERIVED_FLAG_RE.search(key) and value not in (False, None, "", [], {}):
            raise CapabilityParityError(f"Legacy derived asset flag is truthy at {path}")
        if not isinstance(value, str) or not _DERIVED_KEY_RE.search(key):
            continue
        if _LEGACY_VERSION_RE.search(value) or _LEGACY_WORD_RE.search(value):
            raise CapabilityParityError(f"Legacy checkpoint/prediction reference at {path}: {value}")


def _reject_unavailable_sentinels(
    manifest: Mapping[str, Any], sentinels: set[float]
) -> None:
    for path, value in _walk(manifest):
        if not isinstance(value, Mapping):
            continue
        status = str(value.get("availability", value.get("status", ""))).strip().lower()
        unavailable = value.get("available") is False or status in _UNAVAILABLE_STATES
        for key in _UNAVAILABLE_VALUE_KEYS:
            if key not in value:
                continue
            candidate = value[key]
            if _is_forbidden_sentinel(candidate, sentinels):
                raise CapabilityParityError(
                    f"Unavailable value is disguised as sentinel {candidate!r} at {path}.{key}"
                )
            if unavailable and candidate is not None:
                raise CapabilityParityError(
                    f"Unavailable value must be null, not {candidate!r}, at {path}.{key}"
                )


def _reject_family_broadcast(manifest: Mapping[str, Any]) -> None:
    for path, value in _walk(manifest):
        if not isinstance(value, Mapping):
            continue
        for key in ("family_to_exact_broadcast", "family_broadcast", "broadcast_from_family"):
            if key in value and value[key] not in (False, None, "", "forbidden"):
                raise CapabilityParityError(f"Family score broadcast into exact pathway at {path}.{key}")
        target_level = str(value.get("target_level", "")).strip().lower()
        source_level = str(value.get("source_target_level", "")).strip().lower()
        derivation = str(value.get("derivation", "")).strip().lower()
        if target_level == "exact_pathway" and source_level in {"family", "pathway_family"}:
            raise CapabilityParityError(f"Exact output uses a pathway-family source at {path}")
        if target_level == "exact_pathway" and "family" in derivation and "broadcast" in derivation:
            raise CapabilityParityError(f"Exact output is a family broadcast at {path}")


def _validate_new_training(
    capability_id: str,
    observed: Mapping[str, Any],
    required: Mapping[str, Any],
    reusable_roles: set[str],
) -> None:
    training = _mapping(observed.get("new_training"), f"{capability_id}.new_training evidence")
    if training.get("completed") is not True:
        raise CapabilityParityError(f"Capability {capability_id} lacks completed V3.2 new training")
    if not _is_v32(training.get("model_version")):
        raise CapabilityParityError(f"Capability {capability_id} training lineage is not V3.2")
    if not str(training.get("run_id", "")).strip():
        raise CapabilityParityError(f"Capability {capability_id} new training lacks run_id")
    initialization = str(training.get("initialization", "")).strip()
    allowed = set(
        _nonempty_strings(
            _mapping(required.get("new_training"), f"{capability_id}.new_training config").get(
                "allowed_initializations"
            ),
            f"{capability_id}.allowed_initializations",
        )
    )
    if initialization not in allowed:
        raise CapabilityParityError(
            f"Capability {capability_id} initialization {initialization!r} is not allowed"
        )
    if training.get("new_parameters_from_scratch") is not True:
        raise CapabilityParityError(
            f"Capability {capability_id} must train its V3.2 parameters from scratch"
        )
    if initialization == "v32_core_frozen_new_head" and not _is_v32(
        training.get("core_model_version")
    ):
        raise CapabilityParityError(f"Capability {capability_id} does not reference a V3.2 core")
    digest = str(training.get("checkpoint_sha256", ""))
    if not _SHA256_RE.fullmatch(digest):
        raise CapabilityParityError(f"Capability {capability_id} lacks a valid new checkpoint hash")
    legacy_inputs = training.get("legacy_derived_inputs", [])
    if list(_sequence(legacy_inputs, f"{capability_id}.legacy_derived_inputs")):
        raise CapabilityParityError(f"Capability {capability_id} reuses legacy derived inputs")
    input_roles = set(
        _nonempty_strings(training.get("input_roles"), f"{capability_id}.input_roles")
    )
    unknown_roles = input_roles - reusable_roles
    if unknown_roles:
        raise CapabilityParityError(
            f"Capability {capability_id} uses forbidden/unknown input roles: {sorted(unknown_roles)}"
        )


def _artifact_path(root: Path, value: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise CapabilityParityError(f"Artifact escapes release root: {value}") from exc
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _validate_artifacts(
    capability_id: str,
    observed: Mapping[str, Any],
    required: Mapping[str, Any],
    *,
    release_root: Path | None,
) -> None:
    gates = _mapping(required.get("gates"), f"{capability_id}.gates")
    gate = _mapping(gates.get("artifact"), f"{capability_id}.artifact gate")
    expected_ids = set(_nonempty_strings(gate.get("ids"), f"{capability_id}.artifact ids"))
    artifacts = _mapping(observed.get("artifacts"), f"{capability_id}.artifacts")
    missing = sorted(expected_ids - set(artifacts))
    if missing:
        raise CapabilityParityError(f"Capability {capability_id} is missing artifacts: {missing}")
    target_level = str(required.get("target_level"))
    for artifact_id in sorted(expected_ids):
        artifact = _mapping(artifacts[artifact_id], f"{capability_id}.{artifact_id}")
        if artifact.get("present") is not True:
            raise CapabilityParityError(f"Artifact {capability_id}.{artifact_id} is not present")
        if not str(artifact.get("path", "")).strip():
            raise CapabilityParityError(f"Artifact {capability_id}.{artifact_id} lacks path")
        digest = str(artifact.get("sha256", ""))
        if not _SHA256_RE.fullmatch(digest):
            raise CapabilityParityError(f"Artifact {capability_id}.{artifact_id} lacks SHA-256")
        if gate.get("require_nonempty") is True and not (
            _is_positive_number(artifact.get("rows"))
            or _is_positive_number(artifact.get("bytes"))
        ):
            raise CapabilityParityError(f"Artifact {capability_id}.{artifact_id} is empty")
        if not _is_v32(artifact.get("model_version")):
            raise CapabilityParityError(f"Artifact {capability_id}.{artifact_id} is not V3.2")
        if artifact.get("provenance") != "v32_new_training":
            raise CapabilityParityError(
                f"Artifact {capability_id}.{artifact_id} lacks V3.2 new-training provenance"
            )
        if artifact.get("target_level") != target_level:
            raise CapabilityParityError(
                f"Artifact {capability_id}.{artifact_id} target_level mismatch"
            )
        if artifact.get("availability_encoding") != "null_with_reason":
            raise CapabilityParityError(
                f"Artifact {capability_id}.{artifact_id} must encode unavailable as null_with_reason"
            )
        if "unavailable_fill_value" not in artifact or artifact["unavailable_fill_value"] is not None:
            raise CapabilityParityError(
                f"Artifact {capability_id}.{artifact_id} has a non-null unavailable fill"
            )
        if target_level == "exact_pathway":
            if artifact.get("source_target_level") != "exact_pathway":
                raise CapabilityParityError(
                    f"Artifact {capability_id}.{artifact_id} lacks exact-pathway source lineage"
                )
            if artifact.get("family_broadcast") is not False:
                raise CapabilityParityError(
                    f"Artifact {capability_id}.{artifact_id} may broadcast family scores"
                )
        if release_root is not None:
            path = _artifact_path(release_root, str(artifact["path"]))
            if not path.is_file():
                raise CapabilityParityError(f"Artifact file is missing: {path}")
            if path.stat().st_size <= 0:
                raise CapabilityParityError(f"Artifact file is empty: {path}")
            if _file_sha256(path).lower() != digest.lower():
                raise CapabilityParityError(f"Artifact hash mismatch: {path}")


def _validate_surface(
    capability_id: str,
    observed: Mapping[str, Any],
    required: Mapping[str, Any],
    gate_kind: str,
) -> None:
    gate = _mapping(
        _mapping(required.get("gates"), f"{capability_id}.gates").get(gate_kind),
        f"{capability_id}.{gate_kind} gate",
    )
    if gate.get("required") is not True:
        return
    evidence = _mapping(observed.get(gate_kind), f"{capability_id}.{gate_kind} evidence")
    if evidence.get("implemented") is not True:
        raise CapabilityParityError(f"Capability {capability_id} lacks {gate_kind} implementation")
    expected = set(_nonempty_strings(gate.get("ids"), f"{capability_id}.{gate_kind} required ids"))
    actual = set(
        _nonempty_strings(evidence.get("ids"), f"{capability_id}.{gate_kind} observed ids")
    )
    missing = sorted(expected - actual)
    if missing:
        raise CapabilityParityError(
            f"Capability {capability_id} is missing {gate_kind} surfaces: {missing}"
        )


def validate_capability_parity(
    config: Mapping[str, Any] | str | Path,
    release_manifest: Mapping[str, Any] | str | Path,
    *,
    release_root: str | Path | None = None,
) -> ParityValidationReport:
    """Validate a V3.2 release against every historical capability gate.

    When ``release_root`` is supplied, artifact paths are additionally checked
    for containment, existence, non-zero size and SHA-256 agreement.
    """

    config_value = _coerce_mapping(config, "config")
    validate_parity_config(config_value)
    manifest = _coerce_mapping(release_manifest, "release_manifest")
    if manifest.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise CapabilityParityError("Release evidence schema_version mismatch")
    if not _is_v32(manifest.get("model_version")):
        raise CapabilityParityError("Release evidence is not V3.2")
    if manifest.get("target_level") != "exact_pathway":
        raise CapabilityParityError("Release evidence target must be exact_pathway")
    release_id = str(manifest.get("release_id", "")).strip()
    if not release_id:
        raise CapabilityParityError("Release evidence lacks release_id")

    _reject_legacy_derived_assets(manifest)
    policy = _mapping(config_value["lineage_policy"], "lineage_policy")
    sentinels = {float(item) for item in policy["forbidden_unavailable_sentinels"]}
    _reject_unavailable_sentinels(manifest, sentinels)
    _reject_family_broadcast(manifest)

    required_capabilities = _mapping(config_value["capabilities"], "capabilities")
    observed_capabilities = _mapping(manifest.get("capabilities"), "release capabilities")
    required_ids = {
        capability_id
        for capability_id, value in required_capabilities.items()
        if isinstance(value, Mapping) and value.get("required") is True
    }
    missing = sorted(required_ids - set(observed_capabilities))
    if missing:
        raise CapabilityParityError(f"Release is missing required capabilities: {missing}")

    reusable_roles = set(policy["reusable_input_roles"])
    allowed_outcomes = set(config_value["performance_policy"]["allowed_outcomes"])
    diagnostic: list[str] = []
    root = Path(release_root).resolve() if release_root is not None else None
    for capability_id in sorted(required_ids):
        required = _mapping(required_capabilities[capability_id], capability_id)
        observed = _mapping(observed_capabilities[capability_id], f"release.{capability_id}")
        status = str(observed.get("status", "")).strip().upper()
        if status not in {"READY", "DIAGNOSTIC_ONLY"}:
            raise CapabilityParityError(
                f"Required capability {capability_id} has unavailable/disabled status {status!r}"
            )
        performance = _mapping(observed.get("performance"), f"{capability_id}.performance")
        outcome = str(performance.get("outcome", "")).strip().upper()
        if outcome not in allowed_outcomes:
            raise CapabilityParityError(
                f"Capability {capability_id} has invalid performance outcome {outcome!r}"
            )
        if outcome == "NO_INCREMENT" and status != "DIAGNOSTIC_ONLY":
            raise CapabilityParityError(
                f"Capability {capability_id} has NO_INCREMENT and must be DIAGNOSTIC_ONLY"
            )
        if status == "DIAGNOSTIC_ONLY":
            diagnostic.append(capability_id)
        _validate_new_training(capability_id, observed, required, reusable_roles)
        _validate_artifacts(capability_id, observed, required, release_root=root)
        for gate_kind in ("api", "ui", "download"):
            _validate_surface(capability_id, observed, required, gate_kind)

    return ParityValidationReport(
        schema_version=EVIDENCE_SCHEMA_VERSION,
        release_id=release_id,
        model_version=str(manifest["model_version"]),
        status="PASS",
        capability_count=len(required_ids),
        checked_capabilities=tuple(sorted(required_ids)),
        diagnostic_only_capabilities=tuple(sorted(diagnostic)),
    )


# Explicit aliases make the intended release-gate use discoverable.
validate_release_manifest = validate_capability_parity
validate_release_parity = validate_capability_parity


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "EVIDENCE_SCHEMA_VERSION",
    "DEFAULT_CONFIG_PATH",
    "EXPECTED_CAPABILITY_IDS",
    "EXPECTED_CLINICAL_ENDPOINTS",
    "EXPECTED_STATE_PROGRAMS",
    "CapabilityParityError",
    "ParityValidationReport",
    "load_parity_config",
    "validate_parity_config",
    "validate_capability_parity",
    "validate_release_manifest",
    "validate_release_parity",
]
