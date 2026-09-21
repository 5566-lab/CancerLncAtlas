"""Validated release semantics for the intentionally partial V3.2 scRNA scope.

This policy separates the 33-cancer website universe from the subset for which
patient/donor-aware single-cell evidence can be generated without inventing
metadata.  The remaining cancers are typed unavailable, never negative or zero.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping


FORMAT = "CANCERLNCATLAS_V32_SINGLE_CELL_RELEASE_SCOPE_V1"
STATUS = "ACCEPTED_PARTIAL_FORMAL_SCOPE"
POLICY_ID = "V32_SC_FORMAL23_TYPED10_20260903_R2"
DEFAULT_POLICY = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "v32_single_cell_release_scope_20260903_r2.json"
)
FORMAL_ELIGIBLE_CANCERS = frozenset(
    {
        "ACC", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA", "GBM",
        "HNSC", "KIRC", "LAML", "LGG", "LUSC", "MESO", "OV", "PCPG",
        "READ", "SARC", "SKCM", "TGCT", "THYM", "UCEC", "UVM",
    }
)
TYPED_UNAVAILABLE_CANCERS = frozenset(
    {"BLCA", "KICH", "KIRP", "LIHC", "LUAD", "PAAD", "PRAD", "STAD", "THCA", "UCS"}
)


class SingleCellReleaseScopeError(RuntimeError):
    """Raised when the accepted scope is internally inconsistent."""


def validate_single_cell_release_scope(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if result.get("format") != FORMAT or result.get("status") != STATUS:
        raise SingleCellReleaseScopeError("single-cell release-scope format/status drift")
    if result.get("policy_id") != POLICY_ID:
        raise SingleCellReleaseScopeError("single-cell release-scope policy identity drift")
    eligible = tuple(str(x).upper() for x in result.get("formal_eligible_cancers", ()))
    unavailable_raw = result.get("typed_unavailable")
    if not isinstance(unavailable_raw, Mapping):
        raise SingleCellReleaseScopeError("typed_unavailable must be a cancer-to-reason mapping")
    unavailable = {str(k).upper(): str(v) for k, v in unavailable_raw.items()}
    if len(eligible) != 23 or len(set(eligible)) != 23:
        raise SingleCellReleaseScopeError("formal eligible scope must contain 23 unique cancers")
    if len(unavailable) != 10 or any(not reason for reason in unavailable.values()):
        raise SingleCellReleaseScopeError("typed unavailable scope must contain 10 explicit reasons")
    if set(eligible) & set(unavailable):
        raise SingleCellReleaseScopeError("eligible and typed-unavailable scopes overlap")
    if set(eligible) != FORMAL_ELIGIBLE_CANCERS:
        raise SingleCellReleaseScopeError("formal eligible cancer identities drift")
    if set(unavailable) != TYPED_UNAVAILABLE_CANCERS:
        raise SingleCellReleaseScopeError("typed-unavailable cancer identities drift")
    if len(set(eligible) | set(unavailable)) != 33:
        raise SingleCellReleaseScopeError("single-cell release policy must partition 33 cancers")
    if int(result.get("declared_cancer_universe_count", -1)) != 33:
        raise SingleCellReleaseScopeError("declared cancer universe is not 33")
    if int(result.get("formal_eligible_cancer_count", -1)) != len(eligible):
        raise SingleCellReleaseScopeError("formal eligible count drift")
    if int(result.get("typed_unavailable_cancer_count", -1)) != len(unavailable):
        raise SingleCellReleaseScopeError("typed unavailable count drift")
    if result.get("full_33_single_cell_coverage_claimed") is not False:
        raise SingleCellReleaseScopeError("policy must not claim full-33 single-cell coverage")
    if result.get("full_33_single_cell_completion_is_release_gate") is not False:
        raise SingleCellReleaseScopeError("full-33 completion must not be a release gate")
    semantics = result.get("release_semantics")
    if not isinstance(semantics, Mapping):
        raise SingleCellReleaseScopeError("release_semantics missing")
    required = {
        "eligible_outputs_required_for_single_cell_module_available": 23,
        "typed_unavailable_rows_must_remain_null": True,
        "typed_unavailable_rows_must_not_reduce_primary_score": True,
        "single_cell_is_an_optional_independent_head": True,
        "repeat_full33_rescue_as_standalone_release_blocker": False,
    }
    drift = {key: semantics.get(key) for key, value in required.items() if semantics.get(key) != value}
    if drift:
        raise SingleCellReleaseScopeError(f"single-cell release semantics drift: {drift}")
    evidence = result.get("evidence_basis")
    if not isinstance(evidence, Mapping):
        raise SingleCellReleaseScopeError("evidence_basis missing")
    success_rel = evidence.get("rescued_contract_success")
    success_sha = str(evidence.get("rescued_contract_success_sha256", "")).lower()
    if not success_rel or len(success_sha) != 64:
        raise SingleCellReleaseScopeError("rescued contract hash binding missing")
    success_path = Path(__file__).resolve().parents[2] / str(success_rel)
    if not success_path.is_file():
        raise SingleCellReleaseScopeError("rescued contract local copy missing")
    observed_sha = hashlib.sha256(success_path.read_bytes()).hexdigest()
    if observed_sha != success_sha:
        raise SingleCellReleaseScopeError("rescued contract local copy hash drift")
    success = json.loads(success_path.read_text(encoding="utf-8"))
    if (
        success.get("status") != "SUCCESS"
        or success.get("formal_eligible_cancer_count") != 23
        or success.get("typed_unavailable_cancer_count") != 10
        or success.get("dataset_manifest_sha256") != evidence.get("dataset_manifest_sha256")
        or success.get("preflight_sha256") != evidence.get("server_preflight_sha256")
        or success.get("run_status_sha256") != evidence.get("run_status_sha256")
    ):
        raise SingleCellReleaseScopeError("rescued contract evidence disagrees with policy")
    result["formal_eligible_cancers"] = list(eligible)
    result["typed_unavailable"] = unavailable
    return result


def load_single_cell_release_scope(path: str | Path = DEFAULT_POLICY) -> dict[str, Any]:
    source = Path(path).resolve()
    return validate_single_cell_release_scope(json.loads(source.read_text(encoding="utf-8")))


def final_binding_scope_is_available(module: Mapping[str, Any]) -> bool:
    """Accept full-33 data or the frozen 23+10 typed-unavailable policy."""

    if str(module.get("status", "")).strip().lower() != "available":
        return False
    if module.get("fresh_recompute") is not True:
        return False
    if module.get("historical_assets_relabelled_fresh") is not False:
        return False
    if int(module.get("cancer_count", -1)) != 33:
        return False
    fresh = int(module.get("fresh_derived_cancer_count", -1))
    if fresh == 33:
        return True
    return bool(
        module.get("scope_policy_id") == POLICY_ID
        and fresh == 23
        and int(module.get("formal_eligible_cancer_count", -1)) == 23
        and int(module.get("typed_unavailable_cancer_count", -1)) == 10
        and module.get("full_33_single_cell_coverage_claimed") is False
        and module.get("typed_unavailable_rows_are_null") is True
        and module.get("typed_unavailable_changes_primary_score") is False
    )


__all__ = [
    "DEFAULT_POLICY",
    "POLICY_ID",
    "SingleCellReleaseScopeError",
    "final_binding_scope_is_available",
    "load_single_cell_release_scope",
    "validate_single_cell_release_scope",
]
