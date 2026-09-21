"""Implementation-level, fail-closed V3.2 staging completeness audit.

Unlike the earlier catalog-parity audit, this evaluator keeps three states
separate: ``CATALOG_DECLARED``, ``IMPLEMENTED``, and ``SERVABLE``.  A required
gate passes only when implementation and runtime evidence are both present.
Typed gaps and expected 503 responses are useful evidence, but never PASS.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from .capability_parity import EXPECTED_CAPABILITY_IDS, load_parity_config
from .unified_staging_bindings import create_app_from_unified_bindings


REPORT_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_AUDIT_V2"
BINDING_FORMAT = "CANCERLNCATLAS_V32_INTEGRATED_COMPLETENESS_STRICT_BINDING_V2"
MODEL_VERSION = "V3.2"
GATE_KINDS = ("artifact", "api", "ui", "download")
DOWNLOAD_PASS_STATUSES = frozenset({"READY_FILE", "READY_PARTS", "DYNAMIC_QUERY_EXPORT"})
REQUIRED_R4_DOWNLOAD_BINDING_SHA256 = (
    "e1d70c44dad440c1812943854e99331327915a07fb5dd0b58ee137021946a4af"
)
REQUIRED_R4_DOWNLOAD_AUDIT_SHA256 = (
    "cfa75dab38c2fd9e7a6823dd6fce9ddd62526715773955dbd978f888e2b53d41"
)
REQUIRED_R3_HISTORICAL_BINDING_SHA256 = (
    "b128681bff1856ab09df21c387013b431e0da956bce6f32243b1361661e0c4d8"
)
REQUIRED_R3_HISTORICAL_AUDIT_SHA256 = (
    "b652fcb0cdecddfd698983322c6b485025a4a656f0b86f7856b52ef20840e3ae"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_BLOCKING_STATUS_RE = re.compile(r"GAP|PARTIAL|PENDING|UNAVAILABLE|NOT_MATERIALIZED")


class StrictCompletenessError(RuntimeError):
    """Raised when source integrity is too weak to issue a strict audit."""


# A legacy/public API is considered implemented only through these explicit
# staging equivalents.  ``None`` deliberately means no equivalent was found.
API_EQUIVALENTS: dict[str, dict[str, tuple[str, str] | None]] = {
    "exact_pathway": {
        "GET /api/site/stats": ("GET", "/v3.2-staging/exact-pathway/stats"),
        "GET /api/site/search": None,
        "GET /api/site/lncrna/{lncrna}": ("GET", "/v3.2-staging/exact-pathway/associations"),
        "GET /api/site/genesets": ("GET", "/v3.2-staging/gene-sets"),
        "GET /api/site/datasets": None,
        "GET /api/site/cancers": None,
        "GET /api/site/cancer/{cancer}": ("GET", "/v3.2-staging/exact-pathway/associations"),
    },
    "gene_set": {"GET /api/site/genesets": ("GET", "/v3.2-staging/gene-sets")},
    "ranked_subtype": {"GET /api/site/subtypes": ("GET", "/v3.2-staging/ranked-subtypes/overview")},
    "network": {"GET /api/site/network/{lncrna}": ("GET", "/v3.2-staging/network/lncrna/{lncrna_id}")},
    "clinical": {
        "GET /api/site/clinical/endpoints": ("GET", "/v3.2-staging/clinical"),
        "GET /api/site/clinical/lncrna/{lncrna}": ("GET", "/v3.2-staging/clinical"),
        "GET /api/site/clinical/priority": ("GET", "/v3.2-staging/clinical/translational-priority"),
    },
    "expression_landscape": {
        "GET /api/site/lncrna/{lncrna}/expression": ("GET", "/v3.2-staging/lncrna/{lncrna_id}/expression"),
        "GET /api/site/lncrna/{lncrna}/coverage": ("GET", "/v3.2-staging/lncrna/{lncrna_id}/coverage"),
    },
    "survival_kaplan_meier": {"GET /api/site/lncrna/{lncrna}/survival": ("GET", "/v3.2-staging/lncrna/{lncrna_id}/survival")},
    "bulk_coexpression": {"GET /api/site/lncrna/{lncrna}/coexpression": ("GET", "/v3.2-staging/lncrna/{lncrna_id}/coexpression")},
    "external_validation": {
        "GET /api/site/validation/external": ("GET", "/v3.2-staging/validation/external"),
        "GET /api/site/lncrna/{lncrna}/external-validation": ("GET", "/v3.2-staging/lncrna/{lncrna_id}/external-validation"),
    },
    "continuous_pathway_activity": {"GET /api/site/activity/continuous/{cancer}": ("GET", "/v3.2-staging/activity/continuous/{cancer_id}")},
    "single_cell": {
        "GET /api/site/single-cell/{lncrna}": ("GET", "/v3.2-staging/single-cell/exact-pathway-associations"),
        "GET /api/site/sc-activity/{cancer}": None,
        "GET /api/site/sc-trajectory/{cancer}": ("GET", "/v3.2-staging/single-cell/hnsc/pseudotime-availability"),
        "GET /api/site/sc-figures/{cancer}": None,
        "GET /api/site/sc-figure/{cancer}/{figure}": None,
    },
    "mutation": {
        "GET /api/site/mutation/{cancer}": ("GET", "/v3.2-staging/genomic"),
        "POST /api/site/predict/mutation-context": None,
    },
    "cnv": {"GET /api/site/cnv/{cancer}": ("GET", "/v3.2-staging/genomic")},
    "drug": {
        "GET /api/site/drug/{lncrna}": ("GET", "/v3.2-staging/drug/response-actionability"),
        "GET /api/site/drug/{lncrna}/mechanisms": ("GET", "/v3.2-staging/drug/structural-mechanisms"),
        "GET /api/site/cancer/{cancer}": ("GET", "/v3.2-staging/drug/response-actionability"),
    },
    "physical_interaction": {
        "GET /api/site/lncrna/{lncrna}/relationships": ("GET", "/v3.2-staging/interaction/relationships"),
        "GET /api/site/lncrna/{lncrna}/interaction-pathways": ("GET", "/v3.2-staging/interaction/exact-pathway-enrichment"),
        "GET /api/site/relationship/{relationship_id}/evidence": ("GET", "/v3.2-staging/interaction/evidence"),
        "GET /api/site/network/{lncrna}": ("GET", "/v3.2-staging/network/lncrna/{lncrna_id}"),
    },
    "experiment_perturbation": {
        "GET /api/site/lncrna/{lncrna}/relationships": ("GET", "/v3.2-staging/experiment/perturbation/evidence-bridge"),
        "GET /api/site/relationship/{relationship_id}/evidence": ("GET", "/v3.2-staging/experiment/perturbation/evidence-bridge/events"),
    },
    "evidence_transformer": {"GET /api/site/lncrna/{lncrna}/relationships": ("GET", "/v3.2-staging/evidence/confidence")},
    "mixed_lncrna_protein_pathway_query": {
        "POST /api/site/query/mixed-lncrna-protein-pathway": ("POST", "/v3.2-staging/enrichment/mixed-exact-pathway"),
        "POST /api/site/query/custom-gene-set-lncrna": ("POST", "/v3.2-staging/enrichment/mixed-exact-pathway"),
        "POST /api/site/query/protein-set-lncrna": ("POST", "/v3.2-staging/enrichment/mixed-exact-pathway"),
    },
}
for _state_capability in (
    "state_rnass", "state_dnass", "state_extend", "state_ereg_expss",
    "state_dmpss", "state_enhss", "state_ereg_methss",
):
    API_EQUIVALENTS[_state_capability] = {
        "GET /api/site/states/{state_id}": ("GET", "/v3.2-staging/state")
    }


CAPABILITY_UI_ACTIONS: dict[str, tuple[str, ...]] = {
    "exact_pathway": ("exact",), "gene_set": ("gene-set",),
    "ranked_subtype": ("subtype",), "network": ("network",),
    "clinical": ("clinical", "clinical-priority"),
    "expression_landscape": ("expression",),
    "survival_kaplan_meier": ("survival",),
    "bulk_coexpression": ("coexpression",),
    "external_validation": ("validation",),
    "continuous_pathway_activity": ("activity",),
    "single_cell": ("sc", "sc-expression", "sc-audit", "sc-gaps", "ucell", "pseudotime", "sc-figures"),
    "mutation": ("mutation", "mutation-subgroup"),
    "cnv": ("cnv", "cnv-coverage"),
    "drug": ("drug", "drug-mechanism"),
    "physical_interaction": ("physical", "interaction-evidence"),
    "experiment_perturbation": ("experiment",),
    "evidence_transformer": ("evidence", "evidence-direction"),
    "mixed_lncrna_protein_pathway_query": ("mixed",),
}
for _state_capability in (
    "state_rnass", "state_dnass", "state_extend", "state_ereg_expss",
    "state_dmpss", "state_enhss", "state_ereg_methss",
):
    CAPABILITY_UI_ACTIONS[_state_capability] = ("state", "state-gene-set")


# Explicit aliases are used only when the contract ID is not itself present in
# an authority document.  Selectors point into a hash-mounted JSON authority.
ARTIFACT_ALIASES: dict[str, tuple[Any, ...]] = {
    "v32_exact_pathway_model": ("exact_model",),
    "v32_exact_pathway_predictions": ("record", "gene_set_ranked_subtype", "artifacts", "exact_pathway_associations"),
    # No alias for v32_exact_pathway_report_manifest: the current tree has a
    # model audit/report, but no hash-bound artifact with the contracted role.
    "v32_ranked_subtypes": ("record", "gene_set_ranked_subtype", "artifacts", "ranked_subtypes"),
    "v32_subtype_stability": ("record", "gene_set_ranked_subtype", "artifacts", "subtype_stability"),
    "v32_network_nodes": ("record", "unified_network", "artifacts", "v32_network_nodes.parquet"),
    "v32_network_edges": ("record", "unified_network", "artifacts", "v32_network_edges.parquet"),
    "v32_state_oof": ("record", "state_gene_sets", "source_artifacts", "state_predictions"),
    "v32_lncrna_state_release": ("record", "state_gene_sets", "source_artifacts", "state_predictions"),
    "v32_state_gene_set_gmt": ("record", "state_gene_sets", "artifacts", "gmt"),
    "v32_state_report_manifest": ("record", "state_gene_sets", "artifacts", "report_manifest"),
    "v32_bulk_lncrna_expression_summary": ("record", "bulk_expression", "artifacts", "expression_summary"),
    "v32_lncrna_model_coverage": ("record", "bulk_expression", "artifacts", "expression_summary"),
    "v32_kaplan_meier_curves": ("download", "kaplan_meier_curves"),
    "v32_kaplan_meier_statistics": ("record", "clinical_kaplan_meier", "artifacts", "statistics"),
    "v32_tumor_lncrna_gene_coexpression": ("download", "tumor_lncrna_gene_coexpression"),
    "v32_coexpression_clusters": ("download", "coexpression_clusters"),
    "v32_external_validation_metrics": ("record", "external_validation", "artifacts", "metrics"),
    "v32_external_validation_details": ("record", "external_validation", "artifacts", "details"),
    "v32_external_validation_overlap_audit": ("record", "external_validation", "artifacts", "overlap_audit"),
    "v32_drug_response_predictions": ("download", "drug_response_predictions"),
    "v32_drug_replication": ("download", "drug_evidence"),
    "v32_drug_evidence_provenance": ("download", "drug_evidence"),
    "v32_lncrna_protein_drug_mechanism": ("download", "drug_mechanisms"),
    "v32_physical_interactions": ("record", "physical_interaction", "artifacts", "physical_interaction_relationships.parquet"),
    "v32_interaction_confidence": ("record", "physical_interaction", "artifacts", "physical_interaction_relationships.parquet"),
    "v32_interaction_provenance": ("record", "physical_interaction", "artifacts", "relationship_evidence.parquet"),
    "v32_interaction_pathway_enrichment": ("record", "physical_interaction", "artifacts", "interaction_exact_pathway_enrichment.parquet"),
    "v32_evidence_direction_probability": ("record", "evidence_direction_probabilities", "artifact"),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _safe_path(path: str | Path, root: Path, label: str, *, file_only: bool = True) -> Path:
    requested = Path(path)
    if requested.is_symlink():
        raise StrictCompletenessError(f"{label} may not be a symlink")
    resolved = requested.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise StrictCompletenessError(f"{label} escapes repository: {resolved}") from exc
    valid = resolved.is_file() if file_only else resolved.exists()
    if not valid or (resolved.is_file() and resolved.stat().st_size <= 0):
        raise StrictCompletenessError(f"{label} is missing or empty: {resolved}")
    return resolved


def _resolve(value: Any, root: Path, base: Path, label: str, *, file_only: bool = True) -> Path:
    raw = Path(str(value or ""))
    return _safe_path(raw if raw.is_absolute() else base / raw, root, label, file_only=file_only)


def _read_json(path: str | Path, root: Path, label: str) -> tuple[Path, dict[str, Any]]:
    source = _safe_path(path, root, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StrictCompletenessError(f"{label} is invalid JSON: {source}") from exc
    if not isinstance(value, dict):
        raise StrictCompletenessError(f"{label} must be a JSON object")
    return source, value


def _verify_declaration(declaration: Mapping[str, Any], root: Path, base: Path, label: str) -> Path:
    source = _resolve(declaration.get("path"), root, base, label)
    expected = str(declaration.get("sha256", "")).lower()
    if not _SHA256_RE.fullmatch(expected) or sha256_file(source) != expected:
        raise StrictCompletenessError(f"{label} SHA256 drift")
    if "bytes" in declaration and declaration.get("bytes") not in (None, source.stat().st_size):
        raise StrictCompletenessError(f"{label} byte-count drift")
    return source


def _source_record(path: Path, root: Path) -> dict[str, Any]:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _selector(value: Mapping[str, Any], keys: Sequence[str]) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


class _FrontendParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.query_actions: set[str] = set()
        self.scripts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if values.get("data-query"):
            self.query_actions.add(str(values["data-query"]))
        if tag == "script" and values.get("src"):
            self.scripts.append(str(values["src"]))


def _js_has_handler(source: str, action: str) -> bool:
    quoted = re.search(rf"[\"']{re.escape(action)}[\"']\s*:\s*", source)
    bare = re.search(rf"(?:^|\n)\s*(?:async\s+)?{re.escape(action)}\s*(?::|\()", source)
    return bool(quoted or bare)


def _collect_authority_documents(
    *,
    root: Path,
    unified: Mapping[str, Any],
    download_catalog: Mapping[str, Any],
    required_artifact_ids: set[str],
) -> tuple[
    dict[str, tuple[Path, dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, Path],
    dict[str, dict[str, Any]],
]:
    authorities: dict[str, tuple[Path, dict[str, Any]]] = {}
    all_documents: dict[str, Path] = {}
    binding_states: dict[str, dict[str, Any]] = {}
    bindings = unified.get("bindings")
    if not isinstance(bindings, Mapping):
        raise StrictCompletenessError("Unified bindings map is missing")
    for authority_id, raw in sorted(bindings.items()):
        if not isinstance(raw, Mapping):
            raise StrictCompletenessError(f"Binding {authority_id} is not an object")
        status = str(raw.get("status", ""))
        if status != "MOUNTED_HASH_PINNED":
            binding_states[str(authority_id)] = {
                "status": "PENDING_OR_UNMOUNTED",
                "declared_status": status or None,
                "reason": raw.get("reason"),
                "path": None,
            }
            continue
        path = _verify_declaration(raw, root, root, f"binding {authority_id}")
        _, payload = _read_json(path, root, f"binding {authority_id}")
        is_independent_audit = (
            "independent_audit" in str(authority_id).lower()
            or authority_id == "experiment_evidence_bridge_audit"
        )
        if is_independent_audit:
            audit_status = str(payload.get("status", "")).upper()
            failed = payload.get("fail_count", payload.get("failed_checks", 0))
            if not audit_status.startswith("PASS") or failed not in (0, None):
                raise StrictCompletenessError(
                    f"Independent audit binding {authority_id} does not report PASS"
                )
        authorities[str(authority_id)] = (path, payload)
        all_documents[f"unified_binding:{authority_id}"] = path
        binding_states[str(authority_id)] = {
            "status": "HASH_BOUND",
            "declared_status": status,
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }

    catalog_authorities = download_catalog.get("authorities")
    if not isinstance(catalog_authorities, Mapping):
        raise StrictCompletenessError("Download catalog authority map is missing")
    for authority_id, raw in sorted(catalog_authorities.items()):
        if not isinstance(raw, Mapping):
            raise StrictCompletenessError(f"Catalog authority {authority_id} is invalid")
        path = _verify_declaration(raw, root, root, f"catalog authority {authority_id}")
        if path.suffix.lower() != ".json":
            continue
        _, payload = _read_json(path, root, f"catalog authority {authority_id}")
        key = f"catalog:{authority_id}"
        authorities.setdefault(key, (path, payload))
        all_documents[f"catalog_authority:{authority_id}"] = path

    artifact_index: dict[str, list[dict[str, Any]]] = {
        artifact_id: [] for artifact_id in required_artifact_ids
    }
    seen: set[tuple[str, Path]] = set()

    def visit(authority_id: str, path: Path, payload: Any, depth: int) -> None:
        resolved = path.resolve()
        marker = (authority_id, resolved)
        if marker in seen or depth > 5:
            return
        seen.add(marker)
        all_documents.setdefault(
            f"authority_document:{authority_id}:{len(all_documents):04d}", resolved
        )

        def walk(value: Any, json_path: str) -> None:
            if isinstance(value, Mapping):
                for key, item in value.items():
                    child_path = f"{json_path}.{key}"
                    if key in required_artifact_ids:
                        artifact_index[str(key)].append(
                            {
                                "authority_id": authority_id,
                                "authority_document": resolved,
                                "json_path": child_path,
                                "record": item,
                            }
                        )
                    walk(item, child_path)
                record_path = value.get("path")
                record_sha = value.get("sha256")
                if record_path and record_sha:
                    raw_path = Path(str(record_path))
                    candidate = raw_path if raw_path.is_absolute() else resolved.parent / raw_path
                    if candidate.suffix.lower() == ".json" and candidate.exists():
                        try:
                            child = _verify_declaration(
                                value,
                                root,
                                resolved.parent,
                                f"transitive authority {authority_id}",
                            )
                            _, child_payload = _read_json(
                                child, root, f"transitive authority {authority_id}"
                            )
                        except StrictCompletenessError:
                            # A drifted incidental link is never indexed as evidence.
                            # If it is required, artifact resolution remains UNRESOLVED.
                            return
                        visit(authority_id, child, child_payload, depth + 1)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for index, item in enumerate(value):
                    walk(item, f"{json_path}[{index}]")

        walk(payload, "$")

    for authority_id, (path, payload) in sorted(authorities.items()):
        visit(authority_id, path, payload, 0)
    return authorities, artifact_index, all_documents, binding_states


def _artifact_state_from_status(status: Any) -> tuple[str | None, str | None]:
    normalized = str(status or "").upper()
    if not normalized:
        return None, None
    if "PENDING" in normalized:
        return "PENDING", normalized
    if _BLOCKING_STATUS_RE.search(normalized):
        return "TYPED_GAP", normalized
    return None, normalized


def _validate_file_record(
    record: Mapping[str, Any], *, root: Path, document: Path, label: str
) -> dict[str, Any]:
    blocked, declared_status = _artifact_state_from_status(record.get("status"))
    if blocked:
        return {
            "state": blocked,
            "declared_status": declared_status,
            "reason": record.get("unavailable_reason", record.get("reason")),
            "artifact_path": record.get("path"),
            "sha256": record.get("sha256"),
            "rows": record.get("rows"),
        }
    part_values = record.get("part_artifacts")
    if isinstance(part_values, Sequence) and not isinstance(part_values, (str, bytes)):
        if not part_values:
            return {"state": "UNRESOLVED", "reason": "empty part_artifacts"}
        validated_parts: list[dict[str, Any]] = []
        for index, part in enumerate(part_values):
            if not isinstance(part, Mapping):
                return {"state": "UNRESOLVED", "reason": "invalid part record"}
            source = _verify_declaration(
                part, root, document.parent, f"{label}.part[{index}]"
            )
            validated_parts.append(
                {
                    "path": source.relative_to(root).as_posix(),
                    "sha256": sha256_file(source),
                    "bytes": source.stat().st_size,
                }
            )
        return {
            "state": "HASH_BOUND",
            "declared_status": declared_status,
            "parts": len(validated_parts),
            "rows": record.get("rows"),
            "sha256_tree": record.get("sha256_tree"),
            "sample_parts": validated_parts[:3],
        }
    if record.get("path") and record.get("sha256"):
        raw = Path(str(record.get("path")))
        candidate = raw if raw.is_absolute() else document.parent / raw
        if candidate.resolve().is_dir():
            return {
                "state": "UNRESOLVED",
                "declared_status": declared_status,
                "reason": "directory declaration lacks independently enumerated parts",
                "artifact_path": str(record.get("path")),
            }
        source = _verify_declaration(record, root, document.parent, label)
        if record.get("rows") == 0:
            return {
                "state": "UNRESOLVED",
                "declared_status": declared_status,
                "reason": "required artifact declares zero rows",
                "artifact_path": source.relative_to(root).as_posix(),
                "sha256": sha256_file(source),
            }
        return {
            "state": "HASH_BOUND",
            "declared_status": declared_status,
            "artifact_path": source.relative_to(root).as_posix(),
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
            "rows": record.get("rows"),
        }
    return {
        "state": "UNRESOLVED",
        "declared_status": declared_status,
        "reason": "record has no verifiable file/parts declaration",
    }


def _download_artifact_record(download: Mapping[str, Any]) -> dict[str, Any]:
    status = str(download.get("status", ""))
    if status == "PENDING_FORMAL_SUCCESS":
        state = "PENDING"
    elif status in {"PARTIAL_READY_PARTS", "GAP_TYPED_UNAVAILABLE", "DATA_PRESENT_DOWNLOAD_NOT_IMPLEMENTED"}:
        state = "TYPED_GAP"
    elif status in {"READY_FILE", "READY_PARTS"}:
        state = "HASH_BOUND"
    else:
        state = "UNRESOLVED"
    return {
        "state": state,
        "authority_id": "download_catalog_independent_audit",
        "download_id": download.get("download_id"),
        "declared_status": status,
        "reason": download.get("known_gap", download.get("unavailable_reason")),
        "file_sha256": (
            download.get("file", {}).get("sha256")
            if isinstance(download.get("file"), Mapping)
            else None
        ),
        "part_count": len(download.get("parts", [])) if isinstance(download.get("parts"), list) else 0,
    }


def _resolve_exact_model(
    *, root: Path, registry_path: Path, registry: Mapping[str, Any]
) -> dict[str, Any]:
    module = registry.get("modules", {}).get("exact_pathway", {})
    if not isinstance(module, Mapping) or module.get("status") != "SUCCESS_NEWLY_TRAINED":
        return {"state": "UNRESOLVED", "reason": "exact-pathway registry module is not successful"}
    declaration = {
        "path": registry_path.parent / str(module.get("lineage_path", "")),
        "sha256": module.get("lineage_sha256"),
    }
    lineage_path = _verify_declaration(
        declaration, root, registry_path.parent, "exact-pathway lineage"
    )
    _, lineage = _read_json(lineage_path, root, "exact-pathway lineage")
    fold_count = lineage.get("folds")
    folds = lineage.get("ensemble_source_records")
    if fold_count != 5 or not isinstance(folds, list) or len(folds) != 5:
        return {"state": "UNRESOLVED", "reason": "exact-pathway five-fold model is incomplete"}
    checkpoints: list[dict[str, Any]] = []
    for index, fold in enumerate(folds):
        if not isinstance(fold, Mapping):
            return {"state": "UNRESOLVED", "reason": f"invalid exact fold {index}"}
        fold_id = fold.get("patient_fold", index)
        checkpoint = _verify_declaration(
            {"path": fold.get("checkpoint_path"), "sha256": fold.get("checkpoint_sha256")},
            root,
            lineage_path.parent,
            f"exact checkpoint fold {fold_id}",
        )
        checkpoints.append(
            {
                "fold": str(fold_id),
                "path": checkpoint.relative_to(root).as_posix(),
                "sha256": sha256_file(checkpoint),
                "bytes": checkpoint.stat().st_size,
            }
        )
    return {
        "state": "HASH_BOUND",
        "authority_id": "release_registry:exact_pathway",
        "lineage_path": lineage_path.relative_to(root).as_posix(),
        "lineage_sha256": sha256_file(lineage_path),
        "checkpoint_count": len(checkpoints),
        "checkpoints": checkpoints,
    }


def _resolve_artifact(
    artifact_id: str,
    *,
    root: Path,
    authorities: Mapping[str, tuple[Path, dict[str, Any]]],
    artifact_index: Mapping[str, list[dict[str, Any]]],
    downloads: Mapping[str, Mapping[str, Any]],
    registry_path: Path,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    alias = ARTIFACT_ALIASES.get(artifact_id)
    if alias:
        kind = str(alias[0])
        if kind == "exact_model":
            value = _resolve_exact_model(root=root, registry_path=registry_path, registry=registry)
            value["artifact_id"] = artifact_id
            value["resolution"] = "EXPLICIT_SPECIAL"
            return value
        if kind == "download":
            download_id = str(alias[1])
            value = _download_artifact_record(downloads[download_id])
            value["artifact_id"] = artifact_id
            value["resolution"] = "DOWNLOAD_CATALOG_HASH_AUDITED"
            return value
        if kind == "record":
            authority_id = str(alias[1])
            authority = authorities.get(authority_id)
            if authority is None:
                return {
                    "artifact_id": artifact_id,
                    "state": "PENDING" if authority_id.startswith("drug") else "UNRESOLVED",
                    "resolution": "EXPLICIT_SELECTOR",
                    "authority_id": authority_id,
                    "reason": "authority is not mounted",
                }
            document, payload = authority
            record = _selector(payload, [str(value) for value in alias[2:]])
            if not isinstance(record, Mapping):
                return {
                    "artifact_id": artifact_id,
                    "state": "UNRESOLVED",
                    "resolution": "EXPLICIT_SELECTOR",
                    "authority_id": authority_id,
                    "reason": "selector did not resolve to a record",
                }
            value = _validate_file_record(
                record, root=root, document=document, label=artifact_id
            )
            value.update(
                {
                    "artifact_id": artifact_id,
                    "resolution": "EXPLICIT_SELECTOR",
                    "authority_id": authority_id,
                    "authority_document": document.relative_to(root).as_posix(),
                    "json_path": ".".join(str(item) for item in alias[2:]),
                }
            )
            return value

    candidates = artifact_index.get(artifact_id, [])
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        record = candidate.get("record")
        if not isinstance(record, Mapping):
            continue
        value = _validate_file_record(
            record,
            root=root,
            document=Path(candidate["authority_document"]),
            label=artifact_id,
        )
        value.update(
            {
                "artifact_id": artifact_id,
                "resolution": "EXACT_CONTRACT_ID_IN_AUTHORITY",
                "authority_id": candidate["authority_id"],
                "authority_document": Path(candidate["authority_document"]).relative_to(root).as_posix(),
                "json_path": candidate["json_path"],
            }
        )
        evaluated.append(value)
    priority = {"HASH_BOUND": 0, "TYPED_GAP": 1, "PENDING": 2, "UNRESOLVED": 3}
    if evaluated:
        return sorted(evaluated, key=lambda item: priority.get(str(item.get("state")), 9))[0]
    return {
        "artifact_id": artifact_id,
        "state": "UNRESOLVED",
        "resolution": "NO_HASH_BOUND_CONTRACT_ID_OR_ALIAS",
        "reason": "required artifact ID was not resolved by any mounted authority",
    }


def _route_key(method: str, path: str) -> tuple[str, str]:
    return method.upper(), path.split("?", 1)[0]


def _body_has_payload(body: Any) -> bool:
    if not isinstance(body, Mapping):
        return False
    for key in ("rows", "results", "pathways", "associations"):
        value = body.get(key)
        if isinstance(value, list) and value:
            return True
    for key in ("returned_rows", "total_rows", "count"):
        value = body.get(key)
        if isinstance(value, int) and value > 0:
            return True
    for section in ("entity_results", "patient_results"):
        value = body.get(section)
        if isinstance(value, Mapping) and int(value.get("total_rows", 0)) > 0:
            return True
    status = str(body.get("status", "")).upper()
    return status.startswith(("PASS", "ENABLED", "SUCCESS", "READY"))


def _response_record(
    *, name: str, method: str, path: str, response: Any, semantic: str
) -> dict[str, Any]:
    try:
        body = response.json()
    except Exception:
        body = None
    if semantic == "typed_drug_pending":
        valid = response.status_code == 503 and "not mounted" in str(body).lower()
        servable = False
    elif semantic == "typed_single_cell_gap":
        rows = body.get("rows", []) if isinstance(body, Mapping) else []
        valid = response.status_code == 200 and any(
            isinstance(row, Mapping)
            and row.get("artifact_id") == "v32_sc_pseudotime"
            and row.get("status") == "GAP_TYPED_UNAVAILABLE"
            and row.get("numeric_rows") == 0
            for row in rows
        )
        servable = False
    elif semantic == "typed_pseudotime":
        rows = body.get("rows", []) if isinstance(body, Mapping) else []
        valid = (
            response.status_code == 200
            and isinstance(body, Mapping)
            and body.get("pseudotime_numeric_values") == 0
            and all(
                isinstance(row, Mapping)
                and row.get("pseudotime_available") is False
                for row in rows
            )
        )
        servable = False
    elif semantic == "json_200":
        valid = response.status_code == 200 and isinstance(body, Mapping)
        servable = valid
    else:
        valid = response.status_code == 200 and _body_has_payload(body)
        servable = valid
    summary: dict[str, Any] = {}
    if isinstance(body, Mapping):
        for key in ("module", "module_id", "query_kind", "status", "returned_rows", "total_rows", "count", "availability", "pseudotime_available"):
            if key in body:
                summary[key] = body.get(key)
        if "detail" in body:
            summary["detail"] = str(body.get("detail"))[:500]
    return {
        "name": name,
        "method": method,
        "path": path,
        "http_status": response.status_code,
        "semantic": semantic,
        "observation_valid": bool(valid),
        "servable_success": bool(servable),
        "body_summary": summary,
    }


def _run_api_runtime(
    *,
    root: Path,
    unified_path: Path,
    contract: Mapping[str, Any],
    web_rows: Mapping[str, Mapping[str, Any]],
    drug_mounted: bool,
) -> dict[str, Any]:
    app = create_app_from_unified_bindings(unified_path, repo_root=root)
    route_set = sorted(
        {
            (method, route.path)
            for route in app.routes
            for method in (route.methods or set())
            if method in {"GET", "POST"}
        }
    )
    route_lookup = set(route_set)
    declared_route_checks: dict[str, dict[str, Any]] = {}
    mapping_checks: dict[str, dict[str, Any]] = {}
    for capability_id, capability in contract["capabilities"].items():
        declared: list[dict[str, Any]] = []
        for endpoint in web_rows[capability_id].get("staging_endpoints", []):
            parts = str(endpoint).split(maxsplit=1)
            if len(parts) != 2:
                declared.append({"declaration": endpoint, "implemented": False})
                continue
            key = _route_key(parts[0], parts[1])
            declared.append(
                {
                    "declaration": endpoint,
                    "method": key[0],
                    "path": key[1],
                    "implemented": key in route_lookup,
                }
            )
        declared_route_checks[capability_id] = {
            "status": "IMPLEMENTED" if declared and all(row["implemented"] for row in declared) else "CATALOG_ONLY_OR_MISSING",
            "routes": declared,
        }
        required_ids = [str(value) for value in capability["gates"]["api"]["ids"]]
        mapping = API_EQUIVALENTS.get(capability_id, {})
        rows: list[dict[str, Any]] = []
        for contract_id in required_ids:
            equivalent = mapping.get(contract_id)
            implemented = equivalent in route_lookup if equivalent is not None else False
            rows.append(
                {
                    "contract_id": contract_id,
                    "equivalent": (
                        {"method": equivalent[0], "path": equivalent[1]}
                        if equivalent is not None
                        else None
                    ),
                    "implemented": implemented,
                }
            )
        mapping_checks[capability_id] = {
            "status": "IMPLEMENTED" if rows and all(row["implemented"] for row in rows) else "INCOMPLETE",
            "contracts": rows,
        }

    probes: dict[str, list[dict[str, Any]]] = {capability_id: [] for capability_id in contract["capabilities"]}

    def add(capability_id: str, name: str, method: str, path: str, *, params: Mapping[str, Any] | None = None, json_body: Any = None, semantic: str = "nonempty_200") -> Any:
        response = client.request(method, path, params=params, json=json_body)
        probes[capability_id].append(
            _response_record(
                name=name, method=method, path=path, response=response, semantic=semantic
            )
        )
        return response

    with TestClient(app) as client:
        exact = add(
            "exact_pathway", "exact_association", "GET",
            "/v3.2-staging/exact-pathway/associations",
            params={"cancer_id": "BRCA", "limit": 1},
        )
        exact_body = exact.json() if exact.status_code == 200 else {}
        exact_rows = exact_body.get("rows", []) if isinstance(exact_body, Mapping) else []
        seed_lnc = str(exact_rows[0].get("lncrna_id")) if exact_rows else "LNC:ENSG00000117242"
        seed_pathway = str(exact_rows[0].get("pathway_id")) if exact_rows else None
        add("gene_set", "gene_set_catalog", "GET", "/v3.2-staging/gene-sets", params={"cancer_id": "BRCA", "limit": 1})
        add("ranked_subtype", "subtype_overview", "GET", "/v3.2-staging/ranked-subtypes/overview", params={"cancer_id": "BRCA"})
        add("network", "network_lncrna", "GET", f"/v3.2-staging/network/lncrna/{seed_lnc}", params={"limit": 1})

        state_ids = {
            "state_rnass": "stemness_rna::RNAss", "state_dnass": "stemness_dna::DNAss",
            "state_extend": "EXTEND::published_score", "state_ereg_expss": "stemness_rna::EREG.EXPss",
            "state_dmpss": "stemness_dna::DMPss", "state_enhss": "stemness_dna::ENHss",
            "state_ereg_methss": "stemness_dna::EREG-METHss",
        }
        for capability_id, state_id in state_ids.items():
            add(capability_id, "state_query", "GET", "/v3.2-staging/state", params={"state_id": state_id, "limit": 1})

        clinical = add(
            "clinical", "clinical_lncrna", "GET", "/v3.2-staging/clinical",
            params={"clinical_endpoint": "OS", "cancer_id": "BRCA", "subject_type": "lncrna", "limit": 1},
        )
        clinical_body = clinical.json() if clinical.status_code == 200 else {}
        clinical_rows = clinical_body.get("entity_results", {}).get("results", []) if isinstance(clinical_body, Mapping) else []
        clinical_lnc = str(clinical_rows[0].get("subject_id")) if clinical_rows else seed_lnc
        add("expression_landscape", "expression_coverage", "GET", "/v3.2-staging/expression/cancer-coverage", params={"cancer_id": "BRCA", "limit": 1})
        add("survival_kaplan_meier", "kaplan_meier", "GET", f"/v3.2-staging/lncrna/{clinical_lnc}/survival", params={"cancer_id": "BRCA", "clinical_endpoint": "OS", "limit": 1}, semantic="json_200")
        add("bulk_coexpression", "coexpression_clusters", "GET", "/v3.2-staging/coexpression/clusters", params={"cancer_id": "BRCA", "limit": 1})
        add("external_validation", "external_validation", "GET", "/v3.2-staging/validation/external", params={"limit": 1})
        add("continuous_pathway_activity", "continuous_activity", "GET", "/v3.2-staging/activity/continuous/BRCA", params={"pathway_id": seed_pathway, "limit": 1})
        add("single_cell", "single_cell_exact", "GET", "/v3.2-staging/single-cell/exact-pathway-associations", params={"cancer_id": "BRCA", "limit": 1})
        add("single_cell", "single_cell_gap", "GET", "/v3.2-staging/single-cell/audit/gaps", semantic="typed_single_cell_gap")
        add("single_cell", "pseudotime_typed", "GET", "/v3.2-staging/single-cell/hnsc/pseudotime-availability", params={"level": "CELL", "limit": 1}, semantic="typed_pseudotime")
        add("mutation", "mutation_query", "GET", "/v3.2-staging/genomic", params={"modality": "mutation", "cancer_id": "BRCA", "limit": 1})
        add("cnv", "cnv_query", "GET", "/v3.2-staging/genomic", params={"modality": "cnv", "cancer_id": "BRCA", "limit": 1})
        add(
            "drug",
            "drug_runtime",
            "GET",
            "/v3.2-staging/drug/response-actionability",
            params={
                "cancer_id": "BRCA",
                "lncrna_id": seed_lnc,
                "drug_id": "STRICT_AUDIT_PROBE",
            },
            semantic="nonempty_200" if drug_mounted else "typed_drug_pending",
        )
        physical = add("physical_interaction", "physical_relationship", "GET", "/v3.2-staging/interaction/relationships", params={"lncrna_id": seed_lnc, "limit": 1})
        add("experiment_perturbation", "experiment_bridge", "GET", "/v3.2-staging/experiment/perturbation/evidence-bridge", params={"cancer_id": "BRCA", "lncrna_id": seed_lnc, "limit": 1}, semantic="json_200")
        add("evidence_transformer", "evidence_direction", "GET", "/v3.2-staging/evidence/direction/probabilities", params={"available": "true", "limit": 1})
        add("mixed_lncrna_protein_pathway_query", "mixed_query", "POST", "/v3.2-staging/enrichment/mixed-exact-pathway", json_body={"members": ["TP53", "EGFR"], "cancer_id": "BRCA", "top_k": 1})

    runtime: dict[str, dict[str, Any]] = {}
    for capability_id, rows in probes.items():
        observation_valid = bool(rows) and all(row["observation_valid"] for row in rows)
        servable_success = bool(rows) and all(row["servable_success"] for row in rows)
        runtime[capability_id] = {
            "observation_valid": observation_valid,
            "servable_success": servable_success,
            "probes": rows,
        }
    return {
        "app_constructed": True,
        "route_count": len(route_set),
        "route_set": [{"method": method, "path": path} for method, path in route_set],
        "catalog_declared_route_checks": declared_route_checks,
        "required_contract_mapping_checks": mapping_checks,
        "runtime": runtime,
    }


def _run_main_site_probe(
    *, root: Path, html_path: Path, js_path: Path, catalog_path: Path
) -> dict[str, Any]:
    program = r'''
import hashlib
import json
from fastapi.testclient import TestClient
from website.backend import app as main_module

paths = [
    "/v32-staging.html",
    "/assets/v32-staging.js",
    "/v32-capability-catalog.json",
    "/v3.2-staging/health",
]
rows = []
with TestClient(main_module.app) as client:
    for path in paths:
        response = client.get(path)
        rows.append({
            "path": path,
            "http_status": response.status_code,
            "content_type": response.headers.get("content-type"),
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "bytes": len(response.content),
        })
print(json.dumps({"constructed": True, "responses": rows}, sort_keys=True))
'''
    environment = dict(os.environ)
    environment["CANCERLNCATLAS_REQUIRE_V31_EXACT_PATHWAY"] = "0"
    environment["CANCERLNCATLAS_ENABLE_V32_STAGING"] = "1"
    python_path = os.pathsep.join(
        [str(root), str(root / "src"), environment.get("PYTHONPATH", "")]
    )
    environment["PYTHONPATH"] = python_path
    try:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=90,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "constructed": False,
            "servable": False,
            "failure": f"{type(exc).__name__}: {exc}",
            "responses": [],
        }
    payload: dict[str, Any] = {}
    if completed.returncode == 0:
        try:
            parsed = json.loads(completed.stdout.strip().splitlines()[-1])
            if isinstance(parsed, dict):
                payload = parsed
        except (IndexError, json.JSONDecodeError):
            payload = {}
    responses = payload.get("responses", []) if isinstance(payload, Mapping) else []
    by_path = {
        row.get("path"): row for row in responses if isinstance(row, Mapping)
    }
    expected = {
        "/v32-staging.html": sha256_file(html_path),
        "/assets/v32-staging.js": sha256_file(js_path),
        "/v32-capability-catalog.json": sha256_file(catalog_path),
    }
    exact_static = all(
        by_path.get(path, {}).get("http_status") == 200
        and by_path.get(path, {}).get("sha256") == digest
        for path, digest in expected.items()
    )
    health = by_path.get("/v3.2-staging/health", {})
    staging_health_json = (
        health.get("http_status") == 200
        and str(health.get("content_type", "")).startswith("application/json")
    )
    return {
        "constructed": completed.returncode == 0 and payload.get("constructed") is True,
        "servable": bool(exact_static),
        "staging_api_mounted_on_main_app": bool(staging_health_json),
        "expected_static_sha256": expected,
        "responses": responses,
        "process_returncode": completed.returncode,
        "failure": None if completed.returncode == 0 else completed.stderr[-2000:],
    }


def _run_frontend_checks(
    *,
    root: Path,
    html_path: Path,
    js_path: Path,
    catalog_path: Path,
    contract: Mapping[str, Any],
    web_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    html_source = html_path.read_text(encoding="utf-8")
    js_source = js_path.read_text(encoding="utf-8")
    parser = _FrontendParser()
    parser.feed(html_source)
    try:
        node = subprocess.run(
            ["node", "--check", str(js_path)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        js_syntax_ok = node.returncode == 0
        node_detail = (node.stderr or node.stdout)[-1000:]
    except (OSError, subprocess.TimeoutExpired) as exc:
        js_syntax_ok = False
        node_detail = f"{type(exc).__name__}: {exc}"
    main_site = _run_main_site_probe(
        root=root, html_path=html_path, js_path=js_path, catalog_path=catalog_path
    )
    capability_checks: dict[str, dict[str, Any]] = {}
    for capability_id, capability in contract["capabilities"].items():
        actions = CAPABILITY_UI_ACTIONS[capability_id]
        action_rows = [
            {
                "action": action,
                "html_control_present": action in parser.query_actions,
                "javascript_handler_present": _js_has_handler(js_source, action),
            }
            for action in actions
        ]
        required_surfaces = [str(value) for value in capability["gates"]["ui"]["ids"]]
        declared_surfaces = [str(value) for value in web_rows[capability_id].get("ui_surface_ids", [])]
        catalog_surface_closure = sorted(required_surfaces) == sorted(declared_surfaces)
        implemented_static = (
            js_syntax_ok
            and bool(action_rows)
            and all(
                row["html_control_present"] and row["javascript_handler_present"]
                for row in action_rows
            )
        )
        servable = (
            implemented_static
            and main_site["servable"]
            and not bool(web_rows[capability_id].get("known_gap"))
        )
        capability_checks[capability_id] = {
            "catalog_surface_closure": catalog_surface_closure,
            "required_surface_ids": required_surfaces,
            "declared_surface_ids": declared_surfaces,
            "actions": action_rows,
            "implementation_state": (
                "IMPLEMENTED_STATIC" if implemented_static else "CATALOG_DECLARED_OR_INCOMPLETE_STATIC"
            ),
            "servable": servable,
        }
    return {
        "html_path": html_path.relative_to(root).as_posix(),
        "html_sha256": sha256_file(html_path),
        "javascript_path": js_path.relative_to(root).as_posix(),
        "javascript_sha256": sha256_file(js_path),
        "javascript_syntax_ok": js_syntax_ok,
        "javascript_syntax_detail": node_detail,
        "html_control_ids": sorted(parser.ids),
        "html_query_actions": sorted(parser.query_actions),
        "script_sources": parser.scripts,
        "main_site": main_site,
        "capabilities": capability_checks,
    }


def _gate(
    passed: bool, reasons: Sequence[str], evidence: Mapping[str, Any]
) -> dict[str, Any]:
    reason_values = [str(reason) for reason in reasons if str(reason)]
    return {
        "status": "PASS" if passed and not reason_values else "FAIL",
        "reasons": reason_values,
        "evidence": dict(evidence),
    }


def _load_strict_release_inputs(
    *, root: Path, unified: Mapping[str, Any]
) -> tuple[
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
]:
    if unified.get("schema_version") != "CANCERLNCATLAS_V32_UNIFIED_STAGING_BINDINGS_V1":
        raise StrictCompletenessError("Unexpected unified bindings schema")
    if unified.get("environment") != "staging" or unified.get("production_deployed") is not False:
        raise StrictCompletenessError("Unified bindings must be staging-only")
    registry_raw = unified.get("registry")
    bindings = unified.get("bindings")
    if not isinstance(registry_raw, Mapping) or not isinstance(bindings, Mapping):
        raise StrictCompletenessError("Unified registry/bindings declaration is missing")
    registry_path = _verify_declaration(registry_raw, root, root, "release registry")
    _, registry = _read_json(registry_path, root, "release registry")

    required = {
        "download_catalog": REQUIRED_R4_DOWNLOAD_BINDING_SHA256,
        "download_catalog_independent_audit": REQUIRED_R4_DOWNLOAD_AUDIT_SHA256,
        "historical_artifact_remediation": REQUIRED_R3_HISTORICAL_BINDING_SHA256,
        "historical_artifact_remediation_independent_audit": REQUIRED_R3_HISTORICAL_AUDIT_SHA256,
    }
    loaded: dict[str, tuple[Path, dict[str, Any]]] = {}
    for binding_id, expected_sha in required.items():
        declaration = bindings.get(binding_id)
        if not isinstance(declaration, Mapping):
            raise StrictCompletenessError(f"Required binding {binding_id} is absent")
        if declaration.get("status") != "MOUNTED_HASH_PINNED":
            raise StrictCompletenessError(f"Required binding {binding_id} is not mounted")
        if str(declaration.get("sha256", "")).lower() != expected_sha:
            raise StrictCompletenessError(
                f"Required binding {binding_id} is not the pinned r4/r3 revision"
            )
        path = _verify_declaration(declaration, root, root, binding_id)
        _, payload = _read_json(path, root, binding_id)
        loaded[binding_id] = (path, payload)

    download_binding_path, download_binding = loaded["download_catalog"]
    if download_binding.get("catalog_version") != "v3.2-20260826-r4":
        raise StrictCompletenessError("Download binding is not catalog r4")
    if download_binding.get("production_deployed") is not False:
        raise StrictCompletenessError("Download binding falsely claims production")
    catalog_declaration = download_binding.get("catalog")
    if not isinstance(catalog_declaration, Mapping):
        raise StrictCompletenessError("Download binding lacks catalog declaration")
    download_catalog_path = _verify_declaration(
        catalog_declaration, root, download_binding_path.parent, "download catalog"
    )
    _, download_catalog = _read_json(download_catalog_path, root, "download catalog")
    if (
        download_catalog.get("catalog_version") != "v3.2-20260826-r4"
        or download_catalog.get("environment") != "staging"
        or download_catalog.get("production_deployed") is not False
    ):
        raise StrictCompletenessError("Download catalog r4 staging identity is invalid")

    download_audit_path, download_audit = loaded[
        "download_catalog_independent_audit"
    ]
    if download_audit.get("status") != "PASS" or download_audit.get("fail_count") != 0:
        raise StrictCompletenessError("Download catalog r4 independent audit does not PASS")
    if (
        _selector(download_audit, ("release_binding", "sha256"))
        != REQUIRED_R4_DOWNLOAD_BINDING_SHA256
        or _selector(download_audit, ("catalog", "sha256"))
        != sha256_file(download_catalog_path)
    ):
        raise StrictCompletenessError("Download catalog r4 audit is not bound to current inputs")

    historical_path, historical = loaded["historical_artifact_remediation"]
    if historical.get("production_deployed") is not False:
        raise StrictCompletenessError("Historical remediation falsely claims production")
    historical_audit_path, historical_audit = loaded[
        "historical_artifact_remediation_independent_audit"
    ]
    if (
        not str(historical_audit.get("status", "")).startswith("PASS")
        or historical_audit.get("failed_checks") != 0
        or historical_audit.get("release_binding_sha256")
        != REQUIRED_R3_HISTORICAL_BINDING_SHA256
    ):
        raise StrictCompletenessError("Historical remediation r3 audit does not PASS")
    return (
        registry_path,
        registry,
        download_binding_path,
        download_binding,
        download_catalog_path,
        download_catalog,
        download_audit_path,
        download_audit,
        historical_path,
        historical,
    )


def _web_capability_rows(
    web: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if (
        web.get("environment") != "staging"
        or web.get("production_deployed") is not False
        or web.get("model_version") != MODEL_VERSION
    ):
        raise StrictCompletenessError("Web catalog staging identity is invalid")
    raw_rows = web.get("capabilities")
    if not isinstance(raw_rows, list):
        raise StrictCompletenessError("Web catalog capabilities are missing")
    rows = {
        str(row.get("capability_id")): dict(row)
        for row in raw_rows
        if isinstance(row, Mapping) and row.get("capability_id")
    }
    if len(rows) != len(raw_rows) or set(rows) != set(contract["capabilities"]):
        raise StrictCompletenessError("Web catalog capability closure/uniqueness drift")
    for capability_id, capability in contract["capabilities"].items():
        row = rows[capability_id]
        closures = (
            ("required_api_contract", capability["gates"]["api"]["ids"]),
            ("ui_surface_ids", capability["gates"]["ui"]["ids"]),
            ("download_ids", capability["gates"]["download"]["ids"]),
        )
        for key, expected in closures:
            observed = [str(value) for value in row.get(key, [])]
            if sorted(observed) != sorted(str(value) for value in expected):
                raise StrictCompletenessError(
                    f"Web catalog {capability_id}.{key} closure drift"
                )
    return rows


def _download_rows(
    catalog: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    expected: dict[str, list[str]] = {}
    for capability_id, capability in contract["capabilities"].items():
        for download_id in capability["gates"]["download"]["ids"]:
            expected.setdefault(str(download_id), []).append(capability_id)
    raw_rows = catalog.get("downloads")
    if not isinstance(raw_rows, list):
        raise StrictCompletenessError("Download catalog rows are missing")
    rows = {
        str(row.get("download_id")): dict(row)
        for row in raw_rows
        if isinstance(row, Mapping) and row.get("download_id")
    }
    if (
        len(rows) != len(raw_rows)
        or set(rows) != set(expected)
        or catalog.get("download_count") != len(rows)
    ):
        raise StrictCompletenessError("Download catalog ID closure/uniqueness drift")
    for download_id, capability_ids in expected.items():
        observed = sorted(str(value) for value in rows[download_id].get("capability_ids", []))
        if observed != sorted(capability_ids):
            raise StrictCompletenessError(
                f"Download {download_id} capability membership drift"
            )
    return rows


def evaluate_strict_integrated_completeness(
    *,
    repo_root: str | Path,
    parity_contract_path: str | Path,
    unified_bindings_path: str | Path,
    web_catalog_path: str | Path,
    frontend_html_path: str | Path,
    frontend_javascript_path: str | Path,
    main_app_path: str | Path,
    evaluator_code_path: str | Path | None = None,
    runner_code_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate implementation and runtime evidence for every parity gate."""

    root = Path(repo_root).resolve()
    contract_path = _safe_path(parity_contract_path, root, "parity contract")
    unified_path, unified = _read_json(unified_bindings_path, root, "unified bindings")
    web_path, web = _read_json(web_catalog_path, root, "web catalog")
    html_path = _safe_path(frontend_html_path, root, "V3.2 HTML")
    js_path = _safe_path(frontend_javascript_path, root, "V3.2 JavaScript")
    main_path = _safe_path(main_app_path, root, "main website application")
    contract = load_parity_config(contract_path)
    if (
        contract.get("schema_version")
        != "CANCERLNCATLAS_V32_HISTORICAL_CAPABILITY_PARITY_V1"
        or contract.get("release_model_version") != MODEL_VERSION
        or set(contract.get("capabilities", {})) != set(EXPECTED_CAPABILITY_IDS)
        or len(contract.get("capabilities", {})) != 25
    ):
        raise StrictCompletenessError("Parity contract identity/capability closure is invalid")
    web_rows = _web_capability_rows(web, contract)
    (
        registry_path,
        registry,
        download_binding_path,
        download_binding,
        download_catalog_path,
        download_catalog,
        download_audit_path,
        download_audit,
        historical_path,
        historical,
    ) = _load_strict_release_inputs(root=root, unified=unified)
    downloads = _download_rows(download_catalog, contract)

    required_artifact_ids = {
        str(artifact_id)
        for capability in contract["capabilities"].values()
        for artifact_id in capability["gates"]["artifact"]["ids"]
    }
    authorities, artifact_index, all_documents, binding_states = (
        _collect_authority_documents(
            root=root,
            unified=unified,
            download_catalog=download_catalog,
            required_artifact_ids=required_artifact_ids,
        )
    )
    artifact_resolutions = {
        artifact_id: _resolve_artifact(
            artifact_id,
            root=root,
            authorities=authorities,
            artifact_index=artifact_index,
            downloads=downloads,
            registry_path=registry_path,
            registry=registry,
        )
        for artifact_id in sorted(required_artifact_ids)
    }
    drug_binding = unified.get("bindings", {}).get("drug_response_actionability", {})
    drug_mounted = (
        isinstance(drug_binding, Mapping)
        and drug_binding.get("status") == "MOUNTED_HASH_PINNED"
    )
    api_runtime = _run_api_runtime(
        root=root,
        unified_path=unified_path,
        contract=contract,
        web_rows=web_rows,
        drug_mounted=drug_mounted,
    )
    frontend = _run_frontend_checks(
        root=root,
        html_path=html_path,
        js_path=js_path,
        catalog_path=web_path,
        contract=contract,
        web_rows=web_rows,
    )
    route_lookup = {
        (str(row["method"]), str(row["path"])) for row in api_runtime["route_set"]
    }

    capability_rows: list[dict[str, Any]] = []
    for capability_id in sorted(contract["capabilities"]):
        capability = contract["capabilities"][capability_id]
        web_row = web_rows[capability_id]
        known_gap = web_row.get("known_gap")

        artifact_ids = [str(value) for value in capability["gates"]["artifact"]["ids"]]
        artifact_records = [artifact_resolutions[value] for value in artifact_ids]
        artifact_reasons = [
            f"{row['artifact_id']}: {row.get('state')} ({row.get('reason') or row.get('declared_status') or 'not hash-bound'})"
            for row in artifact_records
            if row.get("state") != "HASH_BOUND"
        ]
        artifact_gate = _gate(
            not artifact_reasons,
            artifact_reasons,
            {
                "required_artifact_ids": artifact_ids,
                "resolutions": artifact_records,
                "implementation_state": (
                    "HASH_BOUND" if not artifact_reasons else "UNRESOLVED_OR_TYPED_GAP"
                ),
            },
        )

        declared_check = api_runtime["catalog_declared_route_checks"][capability_id]
        mapping_check = api_runtime["required_contract_mapping_checks"][capability_id]
        runtime_check = api_runtime["runtime"][capability_id]
        api_reasons: list[str] = []
        if declared_check["status"] != "IMPLEMENTED":
            api_reasons.append("one or more catalog-declared routes are absent from FastAPI")
        if mapping_check["status"] != "IMPLEMENTED":
            api_reasons.append("one or more required legacy API contracts lack a staging equivalent")
        if not runtime_check["observation_valid"]:
            api_reasons.append("representative runtime observation is invalid")
        if not runtime_check["servable_success"]:
            api_reasons.append("representative runtime is a typed gap/error, not a usable result")
        if web_row.get("staging_status") != "QUERYABLE_STAGING":
            api_reasons.append(f"catalog staging status is {web_row.get('staging_status')}")
        if known_gap:
            api_reasons.append(f"catalogued known gap: {known_gap}")
        api_gate = _gate(
            not api_reasons,
            api_reasons,
            {
                "catalog_state": "CATALOG_DECLARED",
                "declared_route_check": declared_check,
                "required_contract_mapping": mapping_check,
                "implementation_state": (
                    "IMPLEMENTED"
                    if declared_check["status"] == mapping_check["status"] == "IMPLEMENTED"
                    else "INCOMPLETE"
                ),
                "runtime": runtime_check,
                "servable": bool(runtime_check["servable_success"]),
            },
        )

        ui_check = frontend["capabilities"][capability_id]
        ui_reasons: list[str] = []
        if not ui_check["catalog_surface_closure"]:
            ui_reasons.append("UI surface declaration differs from parity contract")
        if ui_check["implementation_state"] != "IMPLEMENTED_STATIC":
            ui_reasons.append("required HTML controls/JavaScript handlers are incomplete")
        if not frontend["main_site"]["servable"]:
            ui_reasons.append("main website does not serve the exact V3.2 HTML/JS/catalog assets")
        if not frontend["main_site"]["staging_api_mounted_on_main_app"]:
            ui_reasons.append("main website does not prove the V3.2 staging API mount")
        if known_gap:
            ui_reasons.append(f"catalogued known gap: {known_gap}")
        ui_gate = _gate(
            not ui_reasons,
            ui_reasons,
            {
                "catalog_state": "CATALOG_DECLARED",
                **ui_check,
                "main_site": frontend["main_site"],
            },
        )

        download_ids = [str(value) for value in capability["gates"]["download"]["ids"]]
        download_records: list[dict[str, Any]] = []
        download_reasons: list[str] = []
        for download_id in download_ids:
            row = downloads[download_id]
            status = str(row.get("status", ""))
            reasons: list[str] = []
            if status not in DOWNLOAD_PASS_STATUSES:
                reasons.append(f"non-deliverable status {status or 'MISSING'}")
            if row.get("contract_complete") is not True:
                reasons.append("contract_complete is not true")
            if row.get("download_implemented") is not True:
                reasons.append("download_implemented is not true")
            if status in {"READY_FILE", "READY_PARTS"} and row.get("data_present") is not True:
                reasons.append("ready record does not attest data_present")
            export_route_present = None
            if status == "DYNAMIC_QUERY_EXPORT":
                endpoint = str(row.get("export_endpoint", ""))
                parts = endpoint.split(maxsplit=1)
                export_route_present = (
                    len(parts) == 2 and _route_key(parts[0], parts[1]) in route_lookup
                )
                if not export_route_present:
                    reasons.append("dynamic export endpoint is absent from FastAPI")
                if not row.get("export_formats"):
                    reasons.append("dynamic export formats are absent")
                if not runtime_check["servable_success"]:
                    reasons.append("dynamic export capability did not pass a runtime probe")
            download_records.append(
                {
                    "download_id": download_id,
                    "status": status,
                    "implementation_state": (
                        "SERVABLE" if not reasons else "PENDING_OR_GAP"
                    ),
                    "export_route_present": export_route_present,
                    "reasons": reasons,
                }
            )
            download_reasons.extend(f"{download_id}: {reason}" for reason in reasons)
        download_gate = _gate(
            not download_reasons,
            download_reasons,
            {
                "catalog_state": "CATALOG_DECLARED",
                "catalog_version": download_catalog.get("catalog_version"),
                "required_download_ids": download_ids,
                "records": download_records,
                "independent_audit_status": download_audit.get("status"),
            },
        )
        gates = {
            "artifact": artifact_gate,
            "api": api_gate,
            "ui": ui_gate,
            "download": download_gate,
        }
        complete = all(gates[kind]["status"] == "PASS" for kind in GATE_KINDS)
        capability_rows.append(
            {
                "capability_id": capability_id,
                "status": "COMPLETE" if complete else "PARTIAL",
                "all_four_gates_complete": complete,
                "target_level": capability.get("target_level"),
                "known_gap": known_gap,
                "gates": gates,
            }
        )

    complete_ids = [
        row["capability_id"] for row in capability_rows if row["all_four_gates_complete"]
    ]
    blocking_ids = [
        row["capability_id"] for row in capability_rows if not row["all_four_gates_complete"]
    ]
    evaluator = _safe_path(
        evaluator_code_path or Path(__file__), root, "strict evaluator code"
    )
    runner = _safe_path(
        runner_code_path
        or root / "scripts/audit_v32_integrated_completeness_strict.py",
        root,
        "strict evaluator runner",
    )
    input_paths: dict[str, Path] = {
        "parity_contract": contract_path,
        "unified_staging_bindings": unified_path,
        "web_catalog": web_path,
        "frontend_html": html_path,
        "frontend_javascript": js_path,
        "main_website_application": main_path,
        "unified_staging_loader_code": _safe_path(
            root / "cc_hhgt/v32/unified_staging_bindings.py",
            root,
            "unified staging loader code",
        ),
        "v32_staging_api_code": _safe_path(
            root / "website/backend/v32_staging_api.py",
            root,
            "V3.2 staging API code",
        ),
        "legacy_compatibility_api_code": _safe_path(
            root / "src/cc_hhgt_v26/api.py",
            root,
            "legacy compatibility API code",
        ),
        "legacy_query_models_code": _safe_path(
            root / "src/cc_hhgt_v26/query_models.py",
            root,
            "legacy query models code",
        ),
        "exact_pathway_report_query_code": _safe_path(
            root / "cc_hhgt/v32/exact_pathway_report_query.py",
            root,
            "exact-pathway report query code",
        ),
        "release_registry": registry_path,
        "download_catalog_r4_binding": download_binding_path,
        "download_catalog_r4": download_catalog_path,
        "download_catalog_r4_independent_audit": download_audit_path,
        "historical_remediation_r3_binding": historical_path,
        "evaluator_code": evaluator,
        "runner_code": runner,
    }
    historical_audit_declaration = unified["bindings"][
        "historical_artifact_remediation_independent_audit"
    ]
    input_paths["historical_remediation_r3_independent_audit"] = _verify_declaration(
        historical_audit_declaration, root, root, "historical remediation r3 audit"
    )
    for key, path in sorted(all_documents.items()):
        input_paths.setdefault(key, path)
    inputs = {
        key: _source_record(path, root) for key, path in sorted(input_paths.items())
    }
    gate_counts = {
        kind: {
            "pass": sum(row["gates"][kind]["status"] == "PASS" for row in capability_rows),
            "fail": sum(row["gates"][kind]["status"] == "FAIL" for row in capability_rows),
        }
        for kind in GATE_KINDS
    }
    report = {
        "format": REPORT_FORMAT,
        "analysis_version": "CancerLncAtlas_V3.2_FULL_MULTITASK",
        "model_version": MODEL_VERSION,
        "environment": "staging",
        "status": "COMPLETE" if not blocking_ids else "PARTIAL",
        "all_25_capabilities_four_gate_complete": not blocking_ids,
        "release_ready": not blocking_ids,
        "production_deployed": False,
        "fail_closed": True,
        "catalog_declared_is_not_implemented": True,
        "typed_gaps_count_as_complete": False,
        "expected_errors_count_as_servable": False,
        "required_capability_count": len(capability_rows),
        "complete_capability_count": len(complete_ids),
        "partial_capability_count": len(blocking_ids),
        "complete_capability_ids": complete_ids,
        "blocking_capability_ids": blocking_ids,
        "gate_counts": gate_counts,
        "required_release_revisions": {
            "download_catalog_r4_binding_sha256": REQUIRED_R4_DOWNLOAD_BINDING_SHA256,
            "download_catalog_r4_audit_sha256": REQUIRED_R4_DOWNLOAD_AUDIT_SHA256,
            "historical_remediation_r3_binding_sha256": REQUIRED_R3_HISTORICAL_BINDING_SHA256,
            "historical_remediation_r3_audit_sha256": REQUIRED_R3_HISTORICAL_AUDIT_SHA256,
        },
        "implementation_evidence": {
            "api": api_runtime,
            "frontend": frontend,
            "artifact_resolution_count": len(artifact_resolutions),
            "artifact_resolutions": artifact_resolutions,
            "unified_binding_states": binding_states,
            "historical_remediation_status": historical.get("status"),
            "download_catalog_status_counts": download_catalog.get("status_counts"),
        },
        "capabilities": capability_rows,
        "input_sha256": inputs,
    }
    return report, inputs


def materialize_strict_integrated_completeness_audit(
    *,
    repo_root: str | Path,
    parity_contract_path: str | Path,
    unified_bindings_path: str | Path,
    web_catalog_path: str | Path,
    frontend_html_path: str | Path,
    frontend_javascript_path: str | Path,
    main_app_path: str | Path,
    output_root: str | Path,
    evaluator_code_path: str | Path | None = None,
    runner_code_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise StrictCompletenessError("Audit output must remain inside repository") from exc
    if output.exists() and any(output.iterdir()):
        raise StrictCompletenessError(f"Refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report, inputs = evaluate_strict_integrated_completeness(
        repo_root=root,
        parity_contract_path=parity_contract_path,
        unified_bindings_path=unified_bindings_path,
        web_catalog_path=web_catalog_path,
        frontend_html_path=frontend_html_path,
        frontend_javascript_path=frontend_javascript_path,
        main_app_path=main_app_path,
        evaluator_code_path=evaluator_code_path,
        runner_code_path=runner_code_path,
    )
    report_path = output / "INTEGRATED_COMPLETENESS_STRICT_REPORT.json"
    _atomic_json(report_path, report)
    binding = {
        "format": BINDING_FORMAT,
        "analysis_version": report["analysis_version"],
        "model_version": MODEL_VERSION,
        "environment": "staging",
        "status": report["status"],
        "all_25_capabilities_four_gate_complete": report[
            "all_25_capabilities_four_gate_complete"
        ],
        "release_ready": report["release_ready"],
        "production_deployed": False,
        "fail_closed": True,
        "catalog_declared_is_not_implemented": True,
        "required_capability_count": report["required_capability_count"],
        "complete_capability_count": report["complete_capability_count"],
        "partial_capability_count": report["partial_capability_count"],
        "blocking_capability_ids": report["blocking_capability_ids"],
        "gate_counts": report["gate_counts"],
        "report": _source_record(report_path, root),
        "inputs": inputs,
    }
    binding_path = output / "INTEGRATED_COMPLETENESS_STRICT_BINDING.json"
    _atomic_json(binding_path, binding)
    return {
        "binding_path": str(binding_path),
        "binding_sha256": sha256_file(binding_path),
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "status": report["status"],
        "complete_capability_count": report["complete_capability_count"],
        "partial_capability_count": report["partial_capability_count"],
        "blocking_capability_ids": report["blocking_capability_ids"],
        "gate_counts": report["gate_counts"],
    }
