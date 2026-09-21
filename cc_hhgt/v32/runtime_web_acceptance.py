"""Bounded black-box acceptance audit for a CancerLncAtlas web candidate.

The audit deliberately reports two independent dimensions:

* HTTP remediation: whether the public route can be called without an
  untyped transport/server failure.
* scientific availability: whether the response says that the corresponding
  scientific artifact is actually available.

A ``200`` response containing ``source_unavailable`` therefore passes the
HTTP transport check but does *not* become evidence that the data exist.
Likewise, a well-formed ``503 release_unavailable`` remains an HTTP failure,
although the typed reason is retained in the report.

Only loopback candidate ports are accepted.  Port 8260, the known production
port, is rejected before a request is made.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlparse
from urllib.request import ProxyHandler, Request, build_opener

from .single_cell_release_scope import final_binding_scope_is_available


REPORT_FORMAT = "CANCERLNCATLAS_V32_RUNTIME_ACCEPTANCE_V1"
FINAL_BINDING_FORMAT = "CANCERLNCATLAS_V32_FINAL_WEB_BINDING_V1"
ANALYSIS_VERSION = "CancerLncAtlas_V3.2_FULL_MULTITASK"

REQUIRED_FINAL_BINDING_ARTIFACT_ROLES = frozenset(
    {
        "winner_selection_authority",
        "web_relationship_materialization",
        "download_catalog_binding",
        "clinical_binding",
        "exact_geneset_binding",
        "mutation_binding",
        "single_cell_binding",
    }
)
REQUIRED_FINAL_BINDING_MODULES = frozenset(
    {
        "web_relationships",
        "downloads",
        "clinical",
        "exact_geneset",
        "mutation",
        "single_cell",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

TCGA_CANCERS: tuple[str, ...] = (
    "ACC",
    "BLCA",
    "BRCA",
    "CESC",
    "CHOL",
    "COAD",
    "DLBC",
    "ESCA",
    "GBM",
    "HNSC",
    "KICH",
    "KIRC",
    "KIRP",
    "LAML",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "MESO",
    "OV",
    "PAAD",
    "PCPG",
    "PRAD",
    "READ",
    "SARC",
    "SKCM",
    "STAD",
    "TGCT",
    "THCA",
    "THYM",
    "UCEC",
    "UCS",
    "UVM",
)

# The 51-route public contract described in the user's 2026-08-28 HTML issue
# list.  Newer auxiliary routes (sc-original-*, sc-query-umap-*, sc-ucell-qa,
# docs, and the duplicate /api/downloads alias) are deliberately not used to
# inflate this historical denominator.  They can be added as supplementary
# probes without changing this contract.
HTML_51_ROUTE_CONTRACT: tuple[tuple[str, str], ...] = (
    ("GET", "/v2.6/health"),
    ("GET", "/v2.6/version"),
    ("POST", "/v2.6/predict/mixed-gene-set"),
    ("POST", "/v2.6/predict/custom-gene-set-lncrna"),
    ("POST", "/v2.6/predict/protein-set-lncrna"),
    ("GET", "/api/meta"),
    ("GET", "/api/health"),
    ("GET", "/api/site/stats"),
    ("GET", "/api/site/search"),
    ("GET", "/api/site/lncrna/{lncrna}"),
    ("GET", "/api/site/lncrna/{lncrna}/visuals"),
    ("GET", "/api/site/lncrna/{lncrna}/relationships"),
    ("GET", "/api/site/genesets"),
    ("GET", "/api/site/genesets/{geneset_id}"),
    ("POST", "/api/site/gene-set-query"),
    ("GET", "/api/site/pathway-performance"),
    ("GET", "/api/site/single-cell"),
    ("GET", "/api/site/network"),
    ("GET", "/api/site/sc-activity/{cancer}"),
    ("GET", "/api/site/sc-summary/{cancer}"),
    ("GET", "/api/site/sc-umap/{cancer}"),
    ("GET", "/api/site/sc-umap-figure/{cancer}/{mode}"),
    ("GET", "/api/site/sc-figures/{cancer}"),
    ("GET", "/api/site/sc-figure/{cancer}/{figure}"),
    ("GET", "/api/site/datasets"),
    ("GET", "/api/site/cancers"),
    ("GET", "/api/site/bulk-composition/{cancer_id}"),
    ("GET", "/api/site/cancer/{cancer_id}"),
    ("GET", "/api/site/cancer/{cancer_id}/candidates"),
    (
        "GET",
        "/api/site/cancer/{cancer_id}/candidate/{lncrna}/{pathway_family_id}",
    ),
    ("GET", "/api/site/cancer/{cancer_id}/export"),
    ("GET", "/api/site/clinical/endpoints"),
    ("GET", "/api/site/clinical/lncrna/{lncrna}"),
    ("GET", "/api/site/clinical/priority"),
    ("GET", "/api/site/clinical/associations"),
    ("GET", "/api/site/clinical/visuals"),
    ("GET", "/api/site/mutation/status"),
    ("GET", "/api/site/mutation/cancer/{cancer_id}"),
    ("GET", "/api/site/mutation/lncrna/{lncrna}"),
    ("GET", "/v2.7/health"),
    ("GET", "/v2.7/version"),
    ("GET", "/v2.7/mutation/cancer/{cancer_id}"),
    ("GET", "/v2.7/mutation/lncrna/{lncrna_id}"),
    ("POST", "/v2.7/predict/mutation-context"),
    ("POST", "/v2.7/predict/mutation-subgroup-lncrna"),
    ("GET", "/api/site/downloads"),
    ("GET", "/api/site/download/{key}"),
    ("GET", "/api/search"),
    ("GET", "/api/cancers"),
    ("GET", "/api/cancers/{cancer_id}"),
    ("GET", "/api/lncrnas/{lncrna}"),
)

if len(HTML_51_ROUTE_CONTRACT) != 51:  # pragma: no cover - import invariant
    raise RuntimeError("The historical route contract must contain exactly 51 routes")
if len(TCGA_CANCERS) != 33:  # pragma: no cover - import invariant
    raise RuntimeError("The TCGA cancer contract must contain exactly 33 cancers")


UNAVAILABLE_STATUSES = {
    "artifact_unavailable",
    "blocked",
    "disabled",
    "missing",
    "not_available",
    "not_implemented",
    "not_mounted",
    "not_ready",
    "pending",
    "release_unavailable",
    "source_unavailable",
    "unavailable",
}
AVAILABLE_STATUSES = {
    "available",
    "healthy",
    "ok",
    "pass",
    "ready",
    "success",
}
EMPTY_BUT_EVALUATED_STATUSES = {
    "insufficient_heldout_labels",
    "no_complete_five_fold_oof_candidates",
    "no_evidence",
    "not_applicable",
}


@dataclass(frozen=True)
class ProbeSpec:
    """A single bounded HTTP probe."""

    probe_id: str
    route_template: str
    path: str
    method: str = "GET"
    group: str = "route_contract"
    body: Mapping[str, Any] | None = None
    expected_statuses: tuple[int, ...] = (200,)
    headers: Mapping[str, str] = field(default_factory=dict)
    cancer_id: str | None = None
    required_http: bool = True
    response_kind: str = "json"
    dependency_fallback: bool = False


@dataclass
class ProbeOutcome:
    """Observed response plus a private parsed payload used by validators."""

    spec: ProbeSpec
    status: int = 0
    elapsed_ms: float = 0.0
    content_type: str | None = None
    bytes_sampled: int = 0
    body_sample_sha256: str | None = None
    body_truncated: bool = False
    transport_error: str | None = None
    timed_out: bool = False
    payload: Any = field(default=None, repr=False)
    payload_parse_error: str | None = None

    @property
    def http_pass(self) -> bool:
        return not self.transport_error and self.status in self.spec.expected_statuses

    @property
    def typed_unavailable_paths(self) -> list[str]:
        return typed_unavailable_paths(self.payload)

    @property
    def scientific_state(self) -> str:
        return classify_scientific_state(self)

    def to_record(self) -> dict[str, Any]:
        summary = payload_summary(self.payload)
        return {
            "probe_id": self.spec.probe_id,
            "group": self.spec.group,
            "method": self.spec.method,
            "route_template": self.spec.route_template,
            "path": self.spec.path,
            "cancer_id": self.spec.cancer_id,
            "required_http": self.spec.required_http,
            "expected_statuses": list(self.spec.expected_statuses),
            "status": self.status,
            "http_pass": self.http_pass,
            "scientific_state": self.scientific_state,
            "typed_unavailable_paths": self.typed_unavailable_paths,
            "elapsed_ms": round(self.elapsed_ms, 3),
            "content_type": self.content_type,
            "bytes_sampled": self.bytes_sampled,
            "body_sample_sha256": self.body_sample_sha256,
            "body_truncated": self.body_truncated,
            "transport_error": self.transport_error,
            "timed_out": self.timed_out,
            "payload_parse_error": self.payload_parse_error,
            "payload_summary": summary,
            "dependency_fallback": self.spec.dependency_fallback,
        }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_sha256(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    return normalized if _SHA256_RE.fullmatch(normalized) else None


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def load_final_binding_evidence(
    binding_path: str | Path,
    *,
    expected_sha256: str,
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Load and rehash a final candidate binding and every declared artifact.

    The binding must use paths contained by ``artifact_root``.  This makes the
    runtime report independently check the files selected by the deployment
    rather than accepting self-reported hashes from HTTP payloads alone.
    Semantic readiness is evaluated separately by
    :func:`check_final_binding_attestation` so an invalid binding can still be
    represented as an explicit failed gate.
    """

    root = Path(artifact_root).resolve()
    source = Path(binding_path).resolve()
    expected = _normalized_sha256(expected_sha256)
    if expected is None:
        raise ValueError("Expected final-binding SHA-256 must contain 64 hex digits")
    if not source.is_file():
        raise FileNotFoundError(f"Final binding is missing: {source}")
    if not _path_within(source, root):
        raise ValueError("Final binding must be contained by artifact_root")
    observed = _sha256_file(source)
    if observed != expected:
        raise ValueError(
            f"Final binding SHA-256 drift: expected {expected}, observed {observed}"
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Final binding must be a JSON object")

    records: list[dict[str, Any]] = []
    declarations = payload.get("artifacts")
    if isinstance(declarations, Sequence) and not isinstance(
        declarations, (str, bytes)
    ):
        for index, declaration in enumerate(declarations):
            role = ""
            raw_path = ""
            declared_sha: str | None = None
            if isinstance(declaration, Mapping):
                role = str(declaration.get("role") or "").strip()
                raw_path = str(declaration.get("path") or "").strip()
                declared_sha = _normalized_sha256(declaration.get("sha256"))
            record: dict[str, Any] = {
                "index": index,
                "role": role,
                "path": raw_path,
                "declared_sha256": declared_sha,
                "contained_by_artifact_root": False,
                "file_exists": False,
                "sha256_matches": False,
            }
            if raw_path:
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = root / candidate
                candidate = candidate.resolve()
                record["resolved_path"] = str(candidate)
                record["contained_by_artifact_root"] = _path_within(candidate, root)
                if record["contained_by_artifact_root"] and candidate.is_file():
                    record["file_exists"] = True
                    record["observed_sha256"] = _sha256_file(candidate)
                    record["sha256_matches"] = (
                        declared_sha is not None
                        and record["observed_sha256"] == declared_sha
                    )
            records.append(record)
    return {
        "path": str(source),
        "sha256": observed,
        "artifact_root": str(root),
        "payload": dict(payload),
        "artifact_verification": records,
    }


def check_final_binding_attestation(
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate the immutable winner/release binding used by the candidate.

    This gate is intentionally stricter than HTTP probing.  It requires a
    validation-only routing winner, five fold predictions and checkpoints,
    explicit fresh-V3.2/no-old-artifact lineage, module-specific scientific
    claims, and rehashed files for every required artifact role.
    """

    if not isinstance(evidence, Mapping):
        return {
            "name": "final_winner_and_scientific_binding",
            "pass": False,
            "provided": False,
            "reason": "FINAL_BINDING_NOT_PROVIDED",
            "missing_artifact_roles": sorted(REQUIRED_FINAL_BINDING_ARTIFACT_ROLES),
            "missing_modules": sorted(REQUIRED_FINAL_BINDING_MODULES),
        }
    payload = evidence.get("payload")
    if not isinstance(payload, Mapping):
        payload = {}
    records = evidence.get("artifact_verification")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        records = []

    roles = [
        str(record.get("role") or "").strip()
        for record in records
        if isinstance(record, Mapping)
    ]
    duplicate_roles = sorted({role for role in roles if role and roles.count(role) > 1})
    missing_roles = sorted(REQUIRED_FINAL_BINDING_ARTIFACT_ROLES - set(roles))
    invalid_artifacts: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            invalid_artifacts.append(
                {
                    "role": None,
                    "path": None,
                    "contained_by_artifact_root": False,
                    "file_exists": False,
                    "sha256_matches": False,
                }
            )
            continue
        if (
            record.get("contained_by_artifact_root") is not True
            or record.get("file_exists") is not True
            or record.get("sha256_matches") is not True
        ):
            invalid_artifacts.append(
                {
                    "role": record.get("role"),
                    "path": record.get("path"),
                    "contained_by_artifact_root": record.get(
                        "contained_by_artifact_root"
                    ),
                    "file_exists": record.get("file_exists"),
                    "sha256_matches": record.get("sha256_matches"),
                }
            )
    artifact_integrity = not missing_roles and not duplicate_roles and not invalid_artifacts

    winner = payload.get("winner_selection")
    if not isinstance(winner, Mapping):
        winner = {}
    compared = winner.get("compared_models")
    compared_set = {
        _normalized_status(value)
        for value in compared
    } if isinstance(compared, Sequence) and not isinstance(compared, (str, bytes)) else set()
    fold_hashes = winner.get("fold_prediction_sha256")
    checkpoint_hashes = winner.get("checkpoint_sha256")
    if not isinstance(fold_hashes, Sequence) or isinstance(fold_hashes, (str, bytes)):
        fold_hashes = []
    if not isinstance(checkpoint_hashes, Sequence) or isinstance(
        checkpoint_hashes, (str, bytes)
    ):
        checkpoint_hashes = []
    normalized_fold_hashes = [_normalized_sha256(value) for value in fold_hashes]
    normalized_checkpoint_hashes = [
        _normalized_sha256(value) for value in checkpoint_hashes
    ]
    folds_attested = bool(
        len(normalized_fold_hashes) == 5
        and all(value is not None for value in normalized_fold_hashes)
        and len(set(normalized_fold_hashes)) == 5
    )
    checkpoints_attested = bool(
        len(normalized_checkpoint_hashes) == 5
        and all(value is not None for value in normalized_checkpoint_hashes)
        and len(set(normalized_checkpoint_hashes)) == 5
    )
    winner_score = _finite_number(winner.get("winner_score"))
    winner_attested = bool(
        str(winner.get("winner_id") or "").strip()
        and _normalized_status(winner.get("selection_split")) == "validation_only"
        and winner.get("heldout_test_used_for_selection") is False
        and winner.get("fair_comparison") is True
        and {"external_router", "hierarchical_end_to_end"} <= compared_set
        and winner_score is not None
        and folds_attested
        and checkpoints_attested
    )

    modules = payload.get("modules")
    if not isinstance(modules, Mapping):
        modules = {}
    missing_modules = sorted(REQUIRED_FINAL_BINDING_MODULES - set(modules))

    def module(name: str) -> Mapping[str, Any]:
        value = modules.get(name)
        return value if isinstance(value, Mapping) else {}

    web = module("web_relationships")
    downloads = module("downloads")
    clinical = module("clinical")
    exact = module("exact_geneset")
    mutation = module("mutation")
    single_cell = module("single_cell")
    download_catalog_payload_sha256 = _normalized_sha256(
        downloads.get("catalog_payload_sha256")
    )
    download_file_count = _integerish(downloads.get("file_count"))
    module_semantics = {
        "web_relationships": bool(
            _normalized_status(web.get("status")) == "available"
            and _integerish(web.get("cancer_count")) == 33
            and (_integerish(web.get("predicted_candidate_total")) or 0) > 0
            and web.get("exclusive_relationship_routing") is True
        ),
        "downloads": bool(
            _normalized_status(downloads.get("status")) == "available"
            and downloads.get("scientifically_attested") is True
            and download_catalog_payload_sha256 is not None
            and (download_file_count or 0) > 0
        ),
        "clinical": bool(
            _normalized_status(clinical.get("status")) == "available"
            and clinical.get("events_le_patients_invariant") is True
        ),
        "exact_geneset": bool(
            _normalized_status(exact.get("status")) == "available"
            and _normalized_status(exact.get("target_level")) == "exact_pathway"
            and exact.get("family_broadcast_used") is False
        ),
        "mutation": bool(
            _normalized_status(mutation.get("status")) == "available"
            and _integerish(mutation.get("cancer_count")) == 33
            and mutation.get("fresh_training") is True
            and mutation.get("old_predictions_used") is False
            and mutation.get("old_rankings_used") is False
        ),
        "single_cell": final_binding_scope_is_available(single_cell),
    }
    modules_attested = not missing_modules and all(module_semantics.values())
    top_level_lineage = bool(
        payload.get("format") == FINAL_BINDING_FORMAT
        and payload.get("analysis_version") == ANALYSIS_VERSION
        and str(payload.get("model_version") or "").upper() == "V3.2"
        and _normalized_status(payload.get("status"))
        in {"final_candidate_bound", "release_ready"}
        and payload.get("release_ready") is True
        and payload.get("production_deployed") is False
        and payload.get("fresh_training") is True
        and payload.get("old_predictions_used") is False
        and payload.get("old_rankings_used") is False
    )
    passed = artifact_integrity and winner_attested and modules_attested and top_level_lineage
    return {
        "name": "final_winner_and_scientific_binding",
        "pass": passed,
        "provided": True,
        "binding_path": evidence.get("path"),
        "binding_sha256": evidence.get("sha256"),
        "artifact_root": evidence.get("artifact_root"),
        "top_level_lineage_pass": top_level_lineage,
        "artifact_integrity_pass": artifact_integrity,
        "winner_selection_pass": winner_attested,
        "modules_attested": modules_attested,
        "module_semantics": module_semantics,
        "missing_artifact_roles": missing_roles,
        "duplicate_artifact_roles": duplicate_roles,
        "invalid_artifacts": invalid_artifacts,
        "missing_modules": missing_modules,
        "winner_id": winner.get("winner_id"),
        "winner_score": winner_score,
        "fold_prediction_hashes_attested": folds_attested,
        "checkpoint_hashes_attested": checkpoints_attested,
        "heldout_test_used_for_selection": winner.get(
            "heldout_test_used_for_selection"
        ),
        "download_catalog_payload_sha256": download_catalog_payload_sha256,
        "download_file_count": download_file_count,
        "predicted_candidate_total": _integerish(
            web.get("predicted_candidate_total")
        ),
    }


Requester = Callable[[str, ProbeSpec, float, int], ProbeOutcome]


def validate_candidate_base_url(base_url: str) -> str:
    """Return a normalized candidate URL or fail before any network access."""

    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Candidate base URL must use http or https")
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Runtime acceptance is restricted to a loopback candidate")
    if parsed.username or parsed.password:
        raise ValueError("Credentials are not allowed in the candidate base URL")
    if parsed.query or parsed.fragment or parsed.params:
        raise ValueError("Candidate base URL cannot contain query or fragment data")
    if parsed.path not in {"", "/"}:
        raise ValueError("Candidate base URL cannot contain an application path")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Candidate port is invalid") from exc
    if port is None:
        raise ValueError("An explicit candidate port is required")
    if port == 8260:
        raise ValueError("Port 8260 is the production port and is forbidden")
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


def _request(
    base_url: str,
    spec: ProbeSpec,
    timeout_seconds: float,
    max_body_bytes: int,
) -> ProbeOutcome:
    """Execute one request, reading only a bounded response sample."""

    started = time.monotonic()
    body_bytes = None
    headers = {
        "Accept": "application/json, text/csv, image/*;q=0.8, */*;q=0.1",
        "User-Agent": "CancerLncAtlas-V3.2-runtime-acceptance/1",
        **dict(spec.headers),
    }
    if spec.body is not None:
        body_bytes = json.dumps(spec.body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(
        base_url + spec.path,
        data=body_bytes,
        headers=headers,
        method=spec.method,
    )
    status = 0
    raw = b""
    content_type = None
    transport_error = None
    timed_out = False
    try:
        # Do not inherit HTTP(S)_PROXY for a loopback-only audit.  Besides being
        # unnecessary, a proxy can turn "candidate is not listening" into an
        # unrelated gateway 502 and destroy the evidentiary value of the probe.
        with build_opener(ProxyHandler({})).open(
            request, timeout=timeout_seconds
        ) as response:
            status = int(response.status)
            content_type = response.headers.get("Content-Type")
            raw = response.read(max_body_bytes + 1)
    except HTTPError as exc:
        status = int(exc.code)
        content_type = exc.headers.get("Content-Type") if exc.headers else None
        raw = exc.read(max_body_bytes + 1)
    except (TimeoutError, socket.timeout) as exc:
        timed_out = True
        transport_error = f"timeout: {exc}"
    except (URLError, OSError) as exc:
        reason = getattr(exc, "reason", None)
        timed_out = isinstance(reason, (TimeoutError, socket.timeout))
        prefix = "timeout" if timed_out else "transport_error"
        transport_error = f"{prefix}: {exc}"

    truncated = len(raw) > max_body_bytes
    sample = raw[:max_body_bytes]
    payload = None
    parse_error = None
    looks_json = bool(
        sample
        and (
            "json" in (content_type or "").lower()
            or sample.lstrip().startswith((b"{", b"["))
        )
    )
    if looks_json:
        try:
            payload = json.loads(sample.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
    return ProbeOutcome(
        spec=spec,
        status=status,
        elapsed_ms=(time.monotonic() - started) * 1000.0,
        content_type=content_type,
        bytes_sampled=len(sample),
        body_sample_sha256=hashlib.sha256(sample).hexdigest() if sample else None,
        body_truncated=truncated,
        transport_error=transport_error,
        timed_out=timed_out,
        payload=payload,
        payload_parse_error=parse_error,
    )


def run_specs(
    base_url: str,
    specs: Sequence[ProbeSpec],
    *,
    workers: int,
    timeout_seconds: float,
    max_body_bytes: int,
    requester: Requester = _request,
) -> list[ProbeOutcome]:
    """Run probes with a strict maximum number of concurrent workers."""

    if workers < 1 or workers > 32:
        raise ValueError("workers must be between 1 and 32")
    if timeout_seconds <= 0 or timeout_seconds > 120:
        raise ValueError("timeout_seconds must be in (0, 120]")
    if max_body_bytes < 1024 or max_body_bytes > 16 * 1024 * 1024:
        raise ValueError("max_body_bytes must be between 1 KiB and 16 MiB")
    normalized = validate_candidate_base_url(base_url)
    indexed: dict[Any, int] = {}
    outcomes: list[ProbeOutcome | None] = [None] * len(specs)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="web-audit") as pool:
        for index, spec in enumerate(specs):
            future = pool.submit(
                requester,
                normalized,
                spec,
                timeout_seconds,
                max_body_bytes,
            )
            indexed[future] = index
        for future in as_completed(indexed):
            index = indexed[future]
            try:
                outcomes[index] = future.result()
            except Exception as exc:  # requester bugs remain visible per probe
                outcomes[index] = ProbeOutcome(
                    spec=specs[index],
                    transport_error=f"requester_exception: {type(exc).__name__}: {exc}",
                )
    return [outcome for outcome in outcomes if outcome is not None]


def _normalized_status(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def typed_unavailable_paths(payload: Any) -> list[str]:
    """Locate explicit unavailability statuses without interpreting empty data."""

    paths: list[str] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if key in {
                    "status",
                    "availability",
                    "release_status",
                    "mutation_release_status",
                }:
                    status = _normalized_status(child)
                    if status in UNAVAILABLE_STATUSES:
                        paths.append(child_path)
                if key.endswith("_available") and child is False:
                    paths.append(child_path)
                visit(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value[:500]):
                visit(child, f"{path}[{index}]")

    visit(payload, "")
    return sorted(set(paths))


def _status_values(payload: Any) -> set[str]:
    values: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if key in {
                    "status",
                    "availability",
                    "release_status",
                    "mutation_release_status",
                }:
                    status = _normalized_status(child)
                    if status:
                        values.add(status)
                visit(child)
        elif isinstance(value, list):
            for child in value[:500]:
                visit(child)

    visit(payload)
    return values


def classify_scientific_state(outcome: ProbeOutcome) -> str:
    if outcome.transport_error:
        return "NOT_OBSERVED_TRANSPORT_FAILURE"
    unavailable = outcome.typed_unavailable_paths
    statuses = _status_values(outcome.payload)
    if unavailable:
        if outcome.status in outcome.spec.expected_statuses and (
            statuses & (AVAILABLE_STATUSES | EMPTY_BUT_EVALUATED_STATUSES)
        ):
            return "PARTIAL_TYPED_UNAVAILABLE"
        return "UNAVAILABLE_TYPED"
    if statuses & AVAILABLE_STATUSES:
        return "AVAILABLE"
    if statuses & EMPTY_BUT_EVALUATED_STATUSES:
        return "EVALUATED_EMPTY_OR_NOT_APPLICABLE"
    if outcome.http_pass:
        if outcome.spec.response_kind in {"binary", "download", "export"}:
            return "HTTP_ASSET_AVAILABLE_NOT_SCIENTIFICALLY_ATTESTED"
        return "HTTP_AVAILABLE_SCIENCE_NOT_ATTESTED"
    return "NOT_OBSERVED_HTTP_FAILURE"


def payload_summary(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, Mapping):
        result: dict[str, Any] = {"type": "object", "keys": sorted(payload)[:80]}
        for key in (
            "status",
            "analysis_version",
            "model_version",
            "release_id",
            "mutation_release_available",
            "mutation_release_status",
            "new_training",
            "new_training_attestation",
            "old_predictions_used",
            "production_deployed",
            "total",
            "total_rows",
        ):
            value = payload.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                if key in payload:
                    result[key] = value
        for key in ("results", "rows", "files", "cancers", "endpoints", "figures"):
            value = payload.get(key)
            if isinstance(value, list):
                result[f"{key}_count"] = len(value)
        return result
    if isinstance(payload, list):
        return {"type": "array", "length": len(payload)}
    return {"type": type(payload).__name__}


def _first_rows(payload: Any, keys: Sequence[str]) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, Mapping)]
    if not isinstance(payload, Mapping):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, Mapping)]
    return []


def _first_string(row: Mapping[str, Any] | None, keys: Sequence[str]) -> str | None:
    if row is None:
        return None
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def discovery_specs() -> list[ProbeSpec]:
    return [
        ProbeSpec(
            "discovery.candidate",
            "/api/site/cancer/{cancer_id}/candidates",
            "/api/site/cancer/BRCA/candidates?limit=1",
            group="discovery",
            required_http=False,
        ),
        ProbeSpec(
            "discovery.geneset",
            "/api/site/genesets",
            "/api/site/genesets?limit=1",
            group="discovery",
            required_http=False,
        ),
        ProbeSpec(
            "discovery.figure",
            "/api/site/sc-figures/{cancer}",
            "/api/site/sc-figures/BLCA",
            group="discovery",
            required_http=False,
        ),
        ProbeSpec(
            "discovery.download",
            "/api/site/downloads",
            "/api/site/downloads",
            group="discovery",
            required_http=False,
        ),
    ]


def derive_context(discovery: Sequence[ProbeOutcome]) -> dict[str, Any]:
    by_id = {outcome.spec.probe_id: outcome for outcome in discovery}
    candidate_rows = _first_rows(
        getattr(by_id.get("discovery.candidate"), "payload", None),
        ("results", "rows", "candidates"),
    )
    geneset_rows = _first_rows(
        getattr(by_id.get("discovery.geneset"), "payload", None),
        ("results", "rows", "gene_sets", "genesets", "pathways"),
    )
    figure_rows = _first_rows(
        getattr(by_id.get("discovery.figure"), "payload", None),
        ("figures", "results", "rows"),
    )
    download_rows = _first_rows(
        getattr(by_id.get("discovery.download"), "payload", None),
        ("files", "downloads", "results"),
    )
    candidate = candidate_rows[0] if candidate_rows else None
    geneset = geneset_rows[0] if geneset_rows else None
    figure = figure_rows[0] if figure_rows else None
    download = download_rows[0] if download_rows else None
    return {
        "candidate_lncrna": _first_string(
            candidate, ("lncrna_id", "lncrna", "gene_symbol")
        )
        or "MALAT1",
        "candidate_pathway": _first_string(
            candidate,
            ("pathway_family_id", "pathway_id", "geneset_id", "family_id"),
        )
        or "PF:0001",
        "geneset_id": _first_string(
            geneset, ("geneset_id", "pathway_id", "id", "pathway_family_id")
        )
        or "PF:0001",
        "figure_id": _first_string(
            figure, ("figure_id", "figure", "id", "name", "key")
        )
        or "__NO_FIGURE_DISCOVERED__",
        "download_key": _first_string(
            download, ("key", "download_id", "id", "name")
        )
        or "__NO_DOWNLOAD_DISCOVERED__",
        "fallbacks": {
            "candidate": not bool(candidate_rows),
            "geneset": not bool(geneset_rows),
            "figure": not bool(figure_rows),
            "download": not bool(download_rows),
        },
    }


def _query(path: str, **params: Any) -> str:
    values = {key: value for key, value in params.items() if value is not None}
    return path + ("?" + urlencode(values) if values else "")


def build_route_contract_specs(context: Mapping[str, Any]) -> list[ProbeSpec]:
    """Materialize one safe representative request for every historical route."""

    lnc = quote(str(context["candidate_lncrna"]), safe="")
    pathway = quote(str(context["candidate_pathway"]), safe="")
    geneset = quote(str(context["geneset_id"]), safe="")
    figure = quote(str(context["figure_id"]), safe="")
    download = quote(str(context["download_key"]), safe="")
    fallbacks = dict(context.get("fallbacks", {}))
    mixed = {"members": ["TP53", "MALAT1"], "cancer_id": "LUAD", "top_k": 1}
    specs = [
        ProbeSpec("route.01", "/v2.6/health", "/v2.6/health"),
        ProbeSpec("route.02", "/v2.6/version", "/v2.6/version"),
        ProbeSpec(
            "route.03",
            "/v2.6/predict/mixed-gene-set",
            "/v2.6/predict/mixed-gene-set",
            method="POST",
            body={**mixed, "network_nodes": 1},
        ),
        ProbeSpec(
            "route.04",
            "/v2.6/predict/custom-gene-set-lncrna",
            "/v2.6/predict/custom-gene-set-lncrna",
            method="POST",
            body=mixed,
        ),
        ProbeSpec(
            "route.05",
            "/v2.6/predict/protein-set-lncrna",
            "/v2.6/predict/protein-set-lncrna",
            method="POST",
            body={"members": ["P04637"], "cancer_id": "LUAD", "top_k": 1},
        ),
        ProbeSpec("route.06", "/api/meta", "/api/meta"),
        ProbeSpec("route.07", "/api/health", "/api/health"),
        ProbeSpec("route.08", "/api/site/stats", "/api/site/stats"),
        ProbeSpec(
            "route.09",
            "/api/site/search",
            _query("/api/site/search", q="肺癌", limit=12),
        ),
        ProbeSpec("route.10", "/api/site/lncrna/{lncrna}", f"/api/site/lncrna/{lnc}"),
        ProbeSpec(
            "route.11",
            "/api/site/lncrna/{lncrna}/visuals",
            _query(f"/api/site/lncrna/{lnc}/visuals", cancer="BLCA"),
        ),
        ProbeSpec(
            "route.12",
            "/api/site/lncrna/{lncrna}/relationships",
            _query(
                f"/api/site/lncrna/{lnc}/relationships",
                kind="pathway",
                cancer="BLCA",
                limit=1,
            ),
        ),
        ProbeSpec("route.13", "/api/site/genesets", "/api/site/genesets?limit=1"),
        ProbeSpec(
            "route.14",
            "/api/site/genesets/{geneset_id}",
            f"/api/site/genesets/{geneset}",
            dependency_fallback=bool(fallbacks.get("geneset")),
        ),
        ProbeSpec(
            "route.15",
            "/api/site/gene-set-query",
            "/api/site/gene-set-query",
            method="POST",
            body=mixed,
        ),
        ProbeSpec(
            "route.16",
            "/api/site/pathway-performance",
            "/api/site/pathway-performance?limit=1",
        ),
        ProbeSpec(
            "route.17",
            "/api/site/single-cell",
            _query(
                "/api/site/single-cell",
                cancer="BRCA",
                lnc=str(context["candidate_lncrna"]),
                limit=1,
            ),
        ),
        ProbeSpec(
            "route.18",
            "/api/site/network",
            "/api/site/network?lnc=MALAT1&cancer=BLCA",
        ),
        ProbeSpec(
            "route.19", "/api/site/sc-activity/{cancer}", "/api/site/sc-activity/BLCA"
        ),
        ProbeSpec(
            "route.20", "/api/site/sc-summary/{cancer}", "/api/site/sc-summary/BLCA"
        ),
        ProbeSpec(
            "route.21",
            "/api/site/sc-umap/{cancer}",
            "/api/site/sc-umap/BLCA?max_points=1000",
        ),
        ProbeSpec(
            "route.22",
            "/api/site/sc-umap-figure/{cancer}/{mode}",
            "/api/site/sc-umap-figure/BLCA/cell_type",
            response_kind="binary",
        ),
        ProbeSpec(
            "route.23", "/api/site/sc-figures/{cancer}", "/api/site/sc-figures/BLCA"
        ),
        ProbeSpec(
            "route.24",
            "/api/site/sc-figure/{cancer}/{figure}",
            f"/api/site/sc-figure/BLCA/{figure}?page=1",
            response_kind="binary",
            dependency_fallback=bool(fallbacks.get("figure")),
        ),
        ProbeSpec("route.25", "/api/site/datasets", "/api/site/datasets"),
        ProbeSpec("route.26", "/api/site/cancers", "/api/site/cancers"),
        ProbeSpec(
            "route.27",
            "/api/site/bulk-composition/{cancer_id}",
            "/api/site/bulk-composition/BLCA?method=EPIC&state=all&max_samples=10",
        ),
        ProbeSpec(
            "route.28", "/api/site/cancer/{cancer_id}", "/api/site/cancer/BLCA?limit=5"
        ),
        ProbeSpec(
            "route.29",
            "/api/site/cancer/{cancer_id}/candidates",
            "/api/site/cancer/BRCA/candidates?limit=1",
        ),
        ProbeSpec(
            "route.30",
            "/api/site/cancer/{cancer_id}/candidate/{lncrna}/{pathway_family_id}",
            f"/api/site/cancer/BRCA/candidate/{lnc}/{pathway}",
            dependency_fallback=bool(fallbacks.get("candidate")),
        ),
        ProbeSpec(
            "route.31",
            "/api/site/cancer/{cancer_id}/export",
            "/api/site/cancer/BRCA/export?q=MALAT1",
            response_kind="export",
            headers={"Range": "bytes=0-65535"},
        ),
        ProbeSpec(
            "route.32", "/api/site/clinical/endpoints", "/api/site/clinical/endpoints"
        ),
        ProbeSpec(
            "route.33",
            "/api/site/clinical/lncrna/{lncrna}",
            f"/api/site/clinical/lncrna/{lnc}?cancer=BLCA",
        ),
        ProbeSpec(
            "route.34",
            "/api/site/clinical/priority",
            "/api/site/clinical/priority?cancer=BLCA&limit=1",
        ),
        ProbeSpec(
            "route.35",
            "/api/site/clinical/associations",
            "/api/site/clinical/associations?cancer=BLCA&limit=1",
        ),
        ProbeSpec(
            "route.36",
            "/api/site/clinical/visuals",
            "/api/site/clinical/visuals?cancer=BLCA&lnc=MALAT1",
        ),
        ProbeSpec(
            "route.37", "/api/site/mutation/status", "/api/site/mutation/status"
        ),
        ProbeSpec(
            "route.38",
            "/api/site/mutation/cancer/{cancer_id}",
            "/api/site/mutation/cancer/BLCA?limit=1",
        ),
        ProbeSpec(
            "route.39",
            "/api/site/mutation/lncrna/{lncrna}",
            f"/api/site/mutation/lncrna/{lnc}?cancer=BLCA&limit=1",
        ),
        ProbeSpec("route.40", "/v2.7/health", "/v2.7/health"),
        ProbeSpec("route.41", "/v2.7/version", "/v2.7/version"),
        ProbeSpec(
            "route.42",
            "/v2.7/mutation/cancer/{cancer_id}",
            "/v2.7/mutation/cancer/BLCA?limit=1",
        ),
        ProbeSpec(
            "route.43",
            "/v2.7/mutation/lncrna/{lncrna_id}",
            f"/v2.7/mutation/lncrna/{lnc}?cancer=BLCA&limit=1",
        ),
        ProbeSpec(
            "route.44",
            "/v2.7/predict/mutation-context",
            "/v2.7/predict/mutation-context",
            method="POST",
            body={"cancer_id": "BLCA", "top_k": 1},
        ),
        ProbeSpec(
            "route.45",
            "/v2.7/predict/mutation-subgroup-lncrna",
            "/v2.7/predict/mutation-subgroup-lncrna",
            method="POST",
            body={"cancer_id": "BLCA", "top_k": 1},
        ),
        ProbeSpec("route.46", "/api/site/downloads", "/api/site/downloads"),
        ProbeSpec(
            "route.47",
            "/api/site/download/{key}",
            f"/api/site/download/{download}",
            expected_statuses=(200, 206),
            response_kind="download",
            headers={"Range": "bytes=0-0"},
            dependency_fallback=bool(fallbacks.get("download")),
        ),
        ProbeSpec("route.48", "/api/search", "/api/search?q=MALAT1&limit=1"),
        ProbeSpec("route.49", "/api/cancers", "/api/cancers"),
        ProbeSpec(
            "route.50", "/api/cancers/{cancer_id}", "/api/cancers/BLCA?limit=1"
        ),
        ProbeSpec(
            "route.51",
            "/api/lncrnas/{lncrna}",
            f"/api/lncrnas/{lnc}?cancer=BLCA",
        ),
    ]
    observed = tuple((spec.method, spec.route_template) for spec in specs)
    if observed != HTML_51_ROUTE_CONTRACT:
        raise RuntimeError("Materialized route specs drifted from the 51-route contract")
    return specs


def build_cancer_sweep_specs(cancers: Sequence[str] = TCGA_CANCERS) -> list[ProbeSpec]:
    """Build the 33-cancer matrix for every issue-bearing route family."""

    specs: list[ProbeSpec] = []
    families = (
        (
            "sc_summary",
            "/api/site/sc-summary/{cancer}",
            lambda cancer: f"/api/site/sc-summary/{cancer}",
        ),
        (
            "cancer_detail",
            "/api/site/cancer/{cancer_id}",
            lambda cancer: f"/api/site/cancer/{cancer}?limit=5",
        ),
        (
            "sc_umap",
            "/api/site/sc-umap/{cancer}",
            lambda cancer: f"/api/site/sc-umap/{cancer}?max_points=1000",
        ),
        (
            "clinical_visuals",
            "/api/site/clinical/visuals",
            lambda cancer: f"/api/site/clinical/visuals?cancer={cancer}",
        ),
        (
            "legacy_cancer",
            "/api/cancers/{cancer_id}",
            lambda cancer: f"/api/cancers/{cancer}?limit=1",
        ),
        (
            "candidate_page",
            "/api/site/cancer/{cancer_id}/candidates",
            lambda cancer: f"/api/site/cancer/{cancer}/candidates?limit=1",
        ),
        (
            "predicted_candidate",
            "/api/site/cancer/{cancer_id}/candidates",
            lambda cancer: (
                f"/api/site/cancer/{cancer}/candidates"
                "?confidence=predicted_candidate&limit=1"
            ),
        ),
        (
            "mutation_site",
            "/api/site/mutation/cancer/{cancer_id}",
            lambda cancer: f"/api/site/mutation/cancer/{cancer}?limit=1",
        ),
        (
            "mutation_compat",
            "/v2.7/mutation/cancer/{cancer_id}",
            lambda cancer: f"/v2.7/mutation/cancer/{cancer}?limit=1",
        ),
    )
    for family, template, path_builder in families:
        for cancer in cancers:
            specs.append(
                ProbeSpec(
                    probe_id=f"cancer.{family}.{cancer}",
                    route_template=template,
                    path=path_builder(cancer),
                    group=f"cancer_sweep:{family}",
                    cancer_id=cancer,
                )
            )
    return specs


def candidate_marker_spec() -> ProbeSpec:
    return ProbeSpec(
        "preflight.candidate_marker",
        "/__candidate/health",
        "/__candidate/health",
        group="preflight",
    )


def openapi_spec() -> ProbeSpec:
    return ProbeSpec(
        "preflight.openapi", "/openapi.json", "/openapi.json", group="preflight"
    )


def _normalize_openapi_path(path: str) -> str:
    return re.sub(r"\{([^}:]+):[^}]+\}", r"{\1}", path)


def check_openapi(outcome: ProbeOutcome) -> dict[str, Any]:
    pairs: set[tuple[str, str]] = set()
    if isinstance(outcome.payload, Mapping):
        paths = outcome.payload.get("paths")
        if isinstance(paths, Mapping):
            for path, operations in paths.items():
                if not isinstance(operations, Mapping):
                    continue
                for method in operations:
                    upper = str(method).upper()
                    if upper in {"GET", "POST", "PUT", "DELETE", "PATCH"}:
                        pairs.add((upper, _normalize_openapi_path(str(path))))
    expected = {
        (method, _normalize_openapi_path(path))
        for method, path in HTML_51_ROUTE_CONTRACT
    }
    missing = sorted(expected - pairs)
    return {
        "name": "openapi_51_route_contract",
        "pass": outcome.http_pass and not missing,
        "http_pass": outcome.http_pass,
        "expected_count": 51,
        "observed_operation_count": len(pairs),
        "missing": [{"method": method, "path": path} for method, path in missing],
    }


def check_candidate_marker(outcome: ProbeOutcome) -> dict[str, Any]:
    payload = outcome.payload if isinstance(outcome.payload, Mapping) else {}
    production_flag = payload.get("production_deployed")
    return {
        "name": "isolated_candidate_marker",
        "pass": outcome.http_pass and production_flag is False,
        "http_pass": outcome.http_pass,
        "production_deployed": production_flag,
        "reason": None
        if outcome.http_pass and production_flag is False
        else "candidate marker missing, unreachable, or production_deployed is not false",
    }


def check_runtime_version_lineage(
    marker_outcome: ProbeOutcome,
    final_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Require the live candidate marker to echo the hash-bound V3.2 winner."""

    payload = marker_outcome.payload if isinstance(marker_outcome.payload, Mapping) else {}
    expected_sha = _normalized_sha256(final_binding.get("binding_sha256"))
    observed_sha = _normalized_sha256(payload.get("final_binding_sha256"))
    winner_id = str(final_binding.get("winner_id") or "").strip()
    marker_winner_id = str(payload.get("winner_id") or "").strip()
    fresh = payload.get("fresh_training") is True or payload.get("new_training") is True
    old_predictions_false = payload.get("old_predictions_used") is False
    old_rankings_false = payload.get("old_rankings_used") is False
    passed = bool(
        marker_outcome.http_pass
        and final_binding.get("pass") is True
        and payload.get("analysis_version") == ANALYSIS_VERSION
        and str(payload.get("model_version") or "").upper() == "V3.2"
        and expected_sha is not None
        and observed_sha == expected_sha
        and winner_id
        and marker_winner_id == winner_id
        and fresh
        and old_predictions_false
        and old_rankings_false
        and payload.get("production_deployed") is False
    )
    return {
        "name": "runtime_version_and_winner_lineage",
        "pass": passed,
        "http_pass": marker_outcome.http_pass,
        "analysis_version": payload.get("analysis_version"),
        "model_version": payload.get("model_version"),
        "expected_final_binding_sha256": expected_sha,
        "observed_final_binding_sha256": observed_sha,
        "expected_winner_id": winner_id or None,
        "observed_winner_id": marker_winner_id or None,
        "fresh_training_attested": fresh,
        "old_predictions_explicitly_false": old_predictions_false,
        "old_rankings_explicitly_false": old_rankings_false,
        "production_deployed": payload.get("production_deployed"),
    }


def check_route_contract(outcomes: Sequence[ProbeOutcome]) -> dict[str, Any]:
    required = [outcome for outcome in outcomes if outcome.spec.required_http]
    failures = [outcome.to_record() for outcome in required if not outcome.http_pass]
    return {
        "name": "historical_51_route_http_contract",
        "pass": len(required) == 51 and not failures,
        "expected_count": 51,
        "observed_count": len(required),
        "http_pass_count": sum(outcome.http_pass for outcome in required),
        "http_failure_count": len(failures),
        "failures": failures,
    }


def check_cancer_matrix(outcomes: Sequence[ProbeOutcome]) -> dict[str, Any]:
    by_family: dict[str, list[ProbeOutcome]] = {}
    for outcome in outcomes:
        family = outcome.spec.group.split(":", 1)[-1]
        by_family.setdefault(family, []).append(outcome)
    families: dict[str, Any] = {}
    matrix_pass = True
    for family, rows in sorted(by_family.items()):
        seen = {row.spec.cancer_id for row in rows if row.spec.cancer_id}
        missing = sorted(set(TCGA_CANCERS) - seen)
        failures = [row for row in rows if not row.http_pass]
        typed_failures = [row for row in failures if row.typed_unavailable_paths]
        untyped_failures = [row for row in failures if not row.typed_unavailable_paths]
        family_pass = len(rows) == 33 and not missing and not failures
        matrix_pass = matrix_pass and family_pass
        families[family] = {
            "pass": family_pass,
            "expected_cancers": 33,
            "observed_cancers": len(seen),
            "http_2xx": sum(row.http_pass for row in rows),
            "typed_unavailable_http_failures": len(typed_failures),
            "untyped_http_failures": len(untyped_failures),
            "missing_cancers": missing,
            "failed_cancers": [
                {
                    "cancer_id": row.spec.cancer_id,
                    "status": row.status,
                    "transport_error": row.transport_error,
                    "typed_unavailable_paths": row.typed_unavailable_paths,
                }
                for row in failures
            ],
            "scientific_states": _count_states(rows),
        }
    return {
        "name": "nine_route_families_x_33_cancers",
        "pass": matrix_pass and len(families) == 9,
        "expected_families": 9,
        "observed_families": len(families),
        "expected_probe_count": 9 * 33,
        "observed_probe_count": len(outcomes),
        "families": families,
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in {float("inf"), float("-inf")}:
        return None
    return number


def check_clinical_invariant(outcome: ProbeOutcome) -> dict[str, Any]:
    rows = _first_rows(outcome.payload, ("endpoints", "results", "rows"))
    violations: list[dict[str, Any]] = []
    malformed: list[dict[str, Any]] = []
    duplicate_keys: list[str] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        cancer = str(row.get("cancer_id") or "")
        endpoint = str(row.get("endpoint") or "")
        key = (cancer, endpoint)
        if key in seen:
            duplicate_keys.append(f"{cancer}:{endpoint}")
        seen.add(key)
        patients = _finite_number(row.get("n_patients"))
        events = _finite_number(row.get("n_events"))
        if patients is None or events is None:
            malformed.append(
                {
                    "cancer_id": cancer,
                    "endpoint": endpoint,
                    "n_patients": row.get("n_patients"),
                    "n_events": row.get("n_events"),
                }
            )
        elif events < 0 or patients < 0 or events > patients:
            violations.append(
                {
                    "cancer_id": cancer,
                    "endpoint": endpoint,
                    "n_patients": patients,
                    "n_events": events,
                }
            )
    passed = bool(rows) and outcome.http_pass and not violations and not malformed and not duplicate_keys
    return {
        "name": "clinical_events_le_patients_and_unique_endpoint",
        "pass": passed,
        "http_pass": outcome.http_pass,
        "row_count": len(rows),
        "violation_count": len(violations),
        "malformed_count": len(malformed),
        "duplicate_key_count": len(set(duplicate_keys)),
        "violations": violations[:100],
        "malformed": malformed[:100],
        "duplicate_keys": sorted(set(duplicate_keys))[:100],
    }


def _integerish(value: Any) -> int | None:
    number = _finite_number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def check_candidate_classification(
    cancer_overview: ProbeOutcome,
    predicted_sweep: Sequence[ProbeOutcome],
    *,
    expected_bound_total: int | None = None,
) -> dict[str, Any]:
    rows = _first_rows(cancer_overview.payload, ("cancers", "results", "rows"))
    overview_counts: dict[str, int] = {}
    missing_field: list[str] = []
    overview_duplicates: list[str] = []
    for row in rows:
        cancer = str(row.get("cancer_id") or "").upper()
        if cancer in overview_counts:
            overview_duplicates.append(cancer)
        value = _integerish(row.get("predicted_candidate_relations"))
        if value is None:
            missing_field.append(cancer or "<unknown>")
        else:
            overview_counts[cancer] = value

    sweep_counts: dict[str, int] = {}
    wrong_classes: list[dict[str, Any]] = []
    sweep_http_failures: list[str] = []
    sweep_duplicates: list[str] = []
    for outcome in predicted_sweep:
        cancer = str(outcome.spec.cancer_id or "").upper()
        if not outcome.http_pass:
            sweep_http_failures.append(cancer)
            continue
        payload = outcome.payload if isinstance(outcome.payload, Mapping) else {}
        total = _integerish(payload.get("total"))
        if total is None:
            result_rows = _first_rows(payload, ("results", "rows", "candidates"))
            total = len(result_rows)
        if cancer in sweep_counts:
            sweep_duplicates.append(cancer)
        sweep_counts[cancer] = total
        for row in _first_rows(payload, ("results", "rows", "candidates")):
            observed = _normalized_status(
                row.get("relationship_class") or row.get("confidence_tier")
            )
            if observed != "predicted_candidate":
                wrong_classes.append(
                    {"cancer_id": cancer, "observed_class": observed}
                )

    overview_total = sum(overview_counts.values())
    sweep_total = sum(sweep_counts.values())
    nonzero_cancers = sorted(
        cancer
        for cancer in set(overview_counts) | set(sweep_counts)
        if max(overview_counts.get(cancer, 0), sweep_counts.get(cancer, 0)) > 0
    )
    has_classified_rows = max(overview_total, sweep_total) > 0
    expected_cancers = set(TCGA_CANCERS)
    overview_cancers = set(overview_counts)
    sweep_cancers = set(sweep_counts)
    count_disagreements = [
        {
            "cancer_id": cancer,
            "overview_count": overview_counts.get(cancer),
            "sweep_count": sweep_counts.get(cancer),
        }
        for cancer in TCGA_CANCERS
        if overview_counts.get(cancer) != sweep_counts.get(cancer)
    ]
    exact_33_coverage = bool(
        len(rows) == 33
        and len(predicted_sweep) == 33
        and overview_cancers == expected_cancers
        and sweep_cancers == expected_cancers
        and not overview_duplicates
        and not sweep_duplicates
    )
    binding_total_matches = bool(
        expected_bound_total is not None
        and expected_bound_total > 0
        and overview_total == expected_bound_total
        and sweep_total == expected_bound_total
    )
    passed = (
        cancer_overview.http_pass
        and exact_33_coverage
        and not missing_field
        and not sweep_http_failures
        and not wrong_classes
        and not count_disagreements
        and has_classified_rows
    )
    return {
        "name": "predicted_candidate_not_swallowed_by_supported_branch",
        "pass": passed,
        "overview_http_pass": cancer_overview.http_pass,
        "overview_row_count": len(rows),
        "overview_predicted_total": overview_total,
        "overview_missing_field_cancers": missing_field,
        "overview_duplicate_cancers": sorted(set(overview_duplicates)),
        "sweep_predicted_total": sweep_total,
        "sweep_http_failure_cancers": sorted(sweep_http_failures),
        "sweep_duplicate_cancers": sorted(set(sweep_duplicates)),
        "exact_33_cancer_coverage": exact_33_coverage,
        "count_disagreements": count_disagreements,
        "expected_bound_total": expected_bound_total,
        "binding_total_matches": binding_total_matches,
        "nonzero_cancers": nonzero_cancers,
        "wrong_class_rows": wrong_classes[:100],
        "semantics": (
            "At least one predicted_candidate must remain after the mutually "
            "exclusive observed -> predicted -> supported -> exploratory routing."
        ),
    }


def check_chinese_search(outcome: ProbeOutcome) -> dict[str, Any]:
    rows = _first_rows(outcome.payload, ("results", "rows"))
    result_ids = [
        str(row.get("id") or row.get("cancer_id") or "").strip().upper()
        for row in rows
    ]
    return {
        "name": "chinese_search_lung_cancer_resolves_luad",
        "pass": outcome.http_pass and "LUAD" in result_ids,
        "http_pass": outcome.http_pass,
        "query": "肺癌",
        "result_ids": result_ids,
    }


def _typed_component(
    payload: Mapping[str, Any], key: str, *, allow_scalar_status: bool = False
) -> tuple[bool, dict[str, Any]]:
    value = payload.get(key)
    if allow_scalar_status and isinstance(value, str):
        status = _normalized_status(value)
        return status in AVAILABLE_STATUSES | UNAVAILABLE_STATUSES, {
            "status": status,
            "has_reason": False,
        }
    if not isinstance(value, Mapping):
        return False, {"status": None, "has_reason": False}
    status = _normalized_status(value.get("status"))
    allowed = AVAILABLE_STATUSES | UNAVAILABLE_STATUSES | EMPTY_BUT_EVALUATED_STATUSES
    typed = status in allowed
    if status in UNAVAILABLE_STATUSES:
        typed = typed and any(key in value for key in ("reason", "detail", "reason_code"))
    return typed, {
        "status": status,
        "has_reason": any(key in value for key in ("reason", "detail", "reason_code")),
    }


def check_optional_components(
    lnc_visuals: ProbeOutcome, clinical_visuals: ProbeOutcome
) -> dict[str, Any]:
    lnc_payload = lnc_visuals.payload if isinstance(lnc_visuals.payload, Mapping) else {}
    clinical_payload = (
        clinical_visuals.payload if isinstance(clinical_visuals.payload, Mapping) else {}
    )
    lnc_components: dict[str, Any] = {}
    lnc_pass = lnc_visuals.http_pass and lnc_payload.get(
        "partial_components_do_not_fail_request"
    ) is True
    for key in ("pathway_heatmap", "bulk_scatter", "survival_curve", "drug_evidence"):
        typed, detail = _typed_component(lnc_payload, key)
        lnc_components[key] = detail
        lnc_pass = lnc_pass and typed
    km_typed, km_detail = _typed_component(clinical_payload, "kaplan_meier")
    event_typed, event_detail = _typed_component(
        clinical_payload, "event_count_status", allow_scalar_status=True
    )
    clinical_pass = clinical_visuals.http_pass and km_typed and event_typed
    unavailable_component_count = sum(
        detail.get("status") in UNAVAILABLE_STATUSES
        for detail in [*lnc_components.values(), km_detail, event_detail]
    )
    scientific_available = (
        lnc_visuals.http_pass
        and clinical_visuals.http_pass
        and unavailable_component_count == 0
    )
    return {
        "name": "optional_components_fail_soft_with_typed_status",
        "pass": lnc_pass and clinical_pass,
        "scientific_available": scientific_available,
        "typed_unavailable_component_count": unavailable_component_count,
        "lncrna_visuals": {
            "pass": lnc_pass,
            "http_pass": lnc_visuals.http_pass,
            "top_level_status": lnc_payload.get("status"),
            "partial_components_do_not_fail_request": lnc_payload.get(
                "partial_components_do_not_fail_request"
            ),
            "components": lnc_components,
        },
        "clinical_visuals": {
            "pass": clinical_pass,
            "http_pass": clinical_visuals.http_pass,
            "top_level_status": clinical_payload.get("status"),
            "kaplan_meier": km_detail,
            "event_count_status": event_detail,
        },
    }


def _recursive_values(payload: Any, keys: set[str]) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key in keys:
                values.append(value)
            values.extend(_recursive_values(value, keys))
    elif isinstance(payload, list):
        for value in payload[:500]:
            values.extend(_recursive_values(value, keys))
    return values


def check_mutation_status(
    status_outcome: ProbeOutcome,
    mutation_sweep: Sequence[ProbeOutcome],
) -> dict[str, Any]:
    payload = status_outcome.payload if isinstance(status_outcome.payload, Mapping) else {}
    availability_values = _recursive_values(
        payload, {"mutation_release_available", "available", "enabled"}
    )
    explicit_available = any(value is True for value in availability_values)
    statuses = _status_values(payload)
    if "available" in statuses:
        explicit_available = True
    versions = [
        str(value)
        for value in _recursive_values(
            payload,
            {"analysis_version", "model_version", "mutation_model_release", "release_id"},
        )
        if value is not None
    ]
    fresh_values = _recursive_values(
        payload, {"new_training", "new_training_attestation"}
    )
    old_values = _recursive_values(
        payload,
        {
            "old_predictions_used",
            "old_predictions_used_as_features",
            "old_rankings_used_as_outputs",
            "old_checkpoint_loaded",
        },
    )
    version_is_v32 = any(
        re.search(r"(?:V?3[._]2|V3\.2)", value, re.IGNORECASE) for value in versions
    )
    fresh_attested = bool(fresh_values) and any(value is True for value in fresh_values)
    old_forbidden = bool(old_values) and all(value is False for value in old_values)
    sweep_http_pass = len(mutation_sweep) == 66 and all(
        outcome.http_pass for outcome in mutation_sweep
    )
    family_cancers: dict[str, set[str]] = {}
    for outcome in mutation_sweep:
        family = outcome.spec.group.split(":", 1)[-1]
        family_cancers.setdefault(family, set()).add(
            str(outcome.spec.cancer_id or "").upper()
        )
    exact_sweep_coverage = bool(
        set(family_cancers) == {"mutation_site", "mutation_compat"}
        and all(cancers == set(TCGA_CANCERS) for cancers in family_cancers.values())
        and len(mutation_sweep) == 66
    )
    sweep_scientific_available = bool(
        exact_sweep_coverage
        and all(
            outcome.http_pass
            and outcome.scientific_state == "AVAILABLE"
            and not outcome.typed_unavailable_paths
            for outcome in mutation_sweep
        )
    )
    science_available = bool(
        status_outcome.http_pass
        and explicit_available
        and version_is_v32
        and fresh_attested
        and old_forbidden
        and sweep_scientific_available
    )
    return {
        "name": "mutation_v32_http_and_scientific_status",
        "http_pass": status_outcome.http_pass and sweep_http_pass,
        "scientific_available": science_available,
        "pass": status_outcome.http_pass and sweep_http_pass and science_available,
        "status_endpoint_http_pass": status_outcome.http_pass,
        "sweep_expected": 66,
        "sweep_observed": len(mutation_sweep),
        "sweep_http_pass_count": sum(outcome.http_pass for outcome in mutation_sweep),
        "sweep_exact_33_cancer_coverage": exact_sweep_coverage,
        "sweep_scientific_available": sweep_scientific_available,
        "sweep_scientific_states": _count_states(mutation_sweep),
        "explicit_available": explicit_available,
        "version_is_v32": version_is_v32,
        "versions": versions,
        "fresh_training_attested": fresh_attested,
        "old_artifacts_explicitly_forbidden": old_forbidden,
        "typed_unavailable_paths": status_outcome.typed_unavailable_paths,
        "interpretation": (
            "HTTP pass and scientific availability are independent; a 200 status "
            "documenting release_unavailable is not a published mutation result."
        ),
    }


def build_download_specs(catalog_outcome: ProbeOutcome, base_url: str) -> tuple[list[ProbeSpec], list[str]]:
    rows = _first_rows(catalog_outcome.payload, ("files", "downloads", "results"))
    specs: list[ProbeSpec] = []
    unsafe: list[str] = []
    normalized_base = validate_candidate_base_url(base_url)
    base_parsed = urlparse(normalized_base)
    for index, row in enumerate(rows):
        key = _first_string(row, ("key", "download_id", "id", "name"))
        url = _first_string(row, ("download_url", "url", "href"))
        if not key:
            unsafe.append(f"row[{index}]:missing_key")
            continue
        if url:
            parsed = urlparse(url)
            if parsed.scheme or parsed.netloc:
                if (
                    parsed.scheme != base_parsed.scheme
                    or parsed.hostname != base_parsed.hostname
                    or parsed.port != base_parsed.port
                ):
                    unsafe.append(f"{key}:external_or_wrong_origin_url")
                    continue
                path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
            else:
                if not url.startswith("/"):
                    unsafe.append(f"{key}:non_absolute_path")
                    continue
                path = url
        else:
            path = f"/api/site/download/{quote(key, safe='')}"
        specs.append(
            ProbeSpec(
                probe_id=f"download.{index:04d}.{key}",
                route_template="/api/site/download/{key}",
                path=path,
                group="download_keys",
                expected_statuses=(200, 206),
                headers={"Range": "bytes=0-0"},
                response_kind="download",
            )
        )
    return specs, unsafe


def check_downloads(
    catalog_outcome: ProbeOutcome,
    download_outcomes: Sequence[ProbeOutcome],
    unsafe_entries: Sequence[str],
    *,
    expected_catalog_payload_sha256: str | None = None,
    expected_file_count: int | None = None,
) -> dict[str, Any]:
    rows = _first_rows(catalog_outcome.payload, ("files", "downloads", "results"))
    keys = [
        _first_string(row, ("key", "download_id", "id", "name")) for row in rows
    ]
    non_null_keys = [key for key in keys if key]
    duplicates = sorted({key for key in non_null_keys if non_null_keys.count(key) > 1})
    broken = [outcome.to_record() for outcome in download_outcomes if not outcome.http_pass]
    http_pass = (
        catalog_outcome.http_pass
        and bool(rows)
        and len(download_outcomes) == len(rows)
        and not unsafe_entries
        and not duplicates
        and not broken
    )
    unattested_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        status = _normalized_status(
            row.get("scientific_status") or row.get("status")
        )
        sha256 = _normalized_sha256(row.get("sha256"))
        attested = bool(
            status == "available"
            and row.get("scientifically_attested") is True
            and row.get("analysis_version") == ANALYSIS_VERSION
            and sha256 is not None
        )
        if not attested:
            unattested_rows.append(
                {
                    "index": index,
                    "key": _first_string(row, ("key", "download_id", "id", "name")),
                    "scientific_status": status or None,
                    "scientifically_attested": row.get("scientifically_attested"),
                    "analysis_version": row.get("analysis_version"),
                    "sha256_valid": sha256 is not None,
                }
            )
    expected_catalog_sha = _normalized_sha256(expected_catalog_payload_sha256)
    observed_catalog_sha = _normalized_sha256(catalog_outcome.body_sample_sha256)
    catalog_hash_matches_binding = bool(
        expected_catalog_sha is not None
        and observed_catalog_sha == expected_catalog_sha
        and catalog_outcome.body_truncated is False
    )
    expected_count_matches = bool(
        expected_file_count is not None
        and expected_file_count > 0
        and len(rows) == expected_file_count
    )
    scientific_available = bool(
        rows
        and not unattested_rows
        and catalog_hash_matches_binding
        and expected_count_matches
    )
    return {
        "name": "download_catalog_every_key_resolves",
        "pass": http_pass,
        "http_pass": http_pass,
        "scientific_available": scientific_available,
        "catalog_http_pass": catalog_outcome.http_pass,
        "listed_count": len(rows),
        "unique_key_count": len(set(non_null_keys)),
        "probed_count": len(download_outcomes),
        "duplicates": duplicates,
        "unsafe_entries": list(unsafe_entries),
        "broken_count": len(broken),
        "broken": broken,
        "scientifically_attested_count": len(rows) - len(unattested_rows),
        "unattested_count": len(unattested_rows),
        "unattested": unattested_rows,
        "expected_catalog_payload_sha256": expected_catalog_sha,
        "observed_catalog_payload_sha256": observed_catalog_sha,
        "catalog_payload_truncated": catalog_outcome.body_truncated,
        "catalog_hash_matches_binding": catalog_hash_matches_binding,
        "expected_file_count": expected_file_count,
        "expected_file_count_matches": expected_count_matches,
        "range_probe": True,
    }


def check_geneset_exact(outcome: ProbeOutcome) -> dict[str, Any]:
    payload = outcome.payload if isinstance(outcome.payload, Mapping) else {}
    broadcast_values = _recursive_values(
        payload, {"legacy_family_broadcast", "family_broadcast_used"}
    )
    target_values = [
        _normalized_status(value)
        for value in _recursive_values(
            payload, {"pathway_target_level", "target_level", "granularity"}
        )
    ]
    exact = any(value in {"exact", "exact_pathway", "pathway_id"} for value in target_values)
    no_broadcast = bool(broadcast_values) and all(
        value is False for value in broadcast_values
    )
    state = outcome.scientific_state
    top_status = _normalized_status(payload.get("status"))
    science_available = (
        outcome.http_pass
        and top_status == "available"
        and not outcome.typed_unavailable_paths
    )
    return {
        "name": "geneset_exact_no_family_broadcast",
        "http_pass": outcome.http_pass,
        "scientific_available": science_available,
        "exact_target_attested": exact,
        "family_broadcast_false": no_broadcast,
        "pass": science_available and exact and no_broadcast,
        "target_values": target_values,
        "broadcast_values": broadcast_values,
        "scientific_state": state,
    }


def check_single_cell_science(
    single_cell_sweep: Sequence[ProbeOutcome],
) -> dict[str, Any]:
    """Check runtime single-cell coverage without promoting typed partial HTTP 200s."""

    expected_families = {"sc_summary", "sc_umap"}
    by_family: dict[str, list[ProbeOutcome]] = {}
    for outcome in single_cell_sweep:
        family = outcome.spec.group.split(":", 1)[-1]
        by_family.setdefault(family, []).append(outcome)
    family_results: dict[str, Any] = {}
    all_pass = set(by_family) == expected_families
    for family in sorted(expected_families):
        rows = by_family.get(family, [])
        cancers = [str(row.spec.cancer_id or "").upper() for row in rows]
        exact_coverage = bool(
            len(rows) == 33
            and len(set(cancers)) == 33
            and set(cancers) == set(TCGA_CANCERS)
        )
        available = [
            row
            for row in rows
            if row.http_pass
            and row.scientific_state == "AVAILABLE"
            and not row.typed_unavailable_paths
        ]
        http_pass = exact_coverage and all(row.http_pass for row in rows)
        science_pass = exact_coverage and len(available) == 33
        all_pass = all_pass and http_pass and science_pass
        family_results[family] = {
            "expected_cancers": 33,
            "observed_cancers": len(set(cancers)),
            "http_pass_count": sum(row.http_pass for row in rows),
            "scientifically_available_count": len(available),
            "exact_33_cancer_coverage": exact_coverage,
            "scientific_states": _count_states(rows),
            "typed_unavailable_cancers": sorted(
                {
                    str(row.spec.cancer_id or "").upper()
                    for row in rows
                    if row.typed_unavailable_paths
                    or row.scientific_state != "AVAILABLE"
                }
            ),
        }
    return {
        "name": "single_cell_two_route_families_x_33_scientifically_available",
        "pass": all_pass,
        "http_pass": bool(
            set(by_family) == expected_families
            and all(
                detail["exact_33_cancer_coverage"]
                and detail["http_pass_count"] == 33
                for detail in family_results.values()
            )
        ),
        "scientific_available": all_pass,
        "expected_probe_count": 66,
        "observed_probe_count": len(single_cell_sweep),
        "families": family_results,
        "interpretation": (
            "A typed partial or unavailable HTTP 200 is covered by the route audit "
            "but is not a fresh single-cell scientific result."
        ),
    }


def _count_states(outcomes: Iterable[ProbeOutcome]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.scientific_state] = counts.get(outcome.scientific_state, 0) + 1
    return dict(sorted(counts.items()))


def _find(outcomes: Sequence[ProbeOutcome], probe_id: str) -> ProbeOutcome:
    for outcome in outcomes:
        if outcome.spec.probe_id == probe_id:
            return outcome
    raise KeyError(probe_id)


def _issue_status_from_family(family: Mapping[str, Any]) -> tuple[str, str]:
    if family.get("http_2xx") == 33:
        states = family.get("scientific_states", {})
        unavailable = sum(
            count
            for state, count in states.items()
            if "UNAVAILABLE" in state or "NOT_OBSERVED" in state
        )
        return "HTTP_REPAIRED", "AVAILABLE_OR_TYPED_PARTIAL" if unavailable == 0 else "PARTIAL"
    if family.get("http_2xx", 0) > 0:
        return "HTTP_PARTIAL", "UNRESOLVED"
    return "HTTP_NOT_REPAIRED", "UNRESOLVED"


def build_issue_matrix(
    route_outcomes: Sequence[ProbeOutcome],
    cancer_check: Mapping[str, Any],
    checks: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    families = cancer_check.get("families", {})
    rows: list[dict[str, Any]] = []
    for issue, family_name, route in (
        ("single_cell_summary_33", "sc_summary", "/api/site/sc-summary/{cancer}"),
        ("cancer_detail_33", "cancer_detail", "/api/site/cancer/{cancer_id}"),
        ("single_cell_umap_33", "sc_umap", "/api/site/sc-umap/{cancer}"),
        ("clinical_visuals_33", "clinical_visuals", "/api/site/clinical/visuals"),
        ("legacy_cancer_33", "legacy_cancer", "/api/cancers/{cancer_id}"),
    ):
        http, science = _issue_status_from_family(families.get(family_name, {}))
        rows.append(
            {
                "issue": issue,
                "route": route,
                "http_conclusion": http,
                "scientific_conclusion": science,
                "evidence": families.get(family_name, {}),
            }
        )
    for issue, probe_id, check_name in (
        ("lncrna_visuals", "route.11", "optional_components"),
        ("geneset_detail", "route.14", "geneset_exact"),
        ("downloads", "route.46", "downloads"),
        ("mutation", "route.37", "mutation"),
        ("clinical_event_counts", "route.32", "clinical_invariant"),
        ("predicted_candidate_zero", "route.26", "candidate_classification"),
        ("chinese_search_empty", "route.09", "chinese_search"),
    ):
        outcome = _find(route_outcomes, probe_id)
        check = checks[check_name]
        science_available = check.get("scientific_available")
        if science_available is None:
            science_available = check.get("pass")
        rows.append(
            {
                "issue": issue,
                "route": outcome.spec.route_template,
                "http_conclusion": "HTTP_REPAIRED" if outcome.http_pass else "HTTP_NOT_REPAIRED",
                "scientific_conclusion": (
                    "AVAILABLE_OR_INVARIANT_FIXED" if science_available else "UNAVAILABLE_OR_INVALID"
                ),
                "evidence": check,
            }
        )
    return rows


def execute_runtime_audit(
    base_url: str,
    *,
    workers: int = 6,
    timeout_seconds: float = 15.0,
    max_body_bytes: int = 2 * 1024 * 1024,
    requester: Requester = _request,
    final_binding_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the complete audit and return a machine-readable report."""

    base = validate_candidate_base_url(base_url)
    started = time.monotonic()
    preflight = run_specs(
        base,
        [candidate_marker_spec(), openapi_spec()],
        workers=min(workers, 2),
        timeout_seconds=timeout_seconds,
        max_body_bytes=max_body_bytes,
        requester=requester,
    )
    discovery = run_specs(
        base,
        discovery_specs(),
        workers=min(workers, 4),
        timeout_seconds=timeout_seconds,
        max_body_bytes=max_body_bytes,
        requester=requester,
    )
    context = derive_context(discovery)
    route_outcomes = run_specs(
        base,
        build_route_contract_specs(context),
        workers=workers,
        timeout_seconds=timeout_seconds,
        max_body_bytes=max_body_bytes,
        requester=requester,
    )
    cancer_outcomes = run_specs(
        base,
        build_cancer_sweep_specs(),
        workers=workers,
        timeout_seconds=timeout_seconds,
        max_body_bytes=max_body_bytes,
        requester=requester,
    )
    download_specs, unsafe_downloads = build_download_specs(
        _find(route_outcomes, "route.46"), base
    )
    download_outcomes = run_specs(
        base,
        download_specs,
        workers=workers,
        timeout_seconds=timeout_seconds,
        max_body_bytes=max_body_bytes,
        requester=requester,
    )

    marker_outcome = _find(preflight, "preflight.candidate_marker")
    candidate_marker = check_candidate_marker(marker_outcome)
    final_binding = check_final_binding_attestation(final_binding_evidence)
    runtime_lineage = check_runtime_version_lineage(marker_outcome, final_binding)
    openapi = check_openapi(_find(preflight, "preflight.openapi"))
    route_contract = check_route_contract(route_outcomes)
    cancer_matrix = check_cancer_matrix(cancer_outcomes)
    clinical_invariant = check_clinical_invariant(_find(route_outcomes, "route.32"))
    predicted = [
        outcome
        for outcome in cancer_outcomes
        if outcome.spec.group == "cancer_sweep:predicted_candidate"
    ]
    candidate_classification = check_candidate_classification(
        _find(route_outcomes, "route.26"),
        predicted,
        expected_bound_total=final_binding.get("predicted_candidate_total"),
    )
    chinese_search = check_chinese_search(_find(route_outcomes, "route.09"))
    optional_components = check_optional_components(
        _find(route_outcomes, "route.11"), _find(route_outcomes, "route.36")
    )
    mutation_sweep = [
        outcome
        for outcome in cancer_outcomes
        if outcome.spec.group
        in {"cancer_sweep:mutation_site", "cancer_sweep:mutation_compat"}
    ]
    mutation = check_mutation_status(
        _find(route_outcomes, "route.37"), mutation_sweep
    )
    downloads = check_downloads(
        _find(route_outcomes, "route.46"),
        download_outcomes,
        unsafe_downloads,
        expected_catalog_payload_sha256=final_binding.get(
            "download_catalog_payload_sha256"
        ),
        expected_file_count=final_binding.get("download_file_count"),
    )
    geneset_exact = check_geneset_exact(_find(route_outcomes, "route.14"))
    single_cell_sweep = [
        outcome
        for outcome in cancer_outcomes
        if outcome.spec.group in {"cancer_sweep:sc_summary", "cancer_sweep:sc_umap"}
    ]
    single_cell = check_single_cell_science(single_cell_sweep)

    named_checks = {
        "candidate_marker": candidate_marker,
        "final_binding": final_binding,
        "runtime_lineage": runtime_lineage,
        "openapi": openapi,
        "route_contract": route_contract,
        "cancer_matrix": cancer_matrix,
        "clinical_invariant": clinical_invariant,
        "candidate_classification": candidate_classification,
        "chinese_search": chinese_search,
        "optional_components": optional_components,
        "mutation": mutation,
        "downloads": downloads,
        "geneset_exact": geneset_exact,
        "single_cell": single_cell,
    }
    http_gate_names = (
        "candidate_marker",
        "openapi",
        "route_contract",
        "cancer_matrix",
        "optional_components",
        "downloads",
    )
    data_quality_names = (
        "clinical_invariant",
        "candidate_classification",
        "chinese_search",
        "geneset_exact",
    )
    http_gate = all(named_checks[name]["pass"] for name in http_gate_names)
    data_quality_gate = all(named_checks[name]["pass"] for name in data_quality_names)
    scientific_gate = bool(
        final_binding["pass"]
        and runtime_lineage["pass"]
        and candidate_classification["binding_total_matches"]
        and mutation["pass"]
        and geneset_exact["pass"]
        and downloads["scientific_available"]
        and single_cell["pass"]
        and optional_components["scientific_available"]
    )
    all_outcomes = [
        *preflight,
        *discovery,
        *route_outcomes,
        *cancer_outcomes,
        *download_outcomes,
    ]
    report: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            "base_url": base,
            "loopback_only": True,
            "production_port_8260_touched": False,
            "candidate_marker_required": True,
        },
        "configuration": {
            "workers": workers,
            "request_timeout_seconds": timeout_seconds,
            "max_body_bytes": max_body_bytes,
            "historical_route_contract_count": len(HTML_51_ROUTE_CONTRACT),
            "cancer_count": len(TCGA_CANCERS),
            "cancer_route_family_count": 9,
            "download_strategy": "GET Range bytes=0-0; bounded response read",
            "final_binding_required_for_full_gate": True,
            "final_binding_provided": final_binding["provided"],
        },
        "coverage": {
            "historical_routes_expected": 51,
            "historical_routes_observed": len(route_outcomes),
            "cancers_expected": 33,
            "cancers": list(TCGA_CANCERS),
            "cancer_matrix_probes_expected": 9 * 33,
            "cancer_matrix_probes_observed": len(cancer_outcomes),
            "download_keys_listed": downloads["listed_count"],
            "download_keys_probed": downloads["probed_count"],
        },
        "verdicts": {
            "http_remediation_pass": http_gate,
            "data_processing_invariants_pass": data_quality_gate,
            "scientific_artifacts_available": scientific_gate,
            "full_candidate_acceptance_pass": (
                http_gate and data_quality_gate and scientific_gate
            ),
            "important_semantics": (
                "HTTP remediation does not imply scientific artifact availability."
            ),
        },
        "checks": named_checks,
        "issue_matrix": build_issue_matrix(
            route_outcomes, cancer_matrix, named_checks
        ),
        "scientific_state_counts": _count_states(all_outcomes),
        "probe_results": {
            "preflight": [outcome.to_record() for outcome in preflight],
            "discovery": [outcome.to_record() for outcome in discovery],
            "route_contract": [outcome.to_record() for outcome in route_outcomes],
            "cancer_matrix": [outcome.to_record() for outcome in cancer_outcomes],
            "download_keys": [outcome.to_record() for outcome in download_outcomes],
        },
        "representative_context": context,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    return report


def render_markdown(report: Mapping[str, Any]) -> str:
    verdicts = report["verdicts"]
    coverage = report["coverage"]
    checks = report["checks"]
    lines = [
        "# CancerLncAtlas V3.2 candidate runtime acceptance",
        "",
        f"Generated: `{report['generated_at_utc']}`  ",
        f"Target: `{report['target']['base_url']}`  ",
        "Production port 8260 touched: **NO**",
        "",
        "## Independent verdicts",
        "",
        "| Dimension | Verdict |",
        "|---|---|",
        f"| HTTP remediation | {'PASS' if verdicts['http_remediation_pass'] else 'FAIL'} |",
        f"| Data-processing invariants | {'PASS' if verdicts['data_processing_invariants_pass'] else 'FAIL'} |",
        f"| Scientific artifacts available | {'PASS' if verdicts['scientific_artifacts_available'] else 'FAIL'} |",
        f"| Full candidate acceptance | {'PASS' if verdicts['full_candidate_acceptance_pass'] else 'FAIL'} |",
        "",
        "> A HTTP 200 response containing `source_unavailable` proves fail-soft UI/API behavior only; it does not prove that the scientific artifact is published.",
        "",
        "## Coverage",
        "",
        f"- Historical OpenAPI contract: {coverage['historical_routes_observed']}/{coverage['historical_routes_expected']} routes.",
        f"- Cancer matrix: {coverage['cancer_matrix_probes_observed']}/{coverage['cancer_matrix_probes_expected']} probes across 33 cancers and 9 issue-bearing route families.",
        f"- Download keys: {coverage['download_keys_probed']}/{coverage['download_keys_listed']} probed with a bounded Range request.",
        "",
        "## Original issue list disposition",
        "",
        "| Issue | Route | HTTP | Scientific/data |",
        "|---|---|---|---|",
    ]
    for row in report["issue_matrix"]:
        lines.append(
            f"| {row['issue']} | `{row['route']}` | {row['http_conclusion']} | {row['scientific_conclusion']} |"
        )
    lines.extend(
        [
            "",
            "## Gate details",
            "",
            "| Check | Pass | Key evidence |",
            "|---|---:|---|",
        ]
    )
    for name, check in checks.items():
        evidence_bits = []
        for key in (
            "expected_count",
            "observed_count",
            "http_pass_count",
            "row_count",
            "violation_count",
            "listed_count",
            "probed_count",
            "broken_count",
            "sweep_http_pass_count",
            "scientific_available",
        ):
            if key in check:
                evidence_bits.append(f"{key}={check[key]}")
        lines.append(
            f"| {name} | {'PASS' if check.get('pass') else 'FAIL'} | {'; '.join(evidence_bits)} |"
        )
    lines.extend(
        [
            "",
            "## 33-cancer HTTP matrix summary",
            "",
            "| Route family | HTTP 2xx | Typed unavailable HTTP failures | Untyped failures |",
            "|---|---:|---:|---:|",
        ]
    )
    for family, detail in checks["cancer_matrix"]["families"].items():
        lines.append(
            f"| {family} | {detail['http_2xx']}/33 | {detail['typed_unavailable_http_failures']} | {detail['untyped_http_failures']} |"
        )
    lines.extend(
        [
            "",
            "## Historical 51-route results",
            "",
            "| # | Method | Route | HTTP | Science classification |",
            "|---:|---|---|---:|---|",
        ]
    )
    for index, row in enumerate(report["probe_results"]["route_contract"], start=1):
        status = row["status"] if row["status"] else row["transport_error"]
        lines.append(
            f"| {index} | {row['method']} | `{row['route_template']}` | {status} | {row['scientific_state']} |"
        )
    failures = [
        row
        for row in report["probe_results"]["cancer_matrix"]
        if not row["http_pass"]
    ]
    if failures:
        lines.extend(
            [
                "",
                "## Cancer-route failures",
                "",
                "| Family | Cancer | HTTP | Typed unavailable | Error |",
                "|---|---|---:|---|---|",
            ]
        )
        for row in failures:
            typed = ", ".join(row["typed_unavailable_paths"][:4])
            error = str(row["transport_error"] or "").replace("|", "\\|")
            lines.append(
                f"| {row['group']} | {row['cancer_id']} | {row['status']} | {typed} | {error} |"
            )
    return "\n".join(lines) + "\n"


def report_exit_code(report: Mapping[str, Any], gate: str) -> int:
    if gate == "report-only":
        return 0
    verdicts = report["verdicts"]
    if gate == "http":
        return 0 if (
            verdicts["http_remediation_pass"]
            and verdicts["data_processing_invariants_pass"]
        ) else 2
    if gate == "full":
        return 0 if verdicts["full_candidate_acceptance_pass"] else 3
    raise ValueError(f"Unknown gate: {gate}")
